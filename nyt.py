import os
import json
import numpy as np
from sklearn.cluster import KMeans
from pinecone import Pinecone
import anthropic
import time
from typing import List, Dict, Any, Optional
import sys

# Load environment variables
import dotenv
dotenv.load_dotenv()

# Set API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# Print API key status for debugging
print("API Key Status:")
print(f"Anthropic API Key available: {'Yes' if anthropic_api_key else 'No'}")
print(f"Pinecone API Key available: {'Yes' if pinecone_api_key else 'No'}")
print(f"OpenAI API Key available: {'Yes' if openai_api_key else 'No'}")

# Initialize API clients
try:
    from openai import OpenAI
    client = OpenAI(api_key=openai_api_key)
except Exception as e:
    print(f"Warning: OpenAI client initialization error - {str(e)}")
    print("This script requires OpenAI embeddings to work with the Pinecone database.")

try:
    claude_client = anthropic.Anthropic(api_key=anthropic_api_key)
    print("Claude client initialized successfully")
except Exception as e:
    print(f"Error initializing Claude client: {str(e)}")
    print("Please check your Anthropic API key.")

# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key)
index_name = 'nyt-archive'
index = pc.Index(index_name)

# ----- UTILITY FUNCTIONS -----

def get_namespaces():
    """Get all namespaces in the index and their vector counts"""
    index_stats = index.describe_index_stats()
    if 'namespaces' in index_stats and index_stats['namespaces']:
        # Create a list of (namespace, vector_count) tuples and sort by namespace name (descending)
        sorted_namespaces = sorted(
            [(namespace, stats['vector_count']) for namespace, stats in index_stats['namespaces'].items()],
            key=lambda x: x[0], 
            reverse=True
        )
        return sorted_namespaces
    return []

def create_embedding(text):
    """Create an embedding vector for a text using OpenAI's API"""
    response = client.embeddings.create(
        input=[text],
        model='text-embedding-3-small'
    )
    return response.data[0].embedding

def query_pinecone(query_text, namespace=None, top_k=200, threshold=0.3, filter_dict=None):
    """Query Pinecone index with a text query and return matches above threshold"""
    # Generate embedding for query
    query_embedding = create_embedding(query_text)
    
    # Build query parameters
    query_params = {
        "vector": query_embedding,
        "top_k": top_k,
        "include_values": True,
        "include_metadata": True,
    }
    
    if namespace:
        query_params["namespace"] = namespace
    
    if filter_dict:
        query_params["filter"] = filter_dict
    
    # Perform query
    query_response = index.query(**query_params)
    
    # Filter results by threshold
    filtered_matches = [match for match in query_response['matches'] if match['score'] > threshold]
    
    return {
        "matches": filtered_matches,
        "total_matches": len(filtered_matches),
        "query_text": query_text,
        "namespace": namespace,
        "threshold": threshold
    }

def extract_article_data(articles):
    """Extract relevant metadata from articles for analysis"""
    extracted_data = []
    
    for article in articles:
        metadata = article.get('metadata', {})
        headline = metadata.get('headline_main', 'No headline')
        abstract = metadata.get('abstract', 'No abstract')
        pub_date = metadata.get('pub_date', 'Unknown date')
        
        extracted_data.append({
            "headline": headline,
            "abstract": abstract,
            "pub_date": pub_date,
            "score": article.get('score', 0)
        })
    
    return extracted_data

# ----- FUNCTION CALLING SCHEMA -----

functions_schema = [
    {
        "name": "query_legislative_issues",
        "description": "Search the New York Times articles for issues related to the legislative process in the US Congress",
        "parameters": {
            "type": "object",
            "properties": {
                "query_text": {
                    "type": "string",
                    "description": "The specific legislative topic to search for (e.g., 'immigration reform', 'healthcare policy', 'tax legislation')"
                },
                "year": {
                    "type": "string",
                    "description": "The year to search in (YYYY format)"
                },
                "namespace": {
                    "type": "string", 
                    "description": "Specific namespace to search (format: YYYY_MM, e.g. 2023_5)"
                },
                "threshold": {
                    "type": "number",
                    "description": "Minimum similarity score (0.0-1.0) to include an article"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of articles to return"
                }
            },
            "required": ["query_text"]
        }
    },
    {
        "name": "get_available_periods",
        "description": "Get information about available time periods (namespaces) in the NYT archive",
        "parameters": {
            "type": "object",
            "properties": {
                "year": {
                    "type": "string",
                    "description": "Optional year to filter namespaces (YYYY format)"
                }
            }
        }
    }
]

# ----- FUNCTION IMPLEMENTATION -----

def handle_query_legislative_issues(parameters):
    """Handle query for legislative issues"""
    query_text = parameters.get("query_text")
    year = parameters.get("year")
    namespace = parameters.get("namespace")  # Allow direct namespace specification
    threshold = parameters.get("threshold", 0.35)
    limit = parameters.get("limit", 100)  # Increased default limit to get more articles
    
    print(f"Querying for: {query_text}")
    all_results = []
    queried_namespaces = []
    
    # Get all available namespaces
    all_namespaces = get_namespaces()
    
    # Case 1: Specific namespace provided
    if namespace:
        print(f"Searching in specific namespace: {namespace}")
        results = query_pinecone(query_text, namespace=namespace, top_k=100, threshold=threshold)
        all_results.extend(results["matches"])
        queried_namespaces.append(namespace)
    
    # Case 2: Year specified but no specific namespace
    elif year:
        print(f"Searching in all namespaces for year: {year}")
        year_namespaces = [ns for ns, _ in all_namespaces if ns.startswith(f"{year}_")]
        
        if not year_namespaces:
            print(f"No namespaces found for year {year}")
            return {
                "articles": [],
                "total_found": 0,
                "query": query_text,
                "year": year,
                "error": f"No data available for year {year}"
            }
        
        # Query each namespace for this year
        for ns in year_namespaces:
            print(f"  - Querying namespace: {ns}")
            results = query_pinecone(query_text, namespace=ns, top_k=200, threshold=threshold)
            all_results.extend(results["matches"])
            queried_namespaces.append(ns)
    
    # Case 3: No year or namespace specified - use most recent year
    else:
        # Sort namespaces to find most recent year
        if all_namespaces:
            years = sorted(set([ns.split('_')[0] for ns, _ in all_namespaces]), reverse=True)
            most_recent_year = years[0]
            print(f"No year specified. Using most recent year: {most_recent_year}")
            
            year_namespaces = [ns for ns, _ in all_namespaces if ns.startswith(f"{most_recent_year}_")]
            
            # Query each namespace for this year
            for ns in year_namespaces:
                print(f"  - Querying namespace: {ns}")
                results = query_pinecone(query_text, namespace=ns, top_k=200, threshold=threshold)
                all_results.extend(results["matches"])
                queried_namespaces.append(ns)
    
    # Sort by score and get a substantial number of results for comprehensive analysis
    all_results.sort(key=lambda x: x["score"], reverse=True)
    limited_results = all_results[:min(len(all_results), limit)]
    
    print(f"Found {len(all_results)} total matching articles")
    print(f"Using top {len(limited_results)} results for analysis")
    
    # Extract article data for analysis
    article_data = extract_article_data(limited_results)
    
    return {
        "articles": article_data,
        "total_found": len(limited_results),
        "query": query_text,
        "namespaces_queried": queried_namespaces,
        "threshold": threshold
    }

def handle_get_available_periods(parameters):
    """Get information about available time periods"""
    namespaces = get_namespaces()
    
    # Group by year
    years = {}
    for namespace, count in namespaces:
        if '_' in namespace:
            year, month = namespace.split('_')
            if year not in years:
                years[year] = {"months": [], "total_articles": 0}
            years[year]["months"].append({"month": month, "article_count": count})
            years[year]["total_articles"] += count
    
    # Sort years and months
    result = {
        "years": []
    }
    
    for year in sorted(years.keys(), reverse=True):
        year_data = {
            "year": year,
            "total_articles": years[year]["total_articles"],
            "months": sorted(years[year]["months"], key=lambda x: int(x["month"]), reverse=True)
        }
        result["years"].append(year_data)
    
    return result

# ----- STREAMING IMPLEMENTATION -----

def analyze_legislative_issues(query=None, year=None, namespace=None, recursive=True, discovery_mode=False, iteration=1, max_iterations=3):
    """Analyze legislative issues in Congress with Claude and stream the response"""
    print(f"Analyzing legislative issues in Congress... [Iteration {iteration}/{max_iterations}]\n")
    
    # Default query if none provided
    if not query:
        query = "legislative process US Congress policy issues"
    
    # Step 1: Get available namespaces if needed
    if not year and not namespace:
        print("Getting available time periods...")
        periods_data = handle_get_available_periods({})
        years = sorted([period["year"] for period in periods_data["years"]], reverse=True)
        if years:
            print(f"Available years: {', '.join(years[:5])}...")
    
    # Step 2: Get articles
    print(f"Searching for articles related to: {query}")
    if year:
        print(f"Year: {year}")
    if namespace:
        print(f"Namespace: {namespace}")
    
    # Get articles
    parameters = {
        "query_text": query,
        "year": year,
        "namespace": namespace,
        "threshold": 0.35,
        "limit": 100  # Increased to get more articles for better analysis
    }
    
    articles_data = handle_query_legislative_issues(parameters)
    
    if articles_data.get("error"):
        print(f"Error: {articles_data['error']}")
        return
        
    print(f"Found {articles_data['total_found']} relevant articles")
    print(f"Queried namespaces: {', '.join(articles_data.get('namespaces_queried', []))}\n")
    
    # Step 2: Send to Claude for analysis
    print("Analyzing articles with Claude...\n")
    
    # Prepare system prompt with explicit thinking instructions
    system_prompt = """You are a legislative analyst specializing in US Congressional policy who replicates the Binder (1999) legislative gridlock study approach.

USE THINKING: You MUST show your detailed thought process by using <thinking></thinking> tags. This helps users understand your analysis process in real-time as you consider different ways to cluster articles.

For each cluster of related articles:
1. Give the cluster a descriptive name (5-7 words capturing the policy area)
2. Provide a list of 5-10 key articles in the cluster
3. Write a 3-5 sentence summary of the legislative issues/debates represented
4. Explain how these issues relate to the formal legislative process or gridlock in Congress

When analyzing, pay special attention to:
- Bills being debated or passed in Congress
- Committee activities and hearings
- Partisan dynamics affecting legislation
- Key policy stakeholders (interest groups, agencies)
- Legislative procedures being used or debated
- Signs of legislative gridlock or progress

Your output must be structured as valid JSON surrounded by <json></json> tags with this structure:
{
  "clusters": [
    {
      "name": "string", 
      "articles": ["headline1", "headline2", ...], 
      "summary": "string", 
      "legislative_process": "string"
    },
    ...
  ],
  "continue_exploration": boolean,  // whether you want to explore more namespaces/years
  "next_periods": ["YYYY", "YYYY_MM", ...],  // which periods to explore next (years or specific namespaces)
  "reasoning": "string",  // explanation of why you want to continue or stop
  "recommended_queries": ["query1", "query2", ...] // suggested queries for further exploration
}
"""
    
    # Prepare user message
    user_message = f"""I need you to analyze these New York Times articles about legislative issues in Congress using Sarah Binder's (1999) approach to studying legislative gridlock.

TIME PERIOD: {articles_data.get('year', 'recent years')}
QUERY: {query}
CURRENT ITERATION: {iteration} of {max_iterations}

Begin your analysis by thinking through how to approach these articles. You MUST use <thinking></thinking> tags to show your detailed reasoning process. For example:

<thinking>
I'll first scan through all the articles to identify key legislative issues being covered in the media.
For each article, I'll note:
- What specific legislation or policy issue is being discussed
- Which stage of the legislative process it's in (introduced, committee, floor debate, passed one chamber, etc.)
- What political factors are affecting its progress
- Whether there are signs of gridlock or advancement

Then I'll group the articles into distinct policy clusters based on common themes, bills, or issue areas.
For each cluster, I'll analyze:
- The nature of the legislative activity
- Partisan dynamics affecting the legislation
- Key stakeholders involved
- Whether the issue appears gridlocked or making progress

I'll also consider which additional time periods might contain relevant information that would improve my analysis.
</thinking>

Please identify the main clusters of legislative issues and analyze them according to Binder's approach to studying gridlock through media coverage.

Here are the articles:
{json.dumps(articles_data['articles'], indent=2)}

In your response:
1. Decide if we should continue exploring more namespaces/years ("continue_exploration")
2. Suggest which time periods to explore next ("next_periods") - can be years or specific namespaces
3. Recommend specific queries that might yield better results ("recommended_queries")
4. Explain your reasoning

Remember to structure your final response as valid JSON within <json></json> tags.
"""
    
    # Stream the response
    print("Claude is analyzing the articles through a Binder (1999) legislative gridlock framework...\n")
    
    # Create the message
    try:
        print("Creating Claude 3.7 Sonnet message...")
        message = claude_client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=4000,
            temperature=0.2,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_message}
            ],
            stream=True,
        )
    except Exception as e:
        print(f"Error creating Claude message: {str(e)}")
        # Try alternate model if 3-7-sonnet fails
        print("Trying alternate Claude model...")
        try:
            message = claude_client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=4000,
                temperature=0.2,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_message}
                ],
                stream=True
            )
        except Exception as e2:
            print(f"Error with alternate model: {str(e2)}")
            print("Falling back to simpler model...")
            message = claude_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=4000,
                temperature=0.2,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_message}
                ],
                stream=True
            )
    
    # Stream the response
    print("Streaming Claude's analysis in real-time:\n")
    analysis_result = ""
    
    for chunk in message:
        if chunk.type == "content_block_delta":
            text = chunk.delta.text
            print(text, end="", flush=True)
            analysis_result += text
            time.sleep(0.01)  # Small delay to make streaming visible
    
    # Try to extract JSON data for potential further processing
    json_data = None
    if "<json>" in analysis_result and "</json>" in analysis_result:
        try:
            json_text = analysis_result.split("<json>")[1].split("</json>")[0]
            json_data = json.loads(json_text)
            
            # Extract continuation information if available
            if json_data.get("continue_exploration") and iteration < max_iterations:
                print("\n\nContinuation Recommended:")
                print(f"Reason: {json_data.get('reasoning', 'Not specified')}")
                
                next_periods = json_data.get("next_periods", [])
                if not next_periods and json_data.get("next_namespaces"):  # Backward compatibility
                    next_periods = json_data.get("next_namespaces")
                
                if next_periods:
                    print(f"Suggested Periods to Explore: {', '.join(next_periods)}")
                
                recommended_queries = json_data.get("recommended_queries", [])
                if recommended_queries:
                    print(f"Recommended Queries: {', '.join(recommended_queries)}")
                
                # Handle continuation based on recursive flag
                if recursive:
                    # Auto-continue if --no-recursive not specified, otherwise ask user
                    proceed_auto = True
                    if os.environ.get("INTERACTIVE_MODE", "").lower() == "true":
                        proceed = input("\nContinue with next exploration iteration? (y/n): ")
                        proceed_auto = proceed.lower() in ['y', 'yes']
                    
                    if proceed_auto:
                        print("\n" + "="*80 + "\n")
                        print(f"AUTOMATICALLY CONTINUING TO NEXT ITERATION PER MODEL'S RECOMMENDATION")
                        print("="*80 + "\n")
                        
                        # Determine next query and period
                        next_query = recommended_queries[0] if recommended_queries else query
                        next_period = next_periods[0] if next_periods else None
                        
                        # Determine if next_period is a year or namespace
                        next_year = None
                        next_namespace = None
                        if next_period:
                            if '_' in next_period:  # It's a namespace (e.g., 2023_01)
                                next_namespace = next_period
                            else:  # It's a year (e.g., 2023)
                                next_year = next_period
                        
                        # Launch next iteration with Claude's recommendations
                        analyze_legislative_issues(
                            query=next_query,
                            year=next_year,
                            namespace=next_namespace,
                            recursive=recursive,
                            discovery_mode=discovery_mode,
                            iteration=iteration+1,
                            max_iterations=max_iterations
                        )
                        return
            else:
                if iteration >= max_iterations:
                    print("\n\nReached maximum iteration limit.")
                else:
                    print("\n\nNo further exploration recommended.")
        except Exception as e:
            print(f"\nWarning: Could not parse JSON: {str(e)}")
    
    print("\n\nAnalysis complete!")
    return

# ----- COMMAND LINE INTERFACE -----

def run_full_agent():
    """Run Claude as a fully autonomous agent with function calling capabilities to analyze legislative issues"""
    print("Starting Claude as a fully autonomous legislative research agent...\n")
    
    # Define the system prompt for the agent
    system_prompt = """You are a Legislative Research Assistant specialized in analyzing policy issues in the U.S. Congress using Sarah Binder's (1999) approach to studying legislative gridlock through media analysis.

You have access to the following functions to help you collect and analyze data:
1. get_available_time_periods() - Returns information about all available time periods (years and months) in the database
2. query_articles(query_text, namespace, year, threshold) - Searches for articles matching your query in specified time periods
3. analyze_articles(articles, query_text) - Analyzes a collection of articles to identify legislative gridlock patterns

IMPORTANT: You should use these functions to perform your analysis. Think step by step about what data you need. You can call multiple functions in parallel when appropriate.

Your task is to analyze legislative gridlock patterns in Congress by examining media coverage. Following Binder's approach:
1. Identify key legislative issues being reported in the media
2. Determine which issues are experiencing gridlock vs. progress
3. Analyze the causes and dynamics of gridlock in different policy areas

You should first explore what time periods are available in our database, then search for relevant articles about legislative activity, and finally analyze those articles to identify patterns of gridlock.

Format your final analysis as structured JSON with distinct policy clusters and analysis of legislative dynamics for each cluster.
"""

    # Create initial function calling schema
    function_schema = [
        {
            "name": "get_available_time_periods",
            "description": "Get information about available time periods (years and months) in the NYT article database",
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {
                        "type": "string",
                        "description": "Optional year to filter results (YYYY format)"
                    }
                }
            }
        },
        {
            "name": "query_articles",
            "description": "Search for articles matching specific query text in the NYT database",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_text": {
                        "type": "string",
                        "description": "Query text to search for"
                    },
                    "namespace": {
                        "type": "string",
                        "description": "Specific namespace to search (format YYYY_MM, e.g. 2023_5)"
                    },
                    "year": {
                        "type": "string",
                        "description": "Year to search across all its months (YYYY format)"
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Minimum similarity threshold (0.0-1.0)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of articles to return" 
                    }
                },
                "required": ["query_text"]
            }
        },
        {
            "name": "analyze_articles",
            "description": "Analyze a collection of articles to identify legislative patterns and gridlock",
            "parameters": {
                "type": "object",
                "properties": {
                    "articles": {
                        "type": "array",
                        "description": "Array of article objects to analyze"
                    },
                    "query_text": {
                        "type": "string",
                        "description": "The original query used to find these articles"
                    },
                    "time_period": {
                        "type": "string",
                        "description": "Time period these articles are from"
                    }
                },
                "required": ["articles"]
            }
        }
    ]

    # Create initial prompt for Claude
    user_prompt = """I would like you to analyze legislative gridlock in the U.S. Congress using media coverage analysis.

Please approach this like Sarah Binder's 1999 study on legislative gridlock, where you examine how the media reports on Congressional activity to understand where and why gridlock occurs in the legislative process.

Start by determining what time periods are available in our database, then identify and search for relevant legislative issues, and finally analyze the articles to understand patterns of gridlock.

You can use the available functions to collect data and perform your analysis. Feel free to call multiple functions in parallel when appropriate to speed up your research.
"""

    # Initialize conversation history
    conversation = [
        {"role": "user", "content": user_prompt}
    ]
    
    # Maximum conversation turns to prevent infinite loops
    max_turns = 10
    current_turn = 0
    all_articles = []
    all_analysis = []
    
    print("\nStarting agent conversation...\n")
    print("-" * 80)
    
    while current_turn < max_turns:
        current_turn += 1
        print(f"\n[Turn {current_turn}/{max_turns}]")
        
        try:
            # Call Claude with the current conversation
            response = claude_client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=4000,
                temperature=0.2,
                system=system_prompt,
                messages=conversation,
                tools=function_schema
            )
            
            # Add the assistant's response to the conversation
            conversation.append({
                "role": "assistant", 
                "content": response.content,
                "tool_calls": response.tool_calls if response.tool_calls else []
            })
            
            # Print the assistant's thinking/response
            print("\nAgent's reasoning:")
            print("-" * 40)
            print(response.content)
            print("-" * 40)
            
            # Process tool calls if any
            if response.tool_calls and len(response.tool_calls) > 0:
                print(f"\nAgent is making {len(response.tool_calls)} function calls:")
                
                tool_results = []
                
                for i, tool_call in enumerate(response.tool_calls):
                    function_name = tool_call.name
                    function_args = tool_call.parameters
                    
                    print(f"  Function call {i+1}: {function_name}({json.dumps(function_args, indent=2)})")
                    
                    # Execute the function
                    result = None
                    
                    if function_name == "get_available_time_periods":
                        result = handle_get_available_periods(function_args)
                    
                    elif function_name == "query_articles":
                        # Map parameters to our existing function
                        query_params = {
                            "query_text": function_args.get("query_text"),
                            "namespace": function_args.get("namespace"),
                            "year": function_args.get("year"),
                            "threshold": function_args.get("threshold", 0.35),
                            "limit": function_args.get("limit", 100)  # Increased default limit
                        }
                        result = handle_query_legislative_issues(query_params)
                        
                        # Add articles to our collection
                        if result.get("articles"):
                            all_articles.extend(result.get("articles"))
                    
                    elif function_name == "analyze_articles":
                        # Here we'll use Claude to analyze the articles
                        articles = function_args.get("articles", [])
                        query = function_args.get("query_text", "legislative issues")
                        time_period = function_args.get("time_period", "various periods")
                        
                        # Construct a simple analysis prompt
                        analysis_prompt = f"""Please analyze these {len(articles)} articles about legislative issues from {time_period}.
                        
These articles were found using the query: {query}
                        
Please identify:
1. Distinct clusters of legislative issues
2. Signs of gridlock or progress in each cluster
3. Factors contributing to gridlock or progress
                        
Articles data:
{json.dumps(articles, indent=2)}

Provide your analysis in JSON format with:
- Named policy clusters
- Key articles in each cluster
- Summary of legislative dynamics
- Gridlock analysis for each cluster

Structure your response as valid JSON within <json></json> tags.
"""
                        # Get analysis
                        analysis_message = claude_client.messages.create(
                            model="claude-3-7-sonnet-20250219",
                            max_tokens=4000,
                            temperature=0.2,
                            messages=[{"role": "user", "content": analysis_prompt}]
                        )
                        
                        # Extract JSON if it exists
                        analysis_text = analysis_message.content
                        analysis_json = {}
                        
                        if "<json>" in analysis_text and "</json>" in analysis_text:
                            try:
                                json_text = analysis_text.split("<json>")[1].split("</json>")[0]
                                analysis_json = json.loads(json_text)
                                all_analysis.append(analysis_json)
                            except Exception as e:
                                print(f"Error parsing analysis JSON: {e}")
                        
                        result = {
                            "analysis_summary": "Analysis completed successfully",
                            "clusters_found": len(analysis_json.get("clusters", [])),
                            "full_analysis": analysis_json
                        }
                    
                    else:
                        result = {"error": f"Unknown function: {function_name}"}
                    
                    # Add the function result to the conversation
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result)
                    })
                
                # Add all tool results to the conversation
                conversation.extend(tool_results)
                
                print("\nFunction calls completed. Continuing conversation...")
                
            else:
                # No more function calls - check if we need to continue
                print("\nAgent has completed its analysis.")
                
                # Check for a final JSON response
                if "<json>" in response.content and "</json>" in response.content:
                    try:
                        json_text = response.content.split("<json>")[1].split("</json>")[0]
                        final_analysis = json.loads(json_text)
                        
                        print("\nFinal Analysis Summary:")
                        if "clusters" in final_analysis:
                            print(f"Found {len(final_analysis['clusters'])} legislative issue clusters:")
                            for i, cluster in enumerate(final_analysis["clusters"]):
                                print(f"  {i+1}. {cluster.get('name', 'Unnamed cluster')}")
                        
                        all_analysis.append(final_analysis)
                    except Exception as e:
                        print(f"Error parsing final JSON: {e}")
                
                # End the conversation
                break
                
        except Exception as e:
            print(f"Error in agent conversation: {str(e)}")
            break
    
    print("\n" + "="*80)
    print("Agent conversation complete!")
    
    # Provide summary of all collected data
    print(f"\nTotal articles collected: {len(all_articles)}")
    print(f"Total analyses generated: {len(all_analysis)}")
    
    # Print the final comprehensive analysis if available
    if all_analysis:
        final = all_analysis[-1]
        if "clusters" in final:
            print("\nFinal Legislative Gridlock Analysis:")
            for i, cluster in enumerate(final["clusters"]):
                print(f"\n{i+1}. {cluster.get('name', 'Unnamed Cluster')}")
                print(f"   Summary: {cluster.get('summary', '')[:150]}...")
    
    return
def main():
    """Command line interface for the legislative issues analyzer"""
    print("NYT Legislative Issues Analyzer")
    print("===============================")
    
    # Check if API keys are available
    if not anthropic_api_key:
        print("Error: ANTHROPIC_API_KEY not found in environment variables.")
        print("Please set your Anthropic API key in .env file.")
        return
    
    # Parse command line arguments
    query = None
    year = None
    namespace = None
    list_namespaces = False
    interactive = False
    no_recursive = False
    discovery_mode = False
    agent_mode = False
    
    # Set interactive mode environment variable to control auto-continue behavior
    os.environ["INTERACTIVE_MODE"] = "false"  # Default to auto-continue
    
    if len(sys.argv) > 1:
        # Check for flags
        if "--year" in sys.argv:
            year_index = sys.argv.index("--year") + 1
            if year_index < len(sys.argv):
                year = sys.argv[year_index]
                # Remove the year arguments
                sys.argv.pop(year_index)
                sys.argv.remove("--year")
        
        if "--namespace" in sys.argv:
            ns_index = sys.argv.index("--namespace") + 1
            if ns_index < len(sys.argv):
                namespace = sys.argv[ns_index]
                # Remove the namespace arguments
                sys.argv.pop(ns_index)
                sys.argv.remove("--namespace")
        
        if "--list" in sys.argv:
            list_namespaces = True
            sys.argv.remove("--list")
            
        if "--interactive" in sys.argv:
            interactive = True
            os.environ["INTERACTIVE_MODE"] = "true"  # Set to ask user before continuing
            sys.argv.remove("--interactive")
            
        if "--agent" in sys.argv:
            agent_mode = True
            sys.argv.remove("--agent")
            
        if "--no-recursive" in sys.argv:
            no_recursive = True
            sys.argv.remove("--no-recursive")
            
        if "--discover" in sys.argv:
            discovery_mode = True
            sys.argv.remove("--discover")
        
        # Rest of arguments form the query
        if len(sys.argv) > 1:
            query = " ".join(sys.argv[1:])
    
    # Handle listing namespaces
    if list_namespaces:
        print("Available time periods in the NYT archive:")
        periods_data = handle_get_available_periods({"year": year})
        
        # Print years and months
        for year_data in periods_data["years"]:
            print(f"\n{year_data['year']}: {year_data['total_articles']} total articles")
            # Group by quarters for cleaner display
            months = year_data["months"]
            for month in months:
                print(f"  {year_data['year']}_{month['month']}: {month['article_count']} articles")
        
        return
    
    # Agent mode - fully autonomous with function calling
    if agent_mode:
        print("\nRunning in full agent mode - Claude will control the entire research process")
        print("Using Sarah Binder's (1999) legislative gridlock framework")
        try:
            run_full_agent()
        except Exception as e:
            print(f"Error in agent mode: {str(e)}")
            import traceback
            traceback.print_exc()
        return
        
    # Interactive mode
    if interactive:
        print("Interactive Mode: Claude will decide which namespaces to explore")
        
        if not query and not discovery_mode:
            query_choice = input("Enter your query about legislative issues (or press Enter for default): ")
            if query_choice:
                query = query_choice
            
        if not year and not namespace:
            year_choice = input("Enter a specific year to focus on (or press Enter for most recent): ")
            if year_choice:
                year = year_choice
        
        discovery_choice = input("Use discovery mode to automatically find legislative issues? (y/n): ")
        if discovery_choice.lower() in ['y', 'yes']:
            discovery_mode = True
            print("Discovery mode enabled: Claude will identify legislative issues without a specific query")
                
        print(f"Query: {query or ('Auto-discovery' if discovery_mode else 'legislative process in Congress')}")
        print(f"Year: {year or 'most recent'}")
        print(f"Namespace: {namespace or 'automatically selected'}")
        print(f"Recursive exploration: {'No' if no_recursive else 'Yes'}")
        print(f"Discovery mode: {'Yes' if discovery_mode else 'No'}")
        print()
    
    # Show Binder study reference
    if discovery_mode:
        print("\nOperating in discovery mode - replicating Binder (1999) legislative gridlock study")
        print("This mode automatically identifies legislative issues being discussed in the media")
        print("and clusters them to analyze Congressional activity and gridlock\n")
    
    # Run the analysis
    try:
        analyze_legislative_issues(
            query=query, 
            year=year, 
            namespace=namespace, 
            recursive=not no_recursive,
            discovery_mode=discovery_mode
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
