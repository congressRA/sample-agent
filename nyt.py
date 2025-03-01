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

def query_pinecone(query_text, namespace=None, top_k=100, threshold=0.3, filter_dict=None):
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
    limit = parameters.get("limit", 50)
    
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
            results = query_pinecone(query_text, namespace=ns, top_k=100, threshold=threshold)
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
                results = query_pinecone(query_text, namespace=ns, top_k=100, threshold=threshold)
                all_results.extend(results["matches"])
                queried_namespaces.append(ns)
    
    # Sort by score and limit results
    all_results.sort(key=lambda x: x["score"], reverse=True)
    limited_results = all_results[:limit]
    
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

def analyze_legislative_issues(query=None, year=None, namespace=None, recursive=True, discovery_mode=False):
    """Analyze legislative issues in Congress with Claude and stream the response"""
    print("Analyzing legislative issues in Congress...\n")
    
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
        "limit": 50
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
  "continue_exploration": boolean,  // whether you want to explore more namespaces
  "next_namespaces": ["YYYY_MM", ...],  // which namespaces to explore next (if continue_exploration is true)
  "reasoning": "string",  // brief explanation of why you want to continue or stop
  "recommended_queries": ["query1", "query2", ...] // suggested queries for further exploration
}
"""
    
    # Prepare user message
    user_message = f"""I need you to analyze these New York Times articles about legislative issues in Congress using Sarah Binder's (1999) approach to studying legislative gridlock.

TIME PERIOD: {articles_data.get('year', 'recent years')}
QUERY: {query}

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
1. Decide if we should continue exploring more namespaces ("continue_exploration")
2. Suggest which time periods to explore next ("next_namespaces")
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
            if json_data.get("continue_exploration"):
                print("\n\nContinuation Recommended:")
                print(f"Reason: {json_data.get('reasoning', 'Not specified')}")
                
                if json_data.get("next_namespaces"):
                    print(f"Suggested Namespaces: {', '.join(json_data['next_namespaces'])}")
                
                if json_data.get("recommended_queries"):
                    print(f"Recommended Queries: {', '.join(json_data['recommended_queries'])}")
        except Exception as e:
            print(f"\nWarning: Could not parse JSON: {str(e)}")
    
    print("\n\nAnalysis complete!")
    return

# ----- COMMAND LINE INTERFACE -----

def run_claude_agent():
    """Run Claude as a fully autonomous agent to analyze legislative issues"""
    print("Starting Claude legislative research agent...\n")
    
    # Define the system prompt for the agent
    system_prompt = """You are a Legislative Research Assistant specialized in analyzing policy issues in the U.S. Congress using Sarah Binder's (1999) approach to studying legislative gridlock through media analysis.

Your task is to analyze New York Times articles to identify patterns in legislative activities and understand gridlock. Show your thought process as you work through this task.

You have access to a database of New York Times articles that cover legislative activities. 

IMPORTANT: Begin by thinking about what you're trying to accomplish and what information you need. Structure your thinking clearly.

You should:
1. First determine what time period you want to analyze 
2. Decide what specific search terms would best identify articles about legislative gridlock
3. Collect and analyze the articles
4. Identify clusters of related legislative issues
5. Analyze each cluster to understand signs of gridlock or progress
6. Determine if you need to explore more time periods

Your final output should be a structured JSON with distinct policy clusters, analysis of legislative dynamics, and recommendations for further exploration.
"""

    # Create initial prompt for Claude
    user_prompt = """Please help me analyze legislative gridlock in the U.S. Congress using media coverage analysis.

I'd like you to approach this like Sarah Binder's 1999 study on legislative gridlock, where you examine how the media reports on Congressional activity to understand where and why gridlock occurs in the legislative process.

Begin by thinking through your approach to this research question. Which time periods should we examine? What search terms would be most effective? How should we identify and cluster the legislative issues?

Feel free to explore recent years (2019-2024) to understand current patterns of gridlock.
"""

    # Create Claude agent
    try:
        print("Creating Claude 3.7 Sonnet agent...")
        message = claude_client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=4000,
            temperature=0.3,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ],
            stream=True
        )
    except Exception as e:
        print(f"Error creating Claude agent: {str(e)}")
        print("Trying alternate Claude model...")
        message = claude_client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=4000,
            temperature=0.3,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ],
            stream=True
        )
    
    # Stream the agent's thinking
    print("\nClaude agent is thinking about the research approach:\n")
    initial_response = ""
    
    for chunk in message:
        if chunk.type == "content_block_delta":
            text = chunk.delta.text
            print(text, end="", flush=True)
            initial_response += text
            time.sleep(0.01)  # Small delay for readability
    
    # Extract specific years and search terms from Claude's response
    years_to_analyze = ["2023", "2024"]  # Default to recent years
    search_terms = ["Congress legislation gridlock", "legislative process Congress"]  # Default search terms
    
    # Look for recommendations in Claude's response
    if "I recommend analyzing" in initial_response and "years" in initial_response:
        # Try to extract years Claude wants to analyze
        import re
        year_matches = re.findall(r'20\d\d', initial_response)
        if year_matches:
            years_to_analyze = list(set(year_matches))[:3]  # Take up to 3 unique years
    
    if "search terms" in initial_response:
        # Try to extract search terms
        lines = initial_response.split('\n')
        for i, line in enumerate(lines):
            if "search terms" in line.lower() and i < len(lines)-1:
                terms = lines[i+1:i+4]
                potential_terms = []
                for term in terms:
                    term = term.strip('- "\'')
                    if term and len(term.split()) >= 2:
                        potential_terms.append(term)
                if potential_terms:
                    search_terms = potential_terms[:2]  # Take up to 2 search terms
    
    # Now use Claude's recommendations to run actual analysis
    print("\n\nBased on Claude's analysis, we will:")
    print(f"- Analyze years: {', '.join(years_to_analyze)}")
    print(f"- Use search terms: {search_terms}")
    print("\nRunning detailed legislative analysis...\n")
    
    all_articles = []
    
    # For each year and search term, collect articles
    for year in years_to_analyze:
        for search_term in search_terms:
            print(f"\nSearching for '{search_term}' in year {year}...")
            try:
                parameters = {
                    "query_text": search_term,
                    "year": year,
                    "threshold": 0.35,
                    "limit": 25  # Limit per search to avoid overwhelming
                }
                
                articles_data = handle_query_legislative_issues(parameters)
                
                if articles_data.get("total_found", 0) > 0:
                    print(f"Found {articles_data['total_found']} relevant articles")
                    all_articles.extend(articles_data.get("articles", []))
                else:
                    print("No relevant articles found with this search term and year")
            except Exception as e:
                print(f"Error searching for articles: {str(e)}")
    
    # Deduplicate articles
    seen_headlines = set()
    unique_articles = []
    for article in all_articles:
        headline = article.get("headline", "")
        if headline and headline not in seen_headlines:
            seen_headlines.add(headline)
            unique_articles.append(article)
    
    print(f"\nTotal unique articles collected: {len(unique_articles)}")
    
    # Exit if we don't have enough articles
    if len(unique_articles) < 10:
        print("Not enough articles found to perform meaningful analysis. Please try different search terms or years.")
        return
    
    # Limit articles to avoid overwhelming Claude
    analysis_articles = unique_articles[:50]
    
    # Create a new prompt for Claude to analyze the articles
    analysis_prompt = f"""I've collected {len(analysis_articles)} articles about legislative activities in Congress from the years {', '.join(years_to_analyze)}.

Please analyze these articles using Binder's approach to studying legislative gridlock. Identify distinct clusters of legislative issues, analyze the signs of gridlock or progress for each cluster, and determine the factors contributing to gridlock.

For each cluster, please provide:
1. A descriptive name for the policy area
2. Key articles in the cluster
3. Analysis of the legislative dynamics
4. Signs of gridlock or progress
5. Factors contributing to the outcome

Here are the articles:
{json.dumps(analysis_articles, indent=2)}

Please structure your final analysis as JSON within <json></json> tags.
"""
    
    # Create Claude analysis
    try:
        print("\nAnalyzing collected articles with Claude...")
        analysis_message = claude_client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=4000,
            temperature=0.2,
            messages=[
                {"role": "user", "content": analysis_prompt}
            ],
            stream=True
        )
    except Exception as e:
        print(f"Error creating analysis message: {str(e)}")
        analysis_message = claude_client.messages.create(
            model="claude-3-opus-20240229", 
            max_tokens=4000,
            temperature=0.2,
            messages=[
                {"role": "user", "content": analysis_prompt}
            ],
            stream=True
        )
    
    # Stream the analysis
    print("\nStreaming Claude's legislative gridlock analysis:\n")
    analysis_result = ""
    
    for chunk in analysis_message:
        if chunk.type == "content_block_delta":
            text = chunk.delta.text
            print(text, end="", flush=True)
            analysis_result += text
            time.sleep(0.01)  # Small delay for readability
    
    # Extract JSON if available
    if "<json>" in analysis_result and "</json>" in analysis_result:
        try:
            json_text = analysis_result.split("<json>")[1].split("</json>")[0]
            json_data = json.loads(json_text)
            
            # Print summary
            print("\n\nAnalysis complete!")
            
            if json_data.get("clusters"):
                print(f"\nIdentified {len(json_data['clusters'])} legislative issue clusters")
                
                # Print each cluster name
                for i, cluster in enumerate(json_data["clusters"]):
                    print(f"  {i+1}. {cluster.get('name', 'Unnamed cluster')}")
            
            if json_data.get("continue_exploration"):
                print("\nClaude recommends continuing exploration:")
                print(f"  Reason: {json_data.get('reasoning', 'Not specified')}")
        except Exception as e:
            print(f"\nWarning: Could not parse JSON: {str(e)}")
    
    print("\nLegislative gridlock analysis complete!")

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
    
    # Agent mode
    if agent_mode:
        print("\nRunning in full agent mode - Claude will control the entire research process")
        print("Using Sarah Binder's (1999) legislative gridlock framework")
        try:
            run_claude_agent()
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