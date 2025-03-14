"""
NYT Clusters Module

This module extracts the main functionality from nyt.py to find policy clusters
based on New York Times articles. It can be used as a standalone tool or
imported into other gridlock computation scripts.

Congressional number to year mapping:
113 - 2013, 2014
114 - 2015, 2016
115 - 2017, 2018
116 - 2019, 2020
117 - 2021, 2022
118 - 2023, 2024
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, Union
import anthropic
from pinecone import Pinecone
from openai import OpenAI
import dotenv

# Load environment variables
dotenv.load_dotenv()

# API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# Initialize clients
anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
try:
    openai_client = OpenAI(api_key=openai_api_key)
except Exception as e:
    print(f"Warning: OpenAI client initialization error - {str(e)}")

try:
    pc = Pinecone(api_key=pinecone_api_key)
    index_name = 'nyt-archive'
    index = pc.Index(index_name)
except Exception as e:
    print(f"Warning: Pinecone client initialization error - {str(e)}")

# Congressional number to year mapping
CONGRESS_TO_YEARS = {
    "113": ["2013", "2014"],
    "114": ["2015", "2016"],
    "115": ["2017", "2018"],
    "116": ["2019", "2020"],
    "117": ["2021", "2022"],
    "118": ["2023", "2024"]
}

# Year to congressional number mapping
YEAR_TO_CONGRESS = {
    "2013": "113", "2014": "113",
    "2015": "114", "2016": "114",
    "2017": "115", "2018": "115",
    "2019": "116", "2020": "116",
    "2021": "117", "2022": "117",
    "2023": "118", "2024": "118"
}

# Claude model to use
DEFAULT_CLAUDE_MODEL = "claude-3-7-sonnet-20250219"  # Latest Claude model

def create_embedding(text):
    """Create an embedding vector for a text using OpenAI's API"""
    try:
        response = openai_client.embeddings.create(
            input=[text],
            model='text-embedding-3-small'
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error creating embedding: {str(e)}")
        return None

def query_pinecone(query_text, namespace=None, top_k=300, threshold=0.35, filter_dict=None):
    """Query Pinecone index with a text query and return matches above threshold"""
    # Generate embedding for query
    query_embedding = create_embedding(query_text)
    
    if not query_embedding:
        return {"error": "Failed to create embedding", "matches": [], "total_matches": 0}
    
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
    try:
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
    except Exception as e:
        print(f"Error querying Pinecone: {str(e)}")
        return {"error": str(e), "matches": [], "total_matches": 0}

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

def get_namespaces():
    """Get all namespaces in the index and their vector counts"""
    try:
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
    except Exception as e:
        print(f"Error getting namespaces: {str(e)}")
        return []

def get_year_namespaces(year):
    """Get all namespaces for a specific year"""
    all_namespaces = get_namespaces()
    return [ns for ns, _ in all_namespaces if ns.startswith(f"{year}_")]

def get_articles_for_congress(congress_num, query_text="news articles about policy issues", threshold=0.35, top_k=300, limit=100):
    """Get articles from all namespaces for a specific congressional number"""
    years = CONGRESS_TO_YEARS.get(str(congress_num))
    if not years:
        print(f"Invalid congressional number: {congress_num}")
        return {
            "articles": [],
            "total_found": 0,
            "query": query_text,
            "congress": congress_num,
            "error": f"Invalid congressional number: {congress_num}"
        }
    
    print(f"Querying for: {query_text}")
    print(f"Congressional number: {congress_num} (years: {', '.join(years)})")
    
    all_results = []
    queried_namespaces = []
    
    # Get all available namespaces
    all_namespaces = get_namespaces()
    
    # Get all namespaces for all years in this congressional term
    for year in years:
        print(f"Searching for articles from {year}...")
        year_namespaces = [ns for ns, _ in all_namespaces if ns.startswith(f"{year}_")]
        
        if year_namespaces:
            print(f"Found {len(year_namespaces)} namespaces for {year}")
            # Query each namespace for this year
            for ns in year_namespaces:
                print(f"  - Querying namespace: {ns}")
                results = query_pinecone(query_text, namespace=ns, top_k=top_k, threshold=threshold)
                if "error" not in results:
                    all_results.extend(results["matches"])
                    queried_namespaces.append(ns)
        else:
            print(f"No namespaces found for year {year}")
    
    if not all_results:
        print(f"No articles found for {congress_num}th Congress")
        return {
            "articles": [],
            "total_found": 0,
            "query": query_text,
            "congress": congress_num,
            "error": f"No articles found for {congress_num}th Congress"
        }
    
    # Sort by score and get a substantial number of results for comprehensive analysis
    all_results.sort(key=lambda x: x["score"], reverse=True)
    limited_results = all_results[:min(len(all_results), limit)]
    
    print(f"Found {len(all_results)} total matching articles across {len(queried_namespaces)} namespaces")
    print(f"Using top {len(limited_results)} results for analysis")
    
    # Extract article data for analysis
    article_data = extract_article_data(limited_results)
    
    return {
        "articles": article_data,
        "total_found": len(limited_results),
        "query": query_text,
        "congress": congress_num,
        "years": years,
        "namespaces_queried": queried_namespaces,
        "threshold": threshold
    }

def get_articles_for_query(query_text, year=None, namespace=None, threshold=0.35, top_k=300, limit=200):
    """Get articles for a specific query, year, or namespace"""
    print(f"Querying for: {query_text}")
    all_results = []
    queried_namespaces = []
    
    # Get all available namespaces
    all_namespaces = get_namespaces()
    
    # Case 1: Specific namespace provided
    if namespace:
        print(f"Searching in specific namespace: {namespace}")
        results = query_pinecone(query_text, namespace=namespace, top_k=top_k, threshold=threshold)
        if "error" not in results:
            all_results.extend(results["matches"])
            queried_namespaces.append(namespace)
    
    # Case 2: Year specified but no specific namespace
    elif year:
        print(f"Searching in all namespaces for year: {year}")
        year_namespaces = get_year_namespaces(year)
        
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
            results = query_pinecone(query_text, namespace=ns, top_k=top_k, threshold=threshold)
            if "error" not in results:
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
                results = query_pinecone(query_text, namespace=ns, top_k=top_k, threshold=threshold)
                if "error" not in results:
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
        "threshold": threshold,
        "year": year
    }

def analyze_articles_claude(articles_data, year=None, congress=None, model=DEFAULT_CLAUDE_MODEL, stream=True):
    """Analyze articles with Claude to identify policy clusters"""
    # Determine time period (year or congress)
    if congress and not year:
        years = CONGRESS_TO_YEARS.get(str(congress), [])
        if years:
            years_str = ", ".join(years)
            congress_period = f"{congress}th Congress ({years_str})"
        else:
            congress_period = f"{congress}th Congress"
    elif not year and articles_data.get('congress'):
        congress = articles_data.get('congress')
        years = CONGRESS_TO_YEARS.get(str(congress), [])
        if years:
            years_str = ", ".join(years)
            congress_period = f"{congress}th Congress ({years_str})"
        else:
            congress_period = f"{congress}th Congress"
    elif not year and articles_data.get('year'):
        year = articles_data.get('year')
        congress_period = f"{year}"
    elif not year:
        # Find most common year in pub_dates
        years = {}
        for article in articles_data.get('articles', []):
            pub_date = article.get('pub_date', '')
            if pub_date and len(pub_date) >= 4:
                year_str = pub_date[:4]
                years[year_str] = years.get(year_str, 0) + 1
        
        if years:
            year = max(years.items(), key=lambda x: x[1])[0]
            congress_period = f"{year}"
        else:
            year = "2024"  # Default
            congress_period = f"{year}"
    else:
        congress_period = f"{year}"
    
    # Define function schemas for function calling
    function_schema = None  # Don't use function calling - it interferes with streaming the reasoning
                           # We'll extract the JSON from the streamed output instead
    
    # Prepare system prompt with explicit thinking instructions
    system_prompt = f"""

NOTE: Please consider only the news articles that cover events in the U.S. Congress during the time period of {congress_period}.

REMEMBER: You must EXHAUSTIVELY identify and analyze ALL legislative issue clusters for the {congress_period}. Focus on this time period ONLY.

USE "THINKING": You MUST show your detailed thought process by using <thinking></thinking> tags. This helps users understand your analysis process in real-time as you consider different ways to cluster articles.

For each cluster:
1. Give the cluster a descriptive name (5-7 words capturing the policy area)
2. Provide a list of key articles in the cluster (proper size is critical)
3. Write a 3-5 sentence summary of the legislative issues/debates represented
4. Explain how these issues relate to the formal legislative process or gridlock in Congress
5. Include an "article_count" field with the number of articles in this cluster
6. Add a "query" field with a search string that can be used to find related bills

When analyzing, pay special attention to:
- Specific bills being debated or passed in Congress
- Committee activities and hearings
- Partisan dynamics affecting legislation 
- Key policy stakeholders (interest groups, agencies)
- Legislative procedures being used or debated
- Signs of gridlock or progress for each issue

MOST IMPORTANT: After completing your analysis, you MUST provide a final JSON output enclosed in <json></json> tags that follows the specified structure.
"""
    
    # Prepare user message
    user_message = f"""I need you to analyze these New York Times articles about legislative issues in Congress using Sarah Binder's (1999) approach to studying legislative gridlock.

TIME PERIOD: {congress_period}
QUERY: {articles_data.get('query', 'legislative issues')}
TOTAL ARTICLES: {len(articles_data.get('articles', []))}

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
</thinking>

Please identify the main clusters of legislative issues and analyze them according to Binder's approach to studying gridlock through media coverage.

Here are the articles:
{json.dumps(articles_data['articles'], indent=2)}

CRITICAL INSTRUCTION: After completing your analysis, you MUST end your response with JSON data enclosed in <json></json> tags. The JSON must follow exactly this structure:

{{
  "clusters": [
    {{
      "name": "Government Funding and Shutdown Politics",
      "articles": ["Congress Returns for Another Big Spending Fight", "Debt Deal Behind Them, Lawmakers Plunge Into Bitter Spending Fight"],
      "article_count": 15,
      "summary": "These articles highlight the intense debates over federal spending and shutdown threats.",
      "legislative_process": "This represents gridlock in Congress's core constitutional function of appropriations.",
      "query": "government funding shutdown appropriations budget continuing resolution"
    }},
    ... additional clusters ...
  ]
}}

You MUST include this final JSON structure after your analysis. Without it, your response will be considered incomplete.
"""
    
    # Create the message
    try:
        print(f"Analyzing articles with {model}...")
        
        # Always use streaming to capture the full reasoning process
        print("Streaming Claude's analysis in real-time:\n")
        analysis_result = ""
        
        message = anthropic_client.messages.create(
            model=model,
            max_tokens=4000,
            temperature=0.2,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_message}
            ],
            stream=True
        )
        
        # Save the full streaming output to a file in real-time
        full_output_filename = f"cluster_analysis_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        with open(full_output_filename, 'w', encoding='utf-8') as f:
            f.write(f"CLAUDE ANALYSIS: {congress_period}\n")
            f.write(f"MODEL: {model}\n")
            f.write(f"TIME: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # Collect the streaming output
            for chunk in message:
                if chunk.type == "content_block_delta":
                    text = chunk.delta.text
                    print(text, end="", flush=True)
                    f.write(text)
                    f.flush()  # Ensure it's written immediately
                    analysis_result += text
                    time.sleep(0.01)  # Small delay to make streaming visible
        
        print(f"\n\nFull analysis saved to {full_output_filename}")
        
        # Extract JSON data from the streaming response
        if "<json>" in analysis_result and "</json>" in analysis_result:
            try:
                json_text = analysis_result.split("<json>")[1].split("</json>")[0]
                json_data = json.loads(json_text)
                return json_data
            except Exception as e:
                print(f"Error parsing JSON from response: {str(e)}")
                print("Will try to extract again with a full response...")
        
        # If streaming didn't capture a valid JSON, try one more time with non-streaming
        if "<json>" not in analysis_result or "</json>" not in analysis_result:
            print("\nRe-fetching complete response to extract JSON data...")
            # Use a more forceful prompt that only asks for the JSON
            direct_json_prompt = """You previously analyzed articles to identify policy clusters but didn't provide proper JSON output.
            
Based on your previous analysis, please ONLY provide a JSON response surrounded by <json></json> tags with the policy clusters you identified.
Your response should contain NOTHING except the JSON output in the specified format."""
            
            full_message = anthropic_client.messages.create(
                model=model,
                max_tokens=4000,
                temperature=0.2,
                system="You are a helpful AI assistant that provides properly formatted JSON data.",
                messages=[
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": analysis_result[:6000]},  # Send first part of previous analysis
                    {"role": "user", "content": direct_json_prompt}
                ]
            )
            
            content = full_message.content[0].text
            if "<json>" in content and "</json>" in content:
                try:
                    json_text = content.split("<json>")[1].split("</json>")[0]
                    json_data = json.loads(json_text)
                    # Append the full message to the output file as well
                    with open(full_output_filename, 'a', encoding='utf-8') as f:
                        f.write("\n\n=== COMPLETE RESPONSE (NON-STREAMING) ===\n\n")
                        f.write(content)
                    return json_data
                except Exception as e:
                    print(f"Error parsing JSON from full response: {str(e)}")
        
        print("\nNo valid JSON data found in Claude's response.")
        
        # Create fallback JSON from the analysis to avoid complete failure
        try:
            print("Attempting to create fallback JSON from analysis text...")
            
            # Try to extract clusters from the analysis
            import re
            
            # Look for sections that might be clusters
            cluster_sections = re.split(r'##\s+', analysis_result)
            
            if len(cluster_sections) <= 1:
                # Try another pattern if headers aren't markdown style
                cluster_sections = re.split(r'\n\d+\.\s+', analysis_result)
            
            clusters = []
            
            # Skip the first section which is typically introduction
            for section in cluster_sections[1:]:
                # Stop if we have enough clusters
                if len(clusters) >= 8:
                    break
                    
                # Try to extract cluster information
                section = section.strip()
                lines = section.split('\n')
                
                if not lines:
                    continue
                    
                name = lines[0].strip()
                
                # Extract articles - look for bullet points
                articles = []
                for line in lines:
                    if line.strip().startswith('- "') or line.strip().startswith('- \''):
                        # Extract the article title
                        article_match = re.search(r'- ["\']([^"\']+)["\']', line)
                        if article_match:
                            articles.append(article_match.group(1))
                
                # If no articles found with bullets, try to find quotes in the text
                if not articles:
                    article_matches = re.findall(r'["\']([^"\']+)["\']', section)
                    articles = article_matches[:5]  # Take up to 5 matches as articles
                
                # Create a query from the name
                query = name.lower()
                
                # Try to find a summary paragraph
                summary = ""
                for i in range(1, min(10, len(lines))):
                    if len(lines[i]) > 100:  # Look for a long paragraph
                        summary = lines[i]
                        break
                
                # If no summary found, use a default
                if not summary:
                    summary = f"Legislative issues related to {name}."
                
                # Create a basic cluster
                if name and not name.startswith('thinking'):
                    clusters.append({
                        "name": name[:50],  # Truncate long names
                        "articles": articles[:10],  # Limit to 10 articles
                        "article_count": len(articles),
                        "summary": summary[:200],  # Truncate long summaries
                        "legislative_process": f"Congressional activity related to {name}.",
                        "query": query
                    })
            
            # If we found at least one cluster, return them
            if clusters:
                fallback_data = {"clusters": clusters}
                print(f"Created fallback JSON with {len(clusters)} clusters")
                return fallback_data
            
        except Exception as e:
            print(f"Failed to create fallback JSON: {str(e)}")
        
        return None
            
    except Exception as e:
        print(f"Error analyzing articles with Claude: {str(e)}")
        return None

def get_policy_clusters(congress_num=None, year=None, query="legislative process US Congress policy issues", 
                        top_k=300, threshold=0.35, limit=100, model=DEFAULT_CLAUDE_MODEL, stream=True):
    """Get policy clusters for a specific congressional number or year"""
    
    articles_data = None
    
    # If congress number is provided, get articles for all years in that congress
    if congress_num:
        print(f"Getting policy clusters for the {congress_num}th Congress")
        articles_data = get_articles_for_congress(
            congress_num=congress_num,
            query_text=query,
            threshold=threshold,
            top_k=top_k,
            limit=limit
        )
    # If only year is provided, get articles for that year
    elif year:
        print(f"Getting policy clusters for the year {year}")
        # Map year to congressional number for reference
        congress_num = YEAR_TO_CONGRESS.get(str(year))
        if not congress_num:
            print(f"Invalid year: {year}")
            return None
            
        articles_data = get_articles_for_query(
            query_text=query,
            year=year,
            namespace=None,
            threshold=threshold,
            top_k=top_k,
            limit=limit
        )
    else:
        print("Either congress_num or year must be provided")
        return None
    
    if articles_data.get("error") or articles_data.get("total_found", 0) == 0:
        print(f"Error or no articles found: {articles_data.get('error', 'No articles found')}")
        return None
    
    # Analyze articles with Claude
    try:
        print(f"Analyzing {articles_data['total_found']} articles with Claude...")
        clusters = analyze_articles_claude(
            articles_data, 
            year=year, 
            congress=congress_num,
            model=model,
            stream=stream
        )
        
        # Add metadata to the results
        if clusters:
            return {
                "congress": congress_num,
                "year": year if year else None,
                "years": articles_data.get("years", []),
                "query": query,
                "total_articles": articles_data["total_found"],
                "namespaces_queried": articles_data["namespaces_queried"],
                "clusters": clusters.get("clusters", [])
            }
        return None
    except Exception as e:
        print(f"Error analyzing policy clusters: {str(e)}")
        return None

def save_clusters_to_file(clusters_data, output_file):
    """Save policy clusters to a JSON file"""
    try:
        # Save to JSON file
        with open(output_file, "w") as f:
            json.dump(clusters_data, f, indent=2)
        print(f"Saved clusters to {output_file}")
        
        # Also save a human-readable summary
        summary_file = output_file.replace('.json', '_summary.txt')
        with open(summary_file, "w") as f:
            f.write(f"POLICY CLUSTERS SUMMARY\n")
            f.write(f"=====================\n\n")
            if 'congress' in clusters_data:
                f.write(f"Congress: {clusters_data['congress']}\n")
            if 'years' in clusters_data:
                f.write(f"Years: {', '.join(clusters_data['years'])}\n")
            f.write(f"Total Articles Analyzed: {clusters_data.get('total_articles', 'N/A')}\n")
            f.write(f"Total Clusters Identified: {len(clusters_data.get('clusters', []))}\n\n")
            
            f.write("CLUSTERS:\n")
            for i, cluster in enumerate(clusters_data.get('clusters', [])):
                f.write(f"\n{i+1}. {cluster.get('name', 'Unnamed Cluster')}\n")
                f.write(f"   Articles: {cluster.get('article_count', 0)}\n")
                f.write(f"   Summary: {cluster.get('summary', '')}\n")
                f.write(f"   Legislative Process: {cluster.get('legislative_process', '')}\n")
                f.write(f"   Query: {cluster.get('query', '')}\n")
        
        print(f"Saved summary to {summary_file}")
        return True
    except Exception as e:
        print(f"Error saving clusters to file: {str(e)}")
        return False

def load_clusters_from_file(input_file):
    """Load policy clusters from a JSON file"""
    try:
        with open(input_file, "r") as f:
            clusters_data = json.load(f)
        print(f"Loaded clusters from {input_file}")
        return clusters_data
    except Exception as e:
        print(f"Error loading clusters from file: {str(e)}")
        return None

def main():
    """Command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Get policy clusters from NYT articles")
    parser.add_argument("--congress", type=str, help="Congressional number (e.g., 117)")
    parser.add_argument("--year", type=str, help="Year to analyze (e.g., 2022)")
    parser.add_argument("--query", type=str, default="legislative process US Congress policy issues", 
                        help="Query to search for articles")
    parser.add_argument("--top-k", type=int, default=300, help="Maximum number of vectors to retrieve from Pinecone")
    parser.add_argument("--threshold", type=float, default=0.35, help="Similarity threshold for matching articles")
    parser.add_argument("--limit", type=int, default=200, help="Maximum number of articles to analyze")
    parser.add_argument("--output", type=str, help="Output JSON file to save clusters")
    parser.add_argument("--model", type=str, default=DEFAULT_CLAUDE_MODEL, help="Claude model to use for analysis")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming of Claude's analysis")
    
    args = parser.parse_args()
    
    if not args.congress and not args.year:
        parser.error("Either --congress or --year must be specified")
    
    # Generate output file name if not specified
    output_file = args.output
    if not output_file:
        if args.congress:
            output_file = f"clusters_congress_{args.congress}.json"
        elif args.year:
            output_file = f"clusters_year_{args.year}.json"
    
    # Get policy clusters
    clusters = get_policy_clusters(
        congress_num=args.congress,
        year=args.year,
        query=args.query,
        top_k=args.top_k,
        threshold=args.threshold,
        limit=args.limit,
        model=args.model,
        stream=not args.no_stream
    )
    
    if clusters:
        print(f"\nIdentified {len(clusters.get('clusters', []))} policy clusters")
        
        # Display clusters
        for i, cluster in enumerate(clusters.get("clusters", [])):
            print(f"\n{i+1}. {cluster.get('name', 'Unnamed Cluster')}")
            print(f"   Articles: {cluster.get('article_count', 0)}")
            print(f"   Summary: {cluster.get('summary', '')[:150]}...")
            print(f"   Query: {cluster.get('query', '')}")
        
        # Save to file
        if output_file:
            save_clusters_to_file(clusters, output_file)
            print(f"\nSaved clusters to {output_file}")
    else:
        print("Failed to identify policy clusters")

if __name__ == "__main__":
    main()