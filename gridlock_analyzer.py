"""
Gridlock Analyzer

This script analyzes legislative gridlock by:
1. Finding policy clusters from NYT articles for a specific Congress
2. Identifying bills related to those clusters
3. Checking which bills were enacted
4. Calculating a gridlock score

It can either generate new clusters or use cached clusters from a file.
"""

import os
import json
import time
import argparse
from typing import List, Dict, Any, Optional
import anthropic
from pinecone import Pinecone
from openai import OpenAI
import dotenv

# Import our NYT clusters module
import nyt_clusters

# Load environment variables
dotenv.load_dotenv()

# API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# Initialize clients
anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
openai_client = OpenAI(api_key=openai_api_key)
pc = Pinecone(api_key=pinecone_api_key)

# Connect to Pinecone bills index
bill_index_name = 'bill-summaries'
bill_index = pc.Index(bill_index_name)

# Load bill status data
def load_bill_status():
    """Load bill enactment status from JSON file"""
    try:
        with open('bill_status_dict.json', 'r') as f:
            # File is already in the right format (bill_id: enacted_status)
            bill_status_data = json.load(f)
            print(f"Successfully loaded bill status data: {len(bill_status_data)} bills")
            return bill_status_data
    except FileNotFoundError:
        print("Error: bill_status_dict.json not found. Cannot proceed.")
        return None

# Identify bills related to a policy using Claude
def identify_related_bills(policy, congress="118", limit=20):
    """Use Claude to identify bills related to a policy area"""
    print(f"Identifying bills related to: {policy['name']}")
    
    system_prompt = f"""You are a congressional research expert who specializes in identifying relevant legislation.
    You are working with a database of bills from the {congress}th Congress.
    
    Your task is to identify bills that are closely related to a specific policy area.
    You will search for bills using the bill_summaries database and return the most relevant bill IDs.
    
    Guidelines:
    1. Focus only on bills that are clearly and directly related to the policy area
    2. Aim for precision over recall - only include bills that are definitely related
    3. Return only bill IDs in the format they appear in the database
    4. Do not make up bill IDs
    5. IMPORTANT: Format bill IDs as: "{congress}-<bill_type>-<bill_number>" (e.g., "{congress}-hr-1234", "{congress}-s-789")
    """
    
    # Extract query from policy or use default
    query = policy.get("query", "")
    if not query and "articles" in policy:
        # Fallback: Create a query from the first few article headlines
        articles = policy.get("articles", [])[:3]
        query = " ".join(articles)
    
    # If no query or articles, use the name and summary
    if not query:
        name = policy.get("name", "")
        summary = policy.get("summary", "")
        query = f"{name} {summary}"
    
    user_prompt = f"""Please identify bills from the {congress}th Congress that are related to this policy area:
    
    POLICY NAME: {policy.get('name', 'Unnamed policy')}
    POLICY SUMMARY: {policy.get('summary', 'No summary provided')}
    SEARCH QUERY: {query}
    
    First, I will search for relevant bills using this query. Then analyze the results to determine which bills are genuinely related to this policy area.
    
    For each bill you identify as related:
    1. Explain briefly why it's related to the policy area
    2. Provide the bill ID in the exact format: "{congress}-<bill_type>-<bill_number>" (e.g., "{congress}-hr-1234", "{congress}-s-789")
    
    Format your response with a final "BILL_IDS:" section that lists only the IDs of related bills, separated by commas.
    For example:
    
    BILL_IDS: {congress}-hr-1234, {congress}-s-789, {congress}-hjres-45
    """
    
    # Search for bills using vector search
    try:
        # Generate embedding for query
        embed_response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=[query]
        )
        embedding = embed_response.data[0].embedding
        
        # Query Pinecone
        query_response = bill_index.query(
            vector=embedding,
            top_k=50,  # Get a good number of candidates
            include_metadata=True,
            filter={"congress": congress}
        )
        
        # Format bill data for Claude
        bill_data = []
        for match in query_response.matches:
            # Transform bill ID to match the format in bill_status_dict.json
            bill_type = match.metadata.get('bill_type', '').lower()
            bill_number = match.metadata.get('bill_number', '')
            bill_id_format = f"{congress}-{bill_type}-{bill_number}"
            
            bill_info = (
                f"ID: {bill_id_format}\n"
                f"Score: {match.score}\n"
                f"Title: {match.metadata.get('title', 'N/A')}\n"
                f"Bill Type: {match.metadata.get('bill_type', 'N/A')}\n"
                f"Bill Number: {match.metadata.get('bill_number', 'N/A')}\n"
                f"Summary: {match.metadata.get('summary_text', 'N/A')[:300]}...\n"
                f"----------------------\n"
            )
            bill_data.append(bill_info)
        
        # Combine bill data
        bill_context = "\n".join(bill_data)
        
        # Call Claude with bill data
        claude_prompt = f"{user_prompt}\n\nHere are the search results:\n\n{bill_context}"
        
        response = anthropic_client.messages.create(
            model="claude-3-haiku-20240307",  # Using Haiku to save tokens
            max_tokens=1000,
            temperature=0.2,
            system=system_prompt,
            messages=[
                {"role": "user", "content": claude_prompt}
            ]
        )
        
        # Extract bill IDs from response
        content = response.content[0].text
        
        # Look for the BILL_IDS: section
        if "BILL_IDS:" in content:
            bill_ids_section = content.split("BILL_IDS:")[1].strip()
            # If there are multiple lines, get just the first line with IDs
            bill_ids_section = bill_ids_section.split("\n")[0].strip()
            # Split by commas and clean up whitespace
            bill_ids = [bid.strip() for bid in bill_ids_section.split(",")]
            # Filter out any empty strings
            bill_ids = [bid for bid in bill_ids if bid]
            
            print(f"  Found {len(bill_ids)} related bills")
            return bill_ids
        else:
            print("  No bills explicitly identified in the response")
            return []
    
    except Exception as e:
        print(f"  Error identifying related bills: {str(e)}")
        return []

# Check bill enactment status
def check_enactment_status(bill_ids, bill_status):
    """Check which bills are enacted based on bill_status data"""
    total_bills = len(bill_ids)
    enacted_bills = []
    
    for bill_id in bill_ids:
        enacted = bill_status.get(bill_id, None)
        if enacted == 1:
            enacted_bills.append(bill_id)
    
    return {
        "total_bills": total_bills,
        "enacted_count": len(enacted_bills),
        "enacted_bills": enacted_bills
    }

# Calculate gridlock score
def calculate_gridlock(policy_results):
    """Calculate gridlock score based on policy results"""
    total_policies = len(policy_results)
    enacted_policies = sum(1 for p in policy_results if p["enacted_count"] > 0)
    
    return {
        "total_policies": total_policies,
        "enacted_policies": enacted_policies,
        "gridlock_rate": 1 - (enacted_policies / total_policies if total_policies > 0 else 0)
    }

# Analyze gridlock for a specific Congress
def analyze_gridlock(congress_num, use_cached_clusters=False, clusters_file=None, save_clusters=True, 
                     top_k=300, threshold=0.35, limit=100, model=nyt_clusters.DEFAULT_CLAUDE_MODEL):
    """Analyze gridlock for a specific Congress"""
    print(f"Analyzing gridlock for the {congress_num}th Congress")
    print(f"Years: {', '.join(nyt_clusters.CONGRESS_TO_YEARS.get(str(congress_num), []))}")
    
    # Load bill status data
    bill_status = load_bill_status()
    if not bill_status:
        print("Failed to load bill status data. Cannot continue.")
        return None
    
    # Step 1: Get policy clusters
    clusters_data = None
    
    if use_cached_clusters and clusters_file and os.path.exists(clusters_file):
        # Load clusters from file
        print(f"Loading cached clusters from {clusters_file}")
        clusters_data = nyt_clusters.load_clusters_from_file(clusters_file)
    else:
        # Generate new clusters - using all years for this congressional session
        print(f"Generating new policy clusters for the {congress_num}th Congress...")
        clusters_data = nyt_clusters.get_policy_clusters(
            congress_num=congress_num,
            top_k=top_k,
            threshold=threshold,
            limit=limit,
            model=model
        )
        
        # Save clusters if requested
        if save_clusters and clusters_data:
            output_file = clusters_file or f"clusters_congress_{congress_num}.json"
            nyt_clusters.save_clusters_to_file(clusters_data, output_file)
    
    if not clusters_data:
        print("Failed to obtain policy clusters. Cannot continue.")
        return None
    
    # Step 2: Process each policy cluster
    print(f"\nFound {len(clusters_data.get('clusters', []))} policy clusters to analyze")
    
    policy_results = []
    
    for i, policy in enumerate(clusters_data.get("clusters", [])):
        print(f"\nProcessing policy {i+1}/{len(clusters_data.get('clusters', []))}: {policy.get('name', 'Unnamed policy')}")
        
        # Identify related bills
        bill_ids = identify_related_bills(policy, congress=congress_num)
        
        # Check enactment status
        status_results = check_enactment_status(bill_ids, bill_status)
        
        # Store results
        policy_result = {
            "policy": policy,
            "bill_ids": bill_ids,
            "total_bills": status_results["total_bills"],
            "enacted_count": status_results["enacted_count"],
            "enacted_bills": status_results["enacted_bills"]
        }
        
        policy_results.append(policy_result)
        
        # Print results for this policy
        print(f"  Related bills: {status_results['total_bills']}")
        print(f"  Enacted bills: {status_results['enacted_count']}")
        if status_results["enacted_count"] > 0:
            print(f"  Enacted bill IDs: {', '.join(status_results['enacted_bills'])}")
        
        # Small delay to avoid rate limits
        time.sleep(1)
    
    # Step 3: Calculate gridlock score
    gridlock = calculate_gridlock(policy_results)
    
    # Step 4: Prepare results
    results = {
        "congress": congress_num,
        "years": nyt_clusters.CONGRESS_TO_YEARS.get(str(congress_num), []),
        "total_policies": gridlock["total_policies"],
        "enacted_policies": gridlock["enacted_policies"],
        "gridlock_rate": gridlock["gridlock_rate"],
        "policy_results": policy_results
    }
    
    # Print final results
    print("\n=== FINAL RESULTS ===")
    print(f"Congress: {congress_num}")
    print(f"Years: {', '.join(results['years'])}")
    print(f"Total policy areas analyzed: {gridlock['total_policies']}")
    print(f"Policy areas with enacted legislation: {gridlock['enacted_policies']}")
    print(f"Gridlock rate: {gridlock['gridlock_rate']:.2f}")
    
    # Save results to file
    results_file = f"gridlock_results_congress_{congress_num}.json"
    try:
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved detailed results to {results_file}")
    except Exception as e:
        print(f"Error saving results: {str(e)}")
    
    # Also save a more readable text summary
    summary_file = f"gridlock_summary_congress_{congress_num}.txt"
    try:
        with open(summary_file, "w") as f:
            f.write(f"GRIDLOCK ANALYSIS: {congress_num}th CONGRESS\n")
            f.write("======================================\n\n")
            f.write(f"Years: {', '.join(results['years'])}\n")
            f.write(f"Total policy areas analyzed: {gridlock['total_policies']}\n")
            f.write(f"Policy areas with enacted legislation: {gridlock['enacted_policies']}\n")
            f.write(f"Gridlock rate: {gridlock['gridlock_rate']:.2f}\n\n")
            
            f.write("POLICY DETAILS\n")
            f.write("=============\n\n")
            
            for i, result in enumerate(policy_results):
                policy = result["policy"]
                f.write(f"{i+1}. {policy.get('name', 'Unnamed policy')}\n")
                f.write(f"   Summary: {policy.get('summary', 'No summary provided')}\n")
                f.write(f"   Related bills: {result['total_bills']}\n")
                f.write(f"   Enacted bills: {result['enacted_count']}\n")
                
                if result["enacted_count"] > 0:
                    f.write(f"   Enacted bill IDs: {', '.join(result['enacted_bills'])}\n")
                
                f.write("\n")
        
        print(f"Saved summary to {summary_file}")
    except Exception as e:
        print(f"Error saving summary: {str(e)}")
    
    return results

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="Analyze legislative gridlock for a specific Congress")
    
    # Required arguments
    parser.add_argument("--congress", type=str, required=True, help="Congressional number (e.g., 118)")
    
    # Cluster source options
    parser.add_argument("--cached", action="store_true", help="Use cached clusters if available")
    parser.add_argument("--clusters-file", type=str, help="Path to cached clusters file (used with --cached)")
    parser.add_argument("--no-save", action="store_true", help="Don't save generated clusters to file")
    
    # Article retrieval parameters
    parser.add_argument("--top-k", type=int, default=300, help="Maximum number of vectors to retrieve from Pinecone")
    parser.add_argument("--threshold", type=float, default=0.35, help="Similarity threshold for matching articles")
    parser.add_argument("--limit", type=int, default=100, help="Maximum number of articles to analyze")
    
    # Claude options
    parser.add_argument("--model", type=str, default=nyt_clusters.DEFAULT_CLAUDE_MODEL, 
                        help="Claude model to use for analysis")
    
    args = parser.parse_args()
    
    # Analyze gridlock
    analyze_gridlock(
        congress_num=args.congress,
        use_cached_clusters=args.cached,
        clusters_file=args.clusters_file,
        save_clusters=not args.no_save,
        top_k=args.top_k,
        threshold=args.threshold,
        limit=args.limit,
        model=args.model
    )

if __name__ == "__main__":
    main()