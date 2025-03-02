"""
Gridlock Score Calculator with AI Agent

This script uses Claude to identify bills related to policy clusters found in NYT analysis,
then calculates a gridlock score based on which policies have any enacted legislation.
"""

import os
import json
import time
from typing import List, Dict, Any
import anthropic
from pinecone import Pinecone
from openai import OpenAI
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Set up API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# Initialize clients
anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
openai_client = OpenAI(api_key=openai_api_key)
pc = Pinecone(api_key=pinecone_api_key)

# Connect to Pinecone index
index_name = 'bill-summaries'
index = pc.Index(index_name)

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
        print("Warning: bill_status_dict.json not found. Creating empty dictionary.")
        return {}

# Get policy clusters from NYT analysis
def get_nyt_clusters(year="2024"):
    """Get policy clusters from NYT analysis using Claude"""
    print(f"Getting policy clusters for {year} from NYT analysis...")
    
    system_prompt = f"""You are a legislative policy expert analyzing media coverage of Congress for {year}.
    Your task is to identify distinct policy clusters from NYT articles about legislative activity in Congress.
    You should focus on major policy areas where Congress has been active or where there has been significant debate.
    For each policy cluster, provide a name, summary, and list of key articles.
    """
    
    user_prompt = f"""Please analyze recent New York Times coverage of Congressional legislative activity for {year}
    and identify 8-12 distinct policy clusters or areas where Congress has been working on legislation.
    
    For each policy cluster:
    1. Provide a short name (3-6 words max)
    2. Write a 2-3 sentence summary of the legislative issues in this area
    3. Generate a sample query to find relevant bills in this policy area (this should be a concise query for vector search)
    
    Format your response as a JSON array where each element is an object with these fields:
    - "name": Short name of the policy area
    - "summary": Brief description of legislative issues
    - "query": Search query to find relevant bills
    
    Return ONLY the JSON array without explanation, surrounded by triple backticks.
    Example:
    ```
    [
      {
        "name": "Immigration Reform",
        "summary": "Legislation addressing border security, asylum processes, and immigration system reforms.",
        "query": "immigration border security asylum reform pathways citizenship"
      }
    ]
    ```
    """
    
    # Call Claude to get policy clusters
    try:
        response = anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1500,
            temperature=0.3,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        
        # Extract JSON from response
        content = response.content[0].text
        json_match = content.split("```")[1] if "```" in content else content
        json_match = json_match.strip()
        if json_match.startswith("json"):
            json_match = json_match[4:].strip()
            
        clusters = json.loads(json_match)
        print(f"Identified {len(clusters)} policy clusters.")
        return clusters
    
    except Exception as e:
        print(f"Error getting policy clusters: {str(e)}")
        # Fallback to default clusters if API call fails
        return [
            {
                "name": "Immigration Reform",
                "summary": "Legislation addressing border security, asylum processes, and immigration system reforms.",
                "query": "immigration border security asylum reform pathways citizenship"
            },
            {
                "name": "Debt Ceiling and Spending",
                "summary": "Bills related to government funding, debt ceiling increases, and federal budget negotiations.",
                "query": "debt ceiling budget spending appropriations government shutdown"
            },
            {
                "name": "Healthcare Policy",
                "summary": "Legislation on healthcare access, prescription drug pricing, and insurance reforms.",
                "query": "healthcare prescription drugs medicare medicaid affordable care act"
            }
        ]

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
    5. IMPORTANT: Format bill IDs as: "118-<bill_type>-<bill_number>" (e.g., "118-hr-1234", "118-s-789")
    """
    
    user_prompt = f"""Please identify bills from the {congress}th Congress that are related to this policy area:
    
    POLICY NAME: {policy['name']}
    POLICY SUMMARY: {policy['summary']}
    SEARCH QUERY: {policy['query']}
    
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
            input=[policy['query']]
        )
        embedding = embed_response.data[0].embedding
        
        # Query Pinecone
        query_response = index.query(
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

# Main function
def main():
    """Main function to analyze gridlock"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Calculate gridlock score')
    parser.add_argument('--congress', type=str, default="118", help='Congress number to analyze')
    parser.add_argument('--year', type=str, default="2023", help='Year to analyze (for NYT clusters)')
    args = parser.parse_args()
    
    congress = args.congress
    year = args.year
    
    print("Gridlock Score Calculator")
    print("========================\n")
    print(f"Analyzing legislative gridlock for the {congress}th Congress")
    print(f"Using policy clusters from NYT {year} coverage\n")
    
    # Load bill status data
    bill_status = load_bill_status()
    if not bill_status:
        print("Error: No bill status data available. Please make sure bill_status_dict.json exists.")
        return
    
    print(f"Loaded status data for {len(bill_status)} bills\n")
    
    # Get policy clusters
    policy_clusters = get_nyt_clusters(year=year)
    
    # Process each policy cluster
    policy_results = []
    
    for i, policy in enumerate(policy_clusters):
        print(f"\nProcessing policy {i+1}/{len(policy_clusters)}: {policy['name']}")
        print(f"Summary: {policy['summary']}")
        
        # Identify related bills
        bill_ids = identify_related_bills(policy, congress)
        
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
    
    # Calculate gridlock score
    gridlock = calculate_gridlock(policy_results)
    
    # Print final results
    print("\n=== FINAL RESULTS ===")
    print(f"Total policy areas analyzed: {gridlock['total_policies']}")
    print(f"Policy areas with enacted legislation: {gridlock['enacted_policies']}")
    print(f"Gridlock rate: {gridlock['gridlock_rate']:.2f}")
    
    # Write results to file
    with open('gridlock_results.txt', 'w', encoding='utf-8') as f:
        f.write("GRIDLOCK ANALYSIS RESULTS\n")
        f.write("========================\n\n")
        f.write(f"Congress: {congress}\n")
        f.write(f"Policies analyzed: {gridlock['total_policies']}\n")
        f.write(f"Policies with enacted legislation: {gridlock['enacted_policies']}\n")
        f.write(f"Gridlock rate: {gridlock['gridlock_rate']:.2f}\n\n")
        
        f.write("POLICY DETAILS\n")
        f.write("=============\n\n")
        
        for i, result in enumerate(policy_results):
            f.write(f"{i+1}. {result['policy']['name']}\n")
            f.write(f"   Summary: {result['policy']['summary']}\n")
            f.write(f"   Related bills: {result['total_bills']}\n")
            f.write(f"   Enacted bills: {result['enacted_count']}\n")
            
            if result["enacted_count"] > 0:
                f.write(f"   Enacted bill IDs: {', '.join(result['enacted_bills'])}\n")
            
            if result["bill_ids"]:
                f.write(f"   All related bill IDs: {', '.join(result['bill_ids'])}\n")
            
            f.write("\n")
    
    print(f"\nDetailed results written to gridlock_results.txt")

if __name__ == "__main__":
    main()