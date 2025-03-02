"""
Gridlock Score Calculator - Direct Version

This script uses Claude to identify bills related to specific policy clusters,
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

# Predefined policy clusters from the original analysis
POLICY_CLUSTERS = [
    {
        "name": "Debt Ceiling and Government Shutdowns",
        "summary": "These articles highlight the intense debates over federal spending, the debt ceiling, and the risk of government shutdowns. The Republican-controlled House pushed for deep spending cuts and policy riders on appropriations bills.",
        "query": "debt ceiling budget spending appropriations government shutdown"
    },
    {
        "name": "Immigration Policy and Border Security",
        "summary": "These articles address the contentious debates over U.S. immigration policy, including asylum procedures, border security, and migrant labor issues.",
        "query": "immigration border security asylum reform pathways citizenship"
    },
    {
        "name": "Regulation of Artificial Intelligence",
        "summary": "These articles highlight the growing concern among lawmakers about the rapid advancement of artificial intelligence and the need for regulation.",
        "query": "artificial intelligence AI regulation technology algorithm framework"
    },
    {
        "name": "Abortion Rights and Legislation",
        "summary": "These articles discuss the legislative battles over abortion rights following the Supreme Court's decision to overturn Roe v. Wade.",
        "query": "abortion reproductive rights women's health protection access restrictions"
    },
    {
        "name": "Defense Policy and Military Funding",
        "summary": "These articles focus on the National Defense Authorization Act (NDAA) and related defense spending bills.",
        "query": "defense military NDAA armed forces pentagon security appropriations"
    },
    {
        "name": "Investigations and Oversight",
        "summary": "These articles cover the investigative efforts by House Republicans into the Biden administration, including potential impeachment inquiries.",
        "query": "oversight investigation impeachment hearing subpoena committee accountability"
    },
    {
        "name": "Voting Rights and Election Laws",
        "summary": "These articles examine efforts to change voting laws and redistricting ahead of the 2024 elections.",
        "query": "voting election rights ballot access redistricting gerrymandering"
    },
    {
        "name": "Transgender Rights and Legislation",
        "summary": "These articles focus on legislation at both federal and state levels targeting transgender individuals, particularly youth.",
        "query": "transgender gender identity sports healthcare medical treatment youth"
    },
    {
        "name": "Supreme Court Ethics and Reforms",
        "summary": "These articles discuss efforts by lawmakers to impose an ethics code on the Supreme Court following reports of potential conflicts of interest.",
        "query": "supreme court ethics recusal transparency judicial reform"
    },
    {
        "name": "Gun Control Legislation",
        "summary": "These articles address the challenges of passing gun control legislation in Congress despite public support for certain measures.",
        "query": "gun control firearm safety background check regulation violence"
    },
    {
        "name": "Health Care Legislation",
        "summary": "These articles highlight legislative actions related to health care, including mental health, pandemic preparedness, Medicare and Medicaid benefits.",
        "query": "healthcare prescription drugs medicare medicaid insurance affordable care"
    },
    {
        "name": "Surveillance Laws and Privacy",
        "summary": "These articles focus on debates over the reauthorization of Section 702 of the Foreign Intelligence Surveillance Act (FISA).",
        "query": "surveillance FISA privacy intelligence national security data collection"
    },
    {
        "name": "Climate and Energy Policy",
        "summary": "These articles discuss legislative efforts related to climate change and energy policy.",
        "query": "climate change environment energy carbon emissions regulations"
    }
]

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
        return {}

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
    args = parser.parse_args()
    
    congress = args.congress
    
    print("Gridlock Score Calculator - Direct Version")
    print("=======================================\n")
    print(f"Analyzing legislative gridlock for the {congress}th Congress")
    print(f"Using {len(POLICY_CLUSTERS)} predefined policy clusters\n")
    
    # Load bill status data
    bill_status = load_bill_status()
    if not bill_status:
        print("Error: No bill status data available. Cannot proceed.")
        return
    
    print(f"Loaded status data for {len(bill_status)} bills\n")
    
    # Process each policy cluster
    policy_results = []
    
    for i, policy in enumerate(POLICY_CLUSTERS):
        print(f"\nProcessing policy {i+1}/{len(POLICY_CLUSTERS)}: {policy['name']}")
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
            f.write(f"   Query: {result['policy']['query']}\n")
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