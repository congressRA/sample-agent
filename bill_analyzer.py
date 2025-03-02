import json
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from pinecone import Pinecone
from openai import OpenAI
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI and Pinecone
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Constants
BILL_SUMMARIES_INDEX = "bill-summaries"
EMBEDDING_MODEL = "text-embedding-3-small"

class BillAnalyzer:
    def __init__(self):
        """Initialize the BillAnalyzer with necessary connections."""
        # Connect to Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.bill_index = self.pc.Index(BILL_SUMMARIES_INDEX)
        self.client = OpenAI(api_key=openai_api_key)
        
        # Load bill status dictionary
        try:
            with open("bill_status_dict.json", "r") as f:
                self.bill_status_dict = json.load(f)
            logger.info(f"Loaded status data for {len(self.bill_status_dict)} bills")
        except FileNotFoundError:
            # Try alternate filename used in legacy code
            try:
                with open("bill_status.json", "r") as f:
                    bill_status_data = json.load(f)
                    # Convert the list of dicts to a dictionary {id: enacted}
                    self.bill_status_dict = {item['id']: item['enacted'] for item in bill_status_data}
                logger.info(f"Loaded status data for {len(self.bill_status_dict)} bills from bill_status.json")
            except FileNotFoundError:
                logger.error("Neither bill_status_dict.json nor bill_status.json found")
                self.bill_status_dict = {}
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for the given text using OpenAI's API."""
        response = self.client.embeddings.create(
            input=[text],
            model=EMBEDDING_MODEL
        )
        return response.data[0].embedding
    
    def query_bills_for_cluster(self, 
                               cluster: Dict[str, Any], 
                               congress_number: str,
                               top_k: int = 10,
                               threshold: float = 0.4) -> List[Dict[str, Any]]:
        """
        Query the bill-summaries index for bills related to a given policy cluster.
        
        Args:
            cluster: A dictionary containing cluster information (name, query, summary)
            congress_number: The number of the Congress (e.g., "118")
            top_k: Number of bills to retrieve
            threshold: Minimum similarity score threshold for matches
            
        Returns:
            List of matching bill metadata
        """
        # Construct a query string combining relevant cluster information
        query_text = f"{cluster['name']}. {cluster['summary']} {cluster.get('query', '')}"
        logger.info(f"Searching bills for cluster: {cluster['name']}")
        
        # Generate embedding for the query
        query_embedding = self.generate_embedding(query_text)
        
        # Filter by congress number
        filter_dict = {"congress": congress_number}
        
        # Query the index
        results = self.bill_index.query(
            vector=query_embedding,
            filter=filter_dict,
            top_k=top_k,
            include_metadata=True,
            include_values=False
        )
        
        # Filter results by threshold
        filtered_matches = [match for match in results.matches if match.score >= threshold]
        
        # Extract and format results
        bills = []
        for match in filtered_matches:
            bill_data = {
                "bill_id": match.id,
                "score": match.score,
                "metadata": match.metadata
            }
            bills.append(bill_data)
        
        # Sort by score (highest first)
        bills.sort(key=lambda x: x["score"], reverse=True)
        
        logger.info(f"Found {len(bills)} bills for cluster: {cluster['name']} (threshold: {threshold})")
        return bills
    
    def check_bill_status(self, bill_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Check if bills are enacted based on the bill_status_dict.
        
        Args:
            bill_ids: List of bill IDs to check
            
        Returns:
            Dictionary mapping bill IDs to their status information
        """
        logger.info(f"Checking status for {len(bill_ids)} bills")
        status_results = {}
        
        for bill_id in bill_ids:
            if bill_id in self.bill_status_dict:
                # Handle both formats of bill_status_dict
                status_info = self.bill_status_dict[bill_id]
                
                if isinstance(status_info, dict):
                    # New format with detailed status info
                    status_results[bill_id] = {
                        "enacted": status_info.get("enacted", False),
                        "status": status_info.get("status", "unknown"),
                        "title": status_info.get("title", ""),
                        "latest_action": status_info.get("latest_action", "")
                    }
                else:
                    # Legacy format where value is directly the enacted status (0 or 1)
                    enacted = bool(status_info)
                    status_results[bill_id] = {
                        "enacted": enacted,
                        "status": "Enacted" if enacted else "Not Enacted",
                        "title": "",
                        "latest_action": ""
                    }
            else:
                status_results[bill_id] = {
                    "enacted": False,
                    "status": "unknown",
                    "title": "",
                    "latest_action": "Status not found in database"
                }
        
        return status_results
    
    def analyze_cluster_legislation(self, 
                                   cluster: Dict[str, Any], 
                                   congress_number: str,
                                   top_k: int = 100,
                                   threshold: float = 0.4) -> Dict[str, Any]:
        """
        Comprehensive analysis of legislation related to a policy cluster.
        
        Args:
            cluster: A dictionary containing cluster information
            congress_number: The number of the Congress
            top_k: Number of bills to retrieve from vector DB
            threshold: Minimum similarity score for bill matches
            
        Returns:
            Analysis results including bills and their enactment status
        """
        # Find related bills - use threshold from cluster if available
        custom_threshold = cluster.get("threshold", threshold)
        related_bills = self.query_bills_for_cluster(
            cluster, 
            congress_number, 
            top_k=top_k, 
            threshold=custom_threshold
        )
        
        # Extract bill IDs
        bill_ids = [bill["bill_id"] for bill in related_bills]
        
        # Check status of bills
        status_results = self.check_bill_status(bill_ids)
        
        # Count enacted bills
        enacted_count = sum(1 for status in status_results.values() if status["enacted"])
        
        # Combine bills with their status
        bills_with_status = []
        for bill in related_bills:
            bill_id = bill["bill_id"]
            status_info = status_results.get(bill_id, {"enacted": False, "status": "unknown"})
            
            bill_data = {
                "bill_id": bill_id,
                "title": bill["metadata"].get("title", ""),
                "summary": bill["metadata"].get("summary_text", ""),
                "bill_type": bill["metadata"].get("bill_type", ""),
                "bill_number": bill["metadata"].get("bill_number", ""),
                "score": bill["score"],
                "enacted": status_info["enacted"],
                "status": status_info["status"]
            }
            
            if "latest_action" in status_info and status_info["latest_action"]:
                bill_data["latest_action"] = status_info["latest_action"]
                
            bills_with_status.append(bill_data)
        
        # Sort by relevance score
        bills_with_status.sort(key=lambda x: x["score"], reverse=True)
        
        has_enacted_legislation = enacted_count > 0
        
        return {
            "cluster_name": cluster["name"],
            "query": cluster.get("query", ""),
            "threshold": custom_threshold,
            "total_bills_found": len(related_bills),
            "enacted_bills": enacted_count,
            "has_enacted_legislation": has_enacted_legislation,
            "enactment_rate": enacted_count / len(related_bills) if related_bills else 0,
            "bills": bills_with_status
        }


def analyze_clusters(clusters: List[Dict[str, Any]], congress_number: str, top_k: int = 100, threshold: float = 0.4) -> Dict[str, Any]:
    """
    Analyze all clusters from NYT analysis to determine legislative gridlock.
    
    Args:
        clusters: List of policy clusters from NYT analysis
        congress_number: The congressional session number
        top_k: Maximum number of bills to retrieve per cluster
        threshold: Default similarity threshold for bill matches
        
    Returns:
        Comprehensive analysis of legislation related to policy clusters
    """
    analyzer = BillAnalyzer()
    results = []
    
    for cluster in clusters:
        logger.info(f"Analyzing cluster: {cluster['name']}")
        
        # Use custom threshold from cluster if available
        custom_threshold = cluster.get("threshold", threshold)
        
        cluster_analysis = analyzer.analyze_cluster_legislation(
            cluster, 
            congress_number,
            top_k=top_k,
            threshold=custom_threshold
        )
        results.append(cluster_analysis)
    
    # Calculate overall gridlock metrics
    total_clusters = len(results)
    clusters_with_enacted_bills = sum(1 for r in results if r["enacted_bills"] > 0)
    gridlock_rate = 1 - (clusters_with_enacted_bills / total_clusters) if total_clusters else 0
    
    logger.info(f"Analysis complete: {clusters_with_enacted_bills}/{total_clusters} clusters have enacted legislation")
    logger.info(f"Gridlock rate: {gridlock_rate:.2f}")
    
    return {
        "congress": congress_number,
        "total_policy_clusters": total_clusters,
        "clusters_with_enacted_bills": clusters_with_enacted_bills,
        "clusters_with_no_enacted_bills": total_clusters - clusters_with_enacted_bills,
        "gridlock_rate": gridlock_rate,
        "cluster_results": results
    }


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze legislative gridlock based on NYT policy clusters")
    parser.add_argument("--clusters-file", type=str, help="Path to the JSON file containing policy clusters")
    parser.add_argument("--congress", type=str, required=True, help="Congress number (e.g., 118)")
    parser.add_argument("--output", type=str, help="Path to save analysis results")
    parser.add_argument("--top-k", type=int, default=100, help="Maximum number of bills to retrieve per cluster")
    parser.add_argument("--threshold", type=float, default=0.4, help="Default similarity threshold for bill matches")
    
    args = parser.parse_args()
    
    # Log the configuration
    logger.info(f"Analyzing clusters for Congress {args.congress}")
    logger.info(f"Bill search parameters: top_k={args.top_k}, threshold={args.threshold}")
    
    # Load clusters from file if provided, otherwise use stdin
    if args.clusters_file:
        logger.info(f"Loading clusters from file: {args.clusters_file}")
        with open(args.clusters_file, "r") as f:
            data = json.load(f)
            clusters = data.get("clusters", [])
    else:
        logger.info("Loading clusters from stdin")
        data = json.load(sys.stdin)
        clusters = data.get("clusters", [])
    
    logger.info(f"Loaded {len(clusters)} policy clusters for analysis")
    
    # Run analysis
    results = analyze_clusters(
        clusters, 
        args.congress,
        top_k=args.top_k,
        threshold=args.threshold
    )
    
    # Output results
    if args.output:
        logger.info(f"Saving results to: {args.output}")
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
    else:
        print(json.dumps(results, indent=2))
    
    # Print summary to console
    print(f"\nGRIDLOCK ANALYSIS SUMMARY - {args.congress}th Congress")
    print(f"----------------------------------------")
    print(f"Total policy clusters analyzed: {results['total_policy_clusters']}")
    print(f"Clusters with enacted legislation: {results['clusters_with_enacted_bills']} ({results['clusters_with_enacted_bills']/results['total_policy_clusters']*100:.1f}%)")
    print(f"Clusters with no enacted legislation: {results['clusters_with_no_enacted_bills']} ({results['clusters_with_no_enacted_bills']/results['total_policy_clusters']*100:.1f}%)")
    print(f"Gridlock rate: {results['gridlock_rate']*100:.1f}%")