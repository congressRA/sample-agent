#!/usr/bin/env python3
"""
Run the complete gridlock analysis pipeline:
1. Extract policy clusters from NYT articles using nyt_clusters.py
2. For each cluster, find related bills from the bill-summaries index
3. Check which bills were enacted using bill_status_dict.json
4. Calculate gridlock metrics and generate a report
"""

import os
import json
import argparse
import logging
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_nyt_clusters(congress: str, output_file: Optional[str] = None) -> str:
    """
    Run the nyt_clusters.py script to extract policy clusters.
    
    Args:
        congress: Congress number (e.g., "118")
        output_file: Optional file path to save the results
        
    Returns:
        Path to the output file containing clusters
    """
    if not output_file:
        output_file = f"clusters_congress_{congress}.json"
    
    logger.info(f"Extracting policy clusters for Congress {congress}")
    
    cmd = ["python", "nyt_clusters.py", "--congress", congress, "--output", output_file]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Error running nyt_clusters.py: {result.stderr}")
        raise RuntimeError(f"Failed to extract policy clusters: {result.stderr}")
    
    logger.info(f"Policy clusters saved to {output_file}")
    return output_file

def run_bill_analyzer(clusters_file: str, congress: str, output_file: Optional[str] = None) -> str:
    """
    Run the bill_analyzer.py script to find related bills for each cluster.
    
    Args:
        clusters_file: Path to the JSON file containing policy clusters
        congress: Congress number (e.g., "118")
        output_file: Optional file path to save the results
        
    Returns:
        Path to the output file containing bill analysis
    """
    if not output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"gridlock_analysis_{congress}_{timestamp}.json"
    
    logger.info(f"Analyzing bills for clusters in {clusters_file}")
    
    cmd = ["python", "bill_analyzer.py", "--clusters-file", clusters_file, "--congress", congress, "--output", output_file]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Error running bill_analyzer.py: {result.stderr}")
        raise RuntimeError(f"Failed to analyze bills: {result.stderr}")
    
    logger.info(f"Bill analysis saved to {output_file}")
    return output_file

def run_gridlock_analyzer(analysis_file: str, congress: str, output_dir: str = "gridlock_results", generate_report: bool = True) -> str:
    """
    Run the gridlock_analyzer.py script to generate the final analysis and report.
    
    Args:
        analysis_file: Path to the JSON file containing bill analysis
        congress: Congress number (e.g., "118")
        output_dir: Directory to save the results
        generate_report: Whether to generate a human-readable report
        
    Returns:
        Path to the output directory containing results
    """
    logger.info(f"Generating gridlock analysis for Congress {congress}")
    
    cmd = ["python", "gridlock_analyzer.py", "--clusters-file", analysis_file, "--congress", congress, "--output-dir", output_dir]
    
    if generate_report:
        cmd.append("--generate-report")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Error running gridlock_analyzer.py: {result.stderr}")
        raise RuntimeError(f"Failed to generate gridlock analysis: {result.stderr}")
    
    logger.info(f"Gridlock analysis saved to {output_dir}")
    print(result.stdout)  # Print the summary output
    
    return output_dir

def main():
    parser = argparse.ArgumentParser(description="Run the complete gridlock analysis pipeline")
    parser.add_argument("--congress", type=str, required=True, help="Congress number (e.g., 118)")
    parser.add_argument("--clusters-file", type=str, help="Path to existing clusters file (skips cluster extraction)")
    parser.add_argument("--analysis-file", type=str, help="Path to existing bill analysis file (skips bill analysis)")
    parser.add_argument("--output-dir", type=str, default="gridlock_results", help="Directory to save results")
    parser.add_argument("--skip-report", action="store_true", help="Skip generating human-readable report")
    
    args = parser.parse_args()
    
    # Step 1: Extract policy clusters (if not provided)
    clusters_file = args.clusters_file
    if not clusters_file:
        # Check if clusters file already exists with the standard naming convention
        default_clusters_file = f"clusters_congress_{args.congress}.json"
        if os.path.exists(default_clusters_file):
            logger.info(f"Using existing clusters file: {default_clusters_file}")
            clusters_file = default_clusters_file
        else:
            clusters_file = run_nyt_clusters(args.congress)
    
    # Step 2: Analyze bills for each cluster (if not provided)
    analysis_file = args.analysis_file
    if not analysis_file:
        analysis_file = run_bill_analyzer(clusters_file, args.congress)
    
    # Step 3: Generate gridlock analysis and report
    run_gridlock_analyzer(analysis_file, args.congress, args.output_dir, not args.skip_report)
    
    logger.info("Gridlock analysis pipeline completed successfully")

if __name__ == "__main__":
    main()