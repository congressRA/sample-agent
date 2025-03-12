#!/usr/bin/env python3
"""
Script to count vectors in each namespace of a Pinecone index.
Useful for understanding data distribution across different time periods.
"""

import os
import argparse
import logging
from collections import defaultdict
from typing import Dict, List, Tuple
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
INDEX_NAME = "nyt-archive"

class NamespaceCounter:
    def __init__(self, index_name: str = INDEX_NAME):
        """
        Initialize the namespace counter.
        
        Args:
            index_name: Name of the Pinecone index to analyze
        """
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        
        if not self.pinecone_api_key:
            raise ValueError("Missing Pinecone API key in environment variables")
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.index_name = index_name
        
        # Connect to index
        self._connect_to_index()
    
    def _connect_to_index(self):
        """Connect to existing index."""
        try:
            # Check if index exists
            indexes = self.pc.list_indexes()
            
            if self.index_name not in [idx.name for idx in indexes]:
                raise ValueError(f"Index '{self.index_name}' does not exist")
            
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Error connecting to index: {e}")
            raise

    def list_namespaces(self) -> List[str]:
        """
        List all namespaces in the index.
        
        Returns:
            List of namespace names
        """
        try:
            # Get stats about the index
            index_stats = self.index.describe_index_stats()
            
            # Extract namespaces
            if hasattr(index_stats, 'namespaces') and index_stats.namespaces:
                namespaces = list(index_stats.namespaces.keys())
            else:
                # For older Pinecone API versions
                namespaces = list(index_stats.get('namespaces', {}).keys())
                
            logger.info(f"Found {len(namespaces)} namespaces in index")
            return namespaces
        
        except Exception as e:
            logger.error(f"Error listing namespaces: {e}")
            raise

    def count_vectors_by_namespace(self) -> Dict[str, int]:
        """
        Count the number of vectors in each namespace.
        
        Returns:
            Dictionary mapping namespace names to vector counts
        """
        try:
            # Get stats about the index
            index_stats = self.index.describe_index_stats()
            
            # Extract namespace vector counts
            namespace_counts = {}
            
            # Handle both object and dict API responses
            if hasattr(index_stats, 'namespaces') and index_stats.namespaces:
                # New API version
                for ns, stats in index_stats.namespaces.items():
                    namespace_counts[ns] = stats.vector_count
            else:
                # Older API version
                for ns, stats in index_stats.get('namespaces', {}).items():
                    namespace_counts[ns] = stats.get('vector_count', 0)
            
            logger.info(f"Retrieved vector counts for {len(namespace_counts)} namespaces")
            return namespace_counts
        
        except Exception as e:
            logger.error(f"Error counting vectors by namespace: {e}")
            raise

    def sort_namespaces_by_year_month(self, namespace_data: Dict[str, int]) -> List[Tuple[str, int]]:
        """
        Sort namespaces by year and month (assuming namespace format 'YYYY_MM').
        
        Args:
            namespace_data: Dictionary mapping namespace names to vector counts
            
        Returns:
            List of (namespace, count) tuples sorted by year and month
        """
        sorted_data = []
        other_data = []
        
        for ns, count in namespace_data.items():
            # Check if namespace follows the expected format
            if '_' in ns and ns.split('_')[0].isdigit() and ns.split('_')[1].isdigit():
                sorted_data.append((ns, count))
            else:
                other_data.append((ns, count))
        
        # Sort by year and month
        sorted_data.sort(key=lambda x: tuple(map(int, x[0].split('_'))))
        
        # Add other namespaces at the end
        return sorted_data + other_data

    def generate_report(self, output_file: str = None) -> Dict[str, List[Tuple[str, int]]]:
        """
        Generate a comprehensive report of vector counts by namespace.
        
        Args:
            output_file: Path to save the report (optional)
            
        Returns:
            Dictionary containing different groupings of the data
        """
        # Get raw data
        namespace_counts = self.count_vectors_by_namespace()
        
        # Sort by year and month
        chronological_data = self.sort_namespaces_by_year_month(namespace_counts)
        
        # Group by year
        yearly_data = defaultdict(int)
        for ns, count in chronological_data:
            if '_' in ns and ns.split('_')[0].isdigit():
                year = ns.split('_')[0]
                yearly_data[year] += count
            else:
                yearly_data["unknown"] += count
        
        # Sort by count (descending)
        sorted_by_count = sorted(namespace_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Prepare report data
        report_data = {
            "chronological": chronological_data,
            "by_count": sorted_by_count,
            "yearly_totals": sorted(yearly_data.items(), key=lambda x: x[0] if x[0] != "unknown" else "9999")
        }
        
        # Calculate total vectors
        total_vectors = sum(namespace_counts.values())
        
        # Generate report text
        report_text = []
        report_text.append(f"=== Namespace Analysis for Index: {self.index_name} ===\n")
        report_text.append(f"Total vectors: {total_vectors:,}\n")
        report_text.append(f"Total namespaces: {len(namespace_counts)}\n")
        
        # Report all namespaces by count
        report_text.append("\n== All Namespaces by Vector Count (Descending) ==")
        for ns, count in sorted_by_count:
            percentage = (count / total_vectors) * 100 if total_vectors > 0 else 0
            report_text.append(f"{ns}: {count:,} vectors ({percentage:.2f}%)")
        
        # Report yearly totals
        report_text.append("\n== Yearly Totals ==")
        for year, count in report_data["yearly_totals"]:
            percentage = (count / total_vectors) * 100 if total_vectors > 0 else 0
            report_text.append(f"{year}: {count:,} vectors ({percentage:.2f}%)")
        
        # Report chronological data
        report_text.append("\n== Chronological Breakdown ==")
        for ns, count in chronological_data:
            percentage = (count / total_vectors) * 100 if total_vectors > 0 else 0
            report_text.append(f"{ns}: {count:,} vectors ({percentage:.2f}%)")
        
        # Compile final report
        final_report = "\n".join(report_text)
        
        # Print to console
        print(final_report)
        
        # Write to file if requested
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    f.write(final_report)
                logger.info(f"Report saved to {output_file}")
            except Exception as e:
                logger.error(f"Error writing report to file: {e}")
        
        return report_data


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Count vectors in each namespace of a Pinecone index")
    parser.add_argument('--index-name', type=str, default=INDEX_NAME, help='Pinecone index name')
    parser.add_argument('--output', type=str, help='Output file path to save the report')
    
    args = parser.parse_args()
    
    counter = NamespaceCounter(index_name=args.index_name)
    counter.generate_report(output_file=args.output)


if __name__ == "__main__":
    main()