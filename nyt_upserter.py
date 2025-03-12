#!/usr/bin/env python3
"""
Efficient upserter for NYT articles to Pinecone vector database.
- Uses batched upserts with retry logic
- Leverages parallel processing
- Supports both REST and gRPC clients
- Includes progress tracking
"""

import os
import json
import time
import logging
import asyncio
import itertools
from typing import List, Dict, Any, Generator, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import argparse

import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
try:
    from pinecone.grpc import PineconeGRPC
    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False

from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
EMBEDDING_MODEL = "text-embedding-3-small"
INDEX_NAME = "nyt-archive"
INDEX_DIMENSION = 1536
DEFAULT_BATCH_SIZE = 100
MAX_RETRIES = 5
RETRY_DELAY = 2  # seconds


class NYTUpserter:
    def __init__(
        self, 
        use_grpc: bool = False, 
        batch_size: int = DEFAULT_BATCH_SIZE,
        index_name: str = INDEX_NAME,
        dimension: int = INDEX_DIMENSION,
        overwrite: bool = False
    ):
        """
        Initialize the NYT article upserter.
        
        Args:
            use_grpc: Whether to use gRPC client for potentially faster uploads
            batch_size: Number of vectors to upsert in each batch
            index_name: Name of the Pinecone index
            dimension: Dimension of the embeddings
            overwrite: Whether to overwrite existing vectors
        """
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_env = os.getenv("PINECONE_ENV")
        
        if not all([self.openai_api_key, self.pinecone_api_key]):
            raise ValueError("Missing required API keys in environment variables")
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        
        # Initialize Pinecone client
        if use_grpc and GRPC_AVAILABLE:
            logger.info("Using gRPC client for Pinecone")
            self.pc = PineconeGRPC(api_key=self.pinecone_api_key)
        else:
            if use_grpc and not GRPC_AVAILABLE:
                logger.warning("gRPC requested but not available. Install with 'pip install pinecone-client[grpc]'")
            logger.info("Using REST client for Pinecone")
            self.pc = Pinecone(api_key=self.pinecone_api_key)
        
        self.index_name = index_name
        self.dimension = dimension
        self.batch_size = batch_size
        self.overwrite = overwrite
        
        # Connect to index or create if it doesn't exist
        self._connect_to_index()

    def _connect_to_index(self):
        """Connect to existing index or create a new one if it doesn't exist."""
        try:
            # Check if index exists
            indexes = self.pc.list_indexes()
            
            if self.index_name not in [idx.name for idx in indexes]:
                logger.info(f"Creating new index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                # Wait for index to be ready
                while not self.pc.describe_index(self.index_name).status.ready:
                    logger.info("Waiting for index to be ready...")
                    time.sleep(2)
            
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Error connecting to index: {e}")
            raise

    def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for the given text using OpenAI's API."""
        try:
            response = self.openai_client.embeddings.create(
                input=[text],
                model=EMBEDDING_MODEL
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def _get_first_element(self, value):
        """Extract first element if value is a list, otherwise return the value."""
        if isinstance(value, list) and value:
            return value[0]
        return value
    
    def _prepare_articles_for_embedding(
        self, 
        articles: List[Dict[str, Any]]
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Prepare articles for embedding by extracting text and creating IDs.
        Only processes articles with publication dates between 2013-2024.
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            List of tuples containing (id, text_to_embed, namespace, metadata)
        """
        prepared_articles = []
        filtered_out = 0
        
        for article in articles:
            try:
                # Extract publication date first to filter early
                pub_date = self._get_first_element(article.get("pub_date", ""))
                
                # Filter articles by publication year (2013-2024 only)
                year = None
                if pub_date:
                    try:
                        date_obj = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
                        year = date_obj.year
                        # Skip articles outside our target range (2013-2024)
                        if year < 2013 or year > 2024:
                            filtered_out += 1
                            continue
                    except (ValueError, TypeError):
                        # If we can't parse the date, we'll still process the article
                        pass
                
                # Handle both regular dict and dict-with-arrays formats
                # Extract primary fields, handling cases where values might be arrays
                headline_obj = article.get("headline", {})
                if isinstance(headline_obj, list) and headline_obj:
                    headline_obj = headline_obj[0]
                elif not isinstance(headline_obj, dict):
                    headline_obj = {"main": str(headline_obj)}
                
                headline = self._get_first_element(headline_obj.get("main", ""))
                lead_paragraph = self._get_first_element(article.get("lead_paragraph", ""))
                abstract = self._get_first_element(article.get("abstract", ""))
                snippet = self._get_first_element(article.get("snippet", ""))
                
                # Create combined text for embedding
                text_to_embed = f"{headline}\n{abstract}\n{lead_paragraph}\n{snippet}"
                
                # Create a unique ID
                web_url = self._get_first_element(article.get("web_url", ""))
                article_id = self._get_first_element(article.get("_id", ""))
                
                # If we don't have an ID but have a URL, use URL to create ID
                if not article_id and web_url:
                    # Use last part of URL as ID component
                    url_parts = web_url.rstrip('/').split('/')
                    url_id = url_parts[-1].split('.')[0] if url_parts else ""
                    
                    # Create a unique ID combining publication date and URL ID
                    article_id = f"{pub_date}_{url_id}" if pub_date and url_id else f"nyt_{hash(text_to_embed)}"
                
                # Ensure we have some ID
                if not article_id:
                    article_id = f"nyt_{hash(text_to_embed)}"
                
                # Extract metadata
                metadata = {
                    "headline": headline,
                    "pub_date": pub_date,
                    "web_url": web_url,
                    "source": self._get_first_element(article.get("source", "nyt")),
                    "section_name": self._get_first_element(article.get("section_name", "")),
                    "subsection_name": self._get_first_element(article.get("subsection_name", "")),
                    "document_type": self._get_first_element(article.get("document_type", "")),
                    "news_desk": self._get_first_element(article.get("news_desk", "")),
                    "word_count": self._get_first_element(article.get("word_count", 0)),
                    "lead_paragraph": lead_paragraph,
                    "abstract": abstract,
                    "snippet": snippet
                }
                
                # Add year and month to metadata if we have them
                if year is not None:
                    month = date_obj.month
                    metadata["year"] = year
                    metadata["month"] = month
                    namespace = f"{year}_{month}"
                else:
                    namespace = "unknown_date"
                
                # Add to list
                prepared_articles.append((article_id, text_to_embed, namespace, metadata))
                
            except Exception as e:
                logger.error(f"Error processing article: {e}")
                logger.debug(f"Problematic article: {article}")
                continue
        
        if filtered_out > 0:
            logger.info(f"Filtered out {filtered_out} articles with publication dates outside 2013-2024 range")
            
        return prepared_articles

    async def _batch_generate_embeddings(
        self, 
        prepared_articles: List[Tuple[str, str, str, Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Generate embeddings for a batch of articles using thread pool for parallelism.
        
        Args:
            prepared_articles: List of tuples (id, text, namespace, metadata)
            
        Returns:
            List of vector records ready for upserting
        """
        # Define a function to generate embeddings for one article
        def _embed_single_article(article_tuple):
            article_id, text, namespace, metadata = article_tuple
            try:
                vector = self.generate_embedding(text)
                return {
                    "id": article_id,
                    "values": vector,
                    "metadata": metadata,
                    "namespace": namespace
                }
            except Exception as e:
                logger.error(f"Error generating embedding for article {article_id}: {e}")
                return None
        
        # Use a ThreadPoolExecutor to parallelize embedding generation
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            tasks = [
                loop.run_in_executor(executor, _embed_single_article, article)
                for article in prepared_articles
            ]
            
            # Wait for all tasks to complete
            vector_records = await asyncio.gather(*tasks)
        
        # Filter out any None values (failed embeddings)
        return [record for record in vector_records if record is not None]

    async def _upsert_batch_with_retry(
        self, 
        vector_records: List[Dict[str, Any]]
    ) -> bool:
        """
        Upsert a batch of vectors with retry logic.
        
        Args:
            vector_records: List of vector records to upsert
            
        Returns:
            True if successful, False otherwise
        """
        if not vector_records:
            return True
        
        # Group by namespace
        namespace_groups = {}
        for record in vector_records:
            namespace = record.pop("namespace", "")
            if namespace not in namespace_groups:
                namespace_groups[namespace] = []
            namespace_groups[namespace].append(record)
        
        for namespace, records in namespace_groups.items():
            for attempt in range(MAX_RETRIES):
                try:
                    # Upsert the batch to this namespace
                    self.index.upsert(vectors=records, namespace=namespace, batch_size=self.batch_size)
                    logger.debug(f"Successfully upserted {len(records)} vectors to namespace {namespace}")
                    break
                except Exception as e:
                    if attempt < MAX_RETRIES - 1:
                        wait_time = RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"Upsert attempt {attempt+1} failed: {e}, retrying in {wait_time}s")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Failed to upsert batch after {MAX_RETRIES} attempts: {e}")
                        return False
        
        return True

    def chunks(self, iterable, batch_size: int) -> Generator:
        """Split an iterable into chunks of specified size."""
        it = iter(iterable)
        chunk = list(itertools.islice(it, batch_size))
        while chunk:
            yield chunk
            chunk = list(itertools.islice(it, batch_size))

    async def upsert_articles(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Upsert a list of NYT articles to Pinecone.
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            Dictionary with upsert statistics
        """
        start_time = time.time()
        total_articles = len(articles)
        logger.info(f"Starting upsert of {total_articles} articles")
        
        # Prepare articles for embedding
        prepared_articles = self._prepare_articles_for_embedding(articles)
        
        # Track statistics
        successful_upserts = 0
        failed_upserts = 0
        
        # Process in batches
        for i, batch in enumerate(self.chunks(prepared_articles, self.batch_size)):
            batch_start = time.time()
            logger.info(f"Processing batch {i+1}/{(total_articles // self.batch_size) + 1} ({len(batch)} articles)")
            
            # Generate embeddings for this batch
            vector_records = await self._batch_generate_embeddings(batch)
            
            # Upsert the batch
            success = await self._upsert_batch_with_retry(vector_records)
            
            if success:
                successful_upserts += len(vector_records)
                failed_count = len(batch) - len(vector_records)
                if failed_count > 0:
                    failed_upserts += failed_count
                    logger.warning(f"{failed_count} articles failed during embedding generation")
            else:
                failed_upserts += len(batch)
                logger.error(f"Batch {i+1} failed during upsert")
            
            batch_duration = time.time() - batch_start
            logger.info(f"Batch {i+1} completed in {batch_duration:.2f}s ({len(batch)/batch_duration:.2f} articles/s)")
        
        total_duration = time.time() - start_time
        throughput = successful_upserts / total_duration if total_duration > 0 else 0
        
        results = {
            "total_articles": total_articles,
            "successful_upserts": successful_upserts,
            "failed_upserts": failed_upserts,
            "duration_seconds": total_duration,
            "throughput_per_second": throughput
        }
        
        logger.info(f"Upsert completed: {successful_upserts}/{total_articles} articles successful, " 
                    f"{total_duration:.2f}s total ({throughput:.2f} articles/s)")
        
        return results

    async def upsert_from_file(self, filepath: str) -> Dict[str, Any]:
        """
        Upsert articles from a JSON file.
        
        Args:
            filepath: Path to JSON file containing articles
            
        Returns:
            Dictionary with upsert statistics
        """
        logger.info(f"Loading articles from {filepath}")
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Handle different JSON formats:
            articles = []
            
            # Format 1: Dict with 'response' -> 'docs' (standard NYT API format)
            if isinstance(data, dict) and 'response' in data and 'docs' in data['response']:
                articles = data['response']['docs']
            
            # Format 2: Direct list of articles
            elif isinstance(data, list):
                articles = data
            
            # Format 3: Single article object
            elif isinstance(data, dict) and any(k in data for k in ['headline', 'abstract', 'lead_paragraph', '_id', 'web_url']):
                articles = [data]
                
            # Unknown format
            else:
                logger.warning(f"Unrecognized JSON format in {filepath}, attempting to process anyway")
                articles = [data]
            
            logger.info(f"Loaded {len(articles)} articles from file")
            return await self.upsert_articles(articles)
            
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in file {filepath}")
            return {"total_articles": 0, "successful_upserts": 0, "failed_upserts": 1, "duration_seconds": 0}
        except Exception as e:
            logger.error(f"Error loading or processing file {filepath}: {e}")
            return {"total_articles": 0, "successful_upserts": 0, "failed_upserts": 1, "duration_seconds": 0}

    def upsert_from_dataframe(self, df: pd.DataFrame, text_column: str = None) -> Dict[str, Any]:
        """
        Upsert articles from a pandas DataFrame.
        
        Args:
            df: DataFrame containing article data
            text_column: Column name containing text to embed (if None, will use combined fields)
            
        Returns:
            Dictionary with upsert statistics
        """
        articles = df.to_dict(orient='records')
        return asyncio.run(self.upsert_articles(articles))

    async def batch_load_articles(self, filepaths: List[str], max_files: int = 1000) -> List[Dict[str, Any]]:
        """
        Load and combine articles from multiple files in batches
        
        Args:
            filepaths: List of file paths to JSON files
            max_files: Maximum number of files to load in one batch
            
        Returns:
            List of article dictionaries
        """
        all_articles = []
        batch_size = min(len(filepaths), max_files)
        processed_files = 0
        skipped_files = 0
        
        logger.info(f"Loading batch of up to {batch_size} files")
        
        for filepath in filepaths[:batch_size]:
            # Skip video files
            if "nyt_video_" in filepath:
                skipped_files += 1
                continue
                
            # Only process article files if they contain "nyt_article_"
            if "nyt_article_" not in filepath:
                skipped_files += 1
                continue
            
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Handle different JSON formats:
                # Format 1: Dict with 'response' -> 'docs' (standard NYT API format)
                if isinstance(data, dict) and 'response' in data and 'docs' in data['response']:
                    all_articles.extend(data['response']['docs'])
                
                # Format 2: Direct list of articles
                elif isinstance(data, list):
                    all_articles.extend(data)
                
                # Format 3: Single article object
                elif isinstance(data, dict) and any(k in data for k in ['headline', 'abstract', 'lead_paragraph', '_id', 'web_url']):
                    all_articles.append(data)
                    
                # Unknown format
                else:
                    logger.warning(f"Unrecognized JSON format in {filepath}, attempting to process anyway")
                    all_articles.append(data)
                
                processed_files += 1
                    
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON in file {filepath}")
            except Exception as e:
                logger.error(f"Error loading file {filepath}: {e}")
        
        logger.info(f"Loaded {len(all_articles)} articles from {processed_files} files (skipped {skipped_files} non-article files)")
        return all_articles

    async def upsert_from_directory_async(self, directory: str, pattern: str = "*.json", batch_size: int = 100) -> Dict[str, Any]:
        """
        Upsert articles from all JSON files in a directory (async version).
        Uses batching for much higher throughput.
        
        Args:
            directory: Directory containing JSON files
            pattern: File pattern to match
            batch_size: Number of files to process in each batch
            
        Returns:
            Dictionary with upsert statistics
        """
        import glob
        start_time = time.time()
        
        # Determine whether to use recursive globbing
        if '**' in pattern:
            # The '**' pattern requires recursive=True flag
            files = glob.glob(os.path.join(directory, pattern), recursive=True)
        else:
            files = glob.glob(os.path.join(directory, pattern))
        
        if not files:
            logger.warning(f"No files matching pattern {pattern} found in {directory}")
            return {"total_articles": 0, "successful_upserts": 0, "failed_upserts": 0}
        
        logger.info(f"Found {len(files)} files to process")
        
        # Track total statistics
        total_stats = {
            "total_articles": 0,
            "successful_upserts": 0,
            "failed_upserts": 0,
            "duration_seconds": 0,
            "files_processed": 0,
            "files_failed": 0,
            "batches_processed": 0
        }
        
        # Process files in batches for much higher throughput
        for i in tqdm(range(0, len(files), batch_size), desc="Processing batches"):
            batch_files = files[i:i+batch_size]
            batch_start_time = time.time()
            
            try:
                # Load articles from all files in this batch
                articles = await self.batch_load_articles(batch_files)
                
                if articles:
                    # Upsert all articles in one go
                    batch_stats = await self.upsert_articles(articles)
                    
                    # Accumulate statistics
                    for key in ["total_articles", "successful_upserts", "failed_upserts", "duration_seconds"]:
                        total_stats[key] += batch_stats.get(key, 0)
                    
                    total_stats["files_processed"] += len(batch_files)
                    total_stats["batches_processed"] += 1
                    
                    # Calculate batch throughput
                    batch_duration = time.time() - batch_start_time
                    batch_throughput = batch_stats["successful_upserts"] / batch_duration if batch_duration > 0 else 0
                    
                    logger.info(f"Batch {total_stats['batches_processed']} completed: "
                                f"{batch_stats['successful_upserts']}/{batch_stats['total_articles']} articles successful "
                                f"({batch_throughput:.2f} articles/s)")
                else:
                    logger.warning(f"No articles found in batch {i//batch_size + 1}")
                    total_stats["files_failed"] += len(batch_files)
                
            except Exception as e:
                total_stats["files_failed"] += len(batch_files)
                logger.error(f"Failed to process batch {i//batch_size + 1}: {e}")
        
        # Calculate overall throughput and duration
        total_stats["duration_seconds"] = time.time() - start_time
        if total_stats["duration_seconds"] > 0:
            total_stats["throughput_per_second"] = total_stats["successful_upserts"] / total_stats["duration_seconds"]
        else:
            total_stats["throughput_per_second"] = 0
        
        # Final summary
        logger.info(f"Directory processing complete: {total_stats['files_processed']}/{len(files)} files processed successfully")
        logger.info(f"Articles: {total_stats['successful_upserts']}/{total_stats['total_articles']} successful")
        logger.info(f"Overall throughput: {total_stats['throughput_per_second']:.2f} articles/s")
        
        return total_stats
        
    def upsert_from_directory(self, directory: str, pattern: str = "*.json", 
                             file_batch_size: int = 100) -> Dict[str, Any]:
        """
        Upsert articles from all JSON files in a directory.
        
        Args:
            directory: Directory containing JSON files
            pattern: File pattern to match
            file_batch_size: Number of files to process in each batch
            
        Returns:
            Dictionary with upsert statistics
        """
        # Create a new event loop for this function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            return loop.run_until_complete(
                self.upsert_from_directory_async(directory, pattern, file_batch_size)
            )
        finally:
            loop.close()


def main():
    """Main entry point for the script, non-async version to avoid asyncio issues."""
    parser = argparse.ArgumentParser(description="Upsert NYT articles to Pinecone vector database")
    parser.add_argument('--input', type=str, help='Input file or directory path')
    parser.add_argument('--grpc', action='store_true', help='Use gRPC client for faster uploads')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size for vector upserts')
    parser.add_argument('--file-batch-size', type=int, default=100, help='Number of files to process in each batch')
    parser.add_argument('--recursive', action='store_true', help='Process directories recursively')
    parser.add_argument('--pattern', type=str, default="nyt_article_*.json", help='File pattern to match (e.g., "*.json", "nyt_article_*.json")')
    parser.add_argument('--dimension', type=int, default=INDEX_DIMENSION, help='Vector dimension')
    parser.add_argument('--index-name', type=str, default=INDEX_NAME, help='Pinecone index name')
    
    args = parser.parse_args()
    
    upserter = NYTUpserter(
        use_grpc=args.grpc,
        batch_size=args.batch_size,
        index_name=args.index_name,
        dimension=args.dimension
    )
    
    if not args.input:
        parser.print_help()
        return
    
    stats = None
    
    if os.path.isfile(args.input):
        # Process single file - need to create a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            stats = loop.run_until_complete(upserter.upsert_from_file(args.input))
        finally:
            loop.close()
    elif os.path.isdir(args.input):
        # Process directory
        pattern = args.pattern
        if args.recursive:
            # For recursive search, make sure pattern includes directory wildcard
            if '/' not in pattern:
                pattern = f"**/{pattern}"
            
        logger.info(f"Searching for files with pattern: {pattern} in {args.input}")
        stats = upserter.upsert_from_directory(
            args.input, 
            pattern=pattern,
            file_batch_size=args.file_batch_size
        )
    else:
        logger.error(f"Input path {args.input} does not exist")
        return
    
    if stats:
        # Print statistics
        print("\nUpsert Statistics:")
        print(f"Total articles processed: {stats['total_articles']}")
        
        if stats['total_articles'] > 0:
            success_pct = stats['successful_upserts']/stats['total_articles']*100
            failed_pct = stats['failed_upserts']/stats['total_articles']*100
            print(f"Successfully upserted: {stats['successful_upserts']} ({success_pct:.2f}%)")
            print(f"Failed upserts: {stats['failed_upserts']} ({failed_pct:.2f}%)")
        else:
            print("Successfully upserted: 0 (0.00%)")
            print("Failed upserts: 0 (0.00%)")
            
        print(f"Total duration: {stats['duration_seconds']:.2f} seconds")
        print(f"Throughput: {stats['throughput_per_second']:.2f} articles/second")
        
        if 'files_processed' in stats:
            print(f"Files processed: {stats['files_processed']}")
            print(f"Files failed: {stats.get('files_failed', 0)}")


if __name__ == "__main__":
    # Run the non-async main function directly 
    main()