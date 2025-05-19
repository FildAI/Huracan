"""
Example script demonstrating how to index company data from a CSV file.

This script shows how to:
1. Load company data from a CSV file
2. Create embeddings for each company
3. Index the data in Qdrant
"""

import os
import pandas as pd
from dotenv import load_dotenv
import sys
import argparse
from pathlib import Path
import time

# Add project root to path to allow imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from fild.huracan import (
    VectorStore,
    OpenAIEmbedder,
    SentenceTransformerShort
)
from fild.huracan.utils import get_logger, load_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Index company data from CSV")
    parser.add_argument("--file", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--collection", type=str, default="companies", help="Collection name")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for indexing")
    parser.add_argument("--reset", action="store_true", help="Reset collection before indexing")
    return parser.parse_args()


def main():
    """Run the indexing example."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    logger = get_logger("indexing_example")
    logger.info("Starting indexing example")
    
    # Load environment variables
    load_dotenv()
    
    # Load configuration
    config = load_config()
    
    # Check if file exists
    file_path = Path(args.file)
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        print(f"File not found: {file_path}")
        return
    
    # Create embedders
    embedders = {
        "dense": OpenAIEmbedder(api_key=os.getenv("OPENAI_API_KEY")),
        "mini": SentenceTransformerShort()
    }
    
    # Create vector store
    vector_store = VectorStore(
        collection_name=args.collection,
        embedders=embedders,
        url=config["qdrant"]["url"],
        api_key=config["qdrant"]["api_key"]
    )
    
    # Reset collection if requested
    if args.reset:
        logger.info(f"Resetting collection: {args.collection}")
        vector_store.delete_collection()
    
    # Create collection if it doesn't exist
    vector_store.create_collection_if_not_exists()
    
    # Load the CSV file
    try:
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} companies")
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        print(f"Error loading CSV: {e}")
        return
    
    # Process in batches
    batch_size = args.batch_size
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    print(f"Indexing {len(df)} companies in {total_batches} batches of size {batch_size}")
    
    start_time = time.time()
    total_indexed = 0
    
    try:
        # Convert DataFrame to list of dictionaries
        records = df.to_dict(orient="records")
        
        # Index in batches
        for i in range(0, len(records), batch_size):
            batch_start = time.time()
            batch = records[i:i+batch_size]
            
            # Index batch
            vector_store.upsert(batch)
            
            # Update progress
            total_indexed += len(batch)
            batch_end = time.time()
            batch_time = batch_end - batch_start
            
            # Log progress
            batch_num = i // batch_size + 1
            logger.info(f"Batch {batch_num}/{total_batches} complete: {len(batch)} records in {batch_time:.2f}s")
            print(f"Batch {batch_num}/{total_batches} complete: {len(batch)} records in {batch_time:.2f}s")
            print(f"Progress: {total_indexed}/{len(df)} records ({(total_indexed/len(df)*100):.1f}%)")
    
    except Exception as e:
        logger.error(f"Error indexing data: {e}")
        print(f"Error indexing data: {e}")
    
    # Log completion
    end_time = time.time()
    total_time = end_time - start_time
    
    logger.info(f"Indexing complete: {total_indexed} records in {total_time:.2f}s")
    print(f"\nIndexing complete: {total_indexed} records in {total_time:.2f}s")
    print(f"Average time per record: {(total_time/total_indexed if total_indexed else 0):.4f}s")


if __name__ == "__main__":
    main() 