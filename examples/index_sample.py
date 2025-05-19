"""
Script to index the subsample_10.csv file with both dense embeddings and BM25.
"""

import os
import pandas as pd
from dotenv import load_dotenv
import sys
from pathlib import Path
import time

# Add project root to path to allow imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from fild.huracan import (
    VectorStore,
    OpenAIEmbedder,
    SentenceTransformerShort,
    BM25Embedder
)
from fild.huracan.utils import get_logger, load_config

# Load environment variables
load_dotenv()

def main():
    """Index the subsample_10.csv data."""
    # Setup logging
    logger = get_logger("index_sample")
    logger.info("Starting sample indexing")
    
    # Load configuration
    config = load_config()
    
    # Check if file exists
    file_path = project_root / "workbooks" / "lead_generation" / "subsample_10.csv"
    if not file_path.exists():
        logger.error(f"Sample data file not found: {file_path}")
        print(f"File not found: {file_path}")
        return
    
    # Load the CSV file
    try:
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} companies")
        
        # Prepare corpus for BM25
        corpus = []
        for _, row in df.iterrows():
            if 'description' in row and pd.notna(row['description']):
                corpus.append(str(row['description']))
            else:
                corpus.append("")
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        print(f"Error loading CSV: {e}")
        return
    
    # Create embedders
    embedders = {
        "dense": OpenAIEmbedder(api_key=os.getenv("OPENAI_API_KEY")),
        "mini": SentenceTransformerShort(),
        "bm25": BM25Embedder(corpus=corpus)  # Initialize with our corpus
    }
    
    # Create vector store
    vector_store = VectorStore(
        collection_name="companies_sample",
        embedders=embedders,
        url=config["qdrant"]["url"],
        api_key=config["qdrant"]["api_key"]
    )
    
    # Reset collection
    logger.info(f"Resetting collection: companies_sample")
    vector_store.delete_collection()
    
    # Create collection if it doesn't exist
    vector_store.create_collection_if_not_exists()
    
    # Process in batches
    print(f"Indexing {len(df)} companies")
    
    start_time = time.time()
    
    try:
        # Convert DataFrame to list of dictionaries
        records = df.to_dict(orient="records")
        
        # Index the records
        vector_store.upsert(records)
        
    except Exception as e:
        logger.error(f"Error indexing data: {e}")
        print(f"Error indexing data: {e}")
        return
    
    # Log completion
    end_time = time.time()
    total_time = end_time - start_time
    
    logger.info(f"Indexing complete: {len(df)} records in {total_time:.2f}s")
    print(f"\nIndexing complete: {len(df)} records in {total_time:.2f}s")
    print(f"Average time per record: {(total_time/len(df)):.4f}s")
    
    print("\nNow you can use the search_example.py script to search for companies.")

if __name__ == "__main__":
    main() 