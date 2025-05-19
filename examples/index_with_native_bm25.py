"""
Script to index the subsample_10.csv file using Qdrant's native BM25 capabilities.
"""

import os
import pandas as pd
from dotenv import load_dotenv
import sys
from pathlib import Path
import time
import logging
import traceback

# Add project root to path to allow imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from qdrant_client import QdrantClient, models
from openai import OpenAI, OpenAIError
from fild.huracan.utils import get_logger, load_config

# Load environment variables
load_dotenv()

def create_embedding(text, model="text-embedding-3-small", max_retries=3):
    """Create embedding with retry logic"""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=30.0)
    
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                input=text,
                model=model
            )
            return response.data[0].embedding
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Retry {attempt+1}/{max_retries} creating embedding: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"Failed to create embedding after {max_retries} attempts: {e}")
                raise

def main():
    """Index the subsample_10.csv data with Qdrant's native BM25."""
    # Setup logging
    logger = get_logger("index_native_bm25")
    logger.info("Starting sample indexing with native BM25")
    
    # Load configuration
    config = load_config()
    
    # Check if file exists
    file_path = project_root / "workbooks" / "lead_generation" / "subsample_10.csv"
    if not file_path.exists():
        # Try alternate path
        file_path = project_root / "fild" / "workbooks" / "lead_generation" / "subsample_10.csv"
        if not file_path.exists():
            logger.error(f"Sample data file not found: {file_path}")
            print(f"File not found: {file_path}")
            return
    
    # Load the CSV file
    try:
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} companies")
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        print(f"Error loading CSV: {e}")
        return
    
    # Initialize Qdrant client with increased timeout
    qdrant_url = config["qdrant"]["url"]
    qdrant_api_key = config["qdrant"]["api_key"]
    
    try:
        client = QdrantClient(
            url=qdrant_url, 
            api_key=qdrant_api_key,
            timeout=30.0  # Increase timeout
        )
        logger.info(f"Connected to Qdrant at {qdrant_url}")
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        print(f"Failed to connect to Qdrant: {e}")
        return
    
    collection_name = "companies_bm25"
    
    # Delete existing collection if it exists
    try:
        client.delete_collection(collection_name=collection_name)
        logger.info(f"Deleted existing collection: {collection_name}")
    except Exception:
        logger.info(f"Collection {collection_name} does not exist yet")
    
    # Create collection with dense vectors
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=1536,  # OpenAI embedding size
                    distance=models.Distance.COSINE
                )
            }
        )
        logger.info(f"Created collection: {collection_name}")
    except Exception as e:
        logger.error(f"Error creating collection: {e}")
        print(f"Error creating collection: {e}")
        return
    
    # Create text indexes for relevant fields
    fields_to_index = ["name", "description", "primary_industry", "country"]
    
    for field in fields_to_index:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=models.TextIndexParams(
                    type="text",
                    tokenizer=models.TokenizerType.WORD,
                    min_token_len=2,
                    max_token_len=20,
                    lowercase=True,
                )
            )
            logger.info(f"Created text index for field: {field}")
        except Exception as e:
            logger.error(f"Error creating index for {field}: {e}")
    
    # Index the records one by one instead of batch
    start_time = time.time()
    print(f"Indexing {len(df)} companies")
    
    success_count = 0
    
    # Process each record individually
    records = df.to_dict(orient="records")
    for i, record in enumerate(records):
        try:
            # Create a unified text for embedding
            text_for_embedding = " ".join(
                str(v) for k, v in record.items() 
                if pd.notna(v) and v and k in ["name", "description", "primary_industry", "country"]
            )
            
            # Create OpenAI embedding with retry logic
            vector = create_embedding(text_for_embedding)
            
            # Create point with numeric ID (not string)
            point_id = i  # Use integer ID instead of string
            
            # Upload point individually
            client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,  # Integer ID
                        vector={"dense": vector},
                        payload=record
                    )
                ]
            )
            
            success_count += 1
            print(f"Indexed company {i+1}/{len(df)}: {record.get('name', 'Unknown')}")
            
            # Small pause between requests
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error indexing company {i+1}: {e}")
            print(f"Error indexing company {i+1}: {str(e)}")
            # Continue with next record
    
    # Log completion
    end_time = time.time()
    total_time = end_time - start_time
    
    logger.info(f"Indexing complete: {success_count}/{len(df)} records in {total_time:.2f}s")
    print(f"\nIndexing complete: {success_count}/{len(df)} records in {total_time:.2f}s")
    
    if success_count > 0:
        print(f"Average time per record: {(total_time/success_count):.4f}s")
    
    print("\nCollection has been created with the following capabilities:")
    print("1. Dense vectors from OpenAI embeddings")
    print("2. Native BM25 text search on fields: name, description, primary_industry, country")
    print("\nYou can now run search_with_native_bm25.py to search the companies")

if __name__ == "__main__":
    main() 