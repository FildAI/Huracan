"""
Script to search the indexed companies using Qdrant's native BM25 capabilities.
"""

import os
import pandas as pd
from dotenv import load_dotenv
import sys
from pathlib import Path

# Add project root to path to allow imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from qdrant_client import QdrantClient, models
from fild.huracan import OpenAIEmbedder
from fild.huracan.utils import get_logger, load_config

# Load environment variables
load_dotenv()

def main():
    """Search the indexed companies using Qdrant's native BM25."""
    # Setup logging
    logger = get_logger("search_native_bm25")
    logger.info("Starting search with native BM25")
    
    # Load configuration
    config = load_config()
    
    # Initialize OpenAI embedder for dense vectors
    openai_embedder = OpenAIEmbedder(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Initialize Qdrant client
    qdrant_url = config["qdrant"]["url"]
    qdrant_api_key = config["qdrant"]["api_key"]
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    collection_name = "companies_bm25"
    
    # Check if collection exists
    try:
        client.get_collection(collection_name=collection_name)
    except Exception:
        logger.error(f"Collection {collection_name} does not exist")
        print(f"Collection {collection_name} does not exist. Please run index_with_native_bm25.py first.")
        return
    
    # Run search loop
    print("\nSearch companies by typing natural language queries.")
    print("Available search modes:")
    print("1. Dense vector search (use prefix 'dense:')")
    print("2. BM25 text search on specific field (use prefix 'bm25-field:')")
    print("3. Combined search (use prefix 'hybrid:')")
    print("Type 'quit' or 'exit' to end the program.\n")
    
    while True:
        user_input = input("\nEnter search query: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        # Parse search mode and query
        if user_input.startswith("dense:"):
            mode = "dense"
            query = user_input[6:].strip()
        elif user_input.startswith("bm25-field:"):
            mode = "bm25-field"
            query = user_input[11:].strip()
        elif user_input.startswith("hybrid:"):
            mode = "hybrid"
            query = user_input[7:].strip()
        else:
            # Default to hybrid search
            mode = "hybrid"
            query = user_input
        
        print(f"\nSearch mode: {mode}")
        print(f"Query: {query}")
        
        try:
            results = []
            
            if mode == "dense":
                # Dense vector search
                vector = openai_embedder.create_embedding(query)
                search_result = client.search(
                    collection_name=collection_name,
                    query_vector={"dense": vector},
                    limit=10,
                    with_payload=True
                )
                results = search_result
                print("Performed dense vector search")
                
            elif mode == "bm25-field":
                # Ask for field to search in
                print("Available fields: name, description, primary_industry, country")
                field = input("Enter field to search in: ").strip()
                if not field:
                    field = "description"  # Default to description
                
                # BM25 search on specific field
                search_result = client.scroll(
                    collection_name=collection_name,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key=field,
                                match=models.MatchText(
                                    text=query,
                                    # BM25 is the default algorithm
                                )
                            )
                        ]
                    ),
                    limit=10,
                    with_payload=True
                )
                results = search_result[0]  # First element is points, second is next_page_offset
                print(f"Performed BM25 text search on field: {field}")
                
            elif mode == "hybrid":
                # First, create dense vector embedding
                vector = openai_embedder.create_embedding(query)
                
                # Then set up text search condition on description field
                text_condition = models.FieldCondition(
                    key="description",
                    match=models.MatchText(text=query)
                )
                
                # Use hybrid search - this combines vector search with filter
                search_result = client.search(
                    collection_name=collection_name,
                    query_vector={"dense": vector},
                    query_filter=models.Filter(should=[text_condition]),
                    limit=10,
                    with_payload=True
                )
                results = search_result
                print("Performed hybrid search (dense vectors + BM25 text filter)")
            
            # Display results
            if not results:
                print("No results found.")
            else:
                print(f"\nFound {len(results)} results:")
                
                # Format results as DataFrame
                rows = []
                for i, point in enumerate(results, 1):
                    payload = point.payload
                    row = {
                        "rank": i,
                        "score": getattr(point, "score", 0.0),
                        "name": payload.get("name", "Unknown"),
                        "industry": payload.get("primary_industry", "Unknown"),
                        "country": payload.get("country", "Unknown")
                    }
                    rows.append(row)
                
                results_df = pd.DataFrame(rows)
                print(results_df[["rank", "score", "name", "industry", "country"]])
                
                # Show details for top result
                if rows:
                    top_result = results[0].payload
                    print("\nTop result details:")
                    print(f"Name: {top_result.get('name', 'N/A')}")
                    print(f"Industry: {top_result.get('primary_industry', 'N/A')}")
                    print(f"Country: {top_result.get('country', 'N/A')}")
                    print(f"Size: {top_result.get('size', 'N/A')}")
                    print(f"Type: {top_result.get('type', 'N/A')}")
                    
                    if "description" in top_result and top_result["description"]:
                        desc = top_result["description"]
                        print(f"\nDescription: {desc[:300]}..." if len(desc) > 300 else f"\nDescription: {desc}")
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            print(f"Error searching: {e}")
    
    print("Goodbye!")

if __name__ == "__main__":
    main() 