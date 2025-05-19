"""
Script to search the indexed companies using BM25 and dense embeddings.
"""

import os
import pandas as pd
from dotenv import load_dotenv
import sys
from pathlib import Path

# Add project root to path to allow imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from fild.huracan import (
    VectorStore,
    OpenAIEmbedder,
    SentenceTransformerShort,
    BM25Embedder,
    QueryParser
)
from fild.huracan.utils import get_logger, load_config

# Load environment variables
load_dotenv()

def main():
    """Search the indexed companies."""
    # Setup logging
    logger = get_logger("search_with_bm25")
    logger.info("Starting search with BM25")
    
    # Load configuration
    config = load_config()
    
    # We need a small corpus to initialize BM25
    # In a real application, this would be loaded from the same dataset used for indexing
    mini_corpus = ["Sample text for BM25 initialization"]
    
    # Create embedders
    embedders = {
        "dense": OpenAIEmbedder(api_key=os.getenv("OPENAI_API_KEY")),
        "mini": SentenceTransformerShort(),
        "bm25": BM25Embedder(corpus=mini_corpus)
    }
    
    # Create vector store
    vector_store = VectorStore(
        collection_name="companies_sample",
        embedders=embedders,
        url=config["qdrant"]["url"],
        api_key=config["qdrant"]["api_key"]
    )
    
    # Create query parser
    query_parser = QueryParser(llm_api_key=os.getenv("OPENAI_API_KEY"))
    
    # Available fields for filtering
    available_fields = ["name", "description", "primary_industry", "country", "size", "type"]
    
    # Run search loop
    print("\nSearch companies by typing natural language queries.")
    print("You can search with specific embedders by adding '!dense', '!mini', or '!bm25' to your query.")
    print("Type 'quit' or 'exit' to end the program.\n")
    
    while True:
        query = input("\nEnter search query: ")
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        # Check for embedder specification
        embedder_name = None
        if "!dense" in query:
            embedder_name = "dense"
            query = query.replace("!dense", "").strip()
        elif "!mini" in query:
            embedder_name = "mini"
            query = query.replace("!mini", "").strip()
        elif "!bm25" in query:
            embedder_name = "bm25"
            query = query.replace("!bm25", "").strip()
        
        try:
            # Parse query to get filter and keywords
            filter_obj, keywords = query_parser.parse_query(query, available_fields)
            
            print(f"Searching for: '{query}'")
            print(f"Using filters: {filter_obj is not None}")
            print(f"Keywords: {keywords}")
            print(f"Embedder: {embedder_name or 'default'}")
            
            # Execute search
            search_text = " ".join(keywords) if keywords else query
            results = vector_store.search(
                query=search_text,
                filter_conditions=filter_obj,
                limit=10,
                embedder_name=embedder_name
            )
            
            # Display results
            if not results:
                print("No results found.")
            else:
                print(f"\nFound {len(results)} results:")
                
                # Format results as DataFrame
                rows = []
                for i, result in enumerate(results, 1):
                    metadata = result["metadata"]
                    row = {
                        "rank": i,
                        "score": result["score"],
                        "name": metadata.get("name", "Unknown"),
                        "industry": metadata.get("primary_industry", "Unknown"),
                        "country": metadata.get("country", "Unknown")
                    }
                    rows.append(row)
                
                results_df = pd.DataFrame(rows)
                print(results_df[["rank", "score", "name", "industry", "country"]])
                
                # Show details for top result
                if rows:
                    top_result = results[0]["metadata"]
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