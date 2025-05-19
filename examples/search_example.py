"""
Example script demonstrating vector search with Huracan.

This script shows how to:
1. Create a SearchPipeline with default configurations
2. Index a DataFrame of company data
3. Perform vector searches with natural language queries
"""

import os
import pandas as pd
from dotenv import load_dotenv
import sys
from pathlib import Path

# Add project root to path to allow imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from fild.huracan import SearchPipeline, load_config
from fild.huracan.utils import get_logger

# Load environment variables
load_dotenv()


def load_sample_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} companies from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        # Return empty DataFrame
        return pd.DataFrame()


def main():
    """Run the search example."""
    logger = get_logger("search_example")
    logger.info("Starting search example")
    
    # Load configuration
    config = load_config()
    
    # Create search pipeline
    pipeline = SearchPipeline.create_default_pipeline(
        collection_name="companies",
        available_fields=["name", "description", "industry", "country", "employee_count"],
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        qdrant_url=config["qdrant"]["url"],
        qdrant_api_key=config["qdrant"]["api_key"],
        default_limit=10
    )
    
    # Check if we need to index data
    should_index = input("Index sample data? (y/n): ").lower() == 'y'
    
    if should_index:
        # Load sample data
        sample_data_path = project_root / "data" / "companies_sample.csv"
        if not sample_data_path.exists():
            logger.error(f"Sample data file not found: {sample_data_path}")
            print(f"Please place a companies_sample.csv file in {sample_data_path.parent}")
            return
        
        df = load_sample_data(str(sample_data_path))
        if len(df) == 0:
            logger.error("No data to index")
            return
        
        # Index data
        try:
            logger.info(f"Indexing {len(df)} companies")
            pipeline.vector_store.delete_collection()  # Optional: reset collection
            total_indexed = pipeline.index_dataframe(df)
            logger.info(f"Indexed {total_indexed} companies")
        except Exception as e:
            logger.error(f"Error indexing data: {e}")
            return
    
    # Run search loop
    print("\nSearch companies by typing natural language queries.")
    print("Type 'quit' or 'exit' to end the program.\n")
    
    while True:
        query = input("\nEnter search query: ")
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        try:
            # Perform search
            results_df, metadata = pipeline.search(query)
            
            # Display results
            if len(results_df) == 0:
                print("No results found.")
            else:
                print(f"\nFound {len(results_df)} results:")
                print(results_df[["score", "name", "industry", "country"]].head(10))
                
                # Show additional details for top result
                if len(results_df) > 0:
                    top_result = results_df.iloc[0]
                    print("\nTop result details:")
                    print(f"Name: {top_result.get('name', 'N/A')}")
                    print(f"Industry: {top_result.get('industry', 'N/A')}")
                    print(f"Country: {top_result.get('country', 'N/A')}")
                    print(f"Employees: {top_result.get('employee_count', 'N/A')}")
                    if 'description' in top_result:
                        print(f"Description: {top_result['description'][:200]}...")
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            print(f"Error searching: {e}")
    
    print("Goodbye!")


if __name__ == "__main__":
    main() 