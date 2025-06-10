#!/usr/bin/env python3
"""
Example script demonstrating how to use the CompaniesBulkService
for bulk inserting company data into both companies and vocab collections.
"""

import sys
import os
from pathlib import Path
import pandas as pd

# Add the parent directory to the path so we can import from core
sys.path.append(str(Path(__file__).parent.parent))

from core.services.companies.bulk_service import CompaniesBulkService


def create_sample_companies_data():
    """Create sample company data for demonstration"""
    sample_companies = [
        {
            "name": "Apple Inc.",
            "description": "Technology company specializing in consumer electronics, software, and online services",
            "primary_industry": "Technology",
            "country": "United States",
            "size": "Large",
            "website": "https://apple.com",
            "founded": 1976
        },
        {
            "name": "Microsoft Corporation",
            "description": "Multinational technology corporation producing computer software, consumer electronics, and personal computers",
            "primary_industry": "Technology",
            "country": "United States", 
            "size": "Large",
            "website": "https://microsoft.com",
            "founded": 1975
        },
        {
            "name": "Tesla Inc.",
            "description": "Electric vehicle and clean energy company",
            "primary_industry": "Automotive",
            "country": "United States",
            "size": "Large",
            "website": "https://tesla.com",
            "founded": 2003
        },
        {
            "name": "OpenAI",
            "description": "AI research and deployment company focused on artificial general intelligence",
            "primary_industry": "Artificial Intelligence",
            "country": "United States",
            "size": "Medium",
            "website": "https://openai.com",
            "founded": 2015
        },
        {
            "name": "SpaceX",
            "description": "Aerospace manufacturer and space transportation services company",
            "primary_industry": "Aerospace",
            "country": "United States",
            "size": "Large",
            "website": "https://spacex.com",
            "founded": 2002
        }
    ]
    return sample_companies


def example_single_insert():
    """Example of adding a single company"""
    print("=== Single Company Insert Example ===")
    
    # Initialize the service
    service = CompaniesBulkService(batch_size=50)
    
    # Setup collections (this will create them if they don't exist)
    service.setup_collections(force_recreate=False)
    
    # Add a single company
    single_company = {
        "name": "GitHub Inc.",
        "description": "Web-based version control and collaboration platform for software developers",
        "primary_industry": "Technology",
        "country": "United States",
        "size": "Large",
        "website": "https://github.com",
        "founded": 2008
    }
    
    service.add_single_company(single_company)
    print("Single company added successfully!\n")


def example_bulk_insert():
    """Example of bulk inserting companies"""
    print("=== Bulk Insert Example ===")
    
    # Initialize the service with custom batch size
    service = CompaniesBulkService(batch_size=100)
    
    # Setup collections
    service.setup_collections(force_recreate=False)
    
    # Get sample data
    companies_data = create_sample_companies_data()
    
    # Bulk insert companies
    service.add_companies_bulk(companies_data, disable_indexing=True)
    print("Bulk insert completed successfully!\n")


def example_csv_insert():
    """Example of inserting companies from CSV file"""
    print("=== CSV Insert Example ===")
    
    # Create a sample CSV file
    csv_path = "sample_companies.csv"
    sample_data = create_sample_companies_data()
    df = pd.DataFrame(sample_data)
    df.to_csv(csv_path, index=False)
    print(f"Created sample CSV file: {csv_path}")
    
    # Initialize the service
    service = CompaniesBulkService(batch_size=50)
    
    # Setup collections
    service.setup_collections(force_recreate=False)
    
    # Insert from CSV
    service.add_companies_from_csv(csv_path, disable_indexing=True)
    
    # Clean up the sample CSV file
    os.remove(csv_path)
    print("CSV insert completed and sample file cleaned up!\n")


def example_dataframe_insert():
    """Example of inserting companies from pandas DataFrame"""
    print("=== DataFrame Insert Example ===")
    
    # Initialize the service
    service = CompaniesBulkService(batch_size=25)
    
    # Setup collections
    service.setup_collections(force_recreate=False)
    
    # Create DataFrame
    sample_data = create_sample_companies_data()
    df = pd.DataFrame(sample_data)
    
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame columns: {list(df.columns)}")
    
    # Insert from DataFrame
    service.add_companies_from_dataframe(df, disable_indexing=True)
    print("DataFrame insert completed successfully!\n")


def example_search():
    """Example of searching companies after insertion"""
    print("=== Search Example ===")
    
    # Initialize the service
    service = CompaniesBulkService()
    
    # Search queries
    queries = [
        "technology companies",
        "electric vehicles",
        "artificial intelligence",
        "space exploration",
        "companies in United States"
    ]
    
    for query in queries:
        print(f"\nSearching for: '{query}'")
        try:
            results = service.search_companies(query, limit=3, verbose=False)
            
            if hasattr(results, 'points') and results.points:
                print(f"Found {len(results.points)} results:")
                for i, point in enumerate(results.points, 1):
                    company_name = point.payload.get('name', 'Unknown')
                    industry = point.payload.get('primary_industry', 'Unknown')
                    score = getattr(point, 'score', 0)
                    print(f"  {i}. {company_name} ({industry}) - Score: {score:.3f}")
            else:
                print("  No results found")
        except Exception as e:
            print(f"  Search failed: {e}")
    
    print()


def example_collection_info():
    """Example of getting collection information"""
    print("=== Collection Info Example ===")
    
    # Initialize the service
    service = CompaniesBulkService()
    
    # Get collection information
    info = service.get_collection_info()
    
    print("Collection Information:")
    print(f"Companies Collection:")
    if 'error' in info.get('companies', {}):
        print(f"  Error: {info['companies']['error']}")
    else:
        companies_info = info.get('companies', {})
        print(f"  Name: {companies_info.get('name', 'Unknown')}")
        print(f"  Points Count: {companies_info.get('points_count', 0)}")
        print(f"  Status: {companies_info.get('status', 'Unknown')}")
    
    print(f"\nVocab Collections:")
    vocab_collections = info.get('vocab_collections', {})
    for col_name, col_info in vocab_collections.items():
        if 'error' in col_info:
            print(f"  {col_name}: Error - {col_info['error']}")
        else:
            print(f"  {col_name}: {col_info.get('points_count', 0)} points ({col_info.get('status', 'Unknown')})")
    
    print()


def main():
    """Main function demonstrating all examples"""
    print("CompaniesBulkService Usage Examples")
    print("=" * 50)
    
    try:
        # Run examples
        example_single_insert()
        example_bulk_insert()
        example_dataframe_insert()
        example_csv_insert()
        example_search()
        example_collection_info()
        
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 