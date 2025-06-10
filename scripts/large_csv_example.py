#!/usr/bin/env python3

import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from core.services.companies.bulk_service import CompaniesBulkService


def create_large_sample_csv(filename: str, num_rows: int = 50000):
    print(f"Creating sample CSV with {num_rows} rows...")
    
    sample_companies = [
        "Apple Inc.", "Microsoft Corporation", "Tesla Inc.", "OpenAI", "SpaceX",
        "Google LLC", "Amazon.com Inc.", "Meta Platforms Inc.", "NVIDIA Corporation",
        "Intel Corporation", "IBM", "Oracle Corporation", "Salesforce Inc.",
        "Adobe Inc.", "Netflix Inc.", "Uber Technologies Inc.", "Airbnb Inc.",
        "Zoom Video Communications", "Slack Technologies", "Dropbox Inc."
    ]
    
    industries = ["Technology", "Automotive", "Aerospace", "AI/ML", "E-commerce", 
                 "Social Media", "Cloud Computing", "Software", "Hardware"]
    
    countries = ["United States", "Canada", "United Kingdom", "Germany", "France", 
                "Japan", "South Korea", "China", "India", "Australia"]
    
    sizes = ["Startup", "Small", "Medium", "Large", "Enterprise"]
    
    data = []
    for i in range(num_rows):
        company_base = sample_companies[i % len(sample_companies)]
        data.append({
            "name": f"{company_base} {i+1}",
            "description": f"Company {i+1} specializing in innovative solutions",
            "primary_industry": industries[i % len(industries)],
            "country": countries[i % len(countries)],
            "size": sizes[i % len(sizes)],
            "website": f"https://company{i+1}.com",
            "founded": 1990 + (i % 34)
        })
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Created {filename} with {len(df)} rows")
    return filename


def example_large_csv_processing():
    print("=== Large CSV Processing Example ===")
    
    csv_filename = "large_companies_sample.csv"
    
    try:
        create_large_sample_csv(csv_filename, 50000)
        
        service = CompaniesBulkService(batch_size=200)
        service.setup_collections(force_recreate=False)
        
        result = service.add_companies_from_large_csv(
            csv_path=csv_filename,
            chunk_size=5000,
            disable_indexing=True,
            max_retries=3
        )
        
        print("\nProcessing Results:")
        print(f"Total processed: {result['total_processed']}")
        print(f"Total errors: {result['total_errors']}")
        print(f"Total chunks: {result['chunk_count']}")
        
        info = service.get_collection_info()
        print(f"\nFinal collection size: {info['companies']['points_count']} companies")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        import os
        if os.path.exists(csv_filename):
            os.remove(csv_filename)
            print(f"Cleaned up {csv_filename}")


def example_with_different_chunk_sizes():
    print("=== Testing Different Chunk Sizes ===")
    
    csv_filename = "test_chunks.csv"
    
    try:
        create_large_sample_csv(csv_filename, 20000)
        
        chunk_sizes = [1000, 5000, 10000]
        
        for chunk_size in chunk_sizes:
            print(f"\nTesting chunk size: {chunk_size}")
            
            service = CompaniesBulkService(batch_size=100)
            service.setup_collections(force_recreate=True)
            
            import time
            start_time = time.time()
            
            result = service.add_companies_from_large_csv(
                csv_path=csv_filename,
                chunk_size=chunk_size,
                disable_indexing=True
            )
            
            processing_time = time.time() - start_time
            
            print(f"Chunk size {chunk_size}: {processing_time:.2f}s, "
                  f"{result['total_processed']/processing_time:.0f} rows/s")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        import os
        if os.path.exists(csv_filename):
            os.remove(csv_filename)


def main():
    print("Large CSV Processing Examples")
    print("=" * 50)
    
    example_large_csv_processing()
    example_with_different_chunk_sizes()
    
    print("\nAll examples completed!")


if __name__ == "__main__":
    main() 