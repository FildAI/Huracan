#!/usr/bin/env python3
"""
Quick test script for the CompaniesBulkService
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import from core
sys.path.append(str(Path(__file__).parent.parent))

from core.services.companies.bulk_service import CompaniesBulkService


def quick_test():
    """Quick test of the service functionality"""
    print("Testing CompaniesBulkService...")
    
    # Initialize service
    service = CompaniesBulkService(batch_size=10)
    
    # Test collection info (should work even if collections don't exist)
    print("\n1. Testing collection info...")
    try:
        info = service.get_collection_info()
        print(f"Collection info retrieved: {len(info)} collections")
    except Exception as e:
        print(f"Collection info failed: {e}")
    
    # Test setup collections
    print("\n2. Testing collection setup...")
    try:
        service.setup_collections(force_recreate=False)
        print("Collections setup completed")
    except Exception as e:
        print(f"Setup failed: {e}")
        return
    
    # Test single insert
    print("\n3. Testing single company insert...")
    test_company = {
        "name": "Test Company",
        "description": "A test company for validation",
        "primary_industry": "Testing",
        "country": "Test Country",
        "size": "Small"
    }
    
    try:
        service.add_single_company(test_company)
        print("Single company insert completed")
    except Exception as e:
        print(f"Single insert failed: {e}")
    
    # Test search
    print("\n4. Testing search...")
    try:
        results = service.search_companies("test", limit=5, verbose=False)
        if hasattr(results, 'points') and results.points:
            print(f"Search found {len(results.points)} results")
            for point in results.points:
                print(f"  - {point.payload.get('name', 'Unknown')}")
        else:
            print("Search returned no results")
    except Exception as e:
        print(f"Search failed: {e}")
    
    # Final collection info
    print("\n5. Final collection info...")
    try:
        info = service.get_collection_info()
        companies_count = info.get('companies', {}).get('points_count', 0)
        print(f"Companies collection now has {companies_count} points")
        
        vocab_total = 0
        for col_name, col_info in info.get('vocab_collections', {}).items():
            count = col_info.get('points_count', 0)
            vocab_total += count
            print(f"  Vocab {col_name}: {count} points")
        print(f"Total vocab points: {vocab_total}")
        
    except Exception as e:
        print(f"Final info failed: {e}")
    
    print("\nTest completed!")


if __name__ == "__main__":
    quick_test() 