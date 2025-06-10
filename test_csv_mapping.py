#!/usr/bin/env python3
"""
Test script to demonstrate CSV column mapping functionality.
"""

from core.services.companies.bulk_service import CompaniesBulkService
import pandas as pd

def test_csv_mapping():
    # Create test CSV data with the format you mentioned
    test_data = {
        "Imported Companies": ["comp1", "comp2", "comp3"],
        "Name": ["Company A", "Company B", "Company C"],
        "Description": ["Tech company", "Finance company", "Healthcare company"],
        "Primary Industry": ["Technology", "Financial Services", "Healthcare"],
        "Size": ["100-500", "500-1000", "50-100"],
        "Type": ["Private", "Public", "Private"],
        "Location": ["San Francisco", "New York", "Boston"],
        "Country": ["USA", "USA", "USA"],
        "Domain": ["companya.com", "companyb.com", "companyc.com"],
        "LinkedIn URL": ["linkedin.com/company/a", "linkedin.com/company/b", "linkedin.com/company/c"]
    }
    
    # Create test CSV file
    df = pd.DataFrame(test_data)
    test_csv_path = "test_companies.csv"
    df.to_csv(test_csv_path, index=False)
    print(f"Created test CSV: {test_csv_path}")
    
    # Initialize the bulk service
    bulk_service = CompaniesBulkService()
    
    # Preview the mapping
    print("\n" + "="*50)
    print("TESTING CSV COLUMN MAPPING")
    print("="*50)
    
    mapping_result = bulk_service.preview_csv_mapping(test_csv_path)
    
    if mapping_result['validation_success']:
        print("\n✅ Column mapping successful!")
        print(f"Original CSV columns: {mapping_result['original_columns']}")
        print(f"Mapped columns: {mapping_result['mapped_columns']}")
        
        print("\nColumn mapping:")
        for orig, mapped in mapping_result['column_mapping'].items():
            print(f"  '{orig}' -> '{mapped}'")
        
        print("\nSample mapped data:")
        for i, row in enumerate(mapping_result['sample_data']):
            print(f"  Row {i+1}: {row}")
            
    else:
        print(f"❌ Validation failed: {mapping_result['error']}")
    
    # Test what happens with missing required fields
    print("\n" + "="*50)
    print("TESTING MISSING REQUIRED FIELDS")
    print("="*50)
    
    # Create CSV without required 'Name' field
    bad_data = {
        "Company": ["Company A", "Company B"],  # This should map to "name" but we call it "Company"
        "Description": ["Tech company", "Finance company"]
    }
    bad_df = pd.DataFrame(bad_data)
    bad_csv_path = "test_bad_companies.csv"
    bad_df.to_csv(bad_csv_path, index=False)
    
    try:
        bad_mapping_result = bulk_service.preview_csv_mapping(bad_csv_path)
        if not bad_mapping_result['validation_success']:
            print(f"✅ Correctly caught missing required fields: {bad_mapping_result['error']}")
        else:
            print("❌ Should have failed validation")
    except Exception as e:
        print(f"✅ Correctly caught missing required fields: {e}")
    
    print("\n" + "="*50)
    print("TESTING COMPLETE")
    print("="*50)

if __name__ == "__main__":
    test_csv_mapping() 