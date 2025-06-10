# Companies Bulk Service

A service for efficiently bulk inserting company data into both the main companies collection and vocab collections for BM25 filtering in Qdrant.

## Features

- **Bulk Insert Optimization**: Processes companies in configurable batches with optional indexing disable/enable for better performance
- **Multiple Input Methods**: Supports inserting from Python dictionaries, pandas DataFrames, or CSV files
- **Automatic Vocab Management**: Automatically maintains vocabulary collections for BM25 text search
- **Collection Setup**: Handles creation and configuration of all required collections
- **Search Integration**: Provides search functionality using combined vector and BM25 approaches
- **Progress Tracking**: Uses tqdm for progress visualization during bulk operations

## Usage

### Basic Setup

```python
from core.services.companies.bulk_service import CompaniesBulkService

# Initialize the service
service = CompaniesBulkService(batch_size=100)

# Setup collections (creates them if they don't exist)
service.setup_collections(force_recreate=False)
```

### Single Company Insert

```python
company_data = {
    "name": "Apple Inc.",
    "description": "Technology company specializing in consumer electronics",
    "primary_industry": "Technology",
    "country": "United States",
    "size": "Large"
}

service.add_single_company(company_data)
```

### Bulk Insert from List

```python
companies_data = [
    {
        "name": "Apple Inc.",
        "description": "Technology company",
        "primary_industry": "Technology",
        "country": "United States",
        "size": "Large"
    },
    # ... more companies
]

service.add_companies_bulk(companies_data, disable_indexing=True)
```

### Insert from CSV

```python
service.add_companies_from_csv("companies.csv", disable_indexing=True)
```

### Insert from DataFrame

```python
import pandas as pd

df = pd.read_csv("companies.csv")
service.add_companies_from_dataframe(df, disable_indexing=True)
```

### Search Companies

```python
results = service.search_companies("technology companies", limit=10, verbose=False)

for point in results.points:
    print(f"{point.payload['name']} - {point.payload['primary_industry']}")
```

### Get Collection Information

```python
info = service.get_collection_info()
print(f"Companies: {info['companies']['points_count']} points")
for col_name, col_info in info['vocab_collections'].items():
    print(f"Vocab {col_name}: {col_info['points_count']} points")
```

## Configuration

### Batch Size
Configure the batch size for processing:

```python
service = CompaniesBulkService(batch_size=500)  # Default is 100
```

### Indexing Control
For large bulk inserts, disable indexing during insertion for better performance:

```python
service.add_companies_bulk(data, disable_indexing=True)  # Default is True
```

## Required Data Fields

The service works with the following company data fields (as configured in `vector_config.py`):

- **name**: Company name (used for BM25 search)
- **description**: Company description 
- **primary_industry**: Industry classification (used for BM25 search)
- **country**: Country location (used for BM25 search)
- **size**: Company size (used for BM25 search)

Additional fields can be included and will be stored in the payload but won't be indexed for BM25 search unless configured in `VocabBM25Config`.

## Performance Tips

1. **Use bulk methods** for inserting multiple companies
2. **Disable indexing** during large bulk inserts (automatically re-enabled)
3. **Adjust batch size** based on your system's memory and performance
4. **Use appropriate data types** - avoid very large text fields that don't need to be searchable

## Example Script

See `scripts/companies_bulk_example.py` for a complete working example with various usage patterns. 