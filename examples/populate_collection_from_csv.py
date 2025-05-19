"""
Script to populate a Qdrant collection from a CSV file (payload only, no embeddings).
"""

import sys
from pathlib import Path

# Add project root to path to allow imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
from core.indexes.companies import CompaniesIndex
from core.qdrant_client import client
from core.embeddings import OpenAIEmbedder


if __name__ == "__main__":
    csv_path = project_root / "data" / "subsample_10.csv"
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records from {csv_path}")

    embedder = OpenAIEmbedder()
    index = CompaniesIndex(client, embedder=embedder)
    index.create_collection(force_recreate=True)
    print("Collection created (recreated if existed)")

    print("Populating collection...")
    index.populate_collection(df)
    print(f"Population complete: {len(df)} records inserted into collection '{index.collection_name}'") 