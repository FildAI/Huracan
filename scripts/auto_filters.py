import os
from dotenv import load_dotenv
from pathlib import Path
import sys

load_dotenv()

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.qdrant_client import get_client
from core.indexes.vocab_bm25 import VocabBM25Index


def main():

    query = "Which banks in New York have over 10,000 employees?"
    client = get_client()
    vocab_index = VocabBM25Index(client)
    results = vocab_index.search_with_auto_filter(query)
    print(results)

if __name__ == "__main__":
    main()