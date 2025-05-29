import sys
from pathlib import Path

# Add project root to path to allow imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.indexes.vocab_bm25 import VocabBM25Index
from core.qdrant_client import get_client


def populate_first_time():
    client = get_client()
    index = VocabBM25Index(client)
    index.add_text_indexes_for_all_columns(force_recreate=True)
    index.populate_vocab_collection()

def repopulate_vocab():
    client = get_client()
    index = VocabBM25Index(client)
    index.populate_vocab_collection()


if __name__ == "__main__":
    repopulate_vocab()
   

    

