import sys
from pathlib import Path

# Add project root to path to allow imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.indexes.vocab_bm25 import VocabBM25Index
from core.qdrant_client import client



if __name__ == "__main__":
    index = VocabBM25Index(client)
    # index.add_text_indexes_for_all_columns(force_recreate=True)
    # index.populate_vocab_collection()

    res = index.find_similar_in_vocab("country", "united Kingdm", limit=1)
    res = index.search_by_text("country", res[0])

    print(res)

