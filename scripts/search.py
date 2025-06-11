import os
from dotenv import load_dotenv
from pathlib import Path
import sys

load_dotenv()

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.qdrant_client import get_client
from core.indexes.companies import CompaniesIndex


def main():

    query = "AHAHAHAHAHAHAHAHAH"
    client = get_client()
    companies_index = CompaniesIndex(client)
    results = companies_index.search(query, verbose=True)
    print("Query results: ", results)

if __name__ == "__main__":
    main()