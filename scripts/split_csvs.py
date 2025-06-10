import os
from dotenv import load_dotenv
from pathlib import Path
import sys

load_dotenv()

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.services.companies.bulk_service import CompaniesBulkService

def main():
    bulk_service = CompaniesBulkService()

    # Split a large CSV into chunks
    chunk_files = bulk_service.split_csv_into_chunks(
        csv_path="./subsample100000.csv",
        output_folder="./company_chunks",
        max_chunk_size=5000  # 5K rows per chunk
    )

    # # Later, process all the chunks
    # bulk_service.process_chunked_csv_folder(
    #     chunks_folder="./company_chunks",
    #     disable_indexing=True
    # )

if __name__ == "__main__":
    main()