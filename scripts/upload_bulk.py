import os
import argparse
import logging
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
import sys

load_dotenv()

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.services.companies.bulk_service import CompaniesBulkService


def setup_logging(log_level: str = "INFO") -> str:
    """Setup logging to both file and console."""
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = logs_dir / f"upload_bulk_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            # File handler
            logging.FileHandler(log_filename, encoding='utf-8'),
            # Console handler
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_filename}")
    return str(log_filename)


def process_single_file(bulk_service: CompaniesBulkService, file_path: str, disable_indexing: bool = True):
    logger = logging.getLogger(__name__)
    logger.info(f"Starting to process single file: {file_path}")
    
    try:
        bulk_service.add_companies_from_csv(
            csv_path=file_path,
            disable_indexing=disable_indexing
        )
        logger.info(f"Successfully processed single file: {file_path}")
    except Exception as e:
        logger.error(f"Failed to process single file {file_path}: {str(e)}")
        raise


def process_directory(bulk_service: CompaniesBulkService, directory_path: str, disable_indexing: bool = True):
    logger = logging.getLogger(__name__)
    logger.info(f"Starting to process directory: {directory_path}")
    
    dir_path = Path(directory_path)
    
    if not dir_path.exists():
        error_msg = f"Directory not found: {directory_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    if not dir_path.is_dir():
        error_msg = f"Path is not a directory: {directory_path}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Find all CSV files in the directory
    csv_files = list(dir_path.glob("*.csv"))
    
    if not csv_files:
        logger.warning(f"No CSV files found in directory: {directory_path}")
        return
    
    logger.info(f"Found {len(csv_files)} CSV files in directory: {directory_path}")
    
    # Track success/failure statistics
    successful_files = 0
    failed_files = 0
    
    # Process each CSV file
    for i, csv_file in enumerate(csv_files, 1):
        logger.info(f"[{i}/{len(csv_files)}] Processing file: {csv_file.name}")
        try:
            bulk_service.add_companies_from_csv(
                csv_path=str(csv_file),
                disable_indexing=disable_indexing
            )
            successful_files += 1
            logger.info(f"✓ Successfully processed: {csv_file.name}")
        except Exception as e:
            failed_files += 1
            logger.error(f"✗ Error processing {csv_file.name}: {str(e)}")
            # Continue with other files instead of stopping
            continue
    
    logger.info(f"Completed processing directory: {directory_path}")
    logger.info(f"Summary - Successful: {successful_files}, Failed: {failed_files}, Total: {len(csv_files)}")


def main():
    parser = argparse.ArgumentParser(
        description="Bulk upload companies from CSV file(s)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload from a single CSV file
  python upload_bulk.py /path/to/companies.csv
  
  # Upload from all CSV files in a directory
  python upload_bulk.py /path/to/csv/directory/
  
  # Upload with indexing enabled (slower but immediate search capability)
  python upload_bulk.py /path/to/companies.csv --enable-indexing
  
  # Enable debug logging
  python upload_bulk.py /path/to/companies.csv --log-level DEBUG
        """
    )
    
    parser.add_argument(
        "path",
        help="Path to CSV file or directory containing CSV files"
    )
    
    parser.add_argument(
        "--enable-indexing",
        action="store_true",
        help="Enable indexing during upload (slower but allows immediate searching)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set the logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Setup logging first
    log_filename = setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("STARTING BULK UPLOAD PROCESS")
    logger.info("="*60)
    logger.info(f"Arguments: {vars(args)}")
    logger.info(f"Log file: {log_filename}")
    
    try:
        # Initialize the bulk service
        logger.info("Initializing CompaniesBulkService...")
        bulk_service = CompaniesBulkService()
        
        path = Path(args.path)
        disable_indexing = not args.enable_indexing
        
        logger.info(f"Processing path: {args.path}")
        logger.info(f"Disable indexing: {disable_indexing}")
        
        if path.is_file():
            # Single file mode
            if not path.suffix.lower() == '.csv':
                error_msg = f"File must be a CSV file: {args.path}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            logger.info("Mode: Single file upload")
            process_single_file(bulk_service, str(path), disable_indexing)
            
        elif path.is_dir():
            # Directory mode
            logger.info("Mode: Directory upload (all CSV files)")
            process_directory(bulk_service, str(path), disable_indexing)
            
        else:
            error_msg = f"Path not found: {args.path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Show final collection info
        logger.info("="*60)
        logger.info("UPLOAD COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        
        logger.info("Retrieving final collection information...")
        info = bulk_service.get_collection_info()
        
        if 'companies' in info:
            companies_info = info['companies']
            if 'error' not in companies_info:
                logger.info(f"Companies collection: {companies_info['points_count']} companies")
            else:
                logger.error(f"Companies collection error: {companies_info['error']}")
        
        # Log vocab collections info
        if 'vocab_collections' in info:
            for col_name, col_info in info['vocab_collections'].items():
                if 'error' not in col_info:
                    logger.info(f"Vocab collection '{col_name}': {col_info['points_count']} entries")
                else:
                    logger.error(f"Vocab collection '{col_name}' error: {col_info['error']}")
        
        logger.info("="*60)
        logger.info(f"Process completed successfully. Full log available at: {log_filename}")
        
    except Exception as e:
        logger.error("="*60)
        logger.error("UPLOAD PROCESS FAILED")
        logger.error("="*60)
        logger.error(f"Error: {str(e)}", exc_info=True)
        logger.error(f"Full error details logged to: {log_filename}")
        sys.exit(1)


if __name__ == "__main__":
    main()