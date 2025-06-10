from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm
from qdrant_client import QdrantClient
import os
from pathlib import Path

from core.indexes.companies import CompaniesIndex
from core.indexes.vocab_bm25 import VocabBM25Index
from core.vector_config import CompanyVocabBM25Builder
from core.qdrant_client import get_client


class CompaniesBulkService:

    # Column mapping from CSV headers to expected field names
    CSV_COLUMN_MAPPING = {
        "Name": "name",
        "Description": "description", 
        "Primary Industry": "primary_industry",
        "Country": "country",
        "Size": "size",
        "Type": "type",
        "Location": "location",
        "Domain": "domain",
        "LinkedIn URL": "linkedin_url",
        "Imported Companies": "imported_companies"
    }
    
    # Required fields that must be present after mapping
    REQUIRED_FIELDS = ["name"]
    
    # Expected fields based on extractors in vector_config.py
    EXPECTED_FIELDS = ["name", "description", "primary_industry", "country", "size"]

    def __init__(self, client: QdrantClient = None, batch_size: int = 1000):
        self.client = client or get_client()
        self.batch_size = batch_size
        
        # Initialize the indexes
        self.companies_index = CompaniesIndex(self.client)
        self.vocab_index = VocabBM25Index(self.client, CompanyVocabBM25Builder)
        
    def validate_csv_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and map CSV columns to expected format.
        
        Args:
            df: Raw DataFrame from CSV
            
        Returns:
            DataFrame with properly mapped column names
            
        Raises:
            ValueError: If required columns are missing
        """
        print("Validating and mapping CSV columns...")
        
        # Get current column names
        current_columns = df.columns.tolist()
        print(f"CSV columns found: {current_columns}")
        
        # Create mapping for columns that exist in our mapping
        column_rename_map = {}
        mapped_columns = []
        
        for csv_col in current_columns:
            if csv_col in self.CSV_COLUMN_MAPPING:
                expected_col = self.CSV_COLUMN_MAPPING[csv_col]
                column_rename_map[csv_col] = expected_col
                mapped_columns.append(expected_col)
                print(f"Mapping: '{csv_col}' -> '{expected_col}'")
            else:
                # Keep unmapped columns as-is (but convert to lowercase and replace spaces with underscores)
                normalized_col = csv_col.lower().replace(' ', '_').replace('-', '_')
                column_rename_map[csv_col] = normalized_col
                mapped_columns.append(normalized_col)
                print(f"Normalizing: '{csv_col}' -> '{normalized_col}'")
        
        # Apply column renaming
        df_renamed = df.rename(columns=column_rename_map)

        df_renamed = df_renamed.drop(columns=['imported_companies'])
        
        # Check for required fields
        missing_required = [field for field in self.REQUIRED_FIELDS if field not in mapped_columns]
        if missing_required:
            raise ValueError(f"Required fields missing from CSV: {missing_required}. "
                           f"Available fields after mapping: {mapped_columns}")
        
        # Check for expected fields and warn if missing
        missing_expected = [field for field in self.EXPECTED_FIELDS if field not in mapped_columns]
        if missing_expected:
            print(f"Warning: Expected fields missing from CSV: {missing_expected}")
            print("These fields will be set to None/empty values.")
        
        print(f"Column validation successful. Mapped columns: {mapped_columns}")
        return df_renamed
    
    def preview_csv_mapping(self, csv_path: str, num_rows: int = 5) -> Dict[str, Any]:
        """
        Preview how CSV columns will be mapped and show sample data.
        
        Args:
            csv_path: Path to CSV file
            num_rows: Number of sample rows to show
            
        Returns:
            Dictionary with mapping info and sample data
        """
        print(f"Previewing CSV mapping for: {csv_path}")
        
        # Read just a few rows for preview
        df_preview = pd.read_csv(csv_path, nrows=num_rows * 2)  # Read extra in case of empty rows
        
        print(f"\nOriginal CSV structure:")
        print(f"Columns: {df_preview.columns.tolist()}")
        print(f"Shape: {df_preview.shape}")
        
        # Apply validation and mapping
        try:
            validated_df = self.validate_csv_columns(df_preview)
            
            print(f"\nAfter mapping:")
            print(f"Columns: {validated_df.columns.tolist()}")
            
            # Show sample data
            sample_data = validated_df.head(num_rows).to_dict(orient='records')
            
            return {
                'original_columns': df_preview.columns.tolist(),
                'mapped_columns': validated_df.columns.tolist(),
                'column_mapping': {orig: mapped for orig, mapped in zip(df_preview.columns, validated_df.columns)},
                'sample_data': sample_data,
                'validation_success': True
            }
            
        except Exception as e:
            return {
                'original_columns': df_preview.columns.tolist(),
                'error': str(e),
                'validation_success': False
            }
    
    def normalize_company_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize company data to ensure all expected fields exist.
        
        Args:
            data: Company data dictionary
            
        Returns:
            Normalized company data dictionary with all expected fields
        """
        normalized = data.copy()
        
        # Ensure all expected fields exist
        for field in self.EXPECTED_FIELDS:
            if field not in normalized:
                normalized[field] = None
        
        return normalized

    def sanitize_company_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize company data to ensure it can be serialized to JSON and used with Qdrant.
        
        Args:
            data: Raw company data dictionary
            
        Returns:
            Sanitized company data dictionary
        """
        # First normalize to ensure expected fields exist
        normalized_data = self.normalize_company_data(data)
        
        sanitized = {}
        
        for key, value in normalized_data.items():
            # Handle numpy types
            if isinstance(value, (np.generic,)):
                value = value.item()
            
            # Handle NaN, inf, -inf values
            if isinstance(value, float) and (pd.isna(value) or np.isinf(value)):
                sanitized[key] = None
            # Handle pandas NaType
            elif pd.isna(value):
                sanitized[key] = None
            # Convert everything else to string if it's not None
            elif value is not None:
                sanitized[key] = str(value).strip() if str(value).strip() else None
            else:
                sanitized[key] = None
                
        return sanitized
    
    def sanitize_companies_data(self, companies_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sanitize a list of company data dictionaries.
        
        Args:
            companies_data: List of raw company data dictionaries
            
        Returns:
            List of sanitized company data dictionaries
        """
        print("Sanitizing company data...")
        sanitized_data = []
        
        for company_data in tqdm(companies_data, desc="Sanitizing companies"):
            sanitized_company = self.sanitize_company_data(company_data)
            sanitized_data.append(sanitized_company)
        
        print(f"Sanitized {len(sanitized_data)} companies")
        return sanitized_data
        
    def setup_collections(self, force_recreate: bool = False):
        print("Setting up companies collection...")
        self.companies_index.create_collection(force_recreate=force_recreate)
        
        print("Setting up vocab collections...")
        self.vocab_index.create_vocab_indexes(force_recreate=force_recreate)
        
        print("Adding text indexes for BM25 search...")
        self.vocab_index.add_text_indexes_for_all_columns(force_recreate=force_recreate)
        
        print("Setup completed successfully!")
    
    def add_single_company(self, company_data: Dict[str, Any]):
        # Sanitize the data first
        sanitized_data = self.sanitize_company_data(company_data)
        
        # Create and insert company point
        company_point = self.companies_index.create_point(sanitized_data)
        self.companies_index.upsert_batch([company_point])
        
        # Add to vocab collections
        self.vocab_index.add_item_to_vocab(sanitized_data)
        
        print(f"Added company: {sanitized_data.get('name', 'Unknown')}")
    
    def add_companies_bulk(self, companies_data: List[Dict[str, Any]], disable_indexing: bool = True):
        if not companies_data:
            print("No company data provided")
            return
        
        print(f"Starting bulk insert of {len(companies_data)} companies...")
        
        # Sanitize all company data first
        sanitized_companies = self.sanitize_companies_data(companies_data)
        
        # Disable indexing for better bulk insert performance
        if disable_indexing:
            print("Disabling indexing for bulk insert...")
            self.companies_index.disable_indexing()
        
        try:
            # Process companies in batches (vocab is added during batch processing)
            self._process_companies_batches(sanitized_companies)
            
        finally:
            # Re-enable indexing
            if disable_indexing:
                print("Re-enabling indexing...")
                self.companies_index.enable_indexing()
        
        print(f"Bulk insert completed successfully! Added {len(sanitized_companies)} companies.")

    
    def add_companies_from_csv(self, csv_path: str, disable_indexing: bool = True):
        print(f"Loading companies from CSV: {csv_path}")
        # Use low_memory=False to avoid dtype warnings and handle mixed types properly
        df = pd.read_csv(csv_path, low_memory=False)
        print(f"Loaded {len(df)} companies from CSV")
        
        # Validate and map columns
        validated_df = self.validate_csv_columns(df)
        
        self.add_companies_from_dataframe(validated_df, disable_indexing=disable_indexing)

    def add_companies_from_dataframe(self, df: pd.DataFrame, disable_indexing: bool = True):
        # Convert to dict and let sanitization handle the data cleaning
        companies_data = df.to_dict(orient="records")
        self.add_companies_bulk(companies_data, disable_indexing=disable_indexing)
    
    def _process_companies_batches(self, companies_data: List[Dict[str, Any]]):
        total_batches = (len(companies_data) + self.batch_size - 1) // self.batch_size
        
        print(f"Processing {len(companies_data)} companies in {total_batches} batches of {self.batch_size}")
        
        for i in tqdm(range(0, len(companies_data), self.batch_size), desc="Processing company batches"):
            batch_data = companies_data[i:i + self.batch_size]
            
            # First, add batch data to vocab collections using bulk method
            self.vocab_index.add_vocab_items_bulk(batch_data)
            
            # Then create company points using efficient batch processing
            company_points = self.companies_index.vector_builder.get_batch_points(batch_data)
            
            # Bulk insert company points
            if company_points:
                self.companies_index.upsert_batch(company_points)
    
    def search_companies(self, query: str, limit: int = 10, verbose: bool = False):
        return self.companies_index.search(query, limit=limit, verbose=verbose)
    
    def get_collection_info(self):
        info = {}
        
        try:
            companies_info = self.client.get_collection(self.companies_index.collection_name)
            info['companies'] = {
                'name': companies_info.name,
                'points_count': companies_info.points_count,
                'status': companies_info.status
            }
        except Exception as e:
            info['companies'] = {'error': str(e)}
        
        # Get vocab collection information
        info['vocab_collections'] = {}
        for col in self.vocab_index.get_columns():
            vocab_collection_name = self.vocab_index.get_vocab_collection_name(col)
            try:
                vocab_info = self.client.get_collection(vocab_collection_name)
                info['vocab_collections'][col] = {
                    'name': vocab_info.name,
                    'points_count': vocab_info.points_count,
                    'status': vocab_info.status
                }
            except Exception as e:
                info['vocab_collections'][col] = {'error': str(e)}
        
        return info 

    def split_csv_into_chunks(self, csv_path: str, output_folder: str, max_chunk_size: int = 5000, chunk_read_size: int = 10000):

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        # Create output folder if it doesn't exist
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Splitting CSV file: {csv_path}")
        print(f"Output folder: {output_folder}")
        print(f"Max chunk size: {max_chunk_size} rows")
        
        chunk_files = []
        chunk_number = 1
        current_chunk_rows = []
        header_saved = False
        header = None
        
        try:
            # Read CSV in chunks to avoid loading entire file into memory
            csv_reader = pd.read_csv(csv_path, chunksize=chunk_read_size)
            
            for chunk_df in tqdm(csv_reader, desc="Processing CSV chunks"):
                # Validate and map columns for first chunk to get proper header
                if not header_saved:
                    validated_chunk = self.validate_csv_columns(chunk_df)
                    header = validated_chunk.columns.tolist()
                    header_saved = True
                else:
                    # Apply the same column mapping to subsequent chunks
                    validated_chunk = self.validate_csv_columns(chunk_df)
                
                # Convert chunk to list of dictionaries for easier handling
                chunk_rows = validated_chunk.to_dict(orient='records')
                
                # Process each row in the current chunk
                for row in chunk_rows:
                    current_chunk_rows.append(row)
                    
                    # If we've reached the max chunk size, save the chunk
                    if len(current_chunk_rows) >= max_chunk_size:
                        chunk_file_path = self._save_chunk_to_file(
                            current_chunk_rows, 
                            output_path, 
                            chunk_number, 
                            header
                        )
                        chunk_files.append(chunk_file_path)
                        
                        # Reset for next chunk
                        current_chunk_rows = []
                        chunk_number += 1
            
            # Save any remaining rows in the last chunk
            if current_chunk_rows:
                chunk_file_path = self._save_chunk_to_file(
                    current_chunk_rows, 
                    output_path, 
                    chunk_number, 
                    header
                )
                chunk_files.append(chunk_file_path)
        
        except Exception as e:
            print(f"Error processing CSV file: {str(e)}")
            raise
        
        print(f"Successfully split CSV into {len(chunk_files)} chunks")
        print(f"Chunk files saved in: {output_folder}")
        
        return chunk_files
    
    def _save_chunk_to_file(self, chunk_rows: List[Dict[str, Any]], output_path: Path, chunk_number: int, header: List[str]) -> str:
        """
        Save a chunk of rows to a CSV file.
        
        Args:
            chunk_rows: List of row dictionaries
            output_path: Path object for output directory
            chunk_number: Current chunk number
            header: List of column names
            
        Returns:
            Path to the saved chunk file
        """
        chunk_filename = f"chunk_{chunk_number:04d}.csv"
        chunk_file_path = output_path / chunk_filename
        
        # Convert back to DataFrame and save
        chunk_df = pd.DataFrame(chunk_rows, columns=header)
        chunk_df.to_csv(chunk_file_path, index=False)
        
        print(f"Saved chunk {chunk_number}: {chunk_file_path} ({len(chunk_rows)} rows)")
        return str(chunk_file_path)

    def process_chunked_csv_folder(self, chunks_folder: str, disable_indexing: bool = True):
        chunks_path = Path(chunks_folder)
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks folder not found: {chunks_folder}")
        
        # Get all CSV files in the folder
        chunk_files = sorted(chunks_path.glob("chunk_*.csv"))
        
        if not chunk_files:
            print(f"No chunk files found in {chunks_folder}")
            return
        
        print(f"Found {len(chunk_files)} chunk files to process")
        
        # Disable indexing for better bulk insert performance
        if disable_indexing:
            print("Disabling indexing for bulk insert...")
            self.companies_index.disable_indexing()
        
        try:
            total_companies = 0
            
            for chunk_file in tqdm(chunk_files, desc="Processing chunk files"):
                print(f"Processing chunk: {chunk_file.name}")
                
                # Read and process each chunk
                chunk_df = pd.read_csv(chunk_file)
                
                # Validate and map columns (chunks should already be properly formatted if created by split_csv_into_chunks)
                validated_df = self.validate_csv_columns(chunk_df)
                companies_data = validated_df.to_dict(orient="records")
                
                # Process this chunk's companies (vocab is added during batch processing)
                self._process_companies_batches(companies_data)
                total_companies += len(companies_data)
                
                print(f"Processed {len(companies_data)} companies from {chunk_file.name}")
            
        finally:
            # Re-enable indexing
            if disable_indexing:
                print("Re-enabling indexing...")
                self.companies_index.enable_indexing()
        
        print(f"Completed processing all chunks! Total companies added: {total_companies}") 