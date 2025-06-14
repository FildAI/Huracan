#!/usr/bin/env python3
import os, sys, argparse
from pymongo import MongoClient, IndexModel, ASCENDING
from pymongo.operations import UpdateOne
import pandas as pd
from pathlib import Path
import os
import argparse
import logging
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
import sys
from tqdm import tqdm
import glob
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

load_dotenv()

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

class CompanyPeopleDB:
    def __init__(self):
        self.client = MongoClient(os.getenv('MONGO_URI', 'mongodb://localhost:27017'))
        self.db = self.client['people_db']
        self.col = self.db['people']
        self.lock = threading.Lock()
        
    def upload(self, csv_path, debug=False):
        try:
            if debug:
                print(f"🔍 DEBUG: Starting to process {csv_path}")
                print(f"🔍 DEBUG: File exists: {Path(csv_path).exists()}")
                print(f"🔍 DEBUG: File size: {Path(csv_path).stat().st_size} bytes")
            
            # Read CSV with more detailed error handling
            try:
                df = pd.read_csv(csv_path)
                if debug:
                    print(f"🔍 DEBUG: Initial CSV shape: {df.shape}")
                    print(f"🔍 DEBUG: Columns found: {list(df.columns)}")
            except Exception as e:
                print(f"❌ ERROR: Failed to read CSV {Path(csv_path).name}: {str(e)}")
                return -1  # Read error
            
            # Check if required columns exist
            required_columns = ['Company Name', 'Full Name']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"❌ ERROR: Missing required columns in {Path(csv_path).name}: {missing_columns}")
                print(f"   Available columns: {list(df.columns)}")
                return -1  # Column error
            
            # Rename columns
            df = df.rename(columns={
                'Company Name': 'company',
                'LinkedIn Profile': 'linkedin'
            })
            
            if debug:
                print(f"🔍 DEBUG: After renaming columns: {list(df.columns)}")
            
            # Drop columns that exist
            columns_to_drop = [col for col in ['Imported Profiles'] if col in df.columns]
            if columns_to_drop:
                df = df.drop(columns=columns_to_drop)
                if debug:
                    print(f"🔍 DEBUG: Dropped columns: {columns_to_drop}")
            
            # Skip empty files
            if df.empty:
                print(f"⚠️  Skipping empty file: {Path(csv_path).name}")
                return 0  # Empty file (successful skip)
            
            # Check for missing required data
            null_companies = df['company'].isnull().sum()
            null_names = df['Full Name'].isnull().sum()
            if debug:
                print(f"🔍 DEBUG: Null companies: {null_companies}, Null names: {null_names}")
            
            # Filter out rows with missing essential data
            initial_count = len(df)
            df = df.dropna(subset=['company', 'Full Name'])
            final_count = len(df)
            
            if debug:
                print(f"🔍 DEBUG: Filtered from {initial_count} to {final_count} rows")
            
            if df.empty:
                print(f"⚠️  No valid data after filtering in {Path(csv_path).name}")
                return 0  # No valid data (successful skip)
            
            # Create search-optimized fields
            df['company_lower'] = df['company'].str.lower()
            df['domain'] = df['company'].str.extract(r'(?:www\.)?([^\.]+)\.com')
            
            if debug:
                print(f"🔍 DEBUG: Sample company_lower values: {df['company_lower'].head(3).tolist()}")
            
            # Batch upsert by company+name
            ops = []
            failed_ops = 0
            for idx, row in df.iterrows():
                try:
                    op = UpdateOne(
                        {'company_lower': row['company_lower'], 'name': row['Full Name']},
                        {'$set': row.to_dict()},
                        upsert=True
                    )
                    ops.append(op)
                except Exception as e:
                    failed_ops += 1
                    if debug:
                        print(f"🔍 DEBUG: Failed to create operation for row {idx}: {str(e)}")
            
            if debug:
                print(f"🔍 DEBUG: Created {len(ops)} operations, {failed_ops} failed")
            
            if not ops:
                print(f"❌ ERROR: No valid operations created for {Path(csv_path).name}")
                return -1  # Processing error
            
            # Execute bulk write
            try:
                with self.lock:
                    result = self.col.bulk_write(ops)
                    self.col.create_indexes([IndexModel([('company_lower', ASCENDING)])])
                    
                if debug:
                    print(f"🔍 DEBUG: Bulk write result - Inserted: {result.inserted_count}, "
                          f"Modified: {result.modified_count}, Upserted: {result.upserted_count}")
                
                return len(ops)  # Successful processing
                
            except Exception as e:
                print(f"❌ ERROR: Database operation failed for {Path(csv_path).name}: {str(e)}")
                return -1  # Database error
            
        except Exception as e:
            print(f"❌ ERROR: Unexpected error processing {Path(csv_path).name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return -1  # Unexpected error
    
    def process_single_file(self, csv_file, processed_files, completed_dir, debug=False):
        """Process a single CSV file and move it to completed directory"""
        try:
            file_key = str(csv_file.relative_to(csv_file.parents[1]))  # Get relative path for tracking
            
            # Skip if already processed
            if file_key in processed_files:
                return 0, f"⏭️  Skipping already processed: {csv_file.name}"
            
            if debug:
                print(f"🔍 DEBUG: Processing file key: {file_key}")
            
            records_processed = self.upload(csv_file, debug=debug)
            
            if records_processed > 0:
                # Successfully processed with records
                # Move file to completed directory
                completed_dir.mkdir(parents=True, exist_ok=True)
                completed_file = completed_dir / csv_file.name
                
                # Handle duplicate names in completed directory
                counter = 1
                while completed_file.exists():
                    stem = csv_file.stem
                    suffix = csv_file.suffix
                    completed_file = completed_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
                
                shutil.move(str(csv_file), str(completed_file))
                
                # Update processed files tracking
                processed_files[file_key] = {
                    'processed_at': datetime.now().isoformat(),
                    'records_count': records_processed,
                    'completed_path': str(completed_file),
                    'status': 'processed'
                }
                
                return records_processed, f"✅ Successfully processed {records_processed} records"
            
            elif records_processed == 0:
                # Empty file or no valid data - successfully skipped
                # Move file to completed directory but mark as skipped
                completed_dir.mkdir(parents=True, exist_ok=True)
                completed_file = completed_dir / csv_file.name
                
                counter = 1
                while completed_file.exists():
                    stem = csv_file.stem
                    suffix = csv_file.suffix
                    completed_file = completed_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
                
                shutil.move(str(csv_file), str(completed_file))
                
                # Update processed files tracking
                processed_files[file_key] = {
                    'processed_at': datetime.now().isoformat(),
                    'records_count': 0,
                    'completed_path': str(completed_file),
                    'status': 'skipped_empty'
                }
                
                return 0, f"⚠️  Successfully skipped empty/invalid file: {csv_file.name}"
            
            else:
                # records_processed < 0 indicates an actual error
                return records_processed, f"❌ Failed to process: {csv_file.name} (processing error)"
                
        except Exception as e:
            print(f"❌ ERROR: Exception in process_single_file for {csv_file.name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return -1, f"❌ Error processing {csv_file.name}: {str(e)}"
    
    def upload_directory(self, directory_path, max_workers=5, debug=False):
        """Upload all CSV files from directory using multithreading"""
        directory = Path(directory_path)
        completed_dir = directory / "completed"
        progress_file = directory / "upload_progress.json"
        
        # Load previous progress
        processed_files = {}
        if progress_file.exists():
            try:
                with open(progress_file, 'r') as f:
                    processed_files = json.load(f)
                print(f"📋 Loaded progress: {len(processed_files)} files already processed")
            except Exception as e:
                print(f"⚠️  Could not load progress file: {e}")
        
        # Find all CSV files recursively, excluding completed directory
        csv_files = []
        for csv_file in directory.rglob("*.csv"):
            if "completed" not in csv_file.parts:
                csv_files.append(csv_file)
        
        if not csv_files:
            print(f"No CSV files found in {directory_path} (excluding completed folder)")
            return
        
        print(f"Found {len(csv_files)} CSV files to process")
        
        total_records = 0
        successful_files = 0
        skipped_files = 0
        failed_files = 0
        
        # Process files with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.process_single_file, csv_file, processed_files, completed_dir, debug): csv_file
                for csv_file in csv_files
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(csv_files), desc="Processing CSV files", unit="file") as pbar:
                for future in as_completed(future_to_file):
                    csv_file = future_to_file[future]
                    try:
                        records_processed, message = future.result()
                        print(f"\n📁 {csv_file.name}: {message}")
                        
                        if records_processed > 0:
                            # Successfully processed with data
                            total_records += records_processed
                            successful_files += 1
                            
                            # Save progress after each successful file
                            with open(progress_file, 'w') as f:
                                json.dump(processed_files, f, indent=2)
                                
                        elif records_processed == 0:
                            # Successfully skipped (empty file)
                            skipped_files += 1
                            
                            # Save progress after each file (including skipped)
                            with open(progress_file, 'w') as f:
                                json.dump(processed_files, f, indent=2)
                        
                        else:
                            # Failed to process (records_processed < 0)
                            failed_files += 1
                            
                    except Exception as e:
                        print(f"\n❌ Error with {csv_file.name}: {str(e)}")
                        failed_files += 1
                    
                    pbar.update(1)
        
        print(f"\n🎉 Summary:")
        print(f"   - Total files found: {len(csv_files)}")
        print(f"   - Successfully processed: {successful_files}")
        print(f"   - Skipped (empty/invalid): {skipped_files}")
        print(f"   - Failed: {failed_files}")
        print(f"   - Total records processed: {total_records}")
        print(f"   - Progress saved to: {progress_file}")
        print(f"   - Completed files moved to: {completed_dir}")
    
    def search(self, companies):
        search_terms = [c.lower().strip() for c in companies]
        return list(self.col.find({
            'company_lower': {'$in': search_terms}
        }, {'_id': 0}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Upload people data from CSV files to MongoDB")
    parser.add_argument('path', help="Path to CSV file or directory containing CSV files")
    parser.add_argument('--single', action='store_true', help="Process single file instead of directory")
    parser.add_argument('--threads', type=int, default=5, help="Number of threads for processing (default: 5)")
    parser.add_argument('--debug', action='store_true', help="Enable debug output")
    args = parser.parse_args()
    
    db = CompanyPeopleDB()
    
    path = Path(args.path)
    
    if args.single or path.is_file():
        # Process single file
        records = db.upload(args.path, debug=args.debug)
        print(f"Uploaded {records} records from {path.name} to MongoDB")
    else:
        # Process directory
        db.upload_directory(args.path, max_workers=args.threads, debug=args.debug) 