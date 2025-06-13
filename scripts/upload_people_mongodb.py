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

load_dotenv()

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))



class CompanyPeopleDB:
    def __init__(self):
        self.client = MongoClient(os.getenv('MONGO_URI', 'mongodb://localhost:27017'))
        self.db = self.client['people_db']
        self.col = self.db['people']
        
    def upload(self, csv_path):
        df = pd.read_csv(csv_path).rename(columns={
            'Company Table Data (US-Software 1)': 'company',
            'LinkedIn Profile': 'linkedin'
        }).drop(columns=['Imported Profiles'])
        
        # Create search-optimized fields
        df['company_lower'] = df['company'].str.lower()
        df['domain'] = df['company'].str.extract(r'(?:www\.)?([^\.]+)\.com')
        
        # Batch upsert by company+name
        ops = [UpdateOne(
            {'company_lower': row['company_lower'], 'name': row['Full Name']},
            {'$set': row.to_dict()},
            upsert=True
        ) for _, row in df.iterrows()]
        
        self.col.bulk_write(ops)
        self.col.create_indexes([IndexModel([('company_lower', ASCENDING)])])
    
    def search(self, companies):
        search_terms = [c.lower().strip() for c in companies]
        return list(self.col.find({
            'company_lower': {'$in': search_terms}
        }, {'_id': 0}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path', help="Path to CSV file")
    args = parser.parse_args()
    
    db = CompanyPeopleDB()
    # db.upload(args.csv_path)
    # db.search(["Volato", "Hal Inc.", "Google"])
    print(f"Uploaded {Path(args.csv_path).name} to MongoDB") 