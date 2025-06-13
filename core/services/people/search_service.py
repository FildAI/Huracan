import os
import math
from typing import List, Optional
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
from pydantic import BaseModel


class CompanySearchRequest(BaseModel):
    companies: List[str]
    limit: Optional[int] = 100
    min_score: Optional[float] = None


class PeopleSearchService:
    """Service for searching people by companies"""
    
    def __init__(self):
        self.mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
    
    def get_people_collection(self):
        """Get MongoDB people collection"""
        client = MongoClient(self.mongo_uri)
        return client['people_db']['people']
    
    def sanitize_json(self, value):
        """Recursively sanitize NaN and Infinity values"""
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return None
            return value
        elif isinstance(value, dict):
            return {k: self.sanitize_json(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self.sanitize_json(v) for v in value]
        return value
    
    async def search_people_by_companies(
        self, 
        companies: List[str], 
        limit: int = 100, 
        min_score: Optional[float] = None
    ) -> dict:
        """
        Search for people by list of company names (case-insensitive)
        
        Args:
            companies: List of company names to search for
            limit: Maximum number of results (default: 100, max: 5000)
            min_score: Optional match score threshold (if using scoring)
            
        Returns:
            Dict with matched_companies, results, and count
        """
        try:
            if not companies or len(companies) > 500:
                raise ValueError("Provide 1-500 company names")
                
            # Clean and lowercase company names
            search_terms = [c.lower().strip() for c in companies if c.strip()]
            
            # Build query with optional score filtering
            query = {"company_lower": {"$in": search_terms}}
            if min_score is not None:
                query["match_score"] = {"$gte": min_score}

            # Get collection and execute search
            col = self.get_people_collection()
            results = col.find(query, {"_id": 0}).limit(min(limit, 5000))
            
            return {
                "matched_companies": len(search_terms),
                "results": [self.sanitize_json(doc) for doc in results],
                "count": col.count_documents(query)
            }

        except ServerSelectionTimeoutError:
            raise ConnectionError("Could not connect to MongoDB")
        except Exception as e:
            raise RuntimeError(f"Search failed: {str(e)}")
    
    async def health_check(self) -> dict:
        """Health check for people search service"""
        try:
            col = self.get_people_collection()
            col.find_one()  # Simple connection test
            return {
                "status": "healthy",
                "stats": {
                    "total_people": col.estimated_document_count(),
                    "indexes": list(col.list_indexes())
                }
            }
        except Exception as e:
            raise RuntimeError(f"Service unavailable: {str(e)}") 