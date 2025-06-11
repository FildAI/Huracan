# routes/people.py
from fastapi import APIRouter, HTTPException, Body
from typing import List, Optional
import os
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
from pydantic import BaseModel
import math

router = APIRouter(prefix="/people", tags=["people"])

# Reuse your MongoDB connection pattern from upload_people_mongodb.py
def get_people_collection():
    client = MongoClient(os.getenv('MONGO_URI', 'mongodb://localhost:27017'))
    return client['people_db']['people']

# Add request model
class CompanySearchRequest(BaseModel):
    companies: List[str]
    limit: Optional[int] = 100
    min_score: Optional[float] = None

def sanitize_json(value):
    """Recursively sanitize NaN and Infinity values"""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    elif isinstance(value, dict):
        return {k: sanitize_json(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [sanitize_json(v) for v in value]
    return value

@router.post("/search/by-companies")
async def search_people_by_companies(
    request: CompanySearchRequest = Body(...)  # Changed to use request model
):
    """
    Search for people by list of company names (case-insensitive)
    
    Request Body:
    - **companies**: List of company names to search for
    - **limit**: Maximum number of results (default: 1000, max: 5000)
    - **min_score**: Optional match score threshold (if using scoring)
    """
    try:
        if not request.companies or len(request.companies) > 500:
            raise HTTPException(status_code=400, 
                detail="Provide 1-500 company names")
                
        # Clean and lowercase company names
        search_terms = [c.lower().strip() for c in request.companies if c.strip()]
        
        # Build query with optional score filtering
        query = {"company_lower": {"$in": search_terms}}
        if request.min_score is not None:
            query["match_score"] = {"$gte": request.min_score}

        # Get collection and execute search
        col = get_people_collection()
        results = col.find(query, {"_id": 0}).limit(min(request.limit, 5000))
        
        return {
            "matched_companies": len(search_terms),
            "results": [sanitize_json(doc) for doc in results],
            "count": col.count_documents(query)
        }

    except ServerSelectionTimeoutError:
        raise HTTPException(status_code=503, 
            detail="Could not connect to MongoDB")
    except Exception as e:
        raise HTTPException(status_code=500, 
            detail=f"Search failed: {str(e)}")

@router.get("/health")
async def people_health_check():
    """Health check for people search service"""
    try:
        col = get_people_collection()
        col.find_one()  # Simple connection test
        return {
            "status": "healthy",
            "stats": {
                "total_people": col.estimated_document_count(),
                "indexes": list(col.list_indexes())
            }
        }
    except Exception as e:
        raise HTTPException(status_code=503, 
            detail=f"Service unavailable: {str(e)}")