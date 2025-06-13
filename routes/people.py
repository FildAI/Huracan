# routes/people.py
from fastapi import APIRouter, HTTPException, Body, Query
from typing import List, Optional
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.services.people.search_service import PeopleSearchService, CompanySearchRequest
from core.qdrant_client import get_client
from core.indexes.companies import CompaniesIndex

router = APIRouter(prefix="/people", tags=["people"])

@router.get("/search")
async def search_people(
    query: str = Query(..., description="Search query for companies to find people"),
    limit: int = Query(100, ge=1, le=5000, description="Maximum number of people results to return"),
    companies_limit: int = Query(50, ge=1, le=500, description="Maximum number of companies to search for"),
    min_score: Optional[float] = Query(None, description="Optional match score threshold"),
    verbose: bool = Query(False, description="Enable verbose output for debugging")
):
    """
    Search for people by searching companies first, then finding people in those companies.
    
    Step 1: Search for companies using semantic search
    Step 2: Search for people working at those companies
    
    - **query**: The search query for companies (e.g., "AI startups in San Francisco")
    - **limit**: Maximum number of people results (1-5000, default: 100)
    - **companies_limit**: Maximum number of companies to search (1-500, default: 50)
    - **min_score**: Optional match score threshold for people results
    - **verbose**: Include debugging information in response
    """
    try:
        # Step 1: Search for companies
        qdrant_client = get_client()
        companies_index = CompaniesIndex(qdrant_client)
        companies_results = companies_index.search(query, limit=companies_limit, verbose=verbose)
        
        if not companies_results.points:
            return {
                "query": query,
                "results": {
                    "matched_companies": 0,
                    "results": [],
                    "count": 0
                },
                "companies_found": 0,
                "companies": []
            }
        
        # Extract company names from search results
        company_names = []
        companies_info = []
        for point in companies_results.points:
            # Assuming company name is in payload - adjust field name as needed
            company_name = point.payload.get('name') or point.payload.get('company_name') or point.payload.get('company')
            if company_name:
                company_names.append(company_name)
                companies_info.append({
                    "id": point.id,
                    "score": point.score,
                    "name": company_name,
                    "metadata": point.payload
                })
        
        if verbose:
            print(f"Found {len(company_names)} companies: {company_names}")
        
        # Step 2: Search for people in those companies
        people_service = PeopleSearchService()
        people_results = await people_service.search_people_by_companies(
            companies=company_names,
            limit=limit,
            min_score=min_score
        )
        
        return {
            "query": query,
            "results": people_results,
            "companies_found": len(companies_info),
            "companies": companies_info
        }
        
    except ConnectionError:
        raise HTTPException(status_code=503, detail="Could not connect to MongoDB")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/search/by-companies")
async def search_people_by_companies(
    request: CompanySearchRequest = Body(...)
):
    """
    Search for people by list of company names (case-insensitive)
    
    Request Body:
    - **companies**: List of company names to search for
    - **limit**: Maximum number of results (default: 100, max: 5000)
    - **min_score**: Optional match score threshold (if using scoring)
    """
    try:
        people_service = PeopleSearchService()
        return await people_service.search_people_by_companies(
            companies=request.companies,
            limit=request.limit,
            min_score=request.min_score
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ConnectionError:
        raise HTTPException(status_code=503, detail="Could not connect to MongoDB")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.get("/health")
async def people_health_check():
    """Health check for people search service"""
    try:
        people_service = PeopleSearchService()
        return await people_service.health_check()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")