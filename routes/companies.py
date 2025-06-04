from fastapi import APIRouter, Query, HTTPException
from typing import Optional
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.qdrant_client import get_client
from core.indexes.companies import CompaniesIndex

router = APIRouter(prefix="/companies", tags=["companies"])

@router.get("/search")
async def search_companies(
    query: str = Query(..., description="Search query for companies"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of results to return"),
    verbose: bool = Query(False, description="Enable verbose output for debugging")
):
    """
    Search for companies using semantic search with BM25 filtering.
    
    - **query**: The search query (e.g., "AI companies in the US")
    - **limit**: Maximum number of results (1-100, default: 10)
    - **verbose**: Include debugging information in response
    """
    try:
        client = get_client()
        companies_index = CompaniesIndex(client)
        results = companies_index.search(query, limit=limit, verbose=verbose)
        
        # Format results for API response
        formatted_results = {
            "query": query,
            "limit": limit,
            "total_results": len(results.points) if results.points else 0,
            "results": []
        }
        
        if results.points:
            for point in results.points:
                result_item = {
                    "id": point.id,
                    "score": point.score,
                    "metadata": point.payload
                }
                formatted_results["results"].append(result_item)
        
        return formatted_results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.get("/health")
async def health_check():
    """
    Health check endpoint for the companies search service.
    """
    try:
        client = get_client()
        companies_index = CompaniesIndex(client)
        # Try to access the collection to verify connection
        collection_info = client.get_collection(companies_index.collection_name)
        return {
            "status": "healthy",
            "collection": companies_index.collection_name,
            "vectors_count": collection_info.vectors_count
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}") 