from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from .company import Company


@dataclass
class SearchResult:
    """Class representing a search result."""
    
    company: Company
    score: float
    match_details: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_qdrant_result(cls, result: Dict[str, Any]) -> 'SearchResult':
        """
        Create a SearchResult from a Qdrant search result.
        
        Args:
            result: Dictionary with Qdrant search result
            
        Returns:
            SearchResult instance
        """
        # Extract company data from payload
        company_data = result.get("metadata", {})
        company = Company.from_dict(company_data)
        
        # Extract score
        score = result.get("score", 0.0)
        
        return cls(
            company=company,
            score=score
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert SearchResult to a dictionary.
        
        Returns:
            Dictionary representation of the search result
        """
        result = {
            "company": self.company.to_dict(),
            "score": self.score
        }
        
        if self.match_details:
            result["match_details"] = self.match_details
        
        return result


@dataclass
class SearchResults:
    """Class representing a collection of search results."""
    
    results: List[SearchResult]
    query: str
    total_found: int
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert SearchResults to a dictionary.
        
        Returns:
            Dictionary representation of the search results
        """
        return {
            "query": self.query,
            "total_found": self.total_found,
            "results": [result.to_dict() for result in self.results],
            "metadata": self.metadata or {}
        }
    
    def get_companies(self) -> List[Company]:
        """
        Get list of companies from search results.
        
        Returns:
            List of Company instances
        """
        return [result.company for result in self.results] 