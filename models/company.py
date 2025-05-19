from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import json


@dataclass
class Company:
    """Class representing a company entity."""
    
    name: str
    description: Optional[str] = None
    industry: Optional[str] = None
    country: Optional[str] = None
    website: Optional[str] = None
    founded_year: Optional[int] = None
    employee_count: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Company':
        """
        Create a Company instance from a dictionary.
        
        Args:
            data: Dictionary with company data
            
        Returns:
            Company instance
        """
        # Extract known fields
        name = data.pop("name", None) or data.pop("Name", None) or "Unknown"
        description = data.pop("description", None) or data.pop("Description", None)
        industry = data.pop("industry", None) or data.pop("Industry", None)
        country = data.pop("country", None) or data.pop("Country", None)
        website = data.pop("website", None) or data.pop("Website", None)
        
        # Try to convert founded_year to int
        founded_year = data.pop("founded_year", None) or data.pop("Founded Year", None)
        if founded_year:
            try:
                founded_year = int(founded_year)
            except (ValueError, TypeError):
                founded_year = None
        
        # Try to convert employee_count to int
        employee_count = data.pop("employee_count", None) or data.pop("Employees", None)
        if employee_count:
            try:
                employee_count = int(employee_count)
            except (ValueError, TypeError):
                employee_count = None
        
        # Store remaining fields in metadata
        return cls(
            name=name,
            description=description,
            industry=industry,
            country=country,
            website=website,
            founded_year=founded_year,
            employee_count=employee_count,
            metadata=data
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert Company instance to a dictionary.
        
        Returns:
            Dictionary representation of the company
        """
        result = {
            "name": self.name
        }
        
        # Add non-None fields
        if self.description:
            result["description"] = self.description
        if self.industry:
            result["industry"] = self.industry
        if self.country:
            result["country"] = self.country
        if self.website:
            result["website"] = self.website
        if self.founded_year:
            result["founded_year"] = self.founded_year
        if self.employee_count:
            result["employee_count"] = self.employee_count
        
        # Add metadata
        result.update(self.metadata)
        
        return result
    
    def get_searchable_text(self) -> str:
        """
        Get a text representation of the company for embedding.
        
        Returns:
            Text string combining important fields
        """
        parts = [self.name]
        
        if self.description:
            parts.append(self.description)
        if self.industry:
            parts.append(f"Industry: {self.industry}")
        if self.country:
            parts.append(f"Country: {self.country}")
        
        # Add selected metadata fields that might be relevant for search
        for key, value in self.metadata.items():
            if isinstance(value, (str, int, float)) and str(value).strip():
                parts.append(f"{key}: {value}")
        
        return " ".join(parts)
    
    def __str__(self) -> str:
        """Return string representation of the company."""
        return f"{self.name} ({self.industry or 'Unknown industry'}, {self.country or 'Unknown location'})" 