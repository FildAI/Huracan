"""Object-oriented company text extractors for embeddings."""

from typing import Callable, List, Optional, Dict, Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from ..types import Document, CompanyField

@dataclass
class CompanyFields:
    """Standard company field definitions."""
    REQUIRED_FIELDS = [
        "name",
        "description", 
        "primary_industry",
        "size",
        "type",
        "location",
        "country",
        "domain",
        "linkedin_url"
    ]
    
    ORDERED_FIELDS = [
        "name",
        "description",
        "primary_industry", 
        "type",
        "size",
        "location",
        "country",
        "domain",
        "linkedin_url"
    ]

    ORDERED_FIELDS_SET = set(ORDERED_FIELDS)
    
    FIELD_LABELS = {
        "name": "Company",
        "description": "Description", 
        "primary_industry": "Industry",
        "type": "Type",
        "size": "Size",
        "location": "Location",
        "country": "Country",
        "domain": "Website",
        "linkedin_url": "LinkedIn"
    }

class CompanyExtractor(ABC):
    """Base class for company text extractors."""
    
    def __init__(self, fields: Optional[CompanyFields] = None):
        self.fields = fields or CompanyFields()
    
    @abstractmethod
    def extract(self, document: Document) -> str:
        pass
    
    def __call__(self, document: Document) -> str:
        return self.extract(document)
    
    def _get_field_value(self, document: Document, field_name: str) -> str:
        value = document.get(field_name, "")
        return str(value).strip() if value is not None else ""
    
    def _has_field_value(self, document: Document, field_name: str) -> bool:
        value = self._get_field_value(document, field_name)
        return bool(value)
    
    def _get_fields_order(self, document: Document) -> List[str]:
        return [key for key in document.keys() if key in self.fields.ORDERED_FIELDS_SET]

class CompanyExtractorFull(CompanyExtractor):
    def extract(self, document: Document) -> str:
        result = ""
        for field in self._get_fields_order(document):
            result += self._get_field_value(document, field) + " "
        return result.strip()

class CompanyExtractorField(CompanyExtractor):
    def __init__(self, field: CompanyField):
        super().__init__()
        self.field = field

    def extract(self, document: Document) -> str:
        return self._get_field_value(document, self.field)
