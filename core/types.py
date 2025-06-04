from dataclasses import dataclass

Document = dict
CompanyField = str

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