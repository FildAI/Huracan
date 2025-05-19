from typing import Dict, List, Any, Tuple, Optional
import re
from qdrant_client.http import models as m
from openai import OpenAI
import os


class QueryParser:
    """Class for parsing natural language queries into structured filters."""
    
    def __init__(self, llm_api_key: Optional[str] = None, model: str = "gpt-4-turbo"):
        """
        Initialize a QueryParser.
        
        Args:
            llm_api_key: OpenAI API key (defaults to env variable OPENAI_API_KEY)
            model: OpenAI model to use for parsing
        """
        self.api_key = llm_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required for QueryParser")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
    
    def extract_keywords(self, query: str) -> List[str]:
        """
        Extract important keywords from a query for semantic search.
        
        Args:
            query: Natural language query
            
        Returns:
            List of important keywords
        """
        # Simple keyword extraction without LLM
        # Remove common words and keep only potential keywords
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about',
                     'from', 'of', 'as', 'that', 'this', 'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should', 'can',
                     'could', 'may', 'might', 'must', 'give', 'me', 'find', 'show', 'tell', 'which', 'who', 'where'}
        
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    def parse_with_llm(self, query: str, available_fields: List[str]) -> Dict[str, Any]:
        """
        Use LLM to parse a query into structured filters.
        
        Args:
            query: Natural language query
            available_fields: List of available fields that can be filtered
            
        Returns:
            Dictionary with extracted filter conditions
        """
        system_prompt = f"""
        You are a helpful assistant that extracts structured filter conditions from natural language queries.
        Given a query about companies, extract specific filter conditions that can be applied to a database.
        
        Available fields for filtering: {', '.join(available_fields)}
        
        For each field that appears in the query, extract:
        1. The field name
        2. The condition type (equals, contains, greater than, less than, etc.)
        3. The value to filter by
        
        Return JSON with this structure:
        {{
            "filters": [
                {{
                    "field": "field_name",
                    "condition": "equals|contains|gt|lt",
                    "value": "value"
                }}
            ],
            "keywords": ["keyword1", "keyword2"]
        }}
        
        If no explicit filters are found, return an empty filters list but extract relevant keywords.
        """
        
        user_prompt = f"Query: {query}"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            parsed_json = response.choices[0].message.content
            import json
            result = json.loads(parsed_json)
            return result
        except Exception as e:
            print(f"Error parsing query with LLM: {e}")
            # Fallback to simple keyword extraction
            return {
                "filters": [],
                "keywords": self.extract_keywords(query)
            }
    
    def build_qdrant_filter(self, parsed_query: Dict[str, Any]) -> Optional[m.Filter]:
        """
        Build a Qdrant filter from parsed query.
        
        Args:
            parsed_query: Dict with filters and keywords
            
        Returns:
            Qdrant filter object or None if no filters
        """
        if not parsed_query.get("filters"):
            return None
        
        conditions = []
        
        for filter_item in parsed_query["filters"]:
            field = filter_item.get("field")
            condition_type = filter_item.get("condition")
            value = filter_item.get("value")
            
            if not all([field, condition_type, value]):
                continue
                
            if condition_type == "equals":
                conditions.append(
                    m.FieldCondition(
                        key=field,
                        match=m.MatchValue(value=value)
                    )
                )
            elif condition_type == "contains":
                conditions.append(
                    m.FieldCondition(
                        key=field,
                        match=m.MatchText(text=value)
                    )
                )
            elif condition_type == "gt":
                try:
                    numeric_value = float(value)
                    conditions.append(
                        m.FieldCondition(
                            key=field,
                            range=m.Range(gt=numeric_value)
                        )
                    )
                except (ValueError, TypeError):
                    continue
            elif condition_type == "lt":
                try:
                    numeric_value = float(value)
                    conditions.append(
                        m.FieldCondition(
                            key=field,
                            range=m.Range(lt=numeric_value)
                        )
                    )
                except (ValueError, TypeError):
                    continue
        
        if not conditions:
            return None
            
        return m.Filter(
            must=conditions
        )
    
    def parse_query(self, query: str, available_fields: List[str]) -> Tuple[Optional[m.Filter], List[str]]:
        """
        Parse a natural language query into a Qdrant filter and keywords.
        
        Args:
            query: Natural language query
            available_fields: List of available fields that can be filtered
            
        Returns:
            Tuple of (Qdrant filter object, list of keywords)
        """
        # Get parsed query with LLM
        parsed_query = self.parse_with_llm(query, available_fields)
        
        # Build Qdrant filter
        filter_obj = self.build_qdrant_filter(parsed_query)
        
        # Get keywords
        keywords = parsed_query.get("keywords", self.extract_keywords(query))
        
        return filter_obj, keywords 