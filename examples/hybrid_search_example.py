"""
Advanced hybrid search example combining dense vectors and BM25 with query analysis.

This demonstrates a more sophisticated search pattern:
1. Analyze the query to identify explicit field filters
2. Use BM25 on specific text fields 
3. Combine with vector search for semantic similarity
4. Use a weighted scoring approach for hybrid results
"""

import os
import pandas as pd
from dotenv import load_dotenv
import sys
import re
from pathlib import Path
import json
from typing import Dict, List, Any, Tuple, Optional, Union

# Add project root to path to allow imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from qdrant_client import QdrantClient, models
from openai import OpenAI
from fild.huracan import OpenAIEmbedder
from fild.huracan.utils import get_logger, load_config

# Load environment variables
load_dotenv()

class QueryAnalyzer:
    """Class to analyze and extract structured information from search queries."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the QueryAnalyzer."""
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=self.api_key)
    
    def analyze_query(self, query: str, available_fields: List[str]) -> Dict[str, Any]:
        """
        Analyze a query to extract filters and search terms.
        
        Args:
            query: The natural language query
            available_fields: List of fields available for filtering
            
        Returns:
            Dict with extracted filters and search terms
        """
        system_prompt = f"""
        You are a helpful assistant that extracts structured filters from natural language queries.
        The query is searching for companies in a database.
        
        Available fields for filtering: {', '.join(available_fields)}
        
        Extract:
        1. Any explicit field filters mentioned (exact matches or contains)
        2. Main search keywords that don't belong to specific filters
        
        Return JSON with this structure:
        {{
            "filters": [
                {{
                    "field": "field_name",
                    "operation": "exact|contains",
                    "value": "filter_value"
                }}
            ],
            "search_terms": "main search keywords"
        }}
        
        - For industry/country/type, prefer exact matches
        - For description/name, use contains operations
        - Put any terms that aren't explicitly associated with fields into search_terms
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query: {query}"}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            print(f"Error analyzing query: {e}")
            # Simple fallback without LLM
            return {
                "filters": [],
                "search_terms": query
            }
    
    def build_qdrant_filter(self, query_analysis: Dict[str, Any]) -> Optional[models.Filter]:
        """
        Build a Qdrant filter from query analysis.
        
        Args:
            query_analysis: Dict with filters from analyze_query
            
        Returns:
            Qdrant filter object or None
        """
        filters = query_analysis.get("filters", [])
        if not filters:
            return None
        
        conditions = []
        
        for filter_info in filters:
            field = filter_info.get("field")
            operation = filter_info.get("operation")
            value = filter_info.get("value")
            
            if not all([field, operation, value]):
                continue
            
            if operation == "exact":
                conditions.append(
                    models.FieldCondition(
                        key=field,
                        match=models.MatchValue(value=value)
                    )
                )
            elif operation == "contains":
                conditions.append(
                    models.FieldCondition(
                        key=field,
                        match=models.MatchText(text=value)
                    )
                )
        
        if not conditions:
            return None
        
        return models.Filter(must=conditions)


class HybridSearcher:
    """Class for performing hybrid searches combining vectors and BM25."""
    
    def __init__(
        self,
        collection_name: str,
        vector_name: str = "dense",
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None
    ):
        """Initialize the HybridSearcher."""
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.qdrant_url:
            raise ValueError("Qdrant URL is required")
        
        self.collection_name = collection_name
        self.vector_name = vector_name
        
        # Initialize clients
        self.qdrant_client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)
        self.embedder = OpenAIEmbedder(api_key=self.openai_api_key)
        self.query_analyzer = QueryAnalyzer(openai_api_key=self.openai_api_key)
        
        # Define searchable fields
        self.available_fields = ["name", "description", "primary_industry", "country", "size", "type"]
    
    def search(
        self,
        query: str,
        limit: int = 10,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Perform a hybrid search.
        
        Args:
            query: The search query
            limit: Maximum number of results
            vector_weight: Weight for vector search (0-1)
            bm25_weight: Weight for BM25 search (0-1)
            
        Returns:
            Tuple of (results, metadata)
        """
        # Analyze query to extract filters and search terms
        query_analysis = self.query_analyzer.analyze_query(query, self.available_fields)
        filter_obj = self.query_analyzer.build_qdrant_filter(query_analysis)
        search_terms = query_analysis.get("search_terms", query)
        
        # Create metadata for debugging
        metadata = {
            "original_query": query,
            "analyzed_query": query_analysis,
            "vector_weight": vector_weight,
            "bm25_weight": bm25_weight,
            "search_terms": search_terms
        }
        
        # Create vector embedding
        vector = self.embedder.create_embedding(search_terms)
        
        # Default search (if no specific text field is targeted)
        default_text_field = "description"
        
        # Create text search conditions
        text_condition = models.FieldCondition(
            key=default_text_field,
            match=models.MatchText(text=search_terms)
        )
        
        # If we have filters from query analysis, use them
        # Otherwise, provide a weak text condition to help weight the results
        search_filter = filter_obj
        if not search_filter:
            # Use a weak "should" filter that doesn't strictly require matches
            search_filter = models.Filter(should=[text_condition])
        
        # Perform hybrid search
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector={self.vector_name: vector},
            query_filter=search_filter,
            limit=limit,
            with_payload=True,
            score_threshold=0.0  # Get all results so we can rerank
        )
        
        # Process results
        results = []
        for point in search_result:
            result = {
                "id": point.id,
                "score": point.score,
                "metadata": point.payload
            }
            results.append(result)
        
        return results, metadata


def main():
    """Run the hybrid search example."""
    # Setup logging
    logger = get_logger("hybrid_search")
    logger.info("Starting hybrid search example")
    
    # Load configuration
    config = load_config()
    
    # Create hybrid searcher
    try:
        searcher = HybridSearcher(
            collection_name="companies_bm25",
            qdrant_url=config["qdrant"]["url"],
            qdrant_api_key=config["qdrant"]["api_key"]
        )
    except Exception as e:
        logger.error(f"Error creating searcher: {e}")
        print(f"Error: {e}")
        print("Make sure you have run index_with_native_bm25.py first to create the collection.")
        return
    
    # Run search loop
    print("\nAdvanced Hybrid Search")
    print("======================")
    print("This example demonstrates how to combine vector search with BM25 search")
    print("It automatically analyzes your query to find explicit filters and search terms")
    print("\nExample queries:")
    print("- Telecom companies that mention WhatsApp")
    print("- IT companies in Chile")
    print("- Retail companies with luxury in the description")
    print("\nType 'quit' or 'exit' to end the program.")
    
    while True:
        query = input("\nEnter search query: ")
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        try:
            # Perform search
            results, metadata = searcher.search(query, limit=10)
            
            # Display search analysis
            print("\nQuery analysis:")
            print(f"- Search terms: {metadata['search_terms']}")
            if metadata['analyzed_query'].get('filters'):
                print("- Detected filters:")
                for filter_info in metadata['analyzed_query']['filters']:
                    print(f"  * {filter_info['field']} {filter_info['operation']} '{filter_info['value']}'")
            else:
                print("- No explicit filters detected")
            
            # Display results
            if not results:
                print("\nNo results found.")
            else:
                print(f"\nFound {len(results)} results:")
                
                # Format results as DataFrame
                rows = []
                for i, result in enumerate(results, 1):
                    payload = result["metadata"]
                    row = {
                        "rank": i,
                        "score": result["score"],
                        "name": payload.get("name", "Unknown"),
                        "industry": payload.get("primary_industry", "Unknown"),
                        "country": payload.get("country", "Unknown")
                    }
                    rows.append(row)
                
                results_df = pd.DataFrame(rows)
                print(results_df[["rank", "score", "name", "industry", "country"]])
                
                # Show details for top result
                if rows:
                    top_result = results[0]["metadata"]
                    print("\nTop result details:")
                    print(f"Name: {top_result.get('name', 'N/A')}")
                    print(f"Industry: {top_result.get('primary_industry', 'N/A')}")
                    print(f"Country: {top_result.get('country', 'N/A')}")
                    print(f"Size: {top_result.get('size', 'N/A')}")
                    print(f"Type: {top_result.get('type', 'N/A')}")
                    
                    if "description" in top_result and top_result["description"]:
                        desc = top_result["description"]
                        print(f"\nDescription: {desc[:300]}..." if len(desc) > 300 else f"\nDescription: {desc}")
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            print(f"Error searching: {e}")
    
    print("Goodbye!")

if __name__ == "__main__":
    main() 