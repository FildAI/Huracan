from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

from .vector_store import VectorStore
from .query_parser import QueryParser
from .embeddings import BaseEmbedder, OpenAIEmbedder, BM25Embedder, SentenceTransformerShort


class SearchPipeline:
    """Class for orchestrating the search process."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        query_parser: QueryParser,
        available_fields: List[str],
        default_limit: int = 10
    ):
        """
        Initialize a SearchPipeline.
        
        Args:
            vector_store: VectorStore instance
            query_parser: QueryParser instance
            available_fields: List of fields available for filtering
            default_limit: Default number of results to return
        """
        self.vector_store = vector_store
        self.query_parser = query_parser
        self.available_fields = available_fields
        self.default_limit = default_limit
    
    @classmethod
    def create_default_pipeline(
        cls,
        collection_name: str,
        available_fields: List[str],
        openai_api_key: Optional[str] = None,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        default_limit: int = 10
    ) -> 'SearchPipeline':
        """
        Create a default search pipeline with standard embedders.
        
        Args:
            collection_name: Name of the Qdrant collection
            available_fields: List of fields available for filtering
            openai_api_key: OpenAI API key
            qdrant_url: Qdrant server URL
            qdrant_api_key: Qdrant API key
            default_limit: Default number of results to return
            
        Returns:
            Configured SearchPipeline instance
        """
        # Create embedders
        openai_embedder = OpenAIEmbedder(api_key=openai_api_key)
        sentence_embedder = SentenceTransformerShort()
        
        # Create vector store with multiple embedders
        vector_store = VectorStore(
            collection_name=collection_name,
            embedders={
                "dense": openai_embedder,
                "mini": sentence_embedder
            },
            url=qdrant_url,
            api_key=qdrant_api_key
        )
        
        # Create query parser
        query_parser = QueryParser(llm_api_key=openai_api_key)
        
        # Create and return pipeline
        return cls(
            vector_store=vector_store,
            query_parser=query_parser,
            available_fields=available_fields,
            default_limit=default_limit
        )
    
    def format_results(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Format search results as a pandas DataFrame.
        
        Args:
            results: List of search results
            
        Returns:
            DataFrame with results
        """
        if not results:
            return pd.DataFrame()
        
        # Extract metadata and scores
        rows = []
        for result in results:
            row = {**result["metadata"], "score": result["score"]}
            rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Reorder columns to put score first
        if "score" in df.columns:
            cols = ["score"] + [col for col in df.columns if col != "score"]
            df = df[cols]
        
        return df
    
    def search(
        self,
        query: str,
        limit: Optional[int] = None,
        embedder_name: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Search for documents matching a query.
        
        Args:
            query: Natural language query
            limit: Maximum number of results to return
            embedder_name: Which embedder to use for query vector
            
        Returns:
            Tuple of (results DataFrame, search metadata)
        """
        if limit is None:
            limit = self.default_limit
        
        # Parse query to get filter and keywords
        filter_obj, keywords = self.query_parser.parse_query(query, self.available_fields)
        
        # Log search parameters
        search_metadata = {
            "original_query": query,
            "parsed_keywords": keywords,
            "has_filters": filter_obj is not None,
            "embedder_used": embedder_name or "default"
        }
        
        print(f"Searching for: '{query}'")
        print(f"Using filters: {filter_obj is not None}")
        print(f"Keywords: {keywords}")
        
        # Execute search
        search_text = " ".join(keywords) if keywords else query
        results = self.vector_store.search(
            query=search_text,
            filter_conditions=filter_obj,
            limit=limit,
            embedder_name=embedder_name
        )
        
        print(f"Found {len(results)} results")
        
        # Format results
        results_df = self.format_results(results)
        
        return results_df, search_metadata
    
    def index_dataframe(self, df: pd.DataFrame, batch_size: int = 100) -> int:
        """
        Index a DataFrame into the vector store.
        
        Args:
            df: DataFrame to index
            batch_size: Batch size for indexing
            
        Returns:
            Number of documents indexed
        """
        # Ensure the collection exists
        self.vector_store.create_collection_if_not_exists()
        
        # Convert DataFrame to list of dictionaries
        records = df.to_dict(orient="records")
        
        # Index in batches
        total_indexed = 0
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            self.vector_store.upsert(batch)
            total_indexed += len(batch)
            print(f"Indexed {total_indexed}/{len(records)} records")
        
        return total_indexed 