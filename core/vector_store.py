from typing import List, Dict, Any, Optional, Union, Tuple
import hashlib
import json
import os
from qdrant_client import QdrantClient
from qdrant_client.http import models as m
from qdrant_client.http.models import Distance, VectorParams, FieldCondition
import pandas as pd

from .embeddings import BaseEmbedder, OpenAIEmbedder, BM25Embedder


class VectorStore:
    """Main class for interacting with Qdrant vector database."""
    
    def __init__(
        self,
        collection_name: str,
        embedders: Dict[str, BaseEmbedder],
        url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize a VectorStore.
        
        Args:
            collection_name: Name of the Qdrant collection
            embedders: Dictionary mapping vector field names to embedders
            url: Qdrant server URL (defaults to env variable QDRANT_URL)
            api_key: Qdrant API key (defaults to env variable QDRANT_API_KEY)
        """
        self.collection_name = collection_name
        self.embedders = embedders
        
        self.url = url or os.getenv("QDRANT_URL")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        
        if not self.url:
            raise ValueError("Qdrant URL must be provided or set as QDRANT_URL env variable")
            
        self.client = QdrantClient(url=self.url, api_key=self.api_key)
        
    def create_collection_if_not_exists(self, vector_size: Dict[str, int] = None):
        """
        Create the collection if it doesn't exist.
        
        Args:
            vector_size: Dictionary mapping vector field names to their sizes
        """
        # Get default vector sizes if not provided
        if vector_size is None:
            vector_size = {}
            # Default dimensionality for OpenAI embeddings
            for name, embedder in self.embedders.items():
                if isinstance(embedder, OpenAIEmbedder):
                    vector_size[name] = 1536  # Default for text-embedding-3-small
        
        # Check if collection exists
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if self.collection_name not in collection_names:
            # Create vector params for each embedder
            vectors_config = {}
            for name, size in vector_size.items():
                vectors_config[name] = VectorParams(
                    size=size,
                    distance=Distance.COSINE
                )
            
            # Create the collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=vectors_config,
                sparse_vectors_config={
                    "text": {}  # Enable sparse BM25 vectors
                }
            )
            
            print(f"Created collection '{self.collection_name}' with vectors: {list(vectors_config.keys())}")
        else:
            print(f"Collection '{self.collection_name}' already exists")
    
    def delete_collection(self):
        """Delete the collection if it exists."""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            print(f"Collection '{self.collection_name}' deleted")
        except Exception as e:
            print(f"Error deleting collection: {e}")
    
    def _generate_id(self, data: Dict) -> str:
        """
        Generate a deterministic ID for a data point.
        
        Args:
            data: Data to generate ID for
            
        Returns:
            SHA256 hash of the data
        """
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode('utf-8')).hexdigest()
    
    def _create_vectors(self, text: str) -> Dict[str, List[float]]:
        """
        Create vectors for all embedders.
        
        Args:
            text: Text to create vectors for
            
        Returns:
            Dictionary mapping vector names to embeddings
        """
        vectors = {}
        
        for name, embedder in self.embedders.items():
            embedding = embedder.create_embedding(text)
            if embedding:
                vectors[name] = embedding
        
        return vectors
    
    def upsert(self, data: List[Dict[str, Any]]):
        """
        Upsert data points into the collection.
        
        Args:
            data: List of data points to upsert
        """
        points = []
        batch_size = 100
        
        for item in data:
            # Extract text for embedding from all fields
            text_values = " ".join(str(v) for v in item.values() if pd.notna(v) and v)
            
            # Create embeddings
            vectors = self._create_vectors(text_values)
            
            # Create point ID
            point_id = self._generate_id(item)
            
            # Create point
            points.append(
                m.PointStruct(
                    id=point_id,
                    payload=item,
                    vector=vectors
                )
            )
            
            # Upsert in batches
            if len(points) >= batch_size:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                print(f"Upserted batch of {len(points)} points")
                points = []
        
        # Upsert remaining points
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            print(f"Upserted final batch of {len(points)} points")
    
    def search(
        self, 
        query: str, 
        filter_conditions: Optional[m.Filter] = None,
        limit: int = 10,
        embedder_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search the collection.
        
        Args:
            query: Query text
            filter_conditions: Qdrant filter conditions
            limit: Maximum number of results to return
            embedder_name: Which embedder to use for query vector (defaults to first available)
            
        Returns:
            List of search results with metadata and scores
        """
        if embedder_name is None:
            embedder_name = next(iter(self.embedders.keys()))
        
        embedder = self.embedders.get(embedder_name)
        if not embedder:
            raise ValueError(f"Embedder '{embedder_name}' not found")
        
        # Create query vector
        query_vector = embedder.create_embedding(query)
        if not query_vector:
            raise ValueError("Failed to create query vector")
        
        # Search
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector={embedder_name: query_vector},
            limit=limit,
            filter=filter_conditions,
            with_payload=True
        )
        
        # Format results
        results = []
        for point in search_result:
            result = {
                "id": point.id,
                "score": point.score,
                "metadata": point.payload
            }
            results.append(result)
        
        return results 