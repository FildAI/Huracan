"""Qdrant implementation of VectorStore."""

from typing import List, Optional, Dict, Any, Union
from uuid import UUID
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct, VectorParams, Distance, Filter as QdrantFilter,
    FieldCondition, MatchValue, Range, SearchRequest, NamedVector,
    FilterSelector, PayloadSelector, RecommendStrategy,
    UpdateStatus, PointIdsList, Batch
)

from ...core.abstractions import VectorStore
from ...core.types import (
    VectorId, VectorPoint, SearchResult, BatchResult,
    Filter, Vector
)
from ...core.exceptions import (
    VectorNotFoundError, VectorError, IndexNotFoundError,
    BatchOperationError
)

class QdrantVectorStore(VectorStore):
    """Qdrant implementation of VectorStore with multi-embedding support."""
    
    def __init__(self, client: QdrantClient):
        """Initialize with Qdrant client."""
        self.client = client
    
    def _convert_to_point_struct(self, point: VectorPoint) -> PointStruct:
        """Convert VectorPoint to Qdrant PointStruct."""
        # Convert vectors to NamedVector format
        vectors = {}
        for name, vector in point.vectors.items():
            if isinstance(vector, np.ndarray):
                vector = vector.tolist()
            vectors[name] = vector
        
        # Convert ID to appropriate format
        point_id = str(point.id) if isinstance(point.id, UUID) else point.id
        
        return PointStruct(
            id=point_id,
            vector=vectors,
            payload=point.payload
        )
    
    def _convert_filter(self, filter: Optional[Filter]) -> Optional[QdrantFilter]:
        """Convert generic filter to Qdrant filter."""
        if not filter:
            return None
        
        conditions = []
        for key, value in filter.items():
            if isinstance(value, dict):
                # Handle range queries
                if "gte" in value or "lte" in value or "gt" in value or "lt" in value:
                    conditions.append(FieldCondition(
                        key=key,
                        range=Range(
                            gte=value.get("gte"),
                            lte=value.get("lte"),
                            gt=value.get("gt"),
                            lt=value.get("lt")
                        )
                    ))
                else:
                    # Nested filter
                    conditions.append(FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    ))
            else:
                # Simple match
                conditions.append(FieldCondition(
                    key=key,
                    match=MatchValue(value=value)
                ))
        
        return QdrantFilter(must=conditions) if conditions else None
    
    def insert(self, index_name: str, point: VectorPoint) -> None:
        """Insert a single vector point."""
        try:
            point_struct = self._convert_to_point_struct(point)
            self.client.upsert(
                collection_name=index_name,
                points=[point_struct],
                wait=True
            )
        except Exception as e:
            if "Not found" in str(e):
                raise IndexNotFoundError(index_name)
            raise VectorError(f"Failed to insert point: {str(e)}")
    
    def insert_batch(self, index_name: str, points: List[VectorPoint]) -> BatchResult:
        """Insert multiple vector points efficiently."""
        successful_ids = []
        failed_ids = []
        errors = {}
        
        try:
            point_structs = [self._convert_to_point_struct(p) for p in points]
            result = self.client.upsert(
                collection_name=index_name,
                points=point_structs,
                wait=True
            )
            
            if result.status == UpdateStatus.COMPLETED:
                successful_ids = [p.id for p in points]
            else:
                # Handle partial failures if Qdrant provides such info
                successful_ids = [p.id for p in points]
                
        except Exception as e:
            if "Not found" in str(e):
                raise IndexNotFoundError(index_name)
            # If batch fails, try individual inserts
            for point in points:
                try:
                    self.insert(index_name, point)
                    successful_ids.append(point.id)
                except Exception as err:
                    failed_ids.append(point.id)
                    errors[point.id] = str(err)
        
        return BatchResult(
            successful_ids=successful_ids,
            failed_ids=failed_ids,
            errors=errors
        )
    
    def search(self, index_name: str, query_vectors: Dict[str, List[float]], 
               limit: int = 10, filter: Optional[Filter] = None,
               **kwargs) -> List[SearchResult]:
        """Search for similar vectors with multi-embedding support."""
        try:
            qdrant_filter = self._convert_filter(filter)
            
            # Single vector search
            if len(query_vectors) == 1:
                vector_name = list(query_vectors.keys())[0]
                vector = list(query_vectors.values())[0]
                
                results = self.client.search(
                    collection_name=index_name,
                    query_vector=NamedVector(
                        name=vector_name,
                        vector=vector
                    ),
                    limit=limit,
                    query_filter=qdrant_filter,
                    with_payload=True,
                    **kwargs
                )
                
                return [
                    SearchResult(
                        id=hit.id,
                        score=hit.score,
                        payload=hit.payload,
                        vector_name=vector_name
                    )
                    for hit in results
                ]
            
            # Multi-vector search using prefetch
            else:
                prefetch = []
                for vector_name, vector in query_vectors.items():
                    prefetch.append(SearchRequest(
                        vector=NamedVector(
                            name=vector_name,
                            vector=vector
                        ),
                        limit=limit * 2,  # Get more candidates for fusion
                        filter=qdrant_filter,
                        with_payload=False
                    ))
                
                # Final fusion search
                results = self.client.search(
                    collection_name=index_name,
                    query_vector=NamedVector(
                        name=list(query_vectors.keys())[0],
                        vector=list(query_vectors.values())[0]
                    ),
                    limit=limit,
                    query_filter=qdrant_filter,
                    with_payload=True,
                    search_params=kwargs.get("search_params"),
                    prefetch=prefetch
                )
                
                return [
                    SearchResult(
                        id=hit.id,
                        score=hit.score,
                        payload=hit.payload
                    )
                    for hit in results
                ]
                
        except Exception as e:
            if "Not found" in str(e):
                raise IndexNotFoundError(index_name)
            raise VectorError(f"Search failed: {str(e)}")
    
    def get(self, index_name: str, id: VectorId) -> Optional[VectorPoint]:
        """Get a vector point by ID."""
        try:
            point_id = str(id) if isinstance(id, UUID) else id
            points = self.client.retrieve(
                collection_name=index_name,
                ids=[point_id],
                with_vectors=True,
                with_payload=True
            )
            
            if not points:
                return None
            
            point = points[0]
            return VectorPoint(
                id=point.id,
                vectors=point.vector if isinstance(point.vector, dict) else {"default": point.vector},
                payload=point.payload
            )
            
        except Exception as e:
            if "Not found" in str(e):
                raise IndexNotFoundError(index_name)
            raise VectorError(f"Failed to get point: {str(e)}")
    
    def update(self, index_name: str, point: VectorPoint) -> None:
        """Update an existing vector point."""
        # Qdrant's upsert handles updates
        self.insert(index_name, point)
    
    def delete(self, index_name: str, id: VectorId) -> None:
        """Delete a vector point by ID."""
        try:
            point_id = str(id) if isinstance(id, UUID) else id
            result = self.client.delete(
                collection_name=index_name,
                points_selector=PointIdsList(
                    points=[point_id]
                ),
                wait=True
            )
            
            if result.status != UpdateStatus.COMPLETED:
                raise VectorError(f"Failed to delete point {id}")
                
        except Exception as e:
            if "Not found" in str(e):
                raise IndexNotFoundError(index_name)
            raise VectorError(f"Failed to delete point: {str(e)}")
    
    def delete_batch(self, index_name: str, ids: List[VectorId]) -> BatchResult:
        """Delete multiple vector points."""
        successful_ids = []
        failed_ids = []
        errors = {}
        
        try:
            point_ids = [str(id) if isinstance(id, UUID) else id for id in ids]
            result = self.client.delete(
                collection_name=index_name,
                points_selector=PointIdsList(points=point_ids),
                wait=True
            )
            
            if result.status == UpdateStatus.COMPLETED:
                successful_ids = ids
            else:
                # Try individual deletes on failure
                for id in ids:
                    try:
                        self.delete(index_name, id)
                        successful_ids.append(id)
                    except Exception as e:
                        failed_ids.append(id)
                        errors[id] = str(e)
                        
        except Exception as e:
            if "Not found" in str(e):
                raise IndexNotFoundError(index_name)
            # Try individual deletes
            for id in ids:
                try:
                    self.delete(index_name, id)
                    successful_ids.append(id)
                except Exception as err:
                    failed_ids.append(id)
                    errors[id] = str(err)
        
        return BatchResult(
            successful_ids=successful_ids,
            failed_ids=failed_ids,
            errors=errors
        )
    
    def count(self, index_name: str, filter: Optional[Filter] = None) -> int:
        """Count vector points in an index."""
        try:
            qdrant_filter = self._convert_filter(filter)
            
            if qdrant_filter:
                # Count with filter
                result = self.client.count(
                    collection_name=index_name,
                    count_filter=qdrant_filter
                )
                return result.count
            else:
                # Get collection info for total count
                info = self.client.get_collection(index_name)
                return info.points_count
                
        except Exception as e:
            if "Not found" in str(e):
                raise IndexNotFoundError(index_name)
            raise VectorError(f"Failed to count points: {str(e)}") 