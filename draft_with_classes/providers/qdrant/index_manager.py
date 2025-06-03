"""Qdrant implementation of IndexManager."""

from typing import List, Optional, Dict, Any
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, CollectionStatus, PayloadSchemaType,
    OptimizersConfigDiff, CollectionInfo
)

from ...core.abstractions import IndexManager
from ...core.types import IndexConfig, IndexInfo, IndexState, DistanceMetric, VectorConfig
from ...core.exceptions import (
    IndexNotFoundError, IndexAlreadyExistsError, 
    IndexConfigurationError, VectorError
)

class QdrantIndexManager(IndexManager):
    """Qdrant implementation of IndexManager."""
    
    def __init__(self, client: QdrantClient):
        """Initialize with Qdrant client."""
        self.client = client
    
    def _convert_distance_metric(self, metric: DistanceMetric) -> Distance:
        """Convert our DistanceMetric to Qdrant Distance."""
        mapping = {
            DistanceMetric.COSINE: Distance.COSINE,
            DistanceMetric.EUCLIDEAN: Distance.EUCLID,
            DistanceMetric.DOT_PRODUCT: Distance.DOT,
            DistanceMetric.MANHATTAN: Distance.MANHATTAN
        }
        return mapping.get(metric, Distance.COSINE)
    
    def _convert_distance_from_qdrant(self, distance: Distance) -> DistanceMetric:
        """Convert Qdrant Distance to our DistanceMetric."""
        mapping = {
            Distance.COSINE: DistanceMetric.COSINE,
            Distance.EUCLID: DistanceMetric.EUCLIDEAN,
            Distance.DOT: DistanceMetric.DOT_PRODUCT,
            Distance.MANHATTAN: DistanceMetric.MANHATTAN
        }
        return mapping.get(distance, DistanceMetric.COSINE)
    
    def _convert_status(self, status: CollectionStatus) -> IndexState:
        """Convert Qdrant collection status to IndexState."""
        if status == CollectionStatus.GREEN:
            return IndexState.READY
        elif status == CollectionStatus.YELLOW:
            return IndexState.OPTIMIZING
        elif status == CollectionStatus.RED:
            return IndexState.ERROR
        else:
            return IndexState.CREATING
    
    def create_index(self, name: str, config: IndexConfig) -> None:
        """Create a new vector index."""
        try:
            # Check if collection already exists
            collections = self.client.get_collections().collections
            if any(col.name == name for col in collections):
                raise IndexAlreadyExistsError(name)
            
            # Convert vector configs to Qdrant format
            vectors_config = {}
            for vector_name, vector_config in config.vectors.items():
                vectors_config[vector_name] = VectorParams(
                    size=vector_config.dimension,
                    distance=self._convert_distance_metric(vector_config.distance_metric)
                )
            
            # Create collection with configurations
            create_config = {
                "collection_name": name,
                "vectors_config": vectors_config,
                "on_disk_payload": config.on_disk
            }
            
            # Add optional configurations
            if config.hnsw_config:
                create_config["hnsw_config"] = config.hnsw_config
            if config.wal_config:
                create_config["wal_config"] = config.wal_config
            if config.optimizers_config:
                create_config["optimizers_config"] = config.optimizers_config
            if config.quantization:
                create_config["quantization_config"] = config.quantization
            
            self.client.create_collection(**create_config)
            
        except IndexAlreadyExistsError:
            raise
        except Exception as e:
            raise IndexConfigurationError(f"Failed to create index '{name}': {str(e)}")
    
    def delete_index(self, name: str) -> None:
        """Delete an existing index."""
        try:
            self.client.delete_collection(collection_name=name)
        except Exception as e:
            if "Not found" in str(e):
                raise IndexNotFoundError(name)
            raise VectorError(f"Failed to delete index '{name}': {str(e)}")
    
    def get_index_info(self, name: str) -> IndexInfo:
        """Get information about an index."""
        try:
            # Get collection info
            collection_info = self.client.get_collection(collection_name=name)
            
            # Convert vectors config back
            vectors = {}
            if hasattr(collection_info.config.params, 'vectors'):
                vectors_params = collection_info.config.params.vectors
                if isinstance(vectors_params, dict):
                    for vector_name, params in vectors_params.items():
                        vectors[vector_name] = VectorConfig(
                            dimension=params.size,
                            distance_metric=self._convert_distance_from_qdrant(params.distance)
                        )
                elif vectors_params:
                    # Single vector case
                    vectors["default"] = VectorConfig(
                        dimension=vectors_params.size,
                        distance_metric=self._convert_distance_from_qdrant(vectors_params.distance)
                    )
            
            # Build IndexConfig
            index_config = IndexConfig(
                vectors=vectors,
                on_disk=getattr(collection_info.config.params, 'on_disk_payload', False),
                hnsw_config=getattr(collection_info.config, 'hnsw_config', None),
                wal_config=getattr(collection_info.config, 'wal_config', None),
                optimizers_config=getattr(collection_info.config, 'optimizer_config', None),
                quantization=getattr(collection_info.config, 'quantization_config', None)
            )
            
            # Get status
            status = self._convert_status(collection_info.status)
            
            return IndexInfo(
                name=name,
                state=status,
                vectors_count=collection_info.vectors_count or 0,
                points_count=collection_info.points_count or 0,
                config=index_config
            )
            
        except Exception as e:
            if "Not found" in str(e):
                raise IndexNotFoundError(name)
            raise VectorError(f"Failed to get index info for '{name}': {str(e)}")
    
    def list_indexes(self) -> List[str]:
        """List all available indexes."""
        try:
            collections = self.client.get_collections()
            return [col.name for col in collections.collections]
        except Exception as e:
            raise VectorError(f"Failed to list indexes: {str(e)}")
    
    def optimize_index(self, name: str, **kwargs) -> None:
        """Optimize an index for better performance."""
        try:
            # Create payload indexes for commonly searched fields
            payload_fields = kwargs.get('payload_fields', [])
            
            for field in payload_fields:
                field_type = kwargs.get(f'{field}_type', 'keyword')
                
                if field_type == 'text':
                    # Create text index for full-text search
                    self.client.create_payload_index(
                        collection_name=name,
                        field_name=field,
                        field_schema=PayloadSchemaType.TEXT,
                        wait=True
                    )
                elif field_type == 'integer':
                    # Create integer index
                    self.client.create_payload_index(
                        collection_name=name,
                        field_name=field,
                        field_schema=PayloadSchemaType.INTEGER,
                        wait=True
                    )
                else:
                    # Default to keyword index
                    self.client.create_payload_index(
                        collection_name=name,
                        field_name=field,
                        field_schema=PayloadSchemaType.KEYWORD,
                        wait=True
                    )
            
            # Update optimizer config if provided
            if 'optimizers_config' in kwargs:
                self.client.update_collection(
                    collection_name=name,
                    optimizers_config=OptimizersConfigDiff(**kwargs['optimizers_config'])
                )
            
            # Trigger optimization
            if kwargs.get('vacuum', False):
                # This would trigger a vacuum operation if Qdrant supports it
                pass
            
        except Exception as e:
            if "Not found" in str(e):
                raise IndexNotFoundError(name)
            raise VectorError(f"Failed to optimize index '{name}': {str(e)}") 