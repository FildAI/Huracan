"""Core type definitions for the vector search system."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union
import numpy as np
from uuid import UUID

# Type aliases
VectorId = Union[str, UUID]
Vector = Union[List[float], np.ndarray]
Document = Dict[str, Any]
Filter = Dict[str, Any]
Metadata = Dict[str, Any]

class IndexState(Enum):
    """State of a vector index."""
    CREATING = auto()
    READY = auto()
    OPTIMIZING = auto()
    ERROR = auto()
    DELETED = auto()

class DistanceMetric(Enum):
    """Distance metrics for vector similarity."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"

@dataclass
class VectorPoint:
    """Vector point with support for multiple embeddings."""
    id: VectorId
    vectors: Dict[str, Vector]  # Multiple named vectors
    payload: Document
    
    def get_vector(self, name: str = "default") -> Optional[Vector]:
        """Get a specific vector by name."""
        return self.vectors.get(name)
    
    def has_vector(self, name: str) -> bool:
        """Check if a vector with given name exists."""
        return name in self.vectors

@dataclass
class SearchResult:
    """Result from a vector search operation."""
    id: VectorId
    score: float
    payload: Document
    vector_name: Optional[str] = None
    
@dataclass
class BatchResult:
    """Result of a batch operation."""
    successful_ids: List[VectorId]
    failed_ids: List[VectorId]
    errors: Dict[VectorId, str] = field(default_factory=dict)
    
    @property
    def success_count(self) -> int:
        return len(self.successful_ids)
    
    @property
    def failure_count(self) -> int:
        return len(self.failed_ids)

@dataclass
class VectorConfig:
    """Configuration for a single vector in an index."""
    dimension: int
    distance_metric: DistanceMetric = DistanceMetric.COSINE
    
@dataclass
class IndexConfig:
    """Configuration for a vector index."""
    vectors: Dict[str, VectorConfig]  # Multiple named vector configs
    on_disk: bool = False
    quantization: Optional[Dict[str, Any]] = None
    hnsw_config: Optional[Dict[str, Any]] = None
    wal_config: Optional[Dict[str, Any]] = None
    optimizers_config: Optional[Dict[str, Any]] = None
    
    @classmethod
    def single_vector(cls, dimension: int, 
                     distance_metric: DistanceMetric = DistanceMetric.COSINE) -> "IndexConfig":
        """Create config for single vector index."""
        return cls(vectors={"default": VectorConfig(dimension, distance_metric)})

@dataclass
class IndexInfo:
    """Information about an existing index."""
    name: str
    state: IndexState
    vectors_count: int
    points_count: int
    config: IndexConfig
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""
    
    @property
    def dimension(self) -> int:
        """Get the dimension of embeddings."""
        ...
    
    def embed_single(self, text: str) -> Vector:
        """Embed a single text."""
        ...
    
    def embed_batch(self, texts: List[str]) -> List[Vector]:
        """Embed multiple texts efficiently."""
        ...

class MultiEmbeddingProcessor(Protocol):
    """Protocol for processors that create multiple embeddings per document."""
    
    def process_document(self, document: Document) -> Dict[str, Vector]:
        """Process document into multiple named embeddings."""
        ...
    
    def process_batch(self, documents: List[Document]) -> List[Dict[str, Vector]]:
        """Process multiple documents efficiently."""
        ...
    
    def get_embedding_configs(self) -> Dict[str, VectorConfig]:
        """Get vector configurations for all embeddings."""
        ... 