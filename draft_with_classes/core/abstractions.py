"""Abstract base classes for the vector search system."""

from abc import ABC, abstractmethod
from typing import Generic, List, Optional, TypeVar, Dict
from .types import (
    VectorId, VectorPoint, SearchResult, BatchResult,
    Filter, IndexConfig, IndexInfo, Document
)

T = TypeVar('T', bound=Document)

class IndexManager(ABC):
    """Abstract base class for index management."""
    
    @abstractmethod
    def create_index(self, name: str, config: IndexConfig) -> None:
        """Create a new vector index."""
        pass
    
    @abstractmethod
    def delete_index(self, name: str) -> None:
        """Delete an existing index."""
        pass
    
    @abstractmethod
    def get_index_info(self, name: str) -> IndexInfo:
        """Get information about an index."""
        pass
    
    @abstractmethod
    def list_indexes(self) -> List[str]:
        """List all available indexes."""
        pass
    
    @abstractmethod
    def optimize_index(self, name: str, **kwargs) -> None:
        """Optimize an index for better performance."""
        pass

class VectorStore(ABC):
    """Abstract base class for vector storage operations."""
    
    @abstractmethod
    def insert(self, index_name: str, point: VectorPoint) -> None:
        """Insert a single vector point."""
        pass
    
    @abstractmethod
    def insert_batch(self, index_name: str, points: List[VectorPoint]) -> BatchResult:
        """Insert multiple vector points efficiently."""
        pass
    
    @abstractmethod
    def search(self, index_name: str, query_vectors: Dict[str, List[float]], 
               limit: int = 10, filter: Optional[Filter] = None,
               **kwargs) -> List[SearchResult]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    def get(self, index_name: str, id: VectorId) -> Optional[VectorPoint]:
        """Get a vector point by ID."""
        pass
    
    @abstractmethod
    def update(self, index_name: str, point: VectorPoint) -> None:
        """Update an existing vector point."""
        pass
    
    @abstractmethod
    def delete(self, index_name: str, id: VectorId) -> None:
        """Delete a vector point by ID."""
        pass
    
    @abstractmethod
    def delete_batch(self, index_name: str, ids: List[VectorId]) -> BatchResult:
        """Delete multiple vector points."""
        pass
    
    @abstractmethod
    def count(self, index_name: str, filter: Optional[Filter] = None) -> int:
        """Count vector points in an index."""
        pass

class Repository(ABC, Generic[T]):
    """Abstract base class for domain repositories."""
    
    @abstractmethod
    def save(self, document: T) -> VectorId:
        """Save a document to the repository."""
        pass
    
    @abstractmethod
    def save_batch(self, documents: List[T]) -> BatchResult:
        """Save multiple documents efficiently."""
        pass
    
    @abstractmethod
    def find_by_similarity(self, query: str, limit: int = 10,
                          filter: Optional[Filter] = None) -> List[T]:
        """Find documents by similarity to query text."""
        pass
    
    @abstractmethod
    def find_by_id(self, id: VectorId) -> Optional[T]:
        """Find a document by ID."""
        pass
    
    @abstractmethod
    def update(self, id: VectorId, document: T) -> None:
        """Update an existing document."""
        pass
    
    @abstractmethod
    def delete(self, id: VectorId) -> None:
        """Delete a document by ID."""
        pass
    
    @abstractmethod
    def delete_batch(self, ids: List[VectorId]) -> BatchResult:
        """Delete multiple documents."""
        pass
    
    @abstractmethod
    def count(self, filter: Optional[Filter] = None) -> int:
        """Count documents in the repository."""
        pass 