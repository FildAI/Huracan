"""Custom exceptions for the vector search system."""

from typing import Optional, List
from .types import VectorId

class VectorSystemError(Exception):
    """Base exception for all vector system errors."""
    pass

class IndexError(VectorSystemError):
    """Base exception for index-related errors."""
    pass

class IndexNotFoundError(IndexError):
    """Raised when an index is not found."""
    def __init__(self, index_name: str):
        self.index_name = index_name
        super().__init__(f"Index '{index_name}' not found")

class IndexAlreadyExistsError(IndexError):
    """Raised when trying to create an index that already exists."""
    def __init__(self, index_name: str):
        self.index_name = index_name
        super().__init__(f"Index '{index_name}' already exists")

class IndexConfigurationError(IndexError):
    """Raised when index configuration is invalid."""
    pass

class VectorError(VectorSystemError):
    """Base exception for vector-related errors."""
    pass

class VectorNotFoundError(VectorError):
    """Raised when a vector is not found."""
    def __init__(self, vector_id: VectorId, index_name: Optional[str] = None):
        self.vector_id = vector_id
        self.index_name = index_name
        message = f"Vector '{vector_id}' not found"
        if index_name:
            message += f" in index '{index_name}'"
        super().__init__(message)

class VectorDimensionMismatchError(VectorError):
    """Raised when vector dimensions don't match expected dimensions."""
    def __init__(self, expected: int, actual: int, vector_name: Optional[str] = None):
        self.expected = expected
        self.actual = actual
        self.vector_name = vector_name
        message = f"Vector dimension mismatch: expected {expected}, got {actual}"
        if vector_name:
            message = f"Vector '{vector_name}' dimension mismatch: expected {expected}, got {actual}"
        super().__init__(message)

class EmbeddingError(VectorSystemError):
    """Base exception for embedding-related errors."""
    pass

class EmbeddingProviderError(EmbeddingError):
    """Raised when embedding provider fails."""
    def __init__(self, provider_name: str, message: str, original_error: Optional[Exception] = None):
        self.provider_name = provider_name
        self.original_error = original_error
        super().__init__(f"Embedding provider '{provider_name}' error: {message}")

class BatchOperationError(VectorSystemError):
    """Raised when a batch operation partially fails."""
    def __init__(self, successful_ids: List[VectorId], failed_ids: List[VectorId], 
                 errors: Optional[dict] = None):
        self.successful_ids = successful_ids
        self.failed_ids = failed_ids
        self.errors = errors or {}
        message = f"Batch operation partially failed: {len(successful_ids)} succeeded, {len(failed_ids)} failed"
        super().__init__(message)

class ConnectionError(VectorSystemError):
    """Raised when connection to vector database fails."""
    pass

class AuthenticationError(VectorSystemError):
    """Raised when authentication fails."""
    pass

class ConfigurationError(VectorSystemError):
    """Raised when configuration is invalid."""
    pass

class SearchError(VectorSystemError):
    """Base exception for search-related errors."""
    pass

class FilterError(SearchError):
    """Raised when search filter is invalid."""
    pass

class QueryError(SearchError):
    """Raised when search query is invalid."""
    pass 