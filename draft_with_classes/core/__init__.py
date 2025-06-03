"""Core module exports."""

from .types import (
    # Type aliases
    VectorId, Vector, Document, Filter, Metadata,
    # Enums
    IndexState, DistanceMetric,
    # Dataclasses
    VectorPoint, SearchResult, BatchResult, VectorConfig, 
    IndexConfig, IndexInfo,
    # Protocols
    EmbeddingProvider, MultiEmbeddingProcessor
)

from .abstractions import (
    IndexManager, VectorStore, Repository
)

from .exceptions import (
    VectorSystemError, IndexError, IndexNotFoundError,
    IndexAlreadyExistsError, IndexConfigurationError,
    VectorError, VectorNotFoundError, VectorDimensionMismatchError,
    EmbeddingError, EmbeddingProviderError, BatchOperationError,
    ConnectionError, AuthenticationError, ConfigurationError,
    SearchError, FilterError, QueryError
)

__all__ = [
    # Types
    "VectorId", "Vector", "Document", "Filter", "Metadata",
    "IndexState", "DistanceMetric",
    "VectorPoint", "SearchResult", "BatchResult", "VectorConfig",
    "IndexConfig", "IndexInfo",
    "EmbeddingProvider", "MultiEmbeddingProcessor",
    # Abstractions
    "IndexManager", "VectorStore", "Repository",
    # Exceptions
    "VectorSystemError", "IndexError", "IndexNotFoundError",
    "IndexAlreadyExistsError", "IndexConfigurationError",
    "VectorError", "VectorNotFoundError", "VectorDimensionMismatchError",
    "EmbeddingError", "EmbeddingProviderError", "BatchOperationError",
    "ConnectionError", "AuthenticationError", "ConfigurationError",
    "SearchError", "FilterError", "QueryError"
]
