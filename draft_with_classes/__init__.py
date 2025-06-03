"""Vector System - A clean, production-ready vector search system with Qdrant."""

# Main facade
from .facades.search_engine import VectorSearchEngine, AsyncVectorSearchEngine

# Providers
from .providers.qdrant import QdrantProvider
from .providers.embedding.openai_provider import OpenAIEmbeddingProvider
from .providers.embedding.sentence_transformer_provider import SentenceTransformerProvider
from .providers.embedding.multi_processor import (
    StandardMultiEmbeddingProcessor, MultiEmbeddingBuilder,
    extract_field, extract_concatenated, extract_with_template
)

# Repository
from .repositories.generic_repository import GenericRepository

# Configuration
from .utils.config import (
    VectorSystemConfig, EmbeddingConfigBuilder,
    PresetConfigs, create_simple_config
)

# Core types for advanced usage
from .core import (
    Document, VectorId, Filter,
    IndexConfig, VectorConfig, DistanceMetric,
    BatchResult, SearchResult
)

__version__ = "0.1.0"

__all__ = [
    # Main interface
    "VectorSearchEngine",
    "AsyncVectorSearchEngine",
    
    # Providers
    "QdrantProvider",
    "OpenAIEmbeddingProvider",
    "SentenceTransformerProvider",
    "StandardMultiEmbeddingProcessor",
    "MultiEmbeddingBuilder",
    
    # Repository
    "GenericRepository",
    
    # Configuration
    "VectorSystemConfig",
    "EmbeddingConfigBuilder",
    "PresetConfigs",
    "create_simple_config",
    
    # Text extractors
    "extract_field",
    "extract_concatenated",
    "extract_with_template",
    
    # Core types
    "Document",
    "VectorId",
    "Filter",
    "IndexConfig",
    "VectorConfig",
    "DistanceMetric",
    "BatchResult",
    "SearchResult"
]
