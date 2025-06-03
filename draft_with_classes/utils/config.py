"""Configuration utilities for the vector system."""

import os
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field

from ..core.types import (
    Document, VectorConfig, IndexConfig, DistanceMetric,
    EmbeddingProvider
)
from ..core.exceptions import ConfigurationError

@dataclass
class VectorSystemConfig:
    """Main configuration for the vector system."""
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: Optional[str] = None
    qdrant_grpc_port: Optional[int] = None
    qdrant_prefer_grpc: bool = False
    
    openai_api_key: Optional[str] = None
    openai_model: str = "text-embedding-3-small"
    
    default_collection: str = "documents"
    default_distance_metric: DistanceMetric = DistanceMetric.COSINE
    
    batch_size: int = 100
    max_retries: int = 3
    timeout: float = 30.0
    
    @classmethod
    def from_env(cls) -> "VectorSystemConfig":
        """Load configuration from environment variables."""
        return cls(
            qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
            qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            qdrant_grpc_port=int(os.getenv("QDRANT_GRPC_PORT", "6334")) if os.getenv("QDRANT_GRPC_PORT") else None,
            qdrant_prefer_grpc=os.getenv("QDRANT_PREFER_GRPC", "false").lower() == "true",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL", "text-embedding-3-small"),
            default_collection=os.getenv("DEFAULT_COLLECTION", "documents"),
            batch_size=int(os.getenv("BATCH_SIZE", "100")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            timeout=float(os.getenv("TIMEOUT", "30.0"))
        )
    
    def validate(self):
        """Validate configuration."""
        if self.batch_size <= 0:
            raise ConfigurationError("Batch size must be positive")
        if self.max_retries < 0:
            raise ConfigurationError("Max retries must be non-negative")
        if self.timeout <= 0:
            raise ConfigurationError("Timeout must be positive")

class EmbeddingConfigBuilder:
    """Fluent builder for embedding configurations."""
    
    def __init__(self):
        self._configs = []
        self._text_extractors = {}
    
    def add_openai_embedding(self,
                           vector_name: str,
                           text_extractor: Callable[[Document], str],
                           api_key: Optional[str] = None,
                           model: str = "text-embedding-3-small",
                           distance_metric: DistanceMetric = DistanceMetric.COSINE) -> "EmbeddingConfigBuilder":
        """Add OpenAI embedding configuration."""
        from ..providers.embedding.openai_provider import OpenAIEmbeddingProvider
        
        # Use provided API key or from environment
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ConfigurationError("OpenAI API key not provided")
        
        provider = OpenAIEmbeddingProvider(api_key=api_key, model=model)
        
        self._configs.append({
            "vector_name": vector_name,
            "provider": provider,
            "text_extractor": text_extractor,
            "distance_metric": distance_metric
        })
        self._text_extractors[vector_name] = text_extractor
        
        return self
    
    def add_sentence_transformer(self,
                               vector_name: str,
                               text_extractor: Callable[[Document], str],
                               model_name: str = "all-MiniLM-L6-v2",
                               distance_metric: DistanceMetric = DistanceMetric.COSINE) -> "EmbeddingConfigBuilder":
        """Add Sentence Transformer embedding configuration."""
        from ..providers.embedding.sentence_transformer_provider import SentenceTransformerProvider
        
        provider = SentenceTransformerProvider(model_name=model_name)
        
        self._configs.append({
            "vector_name": vector_name,
            "provider": provider,
            "text_extractor": text_extractor,
            "distance_metric": distance_metric
        })
        self._text_extractors[vector_name] = text_extractor
        
        return self
    
    def add_custom_provider(self,
                          vector_name: str,
                          provider: EmbeddingProvider,
                          text_extractor: Callable[[Document], str],
                          distance_metric: DistanceMetric = DistanceMetric.COSINE) -> "EmbeddingConfigBuilder":
        """Add custom embedding provider."""
        self._configs.append({
            "vector_name": vector_name,
            "provider": provider,
            "text_extractor": text_extractor,
            "distance_metric": distance_metric
        })
        self._text_extractors[vector_name] = text_extractor
        
        return self
    
    def build(self):
        """Build the multi-embedding processor."""
        if not self._configs:
            raise ConfigurationError("No embedding configurations added")
        
        from ..providers.embedding.multi_processor import StandardMultiEmbeddingProcessor, EmbeddingConfig
        
        embedding_configs = []
        for config in self._configs:
            embedding_configs.append(EmbeddingConfig(
                provider=config["provider"],
                text_extractor=config["text_extractor"],
                vector_name=config["vector_name"],
                distance_metric=config["distance_metric"]
            ))
        
        return StandardMultiEmbeddingProcessor(embedding_configs)
    
    def get_index_config(self) -> IndexConfig:
        """Get index configuration based on embeddings."""
        if not self._configs:
            raise ConfigurationError("No embedding configurations added")
        
        vectors = {}
        for config in self._configs:
            provider = config["provider"]
            vectors[config["vector_name"]] = VectorConfig(
                dimension=provider.dimension,
                distance_metric=config["distance_metric"]
            )
        
        return IndexConfig(vectors=vectors)

# Preset configurations
class PresetConfigs:
    """Common preset configurations."""
    
    @staticmethod
    def single_openai(api_key: Optional[str] = None,
                     model: str = "text-embedding-3-small") -> EmbeddingConfigBuilder:
        """Single OpenAI embedding for all text."""
        from ..providers.embedding.multi_processor import extract_concatenated
        
        builder = EmbeddingConfigBuilder()
        return builder.add_openai_embedding(
            vector_name="default",
            text_extractor=lambda doc: " ".join(str(v) for v in doc.values() if isinstance(v, (str, int, float))),
            api_key=api_key,
            model=model
        )
    
    @staticmethod
    def multi_field_search() -> EmbeddingConfigBuilder:
        """Multi-field search with different embeddings."""
        from ..providers.embedding.multi_processor import extract_field
        
        builder = EmbeddingConfigBuilder()
        
        # Title with OpenAI
        builder.add_openai_embedding(
            vector_name="title",
            text_extractor=extract_field("title")
        )
        
        # Content with Sentence Transformer
        builder.add_sentence_transformer(
            vector_name="content",
            text_extractor=extract_field("content"),
            model_name="all-mpnet-base-v2"
        )
        
        return builder
    
    @staticmethod
    def semantic_and_keyword() -> EmbeddingConfigBuilder:
        """Combination of semantic and keyword search."""
        from ..providers.embedding.multi_processor import extract_field, extract_concatenated
        
        builder = EmbeddingConfigBuilder()
        
        # Semantic search on full content
        builder.add_openai_embedding(
            vector_name="semantic",
            text_extractor=extract_concatenated("title", "description", "content")
        )
        
        # Keyword-focused search on title
        builder.add_sentence_transformer(
            vector_name="keyword",
            text_extractor=extract_field("title"),
            model_name="all-MiniLM-L6-v2"
        )
        
        return builder

# Helper functions
def create_simple_config(
    collection_name: str = "documents",
    embedding_field: str = "content",
    use_openai: bool = True,
    openai_api_key: Optional[str] = None,
    sentence_transformer_model: str = "all-MiniLM-L6-v2"
) -> Dict:
    """Create a simple configuration for quick start."""
    from ..providers.embedding.multi_processor import extract_field
    
    builder = EmbeddingConfigBuilder()
    
    if use_openai:
        builder.add_openai_embedding(
            vector_name="default",
            text_extractor=extract_field(embedding_field),
            api_key=openai_api_key
        )
    else:
        builder.add_sentence_transformer(
            vector_name="default",
            text_extractor=extract_field(embedding_field),
            model_name=sentence_transformer_model
        )
    
    return {
        "collection_name": collection_name,
        "embedding_processor": builder.build(),
        "index_config": builder.get_index_config()
    }

def validate_qdrant_connection(host: str, port: int, api_key: Optional[str] = None) -> bool:
    """Validate Qdrant connection parameters."""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=host, port=port, api_key=api_key)
        # Try to get collections to test connection
        client.get_collections()
        return True
    except Exception:
        return False 