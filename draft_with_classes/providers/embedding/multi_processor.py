"""Multi-embedding processor implementation."""

from typing import List, Dict, Callable, Tuple, Optional
from dataclasses import dataclass

from ...core.types import (
    Document, Vector, VectorConfig, EmbeddingProvider,
    MultiEmbeddingProcessor, DistanceMetric
)
from ...core.exceptions import EmbeddingProviderError, ConfigurationError

@dataclass
class EmbeddingConfig:
    """Configuration for a single embedding in the processor."""
    provider: EmbeddingProvider
    text_extractor: Callable[[Document], str]
    vector_name: str
    distance_metric: DistanceMetric = DistanceMetric.COSINE

class StandardMultiEmbeddingProcessor(MultiEmbeddingProcessor):
    """Processor that creates multiple embeddings per document."""
    
    def __init__(self, embedding_configs: List[EmbeddingConfig]):
        """
        Initialize with embedding configurations.
        
        Args:
            embedding_configs: List of configurations for each embedding
        """
        if not embedding_configs:
            raise ConfigurationError("At least one embedding configuration is required")
        
        self.embedding_configs = embedding_configs
        self._validate_configs()
    
    def _validate_configs(self):
        """Validate that vector names are unique."""
        names = [config.vector_name for config in self.embedding_configs]
        if len(names) != len(set(names)):
            raise ConfigurationError("Vector names must be unique")
    
    def process_document(self, document: Document) -> Dict[str, Vector]:
        """Process document into multiple named embeddings."""
        embeddings = {}
        
        for config in self.embedding_configs:
            try:
                # Extract text using the provided function
                text = config.text_extractor(document)
                
                # Generate embedding
                embedding = config.provider.embed_single(text)
                
                embeddings[config.vector_name] = embedding
                
            except Exception as e:
                raise EmbeddingProviderError(
                    config.vector_name,
                    f"Failed to process document: {str(e)}",
                    e
                )
        
        return embeddings
    
    def process_batch(self, documents: List[Document]) -> List[Dict[str, Vector]]:
        """Process multiple documents efficiently."""
        if not documents:
            return []
        
        # Group texts by provider for batch processing
        provider_texts: Dict[EmbeddingProvider, Dict[str, List[str]]] = {}
        
        for config in self.embedding_configs:
            provider = config.provider
            vector_name = config.vector_name
            
            # Extract texts for all documents
            texts = [config.text_extractor(doc) for doc in documents]
            
            if provider not in provider_texts:
                provider_texts[provider] = {}
            provider_texts[provider][vector_name] = texts
        
        # Batch process embeddings for each provider
        provider_embeddings: Dict[EmbeddingProvider, Dict[str, List[Vector]]] = {}
        
        for provider, vector_texts in provider_texts.items():
            provider_embeddings[provider] = {}
            
            for vector_name, texts in vector_texts.items():
                try:
                    embeddings = provider.embed_batch(texts)
                    provider_embeddings[provider][vector_name] = embeddings
                except Exception as e:
                    raise EmbeddingProviderError(
                        vector_name,
                        f"Failed to batch process {len(texts)} documents: {str(e)}",
                        e
                    )
        
        # Reorganize results by document
        results = []
        for i in range(len(documents)):
            doc_embeddings = {}
            
            for config in self.embedding_configs:
                provider = config.provider
                vector_name = config.vector_name
                doc_embeddings[vector_name] = provider_embeddings[provider][vector_name][i]
            
            results.append(doc_embeddings)
        
        return results
    
    def get_embedding_configs(self) -> Dict[str, VectorConfig]:
        """Get vector configurations for all embeddings."""
        configs = {}
        
        for config in self.embedding_configs:
            configs[config.vector_name] = VectorConfig(
                dimension=config.provider.dimension,
                distance_metric=config.distance_metric
            )
        
        return configs
    
    @classmethod
    def builder(cls) -> 'MultiEmbeddingBuilder':
        """Get a builder for creating multi-embedding processor."""
        return MultiEmbeddingBuilder()

class MultiEmbeddingBuilder:
    """Builder for creating MultiEmbeddingProcessor with fluent interface."""
    
    def __init__(self):
        self._configs: List[EmbeddingConfig] = []
    
    def add_embedding(self, 
                     provider: EmbeddingProvider,
                     text_extractor: Callable[[Document], str],
                     vector_name: str,
                     distance_metric: DistanceMetric = DistanceMetric.COSINE) -> 'MultiEmbeddingBuilder':
        """Add an embedding configuration."""
        config = EmbeddingConfig(
            provider=provider,
            text_extractor=text_extractor,
            vector_name=vector_name,
            distance_metric=distance_metric
        )
        self._configs.append(config)
        return self
    
    def add_openai(self,
                   api_key: str,
                   text_extractor: Callable[[Document], str],
                   vector_name: str,
                   model: str = "text-embedding-3-small",
                   distance_metric: DistanceMetric = DistanceMetric.COSINE) -> 'MultiEmbeddingBuilder':
        """Add OpenAI embedding configuration."""
        from .openai_provider import OpenAIEmbeddingProvider
        
        provider = OpenAIEmbeddingProvider(api_key=api_key, model=model)
        return self.add_embedding(provider, text_extractor, vector_name, distance_metric)
    
    def add_sentence_transformer(self,
                               text_extractor: Callable[[Document], str],
                               vector_name: str,
                               model_name: str = "all-MiniLM-L6-v2",
                               distance_metric: DistanceMetric = DistanceMetric.COSINE) -> 'MultiEmbeddingBuilder':
        """Add Sentence Transformer embedding configuration."""
        from .sentence_transformer_provider import SentenceTransformerProvider
        
        provider = SentenceTransformerProvider(model_name=model_name)
        return self.add_embedding(provider, text_extractor, vector_name, distance_metric)
    
    def build(self) -> StandardMultiEmbeddingProcessor:
        """Build the multi-embedding processor."""
        if not self._configs:
            raise ConfigurationError("No embedding configurations added")
        
        return StandardMultiEmbeddingProcessor(self._configs)

# Common text extractors
def extract_field(field_name: str) -> Callable[[Document], str]:
    """Create a text extractor for a specific field."""
    def extractor(doc: Document) -> str:
        return str(doc.get(field_name, ""))
    return extractor

def extract_concatenated(*field_names: str, separator: str = " ") -> Callable[[Document], str]:
    """Create a text extractor that concatenates multiple fields."""
    def extractor(doc: Document) -> str:
        parts = [str(doc.get(field, "")) for field in field_names]
        return separator.join(filter(None, parts))
    return extractor

def extract_with_template(template: str, **field_mapping) -> Callable[[Document], str]:
    """Create a text extractor using a template."""
    def extractor(doc: Document) -> str:
        format_args = {}
        for key, field in field_mapping.items():
            format_args[key] = doc.get(field, "")
        return template.format(**format_args)
    return extractor 