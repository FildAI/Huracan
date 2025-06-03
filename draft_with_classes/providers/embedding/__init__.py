"""Embedding providers module."""

from .openai_provider import OpenAIEmbeddingProvider
from .sentence_transformer_provider import SentenceTransformerProvider
from .multi_processor import (
    StandardMultiEmbeddingProcessor,
    MultiEmbeddingBuilder,
    EmbeddingConfig,
    extract_field,
    extract_concatenated,
    extract_with_template
)

__all__ = [
    "OpenAIEmbeddingProvider",
    "SentenceTransformerProvider",
    "StandardMultiEmbeddingProcessor",
    "MultiEmbeddingBuilder",
    "EmbeddingConfig",
    "extract_field",
    "extract_concatenated",
    "extract_with_template"
]
