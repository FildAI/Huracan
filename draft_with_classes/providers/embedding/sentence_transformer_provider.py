"""Sentence Transformers embedding provider implementation."""

from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from ...core.types import Vector, EmbeddingProvider
from ...core.exceptions import EmbeddingProviderError

class SentenceTransformerProvider(EmbeddingProvider):
    """Sentence Transformers local embedding provider."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize Sentence Transformer embedding provider.
        
        Args:
            model_name: Hugging Face model name or path
        """
        self.model_name = model_name
        
        try:
            self.model = SentenceTransformer(model_name)
            self._dimension = self.model.get_sentence_embedding_dimension()
        except Exception as e:
            raise EmbeddingProviderError(
                "SentenceTransformer",
                f"Failed to load model '{model_name}': {str(e)}",
                e
            )
    
    @property
    def dimension(self) -> int:
        """Get the dimension of embeddings."""
        return self._dimension
    
    def embed_single(self, text: str) -> Vector:
        """Embed a single text."""
        try:
            # Handle empty text
            if not text or not text.strip():
                return np.zeros(self.dimension).tolist()
            
            # SentenceTransformer returns numpy array
            embedding = self.model.encode(text, convert_to_tensor=False)
            
            # Convert to list if numpy array
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            return embedding
            
        except Exception as e:
            raise EmbeddingProviderError(
                "SentenceTransformer",
                f"Failed to embed text: {str(e)}",
                e
            )
    
    def embed_batch(self, texts: List[str]) -> List[Vector]:
        """Embed multiple texts efficiently."""
        if not texts:
            return []
        
        try:
            # Handle empty texts
            processed_texts = []
            empty_indices = []
            
            for i, text in enumerate(texts):
                if not text or not text.strip():
                    empty_indices.append(i)
                    processed_texts.append("empty")  # Placeholder
                else:
                    processed_texts.append(text)
            
            # Batch encode
            embeddings = self.model.encode(
                processed_texts,
                convert_to_tensor=False,
                show_progress_bar=False
            )
            
            # Convert to list format
            if isinstance(embeddings, np.ndarray):
                embeddings = embeddings.tolist()
            
            # Replace empty text embeddings with zero vectors
            zero_vector = np.zeros(self.dimension).tolist()
            for idx in empty_indices:
                embeddings[idx] = zero_vector
            
            return embeddings
            
        except Exception as e:
            raise EmbeddingProviderError(
                "SentenceTransformer",
                f"Failed to embed batch of {len(texts)} texts: {str(e)}",
                e
            ) 