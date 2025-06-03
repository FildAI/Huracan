"""OpenAI embedding provider implementation."""

from typing import List, Optional
import numpy as np
from openai import OpenAI

from ...core.types import Vector, EmbeddingProvider
from ...core.exceptions import EmbeddingProviderError

class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI API-based embedding provider."""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        """
        Initialize OpenAI embedding provider.
        
        Args:
            api_key: OpenAI API key
            model: Model name (text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002)
        """
        self.model = model
        self._dimension = None  # Lazy loading
        
        try:
            self.client = OpenAI(api_key=api_key)
        except Exception as e:
            raise EmbeddingProviderError("OpenAI", f"Failed to initialize client: {str(e)}", e)
    
    @property
    def dimension(self) -> int:
        """Get the dimension of embeddings (lazy loaded)."""
        if self._dimension is None:
            # Get dimension by embedding a test string
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input="test"
                )
                self._dimension = len(response.data[0].embedding)
            except Exception as e:
                raise EmbeddingProviderError(
                    "OpenAI", 
                    f"Failed to determine embedding dimension: {str(e)}", 
                    e
                )
        return self._dimension
    
    def embed_single(self, text: str) -> Vector:
        """Embed a single text."""
        try:
            # Handle empty text
            if not text or not text.strip():
                return np.zeros(self.dimension).tolist()
            
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            
            embedding = response.data[0].embedding
            return embedding
            
        except Exception as e:
            raise EmbeddingProviderError(
                "OpenAI",
                f"Failed to embed text: {str(e)}",
                e
            )
    
    def embed_batch(self, texts: List[str]) -> List[Vector]:
        """Embed multiple texts efficiently."""
        if not texts:
            return []
        
        try:
            # Clean texts and handle empty strings
            cleaned_texts = []
            empty_indices = []
            
            for i, text in enumerate(texts):
                if not text or not text.strip():
                    empty_indices.append(i)
                    cleaned_texts.append("empty")  # Placeholder
                else:
                    cleaned_texts.append(text)
            
            # OpenAI supports batch embedding natively
            response = self.client.embeddings.create(
                model=self.model,
                input=cleaned_texts
            )
            
            embeddings = [data.embedding for data in response.data]
            
            # Replace empty text embeddings with zero vectors
            zero_vector = np.zeros(self.dimension).tolist()
            for idx in empty_indices:
                embeddings[idx] = zero_vector
            
            return embeddings
            
        except Exception as e:
            raise EmbeddingProviderError(
                "OpenAI",
                f"Failed to embed batch of {len(texts)} texts: {str(e)}",
                e
            ) 