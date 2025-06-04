import os
from abc import ABC, abstractmethod
from typing import Optional, List
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

load_dotenv()

class BaseEmbedder(ABC):
    """Abstract base class for all embedders."""
    
    @abstractmethod
    def create_embedding(self, text: str) -> Optional[List[float]]:
        """Create embedding for the given text."""
        pass

    @abstractmethod
    def create_embeddings_batch(self, texts: list[str], model: str = "text-embedding-3-small", dimensions: Optional[int] = None) -> list[Optional[list[float]]]:
        """Create embeddings for the given texts."""
        pass

    @abstractmethod
    def get_embedding_dimensions(self) -> int:
        """Get the dimensions of the embedding."""
        pass

class OpenAIEmbedder(BaseEmbedder):    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        self.client = OpenAI(api_key=self.api_key)
        self.model = "text-embedding-3-small"
    
    def create_embedding(self, text: str) -> Optional[List[float]]:

        if not text or not text.strip():
            return np.zeros(self.get_embedding_dimensions())
        
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error creating OpenAI embedding: {e}")
            return None
        
    def create_embeddings_batch(self, texts: list[str]) -> list[Optional[list[float]]]:
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"Error creating OpenAI batch embeddings: {e}")
            return [None] * len(texts)
    
    def get_embedding_dimensions(self) -> int:
        return 1536


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    
    def create_embedding(self, text: str) -> Optional[List[float]]:

        if not text or not text.strip():
            return np.zeros(self.get_embedding_dimensions())

        try:
            return self.model.encode(text).tolist()
        except Exception as e:
            print(f"Error creating SentenceTransformer embedding: {e}")
            return None


class SentenceTransformerShort(BaseEmbedder):
    """SentenceTransformer embeddings implementation."""
    
    def __init__(self):
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.model = SentenceTransformer(model_name)
    
    def create_embedding(self, text: str) -> Optional[List[float]]:
        try:
            return self.model.encode(text).tolist()
        except Exception as e:
            print(f"Error creating SentenceTransformer embedding: {e}")
            return None
    
    def get_embedding_dimensions(self) -> int:
        return 384
    
    def create_embeddings_batch(self, texts: list[str]) -> list[Optional[list[float]]]:
        try:
            return self.model.encode(texts).tolist()
        except Exception as e:
            print(f"Error creating SentenceTransformer batch embeddings: {e}")
            return [None] * len(texts)