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
    def create_embeddings_batch(self, texts: list[str], model: str = None, dimensions: Optional[int] = None) -> list[Optional[list[float]]]:
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
        
    def create_embeddings_batch(self, texts: list[str], model: str = None, dimensions: Optional[int] = None) -> list[Optional[list[float]]]:
        # Use instance model if no model specified
        model = model or self.model
        
        # Filter out empty/None texts and keep track of original indices
        valid_texts = []
        text_indices = []
        
        for i, text in enumerate(texts):
            if text and str(text).strip():
                valid_texts.append(str(text).strip())
                text_indices.append(i)
        
        # If no valid texts, return zeros for all
        if not valid_texts:
            return [np.zeros(self.get_embedding_dimensions()).tolist()] * len(texts)
        
        try:
            response = self.client.embeddings.create(
                input=valid_texts,
                model=model
            )
            
            # Create result array with proper ordering
            results = [np.zeros(self.get_embedding_dimensions()).tolist()] * len(texts)
            for i, embedding in enumerate(response.data):
                original_index = text_indices[i]
                results[original_index] = embedding.embedding
                
            return results
            
        except Exception as e:
            print(f"Error creating OpenAI batch embeddings: {e}")
            print(f"Input texts sample: {valid_texts[:3] if valid_texts else 'No valid texts'}")
            return [np.zeros(self.get_embedding_dimensions()).tolist()] * len(texts)
    
    def get_embedding_dimensions(self) -> int:
        return 1536


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    
    def create_embedding(self, text: str) -> Optional[List[float]]:

        if not text or not text.strip():
            return np.zeros(self.get_embedding_dimensions()).tolist()

        try:
            return self.model.encode(text).tolist()
        except Exception as e:
            print(f"Error creating SentenceTransformer embedding: {e}")
            return np.zeros(self.get_embedding_dimensions()).tolist()
    
    def create_embeddings_batch(self, texts: list[str], model: str = None, dimensions: Optional[int] = None) -> list[Optional[list[float]]]:
        # Filter out empty/None texts
        processed_texts = []
        for text in texts:
            if text and str(text).strip():
                processed_texts.append(str(text).strip())
            else:
                processed_texts.append("")  # Use empty string for invalid texts
        
        try:
            embeddings = self.model.encode(processed_texts)
            results = []
            for i, embedding in enumerate(embeddings):
                if processed_texts[i]:  # Valid text
                    results.append(embedding.tolist())
                else:  # Empty/invalid text
                    results.append(np.zeros(self.get_embedding_dimensions()).tolist())
            return results
        except Exception as e:
            print(f"Error creating SentenceTransformer batch embeddings: {e}")
            return [np.zeros(self.get_embedding_dimensions()).tolist()] * len(texts)
    
    def get_embedding_dimensions(self) -> int:
        return self.model.get_sentence_embedding_dimension()


class SentenceTransformerShort(BaseEmbedder):
    """SentenceTransformer embeddings implementation."""
    
    def __init__(self):
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.model = SentenceTransformer(model_name)
    
    def create_embedding(self, text: str) -> Optional[List[float]]:
        if not text or not str(text).strip():
            return np.zeros(self.get_embedding_dimensions()).tolist()
            
        try:
            return self.model.encode(str(text).strip()).tolist()
        except Exception as e:
            print(f"Error creating SentenceTransformer embedding: {e}")
            return np.zeros(self.get_embedding_dimensions()).tolist()
    
    def get_embedding_dimensions(self) -> int:
        return 384
    
    def create_embeddings_batch(self, texts: list[str], model: str = None, dimensions: Optional[int] = None) -> list[Optional[list[float]]]:
        # Filter out empty/None texts
        processed_texts = []
        for text in texts:
            if text and str(text).strip():
                processed_texts.append(str(text).strip())
            else:
                processed_texts.append("")  # Use empty string for invalid texts
        
        try:
            embeddings = self.model.encode(processed_texts)
            results = []
            for i, embedding in enumerate(embeddings):
                if processed_texts[i]:  # Valid text
                    results.append(embedding.tolist())
                else:  # Empty/invalid text
                    results.append(np.zeros(self.get_embedding_dimensions()).tolist())
            return results
        except Exception as e:
            print(f"Error creating SentenceTransformer batch embeddings: {e}")
            return [np.zeros(self.get_embedding_dimensions()).tolist()] * len(texts)