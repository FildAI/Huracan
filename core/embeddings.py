import os
from abc import ABC, abstractmethod
from typing import Optional, List, Dict
import pandas as pd
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

load_dotenv()

class BaseEmbedder(ABC):
    """Abstract base class for all embedders."""
    
    @abstractmethod
    def create_embedding(self, text: str) -> Optional[List[float]]:
        """Create embedding for the given text."""
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
    
    def create_embedding(self, text: str) -> Optional[List[float]]:
        try:
            response = self.client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error creating OpenAI embedding: {e}")
            return None
    
    def get_embedding_dimensions(self) -> int:
        return 1536

    def create_embeddings_batch(self, texts: list[str], model: str = "text-embedding-3-small", dimensions: Optional[int] = None) -> list[Optional[list[float]]]:
        try:
            kwargs = dict(input=texts, model=model)
            if dimensions is not None:
                kwargs["dimensions"] = dimensions
            response = self.client.embeddings.create(**kwargs)
            # The response order matches the input order
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"Error creating OpenAI batch embeddings: {e}")
            return [None] * len(texts)

class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    
    def create_embedding(self, text: str) -> Optional[List[float]]:
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

class BM25Embedder(BaseEmbedder):
    """BM25 embeddings implementation using rank_bm25."""
    
    def __init__(self, corpus: Optional[List[str]] = None):
        """
        Initialize BM25 embedder.
        
        Args:
            corpus: Optional list of documents to fit the BM25 model on.
                   If provided, the model will be fitted during initialization.
        """
        self.bm25 = None
        self.vectorizer = CountVectorizer()
        if corpus:
            self.fit(corpus)
    
    def fit(self, corpus: List[str]) -> None:
        """
        Fit the BM25 model on a corpus of documents.
        
        Args:
            corpus: List of documents to fit the model on.
        """
        # Tokenize the corpus
        tokenized_corpus = [doc.split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # Fit the vectorizer for consistent vocabulary
        self.vectorizer.fit(corpus)
    
    def create_embedding(self, text: str) -> Optional[List[float]]:
        """
        Create BM25 embedding for the given text.
        
        Args:
            text: Input text to create embedding for.
            
        Returns:
            List of floats representing the BM25 scores for each term in the vocabulary.
        """
        if not self.bm25:
            raise ValueError("BM25 model needs to be fitted before creating embeddings")
        
        try:
            # Get the vocabulary size
            vocab_size = len(self.vectorizer.vocabulary_)
            
            # Create a sparse vector of term frequencies
            term_freqs = self.vectorizer.transform([text]).toarray()[0]
            
            # Calculate BM25 scores for each term
            scores = np.zeros(vocab_size)
            for i, freq in enumerate(term_freqs):
                if freq > 0:
                    term = self.vectorizer.get_feature_names_out()[i]
                    scores[i] = self.bm25.get_scores([term])[0][0]
            
            # Normalize the scores
            if np.sum(scores) > 0:
                scores = scores / np.linalg.norm(scores)
            
            return scores.tolist()
        except Exception as e:
            print(f"Error creating BM25 embedding: {e}")
            return None

# Example usage:
# openai_embedder = OpenAIEmbedder()
# st_embedder = SentenceTransformerShort()
# corpus = ["This is the first document", "This document is the second document", "And this is the third one"]
# bm25_embedder = BM25Embedder(corpus)
# embedding = bm25_embedder.create_embedding("This is a new document")