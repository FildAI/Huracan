import sys
import os
from typing import Dict, Any, Optional, List, Union, Iterable
from qdrant_client import models, QdrantClient
from qdrant_client.models import TokenizerType
from core.embeddings import SentenceTransformerShort, OpenAIEmbedder
from utils.hash import hash_uuid
from tqdm import tqdm
from core.vector_config import VocabBM25Builder, CompanyVocabBM25Builder
from utils.ai import get_filter_params, get_keywords_guess


class VocabBM25Index:
    
    def __init__(self, client: QdrantClient, vocab_builder: VocabBM25Builder = CompanyVocabBM25Builder):
        self.client = client
        self.vocab_builder = vocab_builder
        self.auto_filter_model = vocab_builder.auto_filter_model
        self.collection_name = vocab_builder.collection_name
        self.embedder = vocab_builder.embedder

    def get_columns(self):
        return self.vocab_builder.get_columns()

    def get_vocab_collection_name(self, column_name: str):
        return self.vocab_builder.get_vocab_collection_name(column_name)

    def _delete_collection(self, collection_name: str):
        try:
            self.client.get_collection(collection_name)
            self.client.delete_collection(collection_name)
        except Exception:
            pass
    
    def create_vocab_indexes(self, force_recreate: bool = False):
        """Create vocab collections for all configured columns"""
        for col in self.get_columns():
            if force_recreate:
                vocab_collection_name = self.get_vocab_collection_name(col)
                self.create_vocab_collection(vocab_collection_name, force_recreate=force_recreate)
        
    def add_text_indexes_for_all_columns(self, force_recreate: bool = False):
        print(f"Adding text indexes for all columns: {self.get_columns()}")
        
        if force_recreate:
            self.create_vocab_indexes(force_recreate=force_recreate)

        for col in self.get_columns():
            self.add_text_index(self.collection_name, col)

    def add_text_index(self, collection_name: str, field_name: str):
        bm25_config = self.vocab_builder.bm25_config[field_name]
        
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=bm25_config.to_qdrant_model()
        )

    def create_vocab_collection(self, name: str, force_recreate: bool = False):
        if force_recreate:
            self._delete_collection(name)

        self.client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=self.embedder.get_embedding_dimensions(), 
                distance=models.Distance.COSINE
            )
        )
        
        return name
    
    def populate_vocab_collection(self):
        if self.embedder is None:
            raise ValueError("Embedder must be set before populating vocabulary")
            
        # Reset collections for each column
        for col in self.get_columns():
            vocab_collection_name = self.get_vocab_collection_name(col)
            self.create_vocab_collection(vocab_collection_name, force_recreate=True)
        
        elements = {col: [] for col in self.get_columns()}
        
        BATCH = 10_000  # how many points to retrieve in one iteration
        offset = None  # cursor; None means start from the smallest id
        
        print("Fetching data from Qdrant collection...")
        with tqdm(desc="Loading data batches") as pbar:
            while True:
                points, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=BATCH,
                    offset=offset,      # pass the cursor for the next iteration
                    with_payload=True,  # we need the payload data
                    with_vectors=False  # we don't need vectors to save bandwidth
                )
                
                # Process each point
                for item in points:
                    for col in self.get_columns():
                        if col in item.payload and item.payload[col] is not None:
                            elements[col].append(item.payload[col])
                
                pbar.update(len(points))
                
                if offset is None:  # when Qdrant returns offset=None, we've reached the end
                    break
        
        # Add collected elements to vocabulary collections
        print("Adding elements to vocabulary collections...")
        for col in tqdm(self.get_columns(), desc="Processing columns"):
            if elements[col]:  # Only process if we have elements
                elements[col] = list(set(elements[col]))
                print(f"Adding {len(elements[col])} items to {col} vocabulary")
                self.add_vocab_items(self.get_vocab_collection_name(col), elements[col])

    def add_vocab_items(self, vocab_collection_name: str, text_values: List[str]):
        """Add vocabulary items to a specific vocab collection"""        
        if self.embedder is None:
            raise ValueError("Embedder must be set before adding vocabulary items")
        
        max_batch_size = 1000

        for i in tqdm(range(0, len(text_values), max_batch_size), desc=f"Adding items to {vocab_collection_name}", leave=False):
            batch_text_values = text_values[i:i+max_batch_size]
            
            # Use vocab builder to create points
            new_batch = self.vocab_builder.get_vocab_batch_points(batch_text_values)
            
            if new_batch:
                self.client.upsert(
                    collection_name=vocab_collection_name,
                    points=new_batch
                )

    def add_item_to_vocab(self, payload: Dict[str, Any]):
        for col in self.get_columns():
            if col in payload and payload[col] is not None:
                vocab_collection_name = self.get_vocab_collection_name(col)
                self.add_vocab_items(vocab_collection_name, [payload[col]])
    
    def search_vocab_similarity(self, column: str, query_text: str, limit: int = 5):
        if self.embedder is None:
            raise ValueError("Embedder must be set before finding similar values")

        if column not in self.get_columns():
            raise ValueError(f"Column '{column}' is not in the allowed columns: {self.get_columns()}")
        
        vocab_collection_name = self.get_vocab_collection_name(column)
        query_vector = self.embedder.create_embedding(query_text)

        hits = self.client.search(
            collection_name=vocab_collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=True
        )
        
        return [hit.payload['value'] for hit in hits]
    
    def search_with_filter(self, query_filter: models.Filter, limit: int = 20):
        zero_vec = models.NamedVector(name="dense", vector=[0.0] * 1536)
        
        hits = self.client.search(
            collection_name=self.collection_name,
            query_filter=query_filter,
            limit=limit,
            query_vector=zero_vec,
            with_payload=True
        )
        
        return hits
    
    def create_filter(self, query: str):
        # Get available columns for context
        available_columns = self.get_columns()
        
        guessed_columns = get_keywords_guess(query, available_columns, self.auto_filter_model)

        best_columns = {}

        for col in guessed_columns.items:
            similar_values = []
            for keyword in col.keywords:
                values = self.search_vocab_similarity(col.column, keyword, 5)
                print(f"Based on {keyword} found {values} values")
                similar_values += values
            best_columns[col.column] = similar_values

        print("Best columns: ", best_columns)


        filter_params = get_filter_params(query, best_columns)
        print("Filter params: ", filter_params)

        for param in filter_params.items:
            musts = []
            shoulds = []
            for param in filter_params.items:
                filter_cond = models.FieldCondition(
                        key=param.key,
                        match=models.MatchText(text=param.query)
                    )
                if param.type == "must":
                    musts.append(filter_cond)
                else:
                    shoulds.append(filter_cond)
            
        return musts, shoulds
    
    def search_with_auto_filter(self, query: str, limit: int = 20):
        auto_filter = self.create_filter(query)

        musts, shoulds = auto_filter

        filter = models.Filter(
            must=musts,
            should=shoulds
        )

        return self.search_with_filter(filter, limit)
