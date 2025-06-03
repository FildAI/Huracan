import sys
import os
from typing import Dict, Any, Optional, List, Union, Iterable
from qdrant_client import models, QdrantClient
from qdrant_client.models import TokenizerType
from core.embeddings import SentenceTransformerShort, OpenAIEmbedder
from utils.hash import hash_uuid
from tqdm import tqdm


class VocabBM25Index:
    
    COLUMNS = [
        "name",
        # "description",
        "primary_industry",
        "country",
        "size",
    ]
    
    def __init__(self, client: QdrantClient, embedder=None):
        self.client = client
        self.tokenizer = TokenizerType.MULTILINGUAL
        self.collection_name = "companies"

        self.embedder = embedder if embedder is not None else SentenceTransformerShort()
        # self.embedder = OpenAIEmbedder()

    def get_vocab_collection_name(self, column_name: str):
        return f"vocab_{self.collection_name}_{column_name}"


    def set_embedder(self, embedder):
        self.embedder = embedder


    def get_columns(self):
        return self.COLUMNS

    def _delete_collection(self, collection_name: str):
        try:
            self.client.get_collection(collection_name)
            self.client.delete_collection(collection_name)
        except Exception:
            pass
        
    def create_collection(self, collection_name: str, force_recreate: bool = False):
        if force_recreate:
            self._delete_collection(collection_name)

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(size=1536, distance=models.Distance.COSINE)
            },
        )
        
        return collection_name
    
    def create_vocab_indexes(self, force_recreate: bool = False):
        for col in self.COLUMNS:
            if force_recreate:
                vocab_collection_name = self.get_vocab_collection_name(col)
                self.create_vocab_collection(vocab_collection_name, force_recreate=force_recreate)
        
    def add_text_indexes_for_all_columns(self, force_recreate: bool = False):
        print(f"Adding text indexes for all columns: {self.COLUMNS}")
        
        if force_recreate:
            self.create_vocab_indexes(force_recreate=force_recreate)

        for col in self.COLUMNS:
            self.add_text_index(self.collection_name, col)


    def add_text_index(self, collection_name: str, field_name: str):
        
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=models.TextIndexParams(
                type="text",
                tokenizer=self.tokenizer,
                lowercase=True,
                min_token_len=2,
                max_token_len=30,
            )
        )

    
    def create_vocab_collection(self, name: str, force_recreate: bool = False):
        if force_recreate:
            self._delete_collection(name)

        self.client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=self.embedder.get_embedding_dimensions(), distance=models.Distance.COSINE)
        )
        
        return name
    
    
    def populate_vocab_collection(self):
        if self.embedder is None:
            raise ValueError("Embedder must be set before populating vocabulary")
            
        # Reset collections for each column
        for col in self.COLUMNS:
            vocab_collection_name = self.get_vocab_collection_name(col)
            self.create_vocab_collection(vocab_collection_name, force_recreate=True)
        
        elements = {col: [] for col in self.COLUMNS}
        
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
                    for col in self.COLUMNS:
                        if col in item.payload and item.payload[col] is not None:
                            elements[col].append(item.payload[col])
                
                pbar.update(len(points))
                
                if offset is None:  # when Qdrant returns offset=None, we've reached the end
                    break
        
        # Add collected elements to vocabulary collections
        print("Adding elements to vocabulary collections...")
        for col in tqdm(self.COLUMNS, desc="Processing columns"):
            if elements[col]:  # Only process if we have elements
                elements[col] = list(set(elements[col]))
                print(f"Adding {len(elements[col])} items to {col} vocabulary")
                self.add_vocab_items(self.get_vocab_collection_name(col), elements[col])


    
    def add_vocab_items(self, vocab_collection_name: str, text_values: List[str]):        
        if self.embedder is None:
            raise ValueError("Embedder must be set before adding vocabulary items")
        
        max_batch_size = 1000

        for i in tqdm(range(0, len(text_values), max_batch_size), desc=f"Adding items to {vocab_collection_name}", leave=False):
            new_batch = []
            for text_value in text_values[i:i+max_batch_size]:
                # Create embedding for text value
                vector = self.embedder.create_embedding(str(text_value))
                
                new_batch.append(
                    models.PointStruct(
                        id=hash_uuid(text_value),
                        vector=vector,
                        payload={"value": text_value}
                    )
                )
            
            if new_batch:
                self.client.upsert(
                    collection_name=vocab_collection_name,
                    points=new_batch
                )

    def add_object(self, payload: Dict[str, Any]):
        for col in self.COLUMNS:
            if col in payload and payload[col] is not None:
                vocab_collection_name = self.get_vocab_collection_name(col)
                self.add_vocab_items(vocab_collection_name, [payload[col]])
    
    
    def search_by_text(self, key: str, query_text: str, limit: int = 20):
        

        zero_vec = models.NamedVector(name="dense", vector=[0.0] * 1536)
        
        hits = self.client.search(
            collection_name=self.collection_name,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key=key,
                        match=models.MatchText(text=query_text)
                    )
                ]
            ),
            limit=limit,
            query_vector=zero_vec,
            with_payload=True
        )
        
        return hits
    
    def find_similar_in_vocab(self, key: str, query_text: str, limit: int = 5):
    
        if self.embedder is None:
            raise ValueError("Embedder must be set before finding similar values")


        if key not in self.COLUMNS:
            raise ValueError(f"Key {key} is not in the allowed columns")
        
        vocab_collection_name = self.get_vocab_collection_name(key)
        
        query_vector = self.embedder.create_embedding(query_text)

        hits = self.client.search(
            collection_name=vocab_collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=True
        )
        
        return [hit.payload['value'] for hit in hits]
    
    def hybrid_search(self, 
                     collection_name: str,
                     text_query: Optional[str] = None,
                     field_filters: Dict[str, str] = None,
                     vector_text: Optional[str] = None,
                     limit: int = 20):
        
        must_conditions = []
        should_conditions = []
        

        if text_query:
            for col in self.COLUMNS:
                should_conditions.append(
                    models.FieldCondition(
                        key=col,
                        match=models.MatchText(text=text_query)
                    )
                )
        
        if field_filters:
            for field, value in field_filters.items():
                must_conditions.append(
                    models.FieldCondition(
                        key=field,
                        match=models.MatchText(text=value)
                    )
                )
        
        # Создаем финальный фильтр
        query_filter = None
        if must_conditions or should_conditions:
            query_filter = models.Filter(
                must=must_conditions if must_conditions else None,
                should=should_conditions if should_conditions else None,
            )
        
    
        query_vector = None
        if self.embedder is not None:
            if vector_text is not None:
                query_vector = self.embedder.create_embedding(vector_text)
            elif text_query:
                query_vector = self.embedder.create_embedding(text_query)
        
        hits = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True
        )
        
        return hits
