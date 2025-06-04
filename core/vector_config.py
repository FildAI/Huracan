from qdrant_client import models
from qdrant_client.models import Distance, TokenizerType
from core.extractors.company import CompanyExtractorFull, CompanyExtractorField, CompanyField
from core.embeddings import OpenAIEmbedder, SentenceTransformerShort
from typing import Dict, List
from utils import hash_uuid

class VectorConfig:
    def __init__(self, embedding, data_extractor, distance = Distance.COSINE):
        self.embedding = embedding
        self.size = embedding.get_embedding_dimensions()
        self.distance = distance
        self.data_extractor = data_extractor

    def to_qdrant_model(self):
        return models.VectorParams(size=self.size, distance=self.distance, on_disk=True)


class BM25Config:
    def __init__(self, data_extractor, tokenizer=TokenizerType.MULTILINGUAL, lowercase=True, min_token_len=2, max_token_len=30):
        self.data_extractor = data_extractor
        self.tokenizer = tokenizer
        self.lowercase = lowercase
        self.min_token_len = min_token_len
        self.max_token_len = max_token_len

    def to_qdrant_model(self):
        return models.TextIndexParams(
            type="text",
            tokenizer=self.tokenizer,
            lowercase=self.lowercase,
            min_token_len=self.min_token_len,
            max_token_len=self.max_token_len,
        )


class VocabBM25Builder:
    def __init__(self, vocab_config: Dict[str, BM25Config], collection_name: str, embedder, auto_filter_model: str = "gpt-4o-mini"):
        self.vocab_config = vocab_config
        self.collection_name = collection_name
        self.embedder = embedder
        self.auto_filter_model = auto_filter_model

        
    def get_columns(self):
        return list(self.vocab_config.keys())
    
    def get_vocab_collection_name(self, column_name: str):
        return f"vocab_{self.collection_name}_{column_name}"
    
    def get_vocab_point(self, text_value: str):
        """Create a single vocab point from text value"""
        vector = self.embedder.create_embedding(str(text_value))
        return self._turn_vocab_to_point(vector, text_value)
    
    def get_vocab_batch_points(self, text_values: List[str]):
        """Create batch of vocab points from text values"""
        vectors = self.embedder.create_embeddings_batch([str(value) for value in text_values])
        points = [self._turn_vocab_to_point(vectors[i], text_values[i]) for i in range(len(text_values))]
        return points
    
    def _turn_vocab_to_point(self, vector: list, text_value: str):
        """Turn vector and text value into a PointStruct"""
        return models.PointStruct(
            id=hash_uuid(text_value),
            vector=vector,
            payload={"value": text_value}
        )
    


company_extractor = CompanyExtractorFull()


COMPANIES_COLLECTION_NAME = "companies"

full_extractor = lambda x: company_extractor.extract(x)
name_extractor = lambda x: x.get("name", "")
description_extractor = lambda x: x.get("description", "")
primary_industry_extractor = lambda x: x.get("primary_industry", "")
country_extractor = lambda x: x.get("country", "")
size_extractor = lambda x: x.get("size", "")



class VectorBuilder:
    def __init__(self, vector_config: Dict[str, VectorConfig], collection_name: str):
        self.vector_config = vector_config
        self.collection_name = collection_name

    def get_point(self, payload: dict):
        vector = {key: value.embedding.create_embedding(value.data_extractor(payload)) for key, value in self.vector_config.items()}

        return self._turn_vector_to_point(vector, payload)
    
    def _turn_vector_to_point(self, vector: dict, payload: dict):
        return models.PointStruct(
            id=hash_uuid(payload),
            vector=vector,
            payload=payload
        )
    
    def get_batch_points(self, payloads: list[dict]):
        # processing embeddings as a batch
        vectors = {key: value.embedding.create_embeddings_batch([value.data_extractor(payload) for payload in payloads]) for key, value in self.vector_config.items()}
        points = [self._turn_vector_to_point({key: vectors[key][ix] for key in vectors}, payloads[ix]) for ix in range(len(payloads))]
        return points
    
    def query_to_vector(self, query: str):
        # Group configs by their embedder to avoid duplicate computations
        embedder_to_keys = {}
        for key, config in self.vector_config.items():
            embedder = config.embedding
            if embedder not in embedder_to_keys:
                embedder_to_keys[embedder] = []
            embedder_to_keys[embedder].append(key)
        
        vectors = {}
        for embedder, keys in embedder_to_keys.items():
            # Generate embedding once per unique embedder
            vector = embedder.create_embedding(query)
            # Assign to all associated keys
            for key in keys:
                vectors[key] = vector
        return vectors



CompanyVectorConfig = {
    "dense": VectorConfig(embedding=OpenAIEmbedder(), data_extractor=full_extractor, distance=Distance.COSINE),
    # "name": VectorConfig(embedding=OpenAIEmbedder(), data_extractor=name_extractor, distance=Distance.COSINE),
    # "description": VectorConfig(embedding=OpenAIEmbedder(), data_extractor=description_extractor, distance=Distance.COSINE)
}

CompanyVectorBuilder = VectorBuilder(CompanyVectorConfig, COMPANIES_COLLECTION_NAME)

VocabBM25Config = {
    "name": BM25Config(data_extractor=name_extractor),
    "primary_industry": BM25Config(data_extractor=primary_industry_extractor),
    "country": BM25Config(data_extractor=country_extractor),
    "size": BM25Config(data_extractor=size_extractor),
}


# Create vocab builder for companies
CompanyVocabBM25Builder = VocabBM25Builder(
    vocab_config=VocabBM25Config,
    collection_name=COMPANIES_COLLECTION_NAME,
    embedder=SentenceTransformerShort(),
    auto_filter_model="gpt-4o-mini"
)