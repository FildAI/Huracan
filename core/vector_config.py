from qdrant_client import models
from qdrant_client.models import Distance, TokenizerType
from core.extractors.company import CompanyExtractorFull, CompanyExtractorField, CompanyField
from core.embeddings import OpenAIEmbedder
from typing import Dict
from core.utils import hash_uuid

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


company_extractor = CompanyExtractorFull()

full_extractor = lambda x: company_extractor.extract(x)
name_extractor = lambda x: x.get("name", "")
description_extractor = lambda x: x.get("description", "")
primary_industry_extractor = lambda x: x.get("primary_industry", "")
country_extractor = lambda x: x.get("country", "")
size_extractor = lambda x: x.get("size", "")



class VectorBuilder:
    def __init__(self, vector_config: Dict[str, VectorConfig]):
        self.vector_config = vector_config

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


CompanyVectorConfig = {
    "dense": VectorConfig(embedding=OpenAIEmbedder(), data_extractor=full_extractor, distance=Distance.COSINE),
    "name": VectorConfig(embedding=OpenAIEmbedder(), data_extractor=name_extractor, distance=Distance.COSINE),
    "description": VectorConfig(embedding=OpenAIEmbedder(), data_extractor=description_extractor, distance=Distance.COSINE)
}

CompanyVectorBuilder = VectorBuilder(CompanyVectorConfig)

CompanyBM25Config = {
    "name": BM25Config(data_extractor=name_extractor),
    "primary_industry": BM25Config(data_extractor=primary_industry_extractor),
    "country": BM25Config(data_extractor=country_extractor),
    "size": BM25Config(data_extractor=size_extractor),
}