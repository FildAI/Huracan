from qdrant_client import QdrantClient, models
from utils.hash import hash_uuid
from pathlib import Path
import sys
from ..qdrant_client import get_client, get_async_client
from tqdm import tqdm
from ..vector_config import CompanyVectorConfig, CompanyBM25Config, CompanyVectorBuilder

class CompaniesIndex:

    def __init__(self, client: QdrantClient = None):
        self.client = client or get_client()
        self.vector_builder = CompanyVectorBuilder
        self.collection_name = "companies"
        self.full_vectors_config = CompanyVectorConfig
        self.vector_configs = {key: value.to_qdrant_model() for key, value in CompanyVectorConfig.items()}
        self.bm25_config = CompanyBM25Config

    def create_collection(self, force_recreate: bool = False):
        if force_recreate:
            self._delete_collection()

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=self.vector_configs
        )
        
        return self.collection_name
    
    def _delete_collection(self):
        try:
            self.client.get_collection(self.collection_name)
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass

    def create_point(self, metadata: dict):
        return self.vector_builder.get_point(metadata)
    
    def upsert_batch(self, points: list[models.PointStruct]):
        self.client = get_client()
        return self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    def ensure_on_disk(self):
        self.client.update_collection(
            collection_name=self.collection_name,
            vectors_config=self.vector_configs,
        )
    
    def populate_collection(self, df):
        max_batch_size = 10
        # Accept DataFrame directly, process as records
        records = df.to_dict(orient="records")
        print(f"Populating collection with {len(records)} records")
        for i in tqdm(range(0, len(records), max_batch_size)):
    
            print(f"Processing batch {i//max_batch_size + 1} of {len(records)//max_batch_size + 1}")
            batch_records = records[i:i+max_batch_size]
            # Use batch processing for better efficiency
            points = self.vector_builder.get_batch_points(batch_records)
            if points:
                self.upsert_batch(points)

    def search(self, query: str, limit: int = 10):
        pass
    
    async def upsert_batch_async(self, points: list[models.PointStruct]):
        async_client = get_async_client()
        return await async_client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def disable_indexing(self):
        self.client.update_collection(
            collection_name=self.collection_name,
            vectors_config=self.vector_configs,
            hnsw_config=models.HnswConfigDiff(
                m=0,
            ),
            optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=0,
        ),
        )
    def enable_indexing(self):
        self.client.update_collection(
            collection_name=self.collection_name,
            vectors_config=self.vector_configs,
            hnsw_config=models.HnswConfigDiff(
                m=16,
            ),
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=20000,
            ),
        )