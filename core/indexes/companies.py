from qdrant_client import QdrantClient, models
from utils.hash import hash_uuid
from pathlib import Path
import sys
from ..qdrant_client import get_client, get_async_client
from tqdm import tqdm


class CompaniesIndex:

    def __init__(self, client: QdrantClient = None, embedder=None):
        self.client = client or get_client()
        self.embedder = embedder
        self.collection_name = "companies"
        self.vectors_config = {
            "dense": models.VectorParams(size=1536, distance=models.Distance.COSINE, on_disk=True)
        }

    def create_collection(self, force_recreate: bool = False):
        if force_recreate:
            self._delete_collection()

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=self.vectors_config
        )
        
        return self.collection_name
    
    def _delete_collection(self):
        try:
            self.client.get_collection(self.collection_name)
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass

    def create_point(self, text_value: str, metadata: dict):
        embedding = self.embedder.create_embedding(text_value)
        uid = hash_uuid(metadata)
        return models.PointStruct(
            id=uid,
            vector={"dense": embedding},
            payload=metadata
        )
    
    def create_point_with_embedding(self, embedding: list[float], metadata: dict):
        uid = hash_uuid(metadata)
        return models.PointStruct(
            id=uid,
            vector={"dense": embedding},
            payload=metadata
        )
    
    def upsert_batch(self, points: list[models.PointStruct]):
        self.client = get_client()
        return self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    def ensure_on_disk(self):
        self.client.update_collection(
            collection_name=self.collection_name,
            vectors_config=self.vectors_config,
        )
    
    def populate_collection(self, df):
        max_batch_size = 10
        # Accept DataFrame directly, process as records
        records = df.to_dict(orient="records")
        print(f"Populating collection with {len(records)} records")
        for i in tqdm(range(0, len(records), max_batch_size)):
    
            print(f"Processing batch {i//max_batch_size + 1} of {len(records)//max_batch_size}")
            new_batch = []
            for record in tqdm(records[i:i+max_batch_size], desc="Processing records"):
                metadata = record
                text_value = str(metadata).replace("\n", " ").replace("\r", " ").replace("{", "").replace("}", "")
                point = self.create_point(text_value, metadata)
                new_batch.append(point)
            if new_batch:
                self.upsert_batch(new_batch)

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
            vectors_config=self.vectors_config,
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
            vectors_config=self.vectors_config,
            hnsw_config=models.HnswConfigDiff(
                m=16,
            ),
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=20000,
            ),
        )