from qdrant_client import QdrantClient, models
from utils.hash import hash_uuid
from pathlib import Path
import sys
from ..qdrant_client import get_client, get_async_client
from tqdm import tqdm
from ..vector_config import CompanyVectorConfig, CompanyVectorBuilder
from qdrant_client.models import Prefetch
from .vocab_bm25 import VocabBM25Index

class CompaniesIndex:

    def __init__(self, client: QdrantClient = None):
        self.client = client or get_client()
        self.vector_builder = CompanyVectorBuilder
        self.collection_name = self.vector_builder.collection_name
        self.full_vectors_config = CompanyVectorConfig
        self.vector_configs = {key: value.to_qdrant_model() for key, value in CompanyVectorConfig.items()}

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

    def _deduplicate_results(self, results, verbose: bool = False):
        """
        Remove duplicate results based on company ID or domain in payload.
        Keeps the first occurrence (highest score) of each unique company.
        """
        if not results.points:
            return results
        
        seen_ids = set()
        unique_points = []
        
        for point in results.points:
            # Try to get deduplication key from id or metadata.domain
            company_id = point.payload.get('id')
            if not company_id:
                metadata = point.payload.get('metadata', {})
                company_id = metadata.get('domain') if isinstance(metadata, dict) else None
            
            # Skip if no valid deduplication key found
            if not company_id:
                unique_points.append(point)
                continue
            
            if company_id not in seen_ids:
                seen_ids.add(company_id)
                unique_points.append(point)
        
        if verbose:
            original_count = len(results.points)
            unique_count = len(unique_points)
            print(f"Deduplication: {original_count} -> {unique_count} results (removed {original_count - unique_count} duplicates)")
        
        # Create new results object with deduplicated points
        results.points = unique_points
        return results

    def _fallback_search(self, main_vector_query, limit: int = 10, verbose: bool = False):
        """
        Fallback search without prefetch and filters when main search returns no results.
        """
        if verbose:
            print("No results found with filters and prefetch. Attempting fallback search without filters and prefetch.")
        
        # Perform simpler search without prefetch and filters
        fallback_results = self.client.query_points(
            collection_name=self.collection_name,
            query=main_vector_query,
            using="dense",
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
        
        if verbose:
            print(f"Fallback search returned {len(fallback_results.points)} results")
        
        # Apply deduplication to fallback search results
        fallback_results = self._deduplicate_results(fallback_results, verbose)
        
        return fallback_results

    def search(self, query: str, limit: int = 10, verbose: bool = False):
        vocab = VocabBM25Index(self.client)
        musts, shoulds = vocab.create_filter(query)
        if verbose:
            print("Vocab filter: ", musts, shoulds)
        filter = models.Filter(
            must=musts,
            should=shoulds
        )

        query_vector = self.vector_builder.query_to_vector(query)
        if verbose:
            print("Query vector: ", query_vector)
        main_vector_name = "dense"
        main_vector_query = query_vector.pop(main_vector_name)
        prefetch_queries = []
        
        if verbose:
            print("Prefetch queries: ", prefetch_queries)

        for key, vector in query_vector.items():                
                # Create prefetch query for this embedding
                prefetch_query = Prefetch(
                    query=vector,
                    using=key,
                    limit=2 * limit,
                    filter=filter
                )
                prefetch_queries.append(prefetch_query)


        print("Query filter: ", filter)
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetch_queries,
            query=main_vector_query,  # Main query vector is required
            using=main_vector_name,  # Specify which vector space for main query
            limit=limit,
            query_filter=filter,
            with_payload=True,
            with_vectors=False
        )

        # Apply deduplication to main search results
        results = self._deduplicate_results(results, verbose)

        # Fallback search if no results found
        if not results.points:
            return self._fallback_search(main_vector_query, limit, verbose)

        return results


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