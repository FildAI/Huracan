import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
from qdrant_client import models


project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.qdrant_client import get_client
from core.qdrant_http import QdrantHttpClient
from core.indexes.companies import CompaniesIndex
import json
from utils.hash import hash_uuid

# CSV with an "embedding" column (see create_embeddings_csv.py); pass a path or keep it next to this script.
csv_path = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / "subsample100000.csv")

def sanitize_metadata(metadata: dict) -> dict:
    clean = {}
    for k, v in metadata.items():
        if isinstance(v, (np.generic,)):
            v = v.item()
        # Convert NaN, inf, -inf to None
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            clean[k] = None
        else:
            clean[k] = v
    return clean

async def main():
    index = CompaniesIndex()

    index.disable_indexing()
    

    async def upload_batch_async(df_chunk: pd.DataFrame):
        batch = []
        for _, row in df_chunk.iterrows():
            # 1) загружаем embedding
            embedding = json.loads(row["embedding"])
            # Sanitize embedding: replace NaN/inf with 0.0
            embedding = [float(x) if np.isfinite(x) else 0.0 for x in embedding]
            # 2) строим чистые метаданные
            raw_meta = row.drop(labels=["embedding"]).to_dict()
            metadata = sanitize_metadata(raw_meta)
            # 3) строим PointStruct
            point = index.create_point_with_embedding(embedding, metadata)
            batch.append(point)
        # 4) вставляем асинхронно
        await index.upsert_batch_async(batch)

    # Синхронно читаем CSV чанками по 500 строк, но внутри — асинхронная вставка
    for chunk in tqdm(pd.read_csv(csv_path, chunksize=1000), desc="Uploading batches async"):
        await upload_batch_async(chunk)

from itertools import islice



def main_sync():
    batch_size = 500

    df = pd.read_csv(csv_path, chunksize=batch_size)

    # index = CompaniesIndex()
    # index.ensure_on_disk()

    client = QdrantHttpClient()

    for batch in tqdm(df, desc="Uploading batches sync"):
        d_batch = batch.to_dict(orient="records")
        vectors = [json.loads(point.pop("embedding")) for point in d_batch]
        ids = [hash_uuid(point) for point in d_batch]
        
        d_batch = [sanitize_metadata(point) for point in d_batch]

        points= [{
                "id": point_id,
                "vector": {"dense": vector},
                "payload": payload
            } for point_id, vector, payload in zip(ids, vectors, d_batch)]

        

        client.upsert_points("companies", points)
    


if __name__ == "__main__":
    # asyncio.run(main())
    main_sync()


    