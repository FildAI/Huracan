from qdrant_client import QdrantClient, models
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

client = QdrantClient(url="http://34.28.77.18:6333", api_key=os.getenv('QDRANT_API_KEY'))   # or Cloud URL / gRPC


DIM = 384                               # size of your embeddings

# client.create_collection(
#     collection_name="docs",
#     vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE),
#     quantization_config=models.ScalarQuantization(          # ← wrapper
#         scalar=models.ScalarQuantizationConfig(             # ← actual settings
#             type=models.ScalarType.INT8,
#             always_ram=True,        # keep quantized vectors in RAM (optional)
#             quantile=0.99           # optional; defaults to 1.0
#         )
#     )
# )

query_vec = np.random.rand(DIM).tolist()
hits = client.search(
    collection_name="docs",
    query_vector=query_vec,
    limit=5,
    with_payload=True,
)

print(hits)


DIM   = 384                 # same dimension you used when creating the collection
N     = 10                  # how many test vectors
COLL  = "docs"

# --- generate dummy data -------------------------------------------------
ids      = list(range(N))                           # [0, 1, …, 9]
vectors  = np.random.rand(N, DIM).tolist()          # uniform [0,1) floats
payloads = [{"title": f"test_{i}"} for i in ids]    # simple metadata

# --- upsert --------------------------------------------------------------
client.upsert(
    collection_name=COLL,
    points=models.Batch(ids=ids, vectors=vectors, payloads=payloads),
)
print("✅ inserted", N, "vectors")

# quick sanity search
query = vectors[0]          # search for the first vector itself
hits  = client.search(COLL, query_vector=query, limit=3, with_payload=True)
for h in hits:
    print("id:", h.id, "score:", h.score, "payload:", h.payload)