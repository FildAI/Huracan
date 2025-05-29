import os
from qdrant_client import QdrantClient, AsyncQdrantClient
from dotenv import load_dotenv

load_dotenv()

def get_client():
    return QdrantClient(
        url=os.getenv('QDRANT_URL'),
        api_key=os.getenv('QDRANT_API_KEY'),
        prefer_grpc=True,
        timeout=180,
        https=False,
        port="6333",
        grpc_port="6334"
    )

def get_async_client():
    return AsyncQdrantClient(
        url=os.getenv('QDRANT_URL'),
        api_key=os.getenv('QDRANT_API_KEY'),
        prefer_grpc=True,
        timeout=180,
        https=False,
        port="6333",
        grpc_port="6334"
    )