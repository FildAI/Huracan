"""Qdrant provider package."""

from typing import Optional
from qdrant_client import QdrantClient

from .vector_store import QdrantVectorStore
from .index_manager import QdrantIndexManager
from ...core.exceptions import ConnectionError

class QdrantProvider:
    """Convenience class for Qdrant setup and management."""
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 6333,
                 api_key: Optional[str] = None,
                 grpc_port: Optional[int] = None,
                 prefer_grpc: bool = False,
                 timeout: float = 30.0):
        """
        Initialize Qdrant provider.
        
        Args:
            host: Qdrant host address
            port: Qdrant REST API port
            api_key: Optional API key for authentication
            grpc_port: Optional gRPC port for better performance
            prefer_grpc: Whether to prefer gRPC over REST
            timeout: Connection timeout in seconds
        """
        self.host = host
        self.port = port
        self.api_key = api_key
        self.grpc_port = grpc_port
        self.prefer_grpc = prefer_grpc
        self.timeout = timeout
        
        self._client = None
        self._vector_store = None
        self._index_manager = None
    
    @property
    def client(self) -> QdrantClient:
        """Get or create Qdrant client."""
        if self._client is None:
            self._client = self._create_client()
        return self._client
    
    def _create_client(self) -> QdrantClient:
        """Create Qdrant client instance."""
        try:
            client = QdrantClient(
                host=self.host,
                port=self.port,
                api_key=self.api_key,
                grpc_port=self.grpc_port,
                prefer_grpc=self.prefer_grpc,
                timeout=self.timeout
            )
            
            # Test connection
            client.get_collections()
            
            return client
            
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to Qdrant at {self.host}:{self.port}: {str(e)}"
            )
    
    def get_vector_store(self) -> QdrantVectorStore:
        """Get vector store instance."""
        if self._vector_store is None:
            self._vector_store = QdrantVectorStore(self.client)
        return self._vector_store
    
    def get_index_manager(self) -> QdrantIndexManager:
        """Get index manager instance."""
        if self._index_manager is None:
            self._index_manager = QdrantIndexManager(self.client)
        return self._index_manager
    
    def test_connection(self) -> bool:
        """Test connection to Qdrant."""
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False
    
    def get_health(self) -> dict:
        """Get Qdrant health status."""
        try:
            # Get basic info
            collections = self.client.get_collections()
            
            return {
                "status": "healthy",
                "collections_count": len(collections.collections),
                "host": self.host,
                "port": self.port,
                "grpc_enabled": self.grpc_port is not None
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "host": self.host,
                "port": self.port
            }
    
    @classmethod
    def from_url(cls, url: str, api_key: Optional[str] = None) -> "QdrantProvider":
        """
        Create provider from URL.
        
        Args:
            url: Qdrant URL (e.g., "http://localhost:6333")
            api_key: Optional API key
            
        Returns:
            QdrantProvider instance
        """
        # Parse URL
        from urllib.parse import urlparse
        parsed = urlparse(url)
        
        host = parsed.hostname or "localhost"
        port = parsed.port or 6333
        
        return cls(host=host, port=port, api_key=api_key)
    
    @classmethod
    def local(cls, port: int = 6333) -> "QdrantProvider":
        """
        Create provider for local Qdrant instance.
        
        Args:
            port: Local port (default: 6333)
            
        Returns:
            QdrantProvider instance
        """
        return cls(host="localhost", port=port)
    
    @classmethod
    def cloud(cls, cluster_url: str, api_key: str) -> "QdrantProvider":
        """
        Create provider for Qdrant Cloud.
        
        Args:
            cluster_url: Qdrant Cloud cluster URL
            api_key: API key for authentication
            
        Returns:
            QdrantProvider instance
        """
        # Parse cloud URL
        from urllib.parse import urlparse
        parsed = urlparse(cluster_url)
        
        host = parsed.hostname
        port = parsed.port or 443  # Default HTTPS port
        
        return cls(
            host=host,
            port=port,
            api_key=api_key,
            prefer_grpc=True  # Better performance for cloud
        )
    
    def close(self):
        """Close connections and cleanup resources."""
        if self._client:
            # Qdrant client doesn't have explicit close, but we can clear references
            self._client = None
            self._vector_store = None
            self._index_manager = None

# Re-export key classes
__all__ = [
    "QdrantProvider",
    "QdrantVectorStore", 
    "QdrantIndexManager"
]
