"""Search engine facade for simple vector search operations."""

from typing import List, Optional, Dict, Any
from time import time
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

from ..core.abstractions import Repository
from ..core.types import Document, VectorId, Filter, BatchResult
from ..core.exceptions import VectorError

class SearchMetrics:
    """Track performance metrics for search operations."""
    
    def __init__(self):
        self.total_searches = 0
        self.total_adds = 0
        self.total_deletes = 0
        self.search_times = []
        self.add_times = []
        self.query_history = []
    
    def record_search(self, query: str, duration: float, result_count: int):
        """Record a search operation."""
        self.total_searches += 1
        self.search_times.append(duration)
        self.query_history.append({
            "query": query,
            "timestamp": datetime.now(),
            "duration": duration,
            "results": result_count
        })
    
    def record_add(self, duration: float, count: int = 1):
        """Record an add operation."""
        self.total_adds += count
        self.add_times.append(duration)
    
    def get_average_search_time(self) -> float:
        """Get average search time."""
        return sum(self.search_times) / len(self.search_times) if self.search_times else 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get all statistics."""
        return {
            "total_searches": self.total_searches,
            "total_adds": self.total_adds,
            "total_deletes": self.total_deletes,
            "average_search_time": self.get_average_search_time(),
            "recent_queries": self.query_history[-10:]  # Last 10 queries
        }

class VectorSearchEngine:
    """Simple facade for vector search operations."""
    
    def __init__(self, repository: Repository[Document]):
        """
        Initialize search engine.
        
        Args:
            repository: Repository instance to use
        """
        self.repository = repository
        self.metrics = SearchMetrics()
    
    def add(self, document: Document) -> VectorId:
        """
        Add a single document to the search index.
        
        Args:
            document: Document to add
            
        Returns:
            Document ID
        """
        start_time = time()
        try:
            doc_id = self.repository.save(document)
            duration = time() - start_time
            self.metrics.record_add(duration)
            return doc_id
        except Exception as e:
            raise VectorError(f"Failed to add document: {str(e)}")
    
    def add_many(self, documents: List[Document]) -> BatchResult:
        """
        Add multiple documents efficiently.
        
        Args:
            documents: List of documents to add
            
        Returns:
            Batch result with success/failure information
        """
        start_time = time()
        try:
            result = self.repository.save_batch(documents)
            duration = time() - start_time
            self.metrics.record_add(duration, len(result.successful_ids))
            return result
        except Exception as e:
            raise VectorError(f"Failed to add documents: {str(e)}")
    
    def search(self, query: str, limit: int = 10) -> List[Document]:
        """
        Search for documents similar to the query.
        
        Args:
            query: Search query text
            limit: Maximum number of results
            
        Returns:
            List of matching documents with similarity scores
        """
        start_time = time()
        try:
            results = self.repository.find_by_similarity(query, limit)
            duration = time() - start_time
            self.metrics.record_search(query, duration, len(results))
            return results
        except Exception as e:
            raise VectorError(f"Search failed: {str(e)}")
    
    def filtered_search(self, query: str, filter: Filter, limit: int = 10) -> List[Document]:
        """
        Search with metadata filters.
        
        Args:
            query: Search query text
            filter: Metadata filters to apply
            limit: Maximum number of results
            
        Returns:
            List of matching documents
        """
        start_time = time()
        try:
            results = self.repository.find_by_similarity(query, limit, filter)
            duration = time() - start_time
            self.metrics.record_search(f"{query} (filtered)", duration, len(results))
            return results
        except Exception as e:
            raise VectorError(f"Filtered search failed: {str(e)}")
    
    def get(self, id: VectorId) -> Optional[Document]:
        """
        Get a document by ID.
        
        Args:
            id: Document ID
            
        Returns:
            Document if found, None otherwise
        """
        try:
            return self.repository.find_by_id(id)
        except Exception as e:
            raise VectorError(f"Failed to get document {id}: {str(e)}")
    
    def update(self, id: VectorId, document: Document) -> None:
        """
        Update an existing document.
        
        Args:
            id: Document ID to update
            document: New document content
        """
        try:
            self.repository.update(id, document)
        except Exception as e:
            raise VectorError(f"Failed to update document {id}: {str(e)}")
    
    def remove(self, id: VectorId) -> None:
        """
        Remove a document by ID.
        
        Args:
            id: Document ID to remove
        """
        try:
            self.repository.delete(id)
            self.metrics.total_deletes += 1
        except Exception as e:
            raise VectorError(f"Failed to remove document {id}: {str(e)}")
    
    def remove_many(self, ids: List[VectorId]) -> BatchResult:
        """
        Remove multiple documents.
        
        Args:
            ids: List of document IDs to remove
            
        Returns:
            Batch result with success/failure information
        """
        try:
            result = self.repository.delete_batch(ids)
            self.metrics.total_deletes += len(result.successful_ids)
            return result
        except Exception as e:
            raise VectorError(f"Failed to remove documents: {str(e)}")
    
    def count(self, filter: Optional[Filter] = None) -> int:
        """
        Count documents in the index.
        
        Args:
            filter: Optional metadata filter
            
        Returns:
            Number of documents
        """
        try:
            return self.repository.count(filter)
        except Exception as e:
            raise VectorError(f"Failed to count documents: {str(e)}")
    
    def recommend_similar(self, id: VectorId, limit: int = 10) -> List[Document]:
        """
        Recommend documents similar to a given document.
        
        Args:
            id: Document ID to find similar to
            limit: Maximum number of recommendations
            
        Returns:
            List of similar documents
        """
        # Get the document
        doc = self.get(id)
        if not doc:
            raise VectorError(f"Document {id} not found")
        
        # Use its content for similarity search
        # This is a simple approach - could be enhanced
        query_text = " ".join(str(v) for v in doc.values() if isinstance(v, (str, int, float)))
        
        # Search and exclude the original document
        results = self.search(query_text, limit + 1)
        return [r for r in results if r.get("_id") != id][:limit]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return self.metrics.get_stats()
    
    def clear_metrics(self) -> None:
        """Clear performance metrics."""
        self.metrics = SearchMetrics()

class AsyncVectorSearchEngine:
    """Async version of VectorSearchEngine for high-performance scenarios."""
    
    def __init__(self, repository: Repository[Document], max_workers: int = 4):
        """
        Initialize async search engine.
        
        Args:
            repository: Repository instance to use
            max_workers: Maximum number of worker threads
        """
        self.repository = repository
        self.metrics = SearchMetrics()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def add(self, document: Document) -> VectorId:
        """Add a document asynchronously."""
        loop = asyncio.get_event_loop()
        start_time = time()
        
        result = await loop.run_in_executor(
            self.executor,
            self.repository.save,
            document
        )
        
        duration = time() - start_time
        self.metrics.record_add(duration)
        return result
    
    async def add_many(self, documents: List[Document]) -> BatchResult:
        """Add multiple documents asynchronously."""
        loop = asyncio.get_event_loop()
        start_time = time()
        
        result = await loop.run_in_executor(
            self.executor,
            self.repository.save_batch,
            documents
        )
        
        duration = time() - start_time
        self.metrics.record_add(duration, len(result.successful_ids))
        return result
    
    async def search(self, query: str, limit: int = 10) -> List[Document]:
        """Search asynchronously."""
        loop = asyncio.get_event_loop()
        start_time = time()
        
        results = await loop.run_in_executor(
            self.executor,
            self.repository.find_by_similarity,
            query,
            limit
        )
        
        duration = time() - start_time
        self.metrics.record_search(query, duration, len(results))
        return results
    
    async def parallel_search(self, queries: List[str], limit: int = 10) -> List[List[Document]]:
        """Execute multiple searches in parallel."""
        tasks = [self.search(query, limit) for query in queries]
        return await asyncio.gather(*tasks)
    
    def __del__(self):
        """Cleanup executor on deletion."""
        self.executor.shutdown(wait=True) 