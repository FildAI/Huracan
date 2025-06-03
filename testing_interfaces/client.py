from typing import List, Optional, Dict, Any
from time import time
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

from ..core.abstractions import Repository
from ..core.types import Document, VectorId, Filter, BatchResult
from ..core.exceptions import VectorError

class SearchMetrics:
    
    def __init__(self):
        self.total_searches = 0
        self.total_adds = 0
        self.total_deletes = 0
        self.search_times = []
        self.add_times = []
        self.query_history = []
    
    def record_search(self, query: str, duration: float, result_count: int):
        self.total_searches += 1
        self.search_times.append(duration)
        self.query_history.append({
            "query": query,
            "timestamp": datetime.now(),
            "duration": duration,
            "results": result_count
        })
    
    def record_add(self, duration: float, count: int = 1):
        self.total_adds += count
        self.add_times.append(duration)
    
    def get_average_search_time(self) -> float:
        return sum(self.search_times) / len(self.search_times) if self.search_times else 0
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_searches": self.total_searches,
            "total_adds": self.total_adds,
            "total_deletes": self.total_deletes,
            "average_search_time": self.get_average_search_time(),
            "recent_queries": self.query_history[-10:]
        }

class VectorSearchEngine:
    
    def __init__(self, repository: Repository[Document]):
        self.repository = repository
        self.metrics = SearchMetrics()
    
    def add(self, document: Document) -> VectorId:
        start_time = time()
        try:
            doc_id = self.repository.save(document)
            duration = time() - start_time
            self.metrics.record_add(duration)
            return doc_id
        except Exception as e:
            raise VectorError(f"Failed to add document: {str(e)}")
    
    def add_many(self, documents: List[Document]) -> BatchResult:
        start_time = time()
        try:
            result = self.repository.save_batch(documents)
            duration = time() - start_time
            self.metrics.record_add(duration, len(result.successful_ids))
            return result
        except Exception as e:
            raise VectorError(f"Failed to add documents: {str(e)}")
    
    def search(self, query: str, limit: int = 10) -> List[Document]:
        start_time = time()
        try:
            results = self.repository.find_by_similarity(query, limit)
            duration = time() - start_time
            self.metrics.record_search(query, duration, len(results))
            return results
        except Exception as e:
            raise VectorError(f"Search failed: {str(e)}")
    
    def filtered_search(self, query: str, filter: Filter, limit: int = 10) -> List[Document]:
        start_time = time()
        try:
            results = self.repository.find_by_similarity(query, limit, filter)
            duration = time() - start_time
            self.metrics.record_search(f"{query} (filtered)", duration, len(results))
            return results
        except Exception as e:
            raise VectorError(f"Filtered search failed: {str(e)}")
    
    def get(self, id: VectorId) -> Optional[Document]:
        try:
            return self.repository.find_by_id(id)
        except Exception as e:
            raise VectorError(f"Failed to get document {id}: {str(e)}")
    
    def update(self, id: VectorId, document: Document) -> None:
        try:
            self.repository.update(id, document)
        except Exception as e:
            raise VectorError(f"Failed to update document {id}: {str(e)}")
    
    def remove(self, id: VectorId) -> None:
        try:
            self.repository.delete(id)
            self.metrics.total_deletes += 1
        except Exception as e:
            raise VectorError(f"Failed to remove document {id}: {str(e)}")
    
    def remove_many(self, ids: List[VectorId]) -> BatchResult:
        try:
            result = self.repository.delete_batch(ids)
            self.metrics.total_deletes += len(result.successful_ids)
            return result
        except Exception as e:
            raise VectorError(f"Failed to remove documents: {str(e)}")
    
    def count(self, filter: Optional[Filter] = None) -> int:
        try:
            return self.repository.count(filter)
        except Exception as e:
            raise VectorError(f"Failed to count documents: {str(e)}")
    
    def recommend_similar(self, id: VectorId, limit: int = 10) -> List[Document]:
        doc = self.get(id)
        if not doc:
            raise VectorError(f"Document {id} not found")
        
        query_text = " ".join(str(v) for v in doc.values() if isinstance(v, (str, int, float)))
        
        results = self.search(query_text, limit + 1)
        return [r for r in results if r.get("_id") != id][:limit]
    
    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics.get_stats()
    
    def clear_metrics(self) -> None:
        self.metrics = SearchMetrics()
