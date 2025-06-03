"""Generic repository implementation."""

from typing import List, Optional, Dict
from uuid import uuid4

from ..core.abstractions import Repository, VectorStore
from ..core.types import (
    Document, VectorId, VectorPoint, SearchResult, BatchResult,
    Filter, MultiEmbeddingProcessor
)
from ..core.exceptions import VectorNotFoundError, VectorError

class GenericRepository(Repository[Document]):
    """Generic repository for managing documents with vector embeddings."""
    
    def __init__(self, 
                 vector_store: VectorStore,
                 embedding_processor: MultiEmbeddingProcessor,
                 index_name: str):
        """
        Initialize generic repository.
        
        Args:
            vector_store: Vector store instance
            embedding_processor: Multi-embedding processor
            index_name: Name of the index to use
        """
        self.vector_store = vector_store
        self.embedding_processor = embedding_processor
        self.index_name = index_name
    
    def _generate_id(self) -> VectorId:
        """Generate a unique ID."""
        return str(uuid4())
    
    def _ensure_id(self, document: Document) -> VectorId:
        """Ensure document has an ID, generate if missing."""
        if "id" in document:
            return str(document["id"])
        return self._generate_id()
    
    def save(self, document: Document) -> VectorId:
        """Save a document to the repository."""
        # Ensure document has an ID
        doc_id = self._ensure_id(document)
        document["id"] = doc_id
        
        try:
            # Process document into embeddings
            embeddings = self.embedding_processor.process_document(document)
            
            # Create vector point
            point = VectorPoint(
                id=doc_id,
                vectors=embeddings,
                payload=document
            )
            
            # Insert into vector store
            self.vector_store.insert(self.index_name, point)
            
            return doc_id
            
        except Exception as e:
            raise VectorError(f"Failed to save document: {str(e)}")
    
    def save_batch(self, documents: List[Document]) -> BatchResult:
        """Save multiple documents efficiently."""
        if not documents:
            return BatchResult(successful_ids=[], failed_ids=[])
        
        # Ensure all documents have IDs
        for doc in documents:
            doc["id"] = self._ensure_id(doc)
        
        try:
            # Process documents into embeddings
            embeddings_list = self.embedding_processor.process_batch(documents)
            
            # Create vector points
            points = []
            for doc, embeddings in zip(documents, embeddings_list):
                point = VectorPoint(
                    id=doc["id"],
                    vectors=embeddings,
                    payload=doc
                )
                points.append(point)
            
            # Batch insert
            return self.vector_store.insert_batch(self.index_name, points)
            
        except Exception as e:
            # If batch processing fails, try individual saves
            successful_ids = []
            failed_ids = []
            errors = {}
            
            for doc in documents:
                try:
                    doc_id = self.save(doc)
                    successful_ids.append(doc_id)
                except Exception as err:
                    doc_id = doc.get("id", "unknown")
                    failed_ids.append(doc_id)
                    errors[doc_id] = str(err)
            
            return BatchResult(
                successful_ids=successful_ids,
                failed_ids=failed_ids,
                errors=errors
            )
    
    def find_by_similarity(self, query: str, limit: int = 10,
                          filter: Optional[Filter] = None) -> List[Document]:
        """Find documents by similarity to query text."""
        # Create a temporary document for query processing
        query_doc = {"query": query}
        
        try:
            # Process query into embeddings
            query_embeddings = self.embedding_processor.process_document(query_doc)
            
            # Convert to proper format for search
            query_vectors = {name: vec for name, vec in query_embeddings.items()}
            
            # Search
            results = self.vector_store.search(
                index_name=self.index_name,
                query_vectors=query_vectors,
                limit=limit,
                filter=filter
            )
            
            # Extract documents from results
            documents = []
            for result in results:
                doc = result.payload.copy()
                # Add similarity score as metadata
                doc["_score"] = result.score
                doc["_id"] = result.id
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            raise VectorError(f"Failed to search documents: {str(e)}")
    
    def find_by_id(self, id: VectorId) -> Optional[Document]:
        """Find a document by ID."""
        try:
            point = self.vector_store.get(self.index_name, id)
            if point:
                doc = point.payload.copy()
                doc["_id"] = point.id
                return doc
            return None
        except VectorNotFoundError:
            return None
        except Exception as e:
            raise VectorError(f"Failed to get document {id}: {str(e)}")
    
    def update(self, id: VectorId, document: Document) -> None:
        """Update an existing document."""
        # Ensure the ID is preserved
        document["id"] = str(id)
        
        try:
            # Process document into embeddings
            embeddings = self.embedding_processor.process_document(document)
            
            # Create updated vector point
            point = VectorPoint(
                id=id,
                vectors=embeddings,
                payload=document
            )
            
            # Update in vector store
            self.vector_store.update(self.index_name, point)
            
        except Exception as e:
            raise VectorError(f"Failed to update document {id}: {str(e)}")
    
    def delete(self, id: VectorId) -> None:
        """Delete a document by ID."""
        try:
            self.vector_store.delete(self.index_name, id)
        except Exception as e:
            raise VectorError(f"Failed to delete document {id}: {str(e)}")
    
    def delete_batch(self, ids: List[VectorId]) -> BatchResult:
        """Delete multiple documents."""
        if not ids:
            return BatchResult(successful_ids=[], failed_ids=[])
        
        try:
            return self.vector_store.delete_batch(self.index_name, ids)
        except Exception as e:
            # Try individual deletes on batch failure
            successful_ids = []
            failed_ids = []
            errors = {}
            
            for id in ids:
                try:
                    self.delete(id)
                    successful_ids.append(id)
                except Exception as err:
                    failed_ids.append(id)
                    errors[id] = str(err)
            
            return BatchResult(
                successful_ids=successful_ids,
                failed_ids=failed_ids,
                errors=errors
            )
    
    def count(self, filter: Optional[Filter] = None) -> int:
        """Count documents in the repository."""
        try:
            return self.vector_store.count(self.index_name, filter)
        except Exception as e:
            raise VectorError(f"Failed to count documents: {str(e)}") 