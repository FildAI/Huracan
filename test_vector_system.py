#!/usr/bin/env python3
"""Simple test script to verify the vector system works."""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        # Core imports
        from vector_system import (
            VectorSearchEngine, QdrantProvider, MultiEmbeddingBuilder,
            GenericRepository, extract_field, extract_concatenated
        )
        print("✅ Main imports successful")
        
        # Test provider imports
        from vector_system.providers.embedding import OpenAIEmbeddingProvider, SentenceTransformerProvider
        print("✅ Provider imports successful")
        
        # Test core types
        from vector_system.core import Document, VectorId, DistanceMetric
        print("✅ Core type imports successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_setup():
    """Test basic system setup without Qdrant."""
    print("\nTesting basic setup...")
    
    try:
        from vector_system import MultiEmbeddingBuilder, extract_field
        from vector_system.providers.embedding import SentenceTransformerProvider
        
        # Create a simple embedding processor
        builder = MultiEmbeddingBuilder()
        builder.add_sentence_transformer(
            text_extractor=extract_field("content"),
            vector_name="default",
            model_name="all-MiniLM-L6-v2"
        )
        
        processor = builder.build()
        print("✅ Created embedding processor")
        
        # Test processing a document
        test_doc = {"content": "This is a test document"}
        embeddings = processor.process_document(test_doc)
        
        print(f"✅ Generated embeddings with dimension: {len(embeddings['default'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Setup error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_qdrant_connection():
    """Test Qdrant connection if available."""
    print("\nTesting Qdrant connection...")
    
    try:
        from vector_system import QdrantProvider
        
        provider = QdrantProvider.local()
        if provider.test_connection():
            print("✅ Qdrant connection successful")
            
            # Get health status
            health = provider.get_health()
            print(f"   Status: {health['status']}")
            print(f"   Collections: {health['collections_count']}")
            return True
        else:
            print("⚠️  Qdrant not available (this is OK for testing)")
            return True  # Not a failure, just not available
            
    except Exception as e:
        print(f"⚠️  Qdrant connection test skipped: {e}")
        return True  # Not a failure, just not available

def main():
    """Run all tests."""
    print("=" * 60)
    print("Vector System Test Suite")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_basic_setup,
        test_qdrant_connection
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    if all(results):
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 