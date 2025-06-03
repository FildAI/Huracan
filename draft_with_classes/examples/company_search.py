"""Complete example of company search using the vector system."""

import os
from typing import List, Dict
from datetime import datetime

# Import from our vector system
from vector_system import (
    VectorSearchEngine, QdrantProvider, MultiEmbeddingBuilder,
    GenericRepository, extract_field, extract_concatenated,
    IndexConfig, DistanceMetric
)
from vector_system.core import ConnectionError

# Sample company data
SAMPLE_COMPANIES = [
    {
        "name": "TechCorp AI",
        "description": "Leading artificial intelligence company specializing in machine learning and deep learning solutions for enterprise customers.",
        "industry": "Artificial Intelligence",
        "size": "1000-5000",
        "location": "San Francisco, CA",
        "founded": 2018,
        "tags": ["AI", "ML", "Enterprise", "B2B"]
    },
    {
        "name": "CloudScale Solutions",
        "description": "Cloud infrastructure and DevOps automation platform helping companies scale their applications globally.",
        "industry": "Cloud Computing",
        "size": "500-1000", 
        "location": "Seattle, WA",
        "founded": 2019,
        "tags": ["Cloud", "DevOps", "Infrastructure", "SaaS"]
    },
    {
        "name": "DataStream Analytics",
        "description": "Real-time data analytics and visualization platform for big data processing and business intelligence.",
        "industry": "Data Analytics",
        "size": "100-500",
        "location": "Austin, TX",
        "founded": 2020,
        "tags": ["Analytics", "Big Data", "BI", "Visualization"]
    },
    {
        "name": "SecureNet Systems",
        "description": "Cybersecurity company providing next-generation firewall and threat detection solutions for enterprises.",
        "industry": "Cybersecurity",
        "size": "500-1000",
        "location": "Boston, MA",
        "founded": 2017,
        "tags": ["Security", "Enterprise", "Firewall", "Threat Detection"]
    },
    {
        "name": "HealthTech Innovations",
        "description": "Digital health platform using AI to improve patient outcomes and streamline healthcare operations.",
        "industry": "Healthcare Technology",
        "size": "200-500",
        "location": "Chicago, IL",
        "founded": 2019,
        "tags": ["Healthcare", "AI", "Digital Health", "MedTech"]
    },
    {
        "name": "FinanceFlow",
        "description": "Modern fintech platform offering automated financial planning and investment management solutions.",
        "industry": "Financial Technology",
        "size": "100-500",
        "location": "New York, NY",
        "founded": 2020,
        "tags": ["FinTech", "Investment", "Automation", "B2C"]
    },
    {
        "name": "EcoTech Green Solutions",
        "description": "Sustainable technology company developing renewable energy management systems and carbon tracking platforms.",
        "industry": "Clean Technology",
        "size": "50-200",
        "location": "Portland, OR",
        "founded": 2021,
        "tags": ["Sustainability", "Renewable Energy", "Carbon Tracking", "ESG"]
    },
    {
        "name": "RoboLogix Automation",
        "description": "Industrial automation and robotics company creating intelligent manufacturing solutions.",
        "industry": "Robotics",
        "size": "200-500",
        "location": "Detroit, MI",
        "founded": 2018,
        "tags": ["Robotics", "Automation", "Manufacturing", "Industry 4.0"]
    },
    {
        "name": "EduLearn Platform",
        "description": "Online education technology platform providing personalized learning experiences using adaptive AI.",
        "industry": "Education Technology",
        "size": "100-500",
        "location": "Denver, CO",
        "founded": 2019,
        "tags": ["EdTech", "E-Learning", "AI", "Personalization"]
    },
    {
        "name": "SpaceVenture Systems",
        "description": "Aerospace technology startup developing satellite communication systems and space exploration tools.",
        "industry": "Aerospace",
        "size": "50-200",
        "location": "Los Angeles, CA",
        "founded": 2020,
        "tags": ["Space", "Satellite", "Aerospace", "Communication"]
    }
]

def extract_industry_context(doc: Dict) -> str:
    """Extract industry-specific context for better search."""
    industry = doc.get("industry", "")
    tags = " ".join(doc.get("tags", []))
    return f"{industry} {tags}"

def setup_vector_system(use_openai: bool = False, openai_api_key: str = None) -> VectorSearchEngine:
    """
    Set up the complete vector search system.
    
    Args:
        use_openai: Whether to use OpenAI embeddings (requires API key)
        openai_api_key: OpenAI API key if using OpenAI embeddings
    
    Returns:
        Configured VectorSearchEngine
    """
    print("🚀 Setting up vector search system...")
    
    # 1. Create Qdrant provider
    provider = QdrantProvider.local()
    
    # Test connection
    if not provider.test_connection():
        raise ConnectionError("Cannot connect to Qdrant. Make sure it's running on localhost:6333")
    
    print("✅ Connected to Qdrant")
    
    # 2. Create multi-embedding processor
    builder = MultiEmbeddingBuilder()
    
    if use_openai and openai_api_key:
        # Use OpenAI for semantic search
        builder.add_openai(
            api_key=openai_api_key,
            text_extractor=extract_concatenated("name", "description"),
            vector_name="semantic",
            model="text-embedding-3-small"
        )
        print("✅ Using OpenAI embeddings for semantic search")
    else:
        # Use Sentence Transformers for semantic search
        builder.add_sentence_transformer(
            text_extractor=extract_concatenated("name", "description"),
            vector_name="semantic",
            model_name="all-MiniLM-L6-v2"
        )
        print("✅ Using Sentence Transformers for semantic search")
    
    # Add industry-specific embedding
    builder.add_sentence_transformer(
        text_extractor=extract_industry_context,
        vector_name="industry",
        model_name="all-MiniLM-L6-v2",
        distance_metric=DistanceMetric.COSINE
    )
    
    # Add name-only embedding for exact name matches
    builder.add_sentence_transformer(
        text_extractor=extract_field("name"),
        vector_name="name",
        model_name="all-MiniLM-L6-v2"
    )
    
    embedding_processor = builder.build()
    print("✅ Created multi-embedding processor with 3 vector types")
    
    # 3. Create or update index
    index_name = "companies"
    index_manager = provider.get_index_manager()
    
    # Check if index exists
    existing_indexes = index_manager.list_indexes()
    
    if index_name in existing_indexes:
        print(f"ℹ️  Index '{index_name}' already exists")
        # Optionally delete and recreate
        # index_manager.delete_index(index_name)
    else:
        # Create new index with configuration from embedding processor
        index_config = builder.get_index_config()
        index_manager.create_index(index_name, index_config)
        print(f"✅ Created index '{index_name}'")
        
        # Optimize index for common fields
        index_manager.optimize_index(
            index_name,
            payload_fields=["industry", "size", "location", "founded"],
            industry_type="keyword",
            size_type="keyword",
            location_type="keyword",
            founded_type="integer"
        )
        print("✅ Optimized index for filtering")
    
    # 4. Create repository
    repository = GenericRepository(
        vector_store=provider.get_vector_store(),
        embedding_processor=embedding_processor,
        index_name=index_name
    )
    
    # 5. Create search engine
    engine = VectorSearchEngine(repository)
    print("✅ Vector search engine ready!\n")
    
    return engine

def load_sample_data(engine: VectorSearchEngine):
    """Load sample company data into the search engine."""
    print("📊 Loading sample company data...")
    
    # Add companies in batch
    result = engine.add_many(SAMPLE_COMPANIES)
    
    print(f"✅ Added {result.success_count} companies")
    if result.failure_count > 0:
        print(f"⚠️  Failed to add {result.failure_count} companies")
    
    # Get total count
    total = engine.count()
    print(f"📈 Total companies in index: {total}\n")

def demonstrate_search(engine: VectorSearchEngine):
    """Demonstrate various search capabilities."""
    print("🔍 Demonstrating search capabilities...\n")
    
    # 1. Simple semantic search
    print("1️⃣ Semantic Search: 'AI companies for healthcare'")
    results = engine.search("AI companies for healthcare", limit=5)
    for i, company in enumerate(results, 1):
        print(f"   {i}. {company['name']} (Score: {company['_score']:.3f})")
        print(f"      Industry: {company['industry']}")
        print(f"      Description: {company['description'][:100]}...")
    print()
    
    # 2. Industry-specific search
    print("2️⃣ Industry Search: 'cloud infrastructure'")
    results = engine.search("cloud infrastructure", limit=5)
    for i, company in enumerate(results, 1):
        print(f"   {i}. {company['name']} - {company['industry']}")
    print()
    
    # 3. Filtered search
    print("3️⃣ Filtered Search: AI companies with 100-500 employees")
    results = engine.filtered_search(
        query="artificial intelligence",
        filter={"size": "100-500"},
        limit=5
    )
    for i, company in enumerate(results, 1):
        print(f"   {i}. {company['name']} - Size: {company['size']}")
    print()
    
    # 4. Location-based search
    print("4️⃣ Location Search: Tech companies in California")
    results = engine.filtered_search(
        query="technology",
        filter={"location": {"$contains": "CA"}},  # This would need proper filter implementation
        limit=5
    )
    for i, company in enumerate(results, 1):
        print(f"   {i}. {company['name']} - {company['location']}")
    print()
    
    # 5. Similar companies
    print("5️⃣ Similar Companies: Find companies similar to the first result")
    if results:
        first_id = results[0]["_id"]
        similar = engine.recommend_similar(first_id, limit=3)
        print(f"   Companies similar to {results[0]['name']}:")
        for i, company in enumerate(similar, 1):
            print(f"   {i}. {company['name']} - {company['industry']}")
    print()

def show_performance_metrics(engine: VectorSearchEngine):
    """Display performance metrics."""
    print("📊 Performance Metrics:")
    metrics = engine.get_metrics()
    
    print(f"   Total searches: {metrics['total_searches']}")
    print(f"   Total documents added: {metrics['total_adds']}")
    print(f"   Average search time: {metrics['average_search_time']:.3f}s")
    
    if metrics['recent_queries']:
        print("\n   Recent queries:")
        for query_info in metrics['recent_queries'][-5:]:
            print(f"   - '{query_info['query']}' ({query_info['results']} results in {query_info['duration']:.3f}s)")

def cleanup_example(engine: VectorSearchEngine, provider: QdrantProvider):
    """Clean up resources."""
    print("\n🧹 Cleaning up...")
    
    # Optionally delete the index
    # index_manager = provider.get_index_manager()
    # index_manager.delete_index("companies")
    
    # Close connections
    provider.close()
    print("✅ Cleanup complete")

def main():
    """Run the complete example."""
    print("=" * 60)
    print("🏢 Vector Search System - Company Search Example")
    print("=" * 60)
    print()
    
    # Check for OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    use_openai = bool(openai_api_key)
    
    if use_openai:
        print("ℹ️  Found OpenAI API key - will use OpenAI embeddings")
    else:
        print("ℹ️  No OpenAI API key found - will use Sentence Transformers")
    print()
    
    try:
        # Set up the system
        engine = setup_vector_system(use_openai=use_openai, openai_api_key=openai_api_key)
        
        # Get provider for cleanup later
        provider = QdrantProvider.local()
        
        # Load sample data
        load_sample_data(engine)
        
        # Demonstrate search capabilities
        demonstrate_search(engine)
        
        # Show performance metrics
        show_performance_metrics(engine)
        
        # Clean up
        cleanup_example(engine, provider)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure Qdrant is running:")
        print("  docker run -p 6333:6333 qdrant/qdrant")

if __name__ == "__main__":
    main() 