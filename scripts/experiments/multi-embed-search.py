import uuid
import pandas as pd
import numpy as np
import time
import argparse
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, ScoredPoint,
    Filter, FieldCondition, MatchValue, Range,
    UpdateStatus, CollectionInfo, Prefetch
)

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

load_dotenv()


# ─── Enhanced Config for 10 Embeddings ─────────────────────────────────────────

@dataclass
class EmbeddingConfig:
    name: str
    model_name: str
    text_processor: str  # method to extract text for this embedding
    weight: float
    vector_dim: int = 384
    prefetch_limit: int = 20  # For prefetch optimization

@dataclass
class Config:
    collection_name: str = "companies_multi_embed_v2"
    batch_size: int = 100
    prefetch_factor: int = 3  # Prefetch 3x more results for better ranking
    
    # Define 10 different embedding strategies with optimized weights
    embeddings: List[EmbeddingConfig] = field(default_factory=lambda: [
        EmbeddingConfig("name_embed", "all-MiniLM-L6-v2", "extract_name", 0.15, 384, 15),
        EmbeddingConfig("desc_embed", "all-MiniLM-L6-v2", "extract_description", 0.25, 384, 25),
        EmbeddingConfig("industry_embed", "all-MiniLM-L6-v2", "extract_industry", 0.12, 384, 15),
        EmbeddingConfig("location_embed", "all-MiniLM-L6-v2", "extract_location", 0.08, 384, 10),
        EmbeddingConfig("size_embed", "all-MiniLM-L6-v2", "extract_size_info", 0.06, 384, 10),
        EmbeddingConfig("tech_embed", "sentence-transformers/all-mpnet-base-v2", "extract_tech_stack", 0.12, 768, 15),
        EmbeddingConfig("mission_embed", "sentence-transformers/all-mpnet-base-v2", "extract_mission", 0.10, 768, 12),
        EmbeddingConfig("products_embed", "all-MiniLM-L6-v2", "extract_products", 0.08, 384, 10),
        EmbeddingConfig("market_embed", "all-MiniLM-L6-v2", "extract_target_market", 0.06, 384, 8),
        EmbeddingConfig("culture_embed", "all-MiniLM-L6-v2", "extract_culture", 0.05, 384, 8),
    ])


# ─── Text Processors for Different Embeddings ─────────────────────────────────

class TextProcessor:
    @staticmethod
    def extract_name(row: pd.Series) -> str:
        return str(row.get('name', ''))
    
    @staticmethod
    def extract_description(row: pd.Series) -> str:
        return str(row.get('description', ''))
    
    @staticmethod
    def extract_industry(row: pd.Series) -> str:
        # Try to extract industry information from description or dedicated field
        industry = row.get('industry', '')
        if not industry and 'description' in row:
            desc = str(row['description']).lower()
            # Simple industry keyword extraction
            keywords = {
                'software': 'Software Technology',
                'fintech': 'Financial Technology', 
                'healthcare': 'Healthcare Technology',
                'e-commerce': 'E-commerce',
                'ai': 'Artificial Intelligence',
                'saas': 'Software as a Service',
                'platform': 'Platform Technology',
                'analytics': 'Data Analytics',
                'security': 'Cybersecurity',
                'mobile': 'Mobile Technology'
            }
            for keyword, category in keywords.items():
                if keyword in desc:
                    industry = category
                    break
        return f"Industry: {industry}" if industry else "Industry: Technology"
    
    @staticmethod
    def extract_location(row: pd.Series) -> str:
        location = row.get('location', row.get('headquarters', ''))
        return f"Location: {location}" if location else "Location: Global"
    
    @staticmethod
    def extract_size_info(row: pd.Series) -> str:
        size = row.get('company_size', row.get('employees', ''))
        stage = row.get('stage', row.get('funding_stage', ''))
        return f"Size: {size}, Stage: {stage}" if size or stage else "Growing company"
    
    @staticmethod
    def extract_tech_stack(row: pd.Series) -> str:
        # Extract technical information from description
        desc = str(row.get('description', '')).lower()
        tech_keywords = ['api', 'platform', 'cloud', 'ai', 'ml', 'analytics', 'data', 'mobile', 'web', 'automation']
        found_tech = [word for word in tech_keywords if word in desc]
        tech_info = row.get('technology', '')
        return f"Technology: {', '.join(found_tech)} {tech_info}".strip()
    
    @staticmethod
    def extract_mission(row: pd.Series) -> str:
        mission = row.get('mission', '')
        if not mission and 'description' in row:
            # Extract mission-like language from description
            desc = str(row['description'])
            mission_words = ['help', 'enable', 'empower', 'transform', 'improve', 'revolutionize']
            if any(word in desc.lower() for word in mission_words):
                mission = desc[:200]  # Take first part as mission
        return f"Mission: {mission}" if mission else f"Mission: {row.get('description', '')[:100]}"
    
    @staticmethod
    def extract_products(row: pd.Series) -> str:
        products = row.get('products', '')
        if not products and 'description' in row:
            desc = str(row['description']).lower()
            product_keywords = ['software', 'app', 'tool', 'service', 'solution', 'product']
            products = ' '.join([word for word in desc.split() if any(kw in word for kw in product_keywords)])
        return f"Products: {products}" if products else "Products: Software solutions"
    
    @staticmethod
    def extract_target_market(row: pd.Series) -> str:
        market = row.get('target_market', '')
        if not market and 'description' in row:
            desc = str(row['description']).lower()
            market_keywords = ['enterprise', 'small business', 'consumer', 'b2b', 'b2c', 'startups', 'teams']
            found_markets = [word for word in market_keywords if word in desc]
            market = ', '.join(found_markets)
        return f"Target market: {market}" if market else "Target market: Business"
    
    @staticmethod
    def extract_culture(row: pd.Series) -> str:
        culture = row.get('culture', '')
        if not culture and 'description' in row:
            desc = str(row['description']).lower()
            culture_keywords = ['collaborative', 'innovative', 'agile', 'remote', 'inclusive', 'fast-paced']
            found_culture = [word for word in culture_keywords if word in desc]
            culture = ', '.join(found_culture)
        return f"Culture: {culture}" if culture else "Culture: Innovation-focused"


# ─── Data Loading and Processing ────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """Load company data from CSV file"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} companies from {path}")
    
    # Validate required columns
    required_cols = ['name', 'description']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return df


def load_embedding_models(cfg: Config) -> Dict[str, SentenceTransformer]:
    """Load all required embedding models with caching"""
    models = {}
    unique_models = set(embed_cfg.model_name for embed_cfg in cfg.embeddings)
    
    print("Loading embedding models...")
    for model_name in unique_models:
        print(f"  - {model_name}")
        models[model_name] = SentenceTransformer(model_name)
    
    return models


def embed_all_fields_batch(df: pd.DataFrame, models: Dict[str, SentenceTransformer], cfg: Config) -> List[PointStruct]:
    """Create embeddings for all configured fields using batch processing"""
    print(f"Creating {len(cfg.embeddings)} different embeddings for {len(df)} companies...")
    
    points = []
    processor = TextProcessor()
    
    # Process in batches for memory efficiency
    for batch_start in range(0, len(df), cfg.batch_size):
        batch_end = min(batch_start + cfg.batch_size, len(df))
        batch_df = df.iloc[batch_start:batch_end]
        
        print(f"  Processing batch {batch_start//cfg.batch_size + 1}/{(len(df)-1)//cfg.batch_size + 1}")
        
        # Prepare all texts for each embedding type
        batch_embeddings = {}
        for embed_cfg in cfg.embeddings:
            processor_method = getattr(processor, embed_cfg.text_processor)
            texts = [processor_method(row) for _, row in batch_df.iterrows()]
            
            # Batch encode for efficiency
            model = models[embed_cfg.model_name]
            vectors = model.encode(texts).tolist()
            batch_embeddings[embed_cfg.name] = vectors
        
        # Create points for this batch
        for i, (_, row) in enumerate(batch_df.iterrows()):
            vectors = {name: embeddings[i] for name, embeddings in batch_embeddings.items()}
            
            # Add metadata for filtering
            payload = row.to_dict()
            payload['_indexed_at'] = int(time.time())
            
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vectors,
                payload=payload
            )
            points.append(point)
    
    print(f"  Created {len(points)} points with {len(cfg.embeddings)} embeddings each")
    return points


# ─── Advanced Qdrant Setup ─────────────────────────────────────────────────────

def setup_qdrant_advanced(client: QdrantClient, cfg: Config):
    """Setup Qdrant collection with advanced configurations"""
    print(f"Setting up advanced Qdrant collection: {cfg.collection_name}")
    
    # Create vector configurations
    vectors_config = {}
    for embed_cfg in cfg.embeddings:
        vectors_config[embed_cfg.name] = VectorParams(
            size=embed_cfg.vector_dim, 
            distance=Distance.COSINE
        )
    
    # Recreate collection with optimizations
    try:
        client.delete_collection(cfg.collection_name)
    except:
        pass
    
    client.create_collection(
        collection_name=cfg.collection_name,
        vectors_config=vectors_config
    )
    
    print(f"  Created optimized collection with {len(vectors_config)} vector configurations")


def insert_points_batch(client: QdrantClient, cfg: Config, points: List[PointStruct]):
    """Insert points using batch operations for better performance"""
    print(f"Inserting {len(points)} points in batches...")
    
    total_batches = (len(points) - 1) // cfg.batch_size + 1
    
    for i in range(0, len(points), cfg.batch_size):
        batch = points[i:i + cfg.batch_size]
        batch_num = i // cfg.batch_size + 1
        
        print(f"  Inserting batch {batch_num}/{total_batches} ({len(batch)} points)")
        
        result = client.upsert(
            collection_name=cfg.collection_name, 
            points=batch,
            wait=True  # Wait for completion
        )
        
        if result.status != UpdateStatus.COMPLETED:
            print(f"    Warning: Batch {batch_num} status: {result.status}")
    
    print("  Batch insertion completed")


# ─── Advanced Search Functions ─────────────────────────────────────────────────

def create_search_filters(
    industry: Optional[str] = None,
    location: Optional[str] = None,
    stage: Optional[str] = None,
    company_size: Optional[str] = None
) -> Optional[Filter]:
    """Create search filters for targeted queries"""
    conditions = []
    
    if industry:
        conditions.append(
            FieldCondition(key="industry", match=MatchValue(value=industry))
        )
    
    if location:
        conditions.append(
            FieldCondition(key="location", match=MatchValue(value=location))
        )
    
    if stage:
        conditions.append(
            FieldCondition(key="stage", match=MatchValue(value=stage))
        )
    
    if company_size:
        conditions.append(
            FieldCondition(key="company_size", match=MatchValue(value=company_size))
        )
    
    return Filter(must=conditions) if conditions else None


def advanced_hybrid_search_with_prefetch(
    client: QdrantClient,
    cfg: Config,
    query: str,
    models: Dict[str, SentenceTransformer],
    top_k: int = 5,
    filters: Optional[Filter] = None,
    verbose: bool = True
) -> Tuple[List[dict], Dict[str, float]]:
    """Advanced hybrid search using Qdrant's built-in prefetch functionality"""
    
    start_time = time.time()
    processor = TextProcessor()
    
    if verbose:
        print(f"\nAdvanced prefetch search for: '{query}'")
        print("=" * 60)
    
    # Create prefetch queries for each embedding
    prefetch_queries = []
    
    # Use the first embedding as the main query and others as prefetch
    main_embed_cfg = cfg.embeddings[0]  # Use first embedding as main query
    main_model = models[main_embed_cfg.model_name]
    main_vector = main_model.encode([query])[0].tolist()
    
    # Create prefetch queries for the remaining embeddings
    for embed_cfg in cfg.embeddings[1:]:  # Skip the first one as it's the main query
        model = models[embed_cfg.model_name]
        vector = model.encode([query])[0].tolist()
        
        # Create prefetch query for this embedding
        prefetch_query = Prefetch(
            query=vector,
            using=embed_cfg.name,
            limit=embed_cfg.prefetch_limit,
            filter=filters
        )
        prefetch_queries.append(prefetch_query)
    
    # Use Qdrant's built-in prefetch with fusion
    search_start = time.time()
    
    # Use query_points with prefetch - main query + prefetch fusion
    results = client.query_points(
        collection_name=cfg.collection_name,
        prefetch=prefetch_queries,
        query=main_vector,  # Main query vector is required
        using=main_embed_cfg.name,  # Specify which vector space for main query
        limit=top_k,
        with_payload=True,
        with_vectors=False
    )
    
    # Convert to our expected format
    final_results = []
    for point in results.points:
        result_dict = {
            'id': str(point.id),
            'score': point.score,
            'payload': point.payload
        }
        final_results.append(result_dict)
    
    search_time = time.time() - search_start
    
    metrics = {
        'total_time_ms': search_time * 1000,
        'embeddings_searched': len(cfg.embeddings),
        'prefetch_used': True,
        'results_count': len(final_results)
    }
    
    if verbose:
        print(f"Qdrant prefetch search completed in {search_time*1000:.1f}ms")
        print(f"Used main query ({main_embed_cfg.name}) + {len(prefetch_queries)} prefetch queries")
    
    return final_results, metrics


def batch_search(
    client: QdrantClient,
    cfg: Config,
    queries: List[str],
    models: Dict[str, SentenceTransformer],
    top_k: int = 5
) -> List[Tuple[str, List[dict], Dict[str, float]]]:
    """Perform batch search for multiple queries efficiently using prefetch"""
    
    print(f"Performing batch prefetch search for {len(queries)} queries...")
    results = []
    
    start_time = time.time()
    
    for i, query in enumerate(queries):
        query_results, metrics = advanced_hybrid_search_with_prefetch(
            client, cfg, query, models, top_k, verbose=False
        )
        
        results.append((query, query_results, metrics))
        
        if (i + 1) % 5 == 0:
            print(f"  Processed {i + 1}/{len(queries)} queries")
    
    total_time = time.time() - start_time
    avg_time = total_time / len(queries) * 1000
    
    print(f"Batch search completed in {total_time:.2f}s")
    print(f"Average time per query: {avg_time:.1f}ms")
    
    return results


# ─── Database Creation ─────────────────────────────────────────────────────────

def create_database(data_path: str):
    """Create and populate the Qdrant database with company embeddings"""
    print("🚀 CREATING ADVANCED MULTI-EMBEDDING DATABASE")
    print("="*60)
    
    # Initialize configuration
    cfg = Config()
    
    # Load embedding models
    models = load_embedding_models(cfg)
    
    # Connect to Qdrant
    client = QdrantClient(
        host="localhost", 
        port=6333, 
        api_key=os.getenv("QDRANT_API_KEY"), 
        https=False
    )
    
    # Load data
    df = load_data(data_path)
    
    # Setup Qdrant collection with advanced features
    setup_qdrant_advanced(client, cfg)
    
    # Create and insert embeddings using batch processing
    points = embed_all_fields_batch(df, models, cfg)
    insert_points_batch(client, cfg, points)
    
    # Create index for better performance
    print("Optimizing collection...")
    try:
        # Wait for indexing to complete
        collection_info = client.get_collection(cfg.collection_name)
        print(f"Collection status: {collection_info.status}")
    except Exception as e:
        print(f"Note: {e}")
    
    print(f"\n✅ Advanced database creation completed!")
    print(f"📊 Created {len(cfg.embeddings)} different embeddings")
    print(f"🏢 Indexed {len(df)} companies")
    print(f"💾 Collection: {cfg.collection_name}")
    print(f"🔧 Features: Batch processing, Qdrant prefetch, multi-embedding fusion")


# ─── Query Execution ───────────────────────────────────────────────────────────

def run_queries(interactive: bool = False):
    """Run optimized search queries using Qdrant prefetch against the existing database"""
    print("🔍 RUNNING OPTIMIZED MULTI-EMBEDDING SEARCH WITH QDRANT PREFETCH")
    print("="*60)
    
    # Initialize configuration
    cfg = Config()
    
    # Load embedding models
    models = load_embedding_models(cfg)
    
    # Connect to Qdrant
    client = QdrantClient(
        host="localhost", 
        port=6333, 
        api_key=os.getenv("QDRANT_API_KEY"), 
        https=False
    )
    
    # Check collection
    try:
        collection_info = client.get_collection(cfg.collection_name)
        print(f"Connected to collection: {cfg.collection_name}")
        print(f"Points in collection: {collection_info.points_count}")
        print(f"Collection status: {collection_info.status}")
    except Exception as e:
        print(f"❌ Error: Collection '{cfg.collection_name}' not found!")
        print("Please run with --create-db first to create the database.")
        return
    
    if interactive:
        _run_interactive_search(client, cfg, models)
    else:
        _run_predefined_searches(client, cfg, models)


def _run_interactive_search(client: QdrantClient, cfg: Config, models: Dict[str, SentenceTransformer]):
    """Interactive search with advanced prefetch features"""
    print("\n🎯 Interactive Search Mode with Qdrant Prefetch")
    print("Commands:")
    print("  - Regular query: 'collaborative design tools'")
    print("  - Filtered query: 'AI startups | industry:Artificial Intelligence'")
    print("  - Quit: 'quit' or 'exit'")
    
    while True:
        query_input = input("\nSearch query: ").strip()
        if query_input.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query_input:
            continue
        
        # Parse query and filters
        parts = query_input.split(' | ')
        query = parts[0].strip()
        
        filters = None
        
        for part in parts[1:]:
            part = part.strip()
            if ':' in part:
                # Parse filter
                key, value = part.split(':', 1)
                if key.strip() == 'industry':
                    filters = create_search_filters(industry=value.strip())
                elif key.strip() == 'location':
                    filters = create_search_filters(location=value.strip())
                elif key.strip() == 'stage':
                    filters = create_search_filters(stage=value.strip())
        
        # Prefetch search
        results, metrics = advanced_hybrid_search_with_prefetch(
            client, cfg, query, models, top_k=5, filters=filters, verbose=True
        )
        
        _display_results(results, "Search Results")


def _run_predefined_searches(client: QdrantClient, cfg: Config, models: Dict[str, SentenceTransformer]):
    """Run predefined searches with performance analysis"""
    test_queries = [
        "Collaborative design software",
        "AI analytics platform", 
        "Enterprise security solution",
        "Mobile e-commerce platform",
        "Healthcare technology",
        "Machine learning automation",
        "Financial API services",
        "Video editing tools",
        "DevOps automation",
        "Educational technology"
    ]
    
    print(f"\n🎯 Running {len(test_queries)} advanced queries with Qdrant prefetch...")
    
    # Batch search with prefetch
    print("\n📊 Batch Prefetch Search Performance:")
    
    batch_results = batch_search(
        client, cfg, test_queries, models
    )
    
    # Performance analysis
    search_times = [metrics['total_time_ms'] for _, _, metrics in batch_results]
    
    print(f"\n⚡ Performance Summary:")
    print(f"Average search time: {np.mean(search_times):.1f}ms")
    print(f"Fastest search:      {np.min(search_times):.1f}ms")
    print(f"Slowest search:      {np.max(search_times):.1f}ms")
    print(f"Total queries:       {len(test_queries)}")
    
    # Show detailed results for first few queries
    print(f"\n📋 Detailed Results (first 3 queries):")
    for i, (query, results, metrics) in enumerate(batch_results[:3]):
        print(f"\nQuery {i+1}: '{query}'")
        print(f"Time: {metrics['total_time_ms']:.1f}ms")
        _display_results(results[:3], f"Top 3 Results", show_scores=True)
    
    # Test filtered searches
    print(f"\n🔍 Testing Filtered Searches:")
    
    # Industry filter
    ai_filter = create_search_filters(industry="Artificial Intelligence")
    results, metrics = advanced_hybrid_search_with_prefetch(
        client, cfg, "machine learning platform", models, 
        filters=ai_filter, verbose=False
    )
    print(f"\nAI Industry Filter - Found {len(results)} results in {metrics['total_time_ms']:.1f}ms")
    _display_results(results[:2], "AI Companies")


def _display_results(results: List[dict], title: str, show_scores: bool = False):
    """Display search results in a formatted way"""
    if not results:
        print(f"\n{title}: No results found")
        return
    
    print(f"\n{title}:")
    for i, result in enumerate(results, 1):
        payload = result['payload']
        score_text = f" (score: {result['score']:.3f})" if show_scores else ""
        print(f"  {i}. {payload['name']}{score_text}")
        print(f"     Industry: {payload.get('industry', 'N/A')}")
        print(f"     Location: {payload.get('location', 'N/A')}")
        print(f"     Description: {payload['description'][:80]}...")


# ─── Main Execution ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Optimized Multi-Embedding Company Search with Qdrant Prefetch")
    parser.add_argument("--create-db", action="store_true", 
                       help="Create and populate the database")
    parser.add_argument("--query", action="store_true", 
                       help="Run search queries")
    parser.add_argument("--interactive", action="store_true", 
                       help="Run in interactive query mode")
    parser.add_argument("--data-path", type=str, default="companies.csv",
                       help="Path to company data CSV file")
    
    args = parser.parse_args()
    
    if args.create_db:
        create_database(args.data_path)
    elif args.query or args.interactive:
        run_queries(args.interactive)
    else:
        # Default: create database and run queries
        print("Running full optimized demo: create database + run queries")
        create_database(args.data_path)
        print("\n" + "="*80 + "\n")
        run_queries(False)


if __name__ == "__main__":
    main()