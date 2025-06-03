# Vector System 🚀

A clean, production-ready vector search system with multi-embedding support using Qdrant.

## Features ✨

- **Multi-Embedding Support**: Use multiple vector representations per document
- **Flexible Providers**: OpenAI, Sentence Transformers, or custom embeddings
- **Clean Abstractions**: Domain-agnostic interfaces that scale
- **Batch Operations**: Efficient bulk processing
- **Simple Interface**: Easy-to-use facade pattern
- **Performance Tracking**: Built-in metrics and monitoring
- **Type-Safe**: Full type hints and validation

## Installation 📦

```bash
pip install -r requirements.txt
```

### Start Qdrant

```bash
docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

## Quick Start 🏃‍♂️

### Simple Usage

```python
from vector_system import VectorSearchEngine, QdrantProvider, MultiEmbeddingBuilder, GenericRepository, extract_field

# Setup
provider = QdrantProvider.local()
processor = MultiEmbeddingBuilder() \
    .add_sentence_transformer(extract_field("content"), "default") \
    .build()

repo = GenericRepository(provider.get_vector_store(), processor, "documents")
engine = VectorSearchEngine(repo)

# Use
engine.add({"content": "Python is a great programming language"})
results = engine.search("programming languages")
```

### Advanced Multi-Embedding Setup

```python
from vector_system import extract_concatenated, extract_with_template

# Create processor with multiple embeddings
processor = MultiEmbeddingBuilder() \
    .add_openai(
        api_key="your-key",
        text_extractor=extract_concatenated("title", "description"),
        vector_name="semantic"
    ) \
    .add_sentence_transformer(
        text_extractor=extract_field("category"),
        vector_name="category",
        model_name="all-MiniLM-L6-v2"
    ) \
    .build()

# Create index configuration
index_config = processor.get_embedding_configs()
```

## Architecture 🏗️

```
vector_system/
├── core/               # Core abstractions and types
├── providers/          # Qdrant and embedding providers
├── repositories/       # Business logic layer
├── facades/           # Simple user interfaces
└── utils/             # Configuration and helpers
```

## Examples 📚

### Company Search Example

```bash
cd vector_system/examples
python company_search.py
```

This demonstrates:
- Multi-embedding search (name, description, industry)
- Filtered search by metadata
- Similar company recommendations
- Performance metrics

### Custom Document Type

```python
from typing import TypedDict

class Product(TypedDict):
    name: str
    description: str
    price: float
    category: str

# Use with generic repository
repo: Repository[Product] = GenericRepository(...)
```

## Configuration 🔧

### Environment Variables

```bash
# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=your-key  # Optional

# OpenAI
OPENAI_API_KEY=your-key
OPENAI_MODEL=text-embedding-3-small

# Defaults
DEFAULT_COLLECTION=documents
BATCH_SIZE=100
```

### Programmatic Configuration

```python
from vector_system import VectorSystemConfig

config = VectorSystemConfig(
    qdrant_host="localhost",
    qdrant_port=6333,
    openai_api_key="your-key",
    batch_size=100
)
```

## Text Extractors 🔤

Built-in extractors for flexible text processing:

```python
# Single field
extract_field("description")

# Multiple fields
extract_concatenated("title", "description", separator=" - ")

# Template-based
extract_with_template(
    "Company: {name} in {industry}",
    name="company_name",
    industry="sector"
)

# Custom extractor
def extract_keywords(doc):
    return " ".join(doc.get("tags", []))
```

## Performance Tips 🚀

1. **Batch Operations**: Always prefer `add_many()` over multiple `add()` calls
2. **Index Optimization**: Call `optimize_index()` with payload fields for faster filtering
3. **Multi-Embedding**: Use prefetch for better search quality
4. **Async Operations**: Use `AsyncVectorSearchEngine` for high concurrency

## API Reference 📖

### VectorSearchEngine

```python
# Core operations
engine.add(document)              # Add single document
engine.add_many(documents)        # Add batch
engine.search(query, limit=10)    # Search
engine.get(id)                   # Get by ID
engine.remove(id)                # Delete

# Advanced
engine.filtered_search(query, filter)  # Search with filters
engine.recommend_similar(id)           # Find similar docs
engine.get_metrics()                   # Performance stats
```

### Filters

```python
# Exact match
{"category": "technology"}

# Range query
{"price": {"gte": 10, "lte": 100}}

# Multiple conditions
{
    "category": "books",
    "rating": {"gte": 4.0}
}
```

## Error Handling 🚨

```python
from vector_system.core import VectorSystemError, IndexNotFoundError

try:
    engine.search("query")
except IndexNotFoundError:
    # Create index first
    pass
except VectorSystemError as e:
    # Handle other errors
    print(f"Error: {e}")
```

## Contributing 🤝

1. Use type hints everywhere
2. Follow clean architecture principles
3. Add tests for new features
4. Update documentation

## License 📄

MIT License - see LICENSE file for details 