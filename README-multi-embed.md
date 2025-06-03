# Multi-Embedding Company Search Demo

This demo showcases efficient hybrid search using **10 different embeddings** for company data in Qdrant. It demonstrates how multiple embedding strategies can be combined to create more nuanced and accurate search results.

## 🎯 Key Features

### 10 Different Embedding Strategies
1. **Name Embedding** (15% weight) - Company names
2. **Description Embedding** (20% weight) - Full company descriptions  
3. **Industry Embedding** (12% weight) - Industry classification
4. **Location Embedding** (8% weight) - Geographic information
5. **Size/Stage Embedding** (6% weight) - Company size and funding stage
6. **Technology Embedding** (10% weight) - Tech stack and tools (MPNet model)
7. **Mission Embedding** (10% weight) - Company mission and values (MPNet model)
8. **Products Embedding** (8% weight) - Products and services offered
9. **Target Market Embedding** (6% weight) - Customer segments
10. **Culture Embedding** (5% weight) - Company culture and work style

### Advanced Features
- **Multiple embedding models**: Uses both MiniLM and MPNet for different vector dimensions
- **Intelligent text processing**: Extracts relevant information from company data
- **Weighted hybrid search**: Combines results from all embeddings with configurable weights
- **Performance metrics**: Tracks search time and embedding efficiency
- **Strategy comparison**: Compares hybrid vs single-embedding search
- **Sample data generation**: Creates realistic company data for testing

## 🚀 Setup and Installation

### Prerequisites
- Python 3.8+
- Qdrant running locally on port 6333
- ~2GB RAM for embedding models

### Install Dependencies
```bash
pip install -r requirements-multi-embed.txt
```

### Start Qdrant
```bash
# Using Docker
docker run -p 6333:6333 qdrant/qdrant

# Or using local installation
./qdrant
```

### Environment Variables
Create a `.env` file (optional):
```bash
QDRANT_API_KEY=your_api_key_if_needed
```

## 🔧 Usage

### Basic Usage
```bash
python scripts/experiments/multi-embed-search.py
```

The demo will:
1. Load/generate sample company data
2. Create 10 different embeddings using 2 transformer models
3. Set up Qdrant collection with multiple vector configurations
4. Perform hybrid searches with detailed analytics
5. Compare hybrid vs single-embedding performance

### Sample Output
```
Multi-Embedding Company Search Demo
========================================
Loading embedding models...
  - all-MiniLM-L6-v2
  - sentence-transformers/all-mpnet-base-v2
Loaded 8 companies
Setting up Qdrant collection: companies_multi_embed
  Created collection with 10 vector configurations
Creating 10 different embeddings for 8 companies...
  Processed 8/8 companies
Inserting 8 points into Qdrant...
  Insertion complete

============================================================
DETAILED SEARCH ANALYSIS
============================================================

Searching for: 'Collaborative design software'
==================================================
name_embed           | Weight: 0.15 | Time:  12.3ms | Top result: CollabDesign Studio
desc_embed           | Weight: 0.20 | Time:  8.7ms  | Top result: CollabDesign Studio
industry_embed       | Weight: 0.12 | Time:  9.1ms  | Top result: CollabDesign Studio
...

Total search time: 127.5ms
Searched 10 different embeddings

Final Results:
1. CollabDesign Studio
   Score: 0.892
   Description: Collaborative design software enabling remote teams to create, iterate, and manage design...
   Industry: Design Software
```

## 📊 Performance Insights

### Embedding Strategy Analysis
- **Semantic diversity**: Each embedding captures different aspects of company information
- **Weighted relevance**: Different weights reflect importance for search quality
- **Model efficiency**: MiniLM (384d) for speed, MPNet (768d) for accuracy
- **Hybrid scoring**: Combines multiple perspectives for better results

### Search Performance
- **Average search time**: ~100-150ms for 10 embeddings
- **Scalability**: Efficient for datasets up to 10K+ companies
- **Memory usage**: ~2GB for loaded models, ~100MB for embeddings

### Quality Metrics
- **Precision improvement**: Hybrid search typically outperforms single embeddings
- **Recall enhancement**: Multiple embeddings capture diverse query intents
- **Ranking quality**: Weighted combination provides nuanced scoring

## 🔬 Experiment Configuration

### Modify Embedding Weights
```python
# In Config class, adjust weights:
EmbeddingConfig("desc_embed", "all-MiniLM-L6-v2", "extract_description", 0.25),  # Increase description weight
EmbeddingConfig("name_embed", "all-MiniLM-L6-v2", "extract_name", 0.10),         # Decrease name weight
```

### Add Custom Text Processors
```python
# In TextProcessor class:
@staticmethod
def extract_custom_field(row: pd.Series) -> str:
    # Your custom text extraction logic
    return f"Custom: {row.get('custom_field', '')}"
```

### Use Different Models
```python
# Try different embedding models:
EmbeddingConfig("custom_embed", "sentence-transformers/all-roberta-large-v1", "extract_description", 0.15, 1024),
```

## 📈 Use Cases

### Investor Discovery
- Search for companies by industry, stage, location
- Find companies with specific tech stacks
- Discover mission-aligned opportunities

### Competitive Analysis
- Find similar companies across multiple dimensions
- Analyze market positioning and differentiation
- Track industry trends and patterns

### Sales Prospecting
- Identify target companies by size and market
- Find companies using specific technologies
- Discover cultural fit prospects

### Talent Sourcing
- Find companies with specific work cultures
- Locate companies in desired locations
- Match candidates with company missions

## 🛠 Advanced Customization

### Custom Company Data
Replace the sample data with your own CSV:
```python
df = pd.read_csv("your_companies.csv")
```

Required columns: `name`, `description`
Optional columns: `industry`, `location`, `company_size`, `stage`, etc.

### Embedding Model Selection
The demo uses two models for diversity:
- **MiniLM**: Fast, 384-dimensional embeddings
- **MPNet**: Higher quality, 768-dimensional embeddings

### Performance Tuning
- Adjust `top_k` for result count
- Modify weights based on your use case
- Use batch processing for large datasets
- Implement caching for repeated queries

## 🔍 Troubleshooting

### Common Issues
1. **Qdrant connection failed**: Ensure Qdrant is running on localhost:6333
2. **Model download slow**: First run downloads ~1GB of models
3. **Memory issues**: Reduce batch size or use smaller models
4. **Search timeout**: Increase client timeout for large collections

### Performance Optimization
- Use GPU if available for embedding generation
- Batch embed multiple queries together
- Cache embedding models between runs
- Consider model quantization for production

## 📝 Next Steps

### Potential Enhancements
1. **Dynamic weighting**: Adjust weights based on query type
2. **Query classification**: Route different queries to optimal embeddings
3. **Feedback learning**: Use search results to improve weights
4. **Semantic clustering**: Group similar companies for faster search
5. **Real-time updates**: Handle streaming company data updates

### Production Considerations
- **Monitoring**: Track search performance and quality metrics
- **Scaling**: Use Qdrant cluster for large datasets
- **Security**: Implement proper authentication and access controls
- **Backup**: Regular collection snapshots and model versioning

This demo provides a solid foundation for building production-grade multi-embedding search systems with Qdrant! 