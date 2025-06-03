from qdrant_client import QdrantClient
from core.indexes.vocab_bm25 import VocabBM25Index
from core.vector_config import CompanyVocabBM25Builder

# Initialize client and index
client = QdrantClient("localhost", port=6333)
vocab_index = VocabBM25Index(client, CompanyVocabBM25Builder)

# Example usage of the 4 main goals:

# 1. Add item to vocab
company_data = {
    "name": "OpenAI",
    "primary_industry": "Artificial Intelligence",
    "country": "United States",
    "size": "Large"
}
vocab_index.add_item_to_vocab(company_data)

# 2. Get all vocab names (columns)
columns = vocab_index.get_columns()
print("Available columns:", columns)  # ['name', 'primary_industry', 'country', 'size']

# 3. Quick search based on search query in the vocab
similar_industries = vocab_index.search_vocab_similarity("primary_industry", "AI technology", limit=5)
print("Similar industries:", similar_industries)

similar_countries = vocab_index.search_vocab_similarity("country", "USA", limit=3)
print("Similar countries:", similar_countries)

# 4. Create must/should filters

# Create MUST filter (AND condition) for single field
must_filter = vocab_index.create_must_filter("country", "United States")
results = vocab_index.search_with_filter(must_filter, limit=10)

# Create SHOULD filter (OR conditions) with multiple fields
should_filter = vocab_index.create_should_filter({
    "primary_industry": "technology",
    "country": "United States"
})
results = vocab_index.search_with_filter(should_filter, limit=10)

# Create SHOULD filter for single query across multiple columns
should_filter_single = vocab_index.create_should_filter_single_query("technology", ["name", "primary_industry"])
results = vocab_index.search_with_filter(should_filter_single, limit=10)

# You can also use it across all columns by not specifying columns parameter
should_filter_all = vocab_index.create_should_filter_single_query("OpenAI")
results = vocab_index.search_with_filter(should_filter_all, limit=10)

# NEW: Create combined filter with both SHOULD and MUST conditions
combined_filter = vocab_index.create_combined_filter(
    should_filters={"name": "OpenAI", "primary_industry": "AI"},
    must_filters={"country": "United States"}
)
results = vocab_index.search_with_filter(combined_filter, limit=10)

print("Search completed successfully!") 