"""Example usage of company text extractors for embeddings."""

from core.embedding.company_extractors import (
    # Individual field extractors
    extract_company_name,
    extract_company_description, 
    extract_company_industry,
    extract_company_location_full,
    
    # Composite extractors
    extract_company_full_text,
    extract_company_structured_text,
    extract_company_search_text,
    extract_company_metadata,
    
    # Convenience functions
    get_company_extractor,
    create_company_multi_embedding_processor,
    COMPANY_EXTRACTOR_CONFIGS
)

# Example company data
sample_company = {
    "name": "TechCorp Solutions",
    "description": "Leading provider of cloud-based enterprise software solutions for manufacturing and logistics",
    "primary_industry": "Technology", 
    "size": "500-1000 employees",
    "type": "Private Company",
    "location": "San Francisco, CA",
    "country": "United States",
    "domain": "techcorp.com",
    "linkedin_url": "https://linkedin.com/company/techcorp-solutions"
}

def demonstrate_individual_extractors():
    """Show how to use individual field extractors."""
    print("=== Individual Field Extractors ===")
    
    # Extract individual fields
    name_extractor = extract_company_name()
    description_extractor = extract_company_description()
    industry_extractor = extract_company_industry()
    location_extractor = extract_company_location_full()
    
    print(f"Name: {name_extractor(sample_company)}")
    print(f"Description: {description_extractor(sample_company)}")
    print(f"Industry: {industry_extractor(sample_company)}")
    print(f"Location: {location_extractor(sample_company)}")
    print()

def demonstrate_composite_extractors():
    """Show how to use composite text extractors."""
    print("=== Composite Text Extractors ===")
    
    # Full text extraction (without URLs)
    full_text = extract_company_full_text()
    print("Full Text (no URLs):")
    print(full_text(sample_company))
    print()
    
    # Full text with URLs
    full_text_urls = extract_company_full_text(include_urls=True)
    print("Full Text (with URLs):")
    print(full_text_urls(sample_company))
    print()
    
    # Structured text with labels
    structured_text = extract_company_structured_text()
    print("Structured Text:")
    print(structured_text(sample_company))
    print()
    
    # Search-optimized text
    search_text = extract_company_search_text()
    print("Search Optimized:")
    print(search_text(sample_company))
    print()
    
    # Metadata only
    metadata_text = extract_company_metadata()
    print("Metadata Only:")
    print(metadata_text(sample_company))
    print()

def demonstrate_predefined_extractors():
    """Show how to use predefined extractor configurations."""
    print("=== Predefined Extractor Configurations ===")
    
    print("Available extractors:", list(COMPANY_EXTRACTOR_CONFIGS.keys()))
    print()
    
    # Use predefined extractors
    for extractor_name in ["full_text", "structured", "search_optimized", "metadata_only"]:
        extractor = get_company_extractor(extractor_name)
        result = extractor(sample_company)
        print(f"{extractor_name}:")
        print(result)
        print()

def demonstrate_multi_embedding_processor():
    """Show how to create and use multi-embedding processor for companies."""
    print("=== Multi-Embedding Processor ===")
    
    # Create processor with sentence transformers only
    processor_local = create_company_multi_embedding_processor()
    
    print("Vector configurations:")
    configs = processor_local.get_embedding_configs()
    for name, config in configs.items():
        print(f"  {name}: {config.dimension}D, {config.distance_metric}")
    print()
    
    # Process single company
    embeddings = processor_local.process_document(sample_company)
    print("Generated embeddings:")
    for name, vector in embeddings.items():
        print(f"  {name}: {len(vector)}D vector")
    print()
    
    # Example with OpenAI (requires API key)
    # processor_openai = create_company_multi_embedding_processor(
    #     openai_api_key="your-api-key-here"
    # )

def demonstrate_custom_multi_processor():
    """Show how to create custom multi-embedding processor."""
    print("=== Custom Multi-Embedding Processor ===")
    
    from core.embedding.multi_processor import StandardMultiEmbeddingProcessor
    
    # Build custom processor
    builder = StandardMultiEmbeddingProcessor.builder()
    
    # Add specific embeddings for different use cases
    builder.add_sentence_transformer(
        text_extractor=extract_company_name(),
        vector_name="company_name_only",
        model_name="all-MiniLM-L6-v2"
    )
    
    builder.add_sentence_transformer(
        text_extractor=extract_company_description(),
        vector_name="company_description_only", 
        model_name="all-MiniLM-L6-v2"
    )
    
    builder.add_sentence_transformer(
        text_extractor=extract_company_full_text(include_urls=True, field_separator=" | "),
        vector_name="company_custom_format",
        model_name="all-MiniLM-L6-v2"
    )
    
    processor = builder.build()
    
    # Process company
    embeddings = processor.process_document(sample_company)
    print("Custom embeddings generated:")
    for name, vector in embeddings.items():
        print(f"  {name}: {len(vector)}D vector")
    print()

def demonstrate_extensibility():
    """Show how to handle additional fields."""
    print("=== Extensibility with Additional Fields ===")
    
    # Company with additional fields
    extended_company = sample_company.copy()
    extended_company.update({
        "founded_year": "2015",
        "funding_stage": "Series B",
        "employee_count": "750",
        "technologies": ["Python", "AWS", "Kubernetes"]
    })
    
    # Extract with additional fields
    extended_extractor = extract_company_full_text(
        additional_fields=["founded_year", "funding_stage", "employee_count"]
    )
    
    print("Extended company text:")
    print(extended_extractor(extended_company))
    print()
    
    # Custom extractor for technologies
    def extract_technologies(doc):
        techs = doc.get("technologies", [])
        if isinstance(techs, list):
            return ", ".join(techs)
        return str(techs)
    
    # Build processor with technology embedding
    builder = StandardMultiEmbeddingProcessor.builder()
    builder.add_sentence_transformer(
        text_extractor=extract_technologies,
        vector_name="company_technologies",
        model_name="all-MiniLM-L6-v2"
    )
    
    tech_processor = builder.build()
    tech_embeddings = tech_processor.process_document(extended_company)
    
    print("Technology embedding generated:")
    for name, vector in tech_embeddings.items():
        print(f"  {name}: {len(vector)}D vector")

if __name__ == "__main__":
    demonstrate_individual_extractors()
    demonstrate_composite_extractors() 
    demonstrate_predefined_extractors()
    demonstrate_multi_embedding_processor()
    demonstrate_custom_multi_processor()
    demonstrate_extensibility() 