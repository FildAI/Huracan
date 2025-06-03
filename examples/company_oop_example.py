"""Example usage of OOP company text extractors for embeddings."""

from core.extractors.company import (
    # Extractor classes
    SingleFieldExtractor,
    FullTextExtractor,
    StructuredTextExtractor,
    SearchOptimizedExtractor,
    MetadataExtractor,
    LocationExtractor,
    TemplateExtractor,
    CompositeFieldExtractor,
    
    # Factory and registry
    CompanyExtractorFactory,
    CompanyExtractorRegistry,
    company_extractors,
    
    # Multi-embedding processor
    CompanyMultiEmbeddingProcessor,
    
    # Field definitions
    CompanyFields
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

def demonstrate_extractor_classes():
    """Show how to use extractor classes directly."""
    print("=== OOP Extractor Classes ===")
    
    # Single field extractors
    name_extractor = SingleFieldExtractor("name")
    description_extractor = SingleFieldExtractor("description")
    
    print(f"Name: {name_extractor.extract(sample_company)}")
    print(f"Description: {description_extractor.extract(sample_company)}")
    print()
    
    # Full text extractor with options
    full_text_extractor = FullTextExtractor(
        include_urls=True,
        field_separator=" | ",
        additional_fields=["founded_year"]
    )
    print("Full Text:")
    print(full_text_extractor.extract(sample_company))
    print()
    
    # Structured text extractor
    structured_extractor = StructuredTextExtractor(include_urls=False)
    print("Structured Text:")
    print(structured_extractor.extract(sample_company))
    print()
    
    # Search optimized extractor
    search_extractor = SearchOptimizedExtractor()
    print("Search Optimized:")
    print(search_extractor.extract(sample_company))
    print()
    
    # Metadata extractor
    metadata_extractor = MetadataExtractor()
    print("Metadata:")
    print(metadata_extractor.extract(sample_company))
    print()
    
    # Location extractor
    location_extractor = LocationExtractor()
    print("Location:")
    print(location_extractor.extract(sample_company))
    print()

def demonstrate_template_extractor():
    """Show how to use template extractor."""
    print("=== Template Extractor ===")
    
    # Simple template
    template1 = TemplateExtractor(
        template="{name} is a {size} {primary_industry} company based in {location}"
    )
    print("Simple Template:")
    print(template1.extract(sample_company))
    print()
    
    # Template with field mapping
    template2 = TemplateExtractor(
        template="Company: {company_name} | Industry: {industry} | Size: {company_size}",
        field_mapping={
            "company_name": "name",
            "industry": "primary_industry", 
            "company_size": "size"
        }
    )
    print("Template with Mapping:")
    print(template2.extract(sample_company))
    print()

def demonstrate_composite_extractor():
    """Show how to use composite field extractor."""
    print("=== Composite Field Extractor ===")
    
    # Combine specific fields
    contact_extractor = CompositeFieldExtractor(
        field_names=["domain", "linkedin_url"],
        separator=" | "
    )
    print("Contact Info:")
    print(contact_extractor.extract(sample_company))
    print()
    
    # Business info extractor
    business_extractor = CompositeFieldExtractor(
        field_names=["primary_industry", "type", "size"],
        separator=", "
    )
    print("Business Info:")
    print(business_extractor.extract(sample_company))
    print()

def demonstrate_factory_pattern():
    """Show how to use the factory pattern."""
    print("=== Factory Pattern ===")
    
    factory = CompanyExtractorFactory()
    
    # Create different extractors using factory
    extractors = [
        ("Single Field", factory.create_single_field("name")),
        ("Full Text", factory.create_full_text(include_urls=True)),
        ("Structured", factory.create_structured()),
        ("Search Optimized", factory.create_search_optimized()),
        ("Metadata", factory.create_metadata()),
        ("Location", factory.create_location()),
        ("Template", factory.create_template("Company {name} in {primary_industry}")),
        ("Composite", factory.create_composite(["name", "location"]))
    ]
    
    for name, extractor in extractors:
        result = extractor.extract(sample_company)
        print(f"{name}: {result}")
    print()

def demonstrate_registry_pattern():
    """Show how to use the registry pattern."""
    print("=== Registry Pattern ===")
    
    # List available extractors
    print("Available extractors:", company_extractors.list_extractors())
    print()
    
    # Use predefined extractors from registry
    predefined_names = ["full_text", "structured", "search_optimized", "metadata_only"]
    
    for name in predefined_names:
        extractor = company_extractors.get(name)
        result = extractor.extract(sample_company)
        print(f"{name}: {result}")
    print()
    
    # Register custom extractor
    custom_extractor = TemplateExtractor(
        template="🏢 {name} | 🏭 {primary_industry} | 📍 {location}"
    )
    company_extractors.register("emoji_summary", custom_extractor)
    
    # Use custom extractor
    custom_result = company_extractors.get("emoji_summary").extract(sample_company)
    print(f"Custom emoji extractor: {custom_result}")
    print()

def demonstrate_callable_interface():
    """Show how extractors can be used as callable objects."""
    print("=== Callable Interface ===")
    
    # Extractors can be called directly
    name_extractor = SingleFieldExtractor("name")
    description_extractor = FullTextExtractor()
    
    # Using __call__ method
    print(f"Direct call - Name: {name_extractor(sample_company)}")
    print(f"Direct call - Full: {description_extractor(sample_company)}")
    print()

def demonstrate_multi_embedding_processor():
    """Show how to use the OOP multi-embedding processor."""
    print("=== OOP Multi-Embedding Processor ===")
    
    # Create processor
    processor = CompanyMultiEmbeddingProcessor(
        sentence_transformer_model="all-MiniLM-L6-v2"
    )
    
    # Build and get configurations
    built_processor = processor.build_processor()
    configs = built_processor.get_embedding_configs()
    
    print("Vector configurations:")
    for name, config in configs.items():
        print(f"  {name}: {config.dimension}D, {config.distance_metric}")
    print()
    
    # Process document
    embeddings = processor.process_document(sample_company)
    print("Generated embeddings:")
    for name, vector in embeddings.items():
        print(f"  {name}: {len(vector)}D vector")
    print()

def demonstrate_custom_processor():
    """Show how to create custom multi-embedding processor."""
    print("=== Custom Multi-Embedding Processor ===")
    
    from core.embedding.multi_processor import StandardMultiEmbeddingProcessor
    
    # Create custom extractors
    factory = CompanyExtractorFactory()
    
    name_extractor = factory.create_single_field("name")
    description_extractor = factory.create_single_field("description")
    business_extractor = factory.create_composite(
        ["primary_industry", "type", "size"], " "
    )
    
    # Build custom processor
    builder = StandardMultiEmbeddingProcessor.builder()
    
    builder.add_sentence_transformer(
        text_extractor=name_extractor,
        vector_name="company_name_embedding",
        model_name="all-MiniLM-L6-v2"
    )
    
    builder.add_sentence_transformer(
        text_extractor=description_extractor,
        vector_name="company_description_embedding",
        model_name="all-MiniLM-L6-v2"
    )
    
    builder.add_sentence_transformer(
        text_extractor=business_extractor,
        vector_name="company_business_embedding",
        model_name="all-MiniLM-L6-v2"
    )
    
    custom_processor = builder.build()
    
    # Process company
    embeddings = custom_processor.process_document(sample_company)
    print("Custom embeddings generated:")
    for name, vector in embeddings.items():
        print(f"  {name}: {len(vector)}D vector")
    print()

def demonstrate_extensibility():
    """Show how to extend the system with custom extractors."""
    print("=== Extensibility ===")
    
    # Create custom extractor class
    class IndustryFocusedExtractor(SingleFieldExtractor):
        """Custom extractor that focuses on industry information."""
        
        def __init__(self):
            super().__init__("primary_industry")
        
        def extract(self, document):
            industry = super().extract(document)
            # Add custom processing
            if industry:
                return f"Industry: {industry.upper()}"
            return "Industry: Unknown"
    
    # Use custom extractor
    custom_extractor = IndustryFocusedExtractor()
    result = custom_extractor.extract(sample_company)
    print(f"Custom Industry Extractor: {result}")
    print()
    
    # Register in global registry
    company_extractors.register("industry_focused", custom_extractor)
    
    # Use from registry
    registered_result = company_extractors.get("industry_focused").extract(sample_company)
    print(f"From Registry: {registered_result}")
    print()

def demonstrate_batch_processing():
    """Show batch processing with OOP extractors."""
    print("=== Batch Processing ===")
    
    # Multiple companies
    companies = [
        sample_company,
        {
            "name": "DataFlow Inc",
            "description": "Advanced analytics and data processing solutions",
            "primary_industry": "Data Analytics",
            "size": "50-100 employees", 
            "type": "Startup",
            "location": "Austin, TX",
            "country": "United States",
            "domain": "dataflow.com",
            "linkedin_url": "https://linkedin.com/company/dataflow-inc"
        }
    ]
    
    # Process batch with multi-embedding processor
    processor = CompanyMultiEmbeddingProcessor()
    batch_embeddings = processor.process_batch(companies)
    
    print(f"Processed {len(batch_embeddings)} companies")
    for i, embeddings in enumerate(batch_embeddings):
        print(f"  Company {i+1}: {len(embeddings)} embedding types")
    print()

if __name__ == "__main__":
    demonstrate_extractor_classes()
    demonstrate_template_extractor()
    demonstrate_composite_extractor()
    demonstrate_factory_pattern()
    demonstrate_registry_pattern()
    demonstrate_callable_interface()
    demonstrate_multi_embedding_processor()
    demonstrate_custom_processor()
    demonstrate_extensibility()
    demonstrate_batch_processing() 