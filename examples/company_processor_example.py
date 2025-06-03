"""Example usage of the super easy, transparent CompanyProcessor."""

from core.embedding.company import (
    CompanyProcessor,
    CompanyEmbeddingType,
    process_company,
    process_companies,
    get_search_processor,
    get_classification_processor,
    get_comprehensive_processor
)

# Example company data
sample_companies = [
    {
        "name": "TechCorp Solutions",
        "description": "Leading provider of cloud-based enterprise software solutions for manufacturing and logistics",
        "primary_industry": "Technology", 
        "size": "500-1000 employees",
        "type": "Private Company",
        "location": "San Francisco, CA",
        "country": "United States",
        "domain": "techcorp.com",
        "linkedin_url": "https://linkedin.com/company/techcorp-solutions"
    },
    {
        "name": "DataFlow Analytics",
        "description": "Advanced analytics and AI-powered data processing solutions for enterprises",
        "primary_industry": "Data Analytics",
        "size": "50-100 employees", 
        "type": "Startup",
        "location": "Austin, TX",
        "country": "United States",
        "domain": "dataflow.com",
        "linkedin_url": "https://linkedin.com/company/dataflow-analytics"
    },
    {
        "name": "Green Energy Systems",
        "description": "Renewable energy solutions and sustainable technology development",
        "primary_industry": "Energy",
        "size": "1000+ employees",
        "type": "Public Company",
        "location": "Berlin",
        "country": "Germany",
        "domain": "greenenergy.de",
        "linkedin_url": "https://linkedin.com/company/green-energy-systems"
    }
]

def demonstrate_super_simple_usage():
    """The absolute simplest way to use the processor."""
    print("=== SUPER SIMPLE USAGE ===")
    print("# Just one line to get comprehensive embeddings!\n")
    
    # Option 1: One-liner function
    print("## Option 1: One-liner function")
    embeddings = process_company(sample_companies[0], openai_api_key="your-openai-key")
    print(f"Generated {len(embeddings)} embeddings with one line of code!")
    for name, vector in embeddings.items():
        print(f"  {name}: {len(vector)}D vector")
    print()
    
    # Option 2: Quick start class method
    print("## Option 2: Quick start class method")
    processor = CompanyProcessor.quick_start(openai_api_key="your-openai-key")
    embeddings = processor.process(sample_companies[0])
    print(f"Generated {len(embeddings)} embeddings with quick start!")
    print()

def demonstrate_batch_processing():
    """Show how easy batch processing is."""
    print("=== BATCH PROCESSING ===")
    print("# Process multiple companies with one call\n")
    
    # One-liner batch processing
    batch_embeddings = process_companies(
        sample_companies, 
        openai_api_key="your-openai-key",
        profile="comprehensive"
    )
    
    print(f"Processed {len(batch_embeddings)} companies in one call!")
    for i, embeddings in enumerate(batch_embeddings):
        company_name = sample_companies[i]["name"]
        print(f"  {company_name}: {len(embeddings)} embeddings")
    print()

def demonstrate_predefined_profiles():
    """Show different predefined profiles for common use cases."""
    print("=== PREDEFINED PROFILES ===")
    print("# Different profiles for different use cases\n")
    
    # List available profiles
    profiles = CompanyProcessor.list_profiles()
    print("Available profiles:")
    for name, description in profiles.items():
        print(f"  {name}: {description}")
    print()
    
    # Use different profiles
    use_cases = [
        ("search", "Company search and matching"),
        ("classification", "Industry classification"),
        ("lightweight", "Basic embeddings only"),
        ("comprehensive", "All embedding types")
    ]
    
    for profile_name, use_case in use_cases:
        print(f"## {use_case}")
        processor = CompanyProcessor(profile=profile_name)
        embeddings = processor.process(sample_companies[0])
        
        info = processor.get_profile_info()
        print(f"Profile: {info['name'] if 'name' in info else profile_name}")
        print(f"Local embeddings: {info['local_embeddings']}")
        print(f"Generated {len(embeddings)} embeddings")
        print()

def demonstrate_specialized_processors():
    """Show specialized processor factory methods."""
    print("=== SPECIALIZED PROCESSORS ===")
    print("# Pre-configured processors for specific tasks\n")
    
    # Search processor
    print("## Search Processor")
    search_processor = get_search_processor()
    search_embeddings = search_processor.process(sample_companies[0])
    print(f"Search embeddings: {len(search_embeddings)} vectors")
    print("Perfect for: Company search, matching, similarity")
    print()
    
    # Classification processor
    print("## Classification Processor")
    classification_processor = get_classification_processor()
    class_embeddings = classification_processor.process(sample_companies[0])
    print(f"Classification embeddings: {len(class_embeddings)} vectors")
    print("Perfect for: Industry classification, categorization")
    print()
    
    # Comprehensive processor
    print("## Comprehensive Processor")
    comprehensive_processor = get_comprehensive_processor()
    comp_embeddings = comprehensive_processor.process(sample_companies[0])
    print(f"Comprehensive embeddings: {len(comp_embeddings)} vectors")
    print("Perfect for: General-purpose company processing")
    print()

def demonstrate_custom_configuration():
    """Show custom embedding configurations."""
    print("=== CUSTOM CONFIGURATION ===")
    print("# Configure exactly what embeddings you want\n")
    
    # Custom embedding selection
    custom_embeddings = {
        "company_name": CompanyEmbeddingType.NAME,
        "company_description": CompanyEmbeddingType.DESCRIPTION,
        "company_business_info": CompanyEmbeddingType.METADATA
    }
    
    processor = CompanyProcessor(
        custom_embeddings=custom_embeddings,
        openai_api_key="your-openai-key"
    )
    
    embeddings = processor.process(sample_companies[0])
    print("Custom embeddings generated:")
    for name, vector in embeddings.items():
        print(f"  {name}: {len(vector)}D vector")
    print()

def demonstrate_embedding_info():
    """Show how to get information about embeddings."""
    print("=== EMBEDDING INFORMATION ===")
    print("# Get detailed information about your embeddings\n")
    
    processor = CompanyProcessor(
        openai_api_key="your-openai-key",
        profile="comprehensive"
    )
    
    # Get embedding details
    embedding_info = processor.get_embedding_info()
    print("Embedding Details:")
    for name, info in embedding_info.items():
        print(f"  {name}:")
        print(f"    Dimension: {info['dimension']}")
        print(f"    Type: {info['type']}")
        print(f"    Model: {info['model']}")
        print(f"    Distance: {info['distance_metric']}")
    print()
    
    # Get profile information
    profile_info = processor.get_profile_info()
    print("Profile Information:")
    print(f"  Name: {profile_info['name']}")
    print(f"  Description: {profile_info['description']}")
    print(f"  OpenAI enabled: {profile_info['openai_enabled']}")
    print(f"  Local embeddings: {profile_info['local_embeddings']}")
    print(f"  OpenAI embeddings: {profile_info['openai_embeddings']}")
    print()

def demonstrate_no_openai_usage():
    """Show usage without OpenAI (local only)."""
    print("=== LOCAL ONLY (NO OPENAI) ===")
    print("# Works perfectly without OpenAI API key\n")
    
    # No API key provided - only local embeddings
    processor = CompanyProcessor(profile="comprehensive")
    embeddings = processor.process(sample_companies[0])
    
    print(f"Generated {len(embeddings)} local embeddings (no OpenAI key needed)")
    for name, vector in embeddings.items():
        print(f"  {name}: {len(vector)}D vector")
    print()
    
    # Show what's available without OpenAI
    info = processor.get_profile_info()
    print("Available without OpenAI:")
    print(f"  Local embeddings: {info['local_embeddings']}")
    print(f"  OpenAI embeddings: {info['openai_embeddings']}")
    print()

def demonstrate_real_world_scenarios():
    """Show real-world usage scenarios."""
    print("=== REAL-WORLD SCENARIOS ===")
    print("# Practical examples for common tasks\n")
    
    # Scenario 1: Company search system
    print("## Scenario 1: Company Search System")
    search_processor = CompanyProcessor.for_search()
    
    # Process all companies for search index
    search_embeddings = search_processor.process_batch(sample_companies)
    print(f"✅ Built search index for {len(search_embeddings)} companies")
    print("Use case: Find similar companies, company matching")
    print()
    
    # Scenario 2: Industry classification
    print("## Scenario 2: Industry Classification")
    classifier = CompanyProcessor.for_classification()
    
    # Process for classification
    class_embeddings = classifier.process_batch(sample_companies)
    print(f"✅ Generated classification embeddings for {len(class_embeddings)} companies")
    print("Use case: Automatically categorize companies by industry")
    print()
    
    # Scenario 3: Data analysis pipeline
    print("## Scenario 3: Data Analysis Pipeline")
    processor = CompanyProcessor.quick_start()
    
    # Process comprehensive data
    all_embeddings = processor.process_batch(sample_companies)
    print(f"✅ Processed {len(all_embeddings)} companies for analysis")
    print("Use case: ML features, similarity analysis, clustering")
    print()

def demonstrate_error_handling():
    """Show graceful error handling."""
    print("=== ERROR HANDLING ===")
    print("# Graceful handling of common issues\n")
    
    try:
        # Invalid profile
        processor = CompanyProcessor(profile="nonexistent")
    except Exception as e:
        print(f"❌ Invalid profile handled: {e}")
    
    try:
        # Missing fields are handled gracefully
        incomplete_company = {"name": "Test Corp"}  # Missing other fields
        processor = CompanyProcessor()
        embeddings = processor.process(incomplete_company)
        print(f"✅ Incomplete company handled: {len(embeddings)} embeddings generated")
    except Exception as e:
        print(f"Error with incomplete company: {e}")
    
    print()

if __name__ == "__main__":
    print("🚀 COMPANY PROCESSOR - SUPER EASY EMBEDDINGS 🚀\n")
    print("This shows how incredibly easy it is to process companies!\n")
    
    demonstrate_super_simple_usage()
    demonstrate_batch_processing()
    demonstrate_predefined_profiles()
    demonstrate_specialized_processors()
    demonstrate_custom_configuration()
    demonstrate_embedding_info()
    demonstrate_no_openai_usage()
    demonstrate_real_world_scenarios()
    demonstrate_error_handling()
    
    print("🎉 That's it! Super easy company processing with comprehensive embeddings!")
    print("\nNext steps:")
    print("1. Replace 'your-openai-key' with real API key for OpenAI embeddings")
    print("2. Choose the right profile for your use case")
    print("3. Process your company data with one line of code!") 