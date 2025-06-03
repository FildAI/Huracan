"""Super simple CompanyProcessor example - minimal and easy!"""

from core.embedding.company import CompanyProcessor, process_company, process_companies

# Sample company data
company = {
    "name": "TechCorp",
    "description": "AI software company",
    "primary_industry": "Technology",
    "location": "San Francisco"
}

companies = [
    {"name": "TechCorp", "description": "AI software", "primary_industry": "Technology"},
    {"name": "DataFlow", "description": "Analytics platform", "primary_industry": "Data"},
    {"name": "GreenTech", "description": "Solar energy", "primary_industry": "Energy"}
]

def main():
    print("🚀 SUPER SIMPLE COMPANY PROCESSOR 🚀\n")
    
    # METHOD 1: One-liner functions (easiest!)
    print("=== ONE-LINER FUNCTIONS ===")
    
    # Single company
    embeddings = process_company(company, "your-openai-key")
    print(f"✅ Single company: {len(embeddings)} embeddings")
    
    # Multiple companies  
    batch_embeddings = process_companies(companies, "your-openai-key")
    print(f"✅ Batch companies: {len(batch_embeddings)} companies processed")
    print()
    
    # METHOD 2: Class usage (still super simple!)
    print("=== CLASS USAGE ===")
    
    # With OpenAI
    processor = CompanyProcessor("your-openai-key")
    embeddings = processor.process(company)
    print(f"✅ With OpenAI: {len(embeddings)} embeddings")
    
    # Without OpenAI (still works!)
    processor = CompanyProcessor()
    embeddings = processor.process(company)
    print(f"✅ Without OpenAI: {len(embeddings)} embeddings")
    
    # Info about embeddings
    info = processor.info()
    print(f"📊 Total: {info['total_embeddings']}, Local: {info['local_embeddings']}, OpenAI: {info['openai_embeddings']}")
    print()
    
    # METHOD 3: Batch processing
    print("=== BATCH PROCESSING ===")
    
    processor = CompanyProcessor("your-openai-key")
    batch_results = processor.process_batch(companies)
    
    for i, embeddings in enumerate(batch_results):
        company_name = companies[i]["name"]
        print(f"✅ {company_name}: {len(embeddings)} embeddings")
    
    print("\n🎉 That's it! Super simple company processing!")
    print("\nJust remember:")
    print("1. CompanyProcessor('your-openai-key') - with OpenAI")
    print("2. CompanyProcessor() - without OpenAI") 
    print("3. processor.process(company) - single company")
    print("4. processor.process_batch(companies) - multiple companies")

if __name__ == "__main__":
    main() 