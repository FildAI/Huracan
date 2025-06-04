import os
from typing import Dict, Any, Optional, List
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field
from dotenv import load_dotenv

load_dotenv()

class RelevantColumn(BaseModel):
    column: str = Field(..., description="The column name that is relevant to the query")
    keywords: List[str] = Field(..., description="List of keywords that are relevant to the column")

class RelevantColumnsGuess(BaseModel):
    items: List[RelevantColumn] = Field(..., description="List of columns that are relevant to the query")

class FilterParam(BaseModel):
    key: str = Field(..., description="The column name that is relevant to the query")
    query: str = Field(..., description="The query text that is relevant to the column")
    type: str = Field(..., description="The type of the filter: 'must' or 'should'")


class FilterParams(BaseModel):
    items: List[FilterParam] = Field(..., description="List of filters")

def create_structured_output(
    prompt: str,
    response_model: BaseModel,
    model: str = "gpt-4o-mini"
) -> BaseModel:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    client = OpenAI(api_key=api_key)
    
    try:
        completion = client.responses.parse(
            model=model,
            input=[
                {"role": "user", "content": prompt}
            ],
            text_format=response_model
        )
        
        return completion.output_parsed
    
    except Exception as e:
        print(f"Error creating structured output: {e}")
        # Return empty model as fallback
        return response_model()

def get_keywords_guess(query: str, available_columns: List[str], model: str = "gpt-4o-mini") -> List[str]:
    prompt = f"""
        You are a helpful assistant that can find the column filters for a given query.

        You will be given user query and list of columns for companies. You MUST return the list of columns that you think the most relevant dataset filters and keywords for futher semantic search.

        The main idea is user give you a query and then based on columns that you find — we will filter the dataset to limit number of companies that we will use for the search and keywords for futher semantic search (to find the most relevant companies).

        2 things:
        1) Relevant columns based on existing columns in the dataset (NEVER make up new columns or use columns that are not in the dataset)
        2) Keywords for futher semantic search (UP TO 5 keywords)

        User query: {query}
        Available columns: {', '.join(available_columns)}

        Return the list of columns that you think the most relevant dataset filters and keywords for futher semantic search.
        """

    return create_structured_output(prompt, RelevantColumnsGuess, model)


def get_filter_params(
    query: str,
    best_guess_columns: List[str],
    model: str = "gpt-4o-mini"
) -> FilterParams:    
    prompt = f"""
You are an expert at analyzing search queries to create optimal database filters for company data.

You have available columns and valid values: {str(best_guess_columns)}

Your goal is to create must and should filters using these columns for the given query.

Your final filters should be broad enough, focusing only on adding filters that will help to filter out companies that are not relevant to the query.
Therefore, don't add filters that are too specific.

Query: "{query}"

Analyze this query and determine the best filtering strategy:

1. MUST filters (AND conditions) - Use these for:
   - Exact requirements that must be met
   - Location/geographic constraints  
   - Size requirements
   - Specific industry requirements when explicitly stated

2. SHOULD filters (OR conditions) - Use these for:
   - Flexible/alternative matching
   - When query mentions multiple related terms
   - Broader categorical matches
   - Name variations or similar terms

Guidelines:
- If query mentions specific location (country, city, region), add to must_filters
- If query mentions specific company size, add to must_filters  
- If query mentions broad industry terms, consider should_filters for flexibility
- If query mentions company names or partial names, add to should_filters on 'name' field
- If query is very specific about industry, add to must_filters
- If query has multiple industry terms, use should_filters
- Keep reasoning concise but explain your filtering strategy

Return the filtering parameters that would best match the user's intent.
"""
    
    return create_structured_output(prompt, FilterParams, model) 