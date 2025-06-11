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
    best_guess_columns: Dict[str, List[str]],
    model: str = "gpt-4o-mini"
) -> FilterParams:    
    prompt = f"""
You are an expert in translating user queries into precise database filters for a company database.

Available columns and their valid example values:
{best_guess_columns}

Query: "{query}"

Your task is to generate optimal MUST and SHOULD filters to accurately and effectively narrow down search results.

**Important rules for filtering:**
- **Geographic Filters:** 
  - Only apply country filters if explicitly requested in the query (e.g., "in Germany", "EU companies", "based in France").
  - If the query specifies a broader region (e.g., "EU"), include relevant countries explicitly and ensure the logic uses a SHOULD filter with all applicable countries listed (not MUST, as a company can't simultaneously be in multiple countries).
- **Size Filters:** Apply MUST filters if company size is explicitly stated or strongly implied (e.g., "over 10,000 employees", "mid-sized").
- **Industry Filters:** 
  - Apply MUST filters if industries are explicitly mentioned as mandatory.
  - Use SHOULD filters for broader industry categories or when multiple related terms are given.
- **Name or Description Filters:** 
  - Use SHOULD filters for partial matches or keywords that companies might include in their name or description.

**Example:**
- "B2B companies in EU":
  - SHOULD filter on `country` with EU member countries (Germany, France, Spain, Netherlands, etc.).
  - No size filter unless explicitly mentioned.
  - Industry is general ("B2B"), thus typically no specific industry filter unless context strongly indicates one.

Return a concise, logical explanation for your filter choices, clearly differentiating MUST and SHOULD filters based on the rules above.

Provide filtering parameters structured clearly:
- `must_filters`: conditions that must ALL be satisfied (AND logic).
- `should_filters`: at least ONE of these conditions should be satisfied (OR logic).

Ensure filters directly reflect the user's intent and accurately address geographical and categorical nuances.
"""

    return create_structured_output(prompt, FilterParams, model)