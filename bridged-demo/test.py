# test_agent.py
import json
import os
from dotenv import load_dotenv
from main import PineconeQueryAgent, DateGranularity

# Load environment variables
load_dotenv()

def test_regex_extraction():
    print("=== Testing Regex-based Extraction ===\n")
    
    agent = PineconeQueryAgent(
        openai_api_key="dummy-key",
        date_granularity=DateGranularity.YEAR_MONTH_DAY
    )

    test_cases = [
        {
            "query": "Show me articles by Jane Doe from last year about IPL and injuries",
            "expected": {
                "author": "Jane Doe",
                "tags": {"$in": ["IPL", "injuries"]},
                "published_year": {"$eq": 2024}
            }
        },
        {
            "query": "Find posts tagged with IPL published in May 2025",
            "expected": {
                "tags": {"$in": ["IPL"]},
                "published_year": {"$eq": 2025},
                "published_month": {"$eq": 5}
            }
        },
        {
            "query": "Articles from this year by Jane Doe",
            "expected": {
                "author": "Jane Doe",
                "published_year": {"$eq": 2025}
            }
        }
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"Test Case {i}:")
        print(f"Query: {test['query']}")
        result = agent.process_query(test['query'], use_llm=False)
        print("Expected:", json.dumps(test["expected"], indent=2))
        print("Actual:  ", json.dumps(result, indent=2))
        print(f"Match: {result == test['expected']}")
        print("-" * 60)

def test_llm_extraction():
    print("\n=== Testing LLM-based Extraction ===\n")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("OPENAI_API_KEY not set, skipping LLM tests")
        return

    agent = PineconeQueryAgent(
        openai_api_key=openai_api_key,
        date_granularity=DateGranularity.YEAR_MONTH_DAY
    )

    queries = [
        "Who wrote about IPL and injuries in May 2025?",
        "Get all IPL posts by Jane Doe from last year",
        "Anything on injuries by Jane?"
    ]

    for query in queries:
        print(f"Query: {query}")
        result = agent.process_query(query)
        print("LLM Output:", json.dumps(result, indent=2))
        print("-" * 60)

if __name__ == "__main__":
    # test_regex_extraction()
    test_llm_extraction()
