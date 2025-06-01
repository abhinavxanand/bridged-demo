import os
import json
import re
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

import openai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pinecone import Pinecone, ServerlessSpec
import pandas as pd
import numpy as np


class DateGranularity(Enum):
    """Enum for date granularity options"""
    FULL_DATE = "full_date"
    YEAR_MONTH_DAY = "year_month_day"


@dataclass
class QueryIntent:
    """Represents the extracted intent from natural language"""
    author: Optional[str] = None
    tags: Optional[List[str]] = None
    date_filters: Optional[Dict[str, Any]] = None
    year: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None


class NLQueryRequest(BaseModel):
    """FastAPI request model"""
    query: str
    date_format: str = "year_month_day"  # or "full_date"


class NLQueryResponse(BaseModel):
    """FastAPI response model"""
    filter: Dict[str, Any]
    
    
class PineconeQueryAgent:
    """Agent that converts natural language to Pinecone queries using LLM"""
    
    def __init__(self, openai_api_key: str, date_granularity: DateGranularity = DateGranularity.YEAR_MONTH_DAY):
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.date_granularity = date_granularity
        
    def extract_intent(self, query: str) -> QueryIntent:
        """Use LLM to extract structured intent from natural language"""
        
        current_year = datetime.now().year
        current_month = datetime.now().month
        
        system_prompt = f"""You are an expert at extracting structured information from natural language queries about articles/posts.
        Current date context: Year {current_year}, Month {current_month}
        
        Extract the following information if present:
        - author: The name of the author (full name as mentioned)
        - tags: List of topics/categories mentioned (normalize "LLMs" to "LLM")
        - year: Extract year (handle "last year" as {current_year - 1}, "this year" as {current_year})
        - month: Extract month as integer (1-12)
        - day: Extract day if specifically mentioned
        
        Rules:
        - For tags, preserve the original term but normalize variations (e.g., "LLMs" -> "LLM")
        - For relative dates like "last month", "this year", calculate the actual values
        - If no specific date is mentioned, leave date fields as null
        
        Return as JSON with keys: author, tags, year, month, day
        Set to null if not mentioned."""
        
        user_prompt = f"Extract structured information from this query: '{query}'"
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            extracted = json.loads(response.choices[0].message.content)
            
            return QueryIntent(
                author=extracted.get("author"),
                tags=extracted.get("tags"),
                year=extracted.get("year"),
                month=extracted.get("month"),
                day=extracted.get("day")
            )
            
        except Exception as e:
            raise Exception(f"LLM extraction failed: {e}")
    
    def build_pinecone_filter(self, intent: QueryIntent) -> Dict[str, Any]:
        """Convert intent to Pinecone filter format"""
        filter_dict = {}
        
        # Add author filter
        if intent.author:
            filter_dict["author"] = intent.author
            
        # Add tags filter
        if intent.tags:
            # Normalize tags
            normalized_tags = []
            for tag in intent.tags:
                # Handle variations like "LLM" vs "LLMs"
                if tag.lower() in ["llm", "llms"]:
                    normalized_tags.append("LLM")
                else:
                    normalized_tags.append(tag)
                    
            filter_dict["tags"] = {"$in": normalized_tags}
            
        # Add date filters based on granularity
        if self.date_granularity == DateGranularity.FULL_DATE:
            if intent.year:
                start_date = f"{intent.year}-01-01"
                end_date = f"{intent.year + 1}-01-01"
                
                if intent.month:
                    start_date = f"{intent.year}-{intent.month:02d}-01"
                    if intent.month == 12:
                        end_date = f"{intent.year + 1}-01-01"
                    else:
                        end_date = f"{intent.year}-{intent.month + 1:02d}-01"
                        
                filter_dict["published_date"] = {
                    "$gte": start_date,
                    "$lt": end_date
                }
                
        else:  # YEAR_MONTH_DAY format
            if intent.year:
                filter_dict["published_year"] = {"$eq": intent.year}
                
            if intent.month:
                filter_dict["published_month"] = {"$eq": intent.month}
                
            if intent.day:
                filter_dict["published_day"] = {"$eq": intent.day}
                
        return filter_dict
    
    def process_query(self, natural_language_query: str) -> Dict[str, Any]:
        """Main method to process natural language query using LLM"""
        
        # Extract intent using LLM
        intent = self.extract_intent(natural_language_query)
        
        # Build Pinecone filter
        pinecone_filter = self.build_pinecone_filter(intent)
        
        return pinecone_filter


class PineconeIndexer:
    """Helper class to index the dataset into Pinecone"""
    
    def __init__(self, api_key: str, index_name: str = "bridged-demo"):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.dimension = None
        
    def create_index(self, dimension: int = 1024):
        """Create Pinecone index if it doesn't exist"""
        existing_indexes = self.pc.list_indexes().names()
        
        if self.index_name in existing_indexes:
            # Get existing index stats to determine dimension
            index = self.pc.Index(self.index_name)
            stats = index.describe_index_stats()
            self.dimension = stats.get('dimension', dimension)
            print(f"Index '{self.index_name}' already exists with dimension {self.dimension}")
        else:
            self.dimension = dimension
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            print(f"Created new index '{self.index_name}' with dimension {self.dimension}")
            
    def prepare_data(self, df: pd.DataFrame, date_granularity: DateGranularity) -> List[Dict]:
        """Prepare data for Pinecone indexing"""
        vectors = []
        
        # Use the actual dimension of the index
        embedding_dim = self.dimension or 1024
        
        for idx, row in df.iterrows():
            # Generate a simple embedding (in production, use real embeddings)
            embedding = np.random.rand(embedding_dim).tolist()
            
            metadata = {
                "author": row.get('author', 'Unknown'),
                "tags": row['tags'] if isinstance(row.get('tags'), list) else [row['tags']] if pd.notna(row.get('tags')) else [],
                "content": row.get('content', ''),
                "title": row.get('title', ''),
                "page_url": row.get('page_url', '')
            }
            
            # Add date metadata based on granularity
            if date_granularity == DateGranularity.FULL_DATE:
                metadata["published_date"] = row['published_date']
            else:
                date_obj = pd.to_datetime(row['published_date'])
                metadata["published_year"] = date_obj.year
                metadata["published_month"] = date_obj.month
                metadata["published_day"] = date_obj.day
                
            vectors.append({
                "id": f"doc_{idx}",
                "values": embedding,
                "metadata": metadata
            })
            
        return vectors
    
    def index_data(self, vectors: List[Dict]):
        """Index vectors into Pinecone"""
        index = self.pc.Index(self.index_name)
        
        # Index in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            index.upsert(vectors=batch)


# FastAPI app
app = FastAPI(title="Natural Language to Pinecone Query Agent")

# Initialize agent (set your API key)
agent = None

@app.on_event("startup")
async def startup_event():
    global agent
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    agent = PineconeQueryAgent(
        openai_api_key=openai_api_key,
        date_granularity=DateGranularity.YEAR_MONTH_DAY
    )

@app.post("/query", response_model=NLQueryResponse)
async def convert_query(request: NLQueryRequest):
    """Convert natural language query to Pinecone filter"""
    if not agent:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        # Set date granularity based on request
        agent.date_granularity = DateGranularity(request.date_format)
        
        # Process query
        filter_dict = agent.process_query(request.query)
        
        return NLQueryResponse(filter=filter_dict)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    # Example usage
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        print("Please set OPENAI_API_KEY in your .env file")
        exit(1)
    
    agent = PineconeQueryAgent(
        openai_api_key=openai_api_key,
        date_granularity=DateGranularity.YEAR_MONTH_DAY
    )
    
    # Test queries
    test_queries = [
        "Show me articles by Alice Zhang from last year about machine learning.",
        "Find posts tagged with 'LLMs' published in June, 2023.",
        "Anything by John Doe on vector search?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            result = agent.process_query(query)
            print(f"Filter: {json.dumps(result, indent=2)}")
        except Exception as e:
            print(f"Error: {e}")