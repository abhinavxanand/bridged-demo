import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from main import PineconeIndexer, DateGranularity

# Load environment variables
load_dotenv()

def create_sample_dataset():
    """Create a sample dataset matching the expected format"""
    
    # Sample data
    data = [
        {
            "title": "Introduction to Machine Learning",
            "author": "Alice Zhang",
            "published_date": "2024-03-15",
            "tags": ["machine learning", "AI", "tutorial"],
            "content": "A comprehensive guide to machine learning basics..."
        },
        {
            "title": "Deep Learning with Neural Networks",
            "author": "Alice Zhang",
            "published_date": "2024-07-22",
            "tags": ["machine learning", "deep learning", "neural networks"],
            "content": "Exploring the fundamentals of neural networks..."
        },
        {
            "title": "Understanding Large Language Models",
            "author": "Bob Smith",
            "published_date": "2023-06-10",
            "tags": ["LLM", "NLP", "AI"],
            "content": "An overview of how LLMs work and their applications..."
        },
        {
            "title": "Advanced LLM Techniques",
            "author": "Carol Johnson",
            "published_date": "2023-06-25",
            "tags": ["LLM", "fine-tuning", "prompting"],
            "content": "Exploring advanced techniques for working with LLMs..."
        },
        {
            "title": "Vector Search Fundamentals",
            "author": "John Doe",
            "published_date": "2024-01-10",
            "tags": ["vector search", "embeddings", "similarity"],
            "content": "Understanding the basics of vector search..."
        },
        {
            "title": "Building Scalable Vector Databases",
            "author": "John Doe",
            "published_date": "2024-02-15",
            "tags": ["vector search", "databases", "scalability"],
            "content": "How to build and scale vector databases..."
        },
        {
            "title": "RAG Systems Explained",
            "author": "Alice Zhang",
            "published_date": "2024-08-05",
            "tags": ["RAG", "retrieval", "machine learning"],
            "content": "Retrieval Augmented Generation systems..."
        },
        {
            "title": "Prompt Engineering Best Practices",
            "author": "David Lee",
            "published_date": "2023-11-20",
            "tags": ["prompting", "LLM", "best practices"],
            "content": "How to write effective prompts for LLMs..."
        },
        {
            "title": "Transformer Architecture Deep Dive",
            "author": "Emma Wilson",
            "published_date": "2023-09-14",
            "tags": ["transformers", "architecture", "deep learning"],
            "content": "Understanding the transformer architecture..."
        },
        {
            "title": "Fine-tuning LLMs for Specific Tasks",
            "author": "Frank Chen",
            "published_date": "2024-04-20",
            "tags": ["LLM", "fine-tuning", "machine learning"],
            "content": "Guide to fine-tuning language models..."
        }
    ]
    
    return pd.DataFrame(data)

def main():
    """Main function to index data into Pinecone"""
    
    # Get API keys from environment
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "bridged-demo")
    
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY environment variable not set")
    
    print("Initializing Pinecone indexer...")
    indexer = PineconeIndexer(api_key=pinecone_api_key, index_name=index_name)
    
    print("Creating index...")
    indexer.create_index(dimension=1024)  # Use 1024 to match your existing index
    
    print("Loading dataset...")
    # Load the provided dataset or create sample data
    try:
        # Try to load the actual dataset if provided
        df = pd.read_csv("dataset.csv")
        print(f"Loaded {len(df)} records from dataset.csv")
        
        # Rename columns to match expected format
        column_mapping = {
            'publishedDate': 'published_date',
            'pageURL': 'page_url'
        }
        df = df.rename(columns=column_mapping)
        
        # Convert tags from string to list if needed
        if 'tags' in df.columns and df['tags'].dtype == 'object':
            df['tags'] = df['tags'].apply(lambda x: [tag.strip() for tag in x.split(',')] if pd.notna(x) and x else [])
            
    except FileNotFoundError:
        print("No dataset.csv found, using sample data...")
        df = create_sample_dataset()
        # Save sample data for reference
        df.to_csv("sample_dataset.csv", index=False)
        print(f"Created sample dataset with {len(df)} records")
    
    print("Preparing data for indexing...")
    vectors = indexer.prepare_data(df, DateGranularity.YEAR_MONTH_DAY)
    
    print(f"Indexing {len(vectors)} vectors into Pinecone...")
    indexer.index_data(vectors)
    
    print("Indexing complete!")
    
    # Print some statistics
    print("\nDataset Statistics:")
    print(f"Total documents: {len(df)}")
    if 'author' in df.columns:
        print(f"Authors: {df['author'].unique().tolist()}")
    if 'published_date' in df.columns:
        print(f"Date range: {df['published_date'].min()} to {df['published_date'].max()}")
    
    # Count tags
    if 'tags' in df.columns:
        all_tags = []
        for tags in df['tags']:
            if isinstance(tags, list):
                all_tags.extend(tags)
            elif isinstance(tags, str) and tags:
                all_tags.extend([t.strip() for t in tags.split(',')])
        
        from collections import Counter
        tag_counts = Counter(all_tags)
        print(f"Top tags: {tag_counts.most_common(5)}")

if __name__ == "__main__":
    main()