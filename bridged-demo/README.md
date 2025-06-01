# Natural Language to Pinecone Query Agent

A sophisticated agent that converts natural language queries into valid Pinecone metadata filters using OpenAI's LLM. This tool enables intuitive search queries that are automatically translated into Pinecone's filter format.

## Features

- **LLM-Powered Natural Language Processing**: Uses OpenAI GPT-4 to understand complex queries
- **Flexible Date Handling**: 
  - Supports relative dates ("last year", "this month")
  - Handles both full date format and separate year/month/day fields
- **Intelligent Tag Normalization**: Automatically handles variations (e.g., "LLMs" → "LLM")
- **FastAPI Integration**: Production-ready REST API
- **Docker Support**: Easy containerized deployment
- **Comprehensive Test Suite**: Validates all query types and edge cases

## Architecture

### Core Components

1. **PineconeQueryAgent**: Main agent class
   - Extracts structured intent from natural language using OpenAI LLM
   - Converts intent to Pinecone filter format
   - Supports configurable date granularity

2. **FastAPI Application**: REST API server
   - `/query` - Convert natural language to Pinecone filter
   - `/health` - Health check endpoint
   - Interactive docs at `/docs`

3. **PineconeIndexer**: Data indexing utility
   - Creates and manages Pinecone indexes
   - Handles data preparation and vector upload

## Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key
- Pinecone API key
- Dataset in CSV format (or use sample data)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/bridged-demo.git
cd bridged-demo
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env and add your API keys:
# OPENAI_API_KEY=your-openai-api-key
# PINECONE_API_KEY=your-pinecone-api-key
# PINECONE_INDEX_NAME=bridged-demo
```

### Running the Application

1. **Index your data** (first time only):
```bash
python index_data.py
```

2. **Start the API server**:
```bash
uvicorn main:app --reload
```

3. **Test the API**:
```bash
# Health check
curl http://localhost:8000/health

# Convert a query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show me articles by Alice Zhang from last year about machine learning.",
    "date_format": "year_month_day"
  }'
```

## API Usage

### Query Conversion Endpoint

**POST** `/query`

Request body:
```json
{
  "query": "Your natural language query",
  "date_format": "year_month_day"  // or "full_date"
}
```

Response:
```json
{
  "filter": {
    "author": "Alice Zhang",
    "published_year": {"$eq": 2024},
    "tags": {"$in": ["machine learning"]}
  }
}
```

### Example Queries and Outputs

**Query 1**: "Show me articles by Alice Zhang from last year about machine learning."
```json
{
  "filter": {
    "author": "Alice Zhang",
    "published_year": {"$eq": 2024},
    "tags": {"$in": ["machine learning"]}
  }
}
```

**Query 2**: "Find posts tagged with 'LLMs' published in June, 2023."
```json
{
  "filter": {
    "tags": {"$in": ["LLM"]},
    "published_year": {"$eq": 2023},
    "published_month": {"$eq": 6}
  }
}
```

**Query 3**: "Anything by John Doe on vector search?"
```json
{
  "filter": {
    "author": "John Doe",
    "tags": {"$in": ["vector search"]}
  }
}
```

## Date Format Options

The agent supports two date formats:

### 1. YEAR_MONTH_DAY Format (Default)
Separate fields for year, month, and day:
```json
{
  "published_year": {"$eq": 2023},
  "published_month": {"$eq": 6},
  "published_day": {"$eq": 15}
}
```

### 2. FULL_DATE Format
Single date field with range queries:
```json
{
  "published_date": {
    "$gte": "2023-06-01",
    "$lt": "2023-07-01"
  }
}
```

## Testing

Run the comprehensive test suite:
```bash
python test_agent.py
```

The test suite includes:
- Required test cases from specifications
- Complex query handling
- Date format variations
- Edge cases and error handling

## Docker Deployment

Build and run with Docker:
```bash
# Build image
docker build -t bridged-demo .

# Run container
docker run -p 8000:8000 --env-file .env bridged-demo
```

## Using Poetry (Alternative)

If you prefer Poetry for dependency management:
```bash
poetry install
poetry shell
uvicorn main:app --reload
```

## Integration Example

```python
import requests
from pinecone import Pinecone

# Initialize Pinecone
pc = Pinecone(api_key="your-pinecone-api-key")
index = pc.Index("bridged-demo")

# Convert natural language to filter
response = requests.post(
    "http://localhost:8000/query",
    json={
        "query": "Articles by Alice Zhang about AI from 2024",
        "date_format": "year_month_day"
    }
)

filter_dict = response.json()["filter"]

# Use with Pinecone query
results = index.query(
    vector=query_embedding,  # Your query embedding
    filter=filter_dict,
    top_k=10,
    include_metadata=True
)
```

## Dataset Format

The indexer expects a CSV with the following columns:
- `pageURL`: URL of the article
- `title`: Article title
- `publishedDate`: Publication date
- `author`: Author name
- `tags`: Comma-separated tags

Example:
```csv
pageURL,title,publishedDate,author,tags
https://example.com/ml-intro,Introduction to ML,2024-03-15,Alice Zhang,"machine learning,AI,tutorial"
```

## Project Structure

```
bridged-demo/
├── main.py              # Core agent and FastAPI application
├── index_data.py        # Data indexing script
├── test_agent.py        # Test suite
├── requirements.txt     # Python dependencies
├── pyproject.toml       # Poetry configuration
├── Dockerfile           # Docker configuration
├── .env.example         # Environment template
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## Key Features Explained

### LLM-Based Intent Extraction
The agent uses OpenAI's GPT-4 to understand queries with:
- Context-aware date parsing ("last year" → 2024)
- Intelligent tag normalization
- Flexible author name recognition
- Support for complex, conversational queries

### Metadata Schema
Supports the following metadata fields:
- `author`: String - article author
- `tags`: List - topic/category tags
- `published_date` or `published_year/month/day`: Date fields

### Error Handling
- Graceful API key validation
- Detailed error messages
- Automatic retry logic for API calls

## Performance Considerations

- LLM calls add ~500-1000ms latency
- Consider caching frequently used queries
- Batch processing available for multiple queries
- Async processing supported via FastAPI

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see LICENSE file for details

## Support

For questions or issues:
- Email: rithvik@bridged.media
- Create an issue in the GitHub repository

## Acknowledgments

- Built for Bridged AI hiring challenge
- Uses OpenAI GPT-4 for natural language understanding
- Powered by Pinecone vector database