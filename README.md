# IOE-GPT: AI-Powered Question Assistant

An intelligent question retrieval system for IOE Computer Programming past questions, built with LangChain, LangGraph, Milvus vector database, and Groq LLM.

## Features

- 🤖 AI-powered question search using natural language
- 🔍 Semantic and metadata-based filtering
- 📚 Vector database storage with Milvus
- 🔄 Conversation state management with Redis
- ⚡ Fast API endpoints with FastAPI
- 🎯 Structured query processing with LLM

## Quick Start

```bash
# 1. Clone and setup
git clone <your-repo-url> && cd IOE-GPT
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env  # Then add your GROQ_API_KEY

# 3. Start services
brew services start redis  # or: redis-server
docker-compose up -d milvus-standalone

# 4. Run server
uvicorn server:app --reload
```

## Prerequisites

- **Python 3.12+**
- **Docker** (for Milvus and Redis)
- **Groq API Key** (get from [console.groq.com](https://console.groq.com))

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd IOE-GPT
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```env
# Groq API Configuration
GROQ_API_KEY=your_groq_api_key_here

# Milvus Configuration
MILVUS_HOST=127.0.0.1
MILVUS_PORT=19530

# Redis Configuration
REDIS_URI=redis://localhost:6379

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### 5. Start Required Services

#### Start Milvus (Vector Database)

```bash
# Using Docker Compose (recommended)
docker-compose up -d milvus-standalone

# Or using standalone Docker
docker run -d --name milvus_standalone \
  -p 19530:19530 -p 9091:9091 \
  -v $(pwd)/volumes/milvus:/var/lib/milvus \
  milvusdb/milvus:latest
```

#### Start Redis (Session Storage)

```bash
# Using Homebrew (macOS)
brew services start redis

# Or using Docker
docker run -d --name redis -p 6379:6379 redis:latest

# Or if you have redis-server installed
redis-server
```

### 6. Prepare Question Data

Create a `formatted_data` directory and add your question data:

```bash
mkdir -p formatted_data
```

Create `formatted_data/c_question.json` with your questions in this format:

```json
[
  {
    "id": "CT401_1a",
    "question": "What is computer programming?",
    "subject": "computer Programming",
    "year_ad": 2023,
    "year_bs": 2080,
    "type": "theory",
    "format": "short",
    "marks": 2,
    "topic": "introduction_c_programming",
    "unit": 1,
    "question_number": "1a",
    "source": "regular",
    "semester": "first"
  }
]
```

### 7. Start the Server

```bash
uvicorn server:app --reload
```

The server will start at `http://127.0.0.1:8000`

## API Endpoints

### Chat with Assistant
- **POST** `/chat`
- Send a message to the AI assistant
- Request body:
```json
{
  "message": "Show me questions from 2023 about arrays",
  "thread_id": "optional-session-id"
}
```

### Upload Questions
- **POST** `/upload`
- Upload a JSON file containing questions to the vector database
- Multipart form data with file field

### Health Check
- **GET** `/health`
- Check if the server and database connections are healthy

## Usage Examples

### 1. Chat with the Assistant

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Show me programming questions from 2023",
    "thread_id": "user123"
  }'
```

### 2. Upload Questions

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@formatted_data/c_question.json"
```

### 3. Check API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation (Swagger UI)

## Project Structure

```
IOE-GPT/
├── Graph/                      # LangGraph agent components
│   ├── tools/                  # Tool definitions
│   ├── routes/                 # Routing logic
│   └── utils/                  # Utility functions
├── Model/                      # LLM model configurations
├── Prompts/                    # System prompts
├── Schema/                     # Pydantic schemas
├── core/                       # Core functionality
│   ├── assistant.py            # Assistant implementation
│   ├── db_manager.py           # Database manager
│   └── state.py                # State definitions
├── formatted_data/             # Question data (gitignored)
├── volumes/                    # Docker volumes (gitignored)
├── server.py                   # FastAPI server
├── graph_building.py           # Graph construction
├── vector_store.py             # Vector store operations
├── requirements.txt            # Python dependencies
└── .env                        # Environment variables
```

## Troubleshooting

### Redis Connection Error

**Error:** `redis.exceptions.ConnectionError: Error 61 connecting to localhost:6379`

**Solution:** Start Redis server:
```bash
brew services start redis  # macOS
# or
redis-server               # Manual start
# or
docker run -d -p 6379:6379 redis  # Docker
```

### Milvus Connection Error

**Error:** `Cannot connect to Milvus server`

**Solution:** Ensure Milvus is running:
```bash
docker ps | grep milvus  # Check if running
docker-compose up -d milvus-standalone  # Start if not running
```

### Missing GROQ_API_KEY

**Error:** `GROQ_API_KEY not found in environment variables`

**Solution:** Add your Groq API key to `.env` file:
```env
GROQ_API_KEY=your_actual_api_key_here
```

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'langchain_groq'`

**Solution:** Reinstall dependencies:
```bash
pip install --upgrade -r requirements.txt
```

### langmem Dependency Conflict

**Note:** The `langmem` package has been temporarily disabled due to version conflicts. The summarization feature uses a passthrough implementation. To re-enable, resolve the dependency conflicts between `langmem` and `langchain-core`.

## Development

### Running Tests

```bash
# Test Milvus connection
python milvus_collections.py

# Test vector store operations
python vector_store.py
```

### Viewing Collections

```python
from pymilvus import connections, utility

connections.connect(host="127.0.0.1", port="19530")
collections = utility.list_collections()
print(collections)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Add your license here]

## Acknowledgments

- Built with [LangChain](https://langchain.com/)
- Powered by [Groq](https://groq.com/)
- Vector storage by [Milvus](https://milvus.io/) 