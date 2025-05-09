# Question Vector Store API

This project provides a vector store solution for storing and retrieving question data using Milvus vector database. It includes a FastAPI server for managing the vector store operations.

## Prerequisites

- Python 3.8+
- Milvus server running (can be started using Docker)
- Docker (optional, for running Milvus)

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in `.env` file:
```
MILVUS_HOST=localhost
MILVUS_PORT=19530
COLLECTION_NAME=question_vectors
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

3. Start Milvus server (if not already running):
```bash
docker run -d --name milvus_standalone -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest
```

4. Start the FastAPI server:
```bash
python server.py
```

## API Endpoints

### Upload Documents
- **POST** `/upload`
- Upload a JSON file containing questions
- File should be in the format shown in the example data

### Search Documents
- **POST** `/search`
- Search for similar documents
- Request body:
```json
{
    "query": "your search query",
    "k": 5  // optional, number of results to return
}
```

### Delete Documents
- **POST** `/delete`
- Delete documents by their IDs
- Request body:
```json
{
    "ids": ["id1", "id2", "id3"]
}
```

### Health Check
- **GET** `/health`
- Check if the server is running

## Example Usage

1. Upload a JSON file:
```bash
curl -X POST -F "file=@path/to/your/questions.json" http://localhost:8000/upload
```

2. Search for similar questions:
```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"query": "What is computer programming?", "k": 5}' \
     http://localhost:8000/search
```

3. Delete specific questions:
```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"ids": ["CT401_1a", "CT401_1b"]}' \
     http://localhost:8000/delete
```

## Project Structure

- `server.py`: FastAPI server implementation
- `vector_store.py`: Milvus vector store manager
- `requirements.txt`: Project dependencies
- `.env`: Environment variables configuration 