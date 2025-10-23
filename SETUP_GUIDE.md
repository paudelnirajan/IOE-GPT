# Complete Setup Guide for CBOT Application

## Overview
This is a FastAPI application that uses:
- **Milvus** - Vector database for storing question embeddings
- **Redis** - For conversation state/checkpointing
- **LangChain/LangGraph** - AI agent workflows
- **Groq API** - LLM provider

---

## Prerequisites
- Python 3.8+
- Docker & Docker Compose
- Groq API Key (already configured in .env)

---

## Setup Steps

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Database Services (Milvus + Redis)

Start all services in detached mode:

```bash
docker-compose up -d
```

Check if services are running:

```bash
docker-compose ps
```

You should see 4 containers running:
- `milvus-standalone` (port 19530, 9091)
- `milvus-etcd`
- `milvus-minio` (port 9000, 9001)
- `redis-checkpoint` (port 6379)

### 3. Verify Database Connections

**Check Milvus:**
```bash
curl http://localhost:9091/healthz
```
Should return: `OK`

**Check Redis:**
```bash
docker exec cbot-redis redis-cli ping
```
Should return: `PONG`

### 4. Start the FastAPI Server

```bash
python server.py
```

Or with uvicorn directly:
```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

The server will start on: **http://localhost:8000**

---

## API Endpoints

### 1. **Update Vector Store**
Upload questions to Milvus vector database:

```bash
curl -X POST "http://localhost:8000/update-vector-store" \
  -F "collection_name=c_past_questions" \
  -F "file_path=formatted_data/c_question.json"
```

### 2. **Chat/Query**
Send a question to the AI agent:

```bash
curl -X POST "http://localhost:8000/response" \
  -F "query=What is a pointer in C?" \
  -F "sender_id=user123" \
  -F "metadata=optional_metadata"
```

---

## Database Management

### Stop All Services
```bash
docker-compose down
```

### Stop and Remove All Data
```bash
docker-compose down -v
```

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f milvus-standalone
docker-compose logs -f redis
```

### Restart Services
```bash
docker-compose restart
```

---

## Important Files

- **`server.py`** - Main FastAPI application
- **`docker-compose.yml`** - Database services configuration
- **`.env`** - Environment variables (API keys, ports, etc.)
- **`requirements.txt`** - Python dependencies
- **`vector_store.py`** - Milvus vector store manager
- **`graph_building.py`** - LangGraph workflow definition

---

## Troubleshooting

### Port Already in Use
If you get port conflicts:

```bash
# Check what's using the port
lsof -i :19530  # Milvus
lsof -i :6379   # Redis
lsof -i :8000   # FastAPI

# Kill the process
kill -9 <PID>
```

### Milvus Connection Error
```bash
# Restart Milvus
docker-compose restart standalone

# Check logs
docker-compose logs standalone
```

### Redis Connection Error
```bash
# Restart Redis
docker-compose restart redis

# Test connection
docker exec redis-checkpoint redis-cli ping
```

### Python Dependencies Issues
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Mac/Linux
# venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Environment Variables

Your `.env` file contains:
- `GROQ_API_KEY` - Already configured   
- `LANGSMITH_API_KEY` - For LangSmith tracing (optional)
- `MILVUS_HOST` & `MILVUS_PORT` - Database connection
- `C_PAST_QUESTIONS_COLLECTION` - Collection name for C programming questions

---

## Accessing Services

- **FastAPI Docs**: http://localhost:8000/docs
- **Milvus Web UI**: http://localhost:9091
- **MinIO Console**: http://localhost:9001 (user: minioadmin, pass: minioadmin)

---

## Quick Start (All Commands)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start databases
docker-compose up -d

# 3. Wait 30 seconds for services to be ready
sleep 30

# 4. Verify services
curl http://localhost:9091/healthz
docker exec redis-checkpoint redis-cli ping

# 5. Start application
python server.py
```

# 6. Update database
    Remember the collection name should be "ioe_c_past_questions" because that is set up as default.
---

## Shutdown

```bash
# Stop application (Ctrl+C in terminal running server.py)

# Stop databases
docker-compose down
```
