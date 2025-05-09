from fastapi import FastAPI, HTTPException, Form
from typing import List, Dict, Any, Optional
from vector_store import IOEGPTVectorStore
import json
import os
import logging
from dotenv import load_dotenv
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(
    title="IOEGPT Vector Store API",
    description="API for managing and searching vector collections",
    version="1.0.0"
)
vector_store = IOEGPTVectorStore()

# Get admin password from environment variable
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
if not ADMIN_PASSWORD:
    raise ValueError("ADMIN_PASSWORD environment variable is not set")

# Define request models for better documentation
class UpdateRequest(BaseModel):
    collection_id: str
    password: str

class SearchRequest(BaseModel):
    collection_id: str
    query: str
    k: int = 5

class DeleteRequest(BaseModel):
    collection_id: str
    ids: str
    password: str

class DeleteCollectionRequest(BaseModel):
    collection_id: str
    password: str

@app.post("/update")
async def update_documents(
    collection_id: str = Form(..., description="ID of the collection to update"),
    password: str = Form(..., description="Admin password for authentication")
):
    """Update documents for specified collection"""
    try:
        # Verify password
        if password != ADMIN_PASSWORD:
            logger.warning("Invalid admin password attempt")
            raise HTTPException(
                status_code=401,
                detail="Invalid admin password"
            )

        if collection_id not in vector_store.collections:
            logger.error(f"Invalid collection ID: {collection_id}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid collection ID. Available collections: {list(vector_store.collections.keys())}"
            )

        logger.info(f"Updating documents for collection: {collection_id}")
        
        # Load and process documents
        documents = vector_store.load_documents(collection_id)
        vector_store.add_documents(collection_id, documents)
        
        logger.info(f"Successfully processed {len(documents)} documents")
        return {"message": f"Successfully processed {len(documents)} documents"}
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_documents(
    collection_id: str = Form(..., description="ID of the collection to search in"),
    query: str = Form(..., description="Search query"),
    k: int = Form(5, description="Number of results to return", ge=1, le=20)
):
    """Search for similar documents in specified collection"""
    try:
        if collection_id not in vector_store.collections:
            logger.error(f"Invalid collection ID: {collection_id}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid collection ID. Available collections: {list(vector_store.collections.keys())}"
            )

        logger.info(f"Searching in collection {collection_id} for query: {query}")
        results = vector_store.search(collection_id, query, k)
        return {"results": results}
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/delete")
async def delete_documents(
    collection_id: str = Form(..., description="ID of the collection to delete from"),
    ids: str = Form(None, description="Comma-separated list of document IDs to delete (optional, if not provided, entire collection will be deleted)"),
    password: str = Form(..., description="Admin password for authentication")
):
    """Delete documents from specified collection by their IDs or delete entire collection if no IDs provided"""
    try:
        # Verify password
        if password != ADMIN_PASSWORD:
            logger.warning("Invalid admin password attempt")
            raise HTTPException(
                status_code=401,
                detail="Invalid admin password"
            )

        if collection_id not in vector_store.collections:
            logger.error(f"Invalid collection ID: {collection_id}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid collection ID. Available collections: {list(vector_store.collections.keys())}"
            )

        if ids:
            # Delete specific documents
            id_list = [id.strip() for id in ids.split(",")]
            logger.info(f"Deleting documents from collection {collection_id} with IDs: {id_list}")
            vector_store.delete_documents(collection_id, id_list)
            return {"message": f"Successfully deleted {len(id_list)} documents"}
        else:
            # Delete entire collection
            logger.info(f"Deleting entire collection: {collection_id}")
            vector_store.delete_collection(collection_id)
            return {"message": f"Successfully deleted collection: {collection_id}"}
    except Exception as e:
        logger.error(f"Error deleting documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections")
async def list_collections():
    """List all available collections"""
    return {"collections": list(vector_store.collections.keys())}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    logger.info("Health check requested")
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting IOEGPT Vector Store API server")
    uvicorn.run(app, host="0.0.0.0", port=8000) 