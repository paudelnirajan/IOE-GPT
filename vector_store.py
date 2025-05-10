from typing import List, Dict, Any
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore, InMemoryVectorStore
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

class IOEGPTVectorStore:
    def __init__(self):
        self.host = os.getenv("MILVUS_HOST", "localhost")
        self.port = os.getenv("MILVUS_PORT", "19530")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2 model
        
        # Load collection configurations from environment
        self.collections = {
            "c_past_questions": {
                "name": os.getenv("C_PAST_QUESTIONS_COLLECTION", "c_past_questions"),
                "file_path": os.getenv("C_PAST_QUESTIONS_FILE", "data/c_questions.json")
            }
            # Add more collections here as needed
        }
        
        logger.info(f"Initializing IOEGPTVectorStore with collections: {list(self.collections.keys())}")
        
        # Connect to Milvus
        self._connect()
        
        # Initialize collections
        self._init_collections()
        
        logger.info("Vector store initialization completed")

    def _init_collections(self):
        """Initialize all collections"""
        self.collection_instances = {}
        for collection_id, config in self.collections.items():
            collection_name = config["name"]
            if not utility.has_collection(collection_name):
                logger.info(f"Creating new collection: {collection_name}")
                self._create_collection(collection_name)
            
            self.collection_instances[collection_id] = Collection(collection_name)
            self._load_collection(collection_id)

    def _connect(self):
        """Connect to Milvus server"""
        try:
            logger.info(f"Connecting to Milvus at {self.host}:{self.port}")
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
                timeout=20
            )
            logger.info("Successfully connected to Milvus")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {str(e)}")
            raise

    def _create_collection(self, collection_name: str):
        """Create collection with schema"""
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
            FieldSchema(name="metadata", dtype=DataType.JSON),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
        ]
        schema = CollectionSchema(fields=fields, description=f"Collection for {collection_name}")
        collection = Collection(name=collection_name, schema=schema)
        
        # Create index
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        collection.create_index(field_name="vector", index_params=index_params)

    def _load_collection(self, collection_id: str):
        """Load collection into memory"""
        try:
            collection_name = self.collections[collection_id]["name"]
            logger.info(f"Loading collection {collection_name} into memory")
            self.collection_instances[collection_id].load()
            logger.info(f"Collection {collection_name} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load collection {collection_name}: {str(e)}")
            raise

    def load_documents(self, collection_id: str) -> List[Document]:
        """Load documents from JSON file for specified collection"""
        file_path = self.collections[collection_id]["file_path"]
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        loader = JSONLoader(
            file_path=file_path,
            jq_schema=".[]",
            text_content=False
        )
        return loader.load()

    def add_documents(self, collection_id: str, documents: List[Document]) -> None:
        """Add documents to specified Milvus collection"""
        # Validate collection ID
        self._validate_collection(collection_id)
        
        # Ensure collection exists
        self._ensure_collection_exists(collection_id)
        
        logger.info(f"Adding {len(documents)} documents to collection {collection_id}")
        ids = []
        vectors = []
        metadatas = []
        texts = []

        for doc in documents:
            # Generate embedding
            vector = self.embeddings.embed_query(doc.page_content)
            
            # Extract metadata from document
            metadata = doc.metadata
            
            ids.append(metadata.get("id", str(len(ids))))
            vectors.append(vector)
            metadatas.append(metadata)
            texts.append(doc.page_content)

        # Insert data into collection
        self.collection_instances[collection_id].insert([ids, vectors, metadatas, texts])
        self.collection_instances[collection_id].flush()
        logger.info(f"Documents successfully added to collection {collection_id}")

    def delete_documents(self, collection_id: str, ids: List[str]) -> None:
        """Delete documents from specified collection by IDs"""
        logger.info(f"Deleting documents with IDs: {ids} from collection {collection_id}")
        expr = f'id in {ids}'
        self.collection_instances[collection_id].delete(expr)
        logger.info(f"Documents successfully deleted from collection {collection_id}")

    def delete_collection(self, collection_id: str) -> None:
        """Delete entire collection"""
        try:
            collection_name = self.collections[collection_id]["name"]
            logger.info(f"Deleting collection: {collection_name}")
            
            # Drop the collection from Milvus
            utility.drop_collection(collection_name)
            
            # Remove from local tracking
            del self.collection_instances[collection_id]
            # Don't delete from self.collections to allow recreation
            # del self.collections[collection_id]
            
            logger.info(f"Collection {collection_name} successfully deleted")
        except Exception as e:
            logger.error(f"Failed to delete collection {collection_name}: {str(e)}")
            raise

    def _validate_collection(self, collection_id: str) -> None:
        """Validate if collection ID exists in configuration"""
        if collection_id not in self.collections:
            logger.error(f"Invalid collection ID: {collection_id}")
            raise ValueError(f"Invalid collection ID")

    def _ensure_collection_exists(self, collection_id: str) -> None:
        """Ensure collection exists in Milvus, create if it doesn't"""
        try:
            collection_name = self.collections[collection_id]["name"]
            if not utility.has_collection(collection_name):
                logger.info(f"Collection {collection_name} does not exist, creating it")
                self._create_collection(collection_name)
                self.collection_instances[collection_id] = Collection(collection_name)
            elif collection_id not in self.collection_instances:
                logger.info(f"Collection {collection_name} exists but not in instances, adding it")
                self.collection_instances[collection_id] = Collection(collection_name)
        except Exception as e:
            logger.error(f"Failed to ensure collection exists: {str(e)}")
            raise

    def search(self, collection_id: str, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search similar documents in specified collection"""
        # Validate collection ID
        self._validate_collection(collection_id)
        
        # Ensure collection exists
        self._ensure_collection_exists(collection_id)
        
        logger.info(f"Searching for query: {query} with k={k} in collection {collection_id}")
        
        # Ensure collection is loaded
        try:
            self._load_collection(collection_id)
        except Exception as e:
            logger.error(f"Failed to load collection before search: {str(e)}")
            raise
        
        # Generate query embedding
        query_vector = self.embeddings.embed_query(query)
        
        # Search parameters
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }
        
        # Perform search
        results = self.collection_instances[collection_id].search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=k,
            output_fields=["metadata", "text"]
        )
        
        # Format results
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "id": hit.id,
                    "score": hit.score,
                    "metadata": hit.entity.get("metadata"),
                    "text": hit.entity.get("text")
                })
        
        logger.info(f"Found {len(formatted_results)} results in collection {collection_id}")
        return formatted_results

    def get_vector_store(self, collection_id: str) -> VectorStore:
        """
        Get a vector store instance for the specified collection.
        
        Args:
            collection_id: ID of the collection to get vector store for
            
        Returns:
            VectorStore: Vector store instance for the collection
        """
        # Validate collection ID
        self._validate_collection(collection_id)
        
        # Ensure collection exists
        self._ensure_collection_exists(collection_id)
        
        logger.info(f"Getting vector store for collection: {collection_id}")
        
        # Get collection instance
        collection = self.collection_instances[collection_id]
        
        # Query all documents from the collection
        results = collection.query(
            expr="",  # Empty expression to get all documents
            output_fields=["metadata", "text"]
        )
        
        # Convert results to Document objects
        docs = []
        for result in results:
            try:
                # Create Document with metadata and text
                doc = Document(
                    page_content=result["text"],
                    metadata=result["metadata"]
                )
                docs.append(doc)
            except Exception as e:
                logger.error(f"Error processing document: {e}")
                continue
        
        # Create a temporary vector store with the documents
        temp_store = InMemoryVectorStore(embedding=self.embeddings)
        temp_store.add_documents(docs)
        
        logger.info(f"Created vector store with {len(docs)} documents")
        return temp_store

    def close(self):
        """Close connection to Milvus"""
        logger.info("Closing connection to Milvus")
        connections.disconnect(alias="default") 