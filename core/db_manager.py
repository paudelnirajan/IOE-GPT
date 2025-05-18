from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from pymilvus import utility, connections

class DatabaseManager:
    """Manages database connections and provides access to vector stores."""
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern to ensure only one connection instance exists."""
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, host="127.0.0.1", port="19530"):
        if self._initialized:
            return
            
        self.connection_args = {
            "host": host,
            "port": port
        }
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        # Initialize connection to Milvus
        connections.connect(**self.connection_args)
        self._initialized = True
        print("[INFO] Database connection initialized")
    
    def get_vector_store(self, collection_name):
        """Get an existing vector store for a collection"""
        if not utility.has_collection(collection_name):
            raise ValueError(f"Collection name '{collection_name}' does not exist in database")
        return Milvus(
            embedding_function=self.embeddings,
            connection_args=self.connection_args,
            collection_name=collection_name
        )

# Global instance that can be imported and used throughout the application
db_manager = DatabaseManager() 