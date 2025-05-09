from pymilvus import connections, utility
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Milvus connection details from docker-compose
MILVUS_HOST = "localhost"
MILVUS_PORT = "19531"  # This is the exposed port from docker-compose

def list_collections():
    """List all collections in Milvus database"""
    try:
        # Connect to Milvus
        logger.info(f"Connecting to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")
        connections.connect(
            alias="default",
            host=MILVUS_HOST,
            port=MILVUS_PORT
        )
        
        # Get list of collections
        collections = utility.list_collections()
        
        # Print collections
        if collections:
            logger.info("Available collections:")
            for collection in collections:
                logger.info(f"- {collection}")
        else:
            logger.info("No collections found in the database")
            
    except Exception as e:
        logger.error(f"Error connecting to Milvus: {str(e)}")
    finally:
        # Disconnect from Milvus
        connections.disconnect(alias="default")

if __name__ == "__main__":
    list_collections()