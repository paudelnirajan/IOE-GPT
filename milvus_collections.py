from pymilvus import connections, utility

def test_milvus_connection():
    try:
        # Connect to Milvus
        connections.connect(
            alias="default",
            host="127.0.0.1",
            port="19530"
        )
        print("Successfully connected to Milvus!")
        
        # List all collections
        collections = utility.list_collections()
        print("\nCollections in database:")
        for collection in collections:
            print(f"- {collection}")
            
    except Exception as e:
        print(f"Error connecting to Milvus: {str(e)}")
    finally:
        # Close the connection
        connections.disconnect("default")


def remove_collection(collection_name):
    try:
        # Connect to Milvus
        connections.connect(
            alias="default",
            host="127.0.0.1",
            port="19530"
        )
        
        # Check if collection exists
        if utility.has_collection(collection_name):
            # Drop the collection
            utility.drop_collection(collection_name)
            print(f"Successfully removed collection: {collection_name}")
        else:
            print(f"Collection '{collection_name}' does not exist")
            
    except Exception as e:
        print(f"Error removing collection: {str(e)}")
    finally:
        # Close the connection
        connections.disconnect("default")


if __name__ == "__main__":
    test_milvus_connection()