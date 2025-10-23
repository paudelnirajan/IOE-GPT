import json
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from pymilvus import utility
from core.db_manager import db_manager

class IoePastQuestionsVectorStore:
    def __init__(self, host="127.0.0.1", port="19530"):
        self.connection_args = {
            "host": host,
            "port": port
        }
        # Use the embeddings from the global database manager
        self.embeddings = db_manager.embeddings
    
    def load_json_data(self, file_path="formatted_data/c_question.json"):
        """Load data from a JSON file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    
    def create_documents_from_json(self, json_data):
        """Create Document objects from JSON data"""
        docs = []
        for item in json_data:
            metadata = {k: v for k, v in item.items() if (k != 'question' and k != 'tags')}
            doc = Document(
                page_content=item['question'],
                metadata=metadata
            )
            docs.append(doc)
        return docs
    
    def get_vector_store(self, collection_name):
        """Get an existing vector store for a collection"""
        return db_manager.get_vector_store(collection_name)
    
    def update_vector_store(self, collection_name, file_path="formatted_data/c_question.json"):
        """Update a collection with documents from a JSON file"""

        # TODO add this validation check later
        # if not utility.has_collection(collection_name):
        #     raise ValueError(f"Collection '{collection_name}' does not exist in database")

        json_data = self.load_json_data(file_path)
        docs = self.create_documents_from_json(json_data)

        return Milvus.from_documents(
            docs,
            embedding=self.embeddings,
            connection_args=self.connection_args,
            collection_name=collection_name
        )
