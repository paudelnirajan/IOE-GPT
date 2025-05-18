from Model.models import llm
from vector_store import IoePastQuestionsVectorStore
from langchain_core.prompts import ChatPromptTemplate
from Schema.schema import QuestionSearch
from Prompts.agent_prompt import QUESTION_PROMPT
from typing_extensions import Dict, List
from langchain_core.tools import tool

class QuestionProcessor:
    """Handles the processing of natural language queries into structured format."""
    
    def __init__(self):
        self.structured_llm = llm.with_structured_output(QuestionSearch)
        self.structured_chain = ChatPromptTemplate.from_messages([
            ("system", QUESTION_PROMPT),
            ("human", "{question}"),
        ]) | self.structured_llm

    def create_dynamic_filter(self, query_result: QuestionSearch) -> tuple[str, bool]:
        """
        Create a Milvus filter expression from the query result.
        
        Returns:
            Tuple containing:
            - String containing the Milvus filter expression
            - Boolean indicating if metadata_only is True
        """
        # Extract metadata_only before processing
        metadata_only = query_result.metadata_only
        
        # Get the model dump and remove metadata_only
        filter_dict = query_result.model_dump()
        filter_dict.pop('metadata_only', False)
        
        filter_parts = []
        
        for field_name, value in filter_dict.items():
            if value is not None:
                # this field is redundant if we are building RAG for specific subject separately
                if field_name == 'subject':
                    pass
                elif field_name in ['year_ad', 'year_bs'] and isinstance(value, list):
                    if value:  # Only if the list is not empty
                        years_str = ', '.join(map(str, value))
                        filter_parts.append(f"{field_name} in [{years_str}]")
                elif isinstance(value, list):
                    if value:  # Only add if list is not empty
                        values_str = ', '.join(f"'{v}'" for v in value)
                        filter_parts.append(f"{field_name} in [{values_str}]")
                else:
                    if isinstance(value, str):
                        filter_parts.append(f"{field_name} == '{value}'")
                    else:
                        filter_parts.append(f"{field_name} == {value}")
        
        return (" and ".join(filter_parts) if filter_parts else ""), metadata_only

    def process_query(self, question: str) -> QuestionSearch:
        """
        Process the natural language query into a structured format.
        
        Args:
            question: The natural language question from the user
            
        Returns:
            Structured QuestionSearch object
        """
        return self.structured_chain.invoke({"question": question})


class VectorStoreManager:
    """Manages vector store operations and question retrieval."""
    
    def __init__(self, collection_name: str = "ioe_c_past_questions"):
        self.collection_name = collection_name
        self.vector_store_manager = IoePastQuestionsVectorStore()
        self.question_processor = QuestionProcessor()

    def get_filtered_questions(self, question: str, k: int = 3) -> List:
        try:
            vector_store = self.vector_store_manager.get_vector_store(
                collection_name=self.collection_name
            )
            
            # Process the query
            query_result = self.question_processor.process_query(question)
            filter_expression, metadata_only = self.question_processor.create_dynamic_filter(query_result)

            print(f"[INFO] Filter dictionary: {filter_expression}")  # Debug print
            print(f"[INFO] metadata_only field is {metadata_only}")
            if metadata_only == True:
                # Use search_by_metadata instead of as_retriever
                print("[INFO] Returning questions based on <METADATA> filters...")
                search_results = vector_store.search_by_metadata(
                        expr=filter_expression,
                        limit=k
                    )
                    
                # Create a response dictionary with both results and filter info
                response = {
                    "results": [result.page_content for result in search_results],
                    "filter_info": {
                        "filter_expression": filter_expression,
                        "metadata_only": metadata_only, # XXX may be this is not required
                    }
                }
                return response
            else:
                # semantic filtering
                retriever = vector_store.as_retriever(
                    search_kwargs={'k': k},
                )
                print("[INFO] Returning questions with <SEMANTIC> filtering...")
                search_results = retriever.invoke(question)
                # XXX may be, if we also pass the metadata and instruct llm to also describe about the question like-> this question was asked in year ... and ...
                response = {
                    "results": [result.page_content for result in search_results]
                }
                return response
        except Exception as e:
            print(f"Error getting filtered questions: {str(e)}")
            raise
    
@tool
def get_past_questions(question: str, k: int = 3) -> List:
    """
    Public function to get filtered questions based on the user's query.
    
    Args:
        question: Natural language question from user
        k: Maximum number of results to retrieve
    
    Returns:
        List of relevant documents that match the filter criteria
    """
    manager = VectorStoreManager()
    return manager.get_filtered_questions(question, k) 









# # this code below is for direct testing the tool 
# # handles milvus connection on it's own, change the question and check filter dict generated and metadata_only field.

# from vector_store import IoePastQuestionsVectorStore
# from langchain_core.prompts import ChatPromptTemplate
# from Schema.schema import QuestionSearch
# from Prompts.agent_prompt import QUESTION_PROMPT
# from typing_extensions import Dict, List
# from pymilvus import connections, utility
# import sys
# import json
# from Model.models import llm

# class QuestionProcessor:
#     """Handles the processing of natural language queries into structured format."""
    
#     def __init__(self):
#         self.structured_llm = llm.with_structured_output(QuestionSearch)
#         self.structured_chain = ChatPromptTemplate.from_messages([
#             ("system", QUESTION_PROMPT),
#             ("human", "{question}"),
#         ]) | self.structured_llm

#     def create_dynamic_filter(self, query_result: QuestionSearch) -> tuple[str, bool]:
#         """
#         Create a Milvus filter expression from the query result.
        
#         Returns:
#             Tuple containing:
#             - String containing the Milvus filter expression
#             - Boolean indicating if metadata_only is True
#         """
#         # Extract metadata_only before processing
#         metadata_only = query_result.metadata_only
        
#         # Get the model dump and remove metadata_only
#         filter_dict = query_result.model_dump()
#         filter_dict.pop('metadata_only', False)
        
#         filter_parts = []
        
#         for field_name, value in filter_dict.items():
#             if value is not None:
#                 # this field is redundant if we are building RAG for specific subject separately
#                 if field_name == 'subject':
#                     pass
#                 elif field_name in ['year_ad', 'year_bs'] and isinstance(value, list):
#                     if value:  # Only if the list is not empty
#                         years_str = ', '.join(map(str, value))
#                         filter_parts.append(f"{field_name} in [{years_str}]")
#                 elif isinstance(value, list):
#                     if value:  # Only add if list is not empty
#                         values_str = ', '.join(f"'{v}'" for v in value)
#                         filter_parts.append(f"{field_name} in [{values_str}]")
#                 else:
#                     if isinstance(value, str):
#                         filter_parts.append(f"{field_name} == '{value}'")
#                     else:
#                         filter_parts.append(f"{field_name} == {value}")
        
#         return (" and ".join(filter_parts) if filter_parts else ""), metadata_only

#     def process_query(self, question: str) -> QuestionSearch:
#         """
#         Process the natural language query into a structured format.
        
#         Args:
#             question: The natural language question from the user
            
#         Returns:
#             Structured QuestionSearch object
#         """
#         return self.structured_chain.invoke({"question": question})


# class VectorStoreManager:
#     """Manages vector store operations and question retrieval."""
    
#     def __init__(self, collection_name: str = "ioe_c_past_questions"):
#         self.collection_name = collection_name
#         self._connect_to_milvus()
#         self.vector_store_manager = IoePastQuestionsVectorStore()
#         self.question_processor = QuestionProcessor()

#     def _connect_to_milvus(self):
#         """Establish connection to Milvus server."""
#         try:
#             # Check if connection already exists
#             if not connections.has_connection("default"):
#                 connections.connect(
#                     alias="default",
#                     host="127.0.0.1",
#                     port="19530"
#                 )
#                 print("Successfully connected to Milvus server")
            
#             # Verify collection exists
#             if not utility.has_collection(self.collection_name):
#                 print(f"Warning: Collection '{self.collection_name}' does not exist")
#                 print("Please create the collection first using the update-vector-store endpoint")
#                 sys.exit(1)
                
#         except Exception as e:
#             print(f"Error connecting to Milvus: {str(e)}")
#             print("Please ensure Milvus server is running on localhost:19530")
#             sys.exit(1)

#     def __del__(self):
#         """Cleanup: disconnect from Milvus when the object is destroyed"""
#         try:
#             if connections.has_connection("default"):
#                 connections.disconnect("default")
#                 print("Disconnected from Milvus server")
#         except:
#             pass

#     def get_filtered_questions(self, question: str, k: int = 3) -> List:
#         try:
#             vector_store = self.vector_store_manager.get_vector_store(
#                 collection_name=self.collection_name
#             )
            
#             # Process the query
#             query_result = self.question_processor.process_query(question)
#             filter_expression, metadata_only = self.question_processor.create_dynamic_filter(query_result)

#             print(f"[INFO] Filter dictionary: {filter_expression}")  # Debug print
#             print(f"[INFO] metadata_only field is {metadata_only}")
#             if metadata_only == True:
#                 # Use search_by_metadata instead of as_retriever
#                 print("[INFO] Returning questions based on <METADATA> filters...")
#                 search_results = vector_store.search_by_metadata(
#                         expr=filter_expression,
#                         limit=k
#                     )
                    
#                 # Create a response dictionary with both results and filter info
#                 response = {
#                     "results": [result.page_content for result in search_results],
#                     "filter_info": {
#                         "filter_expression": filter_expression,
#                         "metadata_only": metadata_only, # XXX may be this is not required
#                     }
#                 }
#                 return response

#             else:
#                 # semantic filtering
#                 retriever = vector_store.as_retriever(
#                     search_kwargs={'k': k},
#                 )
#                 print("[INFO] Returning questions with <SEMANTIC> filtering...")
#                 search_results = retriever.invoke(question)
#                 # XXX may be, if we also pass the metadata and instruct llm to also describe about the question like-> this question was asked in year ... and ...
#                 response = {
#                     # "results": [result.page_content for result in search_results]
#                     "results" : search_results
#                 }
#                 return response
#         except Exception as e:
#             print(f"Error getting filtered questions: {str(e)}")
#             raise

# # Public API
# def get_filtered_questions(question: str, k: int = 3) -> List:
#     """
#     Public function to get filtered questions based on the user's query.
    
#     Args:
#         question: Natural language question from user
#         k: Maximum number of results to retrieve
    
#     Returns:
#         List of relevant documents that match the filter criteria
#     """
#     manager = VectorStoreManager()
#     return manager.get_filtered_questions(question, k)

# if __name__ == "__main__":
#     # Example usage
#     try:
#         results = get_filtered_questions("list some programming questions from topic function", 2)
        
#         if not results:
#             print("No results found matching your query.")
#         else:
#             print("\nFound the following questions:")
#             print(results)
#             # for i, result in enumerate(results, 1):
#             #     print(f"\n--- Result {i} ---")
#             #     print("Question:", result.page_content)
#                 # metadata_without_vector = {k: v for k, v in result.metadata.items() if k != 'vector'}
#                 # print("Metadata:", metadata_without_vector)
#     except Exception as e:
#         print(f"Error: {str(e)}")
