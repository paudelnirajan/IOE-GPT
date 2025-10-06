from Model.models import llm
from langchain_core.prompts import ChatPromptTemplate
from Schema.schema import QuestionSearch
from Prompts.agent_prompt import QUESTION_PROMPT
from typing_extensions import Dict, List
from core.db_manager import db_manager
from langchain_core.documents import Document

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
        self.question_processor = QuestionProcessor()

    def get_filtered_questions(self, question: str, k: int = 3) -> List:
        try:
            vector_store = db_manager.get_vector_store(
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
                    
                # Filter out the vector field from each result
                filtered_results = []
                for result in search_results:
                    # Create a new metadata dictionary without the vector field
                    filtered_metadata = {k: v for k, v in result.metadata.items() if k != 'vector'}
                    # print(f"[INFO] Filtered metadata\n----\n{filtered_metadata}\n----\n")
                    
                    # Create a new Document with filtered metadata
                    filtered_doc = Document(
                        page_content=result.page_content,
                        metadata=filtered_metadata
                    )
                    filtered_results.append(filtered_doc)
                    
                # Create a response dictionary with both results and filter info
                response = {
                    "results": filtered_results,
                }
                return response
            else:
                # semantic filtering
                retriever = vector_store.as_retriever(
                    search_kwargs={'k': k},
                )
                print("[INFO] Returning questions with <SEMANTIC> filtering...")
                search_results = retriever.invoke(question)
                
                # Filter out the vector field from each result
                filtered_results = []
                for result in search_results:
                    # Create a new metadata dictionary without the vector field
                    filtered_metadata = {k: v for k, v in result.metadata.items() if k != 'vector'}
                    # print(f"[INFO] Filtered metadata\n----\n{filtered_metadata}\n----\n")
                    
                    # Create a new Document with filtered metadata
                    filtered_doc = Document(
                        page_content=result.page_content,
                        metadata=filtered_metadata
                    )
                    filtered_results.append(filtered_doc)
                    
                response = {
                    "results": filtered_results,
                }
                return response
        except Exception as e:
            print(f"Error getting filtered questions: {str(e)}")
            raise 