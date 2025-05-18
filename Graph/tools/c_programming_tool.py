from typing_extensions import List
from langchain_core.tools import tool
from Graph.utils.question_utils import VectorStoreManager

@tool
def get_past_questions(question: str, k: int = 5) -> List:
    """
    Tool to get filtered past questions based on the user's query.
    
    Args:
        question: Natural language question from user
        k: Maximum number of results to retrieve
    
    Returns:
        List of relevant documents that match the filter criteria
    """
    # Ensure k is an integer
    try:
        if isinstance(k, str):
            k = int(k)
    except ValueError:
        k = 5  # Default if conversion fails
        
    manager = VectorStoreManager(collection_name="ioe_c_past_questions")
    return manager.get_filtered_questions(question, k) 