from json import tool
from langchain_core.prompts import ChatPromptTemplate
from vector_store import IOEGPTVectorStore
from Schema.schema import QuestionSearch
from Model.models import llm
import json
import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize the vector store with Milvus
vector_store = IOEGPTVectorStore()
logger.debug("Vector store initialized")

@dataclass
class QueryResult:
    """Container for query results and metadata"""
    query_result: QuestionSearch
    filter_dict: Dict[str, Any]
    is_metadata_only: bool

class QuestionRetriever:
    def __init__(self):
        logger.debug("Initializing QuestionRetriever")
        self.system_prompt = """You are an expert at converting user questions about past exam papers into structured JSON queries.
        You have access to a database (JSON file) containing information about past exam questions for subjects like Computer Programming, Mathematics, and Digital Logic from various years and semesters.
        Given a user's question, your goal is to construct a JSON query object that conforms to the `QuestionSearch` schema to retrieve the most relevant question(s) from the database.

        When users mention multiple years, collect them into a list. For example:
        - "questions from 2075, 2076 BS" → year_bs: [2075, 2076]
        - "questions before year 2076" → year_bs: [2075, 2076] (DO NOT provide year before 2075)

        You must identify key information in the user's request, such as:
        - Subject name (e.g., "computer programming")
        - Year (Specify BS or AD, e.g., "2080 BS", "2023 AD")
        - Question type ("theory" or "programming")
        - Question format ("short" or "long")
        - Marks
        - Topic
        - Unit number
        - Question number (e.g., "1a", "5b")
        - Source ("regular" or "back" exam)
        - Semester ("first", "second", etc.)
        - Keywords within the question text itself.

        Map this extracted information accurately to the corresponding fields in the `QuestionSearch` JSON schema.
        - Pay close attention to the required fields and the allowed values for fields with `Literal` types (like `subject`, `type`, `format`, `source`, `semester`).
        - Use `null` or omit optional fields if the information is not provided in the user's query.
        - Do not invent information or assume details not explicitly stated by the user.
        - If the user uses specific terms, acronyms, or numbers, preserve them accurately in the query values.
        """
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{question}"),
        ])
        self.structured_llm = llm.with_structured_output(QuestionSearch)
        self.structured_chain = self.prompt | self.structured_llm
        logger.debug("QuestionRetriever initialization complete")

    def _is_metadata_only_query(self, query_result: QuestionSearch) -> bool:
        metadata_fields = sum(1 for value in query_result.model_dump().values() if value is not None)
        logger.debug(f"Metadata fields count: {metadata_fields}")
        return metadata_fields <= 2

    def _create_filter_dict(self, query_result: QuestionSearch) -> Dict[str, Any]:
        filter_dict = {}
        
        for field_name, value in query_result.model_dump().items():
            if value is None:
                continue
                
            if field_name == 'subject':
                filter_dict[field_name] = value.lower()
            elif field_name in ['year_ad', 'year_bs'] and isinstance(value, list) and value:
                filter_dict[field_name] = {"$in": value}
            elif isinstance(value, list) and value:
                filter_dict[field_name] = value
            else:
                filter_dict[field_name] = value
        
        logger.debug(f"Created filter dictionary: {filter_dict}")
        return filter_dict

    def _filter_docs_by_metadata(self, docs: List[Document], filter_dict: Dict[str, Any]) -> List[Document]:
        filtered_docs = []
        
        for doc in docs:
            try:
                doc_content = json.loads(doc.page_content)
                if self._matches_filter_criteria(doc_content, filter_dict):
                    filtered_docs.append(doc)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse document content: {doc.page_content[:100]}...")
                continue
                
        logger.debug(f"Filtered {len(filtered_docs)} documents from {len(docs)} total")
        return filtered_docs

    def _matches_filter_criteria(self, doc_content: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        for key, value in filter_dict.items():
            if key in ['year_ad', 'year_bs']:
                if isinstance(value, dict) and '$in' in value:
                    if doc_content.get(key) not in value['$in']:
                        return False
            elif isinstance(value, list):
                if not any(v in doc_content.get(key, []) for v in value):
                    return False
            else:
                if str(doc_content.get(key, '')).lower() != str(value).lower():
                    return False
        return True

    def _process_query(self, question: str) -> QueryResult:
        logger.debug(f"Processing query: {question}")
        query_result = self.structured_chain.invoke({"question": question})
        filter_dict = self._create_filter_dict(query_result)
        is_metadata_only = self._is_metadata_only_query(query_result)
        
        logger.debug(f"Query processed: is_metadata_only={is_metadata_only}")
        return QueryResult(
            query_result=query_result,
            filter_dict=filter_dict,
            is_metadata_only=is_metadata_only
        )

    def get_filtered_questions(self, question: str, k: int = 5) -> List[Document]:
        logger.info(f"Getting filtered questions for query: {question}")
        
        # Process the query
        query_info = self._process_query(question)
        logger.debug(f"Filter dictionary: {query_info.filter_dict}")
        
        # Get vector store for collection
        collection_store = vector_store.get_vector_store("c_past_questions")
        
        if query_info.is_metadata_only:
            logger.debug("Using metadata-only filtering")
            retriever = collection_store.as_retriever(search_kwargs={'k': 100})
            docs = retriever.invoke("")
            results = self._filter_docs_by_metadata(docs, query_info.filter_dict)[:k]
        else:
            logger.debug("Using semantic search with filtering")
            retriever = collection_store.as_retriever(search_kwargs={'k': k})
            results = retriever.invoke(question)
            results = self._filter_docs_by_metadata(results, query_info.filter_dict)
        
        logger.info(f"Found {len(results)} matching questions")
        return results

# Create a singleton instance
question_retriever = QuestionRetriever()

@tool
def c_programming_questions(question: str, k: int = 5) -> List[Document]:
    """Retrieves C programming questions from a vector store based on natural language queries.
    
    This tool processes natural language questions about C programming exam questions and returns
    relevant matches from a database of past exam questions. It supports various query types including:
    - Subject-specific questions
    - Questions from specific years (BS or AD)
    - Questions by type (theory or programming)
    - Questions by format (short or long)
    - Questions by marks, topic, unit number, or question number
    - Questions by source (regular or back exam)
    - Questions by semester
    - Questions containing specific keywords
    
    The tool uses a combination of semantic search and metadata filtering to find the most relevant
    questions. For metadata-only queries (e.g., "questions from 2075"), it filters based on exact
    matches. For semantic queries (e.g., "questions about arrays"), it uses vector similarity search.
    
    Args:
        question (str): A natural language query describing the desired C programming questions.
                       Examples:
                       - "Show me programming questions from 2075 BS"
                       - "Find theory questions about arrays"
                       - "Get questions from first semester regular exam"
        k (int, optional): Maximum number of questions to return. Defaults to 5.
    
    Returns:
        List[Document]: A list of Document objects containing the matched questions. Each Document
                       contains the question content and associated metadata (year, type, format,
                       marks, etc.).
    
    Note:
        The tool automatically handles both BS and AD year formats, and can process queries
        mentioning multiple years or date ranges.
    """
    logger.info(f"Tool called with question: {question}, k: {k}")
    return question_retriever.get_filtered_questions(question, k)
