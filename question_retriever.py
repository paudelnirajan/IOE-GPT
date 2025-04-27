import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel, Field
import json
from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class QuestionSearch(BaseModel):
    """Search over the json file about the question of particular year or some particular metadata."""

    id: Optional[str] = Field(
        None,
        description="ID of a particular question. Will hold a value like 'subject_code+question_number'."
    )

    subject: Literal["computer programming"] = Field(
        description="Subject the question belongs to."
    )

    year_ad: Optional[List[int]] = Field(
        None,
        description="List of years in AD that the questions appeared."
    )

    year_bs: Optional[List[int]] = Field(
        None,
        description="List of years in BS that the questions appeared."
    )

    type: Optional[Literal["theory", "programming"]] = Field(
        None,
        description="Type of the question."
    )

    format: Optional[Literal["short", "long"]] = Field(
        None,
        description="Format of the question (e.g., short answer, long answer)."
    )

    marks: Optional[int] = Field(
        None,
        description="Marks allocated to the question being searched."
    )

    topic: Optional[Literal[
        "programming_fundamentals",
        "algorithm_and_flowchart",
        "introduction_c_programming",
        "data_and_expressions",
        "input_output",
        "control_structures",
        "arrays_strings_pointers",
        "functions",
        "structures",
        "file_handling",
        "oop_overview"
    ]] = Field(
        None,
        description="Topic that the question is from."
    )

    unit: Optional[int] = Field(
        None,
        description="Unit the question is from."
    )

    tags: Optional[List[str]] = Field(
        None,
        description="Tags associated with the question."
    )

    question_number: Optional[str] = Field(
        None,
        description="Question number (e.g., '1a', '2b', 4, 5)."
    )

    source: Optional[Literal["regular", "back"]] = Field(
        None,
        description="Source exam type (regular or back paper)."
    )

    semester: Optional[Literal[
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "sixth",
        "seventh",
        "eighth"
    ]] = Field(
        None,
        description="Semester the question is for."
    )

class QuestionRetriever:
    def __init__(self, docs: List[Document]):
        """
        Initialize the QuestionRetriever with a list of documents.
        
        Args:
            docs: List of Document objects containing question data
        """
        # Parse JSON strings in documents
        self.docs = []
        for doc in docs:
            try:
                # Parse the JSON string into a dictionary
                content = json.loads(doc.page_content)
                # Store the parsed content back as a string
                doc.page_content = json.dumps(content)
                self.docs.append(doc)
            except json.JSONDecodeError:
                print(f"Skipping invalid document: {doc.page_content[:100]}...")
                continue
        
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = InMemoryVectorStore(embedding=self.embeddings)
        self.vector_store.add_documents(self.docs)
        
        # Initialize the LLM for query understanding
        self.llm = ChatGroq(
            api_key=os.getenv('GROQ_API_KEY'),
            model_name="llama-3.3-70b-versatile"
        )
        
        # Create the structured chain for query parsing
        system_prompt = """You are an expert at converting user questions about past exam papers into structured JSON queries.
        Given a user's question, your goal is to construct a JSON query object that conforms to the `QuestionSearch` schema.
        
        When users mention multiple years, collect them into a list. For example:
        - "questions from 2075, 2076 BS" → year_bs: [2075, 2076]
        - "questions between 2019 and 2021 AD" → year_ad: [2019, 2020, 2021]
        
        Identify if the query is metadata-only (e.g., "all questions from 2079") or needs semantic search
        (e.g., "questions about arrays from 2079").
        """
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}")
        ])
        self.structured_llm = self.llm.with_structured_output(QuestionSearch)
        self.structured_chain = self.prompt | self.structured_llm

    def is_metadata_only_query(self, query_result: QuestionSearch) -> bool:
        """
        Determine if a query can be satisfied by metadata filtering alone.
        
        Args:
            query_result: Structured query result
        
        Returns:
            bool: True if query only needs metadata filtering
        """
        # Count non-None metadata fields
        metadata_fields = sum(1 for value in query_result.model_dump().values() if value is not None)
        
        # If only subject and one other field (e.g., year or topic) are specified,
        # it's likely a metadata-only query
        return metadata_fields <= 2

    def filter_docs_by_metadata(self, filter_dict: Dict[str, Any]) -> List[Document]:
        """
        Filter documents based on metadata criteria.
        
        Args:
            filter_dict: Dictionary of metadata filters
            
        Returns:
            List[Document]: Filtered documents
        """
        filtered_docs = []
        for doc in self.docs:
            try:
                doc_content = json.loads(doc.page_content)
                matches = True
                
                for key, value in filter_dict.items():
                    if key in ['year_ad', 'year_bs']:
                        if isinstance(value, dict) and '$in' in value:
                            if doc_content.get(key) not in value['$in']:
                                matches = False
                                break
                    elif isinstance(value, list):
                        if not any(v in doc_content.get(key, []) for v in value):
                            matches = False
                            break
                    else:
                        if str(doc_content.get(key, '')).lower() != str(value).lower():
                            matches = False
                            break
                
                if matches:
                    filtered_docs.append(doc)
            except json.JSONDecodeError:
                continue
                
        return filtered_docs

    def create_dynamic_filter(self, query_result: QuestionSearch) -> dict:
        """
        Create a filter dictionary from the structured query result.
        
        Args:
            query_result: Structured query result
            
        Returns:
            dict: Filter dictionary
        """
        filter_dict = {}
        
        for field_name, value in query_result.model_dump().items():
            if value is not None:
                if field_name == 'subject':
                    filter_dict[field_name] = value.lower()
                elif field_name in ['year_ad', 'year_bs'] and isinstance(value, list):
                    if value:
                        filter_dict[field_name] = {"$in": value}
                elif isinstance(value, list):
                    if value:
                        filter_dict[field_name] = value
                else:
                    filter_dict[field_name] = value
        
        return filter_dict

    def get_questions(self, question: str, k: int = 5) -> List[Document]:
        """
        Get questions based on the user's natural language query.
        
        Args:
            question: Natural language question from user
            k: Maximum number of results to retrieve
            
        Returns:
            List[Document]: Retrieved documents
        """
        # Parse the query into structured format
        query_result = self.structured_chain.invoke({"question": question})
        
        # Create filter dictionary
        filter_dict = self.create_dynamic_filter(query_result)
        
        # Check if this is a metadata-only query
        if self.is_metadata_only_query(query_result):
            # For metadata-only queries, just return filtered results
            return self.filter_docs_by_metadata(filter_dict)[:k]
        else:
            # For semantic queries, first filter then do semantic search
            filtered_docs = self.filter_docs_by_metadata(filter_dict)
            
            if not filtered_docs:
                return []
                
            # Create temporary vector store with filtered docs
            temp_vector_store = InMemoryVectorStore(embedding=self.embeddings)
            temp_vector_store.add_documents(filtered_docs)
            
            # Perform semantic search on filtered docs
            retriever = temp_vector_store.as_retriever(search_kwargs={'k': k})
            return retriever.invoke(question)

    def display_questions(self, question: str, k: int = 5) -> None:
        """
        Display questions with their content.
        
        Args:
            question: Natural language question from user
            k: Maximum number of results to retrieve
        """
        results = self.get_questions(question, k)
        
        if not results:
            print("No documents found matching the specified criteria.")
            return
        
        print(f"\nFound {len(results)} matching documents:")
        for i, doc in enumerate(results, 1):
            try:
                doc_content = json.loads(doc.page_content)
                print(f"\n{i}. Question: {doc_content['question']}")
                print(f"   Year: {doc_content.get('year_bs')} BS / {doc_content.get('year_ad')} AD")
                print(f"   Topic: {doc_content.get('topic')}")
                print(f"   Marks: {doc_content.get('marks')}")
                print(f"   Type: {doc_content.get('type')}")
                print("-" * 80)
            except json.JSONDecodeError:
                print(f"Error: Could not parse document {i}")
                continue

# Example usage:
if __name__ == "__main__":
    from langchain_community.document_loaders import JSONLoader
    import json
    
    # Load documents
    try:
        print("Loading documents...")
        with open("formatted_data/c_question.json", "r") as f:
            data = json.load(f)
            
        # Convert each JSON object to a Document with stringified content
        docs = [Document(page_content=json.dumps(item)) for item in data]
        print(f"Loaded {len(docs)} documents")
        
        # if docs:
            # print("Sample document content:")
            # print(docs[0].page_content)
            
        # Create retriever
        retriever = QuestionRetriever(docs)
        
        while True:
            question = input("Enter any query regarding the past questions of Computer Programming::\n\n\b")
            retriever.display_questions(question)
        
    except FileNotFoundError:
        print("Error: c_question.json file not found in formatted_data directory")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in c_question.json: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise 