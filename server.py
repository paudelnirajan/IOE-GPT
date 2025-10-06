from typing import Optional
from fastapi import FastAPI, HTTPException, Form
from vector_store import IoePastQuestionsVectorStore
from langchain_core.messages import HumanMessage
from graph_building import build_graph
from core.db_manager import db_manager
from dotenv import load_dotenv
from langgraph.checkpoint.redis import RedisSaver
from utilities import should_reset_checkpoint, delete_thread_checkpoints

load_dotenv()

app = FastAPI()
vector_manager = IoePastQuestionsVectorStore()

# Initialize Redis connection
DB_URI = "redis://localhost:6379"
redis_saver = None
graph = None

@app.on_event("startup")
async def startup_event():
    """Initialize database connection when the application starts"""
    global redis_saver, graph
    print("[INFO] Initializing database connection on server startup...")
    # This will trigger the singleton initialization
    db_manager
    
    # Initialize Redis and graph
    with RedisSaver.from_conn_string(DB_URI) as checkpointer:
        checkpointer.setup()
        graph = build_graph(checkpointer)
        redis_saver = checkpointer
    
    print("[INFO] Database connection initialized successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up Redis connection when the application shuts down"""
    print("[INFO] Cleaning up Redis connection...")
    if redis_saver:
        redis_saver.close()
    print("[INFO] Redis connection closed")

@app.post("/update-vector-store")
def update_vector_store(
    collection_name: str = Form(..., description="Name of the collection to update"),
    file_path: str = Form("formatted_data/c_question.json", description="Path to the JSON file")
):
    try:
        print(f"[INFO] Attempting to update vector store for collection: {collection_name}")
        vector_store = vector_manager.update_vector_store(
            collection_name=collection_name,
            file_path=file_path
        )
        print(f"[INFO] Successfully updated vector store for collection: {collection_name}")
        return {
            "status": "success",
            "message": f"Vector store updated successfully for collection: {collection_name}"
        }
    except Exception as e:
        print(f"[ERROR] Failed to update vector store: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update vector store: {str(e)}"
        )
    

@app.post("/response")
def process_query(
    query: str = Form(..., description="Your question about C programming"),
    sender_id: str = Form(..., description="Unique identifier for the sender"),
    metadata: Optional[str] = Form("metadata_from_front_end", description="Metadata information from frontend")
):
    try:
        print(f"[INFO] Received query from sender {sender_id}: {query}")
        
        # Check if we should reset the conversation
        if should_reset_checkpoint(query):
            delete_thread_checkpoints(redis_saver, sender_id)
            return {
                "messages": {
                    "type": "system",
                    "content": "Hello, how can I help you?"
                }
            }
        
        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "context": {}  # Initialize empty context
        }
        print(f"[INFO] Created initial state with query")

        # Create configuration
        config = {
            "configurable": {
                "thread_id": sender_id
            },
            "recursion_limit": 25
        }
        print(f"[INFO] Created configuration with thread_id: {sender_id}")

        # Invoke the graph
        print(f"[INFO] Invoking graph with query")
        result = graph.invoke(
            initial_state,
            config=config
        )
        print(f"[INFO] Graph invocation completed successfully")

        # Extract messages and summary from result
        messages = []
        for message in result["messages"]:
            messages.append({
                "type": message.type,
                "content": message.content
            })
        print(f"[INFO] Processed {len(messages)} messages from result")
        print(f"[INFO] \n---Message --- \n {messages[-1]}\n--------- \n")
        
        # Get the summary from the context
        summary = ""
        if result.get('context') and result['context'].get('running_summary'):
            summary = result['context']['running_summary'].summary
  
            
        print(f"[INFO] \n---Summary --- \n {summary}\n--------- \n")
        
        response = {
            "messages": messages[-1],
            "summary": summary
        }
        print(f"[INFO] Returning response for sender {sender_id}")
        return response

    except Exception as e:
        print(f"[ERROR] Failed to process query: {str(e)}")
        print(f"[ERROR] Query: {query}")
        print(f"[ERROR] Sender ID: {sender_id}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process query: {str(e)}"
        )