from fastapi import FastAPI, HTTPException, Form
from vector_store import IoePastQuestionsVectorStore

app = FastAPI()
vector_manager = IoePastQuestionsVectorStore()

@app.post("/update-vector-store")
def update_vector_store(
    collection_name: str = Form(..., description="Name of the collection to update"),
    file_path: str = Form("formatted_data/c_question.json", description="Path to the JSON file")
):
    try:
        vector_store = vector_manager.update_vector_store(
            collection_name=collection_name,
            file_path=file_path
        )
        return {
            "status": "success",
            "message": f"Vector store updated successfully for collection: {collection_name}"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update vector store: {str(e)}"
        )