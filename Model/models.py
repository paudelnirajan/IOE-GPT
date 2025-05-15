from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

try:
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")

    llm = ChatGroq(
        api_key=groq_api_key,
        model="llama-3.3-70b-versatile", 
        temperature=0.7,                 
        max_tokens=8192                  
    )
except Exception as e:
    raise Exception(f"Failed to initialize ChatGroq: {str(e)}")