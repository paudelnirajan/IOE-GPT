from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()
logger.debug("Environment variables loaded")

try:
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        logger.error("GROQ_API_KEY not found in environment variables")
        raise ValueError("GROQ_API_KEY not found in environment variables")

    logger.info("Initializing ChatGroq with llama-3.3-70b-versatile model")
    llm = ChatGroq(
        api_key=groq_api_key,
        model="llama-3.3-70b-versatile", 
        temperature=0.7,                 
        max_tokens=8192                  
    )
    logger.debug("ChatGroq initialization successful")
except Exception as e:
    logger.error(f"Failed to initialize ChatGroq: {str(e)}")
    raise Exception(f"Failed to initialize ChatGroq: {str(e)}")