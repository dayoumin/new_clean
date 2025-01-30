import os
from dotenv import load_dotenv

def load_config():
    load_dotenv()
    
    return {
        "model_name": "gpt-4",
        "embedding_model": "text-embedding-3-large",
        "vectorstore_path": "faiss_vectorstore",
        "api_keys": {
            "openai": os.getenv("OPENAI_API_KEY")
        }
    } 