import os
from dotenv import load_dotenv

# Load .env file immediately — must happen before any os.getenv() calls
load_dotenv()

class Settings:
    def __init__(self):
        # Database connection settings
        self.DB_HOST = os.getenv("DB_HOST", "localhost")
        self.DB_PORT = int(os.getenv("DB_PORT", "5432"))
        self.DB_NAME = os.getenv("DB_NAME", "arxiv_rag")
        self.DB_USER = os.getenv("DB_USER", "prishita")
        self.DB_PASSWORD = os.getenv("DB_PASSWORD", "")
        
        # Embedding model — runs locally, no API key needed
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        
        # ChromaDB will store its data in this local folder
        self.CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
        
        # ArXiv fetch settings
        self.ARXIV_MAX_RESULTS = int(os.getenv("ARXIV_MAX_RESULTS", "100"))
        self.ARXIV_SEARCH_QUERY = os.getenv("ARXIV_SEARCH_QUERY", "machine learning")
        
        self.RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        self.OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# Single instance — every file imports this one object
settings = Settings()