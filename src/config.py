"""
Centralized configuration using python-dotenv.
All environment variables loaded once here — never scattered across files.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Single source of truth for all configuration."""

    # Embedding
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # ChromaDB
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "rag_chatbot_docs")

    

    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "rag_documents")
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

    # Retrieval
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))

    # API
    API_VERSION: str = "1.0.0"
    API_TITLE: str = "RAG Chatbot API"
    API_DESCRIPTION: str = (
        "Production RAG chatbot API. Upload documents, ask questions, "
        "get answers grounded in your content."
    )

    @classmethod
    def display(cls):
        """Print current config (for debugging — never print secrets)."""
        print(f"Embedding model : {cls.EMBEDDING_MODEL}")
        print(f"Chroma dir      : {cls.CHROMA_PERSIST_DIR}")
        print(f"Collection      : {cls.COLLECTION_NAME}")
        print(f"Top-K           : {cls.TOP_K_RESULTS}")


# Singleton — import this anywhere
config = Config()