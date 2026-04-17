from typing import List, Optional
import chromadb
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import config


class VectorStoreManager:
    """
    Singleton-style manager for ChromaDB.
    Handles initialization, insertion, querying, deletion, and stats.
    """

    def __init__(self):
        self._vectorstore: Optional[Chroma] = None
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._client: Optional[chromadb.PersistentClient] = None

    def initialize(self) -> None:
        """
        Called once at app startup (inside lifespan).
        Sets up embeddings model and ChromaDB persistent client.
        """
        print(f"[VectorStore] Loading embedding model: {config.EMBEDDING_MODEL}")
        self._embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self._client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)

        self._vectorstore = Chroma(
            client=self._client,
            collection_name=config.COLLECTION_NAME,
            embedding_function=self._embeddings,
        )
        print(f"[VectorStore] Ready. Collection: {config.COLLECTION_NAME}")

    @property
    def is_ready(self) -> bool:
        return self._vectorstore is not None

    def add_documents(self, chunks: List[Document], ids: List[str]) -> int:
        """
        Insert chunks into ChromaDB.
        Returns number of chunks added.
        Uses deterministic IDs — duplicates are silently skipped by Chroma.
        """
        if not self.is_ready:
            raise RuntimeError("VectorStore not initialized.")

        self._vectorstore.add_documents(documents=chunks, ids=ids)
        return len(chunks)

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Retrieve top-k most similar chunks for a query string.
        """
        if not self.is_ready:
            raise RuntimeError("VectorStore not initialized.")

        return self._vectorstore.similarity_search(query, k=k)

    def get_stats(self) -> dict:
        """
        Return collection stats: document count and collection name.
        """
        if not self.is_ready:
            return {"document_count": 0, "collection_name": config.COLLECTION_NAME, "ready": False}

        collection = self._client.get_collection(config.COLLECTION_NAME)
        count = collection.count()
        return {
            "document_count": count,
            "collection_name": config.COLLECTION_NAME,
            "ready": True,
        }

    def clear(self) -> None:
        """
        Delete all documents in the collection.
        The collection itself remains — only the contents are wiped.
        """
        if not self.is_ready:
            raise RuntimeError("VectorStore not initialized.")

        self._client.delete_collection(config.COLLECTION_NAME)

        # Re-create the empty collection
        self._vectorstore = Chroma(
            client=self._client,
            collection_name=config.COLLECTION_NAME,
            embedding_function=self._embeddings,
        )
        print("[VectorStore] Collection cleared and re-created.")


# Module-level singleton — imported by main.py and ingestion_pipeline.py
vectorstore_manager = VectorStoreManager()