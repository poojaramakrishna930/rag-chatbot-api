from src.document_processor import process_file, process_text
from src.vectorstore_manager import vectorstore_manager


def ingest_text(text: str, source_name: str = "direct_input") -> dict:
    """
    Ingest raw text into the vectorstore.
    Returns a summary of what was stored.
    """
    chunks, ids = process_text(text, source_name)
    added = vectorstore_manager.add_documents(chunks, ids)
    return {
        "source": source_name,
        "chunks_created": len(chunks),
        "chunks_stored": added,
    }


def ingest_file(file_path: str) -> dict:
    """
    Ingest a file into the vectorstore.
    Supports: .pdf, .txt, .md, .docx
    Returns a summary of what was stored.
    """
    chunks, ids = process_file(file_path)
    added = vectorstore_manager.add_documents(chunks, ids)
    return {
        "source": file_path,
        "chunks_created": len(chunks),
        "chunks_stored": added,
    }