import hashlib
from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders import Docx2txtLoader

from src.config import config


def load_documents_from_file(file_path: str) -> List[Document]:
    """
    Load a file into LangChain Documents based on file extension.
    Supported: .pdf, .txt, .md, .docx
    """
    path = Path(file_path)
    extension = path.suffix.lower()

    if extension == ".pdf":
        loader = PyPDFLoader(file_path)
    elif extension in [".txt", ".md"]:
        loader = TextLoader(file_path, encoding="utf-8")
    elif extension == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {extension}")

    documents = loader.load()

    # Tag every document with the source filename
    for doc in documents:
        doc.metadata["source"] = path.name
        doc.metadata["file_path"] = str(path)

    return documents


def load_documents_from_text(text: str, source_name: str = "direct_input") -> List[Document]:
    """
    Wrap a raw text string into a LangChain Document.
    Used by the POST /ingest endpoint (text body).
    """
    return [Document(
        page_content=text,
        metadata={"source": source_name}
    )]


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into chunks using RecursiveCharacterTextSplitter.
    Adds chunk_index to metadata for traceability.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    # Tag each chunk with its index within its source
    source_counters = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        source_counters[source] = source_counters.get(source, 0) + 1
        chunk.metadata["chunk_index"] = source_counters[source]

    return chunks


def generate_chunk_ids(chunks: List[Document]) -> List[str]:
    """
    Generate deterministic IDs for each chunk.
    Same content + metadata always produces the same ID.
    This prevents duplicate ingestion.
    """
    ids = []
    for chunk in chunks:
        content = chunk.page_content
        source = chunk.metadata.get("source", "")
        chunk_index = chunk.metadata.get("chunk_index", 0)
        raw = f"{source}::{chunk_index}::{content[:100]}"
        doc_id = hashlib.md5(raw.encode()).hexdigest()
        ids.append(doc_id)
    return ids


def process_file(file_path: str) -> Tuple[List[Document], List[str]]:
    """
    Full pipeline: file → documents → chunks → chunk IDs.
    Returns (chunks, ids) ready for ChromaDB insertion.
    """
    documents = load_documents_from_file(file_path)
    chunks = chunk_documents(documents)
    ids = generate_chunk_ids(chunks)
    return chunks, ids


def process_text(text: str, source_name: str = "direct_input") -> Tuple[List[Document], List[str]]:
    """
    Full pipeline: raw text → documents → chunks → chunk IDs.
    """
    documents = load_documents_from_text(text, source_name)
    chunks = chunk_documents(documents)
    ids = generate_chunk_ids(chunks)
    return chunks, ids