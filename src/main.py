"""
RAG Chatbot API — Main Application
"""

import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.schemas import (
    ChatRequest, ChatResponse,
    IngestRequest, IngestResponse,
    HealthResponse, SourceDocument
)
from src.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# Lifespan: startup + shutdown events
# ─────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Code before yield runs on startup.
    Code after yield runs on shutdown.
    Use this to initialize expensive resources once.
    """
    logger.info("🚀 Starting RAG Chatbot API...")
    logger.info(f"   Embedding model : {config.EMBEDDING_MODEL}")
    logger.info(f"   Chroma dir      : {config.CHROMA_PERSIST_DIR}")
    logger.info(f"   Collection      : {config.COLLECTION_NAME}")

    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    app.state.vectorstore = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=config.CHROMA_PERSIST_DIR,
    )
    app.state.embeddings = embeddings
    # TODO Day 13: Initialize retrieval chain here
    # TODO Day 14: Initialize memory here

    logger.info("✅ API ready.")
    yield  # Server is running here

    logger.info("🛑 Shutting down RAG Chatbot API...")


# ─────────────────────────────────────────
# FastAPI app instance
# ─────────────────────────────────────────

app = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION,
    version=config.API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS — allows browser frontends to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Tighten this in real production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────

@app.get("/", tags=["Root"])
async def root():
    """API root — confirms service is running."""
    return {
        "message": "RAG Chatbot API is running",
        "docs": "/docs",
        "version": config.API_VERSION
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    Load balancers and monitoring tools call this to confirm the service is alive.
    """
    return HealthResponse(
        status="healthy",
        version=config.API_VERSION,
        vectorstore_ready=False,  # Will be True after Day 12
        total_documents=0
    )


@app.post("/ingest", response_model=IngestResponse, tags=["Documents"])
async def ingest_text(request: IngestRequest):
    """
    Ingest raw text into the vector store.
    Day 11: Returns a placeholder — real ingestion added Day 12.
    """
    logger.info(f"Ingest request received: source='{request.source_name}'")

    # Placeholder — real pipeline added Day 12
    return IngestResponse(
        message="Ingestion pipeline not yet connected — coming Day 12",
        chunks_created=0,
        source_name=request.source_name
    )


@app.post("/ingest/file", tags=["Documents"])
async def ingest_file(file: UploadFile = File(...)):
    """
    Upload a PDF or text file for ingestion.
    Accepts multipart/form-data file uploads.
    Day 11: Validates file type, returns placeholder.
    """
    logger.info(f"File upload received: {file.filename}, type: {file.content_type}")

    allowed_types = ["application/pdf", "text/plain"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Allowed: PDF, TXT"
        )

    contents = await file.read()
    file_size_kb = len(contents) / 1024

    logger.info(f"File size: {file_size_kb:.1f} KB")

    # Placeholder — real pipeline added Day 12
    return {
        "message": "File received. Ingestion pipeline coming Day 12.",
        "filename": file.filename,
        "size_kb": round(file_size_kb, 2),
        "content_type": file.content_type
    }


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Main chat endpoint.
    Accepts a question, retrieves relevant document chunks,
    generates a grounded answer.
    Day 11: Returns a placeholder — real RAG chain added Day 13.
    """
    start_time = time.time()
    logger.info(f"Chat request: session='{request.session_id}', message='{request.message[:50]}...'")

    # Placeholder answer — real RAG logic added Day 13
    placeholder_answer = (
        f"RAG pipeline not yet connected. Your question was: '{request.message}'. "
        f"Full RAG answers coming Day 13."
    )

    processing_ms = (time.time() - start_time) * 1000

    return ChatResponse(
        answer=placeholder_answer,
        sources=[],
        session_id=request.session_id,
        processing_time_ms=round(processing_ms, 2)
    )


@app.delete("/vectorstore", tags=["Admin"])
async def clear_vectorstore():
    """
    Clear all documents from the vector store.
    Useful during development and testing.
    """
    logger.warning("Vector store clear requested")
    # Placeholder — real implementation Day 12
    return {"message": "Vector store clear coming Day 12"}


@app.get("/stats", tags=["Admin"])
async def get_stats():
    """
    Return API usage statistics.
    """
    return {
        "api_version": config.API_VERSION,
        "embedding_model": config.EMBEDDING_MODEL,
        "collection_name": config.COLLECTION_NAME,
        "vectorstore_ready": False,
        "note": "Full stats available after Day 12"
    }
