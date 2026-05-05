# rag-chatbot-api/src/main.py

import os
import time
import shutil
import tempfile
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from src.vectorstore_manager import vectorstore_manager
from src.ingestion_pipeline import ingest_text, ingest_file
from src.rag_pipeline import rag_pipeline
from src.pdf_seeder import seed_pdfs_if_empty
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
# Lifespan: startup + shutdown
# ─────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting RAG Chatbot API...")
    logger.info(f"   Embedding model : {config.EMBEDDING_MODEL}")
    logger.info(f"   Chroma dir      : {config.CHROMA_PERSIST_DIR}")
    logger.info(f"   Collection      : {config.COLLECTION_NAME}")

    vectorstore_manager.initialize()   # 1. connect ChromaDB
    seed_pdfs_if_empty()               # 2. ingest 3 PDFs if ChromaDB is empty
    rag_pipeline.initialize()          # 3. load LLM

    logger.info("✅ API ready.")
    yield
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "RAG Chatbot API is running",
        "docs": "/docs",
        "version": config.API_VERSION
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    stats = vectorstore_manager.get_stats()
    return HealthResponse(
        status="healthy",
        version=config.API_VERSION,
        vectorstore_ready=stats["ready"],
        document_count=stats["document_count"],
    )


@app.post("/ingest", response_model=IngestResponse, tags=["Documents"])
async def ingest_text_endpoint(request: IngestRequest):
    logger.info(f"Ingest request received: source='{request.source}'")
    result = ingest_text(request.text, source_name=request.source or "direct_input")
    return IngestResponse(
        success=True,
        message=f"Ingested {result['chunks_stored']} chunks from {result['source']}",
        chunks_created=result["chunks_created"],
        source_name=result["source"],
    )


@app.post("/ingest/file", response_model=IngestResponse, tags=["Documents"])
async def ingest_file_endpoint(file: UploadFile = File(...)):
    logger.info(f"File upload received: {file.filename}, type: {file.content_type}")

    allowed_types = ["application/pdf", "text/plain"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Allowed: PDF, TXT"
        )

    # Use system temp directory — safe on Windows
    suffix = os.path.splitext(file.filename)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_path = tmp.name

    logger.info(f"Saved to temp path: {temp_path}")

    try:
        result = ingest_file(temp_path)
    finally:
        os.remove(temp_path)

    return IngestResponse(
        success=True,
        message=f"Ingested {result['chunks_stored']} chunks from {file.filename}",
        chunks_created=result["chunks_created"],
        source_name=file.filename,
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_endpoint(request: ChatRequest):
    start_time = time.time()
    logger.info(f"Chat request: session='{request.session_id}', message='{request.message[:50]}'")

    result = rag_pipeline.answer(
        question=request.message,
        k=config.TOP_K if hasattr(config, "TOP_K") else 4,
    )
    sources = [SourceDocument(**s) for s in result["sources"]]
    processing_ms = (time.time() - start_time) * 1000

    return ChatResponse(
        session_id=request.session_id,
        answer=result["answer"],
        sources=sources,
        processing_time_ms=round(processing_ms, 2)
    )


@app.delete("/vectorstore", tags=["Admin"])
async def clear_vectorstore():
    logger.warning("Vector store clear requested")
    vectorstore_manager.clear()
    return {"message": "Vectorstore cleared successfully"}


@app.get("/stats", tags=["Admin"])
async def get_stats():
    return vectorstore_manager.get_stats()