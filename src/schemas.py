"""
Pydantic schemas for the RAG Chatbot API.
These define the shape of every request and response.
FastAPI uses these for automatic validation and documentation.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class ChatRequest(BaseModel):
    """Incoming chat message from the client."""
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's question or message",
        example="What are the main topics in the uploaded documents?"
    )
    session_id: Optional[str] = Field(
        default="default",
        description="Session identifier for conversation memory"
    )
    top_k: Optional[int] = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of document chunks to retrieve"
    )


class SourceDocument(BaseModel):
    """A single retrieved source document chunk."""
    content: str = Field(description="The text content of the chunk")
    source: str = Field(description="Where this chunk came from")
    page: Optional[int] = Field(default=None, description="Page number if applicable")
    score: Optional[float] = Field(default=None, description="Relevance score")


class ChatResponse(BaseModel):
    """Response sent back to the client."""
    answer: str = Field(description="The generated answer")
    sources: List[SourceDocument] = Field(
        default=[],
        description="Document chunks used to generate this answer"
    )
    session_id: str = Field(description="Session identifier")
    processing_time_ms: Optional[float] = Field(
        default=None,
        description="How long the request took in milliseconds"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class IngestRequest(BaseModel):
    """Request to ingest a text document directly."""
    text: str = Field(
        ...,
        min_length=10,
        description="Raw text to ingest into the vector store"
    )
    source_name: str = Field(
        default="manual_input",
        description="Name/identifier for this document"
    )


class IngestResponse(BaseModel):
    """Response after document ingestion."""
    message: str
    chunks_created: int
    source_name: str


class HealthResponse(BaseModel):
    """API health check response."""
    status: str
    version: str
    vectorstore_ready: bool
    total_documents: Optional[int] = None