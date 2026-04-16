"""
Basic tests for the RAG Chatbot API scaffold.
Run with: python -m pytest tests/ -v
"""

from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "RAG Chatbot API" in response.json()["message"]


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_chat_returns_response():
    response = client.post("/chat", json={
        "message": "What is in the documents?",
        "session_id": "test-session"
    })
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert data["session_id"] == "test-session"


def test_chat_validates_empty_message():
    response = client.post("/chat", json={"message": ""})
    assert response.status_code == 422  # Pydantic validation error


def test_ingest_text():
    response = client.post("/ingest", json={
        "text": "This is a test document about AI engineering.",
        "source_name": "test_doc"
    })
    assert response.status_code == 200
    assert "chunks_created" in response.json()


def test_ingest_file_wrong_type():
    # Should reject non-PDF/TXT files
    response = client.post(
        "/ingest/file",
        files={"file": ("test.jpg", b"fake image data", "image/jpeg")}
    )
    assert response.status_code == 400