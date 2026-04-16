# RAG Chatbot API

A production-ready Retrieval-Augmented Generation (RAG) chatbot API built with
FastAPI, LangChain, ChromaDB, and HuggingFace sentence transformers.

## Stack
- **FastAPI** — API framework
- **LangChain** — RAG orchestration
- **ChromaDB** — Vector store
- **HuggingFace** — Local embeddings (all-MiniLM-L6-v2)

## Run locally
```bash
pip install -r requirements.txt
uvicorn src.main:app --reload --port 8000
```

## API Docs
Visit `http://localhost:8000/docs` for interactive Swagger UI.

## Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | /health | Health check |
| POST | /chat | Ask a question |
| POST | /ingest | Ingest text |
| POST | /ingest/file | Upload PDF or TXT |

## Status
🔨 In active development — Days 11–26 of 70-day AI engineer training.