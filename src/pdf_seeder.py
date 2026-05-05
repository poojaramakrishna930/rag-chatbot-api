# rag-chatbot-api/src/pdf_seeder.py
# Creates 3 reference PDFs and ingests them into ChromaDB.
# Called automatically at server startup if ChromaDB is empty.

import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from src.ingestion_pipeline import ingest_file
from src.vectorstore_manager import vectorstore_manager

# ---------------------------------------------------------------------------
# PDF content
# ---------------------------------------------------------------------------

PDF_CONTENT = {
    "langchain_guide.pdf": {
        "title": "LangChain Developer Guide",
        "sections": [
            ("What is LangChain?",
             "LangChain is an open-source Python framework for building applications powered by large language models. "
             "It provides abstractions for chaining prompts, memory, tools, and retrievers into coherent pipelines. "
             "LangChain supports integrations with OpenAI, HuggingFace, Anthropic, Cohere, and many other providers."),

            ("LangChain Components",
             "LangChain is built around six core components: Models (LLMs and chat models), Prompts (templates and selectors), "
             "Chains (sequences of calls), Memory (state across interactions), Agents (LLMs that choose actions), "
             "and Retrievers (fetching relevant documents). Each component is modular and interchangeable."),

            ("What is LCEL?",
             "LCEL stands for LangChain Expression Language. It uses the pipe operator to chain components together. "
             "For example: chain = prompt | llm | output_parser. "
             "LCEL supports streaming, async execution, and parallel branching out of the box. "
             "It replaced the older sequential chain pattern and is now the recommended way to build pipelines."),

            ("LangChain Memory Types",
             "LangChain provides several memory classes: ConversationBufferMemory stores the full history, "
             "ConversationSummaryMemory summarizes older turns to save tokens, "
             "ConversationBufferWindowMemory keeps only the last k turns, "
             "and VectorStoreRetrieverMemory stores past interactions in a vector DB for semantic recall."),

            ("LangChain Agents",
             "Agents in LangChain use an LLM to decide which tools to call and in what order. "
             "The ReAct agent follows a Thought-Action-Observation loop until it reaches a final answer. "
             "Tools are Python functions decorated with @tool that the agent can invoke. "
             "LangGraph is the recommended framework for building stateful multi-step agents."),
        ]
    },

    "rag_concepts.pdf": {
        "title": "RAG Systems — Core Concepts",
        "sections": [
            ("What is RAG?",
             "RAG stands for Retrieval-Augmented Generation. It is a technique that combines a retrieval system "
             "with a generative language model. Instead of relying solely on the model's training data, "
             "RAG fetches relevant documents at inference time and passes them as context to the LLM. "
             "This reduces hallucination and keeps answers grounded in real source material."),

            ("Chunking Strategies",
             "Documents must be split into smaller chunks before embedding. Fixed-size chunking splits on character count. "
             "Recursive chunking splits on paragraph and sentence boundaries first, falling back to characters. "
             "Semantic chunking groups sentences by meaning. Chunk size and overlap are critical hyperparameters — "
             "too large and retrieval is imprecise, too small and chunks lose context."),

            ("Embedding Models",
             "Embedding models convert text into dense numerical vectors. All-MiniLM-L6-v2 produces 384-dimensional vectors "
             "and runs locally on CPU with no API key required. Text-embedding-ada-002 from OpenAI produces 1536-dimensional vectors. "
             "Both the documents and the query must be embedded using the same model for similarity search to work correctly."),

            ("Vector Databases",
             "Vector databases store embeddings and support fast nearest-neighbour search. ChromaDB is open-source, "
             "persistent, and supports metadata filtering — ideal for local RAG development. "
             "FAISS is a similarity search library from Meta — extremely fast but in-memory only. "
             "Pinecone and Weaviate are managed cloud vector databases suited for production scale."),

            ("RAG Evaluation Metrics",
             "RAGAS provides five metrics for evaluating RAG systems. Context Precision measures whether retrieved chunks "
             "are relevant to the question. Context Recall measures whether all relevant information was retrieved. "
             "Faithfulness measures whether the answer is supported by the context. "
             "Answer Relevancy measures whether the answer addresses the question. "
             "These metrics require a ground truth dataset to compute."),
        ]
    },

    "fastapi_reference.pdf": {
        "title": "FastAPI Reference for AI Engineers",
        "sections": [
            ("What is FastAPI?",
             "FastAPI is a modern Python web framework for building APIs with high performance and automatic validation. "
             "It uses Python type hints and Pydantic models to validate requests and serialize responses automatically. "
             "FastAPI generates interactive OpenAPI documentation at /docs and /redoc without any extra configuration."),

            ("Path Operations and Routing",
             "FastAPI uses decorators to define routes: @app.get, @app.post, @app.put, @app.delete. "
             "Each decorator takes a path string and optional parameters like response_model and status_code. "
             "Route functions can be async or sync. FastAPI runs async functions in an event loop "
             "and sync functions in a thread pool to avoid blocking."),

            ("Pydantic Integration",
             "Pydantic models define the shape of request and response bodies. When a request arrives, "
             "FastAPI automatically parses and validates the JSON body against the Pydantic model. "
             "If validation fails, FastAPI returns a 422 Unprocessable Entity response with detailed error messages. "
             "Optional fields use Optional[type] with a default value of None."),

            ("Dependency Injection with Depends",
             "FastAPI's Depends() system allows shared logic to be injected into route functions. "
             "Common uses include database session management, authentication token validation, "
             "and rate limiting. Dependencies can themselves have dependencies, forming a tree. "
             "Depends() promotes code reuse and makes routes easier to test in isolation."),

            ("Lifespan and Startup Events",
             "The lifespan context manager handles startup and shutdown logic. Code before yield runs at startup — "
             "ideal for loading ML models, initializing database connections, and warming up caches. "
             "Code after yield runs at shutdown — ideal for closing connections and flushing buffers. "
             "This replaced the older @app.on_event('startup') pattern and is now the recommended approach."),
        ]
    },
}

# ---------------------------------------------------------------------------
# PDF directory — stored inside the project, gitignored
# ---------------------------------------------------------------------------

PDF_DIR = os.path.join(os.path.dirname(__file__), "..", "seed_pdfs")


def _create_pdf(filename: str, title: str, sections: list) -> str:
    """Create a single PDF and return its file path."""
    os.makedirs(PDF_DIR, exist_ok=True)
    filepath = os.path.join(PDF_DIR, filename)

    c = canvas.Canvas(filepath, pagesize=A4)
    width, height = A4
    y = height - 60

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, title)
    y -= 50

    for heading, body in sections:
        if y < 120:
            c.showPage()
            y = height - 60

        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, heading)
        y -= 20

        c.setFont("Helvetica", 10)
        words = body.split()
        line = ""
        for word in words:
            test = line + " " + word if line else word
            if c.stringWidth(test, "Helvetica", 10) < width - 100:
                line = test
            else:
                c.drawString(50, y, line)
                y -= 15
                line = word
                if y < 80:
                    c.showPage()
                    y = height - 60
        if line:
            c.drawString(50, y, line)
        y -= 35

    c.save()
    return filepath


def seed_pdfs_if_empty() -> None:
    """
    Called at startup. If ChromaDB already has documents, skip entirely.
    If empty, create the 3 PDFs and ingest them.
    This means restarting the server never re-ingests duplicates.
    """
    stats = vectorstore_manager.get_stats()
    if stats["document_count"] > 0:
        print(f"[Seeder] ChromaDB already has {stats['document_count']} chunks. Skipping seed.")
        return

    print("[Seeder] ChromaDB is empty. Creating and ingesting 3 seed PDFs...")

    for filename, data in PDF_CONTENT.items():
        filepath = _create_pdf(filename, data["title"], data["sections"])
        result = ingest_file(filepath)
        print(f"[Seeder] {filename} → {result['chunks_stored']} chunks stored.")

    final = vectorstore_manager.get_stats()
    print(f"[Seeder] Done. Total chunks in ChromaDB: {final['document_count']}")