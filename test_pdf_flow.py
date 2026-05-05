# rag-chatbot-api/test_pdf_flow.py
# Make sure the server is running before executing this script:
#   uvicorn src.main:app --reload
# Then in a second terminal:
#   python test_pdf_flow.py

import requests

BASE_URL = "http://127.0.0.1:8000"
PDF_PATH = "test_document.pdf"


def check_health():
    print("\n--- Health Check ---")
    r = requests.get(f"{BASE_URL}/health")
    print(r.json())


def upload_pdf(path: str):
    print(f"\n--- Uploading PDF: {path} ---")
    with open(path, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/ingest/file",
            files={"file": (path, f, "application/pdf")},
        )
    result = r.json()
    print(result)
    return result


def query(question: str):
    print(f"\n--- Query: {question} ---")
    r = requests.post(
        f"{BASE_URL}/chat",
        json={"message": question},
    )
    data = r.json()
    print(f"Answer: {data['answer']}")
    print(f"Sources ({len(data['sources'])}):")
    for s in data["sources"]:
        print(f"  [{s['source']} | chunk {s['chunk_index']}] {s['content'][:100]}...")


def check_stats():
    print("\n--- Stats ---")
    r = requests.get(f"{BASE_URL}/stats")
    print(r.json())


if __name__ == "__main__":
    check_health()
    upload_pdf(PDF_PATH)
    check_stats()

    questions = [
        "What is LangChain used for?",
        "How does a RAG pipeline work?",
        "What is ChromaDB?",
        "What is the capital of Japan?",   # out-of-context — should trigger fallback
    ]

    for q in questions:
        query(q)