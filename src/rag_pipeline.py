# rag-chatbot-api/src/rag_pipeline.py

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from src.vectorstore_manager import vectorstore_manager
from src.config import config


# ---------------------------------------------------------------------------
# Prompt Template (optimized for chat-style LLMs)
# ---------------------------------------------------------------------------

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant.

Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}

If the answer is not in the context, say:
"I don't have enough information to answer that."

Answer:"""
)


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:

    def __init__(self):
        self._chain = None

    def initialize(self) -> None:
        print(f"[RAGPipeline] Loading model: {config.LLM_MODEL}")

        
        tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            config.LLM_MODEL
        )

        pipe = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=config.MAX_NEW_TOKENS,
            max_length=None,
            do_sample=True,
            temperature=0.7,
            return_full_text=False,
        )

        llm = HuggingFacePipeline(pipeline=pipe)

        self._chain = RAG_PROMPT | llm | StrOutputParser()

        print("[RAGPipeline] Ready.")

    @property
    def is_ready(self) -> bool:
        return self._chain is not None

    def answer(self, question: str, k: int = 4) -> dict:
        if not self.is_ready:
            raise RuntimeError("RAGPipeline not initialized.")

        if not vectorstore_manager.is_ready:
            raise RuntimeError("VectorStore not initialized.")

        # Retrieve
        docs = vectorstore_manager.similarity_search(question, k=k)

        if not docs:
            return {
                "answer": "I don't have enough information to answer that. Please ingest some documents first.",
                "sources": [],
            }

        # Format context
        context = "\n\n".join([doc.page_content for doc in docs])

        # Generate
        answer = self._chain.invoke({
            "context": context,
            "question": question
        }).strip()

         # Hard fallback (important)
        if not answer or answer.lower().startswith("you are a helpful assistant"):
            answer = "I don't have enough information to answer that."

        # Build sources
        sources = []
        for doc in docs:
            sources.append({
                "content": doc.page_content[:200],
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", None),
                "chunk_index": doc.metadata.get("chunk_index", None),
            })

        return {
            "answer": answer.strip(),
            "sources": sources,
        }


# Singleton
rag_pipeline = RAGPipeline()