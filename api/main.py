from fastapi import FastAPI
from pydantic import BaseModel
import os

from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.rag_pipeline import RAGPipeline


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="RAG Knowledge Assistant API")


# -----------------------------
# Request schema
# -----------------------------
class QuestionRequest(BaseModel):
    question: str


# -----------------------------
# Initialize RAG system
# -----------------------------
print("Loading vector index...")

# Load embedding model
embedder = EmbeddingModel()

# Initialize vector store
vector_store = VectorStore(dimension=384)

# Load saved FAISS index
vector_store.load()

print(f"Loaded {len(vector_store.text_chunks)} chunks from index.")

# Create retriever
retriever = Retriever(vector_store, embedder)

# Initialize RAG pipeline
rag_pipeline = RAGPipeline(
    retriever=retriever,
    api_key=os.getenv("GROQ_API_KEY")
)

print("RAG system ready.")


# -----------------------------
# Root endpoint
# -----------------------------
@app.get("/")
def root():
    return {
        "message": "RAG Knowledge Assistant API is running"
    }


# -----------------------------
# Ask endpoint
# -----------------------------
@app.post("/ask")
def ask_question(request: QuestionRequest):

    try:
        result = rag_pipeline.generate_answer(request.question)

        return {
            "question": request.question,
            "answer": result["answer"],
            "sources": result["sources"]
        }
    except Exception as e:

        return {
            "error": str(e)
        }