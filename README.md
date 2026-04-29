# RAG Knowledge Assistant  
### Production-Style Retrieval Augmented Generation System with FAISS, FastAPI, Docker and Streamlit

---

# Project Overview

Large Language Models can generate impressive responses, but they may hallucinate and often lack access to domain-specific knowledge.

This project builds an end-to-end **Retrieval Augmented Generation (RAG)** system that grounds LLM responses using a custom knowledge base through semantic retrieval.

The system enables users to:

- Ask natural language questions against a knowledge base
- Retrieve semantically relevant context using vector search
- Generate grounded responses using an LLM
- Access the system through both API and chat interface
- Demonstrate production-style AI engineering architecture

---

# Live Demo

### Streamlit Chat Application  
https://rag-knowledge-assistant333.streamlit.app/

### FastAPI Documentation  
https://rag-knowledge-assistant-sw3i.onrender.com/docs

---

# Problem Statement

Traditional LLM applications suffer from:

- Hallucinated responses
- No access to proprietary knowledge
- Limited factual grounding
- No retrieval mechanism for external context

This project solves that by combining:

- Semantic retrieval  
- Vector search  
- Prompt augmentation  
- LLM answer generation

---

# What is Retrieval Augmented Generation (RAG)

RAG augments a user query with retrieved knowledge before sending it to an LLM.

System Flow:

```text
User Question
↓

Embed Query

↓

Retrieve Relevant Chunks (FAISS)

↓

Augment Prompt with Context

↓

LLM Generates Grounded Answer
```

This reduces hallucination and improves answer reliability.

---

# Project Pipeline

The system consists of the following stages:

1. Knowledge Base Preparation  
2. Text Chunking  
3. Embedding Generation  
4. Vector Database Indexing (FAISS)  
5. Semantic Retrieval Pipeline  
6. Prompt Augmentation  
7. LLM Answer Generation  
8. API Deployment  
9. Interactive Chat Interface

---

# Knowledge Base

Initial knowledge base contains foundational AI/ML topics including:

- Machine Learning  
- Supervised vs Unsupervised Learning  
- Retrieval Augmented Generation  
- Vector Databases  
- Embeddings

Current implementation uses:

```text
knowledge_base.txt
```

Designed to be extensible to:

- PDFs  
- Documentation  
- Enterprise Knowledge Bases  
- Support Manuals

---

# Text Chunking

Large documents are split into semantically meaningful chunks using sentence-based chunking.

Approach:

- NLTK sentence tokenization
- Fixed-size chunk grouping
- Chunk overlap logic (extensible)

Purpose:

- Preserve context
- Improve retrieval relevance
- Fit embedding model input constraints

---

# Embedding Generation

Model Used:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Embedding size:

```text
384 dimensions
```

Purpose:

Convert text chunks into dense vector representations capturing semantic meaning.

---

# Vector Database

Vector store implemented using:

```text
FAISS (Facebook AI Similarity Search)
```

Features:

- Dense vector indexing
- Top-k similarity retrieval
- Persistent FAISS index saving/loading

Semantic search replaces keyword matching.

---

# Retrieval Pipeline

For each user question:

- Embed query
- Search FAISS index
- Retrieve top-k relevant chunks
- Pass retrieved context into prompt

Current retrieval:

```text
Top-k semantic search
```

This forms the grounding layer for generation.

---

# LLM Generation

Inference powered using:

```text
Groq API
Model: Llama 3.1 8B Instant
```

Prompt pattern:

- Inject retrieved context
- Restrict model to use only retrieved evidence
- Generate grounded answer

---

# System Architecture

The system is designed as a modular production-style AI pipeline.

## Architecture

```text
User
↓

Streamlit Chat App

↓

FastAPI API

↓

Retriever

↓

FAISS Vector Store

↓

Groq LLM

↓

Grounded Answer
```

---

# API Endpoints

## Health Endpoint

```http
GET /
```

Returns API status.

---

## Ask a Question

```http
POST /ask
```

Example request:

```json
{
 "question":"What is machine learning?"
}
```

Example response:

```json
{
 "answer":"Machine learning is...",
 "sources":[...]
}
```

---

# Interactive Chat Application

Streamlit frontend supports:

- Chat-style question answering
- Source chunk display
- Retrieval-grounded responses
- Public web deployment

---

# Deployment

## Backend
Deployed on:

```text
Render
```

Served through:

- FastAPI
- Docker Container

---

## Frontend
Deployed on:

```text
Streamlit Community Cloud
```

---

# Tech Stack

Python  
FastAPI  
Streamlit  
FAISS  
Sentence Transformers  
NumPy  
NLTK  
Groq API  
Docker  
Render

---

# Project Structure

```text
rag-knowledge-assistant/
│
├── api/
│   └── main.py
│
├── data/
│   └── knowledge_base.txt
│
├── src/
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── rag_pipeline.py
│   └── build_index.py
│
├── app.py
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Key Learnings

This project helped demonstrate:

- Retrieval Augmented Generation design
- Semantic search with embeddings
- Vector databases with FAISS
- Prompt grounding strategies
- Production API deployment
- Containerization with Docker
- Frontend + backend AI app architecture
- Dependency management across deployments

---

# Challenges Solved

During development, key engineering issues solved included:

- Python environment and dependency conflicts
- NLTK tokenizer resource issues
- FAISS persistence inside Docker
- API quota and model deprecation issues
- Deployment dependency resolution debugging
- Python version compatibility (FAISS vs Python 3.14)

---

# Limitations

Current version:

- Uses a small text-based knowledge base
- Basic top-k retrieval only
- No reranking layer
- No conversational memory
- No hybrid search

---

# Future Improvements

Planned upgrades:

- Multi-document / PDF ingestion
- Hybrid search (BM25 + Vector Search)
- Cross-encoder reranking
- Conversational memory
- Citation grounding
- LangChain / LangGraph integration
- Local open-source model inference
- Agentic RAG extensions

---

# Why This Project Matters

This project demonstrates production-style AI engineering skills relevant for:

- AI Engineer roles  
- ML Engineer roles  
- Applied LLM Engineering  
- Generative AI Systems

It combines:

- Retrieval  
- LLMs  
- APIs  
- Vector databases  
- Deployment  
- Full-stack AI application design

---

# Author

**Swati Yadav**  
Machine Learning / AI Engineering Enthusiast

GitHub: https://github.com/swati666