from src.chunking import load_knowledge_base, smart_chunk_text
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore

print("Building FAISS index...")

# Load knowledge base
text = load_knowledge_base("data/knowledge_base.txt")

# Chunk text
chunks = smart_chunk_text(text)

print(f"Chunks created: {len(chunks)}")

# Create embedding model
embedder = EmbeddingModel()

# Generate embeddings
embeddings = embedder.embed_texts(chunks)

# Create vector store
vector_store = VectorStore(dimension=embeddings.shape[1])

# Add embeddings
vector_store.add_embeddings(embeddings, chunks)

# Save index
vector_store.save()

print("FAISS index saved successfully.")