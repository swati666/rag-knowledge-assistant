import faiss
import numpy as np
import pickle
from typing import List


class VectorStore:
    """
    FAISS-based vector store for semantic search.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.text_chunks: List[str] = []

    # -----------------------------
    # Add embeddings to FAISS
    # -----------------------------
    def add_embeddings(self, embeddings: np.ndarray, texts: List[str]):
        """
        Add embeddings and corresponding text chunks to the index.
        """

        self.index.add(embeddings)
        self.text_chunks.extend(texts)

    # -----------------------------
    # Semantic search
    # -----------------------------
    def search(self, query_embedding: np.ndarray, k: int = 3):
        """
        Search top-k most similar chunks.
        """

        distances, indices = self.index.search(query_embedding, k)

        results = []

        for idx in indices[0]:
            if idx < len(self.text_chunks):
                results.append(self.text_chunks[idx])

        return results

    # -----------------------------
    # Save FAISS index to disk
    # -----------------------------
    def save(self, index_path="faiss_index.bin", text_path="chunks.pkl"):
        """
        Save FAISS index and chunks to disk.
        """

        faiss.write_index(self.index, index_path)

        with open(text_path, "wb") as f:
            pickle.dump(self.text_chunks, f)

    # -----------------------------
    # Load FAISS index from disk
    # -----------------------------
    def load(self, index_path="faiss_index.bin", text_path="chunks.pkl"):
        """
        Load FAISS index and chunks from disk.
        """

        self.index = faiss.read_index(index_path)

        with open(text_path, "rb") as f:
            self.text_chunks = pickle.load(f)


# ------------------------------------------------
# Test block
# ------------------------------------------------
if __name__ == "__main__":

    from src.chunking import load_knowledge_base, smart_chunk_text
    from src.embeddings import EmbeddingModel

    print("Loading knowledge base...")

    # Load knowledge base
    text = load_knowledge_base("data/knowledge_base.txt")

    # Create chunks
    chunks = smart_chunk_text(text)

    print(f"Total Chunks: {len(chunks)}")

    # Create embeddings
    embedder = EmbeddingModel()
    embeddings = embedder.embed_texts(chunks)

    # Create FAISS store
    vector_store = VectorStore(dimension=embeddings.shape[1])

    vector_store.add_embeddings(embeddings, chunks)

    print("Vector index created.")

    # Save index
    vector_store.save()

    print("Index saved to disk.")

    # Test query
    query = "What is Retrieval Augmented Generation?"

    query_embedding = embedder.embed_texts([query])

    results = vector_store.search(query_embedding, k=2)

    print("\nTop results:")

    for r in results:
        print("-", r)