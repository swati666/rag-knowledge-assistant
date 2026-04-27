from typing import List
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore


class Retriever:
    """
    Handles semantic retrieval from the vector store.
    """

    def __init__(self, vector_store: VectorStore, embedder: EmbeddingModel):
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """
        Retrieve top-k relevant text chunks for a query.
        """

        # Convert query to embedding
        query_embedding = self.embedder.embed_texts([query])

        # Search vector store
        results = self.vector_store.search(query_embedding, k)

        return results
if __name__ == "__main__":

    from src.chunking import load_knowledge_base, smart_chunk_text

    # Load knowledge base
    text = load_knowledge_base("data/knowledge_base.txt")

    # Create chunks
    chunks = smart_chunk_text(text)

    # Create embeddings
    embedder = EmbeddingModel()
    embeddings = embedder.embed_texts(chunks)

    # Create vector store
    vector_store = VectorStore(dimension=embeddings.shape[1])
    vector_store.add_embeddings(embeddings, chunks)

    # Create retriever
    retriever = Retriever(vector_store, embedder)

    query = "What is machine learning?"

    results = retriever.retrieve(query, k=2)

    print("\nRetrieved Context:\n")

    for r in results:
        print("-", r)