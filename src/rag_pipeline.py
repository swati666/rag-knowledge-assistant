from src.retriever import Retriever
from groq import Groq


class RAGPipeline:
    """
    End-to-end Retrieval Augmented Generation pipeline.
    """

    def __init__(self, retriever: Retriever, api_key: str):
        self.retriever = retriever
        self.client = Groq(api_key=api_key)

    def generate_answer(self, question: str, k: int = 3):

        # Step 1: Retrieve context
        context_chunks = self.retriever.retrieve(question, k)

        context = "\n\n".join(context_chunks)

        # Step 2: Build prompt
        prompt = f"""
Use the context below to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""

        # Step 3: Call LLM
        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant that answers using the provided context."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        answer = response.choices[0].message.content

        return {
            "answer": answer,
            "sources": context_chunks
        }


# ------------------------------------------------
# Test block
# ------------------------------------------------
if __name__ == "__main__":

    import os

    from src.chunking import load_knowledge_base, smart_chunk_text
    from src.embeddings import EmbeddingModel
    from src.vector_store import VectorStore
    from src.retriever import Retriever

    # Load knowledge base
    text = load_knowledge_base("data/knowledge_base.txt")

    # Chunk text
    chunks = smart_chunk_text(text)

    print(f"Chunks created: {len(chunks)}")

    # Create embeddings
    embedder = EmbeddingModel()
    embeddings = embedder.embed_texts(chunks)

    # Build vector store
    vector_store = VectorStore(dimension=embeddings.shape[1])
    vector_store.add_embeddings(embeddings, chunks)

    # Create retriever
    retriever = Retriever(vector_store, embedder)

    # Create pipeline
    rag = RAGPipeline(
        retriever=retriever,
        api_key=os.getenv("GROQ_API_KEY")   # IMPORTANT FIX
    )

    question = "What is machine learning?"

    answer = rag.generate_answer(question)

    print("\nAnswer:\n")
    print(answer)