from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np


class EmbeddingModel:
    """
    Wrapper for sentence-transformers embedding model.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Convert list of texts into embeddings.
        """

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        return embeddings


# ----------------------------
# Test block
# ----------------------------
from pathlib import Path

if __name__ == "__main__":

    from src.chunking import load_knowledge_base, smart_chunk_text

    # Get project root
    project_root = Path(__file__).resolve().parent.parent

    file_path = project_root / "data" / "knowledge_base.txt"

    # Load knowledge base
    text = load_knowledge_base(file_path)

    # Create chunks
    chunks = smart_chunk_text(text)

    print(f"Total Chunks: {len(chunks)}")

    # Initialize embedding model
    embedder = EmbeddingModel()

    embeddings = embedder.embed_texts(chunks)

    print("\nEmbedding shape:")
    print(embeddings.shape)