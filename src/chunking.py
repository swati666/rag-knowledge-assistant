from typing import List
import nltk

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

def load_knowledge_base(file_path: str) -> str:
    """
    Load knowledge base text file.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def smart_chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 2
) -> List[str]:
    """
    Sentence-aware chunking for better semantic retrieval.

    Args:
        text: full knowledge base
        chunk_size: approximate max characters per chunk
        overlap: number of sentences to overlap

    Returns:
        list of chunks
    """

    sentences = nltk.sent_tokenize(text)

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:

        if current_length + len(sentence) > chunk_size:

            chunks.append(" ".join(current_chunk))

            # add overlap sentences
            current_chunk = current_chunk[-overlap:]
            current_length = sum(len(s) for s in current_chunk)

        current_chunk.append(sentence)
        current_length += len(sentence)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

from pathlib import Path
if __name__ == "__main__":

    # Get project root
    project_root = Path(__file__).resolve().parent.parent

    file_path = project_root / "data" / "knowledge_base.txt"


    text = load_knowledge_base(file_path)

    chunks = smart_chunk_text(text)

    print(f"\nTotal Chunks: {len(chunks)}\n")

    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:\n{chunk}\n")