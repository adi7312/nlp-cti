from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from typing import List, Any


def chunk_text(docs: List, strategy: str = "sliding_window", embedding_model: Any = None) -> List:
    """Apply chunking strategy to loaded documents.

    Args:
        docs: List of documents to chunk.
        strategy: Chunking strategy (fixed, sliding_window, sentence, semantic).
        embedding_model: Embedding model for semantic chunking (required for "semantic" strategy).

    Returns:
        List of chunked documents.

    Raises:
        ValueError: If an unknown chunking strategy is specified, or if semantic strategy
                   is used without providing an embedding_model.
    """
    if strategy == "fixed":
        splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
    elif strategy == "sliding_window":
        splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=256)
    elif strategy == "sentence":
        splitter = RecursiveCharacterTextSplitter(
            separators=[".\n", "?\n", "!\n", ". ", "? ", "! "],
            chunk_size=512,
            chunk_overlap=0,
        )
    elif strategy == "semantic":
        if embedding_model is None:
            raise ValueError("embedding_model is required for semantic chunking strategy")
        splitter = SemanticChunker(embedding_model)
    else:
        raise ValueError("Unknown chunking strategy")

    return splitter.split_documents(docs)
