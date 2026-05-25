"""
FAISS vector store helpers.
Handles building, persisting, and querying the document index.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Path where the FAISS index is saved on disk
_DEFAULT_INDEX_PATH = Path(__file__).parent.parent / "faiss_index"

# Embedding model (runs locally, no API key needed)
_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Return a cached HuggingFace embedding model."""
    return HuggingFaceEmbeddings(
        model_name=_EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(documents: List[Document], index_path: Path | str = _DEFAULT_INDEX_PATH) -> FAISS:
    """
    Create a FAISS index from a list of LangChain Documents and save it to disk.

    Args:
        documents:   List of pre-split Document objects.
        index_path:  Directory where the index will be persisted.

    Returns:
        The in-memory FAISS vector store.
    """
    index_path = Path(index_path)
    embeddings = _get_embeddings()
    store = FAISS.from_documents(documents, embeddings)
    index_path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(index_path))
    print(f"[VectorStore] Index saved to {index_path} ({len(documents)} chunks)")
    return store


def load_vectorstore(index_path: Path | str = _DEFAULT_INDEX_PATH) -> FAISS:
    """
    Load a previously built FAISS index from disk.

    Args:
        index_path: Directory containing the saved index.

    Returns:
        The loaded FAISS vector store.

    Raises:
        FileNotFoundError: If the index directory does not exist.
    """
    index_path = Path(index_path)
    if not index_path.exists():
        raise FileNotFoundError(
            f"No FAISS index found at {index_path}. "
            "Run `python ingest.py` first to build the index."
        )
    embeddings = _get_embeddings()
    store = FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)
    print(f"[VectorStore] Index loaded from {index_path}")
    return store


def retrieve(store: FAISS, query: str, k: int = 4) -> List[Document]:
    """
    Perform a similarity search against the vector store.

    Args:
        store: The FAISS vector store to search.
        query: The natural-language query string.
        k:     Number of top documents to return.

    Returns:
        List of the k most-similar Document objects.
    """
    return store.similarity_search(query, k=k)

