"""
Document ingestion script for the Self-Healing RAG pipeline.

Usage
-----
    python ingest.py                          # ingest default sample docs
    python ingest.py --docs path/to/docs/     # ingest a custom folder
    python ingest.py --docs my_file.pdf       # ingest a single PDF

Supported formats: .txt, .md, .pdf (via PyPDFLoader)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

# Project-local import (works when running from repo root)
sys.path.insert(0, str(Path(__file__).parent))
from src.vectorstore import build_vectorstore

# ── Defaults ──────────────────────────────────────────────────────────────
DEFAULT_DOCS_DIR = Path(__file__).parent / "data" / "sample_docs"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def load_documents(source: Path) -> list[Document]:
    """Load all supported documents from a file or directory."""
    documents: list[Document] = []

    if source.is_file():
        if source.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(source))
        else:
            loader = TextLoader(str(source), encoding="utf-8")
        documents = loader.load()

    elif source.is_dir():
        # Text / Markdown files
        txt_loader = DirectoryLoader(
            str(source),
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            silent_errors=True,
        )
        md_loader = DirectoryLoader(
            str(source),
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            silent_errors=True,
        )
        documents.extend(txt_loader.load())
        documents.extend(md_loader.load())

        # PDF files
        for pdf_path in source.rglob("*.pdf"):
            pdf_loader = PyPDFLoader(str(pdf_path))
            documents.extend(pdf_loader.load())

    else:
        raise ValueError(f"Source path does not exist: {source}")

    print(f"[Ingest] Loaded {len(documents)} raw document(s) from {source}")
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Chunk documents into overlapping pieces for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"[Ingest] Split into {len(chunks)} chunk(s) "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest documents into the FAISS vector store."
    )
    parser.add_argument(
        "--docs",
        type=Path,
        default=DEFAULT_DOCS_DIR,
        help="Path to a file or directory of documents to ingest.",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path(__file__).parent / "faiss_index",
        help="Directory to save the FAISS index.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Self-Healing RAG — Document Ingestion")
    print("=" * 60)

    docs = load_documents(args.docs)
    if not docs:
        print("[Ingest] No documents found. Please add files to data/sample_docs/")
        sys.exit(1)

    chunks = split_documents(docs)
    build_vectorstore(chunks, index_path=args.index)

    print("\n[Ingest] Done! You can now run: python main.py")


if __name__ == "__main__":
    main()

