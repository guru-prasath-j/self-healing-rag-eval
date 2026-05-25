"""
Self-Healing RAG Pipeline — Interactive Demo

Usage
-----
    python main.py                    # interactive Q&A loop
    python main.py --question "..."   # answer a single question and exit
    python main.py --retries 3        # allow up to 3 critique cycles

Prerequisites
-------------
1. Copy .env.example to .env and fill in GROQ_API_KEY
2. python ingest.py   ← builds the FAISS index from data/sample_docs/
3. python main.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env (GROQ_API_KEY, etc.)
load_dotenv()

# Add project root to path so `src` imports resolve
sys.path.insert(0, str(Path(__file__).parent))

from src.graph import run_pipeline
from src.vectorstore import load_vectorstore

BANNER = """
╔══════════════════════════════════════════════════════════╗
║          Self-Healing RAG Pipeline  🔍 → 🤖 → 🔁         ║
║  Retrieves · Generates · Critiques · Reformulates         ║
╚══════════════════════════════════════════════════════════╝
"""


def answer_question(question: str, store, max_retries: int) -> str:
    """Run the pipeline and return the final answer."""
    print(f"\n{'─'*60}")
    print(f"Question: {question}")
    print('─'*60)
    result = run_pipeline(question, store, max_retries=max_retries)
    return result["final_answer"]


def interactive_loop(store, max_retries: int) -> None:
    """Run an interactive Q&A session until the user quits."""
    print(BANNER)
    print("Type your question and press Enter. Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        final_answer = answer_question(question, store, max_retries)
        print(f"\n🤖 Answer: {final_answer}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Self-Healing RAG interactive demo."
    )
    parser.add_argument(
        "--question", "-q",
        type=str,
        default=None,
        help="A single question to answer (non-interactive mode).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Maximum number of retrieve-critique cycles (default: 2).",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path(__file__).parent / "faiss_index",
        help="Path to the FAISS index directory.",
    )
    args = parser.parse_args()

    # Load vector store
    try:
        store = load_vectorstore(args.index)
    except FileNotFoundError as exc:
        print(f"[Error] {exc}")
        sys.exit(1)

    if args.question:
        # Single-question mode
        answer = answer_question(args.question, store, args.retries)
        print(f"\n🤖 Answer: {answer}\n")
    else:
        # Interactive loop
        interactive_loop(store, args.retries)


if __name__ == "__main__":
    main()

