"""
LangGraph workflow definition for the Self-Healing RAG pipeline.

Graph topology
--------------

    START
      │
      ▼
   retrieve ──────────────────────────────────────────┐
      │                                               │ (after reformulate)
      ▼                                               │
   generate                                           │
      │                                               │
      ▼                                               │
   critique ──[grounded]──────────────► respond ── END
      │
      └──[not grounded, retries left]──► reformulate ─┘
      │
      └──[not grounded, max retries]───► respond ─── END
"""

from __future__ import annotations

from functools import partial
from typing import Any, Dict

from langgraph.graph import END, START, StateGraph

from .nodes import (
    critique_node,
    generate_node,
    reformulate_node,
    respond_node,
    retrieve_node,
    route_after_critique,
)
from .state import RAGState
from .vectorstore import FAISS


def build_graph(store: FAISS, k: int = 4) -> Any:
    """
    Construct and compile the Self-Healing RAG LangGraph.

    Args:
        store: A loaded FAISS vector store.
        k:     Number of documents to retrieve per query.

    Returns:
        A compiled LangGraph runnable.
    """
    graph = StateGraph(RAGState)

    # ── Bind the vector store into the retrieve node ──
    _retrieve = partial(retrieve_node, store=store, k=k)

    # ── Register nodes ──
    graph.add_node("retrieve", _retrieve)
    graph.add_node("generate", generate_node)
    graph.add_node("critique", critique_node)
    graph.add_node("reformulate", reformulate_node)
    graph.add_node("respond", respond_node)

    # ── Static edges ──
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "critique")
    graph.add_edge("reformulate", "retrieve")
    graph.add_edge("respond", END)

    # ── Conditional edge after critique ──
    graph.add_conditional_edges(
        "critique",
        route_after_critique,
        {
            "respond": "respond",
            "reformulate": "reformulate",
        },
    )

    return graph.compile()


def run_pipeline(
    question: str,
    store: FAISS,
    k: int = 4,
    max_retries: int = 2,
) -> Dict[str, Any]:
    """
    End-to-end entry point: run the pipeline for a single question.

    Args:
        question:    The user's natural-language question.
        store:       Loaded FAISS vector store.
        k:           Retrieval top-k.
        max_retries: Maximum critique→reformulate cycles before giving up.

    Returns:
        The final RAGState dict (use state['final_answer'] for the result).
    """
    app = build_graph(store, k=k)

    initial_state: RAGState = {
        "question": question,
        "reformulated_question": None,
        "retrieved_docs": [],
        "answer": "",
        "critique": "",
        "is_grounded": False,
        "retry_count": 0,
        "max_retries": max_retries,
        "final_answer": "",
    }

    final_state = app.invoke(initial_state)
    return final_state

