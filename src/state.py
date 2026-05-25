"""
State schema for the Self-Healing RAG pipeline.
Defines the shared state object passed between all LangGraph nodes.
"""

from typing import TypedDict, List, Optional
from langchain_core.documents import Document


class RAGState(TypedDict):
    """
    The complete state of the RAG pipeline at any point in the graph.

    Attributes:
        question:              The original user question.
        reformulated_question: The rewritten query after a critique failure.
        retrieved_docs:        Documents fetched from the FAISS vector store.
        answer:                The LLM-generated answer for the current cycle.
        critique:              Free-text reasoning produced by the critic agent.
        is_grounded:           True if the critic approved the answer.
        retry_count:           How many retrieve-generate-critique cycles have run.
        max_retries:           Maximum allowed retries before giving up (default 2).
        final_answer:          The answer surfaced to the user at the END node.
    """
    question: str
    reformulated_question: Optional[str]
    retrieved_docs: List[Document]
    answer: str
    critique: str
    is_grounded: bool
    retry_count: int
    max_retries: int
    final_answer: str

