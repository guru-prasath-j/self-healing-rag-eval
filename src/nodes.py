"""
LangGraph node implementations for the Self-Healing RAG pipeline.

Each function receives the current RAGState, performs one step of the workflow,
and returns a dict of state keys to update (LangGraph merges these in).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict

from langchain_core.documents import Document
from langchain_groq import ChatGroq

from .prompts import CRITIQUE_PROMPT, GENERATION_PROMPT, REFORMULATION_PROMPT
from .tracing import get_callback_handler
from .state import RAGState
from .vectorstore import FAISS, retrieve


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_docs(docs: list[Document]) -> str:
    """Join document page_content into a single context string."""
    return "\n\n---\n\n".join(
        f"[Doc {i+1}] {doc.page_content}" for i, doc in enumerate(docs)
    )


def _build_llm(model: str = "llama3-8b-8192", temperature: float = 0.0) -> ChatGroq:
    """
    Instantiate a ChatGroq LLM (picks up GROQ_API_KEY from env).

    If Langfuse is configured, a tracing callback is attached so every LLM call
    in the graph is recorded as a span for observability.
    """
    handler = get_callback_handler()
    callbacks = [handler] if handler else None
    return ChatGroq(model=model, temperature=temperature, callbacks=callbacks)


# ---------------------------------------------------------------------------
# Node 1 — RETRIEVE
# ---------------------------------------------------------------------------

def retrieve_node(state: RAGState, store: FAISS, k: int = 4) -> Dict[str, Any]:
    """
    Query the FAISS vector store with the current question
    (uses the reformulated question on retries).

    Updates:
        retrieved_docs: Top-k matched documents.
    """
    query = state.get("reformulated_question") or state["question"]
    print(f"\n[Retrieve] Query: '{query}'")
    docs = retrieve(store, query, k=k)
    print(f"[Retrieve] Found {len(docs)} document(s).")
    return {"retrieved_docs": docs}


# ---------------------------------------------------------------------------
# Node 2 — GENERATE
# ---------------------------------------------------------------------------

def generate_node(state: RAGState) -> Dict[str, Any]:
    """
    Ask the LLM to answer the question using ONLY the retrieved context.

    Updates:
        answer: The raw LLM response string.
    """
    question = state.get("reformulated_question") or state["question"]
    context = _format_docs(state["retrieved_docs"])
    llm = _build_llm()
    chain = GENERATION_PROMPT | llm
    response = chain.invoke({"context": context, "question": question})
    answer = response.content.strip()
    print(f"[Generate] Answer: {answer[:200]}{'...' if len(answer) > 200 else ''}")
    return {"answer": answer}


# ---------------------------------------------------------------------------
# Node 3 — CRITIQUE
# ---------------------------------------------------------------------------

def critique_node(state: RAGState) -> Dict[str, Any]:
    """
    Run the critic agent to decide whether the answer is grounded in the docs.

    Updates:
        critique:     Free-text reasoning from the critic.
        is_grounded:  Boolean verdict.
    """
    question = state.get("reformulated_question") or state["question"]
    context = _format_docs(state["retrieved_docs"])
    llm = _build_llm(temperature=0.0)
    chain = CRITIQUE_PROMPT | llm
    response = chain.invoke({
        "context": context,
        "question": question,
        "answer": state["answer"],
    })

    raw = response.content.strip()

    # Parse JSON verdict — be tolerant of markdown code fences
    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    verdict_data: dict = {}
    if json_match:
        try:
            verdict_data = json.loads(json_match.group())
        except json.JSONDecodeError:
            verdict_data = {}

    verdict = verdict_data.get("verdict", "NOT_GROUNDED").upper()
    reasoning = verdict_data.get("reasoning", raw)
    is_grounded = verdict == "GROUNDED"

    print(f"[Critique] Verdict: {verdict}")
    print(f"[Critique] Reasoning: {reasoning}")
    return {"critique": reasoning, "is_grounded": is_grounded}


# ---------------------------------------------------------------------------
# Node 4 — REFORMULATE
# ---------------------------------------------------------------------------

def reformulate_node(state: RAGState) -> Dict[str, Any]:
    """
    Rewrite the user's query to improve retrieval on the next cycle.

    Updates:
        reformulated_question: New search query.
        retry_count:           Incremented by 1.
    """
    llm = _build_llm(temperature=0.3)
    chain = REFORMULATION_PROMPT | llm
    response = chain.invoke({
        "question": state["question"],
        "critique": state["critique"],
    })
    new_query = response.content.strip()
    new_retry = state.get("retry_count", 0) + 1
    print(f"[Reformulate] New query (attempt {new_retry}): '{new_query}'")
    return {"reformulated_question": new_query, "retry_count": new_retry}


# ---------------------------------------------------------------------------
# Node 5 — RESPOND
# ---------------------------------------------------------------------------

def respond_node(state: RAGState) -> Dict[str, Any]:
    """
    Finalise the answer to surface to the user.

    If the answer is grounded → return it as-is.
    If it says INSUFFICIENT_CONTEXT → return a friendly fallback.
    Otherwise (max retries exhausted without grounding) → return an honest
    "I don't have enough information" message.

    Updates:
        final_answer: The string shown to the end user.
    """
    answer = state["answer"]

    if state.get("is_grounded", False):
        if "INSUFFICIENT_CONTEXT" in answer.upper():
            final = (
                "I don't have enough information in my knowledge base to answer "
                "that question accurately. Please try rephrasing or provide more context."
            )
        else:
            final = answer
    else:
        final = (
            "I was unable to find a reliably grounded answer to your question "
            f"after {state.get('retry_count', 0) + 1} attempt(s). "
            "The retrieved documents may not contain the information you need. "
            "Please try rephrasing your question or expanding the document corpus."
        )

    print(f"\n[Respond] Final answer: {final}")
    return {"final_answer": final}


# ---------------------------------------------------------------------------
# Conditional edge — after CRITIQUE
# ---------------------------------------------------------------------------

def route_after_critique(state: RAGState) -> str:
    """
    Routing function called by LangGraph after the critique node.

    Returns:
        'respond'    — if the answer is grounded (pass).
        'reformulate' — if not grounded AND retries remain.
        'respond'    — if not grounded AND max retries exhausted.
    """
    max_retries = state.get("max_retries", 2)
    retry_count = state.get("retry_count", 0)

    if state.get("is_grounded", False):
        print("[Router] Answer grounded → respond")
        return "respond"

    if retry_count < max_retries:
        print(f"[Router] Not grounded, retries left ({retry_count}/{max_retries}) → reformulate")
        return "reformulate"

    print(f"[Router] Not grounded, max retries reached ({retry_count}/{max_retries}) → respond")
    return "respond"

