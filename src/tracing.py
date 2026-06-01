"""
Optional Langfuse tracing/observability for the Self-Healing RAG pipeline.

Everything here is a NO-OP unless the Langfuse environment variables are set
(LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, optionally LANGFUSE_HOST). That keeps
the core pipeline runnable with zero observability dependencies, while letting
you flip on full tracing + scoring just by adding keys.

Two things it provides:
  1. get_callback_handler() -> a LangChain callback handler that records every
     LLM call (generate / critique / reformulate) as a span in Langfuse.
  2. get_langfuse() + score_run() -> used by the eval harness to attach metric
     scores to each traced run, so you can see quality trends per version.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Optional


def _enabled() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


@lru_cache(maxsize=1)
def get_callback_handler() -> Optional[Any]:
    """Return a Langfuse LangChain callback handler, or None if not configured."""
    if not _enabled():
        return None
    try:
        from langfuse.callback import CallbackHandler

        return CallbackHandler()  # reads keys from env
    except Exception as exc:  # pragma: no cover
        print(f"[tracing] Langfuse handler unavailable: {exc}")
        return None


@lru_cache(maxsize=1)
def get_langfuse() -> Optional[Any]:
    """Return a Langfuse client for manual scoring, or None if not configured."""
    if not _enabled():
        return None
    try:
        from langfuse import Langfuse

        return Langfuse()
    except Exception as exc:  # pragma: no cover
        print(f"[tracing] Langfuse client unavailable: {exc}")
        return None


def score_run(client: Any, golden_item: dict, row: dict) -> None:
    """
    Push eval scores for one golden question to Langfuse as a standalone trace.
    Safe to call with a real client; the eval harness guards for None.
    """
    try:
        trace = client.trace(
            name="rag-eval",
            input=golden_item["question"],
            output=row["answer"],
            metadata={"id": golden_item["id"], "retries": row["retry_count"]},
        )
        for metric in ("answer_similarity", "faithfulness", "context_relevance"):
            trace.score(name=metric, value=row[metric])
        trace.score(name="passed", value=1.0 if row["passed"] else 0.0)
    except Exception as exc:  # pragma: no cover
        print(f"[tracing] score_run failed: {exc}")
