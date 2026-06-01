"""
Evaluation metrics for the Self-Healing RAG pipeline.

Design goal: every metric here runs LOCALLY with the same MiniLM embedding model
the pipeline already uses — no extra paid LLM-judge calls required to get a
signal. (RAGAS / DeepEval LLM-judged metrics are available as an optional,
heavier add-on documented in docs/EVAL_LAYER_EXPLAINED.md.)

All similarity metrics return a value in [0, 1] where higher is better.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np


@lru_cache(maxsize=1)
def _embedder():
    """Lazily load the same local embedding model the pipeline uses."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def _embed(texts: List[str]) -> np.ndarray:
    """Embed and L2-normalise a list of texts -> (n, dim) matrix."""
    vecs = _embedder().encode(texts, normalize_embeddings=True)
    return np.asarray(vecs, dtype=np.float32)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two already-normalised vectors, clamped to [0,1]."""
    return float(max(0.0, min(1.0, float(np.dot(a, b)))))


def answer_similarity(generated: str, reference: str) -> float:
    """
    Semantic closeness between the generated answer and the reference answer.
    Proxy for "answer correctness" without an LLM judge.
    """
    if not generated.strip() or not reference.strip():
        return 0.0
    g, r = _embed([generated, reference])
    return _cos(g, r)


def faithfulness(generated: str, context_chunks: List[str]) -> float:
    """
    How well the answer is supported by the retrieved context.
    We embed the answer and take its best alignment with any retrieved chunk
    (max), which approximates "is this answer actually grounded in the docs".
    """
    if not generated.strip() or not context_chunks:
        return 0.0
    ans = _embed([generated])[0]
    ctx = _embed(context_chunks)
    return float(max(_cos(ans, c) for c in ctx))


def context_relevance(question: str, context_chunks: List[str]) -> float:
    """
    How relevant the retrieved chunks are to the question (mean similarity).
    A retrieval-quality signal independent of the generator.
    """
    if not question.strip() or not context_chunks:
        return 0.0
    q = _embed([question])[0]
    ctx = _embed(context_chunks)
    return float(np.mean([_cos(q, c) for c in ctx]))


def aggregate(per_item: List[dict]) -> dict:
    """
    Collapse per-item scores into the summary metrics the CI gate checks.

    Expects each item dict to contain:
        answer_similarity, faithfulness, context_relevance,
        is_grounded (bool), retry_count (int), latency_ms (float),
        passed (bool)
    """
    n = max(len(per_item), 1)

    def mean(key: str) -> float:
        return round(sum(i[key] for i in per_item) / n, 4)

    latencies = sorted(i["latency_ms"] for i in per_item)
    p95_idx = max(0, int(round(0.95 * (len(latencies) - 1))))

    return {
        "n_items": len(per_item),
        "answer_similarity": mean("answer_similarity"),
        "faithfulness": mean("faithfulness"),
        "context_relevance": mean("context_relevance"),
        "groundedness_rate": round(
            sum(1 for i in per_item if i["is_grounded"]) / n, 4
        ),
        "pass_rate": round(sum(1 for i in per_item if i["passed"]) / n, 4),
        "avg_retries": round(sum(i["retry_count"] for i in per_item) / n, 4),
        "avg_latency_ms": round(sum(latencies) / n, 1),
        "p95_latency_ms": round(latencies[p95_idx], 1) if latencies else 0.0,
    }
