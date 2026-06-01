"""
Eval harness for the Self-Healing RAG pipeline.

What it does
------------
1. Loads the golden Q/A set (eval/golden_set.json).
2. Runs the real pipeline (retrieve -> generate -> critique -> [reformulate]) on
   each question, timing it and recording how many self-healing retries it took.
3. Scores each answer locally (semantic similarity to the reference, faithfulness
   to the retrieved context, retrieval relevance).
4. Optionally logs every run as a Langfuse trace with the scores attached.
5. Aggregates the scores and compares them to eval/thresholds.json.
6. Writes eval/results/latest.json and prints a report.
7. Exits non-zero if ANY threshold is breached  ->  this is what makes the
   GitHub Actions workflow fail a PR that regresses quality.

Usage
-----
    python eval/evaluate.py                 # run full eval + gate
    python eval/evaluate.py --no-gate       # report only, never fail
    python eval/evaluate.py --max-retries 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Make `src` importable when run from repo root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv()

from eval import metrics  # noqa: E402
from src.graph import run_pipeline  # noqa: E402
from src.vectorstore import load_vectorstore  # noqa: E402

try:
    from src.tracing import get_langfuse, score_run  # noqa: E402
except Exception:  # tracing is optional
    def get_langfuse():
        return None

    def score_run(*_args, **_kwargs):
        return None


GOLDEN = ROOT / "eval" / "golden_set.json"
THRESHOLDS = ROOT / "eval" / "thresholds.json"
RESULTS = ROOT / "eval" / "results" / "latest.json"


def _refused(answer: str) -> bool:
    """Heuristic: did the pipeline honestly decline instead of inventing an answer?"""
    a = answer.lower()
    needles = [
        "don't have enough information",
        "do not have enough information",
        "unable to find",
        "insufficient",
        "cannot answer",
        "not contain",
    ]
    return any(n in a for n in needles)


def run_eval(max_retries: int, gate: bool) -> int:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    thresholds = json.loads(THRESHOLDS.read_text(encoding="utf-8"))
    items = golden["items"]

    if not os.getenv("GROQ_API_KEY"):
        print(
            "[eval] GROQ_API_KEY not set — skipping eval (neutral). "
            "Set it as a CI secret to enable the regression gate."
        )
        return 0

    store = load_vectorstore()
    langfuse = get_langfuse()
    per_item = []

    print(f"[eval] Running {len(items)} golden questions...\n")
    for item in items:
        q = item["question"]
        t0 = time.perf_counter()
        state = run_pipeline(q, store, max_retries=max_retries)
        latency_ms = (time.perf_counter() - t0) * 1000

        answer = state.get("final_answer", "")
        chunks = [d.page_content for d in state.get("retrieved_docs", [])]

        if item.get("expect_refusal"):
            passed = _refused(answer)
            sim = 1.0 if passed else 0.0
            faith = 1.0 if passed else 0.0
            ctx_rel = metrics.context_relevance(q, chunks) if chunks else 0.0
        else:
            sim = metrics.answer_similarity(answer, item["reference_answer"])
            faith = metrics.faithfulness(answer, chunks)
            ctx_rel = metrics.context_relevance(q, chunks)
            passed = sim >= thresholds["per_item_pass_similarity"]

        row = {
            "id": item["id"],
            "question": q,
            "answer": answer,
            "answer_similarity": round(sim, 4),
            "faithfulness": round(faith, 4),
            "context_relevance": round(ctx_rel, 4),
            "is_grounded": bool(state.get("is_grounded", False)),
            "retry_count": int(state.get("retry_count", 0)),
            "latency_ms": round(latency_ms, 1),
            "passed": bool(passed),
        }
        per_item.append(row)
        flag = "PASS" if passed else "FAIL"
        print(f"  [{flag}] {item['id']:<18} sim={sim:.2f} faith={faith:.2f} "
              f"retries={row['retry_count']} {row['latency_ms']:.0f}ms")

        if langfuse:
            score_run(langfuse, item, row)

    summary = metrics.aggregate(per_item)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps({"summary": summary, "items": per_item}, indent=2),
        encoding="utf-8",
    )

    print("\n[eval] Summary")
    for k, v in summary.items():
        print(f"    {k:<20} {v}")

    # ---- Gate ----
    breaches = []
    checks = {
        "answer_similarity": ("min", thresholds["min_answer_similarity"]),
        "faithfulness": ("min", thresholds["min_faithfulness"]),
        "context_relevance": ("min", thresholds["min_context_relevance"]),
        "pass_rate": ("min", thresholds["min_pass_rate"]),
        "avg_latency_ms": ("max", thresholds["max_avg_latency_ms"]),
    }
    for metric, (kind, limit) in checks.items():
        value = summary[metric]
        if kind == "min" and value < limit:
            breaches.append(f"{metric}={value} < min {limit}")
        if kind == "max" and value > limit:
            breaches.append(f"{metric}={value} > max {limit}")

    if breaches:
        print("\n[eval] THRESHOLD BREACHES:")
        for b in breaches:
            print(f"    ✗ {b}")
        if gate:
            print("\n[eval] FAIL — quality regressed below thresholds.")
            return 1
        print("\n[eval] (--no-gate) reporting only, not failing.")
        return 0

    print("\n[eval] PASS — all thresholds met.")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Run the Self-Healing RAG eval gate.")
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--no-gate", action="store_true",
                   help="Report metrics but never exit non-zero.")
    args = p.parse_args()
    sys.exit(run_eval(max_retries=args.max_retries, gate=not args.no_gate))


if __name__ == "__main__":
    main()
