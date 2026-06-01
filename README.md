# 🔄 Self-Healing RAG Pipeline

> A Retrieval-Augmented Generation system that **critiques its own answers and retries** — built with LangGraph, Groq (LLaMA 3), and FAISS.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-green)](https://github.com/langchain-ai/langgraph)
[![Groq](https://img.shields.io/badge/LLM-Groq%20LLaMA3-orange)](https://console.groq.com)
[![FAISS](https://img.shields.io/badge/VectorStore-FAISS-red)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 Table of Contents

- [What Is This?](#-what-is-this)
- [How It Works](#-how-it-works)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [Usage Examples](#-usage-examples)
- [Adding Your Own Documents](#-adding-your-own-documents)
- [Running Tests](#-running-tests)
- [Design Decisions](#-design-decisions)
- [Limitations & Future Work](#-limitations--future-work)
- [Eval Layer & CI Quality Gate](#-eval-layer--ci-quality-gate-new)

---

## 🤔 What Is This?

Standard RAG pipelines have a critical flaw: they retrieve documents, generate an answer, and return it — even if the LLM **ignored the context** and hallucinated.

This project fixes that with a **self-healing loop**:

1. **Retrieve** relevant chunks from a FAISS vector store.
2. **Generate** an answer using Groq's LLaMA 3 (fast, free tier).
3. **Critique** — a second LLM call asks: *"Is this answer actually supported by the retrieved docs, or did the model make things up?"*
4. If the answer **fails** the critique:
   - **Reformulate** the query to be more specific.
   - **Re-retrieve** with the new query.
   - Repeat up to `MAX_RETRIES` times.
5. If it still can't find a grounded answer → respond honestly: *"I don't have enough information."*

The entire workflow is modelled as a **stateful, cyclical graph** using [LangGraph](https://github.com/langchain-ai/langgraph) — not a simple linear chain.

---

## 🧪 Eval Layer & CI Quality Gate (NEW)

Self-healing fixes answers **at runtime**. This layer proves the system doesn't
**regress over time** — it grades answers automatically and blocks any change
that lowers quality.

- **Golden set** (`eval/golden_set.json`) — fixed Q/A pairs with reference
  answers, including one deliberately unanswerable question to test honest
  refusal.
- **Local metrics** (`eval/metrics.py`) — `answer_similarity`, `faithfulness`,
  and `context_relevance`, computed with the same MiniLM embedder the pipeline
  uses (no paid LLM judge needed), plus retries / latency / groundedness.
- **Regression gate** (`eval/evaluate.py` + `eval/thresholds.json`) — runs the
  golden set and **exits non-zero if any metric drops below threshold**.
- **CI enforcement** (`.github/workflows/eval.yml`) — runs the gate on every PR;
  a breach fails the build so the change can't merge.
- **Observability** (`src/tracing.py`) — optional Langfuse tracing of every LLM
  call + per-run scores; a no-op unless `LANGFUSE_*` keys are set.

```bash
python ingest.py                 # build the index
python eval/evaluate.py          # run the eval + gate (needs GROQ_API_KEY)
python eval/evaluate.py --no-gate  # report scores without failing
```

> New here for the eval layer? Read **[docs/EVAL_LAYER_EXPLAINED.md](docs/EVAL_LAYER_EXPLAINED.md)** —
> a plain-language walkthrough written for a non-AI audience, with interview Q&A.

---

## ⚙️ How It Works

### The Five Nodes

| Node | What It Does |
|---|---|
| `retrieve` | Embeds the current query → similarity search → returns top-k chunks from FAISS |
| `generate` | Sends chunks + question to Groq LLaMA 3 → produces a candidate answer |
| `critique` | Second LLM call: outputs `GROUNDED` or `NOT_GROUNDED` + reasoning in JSON |
| `reformulate` | Rewrites the query using the critic's feedback to improve next retrieval |
| `respond` | Finalises the answer; surfaces either the verified answer or a graceful fallback |

### The Routing Logic

After `critique`, a conditional edge decides what happens next:

```
is_grounded = True          → respond (done ✓)
is_grounded = False
  AND retry_count < max     → reformulate → retrieve → generate → critique (loop)
  AND retry_count >= max    → respond with "insufficient information" message
```

### The State Object

All nodes share a single `RAGState` TypedDict that flows through the graph:

```python
class RAGState(TypedDict):
    question:              str        # original user question
    reformulated_question: str | None # rewritten query (set after reformulate)
    retrieved_docs:        List[Document]
    answer:                str        # current LLM answer
    critique:              str        # critic's reasoning
    is_grounded:           bool       # critic's verdict
    retry_count:           int        # how many loops have run
    max_retries:           int        # ceiling (default 2)
    final_answer:          str        # surfaced to the user
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Self-Healing RAG Graph                    │
│                                                             │
│  START                                                       │
│    │                                                         │
│    ▼                                                         │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐             │
│  │ retrieve │────▶│ generate │────▶│ critique │             │
│  └──────────┘     └──────────┘     └────┬─────┘             │
│       ▲                                  │                   │
│       │                    ┌─────────────┴──────────────┐   │
│       │              GROUNDED?                           │   │
│       │              YES │                  NO           │   │
│       │                  ▼                  │            │   │
│       │           ┌─────────┐    retries    │            │   │
│       │           │ respond │◀──exhausted?──┘            │   │
│       │           └────┬────┘    NO │                    │   │
│       │                │           ▼                     │   │
│       │               END   ┌────────────┐               │   │
│       └─────────────────────│ reformulate│               │   │
│                             └────────────┘               │   │
└─────────────────────────────────────────────────────────────┘

Embeddings: sentence-transformers/all-MiniLM-L6-v2 (local, CPU)
Vector DB:  FAISS (local file)
LLM:        Groq — llama3-8b-8192 (generate, critique, reformulate)
```

---

## 📁 Project Structure

```
self-healing-rag/
│
├── src/
│   ├── __init__.py
│   ├── state.py          ← RAGState TypedDict definition
│   ├── prompts.py        ← All prompt templates (generation, critique, reformulation)
│   ├── vectorstore.py    ← FAISS build / load / retrieve helpers
│   ├── nodes.py          ← All 5 LangGraph node functions + routing function
│   └── graph.py          ← LangGraph StateGraph assembly + run_pipeline()
│
├── data/
│   └── sample_docs/      ← Drop your .txt, .md, or .pdf files here
│       ├── ai_overview.txt
│       ├── langgraph_intro.txt
│       └── rag_fundamentals.txt
│
├── tests/
│   └── test_pipeline.py  ← Pytest unit tests for all nodes and routing logic
│
├── faiss_index/          ← Auto-generated after running ingest.py (gitignored)
│
├── ingest.py             ← Index your documents into FAISS
├── main.py               ← Interactive Q&A demo / CLI entry point
├── requirements.txt
├── .env.example          ← Copy to .env and add your GROQ_API_KEY
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/usergithub02/self-healing-rag.git
cd self-healing-rag

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Set Up Your API Key

```bash
cp .env.example .env
```

Edit `.env` and add your [Groq API key](https://console.groq.com) (free):

```
GROQ_API_KEY=gsk_your_key_here
```

### 3. Ingest Documents

```bash
python ingest.py
```

This reads all `.txt`, `.md`, and `.pdf` files from `data/sample_docs/`, splits them into chunks, embeds them with `sentence-transformers/all-MiniLM-L6-v2`, and saves the FAISS index to `faiss_index/`.

> The embedding model downloads automatically on first run (~90 MB).

### 4. Ask Questions

```bash
python main.py
```

You'll get an interactive prompt:

```
╔══════════════════════════════════════════════════════════╗
║          Self-Healing RAG Pipeline  🔍 → 🤖 → 🔁         ║
╚══════════════════════════════════════════════════════════╝

You: What is RAG and why is FAISS used?

────────────────────────────────────────────────────────────
[Retrieve] Query: 'What is RAG and why is FAISS used?'
[Retrieve] Found 4 document(s).
[Generate] Answer: RAG (Retrieval-Augmented Generation) is a framework...
[Critique] Verdict: GROUNDED
[Critique] Reasoning: All claims are directly supported by the context.
[Router] Answer grounded → respond

🤖 Answer: RAG (Retrieval-Augmented Generation) is a framework that combines...
```

---

## ⚙️ Configuration

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Your Groq API key |
| `GROQ_MODEL` | `llama3-8b-8192` | Groq model to use |
| `RETRIEVAL_K` | `4` | Number of chunks to retrieve |
| `MAX_RETRIES` | `2` | Max critique-reformulate cycles |

**CLI flags** (override env at runtime):

```bash
python main.py --question "What is LangGraph?" --retries 3
python ingest.py --docs /path/to/my/docs/ --index ./my_index
```

---

## 💡 Usage Examples

### Single Question Mode

```bash
python main.py --question "How does the critique node decide if an answer is grounded?"
```

### Batch Questions via Python

```python
from dotenv import load_dotenv
load_dotenv()

from src.graph import run_pipeline
from src.vectorstore import load_vectorstore

store = load_vectorstore()

questions = [
    "What is FAISS?",
    "How does LangGraph handle cyclical workflows?",
    "What are the limitations of standard RAG?",
]

for q in questions:
    result = run_pipeline(q, store, max_retries=2)
    print(f"Q: {q}")
    print(f"A: {result['final_answer']}\n")
```

### Ingest a Custom PDF

```bash
python ingest.py --docs ./my_research_paper.pdf
python main.py
```

---

## 📄 Adding Your Own Documents

1. Place `.txt`, `.md`, or `.pdf` files into `data/sample_docs/` (or any directory).
2. Run `python ingest.py --docs <your-folder>`.
3. The FAISS index is rebuilt and saved to `faiss_index/`.
4. Run `python main.py` as usual.

Supported formats: **Plain text** (`.txt`), **Markdown** (`.md`), **PDF** (`.pdf`).

Chunk size and overlap are configurable in `ingest.py`:

```python
CHUNK_SIZE = 512    # characters per chunk
CHUNK_OVERLAP = 64  # overlap between adjacent chunks
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

The test suite covers:

- **Router logic** — all 4 branching scenarios (grounded / not-grounded + retries remaining / exhausted)
- **Respond node** — grounded answer pass-through, INSUFFICIENT_CONTEXT fallback, exhausted-retries message
- **Generate node** — mocked LLM call, answer key present in output
- **Critique node** — GROUNDED and NOT_GROUNDED JSON parsing
- **Reformulate node** — query rewrite + retry counter increment

Tests use `unittest.mock` so no API key or FAISS index is needed.

---

## 🧠 Design Decisions

### Why LangGraph instead of a simple loop?

LangGraph gives us a **first-class stateful graph** with:
- Typed state shared across all nodes (no hidden globals).
- Conditional routing as a declared edge, not tangled `if` statements.
- Built-in support for checkpointing (pause/resume) and streaming.
- Clear visualisation of the control flow.

A plain `while` loop would work for this toy case, but LangGraph makes the architecture explicit, testable, and trivially extensible (e.g., adding a human-in-the-loop node).

### Why Groq + LLaMA 3?

- **Speed**: Groq's hardware delivers ~800 tokens/second, making the extra critique call feel instant.
- **Free tier**: No credit card required to get started.
- **Open weights**: LLaMA 3 is a capable open model — the same prompts can be pointed at any LangChain-compatible LLM.

### Why sentence-transformers for embeddings?

- Runs **locally on CPU** — no additional API key or cost.
- `all-MiniLM-L6-v2` is small (80 MB), fast, and good enough for semantic search on most document corpora.
- Swap it in `vectorstore.py` for OpenAI `text-embedding-3-small` or any other model with one line change.

### Why FAISS?

- Zero infrastructure — just a directory of files, committed separately or regenerated from documents.
- Scales to millions of vectors on a laptop.
- Facebook's battle-tested similarity search library, wrapped nicely by LangChain.

### Prompt strategy for the critic

The critique prompt uses structured JSON output (`{"verdict": ..., "reasoning": ...}`) with a regex fallback to tolerate markdown code fences from the LLM. This is more robust than asking for a plain yes/no and avoids brittle string parsing.

---

## ⚠️ Limitations & Future Work

| Limitation | Possible Fix |
|---|---|
| Single-turn only; no conversation history | Add `chat_history` to `RAGState` and a memory node |
| FAISS doesn't support metadata filtering | Switch to ChromaDB or Pinecone for filter-by-date, filter-by-source, etc. |
| Embeddings downloaded fresh each session | Cache with `SENTENCE_TRANSFORMERS_HOME` env var |
| No streaming output | Use LangGraph's `.stream()` and yield tokens |
| Critic uses same model as generator | Use a larger model (e.g., `llama3-70b`) for the critic only |
| No web UI | Add a Gradio or Streamlit front-end |
| No citation in final answer | Surface `retrieved_docs[i].metadata['source']` in the respond node |

---

## 📜 License

MIT — see [LICENSE](LICENSE).

---

## 🙏 Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain) — document loading, splitting, embeddings
- [LangGraph](https://github.com/langchain-ai/langgraph) — stateful graph execution
- [Groq](https://groq.com) — fast LLaMA 3 inference
- [FAISS](https://github.com/facebookresearch/faiss) — vector similarity search
- [sentence-transformers](https://www.sbert.net/) — local embeddings

