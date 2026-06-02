# Self-Healing RAG Evaluation Pipeline

> Production-style Retrieval-Augmented Generation with automatic quality evaluation, self-healing critic loop, and full observability.

Built with **LangChain / LangGraph** · **Groq (Llama 3)** · **FAISS** · **Langfuse**

## Architecture

```
Query → FAISS Retrieval → LangGraph RAG Node → Answer
                              ↓
                      Critic Node (LLM evaluates answer quality)
                              ↓
                   [Pass] → Return answer
                   [Fail] → Re-retrieve with refined query → Retry
```

## Key Features

- **LangGraph orchestration** — stateful multi-node pipeline with conditional edges for retry logic
- **Self-healing loop** — a critic LLM scores each answer; low-confidence answers trigger automatic re-retrieval
- **Groq inference** — ultra-fast Llama 3 70B for both generation and critique
- **FAISS vector store** — local cosine-similarity retrieval with sentence-transformers embeddings
- **Langfuse observability** — every trace, span, score, and token count logged for production monitoring

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | LangChain + LangGraph |
| LLM | Groq (Llama 3 70B) |
| Embeddings | sentence-transformers |
| Vector Store | FAISS |
| Observability | Langfuse |
| Language | Python 3.11 |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Add GROQ_API_KEY and LANGFUSE_* keys
python main.py
```

## Environment Variables

```
GROQ_API_KEY=your_groq_key
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com
```

## Why This Matters

Standard RAG pipelines have no feedback loop — bad retrievals produce bad answers silently. This pipeline adds an **evaluation-driven retry mechanism** that catches low-quality answers before returning them to the user. The Langfuse integration gives you full production visibility into retrieval quality, answer scores, and latency.

## Related Projects

- [brainsync-ai-app](https://github.com/guru-prasath-j/brainsync-ai-app) — Full-stack AI Study Companion using RAG + Flutter + FastAPI
- [pocketmind](https://github.com/guru-prasath-j/pocketmind) — On-device AI with local LLM inference on Flutter
