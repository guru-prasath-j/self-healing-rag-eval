"""
Prompt templates for every LLM call in the Self-Healing RAG pipeline.
Keeping prompts centralised makes them easy to tune without touching node logic.
"""

from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# 1. GENERATION prompt
#    Instructs the LLM to answer ONLY from the provided context.
# ---------------------------------------------------------------------------
GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a precise and factual question-answering assistant. "
            "Answer the question using ONLY the information in the provided context. "
            "If the context does not contain enough information to answer the question, "
            "reply with exactly: 'INSUFFICIENT_CONTEXT'.\n\n"
            "Context:\n{context}"
        ),
    ),
    ("human", "{question}"),
])

# ---------------------------------------------------------------------------
# 2. CRITIQUE prompt
#    The critic checks whether the answer is grounded in the retrieved docs.
# ---------------------------------------------------------------------------
CRITIQUE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a rigorous fact-checker for a RAG system. "
            "Your job is to decide whether the given answer is fully supported by the provided context. "
            "Rules:\n"
            "  - If every factual claim in the answer can be traced back to the context → GROUNDED\n"
            "  - If the answer contains any claim NOT found in the context, or contradicts it → NOT_GROUNDED\n"
            "  - If the answer says 'INSUFFICIENT_CONTEXT' and the context truly lacks the information → GROUNDED\n\n"
            "Respond in this exact JSON format:\n"
            '{{"verdict": "GROUNDED" | "NOT_GROUNDED", "reasoning": "<one sentence explanation>"}}'
        ),
    ),
    (
        "human",
        (
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer: {answer}"
        ),
    ),
])

# ---------------------------------------------------------------------------
# 3. REFORMULATION prompt
#    Rewrites the query to be more specific after a critique failure.
# ---------------------------------------------------------------------------
REFORMULATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a search-query optimiser for a retrieval-augmented generation system. "
            "A previous query failed to retrieve documents that could ground the answer. "
            "Rewrite the query to be more specific, use different keywords, "
            "and increase the chance of retrieving relevant documents. "
            "Return ONLY the rewritten query — no explanation, no prefix."
        ),
    ),
    (
        "human",
        (
            "Original question: {question}\n"
            "Critic feedback: {critique}\n\n"
            "Rewritten query:"
        ),
    ),
])

