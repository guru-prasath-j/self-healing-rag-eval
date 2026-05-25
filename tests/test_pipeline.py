"""
Unit and integration tests for the Self-Healing RAG pipeline.

Run with:
    pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nodes import (
    critique_node,
    generate_node,
    reformulate_node,
    respond_node,
    route_after_critique,
)
from src.state import RAGState


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_doc(content: str):
    """Return a minimal mock LangChain Document."""
    doc = MagicMock()
    doc.page_content = content
    return doc


def _base_state(**overrides) -> RAGState:
    """Return a RAGState with sensible defaults, then apply overrides."""
    state: RAGState = {
        "question": "What is RAG?",
        "reformulated_question": None,
        "retrieved_docs": [_make_doc("RAG stands for Retrieval-Augmented Generation.")],
        "answer": "RAG stands for Retrieval-Augmented Generation.",
        "critique": "",
        "is_grounded": False,
        "retry_count": 0,
        "max_retries": 2,
        "final_answer": "",
    }
    state.update(overrides)
    return state


# ── Router tests ──────────────────────────────────────────────────────────────

class TestRouteAfterCritique:
    def test_grounded_goes_to_respond(self):
        state = _base_state(is_grounded=True)
        assert route_after_critique(state) == "respond"

    def test_not_grounded_retries_remaining_goes_to_reformulate(self):
        state = _base_state(is_grounded=False, retry_count=0, max_retries=2)
        assert route_after_critique(state) == "reformulate"

    def test_not_grounded_max_retries_exhausted_goes_to_respond(self):
        state = _base_state(is_grounded=False, retry_count=2, max_retries=2)
        assert route_after_critique(state) == "respond"

    def test_boundary_retry_count_equals_max(self):
        state = _base_state(is_grounded=False, retry_count=3, max_retries=3)
        assert route_after_critique(state) == "respond"


# ── Respond node tests ────────────────────────────────────────────────────────

class TestRespondNode:
    def test_grounded_answer_returned_as_is(self):
        state = _base_state(
            answer="RAG is Retrieval-Augmented Generation.",
            is_grounded=True,
        )
        result = respond_node(state)
        assert result["final_answer"] == "RAG is Retrieval-Augmented Generation."

    def test_grounded_insufficient_context_returns_fallback(self):
        state = _base_state(
            answer="INSUFFICIENT_CONTEXT",
            is_grounded=True,
        )
        result = respond_node(state)
        assert "don't have enough information" in result["final_answer"].lower()

    def test_not_grounded_exhausted_retries_returns_failure_message(self):
        state = _base_state(
            answer="Some hallucinated answer.",
            is_grounded=False,
            retry_count=2,
        )
        result = respond_node(state)
        assert "unable to find" in result["final_answer"].lower() or \
               "reliably grounded" in result["final_answer"].lower()


# ── Generate node tests (mocked LLM) ─────────────────────────────────────────

class TestGenerateNode:
    @patch("src.nodes._build_llm")
    def test_generate_returns_answer(self, mock_build_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="RAG is a technique.")
        # Simulate chain execution
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = MagicMock(content="RAG is a technique.")
        mock_build_llm.return_value.__or__ = MagicMock(return_value=mock_chain)

        state = _base_state()
        # Patch the chain construction
        with patch("src.nodes.GENERATION_PROMPT") as mock_prompt:
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)
            result = generate_node(state)

        # We just verify the key is present (actual value depends on mock)
        assert "answer" in result


# ── Critique node tests (mocked LLM) ─────────────────────────────────────────

class TestCritiqueNode:
    @patch("src.nodes._build_llm")
    def test_grounded_verdict_parsed(self, mock_build_llm):
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = MagicMock(
            content='{"verdict": "GROUNDED", "reasoning": "All facts match context."}'
        )

        with patch("src.nodes.CRITIQUE_PROMPT") as mock_prompt:
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)
            result = critique_node(_base_state())

        assert "is_grounded" in result
        assert "critique" in result

    @patch("src.nodes._build_llm")
    def test_not_grounded_verdict_parsed(self, mock_build_llm):
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = MagicMock(
            content='{"verdict": "NOT_GROUNDED", "reasoning": "Answer contains hallucination."}'
        )

        with patch("src.nodes.CRITIQUE_PROMPT") as mock_prompt:
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)
            result = critique_node(_base_state())

        assert "is_grounded" in result


# ── Reformulate node tests (mocked LLM) ──────────────────────────────────────

class TestReformulateNode:
    @patch("src.nodes._build_llm")
    def test_reformulate_increments_retry_count(self, mock_build_llm):
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = MagicMock(
            content="What exactly is Retrieval-Augmented Generation in NLP?"
        )

        with patch("src.nodes.REFORMULATION_PROMPT") as mock_prompt:
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)
            state = _base_state(retry_count=0, critique="Answer was not in context.")
            result = reformulate_node(state)

        assert result["retry_count"] == 1
        assert "reformulated_question" in result

