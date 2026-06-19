"""Gate tests for quiz agent tools.

Tools are thin wrappers — tests confirm correct delegation and error handling.
No LLM or network calls are made (all mocked).
"""

import json
from unittest.mock import MagicMock, patch

import pytest


# ── search_and_add_to_kb ──────────────────────────────────────────────────────


@patch("app.quiz_agent.ingest_documents")
@patch("app.quiz_agent.search_and_scrape")
def test_search_and_add_to_kb_success(mock_scrape, mock_ingest):
    from langchain_core.documents import Document
    from app.quiz_agent import search_and_add_to_kb

    mock_scrape.return_value = [
        Document(page_content="chunk", metadata={"source": "http://x.com", "page": 0, "id": "a", "chunk_index": 0})
    ]
    mock_ingest.return_value = 1

    result = search_and_add_to_kb.invoke({"topic": "Python decorators", "num_results": 2})
    assert "Added 1" in result
    assert "Python decorators" in result


@patch("app.quiz_agent.search_and_scrape")
def test_search_and_add_to_kb_no_content(mock_scrape):
    from app.quiz_agent import search_and_add_to_kb

    mock_scrape.return_value = []
    result = search_and_add_to_kb.invoke({"topic": "nonexistent topic xyz"})
    assert "No content" in result or "not found" in result.lower()


@patch("app.quiz_agent.search_and_scrape")
def test_search_and_add_to_kb_handles_exception(mock_scrape):
    from app.quiz_agent import search_and_add_to_kb

    mock_scrape.side_effect = RuntimeError("network error")
    result = search_and_add_to_kb.invoke({"topic": "anything"})
    assert "failed" in result.lower() or "error" in result.lower()


# ── make_quiz ─────────────────────────────────────────────────────────────────


@patch("app.quiz_agent.generate_questions")
def test_make_quiz_returns_json(mock_gen):
    from app.quiz_agent import make_quiz

    mock_gen.return_value = [
        {
            "type": "mcq",
            "question": "What is O(log n)?",
            "options": {"A": "Linear", "B": "Logarithmic", "C": "Quadratic", "D": "Constant"},
            "correct": "B",
            "explanation": "...",
            "source": "algo.pdf",
            "difficulty": "medium",
        }
    ]
    result = make_quiz.invoke({"topic": "binary search", "n": 1})
    parsed = json.loads(result)
    assert isinstance(parsed, list)
    assert parsed[0]["correct"] == "B"


@patch("app.quiz_agent.generate_questions")
def test_make_quiz_no_questions(mock_gen):
    from app.quiz_agent import make_quiz

    mock_gen.return_value = []
    result = make_quiz.invoke({"topic": "nothing here"})
    assert "Could not generate" in result or "no questions" in result.lower() or "knowledge base" in result.lower()


def test_make_quiz_clamps_n():
    from app.quiz_agent import make_quiz

    with patch("app.quiz_agent.generate_questions") as mock_gen:
        mock_gen.return_value = []
        make_quiz.invoke({"topic": "x", "n": 999})
        call_kwargs = mock_gen.call_args.kwargs
        called_n = call_kwargs.get("n", mock_gen.call_args.args[2] if len(mock_gen.call_args.args) > 2 else 20)
        assert called_n <= 20


# ── check_answer ──────────────────────────────────────────────────────────────


@patch("app.quiz_agent.validate_answer")
def test_check_answer_correct(mock_validate):
    from app.quiz_agent import check_answer

    mock_validate.return_value = {
        "score": 100, "is_correct": True, "feedback": "Correct!", "key_missed": [], "hint": "",
    }
    result = check_answer.invoke({
        "question": "Q?",
        "correct_answer": "A",
        "student_answer": "A",
        "question_type": "mcq",
    })
    parsed = json.loads(result)
    assert parsed["is_correct"] is True
    assert parsed["score"] == 100


# ── explain ───────────────────────────────────────────────────────────────────


@patch("app.quiz_agent.explain_concept")
def test_explain_delegates(mock_explain):
    from app.quiz_agent import explain

    mock_explain.return_value = "Recursion is when a function calls itself."
    result = explain.invoke({"concept": "recursion"})
    assert "Recursion" in result
    mock_explain.assert_called_once_with("recursion")


# ── list_topics ───────────────────────────────────────────────────────────────


@patch("app.quiz_agent.get_doc_count")
def test_list_topics_empty_kb(mock_count):
    from app.quiz_agent import list_topics

    mock_count.return_value = 0
    result = list_topics.invoke({})
    assert "empty" in result.lower() or "upload" in result.lower()


@patch("app.quiz_agent.get_retriever")
@patch("app.quiz_agent.get_doc_count")
def test_list_topics_with_docs(mock_count, mock_retriever_fn):
    from langchain_core.documents import Document
    from app.quiz_agent import list_topics

    mock_count.return_value = 42
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        Document(page_content="stuff", metadata={"source": "algo.pdf", "page": 0}),
    ]
    mock_retriever_fn.return_value = mock_retriever

    result = list_topics.invoke({})
    assert "42" in result
    assert "algo.pdf" in result
