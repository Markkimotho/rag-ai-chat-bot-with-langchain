"""Gate tests for quiz generation and answer validation logic.

All tests are deterministic and require no LLM calls (mocked).
Run time: < 2s.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from app.quiz import _build_context, _extract_json, generate_questions, validate_answer


# ── _extract_json ─────────────────────────────────────────────────────────────


def test_extract_json_plain_array():
    raw = '[{"type":"mcq","question":"Q?","correct":"A"}]'
    result = _extract_json(raw)
    assert isinstance(result, list)
    assert result[0]["correct"] == "A"


def test_extract_json_strips_markdown_fences():
    raw = '```json\n[{"type":"true_false","correct":true}]\n```'
    result = _extract_json(raw)
    assert isinstance(result, list)
    assert result[0]["correct"] is True


def test_extract_json_embedded_in_text():
    raw = 'Here are the questions:\n[{"type":"mcq","q":"What?"}]\nDone.'
    result = _extract_json(raw)
    assert isinstance(result, list)


def test_extract_json_single_object():
    raw = '{"type":"short_answer","question":"Explain X."}'
    result = _extract_json(raw)
    assert isinstance(result, dict)
    assert result["type"] == "short_answer"


def test_extract_json_raises_on_garbage():
    with pytest.raises(ValueError):
        _extract_json("This is not JSON at all.")


# ── _build_context ────────────────────────────────────────────────────────────


def test_build_context_includes_source_and_page():
    from langchain_core.documents import Document

    docs = [
        Document(page_content="Content A", metadata={"source": "file.pdf", "page": 1}),
        Document(page_content="Content B", metadata={"source": "other.pdf", "page": 3}),
    ]
    ctx = _build_context(docs)
    assert "file.pdf" in ctx
    assert "Content A" in ctx
    assert "other.pdf" in ctx
    assert "Content B" in ctx


# ── validate_answer — deterministic paths ────────────────────────────────────


def test_validate_mcq_correct():
    result = validate_answer(
        question="What is 2+2?",
        correct_answer="A",
        student_answer="A",
        question_type="mcq",
    )
    assert result["is_correct"] is True
    assert result["score"] == 100
    assert result["key_missed"] == []


def test_validate_mcq_incorrect():
    result = validate_answer(
        question="What is 2+2?",
        correct_answer="A",
        student_answer="B",
        question_type="mcq",
    )
    assert result["is_correct"] is False
    assert result["score"] == 0
    assert "A" in result["key_missed"]


def test_validate_mcq_case_insensitive():
    result = validate_answer(
        question="Pick one",
        correct_answer="C",
        student_answer="c",
        question_type="mcq",
    )
    assert result["is_correct"] is True


def test_validate_true_false_correct():
    result = validate_answer(
        question="Python is interpreted.",
        correct_answer="TRUE",
        student_answer="true",
        question_type="true_false",
    )
    assert result["is_correct"] is True


def test_validate_true_false_incorrect():
    result = validate_answer(
        question="Python is compiled.",
        correct_answer="FALSE",
        student_answer="TRUE",
        question_type="true_false",
    )
    assert result["is_correct"] is False


def test_validate_answer_result_keys():
    result = validate_answer(
        question="Q?", correct_answer="A", student_answer="A", question_type="mcq"
    )
    for key in ("score", "is_correct", "feedback", "key_missed", "hint"):
        assert key in result, f"Missing key: {key}"


# ── generate_questions — mocked LLM ──────────────────────────────────────────


@patch("app.quiz.get_retriever")
@patch("app.quiz._get_llm")
def test_generate_questions_returns_list(mock_llm_fn, mock_retriever_fn):
    from langchain_core.documents import Document

    _mock_docs = [
        Document(page_content="Binary search runs in O(log n).", metadata={"source": "algo.pdf", "page": 1})
    ]
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = _mock_docs
    mock_retriever_fn.return_value = mock_retriever

    _mock_llm = MagicMock()
    _mock_llm.invoke.return_value = MagicMock(
        content=json.dumps([
            {
                "type": "mcq",
                "question": "What is the time complexity of binary search?",
                "options": {"A": "O(n)", "B": "O(log n)", "C": "O(n log n)", "D": "O(1)"},
                "correct": "B",
                "explanation": "Binary search halves the search space each step.",
                "source": "algo.pdf",
                "difficulty": "medium",
            }
        ])
    )
    mock_llm_fn.return_value = _mock_llm

    questions = generate_questions(
        topic="binary search",
        question_type="mcq",
        n=1,
    )
    assert isinstance(questions, list)
    assert len(questions) == 1
    assert questions[0]["correct"] == "B"


@patch("app.quiz.get_retriever")
def test_generate_questions_empty_kb(mock_retriever_fn):
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []
    mock_retriever_fn.return_value = mock_retriever

    questions = generate_questions(topic="something obscure", n=5)
    assert questions == []


@patch("app.quiz.get_retriever")
@patch("app.quiz._get_llm")
def test_generate_questions_handles_llm_error(mock_llm_fn, mock_retriever_fn):
    from langchain_core.documents import Document

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        Document(page_content="Some content", metadata={"source": "x.pdf", "page": 0})
    ]
    mock_retriever_fn.return_value = mock_retriever

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = RuntimeError("model offline")
    mock_llm_fn.return_value = mock_llm

    questions = generate_questions(topic="topic", n=3)
    assert questions == []


# ── validate_answer — short_answer via mocked LLM ────────────────────────────


@patch("app.quiz._get_llm")
def test_validate_short_answer_via_llm(mock_llm_fn):
    _grade = {"score": 85, "is_correct": True, "feedback": "Good.", "key_missed": [], "hint": ""}
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content=json.dumps(_grade))
    mock_llm_fn.return_value = mock_llm

    result = validate_answer(
        question="Explain recursion.",
        correct_answer="A function that calls itself.",
        student_answer="Recursion is when a function calls itself to solve sub-problems.",
        question_type="short_answer",
    )
    assert result["score"] == 85
    assert result["is_correct"] is True


@patch("app.quiz._get_llm")
def test_validate_short_answer_llm_error_returns_zero(mock_llm_fn):
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = RuntimeError("offline")
    mock_llm_fn.return_value = mock_llm

    result = validate_answer(
        question="Q?",
        correct_answer="A.",
        student_answer="B.",
        question_type="short_answer",
    )
    assert result["score"] == 0
    assert result["is_correct"] is False
    assert "feedback" in result
