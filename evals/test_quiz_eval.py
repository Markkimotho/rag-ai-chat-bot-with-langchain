"""Eval: quiz generation produces well-formed, gradable questions.

Paid (uses a real Ollama model). Run with:  pytest evals/ -m eval -v
Threshold: at least 60% of requested questions come back structurally valid.
"""

import hashlib

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.api import app
from app.vectorstore import ingest_documents

pytestmark = pytest.mark.eval

client = TestClient(app)

_TOPIC = "binary search"
_MATERIAL = (
    "Binary search is an efficient algorithm for finding a target value within a "
    "sorted array. It works by repeatedly dividing the search interval in half. "
    "If the target is less than the middle element, the search continues in the "
    "lower half; otherwise it continues in the upper half. Binary search runs in "
    "O(log n) time because the search space halves each step. It requires the "
    "input to be sorted in advance. A linear search, by contrast, runs in O(n) "
    "time and does not require sorted input."
)


@pytest.fixture(scope="module", autouse=True)
def seed_kb():
    chunk = Document(
        page_content=_MATERIAL,
        metadata={
            "source": "eval_binary_search.txt",
            "page": 0,
            "chunk_index": 0,
            "id": hashlib.sha256(b"eval-binary-search").hexdigest(),
        },
    )
    ingest_documents([chunk])


def _valid_mcq(q: dict) -> bool:
    opts = q.get("options") or {}
    return (
        bool(q.get("question"))
        and len(opts) >= 2
        and q.get("correct") in opts
    )


def test_mcq_generation_quality():
    n = 5
    resp = client.post(
        "/api/quiz/generate",
        json={
            "topic": _TOPIC,
            "question_type": "mcq",
            "n": n,
            "difficulty": "medium",
            "exam_type": "general exam",
        },
    )
    assert resp.status_code == 200
    questions = resp.json()
    assert isinstance(questions, list) and len(questions) > 0

    valid = [q for q in questions if _valid_mcq(q)]
    ratio = len(valid) / len(questions)
    assert ratio >= 0.6, f"Only {ratio:.0%} of MCQs were well-formed: {questions}"


def test_validation_grades_correct_answer():
    questions = client.post(
        "/api/quiz/generate",
        json={"topic": _TOPIC, "question_type": "mcq", "n": 3},
    ).json()
    mcq = next((q for q in questions if _valid_mcq(q)), None)
    assert mcq is not None, "no well-formed MCQ to grade"

    result = client.post(
        "/api/quiz/validate",
        json={
            "question": mcq["question"],
            "correct_answer": mcq["correct"],
            "student_answer": mcq["correct"],
            "question_type": "mcq",
        },
    ).json()
    assert result["is_correct"] is True
    assert result["score"] == 100
