"""Eval: the programming assistant returns runnable-looking code.

Paid (uses a real Ollama model). Run with:  pytest evals/ -m eval -v
Threshold: response is non-empty and contains a fenced code block.
"""

import json

import pytest
from fastapi.testclient import TestClient

from app.api import app

pytestmark = pytest.mark.eval

client = TestClient(app)


def _collect_stream(path: str, body: dict) -> tuple[str, list[str]]:
    """POST to an SSE endpoint and return (full_text, tool_names)."""
    text_parts: list[str] = []
    tools: list[str] = []
    with client.stream("POST", path, json=body) as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            frame = json.loads(line[5:].strip())
            if frame["type"] == "token":
                text_parts.append(frame["text"])
            elif frame["type"] == "tool":
                tools.append(frame["name"])
    return "".join(text_parts), tools


def test_code_assistant_returns_code_block(chat_model):
    text, _ = _collect_stream(
        "/api/code-chat/stream",
        {
            "message": (
                "Write a Python function fib(n) that returns the nth Fibonacci "
                "number. Show it in a code block."
            ),
            "thread_id": "eval-code",
            "model": chat_model,
        },
    )
    assert text.strip(), "assistant returned no text"
    assert "```" in text, f"expected a fenced code block, got: {text[:200]}"
    assert "def" in text, "expected a Python function definition"


def test_code_assistant_remembers_context(chat_model):
    thread = "eval-code-mem"
    _collect_stream(
        "/api/code-chat/stream",
        {"message": "Remember the number 42.", "thread_id": thread, "model": chat_model},
    )
    text, _ = _collect_stream(
        "/api/code-chat/stream",
        {
            "message": "What number did I ask you to remember?",
            "thread_id": thread,
            "model": chat_model,
        },
    )
    assert "42" in text, f"assistant lost conversation memory: {text[:200]}"
