"""Gate tests for the FastAPI layer (app/api.py).

Every engine call is mocked — no Ollama, no network, no ChromaDB writes.
Tests assert status codes, response shapes, validation, and SSE framing/order.
"""

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


def _sse_frames(text: str) -> list[dict]:
    """Parse an SSE response body into a list of JSON event dicts."""
    frames = []
    for block in text.strip().split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data:"):
                frames.append(json.loads(line[5:].strip()))
    return frames


# ── Health / models ─────────────────────────────────────────────────────────


def test_health_online():
    with patch("app.api.httpx.get") as mock_get:
        mock_get.return_value.json.return_value = {
            "models": [{"name": "qwen2.5:7b"}, {"name": "nomic-embed-text"}]
        }
        mock_get.return_value.raise_for_status.return_value = None
        body = client.get("/api/health").json()
    assert body["online"] is True
    assert body["model_count"] == 1  # embed model excluded


def test_health_offline():
    with patch("app.api.httpx.get", side_effect=RuntimeError("no ollama")):
        body = client.get("/api/health").json()
    assert body == {"online": False, "model_count": 0}


def test_models_lists_supported_and_installed():
    with patch("app.api.get_chat_models", return_value=["qwen2.5:7b", "mistral"]), patch(
        "app.api.get_installed_models", return_value={"qwen2.5:7b"}
    ):
        body = client.get("/api/models").json()
    assert body["supported"] == ["qwen2.5:7b", "mistral"]
    assert body["installed"] == ["qwen2.5:7b"]


def test_pull_model_delegates():
    with patch("app.api.pull_model", return_value=(True, "done")) as m:
        body = client.post("/api/models/pull", json={"model": "mistral"}).json()
    assert body == {"ok": True, "message": "done"}
    m.assert_called_once_with("mistral")


def test_pull_model_rejects_empty():
    r = client.post("/api/models/pull", json={"model": ""})
    assert r.status_code == 422


# ── Knowledge base ────────────────────────────────────────────────────────────


def test_kb_count():
    with patch("app.api.get_doc_count", return_value=42):
        assert client.get("/api/kb/count").json() == {"count": 42}


def test_kb_scrape_adds_chunks():
    with patch("app.api.search_and_scrape", return_value=["c1", "c2"]), patch(
        "app.api.ingest_documents", return_value=2
    ):
        body = client.post(
            "/api/kb/scrape", json={"topic": "python", "num_results": 2}
        ).json()
    assert body["chunks_added"] == 2


def test_kb_scrape_no_content():
    with patch("app.api.search_and_scrape", return_value=[]):
        body = client.post("/api/kb/scrape", json={"topic": "zzz"}).json()
    assert body["chunks_added"] == 0
    assert "No content" in body["detail"]


def test_kb_scrape_failure_returns_502():
    with patch("app.api.search_and_scrape", side_effect=RuntimeError("net")):
        r = client.post("/api/kb/scrape", json={"topic": "x"})
    assert r.status_code == 502


def test_kb_upload_pdf():
    with patch("app.api.load_and_chunk_pdf", return_value=["a", "b", "c"]) as mock_load, patch(
        "app.api.ingest_documents", return_value=3
    ):
        files = {"files": ("notes.pdf", b"%PDF-1.4 fake", "application/pdf")}
        body = client.post("/api/kb/upload", files=files).json()
    assert body["chunks_added"] == 3
    assert "notes.pdf" in body["detail"]
    # The original filename (not the tempfile path) is preserved as the source.
    assert mock_load.call_args.kwargs["source_name"] == "notes.pdf"


def test_kb_upload_skips_non_pdf():
    files = {"files": ("notes.txt", b"hello", "text/plain")}
    body = client.post("/api/kb/upload", files=files).json()
    assert body["chunks_added"] == 0
    assert "skipped" in body["detail"]


def test_kb_clear():
    with patch("app.api.clear_vectorstore") as m:
        assert client.delete("/api/kb").json() == {"cleared": True}
    m.assert_called_once()


def test_kb_sources_lists_files():
    fake = [
        {"source": "algo.pdf", "chunks": 12, "type": "pdf"},
        {"source": "https://en.wikipedia.org/wiki/Binary_search", "chunks": 40, "type": "web"},
    ]
    with patch("app.api.list_sources", return_value=fake):
        body = client.get("/api/kb/sources").json()
    assert body["sources"][0]["source"] == "algo.pdf"
    assert body["sources"][1]["type"] == "web"
    assert body["sources"][1]["chunks"] == 40


def test_kb_sources_empty():
    with patch("app.api.list_sources", return_value=[]):
        assert client.get("/api/kb/sources").json() == {"sources": []}


# ── Chat streaming (RAG) ────────────────────────────────────────────────────


def test_chat_stream_langchain_order():
    def fake_stream(**kwargs):
        yield ("token", "Hello")
        yield ("token", " there")
        yield ("sources", [{"source": "a.pdf", "page": 1}])

    with patch("app.api.lcel_chain.stream", side_effect=fake_stream):
        r = client.post("/api/chat/stream", json={"question": "hi", "mode": "langchain"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    frames = _sse_frames(r.text)
    assert frames[0] == {"type": "token", "text": "Hello"}
    assert {"type": "sources", "sources": [{"source": "a.pdf", "page": 1}]} in frames
    assert frames[-1] == {"type": "done"}


def test_chat_stream_uses_langgraph_when_selected():
    def fake_stream(**kwargs):
        yield ("token", "G")

    with patch("app.api.langgraph_module.stream", side_effect=fake_stream) as m:
        r = client.post("/api/chat/stream", json={"question": "hi", "mode": "langgraph"})
    assert r.status_code == 200
    m.assert_called_once()


def test_chat_stream_validation_error():
    r = client.post("/api/chat/stream", json={"question": ""})
    assert r.status_code == 422


def test_chat_clear():
    with patch("app.api.clear_history") as m:
        assert client.delete("/api/chat/sess1").json() == {"cleared": True}
    m.assert_called_once_with("sess1")


# ── Code chat + agent streaming ─────────────────────────────────────────────


def test_code_chat_stream():
    def fake(**kwargs):
        yield ("token", "code")

    with patch("app.api.code_assistant.stream_code_assistant", side_effect=fake):
        r = client.post("/api/code-chat/stream", json={"message": "write code"})
    frames = _sse_frames(r.text)
    assert {"type": "token", "text": "code"} in frames
    assert frames[-1] == {"type": "done"}


def test_agent_stream_with_tool_event():
    def fake(**kwargs):
        yield ("tool", "make_quiz")
        yield ("token", "Question 1")

    with patch("app.api.quiz_agent.stream_agent", side_effect=fake):
        r = client.post("/api/agent/stream", json={"message": "quiz me"})
    frames = _sse_frames(r.text)
    assert {"type": "tool", "name": "make_quiz"} in frames
    assert {"type": "token", "text": "Question 1"} in frames


def test_stream_surfaces_engine_error():
    def boom(**kwargs):
        raise RuntimeError("engine exploded")
        yield  # pragma: no cover

    with patch("app.api.code_assistant.stream_code_assistant", side_effect=boom):
        r = client.post("/api/code-chat/stream", json={"message": "x"})
    frames = _sse_frames(r.text)
    assert any(f["type"] == "error" for f in frames)
    assert frames[-1] == {"type": "done"}


# ── Quiz ───────────────────────────────────────────────────────────────────


def test_quiz_generate():
    fake_qs = [{"type": "mcq", "question": "Q", "correct": "A"}]
    with patch("app.api.generate_questions", return_value=fake_qs) as m:
        body = client.post(
            "/api/quiz/generate",
            json={"topic": "trees", "question_type": "mcq", "n": 1},
        ).json()
    assert body == fake_qs
    assert m.call_args.kwargs["topic"] == "trees"


def test_quiz_generate_clamps_n():
    r = client.post("/api/quiz/generate", json={"topic": "x", "n": 999})
    assert r.status_code == 422  # n must be <= 20


def test_quiz_validate():
    result = {
        "score": 100,
        "is_correct": True,
        "feedback": "Correct!",
        "key_missed": [],
        "hint": "",
    }
    with patch("app.api.validate_answer", return_value=result):
        body = client.post(
            "/api/quiz/validate",
            json={
                "question": "Q?",
                "correct_answer": "A",
                "student_answer": "A",
                "question_type": "mcq",
            },
        ).json()
    assert body["is_correct"] is True


def test_quiz_explain():
    with patch("app.api.explain_concept", return_value="Recursion is..."):
        body = client.post(
            "/api/quiz/explain", json={"concept": "recursion"}
        ).json()
    assert body["explanation"].startswith("Recursion")


# ── Metrics ──────────────────────────────────────────────────────────────────


def test_metrics_endpoint_exposes_custom_metrics():
    # Drive one streamed request so the token/latency metrics have samples.
    def fake(**kwargs):
        yield ("token", "x")

    with patch("app.api.code_assistant.stream_code_assistant", side_effect=fake):
        client.post("/api/code-chat/stream", json={"message": "hi"})

    body = client.get("/metrics").text
    assert "preppal_stream_tokens_total" in body
    assert "preppal_llm_request_duration_seconds" in body
    assert "preppal_llm_requests_total" in body
    # Default HTTP metrics from the instrumentator are present too.
    assert "http_request" in body


def test_metrics_count_tokens_for_surface():
    from app.metrics import STREAM_TOKENS

    before = STREAM_TOKENS.labels("code")._value.get()

    def fake(**kwargs):
        yield ("token", "a")
        yield ("token", "b")

    with patch("app.api.code_assistant.stream_code_assistant", side_effect=fake):
        client.post("/api/code-chat/stream", json={"message": "hi"})

    after = STREAM_TOKENS.labels("code")._value.get()
    assert after - before == 2


# ── SPA serving (only when a frontend build exists) ─────────────────────────

import app.api as api_module  # noqa: E402


@pytest.mark.skipif(not api_module._DIST.is_dir(), reason="no frontend build")
def test_spa_served_at_root():
    r = client.get("/")
    assert r.status_code == 200
    assert 'id="root"' in r.text


@pytest.mark.skipif(not api_module._DIST.is_dir(), reason="no frontend build")
def test_spa_fallback_for_client_routes():
    # Deep links handled by react-router must fall back to index.html.
    for path in ("/quiz", "/agent", "/code", "/chat"):
        r = client.get(path)
        assert r.status_code == 200
        assert 'id="root"' in r.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
