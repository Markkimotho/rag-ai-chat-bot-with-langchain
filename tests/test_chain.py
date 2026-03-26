"""Tests for the LCEL RAG chain."""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

# Import early so @patch decorators can resolve the module
import app.chain  # noqa: F401


@patch("app.chain.add_messages")
@patch("app.chain.get_chat_history", return_value=[])
@patch("app.chain._build_chain")
def test_invoke_returns_answer_and_sources(mock_build, mock_history, mock_add):
    mock_docs = [
        Document(page_content="Test content", metadata={"source": "test.pdf", "page": 1}),
    ]

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = {
        "answer": "Test answer",
        "context": mock_docs,
    }
    mock_build.return_value = mock_chain

    from app.chain import invoke

    result = invoke("What is this?", session_id="test-chain-1")

    assert "answer" in result
    assert result["answer"] == "Test answer"
    assert "sources" in result
    assert len(result["sources"]) == 1
    assert result["sources"][0]["source"] == "test.pdf"
    assert result["sources"][0]["page"] == 1


@patch("app.chain.add_messages")
@patch("app.chain.get_chat_history", return_value=[])
@patch("app.chain._build_chain")
def test_invoke_handles_empty_context(mock_build, mock_history, mock_add):
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = {
        "answer": "I don't know.",
        "context": [],
    }
    mock_build.return_value = mock_chain

    from app.chain import invoke

    result = invoke("Unknown question?", session_id="test-chain-2")

    assert result["answer"] == "I don't know."
    assert result["sources"] == []
