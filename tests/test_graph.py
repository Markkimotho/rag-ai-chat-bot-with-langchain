"""Tests for the LangGraph RAG graph."""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage


def test_graph_state_typing():
    """Verify the GraphState TypedDict has the expected keys."""
    from app.graph import GraphState

    # TypedDict should have these annotations
    assert "messages" in GraphState.__annotations__
    assert "documents" in GraphState.__annotations__
    assert "question" in GraphState.__annotations__
    assert "answer" in GraphState.__annotations__


@patch("app.graph._get_llm")
def test_contextualize_with_no_history(mock_llm):
    from app.graph import contextualize

    state = {
        "messages": [HumanMessage(content="What is RAG?")],
        "question": "What is RAG?",
        "documents": [],
        "answer": "",
    }

    result = contextualize(state, config={"configurable": {}})
    assert result["question"] == "What is RAG?"
    mock_llm.assert_not_called()


@patch("app.graph._get_llm")
def test_contextualize_with_history(mock_llm):
    from app.graph import contextualize

    mock_response = MagicMock()
    mock_response.content = "What is Retrieval-Augmented Generation?"
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = mock_response
    mock_llm.return_value = mock_llm_instance

    state = {
        "messages": [
            HumanMessage(content="What is RAG?"),
            AIMessage(content="RAG stands for..."),
            HumanMessage(content="Tell me more"),
        ],
        "question": "Tell me more",
        "documents": [],
        "answer": "",
    }

    result = contextualize(state, config={"configurable": {}})
    assert result["question"] == "What is Retrieval-Augmented Generation?"


@patch("app.graph.get_retriever")
def test_retrieve_calls_retriever(mock_get_retriever):
    from app.graph import retrieve

    mock_retriever = MagicMock()
    mock_docs = [
        Document(page_content="RAG is...", metadata={"source": "test.pdf", "page": 1}),
    ]
    mock_retriever.invoke.return_value = mock_docs
    mock_get_retriever.return_value = mock_retriever

    state = {
        "messages": [],
        "question": "What is RAG?",
        "documents": [],
        "answer": "",
    }

    result = retrieve(state, config={"configurable": {}})
    assert len(result["documents"]) == 1
    assert result["documents"][0].metadata["source"] == "test.pdf"


@patch("app.graph._get_llm")
def test_generate_produces_answer(mock_llm):
    from app.graph import generate

    mock_response = MagicMock()
    mock_response.content = "RAG is a technique that..."
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = mock_response
    mock_llm.return_value = mock_llm_instance

    state = {
        "messages": [HumanMessage(content="What is RAG?")],
        "question": "What is RAG?",
        "documents": [
            Document(page_content="RAG overview", metadata={"source": "rag.pdf", "page": 1}),
        ],
        "answer": "",
    }

    result = generate(state, config={"configurable": {}})
    assert result["answer"] == "RAG is a technique that..."
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
