"""Tests for the retriever module."""

from unittest.mock import MagicMock, patch


@patch("app.retriever.get_vectorstore")
@patch("app.retriever.get_settings")
def test_get_retriever_default_top_k(mock_settings, mock_vs):
    mock_settings.return_value = MagicMock(top_k=5)
    mock_vectorstore = MagicMock()
    mock_vs.return_value = mock_vectorstore

    from app.retriever import get_retriever

    retriever = get_retriever()

    mock_vectorstore.as_retriever.assert_called_once_with(
        search_type="similarity",
        search_kwargs={"k": 5},
    )


@patch("app.retriever.get_vectorstore")
@patch("app.retriever.get_settings")
def test_get_retriever_custom_top_k(mock_settings, mock_vs):
    mock_settings.return_value = MagicMock(top_k=5)
    mock_vectorstore = MagicMock()
    mock_vs.return_value = mock_vectorstore

    from app.retriever import get_retriever

    retriever = get_retriever(top_k=3)

    mock_vectorstore.as_retriever.assert_called_once_with(
        search_type="similarity",
        search_kwargs={"k": 3},
    )
