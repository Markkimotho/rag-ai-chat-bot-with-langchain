"""Tests for the PDF ingestion pipeline."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from app.ingestion import _chunk_id, load_and_chunk_pdf, load_and_chunk_pdfs


def test_chunk_id_deterministic():
    id1 = _chunk_id("test.pdf", 0, 0)
    id2 = _chunk_id("test.pdf", 0, 0)
    assert id1 == id2


def test_chunk_id_differs_for_different_inputs():
    id1 = _chunk_id("test.pdf", 0, 0)
    id2 = _chunk_id("test.pdf", 0, 1)
    id3 = _chunk_id("other.pdf", 0, 0)
    assert id1 != id2
    assert id1 != id3


@patch("app.ingestion.get_settings")
@patch("app.ingestion.PyPDFLoader")
def test_load_and_chunk_pdf(mock_loader_cls, mock_settings):
    from langchain_core.documents import Document

    mock_settings.return_value = MagicMock(chunk_size=100, chunk_overlap=20)

    mock_loader = MagicMock()
    mock_loader.load.return_value = [
        Document(page_content="A" * 200, metadata={"page": 0}),
        Document(page_content="B" * 200, metadata={"page": 1}),
    ]
    mock_loader_cls.return_value = mock_loader

    chunks = load_and_chunk_pdf("/fake/test.pdf")

    assert len(chunks) > 0
    for chunk in chunks:
        assert "source" in chunk.metadata
        assert chunk.metadata["source"] == "test.pdf"
        assert "chunk_index" in chunk.metadata
        assert "id" in chunk.metadata
        assert "page" in chunk.metadata


@patch("app.ingestion.get_settings")
def test_load_and_chunk_pdfs_empty_dir(mock_settings, tmp_path):
    mock_settings.return_value = MagicMock(chunk_size=1000, chunk_overlap=200)
    result = load_and_chunk_pdfs(tmp_path)
    assert result == []
