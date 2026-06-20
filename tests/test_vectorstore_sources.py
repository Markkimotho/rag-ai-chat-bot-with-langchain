"""Gate tests for vectorstore.list_sources() — aggregation logic, mocked Chroma."""

from unittest.mock import MagicMock, patch

from app.vectorstore import list_sources


def _fake_vs(metadatas: list[dict]):
    collection = MagicMock()
    collection.count.return_value = len(metadatas)
    collection.get.return_value = {"metadatas": metadatas}
    vs = MagicMock()
    vs._collection = collection
    return vs


def test_list_sources_aggregates_chunk_counts():
    metas = [
        {"source": "algo.pdf", "type": "pdf"},
        {"source": "algo.pdf", "type": "pdf"},
        {"source": "web.com", "type": "web"},
    ]
    with patch("app.vectorstore.get_vectorstore", return_value=_fake_vs(metas)):
        result = list_sources()
    by_source = {r["source"]: r for r in result}
    assert by_source["algo.pdf"]["chunks"] == 2
    assert by_source["web.com"]["chunks"] == 1
    assert by_source["web.com"]["type"] == "web"


def test_list_sources_empty_kb():
    with patch("app.vectorstore.get_vectorstore", return_value=_fake_vs([])):
        assert list_sources() == []


def test_list_sources_sorted_case_insensitive():
    metas = [{"source": "Zebra.pdf"}, {"source": "alpha.pdf"}]
    with patch("app.vectorstore.get_vectorstore", return_value=_fake_vs(metas)):
        result = list_sources()
    assert [r["source"] for r in result] == ["alpha.pdf", "Zebra.pdf"]


def test_list_sources_handles_exception():
    with patch("app.vectorstore.get_vectorstore", side_effect=RuntimeError("boom")):
        assert list_sources() == []
