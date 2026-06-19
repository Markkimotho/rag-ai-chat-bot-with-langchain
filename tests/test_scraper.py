"""Gate tests for the web scraper module.

All network calls are mocked — no real HTTP requests in this suite.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.scraper import _chunk_id, _extract_text, scrape_url, search_and_scrape


# ── _chunk_id ─────────────────────────────────────────────────────────────────


def test_chunk_id_deterministic():
    assert _chunk_id("https://example.com", 0) == _chunk_id("https://example.com", 0)


def test_chunk_id_differs_on_index():
    assert _chunk_id("https://example.com", 0) != _chunk_id("https://example.com", 1)


def test_chunk_id_differs_on_url():
    assert _chunk_id("https://a.com", 0) != _chunk_id("https://b.com", 0)


# ── _extract_text ─────────────────────────────────────────────────────────────


def test_extract_text_basic():
    html = """
    <html><body>
      <nav>Skip this</nav>
      <main>
        <p>This is real content that is more than forty characters long definitely.</p>
        <p>Another paragraph with enough text to pass the threshold here.</p>
      </main>
    </body></html>
    """
    text = _extract_text(html)
    assert "real content" in text
    assert "Another paragraph" in text
    assert "Skip this" not in text


def test_extract_text_removes_script_tags():
    html = """
    <html><body>
      <script>var x = 1;</script>
      <main><p>Good content with more than forty characters here.</p></main>
    </body></html>
    """
    text = _extract_text(html)
    assert "var x" not in text
    assert "Good content" in text


def test_extract_text_empty_on_no_content():
    html = "<html><body><nav>only nav</nav></body></html>"
    text = _extract_text(html)
    assert text == ""


def test_extract_text_falls_back_to_body():
    html = """
    <html><body>
      <p>Body-level paragraph with enough text to pass the minimum length threshold.</p>
    </body></html>
    """
    text = _extract_text(html)
    assert "Body-level paragraph" in text


# ── scrape_url ────────────────────────────────────────────────────────────────


@patch("app.scraper.httpx.get")
@patch("app.scraper.get_settings")
def test_scrape_url_returns_documents(mock_settings, mock_get):
    mock_settings.return_value = MagicMock(chunk_size=1000, chunk_overlap=200)

    _html = (
        "<html><body><main>"
        + "<p>" + "This is substantial content. " * 30 + "</p>"
        + "</main></body></html>"
    )
    mock_resp = MagicMock()
    mock_resp.text = _html
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    docs = scrape_url("https://example.com/page")
    assert len(docs) > 0
    for doc in docs:
        assert doc.metadata["source"] == "https://example.com/page"
        assert "id" in doc.metadata
        assert "chunk_index" in doc.metadata


@patch("app.scraper.httpx.get")
@patch("app.scraper.get_settings")
def test_scrape_url_returns_empty_on_sparse_content(mock_settings, mock_get):
    mock_settings.return_value = MagicMock(chunk_size=1000, chunk_overlap=200)

    mock_resp = MagicMock()
    mock_resp.text = "<html><body><p>Too short.</p></body></html>"
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    docs = scrape_url("https://example.com/empty")
    assert docs == []


@patch("app.scraper.httpx.get")
def test_scrape_url_returns_empty_on_http_error(mock_get):
    import httpx

    mock_get.side_effect = httpx.ConnectError("refused")
    docs = scrape_url("https://unreachable.example")
    assert docs == []


def test_scrape_url_ids_are_deterministic():
    id1 = _chunk_id("https://example.com", 0)
    id2 = _chunk_id("https://example.com", 0)
    assert id1 == id2


# ── search_and_scrape ─────────────────────────────────────────────────────────


@patch("app.scraper.scrape_url")
def test_search_and_scrape_uses_ddg_results(mock_scrape):
    from langchain_core.documents import Document

    mock_scrape.return_value = [
        Document(page_content="chunk", metadata={"source": "http://x.com", "page": 0, "id": "abc", "chunk_index": 0})
    ]

    with patch("app.scraper.DDGS") as mock_ddg_cls:
        mock_ddg = MagicMock()
        mock_ddg.__enter__ = MagicMock(return_value=mock_ddg)
        mock_ddg.__exit__ = MagicMock(return_value=False)
        mock_ddg.text.return_value = [
            {"href": "https://example.com/a"},
            {"href": "https://example.com/b"},
        ]
        mock_ddg_cls.return_value = mock_ddg

        chunks = search_and_scrape("Python loops", num_results=2)

    assert len(chunks) > 0
    assert mock_scrape.call_count >= 1


@patch("app.scraper.scrape_url")
def test_search_and_scrape_wikipedia_fallback(mock_scrape):
    from langchain_core.documents import Document

    mock_scrape.return_value = [
        Document(
            page_content="Wikipedia content about binary search",
            metadata={"source": "https://en.wikipedia.org/wiki/Binary_search", "page": 0, "id": "xyz", "chunk_index": 0},
        )
    ]

    with patch.dict("sys.modules", {"duckduckgo_search": None}):
        with patch("app.scraper.DDGS", side_effect=ImportError("not installed")):
            chunks = search_and_scrape("binary search", num_results=1)

    assert len(chunks) > 0


@patch("app.scraper.scrape_url")
def test_search_and_scrape_respects_max_chunks(mock_scrape):
    from langchain_core.documents import Document

    mock_scrape.return_value = [
        Document(page_content="x" * 100, metadata={"source": "u", "page": 0, "id": str(i), "chunk_index": i})
        for i in range(30)
    ]

    with patch("app.scraper.DDGS") as mock_ddg_cls:
        mock_ddg = MagicMock()
        mock_ddg.__enter__ = MagicMock(return_value=mock_ddg)
        mock_ddg.__exit__ = MagicMock(return_value=False)
        mock_ddg.text.return_value = [{"href": f"https://x.com/{i}"} for i in range(10)]
        mock_ddg_cls.return_value = mock_ddg

        chunks = search_and_scrape("topic", num_results=10, max_chunks=20)

    assert len(chunks) <= 20
