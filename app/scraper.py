"""Web scraping tools for building a knowledge base when no documents are uploaded.

Primary search: DuckDuckGo (no API key required).
Fallback: direct Wikipedia URL for the topic.
"""

import hashlib
import logging
import re

import httpx
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

_DROP_TAGS = frozenset({
    "script", "style", "nav", "header", "footer", "aside",
    "advertisement", "iframe", "noscript", "form", "button",
})


# ── Helpers ───────────────────────────────────────────────────────────────────


def _chunk_id(url: str, idx: int) -> str:
    return hashlib.sha256(f"{url}::{idx}".encode()).hexdigest()


def _extract_text(html: str) -> str:
    """Extract readable body text from raw HTML."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("Run: pip install beautifulsoup4 lxml")

    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(_DROP_TAGS):
        tag.decompose()

    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find(attrs={"role": "main"})
        or soup.find("div", class_=re.compile(r"content|article|post|main|body", re.I))
        or soup.body
    )
    if not main:
        return ""

    lines = []
    for el in main.find_all(["p", "li", "h1", "h2", "h3", "h4", "td", "pre", "blockquote"]):
        t = el.get_text(separator=" ", strip=True)
        if len(t) > 40:
            lines.append(t)

    return "\n\n".join(lines)


# ── Public API ────────────────────────────────────────────────────────────────


def scrape_url(url: str, timeout: int = 20) -> list[Document]:
    """Fetch one URL and return chunked Documents ready for ChromaDB ingestion.

    Returns [] when no useful text could be extracted.
    """
    settings = get_settings()
    try:
        resp = httpx.get(url, headers=_HEADERS, timeout=timeout, follow_redirects=True)
        resp.raise_for_status()
        text = _extract_text(resp.text)
        if not text or len(text.strip()) < 200:
            logger.warning("No useful text extracted from %s", url)
            return []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        chunks = splitter.create_documents(
            [text],
            metadatas=[{"source": url, "page": 0, "type": "web"}],
        )
        for i, c in enumerate(chunks):
            c.metadata["chunk_index"] = i
            c.metadata["id"] = _chunk_id(url, i)

        logger.info("Scraped %s → %d chunks", url, len(chunks))
        return chunks
    except Exception as exc:
        logger.warning("scrape_url(%s) failed: %s", url, exc)
        return []


def search_and_scrape(
    topic: str,
    num_results: int = 4,
    max_chunks: int = 80,
) -> list[Document]:
    """Search DuckDuckGo for a topic, scrape the top results, return Documents.

    Falls back to a Wikipedia URL when DuckDuckGo is unavailable.
    Returned Documents are ready to pass straight to vectorstore.ingest_documents().
    """
    urls: list[str] = []

    if DDGS is not None:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(topic, max_results=num_results * 2))
            urls = [r["href"] for r in results if r.get("href")][:num_results]
            logger.info("DDG search for %r → %d candidate URLs", topic, len(urls))
        except Exception as exc:
            logger.warning("DuckDuckGo search failed (%s); falling back to Wikipedia", exc)

    if not urls:
        slug = topic.strip().replace(" ", "_")
        urls = [f"https://en.wikipedia.org/wiki/{slug}"]
        logger.info("Falling back to Wikipedia for %r", topic)

    all_chunks: list[Document] = []
    for url in urls:
        chunks = scrape_url(url)
        all_chunks.extend(chunks)
        if len(all_chunks) >= max_chunks:
            break

    logger.info(
        "search_and_scrape(%r) → %d total chunks from %d URL(s)",
        topic, len(all_chunks), len(urls),
    )
    return all_chunks[:max_chunks]
