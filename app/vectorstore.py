import concurrent.futures
import logging
import time

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import get_settings
from app.embeddings import get_embeddings

logger = logging.getLogger(__name__)

_vectorstore: Chroma | None = None


def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        settings = get_settings()
        _vectorstore = Chroma(
            collection_name=settings.chroma_collection_name,
            embedding_function=get_embeddings(),
            persist_directory=settings.chroma_persist_dir,
        )
    return _vectorstore


def _existing_ids(collection, ids: list[str]) -> set[str]:
    """Return the subset of ids already present in the collection."""
    found: set[str] = set()
    for i in range(0, len(ids), 1000):
        try:
            res = collection.get(ids=ids[i : i + 1000], include=[])
            found.update(res.get("ids", []))
        except Exception:
            logger.exception("dedup lookup failed; treating batch as new")
    return found


def ingest_documents(
    documents: list[Document],
    batch_size: int | None = None,
    max_workers: int | None = None,
) -> int:
    """Embed and upsert chunks into ChromaDB. Returns the number newly added.

    Speed: embeddings are the bottleneck, so batches are embedded concurrently
    (Ollama serves parallel requests) and chunks whose deterministic id is
    already indexed are skipped entirely — re-uploading the same file is near
    instant instead of re-embedding every page.
    """
    if not documents:
        return 0

    settings = get_settings()
    batch_size = batch_size or settings.ingest_batch_size
    max_workers = max_workers or settings.ingest_workers

    vectorstore = get_vectorstore()
    collection = vectorstore._collection
    embedder = get_embeddings()

    ids = [doc.metadata["id"] for doc in documents]
    existing = _existing_ids(collection, ids)
    new_docs = [d for d in documents if d.metadata["id"] not in existing]
    skipped = len(documents) - len(new_docs)
    if not new_docs:
        logger.info("All %d chunk(s) already indexed — nothing to embed.", len(documents))
        return 0

    batches = [new_docs[i : i + batch_size] for i in range(0, len(new_docs), batch_size)]

    def _embed(batch: list[Document]) -> list[list[float]]:
        return embedder.embed_documents([d.page_content for d in batch])

    t0 = time.perf_counter()
    workers = max(1, min(max_workers, len(batches)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        vectors_per_batch = list(pool.map(_embed, batches))

    for batch, vectors in zip(batches, vectors_per_batch):
        collection.add(
            ids=[d.metadata["id"] for d in batch],
            embeddings=vectors,
            metadatas=[d.metadata for d in batch],
            documents=[d.page_content for d in batch],
        )

    elapsed = time.perf_counter() - t0
    logger.info(
        "Indexed %d chunk(s): %d new in %.1fs (%.0f/s), %d already present",
        len(documents),
        len(new_docs),
        elapsed,
        len(new_docs) / elapsed if elapsed else 0,
        skipped,
    )
    return len(new_docs)


def clear_vectorstore() -> None:
    """Delete all documents from the ChromaDB collection."""
    global _vectorstore
    vs = get_vectorstore()
    collection = vs._collection
    if collection.count() > 0:
        all_ids = collection.get()["ids"]
        collection.delete(ids=all_ids)
    logger.info("Cleared knowledge base")


def get_doc_count() -> int:
    """Return the number of chunks currently indexed in ChromaDB."""
    try:
        return get_vectorstore()._collection.count()
    except Exception:
        return 0


def list_sources() -> list[dict]:
    """Return the distinct sources in the KB with per-source chunk counts.

    Each item: {"source": str, "chunks": int, "type": str}. Sorted by source.
    Used by the UI to show what's in the knowledge base.
    """
    try:
        collection = get_vectorstore()._collection
        if collection.count() == 0:
            return []
        data = collection.get(include=["metadatas"])
        counts: dict[str, dict] = {}
        for md in data.get("metadatas", []) or []:
            src = (md or {}).get("source", "unknown")
            entry = counts.setdefault(
                src, {"source": src, "chunks": 0, "type": (md or {}).get("type", "pdf")}
            )
            entry["chunks"] += 1
        return sorted(counts.values(), key=lambda x: x["source"].lower())
    except Exception:
        logger.exception("list_sources failed")
        return []
