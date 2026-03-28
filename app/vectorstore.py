import logging

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


def ingest_documents(documents: list[Document], batch_size: int = 100) -> int:
    vectorstore = get_vectorstore()

    ids = [doc.metadata["id"] for doc in documents]
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]

    total = len(documents)
    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        vectorstore.add_texts(
            texts=texts[i:end],
            metadatas=metadatas[i:end],
            ids=ids[i:end],
        )
        logger.info("Upserted batch %d-%d of %d", i + 1, end, total)

    return total


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
