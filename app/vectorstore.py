import logging
import time

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from app.config import get_settings
from app.embeddings import get_embeddings

logger = logging.getLogger(__name__)

_vectorstore: PineconeVectorStore | None = None


def _ensure_index() -> None:
    settings = get_settings()
    pc = Pinecone(api_key=settings.pinecone_api_key.get_secret_value())

    existing = [idx.name for idx in pc.list_indexes()]
    if settings.pinecone_index_name not in existing:
        logger.info("Creating Pinecone index '%s'", settings.pinecone_index_name)
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=1536,  # text-embedding-3-small
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Wait for index to be ready
        while not pc.describe_index(settings.pinecone_index_name).status["ready"]:
            time.sleep(1)
        logger.info("Index '%s' created and ready", settings.pinecone_index_name)


def get_vectorstore() -> PineconeVectorStore:
    global _vectorstore
    if _vectorstore is None:
        _ensure_index()
        settings = get_settings()
        _vectorstore = PineconeVectorStore(
            index_name=settings.pinecone_index_name,
            embedding=get_embeddings(),
            pinecone_api_key=settings.pinecone_api_key.get_secret_value(),
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
        logger.info("Upserted batch %d–%d of %d", i + 1, end, total)

    return total
