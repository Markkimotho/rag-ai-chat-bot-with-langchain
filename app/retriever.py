from langchain_core.vectorstores import VectorStoreRetriever

from app.config import get_settings
from app.vectorstore import get_vectorstore


def get_retriever(top_k: int | None = None) -> VectorStoreRetriever:
    settings = get_settings()
    k = top_k if top_k is not None else settings.top_k
    return get_vectorstore().as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
