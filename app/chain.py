"""LangChain LCEL RAG chain with conversation memory."""

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.memory import add_messages, get_chat_history
from app.prompts import CONTEXTUALIZE_PROMPT, QA_PROMPT
from app.retriever import get_retriever


def _build_chain(model: str | None = None, top_k: int | None = None):
    settings = get_settings()
    llm = ChatOpenAI(
        model=model or settings.openai_model,
        api_key=settings.openai_api_key.get_secret_value(),
        temperature=0,
    )
    retriever = get_retriever(top_k=top_k)

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, CONTEXTUALIZE_PROMPT
    )
    question_answer_chain = create_stuff_documents_chain(llm, QA_PROMPT)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)


def invoke(
    question: str,
    session_id: str = "default",
    model: str | None = None,
    top_k: int | None = None,
) -> dict:
    """Ask a question using the LCEL RAG chain. Returns {"answer": str, "sources": list}."""
    chain = _build_chain(model=model, top_k=top_k)
    chat_history = get_chat_history(session_id)

    result = chain.invoke({"input": question, "chat_history": chat_history})

    answer = result["answer"]
    sources = []
    for doc in result.get("context", []):
        sources.append(
            {
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", "?"),
            }
        )

    add_messages(session_id, question, answer)

    return {"answer": answer, "sources": sources}
