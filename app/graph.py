"""LangGraph stateful RAG graph with MemorySaver checkpointer."""

from __future__ import annotations

from typing import Annotated, Sequence, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.config import get_settings
from app.memory import get_memory_saver
from app.prompts import CONTEXTUALIZE_PROMPT, QA_PROMPT
from app.retriever import get_retriever


class GraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    documents: list[Document]
    question: str
    answer: str


def _get_llm(model: str | None = None) -> ChatOllama:
    settings = get_settings()
    return ChatOllama(
        model=model or settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0,
    )


def contextualize(state: GraphState, config: RunnableConfig) -> dict:
    """Rephrase the question to be standalone given chat history."""
    messages = list(state["messages"])
    question = state["question"]

    # Only contextualize if there's prior conversation
    if len(messages) > 1:
        model = config.get("configurable", {}).get("model")
        llm = _get_llm(model)
        prompt_value = CONTEXTUALIZE_PROMPT.invoke(
            {"chat_history": messages[:-1], "input": question}
        )
        response = llm.invoke(prompt_value)
        return {"question": response.content}
    return {"question": question}


def retrieve(state: GraphState, config: RunnableConfig) -> dict:
    """Retrieve relevant documents from the vector store."""
    top_k = config.get("configurable", {}).get("top_k")
    retriever = get_retriever(top_k=top_k)
    documents = retriever.invoke(state["question"])
    return {"documents": documents}


def generate(state: GraphState, config: RunnableConfig) -> dict:
    """Generate answer from retrieved documents."""
    model = config.get("configurable", {}).get("model")
    llm = _get_llm(model)

    context = "\n\n".join(doc.page_content for doc in state["documents"])
    messages = list(state["messages"])

    prompt_value = QA_PROMPT.invoke(
        {
            "context": context,
            "chat_history": messages[:-1] if len(messages) > 1 else [],
            "input": state["question"],
        }
    )
    response = llm.invoke(prompt_value)
    return {"answer": response.content, "messages": [AIMessage(content=response.content)]}


def build_graph():
    """Build and compile the RAG graph with a MemorySaver checkpointer."""
    graph = StateGraph(GraphState)

    graph.add_node("contextualize", contextualize)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)

    graph.add_edge(START, "contextualize")
    graph.add_edge("contextualize", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile(checkpointer=get_memory_saver())


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def invoke(
    question: str,
    thread_id: str = "default",
    model: str | None = None,
    top_k: int | None = None,
) -> dict:
    """Ask a question using the LangGraph RAG graph. Returns {"answer": str, "sources": list}."""
    graph = get_graph()

    result = graph.invoke(
        {
            "messages": [HumanMessage(content=question)],
            "question": question,
            "documents": [],
            "answer": "",
        },
        config={
            "configurable": {
                "thread_id": thread_id,
                "model": model,
                "top_k": top_k,
            }
        },
    )

    sources = []
    for doc in result.get("documents", []):
        sources.append(
            {
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", "?"),
            }
        )

    return {"answer": result["answer"], "sources": sources}
