"""Programming assistant — a coding-focused chat with conversation memory.

Unlike the RAG chat and the quiz agent, this assistant does NOT touch the
document knowledge base. It is a pure code companion: a strong system prompt
plus per-thread conversation memory, backed by a local Ollama model.

A module-level MemorySaver keeps conversation threads alive across requests for
as long as the process lives, keyed by thread_id (mirrors app/quiz_agent.py).
"""

import logging
from collections.abc import Iterator

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

from app.config import get_settings
from app.retriever import get_retriever

logger = logging.getLogger(__name__)

# Shared checkpointer — keeps thread history across requests for the process life.
_checkpointer = MemorySaver()

SYSTEM_PROMPT = """\
You are an expert programming assistant. You help developers write, understand, \
debug, and improve code across any language or stack.

Rules:
- Be concise and correct. Lead with the solution, then a short explanation.
- ALWAYS put code in fenced code blocks with the correct language tag \
(```python, ```ts, ```bash, ...).
- Prefer idiomatic, production-quality code. Handle edge cases. Avoid dead code.
- When the user shares an error, identify the root cause before proposing a fix, \
and say exactly where it breaks.
- When there is a meaningful tradeoff between approaches, name it in one line and \
recommend one — do not dump every option.
- If a request is ambiguous in a way that changes the code, ask one focused \
question instead of guessing.
- Do not invent APIs. If unsure whether something exists, say so.
"""

_graph = None


def _get_llm(model: str | None = None) -> ChatOllama:
    settings = get_settings()
    return ChatOllama(
        model=model or settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0.2,
        keep_alive=settings.ollama_keep_alive,
    )


def _retrieve_context(query: str, top_k: int | None = None) -> str:
    """Pull relevant chunks from the knowledge base for grounding (best-effort)."""
    try:
        docs = get_retriever(top_k=top_k).invoke(query)
    except Exception:
        logger.exception("KB retrieval failed in code assistant")
        return ""
    if not docs:
        return ""
    return "\n\n---\n\n".join(
        f"[{d.metadata.get('source', 'doc')}] {d.page_content}" for d in docs
    )


def _build_graph():
    """A minimal single-node chat graph with persistent message history.

    The system prompt is prepended on every call; the checkpointer stores the
    running human/assistant turns per thread_id. When `use_kb` is set, relevant
    chunks from the uploaded documents are injected as an ephemeral context
    message (not persisted to memory).
    """
    builder = StateGraph(MessagesState)

    def call_model(state: MessagesState, config) -> dict:
        cfg = config.get("configurable", {})
        llm = _get_llm(cfg.get("model"))
        messages = [SystemMessage(content=SYSTEM_PROMPT)]

        if cfg.get("use_kb"):
            last_human = next(
                (m for m in reversed(state["messages"]) if m.type == "human"), None
            )
            if last_human:
                context = _retrieve_context(last_human.content, cfg.get("top_k"))
                if context:
                    messages.append(
                        SystemMessage(
                            content=(
                                "Relevant excerpts from the user's uploaded documents:\n\n"
                                f"{context}\n\n"
                                "Use these when they help answer the question. If they "
                                "are not relevant, answer from your own knowledge."
                            )
                        )
                    )

        messages.extend(state["messages"])
        response = llm.invoke(messages)
        return {"messages": [response]}

    builder.add_node("call_model", call_model)
    builder.add_edge(START, "call_model")
    return builder.compile(checkpointer=_checkpointer)


def get_graph():
    global _graph
    if _graph is None:
        _graph = _build_graph()
    return _graph


def invoke_code_assistant(
    message: str,
    thread_id: str = "default",
    model: str | None = None,
    use_kb: bool = False,
) -> str:
    """Return the assistant's full response for one message (non-streaming)."""
    graph = get_graph()
    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content=message)]},
            config={"configurable": {"thread_id": thread_id, "model": model, "use_kb": use_kb}},
        )
        messages = result.get("messages", [])
        return messages[-1].content if messages else "No response."
    except Exception as exc:
        logger.exception("Code assistant invocation failed")
        return f"Code assistant error: {exc}"


def stream_code_assistant(
    message: str,
    thread_id: str = "default",
    model: str | None = None,
    use_kb: bool = False,
) -> Iterator[tuple[str, object]]:
    """Stream the assistant's response token-by-token.

    Yields ("token", str) for each delta. Errors surface as ("error", str).
    Conversation memory is persisted by the graph checkpointer. When `use_kb`
    is set, responses are grounded in the uploaded documents.
    """
    graph = get_graph()
    try:
        for msg_chunk, _metadata in graph.stream(
            {"messages": [HumanMessage(content=message)]},
            config={"configurable": {"thread_id": thread_id, "model": model, "use_kb": use_kb}},
            stream_mode="messages",
        ):
            token = getattr(msg_chunk, "content", "")
            if token:
                yield ("token", token)
    except Exception as exc:
        logger.exception("Code assistant stream failed")
        yield ("error", f"Code assistant error: {exc}")
