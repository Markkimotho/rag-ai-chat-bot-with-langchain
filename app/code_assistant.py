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
    )


def _build_graph():
    """A minimal single-node chat graph with persistent message history.

    The system prompt is prepended on every call; the checkpointer stores the
    running human/assistant turns per thread_id.
    """
    builder = StateGraph(MessagesState)

    def call_model(state: MessagesState, config) -> dict:
        model = config.get("configurable", {}).get("model")
        llm = _get_llm(model)
        messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
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
) -> str:
    """Return the assistant's full response for one message (non-streaming)."""
    graph = get_graph()
    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content=message)]},
            config={"configurable": {"thread_id": thread_id, "model": model}},
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
) -> Iterator[tuple[str, object]]:
    """Stream the assistant's response token-by-token.

    Yields ("token", str) for each delta. Errors surface as ("error", str).
    Conversation memory is persisted by the graph checkpointer.
    """
    graph = get_graph()
    try:
        for msg_chunk, _metadata in graph.stream(
            {"messages": [HumanMessage(content=message)]},
            config={"configurable": {"thread_id": thread_id, "model": model}},
            stream_mode="messages",
        ):
            token = getattr(msg_chunk, "content", "")
            if token:
                yield ("token", token)
    except Exception as exc:
        logger.exception("Code assistant stream failed")
        yield ("error", f"Code assistant error: {exc}")
