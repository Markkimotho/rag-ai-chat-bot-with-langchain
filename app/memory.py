"""Conversation memory helpers for both LCEL and LangGraph orchestration modes."""

from collections import defaultdict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from app.config import get_settings

# In-memory chat history store, keyed by session/thread ID
_histories: dict[str, list[BaseMessage]] = defaultdict(list)


def get_chat_history(session_id: str) -> list[BaseMessage]:
    settings = get_settings()
    history = _histories[session_id]
    # Keep only the last N turns (each turn = 1 human + 1 AI = 2 messages)
    max_messages = settings.memory_window * 2
    if len(history) > max_messages:
        _histories[session_id] = history[-max_messages:]
    return _histories[session_id]


def add_messages(session_id: str, human_msg: str, ai_msg: str) -> None:
    _histories[session_id].append(HumanMessage(content=human_msg))
    _histories[session_id].append(AIMessage(content=ai_msg))


def clear_history(session_id: str) -> None:
    _histories.pop(session_id, None)


def get_memory_saver() -> MemorySaver:
    """LangGraph checkpointer for stateful conversation threads."""
    return MemorySaver()
