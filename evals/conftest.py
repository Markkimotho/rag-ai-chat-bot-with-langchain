"""Shared fixtures for the paid LLM eval suite.

Evals require a running Ollama with the default chat + embedding models pulled.
When Ollama is unreachable, every eval is skipped (not failed) so the gate-test
lane stays green on machines without a model server.
"""

import httpx
import pytest

from app.config import get_settings

# Preference order for the chat model an eval will use. The first one that is
# actually installed wins, so evals run against whatever is pulled.
_PREFERRED = ["qwen2.5:7b", "llama3.1:8b", "mistral", "gemma2:9b"]


def _installed_models() -> set[str]:
    settings = get_settings()
    try:
        r = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=3)
        r.raise_for_status()
        return {m["name"] for m in r.json().get("models", [])}
    except Exception:
        return set()


def _pick_chat_model(names: set[str]) -> str | None:
    chat = [n for n in names if "embed" not in n]
    for pref in _PREFERRED:
        for n in chat:
            if n.startswith(pref):
                return n
    return chat[0] if chat else None


@pytest.fixture(scope="session")
def chat_model() -> str:
    """An installed, non-embedding chat model to drive evals with."""
    model = _pick_chat_model(_installed_models())
    assert model, "no installed chat model"
    return model


@pytest.fixture(scope="session", autouse=True)
def require_ollama():
    names = _installed_models()
    has_embed = any("embed" in n for n in names)
    has_chat = _pick_chat_model(names) is not None
    if not (has_embed and has_chat):
        pytest.skip(
            "Ollama not ready: need an embedding model + a chat model pulled.",
            allow_module_level=False,
        )
