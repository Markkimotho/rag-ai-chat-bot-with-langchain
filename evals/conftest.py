"""Shared fixtures for the paid LLM eval suite.

Evals require a running Ollama with the default chat + embedding models pulled.
When Ollama is unreachable, every eval is skipped (not failed) so the gate-test
lane stays green on machines without a model server.
"""

import httpx
import pytest

from app.config import get_settings


def _ollama_ready() -> bool:
    settings = get_settings()
    try:
        r = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=3)
        r.raise_for_status()
        names = {m["name"] for m in r.json().get("models", [])}
        # Need the default chat model and an embedding model present.
        has_chat = any(settings.ollama_model.split(":")[0] in n for n in names)
        has_embed = any("embed" in n for n in names)
        return has_chat and has_embed
    except Exception:
        return False


@pytest.fixture(scope="session", autouse=True)
def require_ollama():
    if not _ollama_ready():
        pytest.skip(
            "Ollama not ready (server down or models not pulled) — skipping evals.",
            allow_module_level=False,
        )
