"""Flashcard decks — a simple persisted store plus LLM generation from the KB.

Decks live in a single JSON file (one small store; no DB needed) guarded by a
process lock. Cards can be created by hand or auto-generated from the knowledge
base. The store path is configurable so it can live on a mounted volume.
"""

import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from app.config import get_settings
from app.quiz import _extract_json  # robust JSON parsing shared with quiz gen
from app.retriever import get_retriever

logger = logging.getLogger(__name__)

_lock = threading.Lock()


def _store_path() -> Path:
    settings = get_settings()
    path = Path(settings.flashcards_dir) / "decks.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load() -> list[dict]:
    path = _store_path()
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        logger.exception("Failed to read flashcard store; starting empty")
        return []


def _save(decks: list[dict]) -> None:
    path = _store_path()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(decks, indent=2))
    tmp.replace(path)  # atomic


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _summary(deck: dict) -> dict:
    return {
        "id": deck["id"],
        "name": deck["name"],
        "card_count": len(deck.get("cards", [])),
        "created_at": deck.get("created_at", 0),
    }


# ── CRUD ──────────────────────────────────────────────────────────────────────


def list_decks() -> list[dict]:
    with _lock:
        decks = _load()
    return [_summary(d) for d in sorted(decks, key=lambda d: -d.get("created_at", 0))]


def get_deck(deck_id: str) -> dict | None:
    with _lock:
        for d in _load():
            if d["id"] == deck_id:
                return d
    return None


def create_deck(name: str, cards: list[dict] | None = None) -> dict:
    deck = {
        "id": _new_id(),
        "name": name.strip() or "Untitled deck",
        "created_at": time.time(),
        "cards": [
            {
                "id": _new_id(),
                "front": c["front"],
                "back": c["back"],
                "created_at": time.time(),
            }
            for c in (cards or [])
        ],
    }
    with _lock:
        decks = _load()
        decks.append(deck)
        _save(decks)
    return deck


def delete_deck(deck_id: str) -> bool:
    with _lock:
        decks = _load()
        remaining = [d for d in decks if d["id"] != deck_id]
        if len(remaining) == len(decks):
            return False
        _save(remaining)
    return True


def add_card(deck_id: str, front: str, back: str) -> dict | None:
    card = {"id": _new_id(), "front": front, "back": back, "created_at": time.time()}
    with _lock:
        decks = _load()
        for d in decks:
            if d["id"] == deck_id:
                d.setdefault("cards", []).append(card)
                _save(decks)
                return card
    return None


def update_card(deck_id: str, card_id: str, front: str, back: str) -> dict | None:
    with _lock:
        decks = _load()
        for d in decks:
            if d["id"] == deck_id:
                for c in d.get("cards", []):
                    if c["id"] == card_id:
                        c["front"], c["back"] = front, back
                        _save(decks)
                        return c
    return None


def delete_card(deck_id: str, card_id: str) -> bool:
    with _lock:
        decks = _load()
        for d in decks:
            if d["id"] == deck_id:
                before = len(d.get("cards", []))
                d["cards"] = [c for c in d.get("cards", []) if c["id"] != card_id]
                if len(d["cards"]) != before:
                    _save(decks)
                    return True
    return False


# ── LLM generation from the knowledge base ──────────────────────────────────

_GEN_PROMPT = """\
You are an expert study-aid creator. Using ONLY the context below, write {n} \
flashcards about "{topic}".

Each flashcard has:
- "front": a concise question, term, or prompt
- "back": a clear, correct, self-contained answer (1-3 sentences)

Make them varied and useful for active recall. Do not repeat cards.

Context:
{context}

Return ONLY a JSON array, e.g.:
[{{"front": "What is X?", "back": "X is ..."}}, ...]
"""


def generate_cards(
    topic: str, n: int = 10, model: str | None = None, top_k: int = 8
) -> list[dict]:
    """Generate flashcards from KB context. Returns [] if nothing usable."""
    settings = get_settings()
    docs = get_retriever(top_k=top_k).invoke(topic)
    if not docs:
        return []
    context = "\n\n---\n\n".join(
        f"[{d.metadata.get('source', 'doc')}] {d.page_content}" for d in docs
    )
    llm = ChatOllama(
        model=model or settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0.3,
        keep_alive=settings.ollama_keep_alive,
    )
    prompt = _GEN_PROMPT.format(n=n, topic=topic, context=context)
    try:
        raw = llm.invoke([HumanMessage(content=prompt)]).content
        parsed: Any = _extract_json(raw)
    except Exception:
        logger.exception("Flashcard generation failed")
        return []
    cards = []
    for item in parsed if isinstance(parsed, list) else []:
        front, back = item.get("front"), item.get("back")
        if front and back:
            cards.append({"front": str(front), "back": str(back)})
    return cards[:n]


def generate_deck(
    topic: str, n: int = 10, model: str | None = None, deck_name: str | None = None
) -> dict:
    """Generate cards from the KB and persist them as a new deck."""
    cards = generate_cards(topic, n=n, model=model)
    name = (deck_name or topic).strip() or topic
    return create_deck(name=name, cards=cards)
