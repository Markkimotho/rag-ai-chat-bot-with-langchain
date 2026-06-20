"""Gate tests for the flashcards store + generation (file store, mocked LLM)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import app.flashcards as fc


@pytest.fixture(autouse=True)
def temp_store(tmp_path, monkeypatch):
    """Point the deck store at a throwaway file for each test."""
    store = tmp_path / "decks.json"
    monkeypatch.setattr(fc, "_store_path", lambda: store)
    yield store


def test_create_and_list_deck():
    deck = fc.create_deck("Algorithms")
    decks = fc.list_decks()
    assert len(decks) == 1
    assert decks[0]["name"] == "Algorithms"
    assert decks[0]["card_count"] == 0
    assert decks[0]["id"] == deck["id"]


def test_create_with_cards_and_get():
    deck = fc.create_deck("DS", cards=[{"front": "O(1)?", "back": "constant"}])
    got = fc.get_deck(deck["id"])
    assert got["cards"][0]["front"] == "O(1)?"
    assert "id" in got["cards"][0]


def test_add_update_delete_card():
    deck = fc.create_deck("X")
    card = fc.add_card(deck["id"], "Q", "A")
    assert card is not None
    fc.update_card(deck["id"], card["id"], "Q2", "A2")
    assert fc.get_deck(deck["id"])["cards"][0]["front"] == "Q2"
    assert fc.delete_card(deck["id"], card["id"]) is True
    assert fc.get_deck(deck["id"])["cards"] == []


def test_delete_deck():
    deck = fc.create_deck("Y")
    assert fc.delete_deck(deck["id"]) is True
    assert fc.get_deck(deck["id"]) is None
    assert fc.delete_deck("missing") is False


def test_persists_across_reload(temp_store):
    fc.create_deck("Persisted", cards=[{"front": "a", "back": "b"}])
    assert Path(temp_store).exists()
    # A fresh read (new _load call) sees the deck.
    assert fc.list_decks()[0]["name"] == "Persisted"


def test_missing_deck_ops_return_none():
    assert fc.add_card("nope", "q", "a") is None
    assert fc.update_card("nope", "x", "q", "a") is None
    assert fc.delete_card("nope", "x") is False


# ── Generation (mocked LLM + retriever) ──────────────────────────────────────


def test_generate_cards_from_kb():
    from langchain_core.documents import Document

    fake_docs = [Document(page_content="Binary search is O(log n).", metadata={"source": "a"})]
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(
        content='[{"front":"Time complexity of binary search?","back":"O(log n)"}]'
    )
    with patch("app.flashcards.get_retriever") as mock_ret, patch(
        "app.flashcards.ChatOllama", return_value=fake_llm
    ):
        mock_ret.return_value.invoke.return_value = fake_docs
        cards = fc.generate_cards("binary search", n=5)
    assert cards == [
        {"front": "Time complexity of binary search?", "back": "O(log n)"}
    ]


def test_generate_cards_empty_kb_returns_empty():
    with patch("app.flashcards.get_retriever") as mock_ret:
        mock_ret.return_value.invoke.return_value = []
        assert fc.generate_cards("anything") == []


def test_generate_deck_persists():
    with patch("app.flashcards.generate_cards", return_value=[{"front": "q", "back": "a"}]):
        deck = fc.generate_deck("topic", n=3, deck_name="My Deck")
    assert deck["name"] == "My Deck"
    assert len(deck["cards"]) == 1
    assert fc.get_deck(deck["id"]) is not None
