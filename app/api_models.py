"""Pydantic request/response models for the FastAPI layer.

These define the HTTP contract consumed by the React frontend. Keep them in
sync with frontend/src/api/types.ts.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

# ── Health / models ─────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    online: bool
    model_count: int


class ModelsResponse(BaseModel):
    supported: list[str]
    installed: list[str]


class PullRequest(BaseModel):
    model: str = Field(min_length=1)


class PullResponse(BaseModel):
    ok: bool
    message: str


# ── Knowledge base ────────────────────────────────────────────────────────────


class CountResponse(BaseModel):
    count: int


class ScrapeRequest(BaseModel):
    topic: str = Field(min_length=1)
    num_results: int = Field(default=3, ge=1, le=8)


class IngestResponse(BaseModel):
    chunks_added: int
    detail: str = ""


class ClearResponse(BaseModel):
    cleared: bool


class KBSource(BaseModel):
    source: str
    chunks: int
    type: str = "pdf"


class SourcesResponse(BaseModel):
    sources: list[KBSource]


# ── Chat (Regular Chat, RAG) ────────────────────────────────────────────────


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    session_id: str = "default"
    mode: Literal["langchain", "langgraph"] = "langchain"
    model: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)


# ── Programming assistant ──────────────────────────────────────────────────


class CodeChatRequest(BaseModel):
    message: str = Field(min_length=1)
    thread_id: str = "default"
    model: str | None = None
    use_kb: bool = False  # ground the answer in uploaded documents


# ── Study agent ────────────────────────────────────────────────────────────


class AgentRequest(BaseModel):
    message: str = Field(min_length=1)
    thread_id: str = "default"
    model: str | None = None


# ── Quiz ───────────────────────────────────────────────────────────────────


class QuizGenerateRequest(BaseModel):
    topic: str = Field(min_length=1)
    question_type: Literal["mcq", "true_false", "short_answer", "mixed"] = "mcq"
    n: int = Field(default=10, ge=1, le=20)
    difficulty: Literal["easy", "medium", "hard", "mixed"] = "medium"
    exam_type: str = "general exam"
    model: str | None = None


class QuizValidateRequest(BaseModel):
    question: str = Field(min_length=1)
    correct_answer: Any
    student_answer: str = ""
    question_type: Literal["mcq", "true_false", "short_answer"] = "short_answer"
    model: str | None = None


class ExplainRequest(BaseModel):
    concept: str = Field(min_length=1)
    model: str | None = None


class ExplainResponse(BaseModel):
    explanation: str


# ── Flashcards ─────────────────────────────────────────────────────────────


class Card(BaseModel):
    id: str
    front: str
    back: str
    created_at: float = 0


class Deck(BaseModel):
    id: str
    name: str
    created_at: float = 0
    cards: list[Card] = []


class DeckSummary(BaseModel):
    id: str
    name: str
    card_count: int
    created_at: float = 0


class DecksResponse(BaseModel):
    decks: list[DeckSummary]


class CreateDeckRequest(BaseModel):
    name: str = Field(min_length=1)


class CardInput(BaseModel):
    front: str = Field(min_length=1)
    back: str = Field(min_length=1)


class GenerateDeckRequest(BaseModel):
    topic: str = Field(min_length=1)
    n: int = Field(default=10, ge=1, le=50)
    model: str | None = None
    deck_name: str | None = None
