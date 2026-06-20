"""FastAPI backend for the Prep Pal AI React frontend.

This is a thin adapter: every route delegates to an existing engine module in
`app/` and shapes the result for HTTP. Conversational endpoints stream tokens
over Server-Sent Events (SSE); everything else is plain JSON.

Prometheus metrics are exposed at GET /metrics (see app/metrics.py + monitoring/).

Run (dev):   uvicorn app.api:app --reload
Run (prod):  uvicorn app.api:app --host 0.0.0.0 --port 8000
In production the built React SPA in frontend/dist is served at "/".
"""

import json
import logging
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import chain as lcel_chain
from app import code_assistant
from app import flashcards as fc
from app import graph as langgraph_module
from app import metrics
from app import quiz_agent
from app.analysis import get_chat_models, get_installed_models, pull_model
from app.api_models import (
    AgentRequest,
    CardInput,
    ChatRequest,
    ClearResponse,
    CodeChatRequest,
    CountResponse,
    CreateDeckRequest,
    Deck,
    DecksResponse,
    ExplainRequest,
    ExplainResponse,
    GenerateDeckRequest,
    HealthResponse,
    IngestResponse,
    ModelsResponse,
    PullRequest,
    PullResponse,
    QuizGenerateRequest,
    QuizValidateRequest,
    ScrapeRequest,
    SourcesResponse,
)
from app.config import get_settings
from app.ingestion import IMAGE_EXTENSIONS, load_and_chunk_image, load_and_chunk_pdf
from app.memory import clear_history
from app.quiz import explain_concept, generate_questions, validate_answer
from app.scraper import search_and_scrape
from app.vectorstore import (
    clear_vectorstore,
    get_doc_count,
    ingest_documents,
    list_sources,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Prep Pal AI", version="1.1.0")

# Dev: Vite serves the SPA on :5173 and proxies /api here. Allow it for CORS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Default HTTP metrics (request count + latency per route) at GET /metrics.
# Exposed before the SPA mount so "/" never shadows it.
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


def _set_kb_gauge() -> None:
    try:
        metrics.KB_CHUNKS.set(get_doc_count())
    except Exception:  # noqa: BLE001
        pass


# ── SSE helper ────────────────────────────────────────────────────────────────


def _sse(events: Iterator[tuple[str, object]], surface: str) -> StreamingResponse:
    """Wrap an engine generator of (kind, payload) tuples as an SSE response.

    Emits one JSON object per event, then a terminal {"type":"done"} frame.
    Records per-surface latency, token throughput, and success/error counts.
    """

    def gen() -> Iterator[str]:
        start = time.perf_counter()
        status = "success"
        try:
            for kind, payload in events:
                if kind == "token":
                    metrics.STREAM_TOKENS.labels(surface).inc()
                    frame = {"type": "token", "text": payload}
                elif kind == "sources":
                    frame = {"type": "sources", "sources": payload}
                elif kind == "tool":
                    frame = {"type": "tool", "name": payload}
                elif kind == "error":
                    status = "error"
                    frame = {"type": "error", "message": payload}
                else:
                    continue
                yield f"data: {json.dumps(frame)}\n\n"
        except Exception as exc:  # noqa: BLE001 — surface any engine failure to client
            status = "error"
            logger.exception("SSE stream failed")
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        finally:
            metrics.LLM_LATENCY.labels(surface).observe(time.perf_counter() - start)
            metrics.LLM_REQUESTS.labels(surface, status).inc()
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Health / models ─────────────────────────────────────────────────────────


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    _set_kb_gauge()
    try:
        r = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=3)
        r.raise_for_status()
        models = [m for m in r.json().get("models", []) if "embed" not in m["name"]]
        metrics.OLLAMA_UP.set(1)
        return HealthResponse(online=True, model_count=len(models))
    except Exception:
        metrics.OLLAMA_UP.set(0)
        return HealthResponse(online=False, model_count=0)


@app.get("/api/models", response_model=ModelsResponse)
def models() -> ModelsResponse:
    return ModelsResponse(
        supported=get_chat_models(),
        installed=sorted(get_installed_models()),
    )


@app.post("/api/models/pull", response_model=PullResponse)
def models_pull(req: PullRequest) -> PullResponse:
    ok, message = pull_model(req.model)
    return PullResponse(ok=ok, message=message)


# ── Knowledge base ────────────────────────────────────────────────────────────


@app.get("/api/kb/count", response_model=CountResponse)
def kb_count() -> CountResponse:
    count = get_doc_count()
    metrics.KB_CHUNKS.set(count)
    return CountResponse(count=count)


@app.get("/api/kb/sources", response_model=SourcesResponse)
def kb_sources() -> SourcesResponse:
    """List the files/URLs currently in the knowledge base with chunk counts."""
    return SourcesResponse(sources=list_sources())


@app.post("/api/kb/upload", response_model=IngestResponse)
async def kb_upload(files: list[UploadFile] = File(...)) -> IngestResponse:
    total = 0
    details: list[str] = []
    for f in files:
        name = f.filename or "file"
        ext = Path(name).suffix.lower()
        is_pdf = ext == ".pdf"
        is_image = ext in IMAGE_EXTENSIONS
        if not (is_pdf or is_image):
            details.append(f"{name}: skipped (unsupported type)")
            continue
        data = await f.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        try:
            # Preserve the original filename as the KB source label.
            if is_pdf:
                chunks = load_and_chunk_pdf(tmp_path, source_name=name)
            else:
                chunks = load_and_chunk_image(tmp_path, source_name=name)
            if not chunks:
                details.append(f"{name}: no readable text found")
                continue
            n = ingest_documents(chunks)
            total += n
            details.append(f"{name}: {n} chunks")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Upload failed for %s", name)
            details.append(f"{name}: failed — {exc}")
        finally:
            tmp_path.unlink(missing_ok=True)
    _set_kb_gauge()
    return IngestResponse(chunks_added=total, detail="; ".join(details))


@app.post("/api/kb/scrape", response_model=IngestResponse)
def kb_scrape(req: ScrapeRequest) -> IngestResponse:
    try:
        chunks = search_and_scrape(req.topic.strip(), num_results=req.num_results)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Scrape failed")
        raise HTTPException(status_code=502, detail=f"Scrape failed: {exc}") from exc
    if not chunks:
        return IngestResponse(chunks_added=0, detail="No content found.")
    n = ingest_documents(chunks)
    _set_kb_gauge()
    return IngestResponse(chunks_added=n, detail=f"Added {n} chunks from web.")


@app.delete("/api/kb", response_model=ClearResponse)
def kb_clear() -> ClearResponse:
    clear_vectorstore()
    metrics.KB_CHUNKS.set(0)
    return ClearResponse(cleared=True)


# ── Chat (Regular Chat, RAG) — streaming ────────────────────────────────────


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    if req.mode == "langgraph":
        events = langgraph_module.stream(
            question=req.question,
            thread_id=req.session_id,
            model=req.model,
            top_k=req.top_k,
        )
    else:
        events = lcel_chain.stream(
            question=req.question,
            session_id=req.session_id,
            model=req.model,
            top_k=req.top_k,
        )
    return _sse(events, surface="chat")


@app.delete("/api/chat/{session_id}", response_model=ClearResponse)
def chat_clear(session_id: str) -> ClearResponse:
    clear_history(session_id)
    return ClearResponse(cleared=True)


# ── Programming assistant — streaming ───────────────────────────────────────


@app.post("/api/code-chat/stream")
def code_chat_stream(req: CodeChatRequest) -> StreamingResponse:
    events = code_assistant.stream_code_assistant(
        message=req.message,
        thread_id=req.thread_id,
        model=req.model,
        use_kb=req.use_kb,
    )
    return _sse(events, surface="code")


# ── Study agent — streaming ─────────────────────────────────────────────────


@app.post("/api/agent/stream")
def agent_stream(req: AgentRequest) -> StreamingResponse:
    events = quiz_agent.stream_agent(
        message=req.message, thread_id=req.thread_id, model=req.model
    )
    return _sse(events, surface="agent")


# ── Quiz ───────────────────────────────────────────────────────────────────


@app.post("/api/quiz/generate")
def quiz_generate(req: QuizGenerateRequest) -> list[dict]:
    start = time.perf_counter()
    status = "success"
    try:
        return generate_questions(
            topic=req.topic.strip(),
            question_type=req.question_type,
            n=req.n,
            difficulty=req.difficulty,
            exam_type=req.exam_type,
            model=req.model,
            top_k=get_settings().quiz_top_k,
        )
    except Exception:
        status = "error"
        raise
    finally:
        metrics.LLM_LATENCY.labels("quiz").observe(time.perf_counter() - start)
        metrics.LLM_REQUESTS.labels("quiz", status).inc()


@app.post("/api/quiz/validate")
def quiz_validate(req: QuizValidateRequest) -> dict:
    return validate_answer(
        question=req.question,
        correct_answer=req.correct_answer,
        student_answer=req.student_answer,
        question_type=req.question_type,
        model=req.model,
    )


@app.post("/api/quiz/explain", response_model=ExplainResponse)
def quiz_explain(req: ExplainRequest) -> ExplainResponse:
    return ExplainResponse(explanation=explain_concept(req.concept, model=req.model))


# ── Flashcards ─────────────────────────────────────────────────────────────


@app.get("/api/flashcards", response_model=DecksResponse)
def flashcards_list() -> DecksResponse:
    return DecksResponse(decks=fc.list_decks())


@app.post("/api/flashcards", response_model=Deck)
def flashcards_create(req: CreateDeckRequest) -> Deck:
    return Deck(**fc.create_deck(req.name))


@app.post("/api/flashcards/generate", response_model=Deck)
def flashcards_generate(req: GenerateDeckRequest) -> Deck:
    start = time.perf_counter()
    status = "success"
    try:
        return Deck(**fc.generate_deck(req.topic, n=req.n, model=req.model, deck_name=req.deck_name))
    except Exception:
        status = "error"
        raise
    finally:
        metrics.LLM_LATENCY.labels("flashcards").observe(time.perf_counter() - start)
        metrics.LLM_REQUESTS.labels("flashcards", status).inc()


@app.get("/api/flashcards/{deck_id}", response_model=Deck)
def flashcards_get(deck_id: str) -> Deck:
    deck = fc.get_deck(deck_id)
    if not deck:
        raise HTTPException(status_code=404, detail="Deck not found")
    return Deck(**deck)


@app.delete("/api/flashcards/{deck_id}", response_model=ClearResponse)
def flashcards_delete(deck_id: str) -> ClearResponse:
    if not fc.delete_deck(deck_id):
        raise HTTPException(status_code=404, detail="Deck not found")
    return ClearResponse(cleared=True)


@app.post("/api/flashcards/{deck_id}/cards", response_model=Deck)
def flashcards_add_card(deck_id: str, req: CardInput) -> Deck:
    if fc.add_card(deck_id, req.front, req.back) is None:
        raise HTTPException(status_code=404, detail="Deck not found")
    return Deck(**fc.get_deck(deck_id))


@app.put("/api/flashcards/{deck_id}/cards/{card_id}", response_model=Deck)
def flashcards_update_card(deck_id: str, card_id: str, req: CardInput) -> Deck:
    if fc.update_card(deck_id, card_id, req.front, req.back) is None:
        raise HTTPException(status_code=404, detail="Card not found")
    return Deck(**fc.get_deck(deck_id))


@app.delete("/api/flashcards/{deck_id}/cards/{card_id}", response_model=Deck)
def flashcards_delete_card(deck_id: str, card_id: str) -> Deck:
    if not fc.delete_card(deck_id, card_id):
        raise HTTPException(status_code=404, detail="Card not found")
    return Deck(**fc.get_deck(deck_id))


# ── Static SPA (production) ─────────────────────────────────────────────────
# Mounted last so it never shadows /api routes. Real files (e.g. /assets/*) are
# served directly; any other path falls back to index.html so client-side
# routing (BrowserRouter deep links like /quiz) works. Only mounted if a build
# exists — otherwise the API runs standalone behind the Vite dev server.

_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"


class SPAStaticFiles(StaticFiles):
    """StaticFiles that serves index.html for unknown (non-file) routes."""

    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404:
                return await super().get_response("index.html", scope)
            raise


if _DIST.is_dir():
    app.mount("/", SPAStaticFiles(directory=str(_DIST), html=True), name="spa")
    logger.info("Serving SPA from %s", _DIST)
else:
    logger.info("No SPA build at %s — API-only mode (use Vite dev server).", _DIST)
