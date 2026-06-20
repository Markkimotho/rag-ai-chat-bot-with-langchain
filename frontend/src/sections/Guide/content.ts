// In-app guide content. Each entry is a section with an id (anchor), a short
// title for the table of contents, a group, and Markdown rendered by <Markdown>.

export interface GuideSection {
  id: string;
  title: string;
  group: "Getting started" | "Core features" | "Developer";
  body: string;
}

export const GUIDE: GuideSection[] = [
  {
    id: "overview",
    title: "Overview",
    group: "Getting started",
    body: `
# Prep Pal AI

A fully local AI study tool. Upload PDFs or images (or scrape the web), then
study with quizzes, an AI tutor, flashcards, a code companion, and document
chat. Everything runs on your machine via **Ollama** — no API keys, no cloud.

**Five core sections**, each with its own accent colour:

- **Quiz Prep** — generate and take quizzes from your material
- **Study Agent** — an AI tutor that can search, quiz, grade, and explain
- **Programming Assistant** — a coding chat that can optionally use your docs
- **Regular Chat** — RAG document Q&A with citations
- **Flashcards** — create or auto-generate decks and study with flip cards

Responses stream token-by-token. Your uploaded documents are reachable from
every section.
`,
  },
  {
    id: "getting-started",
    title: "Run it",
    group: "Getting started",
    body: `
## Running the app

### Docker (recommended)

\`\`\`bash
docker compose up --build
\`\`\`

First boot pulls the models (\`qwen2.5:7b\` + \`nomic-embed-text\`), builds the
SPA, and serves everything at **http://localhost:8000**. Later starts are
instant — models are cached in a volume.

### With monitoring (Prometheus + Grafana + Loki)

\`\`\`bash
docker compose -f docker-compose.yml -f docker-compose.monitoring.yml up --build
\`\`\`

Adds Grafana at **http://localhost:3000** (admin / admin).

### Local development

\`\`\`bash
# backend
uvicorn app.api:app --reload          # http://localhost:8000

# frontend (separate terminal)
cd frontend && npm install && npm run dev   # http://localhost:5173
\`\`\`
`,
  },
  {
    id: "knowledge-base",
    title: "Knowledge base",
    group: "Core features",
    body: `
## Knowledge base

Your documents power Quiz Prep, Regular Chat, flashcard generation, the Study
Agent, and (optionally) the Programming Assistant.

Open the **Knowledge base** button in the top bar to:

1. **Upload PDFs or images** — PNG/JPG/WebP/TIFF. Scanned PDFs and images are
   OCR'd automatically (Tesseract).
2. **Scrape a web topic** — type a topic, it searches DuckDuckGo (Wikipedia
   fallback), scrapes, and indexes the pages.
3. **See what's indexed** — every file/URL with its chunk count and a pdf / img
   / web badge.
4. **Clear** the whole knowledge base.

You get a ✓/✗ acknowledgement when indexing finishes, with a live "indexing…"
indicator.

**Good to know**

- Re-uploading the same file is near-instant — already-indexed chunks are
  skipped (deduplicated by content id).
- Large files index faster because embeddings are computed in parallel batches.
`,
  },
  {
    id: "quiz-prep",
    title: "Quiz Prep",
    group: "Core features",
    body: `
## Quiz Prep

Generate a quiz from your material and take it interactively.

1. Pick a **topic**, **question type** (MCQ / True-False / Short Answer / Mixed),
   **count**, **difficulty**, and **exam type**.
2. Click **Generate quiz**.
3. Answer one question at a time — submit to get instant grading, an
   explanation, and what you missed.
4. The results screen shows your score, a per-question breakdown, and weak areas.
   Retake the same quiz or start a new one.

MCQ and True/False are graded deterministically; short answers are graded by the
model (0–100 with feedback). You need content in the knowledge base first.
`,
  },
  {
    id: "study-agent",
    title: "Study Agent",
    group: "Core features",
    body: `
## Study Agent

A conversational tutor with tools. It decides what to do based on what you ask.

It can:

- **Search the web** and add results to your knowledge base
- **Generate a quiz** and walk you through it
- **Grade your answers** with feedback
- **Explain** concepts from your documents
- **List** what's already in your knowledge base

Just talk to it: *"Quiz me on Python async"*, *"Search for system design prep and
test me"*, *"What's in my knowledge base?"*. Tool use shows as chips while it
works. **New session** clears the conversation.

Tool calling needs a capable model — \`qwen2.5:7b\` or \`llama3.1:8b\`.
`,
  },
  {
    id: "programming-assistant",
    title: "Programming Assistant",
    group: "Core features",
    body: `
## Programming Assistant

A coding companion for any language or stack. Ask it to write, explain, debug,
or refactor code. Answers come with syntax-highlighted, copy-able code blocks.

Flip the **"Use my documents"** toggle to ground answers in your uploaded files
— useful for questions about your own codebase or docs. With the toggle off it
answers from the model's own knowledge (no retrieval).
`,
  },
  {
    id: "regular-chat",
    title: "Regular Chat",
    group: "Core features",
    body: `
## Regular Chat

Straight document Q&A over your knowledge base (RAG). Answers include **source
citations** (filename + page) you can expand.

Toggle between two orchestration modes:

- **LangChain** — LCEL chain with history-aware retrieval
- **LangGraph** — a stateful graph (contextualize → retrieve → generate)

Both rewrite follow-up questions to be standalone and stream the answer.
`,
  },
  {
    id: "flashcards",
    title: "Flashcards",
    group: "Core features",
    body: `
## Flashcards

Create decks and study with animated flip cards.

**Make a deck**

- **By hand** — name a deck, then add front/back cards.
- **Generate from your documents** — type a topic and a card count; the model
  writes a deck from your indexed material.

**Study mode**

- Click a card (or press **Space**) to flip between question and answer.
- **→ / Enter** = Got it · **← ** = Again · the progress bar and counter track
  where you are.
- At the end you get a score; study again or go back to edit the deck.

Decks persist on disk (a mounted volume in Docker), so they survive restarts.
`,
  },
  {
    id: "api",
    title: "API reference",
    group: "Developer",
    body: `
## API reference

The FastAPI backend serves JSON + SSE under \`/api\` and auto-docs at
**\`/docs\`** (OpenAPI) and **\`/redoc\`**.

**Health & models**

| Method | Path | Purpose |
| --- | --- | --- |
| GET | \`/api/health\` | Ollama reachable + model count |
| GET | \`/api/models\` | supported + installed models |
| POST | \`/api/models/pull\` | pull a model |

**Knowledge base**

| Method | Path | Purpose |
| --- | --- | --- |
| GET | \`/api/kb/count\` | indexed chunk count |
| GET | \`/api/kb/sources\` | indexed files/URLs + chunk counts |
| POST | \`/api/kb/upload\` | upload PDFs/images (multipart) |
| POST | \`/api/kb/scrape\` | scrape + index a web topic |
| DELETE | \`/api/kb\` | clear the knowledge base |

**Chat (SSE streaming)** — \`text/event-stream\`, events:
\`{"type":"token"}\`, \`{"type":"sources"}\`, \`{"type":"tool"}\`,
\`{"type":"error"}\`, \`{"type":"done"}\`.

| Method | Path | Surface |
| --- | --- | --- |
| POST | \`/api/chat/stream\` | Regular Chat (RAG) |
| POST | \`/api/code-chat/stream\` | Programming Assistant (\`use_kb\` optional) |
| POST | \`/api/agent/stream\` | Study Agent |

**Quiz & flashcards**

| Method | Path |
| --- | --- |
| POST | \`/api/quiz/generate\` · \`/api/quiz/validate\` · \`/api/quiz/explain\` |
| GET/POST | \`/api/flashcards\` (list / create) |
| POST | \`/api/flashcards/generate\` |
| GET/DELETE | \`/api/flashcards/{id}\` |
| POST/PUT/DELETE | \`/api/flashcards/{id}/cards[/{card_id}]\` |

Consuming an SSE endpoint with \`fetch\`:

\`\`\`ts
const res = await fetch("/api/code-chat/stream", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ message: "write a debounce fn" }),
});
const reader = res.body!.getReader();
// decode chunks and split on \\n\\n to parse "data: {json}" frames
\`\`\`
`,
  },
  {
    id: "monitoring",
    title: "Monitoring & Loki",
    group: "Developer",
    body: `
## Monitoring & logs

Metrics live at **\`/metrics\`** (Prometheus format). The monitoring overlay adds
Prometheus, Grafana, Loki, and Promtail.

\`\`\`bash
docker compose -f docker-compose.yml -f docker-compose.monitoring.yml up --build
\`\`\`

| Service | URL |
| --- | --- |
| Grafana | http://localhost:3000 (admin / admin) |
| Prometheus | http://localhost:9090 |
| Metrics | http://localhost:8000/metrics |

Grafana opens with the **"Prep Pal AI — Overview"** dashboard: Ollama up, KB
size, request/error rates, LLM latency p95 by surface, tokens/sec, and a live
log panel.

### Custom metrics

- \`preppal_llm_request_duration_seconds\` (by surface)
- \`preppal_llm_requests_total\` (surface, status)
- \`preppal_stream_tokens_total\` (surface)
- \`preppal_ollama_up\`, \`preppal_kb_chunks\`

### Using Loki (logs) — step by step

**Loki has no UI** — opening http://localhost:3100 returns *404 page not found*.
That's expected. You read logs through Grafana; Promtail feeds them in.

1. Open **Grafana** → http://localhost:3000 (admin / admin).
2. Left sidebar → **Explore** (compass icon).
3. Datasource dropdown → **Loki**.
4. Type a query and **Run query** (Shift+Enter).

Promtail tags each line with these labels:

| Label | Examples |
| --- | --- |
| \`compose_service\` | \`web\`, \`ollama\`, \`grafana\`, \`prometheus\` |
| \`container\` | \`prep-pal-web\`, \`prep-pal-ollama\` |
| \`compose_project\` | \`rag-ai-chat-bot-with-langchain\` |

**LogQL queries you'll use**

\`\`\`logql
{compose_service="web"}                  # all backend/API logs
{compose_service="ollama"}               # model server
{compose_service="web"} |= "error"       # lines containing "error"
{compose_service="web"} != "GET /metrics"  # hide metric scrapes
{compose_service="web"} |~ "(?i)error|fail" # regex, case-insensitive
{compose_service="web"} | json           # parse JSON lines into fields
\`\`\`

Operators: \`|=\` contains · \`!=\` excludes · \`|~\` regex · \`!~\` regex-exclude.

**Live tail**: in Explore, run a query then click **Live** to stream new lines.

**Time range matters**: Loki only shows logs in the selected window (top-right).
Empty panel? Widen the range first — that's the #1 cause.

**Sanity checks (terminal)**

\`\`\`bash
curl -s http://localhost:3100/ready
curl -s "http://localhost:3100/loki/api/v1/label/compose_service/values"
docker compose logs -f web        # stream app logs directly, no Grafana
\`\`\`
`,
  },
  {
    id: "testing",
    title: "Testing",
    group: "Developer",
    body: `
## Testing

Two lanes: **gate tests** (deterministic, mocked, no Ollama) run on every change;
**evals** (paid, real Ollama) run before ship.

\`\`\`bash
# backend gate tests
pytest tests/ -v

# frontend gate tests (Vitest + React Testing Library)
cd frontend && npm run test

# evals — need Ollama with qwen2.5:7b + nomic-embed-text pulled
pytest evals/ -m eval -v
\`\`\`

Gate tests mock every LLM/network call, so they're fast and free. Evals skip
cleanly when models aren't available.
`,
  },
  {
    id: "configuration",
    title: "Configuration",
    group: "Developer",
    body: `
## Configuration

Settings come from \`pydantic-settings\` (\`app/config.py\`); override any in
\`.env\` or as environment variables.

| Variable | Default | Notes |
| --- | --- | --- |
| \`OLLAMA_MODEL\` | \`qwen2.5:7b\` | Default chat/quiz/code model |
| \`OLLAMA_KEEP_ALIVE\` | \`30m\` | Keep the model warm between requests |
| \`TOP_K\` | \`5\` | Chunks retrieved per query |
| \`QUIZ_TOP_K\` | \`8\` | Chunks for quiz/flashcard generation |
| \`INGEST_BATCH_SIZE\` | \`256\` | Chunks per embedding batch |
| \`INGEST_WORKERS\` | \`4\` | Concurrent embedding batches |
| \`FLASHCARDS_DIR\` | \`data/flashcards\` | Where decks are stored |

**Performance tips**

- Keep \`OLLAMA_KEEP_ALIVE\` high so you don't pay cold-load latency each request.
- On slower machines pick a smaller model (\`llama3.2:3b\`/\`1b\`) from the model
  picker for faster responses.
- Large-file indexing is parallel + deduplicated — re-uploads are near-instant.
`,
  },
  {
    id: "architecture",
    title: "Architecture",
    group: "Developer",
    body: `
## Architecture

\`\`\`text
React SPA  ──HTTP/SSE──▶  FastAPI (app/api.py)
                              │  thin adapter, one route per engine fn
                              ▼
   chain.py / graph.py  · RAG chat (LCEL / LangGraph)
   quiz.py              · question gen + grading
   quiz_agent.py        · ReAct tool-calling tutor
   code_assistant.py    · coding chat (optional KB)
   flashcards.py        · deck store + LLM generation
   scraper.py           · DuckDuckGo + BeautifulSoup
   ingestion.py         · PDF 3-stage + image OCR
   vectorstore.py       · ChromaDB (parallel embed + dedup)
                              │
                              ▼
                 Ollama (LLM + embeddings) · ChromaDB (vectors)
\`\`\`

The FastAPI layer is a thin boundary — each route delegates to an engine module
and shapes the result for HTTP. In production it also serves the built React SPA
(with client-side-routing fallback) so the whole app runs on one port.
`,
  },
];
