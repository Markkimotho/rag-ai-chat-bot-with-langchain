"""Prometheus metrics for Prep Pal AI.

Exposed at GET /metrics (set up in app/api.py). Scrape with Prometheus and
visualize in Grafana — see monitoring/ for a ready-made stack.

Default HTTP metrics (request count/latency per route) come from
prometheus-fastapi-instrumentator. This module adds domain metrics:
LLM-surface latency/throughput, streamed-token counts, Ollama availability,
and knowledge-base size.
"""

from prometheus_client import Counter, Gauge, Histogram

# LLM-backed surfaces: "chat", "code", "agent", "quiz".
LLM_REQUESTS = Counter(
    "preppal_llm_requests_total",
    "LLM-backed requests, by surface and outcome.",
    ["surface", "status"],
)

LLM_LATENCY = Histogram(
    "preppal_llm_request_duration_seconds",
    "End-to-end duration of an LLM-backed request, by surface.",
    ["surface"],
    buckets=(0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300),
)

STREAM_TOKENS = Counter(
    "preppal_stream_tokens_total",
    "Tokens streamed to clients, by surface.",
    ["surface"],
)

OLLAMA_UP = Gauge(
    "preppal_ollama_up",
    "Whether the Ollama server is reachable (1) or not (0).",
)

KB_CHUNKS = Gauge(
    "preppal_kb_chunks",
    "Number of chunks currently indexed in the knowledge base.",
)
