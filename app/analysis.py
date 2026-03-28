"""Multi-model parallel analysis — compare models across your document knowledge base."""

import concurrent.futures
import logging
import re
import time

import httpx
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from app.config import SUPPORTED_MODELS, get_settings
from app.retriever import get_retriever

logger = logging.getLogger(__name__)

_ANALYST_PROMPT = """\
You are a senior data analyst with 20 years of experience in business intelligence \
and research. Based solely on the provided context documents, deliver a structured, \
evidence-based analysis.

Focus on:
1. Key facts, figures, and data points from the documents
2. Patterns, trends, and notable correlations
3. Identified risks, gaps, and opportunities
4. Concrete, prioritized recommendations

Context:
{context}

Question: {question}

Provide a thorough, well-structured analysis with specific evidence from the documents. \
Use bullet points, numbered lists, and clear section headings. Be precise and actionable."""

_INSIGHTS_QUESTIONS = {
    "summary": (
        "Executive Summary",
        "Provide a concise executive summary of all documents in the knowledge base. "
        "Include the main topics, scope, key findings, and overall purpose.",
    ),
    "key_metrics": (
        "Key Metrics & Data Points",
        "What are the key metrics, numbers, statistics, dates, percentages, or quantitative "
        "data points present across the documents? Extract and list them all with context.",
    ),
    "risks": (
        "Risk Analysis",
        "What risks, threats, challenges, warnings, or problems are identified in the documents? "
        "Categorize by severity (high/medium/low) and suggest mitigations.",
    ),
    "recommendations": (
        "Actionable Recommendations",
        "Based on the documents, what are the top actionable next steps, recommendations, or "
        "opportunities? Prioritize from most to least urgent with reasoning.",
    ),
}


def get_chat_models() -> list[str]:
    """Return all supported chat models (regardless of installation status)."""
    return list(SUPPORTED_MODELS)


def get_installed_models() -> set[str]:
    """Return the subset of SUPPORTED_MODELS currently pulled in Ollama."""
    settings = get_settings()
    try:
        resp = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        return {
            m["name"]
            for m in resp.json().get("models", [])
            if "embed" not in m["name"]
        }
    except Exception:
        return set()


def pull_model(model: str) -> tuple[bool, str]:
    """Pull a model from Ollama. Blocks until complete. Returns (success, message)."""
    settings = get_settings()
    try:
        with httpx.stream(
            "POST",
            f"{settings.ollama_base_url}/api/pull",
            json={"name": model},
            timeout=900,          # large models can take 10-15 min on slow connections
        ) as resp:
            resp.raise_for_status()
            for _ in resp.iter_lines():
                pass              # consume stream so download actually completes
        return True, f"Successfully pulled {model}."
    except Exception as exc:
        return False, str(exc)


def _build_context(docs: list) -> str:
    return "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}, "
        f"Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    )


def _score_response(text: str) -> int:
    """Heuristic quality score 0-100 based on length, structure, and specificity."""
    if not text or len(text.strip()) < 50:
        return 0
    score = 0
    words = text.split()
    # Length contribution (up to 35 pts)
    score += min(35, len(words) // 8)
    # Structure — bullets, headers, numbered lists (up to 40 pts)
    if re.search(r"^\s*[-•*]\s", text, re.MULTILINE):
        score += 15
    if re.search(r"^#{1,3}\s|\*\*.+\*\*", text, re.MULTILINE):
        score += 15
    if re.search(r"^\s*\d+[.)]\s", text, re.MULTILINE):
        score += 10
    # Specificity — numbers, percentages, currency (up to 25 pts)
    specifics = re.findall(r"\d+[%$]|[$£]\d+|\b\d{4}\b|\d+\.\d+", text)
    score += min(25, len(specifics) * 5)
    return min(100, score)


def _run_single_model(model: str, question: str, context: str) -> dict:
    """Invoke one model and return a result dict with quality metrics."""
    settings = get_settings()
    t0 = time.perf_counter()
    try:
        llm = ChatOllama(model=model, base_url=settings.ollama_base_url, temperature=0.1)
        prompt = _ANALYST_PROMPT.format(context=context, question=question)
        response = llm.invoke([HumanMessage(content=prompt)])
        elapsed = round(time.perf_counter() - t0, 1)
        content = response.content
        return {
            "model": model,
            "response": content,
            "error": None,
            "response_time": elapsed,
            "word_count": len(content.split()),
            "quality_score": _score_response(content),
        }
    except Exception as exc:
        logger.warning("Model %s failed: %s", model, exc)
        return {
            "model": model,
            "response": None,
            "error": str(exc),
            "response_time": round(time.perf_counter() - t0, 1),
            "word_count": 0,
            "quality_score": 0,
        }


def run_multi_model_analysis(
    question: str,
    top_k: int = 5,
    selected_models: list[str] | None = None,
) -> list[dict]:
    """Retrieve context then fan out to selected models in parallel.

    Returns results sorted by quality_score descending.
    """
    models = selected_models if selected_models else get_chat_models()
    retriever = get_retriever(top_k=top_k)
    docs = retriever.invoke(question)

    if not docs:
        return [
            {
                "model": m,
                "response": None,
                "error": "No documents in knowledge base.",
                "response_time": 0,
                "word_count": 0,
                "quality_score": 0,
            }
            for m in models
        ]

    context = _build_context(docs)
    max_workers = min(len(models), 4)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_single_model, m, question, context): m for m in models}
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    return sorted(results, key=lambda x: (-x["quality_score"], x["model"]))


def generate_insights(top_k: int = 10, model: str | None = None) -> dict[str, dict]:
    """Auto-generate structured insights from the knowledge base using a single model."""
    settings = get_settings()
    if model is None:
        model = (get_chat_models() or [settings.ollama_model])[0]
    retriever = get_retriever(top_k=top_k)

    output: dict[str, dict] = {}

    for key, (label, question) in _INSIGHTS_QUESTIONS.items():
        docs = retriever.invoke(question)
        if not docs:
            output[key] = {"label": label, "content": "_No documents in knowledge base._"}
            continue

        context = _build_context(docs)
        try:
            llm = ChatOllama(
                model=model,
                base_url=settings.ollama_base_url,
                temperature=0,
            )
            prompt = _ANALYST_PROMPT.format(context=context, question=question)
            response = llm.invoke([HumanMessage(content=prompt)])
            output[key] = {"label": label, "content": response.content}
        except Exception as exc:
            output[key] = {"label": label, "content": f"_Error: {exc}_"}

    return output
