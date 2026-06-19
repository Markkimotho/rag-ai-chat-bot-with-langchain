"""Quiz generation, answer validation, and concept explanation from the knowledge base."""

import json
import logging
import re
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from app.config import get_settings
from app.retriever import get_retriever

logger = logging.getLogger(__name__)

# ── Prompts ───────────────────────────────────────────────────────────────────

_MCQ_PROMPT = """\
You are an expert exam question writer. Using ONLY the content below, generate {n} \
multiple-choice questions to help someone prepare for: {exam_type}.

Topic focus: {topic}
Difficulty: {difficulty}

Rules:
- Every question must be answerable from the provided content
- 4 options per question labeled A, B, C, D — exactly one is correct
- Explanation must quote or paraphrase the relevant content
- source: the filename or URL the answer comes from

Content:
{context}

Respond with ONLY a valid JSON array — no markdown fences, no extra text:
[{{"type":"mcq","question":"...","options":{{"A":"...","B":"...","C":"...","D":"..."}},"correct":"A","explanation":"...","source":"...","difficulty":"{difficulty}"}}]"""

_TRUE_FALSE_PROMPT = """\
You are an expert exam question writer. Using ONLY the content below, generate {n} \
true/false questions to help someone prepare for: {exam_type}.

Topic focus: {topic}
Difficulty: {difficulty}

Rules:
- Each statement must be factually verifiable from the content
- Aim for ~50% true, ~50% false
- False statements must be plausibly wrong, not obviously wrong
- Explanation must reference the relevant part of the content

Content:
{context}

Respond with ONLY a valid JSON array — no markdown fences, no extra text:
[{{"type":"true_false","question":"...","correct":true,"explanation":"...","source":"...","difficulty":"{difficulty}"}}]"""

_SHORT_ANSWER_PROMPT = """\
You are an expert exam question writer. Using ONLY the content below, generate {n} \
short-answer questions to help someone prepare for: {exam_type}.

Topic focus: {topic}
Difficulty: {difficulty}

Rules:
- Questions should require 2-4 sentence answers
- For technical interviews: include conceptual + practical questions
- For job interviews: include behavioral (STAR-method) questions
- key_concepts must all appear in the sample_answer

Content:
{context}

Respond with ONLY a valid JSON array — no markdown fences, no extra text:
[{{"type":"short_answer","question":"...","sample_answer":"...","key_concepts":["..."],"source":"...","difficulty":"{difficulty}"}}]"""

_MIXED_PROMPT = """\
You are an expert exam question writer. Using ONLY the content below, generate {n} \
questions of mixed types to help someone prepare for: {exam_type}.

Topic focus: {topic}
Difficulty: {difficulty}

Include MCQ, true/false, and short-answer questions in roughly equal proportion.

MCQ schema:    {{"type":"mcq","question":"...","options":{{"A":"...","B":"...","C":"...","D":"..."}},"correct":"A","explanation":"...","source":"...","difficulty":"..."}}
TF schema:     {{"type":"true_false","question":"...","correct":true,"explanation":"...","source":"...","difficulty":"..."}}
SA schema:     {{"type":"short_answer","question":"...","sample_answer":"...","key_concepts":["..."],"source":"...","difficulty":"..."}}

Content:
{context}

Respond with ONLY a valid JSON array — no markdown fences, no extra text."""

_VALIDATE_PROMPT = """\
Grade this student answer fairly and precisely.

Question: {question}
Reference answer / sample: {correct}
Student answer: {student_answer}

Context excerpt (for short-answer scoring):
{context}

Scoring criteria (short answer only):
- Key concept coverage: 50 pts
- Factual accuracy: 30 pts
- Completeness: 20 pts

Respond with ONLY valid JSON — no markdown:
{{"score":85,"is_correct":true,"feedback":"...","key_missed":["..."],"hint":"Review the section about..."}}"""

_EXPLAIN_PROMPT = """\
Based ONLY on the following content, give a clear and thorough explanation of: {concept}

Include: key definitions, how it works, real examples from the content, and common \
misconceptions if present. If the content does not fully cover this topic, say so.

Content:
{context}"""

# ── Helpers ───────────────────────────────────────────────────────────────────


def _extract_json(text: str) -> Any:
    """Robustly extract JSON from an LLM response that may contain extra text."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for pattern in [r"\[[\s\S]*\]", r"\{[\s\S]*\}"]:
        match = re.search(pattern, text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                continue
    raise ValueError(f"No valid JSON in response (first 300 chars): {text[:300]}")


def _build_context(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}, "
        f"Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    )


def _get_llm(model: str | None = None, temperature: float = 0.3) -> ChatOllama:
    settings = get_settings()
    return ChatOllama(
        model=model or settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=temperature,
    )


# ── Public API ────────────────────────────────────────────────────────────────


def generate_questions(
    topic: str,
    question_type: str = "mcq",
    n: int = 5,
    difficulty: str = "medium",
    exam_type: str = "general exam",
    model: str | None = None,
    top_k: int = 8,
) -> list[dict]:
    """Generate quiz questions from the knowledge base on a given topic.

    question_type: "mcq" | "true_false" | "short_answer" | "mixed"
    Returns a list of question dicts (at most n).
    Returns [] when the knowledge base has no relevant content.
    """
    retriever = get_retriever(top_k=top_k)
    docs = retriever.invoke(topic)
    if not docs:
        logger.warning("No docs found for topic %r", topic)
        return []

    context = _build_context(docs)
    prompt_map = {
        "mcq": _MCQ_PROMPT,
        "true_false": _TRUE_FALSE_PROMPT,
        "short_answer": _SHORT_ANSWER_PROMPT,
        "mixed": _MIXED_PROMPT,
    }
    prompt_template = prompt_map.get(question_type, _MCQ_PROMPT)
    prompt = prompt_template.format(
        n=n,
        topic=topic,
        difficulty=difficulty,
        exam_type=exam_type,
        context=context,
    )

    llm = _get_llm(model, temperature=0.3)
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        questions = _extract_json(response.content)
        if isinstance(questions, dict):
            questions = [questions]
        if not isinstance(questions, list):
            logger.error("LLM returned non-list JSON for questions: %r", type(questions))
            return []
        return questions[:n]
    except Exception as exc:
        logger.error("generate_questions failed: %s", exc)
        return []


def validate_answer(
    question: str,
    correct_answer: Any,
    student_answer: str,
    question_type: str = "short_answer",
    model: str | None = None,
    context_docs: list[Document] | None = None,
) -> dict:
    """Grade a student answer.

    Returns: {"score": 0-100, "is_correct": bool, "feedback": str, "key_missed": list, "hint": str}
    MCQ and True/False are graded deterministically; short answer uses the LLM.
    """
    if question_type in ("mcq", "true_false"):
        expected = str(correct_answer).strip().upper()
        given = student_answer.strip().upper()
        is_correct = expected == given
        return {
            "score": 100 if is_correct else 0,
            "is_correct": is_correct,
            "feedback": "Correct!" if is_correct else f"Incorrect. The correct answer is: {correct_answer}",
            "key_missed": [] if is_correct else [str(correct_answer)],
            "hint": "",
        }

    context = _build_context(context_docs[:3]) if context_docs else ""
    llm = _get_llm(model, temperature=0)
    prompt = _VALIDATE_PROMPT.format(
        question=question,
        correct=str(correct_answer)[:1000],
        student_answer=student_answer,
        context=context[:2000],
    )
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        result = _extract_json(response.content)
        result.setdefault("score", 0)
        result.setdefault("is_correct", int(result.get("score", 0)) >= 70)
        result.setdefault("feedback", "")
        result.setdefault("key_missed", [])
        result.setdefault("hint", "")
        return result
    except Exception as exc:
        logger.error("validate_answer failed: %s", exc)
        return {
            "score": 0,
            "is_correct": False,
            "feedback": f"Could not grade answer: {exc}",
            "key_missed": [],
            "hint": "",
        }


def explain_concept(concept: str, model: str | None = None, top_k: int = 6) -> str:
    """Return a detailed explanation of a concept from the knowledge base."""
    retriever = get_retriever(top_k=top_k)
    docs = retriever.invoke(concept)
    if not docs:
        return f"No information found about '{concept}' in the knowledge base."

    context = _build_context(docs)
    llm = _get_llm(model, temperature=0)
    prompt = _EXPLAIN_PROMPT.format(concept=concept, context=context)
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as exc:
        return f"Error generating explanation: {exc}"
