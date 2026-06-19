"""LangGraph tool-calling agent for interactive exam preparation.

The agent orchestrates five tools:
  - search_and_add_to_kb  : scrape + ingest web content on any topic
  - make_quiz             : generate questions from the knowledge base
  - check_answer          : grade a student answer with feedback
  - explain               : deep-dive explanation from knowledge base
  - list_topics           : summarise what's in the knowledge base

A module-level MemorySaver checkpointer means conversation threads survive
across multiple Streamlit reruns as long as the process stays alive.
"""

import json
import logging

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from app.config import get_settings
from app.quiz import explain_concept, generate_questions, validate_answer
from app.retriever import get_retriever
from app.scraper import search_and_scrape
from app.vectorstore import get_doc_count, ingest_documents

logger = logging.getLogger(__name__)

# ── Module-level state ────────────────────────────────────────────────────────

# Single shared checkpointer — keeps thread history across Streamlit reruns.
_checkpointer = MemorySaver()

# ── Tools ─────────────────────────────────────────────────────────────────────


@tool
def search_and_add_to_kb(topic: str, num_results: int = 3) -> str:
    """Search the web for a topic and add the found content to the knowledge base.

    Use when the user wants to study something not yet in the knowledge base,
    or when they say 'scrape', 'find', 'search', or 'look up' content online.

    Args:
        topic: The topic to search for (e.g. "Python async/await", "System Design basics")
        num_results: Number of web pages to scrape (1-5, default 3)
    """
    try:
        chunks = search_and_scrape(topic, num_results=max(1, min(num_results, 5)))
        if not chunks:
            return (
                f"No content found for '{topic}'. "
                "Try a more specific search term or upload a document instead."
            )
        count = ingest_documents(chunks)
        return (
            f"Added {count} chunks about '{topic}' to the knowledge base "
            f"from {num_results} web page(s). You can now generate quiz questions on this topic."
        )
    except Exception as exc:
        logger.exception("search_and_add_to_kb failed")
        return f"Web search failed: {exc}"


@tool
def make_quiz(
    topic: str,
    question_type: str = "mcq",
    n: int = 5,
    difficulty: str = "medium",
    exam_type: str = "general exam",
) -> str:
    """Generate quiz questions from the knowledge base on a given topic.

    Use when the user asks to be quizzed, tested, or wants practice questions.

    Args:
        topic: Subject/topic to generate questions about
        question_type: "mcq" (multiple choice), "true_false", "short_answer", or "mixed"
        n: Number of questions (1-20, default 5)
        difficulty: "easy", "medium", "hard", or "mixed"
        exam_type: e.g. "general exam", "technical interview", "job interview", "certification"
    """
    questions = generate_questions(
        topic=topic,
        question_type=question_type,
        n=max(1, min(n, 20)),
        difficulty=difficulty,
        exam_type=exam_type,
    )
    if not questions:
        return (
            f"Could not generate questions for '{topic}'. "
            "The knowledge base may lack content on this topic. "
            "Try calling search_and_add_to_kb first."
        )
    return json.dumps(questions, indent=2)


@tool
def check_answer(
    question: str,
    correct_answer: str,
    student_answer: str,
    question_type: str = "short_answer",
) -> str:
    """Grade a student's answer and return detailed feedback.

    Use after presenting a question to evaluate the student's response.

    Args:
        question: The full question text
        correct_answer: The correct answer or reference sample answer
        student_answer: What the student answered
        question_type: "mcq", "true_false", or "short_answer"
    """
    result = validate_answer(
        question=question,
        correct_answer=correct_answer,
        student_answer=student_answer,
        question_type=question_type,
    )
    return json.dumps(result)


@tool
def explain(concept: str) -> str:
    """Get a detailed explanation of a concept from the knowledge base.

    Use when the user says 'explain', 'what is', 'describe', or 'help me understand'.

    Args:
        concept: The concept, term, or topic to explain in detail
    """
    return explain_concept(concept)


@tool
def list_topics() -> str:
    """List the main topics available in the current knowledge base.

    Use when the user asks what's available to study, what topics exist,
    or what they can be quizzed on.
    """
    count = get_doc_count()
    if count == 0:
        return (
            "The knowledge base is empty. You can:\n"
            "1. Upload PDF/TXT documents using the sidebar\n"
            "2. Ask me to search the web for a topic (e.g., 'search for Python basics')"
        )
    retriever = get_retriever(top_k=20)
    try:
        docs = retriever.invoke("main topics")
        sources = sorted({doc.metadata.get("source", "unknown") for doc in docs})
        source_list = ", ".join(sources[:12])
        suffix = " (and more...)" if len(sources) > 12 else ""
        return (
            f"Knowledge base: {count} indexed chunks.\n"
            f"Sources: {source_list}{suffix}"
        )
    except Exception:
        return f"Knowledge base: {count} indexed chunks."


# ── Agent ─────────────────────────────────────────────────────────────────────

_TOOLS = [search_and_add_to_kb, make_quiz, check_answer, explain, list_topics]

_SYSTEM_PROMPT = """\
You are an expert exam and interview preparation tutor. Your job is to help \
students study effectively for any topic.

Available tools:
- search_and_add_to_kb : scrape web content on any topic and add it to your knowledge base
- make_quiz            : generate quiz questions (MCQ, true/false, short answer) from knowledge base
- check_answer         : grade a student's answer with detailed feedback
- explain              : give detailed explanations of concepts from knowledge base
- list_topics          : see what's already in the knowledge base

How to help:
1. When a user asks to study a topic, call list_topics first to check coverage.
2. If the topic isn't covered, call search_and_add_to_kb automatically — don't ask.
3. Present questions clearly, one at a time, with a numbered counter (Q3/10).
4. For MCQ: show all 4 options labeled A, B, C, D on separate lines.
5. For True/False: present as a clear statement for the student to evaluate.
6. After receiving an answer, call check_answer and relay the feedback.
7. After 100% wrong answers, offer an explanation using explain.
8. At the end of a quiz session, summarise: score, weak areas, what to review.

Formatting:
- Use **bold** for question text and option labels.
- Use ✅ for correct, ❌ for incorrect answers.
- Keep feedback concise and encouraging.
- Show score as X/Y (e.g., "Score: 7/10").
"""


def invoke_agent(
    message: str,
    thread_id: str = "default",
    model: str | None = None,
) -> str:
    """Send a message to the quiz agent and return its text response.

    The shared _checkpointer keeps conversation state across calls with
    the same thread_id for as long as the process is alive.
    """
    settings = get_settings()
    llm = ChatOllama(
        model=model or settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0.2,
    )
    agent = create_react_agent(
        llm,
        _TOOLS,
        state_modifier=_SYSTEM_PROMPT,
        checkpointer=_checkpointer,
    )
    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=message)]},
            config={"configurable": {"thread_id": thread_id}},
        )
        messages = result.get("messages", [])
        return messages[-1].content if messages else "No response from agent."
    except Exception as exc:
        logger.exception("Agent invocation failed")
        return f"Agent error: {exc}"
