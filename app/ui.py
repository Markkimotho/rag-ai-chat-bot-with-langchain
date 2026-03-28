"""Streamlit UI — minimal, modern, expert data analyst RAG chatbot."""

import logging
import sys
import tempfile
import uuid
from pathlib import Path

# Ensure project root is on sys.path when run via `streamlit run app/ui.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
import streamlit as st

from app import chain as lcel_chain
from app import graph as langgraph_module
from app.analysis import (
    generate_insights,
    get_chat_models,
    get_installed_models,
    pull_model,
    run_multi_model_analysis,
)
from app.config import get_settings
from app.ingestion import load_and_chunk_pdf
from app.memory import clear_history
from app.vectorstore import clear_vectorstore, get_doc_count, ingest_documents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

SUGGESTED_QUESTIONS = [
    "What are the key findings across all documents?",
    "What are the main risks and challenges identified?",
    "Summarize the most important metrics and data points.",
    "What recommendations or action items are mentioned?",
]

_SECTION_ACCENT = {
    "summary": "#818cf8",
    "key_metrics": "#34d399",
    "risks": "#f87171",
    "recommendations": "#fbbf24",
}

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Analyst AI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

/* ── Inter for text — targeted selectors only, no !important, no span
       Streamlit bundles Material Symbols Rounded locally; do not touch it ── */
html, body,
[data-testid="stMain"],
[data-testid="stSidebar"],
[data-testid="stHeader"],
.stMarkdown, .stButton > button,
.stTextInput input, .stTextArea textarea,
.stSelectbox, .stMultiSelect,
.stSlider, .stRadio,
[data-testid="stChatInput"],
[data-testid="stMetric"],
[data-testid="stCaptionContainer"],
[data-baseweb="tab"], [data-baseweb="select"],
p, h1, h2, h3, h4, h5, h6, li, td, th, label, caption {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer { display: none !important; }
[data-testid="stStatusWidget"] { display: none !important; }

/* ── App background ── */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: #0a0a0a !important;
}
[data-testid="stMain"] .block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1100px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #101012 !important;
    border-right: 1px solid #1e1e22 !important;
}
[data-testid="stSidebarContent"] { padding: 1.25rem 0.875rem; }

/* ── Tabs — underline indicator style ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #1e1e22;
    padding: 0;
    gap: 0;
    margin-bottom: 1.5rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 0 !important;
    color: #52525b;
    font-weight: 500;
    font-size: 0.8rem;
    letter-spacing: 0.01em;
    padding: 10px 18px;
    background: transparent !important;
    border-bottom: 2px solid transparent;
    margin-bottom: -1px;
    transition: color 0.15s;
}
.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: #e4e4e7 !important;
    border-bottom: 2px solid #818cf8 !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: #141416;
    border: 1px solid #1e1e22;
    border-radius: 6px;
    padding: 10px 14px;
}
[data-testid="stMetric"] label {
    color: #52525b !important;
    font-size: 0.65rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="stMetricValue"] {
    color: #e4e4e7 !important;
    font-size: 1.15rem !important;
    font-weight: 600 !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    border-radius: 6px;
    margin-bottom: 6px;
    border: 1px solid #1e1e22;
    background: #141416;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 5px !important;
    font-weight: 500 !important;
    font-size: 0.8rem !important;
    border: 1px solid #2a2a2e !important;
    background: #141416 !important;
    color: #a1a1aa !important;
    transition: background 0.15s, border-color 0.15s, color 0.15s !important;
    box-shadow: none !important;
    padding: 0.35rem 0.75rem !important;
}
.stButton > button:hover {
    background: #1e1e22 !important;
    border-color: #3f3f46 !important;
    color: #e4e4e7 !important;
    transform: none !important;
    box-shadow: none !important;
}
.stButton > button[kind="primary"] {
    background: #4f46e5 !important;
    border-color: #4f46e5 !important;
    color: #fff !important;
}
.stButton > button[kind="primary"]:hover {
    background: #4338ca !important;
    border-color: #4338ca !important;
}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stTextArea > div > textarea {
    background: #141416 !important;
    border: 1px solid #2a2a2e !important;
    border-radius: 5px !important;
    color: #e4e4e7 !important;
    font-size: 0.85rem !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.12) !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] textarea {
    background: #141416 !important;
    border: 1px solid #2a2a2e !important;
    border-radius: 6px !important;
    color: #e4e4e7 !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.12) !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: #141416 !important;
    border: 1px solid #2a2a2e !important;
    border-radius: 5px !important;
    color: #e4e4e7 !important;
}

/* ── Multiselect ── */
[data-testid="stMultiSelect"] > div > div {
    background: #141416 !important;
    border: 1px solid #2a2a2e !important;
    border-radius: 5px !important;
}
[data-testid="stMultiSelect"] [data-baseweb="tag"] {
    background: rgba(79,70,229,0.18) !important;
    border-color: rgba(99,102,241,0.4) !important;
    color: #a5b4fc !important;
    border-radius: 3px !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #141416;
    border: 1px dashed #2a2a2e;
    border-radius: 5px;
}

/* ── Radio ── */
[data-testid="stRadio"] label {
    color: #a1a1aa !important;
    font-size: 0.8rem !important;
}

/* ── Expander ── */
details {
    background: #141416 !important;
    border: 1px solid #1e1e22 !important;
    border-radius: 5px !important;
}
summary { color: #a1a1aa !important; font-size: 0.8rem !important; }

/* ── Caption ── */
[data-testid="stCaptionContainer"] p {
    color: #52525b !important;
    font-size: 0.72rem !important;
}

/* ── Code ── */
code {
    background: #1e1e22 !important;
    color: #a5b4fc !important;
    border-radius: 3px;
    padding: 1px 5px;
    font-size: 0.78rem;
    border: none !important;
}

/* ── Progress ── */
[data-testid="stProgress"] > div > div { background: #4f46e5 !important; }

/* ── Divider ── */
hr { border-color: #1e1e22 !important; margin: 0.75rem 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #2a2a2e; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #3f3f46; }

/* ── Alert ── */
[data-testid="stAlert"] {
    border-radius: 5px !important;
    background: #141416 !important;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _format_sources(sources: list[dict]) -> str:
    if not sources:
        return ""
    seen: set = set()
    lines = []
    for s in sources:
        key = (s.get("source", ""), s.get("page", ""))
        if key not in seen:
            seen.add(key)
            lines.append(f"- {s['source']}, page {s['page']}")
    return "\n\n---\n**Sources:**\n" + "\n".join(lines)


def _ask(question: str, session_id: str, orchestration: str, model: str, top_k: int) -> str:
    try:
        if orchestration == "LangGraph":
            result = langgraph_module.invoke(
                question=question, thread_id=session_id, model=model, top_k=top_k,
            )
        else:
            result = lcel_chain.invoke(
                question=question, session_id=session_id, model=model, top_k=top_k,
            )
        return result["answer"] + _format_sources(result.get("sources", []))
    except Exception as exc:
        logger.exception("Error processing question")
        return f"Error: {exc}"


def _ollama_status() -> tuple[bool, str, int]:
    settings = get_settings()
    try:
        r = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=3)
        r.raise_for_status()
        models_data = r.json().get("models", [])
        chat_models = [m for m in models_data if "embed" not in m["name"]]
        return True, "Connected", len(chat_models)
    except Exception:
        return False, "Offline", 0


# ── Session state ─────────────────────────────────────────────────────────────

_defaults: dict = {
    "session_id": str(uuid.uuid4()),
    "messages": [],
    "insights": None,
    "insights_model": None,
    "analysis_results": None,
    "analysis_question": "",
    "pending_question": None,
    "last_question": None,
    "last_model": None,
    "selected_model": None,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    # ── Brand ──────────────────────────────────────────────────────────────
    st.markdown(
        '<div style="padding:0 0 1rem 0;">'
        '<svg width="18" height="18" viewBox="0 0 18 18" fill="none" '
        'style="vertical-align:-4px;margin-right:6px;" xmlns="http://www.w3.org/2000/svg">'
        '<rect x="1" y="10" width="3" height="7" rx="1" fill="#818cf8"/>'
        '<rect x="7" y="6" width="3" height="11" rx="1" fill="#6366f1"/>'
        '<rect x="13" y="2" width="3" height="15" rx="1" fill="#4f46e5"/>'
        '</svg>'
        '<span style="font-size:0.9rem;font-weight:600;color:#e4e4e7;letter-spacing:-0.02em;">Analyst AI</span>'
        '<div style="font-size:0.68rem;color:#3f3f46;margin-top:3px;padding-left:24px;">RAG · Local · Multi-model</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    _online, _status_text, _model_count = _ollama_status()
    _dot_color = "#4ade80" if _online else "#f87171"
    _glow = f"box-shadow:0 0 5px {_dot_color};" if _online else ""
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:7px;margin-bottom:0.875rem;'
        f'padding:6px 10px;background:#141416;border:1px solid #1e1e22;border-radius:5px;">'
        f'<span style="width:6px;height:6px;border-radius:50%;background:{_dot_color};'
        f'flex-shrink:0;display:inline-block;{_glow}"></span>'
        f'<span style="font-size:0.7rem;color:#a1a1aa;">Ollama {_status_text}</span>'
        f'<span style="font-size:0.65rem;color:#52525b;margin-left:auto;">{_model_count} model{"s" if _model_count != 1 else ""}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Settings section ───────────────────────────────────────────────────
    st.markdown(
        '<div style="display:flex;align-items:center;gap:5px;margin-bottom:5px;">'
        '<svg width="12" height="12" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<circle cx="8" cy="8" r="2.5" stroke="#52525b" stroke-width="1.5"/>'
        '<path d="M8 1v2M8 13v2M1 8h2M13 8h2M3.05 3.05l1.42 1.42M11.54 11.54l1.41 1.41'
        'M3.05 12.95l1.42-1.42M11.54 4.46l1.41-1.41" stroke="#52525b" stroke-width="1.5" stroke-linecap="round"/>'
        '</svg>'
        '<span style="font-size:0.62rem;font-weight:600;color:#3f3f46;letter-spacing:0.09em;text-transform:uppercase;">Settings</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    available_models = get_chat_models()   # always all 4 SUPPORTED_MODELS
    installed_models = get_installed_models()
    _prev = st.session_state.selected_model
    _model_idx = available_models.index(_prev) if _prev in available_models else 0
    model = st.selectbox("Chat model", available_models, index=_model_idx, key="selected_model")

    # ── Install status + pull button ──────────────────────────────────────
    _is_installed = model in installed_models
    if _is_installed:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:5px;margin-top:-6px;margin-bottom:6px;">'
            '<span style="width:6px;height:6px;border-radius:50%;background:#4ade80;display:inline-block;"></span>'
            '<span style="font-size:0.68rem;color:#52525b;">Installed</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:5px;margin-top:-6px;margin-bottom:4px;">'
            '<span style="width:6px;height:6px;border-radius:50%;background:#f87171;display:inline-block;"></span>'
            '<span style="font-size:0.68rem;color:#52525b;">Not installed</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        if st.button(f"Pull {model}", use_container_width=True, key="pull_model_btn"):
            with st.spinner(f"Pulling {model}\u2026 this may take several minutes."):
                _ok, _msg = pull_model(model)
            if _ok:
                st.success(_msg)
                st.rerun()
            else:
                st.error(f"Pull failed: {_msg}")

    orchestration = st.radio("Mode", ["LangChain", "LangGraph"], horizontal=True)
    top_k = st.slider("Retrieved chunks", 1, 15, 5)

    st.divider()

    # ── Knowledge Base section ─────────────────────────────────────────────
    st.markdown(
        '<div style="display:flex;align-items:center;gap:5px;margin-bottom:6px;">'
        '<svg width="12" height="12" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<ellipse cx="8" cy="4.5" rx="5.5" ry="2" stroke="#52525b" stroke-width="1.4"/>'
        '<path d="M2.5 4.5v3c0 1.1 2.46 2 5.5 2s5.5-.9 5.5-2v-3" stroke="#52525b" stroke-width="1.4"/>'
        '<path d="M2.5 7.5v3c0 1.1 2.46 2 5.5 2s5.5-.9 5.5-2v-3" stroke="#52525b" stroke-width="1.4"/>'
        '</svg>'
        '<span style="font-size:0.62rem;font-weight:600;color:#3f3f46;letter-spacing:0.09em;text-transform:uppercase;">Knowledge Base</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    doc_count = get_doc_count()
    st.markdown(
        f'<div style="font-size:0.7rem;color:#52525b;margin-bottom:8px;">'
        f'{doc_count} chunk{"s" if doc_count != 1 else ""} indexed</div>',
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "PDFs", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed",
    )

    if st.button("Upload & Index", type="primary", use_container_width=True):
        if uploaded_files:
            _bar = st.progress(0, text="Indexing...")
            _results = []
            for _i, _f in enumerate(uploaded_files):
                _bar.progress((_i + 1) / len(uploaded_files), text=f"Processing {_f.name}...")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as _tmp:
                    _tmp.write(_f.read())
                    _tmp_path = Path(_tmp.name)
                try:
                    _chunks = load_and_chunk_pdf(_tmp_path)
                    _n = ingest_documents(_chunks)
                    _results.append(f"{_f.name}: {_n} chunks")
                except Exception as _exc:
                    _results.append(f"{_f.name}: failed — {_exc}")
            _bar.empty()
            for _r in _results:
                st.caption(_r)
            st.session_state.insights = None
        else:
            st.warning("Select at least one PDF first.")

    _col_a, _col_b = st.columns(2)
    with _col_a:
        if st.button("Clear KB", use_container_width=True):
            clear_vectorstore()
            st.session_state.insights = None
            st.session_state.analysis_results = None
            st.rerun()
    with _col_b:
        if st.button("Clear Chat", use_container_width=True):
            clear_history(st.session_state.session_id)
            st.session_state.messages = []
            st.rerun()

    st.divider()
    st.markdown(
        f'<div style="font-size:0.65rem;color:#2a2a2e;">Session {st.session_state.session_id[:8]}</div>',
        unsafe_allow_html=True,
    )

# ── Header ────────────────────────────────────────────────────────────────────

_SVG_BARS = (
    '<svg width="18" height="18" viewBox="0 0 18 18" fill="none" '
    'style="vertical-align:-4px;margin-right:8px;" xmlns="http://www.w3.org/2000/svg">'
    '<rect x="1" y="10" width="3" height="7" rx="1" fill="#818cf8"/>'
    '<rect x="7" y="6" width="3" height="11" rx="1" fill="#6366f1"/>'
    '<rect x="13" y="2" width="3" height="15" rx="1" fill="#4f46e5"/>'
    '</svg>'
)
st.markdown(
    f'<h1 style="font-size:1.1rem;font-weight:600;color:#e4e4e7;letter-spacing:-0.03em;margin:0 0 3px 0;">'
    f'{_SVG_BARS}Analyst AI</h1>'
    '<p style="font-size:0.78rem;color:#52525b;margin:0 0 1rem 0;padding-left:26px;">'
    'Upload documents, analyze with multiple models, and extract structured insights.</p>',
    unsafe_allow_html=True,
)

_c1, _c2, _c3, _c4 = st.columns(4)
_c1.metric("Messages", len(st.session_state.messages))
_c2.metric("Models", f"{len(installed_models)}/{len(available_models)}")
_c3.metric("Indexed chunks", doc_count)
_c4.metric("Top-K", top_k)

st.markdown("<div style='margin-bottom:0.25rem'></div>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_chat, tab_analyze, tab_insights = st.tabs(["Chat", "Multi-Model Analysis", "Insights"])

# ════════════════════════════ TAB 1 — CHAT ════════════════════════════════════

with tab_chat:
    if not st.session_state.messages:
        st.markdown(
            '<div style="font-size:0.62rem;font-weight:600;color:#3f3f46;letter-spacing:0.09em;'
            'text-transform:uppercase;margin-bottom:8px;">Quick start</div>',
            unsafe_allow_html=True,
        )
        _sq_cols = st.columns(2)
        for _i, _q in enumerate(SUGGESTED_QUESTIONS):
            with _sq_cols[_i % 2]:
                if st.button(_q, key=f"sq_{_i}", use_container_width=True):
                    st.session_state.pending_question = _q
        st.markdown("<div style='margin-bottom:1rem'></div>", unsafe_allow_html=True)

    if st.session_state.pending_question:
        _pq = st.session_state.pending_question
        st.session_state.pending_question = None
        st.session_state.messages.append({"role": "user", "content": _pq})
        with st.spinner(f"Analyzing with {model}..."):
            _presp = _ask(_pq, st.session_state.session_id, orchestration, model, top_k)
        st.session_state.messages.append({"role": "assistant", "content": _presp, "model": model})
        st.session_state.last_question = _pq
        st.session_state.last_model = model
        st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("model"):
                st.markdown(
                    f'<div style="font-size:0.62rem;color:#3f3f46;margin-bottom:4px;">{msg["model"]}</div>',
                    unsafe_allow_html=True,
                )
            st.markdown(msg["content"])

    # ── Regenerate bar ────────────────────────────────────────────────────────
    if (
        st.session_state.messages
        and st.session_state.messages[-1]["role"] == "assistant"
        and st.session_state.last_question
    ):
        _last_m = st.session_state.last_model or model
        _retry_models = [m for m in available_models if m != _last_m]
        if _retry_models:
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:6px;margin-top:2px;">'
                f'<span style="font-size:0.62rem;color:#3f3f46;">Not satisfied? Retry with:</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            _rcols = st.columns(len(_retry_models))
            for _ci, _rm in enumerate(_retry_models):
                with _rcols[_ci]:
                    if st.button(_rm, key=f"retry_{_rm}", use_container_width=True):
                        st.session_state.messages.pop()
                        with st.spinner(f"Regenerating with {_rm}..."):
                            _new = _ask(
                                st.session_state.last_question,
                                st.session_state.session_id,
                                orchestration,
                                _rm,
                                top_k,
                            )
                        st.session_state.messages.append(
                            {"role": "assistant", "content": _new, "model": _rm}
                        )
                        st.session_state.last_model = _rm
                        st.rerun()

    if prompt := st.chat_input(f"Ask a question ({model})...", key="chat_input"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner(f"Analyzing with {model}..."):
                _answer = _ask(prompt, st.session_state.session_id, orchestration, model, top_k)
            st.markdown(_answer)
        st.session_state.messages.append({"role": "assistant", "content": _answer, "model": model})
        st.session_state.last_question = prompt
        st.session_state.last_model = model

# ═════════════════════════ TAB 2 — MULTI-MODEL ANALYSIS ══════════════════════

with tab_analyze:
    st.markdown(
        '<div style="font-size:0.78rem;color:#52525b;margin-bottom:1rem;">'
        'Select models, pose a question, and compare how each analyzes your documents.</div>',
        unsafe_allow_html=True,
    )

    selected_models = st.multiselect(
        "Models to compare",
        options=available_models,
        default=[m for m in available_models if m in installed_models] or available_models[:1],
        placeholder="Choose models...",
        help="Select which models to include in the parallel analysis run.",
    )

    _not_installed = [m for m in selected_models if m not in installed_models]
    if _not_installed:
        st.warning(
            f"Not installed: **{', '.join(_not_installed)}**. "
            "Pull them from the sidebar before running analysis.",
            icon=":material/warning:",
        )

    analysis_q = st.text_area(
        "Question",
        placeholder="e.g. What are the primary financial risks identified in these reports?",
        height=80,
        label_visibility="collapsed",
        key="analysis_q_input",
    )
    st.caption("Tip: specific, focused questions yield the most actionable results.")

    _run_col, _k_col = st.columns([3, 1])
    with _k_col:
        analysis_top_k = st.slider(
            "Top-K", 3, 15, top_k, key="analysis_topk", label_visibility="collapsed",
        )
        st.caption(f"Top-K: {analysis_top_k}")
    with _run_col:
        _n_models = len(selected_models)
        _run_label = f"Run across {_n_models} model{'s' if _n_models != 1 else ''}"
        run_btn = st.button(
            _run_label,
            type="primary",
            use_container_width=True,
            disabled=not analysis_q.strip() or not selected_models or bool(_not_installed),
        )

    if run_btn and analysis_q.strip() and selected_models:
        with st.spinner(f"Running across {len(selected_models)} model(s)..."):
            _results = run_multi_model_analysis(
                analysis_q.strip(), top_k=analysis_top_k, selected_models=selected_models,
            )
        st.session_state.analysis_results = _results
        st.session_state.analysis_question = analysis_q.strip()

    if st.session_state.analysis_results:
        _res = st.session_state.analysis_results
        _ok = [r for r in _res if not r.get("error")]
        _err = [r for r in _res if r.get("error")]
        _best = max(_ok, key=lambda x: x.get("quality_score", 0)) if _ok else None

        st.markdown(
            f'<div style="font-size:0.7rem;color:#52525b;margin:0.75rem 0 0.5rem;">'
            f'Results for: <em>"{st.session_state.analysis_question}"</em></div>',
            unsafe_allow_html=True,
        )

        _ma, _mb, _mc, _md = st.columns(4)
        _ma.metric("Models run", len(_res))
        _mb.metric("Succeeded", len(_ok))
        _mc.metric("Failed", len(_err))
        _md.metric("Best model", _best["model"].split(":")[0] if _best else "—")

        st.markdown("<div style='margin-top:0.75rem'></div>", unsafe_allow_html=True)

        for result in _res:
            _is_best = _best and result["model"] == _best["model"]
            _star = " \u2605" if _is_best else ""
            with st.expander(
                f"{result['model']}{_star}",
                expanded=bool(_is_best and not result.get("error")),
            ):
                if result.get("error"):
                    st.markdown(
                        f'<div style="color:#f87171;font-size:0.82rem;padding:4px 0;">'
                        f'{result["error"]}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    _r1, _r2, _r3, _ = st.columns([1, 1, 1, 5])
                    _r1.metric("Words", result.get("word_count", 0))
                    _r2.metric("Time", f'{result.get("response_time", 0)}s')
                    _r3.metric("Score", f'{result.get("quality_score", 0)}/100')
                    st.markdown("<div style='margin-bottom:0.5rem'></div>", unsafe_allow_html=True)
                    st.markdown(result["response"])

# ═════════════════════════ TAB 3 — INSIGHTS ══════════════════════════════════

with tab_insights:
    st.markdown(
        '<div style="font-size:0.78rem;color:#52525b;margin-bottom:1rem;">'
        'Auto-extract an executive summary, key metrics, risk analysis, and recommendations '
        'from your entire knowledge base using your chosen model.</div>',
        unsafe_allow_html=True,
    )

    _ig1, _ig2, _ig3 = st.columns([2, 3, 1], vertical_alignment="bottom")
    with _ig1:
        insights_model = st.selectbox(
            "Analysis model",
            available_models,
            index=0,
            key="insights_model_select",
        )
    with _ig2:
        gen_btn = st.button("Generate Insights", type="primary", use_container_width=True)
    with _ig3:
        if st.button("Reset", use_container_width=True):
            st.session_state.insights = None
            st.rerun()

    if gen_btn:
        with st.spinner(f"Generating with {insights_model}... this may take a moment."):
            st.session_state.insights = generate_insights(
                top_k=min(top_k + 5, 15), model=insights_model,
            )
            st.session_state.insights_model = insights_model

    if st.session_state.insights:
        _used_model = st.session_state.get("insights_model", "")
        if _used_model:
            st.markdown(
                f'<div style="font-size:0.68rem;color:#3f3f46;margin-bottom:0.75rem;">'
                f'Generated with {_used_model}</div>',
                unsafe_allow_html=True,
            )
        for key, data in st.session_state.insights.items():
            _accent = _SECTION_ACCENT.get(key, "#52525b")
            st.markdown(
                f'<div style="display:flex;align-items:center;padding:10px 14px;'
                f'background:#141416;border:1px solid #1e1e22;'
                f'border-left:3px solid {_accent};border-radius:0 5px 5px 0;margin-bottom:2px;">'
                f'<span style="font-size:0.85rem;font-weight:500;color:#d4d4d8;">{data["label"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            with st.expander("View analysis", expanded=(key == "summary")):
                st.markdown(data["content"])
    elif not gen_btn:
        st.markdown(
            '<div style="background:#141416;border:1px solid #1e1e22;border-radius:6px;'
            'padding:3rem;text-align:center;">'
            '<div style="font-size:0.85rem;font-weight:500;color:#3f3f46;margin-bottom:4px;">'
            'No insights generated yet</div>'
            '<div style="font-size:0.72rem;color:#2a2a2e;">'
            'Upload documents via the sidebar, choose a model, then click Generate Insights.</div>'
            '</div>',
            unsafe_allow_html=True,
        )

