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

# Section accent colors — will be set dynamically based on active theme
_SECTION_ACCENTS_DARK = {
    "summary": "#818cf8",
    "key_metrics": "#34d399",
    "risks": "#f87171",
    "recommendations": "#fbbf24",
}

_SECTION_ACCENTS_LIGHT = {
    "summary": "#5856d6",
    "key_metrics": "#17a2f7",
    "risks": "#ff3b30",
    "recommendations": "#ff9500",
}

# ── Theme Configuration ───────────────────────────────────────────────────────

_THEMES = {
    "dark": {
        "name": "Dark",
        "bg_primary": "#0a0a0a",
        "bg_secondary": "#101012",
        "bg_tertiary": "#141416",
        "bg_hover": "#1e1e22",
        "bg_input": "#141416",
        "bg_border": "#1e1e22",
        "text_primary": "#e4e4e7",
        "text_secondary": "#a1a1aa",
        "text_tertiary": "#52525b",
        "text_disabled": "#3f3f46",
        "text_placeholder": "#2a2a2e",
        "accent": "#818cf8",
        "accent_hover": "#6366f1",
        "accent_active": "#4f46e5",
        "accent_dark": "#4338ca",
        "success": "#4ade80",
        "error": "#f87171",
        "warning": "#fbbf24",
        "info": "#34d399",
        "border": "#1px solid #1e1e22",
        "tag_bg": "rgba(79,70,229,0.18)",
        "tag_border": "rgba(99,102,241,0.4)",
        "tag_text": "#a5b4fc",
    },
    "light": {
        "name": "Light",
        "bg_primary": "#fafafa",
        "bg_secondary": "#f3f3f7",
        "bg_tertiary": "#ebebf0",
        "bg_hover": "#e0e0e8",
        "bg_input": "#f5f5f9",
        "bg_border": "#d9d9e3",
        "text_primary": "#1a1a1a",
        "text_secondary": "#404040",
        "text_tertiary": "#666666",
        "text_disabled": "#999999",
        "text_placeholder": "#b0b0b0",
        "accent": "#5856d6",
        "accent_hover": "#6f6cdf",
        "accent_active": "#8985f2",
        "accent_dark": "#3c3aa8",
        "success": "#34c759",
        "error": "#ff3b30",
        "warning": "#ff9500",
        "info": "#17a2f7",
        "border": "1px solid #d9d9e3",
        "tag_bg": "rgba(88,86,214,0.12)",
        "tag_border": "rgba(88,86,214,0.25)",
        "tag_text": "#3c3aa8",
    },
}

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Analyst AI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Dynamic Theme CSS Generator
# ---------------------------------------------------------------------------


def _get_system_theme_preference() -> str:
    """Detect system theme preference via browser CSS media query."""
    # This is evaluated client-side, we default to 'dark' on server
    return "dark"


def _generate_css(theme: str) -> str:
    """Generate CSS for the given theme."""
    colors = _THEMES.get(theme, _THEMES["dark"])
    return f"""
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
p, h1, h2, h3, h4, h5, h6, li, td, th, label, caption {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer {{ display: none !important; }}
[data-testid="stStatusWidget"] {{ display: none !important; }}

/* ── App background ── */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {{
    background: {colors['bg_primary']} !important;
}}
[data-testid="stMain"] .block-container {{
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1100px !important;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: {colors['bg_secondary']} !important;
    border-right: {colors['border']} !important;
}}
[data-testid="stSidebarContent"] {{ padding: 1.25rem 0.875rem; }}

/* ── Tabs — underline indicator style ── */
.stTabs [data-baseweb="tab-list"] {{
    background: transparent !important;
    border-bottom: 1px solid {colors['bg_border']};
    padding: 0;
    gap: 0;
    margin-bottom: 1.5rem;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 0 !important;
    color: {colors['text_tertiary']};
    font-weight: 500;
    font-size: 0.8rem;
    letter-spacing: 0.01em;
    padding: 10px 18px;
    background: transparent !important;
    border-bottom: 2px solid transparent;
    margin-bottom: -1px;
    transition: color 0.15s;
}}
.stTabs [aria-selected="true"] {{
    background: transparent !important;
    color: {colors['text_primary']} !important;
    border-bottom: 2px solid {colors['accent']} !important;
}}
.stTabs [data-baseweb="tab-highlight"] {{ display: none !important; }}

/* ── Metrics ── */
[data-testid="stMetric"] {{
    background: {colors['bg_tertiary']};
    border: 1px solid {colors['bg_border']};
    border-radius: 6px;
    padding: 10px 14px;
}}
[data-testid="stMetric"] label {{
    color: {colors['text_tertiary']} !important;
    font-size: 0.65rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}
[data-testid="stMetricValue"] {{
    color: {colors['text_primary']} !important;
    font-size: 1.15rem !important;
    font-weight: 600 !important;
}}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {{
    border-radius: 6px;
    margin-bottom: 6px;
    border: 1px solid {colors['bg_border']};
    background: {colors['bg_tertiary']};
}}

/* ── Buttons ── */
.stButton > button {{
    border-radius: 5px !important;
    font-weight: 500 !important;
    font-size: 0.8rem !important;
    border: 1px solid {colors['bg_border']} !important;
    background: {colors['bg_tertiary']} !important;
    color: {colors['text_secondary']} !important;
    transition: background 0.15s, border-color 0.15s, color 0.15s !important;
    box-shadow: none !important;
    padding: 0.35rem 0.75rem !important;
}}
.stButton > button:hover {{
    background: {colors['bg_hover']} !important;
    border-color: {colors['text_disabled']} !important;
    color: {colors['text_primary']} !important;
    transform: none !important;
    box-shadow: none !important;
}}
.stButton > button[kind="primary"] {{
    background: {colors['accent']} !important;
    border-color: {colors['accent']} !important;
    color: #fff !important;
}}
.stButton > button[kind="primary"]:hover {{
    background: {colors['accent_dark']} !important;
    border-color: {colors['accent_dark']} !important;
}}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stTextArea > div > textarea {{
    background: {colors['bg_input']} !important;
    border: 1px solid {colors['bg_border']} !important;
    border-radius: 5px !important;
    color: {colors['text_primary']} !important;
    font-size: 0.85rem !important;
}}
.stTextInput > div > div > input::placeholder,
.stTextArea > div > textarea::placeholder {{
    color: {colors['text_disabled']} !important;
    opacity: 0.7 !important;
}}
.stTextInput > div > div > input::-webkit-input-placeholder,
.stTextArea > div > textarea::-webkit-input-placeholder {{
    color: {colors['text_disabled']} !important;
    opacity: 0.7 !important;
}}
.stTextInput > div > div > input:focus,
.stTextArea > div > textarea:focus {{
    border-color: {colors['accent']} !important;
    box-shadow: 0 0 0 2px {colors['tag_bg']} !important;
}}

/* ── Chat input ── */
[data-testid="stChatInput"] textarea {{
    background: {colors['bg_input']} !important;
    border: 1px solid {colors['bg_border']} !important;
    border-radius: 6px !important;
    color: {colors['text_primary']} !important;
}}
[data-testid="stChatInput"] textarea::placeholder {{
    color: {colors['text_disabled']} !important;
    opacity: 0.7 !important;
}}
[data-testid="stChatInput"] textarea:focus {{
    border-color: {colors['accent']} !important;
    box-shadow: 0 0 0 2px {colors['tag_bg']} !important;
}}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {{
    background: {colors['bg_input']} !important;
    border: 1px solid {colors['bg_border']} !important;
    border-radius: 5px !important;
    color: {colors['text_primary']} !important;
}}
[data-testid="stSelectbox"] input {{
    background: {colors['bg_input']} !important;
    color: {colors['text_primary']} !important;
}}
[data-testid="stSelectbox"] input::placeholder {{
    color: {colors['text_disabled']} !important;
    opacity: 0.7 !important;
}}
[data-testid="stSelectbox"] [role="listbox"] {{
    background: {colors['bg_secondary']} !important;
    border: 1px solid {colors['bg_border']} !important;
}}
[data-testid="stSelectbox"] [role="option"] {{
    color: {colors['text_primary']} !important;
    background: {colors['bg_tertiary']} !important;
}}
[data-testid="stSelectbox"] [role="option"][aria-selected="true"] {{
    background: {colors['accent']} !important;
    color: #fff !important;
}}
[data-testid="stSelectbox"] [role="option"]:hover {{
    background: {colors['bg_hover']} !important;
}}

/* ── Multiselect ── */
[data-testid="stMultiSelect"] > div > div {{
    background: {colors['bg_input']} !important;
    border: 1px solid {colors['bg_border']} !important;
    border-radius: 5px !important;
}}
[data-testid="stMultiSelect"] [data-baseweb="tag"] {{
    background: {colors['tag_bg']} !important;
    border-color: {colors['tag_border']} !important;
    color: {colors['tag_text']} !important;
    border-radius: 3px !important;
}}

/* ── File uploader ── */
[data-testid="stFileUploader"] {{
    background: {colors['bg_tertiary']} !important;
    border: 1px dashed {colors['bg_border']} !important;
    border-radius: 5px !important;
}}
[data-testid="stFileUploader"] div {{
    color: {colors['text_secondary']} !important;
}}
[data-testid="stFileUploader"] p {{
    color: {colors['text_tertiary']} !important;
}}
[data-testid="stFileUploader"] [data-upload-state] {{
    color: {colors['text_secondary']} !important;
}}
[data-testid="stFileUploader"] span {{
    color: {colors['text_tertiary']} !important;
}}

/* ── Radio ── */
[data-testid="stRadio"] label {{
    color: {colors['text_secondary']} !important;
    font-size: 0.8rem !important;
}}

/* ── Expander ── */
details {{
    background: {colors['bg_tertiary']} !important;
    border: 1px solid {colors['bg_border']} !important;
    border-radius: 5px !important;
}}
summary {{ color: {colors['text_secondary']} !important; font-size: 0.8rem !important; }}

/* ── Caption ── */
[data-testid="stCaptionContainer"] p {{
    color: {colors['text_tertiary']} !important;
    font-size: 0.72rem !important;
}}

/* ── Code ── */
code {{
    background: {colors['bg_hover']} !important;
    color: {colors['tag_text']} !important;
    border-radius: 3px;
    padding: 1px 5px;
    font-size: 0.78rem;
    border: none !important;
}}

/* ── Progress ── */
[data-testid="stProgress"] > div > div {{ background: {colors['accent']} !important; }}

/* ── Divider ── */
hr {{ border-color: {colors['bg_border']} !important; margin: 0.75rem 0 !important; }}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 4px; height: 4px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: {colors['bg_border']}; border-radius: 2px; }}
::-webkit-scrollbar-thumb:hover {{ background: {colors['text_tertiary']}; }}

/* ── Alert ── */
[data-testid="stAlert"] {{
    border-radius: 5px !important;
    background: {colors['bg_tertiary']} !important;
}}

/* ── Dropdown/Popover overlays ── */
[data-testid="stPopoverContent"],
[role="listbox"],
.stPopover {{
    background: {colors['bg_secondary']} !important;
    border: 1px solid {colors['bg_border']} !important;
    color: {colors['text_primary']} !important;
}}

/* ── All input elements (general fallback) ── */
input[type="text"],
input[type="email"],
input[type="password"],
input[type="number"],
textarea {{
    background: {colors['bg_input']} !important;
    color: {colors['text_primary']} !important;
    border: 1px solid {colors['bg_border']} !important;
}}

/* ── Placeholder text (cross-browser) ── */
input::placeholder,
textarea::placeholder,
input::-webkit-input-placeholder,
textarea::-webkit-input-placeholder,
input::-moz-placeholder,
textarea::-moz-placeholder,
input:-ms-input-placeholder,
textarea:-ms-input-placeholder {{
    color: {colors['text_disabled']} !important;
    opacity: 0.7 !important;
}}

/* ── Dropdown options ── */
[role="option"] {{
    color: {colors['text_primary']} !important;
    background: {colors['bg_tertiary']} !important;
}}
[role="option"][aria-selected="true"],
[role="option"].selected {{
    background: {colors['accent']} !important;
    color: #fff !important;
}}
[role="option"]:hover {{
    background: {colors['bg_hover']} !important;
}}
</style>
"""


# (CSS will be applied after session state initialization)


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
    "theme_mode": "system",  # "light", "dark", or "system"
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def _get_active_theme() -> str:
    """Determine which theme is currently active.
    
    Returns 'dark' or 'light' based on:
    1. User selection if not "system"
    2. System preference if set to "system"
    3. Default 'dark' if system preference unavailable
    """
    preference = st.session_state.get("theme_mode", "system")
    
    if preference in ("dark", "light"):
        return preference
    
    # For "system" mode, we default to dark on server-side
    # The browser's prefers-color-scheme CSS media query will handle
    # actual system detection on the client
    return "dark"


def _get_section_accents() -> dict:
    """Get section accent colors based on active theme."""
    theme = _get_active_theme()
    return _SECTION_ACCENTS_DARK if theme == "dark" else _SECTION_ACCENTS_LIGHT


# Apply the dynamic theme CSS to the page
# This is called after session state initialization
_active_theme = _get_active_theme()
_active_colors = _THEMES.get(_active_theme, _THEMES["dark"])
st.markdown(_generate_css(_active_theme), unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    # ── Brand ──────────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="padding:0 0 1rem 0;">'
        f'<svg width="18" height="18" viewBox="0 0 18 18" fill="none" '
        f'style="vertical-align:-4px;margin-right:6px;" xmlns="http://www.w3.org/2000/svg">'
        f'<rect x="1" y="10" width="3" height="7" rx="1" fill="{_active_colors["accent_active"]}"/>'
        f'<rect x="7" y="6" width="3" height="11" rx="1" fill="{_active_colors["accent"]}"/>'
        f'<rect x="13" y="2" width="3" height="15" rx="1" fill="{_active_colors["accent_hover"]}"/>'
        f'</svg>'
        f'<span style="font-size:0.9rem;font-weight:600;color:{_active_colors["text_primary"]};letter-spacing:-0.02em;">Analyst AI</span>'
        f'<div style="font-size:0.68rem;color:{_active_colors["text_tertiary"]};margin-top:3px;padding-left:24px;">RAG · Local · Multi-model</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    _online, _status_text, _model_count = _ollama_status()
    _dot_color = _active_colors["success"] if _online else _active_colors["error"]
    _glow = f"box-shadow:0 0 5px {_dot_color};" if _online else ""
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:7px;margin-bottom:0.875rem;'
        f'padding:6px 10px;background:{_active_colors["bg_tertiary"]};border:1px solid {_active_colors["bg_border"]};border-radius:5px;">'
        f'<span style="width:6px;height:6px;border-radius:50%;background:{_dot_color};'
        f'flex-shrink:0;display:inline-block;{_glow}"></span>'
        f'<span style="font-size:0.7rem;color:{_active_colors["text_secondary"]};">Ollama {_status_text}</span>'
        f'<span style="font-size:0.65rem;color:{_active_colors["text_tertiary"]};margin-left:auto;">{_model_count} model{"s" if _model_count != 1 else ""}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Settings section ───────────────────────────────────────────────────
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:5px;margin-bottom:5px;">'
        f'<svg width="12" height="12" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">'
        f'<circle cx="8" cy="8" r="2.5" stroke="{_active_colors["text_tertiary"]}" stroke-width="1.5"/>'
        f'<path d="M8 1v2M8 13v2M1 8h2M13 8h2M3.05 3.05l1.42 1.42M11.54 11.54l1.41 1.41'
        f'M3.05 12.95l1.42-1.42M11.54 4.46l1.41-1.41" stroke="{_active_colors["text_tertiary"]}" stroke-width="1.5" stroke-linecap="round"/>'
        f'</svg>'
        f'<span style="font-size:0.62rem;font-weight:600;color:{_active_colors["text_disabled"]};letter-spacing:0.09em;text-transform:uppercase;">Settings</span>'
        f'</div>',
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
            f'<div style="display:flex;align-items:center;gap:5px;margin-top:-6px;margin-bottom:6px;">'
            f'<span style="width:6px;height:6px;border-radius:50%;background:{_active_colors["success"]};display:inline-block;"></span>'
            f'<span style="font-size:0.68rem;color:{_active_colors["text_tertiary"]};">Installed</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:5px;margin-top:-6px;margin-bottom:4px;">'
            f'<span style="width:6px;height:6px;border-radius:50%;background:{_active_colors["error"]};display:inline-block;"></span>'
            f'<span style="font-size:0.68rem;color:{_active_colors["text_tertiary"]};">Not installed</span>'
            f'</div>',
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

    # ── Theme selector ─────────────────────────────────────────────────────
    theme_choice = st.selectbox(
        "Theme",
        options=["System", "Light", "Dark"],
        index=["system", "light", "dark"].index(st.session_state.theme_mode),
        key="theme_dropdown",
    )
    
    # Map UI choice to theme_mode value
    theme_mode_map = {"System": "system", "Light": "light", "Dark": "dark"}
    if st.session_state.theme_mode != theme_mode_map[theme_choice]:
        st.session_state.theme_mode = theme_mode_map[theme_choice]
        st.rerun()

    st.divider()

    # ── Knowledge Base section ─────────────────────────────────────────────
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:5px;margin-bottom:6px;">'
        f'<svg width="12" height="12" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">'
        f'<ellipse cx="8" cy="4.5" rx="5.5" ry="2" stroke="{_active_colors["text_tertiary"]}" stroke-width="1.4"/>'
        f'<path d="M2.5 4.5v3c0 1.1 2.46 2 5.5 2s5.5-.9 5.5-2v-3" stroke="{_active_colors["text_tertiary"]}" stroke-width="1.4"/>'
        f'<path d="M2.5 7.5v3c0 1.1 2.46 2 5.5 2s5.5-.9 5.5-2v-3" stroke="{_active_colors["text_tertiary"]}" stroke-width="1.4"/>'
        f'</svg>'
        f'<span style="font-size:0.62rem;font-weight:600;color:{_active_colors["text_disabled"]};letter-spacing:0.09em;text-transform:uppercase;">Knowledge Base</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    doc_count = get_doc_count()
    st.markdown(
        f'<div style="font-size:0.7rem;color:{_active_colors["text_tertiary"]};margin-bottom:8px;">'
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
        f'<div style="font-size:0.65rem;color:{_active_colors["text_tertiary"]};">Session {st.session_state.session_id[:8]}</div>',
        unsafe_allow_html=True,
    )

# ── Header ────────────────────────────────────────────────────────────────────

_SVG_BARS = (
    f'<svg width="18" height="18" viewBox="0 0 18 18" fill="none" '
    f'style="vertical-align:-4px;margin-right:8px;" xmlns="http://www.w3.org/2000/svg">'
    f'<rect x="1" y="10" width="3" height="7" rx="1" fill="{_active_colors["accent_active"]}"/>'
    f'<rect x="7" y="6" width="3" height="11" rx="1" fill="{_active_colors["accent"]}"/>'
    f'<rect x="13" y="2" width="3" height="15" rx="1" fill="{_active_colors["accent_hover"]}"/>'
    f'</svg>'
)
st.markdown(
    f'<h1 style="font-size:1.1rem;font-weight:600;color:{_active_colors["text_primary"]};letter-spacing:-0.03em;margin:0 0 3px 0;">'
    f'{_SVG_BARS}Analyst AI</h1>'
    f'<p style="font-size:0.78rem;color:{_active_colors["text_secondary"]};margin:0 0 1rem 0;padding-left:26px;">'
    f'Upload documents, analyze with multiple models, and extract structured insights.</p>',
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
            f'<div style="font-size:0.62rem;font-weight:600;color:{_active_colors["text_disabled"]};letter-spacing:0.09em;'
            f'text-transform:uppercase;margin-bottom:8px;">Quick start</div>',
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
                    f'<div style="font-size:0.62rem;color:{_active_colors["text_disabled"]};margin-bottom:4px;">{msg["model"]}</div>',
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
                f'<span style="font-size:0.62rem;color:{_active_colors["text_disabled"]};">Not satisfied? Retry with:</span>'
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
        f'<div style="font-size:0.78rem;color:{_active_colors["text_secondary"]};margin-bottom:1rem;">'
        f'Select models, pose a question, and compare how each analyzes your documents.</div>',
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
            f'<div style="font-size:0.7rem;color:{_active_colors["text_secondary"]};margin:0.75rem 0 0.5rem;">'
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
                        f'<div style="color:{_active_colors["error"]};font-size:0.82rem;padding:4px 0;">'
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
        f'<div style="font-size:0.78rem;color:{_active_colors["text_secondary"]};margin-bottom:1rem;">'
        f'Auto-extract an executive summary, key metrics, risk analysis, and recommendations '
        f'from your entire knowledge base using your chosen model.</div>',
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
                f'<div style="font-size:0.68rem;color:{_active_colors["text_disabled"]};margin-bottom:0.75rem;">'
                f'Generated with {_used_model}</div>',
                unsafe_allow_html=True,
            )
        for key, data in st.session_state.insights.items():
            _section_accents = _get_section_accents()
            _accent = _section_accents.get(key, _active_colors["text_tertiary"])
            st.markdown(
                f'<div style="display:flex;align-items:center;padding:10px 14px;'
                f'background:{_active_colors["bg_tertiary"]};border:1px solid {_active_colors["bg_border"]};'
                f'border-left:3px solid {_accent};border-radius:0 5px 5px 0;margin-bottom:2px;">'
                f'<span style="font-size:0.85rem;font-weight:500;color:{_active_colors["text_primary"]};\">{data["label"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            with st.expander("View analysis", expanded=(key == "summary")):
                st.markdown(data["content"])
    elif not gen_btn:
        st.markdown(
            f'<div style="background:{_active_colors["bg_tertiary"]};border:1px solid {_active_colors["bg_border"]};border-radius:6px;'
            f'padding:3rem;text-align:center;">'
            f'<div style="font-size:0.85rem;font-weight:500;color:{_active_colors["text_disabled"]};margin-bottom:4px;">'
            f'No insights generated yet</div>'
            f'<div style="font-size:0.72rem;color:{_active_colors["text_tertiary"]};\">'
            f'Upload documents via the sidebar, choose a model, then click Generate Insights.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

