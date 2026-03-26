"""Gradio UI for the RAG chatbot."""

import logging
import tempfile
import uuid
from pathlib import Path

import gradio as gr

from app import chain as lcel_chain
from app import graph as langgraph_module
from app.ingestion import load_and_chunk_pdf
from app.memory import clear_history
from app.vectorstore import ingest_documents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def _format_sources(sources: list[dict]) -> str:
    if not sources:
        return ""
    seen = set()
    lines = []
    for s in sources:
        key = (s["source"], s["page"])
        if key not in seen:
            seen.add(key)
            lines.append(f"- {s['source']}, page {s['page']}")
    return "\n\n**Sources:**\n" + "\n".join(lines)


def chat_fn(
    message: str,
    history: list[dict],
    session_id: str,
    orchestration: str,
    model: str,
    top_k: int,
):
    if not message.strip():
        return ""

    try:
        if orchestration == "LangGraph":
            result = langgraph_module.invoke(
                question=message,
                thread_id=session_id,
                model=model,
                top_k=top_k,
            )
        else:
            result = lcel_chain.invoke(
                question=message,
                session_id=session_id,
                model=model,
                top_k=top_k,
            )

        answer = result["answer"]
        sources = _format_sources(result.get("sources", []))
        return answer + sources

    except Exception as e:
        logger.exception("Error processing question")
        return f"An error occurred: {e}"


def upload_pdf(files, session_id: str):
    if not files:
        return "No files uploaded."

    results = []
    for file_path in files:
        try:
            path = Path(file_path)
            chunks = load_and_chunk_pdf(path)
            count = ingest_documents(chunks)
            results.append(f"[OK] {path.name}: {count} chunks ingested")
        except Exception as e:
            results.append(f"[FAIL] {Path(file_path).name}: {e}")

    return "\n".join(results)


def clear_chat(session_id: str):
    clear_history(session_id)
    return [], "Conversation cleared."


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="RAG AI Chatbot",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("# RAG AI Chatbot\nAsk questions about your PDF knowledge base.")

        session_id = gr.State(value=lambda: str(uuid.uuid4()))

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=500,
                    type="messages",
                )
                msg = gr.Textbox(
                    label="Your question",
                    placeholder="Ask something about your documents…",
                    submit_btn=True,
                )

            with gr.Column(scale=1):
                gr.Markdown("### Settings")
                orchestration = gr.Radio(
                    choices=["LangChain", "LangGraph"],
                    value="LangChain",
                    label="Orchestration",
                )
                model = gr.Dropdown(
                    choices=["gpt-4o-mini", "gpt-4o"],
                    value="gpt-4o-mini",
                    label="Model",
                )
                top_k = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Top-K documents",
                )
                clear_btn = gr.Button("Clear conversation")

                gr.Markdown("### Upload PDFs")
                pdf_upload = gr.File(
                    label="Upload PDF(s)",
                    file_types=[".pdf"],
                    file_count="multiple",
                )
                upload_btn = gr.Button("Ingest uploaded PDFs")
                upload_status = gr.Textbox(
                    label="Upload status",
                    interactive=False,
                )

        # Chat interaction
        def respond(message, history, sid, orch, mdl, k):
            assistant_msg = chat_fn(message, history, sid, orch, mdl, k)
            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": assistant_msg},
            ]
            return "", history

        msg.submit(
            respond,
            inputs=[msg, chatbot, session_id, orchestration, model, top_k],
            outputs=[msg, chatbot],
        )

        # Clear conversation
        def on_clear(sid):
            clear_history(sid)
            return [], ""

        clear_btn.click(
            on_clear,
            inputs=[session_id],
            outputs=[chatbot, msg],
        )

        # PDF upload
        upload_btn.click(
            upload_pdf,
            inputs=[pdf_upload, session_id],
            outputs=[upload_status],
        )

    return demo


def main():
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
