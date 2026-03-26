import hashlib
import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings

logger = logging.getLogger(__name__)


def _chunk_id(source: str, page: int, chunk_index: int) -> str:
    """Deterministic vector ID so re-ingesting the same PDF overwrites, not duplicates."""
    raw = f"{source}::{page}::{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()


def load_and_chunk_pdf(pdf_path: str | Path) -> list[Document]:
    settings = get_settings()
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
    )

    source_name = Path(pdf_path).name
    chunks = splitter.split_documents(pages)

    for i, chunk in enumerate(chunks):
        chunk.metadata["source"] = source_name
        chunk.metadata["chunk_index"] = i
        chunk.metadata["id"] = _chunk_id(
            source_name, chunk.metadata.get("page", 0), i
        )

    return chunks


def load_and_chunk_pdfs(pdf_dir: str | Path) -> list[Document]:
    pdf_dir = Path(pdf_dir)
    all_chunks: list[Document] = []

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in %s", pdf_dir)
        return all_chunks

    for pdf_path in pdf_files:
        logger.info("Processing %s", pdf_path.name)
        chunks = load_and_chunk_pdf(pdf_path)
        all_chunks.extend(chunks)
        logger.info("  → %d chunks from %s", len(chunks), pdf_path.name)

    logger.info("Total: %d chunks from %d PDFs", len(all_chunks), len(pdf_files))
    return all_chunks
