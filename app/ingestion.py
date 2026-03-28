import hashlib
import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings

logger = logging.getLogger(__name__)

# Characters-per-page threshold below which we consider text "sparse" (likely scanned)
_SPARSE_THRESHOLD = 80


def _chunk_id(source: str, page: int, chunk_index: int) -> str:
    """Deterministic vector ID so re-ingesting the same PDF overwrites, not duplicates."""
    raw = f"{source}::{page}::{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _avg_chars_per_page(pages: list[Document]) -> float:
    if not pages:
        return 0.0
    total = sum(len(p.page_content.strip()) for p in pages)
    return total / len(pages)


def _try_pymupdf(pdf_path: Path, source_name: str) -> list[Document]:
    """Extract text with PyMuPDF (fitz) — often better than pypdf for complex layouts."""
    try:
        import fitz  # pymupdf
    except ImportError:
        return []
    pages = []
    with fitz.open(str(pdf_path)) as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text")
            if text.strip():
                pages.append(Document(
                    page_content=text,
                    metadata={"source": source_name, "page": i},
                ))
    return pages


def _try_ocr(pdf_path: Path, source_name: str) -> list[Document]:
    """OCR fallback: convert each page to an image and run Tesseract."""
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except ImportError:
        logger.warning(
            "OCR dependencies not installed. Run: brew install tesseract poppler && "
            "pip install pytesseract pdf2image"
        )
        return []
    try:
        logger.info("Running OCR on %s — this may take a moment...", source_name)
        # pdf2image needs poppler; try Homebrew path if not in system PATH
        import shutil
        poppler_path = shutil.which("pdftoppm")
        if poppler_path:
            poppler_path = str(Path(poppler_path).parent)
        else:
            # Common Homebrew location on macOS
            homebrew_poppler = Path("/opt/homebrew/opt/poppler/bin")
            poppler_path = str(homebrew_poppler) if homebrew_poppler.exists() else None

        images = convert_from_path(str(pdf_path), dpi=300, poppler_path=poppler_path)
        pages = []
        for i, img in enumerate(images):
            text = pytesseract.image_to_string(img, config="--psm 3")
            if text.strip():
                pages.append(Document(
                    page_content=text,
                    metadata={"source": source_name, "page": i},
                ))
        logger.info("OCR extracted %d page(s) from %s", len(pages), source_name)
        return pages
    except Exception as exc:
        logger.warning("OCR failed for %s: %s", source_name, exc)
        return []


def load_and_chunk_pdf(pdf_path: str | Path) -> list[Document]:
    settings = get_settings()
    pdf_path = Path(pdf_path)
    source_name = pdf_path.name

    # Stage 1: PyPDFLoader (fast, standard)
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()

    # Stage 2: PyMuPDF — better text extraction for complex layouts
    if _avg_chars_per_page(pages) < _SPARSE_THRESHOLD:
        logger.info("Sparse text via PyPDF for %s, trying PyMuPDF...", source_name)
        pymupdf_pages = _try_pymupdf(pdf_path, source_name)
        if _avg_chars_per_page(pymupdf_pages) > _avg_chars_per_page(pages):
            pages = pymupdf_pages

    # Stage 3: Tesseract OCR — for scanned/image-based PDFs
    if _avg_chars_per_page(pages) < _SPARSE_THRESHOLD:
        logger.info("Still sparse after PyMuPDF, running OCR on %s...", source_name)
        ocr_pages = _try_ocr(pdf_path, source_name)
        if _avg_chars_per_page(ocr_pages) > _avg_chars_per_page(pages):
            pages = ocr_pages

    if not pages or _avg_chars_per_page(pages) < 10:
        logger.warning("No readable text found in %s after all extraction attempts.", source_name)
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
    )

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
