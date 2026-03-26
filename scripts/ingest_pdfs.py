"""CLI script to ingest PDFs into Pinecone."""

import argparse
import logging
import time

from app.ingestion import load_and_chunk_pdfs
from app.vectorstore import ingest_documents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs into Pinecone")
    parser.add_argument(
        "--pdf-dir",
        default="data/pdfs",
        help="Directory containing PDF files (default: data/pdfs)",
    )
    args = parser.parse_args()

    start = time.perf_counter()

    logger.info("Loading and chunking PDFs from '%s'", args.pdf_dir)
    documents = load_and_chunk_pdfs(args.pdf_dir)

    if not documents:
        logger.warning("No documents to ingest. Add PDFs to '%s' and retry.", args.pdf_dir)
        return

    logger.info("Ingesting %d chunks into Pinecone…", len(documents))
    count = ingest_documents(documents)

    elapsed = time.perf_counter() - start
    logger.info("Done — %d chunks ingested in %.1fs", count, elapsed)


if __name__ == "__main__":
    main()
