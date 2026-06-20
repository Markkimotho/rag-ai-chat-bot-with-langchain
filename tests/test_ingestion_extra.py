"""Gate tests for image OCR ingestion + the fast/dedup ingest path."""

import sys
import types
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from app.ingestion import IMAGE_EXTENSIONS, load_and_chunk_image


def _install_fake_ocr(text: str):
    """Inject fake PIL + pytesseract modules so load_and_chunk_image runs offline."""
    pil = types.ModuleType("PIL")
    image_mod = types.ModuleType("PIL.Image")

    class _Ctx:
        def __enter__(self):
            return MagicMock()

        def __exit__(self, *a):
            return False

    image_mod.open = lambda _p: _Ctx()
    pil.Image = image_mod

    pytesseract = types.ModuleType("pytesseract")
    pytesseract.image_to_string = lambda _img, config="": text
    return {"PIL": pil, "PIL.Image": image_mod, "pytesseract": pytesseract}


def test_image_extensions_cover_common_formats():
    for ext in (".png", ".jpg", ".jpeg", ".webp", ".tiff"):
        assert ext in IMAGE_EXTENSIONS


def test_load_and_chunk_image_ocrs_text():
    fakes = _install_fake_ocr("Newton's second law states that F equals m times a. " * 5)
    with patch.dict(sys.modules, fakes):
        chunks = load_and_chunk_image("/tmp/photo.png", source_name="physics.png")
    assert len(chunks) >= 1
    for c in chunks:
        assert c.metadata["source"] == "physics.png"
        assert c.metadata["type"] == "image"
        assert "id" in c.metadata


def test_load_and_chunk_image_empty_on_no_text():
    fakes = _install_fake_ocr("   ")
    with patch.dict(sys.modules, fakes):
        assert load_and_chunk_image("/tmp/blank.png") == []


# ── ingest_documents dedup + parallel path ──────────────────────────────────


def _doc(i: int) -> Document:
    return Document(
        page_content=f"chunk {i}",
        metadata={"source": "f.pdf", "page": 0, "chunk_index": i, "id": f"id{i}"},
    )


def test_ingest_skips_already_indexed_chunks():
    from app import vectorstore as vs

    docs = [_doc(0), _doc(1), _doc(2)]
    collection = MagicMock()
    # id0 already present → only id1, id2 should be embedded/added.
    collection.get.return_value = {"ids": ["id0"]}
    fake_vs = MagicMock()
    fake_vs._collection = collection
    embedder = MagicMock()
    embedder.embed_documents.side_effect = lambda texts: [[0.1] * 3 for _ in texts]

    with patch("app.vectorstore.get_vectorstore", return_value=fake_vs), patch(
        "app.vectorstore.get_embeddings", return_value=embedder
    ):
        added = vs.ingest_documents(docs)

    assert added == 2
    added_ids = collection.add.call_args.kwargs["ids"]
    assert "id0" not in added_ids
    assert set(added_ids) == {"id1", "id2"}


def test_ingest_noop_when_all_present():
    from app import vectorstore as vs

    docs = [_doc(0)]
    collection = MagicMock()
    collection.get.return_value = {"ids": ["id0"]}
    fake_vs = MagicMock()
    fake_vs._collection = collection
    with patch("app.vectorstore.get_vectorstore", return_value=fake_vs), patch(
        "app.vectorstore.get_embeddings"
    ) as mock_emb:
        added = vs.ingest_documents(docs)
    assert added == 0
    mock_emb.return_value.embed_documents.assert_not_called()


def test_ingest_empty_returns_zero():
    from app import vectorstore as vs

    assert vs.ingest_documents([]) == 0
