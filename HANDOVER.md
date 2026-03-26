# Handover Documentation

This guide explains how to maintain and extend the RAG AI Chatbot after delivery.

---

## Adding New PDFs

### Option A: Via the Gradio UI (easiest)

1. Open the chatbot at `http://localhost:7860`
2. In the right sidebar, under **Upload PDFs**, click the file upload area
3. Select one or more PDF files
4. Click **📤 Ingest uploaded PDFs**
5. Wait for the status message confirming ingestion (e.g., "✓ report.pdf: 47 chunks ingested")
6. The new knowledge is immediately available — start asking questions

### Option B: Via CLI script

1. Copy your PDF files into the `data/pdfs/` directory
2. Run the ingestion script:

```bash
python -m scripts.ingest_pdfs
```

3. To ingest from a different directory:

```bash
python -m scripts.ingest_pdfs --pdf-dir /path/to/your/pdfs
```

### Option C: Via Docker

If running with Docker, the `data/pdfs/` directory is mounted as a volume:

1. Copy PDFs into `data/pdfs/` on the host machine
2. Run ingestion inside the container:

```bash
docker-compose exec chatbot python -m scripts.ingest_pdfs
```

---

## Re-ingesting PDFs

The system uses deterministic vector IDs based on filename + page + chunk index. This means:

- **Re-ingesting the same PDF** overwrites existing vectors (no duplicates)
- **Renaming a PDF** and re-ingesting creates new vectors (the old ones remain)
- To fully reset, delete the Pinecone index from the Pinecone dashboard and re-ingest all PDFs

---

## Switching Models

### In the UI

Use the **Model** dropdown in the sidebar to switch between `gpt-4o-mini` and `gpt-4o`.

### In the environment

Edit `.env`:

```
OPENAI_MODEL=gpt-4o
```

Restart the application.

---

## Tuning Retrieval

| Setting | Where | Effect |
|---------|-------|--------|
| **Top-K** | UI slider or `TOP_K` in `.env` | More docs = more context but slower and more tokens |
| **Chunk Size** | `CHUNK_SIZE` in `.env` | Larger chunks = more context per chunk, fewer chunks total |
| **Chunk Overlap** | `CHUNK_OVERLAP` in `.env` | More overlap = better continuity at chunk boundaries |
| **Memory Window** | `MEMORY_WINDOW` in `.env` | More turns = longer conversation context |

After changing chunk size/overlap, you must **re-ingest all PDFs** for the changes to take effect.

---

## Customizing Prompts

Prompts are in `app/prompts.py`:

- **`CONTEXTUALIZE_SYSTEM`** — Instructions for rephrasing questions given chat history
- **`QA_SYSTEM`** — Main system prompt controlling answer style, citation format, and fallback behavior

Edit these strings and restart the app. No re-ingestion needed.

---

## Switching Orchestration Modes

The chatbot offers two orchestration backends:

| Mode | File | Memory | Best For |
|------|------|--------|----------|
| **LangChain** (LCEL) | `app/chain.py` | Manual (in-memory window) | Simple, reliable RAG |
| **LangGraph** | `app/graph.py` | Automatic (MemorySaver checkpointer) | Complex workflows, extensibility |

Toggle between them using the **Orchestration** radio button in the UI sidebar.

---

## Troubleshooting

### "No PDF files found"
- Ensure PDFs are in `data/pdfs/` (or the directory you specified)
- Files must have a `.pdf` extension

### "Invalid API key" errors
- Verify `OPENAI_API_KEY` and `PINECONE_API_KEY` in `.env`
- Ensure no extra whitespace or quotes around the keys

### Pinecone index errors
- The app auto-creates the index on first use
- If you need to reset, delete the index from the [Pinecone Console](https://app.pinecone.io/) and restart

### Slow responses
- Reduce `TOP_K` to retrieve fewer documents
- Use `gpt-4o-mini` instead of `gpt-4o`
- Reduce `CHUNK_SIZE` for shorter context windows

### Docker issues
- Ensure `.env` file exists in the project root
- Check logs: `docker-compose logs -f chatbot`
- Verify port 7860 is not in use: `lsof -i :7860`
