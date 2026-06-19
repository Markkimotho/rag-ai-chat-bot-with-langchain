# ── Stage 1: build the React SPA ─────────────────────────────────────────────
FROM node:20-alpine AS frontend
WORKDIR /frontend

# Install deps first for better layer caching.
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci

COPY frontend/ ./
RUN npm run build   # emits /frontend/dist

# ── Stage 2: FastAPI backend serving the built SPA ───────────────────────────
FROM python:3.12-slim
WORKDIR /app

# System deps: PDF OCR (tesseract/poppler) + lxml (libxml/xslt) for scraping.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    poppler-utils \
    libxml2-dev \
    libxslt-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY scripts/ ./scripts/

# Built SPA — api.py serves /app/frontend/dist at "/" when present.
COPY --from=frontend /frontend/dist ./frontend/dist

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
