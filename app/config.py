from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

# Curated list of supported Ollama chat models, shown in this order in the UI.
# Best open-source models for structured reasoning and JSON output as of 2026.
SUPPORTED_MODELS: list[str] = [
    "qwen2.5:7b",       # Best structured/JSON output — default for quiz generation
    "llama3.1:8b",      # Strong reasoning, supports tool calling
    "mistral",          # Solid all-rounder
    "gemma2:9b",        # Google quality, good responses
    "deepseek-r1:7b",   # Strong reasoning (R1 distil)
    "llama3.2:3b",      # Lightweight / fast option
    "llama3.2:1b",      # Fastest / lowest memory
]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Ollama (local, free)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    ollama_embedding_model: str = "nomic-embed-text"

    # ChromaDB (local, free)
    chroma_persist_dir: str = "data/chroma"
    chroma_collection_name: str = "rag-chatbot"

    # RAG tuning
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    memory_window: int = 10

    # Quiz / exam prep
    quiz_default_n: int = 10
    quiz_top_k: int = 8


@lru_cache
def get_settings() -> Settings:
    return Settings()
