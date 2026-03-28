from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

# Curated list of supported Ollama chat models, shown in this order in the UI.
SUPPORTED_MODELS: list[str] = ["llama3.2:1b", "llama3.2", "mistral", "gemma2:2b"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Ollama (local, free)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:1b"
    ollama_embedding_model: str = "nomic-embed-text"

    # ChromaDB (local, free)
    chroma_persist_dir: str = "data/chroma"
    chroma_collection_name: str = "rag-chatbot"

    # RAG tuning
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    memory_window: int = 10


@lru_cache
def get_settings() -> Settings:
    return Settings()
