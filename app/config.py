from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # OpenAI
    openai_api_key: SecretStr
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # Pinecone
    pinecone_api_key: SecretStr
    pinecone_index_name: str = "rag-chatbot"

    # RAG tuning
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    memory_window: int = 10


@lru_cache
def get_settings() -> Settings:
    return Settings()
