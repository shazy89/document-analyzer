from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Document Analyzer"
    app_version: str = "0.1.0"
    host: str = Field(
        default="127.0.0.1",
        validation_alias=AliasChoices("DOC_ANALYZER_HOST", "APP_HOST"),
    )
    port: int = Field(
        default=8000,
        validation_alias=AliasChoices("DOC_ANALYZER_PORT", "APP_PORT"),
    )
    together_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("TOGETHER_API_KEY", "DOC_ANALYZER_TOGETHER_API_KEY"),
    )
    together_model: str = Field(
        default="mistralai/Mistral-Small-24B-Instruct-2501",
        validation_alias=AliasChoices("TOGETHER_MODEL", "DOC_ANALYZER_TOGETHER_MODEL"),
    )

    # ── ChromaDB ──────────────────────────────────────────────
    chroma_host: str = Field(
        default="localhost",
        validation_alias=AliasChoices("CHROMA_HOST", "DOC_ANALYZER_CHROMA_HOST"),
    )
    chroma_port: int = Field(
        default=8100,
        validation_alias=AliasChoices("CHROMA_PORT", "DOC_ANALYZER_CHROMA_PORT"),
    )
    chroma_collection: str = Field(
        default="documents",
        validation_alias=AliasChoices("CHROMA_COLLECTION", "DOC_ANALYZER_CHROMA_COLLECTION"),
    )
    documents_path: str = Field(
       default="./documents",
        validation_alias=AliasChoices("DOCUMENTS_PATH", "DOC_ANALYZER_DOCUMENTS_PATH"),
    )

    # ── PostgreSQL (BM25 full-text search) ────────────────────
    postgres_host: str = Field(
        default="localhost",
        validation_alias=AliasChoices("POSTGRES_HOST", "DOC_ANALYZER_POSTGRES_HOST"),
    )
    postgres_port: int = Field(
        default=5434,
        validation_alias=AliasChoices("POSTGRES_PORT", "DOC_ANALYZER_POSTGRES_PORT"),
    )
    postgres_user: str = Field(
        default="ed",
        validation_alias=AliasChoices("POSTGRES_USER", "DOC_ANALYZER_POSTGRES_USER"),
    )
    postgres_password: SecretStr = Field(
        default="123123",
        validation_alias=AliasChoices("POSTGRES_PASSWORD", "DOC_ANALYZER_POSTGRES_PASSWORD"),
    )
    postgres_db: str = Field(
        default="document_analyzer",
        validation_alias=AliasChoices("POSTGRES_DB", "DOC_ANALYZER_POSTGRES_DB"),
    )

    # ── Logging ───────────────────────────────────────────────
    log_level: str = Field(
        default="INFO",
        validation_alias=AliasChoices("LOG_LEVEL", "DOC_ANALYZER_LOG_LEVEL"),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
