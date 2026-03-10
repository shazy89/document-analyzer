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
    host: str = Field(default="127.0.0.1", validation_alias=AliasChoices("DOC_ANALYZER_HOST", "APP_HOST"))
    port: int = Field(default=8000, validation_alias=AliasChoices("DOC_ANALYZER_PORT", "APP_PORT"))
    together_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("TOGETHER_API_KEY", "DOC_ANALYZER_TOGETHER_API_KEY"),
    )
    together_model: str = Field(
        default="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        validation_alias=AliasChoices("TOGETHER_MODEL", "DOC_ANALYZER_TOGETHER_MODEL"),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
