from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ChatRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    role: ChatRole
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1)
    system_prompt: str | None = None
    model: str | None = None


class ChatResponse(BaseModel):
    model: str
    answer: str


class ServiceHealth(BaseModel):
    status: str
    detail: str | None = None




class HealthResponse(BaseModel):
    status: str
    model: str
    ai_service: ServiceHealth
    vector_db: ServiceHealth
    bm25_db: ServiceHealth


class HybridSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    n_results: int = Field(default=5, gt=0)
    vector_weight: float = Field(default=0.5, ge=0.0, le=1.0)


class SearchResult(BaseModel):
    id: str
    document: str
    metadata: dict
    score: float
    sources: list[str]


class HybridSearchResponse(BaseModel):
    results: list[SearchResult]
    strategy: str
