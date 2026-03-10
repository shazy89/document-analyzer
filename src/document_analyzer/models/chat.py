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


class HealthResponse(BaseModel):
    status: str
    model: str
