from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from document_analyzer.core.config import get_settings
from document_analyzer.models.chat import ChatRequest, ChatResponse, HealthResponse
from document_analyzer.services.together_client import MissingTogetherAPIKeyError, TogetherChatService

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(status="ok", model=settings.together_model)


@router.post("/api/v1/chat", response_model=ChatResponse)
def create_chat_response(request: ChatRequest) -> ChatResponse:
    settings = get_settings()
    service = TogetherChatService.from_settings(settings)
    try:
        return service.ask(
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            model=request.model,
        )
    except MissingTogetherAPIKeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
