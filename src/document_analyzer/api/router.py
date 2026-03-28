from __future__ import annotations

from fastapi import APIRouter, HTTPException, status, Depends

from document_analyzer.core.config import get_settings
from document_analyzer.models.chat import ChatRequest, ChatResponse, HealthResponse, ServiceHealth
from document_analyzer.services.together_client import MissingTogetherAPIKeyError, TogetherChatService
from document_analyzer.services.analyze_document import AnalyzeDocumentService
from document_analyzer.services.prompt_builder import PromptBuilder
from document_analyzer.services.chroma_client import ChromaService


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    settings = get_settings()
    ai_service = TogetherChatService.from_settings(settings)
    vector_db = ChromaService.from_settings(settings)

    ai_ok, ai_detail = ai_service.health()
    db_ok = vector_db.heartbeat()
    overall_status = "ok" if ai_ok and db_ok else "degraded"
    
    return HealthResponse(
        status=overall_status,
        model=settings.together_model,
        ai_service=ServiceHealth(
            status="ok" if ai_ok else "error",
            detail=ai_detail,
        ),
        vector_db=ServiceHealth(
            status="ok" if db_ok else "error",
            detail=(
                f"Connected to ChromaDB at {settings.chroma_host}:{settings.chroma_port}."
                if db_ok
                else f"Could not reach ChromaDB at {settings.chroma_host}:{settings.chroma_port}."
            ),
        ),
    )

@router.get("/api/v1/health", response_model=HealthResponse)
def api_health_check() -> HealthResponse:
    return health_check()

@router.post("/api/v1/analyze")
def analyze_document(request: ChatRequest):
    settings = get_settings()
    service = TogetherChatService.from_settings(settings)
    prompt_builder = PromptBuilder(service)
    analyze_service = AnalyzeDocumentService(prompt_builder=prompt_builder)

    try:
        rewritten_query = analyze_service.analyze(request.prompt)
    except MissingTogetherAPIKeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return {
        "original_prompt": request.prompt,
        "rewritten_query": rewritten_query
    }


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


@router.get("/api/v1/chroma/collections")
def get_chroma_collections():
    settings = get_settings()
    chroma_service = ChromaService.from_settings(settings)
    
    collections = chroma_service.list_collections()
    print(f"ChromaDB Collections: {collections}")
    return {"collections": collections}