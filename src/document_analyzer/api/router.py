from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from document_analyzer.core.config import Settings, get_settings
from document_analyzer.models.chat import ChatRequest, ChatResponse, HealthResponse, ServiceHealth
from document_analyzer.models.chunking import ChunkRequest, ChunkResponse
from document_analyzer.services.analyze_document import AnalyzeDocumentService
from document_analyzer.services.chroma_client import ChromaService
from document_analyzer.services.chunking_service import (
    ChunkingService,
    EmptyDocumentError,
    FileNotFoundChunkingError,
    UnsupportedFileTypeError,
)
from document_analyzer.services.prompt_builder import PromptBuilder
from document_analyzer.services.together_client import MissingTogetherAPIKeyError, TogetherChatService


router = APIRouter()


# ── Dependency providers ─────────────────────────────────────


def get_chat_service(settings: Settings = Depends(get_settings)) -> TogetherChatService:
    return TogetherChatService.from_settings(settings)


def get_chroma_service(settings: Settings = Depends(get_settings)) -> ChromaService:
    return ChromaService.from_settings(settings)


def get_chunking_service(settings: Settings = Depends(get_settings)) -> ChunkingService:
    return ChunkingService.from_settings(settings)


def get_analyze_service(
    service: TogetherChatService = Depends(get_chat_service),
) -> AnalyzeDocumentService:
    return AnalyzeDocumentService(prompt_builder=PromptBuilder(service))


# ── Routes ───────────────────────────────────────────────────


@router.get("/health", response_model=HealthResponse)
def health_check(
    settings: Settings = Depends(get_settings),
    ai_service: TogetherChatService = Depends(get_chat_service),
    vector_db: ChromaService = Depends(get_chroma_service),
) -> HealthResponse:
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
def api_health_check(
    settings: Settings = Depends(get_settings),
    ai_service: TogetherChatService = Depends(get_chat_service),
    vector_db: ChromaService = Depends(get_chroma_service),
) -> HealthResponse:
    return health_check(settings=settings, ai_service=ai_service, vector_db=vector_db)

@router.post("/api/v1/analyze")
def analyze_document(
    request: ChatRequest,
    analyze_service: AnalyzeDocumentService = Depends(get_analyze_service),
):
    try:
        rewritten_query = analyze_service.analyze(request.prompt)
    except MissingTogetherAPIKeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return {
        "original_prompt": request.prompt,
        "rewritten_query": rewritten_query,
    }


@router.post("/api/v1/chat", response_model=ChatResponse)
def create_chat_response(
    request: ChatRequest,
    service: TogetherChatService = Depends(get_chat_service),
) -> ChatResponse:
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
def get_chroma_collections(
    chroma_service: ChromaService = Depends(get_chroma_service),
):
    try:
        collections = chroma_service.list_collections()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return {"collections": collections}


@router.get("/api/v1/get_documents")
def get_documents(
    chroma_service: ChromaService = Depends(get_chroma_service),
):
    try:
        collection = chroma_service.get_or_create_collection()
        raw = collection.get(limit=10, offset=0)
        print("DEBUG: Raw ChromaDB response:", raw)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return {"documents": raw}


@router.post("/api/v1/add_documents")
def add_documents(
    chroma_service: ChromaService = Depends(get_chroma_service),
):
    try:
        collection = chroma_service.get_or_create_collection()
        collection.add(
            ids=["doc1", "doc2"],
            documents=["This is the first document.", "This is the second document."],
            metadatas=[{"source": "test"}, {"source": "test"}],
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return {"status": "documents added successfully"}


@router.post("/api/v1/chunk", response_model=ChunkResponse)
def chunk_document(
    request: ChunkRequest,
    service: ChunkingService = Depends(get_chunking_service),
) -> ChunkResponse:
    try:
        return service.chunk_file(
            request.file_name,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )
    except FileNotFoundChunkingError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except UnsupportedFileTypeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except EmptyDocumentError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
        
        
@router.get("/api/v1/upload_file", response_model=ChunkResponse)
def upload_and_chunk_file(
    file_name: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    service: ChunkingService = Depends(get_chunking_service),
) -> ChunkResponse:
    """Endpoint to upload a file and get its chunks in one step."""
    try:
        return service.chunk_file(
            file_name=file_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    except FileNotFoundChunkingError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except UnsupportedFileTypeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except EmptyDocumentError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc        