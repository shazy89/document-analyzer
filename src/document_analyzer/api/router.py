from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from document_analyzer.core.config import Settings, get_settings
from document_analyzer.models.chat import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    HybridSearchRequest,
    ServiceHealth,
)
from document_analyzer.services.postgres_client import PostgresService
from document_analyzer.models.chunking import ChunkRequest, ChunkResponse, UploadFileRequest
from document_analyzer.services.analyze_document import AnalyzeDocumentService
from document_analyzer.services.chroma_client import ChromaService
from document_analyzer.services.chunking_service import (
    ChunkingService,
    EmptyDocumentError,
    FileNotFoundChunkingError,
    UnsupportedFileTypeError,
)
from document_analyzer.services.embedding_service import EmbeddingService
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


def get_postgres_service(settings: Settings = Depends(get_settings)) -> PostgresService:
    pg = PostgresService.from_settings(settings)
    pg.init_schema()
    return pg


# ── Routes ───────────────────────────────────────────────────


@router.get("/health", response_model=HealthResponse)
def health_check(
    settings: Settings = Depends(get_settings),
    ai_service: TogetherChatService = Depends(get_chat_service),
    vector_db: ChromaService = Depends(get_chroma_service),
    bm25_db: PostgresService = Depends(get_postgres_service),
) -> HealthResponse:
    ai_ok, ai_detail = ai_service.health()
    db_ok = vector_db.heartbeat()
    pg_ok = bm25_db.heartbeat()
    overall_status = "ok" if (ai_ok and db_ok and pg_ok) else "degraded"

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
        bm25_db=ServiceHealth(
            status="ok" if pg_ok else "error",
            detail=(
                f"Connected to PostgreSQL at {settings.postgres_host}:{settings.postgres_port}."
                if pg_ok
                else f"Could not reach PostgreSQL at {settings.postgres_host}:{settings.postgres_port}."
            ),
        ),
    )


@router.get("/api/v1/health", response_model=HealthResponse)
def api_health_check(
    settings: Settings = Depends(get_settings),
    ai_service: TogetherChatService = Depends(get_chat_service),
    vector_db: ChromaService = Depends(get_chroma_service),
    bm25_db: PostgresService = Depends(get_postgres_service),
) -> HealthResponse:
    return health_check(
        settings=settings, ai_service=ai_service, vector_db=vector_db, bm25_db=bm25_db,
    )

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
        collection = chroma_service.query(query_texts=["what year was galatasaray established?"], n_results=5)
     
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return {"documents": collection}


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
        response = service.chunk_file(
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

    if request.embed:
        embedding_service = EmbeddingService()
        embedding_service.embed_chunks(response.chunks)

    return response
        
        
@router.post("/api/v1/upload_file")
def upload_and_chunk_file(
    request: UploadFileRequest,
    service: ChunkingService = Depends(get_chunking_service),
    chroma_service: ChromaService = Depends(get_chroma_service),
    postgres_service: PostgresService = Depends(get_postgres_service),
) -> dict:
    """Upload a file, chunk it, embed it, and store in ChromaDB + PostgreSQL."""
    try:
        response = service.chunk_file(
            file_name=request.file_name,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )

        embedding_service = EmbeddingService()
        embedding_service.embed_chunks(response.chunks)

        ids = [f"{request.file_name}_chunk_{chunk.chunk_index}" for chunk in response.chunks]
        documents = [chunk.content for chunk in response.chunks]
        metadatas = [chunk.metadata for chunk in response.chunks]
        embeddings = [chunk.embedding for chunk in response.chunks if chunk.embedding is not None]

        if len(embeddings) != len(ids):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Embedding generation failed for some chunks.",
            )

        chroma_service.add_documents(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings,
        )

        postgres_service.add_documents(
            ids=ids, documents=documents, metadatas=metadatas,
        )

        return {
            "status": "Successfully uploaded, chunked, embedded, and stored in ChromaDB + PostgreSQL.",
            "chunks_stored": len(ids),
        }

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


@router.post("/api/v1/hybrid_search")
def hybrid_search(
    request: HybridSearchRequest,
    chroma_service: ChromaService = Depends(get_chroma_service),
    postgres_service: PostgresService = Depends(get_postgres_service),
) -> dict:
    """Run hybrid search: vector (ChromaDB) + BM25 (PostgreSQL) with RRF fusion."""
    fetch_count = request.n_results * 2
    k = 60  # RRF constant

    # ── Vector search (ChromaDB) ─────────────────────────────────────────
    try:
        vector_results = chroma_service.query(
            query_texts=[request.query], n_results=fetch_count,
        )
    except Exception:
        vector_results = []

    # ── BM25 search (PostgreSQL) ─────────────────────────────────────────
    try:
        bm25_results = postgres_service.query(
            query_texts=[request.query], n_results=fetch_count,
        )
    except Exception:
        bm25_results = []

    # ── Reciprocal Rank Fusion ────────────────────────────────────────────
    vector_weight = request.vector_weight
    bm25_weight = 1.0 - vector_weight

    doc_scores: dict[str, float] = {}
    doc_data: dict[str, dict] = {}
    doc_sources: dict[str, set[str]] = {}

    for rank, result in enumerate(vector_results):
        doc_id = result["id"]
        rrf_score = vector_weight * (1.0 / (k + rank + 1))
        doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + rrf_score
        doc_data[doc_id] = result
        doc_sources.setdefault(doc_id, set()).add("vector")

    for rank, result in enumerate(bm25_results):
        doc_id = result["id"]
        rrf_score = bm25_weight * (1.0 / (k + rank + 1))
        doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + rrf_score
        if doc_id not in doc_data:
            doc_data[doc_id] = result
        doc_sources.setdefault(doc_id, set()).add("bm25")

    # ── Sort by fused score and return top N ─────────────────────────────
    sorted_ids = sorted(doc_scores, key=lambda did: doc_scores[did], reverse=True)
    top_ids = sorted_ids[: request.n_results]

    results = []
    for doc_id in top_ids:
        data = doc_data[doc_id]
        results.append(
            {
                "id": doc_id,
                "document": data.get("document", ""),
                "metadata": data.get("metadata", {}),
                "score": round(doc_scores[doc_id], 6),
                "sources": sorted(doc_sources[doc_id]),
            }
        )

    return {
        "results": results,
        "strategy": f"rrf(vector_weight={vector_weight}, bm25_weight={bm25_weight}, k={k})",
    }