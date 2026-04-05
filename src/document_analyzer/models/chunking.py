from __future__ import annotations

from pydantic import BaseModel, Field


class ChunkRequest(BaseModel):
    """Input model for the chunking endpoint / CLI."""

    file_name: str = Field(min_length=1, description="Name of the file to chunk (relative to base path)")
    chunk_size: int = Field(default=512, gt=0, description="Maximum characters per chunk")
    chunk_overlap: int = Field(default=50, ge=0, description="Overlap in characters between consecutive chunks")
    embed: bool = Field(default=False, description="Whether to compute embeddings for each chunk")


class DocumentChunk(BaseModel):
    """A single chunk produced by the chunking service."""

    content: str
    chunk_index: int
    embedding: list[float] | None = Field(default=None, description="Vector embedding of the chunk content")
    metadata: dict[str, str | int | None] = Field(default_factory=dict)


class ChunkResponse(BaseModel):
    """Output model returned after chunking a document."""

    file_name: str
    strategy: str  # "recursive" for TXT, "page-level" for PDF
    total_chunks: int
    original_length: int
    chunks: list[DocumentChunk]

class UploadFileRequest(BaseModel):
    """Input model for the upload and chunk endpoint."""

    file_name: str = Field(min_length=1, description="Name of the file to upload and chunk (relative to base path)")
    chunk_size: int = Field(default=512, gt=0, description="Maximum characters per chunk")
    chunk_overlap: int = Field(default=50, ge=0, description="Overlap in characters between consecutive chunks")
    embed: bool = Field(default=False, description="Whether to compute embeddings for each chunk")