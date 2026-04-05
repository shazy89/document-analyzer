from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from document_analyzer.core.config import Settings
from document_analyzer.models.chunking import ChunkResponse, DocumentChunk
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

if TYPE_CHECKING:
    from langchain_text_splitters import RecursiveCharacterTextSplitter


# ── Supported file extensions ────────────────────────────────

_TXT_EXTENSIONS = {".txt", ".md", ".csv", ".log"}
_PDF_EXTENSIONS = {".pdf"}
_SUPPORTED_EXTENSIONS = _TXT_EXTENSIONS | _PDF_EXTENSIONS


# ── Custom errors ────────────────────────────────────────────


class FileNotFoundChunkingError(FileNotFoundError):
    """Raised when the requested document does not exist."""


class UnsupportedFileTypeError(ValueError):
    """Raised when the file extension is not supported."""


class EmptyDocumentError(ValueError):
    """Raised when the document contains no extractable text."""


# ── Service ──────────────────────────────────────────────────


class ChunkingService:
    """Reads documents from disk and splits them into chunks.

    * **TXT / plain-text** files → recursive character splitting.
    * **PDF** files → page-level chunking via ``unstructured`` (partition + chunk_by_title).

    Follows the same construction pattern as ``TogetherChatService``:
    primitive-only ``__init__``, ``from_settings`` factory.
    """

    def __init__(self, *, base_path: str) -> None:
        self._base_path = Path(base_path).expanduser().resolve()

    @classmethod
    def from_settings(cls, settings: Settings) -> ChunkingService:
        return cls(base_path=settings.documents_path)

    # ── public API ───────────────────────────────────────────

    def chunk_file(
        self,
        file_name: str,
        *,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> ChunkResponse:
        """Read *file_name* from the base path and return chunks."""
        file_path = self._resolve_path(file_name)
        suffix = file_path.suffix.lower()

        if suffix in _TXT_EXTENSIONS:
            return self._chunk_txt(file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if suffix in _PDF_EXTENSIONS:
            return self._chunk_pdf(file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        raise UnsupportedFileTypeError(
            f"Unsupported file type '{suffix}'. "
            f"Supported: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}"
        )

    # ── TXT / plain-text strategy ────────────────────────────

    def _chunk_txt(
        self,
        path: Path,
        *,
        chunk_size: int,
        chunk_overlap: int,
    ) -> ChunkResponse:
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            raise EmptyDocumentError(f"Document is empty: {path.name}")

        splitter = self._build_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        raw_chunks = splitter.split_text(text)

        chunks: list[DocumentChunk] = []
        offset = 0
        for idx, chunk_text in enumerate(raw_chunks):
            char_offset = text.find(chunk_text, offset)
            if char_offset == -1:
                char_offset = offset
            chunks.append(
                DocumentChunk(
                    content=chunk_text,
                    chunk_index=idx,
                    metadata={
                        "source": path.name,
                        "char_offset": char_offset,
                        "strategy": "recursive",
                    },
                )
            )
            offset = char_offset + len(chunk_text)

        return ChunkResponse(
            file_name=path.name,
            strategy="recursive",
            total_chunks=len(chunks),
            original_length=len(text),
            chunks=chunks,
        )

    # ── PDF page-level strategy (unstructured) ──────────────

    def _chunk_pdf(
        self,
        path: Path,
        *,
        chunk_size: int,
        chunk_overlap: int,
    ) -> ChunkResponse:
        elements = self._partition_pdf(path)
        if not elements:
            raise EmptyDocumentError(f"PDF contains no extractable elements: {path.name}")

        chunked_elements = self._chunk_by_title(
            elements,
            max_characters=chunk_size,
            combine_text_under_n_chars=min(chunk_overlap, chunk_size),
        )

        if not chunked_elements:
            raise EmptyDocumentError(f"PDF contains no extractable text: {path.name}")

        total_length = sum(len(str(el)) for el in elements)
        splitter = self._build_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks: list[DocumentChunk] = []
        chunk_index = 0

        for chunk in chunked_elements:
            chunk_text = str(chunk)
            raw_chunks = splitter.split_text(chunk_text)
            offset = 0
            for ct in raw_chunks:
                char_offset = chunk_text.find(ct, offset)
                if char_offset == -1:
                    char_offset = offset
                chunks.append(
                    DocumentChunk(
                        content=ct,
                        chunk_index=chunk_index,
                        metadata={
                            "source": path.name,
                            "char_offset": char_offset,
                            "strategy": "unstructured+recursive",
                        },
                    )
                )
                offset = char_offset + len(ct)
                chunk_index += 1

        return ChunkResponse(
            file_name=path.name,
            strategy="page-level",
            total_chunks=len(chunks),
            original_length=total_length,
            chunks=chunks,
        )

    # ── internals ────────────────────────────────────────────

    def _resolve_path(self, file_name: str) -> Path:
        file_path = self._base_path / file_name
        if not file_path.is_file():
            raise FileNotFoundChunkingError(
                f"File not found: {file_path}  (base_path={self._base_path})"
            )
        return file_path

    @staticmethod
    def _build_splitter(*, chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    @staticmethod
    def _partition_pdf(path: Path) -> list:
        from unstructured.partition.pdf import partition_pdf

        return partition_pdf(
            filename=str(path),
            strategy="fast",
        )

    @staticmethod
    def _chunk_by_title(
        elements: list,
        *,
        max_characters: int,
        combine_text_under_n_chars: int,
    ) -> list:
        from unstructured.chunking.title import chunk_by_title

        return chunk_by_title(
            elements,
            multipage_sections=False,
            combine_text_under_n_chars=combine_text_under_n_chars,
            max_characters=max_characters,
        )
