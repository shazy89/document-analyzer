from __future__ import annotations

import argparse
from collections.abc import Sequence
import sys

from document_analyzer.core.config import get_settings
from document_analyzer.services.together_client import MissingTogetherAPIKeyError, TogetherChatService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Terminal-first Together AI project scaffold")
    subparsers = parser.add_subparsers(dest="command")

    ask_parser = subparsers.add_parser("ask", help="Send a single prompt to Together AI")
    ask_parser.add_argument("prompt", help="Prompt to send to the model")
    ask_parser.add_argument("--system-prompt", dest="system_prompt", help="Optional system prompt")
    ask_parser.add_argument("--model", dest="model", help="Override the configured Together model")

    chat_parser = subparsers.add_parser("chat", help="Start an interactive terminal chat")
    chat_parser.add_argument("--system-prompt", dest="system_prompt", help="Optional system prompt")
    chat_parser.add_argument("--model", dest="model", help="Override the configured Together model")

    serve_parser = subparsers.add_parser("serve", help="Run the FastAPI server locally")
    serve_parser.add_argument("--host", help="Override the configured host")
    serve_parser.add_argument("--port", type=int, help="Override the configured port")

    chunk_parser = subparsers.add_parser("chunk", help="Chunk a document into smaller pieces")
    chunk_parser.add_argument("file_name", help="Name of the file to chunk (relative to documents_path)")
    chunk_parser.add_argument(
        "--chunk-size", dest="chunk_size", type=int, default=512,
        help="Maximum characters per chunk (default: 512)",
    )
    chunk_parser.add_argument(
        "--chunk-overlap", dest="chunk_overlap", type=int, default=50,
        help="Overlap in characters between chunks (default: 50)",
    )
    chunk_parser.add_argument(
        "--embed", action="store_true", default=False,
        help="Compute embeddings for each chunk",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "chat"

    if command == "serve":
        return run_server(host=args.host, port=args.port)
    if command == "ask":
        return run_single_prompt(prompt=args.prompt, system_prompt=args.system_prompt, model=args.model)
    if command == "chunk":
        return run_chunk(
            file_name=args.file_name,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            embed=args.embed,
        )
    return run_chat_loop(system_prompt=getattr(args, "system_prompt", None), model=getattr(args, "model", None))


def run_server(*, host: str | None, port: int | None) -> int:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "document_analyzer.main:app",
        host=host or settings.host,
        port=port or settings.port,
        reload=False,
    )
    return 0


def run_single_prompt(*, prompt: str, system_prompt: str | None, model: str | None) -> int:
    service = TogetherChatService.from_settings(get_settings())
    try:
        response = service.ask(prompt=prompt, system_prompt=system_prompt, model=model)
    except MissingTogetherAPIKeyError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(response.answer)
    return 0


def run_chunk(*, file_name: str, chunk_size: int, chunk_overlap: int, embed: bool) -> int:
    from document_analyzer.services.chunking_service import (
        ChunkingService,
        EmptyDocumentError,
        FileNotFoundChunkingError,
        UnsupportedFileTypeError,
    )

    settings = get_settings()
    service = ChunkingService.from_settings(settings)

    try:
        result = service.chunk_file(file_name, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    except (FileNotFoundChunkingError, UnsupportedFileTypeError, EmptyDocumentError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if embed:
        from document_analyzer.services.embedding_service import EmbeddingService

        embedding_service = EmbeddingService()
        embedding_service.embed_chunks(result.chunks)

    print(f"File:            {result.file_name}")
    print(f"Strategy:        {result.strategy}")
    print(f"Original length: {result.original_length} characters")
    print(f"Total chunks:    {result.total_chunks}")

    if result.chunks:
        preview = result.chunks[0].content[:200]
        print(f"\nFirst chunk preview:\n{preview}...")
        if embed and result.chunks[0].embedding:
            dims = len(result.chunks[0].embedding)
            print(f"Embedding dims:  {dims}")
            print(f"First 5 values:  {result.chunks[0].embedding[:5]}")

    return 0


def run_chat_loop(*, system_prompt: str | None, model: str | None) -> int:
    service = TogetherChatService.from_settings(get_settings())
    print("Interactive chat started. Type 'exit' or 'quit' to stop.")

    while True:
        try:
            prompt = input("> ").strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            return 0

        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            return 0

        try:
            response = service.ask(prompt=prompt, system_prompt=system_prompt, model=model)
        except MissingTogetherAPIKeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        print(response.answer)


if __name__ == "__main__":
    raise SystemExit(main())
