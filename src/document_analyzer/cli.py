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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "chat"

    if command == "serve":
        return run_server(host=args.host, port=args.port)
    if command == "ask":
        return run_single_prompt(prompt=args.prompt, system_prompt=args.system_prompt, model=args.model)
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
