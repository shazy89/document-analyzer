# Copilot Instructions

## Project Overview

Terminal-first FastAPI scaffold (`document-analyzer`) that wraps **Together AI** for LLM chat. Exposes both a REST API and a CLI. Python 3.11+, installed as an editable package.

## Architecture

```
core/config.py          – pydantic-settings singleton (get_settings, lru_cache)
models/chat.py          – Pydantic models: ChatRequest, ChatResponse, ChatMessage, HealthResponse
services/together_client.py – TogetherChatService, the only place that touches the Together SDK
api/router.py           – FastAPI routes: GET /health, POST /api/v1/chat
main.py                 – App factory: create_app() wires router into FastAPI
cli.py                  – argparse CLI: ask | chat | serve subcommands
```

Data flow: CLI or HTTP request → `TogetherChatService.ask()` → Together SDK → `ChatResponse`.

## Key Patterns

- **Service construction**: always use `TogetherChatService.from_settings(get_settings())` — never construct with raw kwargs outside tests.
- **Lazy client init**: `Together` SDK is imported and instantiated only on the first `ask()` call; `TOGETHER_NO_BANNER=1` is set via `os.environ.setdefault` before that import.
- **Settings aliasing**: env vars accept two names each (e.g. `DOC_ANALYZER_HOST` or `APP_HOST`) via `AliasChoices`. Use the `DOC_ANALYZER_*` prefix for project-specific overrides.
- **Protocol duck-typing**: Together SDK response types are typed with local `Protocol` classes in `together_client.py` to keep the module-level import inside `TYPE_CHECKING`.
- **`from __future__ import annotations`**: used in every module — keep it.

## Developer Workflows

```bash
# Setup (once)
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env   # then fill in TOGETHER_API_KEY

# Start API server
document-analyzer serve          # http://127.0.0.1:8000

# CLI usage
document-analyzer ask "Your prompt"
document-analyzer chat           # interactive loop; exit/quit to stop

# Quick API smoke test
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello"}'
```

Server can also start via `python -m document_analyzer serve`.

## Configuration Reference

| Env var | Alias | Default |
|---|---|---|
| `TOGETHER_API_KEY` | `DOC_ANALYZER_TOGETHER_API_KEY` | *(required for chat)* |
| `TOGETHER_MODEL` | `DOC_ANALYZER_TOGETHER_MODEL` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` |
| `DOC_ANALYZER_HOST` | `APP_HOST` | `127.0.0.1` |
| `DOC_ANALYZER_PORT` | `APP_PORT` | `8000` |

`together_api_key` is stored as `SecretStr`; always call `.get_secret_value()` to read it.

## Adding New Features

- **New endpoint**: add route to `api/router.py`; add request/response models to `models/chat.py`.
- **New service method**: extend `TogetherChatService` in `services/together_client.py`; keep Together SDK usage isolated there.
- **New CLI subcommand**: add subparser in `cli.py:build_parser()` and a `run_*` function; wire in `main()`.
