# Document Analyzer

Minimal FastAPI + Pydantic scaffold for a terminal-first AI project using Together AI.

## Requirements

- Python 3.11+
- A `TOGETHER_API_KEY`

## Local setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Update `.env` with your Together API key before sending chat requests.

## Run the API

```bash
document-analyzer serve
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Chat request:
jz
```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Give me a one-line summary of FastAPI."}'
```

## Run from the terminal

One-shot prompt:

```bash
document-analyzer ask "Explain what this project scaffold is for."
```

Interactive chat loop:

```bash
document-analyzer chat
```

Type `exit` or `quit` to stop the terminal chat loop.
