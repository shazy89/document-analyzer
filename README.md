# Document Analyzer

Minimal FastAPI + Pydantic scaffold for a terminal-first AI project using Together AI with hybrid search (vector + BM25).

## Requirements

- Python 3.11+
- Docker Desktop (for ChromaDB + PostgreSQL)
- A `TOGETHER_API_KEY`

## Local setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Update `.env` with your Together API key before sending chat requests.

## Start infrastructure

```bash
docker compose up -d
```

This starts:
- **ChromaDB** on port 8100 (vector search)
- **PostgreSQL** on port 5434 (BM25 full-text search)

## Run the API

```bash
document-analyzer serve
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Chat request:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Give me a one-line summary of FastAPI."}'
```

## Upload & Search

Upload a document (chunks stored in both ChromaDB and PostgreSQL):

```bash
curl -X POST http://127.0.0.1:8000/api/v1/upload_file \
  -H 'Content-Type: application/json' \
  -d '{"file_name": "my_document.txt"}'
```

Hybrid search (vector + BM25 with Reciprocal Rank Fusion):

```bash
curl -X POST http://127.0.0.1:8000/api/v1/hybrid_search \
  -H 'Content-Type: application/json' \
  -d '{"query": "revenue growth", "n_results": 5, "vector_weight": 0.5}'
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

## Port Map

| Service    | Port |
|------------|------|
| FastAPI    | 8000 |
| ChromaDB   | 8100 |
| PostgreSQL | 5434 |


# Troubleshooting ChromaDB
chroma browse documents --host http://127.0.0.1:8100