# PostgreSQL BM25 Setup Guide

This guide walks through running PostgreSQL locally via Docker for BM25 full-text search alongside ChromaDB.

## Prerequisites

- **Docker Desktop** installed and running ([download](https://www.docker.com/products/docker-desktop/))
- Project installed: `pip install -e .`

## 1. Start PostgreSQL (and ChromaDB)

From the project root:

```bash
docker compose up -d
```

This starts:
- **ChromaDB** on port **8100** (vector search)
- **PostgreSQL** on port **5434** (BM25 full-text search)

## 2. Verify PostgreSQL is running

```bash
psql -h localhost -p 5434 -U ed -d document_analyzer -c "SELECT 1;"
```

Password: `123123`

Expected: a single row with value `1`.

## 3. Configure environment variables

Copy the example env file if you haven't already:

```bash
cp .env.example .env
```

The PostgreSQL defaults (already in `.env.example`):

```dotenv
POSTGRES_HOST=localhost
POSTGRES_PORT=5434
POSTGRES_USER=ed
POSTGRES_PASSWORD=123123
POSTGRES_DB=document_analyzer
```

## 4. Python quick-start

```python
from document_analyzer.core.config import get_settings
from document_analyzer.services.postgres_client import PostgresService

settings = get_settings()
pg = PostgresService.from_settings(settings)

# Initialize schema (creates table + indexes)
pg.init_schema()

# Check connection
print(pg.heartbeat())  # True

# Add documents
pg.add_documents(
    ids=["doc1", "doc2"],
    documents=[
        "Revenue grew 15% in Q4 2025.",
        "Customer acquisition cost decreased by 8%.",
    ],
    metadatas=[
        {"source": "quarterly_report.pdf", "page": "3"},
        {"source": "quarterly_report.pdf", "page": "7"},
    ],
)

# BM25 full-text search
results = pg.query(query_texts=["revenue"], n_results=2)
for r in results:
    print(r["document"], r["score"])
```

## 5. Hybrid Search (Vector + BM25)

After uploading documents via the API (which stores in both ChromaDB and PostgreSQL):

```bash
# Upload a file (chunks go to both stores)
curl -X POST http://127.0.0.1:8000/api/v1/upload_file \
  -H 'Content-Type: application/json' \
  -d '{"file_name": "my_document.txt"}'

# Hybrid search (combines vector + BM25 results via RRF)
curl -X POST http://127.0.0.1:8000/api/v1/hybrid_search \
  -H 'Content-Type: application/json' \
  -d '{"query": "revenue growth", "n_results": 5}'

# Adjust weighting (0.0 = pure BM25, 1.0 = pure vector)
curl -X POST http://127.0.0.1:8000/api/v1/hybrid_search \
  -H 'Content-Type: application/json' \
  -d '{"query": "revenue growth", "n_results": 5, "vector_weight": 0.7}'
```

## 6. Teardown

Stop containers (data persists in Docker volumes):

```bash
docker compose down
```

Delete all stored data:

```bash
docker compose down -v
```

## Port Map

| Service    | Port |
|------------|------|
| FastAPI    | 8000 |
| ChromaDB   | 8100 |
| PostgreSQL | 5434 |

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Connection refused` on 5434 | Run `docker compose up -d` and wait a few seconds |
| Port 5434 already in use | Change `POSTGRES_PORT` in `.env` and update `docker-compose.yml` port mapping |
| `ModuleNotFoundError: psycopg` | Run `pip install -e .` to install the dependency |
| `FATAL: password authentication failed` | Ensure `POSTGRES_PASSWORD` in `.env` matches `docker-compose.yml` |
