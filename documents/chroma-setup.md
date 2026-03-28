# ChromaDB Local Setup Guide

This guide walks through running ChromaDB locally via Docker and connecting to it from the document-analyzer project.

## Prerequisites

- **Docker Desktop** installed and running ([download](https://www.docker.com/products/docker-desktop/))
- Project installed: `pip install -e .`

## 1. Start ChromaDB

From the project root:

```bash
docker compose up -d
```

This starts ChromaDB on **port 8100** with persistent storage.

## 2. Verify ChromaDB is running

```bash
curl http://localhost:8100/api/v1/heartbeat
```

Expected response:

```json
{"nanosecond heartbeat": 1234567890}
```

## 3. Configure environment variables

Copy the example env file if you haven't already:

```bash
cp .env.example .env
```

The ChromaDB defaults (already in `.env.example`):

```dotenv
CHROMA_HOST=localhost
CHROMA_PORT=8100
CHROMA_COLLECTION=documents
DOCUMENTS_PATH=/Users/erdoanshaziman/ai-practice-docs
```

## 4. Python quick-start

```python
from document_analyzer.core.config import get_settings
from document_analyzer.services.chroma_client import ChromaService

settings = get_settings()
chroma = ChromaService.from_settings(settings)

# Check connection
print(chroma.heartbeat())  # True

# Add documents
chroma.add_documents(
    ids=["doc1", "doc2"],
    documents=[
        "Revenue grew 15% in Q4 2025.",
        "Customer acquisition cost decreased by 8%.",
    ],
    metadatas=[
        {"source": "quarterly_report.pdf", "page": 3},
        {"source": "quarterly_report.pdf", "page": 7},
    ],
)

# Query
results = chroma.query(query_texts=["revenue growth"], n_results=2)
for r in results:
    print(r["document"], r["distance"])
```

## 5. Teardown

Stop and remove the container (data persists in the Docker volume):

```bash
docker compose down
```

To also **delete all stored data**:

```bash
docker compose down -v
```

## Port Map

| Service          | Port |
|------------------|------|
| FastAPI          | 8000 |
| ChromaDB         | 8100 |

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Connection refused` on 8100 | Run `docker compose up -d` and wait a few seconds |
| Port 8100 already in use | Change `CHROMA_PORT` in `.env` and update `docker-compose.yml` port mapping |
| `ModuleNotFoundError: chromadb` | Run `pip install -e .` to install the dependency |
