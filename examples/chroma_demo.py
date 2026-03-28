from __future__ import annotations

from datetime import datetime, UTC
from pathlib import Path

import chromadb

from document_analyzer.core.config import get_settings


def build_client() -> tuple[chromadb.ClientAPI, str]:
    settings = get_settings()

    try:
        http_client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
        )
        http_client.heartbeat()
        return http_client, f"http://{settings.chroma_host}:{settings.chroma_port}"
    except Exception:
        local_path = Path(".chroma-demo-data")
        persistent_client = chromadb.PersistentClient(path=str(local_path))
        return persistent_client, str(local_path.resolve())


def main() -> None:
    settings = get_settings()
    client, backend = build_client()

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    collection_name = f"{settings.chroma_collection}_demo_{run_id}"
    collection = client.get_or_create_collection(name=collection_name)

    documents = [
        {
            "id": f"{run_id}-revenue",
            "text": "Quarterly sales revenue grew 18% because enterprise renewals increased.",
            "metadata": {"topic": "revenue", "source": "demo", "run_id": run_id},
            "embedding": [1.0, 0.0, 0.0],
        },
        {
            "id": f"{run_id}-support",
            "text": "Customer support tickets dropped 12% after the onboarding redesign.",
            "metadata": {"topic": "support", "source": "demo", "run_id": run_id},
            "embedding": [0.0, 1.0, 0.0],
        },
        {
            "id": f"{run_id}-costs",
            "text": "Infrastructure costs fell 7% after removing unused compute capacity.",
            "metadata": {"topic": "costs", "source": "demo", "run_id": run_id},
            "embedding": [0.0, 0.0, 1.0],
        },
    ]

    collection.add(
        ids=[item["id"] for item in documents],
        documents=[item["text"] for item in documents],
        metadatas=[item["metadata"] for item in documents],
        embeddings=[item["embedding"] for item in documents],
    )

    query_text = "Which item talks about quarterly sales growth?"
    query_embedding = [[1.0, 0.0, 0.0]]
    result = collection.query(query_embeddings=query_embedding, n_results=3)

    print(f"Backend: {backend}")
    print(f"Collection: {collection_name}")
    print("Ingested data:")
    for item in documents:
        print(f"- id={item['id']}")
        print(f"  text={item['text']}")
        print(f"  metadata={item['metadata']}")
        print(f"  embedding={item['embedding']}")

    print(f"\nQuery: {query_text}")
    print(f"Query embedding: {query_embedding[0]}")

    ids = result.get("ids", [[]])[0]
    texts = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    print("\nResults:")
    for index, doc_id in enumerate(ids, start=1):
        text = texts[index - 1] if index - 1 < len(texts) else ""
        metadata = metadatas[index - 1] if index - 1 < len(metadatas) else {}
        distance = distances[index - 1] if index - 1 < len(distances) else None
        print(f"{index}. id={doc_id}")
        print(f"   distance={distance}")
        print(f"   text={text}")
        print(f"   metadata={metadata}")


if __name__ == "__main__":
    main()
