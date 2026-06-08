from __future__ import annotations

import chromadb
from pyspark.sql import DataFrame

from src.config import CHROMA_DIR, COLLECTION_NAME


def get_chroma_client() -> chromadb.PersistentClient:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection(collection_name: str = COLLECTION_NAME):
    client = get_chroma_client()
    return client.get_or_create_collection(name=collection_name, embedding_function=None)


def store_dataframe(df: DataFrame, collection_name: str = COLLECTION_NAME) -> None:
    collection = get_collection(collection_name)
    rows = df.select("chunk_id", "source", "doc_id", "chunk_index", "chunk_text", "embedding").collect()

    if not rows:
        return

    collection.upsert(
        ids=[row["chunk_id"] for row in rows],
        documents=[row["chunk_text"] for row in rows],
        embeddings=[row["embedding"] for row in rows],
        metadatas=[
            {
                "source": row["source"],
                "doc_id": row["doc_id"],
                "chunk_index": int(row["chunk_index"]),
            }
            for row in rows
        ],
    )


def collection_count(collection_name: str = COLLECTION_NAME) -> int:
    collection = get_collection(collection_name)
    return collection.count()
