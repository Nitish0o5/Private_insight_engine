from __future__ import annotations

from src.config import COLLECTION_NAME
from src.embedder import embed_text
from src.vectordb import get_collection


def retrieve_chunks(question: str, top_k: int = 3, collection_name: str = COLLECTION_NAME) -> list[dict[str, object]]:
    collection = get_collection(collection_name)
    query_embedding = embed_text(question)
    result = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    ids = result.get("ids", [[]])[0]

    retrieved: list[dict[str, object]] = []
    for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
        retrieved.append(
            {
                "chunk_id": chunk_id,
                "chunk_text": document,
                "metadata": metadata or {},
                "distance": distance,
            }
        )

    return retrieved


def build_context(retrieved_chunks: list[dict[str, object]]) -> str:
    sections: list[str] = []
    for index, chunk in enumerate(retrieved_chunks, start=1):
        metadata = chunk.get("metadata", {}) or {}
        source = metadata.get("source", "unknown")
        sections.append(f"[{index}] Source: {source}\n{chunk['chunk_text']}")
    return "\n\n".join(sections)


def build_prompt(question: str, retrieved_chunks: list[dict[str, object]]) -> str:
    context = build_context(retrieved_chunks)
    return (
        "You are a document assistant. Answer only from the provided context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        'If the answer is not in the context, say: "I could not find that in your documents."'
    )
