from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunker import chunk_documents
from src.config import DATA_DIR, ensure_directories, create_spark_session
from src.embedder import embed_texts
from src.ingest import load_documents
from src.llm import ask_ollama
from src.retriever import build_prompt, retrieve_chunks
from src.vectordb import store_dataframe


def index_documents() -> None:
    ensure_directories()
    documents = load_documents(DATA_DIR)
    if not documents:
        print(f"No .txt or .pdf files found in {DATA_DIR}")
        return

    chunk_rows = chunk_documents(documents)
    if not chunk_rows:
        print("No chunks were generated from the documents.")
        return

    embeddings = embed_texts(chunk["chunk_text"] for chunk in chunk_rows)
    enriched_rows = [
        {**chunk, "embedding": embedding}
        for chunk, embedding in zip(chunk_rows, embeddings)
    ]

    spark = create_spark_session()
    try:
        chunks_df = spark.createDataFrame(enriched_rows)
        store_dataframe(chunks_df)
        print(f"Indexed {len(documents)} documents and stored {len(enriched_rows)} chunks in ChromaDB.")
    finally:
        spark.stop()


def ask_question() -> None:
    ensure_directories()
    question = input("Ask a question: ").strip()
    if not question:
        print("Question cannot be empty.")
        return

    retrieved_chunks = retrieve_chunks(question, top_k=3)
    if not retrieved_chunks:
        print("No relevant chunks found. Index your documents first.")
        return

    prompt = build_prompt(question, retrieved_chunks)
    try:
        answer = ask_ollama(prompt)
    except Exception as error:
        print(f"Failed to query Ollama: {error}")
        return

    print("\nAnswer:\n")
    print(answer)
    print("\nSources:")
    for chunk in retrieved_chunks:
        metadata = chunk.get("metadata", {}) or {}
        print(f"- {metadata.get('source', 'unknown')} (chunk {metadata.get('chunk_index', '?')})")


def main() -> None:
    ensure_directories()
    while True:
        print("\nPrivate Insight Engine")
        print("1. Index documents")
        print("2. Ask a question")
        print("3. Exit")

        choice = input("Choose an option: ").strip()
        if choice == "1":
            index_documents()
        elif choice == "2":
            ask_question()
        elif choice == "3":
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
