from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from src.chunker import chunk_documents
from src.config import DATA_DIR, OLLAMA_HOST, OLLAMA_MODEL_NAME, create_spark_session, ensure_directories
from src.ingest import load_documents
from src.llm import ask_ollama


def index_documents_for_ui() -> tuple[int, int]:
    from src.embedder import embed_texts
    from src.vectordb import store_dataframe

    ensure_directories()
    documents = load_documents(DATA_DIR)
    if not documents:
        raise ValueError(f"No .txt or .pdf files found in {DATA_DIR}")

    chunk_rows = chunk_documents(documents)
    if not chunk_rows:
        raise ValueError("No chunks were generated from the documents.")

    embeddings = embed_texts(chunk["chunk_text"] for chunk in chunk_rows)
    enriched_rows = [{**chunk, "embedding": embedding} for chunk, embedding in zip(chunk_rows, embeddings)]

    spark = create_spark_session("PrivateInsightEngineStreamlit")
    try:
        chunks_df = spark.createDataFrame(enriched_rows)
        store_dataframe(chunks_df)
    finally:
        spark.stop()

    return len(documents), len(enriched_rows)


def format_source(chunk: dict[str, object]) -> str:
    metadata = chunk.get("metadata", {}) or {}
    source = metadata.get("source", "unknown")
    chunk_index = metadata.get("chunk_index", "?")
    distance = chunk.get("distance")
    if distance is None:
        return f"{source} | chunk {chunk_index}"
    return f"{source} | chunk {chunk_index} | distance {distance:.4f}"


st.set_page_config(page_title="Private Insight Engine", page_icon="AI", layout="wide")

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .hero {
            background: linear-gradient(135deg, #f7f1e3 0%, #d7e9ea 100%);
            border: 1px solid rgba(11, 31, 58, 0.12);
            border-radius: 24px;
            padding: 1.5rem 1.75rem;
            margin-bottom: 1.25rem;
        }
        .hero h1 {
            color: #0b1f3a;
            font-size: 2.35rem;
            margin-bottom: 0.25rem;
        }
        .hero p {
            color: #31445f;
            font-size: 1.02rem;
            margin-bottom: 0;
        }
        .card {
            background: white;
            border: 1px solid rgba(11, 31, 58, 0.1);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: 0 8px 30px rgba(11, 31, 58, 0.06);
        }
        .muted {
            color: #5b6b7f;
            font-size: 0.95rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

ensure_directories()

documents = load_documents(DATA_DIR)

st.markdown(
    f"""
    <div class="hero">
        <h1>Private Insight Engine</h1>
        <p>Index local files, retrieve relevant chunks, and query your Ollama model without leaving the workspace.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, middle, right = st.columns(3)
left.metric("Documents found", len(documents))
middle.metric("Chunks in Chroma", "Load on use")
right.metric("Model", OLLAMA_MODEL_NAME)

st.caption(f"Ollama host: {OLLAMA_HOST}")

tab_index, tab_ask = st.tabs(["Index documents", "Ask a question"])

with tab_index:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Build the local vector store")
    st.write("Use this when you add or change files in my_data/.")

    if documents:
        preview_sources = [document["source"] for document in documents[:5]]
        st.write("Detected files:")
        st.write(", ".join(preview_sources))
    else:
        st.info("Put .txt or .pdf files in my_data/ before indexing.")

    if st.button("Index documents", type="primary", use_container_width=True):
        with st.spinner("Reading documents, chunking text, generating embeddings, and saving to ChromaDB..."):
            try:
                doc_count, chunk_count = index_documents_for_ui()
            except Exception as error:
                st.error(f"Indexing failed: {error}")
            else:
                st.success(f"Indexed {doc_count} documents and stored {chunk_count} chunks.")
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

with tab_ask:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Ask questions against your documents")
    question = st.text_area("Question", placeholder="What do the notes say about ...?", height=120)

    ask_pressed = st.button("Ask question", type="primary", use_container_width=True)
    if ask_pressed:
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            with st.spinner("Retrieving relevant chunks and querying Ollama..."):
                try:
                    from src.retriever import build_prompt, retrieve_chunks

                    retrieved_chunks = retrieve_chunks(question.strip(), top_k=3)
                    if not retrieved_chunks:
                        st.warning("No relevant chunks found. Index documents first.")
                    else:
                        prompt = build_prompt(question.strip(), retrieved_chunks)
                        answer = ask_ollama(prompt)
                        st.markdown("### Answer")
                        st.write(answer)
                        st.markdown("### Sources")
                        for chunk in retrieved_chunks:
                            st.write(format_source(chunk))
                            st.caption(chunk.get("chunk_text", ""))
                except Exception as error:
                    st.error(f"Question answering failed: {error}")
    st.markdown("</div>", unsafe_allow_html=True)
