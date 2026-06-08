# Private Insight Engine

Private Insight Engine is a fully local retrieval-augmented generation (RAG) prototype that uses PySpark for document processing, Sentence Transformers for embeddings, ChromaDB for vector storage, and Ollama for local answer generation.

## What it does

- Reads local TXT and PDF files from `my_data/`
- Loads documents into Spark DataFrames
- Chunks text with overlap for better retrieval
- Generates embeddings with `all-MiniLM-L6-v2`
- Stores vectors in local ChromaDB
- Answers questions with a local Ollama model

## Project Structure

- `my_data/` local source documents
- `src/` application code
- `chroma_store/` persistent vector database files

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Install Ollama and pull a model:

```bash
ollama pull phi3
```

or

```bash
ollama pull mistral
```

## Run

Start the CLI:

```bash
python src/main.py
```

Then choose:

1. Index documents
2. Ask a question
3. Exit

Start the Streamlit UI:

```bash
streamlit run streamlit_app.py
```

The UI lets you index files from `my_data/` and ask questions directly in the browser.

## Notes

- Put your documents in `my_data/` before indexing.
- The first run will download the embedding model.
- If Ollama is not running, question answering will fail with a clear error.
