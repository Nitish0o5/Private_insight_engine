from __future__ import annotations

import os
import sys
from pathlib import Path

from pyspark.sql import SparkSession

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "my_data"
CHROMA_DIR = BASE_DIR / "chroma_store"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL_NAME = "phi3"
OLLAMA_HOST = "http://localhost:11434"
COLLECTION_NAME = "documents"


def create_spark_session(app_name: str = "PrivateInsightEngine") -> SparkSession:
    python_executable = sys.executable
    os.environ.setdefault("PYSPARK_PYTHON", python_executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", python_executable)

    return (
        SparkSession.builder.master("local[*]")
        .appName(app_name)
        .config("spark.pyspark.python", python_executable)
        .config("spark.pyspark.driver.python", python_executable)
        .getOrCreate()
    )


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
