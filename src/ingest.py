from __future__ import annotations

from pathlib import Path
from typing import Iterable

from pypdf import PdfReader
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StringType, StructField, StructType


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _read_txt_file(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf_file(file_path: Path) -> str:
    reader = PdfReader(str(file_path))
    pages: list[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def load_documents(data_dir: str | Path) -> list[dict[str, str]]:
    source_dir = Path(data_dir)
    documents: list[dict[str, str]] = []

    for file_path in sorted(source_dir.glob("**/*")):
        if file_path.is_dir():
            continue

        suffix = file_path.suffix.lower()
        if suffix == ".txt":
            text = _read_txt_file(file_path)
        elif suffix == ".pdf":
            text = _read_pdf_file(file_path)
        else:
            continue

        documents.append(
            {
                "doc_id": file_path.stem,
                "source": file_path.name,
                "file_path": str(file_path),
                "text": _normalize_text(text),
            }
        )

    return documents


def build_documents_dataframe(spark: SparkSession, data_dir: str | Path) -> DataFrame:
    schema = StructType(
        [
            StructField("doc_id", StringType(), False),
            StructField("source", StringType(), False),
            StructField("file_path", StringType(), False),
            StructField("text", StringType(), False),
        ]
    )
    documents = load_documents(data_dir)
    return spark.createDataFrame(documents, schema=schema)


def combine_documents(documents: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    return [dict(document) for document in documents]
