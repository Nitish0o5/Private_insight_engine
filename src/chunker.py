from __future__ import annotations

from typing import Iterable

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, concat, lit, posexplode, udf
from pyspark.sql.types import ArrayType, StringType


def chunk_text(text: str, chunk_size_words: int = 500, overlap_words: int = 100) -> list[str]:
    words = text.split()
    if not words:
        return []

    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be greater than zero")

    overlap_words = max(0, min(overlap_words, chunk_size_words - 1))
    step = chunk_size_words - overlap_words
    chunks: list[str] = []

    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size_words]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if start + chunk_size_words >= len(words):
            break

    return chunks


_chunk_text_udf = udf(lambda text, size, overlap: chunk_text(text, size, overlap), ArrayType(StringType()))


def chunk_dataframe(df: DataFrame, chunk_size_words: int = 500, overlap_words: int = 100) -> DataFrame:
    chunked = df.withColumn(
        "chunk_rows",
        _chunk_text_udf(col("text"), lit(chunk_size_words), lit(overlap_words)),
    )
    chunked = chunked.select(
        "doc_id",
        "source",
        "file_path",
        "text",
        posexplode(col("chunk_rows")).alias("chunk_index_zero_based", "chunk_text"),
    )
    chunked = chunked.withColumn("chunk_index", col("chunk_index_zero_based") + lit(1)).drop("chunk_index_zero_based")
    return chunked.withColumn(
        "chunk_id",
        concat(col("doc_id"), lit("_chunk_"), col("chunk_index")),
    )


def chunk_documents(documents: Iterable[dict[str, str]], chunk_size_words: int = 500, overlap_words: int = 100) -> list[dict[str, str]]:
    chunked_documents: list[dict[str, str]] = []
    for document in documents:
        for index, chunk in enumerate(chunk_text(document["text"], chunk_size_words, overlap_words), start=1):
            chunked_documents.append(
                {
                    "chunk_id": f"{document['doc_id']}_chunk_{index}",
                    "doc_id": document["doc_id"],
                    "source": document["source"],
                    "chunk_index": index,
                    "chunk_text": chunk,
                }
            )
    return chunked_documents
