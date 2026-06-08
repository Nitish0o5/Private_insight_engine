from __future__ import annotations

import hashlib
import math
from functools import lru_cache
from typing import Iterable

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, udf
from pyspark.sql.types import ArrayType, DoubleType

from src.config import EMBEDDING_MODEL_NAME

EMBEDDING_DIMENSION = 384


@lru_cache(maxsize=1)
def get_embedding_model():
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)
    except Exception:
        return None


def _hash_embedding(text: str) -> list[float]:
    vector = [0.0] * EMBEDDING_DIMENSION
    for token in text.lower().split():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % EMBEDDING_DIMENSION
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def embed_text(text: str) -> list[float]:
    model = get_embedding_model()
    if model is None:
        return _hash_embedding(text)

    embedding = model.encode(text, normalize_embeddings=True)
    return [float(value) for value in embedding.tolist()]


_embed_text_udf = udf(lambda text: embed_text(text), ArrayType(DoubleType()))


def embed_dataframe(df: DataFrame, text_column: str = "chunk_text", embedding_column: str = "embedding") -> DataFrame:
    return df.withColumn(embedding_column, _embed_text_udf(col(text_column)))


def embed_texts(texts: Iterable[str]) -> list[list[float]]:
    text_list = list(texts)
    model = get_embedding_model()
    if model is None:
        return [_hash_embedding(text) for text in text_list]

    embeddings = model.encode(text_list, normalize_embeddings=True)
    return [[float(value) for value in row] for row in embeddings]
