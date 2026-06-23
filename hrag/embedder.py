"""
Embedding Model Wrapper (Section V.A).

Thin wrapper around sentence-transformers for encoding text into
dense embeddings. Uses all-mpnet-base-v2 (768d) by default.

All vectors are L2-normalized for cosine similarity computation.
"""

from __future__ import annotations

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

from hrag.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class Embedder:
    """Wraps a sentence-transformer model for text embedding."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """Initialize the embedding model.

        Args:
            model_name: Name of the sentence-transformers model to load.
        """
        self.model_name = model_name
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info("Model loaded — dimension: %d", self.dimension)

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode a list of texts into normalized embedding vectors.

        Args:
            texts: List of text strings to encode.
            batch_size: Batch size for encoding.

        Returns:
            np.ndarray of shape (len(texts), dimension), L2-normalized.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,  # L2 normalize for cosine sim
        )
        return np.array(embeddings, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string.

        Args:
            query: The query text.

        Returns:
            np.ndarray of shape (dimension,), L2-normalized.
        """
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )
        return np.array(embedding[0], dtype=np.float32)
