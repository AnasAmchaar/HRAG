"""
Humanized-RAG Configuration.

Hyperparameters from the paper:
  "Toward Human-Inspired RAG: Hierarchical Vector Compression
   and Topic-Guided Retrieval"

All settings can be overridden via environment variables prefixed with ``HRAG_``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
# Default to current working directory; users configure via env vars or API.
DATA_DIR = Path(os.getenv("HRAG_DATA_DIR", Path.cwd() / "data" / "sample"))
STORAGE_DIR = Path(os.getenv("HRAG_STORAGE_DIR", Path.cwd() / "storage"))

# ── Embedding Model (Section V.A) ───────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv("HRAG_EMBEDDING_MODEL", "all-mpnet-base-v2")
EMBEDDING_DIM: int = 768

# ── Chunking (Section V.B) ──────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("HRAG_CHUNK_SIZE", "384"))
CHUNK_OVERLAP: float = float(os.getenv("HRAG_CHUNK_OVERLAP", "0.15"))

# ── Clustering (Section V.C) ────────────────────────────────────────────────
# min_cluster_size = max(3, n // 20)  — computed dynamically
MIN_CLUSTER_SIZE_FLOOR: int = 3
MIN_CLUSTER_SIZE_DIVISOR: int = 20

# ── Tree Construction (Section V.D) ─────────────────────────────────────────
MAX_TREE_DEPTH: int = int(os.getenv("HRAG_MAX_TREE_DEPTH", "4"))

# ── Retrieval (Section V.F) ─────────────────────────────────────────────────
SIMILARITY_THRESHOLD: float = float(os.getenv("HRAG_SIMILARITY_THRESHOLD", "0.5"))
BRANCHING_FACTOR: int = int(os.getenv("HRAG_BRANCHING_FACTOR", "3"))
TOP_K: int = int(os.getenv("HRAG_TOP_K", "5"))

# ── LLM Generation ─────────────────────────────────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")

# ── Serialization ──────────────────────────────────────────────────────────
INDEX_FILE = Path(os.getenv("HRAG_INDEX_FILE", STORAGE_DIR / "forest_index.json"))


def compute_min_cluster_size(n: int) -> int:
    """Compute min_cluster_size = max(3, n // 20) as per Section V.C."""
    return max(MIN_CLUSTER_SIZE_FLOOR, n // MIN_CLUSTER_SIZE_DIVISOR)
