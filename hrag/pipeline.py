"""
End-to-End RAG Pipeline.

Orchestrates the full Humanized-RAG workflow:
  - Ingest: chunking → embedding → tree construction → serialization
  - Query: embedding → forest traversal → LLM generation
  - Compare: hierarchical vs flat retrieval side-by-side
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from hrag.config import (
    BRANCHING_FACTOR,
    INDEX_FILE,
    SIMILARITY_THRESHOLD,
    TOP_K,
)
from hrag.builder import build_forest
from hrag.chunker import Chunk, load_and_chunk_directory, load_and_chunk_file
from hrag.embedder import Embedder
from hrag.generator import generate_answer
from hrag.retriever import (
    RetrievalResult,
    RetrievalStats,
    retrieve_flat,
    retrieve_hierarchical,
)
from hrag.tree import TopicForest

logger = logging.getLogger(__name__)


class HumanizedRAGPipeline:
    """End-to-end Humanized-RAG pipeline."""

    def __init__(self, index_path: Path = INDEX_FILE):
        self.index_path = Path(index_path)
        self.embedder: Optional[Embedder] = None
        self.forest: Optional[TopicForest] = None

    def _ensure_embedder(self) -> Embedder:
        if self.embedder is None:
            self.embedder = Embedder()
        return self.embedder

    # ── Ingestion ────────────────────────────────────────────────────────────

    def ingest(
        self,
        source: str | Path,
        save: bool = True,
    ) -> TopicForest:
        """Ingest documents and build the hierarchical index.

        Args:
            source: Path to a file or directory of documents.
            save: Whether to serialize the forest to disk.

        Returns:
            The constructed TopicForest.
        """
        source = Path(source)
        embedder = self._ensure_embedder()

        # Step 1: Chunking
        logger.info("[1/4] Chunking documents from: %s", source)
        t0 = time.time()
        if source.is_dir():
            chunks = load_and_chunk_directory(source)
        elif source.is_file():
            chunks = load_and_chunk_file(source)
        else:
            raise FileNotFoundError(f"Source not found: {source}")

        logger.info("       -> %d chunks created (%.1fs)", len(chunks), time.time() - t0)

        if not chunks:
            raise ValueError("No chunks were created. Check input documents.")

        # Step 2: Embedding
        logger.info("[2/4] Embedding chunks with %s...", embedder.model_name)
        t0 = time.time()
        texts = [c.text for c in chunks]
        vectors = embedder.encode(texts)
        logger.info("       -> Embeddings shape: %s (%.1fs)", vectors.shape, time.time() - t0)

        # Step 3: Tree construction
        logger.info("[3/4] Building hierarchical topic-tree forest...")
        t0 = time.time()
        self.forest = build_forest(chunks, vectors)
        logger.info("       -> %s", self.forest.summary().split("\n")[0])
        logger.info("       -> Built in %.1fs", time.time() - t0)

        # Step 4: Serialization
        if save:
            logger.info("[4/4] Saving index to: %s", self.index_path)
            t0 = time.time()
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            self.forest.save(self.index_path)
            size_mb = self.index_path.stat().st_size / (1024 * 1024)
            logger.info("       -> Index size: %.1f MB (%.1fs)", size_mb, time.time() - t0)

        return self.forest

    def load_index(self) -> TopicForest:
        """Load a previously saved index from disk."""
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"No index found at {self.index_path}. Run `ingest` first."
            )
        logger.info("Loading index from: %s", self.index_path)
        self.forest = TopicForest.load(self.index_path)
        logger.info("-> %s", self.forest.summary().split("\n")[0])
        return self.forest

    # ── Querying ─────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        k: int = TOP_K,
        tau: float = SIMILARITY_THRESHOLD,
        B: int = BRANCHING_FACTOR,
        use_ollama: bool = True,
    ) -> dict:
        """Query the index and generate an answer.

        Args:
            question: The user's question.
            k: Number of chunks to retrieve.
            tau: Similarity threshold for pruning.
            B: Branching factor.
            use_ollama: Whether to use Ollama for generation.

        Returns:
            Dict with 'answer', 'results', 'stats', and 'latency'.
        """
        if self.forest is None:
            self.load_index()

        embedder = self._ensure_embedder()

        # Embed query
        t0 = time.time()
        query_vector = embedder.encode_query(question)
        embed_time = time.time() - t0

        # Hierarchical retrieval
        t0 = time.time()
        results, stats = retrieve_hierarchical(
            self.forest, query_vector, tau=tau, k=k, B=B
        )
        retrieval_time = time.time() - t0

        # Generation
        t0 = time.time()
        answer = generate_answer(question, results, use_ollama=use_ollama)
        generation_time = time.time() - t0

        return {
            "answer": answer,
            "results": results,
            "stats": stats,
            "latency": {
                "embedding_ms": embed_time * 1000,
                "retrieval_ms": retrieval_time * 1000,
                "generation_ms": generation_time * 1000,
                "total_ms": (embed_time + retrieval_time + generation_time) * 1000,
            },
        }

    # ── Comparison ───────────────────────────────────────────────────────────

    def compare(
        self,
        question: str,
        k: int = TOP_K,
    ) -> dict:
        """Compare hierarchical vs flat retrieval on the same query.

        Returns a dict with both sets of results and stats.
        """
        if self.forest is None:
            self.load_index()

        embedder = self._ensure_embedder()
        query_vector = embedder.encode_query(question)

        # Hierarchical retrieval
        t0 = time.time()
        h_results, h_stats = retrieve_hierarchical(self.forest, query_vector, k=k)
        h_time = time.time() - t0

        # Flat retrieval
        t0 = time.time()
        f_results, f_stats = retrieve_flat(self.forest, query_vector, k=k)
        f_time = time.time() - t0

        return {
            "question": question,
            "hierarchical": {
                "results": h_results,
                "stats": h_stats,
                "latency_ms": h_time * 1000,
            },
            "flat": {
                "results": f_results,
                "stats": f_stats,
                "latency_ms": f_time * 1000,
            },
        }

    def stats(self) -> dict:
        """Return statistics about the current index."""
        if self.forest is None:
            self.load_index()

        forest = self.forest
        total_leaves = forest.total_leaves
        total_nodes = forest.total_nodes
        compression_ratio = total_leaves / total_nodes if total_nodes > 0 else 0

        tree_stats = []
        for i, tree in enumerate(forest.trees):
            tree_stats.append({
                "tree_id": tree.tree_id,
                "depth": tree.depth,
                "nodes": tree.node_count,
                "leaves": tree.leaf_count,
            })

        return {
            "tree_count": forest.tree_count,
            "total_nodes": total_nodes,
            "total_leaves": total_leaves,
            "compression_ratio": f"{compression_ratio:.2f}",
            "index_size_mb": (
                self.index_path.stat().st_size / (1024 * 1024)
                if self.index_path.exists()
                else 0
            ),
            "trees": tree_stats,
        }
