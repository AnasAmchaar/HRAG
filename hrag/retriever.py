"""
Top-Down Forest Traversal — Retriever (Algorithm 1, Section IV.D).

Implements the query-time retrieval procedure:
  1. Compare query against all root nodes
  2. Prune trees with sim < τ; keep top-B trees
  3. For each surviving tree: BFS descent, at each node score children,
     enqueue top-B children with sim ≥ τ
  4. Collect reached leaf nodes
  5. Return top-k leaves ranked by similarity

Complexity: O(D + B·L) similarity comparisons per query,
            vs O(N) for flat retrieval over N chunks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from hrag.config import BRANCHING_FACTOR, SIMILARITY_THRESHOLD, TOP_K
from hrag.tree import TopicForest, TreeNode

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieval result with score and provenance."""

    text: str
    score: float
    metadata: dict
    traversal_path: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        preview = self.text[:60] + "..." if len(self.text) > 60 else self.text
        return f'RetrievalResult(score={self.score:.4f}, text="{preview}")'


@dataclass
class RetrievalStats:
    """Statistics for a single retrieval operation."""

    total_similarity_comparisons: int = 0
    trees_scanned: int = 0
    trees_pruned: int = 0
    levels_traversed: int = 0
    leaves_reached: int = 0


def _sim(query: np.ndarray, vector: np.ndarray) -> float:
    """Compute cosine similarity between query and a single vector."""
    return float(cosine_similarity(
        query.reshape(1, -1), vector.reshape(1, -1)
    )[0, 0])


def retrieve_hierarchical(
    forest: TopicForest,
    query_vector: np.ndarray,
    tau: float = SIMILARITY_THRESHOLD,
    k: int = TOP_K,
    B: int = BRANCHING_FACTOR,
) -> tuple[list[RetrievalResult], RetrievalStats]:
    """Execute top-down forest traversal (Algorithm 1).

    Args:
        forest: The TopicForest to search.
        query_vector: Query embedding, shape (d,).
        tau: Similarity threshold τ for pruning.
        k: Number of top results to return.
        B: Maximum branching factor at each level.

    Returns:
        results: Top-k RetrievalResult objects, ranked by similarity.
        stats: RetrievalStats with comparison counts and traversal info.
    """
    stats = RetrievalStats()
    all_candidates: list[tuple[float, TreeNode, list[str]]] = []

    # ── Step 1: Root-level matching ──────────────────────────────────────────
    tree_scores: list[tuple[float, int]] = []
    for i, tree in enumerate(forest.trees):
        score = _sim(query_vector, tree.root.vector)
        stats.total_similarity_comparisons += 1
        if score >= tau:
            tree_scores.append((score, i))

    stats.trees_pruned = len(forest.trees) - len(tree_scores)

    # Sort by score descending, keep top-B trees
    tree_scores.sort(key=lambda x: x[0], reverse=True)
    selected_trees = tree_scores[:B]
    stats.trees_scanned = len(selected_trees)

    # If no trees pass the threshold, fall back to top-B without threshold
    if not selected_trees and forest.trees:
        all_scores = []
        for i, tree in enumerate(forest.trees):
            score = _sim(query_vector, tree.root.vector)
            all_scores.append((score, i))
        all_scores.sort(key=lambda x: x[0], reverse=True)
        selected_trees = all_scores[:B]
        stats.trees_scanned = len(selected_trees)
        stats.trees_pruned = 0

    # ── Step 2: Top-down traversal of each selected tree ─────────────────────
    for root_score, tree_idx in selected_trees:
        tree = forest.trees[tree_idx]

        # BFS queue: (node, path_so_far)
        queue: list[tuple[TreeNode, list[str]]] = [
            (tree.root, [f"root:{tree.tree_id}"])
        ]
        max_level = 0

        while queue:
            node, path = queue.pop(0)

            if node.is_leaf():
                # Reached a leaf — score and collect
                leaf_score = _sim(query_vector, node.vector)
                stats.total_similarity_comparisons += 1
                stats.leaves_reached += 1
                all_candidates.append((leaf_score, node, path))
            else:
                # Score all children
                child_scores: list[tuple[float, TreeNode]] = []
                for child in node.children:
                    score = _sim(query_vector, child.vector)
                    stats.total_similarity_comparisons += 1
                    child_scores.append((score, child))

                # Sort descending, keep top-B above threshold
                child_scores.sort(key=lambda x: x[0], reverse=True)
                selected_children = [
                    (s, c) for s, c in child_scores[:B] if s >= tau
                ]

                # If no child passes threshold, take the best one anyway
                if not selected_children and child_scores:
                    selected_children = [child_scores[0]]

                for score, child in selected_children:
                    child_path = path + [f"{child.level}:{child.node_id}"]
                    queue.append((child, child_path))

                max_level += 1

        stats.levels_traversed = max(stats.levels_traversed, max_level)

    # ── Step 3: Rank and return top-k ────────────────────────────────────────
    all_candidates.sort(key=lambda x: x[0], reverse=True)
    top_k = all_candidates[:k]

    results = []
    for score, node, path in top_k:
        results.append(RetrievalResult(
            text=node.text or "",
            score=score,
            metadata=node.metadata,
            traversal_path=path,
        ))

    logger.debug(
        "Hierarchical retrieval: %d comparisons, %d results",
        stats.total_similarity_comparisons,
        len(results),
    )
    return results, stats


def retrieve_flat(
    forest: TopicForest,
    query_vector: np.ndarray,
    k: int = TOP_K,
) -> tuple[list[RetrievalResult], RetrievalStats]:
    """Flat retrieval baseline: cosine similarity over all leaf chunks.

    This is the standard RAG approach for comparison.

    Args:
        forest: The TopicForest (uses its flat index).
        query_vector: Query embedding, shape (d,).
        k: Number of results to return.

    Returns:
        results: Top-k RetrievalResult objects.
        stats: RetrievalStats with comparison count = N (total chunks).
    """
    stats = RetrievalStats()

    if forest._flat_vectors is None or len(forest._flat_vectors) == 0:
        return [], stats

    # Compute similarity to all leaf vectors at once
    sims = cosine_similarity(
        query_vector.reshape(1, -1), forest._flat_vectors
    ).flatten()
    stats.total_similarity_comparisons = len(sims)
    stats.leaves_reached = len(sims)

    # Get top-k indices
    top_indices = np.argsort(sims)[::-1][:k]

    results = []
    for idx in top_indices:
        results.append(RetrievalResult(
            text=forest._flat_texts[idx],
            score=float(sims[idx]),
            metadata=forest._flat_metadata[idx],
            traversal_path=["flat"],
        ))

    return results, stats
