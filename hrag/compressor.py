"""
Semantic Vector Compression (Section IV.B, Equations 7–8).

Implements saliency-weighted mean pooling to compress a cluster of
semantically related chunk embeddings into a single concept vector.

Algorithm:
  1. Compute unweighted centroid:  v̄ = (1/k) Σ vᵢ
  2. Compute saliency weights:    αᵢ = sim(vᵢ, v̄) = (vᵢ · v̄) / (‖vᵢ‖ ‖v̄‖)
  3. Concept vector:              v_C = Σ αᵢvᵢ / Σ αᵢ

The concept vector is then L2-normalized to stay consistent with
the embedding space used by the encoder.
"""

from __future__ import annotations

import logging

import hdbscan
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from hrag.config import compute_min_cluster_size

logger = logging.getLogger(__name__)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def compute_concept_vector(vectors: np.ndarray) -> np.ndarray:
    """Compute a saliency-weighted concept vector from a cluster (Eq. 7–8).

    Args:
        vectors: np.ndarray of shape (k, d) — the cluster member vectors.

    Returns:
        np.ndarray of shape (d,) — the compressed concept vector, L2-normalized.
    """
    if len(vectors) == 1:
        v = vectors[0].copy()
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    # Step 1: Unweighted centroid (Eq. 5)
    centroid = np.mean(vectors, axis=0)

    # Step 2: Saliency weights — cosine similarity to centroid (Eq. 8)
    # Reshape centroid for sklearn cosine_similarity
    sims = cosine_similarity(vectors, centroid.reshape(1, -1)).flatten()

    # Clamp negative similarities to a small positive value
    # (peripheral outliers should have low but non-zero weight)
    weights = np.maximum(sims, 0.01)

    # Step 3: Saliency-weighted mean pooling (Eq. 7)
    weighted_sum = np.sum(weights[:, np.newaxis] * vectors, axis=0)
    concept_vector = weighted_sum / np.sum(weights)

    # L2-normalize to stay in the same embedding space
    norm = np.linalg.norm(concept_vector)
    if norm > 0:
        concept_vector = concept_vector / norm

    return concept_vector


def cluster_vectors(
    vectors: np.ndarray,
    min_cluster_size: int | None = None,
) -> tuple[list[list[int]], list[int]]:
    """Cluster vectors using HDBSCAN (Section V.C).

    Args:
        vectors: np.ndarray of shape (n, d).
        min_cluster_size: Minimum cluster size. If None, computed as
                          max(3, n // 20) per the paper.

    Returns:
        clusters: List of lists, where each inner list contains the
                  indices of vectors belonging to that cluster.
        noise_indices: List of indices that HDBSCAN labeled as noise (-1).
    """
    n = len(vectors)

    if n <= 3:
        # Too few vectors to cluster — treat all as one cluster
        return [list(range(n))], []

    if min_cluster_size is None:
        min_cluster_size = compute_min_cluster_size(n)

    # Ensure min_cluster_size doesn't exceed n
    min_cluster_size = min(min_cluster_size, n)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",  # Works well with L2-normalized embeddings
        cluster_selection_method="eom",  # Excess of Mass — default
    )
    labels = clusterer.fit_predict(vectors)

    # Organize indices by cluster label
    cluster_map: dict[int, list[int]] = {}
    noise_indices: list[int] = []

    for idx, label in enumerate(labels):
        if label == -1:
            noise_indices.append(idx)
        else:
            cluster_map.setdefault(label, []).append(idx)

    clusters = list(cluster_map.values())

    # If HDBSCAN produced no clusters (everything is noise),
    # fall back to treating all vectors as one cluster
    if not clusters:
        logger.debug("HDBSCAN produced no clusters; treating all as one cluster")
        return [list(range(n))], []

    logger.debug(
        "Clustered %d vectors into %d clusters (%d noise points)",
        n,
        len(clusters),
        len(noise_indices),
    )
    return clusters, noise_indices


def assign_noise_to_nearest_cluster(
    vectors: np.ndarray,
    clusters: list[list[int]],
    noise_indices: list[int],
) -> list[list[int]]:
    """Assign noise points to their nearest cluster centroid (Section V.C).

    Args:
        vectors: Full array of vectors, shape (n, d).
        clusters: Current cluster assignments (lists of indices).
        noise_indices: Indices labeled as noise by HDBSCAN.

    Returns:
        Updated clusters with noise points reassigned.
    """
    if not noise_indices or not clusters:
        return clusters

    # Compute cluster centroids
    centroids = np.array([
        np.mean(vectors[indices], axis=0) for indices in clusters
    ])

    # For each noise point, find the nearest centroid
    noise_vectors = vectors[noise_indices]
    sims = cosine_similarity(noise_vectors, centroids)
    nearest_cluster_ids = np.argmax(sims, axis=1)

    for noise_idx, cluster_id in zip(noise_indices, nearest_cluster_ids):
        clusters[cluster_id].append(noise_idx)

    return clusters


def cluster_and_compress(
    vectors: np.ndarray,
    min_cluster_size: int | None = None,
) -> tuple[np.ndarray, list[list[int]]]:
    """Cluster vectors and compress each cluster into a concept vector.

    This is the main entry point for the compression step at each
    level of the hierarchy.

    Args:
        vectors: np.ndarray of shape (n, d).
        min_cluster_size: Optional override for min_cluster_size.

    Returns:
        concept_vectors: np.ndarray of shape (m, d), where m is the
                         number of clusters.
        cluster_assignments: List of lists mapping each concept vector
                            index to its member vector indices.
    """
    clusters, noise_indices = cluster_vectors(vectors, min_cluster_size)

    # Reassign noise points
    clusters = assign_noise_to_nearest_cluster(vectors, clusters, noise_indices)

    # Compress each cluster into a concept vector
    concept_vectors = []
    for member_indices in clusters:
        cluster_vectors_subset = vectors[member_indices]
        concept_vec = compute_concept_vector(cluster_vectors_subset)
        concept_vectors.append(concept_vec)

    return np.array(concept_vectors, dtype=np.float32), clusters
