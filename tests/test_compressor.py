"""Tests for semantic vector compression."""

import numpy as np

from hrag.compressor import (
    compute_concept_vector,
    cosine_sim,
    cluster_vectors,
    assign_noise_to_nearest_cluster,
    cluster_and_compress,
)


class TestCosineSim:
    def test_identical_vectors(self):
        v = np.array([1.0, 0.0, 0.0])
        assert abs(cosine_sim(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        assert abs(cosine_sim(v1, v2)) < 1e-6

    def test_opposite_vectors(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([-1.0, 0.0])
        assert abs(cosine_sim(v1, v2) + 1.0) < 1e-6


class TestConceptVector:
    def test_single_vector(self):
        """Single vector should be returned as-is (normalized)."""
        v = np.array([[3.0, 4.0]])
        result = compute_concept_vector(v)
        expected = np.array([0.6, 0.8])  # normalized
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_identical_vectors(self):
        """Identical vectors should produce the same vector (normalized)."""
        v = np.array([1.0, 0.0, 0.0])
        vectors = np.stack([v, v, v])
        result = compute_concept_vector(vectors)
        np.testing.assert_allclose(result, v, atol=1e-5)

    def test_output_dimension(self):
        """Output should have the same dimension as input."""
        vectors = np.random.randn(10, 768).astype(np.float32)
        result = compute_concept_vector(vectors)
        assert result.shape == (768,)

    def test_output_normalized(self):
        """Output should be L2-normalized."""
        vectors = np.random.randn(5, 128).astype(np.float32)
        result = compute_concept_vector(vectors)
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 1e-5

    def test_concept_near_centroid(self):
        """Concept vector should be closer to the centroid than random vectors."""
        np.random.seed(42)
        # Create a tight cluster
        center = np.random.randn(128).astype(np.float32)
        center /= np.linalg.norm(center)
        noise = np.random.randn(10, 128).astype(np.float32) * 0.1
        vectors = center + noise
        # Normalize
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        concept = compute_concept_vector(vectors)
        centroid = np.mean(vectors, axis=0)
        centroid /= np.linalg.norm(centroid)

        sim_to_centroid = cosine_sim(concept, centroid)

        # Random vector should be far from centroid
        random_vec = np.random.randn(128).astype(np.float32)
        random_vec /= np.linalg.norm(random_vec)
        sim_random = cosine_sim(random_vec, centroid)

        assert sim_to_centroid > sim_random


class TestClustering:
    def test_too_few_vectors(self):
        """3 or fewer vectors should produce one cluster."""
        vectors = np.random.randn(3, 64).astype(np.float32)
        clusters, noise = cluster_vectors(vectors)
        assert len(clusters) == 1
        assert len(clusters[0]) == 3

    def test_distinct_clusters(self):
        """Well-separated clusters should be detected."""
        np.random.seed(42)
        # Two distinct clusters in 64d
        cluster_a = np.random.randn(20, 64).astype(np.float32) + 5.0
        cluster_b = np.random.randn(20, 64).astype(np.float32) - 5.0
        vectors = np.vstack([cluster_a, cluster_b])

        clusters, noise = cluster_vectors(vectors, min_cluster_size=5)
        # Should find at least 2 clusters
        assert len(clusters) >= 2


class TestClusterAndCompress:
    def test_compression_reduces_count(self):
        """Compression should produce fewer vectors than input."""
        np.random.seed(42)
        # Create well-separated clusters
        cluster_a = np.random.randn(15, 64).astype(np.float32) + 3.0
        cluster_b = np.random.randn(15, 64).astype(np.float32) - 3.0
        vectors = np.vstack([cluster_a, cluster_b])

        concept_vectors, assignments = cluster_and_compress(vectors, min_cluster_size=5)
        assert len(concept_vectors) < len(vectors)
        assert len(concept_vectors) == len(assignments)

    def test_all_indices_covered(self):
        """All original indices should appear in some cluster."""
        np.random.seed(42)
        n = 30
        vectors = np.random.randn(n, 64).astype(np.float32)
        concept_vectors, assignments = cluster_and_compress(vectors)

        all_indices = set()
        for cluster in assignments:
            all_indices.update(cluster)
        assert all_indices == set(range(n))

    def test_concept_vectors_normalized(self):
        """All concept vectors should be L2-normalized."""
        vectors = np.random.randn(20, 64).astype(np.float32)
        concept_vectors, _ = cluster_and_compress(vectors)

        for cv in concept_vectors:
            norm = np.linalg.norm(cv)
            assert abs(norm - 1.0) < 1e-4, f"Norm = {norm}"
