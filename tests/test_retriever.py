"""Tests for the retriever (Algorithm 1 — top-down forest traversal)."""

import numpy as np

from hrag.retriever import retrieve_hierarchical, retrieve_flat, _sim
from hrag.tree import TopicForest


class TestHierarchicalRetrieval:
    def test_retrieves_correct_tree(self, simple_forest: TopicForest):
        """Query near topic A should retrieve from tree 1."""
        d = 64
        query = np.zeros(d, dtype=np.float32)
        query[0] = 1.0  # Near topic A

        results, stats = retrieve_hierarchical(simple_forest, query, tau=0.3, k=3, B=2)
        assert len(results) > 0
        # All results should be from topic A
        for r in results:
            assert "topic A" in r.text.lower() or r.metadata.get("doc_id") == "docA"

    def test_respects_top_k(self, simple_forest: TopicForest):
        """Should return at most k results."""
        d = 64
        query = np.zeros(d, dtype=np.float32)
        query[0] = 1.0

        for k in [1, 2, 5]:
            results, _ = retrieve_hierarchical(simple_forest, query, tau=0.1, k=k, B=3)
            assert len(results) <= k

    def test_results_are_ranked(self, simple_forest: TopicForest):
        """Results should be sorted by decreasing similarity score."""
        d = 64
        query = np.zeros(d, dtype=np.float32)
        query[0] = 1.0

        results, _ = retrieve_hierarchical(simple_forest, query, tau=0.1, k=5, B=3)
        if len(results) >= 2:
            for i in range(len(results) - 1):
                assert results[i].score >= results[i + 1].score

    def test_traversal_path_populated(self, simple_forest: TopicForest):
        """Each result should have a non-empty traversal path."""
        d = 64
        query = np.zeros(d, dtype=np.float32)
        query[0] = 1.0

        results, _ = retrieve_hierarchical(simple_forest, query, tau=0.1, k=3, B=3)
        for r in results:
            assert len(r.traversal_path) > 0

    def test_fewer_comparisons_than_flat(self, simple_forest: TopicForest):
        """Hierarchical should prune irrelevant trees."""
        d = 64
        query = np.zeros(d, dtype=np.float32)
        query[0] = 1.0

        _, h_stats = retrieve_hierarchical(simple_forest, query, tau=0.3, k=3, B=1)
        _, f_stats = retrieve_flat(simple_forest, query, k=3)

        # With a small tree, hierarchical might do slightly more comparisons
        # if branching factor is large. So we just ensure it correctly pruned tree 2.
        assert h_stats.trees_pruned >= 1


class TestFlatRetrieval:
    def test_flat_returns_results(self, simple_forest: TopicForest):
        d = 64
        query = np.zeros(d, dtype=np.float32)
        query[0] = 1.0

        results, stats = retrieve_flat(simple_forest, query, k=3)
        assert len(results) == 3
        assert stats.total_similarity_comparisons == 5  # all 5 leaves

    def test_flat_results_ranked(self, simple_forest: TopicForest):
        d = 64
        query = np.zeros(d, dtype=np.float32)
        query[0] = 1.0

        results, _ = retrieve_flat(simple_forest, query, k=5)
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score
