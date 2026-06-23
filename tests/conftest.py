"""Shared test fixtures and configuration for pytest."""

from __future__ import annotations

import numpy as np
import pytest

from hrag.tree import TopicForest, TopicTree, TreeNode


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def simple_forest() -> TopicForest:
    """Create a small test forest with known structure.

    Tree 1 (topic A — vectors near [1, 0, 0, ...]):
      root → [subtopic1, subtopic2]
      subtopic1 → [leaf_a1, leaf_a2]
      subtopic2 → [leaf_a3]

    Tree 2 (topic B — vectors near [0, 1, 0, ...]):
      root → [leaf_b1, leaf_b2]
    """
    d = 64
    np.random.seed(42)

    # Tree 1: vectors clustered around [1, 0, ...]
    base_a = np.zeros(d, dtype=np.float32)
    base_a[0] = 1.0

    leaf_a1 = TreeNode(
        vector=base_a + np.random.randn(d).astype(np.float32) * 0.05,
        level="leaf",
        text="Leaf A1: about topic A detail 1",
        metadata={"doc_id": "docA", "page_number": 1},
    )
    leaf_a1.vector /= np.linalg.norm(leaf_a1.vector)

    leaf_a2 = TreeNode(
        vector=base_a + np.random.randn(d).astype(np.float32) * 0.05,
        level="leaf",
        text="Leaf A2: about topic A detail 2",
        metadata={"doc_id": "docA", "page_number": 2},
    )
    leaf_a2.vector /= np.linalg.norm(leaf_a2.vector)

    leaf_a3 = TreeNode(
        vector=base_a + np.random.randn(d).astype(np.float32) * 0.05,
        level="leaf",
        text="Leaf A3: about topic A detail 3",
        metadata={"doc_id": "docA", "page_number": 3},
    )
    leaf_a3.vector /= np.linalg.norm(leaf_a3.vector)

    sub1 = TreeNode(
        vector=np.mean([leaf_a1.vector, leaf_a2.vector], axis=0),
        level="subtopic",
        children=[leaf_a1, leaf_a2],
    )
    sub1.vector /= np.linalg.norm(sub1.vector)

    sub2 = TreeNode(
        vector=leaf_a3.vector.copy(),
        level="subtopic",
        children=[leaf_a3],
    )

    root1 = TreeNode(
        vector=np.mean([sub1.vector, sub2.vector], axis=0),
        level="root",
        children=[sub1, sub2],
    )
    root1.vector /= np.linalg.norm(root1.vector)
    tree1 = TopicTree(root=root1)

    # Tree 2: vectors clustered around [0, 1, ...]
    base_b = np.zeros(d, dtype=np.float32)
    base_b[1] = 1.0

    leaf_b1 = TreeNode(
        vector=base_b + np.random.randn(d).astype(np.float32) * 0.05,
        level="leaf",
        text="Leaf B1: about topic B detail 1",
        metadata={"doc_id": "docB", "page_number": 1},
    )
    leaf_b1.vector /= np.linalg.norm(leaf_b1.vector)

    leaf_b2 = TreeNode(
        vector=base_b + np.random.randn(d).astype(np.float32) * 0.05,
        level="leaf",
        text="Leaf B2: about topic B detail 2",
        metadata={"doc_id": "docB", "page_number": 2},
    )
    leaf_b2.vector /= np.linalg.norm(leaf_b2.vector)

    root2 = TreeNode(
        vector=np.mean([leaf_b1.vector, leaf_b2.vector], axis=0),
        level="root",
        children=[leaf_b1, leaf_b2],
    )
    root2.vector /= np.linalg.norm(root2.vector)
    tree2 = TopicTree(root=root2)

    # Build forest with flat index
    forest = TopicForest(trees=[tree1, tree2])
    all_leaves = []
    all_texts = []
    all_meta = []
    for tree in forest.trees:
        for leaf in tree.all_leaves():
            all_leaves.append(leaf.vector)
            all_texts.append(leaf.text or "")
            all_meta.append(leaf.metadata)
    forest._flat_vectors = np.array(all_leaves, dtype=np.float32)
    forest._flat_texts = all_texts
    forest._flat_metadata = all_meta

    return forest
