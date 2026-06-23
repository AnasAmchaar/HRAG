"""Tests for topic tree data structures."""

import tempfile
from pathlib import Path

import numpy as np

from hrag.tree import TreeNode, TopicTree, TopicForest


class TestTreeNode:
    def test_leaf_node(self):
        node = TreeNode(vector=np.array([1.0, 0.0]), level="leaf", text="hello")
        assert node.is_leaf()
        assert node.depth() == 0
        assert node.leaf_count() == 1
        assert node.node_count() == 1

    def test_parent_with_children(self):
        child1 = TreeNode(vector=np.array([1.0, 0.0]), level="leaf", text="a")
        child2 = TreeNode(vector=np.array([0.0, 1.0]), level="leaf", text="b")
        parent = TreeNode(
            vector=np.array([0.5, 0.5]),
            level="subtopic",
            children=[child1, child2],
        )
        assert not parent.is_leaf()
        assert parent.depth() == 1
        assert parent.leaf_count() == 2
        assert parent.node_count() == 3

    def test_deep_tree(self):
        leaf = TreeNode(vector=np.zeros(4), level="leaf", text="x")
        sub = TreeNode(vector=np.zeros(4), level="subtopic", children=[leaf])
        topic = TreeNode(vector=np.zeros(4), level="topic", children=[sub])
        root = TreeNode(vector=np.zeros(4), level="root", children=[topic])

        assert root.depth() == 3
        assert root.leaf_count() == 1
        assert root.node_count() == 4

    def test_all_leaves(self):
        leaf1 = TreeNode(vector=np.zeros(2), level="leaf", text="a")
        leaf2 = TreeNode(vector=np.zeros(2), level="leaf", text="b")
        leaf3 = TreeNode(vector=np.zeros(2), level="leaf", text="c")
        sub1 = TreeNode(vector=np.zeros(2), level="subtopic", children=[leaf1, leaf2])
        sub2 = TreeNode(vector=np.zeros(2), level="subtopic", children=[leaf3])
        root = TreeNode(vector=np.zeros(2), level="root", children=[sub1, sub2])

        leaves = root.all_leaves()
        assert len(leaves) == 3
        texts = {leaf.text for leaf in leaves}
        assert texts == {"a", "b", "c"}

    def test_serialization_roundtrip(self):
        leaf = TreeNode(
            vector=np.array([1.0, 2.0, 3.0]),
            level="leaf",
            text="test passage",
            metadata={"doc_id": "doc1", "page_number": 5},
        )
        parent = TreeNode(
            vector=np.array([0.5, 1.0, 1.5]),
            level="root",
            children=[leaf],
        )

        d = parent.to_dict()
        restored = TreeNode.from_dict(d)

        assert restored.level == "root"
        assert len(restored.children) == 1
        assert restored.children[0].level == "leaf"
        assert restored.children[0].text == "test passage"
        assert restored.children[0].metadata["doc_id"] == "doc1"
        np.testing.assert_allclose(restored.vector, parent.vector)
        np.testing.assert_allclose(
            restored.children[0].vector, leaf.vector
        )


class TestTopicTree:
    def test_tree_properties(self):
        leaf1 = TreeNode(vector=np.zeros(4), level="leaf", text="a")
        leaf2 = TreeNode(vector=np.zeros(4), level="leaf", text="b")
        root = TreeNode(vector=np.zeros(4), level="root", children=[leaf1, leaf2])
        tree = TopicTree(root=root)

        assert tree.depth == 1
        assert tree.leaf_count == 2
        assert tree.node_count == 3


class TestTopicForest:
    def test_forest_properties(self):
        # Build two small trees
        t1_leaf = TreeNode(vector=np.zeros(4), level="leaf", text="a")
        t1_root = TreeNode(vector=np.zeros(4), level="root", children=[t1_leaf])
        tree1 = TopicTree(root=t1_root)

        t2_leaf1 = TreeNode(vector=np.zeros(4), level="leaf", text="b")
        t2_leaf2 = TreeNode(vector=np.zeros(4), level="leaf", text="c")
        t2_root = TreeNode(
            vector=np.zeros(4), level="root", children=[t2_leaf1, t2_leaf2]
        )
        tree2 = TopicTree(root=t2_root)

        forest = TopicForest(trees=[tree1, tree2])

        assert forest.tree_count == 2
        assert forest.total_leaves == 3
        assert forest.total_nodes == 5

    def test_save_load_roundtrip(self):
        """Test that save/load preserves the forest structure."""
        leaf1 = TreeNode(
            vector=np.array([1.0, 2.0]),
            level="leaf",
            text="passage 1",
            metadata={"doc_id": "d1"},
        )
        leaf2 = TreeNode(
            vector=np.array([3.0, 4.0]),
            level="leaf",
            text="passage 2",
            metadata={"doc_id": "d1"},
        )
        root = TreeNode(
            vector=np.array([2.0, 3.0]),
            level="root",
            children=[leaf1, leaf2],
        )
        tree = TopicTree(root=root)
        forest = TopicForest(trees=[tree])

        # Save and reload
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = Path(f.name)

        try:
            forest.save(tmp_path)
            loaded = TopicForest.load(tmp_path)

            assert loaded.tree_count == 1
            assert loaded.total_leaves == 2
            assert loaded.trees[0].root.level == "root"
            assert loaded.trees[0].root.children[0].text == "passage 1"
            np.testing.assert_allclose(
                loaded.trees[0].root.vector,
                np.array([2.0, 3.0]),
            )
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_root_vectors(self):
        t1_root = TreeNode(vector=np.array([1.0, 0.0]), level="root")
        t2_root = TreeNode(vector=np.array([0.0, 1.0]), level="root")
        forest = TopicForest(trees=[
            TopicTree(root=t1_root),
            TopicTree(root=t2_root),
        ])

        root_vecs = forest.root_vectors()
        assert root_vecs.shape == (2, 2)
        np.testing.assert_allclose(root_vecs[0], [1.0, 0.0])
        np.testing.assert_allclose(root_vecs[1], [0.0, 1.0])
