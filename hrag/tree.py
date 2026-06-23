"""
Topic-Tree Data Structures (Section IV.C).

Defines the TreeNode, TopicTree, and TopicForest classes that represent
the hierarchical vector index.

Each node carries:
  (1) its embedding vector (same Rd space at all levels)
  (2) its level label: root | topic | subtopic | leaf
  (3) parent/children pointers
  (4) for leaf nodes: original text and source metadata
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)

LevelType = Literal["root", "topic", "subtopic", "leaf"]


@dataclass
class TreeNode:
    """A single node in the topic tree."""

    node_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    vector: np.ndarray = field(default_factory=lambda: np.zeros(0))
    level: LevelType = "leaf"
    children: list[TreeNode] = field(default_factory=list)
    parent: Optional[TreeNode] = field(default=None, repr=False)

    # Leaf-node specific fields
    text: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def depth(self) -> int:
        """Compute the depth of the subtree rooted at this node."""
        if self.is_leaf():
            return 0
        return 1 + max(child.depth() for child in self.children)

    def leaf_count(self) -> int:
        """Count the number of leaf nodes in the subtree."""
        if self.is_leaf():
            return 1
        return sum(child.leaf_count() for child in self.children)

    def node_count(self) -> int:
        """Count total nodes in the subtree (including self)."""
        return 1 + sum(child.node_count() for child in self.children)

    def all_leaves(self) -> list[TreeNode]:
        """Collect all leaf nodes in the subtree."""
        if self.is_leaf():
            return [self]
        leaves = []
        for child in self.children:
            leaves.extend(child.all_leaves())
        return leaves

    def to_dict(self) -> dict:
        """Serialize to a dictionary (for JSON storage)."""
        d = {
            "node_id": self.node_id,
            "vector": self.vector.tolist() if self.vector.size > 0 else [],
            "level": self.level,
            "text": self.text,
            "metadata": self.metadata,
            "children": [child.to_dict() for child in self.children],
        }
        return d

    @classmethod
    def from_dict(cls, d: dict, parent: Optional[TreeNode] = None) -> TreeNode:
        """Deserialize from a dictionary."""
        node = cls(
            node_id=d["node_id"],
            vector=(
                np.array(d["vector"], dtype=np.float32)
                if d["vector"]
                else np.zeros(0)
            ),
            level=d["level"],
            text=d.get("text"),
            metadata=d.get("metadata", {}),
            parent=parent,
        )
        node.children = [
            cls.from_dict(child_d, parent=node)
            for child_d in d.get("children", [])
        ]
        return node

    def __repr__(self) -> str:
        leaf_info = f', text="{self.text[:40]}..."' if self.text else ""
        return (
            f"TreeNode(id={self.node_id}, level={self.level}, "
            f"children={len(self.children)}{leaf_info})"
        )


@dataclass
class TopicTree:
    """A single topic tree with a root node."""

    root: TreeNode
    tree_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    @property
    def depth(self) -> int:
        return self.root.depth()

    @property
    def leaf_count(self) -> int:
        return self.root.leaf_count()

    @property
    def node_count(self) -> int:
        return self.root.node_count()

    def all_leaves(self) -> list[TreeNode]:
        return self.root.all_leaves()

    def to_dict(self) -> dict:
        return {
            "tree_id": self.tree_id,
            "root": self.root.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> TopicTree:
        root = TreeNode.from_dict(d["root"])
        return cls(root=root, tree_id=d["tree_id"])

    def summary(self) -> str:
        """Return a human-readable summary of the tree."""
        return (
            f"TopicTree(id={self.tree_id}, depth={self.depth}, "
            f"nodes={self.node_count}, leaves={self.leaf_count})"
        )


@dataclass
class TopicForest:
    """A forest of topic trees — the complete hierarchical index."""

    trees: list[TopicTree] = field(default_factory=list)

    # Also store all leaf vectors for flat-retrieval baseline comparison
    _flat_vectors: Optional[np.ndarray] = field(default=None, repr=False)
    _flat_texts: Optional[list[str]] = field(default=None, repr=False)
    _flat_metadata: Optional[list[dict]] = field(default=None, repr=False)

    @property
    def tree_count(self) -> int:
        return len(self.trees)

    @property
    def total_leaves(self) -> int:
        return sum(t.leaf_count for t in self.trees)

    @property
    def total_nodes(self) -> int:
        return sum(t.node_count for t in self.trees)

    def root_vectors(self) -> np.ndarray:
        """Return the root vectors of all trees as an array."""
        return np.array([t.root.vector for t in self.trees], dtype=np.float32)

    def save(self, filepath: Path) -> None:
        """Serialize the forest to a JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "trees": [t.to_dict() for t in self.trees],
        }

        # Store flat index data if available
        if self._flat_vectors is not None:
            data["flat_vectors"] = self._flat_vectors.tolist()
            data["flat_texts"] = self._flat_texts
            data["flat_metadata"] = self._flat_metadata

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        logger.info("Forest saved to %s", filepath)

    @classmethod
    def load(cls, filepath: Path) -> TopicForest:
        """Deserialize a forest from a JSON file."""
        filepath = Path(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        forest = cls(
            trees=[TopicTree.from_dict(td) for td in data["trees"]],
        )

        if "flat_vectors" in data:
            forest._flat_vectors = np.array(data["flat_vectors"], dtype=np.float32)
            forest._flat_texts = data["flat_texts"]
            forest._flat_metadata = data["flat_metadata"]

        logger.info("Forest loaded from %s", filepath)
        return forest

    def summary(self) -> str:
        """Return a human-readable summary of the forest."""
        lines = [
            f"TopicForest: {self.tree_count} trees, "
            f"{self.total_nodes} total nodes, "
            f"{self.total_leaves} leaf chunks",
            "",
        ]
        for i, tree in enumerate(self.trees):
            lines.append(f"  Tree {i}: {tree.summary()}")
        return "\n".join(lines)
