"""
Hierarchical Tree Construction (Algorithm 2, Section V.D).

Builds the topic-tree forest from document chunk embeddings.

Algorithm 2 — Hierarchical Tree Construction:
  1. Start with leaf embeddings L = {v₁, ..., vₙ}
  2. While |level| > 1 and depth < D_max:
     a. Run HDBSCAN on current level vectors
     b. For each cluster: compute saliency-weighted concept vector
     c. Store parent → child links
     d. level = parents; depth += 1
  3. Final aggregation produces root node

Forest construction (Section IV.C.3):
  - After building per-document trees, cluster root vectors to detect
    domain boundaries.
  - Merge documents with similar roots into shared trees.
"""

from __future__ import annotations

import logging

import numpy as np

from hrag.config import MAX_TREE_DEPTH, compute_min_cluster_size
from hrag.chunker import Chunk
from hrag.compressor import cluster_and_compress, compute_concept_vector
from hrag.tree import LevelType, TopicForest, TopicTree, TreeNode

logger = logging.getLogger(__name__)

# Level labels assigned bottom-up during construction
_LEVEL_LABELS: list[LevelType] = ["leaf", "subtopic", "topic", "root"]


def _assign_level_label(depth_from_leaf: int, total_depth: int) -> LevelType:
    """Assign a level label based on position in the tree.

    - depth 0 from leaf = "leaf"
    - top = "root"
    - second from top = "topic"
    - everything else = "subtopic"
    """
    if depth_from_leaf == 0:
        return "leaf"
    if depth_from_leaf == total_depth:
        return "root"
    if depth_from_leaf == total_depth - 1:
        return "topic"
    return "subtopic"


def build_tree_from_chunks(
    chunks: list[Chunk],
    vectors: np.ndarray,
    max_depth: int = MAX_TREE_DEPTH,
) -> TopicTree:
    """Build a single topic tree from a set of chunks and their vectors.

    Implements Algorithm 2 from the paper.

    Args:
        chunks: List of Chunk objects (for leaf node text/metadata).
        vectors: np.ndarray of shape (n, d), the chunk embeddings.
        max_depth: Maximum tree depth (default 4).

    Returns:
        A TopicTree with root, topic, subtopic, and leaf nodes.
    """
    n = len(chunks)
    assert len(vectors) == n, f"Mismatch: {n} chunks but {len(vectors)} vectors"

    # Step 1: Create leaf nodes
    leaf_nodes: list[TreeNode] = []
    for i, chunk in enumerate(chunks):
        node = TreeNode(
            vector=vectors[i],
            level="leaf",
            text=chunk.text,
            metadata={
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number,
                "char_offset": chunk.char_offset,
            },
        )
        leaf_nodes.append(node)

    # Step 2: Iteratively cluster and compress upward
    current_level_nodes = leaf_nodes
    depth = 0

    while len(current_level_nodes) > 1 and depth < max_depth - 1:
        current_vectors = np.array(
            [node.vector for node in current_level_nodes], dtype=np.float32
        )

        # If we have very few nodes, just aggregate into one parent
        if len(current_level_nodes) <= 3:
            parent_vector = compute_concept_vector(current_vectors)
            parent_node = TreeNode(
                vector=parent_vector,
                level="root",  # Will be relabeled later
                children=current_level_nodes,
            )
            for child in current_level_nodes:
                child.parent = parent_node
            current_level_nodes = [parent_node]
            depth += 1
            break

        # Cluster and compress
        concept_vectors, cluster_assignments = cluster_and_compress(current_vectors)

        # If clustering produced only 1 cluster, we're done
        if len(concept_vectors) <= 1:
            parent_vector = compute_concept_vector(current_vectors)
            parent_node = TreeNode(
                vector=parent_vector,
                level="root",
                children=current_level_nodes,
            )
            for child in current_level_nodes:
                child.parent = parent_node
            current_level_nodes = [parent_node]
            depth += 1
            break

        # Create parent nodes for each cluster
        parent_nodes: list[TreeNode] = []
        for cluster_idx, member_indices in enumerate(cluster_assignments):
            children = [current_level_nodes[i] for i in member_indices]
            parent_node = TreeNode(
                vector=concept_vectors[cluster_idx],
                level="subtopic",  # Will be relabeled later
                children=children,
            )
            for child in children:
                child.parent = parent_node
            parent_nodes.append(parent_node)

        current_level_nodes = parent_nodes
        depth += 1

    # Step 3: If we still have multiple nodes at the top, create a root
    if len(current_level_nodes) > 1:
        root_vectors = np.array(
            [node.vector for node in current_level_nodes], dtype=np.float32
        )
        root_vector = compute_concept_vector(root_vectors)
        root_node = TreeNode(
            vector=root_vector,
            level="root",
            children=current_level_nodes,
        )
        for child in current_level_nodes:
            child.parent = root_node
    else:
        root_node = current_level_nodes[0]

    # Step 4: Relabel levels based on final tree depth
    total_depth = root_node.depth()
    _relabel_levels(root_node, total_depth, total_depth)

    return TopicTree(root=root_node)


def _relabel_levels(node: TreeNode, depth_from_root: int, total_depth: int) -> None:
    """Recursively assign level labels based on tree structure."""
    depth_from_leaf = total_depth - (total_depth - depth_from_root)

    if node.is_leaf():
        node.level = "leaf"
    elif depth_from_root == 0:
        node.level = "root"
    elif depth_from_root == 1 and total_depth >= 2:
        node.level = "topic"
    else:
        node.level = "subtopic"

    for i, child in enumerate(node.children):
        _relabel_levels(child, depth_from_root - 1, total_depth)


def build_forest(
    chunks: list[Chunk],
    vectors: np.ndarray,
    max_depth: int = MAX_TREE_DEPTH,
) -> TopicForest:
    """Build a topic forest from all chunks across all documents.

    Forest construction (Section IV.C.3):
      1. Group chunks by document
      2. Build a tree per document
      3. Cluster document root vectors to detect domain boundaries
      4. Merge documents with similar roots into shared trees

    Args:
        chunks: All chunks from all documents.
        vectors: All chunk embeddings, shape (n, d).
        max_depth: Maximum tree depth.

    Returns:
        A TopicForest containing the complete hierarchical index.
    """
    # Group chunks by document
    doc_groups: dict[str, tuple[list[Chunk], list[int]]] = {}
    for i, chunk in enumerate(chunks):
        doc_id = chunk.doc_id
        if doc_id not in doc_groups:
            doc_groups[doc_id] = ([], [])
        doc_groups[doc_id][0].append(chunk)
        doc_groups[doc_id][1].append(i)

    logger.info("Building forest from %d documents", len(doc_groups))

    # Build per-document trees
    doc_trees: list[TopicTree] = []
    for doc_id, (doc_chunks, indices) in doc_groups.items():
        doc_vectors = vectors[indices]
        tree = build_tree_from_chunks(doc_chunks, doc_vectors, max_depth)
        doc_trees.append(tree)
        logger.debug("Built tree for '%s': %s", doc_id, tree.summary())

    # If only one document, the forest is just one tree
    if len(doc_trees) <= 1:
        forest = TopicForest(trees=doc_trees)
    else:
        # Cluster document root vectors to detect domain boundaries
        root_vectors = np.array(
            [t.root.vector for t in doc_trees], dtype=np.float32
        )

        if len(doc_trees) <= 3:
            # Too few documents to cluster — each gets its own tree
            forest = TopicForest(trees=doc_trees)
        else:
            concept_vectors, cluster_assignments = cluster_and_compress(root_vectors)

            # Merge documents in the same cluster into a shared tree
            merged_trees: list[TopicTree] = []
            for cluster_idx, member_indices in enumerate(cluster_assignments):
                if len(member_indices) == 1:
                    # Single document cluster — keep its tree as-is
                    merged_trees.append(doc_trees[member_indices[0]])
                else:
                    # Multiple documents — create a new root above their roots
                    children_roots = [doc_trees[i].root for i in member_indices]
                    domain_root = TreeNode(
                        vector=concept_vectors[cluster_idx],
                        level="root",
                        children=children_roots,
                    )
                    for child in children_roots:
                        child.parent = domain_root
                        if child.level == "root":
                            child.level = "topic"
                    merged_tree = TopicTree(root=domain_root)
                    merged_trees.append(merged_tree)

            forest = TopicForest(trees=merged_trees)

    # Store flat index for baseline comparison
    all_leaves = []
    all_leaf_texts = []
    all_leaf_metadata = []
    for tree in forest.trees:
        for leaf in tree.all_leaves():
            all_leaves.append(leaf.vector)
            all_leaf_texts.append(leaf.text or "")
            all_leaf_metadata.append(leaf.metadata)

    forest._flat_vectors = (
        np.array(all_leaves, dtype=np.float32) if all_leaves else None
    )
    forest._flat_texts = all_leaf_texts
    forest._flat_metadata = all_leaf_metadata

    logger.info("Forest built: %s", forest.summary().split("\n")[0])
    return forest
