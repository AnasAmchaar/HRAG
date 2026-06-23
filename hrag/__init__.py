"""
Humanized-RAG (H-RAG): Hierarchical Vector Compression and Topic-Guided Retrieval.

A novel RAG architecture that organizes document chunks into a hierarchical
topic-tree forest, enabling efficient top-down retrieval that mirrors how
humans navigate knowledge — from broad topics to specific details.

Example usage::

    from hrag import HumanizedRAGPipeline

    pipeline = HumanizedRAGPipeline()
    pipeline.ingest(source="./docs/")
    result = pipeline.query("What is semantic vector compression?")
    print(result["answer"])
"""

__version__ = "0.1.0"

from hrag.pipeline import HumanizedRAGPipeline
from hrag.tree import TopicForest, TopicTree, TreeNode
from hrag.embedder import Embedder
from hrag.retriever import RetrievalResult, RetrievalStats
from hrag.chunker import Chunk

__all__ = [
    "__version__",
    "HumanizedRAGPipeline",
    "TopicForest",
    "TopicTree",
    "TreeNode",
    "Embedder",
    "RetrievalResult",
    "RetrievalStats",
    "Chunk",
]
