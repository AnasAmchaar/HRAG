# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-06-23

### Added

- **Core pipeline**: `HumanizedRAGPipeline` with `ingest`, `query`, `compare`, and `stats` methods
- **Hierarchical indexing**: HDBSCAN-based clustering with saliency-weighted vector compression
- **Topic-tree forest**: Multi-level tree structure (Root → Topic → Subtopic → Leaf)
- **Top-down retrieval**: Algorithm 1 implementation with threshold pruning and branching factor control
- **Flat retrieval baseline**: For benchmarking against standard cosine similarity search
- **Document chunking**: Sentence-boundary-aware sliding window with configurable overlap
- **LLM generation**: Ollama integration with extractive fallback
- **Multi-format support**: PDF, TXT, and Markdown ingestion
- **CLI**: `hrag` command with `ingest`, `query`, `compare`, and `stats` subcommands
- **Configuration**: Environment variable overrides for all hyperparameters
- **Test suite**: Unit tests for chunker, compressor, tree, and retriever modules

[0.1.0]: https://github.com/AnasAmchaar/HRAG/releases/tag/v0.1.0
