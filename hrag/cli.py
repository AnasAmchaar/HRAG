"""
Humanized-RAG: CLI Entry Point.

Usage::

    hrag ingest --source ./data/sample/
    hrag ingest --source ./document.pdf
    hrag query "What is semantic vector compression?"
    hrag compare "How does hierarchical retrieval work?"
    hrag stats
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from hrag import __version__
from hrag.pipeline import HumanizedRAGPipeline


def _setup_logging(verbose: bool = False) -> None:
    """Configure logging for the CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest documents and build the hierarchical index."""
    pipeline = HumanizedRAGPipeline()
    pipeline.ingest(source=args.source)
    print("\n[OK] Ingestion complete.")


def cmd_query(args: argparse.Namespace) -> None:
    """Query the index and generate an answer."""
    pipeline = HumanizedRAGPipeline()

    result = pipeline.query(
        question=args.question,
        k=args.top_k,
        use_ollama=not args.no_llm,
    )

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(result["answer"])

    print("\n" + "-" * 70)
    print("RETRIEVED CHUNKS")
    print("-" * 70)
    for i, r in enumerate(result["results"], 1):
        print(f"\n[{i}] Score: {r.score:.4f} | Path: {' -> '.join(r.traversal_path)}")
        print(
            f"    Source: {r.metadata.get('doc_id', '?')}, "
            f"Page: {r.metadata.get('page_number', '?')}"
        )
        preview = r.text[:200] + "..." if len(r.text) > 200 else r.text
        print(f"    {preview}")

    print("\n" + "-" * 70)
    print("STATISTICS")
    print("-" * 70)
    stats = result["stats"]
    latency = result["latency"]
    print(f"  Similarity comparisons: {stats.total_similarity_comparisons}")
    print(f"  Trees scanned: {stats.trees_scanned}")
    print(f"  Trees pruned: {stats.trees_pruned}")
    print(f"  Levels traversed: {stats.levels_traversed}")
    print(f"  Leaves reached: {stats.leaves_reached}")
    print(f"  Embedding latency: {latency['embedding_ms']:.1f} ms")
    print(f"  Retrieval latency: {latency['retrieval_ms']:.1f} ms")
    print(f"  Generation latency: {latency['generation_ms']:.1f} ms")
    print(f"  Total latency: {latency['total_ms']:.1f} ms")


def cmd_compare(args: argparse.Namespace) -> None:
    """Compare hierarchical vs flat retrieval."""
    pipeline = HumanizedRAGPipeline()

    result = pipeline.compare(question=args.question, k=args.top_k)

    print("\n" + "=" * 70)
    print(f'COMPARISON: "{args.question}"')
    print("=" * 70)

    for mode in ["hierarchical", "flat"]:
        data = result[mode]
        stats = data["stats"]
        print(f"\n{'-' * 35}")
        print(f"  {mode.upper()} RETRIEVAL")
        print(f"{'-' * 35}")
        print(f"  Comparisons: {stats.total_similarity_comparisons}")
        print(f"  Leaves reached: {stats.leaves_reached}")
        print(f"  Latency: {data['latency_ms']:.1f} ms")
        print("  Top results:")
        for i, r in enumerate(data["results"][:3], 1):
            preview = r.text[:100] + "..." if len(r.text) > 100 else r.text
            print(f"    [{i}] {r.score:.4f} — {preview}")


def cmd_stats(args: argparse.Namespace) -> None:
    """Print index statistics."""
    pipeline = HumanizedRAGPipeline()
    stats = pipeline.stats()
    print(json.dumps(stats, indent=2))


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="hrag",
        description=(
            "H-RAG: Hierarchical Vector Compression and "
            "Topic-Guided Retrieval for RAG"
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Ingest documents and build index")
    p_ingest.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to file or directory to ingest",
    )

    # query
    p_query = subparsers.add_parser("query", help="Query the index")
    p_query.add_argument("question", type=str, help="The question to ask")
    p_query.add_argument("--top-k", type=int, default=5, help="Number of results")
    p_query.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM generation (extractive only)",
    )

    # compare
    p_compare = subparsers.add_parser(
        "compare", help="Compare hierarchical vs flat retrieval"
    )
    p_compare.add_argument("question", type=str, help="The question to compare on")
    p_compare.add_argument("--top-k", type=int, default=5, help="Number of results")

    # stats
    subparsers.add_parser("stats", help="Print index statistics")

    args = parser.parse_args()

    _setup_logging(verbose=getattr(args, "verbose", False))

    if args.command is None:
        parser.print_help()
        return

    commands = {
        "ingest": cmd_ingest,
        "query": cmd_query,
        "compare": cmd_compare,
        "stats": cmd_stats,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
