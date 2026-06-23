"""
LLM Answer Generation (Section V.F, Step 5).

Takes retrieved chunks + original query and generates a grounded answer.
Supports Ollama (local) or a simple extractive fallback.
"""

from __future__ import annotations

import logging
from typing import Optional

import requests

from hrag.config import OLLAMA_BASE_URL, OLLAMA_MODEL
from hrag.retriever import RetrievalResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
Use ONLY the information from the context below to answer the question.
If the context does not contain enough information to answer, say so clearly.
Cite which passage(s) you used in your answer."""


def _format_context(results: list[RetrievalResult]) -> str:
    """Format retrieved chunks into a numbered context block."""
    parts = []
    for i, r in enumerate(results, 1):
        source = r.metadata.get("doc_id", "unknown")
        page = r.metadata.get("page_number", "?")
        parts.append(
            f"[Passage {i}] (Source: {source}, Page: {page}, Score: {r.score:.3f})\n"
            f"{r.text}"
        )
    return "\n\n---\n\n".join(parts)


def generate_answer_ollama(
    query: str,
    results: list[RetrievalResult],
    model: str = OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
) -> tuple[str, bool]:
    """Generate an answer using Ollama (local LLM).

    Args:
        query: The user's question.
        results: Retrieved chunks with scores and metadata.
        model: Ollama model name.
        base_url: Ollama API base URL.

    Returns:
        (answer_text, success) — the generated answer and whether
        the LLM call succeeded.
    """
    context = _format_context(results)

    prompt = f"""Context:
{context}

Question: {query}

Answer the question using only the context above. Be thorough and cite passage numbers."""

    try:
        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "system": SYSTEM_PROMPT,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "No response generated."), True
    except (requests.ConnectionError, requests.Timeout):
        logger.warning("Ollama not reachable at %s — falling back to extractive", base_url)
        return "", False
    except Exception as e:
        logger.error("LLM generation failed: %s", e)
        return f"LLM error: {e}", False


def generate_answer_extractive(
    query: str,
    results: list[RetrievalResult],
) -> str:
    """Simple extractive fallback: return the most relevant passages.

    Used when Ollama is not available.
    """
    if not results:
        return "No relevant passages found."

    lines = [f'Based on the retrieved passages for: "{query}"\n']
    for i, r in enumerate(results, 1):
        source = r.metadata.get("doc_id", "unknown")
        page = r.metadata.get("page_number", "?")
        lines.append(
            f"[{i}] (Source: {source}, Page: {page}, Score: {r.score:.3f})\n"
            f"    {r.text}\n"
        )
    return "\n".join(lines)


def generate_answer(
    query: str,
    results: list[RetrievalResult],
    use_ollama: bool = True,
) -> str:
    """Generate an answer — tries Ollama first, falls back to extractive.

    Args:
        query: The user's question.
        results: Retrieved chunks.
        use_ollama: Whether to attempt Ollama generation.

    Returns:
        The generated (or extracted) answer string.
    """
    if not results:
        return "No relevant passages were retrieved. Cannot generate an answer."

    if use_ollama:
        answer, success = generate_answer_ollama(query, results)
        if success and answer:
            return answer

    # Fall back to extractive
    return generate_answer_extractive(query, results)
