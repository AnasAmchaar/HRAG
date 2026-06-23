"""
Document Chunking Module (Section V.B).

Splits documents into overlapping chunks using a sliding-window approach.
Supports PDF, TXT, and Markdown input formats.

Paper spec:
  - Window size: 256–512 tokens
  - Overlap: 10–20%
  - Recommended chunk size: 256–384 tokens
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from hrag.config import CHUNK_OVERLAP, CHUNK_SIZE

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A single document chunk with metadata."""

    text: str
    doc_id: str
    chunk_index: int
    page_number: Optional[int] = None
    char_offset: int = 0
    metadata: dict = field(default_factory=dict)


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using regex-based boundary detection."""
    # Split on sentence-ending punctuation followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _approximate_token_count(text: str) -> int:
    """Approximate token count using whitespace splitting.

    This is a rough heuristic (1 token ≈ 0.75 words for English).
    For production use, replace with the actual tokenizer.
    """
    words = text.split()
    return int(len(words) / 0.75)


def chunk_text(
    text: str,
    doc_id: str,
    chunk_size: int = CHUNK_SIZE,
    overlap_ratio: float = CHUNK_OVERLAP,
    page_number: Optional[int] = None,
) -> list[Chunk]:
    """Split a text string into overlapping chunks at sentence boundaries.

    Args:
        text: The input text to chunk.
        doc_id: Identifier for the source document.
        chunk_size: Target chunk size in approximate tokens.
        overlap_ratio: Fraction of chunk_size to overlap (0.0–1.0).
        page_number: Optional page number for metadata.

    Returns:
        List of Chunk objects with text and metadata.
    """
    sentences = _split_into_sentences(text)
    if not sentences:
        return []

    overlap_tokens = int(chunk_size * overlap_ratio)
    chunks: list[Chunk] = []

    current_sentences: list[str] = []
    current_token_count = 0
    char_offset = 0
    chunk_index = 0

    for sentence in sentences:
        sentence_tokens = _approximate_token_count(sentence)

        # If adding this sentence would exceed chunk_size, finalize current chunk
        if current_sentences and (current_token_count + sentence_tokens) > chunk_size:
            chunk_text_str = " ".join(current_sentences)
            chunks.append(Chunk(
                text=chunk_text_str,
                doc_id=doc_id,
                chunk_index=chunk_index,
                page_number=page_number,
                char_offset=char_offset,
            ))
            chunk_index += 1

            # Compute overlap: keep trailing sentences that fit within overlap_tokens
            overlap_sentences: list[str] = []
            overlap_count = 0
            for s in reversed(current_sentences):
                s_tokens = _approximate_token_count(s)
                if overlap_count + s_tokens > overlap_tokens:
                    break
                overlap_sentences.insert(0, s)
                overlap_count += s_tokens

            # Update char_offset: advance past the non-overlapping portion
            non_overlap_text = " ".join(
                current_sentences[: len(current_sentences) - len(overlap_sentences)]
            )
            char_offset += len(non_overlap_text) + 1  # +1 for the space

            current_sentences = overlap_sentences
            current_token_count = overlap_count

        current_sentences.append(sentence)
        current_token_count += sentence_tokens

    # Don't forget the last chunk
    if current_sentences:
        chunk_text_str = " ".join(current_sentences)
        chunks.append(Chunk(
            text=chunk_text_str,
            doc_id=doc_id,
            chunk_index=chunk_index,
            page_number=page_number,
            char_offset=char_offset,
        ))

    return chunks


def load_and_chunk_file(filepath: Path, chunk_size: int = CHUNK_SIZE) -> list[Chunk]:
    """Load a file (PDF, TXT, or MD) and split into chunks.

    Args:
        filepath: Path to the input file.
        chunk_size: Target chunk size in approximate tokens.

    Returns:
        List of Chunk objects.
    """
    filepath = Path(filepath)
    doc_id = filepath.stem
    all_chunks: list[Chunk] = []

    logger.info("Loading file: %s", filepath)

    if filepath.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(filepath))
        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                page_chunks = chunk_text(
                    page_text,
                    doc_id=doc_id,
                    chunk_size=chunk_size,
                    page_number=page_num,
                )
                all_chunks.extend(page_chunks)

    elif filepath.suffix.lower() in (".txt", ".md", ".markdown"):
        text = filepath.read_text(encoding="utf-8")
        all_chunks = chunk_text(text, doc_id=doc_id, chunk_size=chunk_size)

    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")

    logger.info("Created %d chunks from %s", len(all_chunks), filepath.name)
    return all_chunks


def load_and_chunk_directory(
    directory: Path, chunk_size: int = CHUNK_SIZE
) -> list[Chunk]:
    """Load and chunk all supported files in a directory.

    Args:
        directory: Path to directory containing documents.
        chunk_size: Target chunk size in approximate tokens.

    Returns:
        List of Chunk objects from all files.
    """
    directory = Path(directory)
    supported_extensions = {".pdf", ".txt", ".md", ".markdown"}
    all_chunks: list[Chunk] = []

    for filepath in sorted(directory.iterdir()):
        if filepath.suffix.lower() in supported_extensions:
            file_chunks = load_and_chunk_file(filepath, chunk_size)
            all_chunks.extend(file_chunks)

    logger.info(
        "Loaded %d chunks from %d files in %s",
        len(all_chunks),
        len(list(directory.iterdir())),
        directory,
    )
    return all_chunks
