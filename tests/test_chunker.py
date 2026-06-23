"""Tests for the document chunker."""

from hrag.chunker import Chunk, chunk_text, _approximate_token_count, _split_into_sentences


class TestSentenceSplitting:
    def test_basic_splitting(self):
        text = "Hello world. How are you? I am fine."
        sentences = _split_into_sentences(text)
        assert len(sentences) == 3
        assert sentences[0] == "Hello world."
        assert sentences[1] == "How are you?"
        assert sentences[2] == "I am fine."

    def test_empty_string(self):
        assert _split_into_sentences("") == []

    def test_single_sentence(self):
        sentences = _split_into_sentences("Just one sentence.")
        assert len(sentences) == 1


class TestTokenCount:
    def test_approximate_count(self):
        text = "This is a simple test sentence with some words"
        count = _approximate_token_count(text)
        # ~9 words / 0.75 ≈ 12 tokens
        assert count > 0
        assert count < 100


class TestChunking:
    def test_single_chunk(self):
        """Short text should produce a single chunk."""
        text = "This is a short document. It has only two sentences."
        chunks = chunk_text(text, doc_id="test", chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0].doc_id == "test"
        assert chunks[0].chunk_index == 0

    def test_multiple_chunks(self):
        """Long text should be split into multiple chunks."""
        # Generate a long text
        sentences = [f"Sentence number {i} with some filler words to pad." for i in range(100)]
        text = " ".join(sentences)
        chunks = chunk_text(text, doc_id="long_doc", chunk_size=50)
        assert len(chunks) > 1

    def test_overlap_exists(self):
        """Consecutive chunks should share overlapping content."""
        sentences = [f"Unique sentence {i} about topic {i}." for i in range(50)]
        text = " ".join(sentences)
        chunks = chunk_text(text, doc_id="overlap_test", chunk_size=30, overlap_ratio=0.2)

        if len(chunks) >= 2:
            # Check that there's some text overlap between consecutive chunks
            for i in range(len(chunks) - 1):
                words_a = set(chunks[i].text.split())
                words_b = set(chunks[i + 1].text.split())
                overlap = words_a & words_b
                # There should be SOME shared words due to overlap
                assert len(overlap) > 0, f"No overlap between chunk {i} and {i+1}"

    def test_metadata_preserved(self):
        """Chunks should carry the correct metadata."""
        text = "A test sentence. Another one."
        chunks = chunk_text(text, doc_id="meta_test", page_number=42)
        for chunk in chunks:
            assert chunk.doc_id == "meta_test"
            assert chunk.page_number == 42

    def test_chunk_indices_sequential(self):
        """Chunk indices should be sequential."""
        sentences = [f"Sentence {i}." for i in range(50)]
        text = " ".join(sentences)
        chunks = chunk_text(text, doc_id="seq_test", chunk_size=30)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
