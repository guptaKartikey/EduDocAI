"""Unit tests for intelligent chunking and metadata preservation."""

import pytest
from src.ingestion.document_loader import ExtractedPage
from src.preprocessing.chunking import TextChunker, DocumentChunk


class TestChunking:
    def test_chunk_page_metadata_preservation(self):
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        page = ExtractedPage(
            doc_id="doc123",
            source="manual.pdf",
            page=4,
            text="First sentence of manual. Second sentence with details. Third sentence summarizing findings."
        )

        chunks = chunker.chunk_page(page)
        assert len(chunks) >= 1

        for c in chunks:
            assert c.doc_id == "doc123"
            assert c.source == "manual.pdf"
            assert c.page == 4
            assert c.chunk_id.startswith("doc123_p4_c")
            assert c.word_count > 0
            assert len(c.chunk_text) > 0

    def test_chunking_overlap_and_sliding_window(self):
        # Create 120 words of text
        words = [f"word{i}" for i in range(120)]
        text = ". ".join([" ".join(words[i:i+10]) for i in range(0, 120, 10)]) + "."

        chunker = TextChunker(chunk_size=40, chunk_overlap=10)
        page = ExtractedPage(doc_id="d1", source="doc.txt", page=1, text=text)

        chunks = chunker.chunk_page(page)
        assert len(chunks) > 1

        # Check that consecutive chunks share overlap words
        words_c0 = chunks[0].chunk_text.split()
        words_c1 = chunks[1].chunk_text.split()
        common = set(words_c0[-10:]) & set(words_c1[:15])
        assert len(common) > 0

    def test_invalid_parameters_raise_error(self):
        with pytest.raises(ValueError):
            TextChunker(chunk_size=0, chunk_overlap=10)

        with pytest.raises(ValueError):
            TextChunker(chunk_size=100, chunk_overlap=100)

        with pytest.raises(ValueError):
            TextChunker(chunk_size=100, chunk_overlap=-5)

    def test_empty_page_returns_no_chunks(self):
        chunker = TextChunker()
        page = ExtractedPage(doc_id="d1", source="empty.txt", page=1, text="")
        chunks = chunker.chunk_page(page)
        assert chunks == []
