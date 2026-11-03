"""Intelligent text chunking with metadata preservation for EduDocAI."""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from config.settings import settings
from src.ingestion.document_loader import ExtractedPage
from src.utils.logger import logger


@dataclass
class DocumentChunk:
    """Represents an indexed chunk with source, page, and location metadata."""
    chunk_id: str
    doc_id: str
    source: str
    page: int
    chunk_text: str
    word_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for serialization and evaluation."""
        return asdict(self)


class TextChunker:
    """Configurable sliding-window chunker with sentence-boundary awareness."""

    def __init__(
        self,
        chunk_size: int = settings.DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = settings.DEFAULT_CHUNK_OVERLAP
    ):
        """Initializes chunker with size and overlap constraints.

        Args:
            chunk_size: Target number of words per chunk (e.g. 400).
            chunk_overlap: Number of overlapping words between consecutive chunks (e.g. 60).
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def explain_chunking_parameters() -> Dict[str, str]:
        """Explains the architectural effect of chunk size and overlap on RAG performance."""
        return {
            "chunk_size": (
                "Chunk size determines the context granularity. Smaller chunks (e.g. 100-200 words) "
                "allow high-precision retrieval of specific facts but risk losing broader paragraph context. "
                "Larger chunks (e.g. 500-1000 words) capture comprehensive context but may dilute dense semantic "
                "similarity scores and exceed LLM context budgets."
            ),
            "chunk_overlap": (
                "Overlap (e.g. 10%-15% of chunk size) ensures that sentences, definitions, or numerical tables "
                "occurring near chunk boundaries are not arbitrarily split, maintaining cross-boundary coherence "
                "and retrieval continuity."
            )
        }

    def chunk_page(self, page: ExtractedPage) -> List[DocumentChunk]:
        """Splits an ExtractedPage into overlapping sentence-aware chunks.

        Args:
            page: ExtractedPage with doc_id, source, page number, and text.

        Returns:
            List[DocumentChunk]: Non-empty chunks tagged with source and page.

        Chunking strategy
        -----------------
        We split on **newlines** (one idea per line – resumes, reports) and on
        sentence-ending punctuation (`.!?`).  Sentences are grouped into chunks by
        *word count*; when a chunk reaches ``chunk_size`` words the buffer is
        flushed and the last ``chunk_overlap`` words carry over to the next chunk.
        Chunks are joined with '\\n' to preserve line-level structure for downstream
        sentence splitting in the answer generator.
        """
        text = page.text.strip()
        if not text:
            return []

        # Split into logical lines/sentences respecting both newlines and [.!?]
        sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return []

        chunks: List[DocumentChunk] = []
        # Buffer holds (sentence_str, word_list) tuples
        buf_sentences: List[str] = []
        buf_word_count: int = 0
        current_chunk_idx = 0

        def _flush_buffer() -> None:
            nonlocal buf_sentences, buf_word_count, current_chunk_idx
            if not buf_sentences:
                return
            chunk_text = "\n".join(buf_sentences).strip()
            if chunk_text:
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{page.doc_id}_p{page.page}_c{current_chunk_idx}",
                        doc_id=page.doc_id,
                        source=page.source,
                        page=page.page,
                        chunk_text=chunk_text,
                        word_count=buf_word_count,
                    )
                )
                current_chunk_idx += 1

            # Keep overlap: walk back from end until we've kept ~chunk_overlap words
            if self.chunk_overlap > 0:
                overlap_sents: List[str] = []
                overlap_wc = 0
                for sent in reversed(buf_sentences):
                    w = len(sent.split())
                    if overlap_wc + w > self.chunk_overlap:
                        break
                    overlap_sents.insert(0, sent)
                    overlap_wc += w
                buf_sentences = overlap_sents
                buf_word_count = overlap_wc
            else:
                buf_sentences = []
                buf_word_count = 0

        for sentence in sentences:
            s_words = sentence.split()
            s_wc = len(s_words)
            if s_wc == 0:
                continue

            # A single sentence longer than chunk_size: slide through it word-by-word
            if s_wc > self.chunk_size:
                _flush_buffer()
                step = self.chunk_size - self.chunk_overlap
                for i in range(0, s_wc, step):
                    sub_words = s_words[i: i + self.chunk_size]
                    if not sub_words:
                        continue
                    sub_text = " ".join(sub_words)
                    chunks.append(
                        DocumentChunk(
                            chunk_id=f"{page.doc_id}_p{page.page}_c{current_chunk_idx}",
                            doc_id=page.doc_id,
                            source=page.source,
                            page=page.page,
                            chunk_text=sub_text,
                            word_count=len(sub_words),
                        )
                    )
                    current_chunk_idx += 1
                # Overlap from end of long sentence
                buf_sentences = [" ".join(s_words[-self.chunk_overlap:])] if self.chunk_overlap else []
                buf_word_count = min(s_wc, self.chunk_overlap)
                continue

            # Would adding this sentence overflow the chunk?
            if buf_word_count + s_wc > self.chunk_size and buf_sentences:
                _flush_buffer()

            buf_sentences.append(sentence)
            buf_word_count += s_wc

        _flush_buffer()
        return chunks

    def chunk_documents(self, pages: List[ExtractedPage]) -> List[DocumentChunk]:
        """Chunks a sequence of extracted pages across multiple documents.

        Args:
            pages: List of ExtractedPage objects.

        Returns:
            List[DocumentChunk]: All generated chunks with preserved metadata.
        """
        all_chunks: List[DocumentChunk] = []
        for page in pages:
            page_chunks = self.chunk_page(page)
            all_chunks.extend(page_chunks)

        logger.info(
            f"Chunking complete: {len(pages)} pages partitioned into {len(all_chunks)} chunks "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap})."
        )
        return all_chunks
