"""Unit tests for dense, TF-IDF, and hybrid retrieval systems."""

import pytest
from src.preprocessing.chunking import DocumentChunk
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.tfidf_retriever import TfidfRetriever
from src.retrieval.reranker import HybridReranker


@pytest.fixture
def sample_chunks():
    return [
        DocumentChunk(
            chunk_id="doc1_p1_c0",
            doc_id="doc1",
            source="vacation_policy.pdf",
            page=1,
            chunk_text="Employees receive 20 days of paid vacation annually. Rollover limit is 5 days.",
            word_count=13
        ),
        DocumentChunk(
            chunk_id="doc2_p1_c0",
            doc_id="doc2",
            source="cybersecurity_handbook.pdf",
            page=1,
            chunk_text="Passwords must contain at least 12 characters, including uppercase, lowercase, numbers, and symbols.",
            word_count=13
        ),
        DocumentChunk(
            chunk_id="doc3_p1_c0",
            doc_id="doc3",
            source="engineering_guide.pdf",
            page=2,
            chunk_text="EduDocAI uses PyMuPDF for in-memory PDF extraction and FAISS for vector indexing.",
            word_count=13
        )
    ]


class TestRetrieval:
    def test_dense_retrieval_ranking(self, sample_chunks):
        retriever = DenseRetriever()
        retriever.build_index(sample_chunks)

        results = retriever.search("How many vacation days do employees get?", top_k=2)
        assert len(results) > 0
        top = results[0]
        assert top.chunk.source == "vacation_policy.pdf"
        assert top.rank == 1
        assert -1.0 <= top.score <= 1.0

    def test_tfidf_retrieval_ranking(self, sample_chunks):
        retriever = TfidfRetriever()
        retriever.build_index(sample_chunks)

        results = retriever.search("passwords symbols lowercase uppercase", top_k=1)
        assert len(results) == 1
        assert results[0].chunk.source == "cybersecurity_handbook.pdf"
        assert 0.0 <= results[0].score <= 1.0

    def test_empty_query_and_empty_index(self):
        retriever = DenseRetriever()
        assert retriever.search("") == []
        assert retriever.search("hello") == []

        tfidf = TfidfRetriever()
        assert tfidf.search("") == []
        assert tfidf.search("hello") == []

    def test_hybrid_reranker(self, sample_chunks):
        dense = DenseRetriever()
        dense.build_index(sample_chunks)

        tfidf = TfidfRetriever()
        tfidf.build_index(sample_chunks)

        q = "vector indexing FAISS EduDocAI"
        dense_res = dense.search(q, top_k=3)
        tfidf_res = tfidf.search(q, top_k=3)

        fused = HybridReranker.fuse(dense_res, tfidf_res, top_k=2)
        assert len(fused) <= 2
        assert fused[0].chunk.source == "engineering_guide.pdf"
        assert fused[0].retrieval_method == "hybrid_rrf"
