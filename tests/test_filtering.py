"""Unit and integration tests for document filtering before ranking."""

import pytest
from src.preprocessing.chunking import DocumentChunk
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.tfidf_retriever import TfidfRetriever
from src.retrieval.document_filter import DocumentFilter


@pytest.fixture
def multi_doc_chunks():
    return [
        DocumentChunk(
            chunk_id="resume_c0",
            doc_id="doc_resume",
            source="Resume.pdf",
            page=1,
            chunk_text="Kartik is an AI Engineer specializing in NLP, RAG, PyTorch, and LangChain.",
            word_count=12
        ),
        DocumentChunk(
            chunk_id="policy_c0",
            doc_id="doc_policy",
            source="Employee_Policy.pdf",
            page=1,
            chunk_text="All engineers must comply with the remote security guidelines and AI safety handbook.",
            word_count=13
        ),
        DocumentChunk(
            chunk_id="report_c0",
            doc_id="doc_report",
            source="Project_Report.pdf",
            page=1,
            chunk_text="Project EduDocAI achieved 93% MRR on dense semantic retrieval benchmarks.",
            word_count=10
        )
    ]


class TestDocumentFiltering:
    def test_document_filter_standalone(self, multi_doc_chunks):
        # Filter for Resume only
        filtered = DocumentFilter.filter_chunks(multi_doc_chunks, selected_sources=["Resume.pdf"])
        assert len(filtered) == 1
        assert filtered[0].source == "Resume.pdf"

        # Filter across two documents
        filtered_multi = DocumentFilter.filter_chunks(
            multi_doc_chunks,
            selected_sources=["Resume.pdf", "Project_Report.pdf"]
        )
        assert len(filtered_multi) == 2
        sources = {c.source for c in filtered_multi}
        assert sources == {"Resume.pdf", "Project_Report.pdf"}

        # Empty selection returns all
        filtered_all = DocumentFilter.filter_chunks(multi_doc_chunks, selected_sources=[])
        assert len(filtered_all) == 3

    def test_dense_retrieval_strict_filtering(self, multi_doc_chunks):
        dense = DenseRetriever()
        dense.build_index(multi_doc_chunks)

        # Query about "engineer", which appears in Resume and Policy
        # When filtered to Resume.pdf, MUST NOT return Employee_Policy.pdf
        results = dense.search(
            query="Tell me about the engineer",
            top_k=3,
            selected_sources=["Resume.pdf"]
        )

        assert len(results) == 1
        assert results[0].chunk.source == "Resume.pdf"
        assert all(r.chunk.source == "Resume.pdf" for r in results)

    def test_tfidf_retrieval_strict_filtering(self, multi_doc_chunks):
        tfidf = TfidfRetriever()
        tfidf.build_index(multi_doc_chunks)

        # Query "security guidelines"
        # When filtered to Project_Report.pdf, should NOT return Employee_Policy.pdf
        results = tfidf.search(
            query="security guidelines",
            top_k=3,
            selected_sources=["Project_Report.pdf"]
        )

        for r in results:
            assert r.chunk.source == "Project_Report.pdf"

    def test_filtered_retrieval_multi_select(self, multi_doc_chunks):
        dense = DenseRetriever()
        dense.build_index(multi_doc_chunks)

        results = dense.search(
            query="engineer project report",
            top_k=5,
            selected_sources=["Resume.pdf", "Project_Report.pdf"]
        )

        for r in results:
            assert r.chunk.source in ["Resume.pdf", "Project_Report.pdf"]
            assert r.chunk.source != "Employee_Policy.pdf"
