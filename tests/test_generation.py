"""Unit and integration tests for generation, grounding, and citation validation."""

import pytest
from src.preprocessing.chunking import DocumentChunk
from src.retrieval.dense_retriever import SearchResult
from src.generation.answer_generator import AnswerGenerator
from src.generation.grounding_validator import GroundingValidator
from src.generation.ollama_client import OllamaClient


@pytest.fixture
def retrieved_evidence():
    chunk = DocumentChunk(
        chunk_id="handbook_p1_c0",
        doc_id="doc_hb",
        source="Employee_Handbook.pdf",
        page=1,
        chunk_text="Core collaboration hours are 10:00 AM to 3:00 PM EST. Remote employees receive a $1,200 annual stipend.",
        word_count=18
    )
    return [
        SearchResult(
            chunk=chunk,
            score=0.88,
            retrieval_method="dense",
            rank=1
        )
    ]


class TestGenerationAndGrounding:
    def test_extractive_fallback_generation(self, retrieved_evidence):
        generator = AnswerGenerator()
        ans = generator.generate_answer(
            query="What are the core collaboration hours?",
            retrieved_chunks=retrieved_evidence,
            use_ollama=False
        )

        assert not ans.is_abstained
        assert "10:00 AM to 3:00 PM EST" in ans.answer
        assert len(ans.citations) > 0
        assert ans.citations[0]["source"] == "Employee_Handbook.pdf"
        assert ans.citations[0]["page"] == 1
        assert ans.generation_mode in ("extractive_fallback", "extractive_framed")

    def test_abstention_on_unsupported_query(self, retrieved_evidence):
        generator = AnswerGenerator()
        ans = generator.generate_answer(
            query="What is the CEO personal stock option allocation and vesting date?",
            retrieved_chunks=retrieved_evidence,
            use_ollama=False
        )

        assert ans.is_abstained
        assert "The requested information was not found in the uploaded documents." in ans.answer
        assert ans.citations == []

    def test_abstention_on_empty_retrieved_context(self):
        generator = AnswerGenerator()
        ans = generator.generate_answer(
            query="Any question with zero context?",
            retrieved_chunks=[],
            use_ollama=False
        )

        assert ans.is_abstained
        assert ans.answer == "The requested information was not found in the uploaded documents."

    def test_grounding_validator_detects_hallucination(self, retrieved_evidence):
        # Artificially create an unsupported hallucinated answer
        hallucinated_answer = "TechCorp was founded in Mars in year 3045 by robotic aliens."
        report = GroundingValidator.validate_answer(
            query="When was TechCorp founded?",
            answer=hallucinated_answer,
            retrieved_chunks=retrieved_evidence
        )

        # Grounding validator must detect unsupported claims and force abstention
        assert not report.is_valid or report.is_abstained
        assert "The requested information was not found in the uploaded documents." in report.final_answer

    def test_ollama_client_offline_graceful_fallback(self):
        client = OllamaClient(base_url="http://127.0.0.1:99999", timeout=1)
        assert not client.is_available()
        assert client.list_available_models() == []
        assert client.generate(prompt="Hello", system_prompt=None) is None
