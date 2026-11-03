"""Unit tests for evaluation metrics (Recall@K, MRR, Token F1, Exact Match)."""

import pytest
from src.evaluation.metrics import RetrievalMetrics, GenerationMetrics


class TestEvaluationMetrics:
    def test_recall_at_k(self):
        retrieved = [
            {"source": "report.pdf", "page": 1},
            {"source": "handbook.pdf", "page": 2},
            {"source": "policy.pdf", "page": 1}
        ]

        # Top-1 contains report.pdf
        assert RetrievalMetrics.recall_at_k(retrieved, "report.pdf", 1, k=1) == 1.0
        # handbook.pdf is at rank 2, so Recall@1 is 0.0, Recall@3 is 1.0
        assert RetrievalMetrics.recall_at_k(retrieved, "handbook.pdf", 2, k=1) == 0.0
        assert RetrievalMetrics.recall_at_k(retrieved, "handbook.pdf", 2, k=3) == 1.0
        # Non-existent doc
        assert RetrievalMetrics.recall_at_k(retrieved, "other.pdf", 1, k=3) == 0.0

    def test_reciprocal_rank(self):
        retrieved = [
            {"source": "docA.pdf", "page": 1},
            {"source": "docB.pdf", "page": 1},
            {"source": "docC.pdf", "page": 1}
        ]

        # docA is rank 1 -> 1.0
        assert RetrievalMetrics.reciprocal_rank(retrieved, "docA.pdf", 1) == 1.0
        # docB is rank 2 -> 0.5
        assert RetrievalMetrics.reciprocal_rank(retrieved, "docB.pdf", 1) == 0.5
        # docC is rank 3 -> 1/3
        assert pytest.approx(RetrievalMetrics.reciprocal_rank(retrieved, "docC.pdf", 1), 0.001) == 0.3333
        # not present -> 0.0
        assert RetrievalMetrics.reciprocal_rank(retrieved, "docX.pdf", 1) == 0.0

    def test_exact_match(self):
        pred = "Full-time employees work 40 hours per week."
        truth = "Full-Time Employees Work 40 hours per week!"
        assert GenerationMetrics.exact_match(pred, truth) == 1.0

        diff = "Employees work 35 hours."
        assert GenerationMetrics.exact_match(diff, truth) == 0.0

    def test_token_f1(self):
        pred = "employees work 40 hours weekly"
        truth = "employees work 40 hours per week"
        f1 = GenerationMetrics.token_f1(pred, truth)
        assert 0.6 < f1 < 1.0

    def test_abstention_accuracy(self):
        assert GenerationMetrics.abstention_accuracy(True, True) == 1.0
        assert GenerationMetrics.abstention_accuracy(False, False) == 1.0
        assert GenerationMetrics.abstention_accuracy(True, False) == 0.0
        assert GenerationMetrics.abstention_accuracy(False, True) == 0.0
