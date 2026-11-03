"""Information retrieval and generation evaluation metrics for EduDocAI."""

import re
from typing import List, Dict, Any, Optional, Set


class RetrievalMetrics:
    """Computes ranking and retrieval quality metrics."""

    @staticmethod
    def recall_at_k(retrieved_items: List[Dict[str, Any]], target_doc: str, target_page: Optional[int], k: int) -> float:
        """Calculates Recall@K: 1.0 if target document (and optional page) appears in top-K, else 0.0.

        Args:
            retrieved_items: List of retrieved chunk dictionaries.
            target_doc: Expected document filename (or base name).
            target_page: Expected page number (optional).
            k: Top-K cutoff.

        Returns:
            float: 1.0 if found within top-K, 0.0 otherwise.
        """
        if not target_doc:
            return 1.0  # Not applicable or abstention query

        top_k_items = retrieved_items[:k]
        for item in top_k_items:
            source = item.get("source", "")
            page = item.get("page")
            # Normalize extension differences (e.g. .pdf vs .txt)
            clean_source = source.replace(".pdf", "").replace(".txt", "")
            clean_target = target_doc.replace(".pdf", "").replace(".txt", "")

            if clean_source == clean_target:
                if target_page is None or page == target_page:
                    return 1.0
        return 0.0

    @staticmethod
    def reciprocal_rank(retrieved_items: List[Dict[str, Any]], target_doc: str, target_page: Optional[int]) -> float:
        """Calculates Reciprocal Rank: 1 / rank of first relevant chunk, or 0.0 if not found.

        Args:
            retrieved_items: List of retrieved chunk dictionaries.
            target_doc: Expected document filename.
            target_page: Expected page number.

        Returns:
            float: Reciprocal rank value in [0.0, 1.0].
        """
        if not target_doc:
            return 1.0

        for rank, item in enumerate(retrieved_items, start=1):
            source = item.get("source", "")
            page = item.get("page")
            clean_source = source.replace(".pdf", "").replace(".txt", "")
            clean_target = target_doc.replace(".pdf", "").replace(".txt", "")

            if clean_source == clean_target:
                if target_page is None or page == target_page:
                    return 1.0 / rank
        return 0.0


class GenerationMetrics:
    """Computes exact match, token F1, groundedness, and abstention metrics."""

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalizes text for linguistic metrics."""
        text = text.lower()
        # Remove punctuation
        text = re.sub(r"[^\w\s]", "", text)
        # Collapse whitespace
        text = " ".join(text.split())
        return text

    @classmethod
    def exact_match(cls, prediction: str, ground_truth: str) -> float:
        """Computes Exact Match (EM) score (1.0 or 0.0)."""
        return 1.0 if cls.normalize_text(prediction) == cls.normalize_text(ground_truth) else 0.0

    @classmethod
    def token_f1(cls, prediction: str, ground_truth: str) -> float:
        """Computes token-level precision, recall, and F1 harmonic mean."""
        pred_tokens = cls.normalize_text(prediction).split()
        truth_tokens = cls.normalize_text(ground_truth).split()

        if not pred_tokens or not truth_tokens:
            return 1.0 if pred_tokens == truth_tokens else 0.0

        pred_set = set(pred_tokens)
        truth_set = set(truth_tokens)
        common = pred_set & truth_set

        if not common:
            return 0.0

        precision = len(common) / len(pred_set)
        recall = len(common) / len(truth_set)
        return (2.0 * precision * recall) / (precision + recall)

    @classmethod
    def citation_accuracy(
        cls,
        predicted_citations: List[Dict[str, Any]],
        target_doc: Optional[str],
        target_page: Optional[int]
    ) -> float:
        """Checks if predicted citations include the correct ground-truth document and page."""
        if not target_doc:
            # If query was an abstention, citation list should ideally be empty
            return 1.0 if not predicted_citations else 0.0

        clean_target = target_doc.replace(".pdf", "").replace(".txt", "")
        for cite in predicted_citations:
            clean_source = cite.get("source", "").replace(".pdf", "").replace(".txt", "")
            if clean_source == clean_target:
                if target_page is None or cite.get("page") == target_page:
                    return 1.0
        return 0.0

    @classmethod
    def abstention_accuracy(cls, is_abstained: bool, target_is_abstention: bool) -> float:
        """Checks whether the system correctly abstained when information is missing."""
        return 1.0 if is_abstained == target_is_abstention else 0.0
