"""Grounding and citation verification module for EduDocAI."""

import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from config.settings import settings
from src.retrieval.dense_retriever import SearchResult
from src.utils.logger import logger


@dataclass
class ValidationReport:
    """Detailed report on groundedness and citation fidelity."""
    is_valid: bool
    groundedness_score: float
    citation_validity: bool
    is_abstained: bool
    notes: List[str]
    final_answer: str


class GroundingValidator:
    """Validates that generated answers are strictly grounded in retrieved evidence."""

    STOPWORDS = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
        "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
        "below", "between", "both", "but", "by", "can", "cannot", "could", "did", "do",
        "does", "doing", "down", "during", "each", "few", "for", "from", "further",
        "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
        "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it", "its",
        "itself", "just", "me", "more", "most", "my", "myself", "no", "nor", "not",
        "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
        "ourselves", "out", "over", "own", "same", "she", "should", "so", "some",
        "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then",
        "there", "these", "they", "this", "those", "through", "to", "too", "under",
        "until", "up", "very", "was", "wasn't", "we", "were", "what", "when", "where",
        "which", "while", "who", "whom", "why", "with", "would", "you", "your", "yours"
    }

    @classmethod
    def validate_answer(
        cls,
        query: str,
        answer: str,
        retrieved_chunks: List[SearchResult],
        min_groundedness_threshold: float = 0.30
    ) -> ValidationReport:
        """Evaluates whether an answer is supported by the retrieved document chunks.

        Args:
            query: User's original question.
            answer: Generated answer text.
            retrieved_chunks: List of retrieved SearchResult chunks provided as context.
            min_groundedness_threshold: Minimum overlap ratio required before triggering abstention.

        Returns:
            ValidationReport: Detailed findings including groundedness score and safety decisions.
        """
        notes: List[str] = []

        # 1. Check for explicit abstention
        normalized_answer = answer.strip().lower()
        abstention_patterns = [
            "requested information was not found",
            "not found in the uploaded documents",
            "not mentioned in the provided",
            "not found in the provided context",
            "no information",
            "context does not contain",
            "cannot find"
        ]
        if any(pat in normalized_answer for pat in abstention_patterns):
            notes.append("System correctly abstained: information not present in context.")
            return ValidationReport(
                is_valid=True,
                groundedness_score=1.0,
                citation_validity=True,
                is_abstained=True,
                notes=notes,
                final_answer=settings.ABSTENTION_MESSAGE
            )

        # 2. Check if context is completely empty
        if not retrieved_chunks:
            notes.append("No retrieved context available for answering.")
            return ValidationReport(
                is_valid=False,
                groundedness_score=0.0,
                citation_validity=False,
                is_abstained=True,
                notes=notes,
                final_answer=settings.ABSTENTION_MESSAGE
            )

        # 3. Concatenate and tokenize context
        context_text = " ".join([r.chunk.chunk_text.lower() for r in retrieved_chunks])
        context_words = set(re.findall(r"\b[a-zA-Z0-9_\-\.\%]+\b", context_text))

        # Tokenize answer (excluding punctuation and stopwords)
        answer_tokens = re.findall(r"\b[a-zA-Z0-9_\-\.\%]+\b", answer.lower())
        content_tokens = [t for t in answer_tokens if t not in cls.STOPWORDS and len(t) > 1]

        if not content_tokens:
            notes.append("Answer contains insufficient factual tokens.")
            return ValidationReport(
                is_valid=False,
                groundedness_score=0.0,
                citation_validity=False,
                is_abstained=True,
                notes=notes,
                final_answer=settings.ABSTENTION_MESSAGE
            )

        # 4. Measure token-level grounding ratio
        supported_tokens = [t for t in content_tokens if t in context_words]
        groundedness_score = len(supported_tokens) / len(content_tokens)
        notes.append(
            f"Factual grounding score: {groundedness_score:.2f} ({len(supported_tokens)}/{len(content_tokens)} content tokens verified in context)."
        )

        # 5. Citation verification
        # Look for citations in format [Source: doc.pdf, Page: 1] or similar
        citation_valid = True
        cited_sources = re.findall(r"\[(?:Source|Doc):\s*([^,\]]+)(?:,\s*Page:\s*(\d+))?\]", answer, re.IGNORECASE)
        valid_sources = {r.chunk.source for r in retrieved_chunks}
        valid_pages = {(r.chunk.source, r.chunk.page) for r in retrieved_chunks}

        for source_name, page_str in cited_sources:
            source_clean = source_name.strip()
            if source_clean not in valid_sources:
                citation_valid = False
                notes.append(f"Citation mismatch: Cited source '{source_clean}' is not in retrieved context.")
            elif page_str:
                page_num = int(page_str)
                if (source_clean, page_num) not in valid_pages:
                    notes.append(f"Citation warning: Cited page {page_num} of '{source_clean}' was not in retrieved chunks.")

        # 6. Safety check: abstention fallback if hallucination suspected
        if groundedness_score < min_groundedness_threshold:
            notes.append(
                f"Grounding score {groundedness_score:.2f} falls below threshold {min_groundedness_threshold}. "
                "Overriding with safe abstention."
            )
            return ValidationReport(
                is_valid=False,
                groundedness_score=groundedness_score,
                citation_validity=False,
                is_abstained=True,
                notes=notes,
                final_answer=settings.ABSTENTION_MESSAGE
            )

        return ValidationReport(
            is_valid=True,
            groundedness_score=groundedness_score,
            citation_validity=citation_valid,
            is_abstained=False,
            notes=notes,
            final_answer=answer
        )
