"""Text analytics and zero-shot keyword/document classification for Track C."""

import re
from typing import Dict, Any, List, Tuple
from collections import Counter
from src.generation.grounding_validator import GroundingValidator


class TextAnalyticsEngine:
    """Computes NLP text analytics, key phrase extraction, and domain classification."""

    CATEGORY_KEYWORDS = {
        "Policy & Compliance Q&A": {
            "policy", "handbook", "leave", "pto", "benefits", "hours", "remote",
            "vacation", "insurance", "parental", "confidential", "stipend", "compliance"
        },
        "Resume & Professional Profile": {
            "education", "experience", "skills", "projects", "engineer", "technologies",
            "developer", "python", "degree", "university", "work", "bachelor", "master"
        },
        "Technical & System Architecture": {
            "architecture", "pipeline", "faiss", "embedding", "vector", "tfidf",
            "latency", "retrieval", "tokens", "database", "api", "framework", "index"
        },
        "Support Ticket & Helpdesk": {
            "issue", "error", "bug", "crash", "failed", "help", "problem",
            "support", "ticket", "urgent", "resolve", "assistance", "status"
        },
        "Ethics & Governance": {
            "ethics", "guidelines", "governance", "safety", "groundedness",
            "provenance", "hallucination", "privacy", "ephemeral", "principles"
        }
    }

    @classmethod
    def analyze_text(cls, text: str) -> Dict[str, Any]:
        """Computes linguistic and structural analytics on document text."""
        words = re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", text.lower())
        total_words = len(words)
        if total_words == 0:
            return {
                "total_words": 0,
                "total_chars": len(text),
                "unique_words": 0,
                "lexical_diversity": 0.0,
                "reading_time_mins": 0.0,
                "top_keywords": []
            }

        unique_words = len(set(words))
        diversity = round(unique_words / total_words, 3)
        reading_time = round(total_words / 200, 1)  # standard 200 words/min

        # Filter stopwords for top keywords
        content_words = [w for w in words if w not in GroundingValidator.STOPWORDS and len(w) > 2]
        counts = Counter(content_words).most_common(8)

        return {
            "total_words": total_words,
            "total_chars": len(text),
            "unique_words": unique_words,
            "lexical_diversity": diversity,
            "reading_time_mins": max(reading_time, 0.5),
            "top_keywords": counts
        }

    @classmethod
    def classify_domain(cls, text: str) -> Dict[str, Any]:
        """Classifies document or query into Track C application domains."""
        tokens = set(re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower()))
        scores: Dict[str, int] = {}

        for category, kws in cls.CATEGORY_KEYWORDS.items():
            overlap = len(tokens & kws)
            scores[category] = overlap

        best_cat = max(scores, key=scores.get)
        total_overlap = sum(scores.values())

        if total_overlap == 0:
            return {
                "category": "General Document Intelligence",
                "confidence": 0.50,
                "matched_terms": []
            }

        confidence = round(scores[best_cat] / total_overlap, 2)
        matched_terms = list(tokens & cls.CATEGORY_KEYWORDS[best_cat])

        return {
            "category": best_cat,
            "confidence": min(max(confidence, 0.60), 0.98),
            "matched_terms": matched_terms
        }
