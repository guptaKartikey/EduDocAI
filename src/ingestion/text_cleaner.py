"""Text cleaning and normalization pipeline for EduDocAI."""

import re
import unicodedata


class TextCleaner:
    """Provides modular text cleaning for extracted document pages and queries."""

    @staticmethod
    def clean(text: str) -> str:
        """Cleans and normalizes text while preserving punctuation, numbers, and case.

        Args:
            text: Raw extracted string.

        Returns:
            str: Cleaned and normalized text.
        """
        if not text:
            return ""

        # Normalize unicode (NFKC: decomposes and recomposes compatible characters)
        text = unicodedata.normalize("NFKC", text)

        # Replace replacement character \ufffd (commonly from broken bullet/hyphen glyphs)
        text = text.replace("\ufffd", " - ")

        # Replace non-breaking spaces and unusual whitespace
        text = text.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")

        # Standardize common bullet and dash symbols
        text = text.replace("•", " • ").replace("–", "-").replace("—", "-")

        # Normalize Windows / Mac carriage returns to standard newline
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Replace control characters except newlines and tabs
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

        # Replace hyphens split across linebreaks: e.g., 'trans-\nformer' -> 'transformer'
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

        # Normalize multiple horizontal spaces/tabs into a single space
        text = re.sub(r"[ \t]+", " ", text)

        # Normalize excessive consecutive newlines (3 or more down to 2)
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

        return text.strip()

    @staticmethod
    def clean_query(query: str) -> str:
        """Lightweight cleaning specifically for search queries.

        Args:
            query: Raw user query string.

        Returns:
            str: Normalized single-line query.
        """
        if not query:
            return ""
        cleaned = unicodedata.normalize("NFKC", query)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()
