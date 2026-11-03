"""Input and document validation utilities for EduDocAI."""

from pathlib import Path
from typing import List, Optional
from config.settings import settings


class DocumentValidationError(Exception):
    """Raised when document validation fails."""
    pass


class CorruptedDocumentError(DocumentValidationError):
    """Raised when a document cannot be parsed or opened."""
    pass


class EmptyDocumentError(DocumentValidationError):
    """Raised when an uploaded document contains no extractable text."""
    pass


class UnsupportedFormatError(DocumentValidationError):
    """Raised when an uploaded file format is not supported."""
    pass


def validate_file_extension(filename: str, allowed_extensions: Optional[List[str]] = None) -> str:
    """Validates that the file has an allowed extension.

    Args:
        filename: Name or path of the file.
        allowed_extensions: List of allowed extensions (defaults to config).

    Returns:
        str: Lowercase extension including the dot (e.g., '.pdf').

    Raises:
        UnsupportedFormatError: If extension is not allowed.
    """
    if allowed_extensions is None:
        allowed_extensions = settings.ALLOWED_EXTENSIONS

    ext = Path(filename).suffix.lower()
    if ext not in allowed_extensions:
        raise UnsupportedFormatError(
            f"Unsupported file format '{ext}'. Allowed formats: {', '.join(allowed_extensions)}"
        )
    return ext


def validate_file_size(size_bytes: int, max_size_mb: Optional[int] = None) -> None:
    """Validates that the file size is within acceptable limits.

    Args:
        size_bytes: Size of the file in bytes.
        max_size_mb: Maximum allowed size in megabytes.

    Raises:
        DocumentValidationError: If file size exceeds maximum or is 0 bytes.
    """
    if max_size_mb is None:
        max_size_mb = settings.MAX_FILE_SIZE_MB

    if size_bytes == 0:
        raise EmptyDocumentError("File is empty (0 bytes).")

    max_bytes = max_size_mb * 1024 * 1024
    if size_bytes > max_bytes:
        raise DocumentValidationError(
            f"File size ({size_bytes / (1024*1024):.2f} MB) exceeds maximum allowed size ({max_size_mb} MB)."
        )


def validate_query(query: Optional[str]) -> str:
    """Validates and trims user query string.

    Args:
        query: User input question.

    Returns:
        str: Cleaned non-empty query.

    Raises:
        ValueError: If query is empty or whitespace only.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty or whitespace only.")
    return query.strip()
