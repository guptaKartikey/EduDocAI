"""Document ingestion and text extraction module for EduDocAI."""

import hashlib
import io
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
from dataclasses import dataclass, asdict

import fitz  # PyMuPDF

from config.settings import settings
from src.ingestion.text_cleaner import TextCleaner
from src.utils.logger import logger
from src.utils.validators import (
    validate_file_extension,
    validate_file_size,
    DocumentValidationError,
    CorruptedDocumentError,
    EmptyDocumentError,
)


@dataclass
class ExtractedPage:
    """Represents a single extracted page of a document with full metadata."""
    doc_id: str
    source: str
    page: int
    text: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert page metadata and content to a dictionary."""
        return asdict(self)


class DocumentLoader:
    """Loads and extracts text from PDF, TXT, and Markdown files safely in-memory."""

    def __init__(self, cleaner: Optional[TextCleaner] = None):
        self.cleaner = cleaner or TextCleaner()

    @staticmethod
    def generate_doc_id(source_name: str, content_sample: bytes = b"") -> str:
        """Generates a stable, reproducible doc_id based on source name and content hash.

        Args:
            source_name: Filename of the document.
            content_sample: Sample bytes from content for collision avoidance.

        Returns:
            str: 12-character hex ID.
        """
        hasher = hashlib.sha256(source_name.encode("utf-8"))
        if content_sample:
            hasher.update(content_sample[:2048])
        return hasher.hexdigest()[:12]

    def load_from_bytes(self, file_bytes: Union[bytes, io.BytesIO], filename: str) -> List[ExtractedPage]:
        """Extracts text from raw bytes or BytesIO stream.

        Prevents Windows file-locking and temporary path permission errors.

        Args:
            file_bytes: In-memory byte array or BytesIO object.
            filename: Original filename with extension.

        Returns:
            List[ExtractedPage]: List of extracted pages with metadata.
        """
        if isinstance(file_bytes, io.BytesIO):
            data = file_bytes.getvalue()
        else:
            data = file_bytes

        # Validate format & size
        ext = validate_file_extension(filename)
        validate_file_size(len(data))

        doc_id = self.generate_doc_id(filename, data)
        logger.info(f"Loading document '{filename}' (doc_id={doc_id}, size={len(data)} bytes)...")

        if ext == ".pdf":
            return self._extract_pdf_bytes(data, filename, doc_id)
        elif ext in [".txt", ".md"]:
            return self._extract_text_bytes(data, filename, doc_id)
        else:
            raise DocumentValidationError(f"Unsupported document extension: {ext}")

    def load_from_path(self, file_path: Union[str, Path]) -> List[ExtractedPage]:
        """Extracts text from a local disk path.

        Args:
            file_path: Path to the target file.

        Returns:
            List[ExtractedPage]: List of extracted pages with metadata.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, "rb") as f:
            data = f.read()

        return self.load_from_bytes(data, path.name)

    def _extract_pdf_bytes(self, data: bytes, filename: str, doc_id: str) -> List[ExtractedPage]:
        """Extracts text from PDF data in-memory using PyMuPDF.

        Handles encrypted PDFs, corrupt headers, and scans without text.
        """
        try:
            doc = fitz.open(stream=data, filetype="pdf")
        except Exception as e:
            raise CorruptedDocumentError(f"Could not open PDF '{filename}': {str(e)}")

        if doc.is_encrypted:
            doc.close()
            raise CorruptedDocumentError(f"PDF '{filename}' is password-protected or encrypted.")

        page_count = len(doc)
        if page_count == 0:
            doc.close()
            raise EmptyDocumentError(f"PDF '{filename}' has 0 pages.")

        extracted_pages: List[ExtractedPage] = []
        total_extracted_chars = 0

        for page_idx in range(page_count):
            try:
                page = doc.load_page(page_idx)

                # Use blocks-based extraction to preserve line boundaries between
                # text-frame regions (critical for structured docs like resumes / reports).
                # Each block is a spatial bounding-box group; joining them with "\n"
                # retains the logical paragraph/section breaks that get lost in flat
                # get_text("text") when the PDF renderer merges adjacent columns.
                blocks = page.get_text("blocks")
                # blocks: (x0, y0, x1, y1, text, block_no, block_type)
                # block_type == 0 → text block; 1 → image block
                text_parts = []
                for b in sorted(blocks, key=lambda b: (round(b[1] / 10), b[0])):
                    if b[6] == 0:  # text block
                        text_parts.append(b[4].strip())

                raw_text = "\n".join(t for t in text_parts if t)
                cleaned_text = self.cleaner.clean(raw_text)

                total_extracted_chars += len(cleaned_text)

                extracted_pages.append(
                    ExtractedPage(
                        doc_id=doc_id,
                        source=filename,
                        page=page_idx + 1,  # 1-indexed page numbers
                        text=cleaned_text
                    )
                )
            except Exception as e:
                logger.warning(f"Error reading page {page_idx + 1} of '{filename}': {str(e)}")
                extracted_pages.append(
                    ExtractedPage(
                        doc_id=doc_id,
                        source=filename,
                        page=page_idx + 1,
                        text=""
                    )
                )

        doc.close()

        # Check for empty or scanned PDF
        if total_extracted_chars == 0:
            raise EmptyDocumentError(
                f"PDF '{filename}' contains no extractable digital text. "
                "If this is a scanned document or image-only PDF, optical character recognition (OCR) is required."
            )

        logger.info(f"Successfully extracted {len(extracted_pages)} pages from PDF '{filename}' ({total_extracted_chars} chars).")
        return extracted_pages

    def _extract_text_bytes(self, data: bytes, filename: str, doc_id: str) -> List[ExtractedPage]:
        """Extracts text from plaintext or markdown bytes with encoding fallback."""
        text = ""
        # Try UTF-8 first, fallback to latin-1
        for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
            try:
                text = data.decode(enc)
                break
            except UnicodeDecodeError:
                continue

        cleaned = self.cleaner.clean(text)
        if not cleaned:
            raise EmptyDocumentError(f"Document '{filename}' is empty or contains only whitespace.")

        logger.info(f"Successfully extracted plaintext document '{filename}' ({len(cleaned)} chars).")
        return [
            ExtractedPage(
                doc_id=doc_id,
                source=filename,
                page=1,
                text=cleaned
            )
        ]
