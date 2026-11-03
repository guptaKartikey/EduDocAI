"""Unit and integration tests for document ingestion and text cleaning."""

import io
import pytest
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

from src.ingestion.document_loader import DocumentLoader, ExtractedPage
from src.ingestion.text_cleaner import TextCleaner
from src.utils.validators import (
    UnsupportedFormatError,
    EmptyDocumentError,
    CorruptedDocumentError,
    DocumentValidationError
)


def create_mock_pdf_bytes(pages_text: list[str]) -> bytes:
    """Helper to generate in-memory PDF bytes with text on multiple pages."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    for text in pages_text:
        c.drawString(72, 700, text)
        c.showPage()
    c.save()
    return buf.getvalue()


class TestTextCleaner:
    def test_clean_normalizes_whitespace_and_unicode(self):
        raw = "Hello   world!\r\nThis is\u00a0a\n\n\n\ntest.\x00"
        cleaned = TextCleaner.clean(raw)
        assert "Hello world!" in cleaned
        assert "\u00a0" not in cleaned
        assert "\x00" not in cleaned
        assert "\n\n\n" not in cleaned

    def test_clean_hyphenated_linebreaks(self):
        raw = "This is a trans-\nformer architecture."
        cleaned = TextCleaner.clean(raw)
        assert "transformer" in cleaned

    def test_clean_query(self):
        q = "   What   is   EduDocAI?   "
        assert TextCleaner.clean_query(q) == "What is EduDocAI?"


class TestDocumentLoader:
    def setup_method(self):
        self.loader = DocumentLoader()

    def test_pdf_extraction_and_metadata_preservation(self):
        pages_content = ["Page 1 content about RAG.", "Page 2 content about FAISS."]
        pdf_bytes = create_mock_pdf_bytes(pages_content)

        pages = self.loader.load_from_bytes(pdf_bytes, "test_doc.pdf")
        assert len(pages) == 2

        # Verify metadata
        assert pages[0].source == "test_doc.pdf"
        assert pages[0].page == 1
        assert "Page 1 content about RAG" in pages[0].text
        assert pages[0].doc_id is not None

        assert pages[1].source == "test_doc.pdf"
        assert pages[1].page == 2
        assert "Page 2 content about FAISS" in pages[1].text
        assert pages[1].doc_id == pages[0].doc_id

    def test_txt_extraction(self):
        text_content = b"Sample policy document for testing.\nAll employees must read."
        pages = self.loader.load_from_bytes(text_content, "policy.txt")
        assert len(pages) == 1
        assert pages[0].page == 1
        assert pages[0].source == "policy.txt"
        assert "Sample policy document" in pages[0].text

    def test_empty_file_raises_error(self):
        with pytest.raises(EmptyDocumentError):
            self.loader.load_from_bytes(b"", "empty.txt")

    def test_empty_text_content_raises_error(self):
        with pytest.raises(EmptyDocumentError):
            self.loader.load_from_bytes(b"   \n\t  \r\n", "spaces.txt")

    def test_corrupted_pdf_raises_error(self):
        corrupt_bytes = b"%PDF-1.4 completely invalid corrupt binary stream"
        with pytest.raises(CorruptedDocumentError):
            self.loader.load_from_bytes(corrupt_bytes, "corrupt.pdf")

    def test_unsupported_file_extension(self):
        with pytest.raises(UnsupportedFormatError):
            self.loader.load_from_bytes(b"some content", "spreadsheet.xlsx")

    def test_empty_pdf_no_text_ocr_warning(self):
        # Create PDF with blank pages (simulates scanned image PDF without OCR)
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=letter)
        c.showPage()
        c.save()
        with pytest.raises(EmptyDocumentError) as exc_info:
            self.loader.load_from_bytes(buf.getvalue(), "scanned.pdf")
        assert "OCR" in str(exc_info.value)
