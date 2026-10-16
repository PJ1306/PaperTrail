"""Unit tests for PDF processing and page-level text extraction.

All tests create synthetic PDFs programmatically in memory. No real papers,
external APIs, or network calls are required.
"""

from __future__ import annotations

import io
import pytest
import pymupdf

from src.pdf_processor import (
    CorruptPDFError,
    DocumentStatus,
    EmptyPDFError,
    EncryptedPDFError,
    OversizedFileError,
    PageRecord,
    ProcessedDocument,
    UnsupportedFileTypeError,
    extract_pages,
    generate_document_id,
    process_multiple_pdfs,
    process_pdf,
)


def _make_pdf(pages: list[str]) -> bytes:
    """Helper to create an in-memory PDF with specified text per page."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((50, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def _make_image_only_pdf(count: int = 1) -> bytes:
    """Helper to create an in-memory PDF containing only raster images."""
    doc = pymupdf.open()
    for _ in range(count):
        page = doc.new_page()
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 50, 50), 0)
        page.insert_image(pymupdf.Rect(50, 50, 100, 100), pixmap=pix)
    data = doc.tobytes()
    doc.close()
    return data


def _make_encrypted_pdf(password: str = "secret") -> bytes:
    """Helper to create an encrypted in-memory PDF."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 72), "Confidential content")
    data = doc.tobytes(
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        user_pw=password,
        owner_pw=password,
    )
    doc.close()
    return data


class TestPDFProcessorSuccess:
    """Test standard valid extraction cases."""

    def test_extract_multiple_pages_success(self) -> None:
        raw_pdf = _make_pdf(["Page 1: Introduction", "Page 2: Methods", "Page 3: Results"])
        result = process_pdf(raw_pdf, filename="paper.pdf")

        assert result.status == DocumentStatus.SUCCESS
        assert result.is_success is True
        assert result.filename == "paper.pdf"
        assert result.total_pages == 3
        assert result.pages_with_text == 3
        assert len(result.pages) == 3
        assert result.warning_message is None
        assert result.error_message is None

        # Verify page numbering and contents
        assert result.pages[0].page_number == 1
        assert "Page 1: Introduction" in result.pages[0].text
        assert result.pages[0].has_text is True

        assert result.pages[1].page_number == 2
        assert "Page 2: Methods" in result.pages[1].text
        assert result.pages[1].has_text is True

        assert result.pages[2].page_number == 3
        assert "Page 3: Results" in result.pages[2].text
        assert result.pages[2].has_text is True

    def test_one_based_page_numbers(self) -> None:
        raw_pdf = _make_pdf(["A", "B", "C", "D"])
        result = process_pdf(raw_pdf, filename="test.pdf")

        numbers = [p.page_number for p in result.pages]
        assert numbers == [1, 2, 3, 4]

    def test_stable_document_id_preservation(self) -> None:
        raw_pdf = _make_pdf(["Stable content test"])

        res1 = process_pdf(raw_pdf, filename="first_run.pdf")
        res2 = process_pdf(raw_pdf, filename="second_run.pdf")

        assert res1.doc_id.startswith("doc_")
        assert res1.doc_id == res2.doc_id
        for page in res1.pages:
            assert page.doc_id == res1.doc_id
            assert page.filename == "first_run.pdf"

    def test_file_like_object_bytesio(self) -> None:
        raw_pdf = _make_pdf(["BytesIO stream content"])
        stream = io.BytesIO(raw_pdf)
        stream.name = "streamed.pdf"

        result = process_pdf(stream)
        assert result.status == DocumentStatus.SUCCESS
        assert result.filename == "streamed.pdf"
        assert result.total_pages == 1
        assert "stream content" in result.pages[0].text


class TestPDFProcessorEdgeCases:
    """Test empty pages, scanned docs, and partial text."""

    def test_empty_pages_in_multipage_pdf(self) -> None:
        raw_pdf = _make_pdf(["Has text", "", "Also has text"])
        result = process_pdf(raw_pdf, filename="sparse.pdf")

        assert result.status == DocumentStatus.PARTIAL_TEXT
        assert result.total_pages == 3
        assert result.pages_with_text == 2
        assert result.is_success is True

        assert result.pages[0].has_text is True
        assert result.pages[1].has_text is False
        assert result.pages[1].text == ""
        assert result.pages[2].has_text is True
        assert result.warning_message is not None
        assert "2 of 3" in result.warning_message

    def test_scanned_image_only_pdf_is_not_treated_as_text_paper(self) -> None:
        raw_pdf = _make_image_only_pdf(count=2)
        result = process_pdf(raw_pdf, filename="scanned.pdf")

        assert result.status == DocumentStatus.NO_TEXT
        assert result.is_success is False
        assert result.has_extractable_text is False
        assert result.total_pages == 2
        assert result.pages_with_text == 0
        assert result.warning_message is not None
        assert "scanned" in result.warning_message.lower()


class TestPDFProcessorErrors:
    """Test safety limits, corruption, encryption, and bad file types."""

    def test_invalid_pdf_bytes_fails_gracefully(self) -> None:
        garbage = b"This is not a real PDF document at all"
        result = process_pdf(garbage, filename="fake.pdf")

        assert result.status == DocumentStatus.CORRUPTED
        assert result.is_success is False
        assert result.error_message is not None
        assert "%PDF-" in result.error_message

    def test_empty_file_bytes(self) -> None:
        result = process_pdf(b"", filename="empty.pdf")

        assert result.status == DocumentStatus.EMPTY_DOCUMENT
        assert result.is_success is False
        assert result.error_message is not None
        assert "empty" in result.error_message.lower()

    def test_oversized_file_rejection(self) -> None:
        raw_pdf = _make_pdf(["Some sample text"])
        # Set max limit smaller than the generated PDF
        result = process_pdf(raw_pdf, filename="huge.pdf", max_size_bytes=50)

        assert result.status == DocumentStatus.OVERSIZED
        assert result.is_success is False
        assert result.error_message is not None
        assert "exceeds maximum allowed size" in result.error_message

    def test_encrypted_pdf_detected(self) -> None:
        raw_pdf = _make_encrypted_pdf(password="secret123")
        result = process_pdf(raw_pdf, filename="encrypted.pdf")

        assert result.status == DocumentStatus.ENCRYPTED
        assert result.is_success is False
        assert result.error_message is not None
        assert "password-protected or encrypted" in result.error_message

    def test_unsupported_file_extension(self) -> None:
        result = process_pdf(b"%PDF-1.4 dummy", filename="notes.docx")

        assert result.status == DocumentStatus.UNSUPPORTED_TYPE
        assert result.is_success is False
        assert result.error_message is not None
        assert "Only .pdf files are accepted" in result.error_message


class TestMultipleDocumentProcessing:
    """Test batch processing with mixed document health."""

    def test_multiple_pdfs_isolated_execution(self) -> None:
        doc1 = _make_pdf(["Good paper text"])
        doc2 = b"Corrupted garbage bytes"
        doc3 = _make_image_only_pdf(1)

        results = process_multiple_pdfs([
            io.BytesIO(doc1),
            io.BytesIO(doc2),
            io.BytesIO(doc3),
        ])

        assert len(results) == 3
        assert results[0].status == DocumentStatus.SUCCESS
        assert results[1].status == DocumentStatus.CORRUPTED
        assert results[2].status == DocumentStatus.NO_TEXT


class TestLowLevelExceptions:
    """Test that extract_pages raises typed exceptions."""

    def test_raises_unsupported_file_type(self) -> None:
        with pytest.raises(UnsupportedFileTypeError):
            extract_pages(b"%PDF-1.5", filename="report.txt")

    def test_raises_empty_pdf_error(self) -> None:
        with pytest.raises(EmptyPDFError):
            extract_pages(b"", filename="zero.pdf")

    def test_raises_oversized_file_error(self) -> None:
        raw = _make_pdf(["text"])
        with pytest.raises(OversizedFileError):
            extract_pages(raw, filename="large.pdf", max_size_bytes=10)

    def test_raises_corrupt_pdf_error(self) -> None:
        with pytest.raises(CorruptPDFError):
            extract_pages(b"not a pdf", filename="bad.pdf")

    def test_raises_encrypted_pdf_error(self) -> None:
        raw = _make_encrypted_pdf()
        with pytest.raises(EncryptedPDFError):
            extract_pages(raw, filename="locked.pdf")
