"""PDF upload validation and page-level text extraction using PyMuPDF.

This module is designed to be reusable and independent of Streamlit.
It processes PDFs purely in-memory without creating temporary files on disk
and does not execute any embedded PDF scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
from typing import Any, BinaryIO, Sequence

import pymupdf

from src.config import DEFAULT_MAX_FILE_SIZE_MB, get_max_pdf_size_mb

# Minimum length for PDF header inspection
PDF_MAGIC_BYTES = b"%PDF-"
PDF_HEADER_WINDOW = 1024


class DocumentStatus(str, Enum):
    """Lifecycle and extraction status of an uploaded document."""

    SUCCESS = "success"
    PARTIAL_TEXT = "partial_text"
    NO_TEXT = "no_text"
    EMPTY_DOCUMENT = "empty_document"
    ENCRYPTED = "encrypted"
    CORRUPTED = "corrupted"
    OVERSIZED = "oversized"
    UNSUPPORTED_TYPE = "unsupported_type"


class PDFProcessingError(Exception):
    """Base exception for PDF processing errors."""


class UnsupportedFileTypeError(PDFProcessingError):
    """Raised when an uploaded file is not a PDF."""


class OversizedFileError(PDFProcessingError):
    """Raised when an uploaded file exceeds the configured size limit."""


class EmptyPDFError(PDFProcessingError):
    """Raised when an uploaded file is 0 bytes or contains 0 pages."""


class CorruptPDFError(PDFProcessingError):
    """Raised when a PDF file is corrupted or cannot be parsed."""


class EncryptedPDFError(PDFProcessingError):
    """Raised when a PDF is password-protected or encrypted."""


@dataclass(frozen=True)
class PageRecord:
    """Structured record for a single extracted PDF page.

    Page numbers are 1-based (human readable: page 1, page 2, ...).
    """

    doc_id: str
    filename: str
    page_number: int
    text: str

    @property
    def char_count(self) -> int:
        """Total character count of the extracted text."""
        return len(self.text)

    @property
    def has_text(self) -> bool:
        """True if this page contains non-whitespace extractable text."""
        return bool(self.text.strip())


@dataclass
class ProcessedDocument:
    """Result of processing an uploaded PDF file.

    Maintains document metadata, per-page text records, and extraction health.
    Raw PDF bytes are intentionally released after extraction to conserve memory.
    """

    doc_id: str
    filename: str
    file_size_bytes: int
    total_pages: int = 0
    pages_with_text: int = 0
    pages: list[PageRecord] = field(default_factory=list)
    status: DocumentStatus = DocumentStatus.SUCCESS
    error_message: str | None = None
    warning_message: str | None = None

    @property
    def is_success(self) -> bool:
        """True if the document was parsed and contains at least some text."""
        return self.status in (DocumentStatus.SUCCESS, DocumentStatus.PARTIAL_TEXT)

    @property
    def has_extractable_text(self) -> bool:
        """True if at least one page contained extractable text."""
        return self.pages_with_text > 0


def generate_document_id(raw_bytes: bytes) -> str:
    """Compute a deterministic, stable document identifier based on content hash.

    The same document file will always receive the exact same document ID.
    """
    digest = hashlib.sha256(raw_bytes).hexdigest()[:16]
    return f"doc_{digest}"


def _resolve_filename(source: Any, filename: str | None) -> str:
    """Determine the source filename cleanly."""
    if filename and filename.strip():
        return filename.strip()
    name_attr = getattr(source, "name", None)
    if isinstance(name_attr, str) and name_attr.strip():
        return name_attr.strip()
    return "document.pdf"


def _read_bytes(source: bytes | bytearray | BinaryIO | Any, max_size_bytes: int) -> bytes:
    """Read bytes safely from bytes or a file-like object, checking size limits."""
    # Check size attribute if available before loading large content
    size_attr = getattr(source, "size", None)
    if isinstance(size_attr, int) and size_attr > max_size_bytes:
        max_mb = max_size_bytes / (1024 * 1024)
        actual_mb = size_attr / (1024 * 1024)
        raise OversizedFileError(
            f"File size ({actual_mb:.2f} MB) exceeds maximum allowed size ({max_mb:.0f} MB)."
        )

    if isinstance(source, (bytes, bytearray)):
        raw = bytes(source)
    elif hasattr(source, "read"):
        if hasattr(source, "seek"):
            try:
                source.seek(0)
            except Exception:
                pass
        # Read at most max_size_bytes + 1 to detect oversized files without unbounded memory
        raw = source.read(max_size_bytes + 1)
        if len(raw) > max_size_bytes:
            max_mb = max_size_bytes / (1024 * 1024)
            raise OversizedFileError(
                f"File size exceeds maximum allowed size ({max_mb:.0f} MB)."
            )
    else:
        raise TypeError(
            f"Expected bytes or file-like object, got {type(source).__name__}."
        )

    if len(raw) > max_size_bytes:
        max_mb = max_size_bytes / (1024 * 1024)
        actual_mb = len(raw) / (1024 * 1024)
        raise OversizedFileError(
            f"File size ({actual_mb:.2f} MB) exceeds maximum allowed size ({max_mb:.0f} MB)."
        )

    return raw


def extract_pages(
    source: bytes | bytearray | BinaryIO | Any,
    filename: str | None = None,
    *,
    max_size_bytes: int | None = None,
) -> tuple[str, str, int, list[PageRecord]]:
    """Low-level extractor that opens a PDF in memory and extracts structured page records.

    Raises specific PDFProcessingError subclasses on failure.
    Returns: (doc_id, resolved_filename, total_pages, pages).
    """
    resolved_name = _resolve_filename(source, filename)

    if not resolved_name.lower().endswith(".pdf"):
        raise UnsupportedFileTypeError(
            f"Unsupported file type for '{resolved_name}'. Only .pdf files are accepted."
        )

    limit_bytes = (
        max_size_bytes
        if max_size_bytes is not None
        else get_max_pdf_size_mb() * 1024 * 1024
    )

    raw_bytes = _read_bytes(source, limit_bytes)

    if not raw_bytes:
        raise EmptyPDFError(f"File '{resolved_name}' is empty (0 bytes).")

    # Verify standard PDF magic header in first 1024 bytes
    if PDF_MAGIC_BYTES not in raw_bytes[:PDF_HEADER_WINDOW]:
        raise CorruptPDFError(
            f"File '{resolved_name}' does not appear to be a valid PDF (missing %PDF- header)."
        )

    doc_id = generate_document_id(raw_bytes)

    try:
        pdf_doc = pymupdf.open(stream=raw_bytes, filetype="pdf")
    except Exception as exc:
        raise CorruptPDFError(
            f"Failed to parse '{resolved_name}'. File may be corrupted or malformed: {exc}"
        ) from exc

    try:
        if pdf_doc.is_encrypted or pdf_doc.needs_pass:
            raise EncryptedPDFError(
                f"File '{resolved_name}' is password-protected or encrypted. Please upload an unencrypted PDF."
            )

        total_pages = len(pdf_doc)
        if total_pages == 0:
            raise EmptyPDFError(f"PDF '{resolved_name}' contains no pages.")

        pages: list[PageRecord] = []
        for idx in range(total_pages):
            page_obj = pdf_doc[idx]
            # Extract plain text; PyMuPDF parses glyphs and font encodings safely without code execution
            raw_text = page_obj.get_text() or ""
            clean_text = raw_text.strip()
            pages.append(
                PageRecord(
                    doc_id=doc_id,
                    filename=resolved_name,
                    page_number=idx + 1,  # 1-based page numbering
                    text=clean_text,
                )
            )

        return doc_id, resolved_name, total_pages, pages
    finally:
        pdf_doc.close()


def process_pdf(
    source: bytes | bytearray | BinaryIO | Any,
    filename: str | None = None,
    *,
    max_size_bytes: int | None = None,
) -> ProcessedDocument:
    """Process a single PDF source safely, catching all expected errors into status fields.

    This function is guaranteed not to raise an uncaught exception for user-supplied data.
    """
    resolved_name = _resolve_filename(source, filename)
    file_size = getattr(source, "size", None)
    if not isinstance(file_size, int):
        file_size = len(source) if isinstance(source, (bytes, bytearray)) else 0

    limit_bytes = (
        max_size_bytes
        if max_size_bytes is not None
        else get_max_pdf_size_mb() * 1024 * 1024
    )

    try:
        doc_id, resolved_name, total_pages, pages = extract_pages(
            source, resolved_name, max_size_bytes=limit_bytes
        )
    except UnsupportedFileTypeError as exc:
        return ProcessedDocument(
            doc_id="doc_unknown",
            filename=resolved_name,
            file_size_bytes=file_size,
            status=DocumentStatus.UNSUPPORTED_TYPE,
            error_message=str(exc),
        )
    except OversizedFileError as exc:
        return ProcessedDocument(
            doc_id="doc_unknown",
            filename=resolved_name,
            file_size_bytes=file_size,
            status=DocumentStatus.OVERSIZED,
            error_message=str(exc),
        )
    except EmptyPDFError as exc:
        return ProcessedDocument(
            doc_id="doc_empty",
            filename=resolved_name,
            file_size_bytes=file_size,
            status=DocumentStatus.EMPTY_DOCUMENT,
            error_message=str(exc),
        )
    except CorruptPDFError as exc:
        return ProcessedDocument(
            doc_id="doc_corrupt",
            filename=resolved_name,
            file_size_bytes=file_size,
            status=DocumentStatus.CORRUPTED,
            error_message=str(exc),
        )
    except EncryptedPDFError as exc:
        return ProcessedDocument(
            doc_id="doc_encrypted",
            filename=resolved_name,
            file_size_bytes=file_size,
            status=DocumentStatus.ENCRYPTED,
            error_message=str(exc),
        )
    except Exception as exc:
        return ProcessedDocument(
            doc_id="doc_failed",
            filename=resolved_name,
            file_size_bytes=file_size,
            status=DocumentStatus.CORRUPTED,
            error_message=f"Unexpected error processing '{resolved_name}': {exc}",
        )

    pages_with_text = sum(1 for p in pages if p.has_text)

    # Scanned PDF and layout health checks
    if pages_with_text == 0:
        status = DocumentStatus.NO_TEXT
        warning = (
            f"No extractable text was found on any of the {total_pages} page(s). "
            "This document appears to be an image-only scanned PDF or contains only raster graphics. "
            "Scanned-document OCR is not supported in Phase 2, so this file cannot be used for text analysis."
        )
    elif pages_with_text < total_pages:
        status = DocumentStatus.PARTIAL_TEXT
        empty_count = total_pages - pages_with_text
        warning = (
            f"Extractable text was found on {pages_with_text} of {total_pages} page(s). "
            f"{empty_count} page(s) contain no extractable text (e.g. diagrams, full-page figures, or scanned inserts)."
        )
    else:
        status = DocumentStatus.SUCCESS
        warning = None

    return ProcessedDocument(
        doc_id=doc_id,
        filename=resolved_name,
        file_size_bytes=file_size if file_size > 0 else 0,
        total_pages=total_pages,
        pages_with_text=pages_with_text,
        pages=pages,
        status=status,
        warning_message=warning,
    )


def process_multiple_pdfs(
    sources: Sequence[Any],
    *,
    max_size_bytes: int | None = None,
) -> list[ProcessedDocument]:
    """Process a list of uploaded PDF files or streams.

    Each document is processed independently, ensuring an error in one file
    never blocks or crashes processing of the remaining files.
    """
    results: list[ProcessedDocument] = []
    for source in sources:
        results.append(process_pdf(source, max_size_bytes=max_size_bytes))
    return results
