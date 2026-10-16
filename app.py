"""Streamlit UI for PaperTrail.

Phase 1: Nebius Token Factory connection test.
Phase 2: PDF upload, validation, and page-level text extraction with PyMuPDF.
"""

from __future__ import annotations

import streamlit as st

from src.config import ConfigError, get_max_pdf_size_mb, load_settings
from src.nebius_client import NebiusClient, NebiusError
from src.pdf_processor import (
    DocumentStatus,
    ProcessedDocument,
    process_multiple_pdfs,
)

st.set_page_config(page_title="PaperTrail", page_icon="📄", layout="wide")

st.title("📄 PaperTrail")
st.write(
    "**An AI researcher that audits its own claims.** Upload research papers in PDF format, "
    "inspect extracted text page-by-page, and review document metadata. In later phases, "
    "cross-paper question answering, evidence auditing, and contradiction detection will be enabled."
)

# ---------------------------------------------------------------------------
# Phase 1: Nebius Connection Test (Preserved)
# ---------------------------------------------------------------------------
with st.expander("🔌 Nebius API Connection Test (Phase 1)", expanded=False):
    st.write(
        "Verify that your Nebius Token Factory API key can reach an NVIDIA open-source model. "
        "*(Note: An API key is not required for Phase 2 PDF text extraction, which runs 100% locally.)*"
    )
    st.caption(
        "Success is shown only after the model returns text. A missing key, a wrong "
        "model ID, or a network error will appear as a failure."
    )

    if st.button("Test Nebius API connection", type="primary"):
        with st.spinner("Calling the Nebius Token Factory API..."):
            try:
                settings = load_settings()
                reply = NebiusClient(settings).send_test_prompt()
            except ConfigError as exc:
                st.error(str(exc))
            except NebiusError as exc:
                st.error(f"Connection failed. {exc}")
            else:
                st.success("Connection succeeded. The model returned a response.")
                st.write(f"Configured model: `{settings.model_id}`")
                st.write("Model reply:")
                st.code(reply, language=None)

st.divider()

# ---------------------------------------------------------------------------
# Phase 2: PDF Upload and Page-Level Text Extraction
# ---------------------------------------------------------------------------
st.subheader("📚 PDF Upload and Text Extraction")

max_mb = get_max_pdf_size_mb()

st.info(
    f"ℹ️ **Document Safety & Extraction Details:**\n"
    f"- Maximum file size: **{max_mb} MB** per PDF file.\n"
    f"- Documents are parsed safely in memory without writing temporary files to disk.\n"
    f"- Embedded PDF scripts are never executed.\n"
    f"- **Scanned Papers:** This phase extracts digital text layers. Image-only scanned PDFs require OCR, which is not implemented in this phase."
)

# Initialize session state for persistent document store
if "processed_documents" not in st.session_state:
    st.session_state["processed_documents"] = []

uploaded_files = st.file_uploader(
    "Upload research papers (PDF files)",
    type=["pdf"],
    accept_multiple_files=True,
    help=f"Select one or more PDF files up to {max_mb} MB each.",
)

col_btn1, col_btn2 = st.columns([1, 4])
with col_btn1:
    process_clicked = st.button("Process PDFs", type="primary", disabled=not bool(uploaded_files))

with col_btn2:
    if st.session_state["processed_documents"]:
        if st.button("Clear Processed Documents", type="secondary"):
            st.session_state["processed_documents"] = []
            st.rerun()

if process_clicked and uploaded_files:
    with st.spinner("Processing uploaded PDFs and extracting text..."):
        max_bytes = max_mb * 1024 * 1024
        processed_results = process_multiple_pdfs(uploaded_files, max_size_bytes=max_bytes)
        st.session_state["processed_documents"] = processed_results
    st.success(f"Processed {len(processed_results)} document(s).")

# ---------------------------------------------------------------------------
# Extracted Documents & Page Previews
# ---------------------------------------------------------------------------
docs: list[ProcessedDocument] = st.session_state["processed_documents"]

if docs:
    st.subheader("📋 Document Overview")

    total_docs = len(docs)
    total_pages = sum(d.total_pages for d in docs)
    total_pages_with_text = sum(d.pages_with_text for d in docs)

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Documents Processed", total_docs)
    metric_col2.metric("Total Pages", total_pages)
    metric_col3.metric("Pages with Text", total_pages_with_text)

    st.subheader("🔍 Extracted Text by Page")

    for doc_idx, doc in enumerate(docs, start=1):
        with st.container():
            st.markdown(f"### {doc_idx}. {doc.filename}")

            # Status and feedback banners
            if doc.status == DocumentStatus.SUCCESS:
                st.success(
                    f"✓ **Ready:** Extracted text from all {doc.total_pages} page(s). "
                    f"(ID: `{doc.doc_id}` | Total Pages: {doc.total_pages} | Pages with text: {doc.pages_with_text})"
                )
            elif doc.status == DocumentStatus.PARTIAL_TEXT:
                st.warning(
                    f"⚠️ **Partial Text:** {doc.pages_with_text} of {doc.total_pages} page(s) contain extractable text. "
                    f"(ID: `{doc.doc_id}`)\n\n{doc.warning_message}"
                )
            elif doc.status == DocumentStatus.NO_TEXT:
                st.warning(
                    f"⚠️ **No Extractable Text (Scanned Document):** {doc.warning_message} "
                    f"(ID: `{doc.doc_id}` | Total Pages: {doc.total_pages} | Pages with text: 0)"
                )
            else:
                # Fatal errors: corrupted, oversized, encrypted, unsupported type, empty
                st.error(
                    f"❌ **Failed:** {doc.error_message} (Status: `{doc.status.value}`)"
                )

            # Per-page text preview expanders
            if doc.pages:
                st.caption(
                    f"Showing page-level text for **{doc.filename}** (1-based page numbers):"
                )
                for page in doc.pages:
                    preview_title = (
                        f"📄 {doc.filename} — Page {page.page_number} "
                        f"({page.char_count:,} characters)"
                    )
                    with st.expander(preview_title, expanded=False):
                        if page.has_text:
                            st.text_area(
                                label=f"{doc.filename} — Page {page.page_number}",
                                value=page.text,
                                height=200,
                                disabled=True,
                                key=f"preview_{doc.doc_id}_{page.page_number}_{doc_idx}",
                            )
                        else:
                            st.info(
                                f"ℹ️ **Page {page.page_number} has no extractable text.** "
                                "This page may contain a full-page diagram, image, or scanned figure."
                            )

            st.write("")  # Visual spacing between documents
