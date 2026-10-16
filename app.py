"""Streamlit UI for PaperTrail.

Phase 1: Nebius Token Factory connection test.
Phase 2: PDF upload, validation, and page-level text extraction with PyMuPDF.
Phase 3A: Local keyword search and evidence retrieval (BM25).
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
from src.retrieval import SearchResult, search_documents

st.set_page_config(page_title="PaperTrail", page_icon="📄", layout="wide")

st.title("📄 PaperTrail")
st.write(
    "**An AI researcher that audits its own claims.** Upload research papers, extract text page-by-page, "
    "and retrieve relevant evidence passages with precise source citations. In later phases, "
    "automated claim auditing and contradiction detection will be enabled."
)

# ---------------------------------------------------------------------------
# Phase 1: Nebius Connection Test (Preserved)
# ---------------------------------------------------------------------------
with st.expander("🔌 Nebius API Connection Test (Phase 1)", expanded=False):
    st.write(
        "Verify that your Nebius Token Factory API key can reach an NVIDIA open-source model. "
        "*(Note: An API key is not required for Phase 2 text extraction or Phase 3A local search.)*"
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

# Initialize session state for persistent document store and search results
if "processed_documents" not in st.session_state:
    st.session_state["processed_documents"] = []
if "search_results" not in st.session_state:
    st.session_state["search_results"] = None
if "last_search_query" not in st.session_state:
    st.session_state["last_search_query"] = ""

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
            st.session_state["search_results"] = None
            st.session_state["last_search_query"] = ""
            st.rerun()

if process_clicked and uploaded_files:
    with st.spinner("Processing uploaded PDFs and extracting text..."):
        max_bytes = max_mb * 1024 * 1024
        processed_results = process_multiple_pdfs(uploaded_files, max_size_bytes=max_bytes)
        st.session_state["processed_documents"] = processed_results
        # Invalidate old search results when new documents are processed
        st.session_state["search_results"] = None
        st.session_state["last_search_query"] = ""
    st.success(f"Processed {len(processed_results)} document(s).")

# ---------------------------------------------------------------------------
# Phase 3A: Research Search (Local Evidence Retrieval)
# ---------------------------------------------------------------------------
st.divider()
st.subheader("🔎 Research Search (Local Evidence Retrieval)")
st.caption(
    "Search extracted paper passages for evidence using local BM25 keyword retrieval. "
    "Passages are ranked by lexical term relevance with exact page citations. "
    "*(Deterministic keyword retrieval only; this is not AI-generated text or semantic embeddings.)*"
)

docs: list[ProcessedDocument] = st.session_state["processed_documents"]

if not docs:
    st.info("ℹ️ Upload and process at least one research paper above to enable evidence search.")
else:
    has_text_docs = any(d.has_extractable_text for d in docs)
    if not has_text_docs:
        st.warning(
            "⚠️ None of the processed documents contain extractable text (e.g. scanned or image-only PDFs). "
            "Evidence search requires documents with digital text."
        )
    else:
        search_col1, search_col2 = st.columns([3, 1])
        with search_col1:
            query = st.text_input(
                "Enter research question or keywords:",
                placeholder="e.g. quantum key distribution error threshold",
                key="evidence_search_input",
            )
        with search_col2:
            top_k_val = st.number_input(
                "Max results:",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
                key="evidence_top_k",
            )

        search_clicked = st.button("Search Papers", type="primary", disabled=not bool(query.strip()))

        if search_clicked and query.strip():
            with st.spinner("Searching paper passages with BM25..."):
                results = search_documents(query.strip(), docs, top_k=int(top_k_val))
                st.session_state["search_results"] = results
                st.session_state["last_search_query"] = query.strip()

        # Display search results if available
        if st.session_state["last_search_query"]:
            curr_results: list[SearchResult] | None = st.session_state["search_results"]
            if curr_results is not None:
                if not curr_results:
                    st.warning(
                        f"No matching evidence passages found for **'{st.session_state['last_search_query']}'**. "
                        "Try broader search terms or words that appear directly in the papers."
                    )
                else:
                    st.success(
                        f"Found **{len(curr_results)}** relevant evidence passage(s) for query: "
                        f"**'{st.session_state['last_search_query']}'**"
                    )

                    for item in curr_results:
                        with st.container():
                            st.markdown(
                                f"#### Rank #{item.rank} — `{item.filename}` (Page {item.page_number})"
                            )
                            st.caption(
                                f"**BM25 Relevance Score:** `{item.score:.4f}` | "
                                f"**Document ID:** `{item.doc_id}` | "
                                f"**Passage ID:** `{item.passage_id}`\n\n"
                                "*(BM25 Score Note: Higher scores reflect higher term frequency and uniqueness across passages. Non-matching passages with score 0 are omitted.)*"
                            )
                            st.text_area(
                                label=f"Passage ({item.filename} — Page {item.page_number})",
                                value=item.text,
                                height=160,
                                disabled=True,
                                key=f"res_{item.passage_id}_{item.rank}",
                            )
                            st.write("")

# ---------------------------------------------------------------------------
# Extracted Documents & Page Previews (Phase 2 Overview)
# ---------------------------------------------------------------------------
if docs:
    st.divider()
    st.subheader("📋 Document Overview & Page Previews")

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
