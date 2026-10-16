"""Streamlit UI for PaperTrail.

Phase 1: Nebius Token Factory connection test.
Phase 2: PDF upload, validation, and page-level text extraction with PyMuPDF.
Phase 3A: Local keyword search and evidence retrieval (BM25).
Phase 3B: Evidence-grounded answer generation with citation verification.
"""

from __future__ import annotations

import streamlit as st

from src.answer_generator import AnswerResult, Citation, generate_grounded_answer
from src.config import ConfigError, get_max_pdf_size_mb, load_settings
from src.nebius_client import NebiusClient, NebiusError
from src.pdf_processor import (
    DocumentStatus,
    ProcessedDocument,
    process_multiple_pdfs,
)
from src.retrieval import SearchResult, chunk_documents, search_documents, search_passages

st.set_page_config(page_title="PaperTrail", page_icon="📄", layout="wide")

st.title("📄 PaperTrail")
st.write(
    "**An AI researcher that audits its own claims.** Upload research papers, extract text page-by-page, "
    "retrieve relevant evidence passages, and generate answers with verified, traceable citations. "
    "*(Contradiction detection and automated claim auditing will be enabled in future phases.)*"
)

# ---------------------------------------------------------------------------
# Phase 1: Nebius Connection Test (Preserved)
# ---------------------------------------------------------------------------
with st.expander("🔌 Nebius API Connection Test (Phase 1)", expanded=False):
    st.write(
        "Verify that your Nebius Token Factory API key can reach an NVIDIA open-source model. "
        "*(Note: An API key is not required for Phase 2 text extraction, Phase 3A search, or Phase 3B demo mode.)*"
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

# Initialize session state for persistent document store, search, and Q&A
if "processed_documents" not in st.session_state:
    st.session_state["processed_documents"] = []
if "search_results" not in st.session_state:
    st.session_state["search_results"] = None
if "last_search_query" not in st.session_state:
    st.session_state["last_search_query"] = ""
if "qa_result" not in st.session_state:
    st.session_state["qa_result"] = None
if "last_qa_query" not in st.session_state:
    st.session_state["last_qa_query"] = ""

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
            st.session_state["qa_result"] = None
            st.session_state["last_qa_query"] = ""
            st.rerun()

if process_clicked and uploaded_files:
    with st.spinner("Processing uploaded PDFs and extracting text..."):
        max_bytes = max_mb * 1024 * 1024
        processed_results = process_multiple_pdfs(uploaded_files, max_size_bytes=max_bytes)
        st.session_state["processed_documents"] = processed_results
        st.session_state["search_results"] = None
        st.session_state["last_search_query"] = ""
        st.session_state["qa_result"] = None
        st.session_state["last_qa_query"] = ""
    st.success(f"Processed {len(processed_results)} document(s).")

docs: list[ProcessedDocument] = st.session_state["processed_documents"]
has_text_docs = any(d.has_extractable_text for d in docs) if docs else False

# ---------------------------------------------------------------------------
# Phase 3B: Evidence-Grounded Research Question Answering
# ---------------------------------------------------------------------------
st.divider()
st.subheader("🤖 Research Question Answering (Evidence-Grounded)")
st.caption(
    "Ask a research question across uploaded papers. Passages are retrieved via local BM25, "
    "and answers are generated with strictly validated citations. Passages that were not retrieved "
    "cannot be cited."
)

if not docs:
    st.info("ℹ️ Upload and process at least one research paper above to enable Q&A.")
elif not has_text_docs:
    st.warning("⚠️ Processed documents contain no extractable text. Q&A requires text-based papers.")
else:
    qa_mode_col1, qa_mode_col2 = st.columns([2, 1])
    with qa_mode_col1:
        engine_mode = st.radio(
            "Inference Engine:",
            [
                "Offline Demo Mode (Local Mock — No API key needed)",
                "Live Nemotron (Nebius Token Factory API)",
            ],
            index=0,
            horizontal=True,
            help="Demo mode emulates grounded answer generation locally without API calls or credits.",
        )
    is_mock_mode = "Offline Demo Mode" in engine_mode

    qa_col_q, qa_col_k = st.columns([3, 1])
    with qa_col_q:
        qa_input = st.text_input(
            "Enter research question:",
            placeholder="e.g. What are the main findings regarding surface code error thresholds?",
            key="qa_question_input",
        )
    with qa_col_k:
        qa_top_k = st.number_input(
            "Passages to inspect:",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            key="qa_passages_top_k",
        )

    qa_submit = st.button("Generate Answer", type="primary", disabled=not bool(qa_input.strip()))

    if qa_submit and qa_input.strip():
        with st.spinner("Retrieving evidence passages and generating grounded answer..."):
            all_passages = chunk_documents(docs, target_size=1000, overlap=150)
            retrieval_matches = search_passages(qa_input.strip(), all_passages, top_k=int(qa_top_k))
            candidate_passages = [m.passage for m in retrieval_matches]

            live_client = None
            should_run_mock = is_mock_mode

            if not is_mock_mode:
                try:
                    settings = load_settings()
                    live_client = NebiusClient(settings)
                except ConfigError as exc:
                    st.error(
                        f"Cannot run Live Nemotron: {exc}\n\n"
                        "Switching this request to Offline Demo Mode so you can inspect how answers are formatted."
                    )
                    should_run_mock = True

            answer_res = generate_grounded_answer(
                question=qa_input.strip(),
                passages=candidate_passages,
                client=live_client,
                is_mock=should_run_mock,
            )
            st.session_state["qa_result"] = answer_res
            st.session_state["last_qa_query"] = qa_input.strip()

    # Display Q&A results
    qa_res: AnswerResult | None = st.session_state.get("qa_result")
    if qa_res:
        st.write("")
        # Clear engine identification badge
        if qa_res.is_mock:
            st.warning(
                "⚠️ **DEMO MODE:** Generated locally using mock model emulation because live Nebius API "
                "access is offline. No network calls or credits were consumed."
            )
        else:
            st.success(
                f"🟢 **LIVE INFERENCE:** Generated by `{qa_res.model_id}` via Nebius Token Factory."
            )

        if qa_res.has_error:
            st.error(f"❌ **Model Error:** {qa_res.error_message}")
        elif qa_res.insufficient_evidence:
            st.warning(f"⚠️ **Insufficient Evidence:** {qa_res.answer}")
            if qa_res.limitations:
                st.info(f"**Evidence Gaps:** {qa_res.limitations}")
        else:
            st.markdown("#### Answer")
            st.write(qa_res.answer)

            if qa_res.limitations:
                st.info(f"ℹ️ **Limitations & Evidentiary Gaps:** {qa_res.limitations}")

            st.markdown("#### Verified Citations")
            if not qa_res.citations:
                st.caption("No specific passages were cited in this response.")
            else:
                for cit in qa_res.citations:
                    st.markdown(
                        f"📌 **[{cit.passage_id}]** `{cit.filename}` (Page {cit.page_number})"
                    )
                    with st.expander(f"View source passage: {cit.filename} (Page {cit.page_number})", expanded=False):
                        st.text(cit.passage_text)

            if qa_res.invalid_citation_ids:
                st.error(
                    f"⚠️ **Unverified Citations Rejected:** The model attempted to cite passage IDs not "
                    f"in the retrieved evidence: `{', '.join(qa_res.invalid_citation_ids)}`. These were rejected."
                )

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

if not docs:
    st.info("ℹ️ Upload and process at least one research paper above to enable evidence search.")
elif not has_text_docs:
    st.warning("⚠️ None of the processed documents contain extractable text. Search requires text-based papers.")
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

    search_clicked = st.button("Search Papers", type="secondary", disabled=not bool(query.strip()))

    if search_clicked and query.strip():
        with st.spinner("Searching paper passages with BM25..."):
            results = search_documents(query.strip(), docs, top_k=int(top_k_val))
            st.session_state["search_results"] = results
            st.session_state["last_search_query"] = query.strip()

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
                st.error(
                    f"❌ **Failed:** {doc.error_message} (Status: `{doc.status.value}`)"
                )

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

            st.write("")
