# PaperTrail

PaperTrail is a hackathon project: **an AI researcher that audits its own claims**.

The planned product lets you upload research papers, ask questions across them, inspect supporting evidence, spot contradictory findings, and flag claims that lack enough evidence.

This repository currently implements:
- **Phase 1:** Nebius Token Factory client and connection verification.
- **Phase 2:** PDF upload, validation, and page-level text extraction using PyMuPDF.
- **Phase 3A:** Deterministic passage chunking and local BM25 keyword evidence retrieval.

> [!NOTE]
> **No API Key or Paid Services Needed for Phase 2 & Phase 3A!**  
> PDF upload, parsing, page-level text extraction, passage chunking, BM25 keyword search, and UI previews run **100% locally** on your machine. All unit tests run against in-memory synthetic documents and mocked interfaces without making external network calls.

---

## Current Features (Phases 1, 2, & 3A)

- **Multiple PDF Upload:** Upload one or more research papers in PDF format.
- **In-Memory Text Extraction:** Uses PyMuPDF (`pymupdf`) to safely parse PDFs in memory without writing temporary files to disk.
- **Page-Level Structured Records:** Text is extracted page-by-page with human-readable 1-based page numbering (`Page 1`, `Page 2`, ...).
- **Document Metadata & Stable Identifiers:** Generates a deterministic content-based document ID (`doc_<sha256>`) and preserves source filenames.
- **Scanned Document Detection:** Distinguishes between text papers and scanned/image-only PDFs. Never silently claims an image-only PDF was parsed successfully.
- **Deterministic Passage Chunking:** Splits long pages into bounded passages (~800–1200 characters) with modest overlap, strictly preserving original document ID, filename, and page number. Text from different pages is never mixed.
- **Local BM25 Evidence Retrieval:** Ranks matching paper passages using the standard Okapi BM25 algorithm (Robertson-Spärck Jones IDF and length normalization) with zero external dependencies.
- **Transparent Relevance Scores:** Displays clear BM25 scores and rank for each retrieved passage; non-matching passages (score 0) are cleanly omitted.
- **Interactive UI:** Streamlit interface provides multi-file upload, document health overview, page text accordions, and a dedicated research search box.
- **Nebius Connection Test:** Phase 1 connection test is preserved in a collapsible panel to test API keys whenever you obtain one.

---

## How Local Retrieval Works (Phase 3A)

1. **Page-Bounded Chunking:** Each page is processed independently. If page text exceeds the target size (~1000 characters), it is split along natural paragraph, sentence, or word boundaries with a 150-character overlap. Each passage receives a unique identifier (e.g. `doc_123_p2_c1`).
2. **Case & Punctuation Normalization:** Queries and passage texts are normalized to lower-case word tokens, stripping punctuation noise.
3. **BM25 Scoring Formula:** Uses standard Okapi BM25 parameters ($k_1 = 1.5, b = 0.75$) with smoothed positive IDF. Passages matching multiple query terms or distinctive terms are ranked highest.
4. **No Fabricated Answers:** If no passages contain the searched terms, an explicit "no matching evidence" notification is displayed. The system never fabricates citations or generates hallucinated answers.

---

## Known Retrieval & Extraction Limitations

1. **Lexical Keyword Search vs Semantic Understanding:**  
   Phase 3A implements lexical BM25 retrieval. It searches for specific query terms and morphological variations in the text. It does **not** perform semantic embedding search or conceptual matching. If a paper discusses "machine learning" and you search for "deep learning", BM25 will not match unless the term explicitly appears. Semantic embeddings and neural RAG belong to later phases.
2. **Scanned Documents & OCR:**  
   Extraction targets digital text layers encoded in the PDF. Image-only scanned PDFs require Optical Character Recognition (OCR), which is not yet implemented. Scanned files are flagged with a warning and excluded from search until OCR is added.
3. **Multi-Column & Tabular Layouts:**  
   Two-column academic papers are extracted in linear stream order. Complex LaTeX tables, mathematical equations, and vector graphics may lose grid alignment.
4. **Encrypted Documents:**  
   Password-protected or encrypted PDFs are safely rejected with a clear notification.

---

## Prerequisites

- **Python 3.10+** (tested on Python 3.12.1 on Windows).  
  *Tip for Windows:* Use the launcher `py -3` rather than `python` if `python` opens the Microsoft Store.
- **Git** for version control.
- *(Optional)* A Nebius Token Factory API key if you want to test model connectivity in the Phase 1 panel.

---

## Setup Instructions (Windows PowerShell)

### 1. Clone or open the repository

```powershell
cd C:\Users\pune\PaperTrail
```

### 2. Create and activate a virtual environment

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If script execution is disabled on your system, enable it for this session:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

When active, your prompt will show `(.venv)`.

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Optional: Configure environment variables

Copy `.env.example` to `.env`:

```powershell
Copy-Item .env.example .env
```

| Variable | Description | Default |
| --- | --- | --- |
| `MAX_PDF_SIZE_MB` | Maximum allowed size per uploaded PDF in megabytes | `20` |
| `NEBIUS_API_KEY` | *(Optional)* Nebius Token Factory API key | Unset |
| `NEBIUS_MODEL_ID` | *(Optional)* Model ID for Token Factory | `nvidia/Nemotron-3_5-Lightning` |
| `NEBIUS_BASE_URL` | *(Optional)* Token Factory endpoint | `https://api.tokenfactory.nebius.com/v1/` |

---

## Running the Application

Launch the Streamlit app:

```powershell
python -m streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

### How to use:
1. **Upload & Process Papers:** Under **PDF Upload and Text Extraction**, select one or more PDF files and click **Process PDFs**.
2. **Search for Evidence:** Under **Research Search (Local Evidence Retrieval)**, enter your query or research keywords (e.g., `quantum error threshold`) and click **Search Papers**.
3. **Inspect Evidence:** Ranked matching passages appear with their source filename, exact page number, and BM25 score.
4. **Inspect Raw Pages:** Scroll down to **Document Overview & Page Previews** to expand page-by-page text.

---

## Running the Tests

The test suite runs 100% locally with synthetic PDFs and mocked API clients. No network calls or API keys are required:

```powershell
python -m pytest
```

To run with verbose output:
```powershell
python -m pytest -v
```

---

## Keep Secrets Safe

- Store secrets only in `.env`.
- Never commit `.env` or PDF research files containing private data.
- `.gitignore` is pre-configured to ignore `.env`, virtual environments, and cache directories.

---

## Project Structure

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit UI (multi-PDF uploader, page previews, BM25 research search, connection check) |
| `src/config.py` | Environment configuration and safety limits |
| `src/pdf_processor.py` | PyMuPDF extraction, validation, and structured page records |
| `src/retrieval.py` | Deterministic passage chunking and pure Python BM25 keyword retrieval |
| `src/nebius_client.py` | OpenAI SDK client for Nebius Token Factory |
| `tests/test_retrieval.py` | Unit tests for passage chunking, page attribution, and BM25 ranking |
| `tests/test_pdf_processor.py` | Unit tests for PDF extraction, page numbering, safety limits, and errors |
| `tests/test_config.py` | Tests for configuration loader and environment overrides |
| `tests/test_nebius_client.py` | Mocked tests for Nebius API responses and error handling |
| `.env.example` | Template environment file |
| `requirements.txt` | Project dependencies (`streamlit`, `pymupdf`, `openai`, `pytest`, `python-dotenv`) |

---

## Roadmap

- [x] **Phase 1:** Project setup, configuration, Nebius client, mocked tests.
- [x] **Phase 2:** PDF upload, PyMuPDF page-level text extraction, metadata, page previews.
- [x] **Phase 3A:** Deterministic passage chunking and local BM25 keyword evidence retrieval.
- [ ] **Phase 3B:** Vector embeddings and semantic retrieval.
- [ ] **Phase 4:** Cross-document RAG and claim extraction.
- [ ] **Phase 5:** Evidence auditing and contradiction detection.
