# PaperTrail

PaperTrail is a hackathon project: **an AI researcher that audits its own claims**.

The planned product lets you upload research papers, ask questions across them, inspect supporting evidence, spot contradictory findings, and flag claims that lack enough evidence.

This repository currently implements **Phase 1** (Nebius Token Factory client and connection verification) and **Phase 2** (PDF upload, validation, and page-level text extraction using PyMuPDF).

> [!NOTE]
> **No API Key is Needed for Phase 2!**  
> PDF upload, parsing, page-level text extraction, and UI previews run **100% locally** on your machine using PyMuPDF. All unit tests run against in-memory synthetic documents and mocked interfaces without making external network calls.

---

## Current Features (Phase 1 & 2)

- **Multiple PDF Upload:** Upload one or more research papers in PDF format.
- **In-Memory Text Extraction:** Uses PyMuPDF (`pymupdf`) to safely parse PDFs in memory without writing temporary files to disk.
- **Page-Level Structured Records:** Text is extracted page-by-page with human-readable 1-based page numbering (`Page 1`, `Page 2`, ...).
- **Document Metadata & Stable Identifiers:** Generates a deterministic content-based document ID (`doc_<sha256>`) and preserves source filenames.
- **Scanned Document Detection:** Distinguishes between text papers and scanned/image-only PDFs. Never silently claims an image-only PDF was parsed successfully.
- **Resource Safety & Limits:** Rejects non-PDF files, malformed documents, encrypted/password-protected PDFs, and files exceeding the configurable 20 MB size limit.
- **Interactive UI Preview:** Streamlit interface displays per-document extraction status, total pages, pages with text, and collapsible page-by-page text inspection.
- **Nebius Connection Test:** Phase 1 connection test is preserved in a collapsible panel to test API keys whenever you obtain one.

---

## Known Extraction Limitations

1. **Scanned Documents & OCR:**  
   Phase 2 extracts digital text layers encoded in the PDF. **Scanned image PDFs do not contain digital text layers.** Optical Character Recognition (OCR) is not implemented in this phase. The application explicitly flags scanned documents with a warning rather than failing silently or pretending text was found.
2. **Multi-Column Layouts:**  
   Standard academic two-column papers are extracted linearly. While reading order is generally preserved by PyMuPDF, complex multi-column layouts with inline callouts may occasionally interleave text sections.
3. **Mathematical Formulas and Tables:**  
   LaTeX formulas, special mathematical symbols, and complex multi-row tables are extracted as raw unicode text and may lose grid formatting.
4. **Encrypted Documents:**  
   Password-protected or encrypted PDFs cannot be opened without credentials and are rejected with a clear notification.

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

Then open the local URL printed in the terminal (usually `http://localhost:8501`).

### How to use:
1. Under **PDF Upload and Text Extraction**, drag and drop or browse for PDF files.
2. Click **Process PDFs**.
3. View the document overview cards (Total Documents, Total Pages, Pages with Text).
4. Expand any page accordion (e.g., `📄 paper.pdf — Page 1`) to inspect the extracted text.
5. If you have a Nebius API key, expand the **Nebius API Connection Test** panel at the top to verify model connectivity.

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
| `app.py` | Streamlit UI (multi-PDF uploader, page text preview, connection check) |
| `src/config.py` | Environment configuration and safety limits |
| `src/pdf_processor.py` | PyMuPDF extraction, validation, and structured page records |
| `src/nebius_client.py` | OpenAI SDK client for Nebius Token Factory |
| `tests/test_pdf_processor.py` | Unit tests for extraction, page numbering, safety limits, and errors |
| `tests/test_config.py` | Tests for configuration loader and environment overrides |
| `tests/test_nebius_client.py` | Mocked tests for Nebius API responses and error handling |
| `.env.example` | Template environment file |
| `requirements.txt` | Project dependencies (`streamlit`, `pymupdf`, `openai`, `pytest`, `python-dotenv`) |

---

## Roadmap

- [x] **Phase 1:** Project setup, configuration, Nebius client, mocked tests.
- [x] **Phase 2:** PDF upload, PyMuPDF page-level text extraction, metadata, page previews.
- [ ] **Phase 3:** Chunking, embeddings, and vector indexing.
- [ ] **Phase 4:** Cross-document RAG and claim extraction.
- [ ] **Phase 5:** Evidence auditing and contradiction detection.
