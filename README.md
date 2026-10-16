# PaperTrail

PaperTrail is a hackathon project: **an AI researcher that audits its own claims**.

The planned product lets you upload research papers, ask questions across them, inspect supporting evidence, spot contradictory findings, and flag claims that lack enough evidence.

This repository currently implements:
- **Phase 1:** Nebius Token Factory client and connection verification.
- **Phase 2:** PDF upload, validation, and page-level text extraction using PyMuPDF.
- **Phase 3A:** Deterministic passage chunking and local BM25 keyword evidence retrieval.
- **Phase 3B:** Evidence-grounded answer generation with strict citation validation and offline demo mode.

> [!NOTE]
> **No API Key or Paid Services Needed for Development or Testing!**  
> PDF upload, parsing, passage chunking, BM25 keyword search, citation validation, and offline Q&A demo mode run **100% locally** on your machine. All unit tests run against in-memory synthetic documents and mocked interfaces without making external network calls.

---

## Current Features (Phases 1, 2, 3A, & 3B)

- **Multiple PDF Upload:** Upload one or more research papers in PDF format.
- **In-Memory Text Extraction:** Uses PyMuPDF (`pymupdf`) to safely parse PDFs in memory without writing temporary files to disk.
- **Page-Level Structured Records:** Text is extracted page-by-page with human-readable 1-based page numbering (`Page 1`, `Page 2`, ...).
- **Document Metadata & Stable Identifiers:** Generates a deterministic content-based document ID (`doc_<sha256>`) and preserves source filenames.
- **Scanned Document Detection:** Distinguishes between text papers and scanned/image-only PDFs. Never silently claims an image-only PDF was parsed successfully.
- **Deterministic Passage Chunking:** Splits long pages into bounded passages (~800–1200 characters) with modest overlap, strictly preserving original document ID, filename, and page number. Text from different pages is never mixed.
- **Local BM25 Evidence Retrieval:** Ranks matching paper passages using the standard Okapi BM25 algorithm (Robertson-Spärck Jones IDF and length normalization) with zero external dependencies.
- **Evidence-Grounded Question Answering (Phase 3B):** Accepts a research question, retrieves candidate passages, and produces an answer strictly grounded in the retrieved sources.
- **Strict Citation Validation:** Every model citation is verified against the actual retrieved passage IDs. The model cannot invent filenames or page numbers because metadata is mapped back from trusted memory records. Unknown citations are rejected.
- **Explicit Insufficient Evidence:** If no relevant passages are retrieved, the system immediately returns an insufficient-evidence result without calling a model.
- **Offline Demo Mode vs. Live Nemotron:** Includes a fully functional offline demo mode that emulates grounded answer generation locally without API keys or credit consumption. When Nebius credentials are configured, the app seamlessly connects to NVIDIA Nemotron (`nvidia/Nemotron-3_5-Lightning`).
- **Interactive UI:** Streamlit interface provides multi-file upload, document health overview, page text accordions, BM25 research search, and evidence-grounded Q&A with clickable source inspection.

---

## How Evidence-Grounded Q&A Works (Phase 3B)

1. **Retrieval First:** When a user enters a research question, the BM25 retrieval engine first identifies the top candidate passages from the processed papers.
2. **Zero-Call Guardrail:** If no matching passages are found, answer generation terminates immediately with an explicit "insufficient evidence" status without calling the model.
3. **Structured Prompting:** The model is instructed to answer strictly using the provided passages, cite exact passage IDs (e.g. `[doc_123_p1_c1]`), distinguish evidence from inference, and identify evidentiary gaps in the `limitations` field.
4. **Untrusted Output Verification:** Model responses are treated as untrusted data. The validator parses the structured JSON and verifies every citation ID against the in-memory passage dictionary:
   - **Valid Citations:** Mapped back to the trusted filename, page number, and text.
   - **Invalid/Hallucinated Citations:** Flagged, rejected, and excluded from verified sources.
5. **Execution Transparency:** Every answer is clearly badged as either **DEMO MODE (Offline Mock)** or **LIVE INFERENCE (NVIDIA Nemotron via Nebius)** so users always know the provenance of the text.

---

## Known Limitations

1. **Nebius Live API Status:**  
   Live Nemotron calls require an active Nebius Token Factory account with an API key set in `NEBIUS_API_KEY`. While account onboarding/billing is pending, the application provides the **Offline Demo Mode** so you can develop, test, and demonstrate the full pipeline without credentials.
2. **Lexical Keyword Search vs Semantic Understanding:**  
   Passage retrieval in Phase 3A/3B uses BM25 lexical keyword matching. It matches explicit terms and morphological variations rather than conceptual abstractions. Dense vector embeddings belong to future phases.
3. **Scanned Documents & OCR:**  
   Only digital text layers in PDFs are extracted. Image-only scanned PDFs require OCR (not yet implemented) and are flagged with a warning.
4. **No Cross-Paper Contradiction Dashboards Yet:**  
   Contradiction detection and automated multi-paper claim auditing belong to Phase 4 and Phase 5.

---

## Prerequisites

- **Python 3.10+** (tested on Python 3.12.1 on Windows).  
  *Tip for Windows:* Use the launcher `py -3` rather than `python` if `python` opens the Microsoft Store.
- **Git** for version control.
- *(Optional)* A Nebius Token Factory API key if you want to test live model connectivity.

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
1. **Upload & Process Papers:** Under **PDF Upload and Text Extraction**, select PDF files and click **Process PDFs**.
2. **Ask Research Questions (Phase 3B):** Scroll to **Research Question Answering**, enter a question, select your engine (Offline Demo Mode or Live Nemotron), and click **Generate Answer**.
3. **Inspect Citations:** Expand any verified citation to read the exact quoted source passage directly from the paper.
4. **Keyword Search (Phase 3A):** Use **Research Search** to inspect BM25 ranked passages across papers.
5. **Page-by-Page Inspection (Phase 2):** Use **Document Overview & Page Previews** to inspect full page text.

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
| `app.py` | Streamlit UI (multi-PDF uploader, page previews, BM25 search, grounded Q&A) |
| `src/config.py` | Environment configuration and safety limits |
| `src/pdf_processor.py` | PyMuPDF extraction, validation, and structured page records |
| `src/retrieval.py` | Deterministic passage chunking and pure Python BM25 keyword retrieval |
| `src/answer_generator.py` | Grounded answer generation, prompt construction, citation validation, and mock mode |
| `src/nebius_client.py` | OpenAI SDK client for Nebius Token Factory with generate() method |
| `tests/test_answer_generator.py` | Offline tests for prompt construction, citations, mock mode, and error handling |
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
- [x] **Phase 3B:** Evidence-grounded answer generation with citation verification and offline demo mode.
- [ ] **Phase 4:** Cross-document claim extraction and synthesis.
- [ ] **Phase 5:** Evidence auditing and contradiction detection.
