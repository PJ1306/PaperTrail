"""Evidence-grounded answer generation for PaperTrail (Phase 3B).

Produces traceable research answers with strictly validated citations mapped
to extracted page records. Supports both live Nemotron inference via Nebius
Token Factory and an offline demo/mock mode for local development.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Sequence

from src.nebius_client import NebiusClient, NebiusError
from src.retrieval import PassageRecord

SYSTEM_PROMPT = """You are PaperTrail, an evidence-grounded research assistant.
Your task is to answer the user's research question using ONLY the provided evidence passages.

CRITICAL RULES:
1. Grounding: Answer strictly and only from the supplied passages. Do not use outside knowledge or assumptions.
2. Citation requirement: Whenever you make a factual claim, cite the exact passage ID using brackets, e.g. [doc_abc123_p1_c1].
3. Allowed citations: You must ONLY cite passage IDs that are explicitly present in the provided passages. NEVER invent or extrapolate passage IDs, filenames, or page numbers.
4. Evidence vs. Inference: Clearly distinguish direct evidence from inference or speculation.
5. Limitations & Conflicts: Identify any unsupported aspects, conflicting findings, or evidentiary gaps in the "limitations" field.
6. Insufficient Evidence: If the provided passages do not contain enough facts to answer the question, set "insufficient_evidence": true and state what is missing.

RESPONSE FORMAT:
You must respond with a valid JSON object matching this schema:
{
  "answer": "Clear, objective answer text with inline citations like [doc_id_p1_c1].",
  "cited_passage_ids": ["doc_id_p1_c1"],
  "limitations": "Description of any gaps, conflicts, or unverified claims.",
  "insufficient_evidence": false
}
Do not include any conversational preamble or text outside the JSON object.
"""


@dataclass(frozen=True)
class Citation:
    """A validated source reference linking a claim to its origin."""

    passage_id: str
    doc_id: str
    filename: str
    page_number: int
    passage_text: str


@dataclass
class AnswerResult:
    """Structured response containing the grounded answer and verified citations."""

    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    invalid_citation_ids: list[str] = field(default_factory=list)
    limitations: str = ""
    insufficient_evidence: bool = False
    is_mock: bool = False
    model_id: str = ""
    error_message: str | None = None

    @property
    def has_error(self) -> bool:
        return self.error_message is not None

    @property
    def is_verified(self) -> bool:
        """True if there is an answer, no model error, and no unvalidated citations."""
        return not self.has_error and not self.invalid_citation_ids and bool(self.answer)


def build_evidence_prompt(question: str, passages: Sequence[PassageRecord]) -> str:
    """Format research passages into an evidence block with clear passage IDs."""
    evidence_blocks: list[str] = []
    for p in passages:
        block = (
            f"--- PASSAGE START ---\n"
            f"Passage ID: {p.passage_id}\n"
            f"Source File: {p.filename} (Page {p.page_number})\n"
            f"Content:\n{p.text}\n"
            f"--- PASSAGE END ---"
        )
        evidence_blocks.append(block)

    formatted_evidence = "\n\n".join(evidence_blocks)

    return (
        f"RESEARCH QUESTION:\n{question.strip()}\n\n"
        f"EVIDENCE PASSAGES ({len(passages)} passages available):\n"
        f"{formatted_evidence}\n\n"
        f"Provide your answer in the required JSON format."
    )


def _extract_json_string(text: str) -> str:
    """Extract raw JSON text, stripping markdown code fences if present."""
    clean = text.strip()
    # Remove markdown code blocks like ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Try finding the first { and last }
    first_brace = clean.find("{")
    last_brace = clean.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return clean[first_brace : last_brace + 1]
    return clean


def validate_and_map_citations(
    cited_ids: Sequence[str],
    passage_map: dict[str, PassageRecord],
) -> tuple[list[Citation], list[str]]:
    """Validate cited passage IDs against actual retrieved passages.

    Never permits the model to invent filenames, page numbers, or IDs.
    Returns: (valid_citations, invalid_citation_ids).
    """
    valid_citations: list[Citation] = []
    invalid_ids: list[str] = []
    seen: set[str] = set()

    for raw_id in cited_ids:
        pid = str(raw_id).strip()
        # Strip brackets if model included [doc_id] in the array
        pid = pid.strip("[]")
        if not pid or pid in seen:
            continue
        seen.add(pid)

        if pid in passage_map:
            p = passage_map[pid]
            valid_citations.append(
                Citation(
                    passage_id=p.passage_id,
                    doc_id=p.doc_id,
                    filename=p.filename,
                    page_number=p.page_number,
                    passage_text=p.text,
                )
            )
        else:
            invalid_ids.append(pid)

    return valid_citations, invalid_ids


def parse_model_response(
    raw_response: str,
    question: str,
    passages: Sequence[PassageRecord],
    *,
    is_mock: bool = False,
    model_id: str = "",
) -> AnswerResult:
    """Parse and rigorously validate untrusted model output into an AnswerResult."""
    passage_map = {p.passage_id: p for p in passages}
    json_str = _extract_json_string(raw_response)

    try:
        data = json.loads(json_str)
        if not isinstance(data, dict):
            raise ValueError("Parsed JSON is not an object.")
    except Exception as exc:
        # Fallback: scan for passage IDs in the text and report parsing degradation
        found_ids = re.findall(r"\[(doc_[a-zA-Z0-9_-]+_p\d+_c\d+)\]", raw_response)
        valid_cits, invalid_ids = validate_and_map_citations(found_ids, passage_map)
        return AnswerResult(
            question=question,
            answer=raw_response.strip(),
            citations=valid_cits,
            invalid_citation_ids=invalid_ids,
            limitations="Warning: Model response was not in valid JSON format. Structured verification was degraded.",
            insufficient_evidence=False,
            is_mock=is_mock,
            model_id=model_id,
            error_message=f"Model response could not be parsed as JSON: {exc}",
        )

    answer_text = str(data.get("answer") or "").strip()
    raw_cited_ids = data.get("cited_passage_ids")
    if not isinstance(raw_cited_ids, list):
        raw_cited_ids = []

    # Also capture any inline bracketed passage IDs from the answer text
    inline_ids = re.findall(r"\[(doc_[a-zA-Z0-9_-]+_p\d+_c\d+)\]", answer_text)
    combined_ids = list(raw_cited_ids) + inline_ids

    valid_cits, invalid_ids = validate_and_map_citations(combined_ids, passage_map)

    limitations = str(data.get("limitations") or "").strip()
    if invalid_ids:
        warning = f"Model attempted to cite unverified passage IDs: {', '.join(invalid_ids)}."
        limitations = f"{limitations} ({warning})".strip() if limitations else warning

    insufficient_ev = bool(data.get("insufficient_evidence", False))
    if not answer_text and not insufficient_ev:
        insufficient_ev = True
        answer_text = "Insufficient evidence: The model provided an empty response."

    return AnswerResult(
        question=question,
        answer=answer_text,
        citations=valid_cits,
        invalid_citation_ids=invalid_ids,
        limitations=limitations,
        insufficient_evidence=insufficient_ev,
        is_mock=is_mock,
        model_id=model_id,
    )


def generate_mock_answer(
    question: str,
    passages: Sequence[PassageRecord],
) -> AnswerResult:
    """Generate a clearly labeled mock answer for local testing without network calls."""
    if not passages:
        return AnswerResult(
            question=question,
            answer="Insufficient evidence: No relevant passages were retrieved to answer the question.",
            citations=[],
            insufficient_evidence=True,
            is_mock=True,
            model_id="mock-nemotron-offline",
        )

    top_p = passages[0]
    # Extract first sentence or brief snippet from the top passage
    first_sentence = top_p.text.split(".")[0].strip()
    if not first_sentence:
        first_sentence = top_p.text[:120].strip()

    cited_ids = [top_p.passage_id]
    mock_answer = (
        f"[DEMO MODE — Nebius API access is currently offline]\n\n"
        f"Based on the retrieved evidence from {top_p.filename} (Page {top_p.page_number}), "
        f"{first_sentence} [{top_p.passage_id}].\n\n"
        f"*(Note: This is a local mock emulation demonstrating grounded answer generation "
        f"and citation validation. When live Nebius credentials are configured, NVIDIA Nemotron "
        f"will generate the full synthesis.)*"
    )

    valid_cits = [
        Citation(
            passage_id=top_p.passage_id,
            doc_id=top_p.doc_id,
            filename=top_p.filename,
            page_number=top_p.page_number,
            passage_text=top_p.text,
        )
    ]

    return AnswerResult(
        question=question,
        answer=mock_answer,
        citations=valid_cits,
        invalid_citation_ids=[],
        limitations="Offline demonstration mode: Response generated locally from retrieved passages.",
        insufficient_evidence=False,
        is_mock=True,
        model_id="mock-nemotron-offline",
    )


def generate_grounded_answer(
    question: str,
    passages: Sequence[PassageRecord],
    *,
    client: NebiusClient | None = None,
    is_mock: bool = True,
) -> AnswerResult:
    """Generate an evidence-grounded answer using either live Nemotron or offline mock.

    Guarantees:
    - Never calls the model if passages are empty (returns insufficient evidence).
    - Never leaks API keys or crashes on model errors.
    - All citations are strictly validated against retrieved passage records.
    """
    clean_q = question.strip()
    if not clean_q:
        return AnswerResult(
            question=clean_q,
            answer="Please enter a research question.",
            citations=[],
            insufficient_evidence=True,
            is_mock=is_mock,
        )

    # Rule 1: If no passages retrieved, return insufficient evidence without model call
    if not passages:
        return AnswerResult(
            question=clean_q,
            answer="Insufficient evidence: No matching passages were found across the uploaded documents.",
            citations=[],
            insufficient_evidence=True,
            is_mock=is_mock,
            limitations="No relevant evidence passages were retrieved for this query.",
        )

    # Rule 2: If mock/demo mode is requested or no client provided, use local mock
    if is_mock or client is None:
        return generate_mock_answer(clean_q, passages)

    # Rule 3: Live inference via NebiusClient
    prompt = build_evidence_prompt(clean_q, passages)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    try:
        raw_response = client.generate(messages, max_tokens=1024, temperature=0.0)
    except NebiusError as exc:
        return AnswerResult(
            question=clean_q,
            answer="",
            citations=[],
            limitations="Inference failed.",
            insufficient_evidence=False,
            is_mock=False,
            model_id=client.settings.model_id,
            error_message=str(exc),
        )
    except Exception as exc:
        return AnswerResult(
            question=clean_q,
            answer="",
            citations=[],
            limitations="Inference failed.",
            insufficient_evidence=False,
            is_mock=False,
            model_id=client.settings.model_id,
            error_message=f"Unexpected error during answer generation: {exc}",
        )

    return parse_model_response(
        raw_response=raw_response,
        question=clean_q,
        passages=passages,
        is_mock=False,
        model_id=client.settings.model_id,
    )


# Convenient alias
generate_answer = generate_grounded_answer

