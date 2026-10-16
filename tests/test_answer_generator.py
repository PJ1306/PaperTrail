"""Unit tests for evidence-grounded answer generation (Phase 3B).

All tests run completely offline with synthetic PassageRecords and mocked clients.
No network calls or API keys are required.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock
import pytest

from src.answer_generator import (
    AnswerResult,
    Citation,
    build_evidence_prompt,
    generate_answer,
    generate_grounded_answer,
    generate_mock_answer,
    parse_model_response,
    validate_and_map_citations,
)
from src.config import Settings
from src.nebius_client import NebiusClient, NebiusError
from src.retrieval import PassageRecord


def _make_passage(
    passage_id: str,
    filename: str = "paper.pdf",
    page_number: int = 1,
    text: str = "Evidence text.",
) -> PassageRecord:
    return PassageRecord(
        passage_id=passage_id,
        doc_id=f"doc_{filename[:4]}",
        filename=filename,
        page_number=page_number,
        chunk_index=1,
        text=text,
        char_count=len(text),
    )


class TestPromptConstruction:
    """Tests for prompt structure and formatting."""

    def test_build_evidence_prompt_contains_all_metadata(self) -> None:
        p1 = _make_passage("doc1_p1_c1", filename="quantum.pdf", page_number=2, text="Entanglement is non-local.")
        p2 = _make_passage("doc2_p5_c2", filename="relativity.pdf", page_number=5, text="Speed of light is invariant.")

        prompt = build_evidence_prompt("What is non-locality?", [p1, p2])

        assert "RESEARCH QUESTION:" in prompt
        assert "What is non-locality?" in prompt
        assert "doc1_p1_c1" in prompt
        assert "quantum.pdf (Page 2)" in prompt
        assert "Entanglement is non-local." in prompt
        assert "doc2_p5_c2" in prompt
        assert "relativity.pdf (Page 5)" in prompt
        assert "Speed of light is invariant." in prompt


class TestCitationValidation:
    """Tests for validating citations against actual retrieved passages."""

    def test_valid_citations_mapped_accurately(self) -> None:
        p1 = _make_passage("p1", filename="chem.pdf", page_number=3, text="Water boils at 100C.")
        p2 = _make_passage("p2", filename="phys.pdf", page_number=8, text="Absolute zero is -273.15C.")
        passage_map = {"p1": p1, "p2": p2}

        valid_cits, invalid_ids = validate_and_map_citations(["p1", "p2"], passage_map)

        assert len(valid_cits) == 2
        assert invalid_ids == []
        assert valid_cits[0].filename == "chem.pdf"
        assert valid_cits[0].page_number == 3
        assert valid_cits[0].passage_text == "Water boils at 100C."
        assert valid_cits[1].filename == "phys.pdf"
        assert valid_cits[1].page_number == 8

    def test_unknown_citations_rejected(self) -> None:
        p1 = _make_passage("p1", filename="paper.pdf", page_number=1, text="Known fact.")
        passage_map = {"p1": p1}

        # Model attempts to cite unknown IDs
        valid_cits, invalid_ids = validate_and_map_citations(["p1", "ghost_id", "doc_invented_p9_c1"], passage_map)

        assert len(valid_cits) == 1
        assert valid_cits[0].passage_id == "p1"
        assert set(invalid_ids) == {"ghost_id", "doc_invented_p9_c1"}


class TestAnswerGeneration:
    """Tests for grounded generation, mock mode, and error handling."""

    def test_empty_retrieval_returns_insufficient_evidence_without_calling_model(self) -> None:
        mock_client = MagicMock(spec=NebiusClient)

        result = generate_grounded_answer(
            question="What is the critical temperature?",
            passages=[],
            client=mock_client,
            is_mock=False,
        )

        # Model must NEVER be called when no passages exist
        mock_client.generate.assert_not_called()
        assert result.insufficient_evidence is True
        assert result.citations == []
        assert "Insufficient evidence" in result.answer

    def test_mock_mode_clearly_labelled(self) -> None:
        p1 = _make_passage("p1", filename="superconductors.pdf", page_number=4, text="YBCO superconducts at 93 Kelvin.")

        result = generate_grounded_answer(
            question="What is YBCO critical temperature?",
            passages=[p1],
            is_mock=True,
        )

        assert result.is_mock is True
        assert "[DEMO MODE" in result.answer
        assert result.model_id == "mock-nemotron-offline"
        assert len(result.citations) == 1
        assert result.citations[0].filename == "superconductors.pdf"
        assert result.citations[0].page_number == 4

    def test_live_answer_with_valid_json_response(self) -> None:
        p1 = _make_passage("doc_q_p1_c1", filename="qkd.pdf", page_number=1, text="BB84 uses four polarization states.")
        mock_client = MagicMock(spec=NebiusClient)
        mock_client.settings = Settings(
            api_key="key",
            base_url="https://api.test",
            model_id="nvidia/Nemotron-3_5-Lightning",
            timeout_seconds=30.0,
            max_retries=2,
        )
        model_payload = {
            "answer": "BB84 protocol uses four polarization states [doc_q_p1_c1].",
            "cited_passage_ids": ["doc_q_p1_c1"],
            "limitations": "No eavesdropping simulation was included in the source.",
            "insufficient_evidence": False,
        }
        mock_client.generate.return_value = json.dumps(model_payload)

        result = generate_grounded_answer(
            question="How many states does BB84 use?",
            passages=[p1],
            client=mock_client,
            is_mock=False,
        )

        assert result.is_mock is False
        assert result.has_error is False
        assert result.is_verified is True
        assert "BB84 protocol uses four polarization states" in result.answer
        assert len(result.citations) == 1
        assert result.citations[0].filename == "qkd.pdf"
        assert result.citations[0].page_number == 1
        assert result.citations[0].passage_id == "doc_q_p1_c1"
        assert "No eavesdropping simulation" in result.limitations

    def test_model_detects_insufficient_evidence(self) -> None:
        p1 = _make_passage("p1", filename="cooking.pdf", page_number=1, text="Recipes for pasta dough.")
        mock_client = MagicMock(spec=NebiusClient)
        mock_client.settings = Settings("k", "u", "m", 30.0, 2)
        model_payload = {
            "answer": "The text discusses pasta cooking and contains no information about quantum physics.",
            "cited_passage_ids": [],
            "limitations": "Missing relevant physics data.",
            "insufficient_evidence": True,
        }
        mock_client.generate.return_value = json.dumps(model_payload)

        result = generate_grounded_answer(
            question="What is the speed of light?",
            passages=[p1],
            client=mock_client,
            is_mock=False,
        )

        assert result.insufficient_evidence is True
        assert "Missing relevant physics data" in result.limitations

    def test_malformed_json_surfaced_safely(self) -> None:
        p1 = _make_passage("p1", filename="paper.pdf", page_number=1, text="Text.")
        mock_client = MagicMock(spec=NebiusClient)
        mock_client.settings = Settings("k", "u", "m", 30.0, 2)
        # Invalid JSON returned by model
        mock_client.generate.return_value = "This is not JSON at all: {'answer': ...}"

        result = generate_grounded_answer(
            question="Any question?",
            passages=[p1],
            client=mock_client,
            is_mock=False,
        )

        assert result.has_error is True
        assert "JSON" in result.error_message
        assert "Warning" in result.limitations

    def test_api_failure_handled_gracefully(self) -> None:
        p1 = _make_passage("p1", filename="paper.pdf", page_number=1, text="Text.")
        mock_client = MagicMock(spec=NebiusClient)
        mock_client.settings = Settings("k", "u", "m", 30.0, 2)
        mock_client.generate.side_effect = NebiusError("Connection timed out after 30 seconds.")

        result = generate_grounded_answer(
            question="Any question?",
            passages=[p1],
            client=mock_client,
            is_mock=False,
        )

        assert result.has_error is True
        assert "Connection timed out" in result.error_message
        assert result.answer == ""
