"""Unit tests for passage chunking and local BM25 keyword retrieval (Phase 3A).

Uses synthetic PageRecords and ProcessedDocuments. No external APIs or network calls.
"""

from __future__ import annotations

import pytest

from src.pdf_processor import PageRecord, ProcessedDocument
from src.retrieval import (
    BM25Retriever,
    PassageRecord,
    SearchResult,
    chunk_documents,
    chunk_page,
    chunk_pages,
    search_documents,
    search_pages,
    search_passages,
    split_text_into_chunks,
    tokenize,
)


class TestChunking:
    """Tests for deterministic page-level text chunking."""

    def test_chunk_short_text_single_passage(self) -> None:
        short_text = "This is a short research abstract containing about 100 characters."
        chunks = split_text_into_chunks(short_text, target_size=500, overlap=50)

        assert len(chunks) == 1
        assert chunks[0] == short_text

    def test_chunk_long_text_multiple_passages(self) -> None:
        # Create a long text of ~2500 characters
        paragraph = (
            "Neural networks utilize backpropagation and gradient descent to optimize parameters. "
            "Transformers introduce multi-head self-attention mechanisms for sequence modeling. "
        )
        long_text = paragraph * 15
        assert len(long_text) > 2000

        chunks = split_text_into_chunks(long_text, target_size=800, overlap=100)

        assert len(chunks) >= 3
        for chunk in chunks:
            # Each chunk should be roughly around the target size
            assert 200 <= len(chunk) <= 1100

    def test_chunk_preserves_page_metadata(self) -> None:
        page = PageRecord(
            doc_id="doc_abc123",
            filename="superconductors.pdf",
            page_number=4,
            text=("High-temperature superconductors exhibit zero resistance below critical temperatures. " * 20),
        )

        passages = chunk_page(page, target_size=500, overlap=50)

        assert len(passages) >= 2
        for idx, passage in enumerate(passages, start=1):
            assert passage.doc_id == "doc_abc123"
            assert passage.filename == "superconductors.pdf"
            assert passage.page_number == 4
            assert passage.chunk_index == idx
            assert passage.passage_id == f"doc_abc123_p4_c{idx}"
            assert len(passage.text) > 0
            assert passage.char_count == len(passage.text)

    def test_empty_or_whitespace_page_returns_no_passages(self) -> None:
        page_empty = PageRecord(doc_id="d1", filename="f1.pdf", page_number=1, text="")
        page_whitespace = PageRecord(doc_id="d2", filename="f2.pdf", page_number=2, text="   \n\t  ")

        assert chunk_page(page_empty) == []
        assert chunk_page(page_whitespace) == []
        assert split_text_into_chunks("") == []
        assert split_text_into_chunks("   \n\t  ") == []

    def test_different_pages_never_merged(self) -> None:
        page1 = PageRecord(doc_id="doc1", filename="paper.pdf", page_number=1, text="Text on page one.")
        page2 = PageRecord(doc_id="doc1", filename="paper.pdf", page_number=2, text="Text on page two.")

        passages = chunk_pages([page1, page2])

        assert len(passages) == 2
        assert passages[0].page_number == 1
        assert passages[0].text == "Text on page one."
        assert passages[1].page_number == 2
        assert passages[1].text == "Text on page two."


class TestBM25Retrieval:
    """Tests for BM25 keyword retrieval and passage ranking."""

    def test_case_and_punctuation_normalization(self) -> None:
        tokens = tokenize("Quantum, Computing: In Theory & Practice (v2.0)!")
        assert "quantum" in tokens
        assert "computing" in tokens
        assert "theory" in tokens
        assert "practice" in tokens
        assert "v2" in tokens or "2" in tokens

    def test_relevant_passage_ranked_first(self) -> None:
        p1 = PassageRecord(
            passage_id="p1",
            doc_id="d1",
            filename="quantum.pdf",
            page_number=1,
            chunk_index=1,
            text="Quantum key distribution uses entangled photon pairs to establish unconditionally secure keys.",
            char_count=98,
        )
        p2 = PassageRecord(
            passage_id="p2",
            doc_id="d2",
            filename="biology.pdf",
            page_number=1,
            chunk_index=1,
            text="Cellular mitosis involves prophase, metaphase, anaphase, and telophase stages of division.",
            char_count=92,
        )
        p3 = PassageRecord(
            passage_id="p3",
            doc_id="d3",
            filename="history.pdf",
            page_number=1,
            chunk_index=1,
            text="The Industrial Revolution began in Great Britain in the late eighteenth century.",
            char_count=82,
        )

        results = search_passages("quantum key distribution photon", [p1, p2, p3], top_k=3)

        assert len(results) == 1
        assert results[0].passage_id == "p1"
        assert results[0].filename == "quantum.pdf"
        assert results[0].page_number == 1
        assert results[0].score > 0
        assert results[0].rank == 1

    def test_query_with_no_matches_returns_empty_list(self) -> None:
        p1 = PassageRecord(
            passage_id="p1",
            doc_id="d1",
            filename="astro.pdf",
            page_number=1,
            chunk_index=1,
            text="Exoplanets orbiting red dwarf stars may experience intense coronal mass ejections.",
            char_count=85,
        )

        results = search_passages("chlorophyll photosynthesis plant", [p1])
        assert results == []

    def test_empty_or_whitespace_query_returns_empty_list(self) -> None:
        p1 = PassageRecord(
            passage_id="p1",
            doc_id="d1",
            filename="paper.pdf",
            page_number=1,
            chunk_index=1,
            text="Some relevant research text.",
            char_count=28,
        )

        assert search_passages("", [p1]) == []
        assert search_passages("   \t\n ", [p1]) == []

    def test_multiple_matching_terms_score_higher(self) -> None:
        # Passage A matches both "transformer" and "attention"
        p_both = PassageRecord(
            passage_id="pa",
            doc_id="da",
            filename="nlp.pdf",
            page_number=1,
            chunk_index=1,
            text="The transformer architecture relies entirely on self attention mechanisms without recurrence.",
            char_count=92,
        )
        # Passage B matches only "transformer"
        p_single = PassageRecord(
            passage_id="pb",
            doc_id="db",
            filename="elec.pdf",
            page_number=1,
            chunk_index=1,
            text="An electrical transformer steps voltage up or down via magnetic induction between coils.",
            char_count=88,
        )

        results = search_passages("transformer attention", [p_both, p_single], top_k=2)

        assert len(results) == 2
        assert results[0].passage_id == "pa"
        assert results[0].score > results[1].score

    def test_multiple_papers_correct_attribution(self) -> None:
        doc_a = ProcessedDocument(
            doc_id="doc_chem",
            filename="chemistry.pdf",
            file_size_bytes=1000,
            total_pages=2,
            pages_with_text=2,
            pages=[
                PageRecord(
                    doc_id="doc_chem",
                    filename="chemistry.pdf",
                    page_number=1,
                    text="Covalent bonds involve electron sharing between atoms.",
                ),
                PageRecord(
                    doc_id="doc_chem",
                    filename="chemistry.pdf",
                    page_number=2,
                    text="Catalysts lower the activation energy required for chemical reactions.",
                ),
            ],
        )
        doc_b = ProcessedDocument(
            doc_id="doc_phys",
            filename="physics.pdf",
            file_size_bytes=1000,
            total_pages=1,
            pages_with_text=1,
            pages=[
                PageRecord(
                    doc_id="doc_phys",
                    filename="physics.pdf",
                    page_number=1,
                    text="Thermodynamics governs heat and energy transfer in reactions.",
                ),
            ],
        )

        results = search_documents("activation energy catalysts", [doc_a, doc_b], top_k=5)

        # Both match on "energy", but doc_a matches all 3 terms so it ranks first
        assert len(results) == 2

        # Most relevant result is chemistry.pdf page 2
        top_res = results[0]
        assert top_res.filename == "chemistry.pdf"
        assert top_res.page_number == 2
        assert top_res.doc_id == "doc_chem"
        assert "Catalysts lower the activation energy" in top_res.text

        # Secondary match is physics.pdf page 1
        second_res = results[1]
        assert second_res.filename == "physics.pdf"
        assert second_res.page_number == 1
        assert second_res.doc_id == "doc_phys"
        assert "Thermodynamics" in second_res.text

        # Ranking score of multi-term match must exceed single-term match
        assert top_res.score > second_res.score

    def test_top_k_limit_respected(self) -> None:
        passages = [
            PassageRecord(
                passage_id=f"p{i}",
                doc_id=f"d{i}",
                filename=f"paper{i}.pdf",
                page_number=1,
                chunk_index=1,
                text=f"Machine learning experiment {i} demonstrates accuracy improvements on benchmarks.",
                char_count=80,
            )
            for i in range(10)
        ]

        results = search_passages("machine learning accuracy", passages, top_k=3)
        assert len(results) == 3
        assert [r.rank for r in results] == [1, 2, 3]
