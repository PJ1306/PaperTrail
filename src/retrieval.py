"""Local research search and evidence retrieval for PaperTrail (Phase 3A).

This module provides deterministic page-level passage chunking and pure Python
BM25 (Best Matching 25) keyword retrieval. It operates completely in memory
and does not require an external API, database, or network connection.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Sequence

from src.pdf_processor import PageRecord, ProcessedDocument

# Default chunking parameters (characters)
DEFAULT_TARGET_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150

# Default BM25 scoring parameters (Okapi BM25 standard)
DEFAULT_BM25_K1 = 1.5
DEFAULT_BM25_B = 0.75


@dataclass(frozen=True)
class PassageRecord:
    """A bounded passage chunk extracted from a single page of a document.

    Page attribution is strictly preserved; passages are never merged across
    different pages or documents.
    """

    passage_id: str
    doc_id: str
    filename: str
    page_number: int  # 1-based page number
    chunk_index: int  # 1-based chunk index on this page
    text: str
    char_count: int


@dataclass(frozen=True)
class SearchResult:
    """A scored passage returned from evidence retrieval."""

    passage: PassageRecord
    score: float
    rank: int

    @property
    def filename(self) -> str:
        return self.passage.filename

    @property
    def page_number(self) -> int:
        return self.passage.page_number

    @property
    def doc_id(self) -> str:
        return self.passage.doc_id

    @property
    def passage_id(self) -> str:
        return self.passage.passage_id

    @property
    def text(self) -> str:
        return self.passage.text


def tokenize(text: str) -> list[str]:
    """Extract lowercased word tokens from text, stripping punctuation."""
    # Matches words with letters, digits, or internal hyphens/underscores
    tokens = re.findall(r"\b\w+\b", text.lower())
    # Exclude single character punctuation noise, keeping meaningful tokens
    return [t for t in tokens if len(t) > 1 or t.isalnum()]


def split_text_into_chunks(
    text: str,
    target_size: int = DEFAULT_TARGET_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split a page's text into bounded chunks with modest overlap.

    Respects natural paragraph and sentence boundaries wherever feasible.
    Returns an empty list for whitespace-only or empty text.
    """
    clean_text = text.strip()
    if not clean_text:
        return []

    # Safe boundaries for chunking parameters
    safe_target = max(100, target_size)
    safe_overlap = max(0, min(overlap, safe_target // 2))

    if len(clean_text) <= safe_target:
        return [clean_text]

    chunks: list[str] = []
    start = 0
    text_len = len(clean_text)

    while start < text_len:
        end = min(start + safe_target, text_len)

        if end < text_len:
            # Look for natural split points near end of target window
            window_start = start + int(safe_target * 0.6)

            # 1. Paragraph break
            p_break = clean_text.rfind("\n\n", window_start, end)
            if p_break != -1:
                end = p_break + 2
            else:
                # 2. Sentence boundary
                s_break = -1
                for sep in [".\n", ". ", "!\n", "! ", "?\n", "? "]:
                    found = clean_text.rfind(sep, window_start, end)
                    if found > s_break:
                        s_break = found + len(sep)
                if s_break != -1:
                    end = s_break
                else:
                    # 3. Newline or word break
                    space = clean_text.rfind(" ", window_start, end)
                    newline = clean_text.rfind("\n", window_start, end)
                    best_break = max(space, newline)
                    if best_break != -1:
                        end = best_break + 1

        chunk = clean_text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break

        next_start = end - safe_overlap
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def chunk_page(
    page: PageRecord,
    target_size: int = DEFAULT_TARGET_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[PassageRecord]:
    """Chunk a single page into structured PassageRecord instances.

    Guarantees that page number and document metadata are preserved.
    """
    if not page.has_text:
        return []

    text_chunks = split_text_into_chunks(page.text, target_size=target_size, overlap=overlap)
    passages: list[PassageRecord] = []

    for idx, chunk in enumerate(text_chunks, start=1):
        passage_id = f"{page.doc_id}_p{page.page_number}_c{idx}"
        passages.append(
            PassageRecord(
                passage_id=passage_id,
                doc_id=page.doc_id,
                filename=page.filename,
                page_number=page.page_number,
                chunk_index=idx,
                text=chunk,
                char_count=len(chunk),
            )
        )

    return passages


def chunk_pages(
    pages: Sequence[PageRecord],
    target_size: int = DEFAULT_TARGET_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[PassageRecord]:
    """Chunk a collection of PageRecords into PassageRecord instances."""
    passages: list[PassageRecord] = []
    for page in pages:
        passages.extend(chunk_page(page, target_size=target_size, overlap=overlap))
    return passages


def chunk_documents(
    documents: Sequence[ProcessedDocument],
    target_size: int = DEFAULT_TARGET_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[PassageRecord]:
    """Extract and chunk all pages across a collection of ProcessedDocuments."""
    all_pages: list[PageRecord] = []
    for doc in documents:
        all_pages.extend(doc.pages)
    return chunk_pages(all_pages, target_size=target_size, overlap=overlap)


class BM25Retriever:
    """Okapi BM25 keyword retriever operating entirely in memory.

    Calculates Robertson-Spärck Jones IDF and length-normalized term frequencies
    to rank passages by query relevance.
    """

    def __init__(
        self,
        passages: Sequence[PassageRecord],
        k1: float = DEFAULT_BM25_K1,
        b: float = DEFAULT_BM25_B,
    ) -> None:
        self.passages = list(passages)
        self.k1 = k1
        self.b = b
        self.corpus_size = len(self.passages)

        # Pre-tokenize all passages and collect corpus statistics
        self.doc_tokens: list[list[str]] = [tokenize(p.text) for p in self.passages]
        self.doc_lens: list[int] = [len(tokens) for tokens in self.doc_tokens]

        if self.corpus_size > 0:
            self.avg_doc_len = sum(self.doc_lens) / self.corpus_size
        else:
            self.avg_doc_len = 0.0

        # Document frequencies (number of passages containing term t)
        self.doc_freqs: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            for term in set(tokens):
                self.doc_freqs[term] += 1

    def _idf(self, term: str) -> float:
        """Compute smoothed IDF score for a query term."""
        n_q = self.doc_freqs.get(term, 0)
        # Adding 1.0 inside logarithm guarantees positive IDF score
        return math.log((self.corpus_size - n_q + 0.5) / (n_q + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Rank passages against query.

        Returns only passages with a strictly positive relevance score (> 0).
        If no relevant passages match, returns an empty list.
        """
        if not self.passages or not query.strip():
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        # Find matching passages and compute BM25 scores
        scored_results: list[tuple[float, PassageRecord]] = []

        for idx, passage in enumerate(self.passages):
            tokens = self.doc_tokens[idx]
            dl = self.doc_lens[idx]
            if dl == 0:
                continue

            term_counts = Counter(tokens)
            score = 0.0

            for q_term in query_tokens:
                if q_term in term_counts:
                    f = term_counts[q_term]
                    idf_score = self._idf(q_term)
                    # Standard BM25 term frequency saturation formula
                    num = f * (self.k1 + 1.0)
                    denom = f + self.k1 * (
                        1.0 - self.b + self.b * (dl / (self.avg_doc_len or 1.0))
                    )
                    score += idf_score * (num / denom)

            # Only retain passages with at least one matching term (score > 0)
            if score > 0.0:
                scored_results.append((score, passage))

        # Sort descending by score
        scored_results.sort(key=lambda item: item[0], reverse=True)

        # Slice to requested top_k and wrap into SearchResult
        results: list[SearchResult] = []
        for rank, (score, passage) in enumerate(scored_results[:top_k], start=1):
            results.append(
                SearchResult(
                    passage=passage,
                    score=round(score, 4),
                    rank=rank,
                )
            )

        return results


def search_passages(
    query: str,
    passages: Sequence[PassageRecord],
    top_k: int = 5,
) -> list[SearchResult]:
    """Search pre-chunked passages using BM25 keyword retrieval."""
    retriever = BM25Retriever(passages)
    return retriever.search(query, top_k=top_k)


def search_pages(
    query: str,
    pages: Sequence[PageRecord],
    top_k: int = 5,
    target_size: int = DEFAULT_TARGET_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[SearchResult]:
    """Chunk extracted pages and search them using BM25 keyword retrieval."""
    passages = chunk_pages(pages, target_size=target_size, overlap=overlap)
    return search_passages(query, passages, top_k=top_k)


def search_documents(
    query: str,
    documents: Sequence[ProcessedDocument],
    top_k: int = 5,
    target_size: int = DEFAULT_TARGET_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[SearchResult]:
    """Chunk processed documents and search them using BM25 keyword retrieval."""
    passages = chunk_documents(documents, target_size=target_size, overlap=overlap)
    return search_passages(query, passages, top_k=top_k)
