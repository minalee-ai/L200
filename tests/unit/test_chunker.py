"""Unit tests for 2-level chapter detection and phrase-bounded chunking."""

from __future__ import annotations

import pymupdf
import pytest

from app.library.chunker import HierarchicalPdfChunker
from tests.unit.conftest import (
    build_bookmarked_pdf_bytes,
    build_font_headed_paper_bytes,
)


def test_chunker_extracts_bookmarked_chapters_and_metadata() -> None:
    chunker = HierarchicalPdfChunker(target_words=80, overlap_phrases=1)
    toc, chunks = chunker.chunk_pdf(
        build_bookmarked_pdf_bytes(),
        doc_id="book_test0001",
        fallback_title="fallback.pdf",
    )

    assert toc.title == "Distributed Systems Field Guide"
    assert toc.total_pages == 3
    assert [ch.chapter_id for ch in toc.chapters] == ["ch01", "ch02", "ch03"]
    assert toc.chapters[0].page_start == 1
    assert toc.chapters[1].page_start == 2
    assert toc.chapters[2].page_start == 3
    assert len(chunks) >= 3


def test_chunker_falls_back_to_font_and_numbered_headings() -> None:
    chunker = HierarchicalPdfChunker(target_words=80, overlap_phrases=1)
    toc, chunks = chunker.chunk_pdf(
        build_font_headed_paper_bytes(),
        doc_id="book_paper01",
        fallback_title="attention_paper.pdf",
    )

    titles = [ch.title for ch in toc.chapters]
    assert "1. Introduction" in titles
    assert "2. Scaled Dot-Product Attention" in titles
    assert len(chunks) >= 2


def test_chunker_preserves_complete_phrases_and_sliding_overlap() -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    sentences = [
        f"Sentence {idx} explains invariant number {idx} in clear detail for distributed storage."
        for idx in range(1, 10)
    ]
    page.insert_text((50, 60), "\n\n".join(sentences), fontsize=11)
    pdf_bytes = doc.tobytes()
    doc.close()

    chunker = HierarchicalPdfChunker(target_words=45, overlap_phrases=2)
    _, chunks = chunker.chunk_pdf(
        pdf_bytes,
        doc_id="book_overlap",
        fallback_title="overlap.pdf",
    )

    assert len(chunks) >= 2
    # Every chunk ends on a complete sentence boundary rather than mid-word.
    for chunk in chunks:
        assert chunk.text.endswith(".")

    # Trailing sentences of chunk 0 must reappear at the start of chunk 1 for overlap continuity.
    first_chunk_last_sentence = "explains invariant number 3"
    assert first_chunk_last_sentence in chunks[0].text
    assert first_chunk_last_sentence in chunks[1].text


def test_chunker_rejects_empty_or_blank_pdf() -> None:
    chunker = HierarchicalPdfChunker()
    with pytest.raises(ValueError, match="empty"):
        chunker.chunk_pdf(b"", doc_id="book_empty", fallback_title="empty.pdf")

    blank_doc = pymupdf.open()
    blank_doc.new_page()
    blank_bytes = blank_doc.tobytes()
    blank_doc.close()

    with pytest.raises(ValueError, match="no extractable text"):
        chunker.chunk_pdf(blank_bytes, doc_id="book_blank", fallback_title="blank.pdf")
