"""Unit tests for content-addressed LibraryStore, BM25 phrase search, and HTML rendering."""

from __future__ import annotations

from app.library.chunker import HierarchicalPdfChunker
from app.library.companion_renderer import BookCompanionRenderer
from app.library.models import (
    BookProgress,
    BookToc,
    ChapterEntry,
    ChapterStatus,
    TopicMastery,
)
from app.library.search import PhraseBm25Searcher
from app.library.store import LibraryStore
from tests.unit.conftest import build_bookmarked_pdf_bytes


def test_library_store_persists_toc_chunks_and_reader_progress(
    isolated_store: LibraryStore,
) -> None:
    pdf_bytes = build_bookmarked_pdf_bytes()
    doc_id = isolated_store.compute_doc_id(pdf_bytes)
    assert not isolated_store.has_book(doc_id)

    toc, chunks = HierarchicalPdfChunker().chunk_pdf(
        pdf_bytes, doc_id=doc_id, fallback_title="guide.pdf"
    )
    isolated_store.save_book(toc, chunks)
    assert isolated_store.has_book(doc_id)

    reloaded_toc = isolated_store.load_toc(doc_id)
    assert reloaded_toc.title == "Distributed Systems Field Guide"
    ch02_chunks = isolated_store.load_chunks(doc_id, chapter_id="ch02")
    assert ch02_chunks
    assert all(chunk.chapter_id == "ch02" for chunk in ch02_chunks)

    progress = isolated_store.load_or_init_progress("alice", reloaded_toc)
    assert progress.bookmark_chapter_id == "ch01"
    progress.completed_chapters.append("ch01")
    progress.bookmark_chapter_id = "ch02"
    progress.topic_mastery["Quorum"] = TopicMastery(attempts=2, correct=2)
    isolated_store.save_progress("alice", progress)

    books = isolated_store.list_books("alice")
    assert len(books) == 1
    assert books[0]["bookmark_chapter_id"] == "ch02"
    assert books[0]["completed_chapters_count"] == 1
    assert books[0]["overall_mastery_pct"] == 100


def test_phrase_bm25_searcher_boosts_contiguous_phrases_and_scopes_chapters() -> None:
    pdf_bytes = build_bookmarked_pdf_bytes()
    _, chunks = HierarchicalPdfChunker().chunk_pdf(
        pdf_bytes, doc_id="book_search", fallback_title="guide.pdf"
    )
    searcher = PhraseBm25Searcher()

    hits = searcher.search(
        chunks, "randomized election timeouts", chapter_id="all", limit=3
    )
    assert hits
    assert hits[0].chunk.chapter_id == "ch02"

    scoped_hits = searcher.search(chunks, "election", chapter_id="ch03", limit=3)
    assert scoped_hits == []


def test_companion_renderer_escapes_html_and_formats_citations(
    isolated_store: LibraryStore,
) -> None:
    pdf_bytes = build_bookmarked_pdf_bytes()
    toc, _chunks = HierarchicalPdfChunker().chunk_pdf(
        pdf_bytes, doc_id="book_html", fallback_title="guide.pdf"
    )
    progress = isolated_store.load_or_init_progress("alice", toc)
    progress.completed_chapters.append("ch01")
    progress.chapter_digests["ch01"] = (
        "### The Big Idea\n"
        "Replicas execute commands deterministically <script>alert(1)</script> [p. 1].\n\n"
        '> "Safety guarantees that two replicas never commit conflicting values" — [p. 1]'
    )
    progress.topic_mastery["State Machine Replication"] = TopicMastery(
        attempts=2, correct=1
    )

    rendered = BookCompanionRenderer().render_html(toc, progress)
    assert "Distributed Systems Field Guide" in rendered
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert 'class="cite">[p. 1]</span>' in rendered
    assert "State Machine Replication: 50%" in rendered


def test_list_books_is_read_only(isolated_store: LibraryStore) -> None:
    pdf_bytes = build_bookmarked_pdf_bytes()
    toc, chunks = HierarchicalPdfChunker().chunk_pdf(
        pdf_bytes, doc_id="book_readonly", fallback_title="guide.pdf"
    )
    isolated_store.save_book(toc, chunks)

    books = isolated_store.list_books("alice")

    assert [book["doc_id"] for book in books] == ["book_readonly"]
    assert books[0]["bookmark_chapter_id"] == "ch01"
    assert not any((isolated_store.store_dir / "progress").iterdir())


def test_progress_reads_legacy_current_chapter_id_and_reports_status() -> None:
    progress = BookProgress.model_validate(
        {
            "doc_id": "book_legacy",
            "title": "Legacy",
            "current_chapter_id": "ch02",
            "completed_chapters": ["ch01"],
        }
    )

    assert progress.bookmark_chapter_id == "ch02"
    assert "bookmark_chapter_id" in progress.model_dump()
    assert progress.chapter_status("ch01") is ChapterStatus.DONE
    assert progress.chapter_status("ch02") is ChapterStatus.READING_NOW
    assert progress.chapter_status("ch03") is ChapterStatus.UP_NEXT


def test_bookmark_never_moves_backwards_and_empties_when_done() -> None:
    toc = BookToc(
        doc_id="book_bm",
        title="Bookmarks",
        total_pages=3,
        created_at="now",
        chapters=[
            ChapterEntry(chapter_id=f"ch0{i}", title=f"C{i}", page_start=i, page_end=i)
            for i in (1, 2, 3)
        ],
    )
    progress = BookProgress.new(toc)

    assert progress.save_chapter_digest(toc, "ch01", "d1") == "ch02"
    assert progress.save_chapter_digest(toc, "ch02", "d2") == "ch03"
    # Re-reading an earlier chapter keeps the bookmark on the next unread one.
    assert progress.save_chapter_digest(toc, "ch01", "d1 again") == "ch03"
    assert progress.bookmark_chapter_id == "ch03"
    assert progress.chapter_status("ch03") is ChapterStatus.READING_NOW

    assert progress.save_chapter_digest(toc, "ch03", "d3") is None
    assert progress.bookmark_chapter_id == ""

    # Skipping ahead leaves earlier gaps as the next unread chapter.
    gapped = BookProgress.new(toc)
    assert gapped.save_chapter_digest(toc, "ch03", "d3") == "ch01"
    assert gapped.bookmark_chapter_id == "ch01"


def test_record_quiz_results_advances_only_on_full_pass() -> None:
    toc = BookToc(
        doc_id="book_quiz",
        title="Quiz",
        total_pages=2,
        created_at="now",
        chapters=[
            ChapterEntry(chapter_id="ch01", title="A", page_start=1, page_end=1),
            ChapterEntry(chapter_id="ch02", title="B", page_start=2, page_end=2),
        ],
    )
    progress = BookProgress.new(toc)

    progress.record_quiz_results(toc, "ch01", {"Quorum": True, "Leases": False})
    assert progress.completed_chapters == ["ch01"]
    assert progress.bookmark_chapter_id == "ch01", "partial pass keeps the bookmark"
    assert progress.weak_topics == ["Leases"]

    progress.record_quiz_results(toc, "ch01", {"Quorum": True, "Leases": True})
    assert progress.bookmark_chapter_id == "ch02"
    assert progress.topic_mastery["Leases"].accuracy_pct == 50
