"""ADK function tools for BookCompanion's library, search, and progress features.

Blocking work (PDF parsing, store file I/O, BM25 scoring) runs on worker
threads via `asyncio.to_thread` so the serving event loop stays responsive.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from google.adk.tools import ToolContext

from app.library.chunker import HierarchicalPdfChunker
from app.library.models import BookChunk, BookProgress, BookToc, SearchHit
from app.library.search import PhraseBm25Searcher
from app.library.store import LibraryStore
from app.runtime import (
    COMPANION_ARTIFACT_FILENAME,
    MAX_SEARCH_LIMIT,
    get_library_store,
    render_companion_part,
)
from app.state import QUIZ_RESET_DELTA, StateKey, reader_state_delta

_OPENING_EXCERPT_CHUNKS = 2
_OPENING_EXCERPT_MAX_CHARS = 900
_LATEST_UPLOAD_ALIASES = frozenset({"", "latest", "uploaded"})

_chunker = HierarchicalPdfChunker()
_searcher = PhraseBm25Searcher()


def _error(message: str) -> dict[str, Any]:
    return {"status": "error", "message": message}


def _publish_reader_state(
    tool_context: ToolContext,
    toc: BookToc,
    progress: BookProgress,
) -> None:
    """Mirrors persisted progress into session state for the root prompt."""
    tool_context.state.update(
        reader_state_delta(
            toc,
            progress,
            saved_bookshelf=tool_context.state.get(StateKey.SAVED_BOOKSHELF_OVERVIEW),
        )
    )


async def _save_companion_html_artifact(
    tool_context: ToolContext,
    toc: BookToc,
    progress: BookProgress,
) -> int | None:
    """Renders book_companion.html and saves it if ArtifactService is configured."""
    try:
        return await tool_context.save_artifact(
            COMPANION_ARTIFACT_FILENAME, render_companion_part(toc, progress)
        )
    except ValueError:
        # Headless test runners without an ArtifactService continue cleanly.
        return None


async def _load_uploaded_pdf_bytes(
    source: str,
    tool_context: ToolContext,
) -> tuple[bytes, str]:
    """Resolves PDF bytes from ADK ArtifactService first, falling back to local path."""
    cleaned = source.strip().strip('"').strip("'")

    try:
        artifact_names = [
            name
            for name in await tool_context.list_artifacts()
            if name != COMPANION_ARTIFACT_FILENAME
        ]
    except ValueError:
        artifact_names = []

    matched_artifact: str | None = None
    if cleaned in artifact_names:
        matched_artifact = cleaned
    elif cleaned.lower() in _LATEST_UPLOAD_ALIASES and artifact_names:
        pdf_artifacts = [n for n in artifact_names if n.lower().endswith(".pdf")]
        matched_artifact = pdf_artifacts[-1] if pdf_artifacts else artifact_names[-1]
    else:
        basename = Path(cleaned).name
        if basename in artifact_names:
            matched_artifact = basename

    if matched_artifact is not None:
        artifact_part = await tool_context.load_artifact(matched_artifact)
        if (
            artifact_part is not None
            and artifact_part.inline_data is not None
            and artifact_part.inline_data.data
        ):
            return artifact_part.inline_data.data, matched_artifact

    file_path = Path(cleaned).expanduser()
    if not file_path.is_file() and not file_path.is_absolute():
        candidate = Path(__file__).resolve().parents[1] / file_path
        if candidate.is_file():
            file_path = candidate
    if not file_path.is_file():
        raise FileNotFoundError(
            f"Could not find uploaded PDF artifact or local file '{cleaned}'."
        )
    if file_path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Unsupported file extension '{file_path.suffix}'; expected a .pdf file."
        )
    return await asyncio.to_thread(file_path.read_bytes), file_path.name


def _index_pdf(
    store: LibraryStore, pdf_bytes: bytes, display_name: str
) -> tuple[BookToc, bool]:
    """Returns the cached index for a PDF, chunking and persisting it on a miss.

    Blocking (PyMuPDF parsing + file I/O); call via `asyncio.to_thread`.

    Returns:
        The book's TOC and whether it was already cached.
    """
    doc_id = store.compute_doc_id(pdf_bytes)
    if store.has_book(doc_id):
        return store.load_toc(doc_id), True
    toc, chunks = _chunker.chunk_pdf(
        pdf_bytes, doc_id=doc_id, fallback_title=display_name
    )
    store.save_book(toc, chunks)
    return toc, False


async def ingest_or_load_book(source: str, tool_context: ToolContext) -> dict[str, Any]:
    """Indexes an uploaded PDF by chapter and phrase boundaries or reloads a saved book.

    Call this when the user attaches a PDF, provides a PDF filename/path, or selects
    a saved `doc_id` (such as `book_a1b2c3d4e5f6`) from their bookshelf. Pass `"latest"`
    if the user attached a PDF without typing its filename.

    Args:
        source: Artifact filename, `"latest"`, saved `doc_id`, or local `.pdf` file path.

    Returns:
        Dictionary containing `status`, `doc_id`, `title`, `cache_hit`, `chapters`,
        `bookmark_chapter_id`, and `opening_excerpt`.
    """
    store = get_library_store()
    user_id = tool_context.user_id
    cleaned_source = source.strip()

    try:
        if cleaned_source.startswith("book_") and store.has_book(cleaned_source):
            toc = await asyncio.to_thread(store.load_toc, cleaned_source)
            cache_hit = True
        else:
            pdf_bytes, display_name = await _load_uploaded_pdf_bytes(
                cleaned_source, tool_context
            )
            toc, cache_hit = await asyncio.to_thread(
                _index_pdf, store, pdf_bytes, display_name
            )
    except (FileNotFoundError, ValueError) as err:
        return _error(str(err))

    progress = await asyncio.to_thread(store.load_or_init_progress, user_id, toc)
    bookshelf = await asyncio.to_thread(store.list_books, user_id)

    if tool_context.state.get(StateKey.ACTIVE_BOOK_ID) != toc.doc_id:
        # Switching books invalidates the chapter and quiz context of the old one.
        tool_context.state.update({StateKey.ACTIVE_CHAPTER_ID: "", **QUIZ_RESET_DELTA})
    tool_context.state.update(
        reader_state_delta(toc, progress, saved_bookshelf=bookshelf)
    )
    artifact_version = await _save_companion_html_artifact(tool_context, toc, progress)

    opening_excerpt = ""
    if toc.first_chapter_id:
        opening_chunks = await asyncio.to_thread(
            store.load_chunks, toc.doc_id, chapter_id=toc.first_chapter_id
        )
        opening_excerpt = " ".join(
            chunk.text for chunk in opening_chunks[:_OPENING_EXCERPT_CHUNKS]
        )[:_OPENING_EXCERPT_MAX_CHARS]

    return {
        "status": "success",
        "doc_id": toc.doc_id,
        "title": toc.title,
        "total_pages": toc.total_pages,
        "cache_hit": cache_hit,
        "bookmark_chapter_id": progress.bookmark_chapter_id,
        "completed_chapters": progress.completed_chapters,
        "overall_mastery_pct": progress.overall_mastery_pct,
        "companion_artifact": COMPANION_ARTIFACT_FILENAME
        if artifact_version is not None
        else None,
        "chapters": [ch.model_dump() for ch in toc.chapters],
        "opening_excerpt": opening_excerpt,
    }


async def list_saved_books(tool_context: ToolContext) -> dict[str, Any]:
    """Lists all previously chunked and saved books on the reader's bookshelf.

    Returns:
        Dictionary with `status` and `books` containing saved titles, chapter counts,
        bookmarks, and mastery percentages.
    """
    books = await asyncio.to_thread(
        get_library_store().list_books, tool_context.user_id
    )
    tool_context.state[StateKey.SAVED_BOOKSHELF_OVERVIEW] = books
    return {"status": "success", "count": len(books), "books": books}


def _load_toc_and_chunks(
    doc_id: str, chapter_id: str | None
) -> tuple[BookToc, list[BookChunk]]:
    store = get_library_store()
    return store.load_toc(doc_id), store.load_chunks(doc_id, chapter_id=chapter_id)


async def get_chapter_chunks(
    doc_id: str,
    chapter_id: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Loads saved phrase-bounded chunks and page numbers for a specific chapter.

    Args:
        doc_id: Identifier of the saved book (e.g., `book_...`).
        chapter_id: Target chapter identifier (e.g., `ch01`, `ch02`).

    Returns:
        Dictionary with chapter metadata and ordered chunks tagged with page numbers.
    """
    try:
        toc, chunks = await asyncio.to_thread(_load_toc_and_chunks, doc_id, chapter_id)
    except FileNotFoundError as err:
        return _error(str(err))

    if not chunks:
        return _error(
            f"Chapter '{chapter_id}' not found. Available chapters: {toc.chapter_ids}."
        )

    tool_context.state[StateKey.ACTIVE_CHAPTER_ID] = chapter_id
    return {
        "status": "success",
        "doc_id": doc_id,
        "chapter_id": chapter_id,
        "chapter_title": chunks[0].chapter_title,
        "chunk_count": len(chunks),
        "chunks": [
            {
                "chunk_id": ch.chunk_id,
                "pages": f"{ch.page_start}-{ch.page_end}"
                if ch.page_start != ch.page_end
                else f"{ch.page_start}",
                "text": ch.text,
            }
            for ch in chunks
        ],
    }


def _search_chunks(
    doc_id: str, query: str, chapter_id: str, limit: int
) -> list[SearchHit]:
    chunks = get_library_store().load_chunks(doc_id)
    return _searcher.search(chunks, query, chapter_id=chapter_id, limit=limit)


async def search_book_phrases(
    doc_id: str,
    query: str,
    chapter_id: str,
    limit: int,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Searches saved book chunks using phrase-boosted BM25 retrieval.

    Use this for `"deep dive"` questions, locating specific author quotes, or checking
    whether a concept is discussed in the book.

    Args:
        doc_id: Identifier of the saved book.
        query: Specific phrase, term, or concept question to look up.
        chapter_id: Chapter filter such as `ch02`, or `"all"` to search the entire book.
        limit: Maximum number of matching chunks to return (1 to 8).

    Returns:
        Dictionary with ranked chunk matches and exact page citations.
    """
    del tool_context
    capped_limit = max(1, min(limit, MAX_SEARCH_LIMIT))
    try:
        hits = await asyncio.to_thread(
            _search_chunks, doc_id, query, chapter_id, capped_limit
        )
    except FileNotFoundError as err:
        return _error(str(err))

    return {
        "status": "success",
        "doc_id": doc_id,
        "query": query,
        "match_count": len(hits),
        "matches": [
            {
                "chunk_id": hit.chunk.chunk_id,
                "chapter_id": hit.chunk.chapter_id,
                "chapter_title": hit.chunk.chapter_title,
                "page_start": hit.chunk.page_start,
                "page_end": hit.chunk.page_end,
                "score": hit.score,
                "text": hit.chunk.text,
            }
            for hit in hits
        ],
    }


async def record_progress(
    doc_id: str,
    chapter_id: str,
    topic: str,
    quiz_passed: bool,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Records one graded concept outside the quiz flow and refreshes the companion HTML.

    `grade_quiz_answers` records check-in results itself; use this only when the
    reader demonstrates (or misses) a concept in free-form discussion.

    Args:
        doc_id: Identifier of the saved book.
        chapter_id: Chapter identifier tested by the check-in question.
        topic: Concise name of the concept being graded.
        quiz_passed: True if the reader's answer demonstrated understanding, False otherwise.

    Returns:
        Dictionary with updated topic accuracy, overall mastery, and the next unread chapter.
    """
    store = get_library_store()
    try:
        toc = await asyncio.to_thread(store.load_toc, doc_id)
    except FileNotFoundError as err:
        return _error(str(err))

    cleaned_topic = topic.strip() or chapter_id

    def _apply(progress: BookProgress) -> str | None:
        progress.record_topic_result(cleaned_topic, quiz_passed)
        return progress.mark_chapter_read(toc, chapter_id, advance_bookmark=quiz_passed)

    progress, next_chapter_id = await asyncio.to_thread(
        store.update_progress, tool_context.user_id, toc, _apply
    )
    _publish_reader_state(tool_context, toc, progress)
    await _save_companion_html_artifact(tool_context, toc, progress)

    return {
        "status": "success",
        "topic": cleaned_topic,
        "topic_accuracy_pct": progress.topic_mastery[cleaned_topic].accuracy_pct,
        "overall_mastery_pct": progress.overall_mastery_pct,
        "completed_chapters": progress.completed_chapters,
        "next_chapter_id": next_chapter_id,
        "weak_topics_to_review": progress.weak_topics,
    }


async def update_book_companion_artifact(
    doc_id: str,
    chapter_id: str,
    chapter_digest_md: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Saves an abridged chapter summary into the book_companion.html reading edition.

    Args:
        doc_id: Identifier of the saved book.
        chapter_id: Chapter identifier (e.g., `ch01`).
        chapter_digest_md: Grounded Markdown summary of the chapter with `[p. X]` citations.

    Returns:
        Dictionary confirming artifact save status and next unread chapter.
    """
    store = get_library_store()
    try:
        toc = await asyncio.to_thread(store.load_toc, doc_id)
    except FileNotFoundError as err:
        return _error(str(err))
    if not toc.has_chapter(chapter_id):
        return _error(
            f"Chapter '{chapter_id}' not found. Available chapters: {toc.chapter_ids}."
        )

    progress, next_chapter_id = await asyncio.to_thread(
        store.update_progress,
        tool_context.user_id,
        toc,
        lambda p: p.save_chapter_digest(toc, chapter_id, chapter_digest_md),
    )
    _publish_reader_state(tool_context, toc, progress)
    version = await _save_companion_html_artifact(tool_context, toc, progress)

    return {
        "status": "success",
        "artifact_filename": COMPANION_ARTIFACT_FILENAME,
        "artifact_version": version,
        "completed_chapters": progress.completed_chapters,
        "next_chapter_id": next_chapter_id,
    }
