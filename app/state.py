"""Session-state keys and the projection of persisted reader progress into state.

`LibraryStore` is the single source of truth for reading progress. Session
state holds only (a) a prompt-friendly view of that progress, rebuilt through
`reader_state_delta`, and (b) transient hand-off values between pipeline steps.
"""

from __future__ import annotations

from typing import Any

from app.library.models import BookProgress, BookToc


class StateKey:
    """Names of session-state entries shared by prompts, tools, and workflows.

    Values must match the `{{ key? }}` placeholders in `app/prompts.py`
    (enforced by `tests/unit/test_prompts.py`). Keys under ADK's `temp:` scope
    live only for the current invocation and are never persisted, which keeps
    chapter text and pipeline scratch values out of session history. Because
    session services drop `temp:` keys on load, they can only be set by an
    event inside the invocation (tests use `SeedStateAgent` for this), never
    via `create_session(state=...)`.
    """

    # Reader view, derived from LibraryStore via reader_state_delta().
    ACTIVE_BOOK_ID = "active_book_id"
    ACTIVE_BOOK_TITLE = "active_book_title"
    BOOKMARK_CHAPTER_ID = "bookmark_chapter_id"  # Next unread chapter.
    DOC_TOC = "doc_toc"
    SAVED_BOOKSHELF_OVERVIEW = "saved_bookshelf_overview"

    # Chapter most recently presented: the default target for check-ins and
    # deep dives. Deliberately separate from the bookmark, which moves past a
    # chapter as soon as it has been summarized.
    ACTIVE_CHAPTER_ID = "active_chapter_id"

    # Quiz awaiting answers. ACTIVE_QUIZ_KEY holds answers and must never be
    # rendered into the root agent's instruction.
    ACTIVE_QUIZ_KEY = "active_quiz_key"
    PENDING_QUIZ_CHAPTER_ID = "pending_quiz_chapter_id"

    # Per-invocation hand-off between pipeline steps.
    ACTIVE_CHAPTER_TITLE = "temp:active_chapter_title"
    ACTIVE_CHAPTER_MATERIAL = "temp:active_chapter_material"
    ACTIVE_CHAPTER_COVERAGE_NOTE = "temp:active_chapter_coverage_note"
    LATEST_CHAPTER_DIGEST = "temp:latest_chapter_digest"
    QUIZ_DRAFT = "temp:quiz_draft"
    QUIZ_GROUNDING_FEEDBACK = "temp:quiz_grounding_feedback"
    QUIZ_SUBMISSION = "temp:quiz_submission"
    QUIZ_GRADE_REPORT = "temp:quiz_grade_report"
    # Reader-facing message set by a pipeline step to skip the remaining steps.
    PIPELINE_ERROR = "temp:pipeline_error"


ALL_STATE_KEYS = frozenset(
    value for name, value in vars(StateKey).items() if name.isupper()
)

# Cleared whenever the reader moves on (new book, new chapter digest) so a
# stale check-in never lingers in the root prompt.
QUIZ_RESET_DELTA: dict[str, Any] = {
    StateKey.ACTIVE_QUIZ_KEY: {},
    StateKey.PENDING_QUIZ_CHAPTER_ID: "",
}


def toc_view(toc: BookToc, progress: BookProgress) -> list[dict[str, str]]:
    """Returns the compact table of contents with per-chapter reading status."""
    return [
        {
            "chapter_id": chapter.chapter_id,
            "title": chapter.title,
            "pages": f"{chapter.page_start}-{chapter.page_end}",
            "status": progress.chapter_status(chapter.chapter_id).value,
        }
        for chapter in toc.chapters
    ]


def reader_state_delta(
    toc: BookToc,
    progress: BookProgress,
    *,
    saved_bookshelf: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Builds the session-state view of one book's persisted progress.

    Args:
        toc: Book whose progress changed.
        progress: Freshly persisted progress for the current reader.
        saved_bookshelf: Current `SAVED_BOOKSHELF_OVERVIEW` value; this book's
            entry is replaced (or appended) so the overview stays current
            without rescanning the store.

    Returns:
        A state delta suitable for `State.update()` or `EventActions.state_delta`.
    """
    entry = progress.bookshelf_entry(toc)
    bookshelf = [
        entry if book.get("doc_id") == toc.doc_id else book
        for book in (saved_bookshelf or [])
    ]
    if not any(book.get("doc_id") == toc.doc_id for book in bookshelf):
        bookshelf.append(entry)

    return {
        StateKey.ACTIVE_BOOK_ID: toc.doc_id,
        StateKey.ACTIVE_BOOK_TITLE: toc.title,
        StateKey.BOOKMARK_CHAPTER_ID: progress.bookmark_chapter_id,
        StateKey.DOC_TOC: toc_view(toc, progress),
        StateKey.SAVED_BOOKSHELF_OVERVIEW: bookshelf,
    }
