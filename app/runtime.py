"""Process-wide dependencies shared by tools, callbacks, and workflow agents."""

from __future__ import annotations

import threading

from google.adk.models import Gemini
from google.genai import types

from app.library.companion_renderer import BookCompanionRenderer
from app.library.models import BookProgress, BookToc
from app.library.store import LibraryStore

COMPANION_ARTIFACT_FILENAME = "book_companion.html"
# Upper bound on chunks returned by search_book_phrases per call (context hygiene).
MAX_SEARCH_LIMIT = 8
_MODEL_RETRY_ATTEMPTS = 3

_store_lock = threading.Lock()
_default_store: LibraryStore | None = None
_renderer = BookCompanionRenderer()


def get_library_store() -> LibraryStore:
    """Returns the active LibraryStore singleton, initializing lazily on first use."""
    global _default_store
    # Tools reach this from worker threads, so guard against double initialization
    # (two stores would mean two progress locks).
    with _store_lock:
        if _default_store is None:
            _default_store = LibraryStore()
        return _default_store


def set_library_store(store: LibraryStore | None) -> None:
    """Overrides the active LibraryStore (used by hermetic unit and integration tests)."""
    global _default_store
    with _store_lock:
        _default_store = store


def make_model(model_name: str) -> Gemini:
    """Builds a Gemini model with the project's standard retry policy."""
    return Gemini(
        model=model_name,
        retry_options=types.HttpRetryOptions(attempts=_MODEL_RETRY_ATTEMPTS),
    )


def render_companion_part(toc: BookToc, progress: BookProgress) -> types.Part:
    """Renders book_companion.html as an artifact-ready `types.Part`."""
    html_bytes = _renderer.render_html(toc, progress).encode("utf-8")
    return types.Part(inline_data=types.Blob(mime_type="text/html", data=html_bytes))
