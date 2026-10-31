"""Content-addressed persistent store for chunked books and reader progress."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypeVar

from app.library.models import BookChunk, BookProgress, BookToc

_SAFE_ID_RE = re.compile(r"[^a-zA-Z0-9_-]")

_T = TypeVar("_T")


def _resolve_default_store_dir() -> Path:
    """Resolves a writable local cache path outside the source repository."""
    if configured := os.environ.get("BOOK_COMPANION_STORE_DIR"):
        path = Path(configured).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path

    candidates = [
        Path.home() / ".cache" / "book_companion" / "library_store",
        Path(tempfile.gettempdir()) / "book_companion" / "library_store",
    ]
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except OSError:
            continue

    raise RuntimeError("Unable to initialize a writable directory for LibraryStore.")


class LibraryStore:
    """Persists parsed TOCs, phrase chunks, and per-user reading bookmarks on disk.

    Methods perform blocking file I/O; async callers should run them via
    `asyncio.to_thread`. Progress read-modify-write cycles are serialized with
    an in-process lock (see `update_progress`); multi-process deployments need
    an external store such as Firestore or Spanner instead.
    """

    def __init__(self, store_dir: Path | None = None) -> None:
        self._store_dir = (
            store_dir if store_dir is not None else _resolve_default_store_dir()
        )
        self._books_dir = self._store_dir / "books"
        self._progress_dir = self._store_dir / "progress"
        self._books_dir.mkdir(parents=True, exist_ok=True)
        self._progress_dir.mkdir(parents=True, exist_ok=True)
        # Re-entrant: update_progress holds it while calling save_progress.
        self._progress_lock = threading.RLock()

    @property
    def store_dir(self) -> Path:
        """Returns root directory backing this library store."""
        return self._store_dir

    def compute_doc_id(self, pdf_bytes: bytes) -> str:
        """Computes deterministic 12-char SHA-256 content hash for a PDF."""
        digest = hashlib.sha256(pdf_bytes).hexdigest()[:12]
        return f"book_{digest}"

    def has_book(self, doc_id: str) -> bool:
        """Returns True when both toc.json and chunks.json are already cached."""
        book_dir = self._book_dir(doc_id)
        return (book_dir / "toc.json").is_file() and (
            book_dir / "chunks.json"
        ).is_file()

    def save_book(self, toc: BookToc, chunks: Sequence[BookChunk]) -> None:
        """Atomically writes TOC metadata and chunk list for a parsed PDF."""
        book_dir = self._book_dir(toc.doc_id)
        book_dir.mkdir(parents=True, exist_ok=True)

        # Chunks first: has_book() requires both files, and toc.json is the
        # marker list_books() scans for, so a crash never exposes a partial book.
        self._write_json(
            book_dir / "chunks.json",
            [chunk.model_dump(mode="json") for chunk in chunks],
        )
        self._write_json(book_dir / "toc.json", toc.model_dump(mode="json"))

    def load_toc(self, doc_id: str) -> BookToc:
        """Loads saved BookToc for doc_id or raises FileNotFoundError."""
        toc_path = self._book_dir(doc_id) / "toc.json"
        if not toc_path.is_file():
            raise FileNotFoundError(
                f"Book '{doc_id}' was not found in the saved library."
            )
        payload = json.loads(toc_path.read_text(encoding="utf-8"))
        return BookToc.model_validate(payload)

    def load_chunks(
        self,
        doc_id: str,
        *,
        chapter_id: str | None = None,
    ) -> list[BookChunk]:
        """Loads saved chunks for doc_id, filtered optionally by chapter_id."""
        chunks_path = self._book_dir(doc_id) / "chunks.json"
        if not chunks_path.is_file():
            raise FileNotFoundError(f"Chunks for book '{doc_id}' were not found.")
        raw_items = json.loads(chunks_path.read_text(encoding="utf-8"))
        chunks = [BookChunk.model_validate(item) for item in raw_items]
        if chapter_id and chapter_id != "all":
            return [chunk for chunk in chunks if chunk.chapter_id == chapter_id]
        return chunks

    def load_progress(self, user_id: str, doc_id: str) -> BookProgress | None:
        """Returns saved reader progress for a book, or None if never saved."""
        progress_path = self._progress_path(user_id, doc_id)
        if not progress_path.is_file():
            return None
        payload = json.loads(progress_path.read_text(encoding="utf-8"))
        return BookProgress.model_validate(payload)

    def load_or_init_progress(self, user_id: str, toc: BookToc) -> BookProgress:
        """Returns saved progress, or a fresh unsaved bookmark at the first chapter."""
        return self.load_progress(user_id, toc.doc_id) or BookProgress.new(toc)

    def save_progress(self, user_id: str, progress: BookProgress) -> None:
        """Persists updated reading bookmark, mastery scores, and chapter digests.

        Prefer `update_progress` for read-modify-write cycles; this only
        serializes the write itself.
        """
        progress_path = self._progress_path(user_id, progress.doc_id)
        with self._progress_lock:
            self._write_json(progress_path, progress.model_dump(mode="json"))

    def update_progress(
        self,
        user_id: str,
        toc: BookToc,
        mutate: Callable[[BookProgress], _T],
    ) -> tuple[BookProgress, _T]:
        """Atomically loads, mutates, and saves a reader's progress for one book.

        Serializes concurrent read-modify-write cycles (for example, parallel
        `record_progress` calls running on worker threads) so no update is lost.

        Args:
            user_id: Reader whose progress is updated.
            toc: Book the progress belongs to.
            mutate: Callback that edits the progress in place; its return value
                is passed through.

        Returns:
            The saved progress and the value returned by `mutate`.
        """
        with self._progress_lock:
            progress = self.load_or_init_progress(user_id, toc)
            result = mutate(progress)
            self.save_progress(user_id, progress)
        return progress, result

    def list_books(self, user_id: str) -> list[dict[str, Any]]:
        """Lists cached books with this reader's bookmark and mastery (read-only)."""
        books: list[dict[str, Any]] = []
        if not self._books_dir.exists():
            return books

        for entry in sorted(self._books_dir.iterdir()):
            toc_file = entry / "toc.json"
            if not toc_file.is_file():
                continue
            toc = BookToc.model_validate(
                json.loads(toc_file.read_text(encoding="utf-8"))
            )
            progress = self.load_or_init_progress(user_id, toc)
            books.append(progress.bookshelf_entry(toc))
        return books

    def _book_dir(self, doc_id: str) -> Path:
        safe_doc = _SAFE_ID_RE.sub("_", doc_id)
        return self._books_dir / safe_doc

    def _progress_path(self, user_id: str, doc_id: str) -> Path:
        safe_user = _SAFE_ID_RE.sub("_", user_id or "reader")
        safe_doc = _SAFE_ID_RE.sub("_", doc_id)
        return self._progress_dir / f"{safe_user}__{safe_doc}.json"

    def _write_json(self, target: Path, data: Any) -> None:
        """Writes JSON via a unique temp file + atomic rename to avoid torn files."""
        fd, tmp_name = tempfile.mkstemp(
            dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, ensure_ascii=False)
            os.replace(tmp_name, target)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
