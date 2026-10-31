"""Root-agent lifecycle callbacks: state bootstrap, memory sync, and tool guardrails."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import BaseTool, ToolContext

from app.runtime import MAX_SEARCH_LIMIT, get_library_store
from app.state import StateKey

logger = logging.getLogger(__name__)

_ROOT_PROMPT_DEFAULTS: dict[str, Any] = {
    StateKey.ACTIVE_BOOK_ID: "",
    StateKey.ACTIVE_BOOK_TITLE: "",
    StateKey.ACTIVE_CHAPTER_ID: "",
    StateKey.BOOKMARK_CHAPTER_ID: "",
    StateKey.PENDING_QUIZ_CHAPTER_ID: "",
    StateKey.DOC_TOC: [],
}
_CHAPTER_SCOPED_TOOLS = frozenset(
    {"get_chapter_chunks", "record_progress", "update_book_companion_artifact"}
)
_BOOK_SCOPED_TOOLS = frozenset({"reset_book_progress", "delete_saved_book"})


async def initialize_reader_state_callback(
    callback_context: CallbackContext,
) -> None:
    """Initializes root-prompt state keys and loads the bookshelf once per session."""
    state = callback_context.state
    for key, empty_value in _ROOT_PROMPT_DEFAULTS.items():
        if key not in state:
            state[key] = empty_value

    if StateKey.SAVED_BOOKSHELF_OVERVIEW not in state:
        # Scanned once per session; tools keep it current via reader_state_delta().
        state[StateKey.SAVED_BOOKSHELF_OVERVIEW] = await asyncio.to_thread(
            get_library_store().list_books, callback_context.user_id
        )


async def save_session_memory_callback(
    callback_context: CallbackContext,
) -> None:
    """Persists session events to ADK MemoryService when configured on the Runner."""
    try:
        await callback_context.add_session_to_memory()
    except ValueError:
        logger.debug("MemoryService not attached to Runner; skipping memory sync.")


async def validate_tool_inputs_callback(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
) -> dict[str, Any] | None:
    """Guardrail validating tool inputs before execution to prevent bad state or path abuse."""
    del tool_context
    name = tool.name

    if name == "ingest_or_load_book":
        source = str(args.get("source", "")).strip()
        if ".." in Path(source).parts:
            return {
                "status": "error",
                "message": "Relative parent traversal ('..') is not allowed in file paths.",
            }
        if (
            source
            and not source.startswith("book_")
            and source.lower() not in {"latest", "uploaded"}
            and "." in Path(source).name
            and not source.lower().endswith(".pdf")
        ):
            return {
                "status": "error",
                "message": "Only .pdf documents are supported by BookCompanion.",
            }

    if name == "search_book_phrases":
        if not str(args.get("query", "")).strip():
            return {"status": "error", "message": "Search query must not be empty."}
        raw_limit = args.get("limit", 5)
        if isinstance(raw_limit, int):
            args["limit"] = max(1, min(raw_limit, MAX_SEARCH_LIMIT))

    if name in _CHAPTER_SCOPED_TOOLS | _BOOK_SCOPED_TOOLS:
        if not str(args.get("doc_id", "")).strip():
            return {"status": "error", "message": "doc_id is required."}
    if name in _CHAPTER_SCOPED_TOOLS:
        if not str(args.get("chapter_id", "")).strip():
            return {"status": "error", "message": "chapter_id is required."}

    return None
