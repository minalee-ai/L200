"""Human-in-the-loop confirmation tests: destructive tools pause until the reader approves."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Any

import pytest
from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.events import Event
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.genai import types
from pydantic import Field

from app.library.chunker import HierarchicalPdfChunker
from app.library.models import BookToc
from app.library.store import LibraryStore
from app.observability import AuditTrailPlugin
from app.tools import delete_saved_book, reset_book_progress
from tests.unit.conftest import build_bookmarked_pdf_bytes

_APP = "hitl_app"
_USER = "reader_test"
_CONFIRMATION_CALL = "adk_request_confirmation"


class ToolCallingLlm(BaseLlm):
    """Fake LLM that calls one tool on its first turn and replies with text afterwards."""

    model: str = "tool-calling-llm"
    tool_name: str
    tool_args: dict[str, Any] = Field(default_factory=dict)
    calls: int = 0

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del llm_request, stream
        self.calls += 1
        part = (
            types.Part(
                function_call=types.FunctionCall(
                    name=self.tool_name, args=self.tool_args
                )
            )
            if self.calls == 1
            else types.Part(text="All set.")
        )
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


@pytest.fixture
def saved_book(isolated_store: LibraryStore) -> BookToc:
    pdf_bytes = build_bookmarked_pdf_bytes()
    toc, chunks = HierarchicalPdfChunker().chunk_pdf(
        pdf_bytes,
        doc_id=isolated_store.compute_doc_id(pdf_bytes),
        fallback_title="guide.pdf",
    )
    isolated_store.save_book(toc, chunks)
    return toc


class _ConfirmationHarness:
    """Runs a single-tool agent inside a resumable App with the audit plugin."""

    def __init__(self, tool: FunctionTool, llm: ToolCallingLlm) -> None:
        agent = Agent(name="book_companion", model=llm, tools=[tool])
        self.sessions = InMemorySessionService()
        self.runner = Runner(
            app=App(
                name=_APP,
                root_agent=agent,
                plugins=[AuditTrailPlugin()],
                resumability_config=ResumabilityConfig(is_resumable=True),
            ),
            session_service=self.sessions,
        )

    async def run(self, session_id: str, message: types.Content) -> list[Event]:
        return [
            event
            async for event in self.runner.run_async(
                user_id=_USER, session_id=session_id, new_message=message
            )
        ]


def _confirmation_request_id(events: list[Event]) -> str:
    for event in events:
        for call in event.get_function_calls():
            if call.name == _CONFIRMATION_CALL and call.id:
                return call.id
    raise AssertionError("expected an adk_request_confirmation event")


def _confirmation_reply(request_id: str, *, confirmed: bool) -> types.Content:
    return types.Content(
        role="user",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    id=request_id,
                    name=_CONFIRMATION_CALL,
                    response={"confirmed": confirmed},
                )
            )
        ],
    )


def _audit_events(caplog: pytest.LogCaptureFixture, event_type: str) -> list[dict]:
    return [
        record.json_fields
        for record in caplog.records
        if record.name == "book_companion.audit"
        and record.json_fields.get("event_type") == event_type
    ]


@pytest.mark.asyncio
async def test_delete_pauses_for_confirmation_then_runs_once_approved(
    saved_book: BookToc,
    isolated_store: LibraryStore,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="book_companion.audit")
    llm = ToolCallingLlm(
        tool_name="delete_saved_book", tool_args={"doc_id": saved_book.doc_id}
    )
    harness = _ConfirmationHarness(
        FunctionTool(delete_saved_book, require_confirmation=True), llm
    )
    session = await harness.sessions.create_session(app_name=_APP, user_id=_USER)

    paused = await harness.run(
        session.id,
        types.Content(
            role="user",
            parts=[
                types.Part(text="Delete this book and email me at reader@example.com")
            ],
        ),
    )

    request_id = _confirmation_request_id(paused)
    assert isolated_store.has_book(saved_book.doc_id), "must not delete before approval"
    [paused_outcome] = _audit_events(caplog, "outcome")
    assert paused_outcome["intent"] == "delete_book"
    assert paused_outcome["status"] == "awaiting_confirmation"
    stored = await harness.sessions.get_session(
        app_name=_APP, user_id=_USER, session_id=session.id
    )
    assert stored is not None
    user_text = stored.events[0].content.parts[0].text or ""
    assert "reader@example.com" not in user_text
    assert "[REDACTED_EMAIL]" in user_text

    caplog.clear()
    await harness.run(session.id, _confirmation_reply(request_id, confirmed=True))

    assert not isolated_store.has_book(saved_book.doc_id)
    outcome = _audit_events(caplog, "outcome")[-1]
    # A resumed run keeps its invocation, so the original intent is preserved.
    assert outcome["intent"] == "delete_book"
    assert outcome["status"] == "success"
    assert "delete_saved_book" in outcome["tools_invoked"]
    assert outcome["response_preview"] == "All set."


@pytest.mark.asyncio
async def test_rejected_reset_leaves_progress_untouched(
    saved_book: BookToc,
    isolated_store: LibraryStore,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="book_companion.audit")
    isolated_store.update_progress(
        _USER, saved_book, lambda p: p.record_topic_result("Quorums", True)
    )
    llm = ToolCallingLlm(
        tool_name="reset_book_progress", tool_args={"doc_id": saved_book.doc_id}
    )
    harness = _ConfirmationHarness(
        FunctionTool(reset_book_progress, require_confirmation=True), llm
    )
    session = await harness.sessions.create_session(app_name=_APP, user_id=_USER)

    paused = await harness.run(
        session.id,
        types.Content(role="user", parts=[types.Part(text="Reset my progress")]),
    )
    await harness.run(
        session.id,
        _confirmation_reply(_confirmation_request_id(paused), confirmed=False),
    )

    progress = isolated_store.load_progress(_USER, saved_book.doc_id)
    assert progress is not None and "Quorums" in progress.topic_mastery
    assert _audit_events(caplog, "outcome")[-1]["status"] == "rejected_by_reader"
