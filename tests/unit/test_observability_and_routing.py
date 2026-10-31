"""Tests for PII redaction, structured JSON logging, intent classification, and routing."""

from __future__ import annotations

import json
import logging
import sys
from unittest.mock import MagicMock

import pytest
from google.adk.models import LlmRequest, LlmResponse
from google.genai import types
from opentelemetry import trace
from opentelemetry.trace import NonRecordingSpan, SpanContext

from app.observability import (
    AuditTrailPlugin,
    JsonLogFormatter,
    PiiRedactionFilter,
    PiiRedactor,
    ReaderIntent,
    classify_intent,
)
from app.routing import (
    DEFAULT_TIER_MODELS,
    LONG_CHAPTER_WORD_THRESHOLD,
    ModelRouter,
    ModelTier,
)
from app.state import StateKey


def test_redactor_scrubs_common_pii_and_secrets() -> None:
    text = (
        "Mail reader@example.com or call (415) 555-0134. SSN 123-45-6789, "
        "card 4111 1111 1111 1111, Authorization: Bearer abc.def-123, "
        "api_key=sk_live_42, file /usr/local/google/home/minaml/notes.pdf"
    )

    redacted = PiiRedactor().redact_text(text)

    for leaked in (
        "reader@example.com",
        "555-0134",
        "123-45-6789",
        "4111 1111 1111 1111",
        "abc.def-123",
        "sk_live_42",
        "minaml",
    ):
        assert leaked not in redacted
    for marker in ("EMAIL", "PHONE", "SSN", "CARD", "SECRET", "USER"):
        assert f"[REDACTED_{marker}" in redacted


def test_redactor_preserves_reading_identifiers() -> None:
    text = "Summarize ch02 of book_a1b2c3d4e5f6 [p. 12] pages 3-14 on 2026-10-31."
    assert PiiRedactor().redact_text(text) == text


def test_redactor_handles_nested_payloads_and_sensitive_keys() -> None:
    payload = {"password": "hunter2", "notes": ["ping me at a@b.io"], "page": 4}

    redacted = PiiRedactor().redact(payload)

    assert redacted == {
        "password": "[REDACTED_SECRET]",
        "notes": ["ping me at [REDACTED_EMAIL]"],
        "page": 4,
    }


def test_message_redactor_keeps_local_pdf_paths_resolvable() -> None:
    path = "/home/reader/books/guide.pdf"
    assert PiiRedactor(redact_paths=False).redact_text(path) == path


def test_json_formatter_emits_redacted_structured_record_with_trace_ids() -> None:
    record = logging.LogRecord(
        name="book_companion.audit",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="outcome for %s",
        args=("reader@example.com",),
        exc_info=None,
    )
    record.json_fields = {
        "event_type": "outcome",
        "intent": "read_chapter",
        "who": "a@b.io",
    }
    span = NonRecordingSpan(SpanContext(trace_id=0xABC, span_id=0xDEF, is_remote=False))

    assert PiiRedactionFilter().filter(record)
    with trace.use_span(span):
        payload = json.loads(JsonLogFormatter().format(record))

    assert payload["severity"] == "INFO"
    assert payload["logger"] == "book_companion.audit"
    assert payload["message"] == "outcome for [REDACTED_EMAIL]"
    assert payload["event_type"] == "outcome"
    assert payload["intent"] == "read_chapter"
    assert payload["who"] == "[REDACTED_EMAIL]"
    assert payload["trace_id"] == f"{0xABC:032x}"
    assert payload["span_id"] == f"{0xDEF:016x}"


def test_json_formatter_redacts_exception_tracebacks() -> None:
    try:
        raise RuntimeError("token=sk_live_42 for reader@example.com")
    except RuntimeError:
        record = logging.LogRecord(
            name="app",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="ingest failed",
            args=None,
            exc_info=sys.exc_info(),
        )

    assert PiiRedactionFilter().filter(record)
    payload = json.loads(JsonLogFormatter().format(record))

    assert "RuntimeError" in payload["exception"]
    assert "sk_live_42" not in payload["exception"]
    assert "reader@example.com" not in payload["exception"]


def _invocation_context(invocation_id: str) -> MagicMock:
    context = MagicMock()
    context.invocation_id = invocation_id
    context.agent.name = "book_companion"
    context.user_content = types.Content(parts=[types.Part(text="next")])
    context.session.state = {}
    context.session.id = "session"
    context.user_id = "reader"
    return context


@pytest.mark.asyncio
async def test_audit_trail_records_model_served_after_routing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Agent-level routing runs after the plugin's before_model_callback."""
    plugin = AuditTrailPlugin()
    invocation = _invocation_context("inv-1")
    callback_context = MagicMock(invocation_id="inv-1")
    request = LlmRequest(model="gemini-standard")

    await plugin.before_run_callback(invocation_context=invocation)
    await plugin.before_model_callback(
        callback_context=callback_context, llm_request=request
    )
    request.model = "gemini-reasoning"  # Simulates escalate_long_chapters.
    await plugin.after_model_callback(
        callback_context=callback_context, llm_response=LlmResponse()
    )
    invocation.session.events = []
    with caplog.at_level(logging.INFO, logger="book_companion.audit"):
        await plugin.after_run_callback(invocation_context=invocation)

    outcome = next(r for r in caplog.records if r.getMessage() == "outcome")
    assert outcome.json_fields["models_used"] == ["gemini-reasoning"]


@pytest.mark.asyncio
async def test_audit_trail_bounds_tracked_runs() -> None:
    plugin = AuditTrailPlugin()
    limit = plugin._MAX_TRACKED_RUNS

    for index in range(limit + 5):
        await plugin.before_run_callback(
            invocation_context=_invocation_context(f"inv-{index}")
        )

    assert len(plugin._runs) == limit
    assert "inv-0" not in plugin._runs
    assert f"inv-{limit + 4}" in plugin._runs


@pytest.mark.parametrize(
    ("text", "quiz_pending", "expected"),
    [
        ('[Uploaded Artifact: "guide.pdf"] read this', False, ReaderIntent.LOAD_BOOK),
        ("next", False, ReaderIntent.READ_CHAPTER),
        ("check-in please", False, ReaderIntent.REQUEST_CHECK_IN),
        ("deep dive quorum leases", False, ReaderIntent.DEEP_DIVE),
        ("B) strict majority", True, ReaderIntent.ANSWER_CHECK_IN),
        ("Please delete this book", False, ReaderIntent.DELETE_BOOK),
        ("I want to start over", False, ReaderIntent.RESET_PROGRESS),
        ("What's on my bookshelf?", False, ReaderIntent.BROWSE_BOOKSHELF),
        ("hello", False, ReaderIntent.GENERAL),
    ],
)
def test_classify_intent(text: str, quiz_pending: bool, expected: ReaderIntent) -> None:
    assert classify_intent(text, quiz_pending=quiz_pending) is expected


def test_router_uses_distinct_models_per_tier_with_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert len(set(DEFAULT_TIER_MODELS.values())) == len(ModelTier)

    monkeypatch.setenv("BOOK_COMPANION_FAST_MODEL", "gemini-custom-lite")
    router = ModelRouter.from_env()

    assert router.model_name(ModelTier.FAST) == "gemini-custom-lite"
    assert (
        router.model_name(ModelTier.STANDARD) == DEFAULT_TIER_MODELS[ModelTier.STANDARD]
    )
    assert (
        router.model_name(ModelTier.REASONING)
        == DEFAULT_TIER_MODELS[ModelTier.REASONING]
    )


@pytest.mark.parametrize(
    ("words", "expected_tier"),
    [(100, ModelTier.STANDARD), (LONG_CHAPTER_WORD_THRESHOLD + 1, ModelTier.REASONING)],
)
def test_long_chapters_escalate_to_reasoning_tier(
    words: int, expected_tier: ModelTier
) -> None:
    router = ModelRouter(DEFAULT_TIER_MODELS)
    callback_context = MagicMock()
    callback_context.state = {StateKey.ACTIVE_CHAPTER_MATERIAL: "word " * words}
    callback_context.agent_name = "narrative_summarizer"
    request = LlmRequest(model=router.model_name(ModelTier.STANDARD))

    assert router.escalate_long_chapters(callback_context, request) is None
    assert request.model == router.model_name(expected_tier)
