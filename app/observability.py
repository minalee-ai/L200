"""Structured JSON logging, PII redaction, and intent-vs-outcome audit telemetry.

Three layers, all wired in `app/agent.py`:

1. `PiiRedactor` scrubs emails, phone numbers, SSNs, payment cards, secrets,
   and local home-directory paths from free text and nested payloads.
2. `configure_structured_logging()` installs a `JsonLogFormatter` (one JSON
   object per line, Cloud Logging `severity` plus OpenTelemetry `trace_id` /
   `span_id`) behind a `PiiRedactionFilter`, so nothing reaches stdout unredacted.
3. `AuditTrailPlugin` records, for every run, the reader's classified *intent*
   at the start and the observed *outcome* at the end (status, latency, tools,
   models, confirmation state), and redacts PII from user messages before they
   reach the model or session storage.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools import BaseTool, ToolContext
from google.genai import types
from opentelemetry import trace

from app.state import StateKey

_AUDIT_LOGGER = logging.getLogger("book_companion.audit")
_MAX_LOGGED_VALUE_CHARS = 240
# Marks the handler installed by configure_structured_logging (idempotency).
_HANDLER_MARKER = "_book_companion_json_handler"


# --- PII redaction -----------------------------------------------------------


def _passes_luhn(digits: str) -> bool:
    """Returns True for card-number-shaped digit strings with a valid checksum."""
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value = value * 2 - 9 if value > 4 else value * 2
        total += value
    return total % 10 == 0


class PiiRedactor:
    """Replaces personal data and credentials with typed placeholders.

    Args:
        redact_paths: Also mask user home directories (`/home/<user>`). Enabled
            for logs; disabled for user messages, where a reader may legitimately
            reference a local PDF path the ingestion tool must resolve.
    """

    _SENSITIVE_KEYS = frozenset(
        {"password", "passwd", "secret", "token", "api_key", "apikey", "authorization"}
    )
    _CARD_RE = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
    _PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
        (re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/-]+=*"), "[REDACTED_SECRET]"),
        (re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"), "[REDACTED_SECRET]"),
        (
            re.compile(r"(?i)\b(api[_-]?key|token|secret|password)\b(\s*[:=]\s*)\S+"),
            r"\1\2[REDACTED_SECRET]",
        ),
        (re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"), "[REDACTED_EMAIL]"),
        (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
        (
            re.compile(r"(?<!\w)(?:\+?\d{1,2}[ .-])?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}\b"),
            "[REDACTED_PHONE]",
        ),
    )
    _HOME_PATH_RE = re.compile(r"(/(?:usr/local/google/)?home/|/Users/)[^/\s]+")

    def __init__(self, *, redact_paths: bool = True) -> None:
        self._redact_paths = redact_paths

    def redact_text(self, text: str) -> str:
        """Returns `text` with every detected PII span replaced."""
        redacted = self._CARD_RE.sub(self._mask_card, text)
        for pattern, replacement in self._PATTERNS:
            redacted = pattern.sub(replacement, redacted)
        if self._redact_paths:
            redacted = self._HOME_PATH_RE.sub(r"\1[REDACTED_USER]", redacted)
        return redacted

    def redact(self, value: Any) -> Any:
        """Recursively redacts strings inside dicts, lists, and tuples."""
        if isinstance(value, str):
            return self.redact_text(value)
        if isinstance(value, Mapping):
            return {
                key: "[REDACTED_SECRET]"
                if str(key).lower() in self._SENSITIVE_KEYS
                else self.redact(item)
                for key, item in value.items()
            }
        if isinstance(value, list | tuple):
            return [self.redact(item) for item in value]
        return value

    @staticmethod
    def _mask_card(match: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", match.group())
        return "[REDACTED_CARD]" if _passes_luhn(digits) else match.group()


_LOG_REDACTOR = PiiRedactor()


class PiiRedactionFilter(logging.Filter):
    """Logging filter that redacts the rendered message and structured fields."""

    def __init__(self, redactor: PiiRedactor | None = None) -> None:
        super().__init__()
        self._redactor = redactor or _LOG_REDACTOR

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self._redactor.redact_text(record.getMessage())
        record.args = None
        fields = getattr(record, "json_fields", None)
        if isinstance(fields, Mapping):
            record.json_fields = self._redactor.redact(fields)
        return True


# --- Structured JSON logging -------------------------------------------------


class JsonLogFormatter(logging.Formatter):
    """Formats records as single-line JSON compatible with Cloud Logging.

    Structured payloads are passed with `extra={"json_fields": {...}}` and are
    merged into the top-level object. The active OpenTelemetry span is attached
    so log lines join the Cloud Trace view of the same request.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "severity": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        span_context = trace.get_current_span().get_span_context()
        if span_context.is_valid:
            payload["trace_id"] = format(span_context.trace_id, "032x")
            payload["span_id"] = format(span_context.span_id, "016x")
        fields = getattr(record, "json_fields", None)
        if isinstance(fields, Mapping):
            payload.update(fields)
        if record.exc_info:
            # Tracebacks are rendered after filters run, so redact them here.
            payload["exception"] = _LOG_REDACTOR.redact_text(
                self.formatException(record.exc_info)
            )
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_structured_logging(level: str | None = None) -> None:
    """Routes root logging through a redacting JSON handler (safe to call repeatedly)."""
    root = logging.getLogger()
    if any(getattr(handler, _HANDLER_MARKER, False) for handler in root.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter())
    handler.addFilter(PiiRedactionFilter())
    setattr(handler, _HANDLER_MARKER, True)
    root.addHandler(handler)
    root.setLevel(level or os.environ.get("BOOK_COMPANION_LOG_LEVEL", "INFO"))


# --- Intent vs. outcome auditing ---------------------------------------------


class ReaderIntent(StrEnum):
    """What the reader is trying to accomplish in one turn."""

    LOAD_BOOK = "load_book"
    BROWSE_BOOKSHELF = "browse_bookshelf"
    READ_CHAPTER = "read_chapter"
    DEEP_DIVE = "deep_dive"
    REQUEST_CHECK_IN = "request_check_in"
    ANSWER_CHECK_IN = "answer_check_in"
    RESET_PROGRESS = "reset_progress"
    DELETE_BOOK = "delete_book"
    CONFIRMATION_REPLY = "confirmation_reply"
    GENERAL = "general"


class OutcomeStatus(StrEnum):
    """How a turn actually ended."""

    SUCCESS = "success"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    REJECTED_BY_READER = "rejected_by_reader"
    TOOL_ERROR = "tool_error"
    FAILED = "failed"


# First matching rule wins; ordered from most to least specific.
_INTENT_RULES: tuple[tuple[ReaderIntent, re.Pattern[str]], ...] = (
    (
        ReaderIntent.DELETE_BOOK,
        re.compile(
            r"\b(delete|remove)\b.*\bbook\b|\bbook\b.*\b(delete|remove)\b", re.I
        ),
    ),
    (
        ReaderIntent.RESET_PROGRESS,
        re.compile(r"\breset\b|\bstart (?:it )?over\b|\brestart\b", re.I),
    ),
    (
        ReaderIntent.LOAD_BOOK,
        re.compile(r"\.pdf\b|uploaded artifact|\bload\b|\bupload", re.I),
    ),
    (ReaderIntent.REQUEST_CHECK_IN, re.compile(r"\bcheck-?in\b|\bquiz\b", re.I)),
    (
        ReaderIntent.DEEP_DIVE,
        re.compile(r"\bdeep dive\b|\bexplain\b|\bwhy\b|\bhow\b", re.I),
    ),
    (
        ReaderIntent.READ_CHAPTER,
        re.compile(r"\bnext\b|\bch\d{2,}\b|\bchapter\b|\bsummar", re.I),
    ),
    (
        ReaderIntent.BROWSE_BOOKSHELF,
        re.compile(r"\bbookshelf\b|\blibrary\b|\bcontinue\b|\bbooks\b", re.I),
    ),
)

# AgentTool sub-runs start their own invocation; their intent is the workflow.
_WORKFLOW_INTENTS: Mapping[str, ReaderIntent] = {
    "summarize_chapter": ReaderIntent.READ_CHAPTER,
    "generate_chapter_quiz": ReaderIntent.REQUEST_CHECK_IN,
    "grade_quiz_answers": ReaderIntent.ANSWER_CHECK_IN,
}


def classify_intent(
    text: str,
    *,
    quiz_pending: bool = False,
    is_function_response: bool = False,
) -> ReaderIntent:
    """Maps a reader message to its declared intent with lightweight rules."""
    if is_function_response:
        return ReaderIntent.CONFIRMATION_REPLY
    for intent, pattern in _INTENT_RULES:
        if pattern.search(text):
            return intent
    if quiz_pending and text.strip():
        return ReaderIntent.ANSWER_CHECK_IN
    return ReaderIntent.GENERAL


@dataclass
class _RunRecord:
    """Per-invocation telemetry accumulated between intent and outcome."""

    intent: ReaderIntent
    started: float = field(default_factory=time.perf_counter)
    tools: list[str] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    tool_errors: list[str] = field(default_factory=list)
    awaiting_confirmation: bool = False
    rejected: bool = False
    failure: str | None = None
    # The in-flight request; agent-level routing may rewrite its `model` after
    # the plugin's before_model_callback has run.
    active_request: LlmRequest | None = None


def _user_ref(user_id: str) -> str:
    """Pseudonymous, stable reference to a user (raw IDs never hit the logs)."""
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:12]


def _truncate(value: Any) -> Any:
    if isinstance(value, str) and len(value) > _MAX_LOGGED_VALUE_CHARS:
        return value[:_MAX_LOGGED_VALUE_CHARS] + "…"
    return value


class AuditTrailPlugin(BasePlugin):
    """App-wide plugin pairing each run's declared intent with its observed outcome.

    Every callback returns None, so the plugin never short-circuits ADK or the
    plugins registered after it. Records are keyed by invocation ID; `AgentTool`
    sub-runs inherit the plugin and are audited as their own invocations.
    """

    # Upper bound on tracked runs: a client that disconnects mid-stream cancels
    # the run without reaching after_run/on_run_error, so stale records are
    # evicted oldest-first instead of accumulating for the server's lifetime.
    _MAX_TRACKED_RUNS = 512

    def __init__(self, name: str = "audit_trail") -> None:
        super().__init__(name)
        self._message_redactor = PiiRedactor(redact_paths=False)
        self._runs: dict[str, _RunRecord] = {}

    async def on_user_message_callback(
        self, *, invocation_context: InvocationContext, user_message: types.Content
    ) -> types.Content | None:
        # Redact in place (instead of returning new content) so the next
        # plugin, SaveFilesAsArtifactsPlugin, still sees the message.
        redacted_parts = 0
        for part in user_message.parts or []:
            if part.text:
                clean = self._message_redactor.redact_text(part.text)
                if clean != part.text:
                    part.text = clean
                    redacted_parts += 1
        if redacted_parts:
            self._log(
                "pii_redacted",
                invocation_context.invocation_id,
                source="user_message",
                parts_redacted=redacted_parts,
            )
        return None

    async def before_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> types.Content | None:
        agent_name = invocation_context.agent.name
        content = invocation_context.user_content
        parts = content.parts if content and content.parts else []
        text = " ".join(part.text for part in parts if part.text)
        intent = _WORKFLOW_INTENTS.get(agent_name) or classify_intent(
            text,
            quiz_pending=bool(
                invocation_context.session.state.get(StateKey.PENDING_QUIZ_CHAPTER_ID)
            ),
            is_function_response=any(part.function_response for part in parts),
        )
        while len(self._runs) >= self._MAX_TRACKED_RUNS:
            self._runs.pop(next(iter(self._runs)))
        self._runs[invocation_context.invocation_id] = _RunRecord(intent=intent)
        trace.get_current_span().set_attribute("book_companion.intent", intent.value)
        self._log(
            "intent",
            invocation_context.invocation_id,
            intent=intent.value,
            entry_agent=agent_name,
            user_ref=_user_ref(invocation_context.user_id),
            session_id=invocation_context.session.id,
            active_book_id=invocation_context.session.state.get(
                StateKey.ACTIVE_BOOK_ID
            ),
            request_preview=_truncate(text),
        )
        return None

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> LlmResponse | None:
        # Plugin callbacks run before agent callbacks, so the model is recorded
        # in after_model_callback, once routing has had its say.
        if record := self._runs.get(callback_context.invocation_id):
            record.active_request = llm_request
        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> LlmResponse | None:
        if record := self._runs.get(callback_context.invocation_id):
            record.models.append(self._served_model(record, llm_response))
        return None

    async def on_model_error_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
        error: Exception,
    ) -> LlmResponse | None:
        if record := self._runs.get(callback_context.invocation_id):
            record.models.append(llm_request.model or "unknown")
            record.failure = f"model_error:{type(error).__name__}"
        return None

    async def before_tool_callback(
        self, *, tool: BaseTool, tool_args: dict[str, Any], tool_context: ToolContext
    ) -> dict[str, Any] | None:
        if record := self._runs.get(tool_context.invocation_id):
            record.tools.append(tool.name)
        self._log(
            "tool_intent",
            tool_context.invocation_id,
            tool=tool.name,
            args={key: _truncate(value) for key, value in tool_args.items()},
        )
        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        # AgentTool workflows return plain text rather than a dict.
        payload: Mapping[str, Any] = result if isinstance(result, Mapping) else {}
        status = self._tool_status(tool_context, payload)
        record = self._runs.get(tool_context.invocation_id)
        if record:
            if status is OutcomeStatus.AWAITING_CONFIRMATION:
                record.awaiting_confirmation = True
            elif status is OutcomeStatus.REJECTED_BY_READER:
                record.rejected = True
            elif status is OutcomeStatus.TOOL_ERROR:
                record.tool_errors.append(tool.name)
        self._log(
            "tool_outcome",
            tool_context.invocation_id,
            tool=tool.name,
            status=status.value,
            detail=_truncate(payload.get("message") or payload.get("error")),
        )
        return None

    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        error: Exception,
    ) -> dict[str, Any] | None:
        if record := self._runs.get(tool_context.invocation_id):
            record.tool_errors.append(tool.name)
        self._log(
            "tool_outcome",
            tool_context.invocation_id,
            tool=tool.name,
            status=OutcomeStatus.FAILED.value,
            detail=f"{type(error).__name__}: {_truncate(str(error))}",
            level=logging.ERROR,
        )
        return None

    async def on_run_error_callback(
        self, *, invocation_context: InvocationContext, error: Exception
    ) -> None:
        if record := self._runs.get(invocation_context.invocation_id):
            record.failure = f"run_error:{type(error).__name__}"
        self._emit_outcome(invocation_context)

    async def after_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> None:
        self._emit_outcome(invocation_context)

    def _emit_outcome(self, invocation_context: InvocationContext) -> None:
        record = self._runs.pop(invocation_context.invocation_id, None)
        if record is None:
            return
        status = self._run_status(record)
        trace.get_current_span().set_attribute("book_companion.outcome", status.value)
        self._log(
            "outcome",
            invocation_context.invocation_id,
            level=logging.ERROR if status is OutcomeStatus.FAILED else logging.INFO,
            intent=record.intent.value,
            status=status.value,
            latency_ms=round((time.perf_counter() - record.started) * 1000),
            tools_invoked=record.tools,
            models_used=sorted(set(record.models)),
            tool_errors=record.tool_errors,
            failure=record.failure,
            response_preview=_truncate(self._final_text(invocation_context)),
        )

    @staticmethod
    def _served_model(record: _RunRecord, llm_response: LlmResponse) -> str:
        """Prefers the backend-reported version; falls back to the routed request."""
        if llm_response.model_version:
            return llm_response.model_version
        if record.active_request is not None:
            return record.active_request.model or "unknown"
        return "unknown"

    @staticmethod
    def _run_status(record: _RunRecord) -> OutcomeStatus:
        if record.failure:
            return OutcomeStatus.FAILED
        if record.awaiting_confirmation:
            return OutcomeStatus.AWAITING_CONFIRMATION
        if record.rejected:
            return OutcomeStatus.REJECTED_BY_READER
        if record.tool_errors:
            return OutcomeStatus.TOOL_ERROR
        return OutcomeStatus.SUCCESS

    @staticmethod
    def _tool_status(
        tool_context: ToolContext, result: Mapping[str, Any]
    ) -> OutcomeStatus:
        if tool_context.actions.requested_tool_confirmations:
            return OutcomeStatus.AWAITING_CONFIRMATION
        error = str(result.get("error") or "")
        if "rejected" in error:
            return OutcomeStatus.REJECTED_BY_READER
        if error or result.get("status") == "error":
            return OutcomeStatus.TOOL_ERROR
        return OutcomeStatus.SUCCESS

    @staticmethod
    def _final_text(invocation_context: InvocationContext) -> str:
        for event in reversed(invocation_context.session.events):
            if event.invocation_id != invocation_context.invocation_id:
                break
            if event.author != "user" and event.content and event.content.parts:
                text = " ".join(p.text for p in event.content.parts if p.text)
                if text.strip():
                    return text
        return ""

    @staticmethod
    def _log(
        event_type: str, invocation_id: str, *, level: int = logging.INFO, **fields: Any
    ) -> None:
        # Fields are redacted here and again by PiiRedactionFilter on the JSON
        # handler, so audit data is clean for every sink, not just stdout.
        _AUDIT_LOGGER.log(
            level,
            event_type,
            extra={
                "json_fields": _LOG_REDACTOR.redact(
                    {"event_type": event_type, "invocation_id": invocation_id, **fields}
                )
            },
        )
