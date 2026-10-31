"""Deterministic (non-LLM) steps shared by the BookCompanion workflow pipelines.

`SequentialAgent` ignores `EventActions.escalate`, so a step that cannot
continue records a reader-facing message under `StateKey.PIPELINE_ERROR`, and
every downstream step carries a skip callback as its `before_agent_callback`:
`skip_if_pipeline_error` on the final step (re-emitting the error, so it becomes
the pipeline's response that `AgentTool` returns) and
`skip_silently_if_pipeline_error` on every step in between.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import AsyncGenerator, Callable, Mapping
from typing import Any

from google.adk.agents import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types
from pydantic import ValidationError

from app.library.models import (
    BookChunk,
    BookProgress,
    BookToc,
    QuestionGrade,
    QuizGradeReport,
    QuizSetDraft,
)
from app.runtime import (
    COMPANION_ARTIFACT_FILENAME,
    get_library_store,
    render_companion_part,
)
from app.state import QUIZ_RESET_DELTA, StateKey, reader_state_delta

logger = logging.getLogger(__name__)

_CHAPTER_ID_RE = re.compile(r"\b(ch\d{2,})\b", re.IGNORECASE)
_BOOK_ID_RE = re.compile(r"\b(book_[a-f0-9]{8,})\b", re.IGNORECASE)
MAX_CHAPTER_WORDS = 4000

_NO_BOOK_MESSAGE = (
    "No active book is loaded yet. Please attach a PDF or call "
    "`ingest_or_load_book` / `list_saved_books` first."
)
_QUIZ_UNAVAILABLE_MESSAGE = (
    "I couldn't build a check-in quiz that is fully grounded in this chapter's "
    "text. Reply **check-in** to try again, or **deep dive <topic>** to explore "
    "a passage instead."
)
_NO_PENDING_QUIZ_MESSAGE = (
    "There is no check-in quiz awaiting answers. Reply **check-in** to start one."
)
_QUIZ_BOOK_MISSING_MESSAGE = (
    "I can't grade this check-in because its book is no longer loaded, so nothing "
    "was recorded. Reload the book (or pick it from your bookshelf) and reply "
    "**check-in** to start a fresh quiz."
)


class PipelineInputError(Exception):
    """Raised by staging helpers; the message is shown to the reader verbatim."""


def skip_if_pipeline_error(callback_context: CallbackContext) -> types.Content | None:
    """Skips the pipeline's final step, re-emitting the error as its response."""
    message = callback_context.state.get(StateKey.PIPELINE_ERROR)
    return text_content(message) if message else None


def skip_silently_if_pipeline_error(
    callback_context: CallbackContext,
) -> types.Content | None:
    """Skips an intermediate step without emitting text when an earlier step failed.

    Used for LLM steps (ADK would otherwise parse callback text into their
    `output_key`, validating it against `output_schema`) and for intermediate
    workflow steps, so the reader sees the error exactly once.
    """
    if callback_context.state.get(StateKey.PIPELINE_ERROR):
        return types.Content(role="model", parts=[])
    return None


def text_content(text: str) -> types.Content:
    """Wraps plain text as a model-authored `types.Content`."""
    return types.Content(role="model", parts=[types.Part(text=text)])


def _error_event(author: str, message: str) -> Event:
    """Records a pipeline error in state without emitting text.

    The pipeline's final step (wired with `skip_if_pipeline_error`) surfaces the
    message, so the reader sees it exactly once even in streamed runs.
    """
    return Event(
        author=author,
        actions=EventActions(state_delta={StateKey.PIPELINE_ERROR: message}),
    )


def _request_text(ctx: InvocationContext) -> str:
    """Returns the request text passed into the nested workflow invocation."""
    if ctx.user_content and ctx.user_content.parts:
        return " ".join(p.text for p in ctx.user_content.parts if p.text)
    for event in reversed(ctx.session.events):
        if event.author == "user" and event.content and event.content.parts:
            return " ".join(p.text for p in event.content.parts if p.text)
    return ""


def parse_json_state(raw_value: Any) -> dict[str, Any] | None:
    """Returns a dict from a state value stored as a dict or a JSON string."""
    if isinstance(raw_value, dict):
        return raw_value
    if isinstance(raw_value, str) and raw_value.strip():
        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _page_label(page_start: int, page_end: int) -> str:
    return (
        f"p. {page_start}" if page_start == page_end else f"pp. {page_start}-{page_end}"
    )


def _format_chapter_material(
    chunks: list[BookChunk], max_words: int
) -> tuple[str, int]:
    """Formats chunks as cited source blocks within a word budget.

    Returns:
        The formatted material and how many leading chunks it includes.
    """
    blocks: list[str] = []
    word_total = 0
    for chunk in chunks:
        words = len(chunk.text.split())
        if blocks and word_total + words > max_words:
            break
        pages = _page_label(chunk.page_start, chunk.page_end)
        blocks.append(
            f"[{chunk.chunk_id} | {chunk.chapter_title} | {pages}]\n{chunk.text}"
        )
        word_total += words
    return "\n\n".join(blocks), len(blocks)


async def _persist_progress(
    ctx: InvocationContext,
    toc: BookToc,
    mutate: Callable[[BookProgress], Any],
) -> tuple[BookProgress, dict[str, int]]:
    """Applies `mutate` to the reader's progress, saves it, and refreshes the HTML artifact.

    Returns:
        The saved progress and an `artifact_delta` for the emitting event (empty
        when no ArtifactService is configured, e.g. headless eval runners).
    """
    progress, _ = await asyncio.to_thread(
        get_library_store().update_progress, ctx.user_id, toc, mutate
    )
    artifact_delta: dict[str, int] = {}
    if ctx.artifact_service is not None:
        try:
            version = await ctx.artifact_service.save_artifact(
                app_name=ctx.app_name,
                user_id=ctx.user_id,
                session_id=ctx.session.id,
                filename=COMPANION_ARTIFACT_FILENAME,
                artifact=render_companion_part(toc, progress),
            )
            artifact_delta[COMPANION_ARTIFACT_FILENAME] = version
        except ValueError:
            # AgentTool always injects a ForwardingArtifactService; it raises when
            # the parent Runner has none. Progress is already saved either way.
            logger.debug("ArtifactService not configured; skipped companion save.")
    return progress, artifact_delta


class ChapterChunkLoaderAgent(BaseAgent):
    """Resolves the target book and chapter and stages its source text in state.

    Chapter resolution order: an explicit `chNN` in the request (rejected if it
    does not exist), then the state key named by `default_chapter_key`, then the
    bookmark, then the book's first chapter.
    """

    default_chapter_key: str = StateKey.BOOKMARK_CHAPTER_ID

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        try:
            state_delta = await asyncio.to_thread(
                self._stage_chapter,
                _request_text(ctx),
                dict(ctx.session.state),
                ctx.user_id,
            )
        except PipelineInputError as err:
            yield _error_event(self.name, str(err))
            return
        yield Event(author=self.name, actions=EventActions(state_delta=state_delta))

    def _stage_chapter(
        self, request_text: str, state: Mapping[str, Any], user_id: str
    ) -> dict[str, Any]:
        """Loads chapter chunks and returns the state delta (blocking I/O)."""
        store = get_library_store()
        doc_id = self._resolve_doc_id(request_text, state, user_id)
        toc = store.load_toc(doc_id)
        chapter_id = self._resolve_chapter_id(toc, request_text, state)
        chunks = store.load_chunks(doc_id, chapter_id=chapter_id)
        if not chunks:
            raise PipelineInputError(f"No text chunks found for {chapter_id}.")

        material, included = _format_chapter_material(chunks, MAX_CHAPTER_WORDS)
        coverage_note = ""
        if included < len(chunks):
            coverage_note = (
                f"This chapter was too long to include in full: the source covers "
                f"pages {chunks[0].page_start}-{chunks[included - 1].page_end} of "
                f"pages {chunks[0].page_start}-{chunks[-1].page_end}."
            )

        return {
            StateKey.PIPELINE_ERROR: "",
            StateKey.ACTIVE_BOOK_ID: doc_id,
            StateKey.ACTIVE_BOOK_TITLE: toc.title,
            StateKey.ACTIVE_CHAPTER_ID: chapter_id,
            StateKey.ACTIVE_CHAPTER_TITLE: chunks[0].chapter_title,
            StateKey.ACTIVE_CHAPTER_MATERIAL: material,
            StateKey.ACTIVE_CHAPTER_COVERAGE_NOTE: coverage_note,
            # Reset downstream LLM outputs: ADK leaves output_key untouched when
            # a model returns only whitespace, which would replay stale values.
            StateKey.LATEST_CHAPTER_DIGEST: "",
            StateKey.QUIZ_DRAFT: {},
            StateKey.QUIZ_GROUNDING_FEEDBACK: "",
        }

    def _resolve_doc_id(
        self, request_text: str, state: Mapping[str, Any], user_id: str
    ) -> str:
        store = get_library_store()
        book_match = _BOOK_ID_RE.search(request_text)
        doc_id = (
            book_match.group(1).lower()
            if book_match
            else str(state.get(StateKey.ACTIVE_BOOK_ID) or "")
        )
        if not doc_id:
            saved_books = store.list_books(user_id)
            if len(saved_books) == 1:
                doc_id = saved_books[0]["doc_id"]
        if not doc_id or not store.has_book(doc_id):
            raise PipelineInputError(_NO_BOOK_MESSAGE)
        return doc_id

    def _resolve_chapter_id(
        self, toc: BookToc, request_text: str, state: Mapping[str, Any]
    ) -> str:
        chapter_match = _CHAPTER_ID_RE.search(request_text)
        if chapter_match:
            chapter_id = chapter_match.group(1).lower()
            if not toc.has_chapter(chapter_id):
                raise PipelineInputError(
                    f"Chapter `{chapter_id}` doesn't exist in '{toc.title}'. "
                    f"Available chapters: {', '.join(toc.chapter_ids)}."
                )
            return chapter_id

        for key in (self.default_chapter_key, StateKey.BOOKMARK_CHAPTER_ID):
            candidate = str(state.get(key) or "")
            if toc.has_chapter(candidate):
                return candidate
        if toc.first_chapter_id:
            return toc.first_chapter_id
        raise PipelineInputError(f"'{toc.title}' has no readable chapters.")


class CompanionSyncAgent(BaseAgent):
    """Persists the chapter digest, advances the bookmark, and refreshes book_companion.html."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        doc_id = str(state.get(StateKey.ACTIVE_BOOK_ID) or "")
        chapter_id = str(state.get(StateKey.ACTIVE_CHAPTER_ID) or "")
        digest_md = str(state.get(StateKey.LATEST_CHAPTER_DIGEST) or "").strip()
        if not (doc_id and chapter_id and digest_md):
            logger.warning("companion sync skipped: missing book, chapter, or digest.")
            return

        toc = await asyncio.to_thread(get_library_store().load_toc, doc_id)
        progress, artifact_delta = await _persist_progress(
            ctx, toc, lambda p: p.save_chapter_digest(toc, chapter_id, digest_md)
        )

        yield Event(
            author=self.name,
            actions=EventActions(
                state_delta={
                    **reader_state_delta(
                        toc,
                        progress,
                        saved_bookshelf=state.get(StateKey.SAVED_BOOKSHELF_OVERVIEW),
                    ),
                    # Moving to a new chapter retires any unanswered check-in.
                    **QUIZ_RESET_DELTA,
                },
                artifact_delta=artifact_delta,
            ),
        )


class QuizEscalationChecker(BaseAgent):
    """Terminates the quiz verification LoopAgent once grounding_critic grades 'pass'."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        review = parse_json_state(
            ctx.session.state.get(StateKey.QUIZ_GROUNDING_FEEDBACK)
        )
        if review and review.get("grade") == "pass":
            yield Event(author=self.name, actions=EventActions(escalate=True))
        else:
            yield Event(author=self.name)


class PublicQuizFormatterAgent(BaseAgent):
    """Publishes a verified quiz's questions while keeping its answer key in state.

    Only a draft that grounding_critic passed becomes `ACTIVE_QUIZ_KEY`; if the
    verification loop ran out of iterations, nothing is published.
    """

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        quiz = self._verified_quiz(state)
        if quiz is None:
            yield Event(
                author=self.name,
                content=text_content(_QUIZ_UNAVAILABLE_MESSAGE),
                actions=EventActions(
                    state_delta={StateKey.QUIZ_DRAFT: {}, **QUIZ_RESET_DELTA}
                ),
            )
            return

        yield Event(
            author=self.name,
            content=text_content(self._render_questions(quiz)),
            actions=EventActions(
                state_delta={
                    StateKey.QUIZ_DRAFT: {},
                    StateKey.ACTIVE_QUIZ_KEY: quiz.model_dump(mode="json"),
                    StateKey.PENDING_QUIZ_CHAPTER_ID: quiz.chapter_id,
                }
            ),
        )

    def _verified_quiz(self, state: Mapping[str, Any]) -> QuizSetDraft | None:
        review = parse_json_state(state.get(StateKey.QUIZ_GROUNDING_FEEDBACK))
        draft = parse_json_state(state.get(StateKey.QUIZ_DRAFT))
        if not (review and review.get("grade") == "pass" and draft):
            logger.info("quiz withheld: no draft passed grounding verification.")
            return None
        try:
            quiz = QuizSetDraft.model_validate(draft)
        except ValidationError:
            logger.warning("quiz withheld: verified draft failed schema validation.")
            return None
        return quiz if quiz.questions else None

    def _render_questions(self, quiz: QuizSetDraft) -> str:
        lines = [f"### Reflection Check-In: {quiz.chapter_title} (`{quiz.chapter_id}`)"]
        for item in quiz.questions:
            lines.append(
                f"\n**Question {item.question_number} ({item.topic}):** {item.question_text}"
            )
            lines.extend(f"- {option}" for option in item.options)
        return "\n".join(lines)


class QuizSubmissionLoaderAgent(BaseAgent):
    """Stages the reader's answers for grading.

    Stops the pipeline (without a model call) when no quiz is pending, the reply
    is empty, or the quiz's book is no longer in the library.
    """

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        quiz = parse_json_state(state.get(StateKey.ACTIVE_QUIZ_KEY))
        submission = _request_text(ctx).strip()
        if not (quiz and quiz.get("questions")):
            yield _error_event(self.name, _NO_PENDING_QUIZ_MESSAGE)
            return
        if not submission:
            yield _error_event(self.name, "I didn't receive any answers to grade.")
            return

        doc_id = str(state.get(StateKey.ACTIVE_BOOK_ID) or "")
        has_book = bool(doc_id) and await asyncio.to_thread(
            get_library_store().has_book, doc_id
        )
        if not has_book:
            # Fail before the grader runs: results could not be recorded anyway,
            # and a quiz whose book is gone should not linger in state.
            yield Event(
                author=self.name,
                actions=EventActions(
                    state_delta={
                        StateKey.PIPELINE_ERROR: _QUIZ_BOOK_MISSING_MESSAGE,
                        **QUIZ_RESET_DELTA,
                    }
                ),
            )
            return
        yield Event(
            author=self.name,
            actions=EventActions(
                state_delta={
                    StateKey.PIPELINE_ERROR: "",
                    StateKey.QUIZ_SUBMISSION: submission,
                    StateKey.QUIZ_GRADE_REPORT: {},
                }
            ),
        )


class QuizFeedbackAgent(BaseAgent):
    """Records check-in results in the progress store and renders per-question feedback.

    Recording happens here, deterministically, rather than asking the root agent
    to call `record_progress` per question: one store update, one artifact
    refresh, and no reliance on the model re-reading markdown.
    """

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        try:
            quiz = QuizSetDraft.model_validate(
                parse_json_state(state.get(StateKey.ACTIVE_QUIZ_KEY)) or {}
            )
            report = QuizGradeReport.model_validate(
                parse_json_state(state.get(StateKey.QUIZ_GRADE_REPORT)) or {}
            )
        except ValidationError:
            logger.warning("quiz grading output failed schema validation.")
            yield Event(
                author=self.name,
                content=text_content(
                    "I couldn't grade those answers. Please send them again."
                ),
            )
            return

        grades = {grade.question_number: grade for grade in report.grades}
        results = {
            question.topic: bool(
                (grade := grades.get(question.question_number)) and grade.passed
            )
            for question in quiz.questions
        }

        # QuizSubmissionLoaderAgent verified the active book exists before the
        # grader ran, so a missing book here is a genuine fault, not a user path.
        doc_id = str(state.get(StateKey.ACTIVE_BOOK_ID) or "")
        toc = await asyncio.to_thread(get_library_store().load_toc, doc_id)
        progress, artifact_delta = await _persist_progress(
            ctx, toc, lambda p: p.record_quiz_results(toc, quiz.chapter_id, results)
        )

        yield Event(
            author=self.name,
            content=text_content(
                self._render_feedback(
                    quiz, grades, self._render_mastery(progress, results)
                )
            ),
            actions=EventActions(
                state_delta={
                    **reader_state_delta(
                        toc,
                        progress,
                        saved_bookshelf=state.get(StateKey.SAVED_BOOKSHELF_OVERVIEW),
                    ),
                    **QUIZ_RESET_DELTA,
                    StateKey.QUIZ_SUBMISSION: "",
                },
                artifact_delta=artifact_delta,
            ),
        )

    def _render_feedback(
        self,
        quiz: QuizSetDraft,
        grades: Mapping[int, QuestionGrade],
        mastery_lines: list[str],
    ) -> str:
        lines = [f"### Check-In Results: {quiz.chapter_title} (`{quiz.chapter_id}`)"]
        for question in quiz.questions:
            grade = grades.get(question.question_number)
            verdict = "✅ Correct" if grade and grade.passed else "❌ Not quite"
            feedback = grade.feedback if grade else "This question wasn't answered."
            lines.extend(
                [
                    f"\n**Question {question.question_number} ({question.topic}) — "
                    f"{verdict}.** {feedback}",
                    f"- **Expected answer:** {question.correct_answer}",
                    f'> "{question.supporting_quote}" — [p. {question.page_number}]',
                ]
            )
        if mastery_lines:
            lines.append("\n**Mastery update:**")
            lines.extend(mastery_lines)
        return "\n".join(lines)

    def _render_mastery(
        self, progress: BookProgress, results: Mapping[str, bool]
    ) -> list[str]:
        lines = [
            f"- {topic}: {progress.topic_mastery[topic].accuracy_pct}% "
            f"({'passed' if passed else 'missed'} this time)"
            for topic, passed in results.items()
        ]
        lines.append(f"- Overall mastery: {progress.overall_mastery_pct}%")
        if progress.weak_topics:
            lines.append(f"- Worth revisiting: {', '.join(progress.weak_topics)}")
        return lines
