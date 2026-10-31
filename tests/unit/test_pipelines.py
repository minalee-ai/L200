"""Pipeline regression tests driven by a scripted fake LLM (no network calls)."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator

import pytest
from google.adk.agents import BaseAgent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
from google.genai import types
from pydantic import Field

from app.library.chunker import HierarchicalPdfChunker
from app.library.models import BookToc, QuizQuestionDraft, QuizSetDraft
from app.library.store import LibraryStore
from app.routing import ModelRouter
from app.runtime import COMPANION_ARTIFACT_FILENAME
from app.state import StateKey
from app.workflows import (
    create_chapter_digest_pipeline,
    create_quiz_grading_pipeline,
    create_quiz_verification_pipeline,
)
from tests.unit.conftest import build_bookmarked_pdf_bytes

_APP = "test_app"
_USER = "reader_test"
_TEXT = "text"
_SECRET_ANSWER = "Strict majority quorum"
_SECRET_QUOTE = "consequently, non-faulty nodes converge on identical state."


class ScriptedLlm(BaseLlm):
    """Fake LLM that replies from per-output-schema queues and records each call.

    Replies are keyed by the requested `output_schema` class name, or `"text"`
    for agents without one.
    """

    model: str = "scripted-llm"
    replies: dict[str, list[str]] = Field(default_factory=dict)
    calls: list[str] = Field(default_factory=list)
    instructions: list[str] = Field(default_factory=list)

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del stream
        schema = llm_request.config.response_schema if llm_request.config else None
        key = getattr(schema, "__name__", _TEXT) if schema else _TEXT
        self.calls.append(key)
        self.instructions.append(str(llm_request.config.system_instruction or ""))
        yield LlmResponse(
            content=types.Content(
                role="model", parts=[types.Part(text=self.replies[key].pop(0))]
            )
        )


def _quiz_draft(chapter_id: str = "ch01") -> QuizSetDraft:
    return QuizSetDraft(
        chapter_id=chapter_id,
        chapter_title="Chapter 1: Foundations of Distributed Consensus",
        questions=[
            QuizQuestionDraft(
                question_number=1,
                topic="Replica Convergence",
                question_text="Why do non-faulty replicas end up with the same state?",
                options=[],
                correct_answer=_SECRET_ANSWER,
                supporting_quote=_SECRET_QUOTE,
                page_number=1,
            )
        ],
    )


def _grounding(grade: str) -> str:
    return json.dumps({"grade": grade, "feedback": f"grade={grade}"})


class _Harness:
    """Runs pipelines against shared in-memory session and artifact services."""

    def __init__(self) -> None:
        self.sessions = InMemorySessionService()
        self.artifacts = InMemoryArtifactService()

    async def new_session(self, state: dict | None = None) -> Session:
        return await self.sessions.create_session(
            app_name=_APP, user_id=_USER, state=state or {}
        )

    async def run(
        self,
        agent: BaseAgent,
        session: Session,
        message: str,
        *,
        with_artifacts: bool = True,
    ) -> list[str]:
        runner = Runner(
            app_name=_APP,
            agent=agent,
            session_service=self.sessions,
            artifact_service=self.artifacts if with_artifacts else None,
        )
        texts: list[str] = []
        async for event in runner.run_async(
            user_id=_USER,
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text=message)]),
        ):
            if event.content and event.content.parts:
                texts.extend(p.text for p in event.content.parts if p.text)
        return texts

    async def state(self, session: Session) -> dict:
        refreshed = await self.sessions.get_session(
            app_name=_APP, user_id=_USER, session_id=session.id
        )
        assert refreshed is not None
        return refreshed.state


@pytest.fixture
def seeded_book(isolated_store: LibraryStore) -> BookToc:
    pdf_bytes = build_bookmarked_pdf_bytes()
    doc_id = isolated_store.compute_doc_id(pdf_bytes)
    toc, chunks = HierarchicalPdfChunker().chunk_pdf(
        pdf_bytes, doc_id=doc_id, fallback_title="guide.pdf"
    )
    isolated_store.save_book(toc, chunks)
    return toc


@pytest.mark.asyncio
async def test_digest_pipeline_stops_when_no_book_is_loaded(
    isolated_store: LibraryStore,
) -> None:
    del isolated_store
    llm = ScriptedLlm()
    harness = _Harness()
    session = await harness.new_session()

    texts = await harness.run(
        create_chapter_digest_pipeline(ModelRouter.uniform(llm)),
        session,
        "Summarize ch01",
    )

    assert llm.calls == [], "summarizer must not run without source material"
    assert "No active book is loaded" in texts[-1]


@pytest.mark.asyncio
async def test_digest_pipeline_rejects_unknown_chapter(seeded_book: BookToc) -> None:
    llm = ScriptedLlm()
    harness = _Harness()
    session = await harness.new_session({StateKey.ACTIVE_BOOK_ID: seeded_book.doc_id})

    texts = await harness.run(
        create_chapter_digest_pipeline(ModelRouter.uniform(llm)),
        session,
        "Summarize ch99",
    )

    assert llm.calls == []
    assert "`ch99` doesn't exist" in texts[-1]
    assert "ch01, ch02, ch03" in texts[-1]

    quiz_texts = await harness.run(
        create_quiz_verification_pipeline(ModelRouter.uniform(llm)),
        session,
        "check-in for ch99",
    )
    assert llm.calls == []
    assert len(quiz_texts) == 1, "loop and formatter must not both re-emit the error"
    assert "`ch99` doesn't exist" in quiz_texts[0]


@pytest.mark.asyncio
async def test_check_in_after_reading_targets_the_chapter_just_read(
    seeded_book: BookToc,
    isolated_store: LibraryStore,
) -> None:
    digest = "### The Big Idea\nReplicas apply the same log in order [p. 1]."
    llm = ScriptedLlm(
        replies={
            _TEXT: [digest],
            "QuizSetDraft": [_quiz_draft("ch01").model_dump_json()],
            "GroundingReview": [_grounding("pass")],
        }
    )
    harness = _Harness()
    session = await harness.new_session({StateKey.ACTIVE_BOOK_ID: seeded_book.doc_id})

    await harness.run(
        create_chapter_digest_pipeline(ModelRouter.uniform(llm)),
        session,
        "Summarize ch01",
    )

    state = await harness.state(session)
    assert state[StateKey.ACTIVE_CHAPTER_ID] == "ch01"
    assert state[StateKey.BOOKMARK_CHAPTER_ID] == "ch02"
    progress = isolated_store.load_progress(_USER, seeded_book.doc_id)
    assert progress is not None
    assert progress.chapter_digests["ch01"] == digest
    assert progress.bookmark_chapter_id == "ch02"
    artifact_keys = await harness.artifacts.list_artifact_keys(
        app_name=_APP, user_id=_USER, session_id=session.id
    )
    assert COMPANION_ARTIFACT_FILENAME in artifact_keys

    # A bare "check-in" must quiz the chapter just read, not the bookmark (ch02).
    texts = await harness.run(
        create_quiz_verification_pipeline(ModelRouter.uniform(llm)), session, "check-in"
    )

    drafter_instruction = llm.instructions[llm.calls.index("QuizSetDraft")]
    assert "Foundations of Distributed Consensus" in drafter_instruction
    assert "Leader Election" not in drafter_instruction
    assert "Reflection Check-In" in texts[-1]
    assert _SECRET_ANSWER not in texts[-1]
    assert (await harness.state(session))[StateKey.PENDING_QUIZ_CHAPTER_ID] == "ch01"


@pytest.mark.asyncio
async def test_quiz_pipeline_withholds_quiz_that_never_passes_grounding(
    seeded_book: BookToc,
) -> None:
    draft_json = _quiz_draft().model_dump_json()
    llm = ScriptedLlm(
        replies={
            "QuizSetDraft": [draft_json, draft_json],
            "GroundingReview": [_grounding("fail"), _grounding("fail")],
        }
    )
    harness = _Harness()
    session = await harness.new_session({StateKey.ACTIVE_BOOK_ID: seeded_book.doc_id})

    texts = await harness.run(
        create_quiz_verification_pipeline(ModelRouter.uniform(llm)),
        session,
        "check-in for ch01",
    )

    assert llm.calls == ["QuizSetDraft", "GroundingReview"] * 2
    assert "couldn't build a check-in quiz" in texts[-1]
    state = await harness.state(session)
    assert state[StateKey.ACTIVE_QUIZ_KEY] == {}
    assert state[StateKey.PENDING_QUIZ_CHAPTER_ID] == ""


@pytest.mark.asyncio
async def test_grading_pipeline_requires_a_pending_quiz(
    isolated_store: LibraryStore,
) -> None:
    del isolated_store
    llm = ScriptedLlm()
    harness = _Harness()
    session = await harness.new_session()

    texts = await harness.run(
        create_quiz_grading_pipeline(ModelRouter.uniform(llm)), session, "1) majority"
    )

    assert llm.calls == []
    assert "no check-in quiz awaiting answers" in texts[-1]


@pytest.mark.asyncio
async def test_grading_pipeline_fails_loudly_when_quiz_book_is_gone(
    isolated_store: LibraryStore,
) -> None:
    del isolated_store  # Empty library: the quiz's book no longer exists.
    llm = ScriptedLlm()
    harness = _Harness()
    session = await harness.new_session(
        {
            StateKey.ACTIVE_BOOK_ID: "book_deadbeef0000",
            StateKey.ACTIVE_QUIZ_KEY: _quiz_draft().model_dump(mode="json"),
            StateKey.PENDING_QUIZ_CHAPTER_ID: "ch01",
        }
    )

    texts = await harness.run(
        create_quiz_grading_pipeline(ModelRouter.uniform(llm)),
        session,
        "B) Strict majority",
    )

    assert llm.calls == [], "grader must not run when results cannot be recorded"
    assert "nothing was recorded" in texts[-1]
    state = await harness.state(session)
    assert state[StateKey.ACTIVE_QUIZ_KEY] == {}
    assert state[StateKey.PENDING_QUIZ_CHAPTER_ID] == ""


@pytest.mark.asyncio
async def test_grading_pipeline_records_mastery_and_closes_quiz(
    seeded_book: BookToc,
    isolated_store: LibraryStore,
) -> None:
    report = {
        "grades": [{"question_number": 1, "passed": True, "feedback": "Exactly right."}]
    }
    llm = ScriptedLlm(replies={"QuizGradeReport": [json.dumps(report)]})
    harness = _Harness()
    session = await harness.new_session(
        {
            StateKey.ACTIVE_BOOK_ID: seeded_book.doc_id,
            StateKey.ACTIVE_QUIZ_KEY: _quiz_draft().model_dump(mode="json"),
            StateKey.PENDING_QUIZ_CHAPTER_ID: "ch01",
        }
    )

    texts = await harness.run(
        create_quiz_grading_pipeline(ModelRouter.uniform(llm)),
        session,
        "They all apply the same commands in the same order.",
    )

    assert "same commands in the same order" in llm.instructions[0]
    feedback = texts[-1]
    assert "✅ Correct" in feedback
    assert _SECRET_QUOTE in feedback
    assert "Replica Convergence: 100%" in feedback
    state = await harness.state(session)
    assert state[StateKey.ACTIVE_QUIZ_KEY] == {}
    assert state[StateKey.PENDING_QUIZ_CHAPTER_ID] == ""
    assert state[StateKey.BOOKMARK_CHAPTER_ID] == "ch02"

    progress = isolated_store.load_progress(_USER, seeded_book.doc_id)
    assert progress is not None
    assert progress.topic_mastery["Replica Convergence"].attempts == 1
    assert progress.completed_chapters == ["ch01"]
    artifact_keys = await harness.artifacts.list_artifact_keys(
        app_name=_APP, user_id=_USER, session_id=session.id
    )
    assert COMPANION_ARTIFACT_FILENAME in artifact_keys


@pytest.mark.asyncio
async def test_digest_pipeline_survives_missing_artifact_service(
    seeded_book: BookToc,
    isolated_store: LibraryStore,
) -> None:
    llm = ScriptedLlm(replies={_TEXT: ["### The Big Idea\nLogs are ordered [p. 1]."]})
    harness = _Harness()
    session = await harness.new_session({StateKey.ACTIVE_BOOK_ID: seeded_book.doc_id})

    await harness.run(
        create_chapter_digest_pipeline(ModelRouter.uniform(llm)),
        session,
        "Summarize ch01",
        with_artifacts=False,
    )

    progress = isolated_store.load_progress(_USER, seeded_book.doc_id)
    assert progress is not None and "ch01" in progress.chapter_digests


@pytest.mark.asyncio
async def test_new_digest_retires_pending_quiz_and_keeps_scratch_out_of_session(
    seeded_book: BookToc,
) -> None:
    llm = ScriptedLlm(replies={_TEXT: ["### The Big Idea\nQuorums matter [p. 2]."]})
    harness = _Harness()
    session = await harness.new_session(
        {
            StateKey.ACTIVE_BOOK_ID: seeded_book.doc_id,
            StateKey.ACTIVE_QUIZ_KEY: _quiz_draft().model_dump(mode="json"),
            StateKey.PENDING_QUIZ_CHAPTER_ID: "ch01",
        }
    )

    await harness.run(
        create_chapter_digest_pipeline(ModelRouter.uniform(llm)),
        session,
        "Summarize ch02",
    )

    state = await harness.state(session)
    assert state[StateKey.ACTIVE_QUIZ_KEY] == {}
    assert state[StateKey.PENDING_QUIZ_CHAPTER_ID] == ""
    assert state[StateKey.ACTIVE_CHAPTER_ID] == "ch02"
    refreshed = await harness.sessions.get_session(
        app_name=_APP, user_id=_USER, session_id=session.id
    )
    assert refreshed is not None
    persisted_deltas = {
        key
        for event in refreshed.events
        if event.actions and event.actions.state_delta
        for key in event.actions.state_delta
    }
    assert not {k for k in persisted_deltas if k.startswith("temp:")}
    assert StateKey.ACTIVE_CHAPTER_MATERIAL not in persisted_deltas
