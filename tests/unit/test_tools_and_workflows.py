"""Unit tests for ADK function tools, artifact ingestion, and anti-spoiler quiz formatting."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from google.adk.agents import SequentialAgent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.callbacks import validate_tool_inputs_callback
from app.library.models import QuizQuestionDraft, QuizSetDraft
from app.library.store import LibraryStore
from app.runtime import COMPANION_ARTIFACT_FILENAME
from app.state import StateKey
from app.tools import (
    ingest_or_load_book,
    record_progress,
    search_book_phrases,
)
from app.workflows import PublicQuizFormatterAgent
from tests.unit.conftest import SeedStateAgent, build_bookmarked_pdf_bytes


def _make_fake_tool_context(state: dict | None = None) -> MagicMock:
    ctx = MagicMock()
    ctx.state = state if state is not None else {}
    ctx.user_id = "reader_test"
    ctx.list_artifacts.side_effect = ValueError("No ArtifactService")
    ctx.save_artifact.side_effect = ValueError("No ArtifactService")
    return ctx


@pytest.mark.asyncio
async def test_ingest_or_load_book_caches_chunks_across_calls(
    isolated_store: LibraryStore,
    tmp_path: Path,
) -> None:
    del isolated_store
    pdf_file = tmp_path / "consensus.pdf"
    pdf_file.write_bytes(build_bookmarked_pdf_bytes())

    ctx = _make_fake_tool_context()
    first = await ingest_or_load_book(str(pdf_file), ctx)
    assert first["status"] == "success"
    assert first["cache_hit"] is False
    doc_id = first["doc_id"]
    assert ctx.state[StateKey.ACTIVE_BOOK_ID] == doc_id
    assert len(first["chapters"]) == 3

    # Second call hits persistent store without re-parsing.
    second = await ingest_or_load_book(doc_id, ctx)
    assert second["status"] == "success"
    assert second["cache_hit"] is True
    assert second["doc_id"] == doc_id


@pytest.mark.asyncio
async def test_ingest_loads_uploaded_pdf_from_adk_artifact_service(
    isolated_store: LibraryStore,
) -> None:
    del isolated_store
    artifact_service = InMemoryArtifactService()
    pdf_part = types.Part(
        inline_data=types.Blob(
            mime_type="application/pdf",
            data=build_bookmarked_pdf_bytes(),
        )
    )
    await artifact_service.save_artifact(
        app_name="app",
        user_id="reader_ui",
        session_id="s1",
        filename="uploaded_book.pdf",
        artifact=pdf_part,
    )

    ctx = MagicMock()
    ctx.state = {}
    ctx.user_id = "reader_ui"

    async def _list() -> list[str]:
        return await artifact_service.list_artifact_keys(
            app_name="app", user_id="reader_ui", session_id="s1"
        )

    async def _load(name: str):
        return await artifact_service.load_artifact(
            app_name="app", user_id="reader_ui", session_id="s1", filename=name
        )

    async def _save(name: str, part: types.Part) -> int:
        return await artifact_service.save_artifact(
            app_name="app",
            user_id="reader_ui",
            session_id="s1",
            filename=name,
            artifact=part,
        )

    ctx.list_artifacts.side_effect = _list
    ctx.load_artifact.side_effect = _load
    ctx.save_artifact.side_effect = _save

    result = await ingest_or_load_book("latest", ctx)
    assert result["status"] == "success"
    assert result["companion_artifact"] == COMPANION_ARTIFACT_FILENAME

    saved_keys = await artifact_service.list_artifact_keys(
        app_name="app", user_id="reader_ui", session_id="s1"
    )
    assert COMPANION_ARTIFACT_FILENAME in saved_keys


@pytest.mark.asyncio
async def test_record_progress_updates_mastery_and_advances_bookmark(
    isolated_store: LibraryStore,
    tmp_path: Path,
) -> None:
    del isolated_store
    pdf_file = tmp_path / "consensus.pdf"
    pdf_file.write_bytes(build_bookmarked_pdf_bytes())
    ctx = _make_fake_tool_context()
    ingested = await ingest_or_load_book(str(pdf_file), ctx)
    doc_id = ingested["doc_id"]

    progress_res = await record_progress(
        doc_id=doc_id,
        chapter_id="ch01",
        topic="State Machine Replication",
        quiz_passed=True,
        tool_context=ctx,
    )
    assert progress_res["status"] == "success"
    assert progress_res["topic_accuracy_pct"] == 100
    assert progress_res["next_chapter_id"] == "ch02"
    assert ctx.state[StateKey.BOOKMARK_CHAPTER_ID] == "ch02"


@pytest.mark.asyncio
async def test_public_quiz_formatter_hides_answers_and_quotes() -> None:
    secret_answer = "Strict majority quorum"
    secret_quote = "A candidate must collect votes from a strict majority quorum"
    draft = QuizSetDraft(
        chapter_id="ch02",
        chapter_title="Chapter 2: Leader Election",
        questions=[
            QuizQuestionDraft(
                question_number=1,
                topic="Quorum Votes",
                question_text="What vote threshold is required to win election?",
                options=["A) Unanimous", "B) Strict majority", "C) Single peer"],
                correct_answer=secret_answer,
                supporting_quote=secret_quote,
                page_number=2,
            )
        ],
    )

    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name="test", user_id="u1")

    pipeline = SequentialAgent(
        name="formatter_test",
        sub_agents=[
            SeedStateAgent(
                name="seed",
                state_delta={
                    StateKey.QUIZ_DRAFT: draft.model_dump(mode="json"),
                    StateKey.QUIZ_GROUNDING_FEEDBACK: {
                        "grade": "pass",
                        "feedback": "ok",
                    },
                },
            ),
            PublicQuizFormatterAgent(name="public_quiz_presenter"),
        ],
    )
    runner = Runner(agent=pipeline, session_service=session_service, app_name="test")

    rendered_texts: list[str] = []
    async for event in runner.run_async(
        user_id="u1",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text="check-in")]),
    ):
        if event.content and event.content.parts:
            rendered_texts.extend(p.text for p in event.content.parts if p.text)

    combined = "\n".join(rendered_texts)
    assert "What vote threshold is required" in combined
    assert secret_answer not in combined
    assert secret_quote not in combined

    final = await session_service.get_session(
        app_name="test", user_id="u1", session_id=session.id
    )
    assert final.state[StateKey.ACTIVE_QUIZ_KEY] == draft.model_dump(mode="json")
    assert final.state[StateKey.PENDING_QUIZ_CHAPTER_ID] == "ch02"


@pytest.mark.asyncio
async def test_input_guardrail_blocks_bad_paths_and_clamps_limit() -> None:
    fake_tool = MagicMock()
    ctx = _make_fake_tool_context()

    fake_tool.name = "ingest_or_load_book"
    traversal = await validate_tool_inputs_callback(
        fake_tool, {"source": "../secret.pdf"}, ctx
    )
    assert traversal is not None and traversal["status"] == "error"

    bad_ext = await validate_tool_inputs_callback(
        fake_tool, {"source": "notes.txt"}, ctx
    )
    assert bad_ext is not None and bad_ext["status"] == "error"

    fake_tool.name = "search_book_phrases"
    search_args = {
        "doc_id": "book_123",
        "query": "quorum",
        "chapter_id": "all",
        "limit": 99,
    }
    allowed = await validate_tool_inputs_callback(fake_tool, search_args, ctx)
    assert allowed is None
    assert search_args["limit"] == 8
    assert callable(search_book_phrases)


@pytest.mark.asyncio
async def test_parallel_record_progress_calls_do_not_lose_updates(
    isolated_store: LibraryStore,
    tmp_path: Path,
) -> None:
    pdf_file = tmp_path / "consensus.pdf"
    pdf_file.write_bytes(build_bookmarked_pdf_bytes())
    ctx = _make_fake_tool_context()
    doc_id = (await ingest_or_load_book(str(pdf_file), ctx))["doc_id"]

    topics = [f"Topic {idx}" for idx in range(8)]
    results = await asyncio.gather(
        *(
            record_progress(
                doc_id=doc_id,
                chapter_id="ch01",
                topic=topic,
                quiz_passed=True,
                tool_context=ctx,
            )
            for topic in topics
        )
    )

    assert all(result["status"] == "success" for result in results)
    toc = isolated_store.load_toc(doc_id)
    progress = isolated_store.load_or_init_progress("reader_test", toc)
    assert sorted(progress.topic_mastery) == topics
