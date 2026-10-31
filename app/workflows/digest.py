"""Chapter digest pipeline: load saved chunks, write an abridged chapter, sync HTML."""

from __future__ import annotations

from google.adk.agents import Agent, SequentialAgent

from app import prompts
from app.routing import ModelRouter, ModelTier
from app.state import StateKey
from app.workflows.steps import (
    ChapterChunkLoaderAgent,
    CompanionSyncAgent,
    skip_if_pipeline_error,
    skip_silently_if_pipeline_error,
)


def create_chapter_digest_pipeline(router: ModelRouter) -> SequentialAgent:
    """Creates the `summarize_chapter` pipeline exposed to the root agent as a tool."""
    summarizer = Agent(
        name="narrative_summarizer",
        # Standard tier by default; long chapters are escalated per request.
        model=router.model(ModelTier.STANDARD),
        include_contents="none",
        output_key=StateKey.LATEST_CHAPTER_DIGEST,
        description="Composes an engaging abridged-book chapter digest with page citations.",
        instruction=prompts.NARRATIVE_SUMMARIZER_INSTRUCTION,
        before_agent_callback=skip_silently_if_pipeline_error,
        before_model_callback=router.escalate_long_chapters,
    )

    return SequentialAgent(
        name="summarize_chapter",
        description=(
            "Generates a cited, narrative abridged-book summary for a chapter (pass e.g. "
            "'ch01' or 'ch02' in request) and updates the book_companion.html edition."
        ),
        sub_agents=[
            ChapterChunkLoaderAgent(
                name="chapter_retriever",
                default_chapter_key=StateKey.BOOKMARK_CHAPTER_ID,
            ),
            summarizer,
            CompanionSyncAgent(
                name="companion_sync",
                before_agent_callback=skip_if_pipeline_error,
            ),
        ],
    )
