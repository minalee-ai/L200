# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Root BookCompanion agent and ADK application definition."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.apps.app import EventsCompactionConfig, ResumabilityConfig
from google.adk.plugins.save_files_as_artifacts_plugin import (
    SaveFilesAsArtifactsPlugin,
)
from google.adk.tools import FunctionTool
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.preload_memory_tool import PreloadMemoryTool

from app import prompts
from app.callbacks import (
    initialize_reader_state_callback,
    save_session_memory_callback,
    validate_tool_inputs_callback,
)
from app.observability import AuditTrailPlugin, configure_structured_logging
from app.routing import ModelRouter, ModelTier
from app.tools import (
    delete_saved_book,
    get_chapter_chunks,
    ingest_or_load_book,
    list_saved_books,
    record_progress,
    reset_book_progress,
    search_book_phrases,
    update_book_companion_artifact,
)
from app.workflows import (
    create_chapter_digest_pipeline,
    create_quiz_grading_pipeline,
    create_quiz_verification_pipeline,
)

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
configure_structured_logging()

router = ModelRouter.from_env()

root_agent = Agent(
    # Keep in sync with create_params.root_agent_name in agents-cli-manifest.yaml.
    name="book_companion",
    # The coordinator mostly routes turns to tools, so the standard tier suffices.
    model=router.model(ModelTier.STANDARD),
    description=(
        "Interactive summarized-book companion that chunks PDFs by chapter and phrase "
        "boundaries, delivers cited narrative digests, and runs grounded check-in quizzes."
    ),
    instruction=prompts.ROOT_INSTRUCTION,
    tools=[
        ingest_or_load_book,
        list_saved_books,
        get_chapter_chunks,
        search_book_phrases,
        record_progress,
        update_book_companion_artifact,
        # Human-in-the-loop: ADK pauses the run and asks the reader to approve
        # before either destructive action executes.
        FunctionTool(reset_book_progress, require_confirmation=True),
        FunctionTool(delete_saved_book, require_confirmation=True),
        AgentTool(agent=create_chapter_digest_pipeline(router)),
        AgentTool(agent=create_quiz_verification_pipeline(router)),
        AgentTool(agent=create_quiz_grading_pipeline(router)),
        PreloadMemoryTool(),
    ],
    before_agent_callback=initialize_reader_state_callback,
    after_agent_callback=save_session_memory_callback,
    before_tool_callback=validate_tool_inputs_callback,
)

app = App(
    root_agent=root_agent,
    name="app",
    # AuditTrailPlugin goes first so PII is redacted before any other plugin
    # (or the model) sees the user message.
    plugins=[AuditTrailPlugin(), SaveFilesAsArtifactsPlugin()],
    events_compaction_config=EventsCompactionConfig(
        token_threshold=32000,
        event_retention_size=6,
    ),
    # Lets a run paused on a confirmation request resume where it stopped.
    resumability_config=ResumabilityConfig(is_resumable=True),
)
