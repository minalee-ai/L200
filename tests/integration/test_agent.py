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

"""Integration tests verifying BookCompanion agent wiring and streaming execution."""

from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.plugins.save_files_as_artifacts_plugin import (
    SaveFilesAsArtifactsPlugin,
)
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.preload_memory_tool import PreloadMemoryTool
from google.genai import types

from app.agent import app, root_agent


def test_agent_architecture_wiring() -> None:
    """Verifies memory, artifact plugin, compaction, and workflow AgentTools are wired."""
    assert root_agent.name == "book_companion"
    assert app.name == "app"
    assert app.events_compaction_config is not None
    assert any(isinstance(p, SaveFilesAsArtifactsPlugin) for p in app.plugins)
    assert any(isinstance(t, PreloadMemoryTool) for t in root_agent.tools)

    agent_tool_names = {t.name for t in root_agent.tools if isinstance(t, AgentTool)}
    assert {
        "summarize_chapter",
        "generate_chapter_quiz",
        "grade_quiz_answers",
    } <= agent_tool_names


def test_agent_stream() -> None:
    """Tests that the agent streams a valid response over InMemorySessionService."""
    session_service = InMemorySessionService()
    session = session_service.create_session_sync(user_id="test_user", app_name="app")
    runner = Runner(app=app, session_service=session_service)

    message = types.Content(
        role="user",
        parts=[
            types.Part.from_text(
                text="Hello! Briefly introduce how BookCompanion helps me read a PDF book."
            )
        ],
    )

    events = list(
        runner.run(
            new_message=message,
            user_id="test_user",
            session_id=session.id,
            run_config=RunConfig(streaming_mode=StreamingMode.SSE),
        )
    )
    assert len(events) > 0, "Expected at least one event"
    assert any(
        event.content
        and event.content.parts
        and any(part.text for part in event.content.parts)
        for event in events
    ), "Expected at least one text response part"
