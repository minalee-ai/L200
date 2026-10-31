"""Check-in quiz pipelines: grounded quiz generation and answer grading."""

from __future__ import annotations

from google.adk.agents import Agent, LoopAgent, SequentialAgent

from app import prompts
from app.library.models import GroundingReview, QuizGradeReport, QuizSetDraft
from app.routing import ModelRouter, ModelTier
from app.state import StateKey
from app.workflows.steps import (
    ChapterChunkLoaderAgent,
    PublicQuizFormatterAgent,
    QuizEscalationChecker,
    QuizFeedbackAgent,
    QuizSubmissionLoaderAgent,
    skip_if_pipeline_error,
    skip_silently_if_pipeline_error,
)

_MAX_QUIZ_DRAFT_ATTEMPTS = 2


def create_quiz_verification_pipeline(router: ModelRouter) -> SequentialAgent:
    """Creates the `generate_chapter_quiz` pipeline (drafter-critic loop + formatter).

    The critic runs on the reasoning tier: it is the last line of defense
    against ungrounded quotes and answer leaks, so it gets the strongest model.
    """
    drafter = Agent(
        name="quiz_drafter",
        model=router.model(ModelTier.STANDARD),
        include_contents="none",
        output_schema=QuizSetDraft,
        output_key=StateKey.QUIZ_DRAFT,
        description="Drafts 2 comprehension questions grounded in saved chapter chunks.",
        instruction=prompts.QUIZ_DRAFTER_INSTRUCTION,
    )

    critic = Agent(
        name="grounding_critic",
        model=router.model(ModelTier.REASONING),
        include_contents="none",
        output_schema=GroundingReview,
        output_key=StateKey.QUIZ_GROUNDING_FEEDBACK,
        description="Verifies quiz answers against exact PDF phrases and checks anti-spoiler rules.",
        instruction=prompts.GROUNDING_CRITIC_INSTRUCTION,
    )

    verification_loop = LoopAgent(
        name="quiz_verification_loop",
        max_iterations=_MAX_QUIZ_DRAFT_ATTEMPTS,
        sub_agents=[
            drafter,
            critic,
            QuizEscalationChecker(name="quiz_escalation_checker"),
        ],
        before_agent_callback=skip_silently_if_pipeline_error,
    )

    return SequentialAgent(
        name="generate_chapter_quiz",
        description=(
            "Generates a 2-question grounding-verified check-in quiz for a chapter (pass e.g. "
            "'ch01' in request), storing the hidden answer key in session state."
        ),
        sub_agents=[
            ChapterChunkLoaderAgent(
                name="quiz_chapter_retriever",
                default_chapter_key=StateKey.ACTIVE_CHAPTER_ID,
            ),
            verification_loop,
            PublicQuizFormatterAgent(
                name="public_quiz_presenter",
                before_agent_callback=skip_if_pipeline_error,
            ),
        ],
    )


def create_quiz_grading_pipeline(router: ModelRouter) -> SequentialAgent:
    """Creates the `grade_quiz_answers` pipeline, which reads the hidden answer key.

    Grading happens here so the root agent never holds the answer key before the
    reader has answered. Comparing two short answers against a key is a
    classification task, so the grader runs on the fast tier.
    """
    grader = Agent(
        name="quiz_grader",
        model=router.model(ModelTier.FAST),
        include_contents="none",
        output_schema=QuizGradeReport,
        output_key=StateKey.QUIZ_GRADE_REPORT,
        description="Grades the reader's answers against the hidden answer key.",
        instruction=prompts.QUIZ_GRADER_INSTRUCTION,
        before_agent_callback=skip_silently_if_pipeline_error,
    )

    return SequentialAgent(
        name="grade_quiz_answers",
        description=(
            "Grades the reader's answers to the pending check-in quiz. Pass the "
            "reader's reply verbatim as the request."
        ),
        sub_agents=[
            QuizSubmissionLoaderAgent(name="quiz_submission_loader"),
            grader,
            QuizFeedbackAgent(
                name="quiz_feedback_presenter",
                before_agent_callback=skip_if_pipeline_error,
            ),
        ],
    )
