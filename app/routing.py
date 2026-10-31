"""Task-complexity model routing across fast, standard, and reasoning Gemini tiers.

Each agent is pinned to the cheapest tier that handles its job well:

| Tier        | Default model            | Used by                                      |
| :---------- | :----------------------- | :------------------------------------------- |
| `FAST`      | `gemini-3.5-flash-lite`  | `quiz_grader` (rubric classification)        |
| `STANDARD`  | `gemini-3.8-flash`       | root coordinator, `quiz_drafter`, short digests |
| `REASONING` | `gemini-3.1-pro-preview` | `grounding_critic`, long-chapter digests     |

On top of that static choice, `escalate_long_chapters` sends a single request up
to `REASONING` when the staged chapter is too long for the standard tier to
summarize faithfully. Every routing decision is logged as a structured event.
Override a tier's model with `BOOK_COMPANION_<TIER>_MODEL`.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from enum import StrEnum

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import BaseLlm, LlmRequest, LlmResponse

from app.runtime import make_model
from app.state import StateKey

logger = logging.getLogger(__name__)

# Chapters longer than this go to the reasoning tier for summarization.
LONG_CHAPTER_WORD_THRESHOLD = 2500


class ModelTier(StrEnum):
    """Capability tiers, ordered from cheapest/fastest to most capable."""

    FAST = "fast"
    STANDARD = "standard"
    REASONING = "reasoning"


DEFAULT_TIER_MODELS: Mapping[ModelTier, str] = {
    ModelTier.FAST: "gemini-3.5-flash-lite",
    ModelTier.STANDARD: "gemini-3.8-flash",
    ModelTier.REASONING: "gemini-3.1-pro-preview",
}


class ModelRouter:
    """Resolves model tiers to `BaseLlm` instances and routes by task complexity."""

    def __init__(self, tier_models: Mapping[ModelTier, str | BaseLlm]) -> None:
        missing = set(ModelTier) - set(tier_models)
        if missing:
            raise ValueError(f"No model configured for tiers: {sorted(missing)}")
        self._models: dict[ModelTier, BaseLlm] = {
            tier: model if isinstance(model, BaseLlm) else make_model(model)
            for tier, model in tier_models.items()
        }

    @classmethod
    def from_env(cls) -> ModelRouter:
        """Builds the default router, honoring `BOOK_COMPANION_<TIER>_MODEL` overrides."""
        return cls(
            {
                tier: os.environ.get(f"BOOK_COMPANION_{tier.name}_MODEL", default)
                for tier, default in DEFAULT_TIER_MODELS.items()
            }
        )

    @classmethod
    def uniform(cls, model: str | BaseLlm) -> ModelRouter:
        """Builds a router that serves one model for every tier (tests, local fakes)."""
        return cls(dict.fromkeys(ModelTier, model))

    def model(self, tier: ModelTier) -> BaseLlm:
        """Returns the model serving `tier`."""
        return self._models[tier]

    def model_name(self, tier: ModelTier) -> str:
        """Returns the model identifier serving `tier`."""
        return self._models[tier].model

    def escalate_long_chapters(
        self,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
    ) -> LlmResponse | None:
        """`before_model_callback` that upgrades long-chapter requests to `REASONING`.

        ADK copies the agent's model name into `llm_request.model` before this
        callback runs, and `Gemini` honors that field, so rewriting it reroutes
        just this request without rebuilding the agent.
        """
        material = str(
            callback_context.state.get(StateKey.ACTIVE_CHAPTER_MATERIAL) or ""
        )
        word_count = len(material.split())
        if word_count > LONG_CHAPTER_WORD_THRESHOLD:
            tier, reason = ModelTier.REASONING, "long_chapter"
            llm_request.model = self.model_name(tier)
        else:
            tier, reason = ModelTier.STANDARD, "within_standard_budget"

        logger.info(
            "model routing decision",
            extra={
                "json_fields": {
                    "event_type": "model_routing",
                    "agent": callback_context.agent_name,
                    "tier": tier.value,
                    "routed_model": llm_request.model,
                    "reason": reason,
                    "chapter_words": word_count,
                }
            },
        )
        return None
