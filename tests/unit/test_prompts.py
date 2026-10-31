"""Guards the contract between prompt placeholders and session-state keys."""

from __future__ import annotations

import re

import pytest

from app import prompts
from app.state import ALL_STATE_KEYS, StateKey

_PLACEHOLDER_RE = re.compile(r"\{+\s*([A-Za-z_][\w:]*)\??\s*\}+")
_PROMPT_NAMES = [name for name in vars(prompts) if name.endswith("_INSTRUCTION")]


@pytest.mark.parametrize("prompt_name", _PROMPT_NAMES)
def test_prompt_placeholders_are_known_state_keys(prompt_name: str) -> None:
    placeholders = set(_PLACEHOLDER_RE.findall(getattr(prompts, prompt_name)))
    assert placeholders, f"{prompt_name} has no state placeholders"
    assert placeholders <= ALL_STATE_KEYS, placeholders - ALL_STATE_KEYS


def test_root_prompt_never_receives_quiz_answers() -> None:
    for answer_bearing_key in (StateKey.ACTIVE_QUIZ_KEY, StateKey.QUIZ_DRAFT):
        assert answer_bearing_key not in prompts.ROOT_INSTRUCTION
