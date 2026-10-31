"""Local LLM-as-judge for `custom_response_quality` (see eval_config.yaml)."""

import threading
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv()

_local = threading.local()


class _Verdict(BaseModel):
    score: int  # 1-5
    explanation: str


def _client() -> genai.Client:
    """One client per grading thread."""
    client = getattr(_local, "client", None)
    if client is None:
        client = _local.client = genai.Client()
    return client


def _extract_text(value: Any) -> str:
    """Extracts plain text from string, dict Content, or ResponseCandidate payloads."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if "response" in value:
            return _extract_text(value["response"])
        parts = value.get("parts") or []
        texts = [
            p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text")
        ]
        return "\n".join(texts) if texts else str(value)
    return str(value)


def evaluate(instance: dict[str, Any]) -> dict[str, Any]:
    prompt_text = _extract_text(instance.get("prompt"))
    response_text = _extract_text(instance.get("response"))
    reference_text = _extract_text(instance.get("reference"))

    rubric = (
        "Grade BookCompanion's final response on a 1-5 scale (1 poor, 5 excellent) for:\n"
        "1. Grounded accuracy against the tool outputs in the trace (no hallucinated facts).\n"
        "2. Warm, literary 'summarized book' readability and clear organization.\n"
        "3. Page citation discipline ([p. X]) when summarizing book chapters.\n"
        "4. Strict anti-spoiler discipline when presenting a check-in quiz (must NOT leak correct answers before the user responds)."
    )
    if reference_text:
        rubric += (
            "\nThe response should satisfy the expected behavior summary below; "
            "penalize missing core requirements or factual disagreement."
        )

    prompt = (
        f"You are an expert QA evaluator for an AI reading companion.\n{rubric}\n\n"
        f"User Prompt:\n{prompt_text}\n\n"
        f"Final Response:\n{response_text}\n\n"
    )
    if reference_text:
        prompt += f"Expected Behavior Summary:\n{reference_text}\n\n"
    prompt += f"Full Agent Trace:\n{instance.get('agent_data', '')}\n"

    response = _client().models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=_Verdict,
        ),
    )
    verdict = response.parsed
    if verdict is None:
        return {"score": 0, "explanation": response.text or ""}
    return {
        "score": max(1, min(5, verdict.score)),
        "explanation": verdict.explanation,
    }
