"""LLM-as-judge functions for the head-to-head eval.

The judge takes two outputs (A, B) for the same question and scores each
on four axes (1-10), then declares an overall winner. Order is
counterbalanced by the caller so the judge cannot exploit position bias.

Reproducibility: the prompts in this module ARE the artifact for the
paper supplement. Do not edit them silently — bump JUDGE_PROMPT_VERSION
when changing prompts so prior result rows can be filtered.

Failure handling: on any LLM error or JSON parse failure, return
{"status": "error", "error": ...}. The harness records error rows and
moves on.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

JUDGE_DEFAULT_MODEL = "gemini-3.1-pro-preview"
JUDGE_PROMPT_VERSION = "v1-2026-05-19"

JUDGE_AXES = ("specificity", "grounding", "actionability", "honesty")


def _judge_model() -> str:
    return os.environ.get("FERMDOCS_EVAL_JUDGE_MODEL", JUDGE_DEFAULT_MODEL)


HEAD_TO_HEAD_PROMPT = """\
You are an expert reviewer of fermentation-process analyses. Two systems
were given the SAME bundle and the SAME question. They produced the two
answers below. Score each answer on four axes (1-10) and pick an overall
winner.

Axes (each 1-10, integers only):
  SPECIFICITY: names specific instruments, time windows, run IDs,
    numerical values, finding identifiers. Higher = more specific.
  GROUNDING: claims tie to evidence visible in the bundle. Penalize
    inventions or claims that go beyond what the data could support.
  ACTIONABILITY: ends with concrete next steps a process engineer could
    execute. Honest "insufficient evidence" counts as actionable when
    the question doesn't admit a recommendation.
  HONESTY: acknowledges uncertainty, schema artifacts, or limits in the
    data. Penalize overclaiming.

Output STRICT JSON (no prose around it):
{
  "scores": {
    "A": {"specificity": <1-10>, "grounding": <1-10>, "actionability": <1-10>, "honesty": <1-10>},
    "B": {"specificity": <1-10>, "grounding": <1-10>, "actionability": <1-10>, "honesty": <1-10>}
  },
  "winner": "A" | "B" | "tie",
  "rationale": "<one short sentence>"
}

QUESTION:
{question}

ANSWER A:
{a_text}

ANSWER B:
{b_text}
"""


def _strip_json(text: str) -> str:
    """Strip code fences and find the first JSON object in text."""
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```\s*$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text


def _call_gemini(prompt: str, model: str) -> str:
    """Single-shot Gemini call. Lazily imported so tests can monkeypatch."""
    from google import genai  # type: ignore

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text or ""


def _validate_scores(block: dict) -> bool:
    if not isinstance(block, dict):
        return False
    for axis in JUDGE_AXES:
        v = block.get(axis)
        if not isinstance(v, int) or not (1 <= v <= 10):
            return False
    return True


def judge_head_to_head(
    *,
    question: str,
    a_text: str,
    b_text: str,
    seed_label: str = "s0",
) -> dict[str, Any]:
    """Score two answers A/B on four axes and pick a winner.

    Counterbalancing of treatment/baseline -> A/B is the caller's job.
    seed_label is recorded for variance estimation.
    """
    if not (a_text.strip() and b_text.strip()):
        return {"status": "error", "error": "empty answer text"}
    if not question.strip():
        return {"status": "error", "error": "empty question"}

    prompt = HEAD_TO_HEAD_PROMPT.format(
        question=question, a_text=a_text, b_text=b_text
    )
    try:
        raw = _call_gemini(prompt, _judge_model())
        parsed = json.loads(_strip_json(raw))
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    scores = parsed.get("scores") or {}
    if not _validate_scores(scores.get("A")) or not _validate_scores(scores.get("B")):
        return {
            "status": "error",
            "error": "judge returned malformed or out-of-range scores",
            "raw": raw[:1000],
        }

    winner = parsed.get("winner")
    if winner not in ("A", "B", "tie"):
        return {
            "status": "error",
            "error": f"judge returned invalid winner: {winner!r}",
            "raw": raw[:1000],
        }

    return {
        "status": "ok",
        "scores": scores,  # {"A": {...}, "B": {...}}
        "winner": winner,
        "rationale": str(parsed.get("rationale", ""))[:500],
        "seed_label": seed_label,
        "judge_model": _judge_model(),
        "prompt_version": JUDGE_PROMPT_VERSION,
    }
