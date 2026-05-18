"""LLM-as-judge functions for eval suites.

Prompts are kept inline here but mirrored to eval/prompts/*.md as the
canonical reproducibility artifact. When you edit a prompt, edit both.

Judges return structured dicts so downstream metrics code is deterministic.
Failure handling: on any LLM error, return {"status": "error", "error": str},
never raise. The harness records the error row and moves on.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

# Default judge model is intentionally different from the pipeline mapper
# model (gemini-3-flash) and from the E3 baseline (gemini-3.1-pro-preview)
# to reduce same-model self-preference bias. Override via FERMDOCS_EVAL_JUDGE_MODEL.
DEFAULT_JUDGE_MODEL = "gemini-3-pro"


def _judge_model() -> str:
    return os.getenv("FERMDOCS_EVAL_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)


SPECIFICITY_PROMPT = """\
You are scoring a fermentation-failure hypothesis for SPECIFICITY and EVIDENCE GROUNDING.

A hypothesis is HIGH-specificity when it names concrete instruments, parameter
values, time windows, or observed anomalies tied to evidence in the bundle.
A hypothesis is LOW-specificity when it makes generic claims that could apply
to any fermentation run.

Score on a 1-5 integer scale:
1 = generic claims, no evidence anchors
2 = mostly generic, one or two named anchors
3 = mix of generic and specific
4 = mostly specific, well-anchored to evidence
5 = highly specific, every claim tied to named instruments/timepoints/values

Output STRICT JSON with two keys, no prose around it:
{"score": <1-5>, "rationale": "<one sentence>"}

HYPOTHESIS:
{hypothesis_text}
"""


PREFERENCE_PROMPT = """\
You are blindly comparing two fermentation-failure hypotheses for the SAME run.
Pick the one that is more specific, evidence-grounded, and actionable.

Output STRICT JSON with three keys, no prose around it:
{"winner": "A" | "B" | "tie", "rationale": "<one sentence>", "axes": {"specificity": "A"|"B"|"tie", "grounding": "A"|"B"|"tie", "actionability": "A"|"B"|"tie"}}

HYPOTHESIS A:
{a_text}

HYPOTHESIS B:
{b_text}
"""


def _strip_json(text: str) -> str:
    """Strip code fences and find the first JSON object in text."""
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```\s*$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text


def _call_gemini(prompt: str, model: str) -> str:
    """Single-shot Gemini call. Imported lazily so tests can monkeypatch."""
    from google import genai  # type: ignore

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text or ""


def judge_specificity(hypothesis_text: str) -> dict[str, Any]:
    """Score hypothesis specificity 1-5. Returns {"status", "score", "rationale"} or error."""
    if not hypothesis_text.strip():
        return {"status": "error", "error": "empty hypothesis_text"}
    prompt = SPECIFICITY_PROMPT.format(hypothesis_text=hypothesis_text)
    try:
        raw = _call_gemini(prompt, _judge_model())
        parsed = json.loads(_strip_json(raw))
        score = int(parsed["score"])
        if not 1 <= score <= 5:
            return {"status": "error", "error": f"score out of range: {score}", "raw": raw}
        return {
            "status": "ok",
            "score": score,
            "rationale": str(parsed.get("rationale", "")),
            "model": _judge_model(),
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


def judge_preference(a_text: str, b_text: str, *, seed_label: str = "s0") -> dict[str, Any]:
    """Blind A/B preference judge. seed_label is recorded for variance estimation."""
    if not a_text.strip() or not b_text.strip():
        return {"status": "error", "error": "empty hypothesis text"}
    prompt = PREFERENCE_PROMPT.format(a_text=a_text, b_text=b_text)
    try:
        raw = _call_gemini(prompt, _judge_model())
        parsed = json.loads(_strip_json(raw))
        winner = parsed["winner"]
        if winner not in ("A", "B", "tie"):
            return {"status": "error", "error": f"bad winner: {winner}", "raw": raw}
        return {
            "status": "ok",
            "winner": winner,
            "rationale": str(parsed.get("rationale", "")),
            "axes": parsed.get("axes", {}),
            "model": _judge_model(),
            "seed_label": seed_label,
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
