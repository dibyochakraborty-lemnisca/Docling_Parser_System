"""LLM-backed classifier for UserQuestion.

Plan ref: plans/2026-05-04-user-question-and-hitl.md (PR-A commit 2).

Takes the user's raw question text + bundle metadata (run_ids,
variables) and returns a UserQuestion with `shape`, `affected_runs`,
and `affected_variables` populated. The classifier runs ONCE per run,
at API entry-point time, before any subprocess fires.

Error posture: ADDITIVE. A failed classifier produces a UserQuestion
with shape='open' and empty hints — the question still threads through
every downstream agent, just without shape-aware prompt routing. The
classifier never raises across the API boundary; it always returns a
valid UserQuestion.

Validation: the LLM may return shape values outside the literal set
or run_id strings that don't match the bundle. The classifier validates
output against the bundle's actual `available_run_ids` and
`available_variables` sets. Unknown shape → 'open'. Unknown
run_id/variable → silently dropped (logged).

Architecture mirror: same Protocol + structured-output pattern as
GeminiIdentityClient. Stub mode (client=None) returns shape='open'
+ empty hints — used by tests and offline runs.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol

from fermdocs.domain.user_question import (
    QuestionShape,
    UserQuestion,
)

_log = logging.getLogger(__name__)

VALID_SHAPES = ("scoping", "mechanistic", "comparative", "open")


# -----------------------------------------------------------------------------
# Client protocol
# -----------------------------------------------------------------------------


class QuestionClassifierLLMClient(Protocol):
    """Minimal protocol: tests supply a scripted client; production wraps Gemini.

    The client takes a system prompt + user prompt and returns a JSON-shaped
    dict matching the schema below. Errors propagate up to classify_user_question
    which catches and falls back to shape='open'.
    """

    def call(self, system: str, user: str) -> dict[str, Any]: ...


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------


def classify_user_question(
    text: str,
    *,
    available_run_ids: list[str],
    available_variables: list[str],
    client: QuestionClassifierLLMClient | None,
    raised_by: str = "user",
) -> UserQuestion:
    """Classify a user question against a bundle's actual data.

    `text` is the human-typed question (1-2000 chars).
    `available_run_ids` is the set of run_ids in the bundle (e.g.
        ["RUN-0001", "RUN-0002"]).
    `available_variables` is the set of golden_schema column names present
        in the bundle (e.g. ["biomass_g_l", "wcw_g_l"]).
    `client` is None for stub mode; non-None for production.
    `raised_by` flows through to the returned UserQuestion (defaults to
        "user"; "user_followup" or "operator_mid_debate" override the source).

    Returns a UserQuestion with shape + hints populated. Never raises;
    LLM failure → fallback UserQuestion(text=text, shape='open',
    affected_runs=[], affected_variables=[]).
    """
    if not text or not text.strip():
        raise ValueError("classify_user_question: text must be non-empty")

    if client is None:
        return _fallback(text, raised_by, reason="stub mode (no LLM client)")

    try:
        payload = client.call(
            _SYSTEM_PROMPT,
            _build_user_prompt(text, available_run_ids, available_variables),
        )
    except Exception as exc:
        _log.warning(
            "question classifier LLM call failed (%s: %s); falling back to shape='open'",
            exc.__class__.__name__,
            str(exc)[:200],
        )
        return _fallback(text, raised_by, reason=f"LLM error {exc.__class__.__name__}")

    return _validate_and_build(
        payload=payload,
        text=text,
        available_run_ids=available_run_ids,
        available_variables=available_variables,
        raised_by=raised_by,
    )


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------


def _fallback(text: str, raised_by: str, *, reason: str) -> UserQuestion:
    """Construct a safe-default UserQuestion when the LLM path can't be used.

    `reason` goes only to the log, not the returned object — the user
    question is the directive, not a debug artifact.
    """
    _log.info("question classifier fallback: %s", reason)
    return UserQuestion(
        text=text,
        shape="open",
        affected_variables=[],
        affected_runs=[],
        raised_by=raised_by,  # type: ignore[arg-type]
    )


def _validate_and_build(
    *,
    payload: dict[str, Any],
    text: str,
    available_run_ids: list[str],
    available_variables: list[str],
    raised_by: str,
) -> UserQuestion:
    """Coerce a raw LLM response dict into a UserQuestion.

    Hard contract:
      - shape must be in VALID_SHAPES; else fall back to 'open'
      - affected_runs are kept only if they appear in available_run_ids
        (case-insensitive)
      - affected_variables are kept only if they appear in
        available_variables (case-insensitive substring on either side
        — picks up 'biomass' when the bundle has 'biomass_g_l')
      - max 10 variables / 20 runs (UserQuestion's own caps)
    """
    raw_shape = (payload.get("shape") or "").strip().lower()
    shape: QuestionShape = (
        raw_shape if raw_shape in VALID_SHAPES else "open"  # type: ignore[assignment]
    )

    raw_runs = payload.get("affected_runs") or []
    if not isinstance(raw_runs, list):
        raw_runs = []
    valid_runs = _filter_runs(raw_runs, available_run_ids)

    raw_vars = payload.get("affected_variables") or []
    if not isinstance(raw_vars, list):
        raw_vars = []
    valid_vars = _filter_variables(raw_vars, available_variables)

    return UserQuestion(
        text=text,
        shape=shape,
        affected_runs=valid_runs[:20],
        affected_variables=valid_vars[:10],
        raised_by=raised_by,  # type: ignore[arg-type]
    )


def _filter_runs(
    raw_runs: list[Any], available_run_ids: list[str]
) -> list[str]:
    """Keep only runs that match (case-insensitive) the bundle's run_ids.

    Returns canonical-cased run_ids from `available_run_ids` so downstream
    consumers can do exact set membership. Unknown runs are silently
    dropped — the classifier hallucinated them, no need to surface as
    an error.
    """
    canonical = {r.upper(): r for r in available_run_ids if isinstance(r, str)}
    out: list[str] = []
    seen: set[str] = set()
    for r in raw_runs:
        if not isinstance(r, str):
            continue
        canon = canonical.get(r.upper())
        if canon and canon not in seen:
            seen.add(canon)
            out.append(canon)
    return out


def _filter_variables(
    raw_vars: list[Any], available_variables: list[str]
) -> list[str]:
    """Keep only variables that appear in the bundle's variable list.

    Substring match in either direction so 'biomass' picks up
    'biomass_g_l', and 'wcw_g_l' matches a candidate hint of 'wcw'.
    Returns canonical names from `available_variables`.
    """
    available_lower = {v.lower(): v for v in available_variables if isinstance(v, str)}
    out: list[str] = []
    seen: set[str] = set()
    for v in raw_vars:
        if not isinstance(v, str):
            continue
        v_low = v.lower().strip()
        if not v_low:
            continue
        # Direct match first.
        if v_low in available_lower:
            canon = available_lower[v_low]
            if canon not in seen:
                seen.add(canon)
                out.append(canon)
            continue
        # Substring fallback: any available variable containing the hint
        # OR being contained in it.
        for cand_low, canon in available_lower.items():
            if v_low in cand_low or cand_low in v_low:
                if canon not in seen:
                    seen.add(canon)
                    out.append(canon)
                break
    return out


# -----------------------------------------------------------------------------
# Prompts
# -----------------------------------------------------------------------------


_SYSTEM_PROMPT = """\
You classify a fermentation-experiment user question into a small typed
shape so the downstream debate pipeline can route prompts correctly.

Return JSON only. Do not explain.

shape:
  - scoping: question narrows to a specific run, time window, or variable.
    Examples: "Why did RUN-0034 plateau early?" "What happened around 30h?"
  - mechanistic: question proposes a mechanism for the system to test.
    Examples: "Was DO limitation responsible for the biomass drop?"
              "Did overflow metabolism cause the carbon balance failure?"
  - comparative: question points at a comparison axis to surface.
    Examples: "What's different between high-yield and low-yield batches?"
              "Compare BATCH-04 against BATCH-05"
  - open: any question that doesn't fit the above. Default when unclear.

affected_runs: list of RUN-XXXX identifiers MENTIONED IN the question.
  Use the exact form from the AVAILABLE_RUN_IDS list. Case-insensitive
  matching is fine. Empty list when the question doesn't reference
  specific runs.

affected_variables: list of VARIABLE NAMES MENTIONED IN the question that
  appear in AVAILABLE_VARIABLES. Use the exact column name from the list.
  Don't invent names. Empty list when the question doesn't reference
  specific variables.

Output JSON shape:
{
  "shape": "scoping" | "mechanistic" | "comparative" | "open",
  "affected_runs": [ "RUN-..." ],
  "affected_variables": [ "..." ]
}
"""


def _build_user_prompt(
    text: str,
    available_run_ids: list[str],
    available_variables: list[str],
) -> str:
    runs_str = ", ".join(available_run_ids[:50]) if available_run_ids else "(none)"
    vars_str = ", ".join(available_variables[:50]) if available_variables else "(none)"
    return (
        f"AVAILABLE_RUN_IDS: {runs_str}\n"
        f"AVAILABLE_VARIABLES: {vars_str}\n\n"
        f"USER_QUESTION:\n{text}\n"
    )


# -----------------------------------------------------------------------------
# Gemini-structured-output schema (consumed by GeminiQuestionClassifierClient)
# -----------------------------------------------------------------------------


_GEMINI_CLASSIFIER_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "shape": {
            "type": "STRING",
            "enum": list(VALID_SHAPES),
        },
        "affected_runs": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
        "affected_variables": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
    },
    "required": ["shape", "affected_runs", "affected_variables"],
}


# -----------------------------------------------------------------------------
# Production Gemini client
# -----------------------------------------------------------------------------


class GeminiQuestionClassifierClient:
    """Production implementation of QuestionClassifierLLMClient.

    Wraps Google Gemini structured output, mirroring the
    GeminiIdentityClient pattern. Lazy-imports the SDK so projects
    without the gemini extra installed still load the module.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        import os

        _DEFAULT_MODEL = "gemini-3-flash"
        self._model = (
            model
            or os.environ.get("FERMDOCS_QUESTION_CLASSIFIER_MODEL")
            or os.environ.get("FERMDOCS_GEMINI_MODEL", _DEFAULT_MODEL)
        )
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")

    def call(self, system: str, user: str) -> dict[str, Any]:
        from google import genai  # lazy
        from google.genai import types

        client = genai.Client(api_key=self._api_key)
        response = client.models.generate_content(
            model=self._model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_schema=_GEMINI_CLASSIFIER_SCHEMA,
                temperature=0.0,
            ),
        )
        text = response.text
        if not text:
            raise ValueError("Gemini returned empty classifier response")
        return json.loads(text)
