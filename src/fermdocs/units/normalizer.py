from __future__ import annotations

import json
import os
import re
from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel, Field


class NormalizationAction(str, Enum):
    USE_PINT_EXPR = "use_pint_expr"
    DIMENSIONLESS = "dimensionless"
    UNCONVERTIBLE = "unconvertible"


class NormalizationHint(BaseModel):
    """What a normalizer tells the converter to do.

    The contract is intentionally narrow: the LLM never emits a converted value.
    It either rewrites the unit string for pint to retry, declares the value
    dimensionless (store as-is), or gives up.
    """

    action: NormalizationAction
    pint_expr: str | None = None
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)
    source: str = "unknown"  # "rule_based" | "llm" | "chain"


class UnitNormalizer(Protocol):
    def normalize(
        self, unit_raw: str, canonical_unit: str, sample_value: Any | None = None
    ) -> NormalizationHint: ...


_OF_X_PATTERN = re.compile(r"\s+of\s+[A-Za-z][A-Za-z0-9_ ]*\b", re.IGNORECASE)

_SUPERSCRIPTS = {
    "⁻¹": "^-1",
    "⁻²": "^-2",
    "⁻³": "^-3",
    "¹": "^1",
    "²": "^2",
    "³": "^3",
    "⁰": "^0",
    "⁴": "^4",
    "⁵": "^5",
}

_DIMENSIONLESS_TOKENS = {
    "od600",
    "od_600",
    "a600",
    "ph",
    "fold change",
    "fold_change",
    "log cfu/ml",
    "log_cfu_per_ml",
    "%",
    "% w/w",
    "% v/v",
    "percent",
}


class RuleBasedNormalizer:
    """Deterministic transforms for predictable failure modes.

    Returns USE_PINT_EXPR when it can plausibly fix the string, DIMENSIONLESS
    when the unit is in the known set, UNCONVERTIBLE otherwise (so the chain
    can fall through to LLM).
    """

    def normalize(
        self, unit_raw: str, canonical_unit: str, sample_value: Any | None = None
    ) -> NormalizationHint:
        if unit_raw is None:
            return _unconvertible("unit_raw is None", source="rule_based")

        normalized = unit_raw.strip()
        original = normalized

        # Known dimensionless tokens (case-insensitive, allowing trailing words like 'set').
        token = normalized.lower().split("(")[0].strip()
        if token in _DIMENSIONLESS_TOKENS:
            return NormalizationHint(
                action=NormalizationAction.DIMENSIONLESS,
                rationale=f"recognized dimensionless token: {original!r}",
                confidence=0.95,
                source="rule_based",
            )

        # Strip 'of <annotation>' suffix: 'µg/100mg of pellet' -> 'µg/100mg'.
        stripped = _OF_X_PATTERN.sub("", normalized).strip()

        # Replace Unicode superscripts and middle dot.
        replaced = stripped
        for sup, ascii_form in _SUPERSCRIPTS.items():
            replaced = replaced.replace(sup, ascii_form)
        replaced = replaced.replace("·", "*")  # middle dot

        # Insert explicit multiplication where pint expects it: 'g L^-1' -> 'g*L^-1'.
        replaced = re.sub(r"([A-Za-z\)\]])\s+([A-Za-z\(\[])", r"\1*\2", replaced)

        if replaced != normalized:
            return NormalizationHint(
                action=NormalizationAction.USE_PINT_EXPR,
                pint_expr=replaced,
                rationale=f"rewrote {original!r} -> {replaced!r}",
                confidence=0.85,
                source="rule_based",
            )

        return _unconvertible(f"no rule matched {original!r}", source="rule_based")


class ChainNormalizer:
    """Try each normalizer in order; first non-unconvertible result wins."""

    def __init__(self, normalizers: list[UnitNormalizer]) -> None:
        if not normalizers:
            raise ValueError("ChainNormalizer requires at least one normalizer")
        self._normalizers = normalizers

    def normalize(
        self, unit_raw: str, canonical_unit: str, sample_value: Any | None = None
    ) -> NormalizationHint:
        last: NormalizationHint | None = None
        for normalizer in self._normalizers:
            hint = normalizer.normalize(unit_raw, canonical_unit, sample_value)
            if hint.action != NormalizationAction.UNCONVERTIBLE:
                return hint
            last = hint
        return last or _unconvertible("no normalizers ran", source="chain")


class LLMUnitNormalizer:
    """LLM fallback for unit strings rules can't handle.

    Opt-in (FERMDOCS_USE_LLM_NORMALIZER=true). Caches per-(unit_raw, canonical_unit)
    over the lifetime of one instance so repeated cells don't re-call the LLM.
    Failures (network, malformed JSON, auth) degrade to UNCONVERTIBLE rather
    than crashing the pipeline.
    """

    _SYSTEM_PROMPT = (
        "You normalize unit strings that pint (a Python units library) failed to parse. "
        "You return EXACTLY ONE of three actions:\n"
        "1. use_pint_expr: rewrite the unit string into something pint can parse "
        "(strip annotations like 'of pellet', replace Unicode superscripts with '^N', "
        "insert explicit multiplication).\n"
        "2. dimensionless: the unit is conceptually dimensionless (pH, OD600, fold change, "
        "mass fractions like 'µg/100mg of pellet' -- when treating as fraction).\n"
        "3. unconvertible: you cannot make sense of it.\n\n"
        "You MUST NOT compute a converted numeric value. Code does the math.\n"
        "You MUST emit valid JSON matching the response schema."
    )

    def __init__(self, provider: str | None = None) -> None:
        self._provider = (
            provider
            or os.environ.get("FERMDOCS_NORMALIZER_PROVIDER")
            or os.environ.get("FERMDOCS_MAPPER_PROVIDER")
            or "anthropic"
        ).lower()
        self._cache: dict[tuple[str, str], NormalizationHint] = {}

    def normalize(
        self, unit_raw: str, canonical_unit: str, sample_value: Any | None = None
    ) -> NormalizationHint:
        key = (unit_raw or "", canonical_unit or "")
        if key in self._cache:
            return self._cache[key]
        try:
            hint = self._call_llm(unit_raw, canonical_unit, sample_value)
        except Exception as e:
            hint = _unconvertible(f"llm_error: {e}", source="llm")
        self._cache[key] = hint
        return hint

    def _call_llm(
        self, unit_raw: str, canonical_unit: str, sample_value: Any | None
    ) -> NormalizationHint:
        user_prompt = (
            f"unit_raw: {unit_raw!r}\n"
            f"canonical_unit: {canonical_unit!r}\n"
            f"sample_value: {sample_value!r}\n\n"
            "Return JSON matching the schema."
        )
        if self._provider == "anthropic":
            payload = self._call_anthropic(user_prompt)
        elif self._provider == "gemini":
            payload = self._call_gemini(user_prompt)
        else:
            raise ValueError(f"unknown normalizer provider: {self._provider}")
        hint = NormalizationHint.model_validate({**payload, "source": "llm"})
        return hint

    def _call_anthropic(self, user_prompt: str) -> dict[str, Any]:
        from anthropic import Anthropic

        client = Anthropic()
        response = client.messages.create(
            model=os.environ.get("FERMDOCS_MAPPER_MODEL", "claude-haiku-4-5-20251001"),
            max_tokens=512,
            system=self._SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            tools=[
                {
                    "name": "emit_normalization",
                    "description": "Emit a unit normalization hint.",
                    "input_schema": _ANTHROPIC_NORMALIZER_SCHEMA,
                }
            ],
            tool_choice={"type": "tool", "name": "emit_normalization"},
        )
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                return dict(block.input)
        raise ValueError("normalizer response missing tool_use block")

    def _call_gemini(self, user_prompt: str) -> dict[str, Any]:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model=os.environ.get("FERMDOCS_GEMINI_MODEL", "gemini-2.5-flash"),
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=self._SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=_GEMINI_NORMALIZER_SCHEMA,
                temperature=0.0,
            ),
        )
        text = response.text
        if not text:
            raise ValueError("empty gemini response")
        return json.loads(text)


_ANTHROPIC_NORMALIZER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["use_pint_expr", "dimensionless", "unconvertible"],
        },
        "pint_expr": {"type": ["string", "null"]},
        "rationale": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["action", "rationale", "confidence"],
}

_GEMINI_NORMALIZER_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "action": {
            "type": "STRING",
            "enum": ["use_pint_expr", "dimensionless", "unconvertible"],
        },
        "pint_expr": {"type": "STRING", "nullable": True},
        "rationale": {"type": "STRING"},
        "confidence": {"type": "NUMBER"},
    },
    "required": ["action", "rationale", "confidence"],
}


def _unconvertible(reason: str, *, source: str) -> NormalizationHint:
    return NormalizationHint(
        action=NormalizationAction.UNCONVERTIBLE,
        rationale=reason,
        confidence=0.0,
        source=source,
    )


def build_default_normalizer(use_llm: bool = False, provider: str | None = None) -> UnitNormalizer:
    """Convenience constructor for the standard chain.

    Default: rule-based only (deterministic, no LLM cost).
    With use_llm=True: rule-based first, LLM fallback.
    """
    chain: list[UnitNormalizer] = [RuleBasedNormalizer()]
    if use_llm:
        chain.append(LLMUnitNormalizer(provider=provider))
    return ChainNormalizer(chain)
