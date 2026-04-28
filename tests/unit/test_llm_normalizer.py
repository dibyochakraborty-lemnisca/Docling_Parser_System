from __future__ import annotations

import pytest

from fermdocs.units.normalizer import (
    LLMUnitNormalizer,
    NormalizationAction,
    NormalizationHint,
)


class _CountingNormalizer(LLMUnitNormalizer):
    """Replaces _call_llm with a counter so we can assert cache behavior."""

    def __init__(self, response: dict | Exception):
        super().__init__(provider="anthropic")
        self.calls = 0
        self._response = response

    def _call_llm(self, unit_raw, canonical_unit, sample_value):
        self.calls += 1
        if isinstance(self._response, Exception):
            raise self._response
        return NormalizationHint.model_validate({**self._response, "source": "llm"})


def test_valid_response_returns_hint():
    n = _CountingNormalizer(
        {"action": "use_pint_expr", "pint_expr": "ug/(100*mg)",
         "rationale": "stripped annotation", "confidence": 0.85}
    )
    hint = n.normalize("µg/100mg of pellet", "g/L")
    assert hint.action == NormalizationAction.USE_PINT_EXPR
    assert hint.pint_expr == "ug/(100*mg)"
    assert hint.source == "llm"


def test_llm_exception_degrades_to_unconvertible():
    n = _CountingNormalizer(RuntimeError("network down"))
    hint = n.normalize("weird", "g/L")
    assert hint.action == NormalizationAction.UNCONVERTIBLE
    assert "network down" in hint.rationale


def test_cache_prevents_repeat_calls():
    n = _CountingNormalizer(
        {"action": "dimensionless", "rationale": "pH", "confidence": 0.95}
    )
    n.normalize("pH", "pH")
    n.normalize("pH", "pH")
    n.normalize("pH", "pH")
    assert n.calls == 1


def test_cache_keyed_by_canonical_too():
    n = _CountingNormalizer(
        {"action": "unconvertible", "rationale": "x", "confidence": 0.0}
    )
    n.normalize("weird", "g/L")
    n.normalize("weird", "mol/L")
    assert n.calls == 2


def test_invalid_action_rejected_by_pydantic():
    n = _CountingNormalizer(
        {"action": "use_factor", "rationale": "shouldnt happen", "confidence": 1.0}
    )
    # Pydantic validation rejects non-enum value -> caught -> unconvertible
    hint = n.normalize("weird", "g/L")
    assert hint.action == NormalizationAction.UNCONVERTIBLE
