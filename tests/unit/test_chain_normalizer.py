from __future__ import annotations

from fermdocs.units.normalizer import (
    ChainNormalizer,
    NormalizationAction,
    NormalizationHint,
)


class _Stub:
    def __init__(self, hint: NormalizationHint, name: str):
        self.hint = hint
        self.name = name
        self.calls = 0

    def normalize(self, unit_raw, canonical_unit, sample_value=None):
        self.calls += 1
        return self.hint


def _hint(action: NormalizationAction, source: str = "test") -> NormalizationHint:
    return NormalizationHint(
        action=action, rationale="x", confidence=0.9, source=source
    )


def test_first_normalizer_hits_short_circuits():
    rule = _Stub(_hint(NormalizationAction.USE_PINT_EXPR), "rule")
    llm = _Stub(_hint(NormalizationAction.UNCONVERTIBLE), "llm")
    chain = ChainNormalizer([rule, llm])
    chain.normalize("x", "g/L")
    assert rule.calls == 1
    assert llm.calls == 0


def test_first_misses_falls_through_to_second():
    rule = _Stub(_hint(NormalizationAction.UNCONVERTIBLE), "rule")
    llm = _Stub(_hint(NormalizationAction.DIMENSIONLESS), "llm")
    chain = ChainNormalizer([rule, llm])
    result = chain.normalize("x", "g/L")
    assert result.action == NormalizationAction.DIMENSIONLESS
    assert rule.calls == 1
    assert llm.calls == 1


def test_all_unconvertible_returns_last():
    rule = _Stub(_hint(NormalizationAction.UNCONVERTIBLE, "rule_based"), "rule")
    llm = _Stub(_hint(NormalizationAction.UNCONVERTIBLE, "llm"), "llm")
    chain = ChainNormalizer([rule, llm])
    result = chain.normalize("x", "g/L")
    assert result.action == NormalizationAction.UNCONVERTIBLE
