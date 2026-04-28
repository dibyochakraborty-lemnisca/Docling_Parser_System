from __future__ import annotations

from fermdocs.domain.models import ConversionStatus
from fermdocs.units.converter import UnitConverter
from fermdocs.units.normalizer import (
    ChainNormalizer,
    NormalizationAction,
    NormalizationHint,
    RuleBasedNormalizer,
)


class _ScriptedNormalizer:
    def __init__(self, hint: NormalizationHint):
        self._hint = hint
        self.calls = 0

    def normalize(self, unit_raw, canonical_unit, sample_value=None):
        self.calls += 1
        return self._hint


def test_pint_succeeds_normalizer_not_called():
    conv = UnitConverter()
    norm = _ScriptedNormalizer(
        NormalizationHint(
            action=NormalizationAction.UNCONVERTIBLE, rationale="x",
            confidence=0.0, source="test",
        )
    )
    result = conv.convert("14.2", "g/L", "g/L", normalizer=norm)
    assert result.status == ConversionStatus.OK
    assert result.via == "pint"
    assert norm.calls == 0


def test_pint_fails_normalizer_use_pint_expr_retries():
    conv = UnitConverter()
    norm = _ScriptedNormalizer(
        NormalizationHint(
            action=NormalizationAction.USE_PINT_EXPR,
            pint_expr="g/L",  # pretend the LLM rewrote 'flerbs/L' -> 'g/L'
            rationale="x", confidence=0.9, source="llm",
        )
    )
    result = conv.convert("14.2", "flerbs", "g/L", normalizer=norm)
    assert result.status == ConversionStatus.OK
    assert result.via == "llm"
    assert result.value_canonical == 14.2
    assert result.hint is not None


def test_pint_fails_normalizer_dimensionless():
    conv = UnitConverter()
    norm = _ScriptedNormalizer(
        NormalizationHint(
            action=NormalizationAction.DIMENSIONLESS,
            rationale="mass fraction", confidence=0.9, source="rule_based",
        )
    )
    result = conv.convert("0.5", "weird/units", "g/L", normalizer=norm)
    assert result.status == ConversionStatus.OK
    assert result.via == "rule_based"
    assert result.value_canonical == 0.5


def test_pint_fails_normalizer_unconvertible():
    conv = UnitConverter()
    norm = _ScriptedNormalizer(
        NormalizationHint(
            action=NormalizationAction.UNCONVERTIBLE,
            rationale="no idea", confidence=0.0, source="llm",
        )
    )
    result = conv.convert("14.2", "flerbs", "g/L", normalizer=norm)
    assert result.status == ConversionStatus.FAILED
    assert result.via == "llm"
    assert "unconvertible" in (result.error or "")


def test_use_pint_expr_with_missing_expr_fails():
    conv = UnitConverter()
    norm = _ScriptedNormalizer(
        NormalizationHint(
            action=NormalizationAction.USE_PINT_EXPR,
            pint_expr=None,
            rationale="oops", confidence=0.9, source="llm",
        )
    )
    result = conv.convert("1", "weird", "g/L", normalizer=norm)
    assert result.status == ConversionStatus.FAILED
    assert "missing pint_expr" in (result.error or "")


def test_end_to_end_with_real_rule_based_strips_of_annotation():
    conv = UnitConverter()
    norm = ChainNormalizer([RuleBasedNormalizer()])
    # 'g/L of broth' -> pint fails ('of' not a unit) -> rule strips 'of broth'
    # -> retry pint with 'g/L' -> success.
    result = conv.convert("5.0", "g/L of broth", "g/L", normalizer=norm)
    assert result.status == ConversionStatus.OK
    assert result.via == "rule_based"
    assert result.value_canonical == 5.0
    assert result.hint is not None
    assert "of" not in (result.hint.pint_expr or "")
