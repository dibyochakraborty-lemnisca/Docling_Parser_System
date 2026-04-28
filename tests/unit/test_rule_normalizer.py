from __future__ import annotations

from fermdocs.units.normalizer import (
    NormalizationAction,
    RuleBasedNormalizer,
)


def test_strip_of_pellet_annotation():
    n = RuleBasedNormalizer()
    hint = n.normalize("µg/100mg of pellet", "g/L")
    assert hint.action == NormalizationAction.USE_PINT_EXPR
    assert "of" not in hint.pint_expr
    assert hint.source == "rule_based"


def test_strip_of_ipm_layer():
    n = RuleBasedNormalizer()
    hint = n.normalize("µg/mL of IPM Layer", "g/L")
    assert hint.action == NormalizationAction.USE_PINT_EXPR
    assert "IPM" not in hint.pint_expr


def test_unicode_superscript_replacement():
    n = RuleBasedNormalizer()
    hint = n.normalize("g·L⁻¹·h⁻¹", "g/(L*h)")
    assert hint.action == NormalizationAction.USE_PINT_EXPR
    assert "⁻" not in hint.pint_expr
    assert "·" not in hint.pint_expr
    assert "^-1" in hint.pint_expr


def test_known_dimensionless_ph():
    n = RuleBasedNormalizer()
    hint = n.normalize("pH", "pH")
    assert hint.action == NormalizationAction.DIMENSIONLESS
    assert hint.confidence >= 0.9


def test_known_dimensionless_od600():
    n = RuleBasedNormalizer()
    hint = n.normalize("OD600", "OD600")
    assert hint.action == NormalizationAction.DIMENSIONLESS


def test_unknown_unit_returns_unconvertible():
    n = RuleBasedNormalizer()
    hint = n.normalize("flerbs", "g/L")
    assert hint.action == NormalizationAction.UNCONVERTIBLE


def test_none_unit_unconvertible():
    n = RuleBasedNormalizer()
    hint = n.normalize(None, "g/L")  # type: ignore[arg-type]
    assert hint.action == NormalizationAction.UNCONVERTIBLE


def test_implicit_multiplication_inserted():
    n = RuleBasedNormalizer()
    hint = n.normalize("g L^-1", "g/L")
    # 'g L^-1' -> 'g*L^-1' (so pint can parse)
    assert hint.action == NormalizationAction.USE_PINT_EXPR
    assert "*" in hint.pint_expr
