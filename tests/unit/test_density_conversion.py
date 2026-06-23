"""De-LABS #0a — density-aware concentration conversion + closing the two silent
pass-throughs.

The defect these tests target is SILENCE: a gravimetric/fraction concentration
(%w/w, g/kg) was stored verbatim as g/L with status OK, ~10x wrong, no flag. So the
tests assert on the *legibility* of the refusal, not just that it happens — and on
the scoped guard that must NOT catch a legitimately-dimensionless density input.

Density is supplied as a TEST INPUT (never read from anywhere) so 0a's correctness is
provable in complete isolation from extraction. Fixtures use praaj's own stated g/L:
  B474: 10.2 %w/w x 1.0654 g/mL x 10 = 108.67 g/L (sheet states 108.666)
  B541: 15.04 %w/w x 1.0947 g/mL x 10 = 164.65 g/L (sheet states 164.65)
"""
from __future__ import annotations

import pytest

from fermdocs.domain.models import ConversionStatus
from fermdocs.units.converter import UnitConverter
from fermdocs.units.normalizer import (
    NormalizationAction,
    NormalizationHint,
    RuleBasedNormalizer,
)


class _ScriptedNormalizer:
    def __init__(self, hint):
        self._hint = hint
        self.calls = 0

    def normalize(self, unit_raw, canonical_unit, sample_value=None):
        self.calls += 1
        return self._hint


# --- correct conversion when density IS given (reproduces the sheets' g/L) --------

@pytest.mark.parametrize("pct, density, expected", [
    (10.2, 1.065354, 108.666),   # B474
    (15.04, 1.094747, 164.650),  # B541
])
def test_pct_ww_to_gl_with_density(pct, density, expected):
    conv = UnitConverter()
    r = conv.convert(pct, "%w/w", "g/L", density_g_per_ml=density)
    assert r.status == ConversionStatus.OK
    assert r.via == "density_rule"
    assert r.value_canonical == pytest.approx(expected, abs=0.01)


def test_g_per_kg_to_gl_with_density():
    # B474 summary g/Kg row: 102 g/kg x 1.065354 = 108.67 g/L (same answer, other unit)
    conv = UnitConverter()
    r = conv.convert(102.0, "g/kg", "g/L", density_g_per_ml=1.065354)
    assert r.status == ConversionStatus.OK
    assert r.value_canonical == pytest.approx(108.666, abs=0.01)


def test_unit_string_spacing_and_case_insensitive():
    conv = UnitConverter()
    for u in ["% w/w", "%W/W", "g/Kg", "g / kg"]:
        r = conv.convert(10.0, u, "g/L", density_g_per_ml=1.0)
        assert r.status == ConversionStatus.OK, u


# --- REFUSAL when density is absent: legible, not an exception, not a fallback -----

def test_pct_ww_to_gl_without_density_refuses_legibly():
    conv = UnitConverter()
    r = conv.convert(15.04, "%w/w", "g/L")  # density withheld
    assert r.status == ConversionStatus.FAILED          # not OK
    assert r.value_canonical is None                    # no fallback value
    assert "requires per-run density" in (r.error or "")  # names the missing input
    assert "g/L" in (r.error or "") and "%w/w" in (r.error or "")


def test_refusal_does_not_silently_pass_through_raw():
    # The historical bug: 15.04 %w/w stored as 15.04 g/L. Must NOT happen now.
    conv = UnitConverter()
    r = conv.convert(15.04, "%w/w", "g/L")
    assert r.value_canonical != 15.04


def test_dimensionless_hint_into_concentration_target_refuses():
    # Even if a normalizer declares the source DIMENSIONLESS, a g/L target must
    # refuse rather than store the bare number (the second silent pass-through).
    conv = UnitConverter()
    norm = _ScriptedNormalizer(NormalizationHint(
        action=NormalizationAction.DIMENSIONLESS, rationale="x",
        confidence=1.0, source="test"))
    r = conv.convert(15.04, "weird_frac_unit", "g/L", normalizer=norm)
    assert r.status == ConversionStatus.FAILED
    assert "dimensionless" in (r.error or "")


# --- the scoped guard must NOT catch legitimately-dimensionless cases (protects 0b)

def test_dimensionless_target_still_passes():
    # specific-gravity-like: a dimensionless value into a dimensionless/null target
    # must still resolve (this is the path 0b's density-as-specific-gravity will use).
    conv = UnitConverter()
    norm = _ScriptedNormalizer(NormalizationHint(
        action=NormalizationAction.DIMENSIONLESS, rationale="sg",
        confidence=1.0, source="test"))
    r = conv.convert(1.05, "sg", "OD600", normalizer=norm)
    assert r.status == ConversionStatus.OK          # NOT refused — target isn't g/L
    assert r.value_canonical == pytest.approx(1.05)


# --- non-concentration conversions are unaffected ---------------------------------

def test_ordinary_conversions_unaffected():
    conv = UnitConverter()
    assert conv.convert("14.2", "g/L", "g/L").status == ConversionStatus.OK
    assert conv.convert(5000.0, "mg/L", "g/L").value_canonical == pytest.approx(5.0)
    # real rule-based normalizer path for a genuinely dimensionless pH stays OK
    r = conv.convert(7.0, "pH", "pH", normalizer=RuleBasedNormalizer())
    assert r.status == ConversionStatus.OK
