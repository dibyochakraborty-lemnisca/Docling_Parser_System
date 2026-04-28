from __future__ import annotations

from fermdocs.domain.models import ConversionStatus
from fermdocs.units.converter import UnitConverter


def test_no_canonical_unit_is_not_applicable():
    c = UnitConverter()
    r = c.convert("abc", None, None)
    assert r.status == ConversionStatus.NOT_APPLICABLE
    assert r.value_canonical is None


def test_passthrough_when_units_match():
    c = UnitConverter()
    r = c.convert("14.2", "g/L", "g/L")
    assert r.status == ConversionStatus.OK
    assert r.value_canonical == 14.2


def test_mg_per_ml_converts_to_g_per_l():
    c = UnitConverter()
    r = c.convert("14.2", "mg/mL", "g/L")
    assert r.status == ConversionStatus.OK
    assert abs(r.value_canonical - 14.2) < 1e-9


def test_fahrenheit_to_celsius():
    c = UnitConverter()
    r = c.convert("86", "degF", "degC")
    assert r.status == ConversionStatus.OK
    assert abs(r.value_canonical - 30.0) < 1e-6


def test_ph_passthrough_dimensionless():
    c = UnitConverter()
    r = c.convert("7.0", "pH", "pH")
    assert r.status == ConversionStatus.OK
    assert r.value_canonical == 7.0


def test_unknown_unit_fails_cleanly():
    c = UnitConverter()
    r = c.convert("1", "flerbs", "g/L")
    assert r.status == ConversionStatus.FAILED
    assert r.error


def test_non_numeric_value_fails_cleanly():
    c = UnitConverter()
    r = c.convert("not a number", "g/L", "g/L")
    assert r.status == ConversionStatus.FAILED
