"""Tests for ingestion-side time-column detection and run_id coercion.

Run-id resolution itself moved to a strategy chain in
parsing/run_id_resolver.py — see test_run_id_resolver.py for that. This
file keeps the simple time-column header heuristic and the value-coercion
helpers used by both the column-strategy path and the verbatim-strategy path.
"""

from __future__ import annotations

import math

import pytest

from fermdocs.pipeline import (
    _coerce_run_id,
    _coerce_time_h,
    _detect_time_column,
)


# ---------- _detect_time_column ----------


def test_detects_indpensim_time_header():
    """Canonical IndPenSim CSV uses 'Time (h)'."""
    headers = ["Time (h)", "Fg", "RPM", "Batch_ref", "Batch_ref.1"]
    assert _detect_time_column(headers) == 0


def test_detects_simple_time_column():
    headers = ["time", "biomass", "ph"]
    assert _detect_time_column(headers) == 0


def test_case_insensitive():
    headers = ["TIME"]
    assert _detect_time_column(headers) == 0


def test_whitespace_trimmed():
    headers = ["  Time (h)  "]
    assert _detect_time_column(headers) == 0


def test_no_match_returns_none():
    headers = ["x", "y", "z"]
    assert _detect_time_column(headers) is None


def test_empty_headers():
    assert _detect_time_column([]) is None


# ---------- _coerce_time_h ----------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("0.2", 0.2),
        ("12.5", 12.5),
        ("0", 0.0),
        (1.5, 1.5),
    ],
)
def test_coerce_time_valid(raw, expected):
    assert _coerce_time_h(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", [None, "", "abc", "not_a_number"])
def test_coerce_time_invalid_returns_none(raw):
    assert _coerce_time_h(raw) is None


def test_coerce_time_nan_returns_none():
    assert _coerce_time_h(float("nan")) is None
    assert _coerce_time_h("NaN") is None


def test_coerce_time_inf_returns_none():
    assert _coerce_time_h(float("inf")) is None


# ---------- _coerce_run_id ----------


def test_coerce_run_id_string_passthrough():
    assert _coerce_run_id("RUN-A001") == "RUN-A001"
    assert _coerce_run_id("EXP-2024-01") == "EXP-2024-01"


def test_coerce_run_id_numeric_normalized():
    """Numeric ids like '1' or '1.0' normalize to RUN-NNNN so cohort
    comparisons are stable.
    """
    assert _coerce_run_id("1") == "RUN-0001"
    assert _coerce_run_id("1.0") == "RUN-0001"
    assert _coerce_run_id("42") == "RUN-0042"
    assert _coerce_run_id(2) == "RUN-0002"


def test_coerce_run_id_non_integer_float_kept_as_string():
    assert _coerce_run_id("1.5") == "1.5"


@pytest.mark.parametrize("raw", [None, "", "  ", "NaN", "nan"])
def test_coerce_run_id_empty_or_nan_returns_none(raw):
    assert _coerce_run_id(raw) is None


def test_coerce_run_id_strips_whitespace():
    assert _coerce_run_id("  batch_X  ") == "batch_X"
