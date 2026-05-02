"""Tests for ingestion-side time + run_id auto-detection.

The characterization stage reads locator.run_id and locator.timestamp_h on
every observation. Without ingestion stamping these from the CSV, the
summary builder drops every row. These tests cover the heuristics that
recover those fields from real CSVs.
"""

from __future__ import annotations

import math

import pytest

from fermdocs.pipeline import (
    _coerce_run_id,
    _coerce_time_h,
    _detect_time_and_run_columns,
)


# ---------- _detect_time_and_run_columns ----------


def test_detects_indpensim_headers():
    """Canonical IndPenSim CSV has 'Time (h)' and 'Batch_ref' columns."""
    headers = ["Time (h)", "Fg", "RPM", "Batch_ref", "Batch_ref.1"]
    time_idx, run_idx = _detect_time_and_run_columns(headers)
    assert time_idx == 0
    assert run_idx == 3  # first match wins; .1 dup ignored


def test_detects_simple_time_column():
    headers = ["time", "biomass", "ph"]
    time_idx, _ = _detect_time_and_run_columns(headers)
    assert time_idx == 0


def test_detects_run_id_column():
    headers = ["t", "x", "run_id"]
    _, run_idx = _detect_time_and_run_columns(headers)
    assert run_idx == 2


def test_case_insensitive():
    headers = ["TIME", "BATCH"]
    t, r = _detect_time_and_run_columns(headers)
    assert t == 0
    assert r == 1


def test_whitespace_trimmed():
    headers = ["  Time (h)  ", " batch_ref "]
    t, r = _detect_time_and_run_columns(headers)
    assert t == 0
    assert r == 1


def test_no_match_returns_none():
    headers = ["x", "y", "z"]
    t, r = _detect_time_and_run_columns(headers)
    assert t is None
    assert r is None


def test_empty_headers():
    t, r = _detect_time_and_run_columns([])
    assert t is None
    assert r is None


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
    """IndPenSim's Batch_ref column is numeric like '1', '1.0', '2'.
    Normalize to RUN-NNNN so cohort comparisons are stable.
    """
    assert _coerce_run_id("1") == "RUN-0001"
    assert _coerce_run_id("1.0") == "RUN-0001"
    assert _coerce_run_id("42") == "RUN-0042"
    assert _coerce_run_id(2) == "RUN-0002"


def test_coerce_run_id_non_integer_float_kept_as_string():
    """A non-integer float would be a weird run id; preserve verbatim."""
    assert _coerce_run_id("1.5") == "1.5"


@pytest.mark.parametrize("raw", [None, "", "  ", "NaN", "nan"])
def test_coerce_run_id_empty_or_nan_returns_none(raw):
    assert _coerce_run_id(raw) is None


def test_coerce_run_id_strips_whitespace():
    assert _coerce_run_id("  batch_X  ") == "batch_X"
