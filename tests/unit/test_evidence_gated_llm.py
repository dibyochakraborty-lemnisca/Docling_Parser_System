"""Tests for the shared evidence_gated_llm primitive."""

from __future__ import annotations

from fermdocs.mapping.evidence_gated_llm import (
    LLM_CONFIDENCE_CAP,
    MAX_EVIDENCE_LEN,
    value_string_forms,
    verify_substring_evidence,
)


def test_substring_evidence_happy_with_value():
    ok, reason = verify_substring_evidence(
        "grown at 30°C", "the cells were grown at 30°C", 30
    )
    assert ok, reason


def test_substring_evidence_happy_without_value():
    """value=None skips the value-form check; useful for non-numeric claims."""
    ok, reason = verify_substring_evidence(
        "Penicillium chrysogenum", "the strain Penicillium chrysogenum was used", None
    )
    assert ok, reason


def test_substring_evidence_empty_rejected():
    ok, reason = verify_substring_evidence("", "some source", 30)
    assert not ok
    assert "empty" in reason


def test_substring_evidence_too_long_rejected():
    long_ev = "x" * (MAX_EVIDENCE_LEN + 1)
    ok, reason = verify_substring_evidence(long_ev, long_ev + " more", 1)
    assert not ok
    assert "too long" in reason


def test_substring_evidence_not_in_source_rejected():
    ok, reason = verify_substring_evidence("hallucinated", "totally different", 1)
    assert not ok
    assert "not in source" in reason


def test_substring_evidence_value_not_in_span_rejected():
    src = "yields varied. final titer was 14.2 g/L."
    ok, reason = verify_substring_evidence("yields varied.", src, 14.2)
    assert not ok
    assert "not within evidence" in reason


def test_substring_evidence_too_many_sentences_rejected():
    src = "one. two has 14.2. three. four."
    ok, reason = verify_substring_evidence(src, src, 14.2)
    assert not ok
    assert "sentences" in reason


def test_value_string_forms_int_variants():
    forms = value_string_forms(30)
    assert "30" in forms
    assert "30.0" in forms


def test_value_string_forms_float_variants():
    forms = value_string_forms(14.2)
    assert "14.2" in forms
    assert "14.20" in forms


def test_value_string_forms_none_returns_empty():
    assert value_string_forms(None) == set()


def test_confidence_cap_constant_is_documented():
    assert 0.0 < LLM_CONFIDENCE_CAP <= 0.85
