from __future__ import annotations

from fermdocs.mapping.narrative_extractor import (
    MAX_EVIDENCE_LEN,
    verify_evidence,
)


def test_valid_evidence_with_value_passes():
    ok, reason = verify_evidence("grown at 30°C", "the cells were grown at 30°C", 30)
    assert ok, reason


def test_evidence_not_in_source_rejected():
    ok, reason = verify_evidence("grown at 30°C", "unrelated text", 30)
    assert not ok
    assert "not in source" in reason


def test_evidence_empty_rejected():
    ok, reason = verify_evidence("", "some source", 30)
    assert not ok
    assert "empty" in reason


def test_evidence_too_long_rejected():
    long_evidence = "x" * (MAX_EVIDENCE_LEN + 1)
    ok, reason = verify_evidence(long_evidence, long_evidence + " more", 1)
    assert not ok
    assert "too long" in reason


def test_value_not_in_evidence_rejected():
    # Source has the value, but evidence span doesn't include it.
    src = "yields varied across runs. final titer was 14.2 g/L."
    evidence = "yields varied across runs."
    ok, reason = verify_evidence(evidence, src, 14.2)
    assert not ok
    assert "not within evidence" in reason


def test_value_int_form_in_evidence_passes():
    src = "the run was conducted at 30 degrees."
    ok, reason = verify_evidence("conducted at 30 degrees", src, 30.0)
    assert ok, reason


def test_evidence_spans_too_many_sentences_rejected():
    src = "sentence one. sentence two has 14.2. sentence three. four."
    # Evidence containing the value but spanning many sentences:
    evidence = "sentence one. sentence two has 14.2. sentence three. four."
    ok, reason = verify_evidence(evidence, src, 14.2)
    assert not ok
    assert "sentences" in reason


def test_value_decimal_variant_matched():
    src = "the titer was 14.20 g/L on average."
    ok, reason = verify_evidence("titer was 14.20 g/L", src, 14.2)
    assert ok, reason
