"""Unit tests for range_violation candidate generator.

Directly exercises severity boundaries: 2.0σ, 3.0σ, 5.0σ, and below-2σ no-op.
"""

import pytest

from fermdocs_characterize.candidates.range_violation import find_range_violations
from fermdocs_characterize.schema import Severity
from fermdocs_characterize.views.summary import Summary, SummaryRow


def _row(value: float, expected: float, std: float) -> SummaryRow:
    return SummaryRow(
        observation_id="obs-1",
        run_id="R1",
        time=24.0,
        variable="X",
        value=value,
        unit="u",
        expected=expected,
        expected_std_dev=std,
    )


def _summary(row: SummaryRow) -> Summary:
    return Summary(rows=[row], dropped=[], run_ids=["R1"], variables=["X"])


@pytest.mark.parametrize(
    "value,expected_severity",
    [
        (1.0, None),  # 0σ
        (1.05, None),  # 1σ
        (1.1, Severity.MINOR),  # exactly 2σ
        (1.149, Severity.MINOR),  # just under 3σ (2.98)
        (1.15, Severity.MAJOR),  # exactly 3σ (FP rounds correctly to 3.0)
        (1.249, Severity.MAJOR),  # just under 5σ (4.98)
        (1.25, Severity.CRITICAL),  # exactly 5σ
        (1.5, Severity.CRITICAL),  # 10σ
        (0.85, Severity.MAJOR),  # -3σ (negative direction)
        (0.7, Severity.CRITICAL),  # -6σ
    ],
)
def test_severity_thresholds(value, expected_severity):
    # nominal=1.0, std=0.05 → sigmas = (value - 1.0) / 0.05
    summary = _summary(_row(value, 1.0, 0.05))
    candidates = find_range_violations(summary, trajectories=[])
    if expected_severity is None:
        assert candidates == []
    else:
        assert len(candidates) == 1
        assert candidates[0].severity == expected_severity


def test_below_threshold_no_finding():
    # 1.5σ → no finding
    summary = _summary(_row(value=1.075, expected=1.0, std=0.05))
    candidates = find_range_violations(summary, trajectories=[])
    assert candidates == []


def test_negative_direction_above_threshold():
    # -4σ (major), value below nominal
    summary = _summary(_row(value=0.8, expected=1.0, std=0.05))
    candidates = find_range_violations(summary, trajectories=[])
    assert len(candidates) == 1
    assert candidates[0].severity == Severity.MAJOR
    assert candidates[0].statistics["sigmas"] == -4.0
    assert "below" in candidates[0].summary


def test_missing_specs_skipped():
    row = SummaryRow(
        observation_id="obs-1",
        run_id="R1",
        time=0.0,
        variable="X",
        value=10.0,
        unit="u",
        expected=None,
        expected_std_dev=None,
    )
    summary = Summary(rows=[row], dropped=[], run_ids=["R1"], variables=["X"])
    candidates = find_range_violations(summary, trajectories=[])
    assert candidates == []


def test_zero_std_dev_skipped():
    summary = _summary(_row(value=10.0, expected=1.0, std=0.0))
    candidates = find_range_violations(summary, trajectories=[])
    assert candidates == []


def test_confidence_matches_severity():
    # 5σ → critical → conf 0.99
    summary = _summary(_row(value=1.25, expected=1.0, std=0.05))
    candidates = find_range_violations(summary, trajectories=[])
    assert candidates[0].confidence == 0.99
    # 3σ → major → 0.95
    summary = _summary(_row(value=1.15, expected=1.0, std=0.05))
    assert find_range_violations(summary, trajectories=[])[0].confidence == 0.95
    # 2σ → minor → 0.85
    summary = _summary(_row(value=1.1, expected=1.0, std=0.05))
    assert find_range_violations(summary, trajectories=[])[0].confidence == 0.85
