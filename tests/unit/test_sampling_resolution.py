"""Item 4: rate/lag findings must declare the data-derived sampling resolution,
and a rate is never reported instantaneously at t=0."""
from __future__ import annotations

from fermdocs_characterize.agents.catalog_runner_adapters import _sampling_resolution_h


def test_sampling_resolution_is_median_spacing():
    # ~8h sampling (the praaj cadence)
    assert _sampling_resolution_h([0, 8, 16, 24, 32]) == 8.0


def test_sampling_resolution_handles_irregular_spacing():
    # spacings 2,2,10,2 -> median 2
    assert _sampling_resolution_h([0, 2, 4, 14, 16]) == 2.0


def test_sampling_resolution_none_when_no_intervals():
    assert _sampling_resolution_h([5.0]) is None
    assert _sampling_resolution_h([3.0, 3.0, 3.0]) is None  # no positive gaps
