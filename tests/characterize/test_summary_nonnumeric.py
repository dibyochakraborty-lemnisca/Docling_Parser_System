"""build_summary must not crash on a non-numeric value (regression).

A stray "-" missing-marker that slips through ingestion used to crash the
whole characterize stage at float("-"). It must be dropped, not fatal.
"""

from __future__ import annotations

from fermdocs_characterize.views.summary import build_summary


class _NoSpecs:
    def get(self, _variable):
        return None


def _obs(obs_id, value):
    return {
        "observation_id": obs_id,
        "value": value,
        "unit": "g/L",
        "source": {"locator": {"run_id": "R1", "timestamp_h": 0.0}},
    }


def test_nonnumeric_value_is_dropped_not_fatal():
    dossier = {
        "golden_columns": {
            "product_g_l": {"observations": [_obs("o1", "-"), _obs("o2", "12.5")]},
        }
    }
    summary = build_summary(dossier, _NoSpecs())
    kept = [r for r in summary.rows]
    assert len(kept) == 1 and kept[0].observation_id == "o2"
    assert kept[0].value == 12.5
    assert any(d.observation_id == "o1" for d in summary.dropped)
