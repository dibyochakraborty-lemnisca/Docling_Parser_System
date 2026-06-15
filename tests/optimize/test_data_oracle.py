"""DataSimulator: verify the optimizer against real runs, not a simulator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fermdocs_optimize.schema import Candidate
from fermdocs_optimize.simulators.data_backed import (
    DataSimulator,
    build_data_oracle,
    optimize_on_data,
)


def _cand(biomass=0.0, total_sub=0.0, malt_frac=0.0, dilution=0.0):
    return Candidate(
        biomass=biomass, total_sub=total_sub, malt_frac=malt_frac, dilution=dilution
    )


def test_knn_returns_nearest_run_titer():
    # two runs: low total_sub -> low titer, high -> high.
    sim = DataSimulator(
        knob_matrix=np.array([[100.0], [160.0]]),
        peaks=np.array([90.0, 150.0]),
        active_knobs=["total_sub"],
        k=1,
    )
    df = sim.simulate([_cand(total_sub=158.0), _cand(total_sub=102.0)], v0=10.0)
    peaks = {int(b): float(g["P"].max()) for b, g in df.groupby("batch")}
    assert peaks[0] == 150.0  # nearest to 160
    assert peaks[1] == 90.0   # nearest to 100


def test_pinned_knob_ignored_in_distance():
    # malt_frac is NOT an active knob; varying it must not change the answer.
    sim = DataSimulator(
        np.array([[100.0], [160.0]]), np.array([90.0, 150.0]),
        active_knobs=["total_sub"], k=1,
    )
    a = sim.simulate([_cand(total_sub=160.0, malt_frac=0.0)], v0=10.0)["P"].max()
    b = sim.simulate([_cand(total_sub=160.0, malt_frac=999.0)], v0=10.0)["P"].max()
    assert a == b == 150.0


def _obs(rows):
    return pd.DataFrame(rows, columns=["run_id", "variable", "time_h", "value"])


def _campaign():
    # 3 runs, total_sub (initial substrate) varies and tracks titer; volume
    # constant (dilution won't be an active knob); no biomass var.
    rows = []
    for rid, sub0, peak in [("R1", 100, 90), ("R2", 130, 120), ("R3", 160, 150)]:
        rows += [
            (rid, "substrate_g_l", 0.0, sub0),
            (rid, "substrate_g_l", 10.0, sub0 * 0.3),
            (rid, "product_g_l", 0.0, 0.0),
            (rid, "product_g_l", 48.0, peak),
            (rid, "volume_l", 0.0, 10.0),
        ]
    return _obs(rows)


def test_builder_derives_varying_knob_and_pins_malt_frac():
    oracle, box, active = build_data_oracle(_campaign())
    assert "total_sub" in active           # varies across runs
    assert "dilution" not in active        # volume constant -> not active
    # malt_frac pinned (praaj has no malt) — never faked into a real value
    assert box.malt_frac == (0.0, 0.0)
    # active knob box is the observed envelope
    assert box.total_sub == (100.0, 160.0)


def test_optimize_on_data_finds_best_within_envelope():
    report, active = optimize_on_data(_campaign(), n_lhs=40, seed=1)
    # best observed titer is 150 at the high-substrate end
    assert report.best_titer == pytest.approx(150.0, abs=1e-6)
    # total_sub should be pushed to the observed upper bound -> flagged on boundary
    assert "total_sub" in report.knobs_on_boundary


def test_too_few_runs_or_no_variation_refuses():
    with pytest.raises(ValueError):
        build_data_oracle(_obs([("R1", "product_g_l", 0.0, 90.0)]))  # 1 run
    flat = _obs([
        ("R1", "substrate_g_l", 0.0, 100.0), ("R1", "product_g_l", 0.0, 90.0),
        ("R2", "substrate_g_l", 0.0, 100.0), ("R2", "product_g_l", 0.0, 95.0),
    ])
    with pytest.raises(ValueError):  # substrate constant -> no varying knob
        build_data_oracle(flat)
