"""Discover a lever->titer equation, validate (leave-run-out CV), optimize.

Levers are discovered from the experiment's OWN data (run_conditions metadata +
varying observation channels), never a hardcoded knob list. These tests exercise
the surrogate path (the mechanistic ODE fit refuses fast on sparse 2-point
trajectories, so the fallback runs) over both numeric and categorical levers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fermdocs_optimize.data_equation import (
    EquationResult,
    _apply_sanity_guards,
    _compile,
    discover_and_optimize,
    discover_surrogate,
    optimize_surrogate,
)
from fermdocs_optimize.lever_discovery import Lever, build_design, discover_levers


def test_compile_rejects_unknown_symbols():
    with pytest.raises(ValueError):
        _compile("p0 + total_sub + sneaky", ["total_sub"], ["p0"])  # 'sneaky' not allowed


def test_compile_fits_and_evaluates():
    f = _compile("p0 + p1*total_sub", ["total_sub"], ["p0", "p1"])
    X = np.array([[100.0], [200.0]])
    out = f(X, np.array([10.0, 0.5]))
    assert list(out) == [60.0, 110.0]


def _quadratic_runs(n=9, vertex=140.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    subs = np.linspace(90, 165, n)
    runs = []
    for i, s in enumerate(subs):
        peak = 130.0 - 0.01 * (s - vertex) ** 2 + (rng.normal(0, noise) if noise else 0.0)
        runs.append({"run_id": f"R{i}", "peak": float(peak), "total_sub": float(s)})
    return runs


def _design_from_numeric(runs, lever_name="substrate_g_l.initial"):
    """Build a Design with one numeric lever (values=total_sub, outcome=peak)."""
    lev = Lever(lever_name, "numeric", "derived",
                {r["run_id"]: r["total_sub"] for r in runs})
    obs = pd.DataFrame(
        [(r["run_id"], "product_g_l", 48.0, r["peak"]) for r in runs],
        columns=["run_id", "variable", "time_h", "value"],
    )
    return build_design([lev], obs)


def test_surrogate_discovers_and_optimizes_interior_numeric_optimum():
    design = _design_from_numeric(_quadratic_runs())
    found = discover_surrogate(design)
    assert found is not None
    spec, theta, cv = found
    assert cv > 0.9  # quadratic generalizes
    knobs, pred, on_b = optimize_surrogate(design, spec, theta)
    # vertex at 140 is interior to [90,165] -> optimum near 140, NOT on boundary
    assert 130 <= knobs["substrate_g_l.initial"] <= 150
    assert "substrate_g_l.initial" not in on_b
    assert pred == pytest.approx(130.0, abs=2.0)


def _obs_from_runs(runs):
    # long-format observations: substrate_g_l initial = total_sub, product peak.
    rows = []
    for r in runs:
        rows += [
            (r["run_id"], "substrate_g_l", 0.0, r["total_sub"]),
            (r["run_id"], "substrate_g_l", 10.0, r["total_sub"] * 0.3),
            (r["run_id"], "product_g_l", 0.0, 0.0),
            (r["run_id"], "product_g_l", 48.0, r["peak"]),
        ]
    return pd.DataFrame(rows, columns=["run_id", "variable", "time_h", "value"])


def test_end_to_end_clears_and_optimizes_on_clean_numeric_data():
    out = discover_and_optimize(_obs_from_runs(_quadratic_runs()))
    assert out.cleared is True
    assert out.family == "surrogate"          # mechanistic refuses on 2-pt trajectories
    # the lever is discovered from the data, not hardcoded
    assert "substrate_g_l.initial" in out.best_knobs
    assert 130 <= out.best_knobs["substrate_g_l.initial"] <= 150
    assert out.cv_r2 > 0.9
    assert [lev["name"] for lev in out.levers] == ["substrate_g_l.initial"]


def test_end_to_end_refuses_on_noise():
    # peaks uncorrelated with the lever -> no equation generalizes -> refuse.
    rng = np.random.default_rng(3)
    runs = [{"run_id": f"R{i}", "peak": float(rng.normal(100, 30)),
             "total_sub": float(s)} for i, s in enumerate(np.linspace(90, 165, 10))]
    out = discover_and_optimize(_obs_from_runs(runs))
    assert out.cleared is False
    assert "did not generalize" in out.rationale or "no surrogate" in out.rationale
    assert out.best_knobs == {}


# --- categorical levers from metadata (the praaj nitrogen-source spine) --------

def _categorical_dossier_and_obs(seed=0):
    """8 runs split across 2 nitrogen sources; one source yields a higher titer.
    The lever lives in run_conditions metadata, not in the observations."""
    rng = np.random.default_rng(seed)
    sources = ["CSL", "YE"]
    base = {"CSL": 100.0, "YE": 135.0}
    run_conditions, rows = {}, []
    for i in range(8):
        src = sources[i % 2]
        run_id = f"R{i}"
        run_conditions[run_id] = {"nitrogen_source": {"value": src}}
        peak = base[src] + rng.normal(0, 1.0)
        rows.append((run_id, "product_g_l", 48.0, float(peak)))
    dossier = {"run_conditions": run_conditions}
    obs = pd.DataFrame(rows, columns=["run_id", "variable", "time_h", "value"])
    return dossier, obs


def test_discover_levers_finds_categorical_metadata_lever():
    dossier, obs = _categorical_dossier_and_obs()
    levers = discover_levers(dossier, obs)
    names = {lev.name for lev in levers}
    assert "nitrogen_source" in names
    lev = next(l for l in levers if l.name == "nitrogen_source")
    assert lev.kind == "categorical"
    assert lev.source == "metadata"
    assert set(lev.categories) == {"CSL", "YE"}


def test_end_to_end_optimizes_categorical_lever_picks_best_observed_level():
    dossier, obs = _categorical_dossier_and_obs()
    out = discover_and_optimize(obs, dossier=dossier)
    assert out.cleared is True
    # picks the better-titer source, and only an OBSERVED level (no invented mix)
    assert out.best_knobs["nitrogen_source"] == "YE"
    assert out.predicted_peak == pytest.approx(135.0, abs=3.0)


# --- item 8: data-relative sanity guards on the optimizer output ---------------

def _obs_with_objective(peaks, *, volume=None):
    """Long obs frame: one product_g_l peak per run, optional volume series."""
    rows = []
    for i, pk in enumerate(peaks):
        rows.append((f"R{i}", "product_g_l", 48.0, float(pk)))
        if volume is not None:
            v0, v1 = volume
            rows += [(f"R{i}", "volume_l", 0.0, float(v0)),
                     (f"R{i}", "volume_l", 48.0, float(v1))]
    return pd.DataFrame(rows, columns=["run_id", "variable", "time_h", "value"])


def test_guard_refuses_prediction_implausible_vs_observed_envelope():
    # observed peaks ~100-150; a model predicting 1809 (unit-corrupt) is rejected.
    obs = _obs_with_objective([100, 120, 140, 150])
    bad = EquationResult(family="mechanistic", cleared=True, cv_r2=0.9,
                         best_knobs={"substrate_g_l.initial": 99.0}, predicted_peak=1809.8)
    out = _apply_sanity_guards(bad, obs, "product_g_l")
    assert out.cleared is False
    assert out.best_knobs == {}
    assert "implausible" in out.rationale.lower()


def test_guard_keeps_plausible_prediction():
    obs = _obs_with_objective([100, 120, 140, 150])
    ok = EquationResult(family="surrogate", cleared=True, cv_r2=0.8,
                        best_knobs={"k": 1.0}, predicted_peak=165.0)  # within 2x of 150
    out = _apply_sanity_guards(ok, obs, "product_g_l")
    assert out.cleared is True
    assert out.predicted_peak == 165.0


def test_guard_marks_boundary_optimum_as_insufficient_data():
    obs = _obs_with_objective([100, 120, 140, 150])
    res = EquationResult(family="surrogate", cleared=True, cv_r2=0.8,
                         best_knobs={"k": 1.0}, predicted_peak=150.0,
                         on_boundary={"k": "upper"})
    out = _apply_sanity_guards(res, obs, "product_g_l")
    assert out.boundary_limited is True
    assert "insufficient data" in out.rationale.lower()


def test_guard_flags_fedbatch_operating_mode_for_mechanistic():
    # volume rises 61 -> 100 L within each run => fed-batch; batch ODE flagged.
    obs = _obs_with_objective([100, 120, 140, 150], volume=(61.0, 100.0))
    res = EquationResult(family="mechanistic", cleared=True, cv_r2=0.8,
                         best_knobs={"substrate_g_l.initial": 99.0}, predicted_peak=150.0)
    out = _apply_sanity_guards(res, obs, "product_g_l")
    assert "fed-batch" in out.rationale.lower()


def test_constant_factor_is_not_a_lever():
    # pH held constant across runs -> not a lever; only the varying source is.
    dossier, obs = _categorical_dossier_and_obs()
    for rid in dossier["run_conditions"]:
        dossier["run_conditions"][rid]["pH"] = {"numeric": 6.0, "value": "6.0"}
    levers = discover_levers(dossier, obs)
    names = {lev.name for lev in levers}
    assert "pH" not in names
    assert "nitrogen_source" in names
