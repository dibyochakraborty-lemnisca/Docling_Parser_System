"""B2 — ranking_effects redirect (conditioned effect on the free objective) and
B3 — residual variance reporting.
"""
from __future__ import annotations

import pandas as pd
import pytest

from fermdocs.analysis import cross_run
from fermdocs.analysis.computation_cache import reset_default_cache
from fermdocs.analysis.objective import Objective
from fermdocs.analysis.residual import residual_report


@pytest.fixture(autouse=True)
def _isolate():
    reset_default_cache()
    yield
    reset_default_cache()


def _praaj_like():
    """nutrient aliased with target (confounded); feed varies within target (real)."""
    rc, rows, strata = {}, [], {}
    for i in range(6):
        for tgt, pre in [(100, "L"), (160, "H")]:
            rid = f"{pre}{i}"
            feed = i % 2
            nutrient = "DBY" if tgt == 100 else "Leiber_H"  # aliased with target
            rc[rid] = {"feed": {"value": feed}, "nutrient": {"value": nutrient},
                       "target": {"value": tgt}}
            strata[rid] = tgt
            # titer pinned to target; productivity (free) moves with feed
            rows.append((rid, "product_g_l", 30.0 + 10 * feed, float(tgt)))
    obs = pd.DataFrame(rows, columns=["run_id", "variable", "time_h", "value"])
    return {"run_conditions": rc}, obs, strata


def test_ranking_unconditioned_falls_back():
    dossier, obs, _ = _praaj_like()
    scores = cross_run.ranking_effects(dossier, obs, objective="product_g_l")
    # without conditioning, returns unconditioned norm_effects for the varying levers
    assert "nutrient" in scores and "target" in scores


def test_ranking_conditioned_sinks_confounded_lever():
    dossier, obs, _ = _praaj_like()
    # condition on target: nutrient is aliased with target -> sinks to 0;
    # feed varies within target -> keeps a real score.
    scores = cross_run.ranking_effects(
        dossier, obs, objective="product_g_l", conditioning=["target"])
    assert scores.get("nutrient", 1.0) == 0.0      # confounded -> sunk
    assert "target" not in scores                  # covariate excluded from ranking


def test_ranking_targets_free_objective_via_outcomes():
    # rank against a FREE rate objective (productivity) instead of the clamped channel.
    dossier, obs, strata = _praaj_like()
    obj = Objective(name="product_g_l_per_h", kind="rate", base_channel="product_g_l")
    outs = obj.outcome_per_run(obs)
    assert outs  # rate computed
    scores = cross_run.ranking_effects(
        dossier, obs, objective=obj.name, outcomes=outs, conditioning=["target"])
    # feed moves the free objective within target -> non-zero; nutrient confounded -> 0
    assert scores.get("feed", 0.0) > 0.0
    assert scores.get("nutrient", 1.0) == 0.0


def test_residual_flags_missing_lever_when_outcome_is_noise():
    # outcome unrelated to any lever -> levers explain ~nothing -> flag missing lever.
    rc, rows = {}, []
    noise = [3.1, 9.4, 1.2, 7.8, 5.5, 2.2, 8.9, 4.4]
    for i in range(8):
        rc[f"R{i}"] = {"feed": {"value": i % 2}, "nutrient": {"value": ["A", "B"][i % 2]}}
        rows.append((f"R{i}", "product_g_l", 40.0, noise[i]))
    dossier = {"run_conditions": rc}
    obs = pd.DataFrame(rows, columns=["run_id", "variable", "time_h", "value"])
    rep = residual_report(dossier, obs, objective="product_g_l")
    assert rep is not None
    assert rep.likely_missing_lever is True
    assert rep.unexplained > 0.4
    assert "unexplained" in rep.note


def test_residual_low_when_lever_explains_outcome():
    rc, rows = {}, []
    for i in range(8):
        feed = i % 2
        rc[f"R{i}"] = {"feed": {"value": feed}}
        rows.append((f"R{i}", "product_g_l", 40.0, 100.0 + 30.0 * feed))  # feed explains it
    dossier = {"run_conditions": rc}
    obs = pd.DataFrame(rows, columns=["run_id", "variable", "time_h", "value"])
    rep = residual_report(dossier, obs, objective="product_g_l")
    assert rep is not None
    assert rep.explained_r2 > 0.9
    assert rep.likely_missing_lever is False
