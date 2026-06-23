"""A2 — deterministic gates, driven by the natural planted faults from the praaj
debugging: the nutrient-vs-target confound, a recommendation on a clamped
objective, and a materiality demotion. Each fault must trip its named gate.
"""
from __future__ import annotations

import pandas as pd
import pytest

from fermdocs.analysis.computation_cache import reset_default_cache
from fermdocs.analysis.gates import (
    GatedClaim,
    claim_blocked,
    confound_gate,
    direction_gate,
    laundering_gate,
    materiality_gate,
    objective_gate,
    run_gates,
)


@pytest.fixture(autouse=True)
def _isolate():
    reset_default_cache()
    yield
    reset_default_cache()


def _aliased_bundle():
    # nitrogen aliased with target: DBY only at 100, Leiber_H only at 160. Titer=target.
    rc, rows, strata = {}, [], {}
    for i in range(5):
        rid = f"L{i}"; rc[rid] = {"nitrogen": {"value": "DBY"}, "target": {"value": 100}}
        strata[rid] = 100; rows.append((rid, "product_g_l", 40.0, 100.0))
    for i in range(5):
        rid = f"H{i}"; rc[rid] = {"nitrogen": {"value": "Leiber_H"}, "target": {"value": 160}}
        strata[rid] = 160; rows.append((rid, "product_g_l", 40.0, 160.0))
    return {"run_conditions": rc}, pd.DataFrame(
        rows, columns=["run_id", "variable", "time_h", "value"]), strata


def test_confound_gate_fails_the_nutrient_claim():
    dossier, obs, _ = _aliased_bundle()
    claim = GatedClaim(
        assertion="Yeast Extract Leiber H maximizes titer",
        claim_type="causal", lever="nitrogen", objective_channel="product_g_l",
        conditioning=["target"])
    v = confound_gate(claim, dossier, obs)
    assert v is not None and v.passed is False
    assert "not separable" in v.reason


def test_confound_gate_na_without_covariates():
    # No covariates to hold constant -> N/A (not a failure): can't be confounded by
    # a covariate that doesn't exist on this bundle (diagnostic path / pre-B1').
    dossier, obs, _ = _aliased_bundle()
    claim = GatedClaim("nitrogen drives titer", "causal", lever="nitrogen",
                       objective_channel="product_g_l", conditioning=[])
    assert confound_gate(claim, dossier, obs) is None


def test_confound_gate_na_for_observational():
    dossier, obs, _ = _aliased_bundle()
    claim = GatedClaim("nitrogen associates with titer", "observational",
                       lever="nitrogen", objective_channel="product_g_l")
    assert confound_gate(claim, dossier, obs) is None  # not required to condition


def test_objective_gate_fails_recommendation_on_clamped_titer():
    dossier, obs, strata = _aliased_bundle()
    claim = GatedClaim("raise nitrogen to lift peak titer", "recommendation",
                       lever="nitrogen", objective_channel="product_g_l",
                       conditioning=["target"])
    v = objective_gate(claim, obs, strata)
    assert v is not None and v.passed is False
    assert "clamped" in v.reason


def test_materiality_gate_downgrades_smaller_lever():
    # two levers vary within one stratum; 'minor' has a small effect, 'major' large.
    rc, rows = {}, []
    for i in range(8):
        rid = f"R{i}"
        rc[rid] = {"minor": {"value": i % 2}, "major": {"value": i % 4}}
        rows.append((rid, "product_g_l", 40.0, 100.0 + 20.0 * (i % 4) + 0.3 * (i % 2)))
    dossier = {"run_conditions": rc}
    obs = pd.DataFrame(rows, columns=["run_id", "variable", "time_h", "value"])
    claim = GatedClaim("minor is the primary lever", "causal", lever="minor",
                       objective_channel="product_g_l", conditioning=["major"])
    v = materiality_gate(claim, dossier, obs)
    assert v is not None and v.passed is False
    assert v.severity == "downgrade" and "major" in v.reason


def test_run_gates_blocks_the_confounded_recommendation():
    dossier, obs, strata = _aliased_bundle()
    claim = GatedClaim("use Leiber H to maximize titer", "recommendation",
                       lever="nitrogen", objective_channel="product_g_l",
                       conditioning=["target"])
    verdicts = run_gates(claim, dossier, obs, strata)
    gates_fired = {v.gate for v in verdicts}
    assert "confound" in gates_fired and "objective" in gates_fired
    assert claim_blocked(verdicts) is True  # at least one hard-fail


def test_direction_gate_catches_manufactured_convergence():
    # nutrient INCREASES titer but DECREASES the cited growth metric per the data:
    # the "growth corroborates" argument is manufactured.
    rc, rows = {}, []
    for i in range(8):
        nutrient = float(i % 4)  # varies
        rc[f"R{i}"] = {"nutrient": {"value": nutrient}}
        rows.append((f"R{i}", "product_g_l", 40.0, 100.0 + 5.0 * nutrient))   # titer UP with nutrient
        rows.append((f"R{i}", "od600_au", 40.0, 10.0 - 1.5 * nutrient))       # growth DOWN with nutrient
    dossier = {"run_conditions": rc}
    obs = pd.DataFrame(rows, columns=["run_id", "variable", "time_h", "value"])
    claim = GatedClaim("nutrient raises titer via growth", "causal", lever="nutrient",
                       objective_channel="product_g_l", corroborating_metric="od600_au")
    v = direction_gate(claim, dossier, obs)
    assert v is not None and v.passed is False
    assert "manufactured" in v.reason


def test_direction_gate_passes_real_corroboration():
    rc, rows = {}, []
    for i in range(8):
        nutrient = float(i % 4)
        rc[f"R{i}"] = {"nutrient": {"value": nutrient}}
        rows.append((f"R{i}", "product_g_l", 40.0, 100.0 + 5.0 * nutrient))   # both UP
        rows.append((f"R{i}", "od600_au", 40.0, 5.0 + 2.0 * nutrient))
    dossier = {"run_conditions": rc}
    obs = pd.DataFrame(rows, columns=["run_id", "variable", "time_h", "value"])
    claim = GatedClaim("nutrient raises titer via growth", "causal", lever="nutrient",
                       objective_channel="product_g_l", corroborating_metric="od600_au")
    v = direction_gate(claim, dossier, obs)
    assert v is not None and v.passed is True


def test_laundering_gate_flags_ruling_out_support():
    claim = GatedClaim(
        "DO crash explains the plateau", "causal", lever="nitrogen",
        objective_channel="product_g_l",
        support_text="DO=0 mid-run; this is not an oxygen limitation, ruling out aeration "
                     "as the cause rather than a biological effect.")
    v = laundering_gate(claim)
    assert v is not None and v.passed is False and v.severity == "downgrade"


def test_laundering_gate_quiet_on_positive_support():
    claim = GatedClaim(
        "nitrogen raises titer", "causal", lever="nitrogen", objective_channel="product_g_l",
        support_text="Across runs higher nitrogen loading associates with higher peak titer.")
    assert laundering_gate(claim) is None


def test_clean_lever_passes_all_gates():
    # feed varies within both strata, genuinely moves a FREE objective, largest effect.
    rc, rows, strata = {}, [], {}
    for i in range(6):
        for tgt, pre in [(100, "L"), (160, "H")]:
            rid = f"{pre}{i}"; feed = i % 2
            rc[rid] = {"feed": {"value": feed}, "target": {"value": tgt}}
            strata[rid] = tgt
            # productivity-like free objective: depends on feed, not pinned to target
            rows.append((rid, "od600_au", 40.0, 5.0 + 4.0 * feed + 0.1 * i))
    dossier = {"run_conditions": rc}
    obs = pd.DataFrame(rows, columns=["run_id", "variable", "time_h", "value"])
    claim = GatedClaim("feed drives od600", "causal", lever="feed",
                       objective_channel="od600_au", conditioning=["target"])
    verdicts = run_gates(claim, dossier, obs, strata)
    assert not claim_blocked(verdicts)
    assert any(v.gate == "confound" and v.passed for v in verdicts)
