"""A3 — the gate bridge: a gate-failed hypothesis is excluded from the verdict
with the failing gate named, regardless of how confidently it was phrased.
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from fermdocs.analysis.computation_cache import reset_default_cache
from fermdocs_hypothesis.gate_bridge import apply_gates, gated_claim_from_hypothesis


@pytest.fixture(autouse=True)
def _isolate():
    reset_default_cache()
    yield
    reset_default_cache()


def _bundle():
    # nitrogen aliased with target (confounded); feed varies within target (real, free obj).
    rc, rows, strata = {}, [], {}
    for i in range(6):
        for tgt, pre in [(100, "L"), (160, "H")]:
            rid = f"{pre}{i}"
            rc[rid] = {"nitrogen": {"value": "DBY" if tgt == 100 else "Leiber_H"},
                       "feed": {"value": i % 2}, "target": {"value": tgt}}
            strata[rid] = tgt
            # titer is clamped to the campaign target (nitrogen, aliased with target,
            # looks big against it). A FREE channel (od600) moves with feed within a
            # campaign and carries no target component -> feed is a real lever there.
            rows.append((rid, "product_g_l", 40.0, float(tgt)))
            rows.append((rid, "od600_au", 40.0, 5.0 + 8.0 * (i % 2)))
    return {"run_conditions": rc}, pd.DataFrame(
        rows, columns=["run_id", "variable", "time_h", "value"]), strata


def _hyp(hyp_id, lever, claim_type, summary="", rec=None):
    return SimpleNamespace(
        hyp_id=hyp_id, summary=summary or f"{lever} drives titer",
        cited_association_ids=[f"WRA-{lever}"], affected_variables=["product_g_l"],
        actionable_recommendation=rec, claim_type=claim_type, gate_failures=[])


def test_confounded_causal_hypothesis_is_blocked():
    dossier, obs, strata = _bundle()
    confounded = _hyp("H-1", "nitrogen", "recommendation",
                      summary="Use Leiber H to maximize peak titer",
                      rec="switch to Leiber H")
    allowed, blocked = apply_gates(
        [confounded], dossier, obs,
        objective_channel="product_g_l", conditioning=["target"], strata=strata)
    assert allowed == []
    assert len(blocked) == 1
    hyp, verdicts = blocked[0]
    assert "confound" in hyp.gate_failures        # named on the claim
    assert any(v.gate == "confound" and not v.passed for v in verdicts)


def test_clean_causal_hypothesis_passes():
    dossier, obs, strata = _bundle()
    # feed moves the FREE objective (od600) within each campaign -> separable,
    # real effect, largest lever -> passes every gate.
    clean = _hyp("H-2", "feed", "causal", summary="feed window affects od600")
    allowed, blocked = apply_gates(
        [clean], dossier, obs,
        objective_channel="od600_au", conditioning=["target"], strata=strata)
    assert len(allowed) == 1
    assert allowed[0].gate_failures == []


def test_observational_claim_not_gated_for_conditioning():
    dossier, obs, strata = _bundle()
    obs_claim = _hyp("H-3", "nitrogen", "observational",
                     summary="nitrogen associates with titer")
    allowed, blocked = apply_gates(
        [obs_claim], dossier, obs,
        objective_channel="product_g_l", conditioning=["target"], strata=strata)
    assert len(allowed) == 1  # observational claims aren't required to condition


def test_claim_projection_resolves_lever_and_support():
    h = _hyp("H-4", "main_fermentation_nitrogen_source", "causal",
             summary="nitrogen drives titer", rec="use YE")
    claim = gated_claim_from_hypothesis(h, objective_channel="product_g_l",
                                        conditioning=["target"])
    assert claim.lever == "main_fermentation_nitrogen_source"   # WRA- stripped
    assert claim.claim_type == "causal"
    assert "use YE" in claim.support_text
