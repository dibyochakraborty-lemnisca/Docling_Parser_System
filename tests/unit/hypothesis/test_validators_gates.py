"""A3 — deterministic gates wired into validate_hypothesis_output.

A confounded causal/recommendation hypothesis, passed the bundle data, is
downgraded to schema_only with the failing gate named — it can no longer be
honored as a trusted verdict. Without bundle data the pass is skipped (graceful).
"""
from __future__ import annotations

from datetime import datetime
from uuid import UUID

import pandas as pd

from fermdocs_characterize.schema import CharacterizationOutput, Meta
from fermdocs_diagnose.schema import ConfidenceBasis
from fermdocs_hypothesis.schema import (
    BudgetSnapshot,
    FinalHypothesis,
    HypothesisMeta,
    HypothesisOutput,
)
from fermdocs_hypothesis.validators import validate_hypothesis_output

CHAR_ID = UUID(int=77)


def _upstream() -> CharacterizationOutput:
    return CharacterizationOutput(
        meta=Meta(
            schema_version="2.0", characterization_version="v1.0.0",
            characterization_id=CHAR_ID, generation_timestamp=datetime(2026, 1, 1),
            source_dossier_ids=["EXP-X"]),
        findings=[], trajectories=[])


def _meta() -> HypothesisMeta:
    return HypothesisMeta(
        schema_version="1.0", hypothesis_version="v0.1.0",
        hypothesis_id=UUID(int=1), supersedes_diagnosis_id=UUID(int=2),
        generation_timestamp=datetime(2026, 5, 3), model="gemini-3-pro",
        provider="gemini", budget_used=BudgetSnapshot())


def _final(claim_type: str, lever: str) -> FinalHypothesis:
    return FinalHypothesis(
        hyp_id="H-0001", summary=f"use {lever} to maximize titer",
        facet_ids=["FCT-0001"], cited_finding_ids=[], cited_narrative_ids=[],
        cited_trajectories=[], cited_association_ids=[f"WRA-{lever}"],
        affected_variables=["product_g_l"], confidence=0.8,
        confidence_basis=ConfidenceBasis.CROSS_RUN,
        critic_flag="green", judge_ruled_criticism_valid=False,
        actionable_recommendation=f"switch to {lever}", claim_type=claim_type)


def _bundle():
    rc, rows, strata = {}, [], {}
    for i in range(5):
        rc[f"L{i}"] = {"nitrogen": {"value": "DBY"}, "target": {"value": 100}}
        strata[f"L{i}"] = 100
        rows.append((f"L{i}", "product_g_l", 40.0, 100.0))
    for i in range(5):
        rc[f"H{i}"] = {"nitrogen": {"value": "Leiber_H"}, "target": {"value": 160}}
        strata[f"H{i}"] = 160
        rows.append((f"H{i}", "product_g_l", 40.0, 160.0))
    return {"run_conditions": rc}, pd.DataFrame(
        rows, columns=["run_id", "variable", "time_h", "value"]), strata


def test_confounded_recommendation_downgraded_by_gates():
    dossier, obs, strata = _bundle()
    out = HypothesisOutput(meta=_meta(),
                           final_hypotheses=[_final("recommendation", "nitrogen")])
    cleaned = validate_hypothesis_output(
        out, upstream=_upstream(),
        dossier=dossier, obs_df=obs, objective_channel="product_g_l",
        conditioning=["target"], strata=strata)
    h = cleaned.final_hypotheses[0]
    assert h.gate_failures                                  # named
    assert "confound" in h.gate_failures
    assert h.confidence_basis == ConfidenceBasis.SCHEMA_ONLY  # demoted
    assert h.provenance_downgraded is True
    assert h.confidence <= 0.1
    assert "gates failed" in h.summary


def test_no_bundle_data_skips_gates_graceful():
    out = HypothesisOutput(meta=_meta(),
                           final_hypotheses=[_final("recommendation", "nitrogen")])
    cleaned = validate_hypothesis_output(out, upstream=_upstream())  # no dossier/obs
    h = cleaned.final_hypotheses[0]
    assert h.gate_failures == []                            # gates not run
    assert h.confidence_basis == ConfidenceBasis.CROSS_RUN


def test_observational_claim_not_downgraded():
    dossier, obs, strata = _bundle()
    out = HypothesisOutput(meta=_meta(),
                           final_hypotheses=[_final("observational", "nitrogen")])
    cleaned = validate_hypothesis_output(
        out, upstream=_upstream(),
        dossier=dossier, obs_df=obs, objective_channel="product_g_l",
        conditioning=["target"], strata=strata)
    h = cleaned.final_hypotheses[0]
    assert h.gate_failures == []                            # observational: not gated
    assert h.confidence_basis == ConfidenceBasis.CROSS_RUN
