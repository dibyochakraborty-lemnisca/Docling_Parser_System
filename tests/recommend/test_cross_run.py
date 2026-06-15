"""Cross-run comparative engine + its promotion in the recommendation output."""

from __future__ import annotations

import json

import pandas as pd

from fermdocs_recommend import cross_run
from fermdocs_recommend.agent import RecommendationAgent


def _obs(rows):
    return pd.DataFrame(rows, columns=["run_id", "variable", "time_h", "value"])


def _runs_with_titer(titers: dict[str, float]) -> pd.DataFrame:
    rows = []
    for rid, peak in titers.items():
        rows.append([rid, "product_g_l", 0.0, 0.0])
        rows.append([rid, "product_g_l", 48.0, peak])
    return _obs(rows)


def _dossier(conditions: dict) -> dict:
    return {"run_conditions": conditions}


def test_numeric_knob_association_clears_and_ranks():
    # initial_sugar correlates with titer; base_type does not move it.
    titers = {"R1": 80.0, "R2": 90.0, "R3": 100.0, "R4": 110.0}
    conds = {
        "R1": {"initial_sugar": {"value": 100.0}, "base_type": {"value": "CaOH2"}},
        "R2": {"initial_sugar": {"value": 120.0}, "base_type": {"value": "NaOH"}},
        "R3": {"initial_sugar": {"value": 140.0}, "base_type": {"value": "CaOH2"}},
        "R4": {"initial_sugar": {"value": 160.0}, "base_type": {"value": "NaOH"}},
    }
    res = cross_run.analyze(_dossier(conds), _runs_with_titer(titers))
    assert res["cleared"] is True
    top = res["interventions"][0]
    assert top["knob"] == "initial_sugar"
    assert top["description"].startswith("Increase initial_sugar")
    assert top["delta"] > 0
    assert top["in_coverage"] is True
    assert "validate experimentally" in top["caveat"]


def test_no_variation_or_too_few_runs_does_not_clear():
    # 4 runs but the knob is constant -> no association.
    titers = {"R1": 80.0, "R2": 90.0, "R3": 100.0, "R4": 110.0}
    conds = {r: {"k": {"value": 5.0}} for r in titers}
    res = cross_run.analyze(_dossier(conds), _runs_with_titer(titers))
    assert res["cleared"] is False
    assert res["interventions"] == []

    # Too few runs with the objective.
    few = cross_run.analyze(
        _dossier({"R1": {"k": {"value": 1.0}}}), _runs_with_titer({"R1": 80.0})
    )
    assert few["cleared"] is False


def test_categorical_knob_best_category():
    titers = {"R1": 70.0, "R2": 72.0, "R3": 100.0, "R4": 104.0}
    conds = {
        "R1": {"base_type": {"value": "NaOH"}},
        "R2": {"base_type": {"value": "NaOH"}},
        "R3": {"base_type": {"value": "CaOH2"}},
        "R4": {"base_type": {"value": "CaOH2"}},
    }
    res = cross_run.analyze(_dossier(conds), _runs_with_titer(titers))
    assert res["cleared"] is True
    top = res["interventions"][0]
    assert top["knob"] == "base_type"
    assert top["predicted_value"] > top["baseline_value"]


def test_no_conditions_returns_none():
    assert cross_run.analyze({}, _runs_with_titer({"R1": 80.0})) is None
    assert cross_run.analyze(None, _runs_with_titer({"R1": 80.0})) is None


# --- agent promotion -------------------------------------------------------


class _RefusingClient:
    """Emits three poor dynamic candidates -> rubric refuses."""

    def call(self, system, messages):
        payload = {
            "candidates": [
                {"model_type": m, "attempted": True,
                 "report": {"fit_quality": {"product_g_l": {"r2": 0.1, "rmse": 9.0, "n": 6}},
                            "fitted_parameters": {}, "loss_reduction_frac": 0.1}}
                for m in ("mechanistic", "surrogate", "hybrid")
            ],
            "interventions": [],
        }
        return {"action": "emit", "payload_json": json.dumps(payload)}


class _Reader:
    def __init__(self, d):
        self.dir = d


def test_cross_run_wins_when_bakeoff_refuses(tmp_path, monkeypatch):
    # Bake-off refuses; cross-run clears -> recommendation becomes
    # cross_run_comparative with its interventions (not an empty refusal).
    agent = RecommendationAgent(client=_RefusingClient())

    cr_result = {
        "cleared": True,
        "interventions": [{
            "knob": "initial_sugar", "description": "Increase initial_sugar toward 160",
            "objective_metric": "product_g_l.peak", "baseline_value": 95.0,
            "predicted_value": 110.0, "delta": 15.0, "in_coverage": True,
            "caveat": "observational; validate experimentally", "rationale": "assoc",
        }],
        "n_runs": 4, "objective": "product_g_l",
        "summary": "cross-run comparative over 4 runs",
    }
    monkeypatch.setattr(agent, "_cross_run_analysis", lambda bundle: cr_result)
    monkeypatch.setattr(agent, "_discover_complementary", lambda bundle: None)

    out = agent.recommend(_Reader(tmp_path))
    assert out.recommended_model == "cross_run_comparative"
    assert out.confident is True
    assert out.refusal_reason is None
    assert len(out.interventions) == 1
    assert out.interventions[0].knob == "initial_sugar"


def test_lever_effects_flags_confounded_levers():
    # 8 runs. reactor and impeller partition the runs IDENTICALLY (aliased);
    # lot is unique per run (run label); nitrogen has a distinct partition.
    titers = {f"R{i}": 100.0 + 5 * i for i in range(8)}
    conds = {}
    for i in range(8):
        grp = "A" if i < 4 else "B"          # reactor / impeller share this split
        conds[f"R{i}"] = {
            "reactor": {"value": grp},
            "impeller": {"value": f"imp_{grp}"},           # tracks reactor exactly
            "lot": {"value": f"LOT-{i}"},                  # one value per run
            "nitrogen": {"value": "CSL" if i % 2 == 0 else "YE"},  # different split
        }
    eff = cross_run.lever_effects(_dossier(conds), _runs_with_titer(titers))
    assert eff["reactor"]["confounded"] is True            # aliased with impeller
    assert eff["impeller"]["confounded"] is True
    assert eff["lot"]["confounded"] is True                # run label
    assert "run index" in eff["lot"]["confounded_with"]
    assert eff["nitrogen"]["confounded"] is False          # independent partition
