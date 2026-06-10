"""Agent tests: the rubric — not the LLM — decides the verdict."""

import json

from fermdocs_recommend.agent import RecommendationAgent


def _report(r2, rmse, n=10, params=None):
    return {
        "fit_quality": {"X": {"r2": r2, "rmse": rmse, "n": n}},
        "fitted_parameters": {
            k: {"value": v, "plausible": True, "known": True, "range": [0, 9]}
            for k, v in (params or {}).items()
        },
        "loss_reduction_frac": 0.9,
    }


class _EmitClient:
    """Emits a fixed payload via payload_json (the real wire shape)."""

    def __init__(self, payload):
        self._payload = payload

    def call(self, system, messages):
        return {"action": "emit", "payload_json": json.dumps(self._payload)}


class _DummyReader:
    def __init__(self, d):
        self.dir = d


def test_rubric_overrides_llm_self_report(tmp_path):
    # LLM *claims* mechanistic, but supplies poor candidate reports -> rubric refuses.
    payload = {
        "recommended_model": "mechanistic",  # should be ignored
        "confident": True,
        "selection_rationale": "I like mechanistic",
        "candidates": [
            {"model_type": m, "attempted": True, "report": _report(0.3, 5.0)}
            for m in ("mechanistic", "surrogate", "hybrid")
        ],
        "interventions": [{"description": "raise DO"}],
        "grounding_hyp_ids": ["H-0001"],
    }
    agent = RecommendationAgent(client=_EmitClient(payload))
    out = agent.recommend(_DummyReader(tmp_path))
    assert out.recommended_model == "none"  # rubric authority
    assert out.confident is False
    assert out.interventions == []  # validator drops on refusal
    assert out.grounding_hyp_ids == ["H-0001"]


def test_good_mechanistic_is_recommended(tmp_path):
    payload = {
        "candidates": [
            {"model_type": "mechanistic", "attempted": True,
             "report": _report(0.97, 0.4, params={"mu_max": 0.4})},
            {"model_type": "surrogate", "attempted": True, "report": _report(0.9, 1.0)},
            {"model_type": "hybrid", "attempted": False},
        ],
        "interventions": [{"description": "increase feed late", "delta": 1.2}],
        "grounding_hyp_ids": ["H-0002"],
    }
    agent = RecommendationAgent(client=_EmitClient(payload))
    out = agent.recommend(_DummyReader(tmp_path))
    assert out.recommended_model == "mechanistic"
    assert out.confident is True
    # 4 families now: the 3 brewtwin + the complementary mechanistic_discovered
    # (here "not attempted" — _DummyReader has no observations.csv to discover on).
    assert len(out.candidates) == 4
    disc = [c for c in out.candidates if c.model_type == "mechanistic_discovered"]
    assert len(disc) == 1 and disc[0].attempted is False
    assert len(out.interventions) == 1


def test_no_client_is_stage_error(tmp_path):
    out = RecommendationAgent(client=None).recommend(_DummyReader(tmp_path))
    assert out.recommended_model == "none"
    assert out.meta.error == "no_llm_client"


def test_budget_exhausted_is_refusal_not_crash(tmp_path):
    class _NeverEmits:
        def call(self, system, messages):
            return {"action": "tool_call", "tool": "get_data_feed", "args": {}}

    agent = RecommendationAgent(client=_NeverEmits(), max_steps=2)
    out = agent.recommend(_DummyReader(tmp_path))
    assert out.recommended_model == "none"
    assert out.refusal_reason in ("compute_budget_exhausted", "stage_error")
