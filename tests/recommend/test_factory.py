"""Tool-bundle tests (no sandbox execution — that needs the JAX stack)."""

import json

from fermdocs_recommend.tools_bundle.factory import make_recommend_tools


class _Reader:
    def __init__(self, d):
        self.dir = d


def _write_obs(bundle_dir):
    cdir = bundle_dir / "characterization"
    cdir.mkdir(parents=True, exist_ok=True)
    rows = ["run_id,variable,time_h,value,imputed,unit"]
    for run in ("RUN-0001", "RUN-0002"):
        for t in (0.0, 1.0, 2.0):
            rows.append(f"{run},biomass_g_l,{t},{1.0 + t},0,g/L")
            rows.append(f"{run},feed_rate_l_per_h,{t},0.5,0,L/h")
    (cdir / "observations.csv").write_text("\n".join(rows))


def test_get_data_feed_classifies(tmp_path):
    _write_obs(tmp_path)
    tools = make_recommend_tools(_Reader(tmp_path))
    fd = tools.get_data_feed()
    assert set(fd["run_ids"]) == {"RUN-0001", "RUN-0002"}
    assert fd["feed_var"] == "feed_rate_l_per_h"
    assert "biomass_g_l" in fd["states"]
    assert fd["leave_one_run_out"]["validate"] == "RUN-0002"


def test_get_hypotheses_reads_real_artifact(tmp_path):
    hyp = tmp_path / "hypothesis_output.json"
    hyp.write_text(json.dumps({
        "final_hypotheses": [
            {"hyp_id": "H-0001", "summary": "DO crash",
             "affected_variables": ["dissolved_o2_mg_l"],
             "actionable_recommendation": "raise DO", "confidence": 0.8}
        ]
    }))
    tools = make_recommend_tools(_Reader(tmp_path), hypothesis_output_path=hyp)
    out = tools.get_hypotheses()
    assert out["n"] == 1
    assert out["hypotheses"][0]["hyp_id"] == "H-0001"


def test_get_hypotheses_missing(tmp_path):
    tools = make_recommend_tools(_Reader(tmp_path), hypothesis_output_path=None)
    out = tools.get_hypotheses()
    assert out["hypotheses"] == []


def test_get_skill_known_and_unknown(tmp_path):
    tools = make_recommend_tools(_Reader(tmp_path))
    ok = tools.get_skill("fit-mechanistic-model")
    assert "content" in ok and "fit-mechanistic-model" in ok["skill"]
    bad = tools.get_skill("nope")
    assert "error" in bad and "available" in bad


def test_submit_terminates(tmp_path):
    tools = make_recommend_tools(_Reader(tmp_path))
    r = tools.submit_recommendation(payload={"candidates": []})
    assert r["ok"] is True
    assert "error" in tools.submit_recommendation(payload={})
