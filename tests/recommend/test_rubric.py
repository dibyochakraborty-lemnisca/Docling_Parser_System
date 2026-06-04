"""Rubric gate + selection tests (pure, no brewtwin/LLM)."""

from fermdocs_recommend import rubric


def _report(r2, rmse, n=10, params=None, loss=(100.0, 1.0)):
    fq = {"X": {"r2": r2, "rmse": rmse, "n": n}}
    plaus = {
        k: {"value": v, "plausible": lo <= v <= hi, "known": True, "range": [lo, hi]}
        for k, (v, lo, hi) in (params or {}).items()
    }
    return {
        "fit_quality": fq,
        "fitted_parameters": plaus,
        "loss_reduction_frac": (loss[0] - loss[1]) / loss[0],
    }


def _c(mt, **kw):
    base = {"model_type": mt, "attempted": True, "disqualified": False, "report": None}
    base.update(kw)
    return base


def test_mechanistic_wins_when_plausible_and_good():
    cands = [
        _c("mechanistic", report=_report(0.97, 0.5, params={"mu_max": (0.4, 0.05, 2.0)})),
        _c("surrogate", report=_report(0.96, 0.9)),
        _c("hybrid", attempted=False),
    ]
    v = rubric.select(cands)
    assert v["recommended_model"] == "mechanistic"
    assert v["confident"] is True


def test_all_poor_refuses():
    cands = [_c(m, report=_report(0.4, 5.0)) for m in ("mechanistic", "surrogate", "hybrid")]
    v = rubric.select(cands)
    assert v["recommended_model"] == "none"
    assert v["refusal_reason"] == rubric.REFUSAL_POOR_FIT


def test_surrogate_wins_when_mechanistic_fails():
    cands = [
        _c("mechanistic", report=_report(0.7, 2.0, params={"mu_max": (0.4, 0.05, 2.0)})),
        _c("surrogate", report=_report(0.98, 0.3)),
        _c("hybrid", disqualified=True, disqualification_reason="EquinoxRuntimeError"),
    ]
    v = rubric.select(cands)
    assert v["recommended_model"] == "surrogate"
    assert v["confident"] is True


def test_implausible_params_block_mechanism():
    cands = [
        _c("mechanistic", report=_report(0.99, 0.2, params={"mu_max": (9.0, 0.05, 2.0)})),
        _c("surrogate", report=_report(0.5, 3.0)),
        _c("hybrid", attempted=False),
    ]
    v = rubric.select(cands)
    assert v["recommended_model"] == "none"
    assert v["refusal_reason"] == rubric.REFUSAL_IMPLAUSIBLE


def test_stalled_optimizer_refuses_as_insufficient_data():
    cands = [
        _c(m, report=_report(0.99, 0.1, loss=(100.0, 99.9)))
        for m in ("mechanistic", "surrogate", "hybrid")
    ]
    v = rubric.select(cands)
    assert v["recommended_model"] == "none"
    assert v["refusal_reason"] == rubric.REFUSAL_NO_DATA


def test_too_few_heldout_points_not_eligible():
    cands = [
        _c("mechanistic", report=_report(0.99, 0.1, n=2, params={"mu_max": (0.4, 0.05, 2.0)})),
        _c("surrogate", report=_report(0.99, 0.1, n=2)),
        _c("hybrid", attempted=False),
    ]
    v = rubric.select(cands)
    assert v["recommended_model"] == "none"
