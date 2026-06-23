"""F1 — clampedness detection + free-variable objective auto-preference.

praaj shape: titer is pinned to a campaign target (clamped within stratum, varies
across) while productivity (peak/time) moves freely. The resolver must auto-select
the free variable on praaj-like data and the channel on free-titer data, with NO
per-dataset constant and the user override always winning.
"""
from __future__ import annotations

import pandas as pd
import pytest

from fermdocs.analysis.clampedness import detect_clamp
from fermdocs.analysis.objective import Objective, resolve_objective_free


def _praaj_like():
    """Two target campaigns (100, 160). Titer pinned to target (clamped). Time-to-
    peak varies within campaign -> productivity is free."""
    rows, strata = [], {}
    for i in range(6):
        # 100-campaign: titer ~100, time-to-peak alternates 25/40h
        rid = f"L{i}"
        strata[rid] = 100
        tpk = 25.0 if i % 2 == 0 else 40.0
        rows += [(rid, "product_g_l", 0.0, 0.0), (rid, "product_g_l", tpk, 100.0 + (i % 2) * 0.5)]
    for i in range(6):
        rid = f"H{i}"
        strata[rid] = 160
        tpk = 30.0 if i % 2 == 0 else 50.0
        rows += [(rid, "product_g_l", 0.0, 0.0), (rid, "product_g_l", tpk, 150.0 + (i % 2) * 0.5)]
    obs = pd.DataFrame(rows, columns=["run_id", "variable", "time_h", "value"])
    return obs, strata


def test_clamp_detects_titer_clamped():
    obs, strata = _praaj_like()
    info = detect_clamp(obs, strata, channels=["product_g_l"])["product_g_l"]
    assert info.clamped is True
    assert info.eta_squared >= 0.8  # almost all variance is between-campaign


def test_resolver_autoprefers_free_rate_on_clamped_titer():
    obs, strata = _praaj_like()
    obj = resolve_objective_free(obs, strata=strata)
    assert isinstance(obj, Objective)
    assert obj.kind == "rate"                 # auto-preferred productivity
    assert obj.base_channel == "product_g_l"
    assert obj.clamped_base is True
    # the rate actually differs across runs that share a target (it's free)
    outs = obj.outcome_per_run(obs)
    lows = [v for r, v in outs.items() if r.startswith("L")]
    assert max(lows) != min(lows)


def test_resolver_uses_channel_when_titer_is_free():
    # titer varies WITHIN each stratum (not pinned to target) -> free -> use channel
    rows, strata = [], {}
    for i in range(6):
        rid = f"L{i}"; strata[rid] = 100
        rows += [(rid, "product_g_l", 40.0, 80.0 + 15 * (i % 3))]  # big within-stratum spread
    for i in range(6):
        rid = f"H{i}"; strata[rid] = 160
        rows += [(rid, "product_g_l", 40.0, 85.0 + 15 * (i % 3))]
    obs = pd.DataFrame(rows, columns=["run_id", "variable", "time_h", "value"])
    obj = resolve_objective_free(obs, strata=strata)
    assert obj.kind == "channel"
    assert obj.clamped_base is False


def test_user_override_wins_even_if_clamped():
    obs, strata = _praaj_like()
    # add an ethanol channel the user asks about
    extra = pd.DataFrame([(f"L{i}", "ethanol_g_l", 40.0, 2.0 + i) for i in range(6)],
                         columns=["run_id", "variable", "time_h", "value"])
    obs2 = pd.concat([obs, extra], ignore_index=True)

    class UQ:
        affected_variables = ["ethanol_g_l"]
    obj = resolve_objective_free(obs2, user_question=UQ(), strata=strata)
    assert obj.base_channel == "ethanol_g_l"
    assert obj.kind == "channel"


def test_no_strata_uses_channel_not_rate():
    # without strata the clamp is unjudgeable -> use the channel as-is, flagged.
    obs, _ = _praaj_like()
    obj = resolve_objective_free(obs, strata=None)
    assert obj.kind == "channel"
    assert "no strata" in obj.reason


def test_refuses_when_no_objective_channel_present():
    obs = pd.DataFrame([("R0", "od600_au", 40.0, 5.0)],
                       columns=["run_id", "variable", "time_h", "value"])
    assert resolve_objective_free(obs, strata={"R0": 1}) is None
