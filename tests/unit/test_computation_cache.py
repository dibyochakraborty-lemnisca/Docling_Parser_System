"""F2 — canonical computation cache: reproducibility + data-version invalidation.

The failure this closes: the same effect re-derived narratively drifted
(+1.532 -> +2.026). With the cache, re-asking the same (function, objective, data)
question returns the identical object — drift becomes impossible — and changing
the data busts the cache so you never serve a stale number.
"""
from __future__ import annotations

import pandas as pd
import pytest

from fermdocs.analysis import cross_run
from fermdocs.analysis.computation_cache import (
    ComputationCache,
    data_version,
    fingerprint_df,
    make_key,
    reset_default_cache,
)


@pytest.fixture(autouse=True)
def _isolate_cache():
    reset_default_cache()
    yield
    reset_default_cache()


def _frame(seed_shift=0.0):
    sources = ["CSL", "YE"]
    base = {"CSL": 100.0, "YE": 135.0}
    rc, rows = {}, []
    for i in range(8):
        src = sources[i % 2]
        rid = f"R{i}"
        rc[rid] = {"nitrogen_source": {"value": src}}
        rows.append((rid, "product_g_l", 48.0, base[src] + seed_shift + (i % 2) * 0.1))
    dossier = {"run_conditions": rc}
    obs = pd.DataFrame(rows, columns=["run_id", "variable", "time_h", "value"])
    return dossier, obs


def test_reask_returns_identical_object():
    dossier, obs = _frame()
    r1 = cross_run.lever_effects(dossier, obs, objective="product_g_l")
    r2 = cross_run.lever_effects(dossier, obs, objective="product_g_l")
    assert r1 is r2  # same cached object — single-sourced, no re-derivation


def test_reask_is_byte_identical():
    import json
    dossier, obs = _frame()
    a = cross_run.lever_effects(dossier, obs, objective="product_g_l")
    b = cross_run.lever_effects(dossier, obs, objective="product_g_l")
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_equivalent_but_distinct_frames_hit_same_cache():
    # Two DataFrames with identical CONTENT must resolve to the same computation.
    d1, o1 = _frame()
    d2, o2 = _frame()
    assert o1 is not o2
    r1 = cross_run.lever_effects(d1, o1, objective="product_g_l")
    r2 = cross_run.lever_effects(d2, o2, objective="product_g_l")
    assert r1 is r2  # content-keyed, not identity-keyed


def _frame_effect(ye_base):
    # Effect (not just level) varies: only the YE group's titer changes, so the
    # category-mean delta genuinely differs (shift-invariance can't hide it).
    rc, rows = {}, []
    base = {"CSL": 100.0, "YE": ye_base}
    for i in range(8):
        src = ["CSL", "YE"][i % 2]
        rid = f"R{i}"
        rc[rid] = {"nitrogen_source": {"value": src}}
        rows.append((rid, "product_g_l", 48.0, base[src] + (i % 2) * 0.1))
    return {"run_conditions": rc}, pd.DataFrame(
        rows, columns=["run_id", "variable", "time_h", "value"])


def test_data_change_busts_cache():
    d1, o1 = _frame_effect(ye_base=135.0)
    d2, o2 = _frame_effect(ye_base=180.0)  # YE effect is larger -> different delta
    r1 = cross_run.lever_effects(d1, o1, objective="product_g_l")
    r2 = cross_run.lever_effects(d2, o2, objective="product_g_l")
    assert r1 is not r2  # recomputed on changed data (data_version busted the key)
    # and the recomputed numbers actually differ (not a stale serve)
    assert r1["nitrogen_source"]["delta"] != r2["nitrogen_source"]["delta"]


def test_objective_is_part_of_key():
    dossier, obs = _frame()
    # add a second channel so a different objective is a distinct computation
    extra = pd.DataFrame([(f"R{i}", "od600_au", 48.0, 5.0 + i) for i in range(8)],
                         columns=["run_id", "variable", "time_h", "value"])
    obs2 = pd.concat([obs, extra], ignore_index=True)
    a = cross_run.lever_effects(dossier, obs2, objective="product_g_l")
    b = cross_run.lever_effects(dossier, obs2, objective="od600_au")
    assert a is not b  # objective discriminates the key


def test_data_version_and_key_helpers():
    _, o1 = _frame()
    _, o2 = _frame()
    assert fingerprint_df(o1) == fingerprint_df(o2)          # content-stable
    assert data_version(o1) == data_version(o2)
    assert data_version(o1, {"a": 1}) != data_version(o1, {"a": 2})
    # conditioning set is order-independent in the key
    k1 = make_key("f", objective="p", data_ver="v", conditioning=["target", "scale"])
    k2 = make_key("f", objective="p", data_ver="v", conditioning=["scale", "target"])
    assert k1 == k2


def test_cache_unit_hits_and_misses():
    c = ComputationCache()
    calls = {"n": 0}
    def compute():
        calls["n"] += 1
        return {"v": 1}
    k = make_key("f", objective="p", data_ver="v1")
    assert c.get_or_compute(k, compute) == {"v": 1}
    assert c.get_or_compute(k, compute) == {"v": 1}
    assert calls["n"] == 1 and c.hits == 1 and c.misses == 1
