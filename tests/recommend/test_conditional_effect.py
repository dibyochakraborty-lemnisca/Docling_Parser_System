"""A1 — conditional (stratified) effect estimator.

The behaviors that close the wrong-topic failure class:
  - a lever aliased with the covariate (e.g. nitrogen source only ever varied
    together with the target campaign) is reported NOT SEPARABLE — its apparent
    effect is not attributable to the lever. This is the nutrient-vs-target
    confound dying mechanically.
  - a lever that genuinely varies WITHIN each stratum keeps its real effect
    (feed-window-vs-productivity survives conditioning on target).
  - small-n stratification flags insufficient power instead of emitting a crisp
    point estimate (don't replace confident-wrong with confident-underpowered).
"""
from __future__ import annotations

import pandas as pd
import pytest

from fermdocs.analysis import cross_run
from fermdocs.analysis.computation_cache import reset_default_cache


@pytest.fixture(autouse=True)
def _isolate_cache():
    reset_default_cache()
    yield
    reset_default_cache()


def _bundle(rows_meta, titer_fn):
    """rows_meta: list of (run_id, {knob: value}). titer_fn(meta) -> peak titer."""
    rc, obs_rows = {}, []
    for rid, meta in rows_meta:
        rc[rid] = {k: {"value": v} for k, v in meta.items()}
        obs_rows.append((rid, "product_g_l", 48.0, titer_fn(meta)))
    return {"run_conditions": rc}, pd.DataFrame(
        obs_rows, columns=["run_id", "variable", "time_h", "value"])


def test_aliased_lever_is_not_separable():
    # nitrogen source is perfectly aliased with target: 'Leiber H' ONLY appears in
    # the 160 campaign, 'DBY'/'CSL' only in the 100 campaign. Titer is set by target.
    rows = []
    for i in range(5):
        rows.append((f"L{i}", {"nitrogen": "DBY", "target": 100}))      # DBY only at 100
    for i in range(5):
        rows.append((f"H{i}", {"nitrogen": "Leiber_H", "target": 160}))  # Leiber_H only at 160
    dossier, obs = _bundle(rows, lambda m: float(m["target"]))
    r = cross_run.lever_effect_conditioned(
        dossier, obs, "nitrogen", objective="product_g_l", conditioning=["target"])
    assert r["separable"] is False
    assert r["confounded_with"] == "target"
    assert r["pooled_delta"] is None
    assert "not separable" in r["separability_note"]


def test_real_within_stratum_lever_survives_conditioning():
    # feed window varies WITHIN each target campaign and genuinely moves the outcome
    # at fixed target -> survives conditioning with a real pooled effect.
    rows = []
    for i in range(6):
        feed = "short" if i % 2 == 0 else "long"
        # within each target, short feed -> higher outcome by a fixed margin
        rows.append((f"A{i}", {"feed": feed, "target": 100}))
        rows.append((f"B{i}", {"feed": feed, "target": 160}))

    def titer(m):
        bump = 8.0 if m["feed"] == "short" else 0.0
        return m["target"] + bump
    dossier, obs = _bundle(rows, titer)
    r = cross_run.lever_effect_conditioned(
        dossier, obs, "feed", objective="product_g_l", conditioning=["target"])
    assert r["separable"] is True
    assert r["pooled_delta"] is not None
    assert abs(r["pooled_delta"]) > 0          # real effect survives stratification
    assert r["n_strata_effective"] == 2        # varied within both campaigns
    assert r["power"] == "ok"


def test_confound_collapses_relative_to_unconditioned():
    # The headline test: a lever whose UNconditioned effect looks big (because it
    # tracks the target) collapses to not-separable once conditioned on target.
    rows = []
    for i in range(5):
        rows.append((f"L{i}", {"nitrogen": "DBY", "target": 100}))
    for i in range(5):
        rows.append((f"H{i}", {"nitrogen": "Leiber_H", "target": 160}))
    dossier, obs = _bundle(rows, lambda m: float(m["target"]))
    uncond = cross_run.lever_effects(dossier, obs, objective="product_g_l")
    assert "nitrogen" in uncond and abs(uncond["nitrogen"]["delta"]) > 10  # looks big
    cond = cross_run.lever_effect_conditioned(
        dossier, obs, "nitrogen", objective="product_g_l", conditioning=["target"])
    assert cond["separable"] is False          # the apparent effect was the target


def test_small_n_flags_insufficient_power():
    # lever varies within strata but cells are tiny -> flag power, don't crown.
    rows = [
        ("A0", {"feed": "short", "target": 100}),
        ("A1", {"feed": "long", "target": 100}),
        ("B0", {"feed": "short", "target": 160}),
        ("B1", {"feed": "long", "target": 160}),
    ]
    dossier, obs = _bundle(rows, lambda m: m["target"] + (5 if m["feed"] == "short" else 0))
    r = cross_run.lever_effect_conditioned(
        dossier, obs, "feed", objective="product_g_l", conditioning=["target"])
    assert r is not None
    assert r["power"] == "insufficient"
    assert r["power_note"]


def test_conditioned_result_is_cached():
    rows = [("A0", {"feed": "short", "target": 100}), ("A1", {"feed": "long", "target": 100}),
            ("A2", {"feed": "short", "target": 100}), ("A3", {"feed": "long", "target": 100}),
            ("B0", {"feed": "short", "target": 160}), ("B1", {"feed": "long", "target": 160})]
    dossier, obs = _bundle(rows, lambda m: m["target"] + (5 if m["feed"] == "short" else 0))
    a = cross_run.lever_effect_conditioned(dossier, obs, "feed", objective="product_g_l",
                                           conditioning=["target"])
    b = cross_run.lever_effect_conditioned(dossier, obs, "feed", objective="product_g_l",
                                           conditioning=["target"])
    assert a is b  # F2 cache, conditioning in the key
