"""Lever discovery: the experiment's controllable inputs, read from its data.

Levers come from run_conditions metadata (design factors, numeric or
categorical) and from varying observation initial conditions — never a hardcoded
knob list. A source must VARY across runs to be a lever; the objective is never
a lever.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fermdocs_optimize.lever_discovery import (
    build_design,
    discover_levers,
)


def _obs(rows):
    return pd.DataFrame(rows, columns=["run_id", "variable", "time_h", "value"])


def test_metadata_numeric_and_categorical_levers():
    dossier = {"run_conditions": {
        "R0": {"nitrogen_source": {"value": "CSL"}, "feed_g_l": {"numeric": 5.0, "value": "5"}},
        "R1": {"nitrogen_source": {"value": "YE"}, "feed_g_l": {"numeric": 9.0, "value": "9"}},
    }}
    obs = _obs([("R0", "product_g_l", 48.0, 100.0), ("R1", "product_g_l", 48.0, 130.0)])
    levers = {l.name: l for l in discover_levers(dossier, obs)}
    assert levers["nitrogen_source"].kind == "categorical"
    assert levers["nitrogen_source"].source == "metadata"
    assert levers["feed_g_l"].kind == "numeric"
    assert levers["feed_g_l"].observed_range == (5.0, 9.0)


def test_constant_metadata_factor_is_skipped():
    dossier = {"run_conditions": {
        "R0": {"pH": {"numeric": 6.0, "value": "6"}, "src": {"value": "A"}},
        "R1": {"pH": {"numeric": 6.0, "value": "6"}, "src": {"value": "B"}},
    }}
    obs = _obs([("R0", "product_g_l", 1.0, 10.0), ("R1", "product_g_l", 1.0, 20.0)])
    names = {l.name for l in discover_levers(dossier, obs)}
    assert "pH" not in names      # constant -> not a lever
    assert "src" in names         # varies -> a lever


def test_derived_initial_condition_lever_from_observations():
    # substrate_g_l initial differs across runs -> a derived numeric lever.
    obs = _obs([
        ("R0", "substrate_g_l", 0.0, 100.0), ("R0", "substrate_g_l", 5.0, 30.0),
        ("R0", "product_g_l", 48.0, 90.0),
        ("R1", "substrate_g_l", 0.0, 150.0), ("R1", "substrate_g_l", 5.0, 40.0),
        ("R1", "product_g_l", 48.0, 120.0),
    ])
    levers = {l.name: l for l in discover_levers(None, obs)}
    assert "substrate_g_l.initial" in levers
    lev = levers["substrate_g_l.initial"]
    assert lev.source == "derived"
    assert lev.observed_range == (100.0, 150.0)
    # the objective is never itself a lever
    assert "product_g_l.initial" not in levers


def test_metadata_lever_shadows_same_named_derived_channel():
    dossier = {"run_conditions": {
        "R0": {"substrate_g_l": {"numeric": 100.0, "value": "100"}},
        "R1": {"substrate_g_l": {"numeric": 150.0, "value": "150"}},
    }}
    obs = _obs([
        ("R0", "substrate_g_l", 0.0, 100.0), ("R0", "product_g_l", 48.0, 90.0),
        ("R1", "substrate_g_l", 0.0, 150.0), ("R1", "product_g_l", 48.0, 120.0),
    ])
    names = [l.name for l in discover_levers(dossier, obs)]
    # the metadata factor wins; we do NOT also emit substrate_g_l.initial
    assert names.count("substrate_g_l") == 1
    assert "substrate_g_l.initial" not in names


def test_no_levers_when_nothing_varies():
    dossier = {"run_conditions": {"R0": {"x": {"value": "A"}}, "R1": {"x": {"value": "A"}}}}
    obs = _obs([("R0", "product_g_l", 1.0, 10.0), ("R1", "product_g_l", 1.0, 11.0)])
    assert discover_levers(dossier, obs) == []


def test_build_design_one_hots_categorical_and_aligns_outcome():
    dossier = {"run_conditions": {
        "R0": {"src": {"value": "A"}}, "R1": {"src": {"value": "B"}},
        "R2": {"src": {"value": "A"}},
    }}
    obs = _obs([
        ("R0", "product_g_l", 48.0, 100.0),
        ("R1", "product_g_l", 48.0, 130.0),
        ("R2", "product_g_l", 48.0, 110.0),
    ])
    levers = discover_levers(dossier, obs)
    design = build_design(levers, obs)
    assert design is not None
    # one-hot: two columns (A, B), rows aligned to sorted run ids R0,R1,R2
    assert design.X.shape == (3, 2)
    assert list(design.y) == [100.0, 130.0, 110.0]
    assert set(design.onehot_groups["src"].values()) == {"A", "B"}
    # each row has exactly one hot column
    assert np.all(design.X.sum(axis=1) == 1.0)


def test_build_design_drops_runs_missing_a_lever_value():
    dossier = {"run_conditions": {
        "R0": {"src": {"value": "A"}}, "R1": {"src": {"value": "B"}},
    }}
    # R2 has an outcome but no run_conditions entry -> not usable for fitting.
    obs = _obs([
        ("R0", "product_g_l", 48.0, 100.0),
        ("R1", "product_g_l", 48.0, 130.0),
        ("R2", "product_g_l", 48.0, 120.0),
    ])
    levers = discover_levers(dossier, obs)
    design = build_design(levers, obs)
    assert design.run_ids == ["R0", "R1"]
