"""F1 — clampedness detection: which channels are SET by design vs FREE to move.

The praaj lesson: peak titer is target-clamped — operators feed to hit a campaign
target, so titer barely varies *within* a campaign (CV ~1-2%) while productivity
moves freely. Optimizing a clamped channel chases ghosts; the objective must be a
free variable.

Signal (data-relative, parameter-light): the fraction of a channel's variance
explained by the strata — eta-squared = SS_between / SS_total. A channel whose
variance is almost entirely *between* strata (eta^2 high) is determined by the
stratum (clamped); one whose variance lives *within* strata (eta^2 low) is free.
This uses ONE statistical threshold (not a domain constant), and that threshold is
exactly what the IndPenSim cross-dataset check (V) must validate — flagged below.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Variance-ratio above which a channel is "set by the stratum" => clamped.
# STATISTICAL threshold (eta^2 in [0,1]), not a domain value. MUST be validated on
# a second dataset (V/IndPenSim) before its generality is trusted — see plan F1/V.
CLAMP_ETA2 = 0.8
# Need at least this many runs with a value to judge clampedness at all.
_MIN_RUNS = 4


@dataclass(frozen=True)
class ClampInfo:
    channel: str
    clamped: bool
    eta_squared: float          # between-stratum variance fraction
    n_runs: int
    n_strata: int
    reason: str


def _peak_per_run(obs_df: pd.DataFrame, channel: str) -> dict[str, float]:
    if not {"run_id", "variable", "value"}.issubset(obs_df.columns):
        return {}
    df = obs_df[obs_df["variable"].astype(str) == channel].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    if df.empty:
        return {}
    return {str(r): float(g["value"].max()) for r, g in df.groupby("run_id")}


def _eta_squared(values: dict[str, float], strata: dict[str, object]) -> tuple[float, int]:
    """Fraction of variance explained by the stratum label. Returns (eta^2, n_strata).

    eta^2 = SS_between / SS_total. 1.0 => the stratum fully determines the value
    (clamped); 0.0 => the stratum explains nothing (free)."""
    groups: dict[object, list[float]] = {}
    for run, v in values.items():
        s = strata.get(run)
        if s is None:
            continue
        groups.setdefault(s, []).append(v)
    allv = [v for g in groups.values() for v in g]
    if len(allv) < _MIN_RUNS or len(groups) < 2:
        return 0.0, len(groups)
    grand = float(np.mean(allv))
    ss_total = float(np.sum([(v - grand) ** 2 for v in allv]))
    if ss_total <= 0:
        return 1.0, len(groups)  # no variance at all => fully "determined" (clamped)
    ss_between = float(np.sum([len(g) * (float(np.mean(g)) - grand) ** 2 for g in groups.values()]))
    return ss_between / ss_total, len(groups)


def derive_strata(dossier: dict | None) -> tuple[dict[str, object], list[str]]:
    """The campaign/target stratum for clampedness + conditioning, read from a
    structured run_conditions knob whose name signals a titer target (B1' produces
    ``titer_target_g_l``). Returns ``({run_id: value}, [knob])`` or ``({}, [])``
    when no target knob is structured yet — callers then run unconditioned
    (graceful pre-B1'). Shared by the loader (B2) and the gate bridge (A3)."""
    conditions = (dossier or {}).get("run_conditions") or {}
    target_keys: set[str] = set()
    for knobs in conditions.values():
        if isinstance(knobs, dict):
            target_keys |= {k for k in knobs if "target" in k.lower()}
    if not target_keys:
        return {}, []
    key = sorted(target_keys)[0]
    strata: dict[str, object] = {}
    for run, knobs in conditions.items():
        if isinstance(knobs, dict) and key in knobs:
            spec = knobs[key]
            v = spec.get("value") if isinstance(spec, dict) else spec
            if v is not None:
                strata[str(run)] = v
    return (strata, [key]) if len(set(strata.values())) >= 2 else ({}, [])


def detect_clamp(
    obs_df: pd.DataFrame,
    strata: dict[str, object],
    channels: list[str] | None = None,
) -> dict[str, ClampInfo]:
    """Classify each measured channel as clamped (set by stratum) or free.

    `strata` maps run_id -> stratum label (e.g. the target/campaign). Without a
    real stratum (fewer than 2), nothing can be judged clamped — clampedness is a
    within-vs-between question, so a stratum is REQUIRED (this is the F1<-B1'
    dependency: target must be structured first)."""
    if channels is None:
        channels = sorted(obs_df["variable"].astype(str).unique()) if "variable" in obs_df else []
    out: dict[str, ClampInfo] = {}
    for ch in channels:
        vals = _peak_per_run(obs_df, ch)
        eta, n_strata = _eta_squared(vals, strata)
        if len(vals) < _MIN_RUNS or n_strata < 2:
            out[ch] = ClampInfo(ch, False, round(eta, 4), len(vals), n_strata,
                                "insufficient strata/runs to judge clampedness")
            continue
        clamped = eta >= CLAMP_ETA2
        out[ch] = ClampInfo(
            ch, clamped, round(eta, 4), len(vals), n_strata,
            (f"{round(100*eta)}% of variance is between-stratum (>= "
             f"{round(100*CLAMP_ETA2)}%): set by the stratum, not free"
             if clamped else
             f"only {round(100*eta)}% of variance is between-stratum: free to move"))
    return out
