"""Cross-run comparative intervention engine.

The dynamic bake-off (brewtwin families + discovered mechanistic) fits ONE
run's trajectory and scores held-out R^2. On sparse fed-batch data (5-9
points/run) those fits are unidentifiable and the rubric honestly refuses.

But the signal in a multi-run dataset lives ACROSS runs: many batches at
different operating conditions, each with an outcome (final titer). 15 runs
= 15 points in knob-space. This module reads the per-run operating-condition
knobs the layout detector pulled from sheet metadata
(``dossier["run_conditions"]``) and relates each knob to the run outcome, then
emits the controllable levers as interventions.

This is OBSERVATIONAL, not causal: it reports associations across runs that
were not designed as a controlled experiment. Every intervention carries that
caveat. The engine clears its gate only when there are enough runs, the knob
actually varies, and the association is large enough to matter — otherwise it
stays silent rather than invent a lever.

Pure + deterministic (numpy only); no LLM, no brewtwin, no I/O beyond the
DataFrame it is handed. Wired into the recommendation in agent.py: it WINS the
recommendation when the dynamic bake-off refuses but it clears its gate, and
SUPPLEMENTS the interventions when a dynamic model wins.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

# Minimum runs with both a knob value and an outcome to attempt a comparison.
MIN_RUNS = 4
# A knob's association must move the outcome by at least this fraction of the
# observed outcome spread to be worth recommending (filters noise).
MIN_EFFECT_FRAC = 0.15
# Default outcome variable (the thing we want to maximize).
DEFAULT_OBJECTIVE = "product_g_l"
# An association moving the objective by less than this fraction of the
# run-to-run spread is WEAK/preliminary — likely within noise on a small,
# confounded campaign. Used to label such associations so they aren't argued as
# confident levers (a +2 g/L swing on a ~60 g/L spread is not a validated effect).
WEAK_EFFECT_FRAC = 0.2


def is_weak_effect(norm_effect: float | None) -> bool:
    return (norm_effect or 0.0) < WEAK_EFFECT_FRAC


def run_outcomes(obs_df: pd.DataFrame, objective: str) -> dict[str, float]:
    """Peak value of the objective variable per run (robust to end dropout)."""
    df = obs_df.copy()
    if "variable" not in df or "run_id" not in df or "value" not in df:
        return {}
    df = df[df["variable"].astype(str) == objective]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    if df.empty:
        return {}
    return {str(rid): float(g["value"].max()) for rid, g in df.groupby("run_id")}


def _numeric_knob_effect(
    pairs: list[tuple[float, float]], outcome_spread: float,
    *, min_frac: float = MIN_EFFECT_FRAC,
) -> dict[str, Any] | None:
    """Linear association of a numeric knob with the outcome across runs.

    `min_frac` is the suppression gate: an effect smaller than this fraction of
    the outcome spread returns None. The recommend stage uses the default (only
    surface levers worth acting on); ranking callers pass 0.0 to get an UNgated
    number for every varying lever."""
    xs = np.array([p[0] for p in pairs], dtype=float)
    ys = np.array([p[1] for p in pairs], dtype=float)
    if len(set(xs.tolist())) < 2:
        return None  # knob does not vary
    slope, intercept = np.polyfit(xs, ys, 1)
    if not math.isfinite(slope):
        return None
    x_lo, x_hi = float(xs.min()), float(xs.max())
    # Predict at the in-range knob end the fit favors.
    best_x = x_hi if slope >= 0 else x_lo
    predicted = float(slope * best_x + intercept)
    baseline = float(np.median(ys))
    delta = predicted - baseline
    if outcome_spread <= 0 or abs(delta) < min_frac * outcome_spread:
        return None
    return {
        "best_setting": round(best_x, 4),
        "baseline_value": round(baseline, 4),
        "predicted_value": round(predicted, 4),
        "delta": round(delta, 4),
        "direction": "increase" if slope >= 0 else "decrease",
        "n": len(pairs),
        "observed_range": [round(x_lo, 4), round(x_hi, 4)],
    }


def _categorical_knob_effect(
    pairs: list[tuple[str, float]], outcome_spread: float,
    *, min_frac: float = MIN_EFFECT_FRAC,
) -> dict[str, Any] | None:
    """Best category vs overall mean for a categorical knob. `min_frac`: see
    `_numeric_knob_effect` (0.0 = ungated, for ranking)."""
    groups: dict[str, list[float]] = {}
    for cat, y in pairs:
        groups.setdefault(cat, []).append(y)
    if len(groups) < 2:
        return None  # knob does not vary
    means = {c: float(np.mean(v)) for c, v in groups.items()}
    overall = float(np.mean([y for _, y in pairs]))
    best_cat = max(means, key=means.get)
    delta = means[best_cat] - overall
    if outcome_spread <= 0 or abs(delta) < min_frac * outcome_spread:
        return None
    return {
        "best_setting": best_cat,
        "baseline_value": round(overall, 4),
        "predicted_value": round(means[best_cat], 4),
        "delta": round(delta, 4),
        "direction": "set_to",
        "n": len(pairs),
        "category_means": {c: round(m, 4) for c, m in means.items()},
    }


def analyze(
    dossier: dict[str, Any] | None,
    obs_df: pd.DataFrame,
    *,
    objective: str = DEFAULT_OBJECTIVE,
) -> dict[str, Any] | None:
    """Relate each per-run knob to the outcome across runs.

    Returns a dict ``{cleared, interventions, summary, n_runs, objective}`` or
    None when there are no usable conditions/outcomes. ``cleared`` is True only
    when at least one knob shows a large-enough association over >= MIN_RUNS
    runs; callers should treat a cleared result as a recommendable verdict and
    an uncleared one as "analyzed, nothing actionable".

    F2: cached on (objective, data) — re-asking returns the identical verdict.
    Returned dict is read-only.
    """
    from fermdocs.analysis.computation_cache import (
        data_version,
        default_cache,
        make_key,
    )

    key = make_key(
        "cross_run.analyze",
        objective=objective,
        data_ver=data_version(obs_df, dossier),
    )
    return default_cache().get_or_compute(
        key, lambda: _analyze_uncached(dossier, obs_df, objective=objective)
    )


def _analyze_uncached(
    dossier: dict[str, Any] | None,
    obs_df: pd.DataFrame,
    *,
    objective: str = DEFAULT_OBJECTIVE,
) -> dict[str, Any] | None:
    conditions = (dossier or {}).get("run_conditions") or {}
    if not conditions:
        return None
    outcomes = run_outcomes(obs_df, objective)
    if len(outcomes) < MIN_RUNS:
        return {
            "cleared": False,
            "interventions": [],
            "n_runs": len(outcomes),
            "objective": objective,
            "summary": (
                f"cross-run analysis skipped: only {len(outcomes)} run(s) have "
                f"{objective}; need >= {MIN_RUNS}."
            ),
        }
    ys = list(outcomes.values())
    outcome_spread = float(max(ys) - min(ys))

    knob_pairs = _gather_knob_pairs(conditions, outcomes)

    interventions: list[dict[str, Any]] = []
    for knob, pairs in sorted(knob_pairs.items()):
        if len(pairs) < MIN_RUNS:
            continue
        numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v, _ in pairs)
        effect = (
            _numeric_knob_effect([(float(v), y) for v, y in pairs], outcome_spread)
            if numeric
            else _categorical_knob_effect([(str(v), y) for v, y in pairs], outcome_spread)
        )
        if effect is None:
            continue
        interventions.append(_to_intervention(knob, objective, effect))

    interventions.sort(key=lambda i: abs(i.get("delta") or 0.0), reverse=True)
    cleared = bool(interventions)
    if cleared:
        top = interventions[0]
        summary = (
            f"cross-run comparative over {len(outcomes)} runs: "
            f"{top['description']} (assoc. {top['delta']:+} {objective})."
        )
    else:
        summary = (
            f"cross-run analysis over {len(outcomes)} runs found no knob with a "
            f"large-enough association to {objective}."
        )
    return {
        "cleared": cleared,
        "interventions": interventions,
        "n_runs": len(outcomes),
        "objective": objective,
        "summary": summary,
    }


def _spec_value(spec: Any) -> Any:
    """Pull a usable value from a run_conditions cell: the extractor's clean
    `numeric` when present (set only when the whole cell was a number), else the
    verbatim `value` (categorical). Shared by analyze + lever_effects."""
    if isinstance(spec, dict):
        v = spec.get("numeric")
        return spec.get("value") if v is None else v
    return spec


def _gather_knob_pairs(
    conditions: dict[str, Any], outcomes: dict[str, float]
) -> dict[str, list[tuple[Any, float]]]:
    """Every knob seen across runs → its (value, outcome) pairs."""
    knob_pairs: dict[str, list[tuple[Any, float]]] = {}
    for run_id, knobs in conditions.items():
        outcome = outcomes.get(str(run_id))
        if outcome is None or not isinstance(knobs, dict):
            continue
        for knob, spec in knobs.items():
            val = _spec_value(spec)
            if val is None:
                continue
            knob_pairs.setdefault(knob, []).append((val, outcome))
    return knob_pairs


def _value_key(v: Any):
    """Stable grouping key: rounded float for numbers, str otherwise."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return round(float(v), 9)
    return str(v)


def _lever_run_values(conditions: dict, outcomes: dict[str, float]) -> dict[str, dict[str, Any]]:
    """{lever: {run_id: value}} over runs that have an outcome — aligned by run
    so we can compare which runs each lever partitions together."""
    rv: dict[str, dict[str, Any]] = {}
    for run_id, knobs in conditions.items():
        rid = str(run_id)
        if rid not in outcomes or not isinstance(knobs, dict):
            continue
        for knob, spec in knobs.items():
            val = _spec_value(spec)
            if val is not None:
                rv.setdefault(knob, {})[rid] = val
    return rv


def _partition(run_values: dict[str, Any]) -> frozenset:
    """The set of run-groups a lever induces (runs sharing a value)."""
    groups: dict[Any, set[str]] = {}
    for run, v in run_values.items():
        groups.setdefault(_value_key(v), set()).add(run)
    return frozenset(frozenset(s) for s in groups.values())


def _confound_flags(rvbl: dict[str, dict[str, Any]]) -> dict[str, tuple[bool, str | None]]:
    """Data-relative confound detection (no hardcoded 'this factor is bad'):
      - a lever with ~one distinct value per run is a run LABEL, not a knob
        (e.g. glycerol lot number) — its 'effect' is just between-run variance;
      - two levers that partition the runs IDENTICALLY are aliased — their
        effects are not separable (e.g. impeller type tracking reactor id).
    Either way the association is not attributable, so it must not be argued as
    a causal lever. Returns {lever: (confounded, reason)}."""
    flags: dict[str, tuple[bool, str | None]] = {}
    levers = list(rvbl)
    for a in levers:
        rva = rvbl[a]
        if len(rva) < MIN_RUNS:
            flags[a] = (False, None)
            continue
        if len({_value_key(v) for v in rva.values()}) >= len(rva):
            flags[a] = (True, "aliases the run index (~one distinct value per run)")
            continue
        match = None
        for b in levers:
            if b == a:
                continue
            shared = set(rva) & set(rvbl[b])
            if len(shared) < MIN_RUNS:
                continue
            pa = _partition({r: rva[r] for r in shared})
            pb = _partition({r: rvbl[b][r] for r in shared})
            if pa == pb and len(pa) > 1:
                match = b
                break
        flags[a] = (match is not None,
                    f"aliased with {match} (identical run grouping)" if match else None)
    return flags


def lever_effects(
    dossier: dict[str, Any] | None,
    obs_df: pd.DataFrame,
    *,
    objective: str = DEFAULT_OBJECTIVE,
    outcomes: dict[str, float] | None = None,
) -> dict[str, dict[str, Any]]:
    """Per-lever association with the objective across runs — UNGATED, so every
    varying metadata lever gets a number for topic ranking (not just the ones
    big enough to recommend). Returns ``{lever_name: {delta, direction, n,
    best_setting, norm_effect, ...}}`` where ``norm_effect`` in [0,1] is
    ``|delta| / outcome_spread``. Empty dict when there aren't enough runs or no
    outcomes — the caller then falls back to neutral ordering. Pure/deterministic.

    F2: routed through the canonical computation cache so every caller asking the
    same (objective, data) question resolves to ONE computed result — no
    re-derivation drift. The returned dict is read-only; callers must not mutate
    it. (A1 will thread the conditioning covariate set into the cache key.)
    """
    from fermdocs.analysis.computation_cache import (
        data_version,
        default_cache,
        fingerprint_obj,
        make_key,
    )

    key = make_key(
        "cross_run.lever_effects",
        objective=objective,
        data_ver=data_version(obs_df, dossier),
        extra=(fingerprint_obj(_sorted_outcomes(outcomes)),) if outcomes is not None else (),
    )
    return default_cache().get_or_compute(
        key,
        lambda: _lever_effects_uncached(
            dossier, obs_df, objective=objective, outcomes=outcomes
        ),
    )


def _sorted_outcomes(outcomes: dict[str, float] | None):
    return None if outcomes is None else sorted((str(k), round(float(v), 6))
                                                for k, v in outcomes.items())


def _lever_effects_uncached(
    dossier: dict[str, Any] | None,
    obs_df: pd.DataFrame,
    *,
    objective: str = DEFAULT_OBJECTIVE,
    outcomes: dict[str, float] | None = None,
) -> dict[str, dict[str, Any]]:
    conditions = (dossier or {}).get("run_conditions") or {}
    if not conditions:
        return {}
    outcomes = outcomes if outcomes is not None else run_outcomes(obs_df, objective)
    if len(outcomes) < MIN_RUNS:
        return {}
    ys = list(outcomes.values())
    spread = float(max(ys) - min(ys))
    cflags = _confound_flags(_lever_run_values(conditions, outcomes))
    out: dict[str, dict[str, Any]] = {}
    for knob, pairs in _gather_knob_pairs(conditions, outcomes).items():
        if len(pairs) < MIN_RUNS:
            continue
        numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v, _ in pairs)
        effect = (
            _numeric_knob_effect([(float(v), y) for v, y in pairs], spread, min_frac=0.0)
            if numeric
            else _categorical_knob_effect([(str(v), y) for v, y in pairs], spread, min_frac=0.0)
        )
        if effect is None:
            continue
        norm = min(abs(float(effect["delta"])) / spread, 1.0) if spread > 0 else 0.0
        confounded, reason = cflags.get(knob, (False, None))
        out[knob] = {**effect, "norm_effect": round(norm, 4),
                     "confounded": confounded, "confounded_with": reason}
    return out


# Minimum runs in a stratum for its within-stratum effect to count toward power.
MIN_STRATUM_RUNS = 3


def lever_effect_conditioned(
    dossier: dict[str, Any] | None,
    obs_df: pd.DataFrame,
    lever: str,
    *,
    objective: str = DEFAULT_OBJECTIVE,
    conditioning: list[str],
    outcomes: dict[str, float] | None = None,
) -> dict[str, Any] | None:
    """A1 — the conditional (stratified) effect of one lever on the objective,
    holding `conditioning` covariates constant.

    This is the machine that kills the confound mechanically: stratify the runs by
    the covariate(s), estimate the lever's effect WITHIN each stratum (reusing the
    univariate estimators), and pool sample-weighted. Two failure modes are made
    explicit rather than hidden:

      - **not separable** — if the lever does not vary within ANY stratum, it is
        aliased with the covariate (e.g. 'Leiber H only ever used in the high-titer
        campaign'). The effect is *not attributable* to the lever; ``separable`` is
        False and ``confounded_with`` names the covariate. This is strictly stronger
        than the aliasing-only flag in ``lever_effects``.
      - **insufficient power** — stratifying small n into smaller cells trades
        confound bias for variance. If too few runs carry a within-stratum lever
        contrast, ``power`` is 'insufficient' and the pooled point estimate must NOT
        be trusted as crisp (report the CI, don't crown a winner on noise).

    Routed through the F2 cache keyed on (lever, objective, conditioning, data).
    Returns None when the objective/conditions are unusable. Read-only result."""
    from fermdocs.analysis.computation_cache import (
        data_version,
        default_cache,
        fingerprint_obj,
        make_key,
    )

    key = make_key(
        "cross_run.lever_effect_conditioned",
        objective=objective,
        data_ver=data_version(obs_df, dossier),
        conditioning=conditioning,
        extra=(lever, fingerprint_obj(_sorted_outcomes(outcomes)) if outcomes is not None else ""),
    )
    return default_cache().get_or_compute(
        key,
        lambda: _lever_effect_conditioned_uncached(
            dossier, obs_df, lever, objective=objective,
            conditioning=conditioning, outcomes=outcomes,
        ),
    )


def _lever_effect_conditioned_uncached(
    dossier: dict[str, Any] | None,
    obs_df: pd.DataFrame,
    lever: str,
    *,
    objective: str,
    conditioning: list[str],
    outcomes: dict[str, float] | None = None,
) -> dict[str, Any] | None:
    conditions = (dossier or {}).get("run_conditions") or {}
    if not conditions or not conditioning:
        return None
    outcomes = outcomes if outcomes is not None else run_outcomes(obs_df, objective)
    if len(outcomes) < MIN_RUNS:
        return None
    ys = list(outcomes.values())
    spread = float(max(ys) - min(ys))

    # Per-run rows: (lever_value, stratum_key, outcome) over runs with everything present.
    rows: list[tuple[Any, tuple, float]] = []
    for run_id, knobs in conditions.items():
        o = outcomes.get(str(run_id))
        if o is None or not isinstance(knobs, dict):
            continue
        lv = _spec_value(knobs.get(lever))
        if lv is None:
            continue
        cov = tuple(_spec_value(knobs.get(c)) for c in conditioning)
        if any(c is None for c in cov):
            continue
        rows.append((lv, cov, o))
    if len(rows) < MIN_RUNS:
        return None

    # Group by stratum; estimate the lever effect WITHIN each stratum.
    strata: dict[tuple, list[tuple[Any, float]]] = {}
    for lv, cov, o in rows:
        strata.setdefault(cov, []).append((lv, o))

    per_stratum: list[dict[str, Any]] = []
    weighted_delta, weight_total, effective_runs, n_effective_strata = 0.0, 0, 0, 0
    for cov, pairs in strata.items():
        distinct_levels = {_value_key(v) for v, _ in pairs}
        if len(distinct_levels) < 2:
            # lever does not vary inside this stratum -> contributes no contrast
            per_stratum.append({"stratum": list(cov), "n": len(pairs),
                                "delta": None, "varies": False})
            continue
        numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool)
                      for v, _ in pairs)
        eff = (_numeric_knob_effect([(float(v), y) for v, y in pairs], spread, min_frac=0.0)
               if numeric
               else _categorical_knob_effect([(str(v), y) for v, y in pairs], spread, min_frac=0.0))
        if eff is None:
            per_stratum.append({"stratum": list(cov), "n": len(pairs),
                                "delta": None, "varies": True})
            continue
        per_stratum.append({"stratum": list(cov), "n": len(pairs),
                            "delta": eff["delta"], "direction": eff["direction"],
                            "best_setting": eff["best_setting"], "varies": True})
        weighted_delta += eff["delta"] * len(pairs)
        weight_total += len(pairs)
        n_effective_strata += 1
        if len(pairs) >= MIN_STRATUM_RUNS:
            effective_runs += len(pairs)

    # Separability: lever never varied within any stratum -> aliased with covariate.
    if weight_total == 0:
        return {
            "lever": lever, "objective": objective, "conditioned_on": list(conditioning),
            "pooled_delta": None, "pooled_norm_effect": 0.0,
            "separable": False,
            "confounded_with": "+".join(conditioning),
            "separability_note": (f"{lever} does not vary within any {conditioning} "
                                  "stratum — effect is not separable from the covariate "
                                  "(aliased)."),
            "per_stratum": per_stratum, "n": len(rows),
            "n_strata_effective": 0,
            "power": "insufficient",
            "power_note": "no within-stratum contrast for the lever",
        }

    pooled = weighted_delta / weight_total
    norm = min(abs(pooled) / spread, 1.0) if spread > 0 else 0.0
    power_ok = effective_runs >= MIN_RUNS and n_effective_strata >= 1
    return {
        "lever": lever, "objective": objective, "conditioned_on": list(conditioning),
        "pooled_delta": round(pooled, 4), "pooled_norm_effect": round(norm, 4),
        "direction": "increase" if pooled >= 0 else "decrease",
        "separable": True, "confounded_with": None,
        "per_stratum": per_stratum, "n": len(rows),
        "n_strata_effective": n_effective_strata,
        "power": "ok" if power_ok else "insufficient",
        "power_note": (None if power_ok else
                       f"only {effective_runs} run(s) carry a within-stratum contrast "
                       f"(need >= {MIN_RUNS}); treat the pooled estimate as a lead, not a "
                       "validated effect."),
    }


def ranking_effects(
    dossier: dict[str, Any] | None,
    obs_df: pd.DataFrame,
    *,
    objective: str = DEFAULT_OBJECTIVE,
    outcomes: dict[str, float] | None = None,
    conditioning: list[str] | None = None,
) -> dict[str, float]:
    """B2 — the per-lever ranking score the ranker should use: the lever's effect
    on the (free) objective, CONDITIONED on `conditioning` when provided.

    This is the redirect that makes the system rank by *importance*, not salience:
      - with conditioning: a lever's score is its conditioned ``pooled_norm_effect``
        only when the effect is separable AND adequately powered; otherwise 0 — so a
        confounded or underpowered lever (the nutrient claim) sinks to the bottom
        while a real one (feed window) keeps its magnitude.
      - without conditioning (no strata yet): falls back to the unconditioned
        ``norm_effect`` (current behavior) — graceful until B1' structures the
        covariate. The objective is still the FREE one resolved by F1 (via
        ``outcomes``), so the ranking already targets the right thing.

    Returns ``{lever: score in [0,1]}``; the conditioning covariates are excluded
    from the ranking (they're the strata, not candidate levers)."""
    base = lever_effects(dossier, obs_df, objective=objective, outcomes=outcomes)
    cov = set(conditioning or [])
    if not cov:
        return {k: float(v.get("norm_effect") or 0.0) for k, v in base.items()}
    scores: dict[str, float] = {}
    for lever in base:
        if lever in cov:
            continue
        est = lever_effect_conditioned(
            dossier, obs_df, lever, objective=objective,
            conditioning=list(cov), outcomes=outcomes)
        if est is None or not est["separable"] or est["power"] != "ok":
            scores[lever] = 0.0   # confounded / underpowered -> sinks
        else:
            scores[lever] = float(est["pooled_norm_effect"])
    return scores


def _to_intervention(knob: str, objective: str, effect: dict[str, Any]) -> dict[str, Any]:
    direction = effect["direction"]
    if direction == "set_to":
        desc = f"Set {knob} to {effect['best_setting']}"
    else:
        desc = f"{direction.capitalize()} {knob} toward {effect['best_setting']}"
    caveat = (
        f"Observational association across {effect['n']} runs, not a controlled "
        "experiment; correlation is not proven causation — validate experimentally."
    )
    return {
        "knob": knob,
        "description": desc,
        "objective_metric": f"{objective}.peak",
        "baseline_value": effect["baseline_value"],
        "predicted_value": effect["predicted_value"],
        "delta": effect["delta"],
        "in_coverage": True,  # best setting is chosen within the observed range
        "caveat": caveat,
        "rationale": (
            f"Across runs, {knob} associates with {objective} "
            f"(n={effect['n']}); best observed setting {effect['best_setting']}."
        ),
    }
