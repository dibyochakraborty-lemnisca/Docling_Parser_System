"""Discover an experiment's actual levers from its data — never hardcoded.

A *lever* is a controllable input the experiment varied. The optimizer and the
opportunity debate both need to know "what could we have turned?" — and the
honest answer is whatever THIS experiment actually turned, not a fixed list of
LABS lactic-acid knobs. Two sources, both data-derived:

  1. METADATA design factors — ``dossier["run_conditions"]``, the per-run
     conditions the layout detector pulled from sheet metadata (nitrogen
     source, feed concentration, feed timing, ...). These are unambiguously
     things the experimenter set. Numeric or categorical.
  2. DERIVED initial conditions — the earliest observed value of an input
     channel per run (e.g. initial substrate loading). Observational: a
     measured starting point, labelled as such so a caller can weight it.

A source becomes a lever only if it VARIES across runs (>= 2 distinct values) —
a constant can't be optimized. The objective variable is never a lever (it's
the outcome). No domain constants live here: which levers exist, their kinds,
and their ranges all come from the data. The fixed ``KNOB_NAMES`` tuple in
``schema.py`` belongs to the LABS simulator config and is deliberately NOT used.

Pure + deterministic (numpy/pandas only); no LLM, no I/O beyond the frames it is
handed. Shared by the optimizer (``data_equation``) and the debate (``topics``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

# A source must take this many distinct values across runs to count as a lever.
_MIN_DISTINCT = 2
# Default outcome variable — never treated as a lever (it's what we maximize).
DEFAULT_OBJECTIVE = "product_g_l"
# Round numeric values to this many places when counting "distinct" (kills
# floating-point dust, not real variation).
_DISTINCT_ROUND = 9


LeverKind = Literal["numeric", "categorical"]
LeverSource = Literal["metadata", "derived"]


@dataclass(frozen=True)
class Lever:
    """One controllable input the experiment varied, with its per-run values."""

    name: str
    kind: LeverKind
    source: LeverSource
    values: dict[str, Any] = field(default_factory=dict)  # run_id -> value

    @property
    def categories(self) -> list[str]:
        """Distinct categorical levels actually observed (categorical only)."""
        seen: list[str] = []
        for v in self.values.values():
            s = str(v)
            if s not in seen:
                seen.append(s)
        return seen

    @property
    def observed_range(self) -> tuple[float, float] | None:
        """[min, max] of observed numeric values (numeric only)."""
        if self.kind != "numeric":
            return None
        nums = [float(v) for v in self.values.values()]
        return (min(nums), max(nums)) if nums else None


def _spec_value(spec: Any) -> Any:
    """Pull a usable value out of a run_conditions cell. Prefer the extractor's
    clean ``numeric`` (set only when the WHOLE cell was a number), else the
    verbatim ``value`` (categorical). Mirrors cross_run's reading exactly."""
    if isinstance(spec, dict):
        v = spec.get("numeric")
        return spec.get("value") if v is None else v
    return spec


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _distinct_count(values: list[Any]) -> int:
    if not values:
        return 0
    if all(_is_number(v) for v in values):
        return len({round(float(v), _DISTINCT_ROUND) for v in values})
    return len({str(v) for v in values})


def _metadata_levers(conditions: dict[str, Any]) -> list[Lever]:
    """Each design factor in run_conditions that varies across runs."""
    # knob -> {run_id: value}
    by_knob: dict[str, dict[str, Any]] = {}
    for run_id, knobs in conditions.items():
        if not isinstance(knobs, dict):
            continue
        for knob, spec in knobs.items():
            val = _spec_value(spec)
            if val is None:
                continue
            by_knob.setdefault(knob, {})[str(run_id)] = val

    levers: list[Lever] = []
    for knob, run_vals in sorted(by_knob.items()):
        vals = list(run_vals.values())
        if _distinct_count(vals) < _MIN_DISTINCT:
            continue
        numeric = all(_is_number(v) for v in vals)
        if numeric:
            values = {rid: float(v) for rid, v in run_vals.items()}
            levers.append(Lever(knob, "numeric", "metadata", values))
        else:
            values = {rid: str(v) for rid, v in run_vals.items()}
            levers.append(Lever(knob, "categorical", "metadata", values))
    return levers


def _derived_levers(
    obs_df: pd.DataFrame, objective: str, *, skip: set[str]
) -> list[Lever]:
    """Initial (earliest-time) value of each non-objective channel that varies
    across runs. Observational inputs: a measured starting point per run."""
    needed = {"run_id", "variable", "time_h", "value"}
    if not needed.issubset(obs_df.columns):
        return []
    df = obs_df.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["time_h"] = pd.to_numeric(df["time_h"], errors="coerce")
    df = df.dropna(subset=["value"])

    levers: list[Lever] = []
    for var, vdf in sorted(df.groupby("variable")):
        name = str(var)
        if name == objective or name in skip:
            continue
        # Earliest-time value per run = the run's initial condition for `var`.
        initials: dict[str, float] = {}
        for run_id, rdf in vdf.groupby("run_id"):
            rdf = rdf.sort_values("time_h")
            initials[str(run_id)] = float(rdf["value"].iloc[0])
        if _distinct_count(list(initials.values())) < _MIN_DISTINCT:
            continue
        levers.append(Lever(f"{name}.initial", "numeric", "derived", initials))
    return levers


def discover_levers(
    dossier: dict[str, Any] | None,
    obs_df: pd.DataFrame,
    *,
    objective: str = DEFAULT_OBJECTIVE,
) -> list[Lever]:
    """The experiment's actual levers: varying metadata design factors first,
    then varying derived initial conditions (skipping any channel already named
    by a metadata lever). Returns [] when nothing controllable varied."""
    conditions = (dossier or {}).get("run_conditions") or {}
    meta = _metadata_levers(conditions) if isinstance(conditions, dict) else []
    meta_names = {lev.name for lev in meta}
    # A metadata factor and an observation channel can share a name (e.g. a
    # substrate dosed in metadata AND measured); don't double-count it.
    derived = _derived_levers(obs_df, objective, skip=meta_names)
    return meta + derived


# -----------------------------------------------------------------------------
# Design matrix: levers -> numeric features (one-hot for categoricals)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Feature:
    """One numeric column of the design matrix and how to read it back."""

    col: str                         # column name fed to the equation compiler
    lever: str                       # the lever it came from
    kind: Literal["numeric", "onehot"]
    category: str | None = None      # the level encoded, for one-hot columns


@dataclass
class Design:
    """The fitted-over design: rows are runs, columns are lever features."""

    X: np.ndarray                    # (n_runs, n_features)
    y: np.ndarray                    # (n_runs,) objective peak per run
    run_ids: list[str]
    features: list[Feature]
    levers: list[Lever]
    # categorical lever -> the (col -> category) one-hot group, for optimization.
    onehot_groups: dict[str, dict[str, str]] = field(default_factory=dict)

    @property
    def feature_names(self) -> list[str]:
        return [f.col for f in self.features]


def _run_outcomes(obs_df: pd.DataFrame, objective: str) -> dict[str, float]:
    if not {"run_id", "variable", "value"}.issubset(obs_df.columns):
        return {}
    df = obs_df[obs_df["variable"].astype(str) == objective].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    return {str(rid): float(g["value"].max()) for rid, g in df.groupby("run_id")}


def _safe_col(name: str, used: set[str]) -> str:
    """A sympy-safe identifier for a lever/category (compiler rejects others)."""
    base = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name)
    if not base or not (base[0].isalpha() or base[0] == "_"):
        base = "k_" + base
    col = base
    i = 1
    while col in used:
        col = f"{base}_{i}"
        i += 1
    used.add(col)
    return col


def build_design(
    levers: list[Lever], obs_df: pd.DataFrame, *, objective: str = DEFAULT_OBJECTIVE
) -> Design | None:
    """Assemble the per-run design matrix from discovered levers + the run
    outcomes. Numeric levers become one column; categorical levers become one
    0/1 column per observed level (one-hot). Returns None if there are no runs
    with both a full lever row and an outcome."""
    outcomes = _run_outcomes(obs_df, objective)
    if not outcomes or not levers:
        return None

    # Runs usable = have an outcome AND a value for every lever.
    run_ids = sorted(
        rid for rid in outcomes
        if all(rid in lev.values for lev in levers)
    )
    if not run_ids:
        return None

    features: list[Feature] = []
    onehot_groups: dict[str, dict[str, str]] = {}
    used_cols: set[str] = set()
    columns: list[list[float]] = []

    for lev in levers:
        if lev.kind == "numeric":
            col = _safe_col(lev.name, used_cols)
            features.append(Feature(col, lev.name, "numeric"))
            columns.append([float(lev.values[rid]) for rid in run_ids])
        else:
            group: dict[str, str] = {}
            for cat in lev.categories:
                col = _safe_col(f"{lev.name}_{cat}", used_cols)
                features.append(Feature(col, lev.name, "onehot", cat))
                group[col] = cat
                columns.append([1.0 if str(lev.values[rid]) == cat else 0.0
                                for rid in run_ids])
            onehot_groups[lev.name] = group

    X = np.array(columns, dtype=float).T  # (n_runs, n_features)
    y = np.array([outcomes[rid] for rid in run_ids], dtype=float)
    return Design(X=X, y=y, run_ids=run_ids, features=features,
                  levers=levers, onehot_groups=onehot_groups)
