"""Tier A cross-run statistics: cohort-level comparisons.

These functions take a per-run KPI table (one row per run, one column
per metric) and surface where runs disagree. Pure pandas/numpy; no
LLM, no smoothing tricks.

The trajectory analyzer assembles the per-run table by calling Tier A
single-run functions (compute_mu, doubling_time, etc.) and rolling
the results up; cross_run consumes that table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# A19 — Cross-run KPI table assembly
# -----------------------------------------------------------------------------


def cross_run_kpi_table(
    per_run_records: list[dict],
    *,
    run_id_field: str = "run_id",
) -> pd.DataFrame:
    """Build a tidy DataFrame from a list of per-run KPI dicts.

    Each dict is one run's KPIs: {"run_id": "RUN-0001", "mu_max": 0.42, ...}.
    Returns a DataFrame indexed by run_id with one column per KPI.
    NaNs are preserved when a KPI is missing for a given run.
    """
    if not per_run_records:
        return pd.DataFrame()
    df = pd.DataFrame(per_run_records)
    if run_id_field not in df.columns:
        raise ValueError(f"every record must include '{run_id_field}'; got {list(df.columns)}")
    df = df.set_index(run_id_field).sort_index()
    return df


# -----------------------------------------------------------------------------
# A20 — Pairwise deviation matrix
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PairwiseDeviation:
    """Most divergent run pair on a given KPI.

    `relative_gap`: |xi - xj| / mean(xi, xj). 0.5 means they differ by
    half their joint average — that's a big spread for a fermentation
    KPI.
    """

    kpi: str
    run_a: str
    run_b: str
    value_a: float
    value_b: float
    relative_gap: float


def pairwise_deviation(
    kpi_table: pd.DataFrame,
    *,
    top_k: int = 3,
) -> list[PairwiseDeviation]:
    """Find the top-K run pairs with the largest relative gap per KPI.

    Returns a flat list across KPIs, sorted by relative_gap desc.
    Numeric columns only — non-numeric columns are skipped silently.
    Pairs where the joint mean is 0 are excluded (relative_gap undefined).
    """
    if kpi_table.empty or len(kpi_table) < 2:
        return []
    runs = list(kpi_table.index)
    out: list[PairwiseDeviation] = []
    for col in kpi_table.columns:
        vals = pd.to_numeric(kpi_table[col], errors="coerce")
        for i, ra in enumerate(runs):
            for rb in runs[i + 1 :]:
                va, vb = vals.loc[ra], vals.loc[rb]
                if not (np.isfinite(va) and np.isfinite(vb)):
                    continue
                joint_mean = (abs(va) + abs(vb)) / 2.0
                if joint_mean == 0:
                    continue
                gap = abs(va - vb) / joint_mean
                out.append(
                    PairwiseDeviation(
                        kpi=str(col),
                        run_a=str(ra),
                        run_b=str(rb),
                        value_a=float(va),
                        value_b=float(vb),
                        relative_gap=float(gap),
                    )
                )
    out.sort(key=lambda d: d.relative_gap, reverse=True)
    return out[:top_k]


# -----------------------------------------------------------------------------
# A21 — Variance decomposition
# -----------------------------------------------------------------------------


def variance_decomposition(
    kpi_table: pd.DataFrame,
    *,
    grouping_column: str | None = None,
) -> pd.DataFrame:
    """Decompose KPI variance into between-group and within-group components.

    When `grouping_column` is None, treats every run as its own group
    (between-group variance = total variance; within-group = 0). When
    provided, groups runs by that column and computes:

      sigma2_between = variance of group means weighted by group size
      sigma2_within  = mean of (group variance × group size)
      total          = sigma2_between + sigma2_within
      between_frac   = sigma2_between / total

    Returns one row per KPI. KPIs with all-NaN columns are skipped.
    """
    if kpi_table.empty:
        return pd.DataFrame(columns=["kpi", "total_var", "between_var", "within_var", "between_frac"])

    rows = []
    for col in kpi_table.columns:
        if grouping_column is not None and col == grouping_column:
            continue
        vals = pd.to_numeric(kpi_table[col], errors="coerce")
        if vals.dropna().empty:
            continue
        if grouping_column is None:
            total = float(vals.var(ddof=0))
            rows.append(
                {
                    "kpi": col,
                    "total_var": total,
                    "between_var": total,
                    "within_var": 0.0,
                    "between_frac": 1.0 if total > 0 else 0.0,
                }
            )
            continue

        groups = kpi_table[grouping_column]
        df = pd.DataFrame({"value": vals, "group": groups}).dropna()
        if df.empty:
            continue
        group_stats = df.groupby("group")["value"].agg(["mean", "var", "count"])
        # var has ddof=1 by default; switch to ddof=0 for population variance.
        group_stats["var"] = df.groupby("group")["value"].var(ddof=0)
        n_total = float(group_stats["count"].sum())
        if n_total == 0:
            continue
        grand_mean = float((group_stats["mean"] * group_stats["count"]).sum() / n_total)
        between = float(
            (group_stats["count"] * (group_stats["mean"] - grand_mean) ** 2).sum()
            / n_total
        )
        within = float((group_stats["count"] * group_stats["var"]).sum() / n_total)
        total = between + within
        rows.append(
            {
                "kpi": col,
                "total_var": total,
                "between_var": between,
                "within_var": within,
                "between_frac": (between / total) if total > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)
