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


# n below this threshold flags `weak_n` on correlation results. Below 8
# the bootstrap CI tends to be wide and the point estimate is unstable;
# downstream agents must caveat the claim or downgrade to data_gap.
# 8 is the smallest n where Pearson r=0.95 has a non-trivial probability
# of arising under the null with two-tailed p<0.05 — below it, even
# strong-looking correlations carry real chance of being noise.
WEAK_N_THRESHOLD = 8


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
# A25 — Pairwise correlation with bootstrap CI
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class CorrelationResult:
    """Pearson r with non-parametric bootstrap CI.

    `weak_n_flag` is True when n < WEAK_N_THRESHOLD; downstream agents
    must surface that fact in any claim derived from this result.
    `ci_low` / `ci_high` are the empirical 2.5/97.5 percentiles of the
    bootstrap distribution. Returns ci_low=ci_high=r when n<3 (no
    meaningful resampling possible).
    """

    r: float
    n: int
    ci_low: float
    ci_high: float
    weak_n_flag: bool


def compute_correlation(
    x,
    y,
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
    ci_alpha: float = 0.05,
) -> CorrelationResult:
    """Pearson r between paired samples with bootstrap CI.

    Drops paired NaNs before computing. Deterministic given the seed.
    Raises ValueError when fewer than 2 finite paired points remain —
    correlation is undefined.
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if xa.shape != ya.shape:
        raise ValueError(f"x and y must be same shape; got {xa.shape} vs {ya.shape}")
    mask = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[mask], ya[mask]
    n = int(xa.size)
    if n < 2:
        raise ValueError(f"need ≥2 finite paired points; got {n}")

    # Point estimate. np.corrcoef returns NaN for zero-variance input;
    # in that case r is undefined — surface as 0.0 with weak_n_flag.
    if xa.std() == 0 or ya.std() == 0:
        return CorrelationResult(
            r=0.0, n=n, ci_low=0.0, ci_high=0.0, weak_n_flag=True
        )
    r = float(np.corrcoef(xa, ya)[0, 1])

    weak = n < WEAK_N_THRESHOLD
    if n < 3:
        return CorrelationResult(
            r=r, n=n, ci_low=r, ci_high=r, weak_n_flag=weak
        )

    rng = np.random.default_rng(seed)
    boot_rs = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        xb, yb = xa[idx], ya[idx]
        if xb.std() == 0 or yb.std() == 0:
            boot_rs[i] = 0.0
        else:
            boot_rs[i] = np.corrcoef(xb, yb)[0, 1]
    ci_low = float(np.percentile(boot_rs, 100 * ci_alpha / 2))
    ci_high = float(np.percentile(boot_rs, 100 * (1 - ci_alpha / 2)))
    return CorrelationResult(
        r=r, n=n, ci_low=ci_low, ci_high=ci_high, weak_n_flag=weak
    )


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
