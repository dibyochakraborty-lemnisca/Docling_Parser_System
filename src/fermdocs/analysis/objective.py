"""Resolve the optimization objective channel for a bundle — from the data, not
a hardcoded constant.

Priority (de-LABS decision, 2026-06-16):
  1. The user's explicit target — a channel named in `user_question.affected_variables`
     that is actually measured in this bundle. The user asking "how do I raise
     ethanol?" makes ethanol_g_l the objective, not the default product channel.
  2. The golden schema's designated objective channel (`GoldenColumn.objective`),
     IF it is present among the measured channels. This is schema-derived: the
     golden schema marks which canonical channel is the product/objective.
  3. None — refuse. The caller (debate / data optimizer) handles a missing
     objective honestly instead of optimizing an assumed channel.

No baked-in `product_g_l` string lives here; the default comes from the schema.
"""
from __future__ import annotations

from collections.abc import Iterable


def _user_target(user_question, channels: set[str]) -> str | None:
    """The first user-named affected variable that is a measured channel, if any."""
    if user_question is None:
        return None
    affected = getattr(user_question, "affected_variables", None) or []
    for var in affected:
        if var in channels:
            return var
    return None


def resolve_objective(
    channels: Iterable[str],
    *,
    user_question=None,
    schema_path: str | None = None,
) -> str | None:
    """Resolve which measured channel the optimizer should maximize.

    `channels` is the set of variable names actually present in the bundle's
    observations. Returns the resolved objective channel name, or None when none
    can be resolved (caller must refuse rather than guess)."""
    chans = {str(c) for c in channels}
    if not chans:
        return None

    user = _user_target(user_question, chans)
    if user is not None:
        return user

    from fermdocs.domain.golden_schema import cached_schema

    try:
        designated = cached_schema(schema_path).objective_channel()
    except Exception:  # noqa: BLE001 — schema unreadable → fall through to refuse
        designated = None
    if designated and designated in chans:
        return designated
    return None


# -----------------------------------------------------------------------------
# F1 — free-variable objective resolution
# -----------------------------------------------------------------------------

from dataclasses import dataclass  # noqa: E402


@dataclass(frozen=True)
class Objective:
    """What the optimizer maximizes. Either a measured channel's peak, or a derived
    rate (e.g. productivity = peak/time-to-peak) when the channel itself is clamped."""

    name: str            # display name, e.g. "product_g_l" or "product_g_l_per_h"
    kind: str            # "channel" | "rate"
    base_channel: str    # the measured channel it derives from
    clamped_base: bool = False  # was the base channel detected as clamped?
    reason: str = ""

    def outcome_per_run(self, obs_df) -> dict[str, float]:
        """Per-run scalar outcome for this objective. channel -> peak; rate ->
        peak / time-to-peak (the free 'how fast did we get there' quantity)."""
        import pandas as pd
        if not {"run_id", "variable", "value"}.issubset(obs_df.columns):
            return {}
        df = obs_df[obs_df["variable"].astype(str) == self.base_channel].copy()
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        if df.empty:
            return {}
        out: dict[str, float] = {}
        for run, g in df.groupby("run_id"):
            peak = float(g["value"].max())
            if self.kind == "channel":
                out[str(run)] = peak
                continue
            if "time_h" not in g.columns:
                continue
            gg = g.copy()
            gg["time_h"] = pd.to_numeric(gg["time_h"], errors="coerce")
            t_peak = gg.loc[gg["value"].idxmax(), "time_h"]
            if pd.notna(t_peak) and float(t_peak) > 0:
                out[str(run)] = peak / float(t_peak)
        return out


def resolve_objective_free(
    obs_df,
    *,
    user_question=None,
    strata: dict[str, object] | None = None,
    schema_path: str | None = None,
) -> Objective | None:
    """Resolve the objective, auto-preferring a FREE variable over a clamped one.

    Order (de-LABS + F1):
      1. user override wins — a user-named measured channel is the objective as a
         channel (even if clamped; the user gets what they asked for, with the
         clamp noted on the descriptor).
      2. else the schema objective channel. If it is detected CLAMPED (set by the
         stratum, e.g. titer pinned to a campaign target) AND a derived rate is
         computable (the channel has a time axis), prefer the rate — the free
         quantity. Otherwise use the channel.
      3. None — refuse.

    `strata` (run_id -> label, e.g. target campaign) is REQUIRED to detect a clamp;
    without it the channel is used as-is (can't tell clamped from free). This is the
    F1<-B1' dependency made explicit, not hidden."""
    import pandas as pd
    chans = (set(obs_df["variable"].astype(str).unique())
             if isinstance(obs_df, pd.DataFrame) and "variable" in obs_df else set())
    if not chans:
        return None

    user = _user_target(user_question, chans)
    if user is not None:
        return Objective(name=user, kind="channel", base_channel=user,
                         reason="user-specified objective channel")

    from fermdocs.domain.golden_schema import cached_schema
    try:
        designated = cached_schema(schema_path).objective_channel()
    except Exception:  # noqa: BLE001
        designated = None
    if not designated or designated not in chans:
        return None

    if not strata:
        return Objective(name=designated, kind="channel", base_channel=designated,
                         reason="no strata to judge clampedness; using channel as-is")

    from fermdocs.analysis.clampedness import detect_clamp
    info = detect_clamp(obs_df, strata, channels=[designated]).get(designated)
    if info is None or not info.clamped:
        return Objective(name=designated, kind="channel", base_channel=designated,
                         clamped_base=False,
                         reason=(info.reason if info else "channel objective"))

    # Channel is clamped — prefer the free rate if the channel has a time axis.
    rate = Objective(name=f"{designated}_per_h", kind="rate", base_channel=designated,
                     clamped_base=True,
                     reason=(f"{designated} is clamped ({info.reason}); optimizing the "
                             "free rate (peak per time-to-peak) instead"))
    if rate.outcome_per_run(obs_df):
        return rate
    # Clamped but no usable time axis -> fall back to the channel, flagged clamped.
    return Objective(name=designated, kind="channel", base_channel=designated,
                     clamped_base=True,
                     reason=(f"{designated} is clamped but no time axis for a rate; "
                             "objective remains the clamped channel (flagged)"))
