"""A2 — deterministic claim gates.

A claim is not trustworthy because it's phrased well; it's trustworthy because it
survives named, deterministic checks against the data. Each gate is a pure function
that returns a verdict with the failing reason named — auditable, testable, not a
prompt. The gates resolve their numbers through the F2 cache (A1 conditioned
estimate, F1 clampedness, cross-run effects), so a claim can't smuggle in a
hand-copied number — the gate recomputes from the canonical source.

Gates built here (inputs already exist): confound, objective, materiality.
Deferred (need claim-text/per-unit structure): direction, laundering — see plan A2.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from fermdocs.analysis import cross_run
from fermdocs.analysis.clampedness import detect_clamp
from fermdocs.analysis.cross_run import WEAK_EFFECT_FRAC


@dataclass(frozen=True)
class GatedClaim:
    """The structured facts a gate checks. Built from a hypothesis/claim, NOT
    trusted to carry its own numbers — gates recompute via the cache."""

    assertion: str
    claim_type: str                  # "observational" | "causal" | "recommendation"
    lever: str | None = None         # the cited design factor (run_conditions knob)
    objective_channel: str | None = None
    conditioning: list[str] = field(default_factory=list)  # covariates to hold constant
    corroborating_metric: str | None = None  # a channel cited as supporting evidence
    support_text: str = ""           # the claim's stated supporting argument (for laundering)


@dataclass(frozen=True)
class GateVerdict:
    gate: str
    passed: bool
    reason: str
    severity: str = "fail"           # "fail" | "downgrade" | "info"


_CAUSAL = {"causal", "recommendation"}


def confound_gate(
    claim: GatedClaim, dossier: dict[str, Any] | None, obs_df: pd.DataFrame
) -> GateVerdict | None:
    """A causal/recommendation claim must hold its confounders constant and its
    effect must survive that conditioning. N/A (None) for observational claims."""
    if claim.claim_type not in _CAUSAL or not claim.lever or not claim.objective_channel:
        return None
    if not claim.conditioning:
        # No covariates available to hold constant (e.g. no structured target on
        # this bundle) — you can't be confounded by a covariate that doesn't exist.
        # N/A rather than fail, so the gate doesn't punish claims on bundles that
        # have nothing to condition on (the diagnostic path / pre-B1').
        return None
    est = cross_run.lever_effect_conditioned(
        dossier, obs_df, claim.lever,
        objective=claim.objective_channel, conditioning=claim.conditioning)
    if est is None:
        return GateVerdict("confound", False,
                           f"could not estimate '{claim.lever}' conditioned on "
                           f"{claim.conditioning} (insufficient data).")
    if not est["separable"]:
        return GateVerdict("confound", False,
                           f"'{claim.lever}' is not separable from "
                           f"{est['confounded_with']}: {est['separability_note']}")
    if est["power"] == "insufficient":
        return GateVerdict("confound", False,
                           f"'{claim.lever}' effect underpowered after conditioning: "
                           f"{est['power_note']}", severity="downgrade")
    if est["pooled_norm_effect"] < WEAK_EFFECT_FRAC:
        return GateVerdict("confound", False,
                           f"'{claim.lever}' effect attenuates to "
                           f"{est['pooled_norm_effect']} (< {WEAK_EFFECT_FRAC}) once "
                           f"conditioned on {claim.conditioning}: the apparent effect "
                           "was the confounder.")
    return GateVerdict("confound", True,
                       f"'{claim.lever}' survives conditioning on {claim.conditioning} "
                       f"(norm effect {est['pooled_norm_effect']}).")


def objective_gate(
    claim: GatedClaim, obs_df: pd.DataFrame, strata: dict[str, object] | None
) -> GateVerdict | None:
    """A recommendation must move a FREE variable, not one clamped by design.
    N/A for non-recommendations or when there's no objective channel."""
    if claim.claim_type != "recommendation" or not claim.objective_channel:
        return None
    if not strata:
        return GateVerdict("objective", True,
                           "no strata to judge clampedness; objective accepted as-is",
                           severity="info")
    info = detect_clamp(obs_df, strata, channels=[claim.objective_channel]).get(
        claim.objective_channel)
    if info is not None and info.clamped:
        return GateVerdict("objective", False,
                           f"recommendation optimizes '{claim.objective_channel}', which "
                           f"is clamped ({info.reason}) — moving a quantity set by design "
                           "is no recommendation; optimize the free variable.")
    return GateVerdict("objective", True,
                       f"'{claim.objective_channel}' is free to move.")


def materiality_gate(
    claim: GatedClaim, dossier: dict[str, Any] | None, obs_df: pd.DataFrame
) -> GateVerdict | None:
    """A 'primary lever' claim whose effect is smaller than a competing lever's is
    downgraded. N/A for observational claims or when there's no lever."""
    if claim.claim_type not in _CAUSAL or not claim.lever or not claim.objective_channel:
        return None
    effects = cross_run.lever_effects(dossier, obs_df, objective=claim.objective_channel)
    if claim.lever not in effects:
        return None
    mine = float(effects[claim.lever].get("norm_effect") or 0.0)
    bigger = {k: float(v.get("norm_effect") or 0.0) for k, v in effects.items()
              if k != claim.lever and float(v.get("norm_effect") or 0.0) > mine}
    if bigger:
        top = max(bigger, key=bigger.get)
        return GateVerdict("materiality", False,
                           f"'{claim.lever}' (effect {round(mine, 3)}) is not the primary "
                           f"lever — '{top}' has a larger effect ({round(bigger[top], 3)}).",
                           severity="downgrade")
    return GateVerdict("materiality", True,
                       f"'{claim.lever}' is the largest-effect lever ({round(mine, 3)}).")


def direction_gate(
    claim: GatedClaim, dossier: dict[str, Any] | None, obs_df: pd.DataFrame
) -> GateVerdict | None:
    """A cited corroborating metric must move with the lever in a direction
    CONSISTENT with the objective — else the convergence is manufactured (the mu
    argument: 'growth supports the nutrient claim' while growth moved the opposite
    way per the data). N/A unless a numeric corroborating metric is cited."""
    if not claim.corroborating_metric or not claim.lever or not claim.objective_channel:
        return None
    eff_obj = cross_run.lever_effects(
        dossier, obs_df, objective=claim.objective_channel).get(claim.lever)
    eff_met = cross_run.lever_effects(
        dossier, obs_df, objective=claim.corroborating_metric).get(claim.lever)
    if not eff_obj or not eff_met:
        return None
    d_obj, d_met = eff_obj.get("direction"), eff_met.get("direction")
    # Only numeric directions carry a sign to compare; categorical ("set_to") skip.
    if d_obj not in ("increase", "decrease") or d_met not in ("increase", "decrease"):
        return None
    if d_obj != d_met:
        return GateVerdict(
            "direction", False,
            f"cited metric '{claim.corroborating_metric}' moves {d_met} with "
            f"'{claim.lever}' while the objective moves {d_obj} — the corroboration is "
            "manufactured (they point opposite ways per unit).")
    return GateVerdict("direction", True,
                       f"'{claim.corroborating_metric}' moves {d_met}, consistent with "
                       "the objective.")


# Negative-evidence patterns: support that's really "we ruled out something".
_LAUNDER_RE = re.compile(
    r"\b(rul(?:e|ed|ing)\s+out|not\s+(?:an?\s+)?[\w-]+\s+(?:limitation|bottleneck|"
    r"cause|constraint)|rather than|is not the|no evidence (?:of|for))\b", re.I)


def laundering_gate(
    claim: GatedClaim, proposed_levers: set[str] | None = None
) -> GateVerdict | None:
    """Flag a claim whose support launders NEGATIVE evidence (ruling out a factor
    nobody proposed) as positive support — the DO clause in every bad hypothesis.
    Heuristic, so a DOWNGRADE not a hard block: fires when the support contains a
    'ruled-out' pattern. (The LLM critic refines; this just denies it free credit.)"""
    if not claim.support_text:
        return None
    if _LAUNDER_RE.search(claim.support_text):
        return GateVerdict(
            "laundering", False,
            "claim support leans on ruling out a factor (negative evidence) rather "
            "than positive support for its own lever — not counted as corroboration.",
            severity="downgrade")
    return None


def run_gates(
    claim: GatedClaim,
    dossier: dict[str, Any] | None,
    obs_df: pd.DataFrame,
    strata: dict[str, object] | None = None,
    proposed_levers: set[str] | None = None,
) -> list[GateVerdict]:
    """Run all applicable gates; return only the verdicts that applied (non-None).
    A claim is BLOCKED iff any verdict is a hard 'fail' (see claim_blocked)."""
    verdicts = [
        confound_gate(claim, dossier, obs_df),
        objective_gate(claim, obs_df, strata),
        materiality_gate(claim, dossier, obs_df),
        direction_gate(claim, dossier, obs_df),
        laundering_gate(claim, proposed_levers),
    ]
    return [v for v in verdicts if v is not None]


def claim_blocked(verdicts: list[GateVerdict]) -> bool:
    """A hard block: any failing gate with severity 'fail'. Downgrades don't block
    but lower the claim's standing (the Judge weights them)."""
    return any((not v.passed) and v.severity == "fail" for v in verdicts)
