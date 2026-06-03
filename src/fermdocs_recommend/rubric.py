"""Pure model-selection rubric and honest-refusal gates.

No LLM, no brewtwin, no I/O — just decisions over the per-candidate scorecards
that `brewtwin_metrics.build_report` produces. Kept pure so every gate is unit
testable against synthetic `fit_metrics` outputs (tests/recommend/test_rubric.py).

Selection philosophy (corrected after adversarial review):
  * All three families are scored on IDENTICAL held-out footing (same real
    observations, same fit_metrics, same held-out window) — never on training
    loss, which a surrogate/hybrid can drive to ~0 by overfitting.
  * Aggregate by the WORST observed species (min R^2 / max RMSE) so a model
    cannot win by nailing one easy channel.
  * Only species with enough held-out points to support R^2 may drive selection
    (a 4-point plateau gives meaningless R^2). Species that cannot validate are
    recorded, not silently dropped.
  * "Prefer mechanistic" is an interpretability tie-break IN COVERAGE, and an
    extrapolation edge ONLY when the objective requires out-of-range prediction
    — it is never a free RMSE handicap. A surrogate that genuinely beats
    mechanistic on held-out data within range wins.
  * When nothing clears the good-fit gate, the honest answer is a refusal, not
    an invented recommendation.

Thresholds: only R^2 > 0.95 ("good fit") and the +/-5-10% measurement floor are
doc-authoritative (analyze-and-interpret SKILL.md). Everything else is a named,
tunable convention flagged below.
"""

from __future__ import annotations

import math
from typing import Any

# --- Fit-quality thresholds -------------------------------------------------
# The skill doc cites R^2 > 0.95 as "good"; lowered to 0.75 by operator decision
# to accept moderately-good fits as recommendable on noisy bioprocess data.
GOOD_FIT_R2 = 0.75
MEASUREMENT_FLOOR_FRAC = 0.10  # skill: assay error +/-5-10%; use the looser 10%

# --- Tunable conventions (NOT doc-grounded; adjust with domain input) -------
MARGINAL_R2 = 0.50            # below this = "poor"; 0.50-0.75 = "marginal"
RMSE_FLOOR_MULTIPLIER = 2.0   # good-fit also needs RMSE <= 2x measurement floor
MIN_HELDOUT_POINTS = 4        # a species needs >= this many held-out pts to score R^2
MIN_FIT_POINTS = 8            # a species needs >= this many pts to attempt a fit
STALL_REDUCTION_FRAC = 0.05   # loss reduced < 5% over the run => optimizer never moved
DISPLACE_MARGIN_FRAC = 0.10   # surrogate must beat mech RMSE by > this frac to displace

# --- Refusal reason codes ---------------------------------------------------
REFUSAL_NO_BREWTWIN = "brewtwin_not_installed"
REFUSAL_NO_DATA = "insufficient_data"
REFUSAL_POOR_FIT = "poor_fit_all_models"
REFUSAL_IMPLAUSIBLE = "implausible_parameters"
REFUSAL_MECH_UNSUPPORTED = "mechanism_not_supported"
REFUSAL_BUDGET = "compute_budget_exhausted"
REFUSAL_STAGE_ERROR = "stage_error"


def score_candidate(report: dict[str, Any]) -> dict[str, Any]:
    """Reduce a build_report's per-species fit_quality to selection scalars.

    Eligible species = finite R^2 with >= MIN_HELDOUT_POINTS held-out points.
    selection_r2 = min over eligible species; selection_rmse = max over eligible.
    """
    metrics: dict[str, dict[str, float]] = report.get("fit_quality", {})
    eligible: list[str] = []
    ineligible: list[str] = []
    for sp, m in metrics.items():
        r2 = m.get("r2", float("nan"))
        n = int(m.get("n", 0))
        if n >= MIN_HELDOUT_POINTS and r2 is not None and math.isfinite(r2):
            eligible.append(sp)
        else:
            ineligible.append(sp)
    if eligible:
        sel_r2 = min(metrics[sp]["r2"] for sp in eligible)
        sel_rmse = max(metrics[sp]["rmse"] for sp in eligible)
    else:
        sel_r2 = float("nan")
        sel_rmse = float("nan")
    return {
        "selection_r2": sel_r2,
        "selection_rmse": sel_rmse,
        "eligible_species": eligible,
        "ineligible_species": ineligible,
        "loss_reduction_frac": float(report.get("loss_reduction_frac", 0.0)),
    }


def gate_good_fit(
    report: dict[str, Any], *, measurement_floor: dict[str, float] | None = None
) -> tuple[bool, str]:
    """Good iff every eligible species has R^2 > 0.95 and RMSE within the floor.

    `measurement_floor` maps species -> absolute RMSE ceiling. When absent, the
    floor degrades to MEASUREMENT_FLOOR_FRAC * observed-scale, which the caller
    should pass in via the report; without it the RMSE half of the gate is
    skipped (R^2 alone decides) and that is recorded in the reason.
    """
    s = score_candidate(report)
    if not s["eligible_species"]:
        return False, "no species had enough held-out points to validate"
    metrics = report["fit_quality"]
    bad_r2 = [sp for sp in s["eligible_species"] if metrics[sp]["r2"] <= GOOD_FIT_R2]
    if bad_r2:
        worst = min(metrics[sp]["r2"] for sp in bad_r2)
        return False, f"R2<={GOOD_FIT_R2:g} on {bad_r2} (worst {worst:.3f})"
    if measurement_floor:
        bad_rmse = [
            sp
            for sp in s["eligible_species"]
            if sp in measurement_floor
            and metrics[sp]["rmse"] > RMSE_FLOOR_MULTIPLIER * measurement_floor[sp]
        ]
        if bad_rmse:
            return False, f"RMSE exceeds {RMSE_FLOOR_MULTIPLIER}x measurement floor on {bad_rmse}"
        return True, f"R2>{GOOD_FIT_R2:g} and RMSE within floor on all eligible species"
    return True, f"R2>{GOOD_FIT_R2:g} on all eligible species (RMSE floor not supplied)"


def gate_plausible(report: dict[str, Any]) -> tuple[bool, list[str]]:
    """Mechanistic/hybrid only. Any known param out of range, or an unknown
    param name, demotes the mechanism. Returns (plausible, offending_names)."""
    plaus: dict[str, dict[str, Any]] = report.get("fitted_parameters", {})
    if not plaus:
        return True, []  # surrogate: no params to check
    offending = [name for name, info in plaus.items() if not info.get("plausible", False)]
    return (len(offending) == 0), offending


def is_stalled(report: dict[str, Any]) -> bool:
    """Optimizer never moved => data does not constrain the params (G3/G5)."""
    return float(report.get("loss_reduction_frac", 0.0)) < STALL_REDUCTION_FRAC


def select(
    candidates: list[dict[str, Any]],
    *,
    objective_in_coverage: bool = True,
    measurement_floor: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Choose the best family or refuse.

    `candidates` = list of normalized dicts:
        {"model_type": "mechanistic"|"surrogate"|"hybrid",
         "attempted": bool, "disqualified": bool,
         "disqualification_reason": str|None, "report": <build_report dict>}

    `objective_in_coverage` True when the user's counterfactual stays within the
    observed envelope (interpretability tie-break); False when it extrapolates
    (mechanistic gets its documented edge).

    Returns a verdict dict consumed by the agent / schema:
        {recommended_model, confident, refusal_reason, selection_rationale,
         scored: {model_type: {...gate results...}}}
    """
    scored: dict[str, Any] = {}
    passing: dict[str, dict[str, Any]] = {}

    for c in candidates:
        mt = c["model_type"]
        if not c.get("attempted") or c.get("disqualified"):
            scored[mt] = {
                "attempted": bool(c.get("attempted")),
                "disqualified": bool(c.get("disqualified")),
                "reason": c.get("disqualification_reason"),
            }
            continue
        report = c["report"]
        s = score_candidate(report)
        good, good_reason = gate_good_fit(report, measurement_floor=measurement_floor)
        plausible, offending = gate_plausible(report)
        stalled = is_stalled(report)
        scored[mt] = {
            "attempted": True,
            "disqualified": False,
            "selection_r2": s["selection_r2"],
            "selection_rmse": s["selection_rmse"],
            "good_fit": good,
            "good_fit_reason": good_reason,
            "plausible": plausible,
            "offending_params": offending,
            "stalled": stalled,
            "eligible_species": s["eligible_species"],
        }
        # A family "passes" if it clears good-fit, did not stall, and (for
        # mechanistic/hybrid) is plausible. Surrogate skips plausibility.
        if good and not stalled and (mt == "surrogate" or plausible):
            passing[mt] = scored[mt]

    if not passing:
        return _refuse(scored)

    # Selection, prefer-mechanistic. ---------------------------------------
    if "mechanistic" in passing:
        mech = passing["mechanistic"]
        challenger = _best_non_mech(passing)
        if challenger is None:
            return _confident("mechanistic", scored,
                              "mechanistic cleared good-fit + plausibility; no challenger passed")
        cmt, cinfo = challenger
        if objective_in_coverage:
            # interpretability tie-break: mechanistic wins unless challenger is
            # clearly better on held-out RMSE.
            if _beats(cinfo, mech):
                return _confident(cmt, scored,
                                  f"{cmt} beat mechanistic on held-out RMSE by >"
                                  f"{int(DISPLACE_MARGIN_FRAC*100)}% within coverage")
            return _confident("mechanistic", scored,
                              "mechanistic preferred (interpretable; challenger not clearly better)")
        # extrapolation in scope: mechanistic keeps its documented edge.
        return _confident("mechanistic", scored,
                          "mechanistic preferred for out-of-coverage objective (extrapolates better)")

    # No mechanistic: pick best passing surrogate, else hybrid.
    if "surrogate" in passing:
        return _confident("surrogate", scored,
                          "surrogate cleared good-fit; mechanistic did not — valid only in training distribution")
    return _confident("hybrid", scored,
                      "hybrid cleared good-fit; mechanistic did not — ML residual is a black-box correction, not a mechanism")


# --- internals --------------------------------------------------------------
def _best_non_mech(passing: dict[str, dict[str, Any]]) -> tuple[str, dict] | None:
    cands = [(mt, info) for mt, info in passing.items() if mt != "mechanistic"]
    if not cands:
        return None
    return min(cands, key=lambda kv: kv[1]["selection_rmse"])


def _beats(challenger: dict[str, Any], mech: dict[str, Any]) -> bool:
    m = mech["selection_rmse"]
    c = challenger["selection_rmse"]
    if not (math.isfinite(m) and math.isfinite(c)) or m <= 0:
        return False
    return (m - c) / m > DISPLACE_MARGIN_FRAC


def _confident(model: str, scored: dict[str, Any], rationale: str) -> dict[str, Any]:
    return {
        "recommended_model": model,
        "confident": True,
        "refusal_reason": None,
        "selection_rationale": rationale,
        "scored": scored,
    }


def _refuse(scored: dict[str, Any]) -> dict[str, Any]:
    """Pick the most informative refusal code from why each family failed."""
    attempted = {mt: s for mt, s in scored.items() if s.get("attempted") and not s.get("disqualified")}
    if not attempted:
        reason = REFUSAL_NO_DATA
        rationale = "no model family could be fit (disqualified or not attempted)"
    elif all(s.get("stalled") for s in attempted.values()):
        reason = REFUSAL_NO_DATA
        rationale = "optimizer did not move on any family — data does not constrain the parameters"
    elif any(s.get("good_fit") and not s.get("plausible") for s in attempted.values()):
        reason = REFUSAL_IMPLAUSIBLE
        rationale = "the only good fits had implausible parameters; no plausible alternative cleared the gate"
    elif all(not s.get("eligible_species") for s in attempted.values()):
        reason = REFUSAL_NO_DATA
        rationale = "no species had enough held-out points to validate any model"
    elif any(s.get("good_fit") for s in attempted.values()):
        reason = REFUSAL_MECH_UNSUPPORTED
        rationale = "a family fit the curve but offered no defensible interpretable mechanism"
    else:
        reason = REFUSAL_POOR_FIT
        rationale = f"no family reached R2>{GOOD_FIT_R2:g} on the held-out window"
    return {
        "recommended_model": "none",
        "confident": False,
        "refusal_reason": reason,
        "selection_rationale": rationale,
        "scored": scored,
    }
