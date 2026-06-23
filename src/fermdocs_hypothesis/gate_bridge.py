"""A3 — bridge the deterministic gates (fermdocs.analysis.gates) onto debated
hypotheses, so a claim that fails a hard gate cannot enter the verdict no matter
how fluently it was phrased.

The bridge is pure: it maps a FinalHypothesis to a GatedClaim (resolving the lever
from its cited within-run association, never trusting a copied number), runs the
gates against the bundle data, attaches the failing-gate names, and partitions the
hypotheses into allowed vs blocked. Wiring this into the live judge/validators loop
is a thin call to `apply_gates`; the conditioning covariates come from the
structured strata (target), which B1' supplies — until then `conditioning` is empty
and the confound gate only enforces the 'causal claim must name its covariates'
rule, degrading gracefully.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from fermdocs.analysis.gates import GatedClaim, claim_blocked, run_gates


def bundle_context(bundle_dir: Any) -> dict[str, Any]:
    """Best-effort gate context for a bundle: dossier, observations, the resolved
    free objective, and the target stratum to condition on.

    Returns the kwargs for ``validate_hypothesis_output``'s gate pass, but ONLY when
    a target stratum is structured (B1') — otherwise ``{}`` so the gates stay off
    (the diagnostic path and pre-B1' optimize runs are unaffected). Never raises."""
    try:
        from pathlib import Path

        import pandas as pd

        from fermdocs.analysis.clampedness import derive_strata
        from fermdocs.analysis.objective import resolve_objective_free
        from fermdocs.bundle import BundleReader

        obs_path = Path(bundle_dir) / "characterization" / "observations.csv"
        if not obs_path.exists():
            return {}
        obs = pd.read_csv(obs_path)
        try:
            dossier = BundleReader(bundle_dir).get_dossier()
        except Exception:  # noqa: BLE001
            dossier = None
        strata, conditioning = derive_strata(dossier)
        if not strata:
            return {}  # no target stratum yet (B1') -> gates stay off (graceful)
        obj = resolve_objective_free(obs, strata=strata)
        return {
            "dossier": dossier,
            "obs_df": obs,
            "objective_channel": obj.base_channel if obj else None,
            "conditioning": conditioning,
            "strata": strata,
        }
    except Exception:  # noqa: BLE001 — gate context is best-effort; never break the run
        return {}


def _lever_of(hyp: Any) -> str | None:
    """The design factor a hypothesis is about: the first cited within-run
    association (WRA-<lever> -> <lever>). None when it cites no association."""
    assoc = getattr(hyp, "cited_association_ids", None) or []
    if isinstance(hyp, dict):
        assoc = hyp.get("cited_association_ids") or []
    for a in assoc:
        return a[4:] if str(a).startswith("WRA-") else str(a)
    return None


def gated_claim_from_hypothesis(
    hyp: Any, *, objective_channel: str | None, conditioning: list[str],
) -> GatedClaim:
    """Project a (final) hypothesis onto the structured facts the gates check."""
    def g(attr, default=None):
        return hyp.get(attr, default) if isinstance(hyp, dict) else getattr(hyp, attr, default)

    support = " ".join(s for s in [g("summary", ""), g("actionable_recommendation", "") or ""] if s)
    return GatedClaim(
        assertion=g("summary", ""),
        claim_type=g("claim_type", "observational"),
        lever=_lever_of(hyp),
        objective_channel=objective_channel,
        conditioning=list(conditioning),
        support_text=support,
    )


def apply_gates(
    finals: list[Any],
    dossier: dict[str, Any] | None,
    obs_df: pd.DataFrame,
    *,
    objective_channel: str | None,
    conditioning: list[str] | None = None,
    strata: dict[str, object] | None = None,
) -> tuple[list[Any], list[tuple[Any, list]]]:
    """Run the gates over every final hypothesis.

    Returns ``(allowed, blocked)`` where blocked is ``[(hyp, verdicts), ...]`` for
    hypotheses with a hard-fail gate. Each hypothesis gets ``gate_failures`` set to
    the names of the gates it hard-failed (mutated in place when the object supports
    it; dicts get the key set), so the reason travels with the claim into the UI."""
    conditioning = conditioning or []
    proposed = {_lever_of(h) for h in finals if _lever_of(h)}
    allowed: list[Any] = []
    blocked: list[tuple[Any, list]] = []
    for hyp in finals:
        claim = gated_claim_from_hypothesis(
            hyp, objective_channel=objective_channel, conditioning=conditioning)
        verdicts = run_gates(claim, dossier, obs_df, strata, proposed_levers=proposed)
        hard_fails = [v.gate for v in verdicts if (not v.passed) and v.severity == "fail"]
        if isinstance(hyp, dict):
            hyp["gate_failures"] = hard_fails
        else:
            try:
                object.__setattr__(hyp, "gate_failures", hard_fails)
            except Exception:  # noqa: BLE001 — frozen/odd objects: skip attaching
                pass
        if claim_blocked(verdicts):
            blocked.append((hyp, verdicts))
        else:
            allowed.append(hyp)
    return allowed, blocked
