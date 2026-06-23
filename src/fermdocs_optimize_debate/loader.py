"""Load a bundle into the optimization-debate input.

Mirrors `fermdocs_hypothesis.bundle_loader` but:
  - seeds OPPORTUNITY topics (topics.py) from characterization, not faults;
  - treats diagnosis as OPTIONAL — the optimizer runs on healthy bundles that
    were never diagnosed. When a prior diagnosis exists we still load it for the
    analyses pool, but we never require it and never seed topics from it.

Reuses the hypothesis loader's pool builders (findings/narratives/trajectories/
priors) unchanged — they read characterization, which is narrative-agnostic.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path

from fermdocs.bundle.reader import BundleReader
from fermdocs_characterize.schema import CharacterizationOutput
from fermdocs_diagnose.schema import DiagnosisOutput
from fermdocs_hypothesis.bundle_loader import (
    _build_analyses_pool,
    _build_findings_pool,
    _build_narratives_pool,
    _build_priors_pool,
    _build_trajectories_pool,
    _extract_organism_and_family,
    _load_user_question,
)
from fermdocs_hypothesis.schema import (
    AnalysisRef,
    FindingRef,
    HypothesisInput,
    NarrativeRef,
    ResolvedPriorRef,
    TrajectoryViewRef,
)

from fermdocs_optimize_debate.topics import extract_opportunity_topics


def _golden_objective_label() -> str:
    """The golden-schema designated objective channel name, as a display/label
    fallback when the bundle's own objective can't be resolved. Schema-derived,
    not a hardcoded domain value."""
    try:
        from fermdocs.domain.golden_schema import cached_schema
        return cached_schema().objective_channel() or "product_g_l"
    except Exception:  # noqa: BLE001 — schema unreadable → canonical product channel
        return "product_g_l"


def _resolve_bundle_objective(reader, user_question) -> str | None:
    """Resolve the objective channel from THIS bundle's measured channels (+ the
    user's question), not a fixed species. None when nothing resolves."""
    try:
        import pandas as pd

        from fermdocs.analysis.objective import resolve_objective

        obs_path = reader.dir / "characterization" / "observations.csv"
        if not obs_path.exists():
            return None
        obs = pd.read_csv(obs_path)
        channels = (set(obs["variable"].astype(str).unique())
                    if "variable" in obs.columns else set())
        return resolve_objective(channels, user_question=user_question)
    except Exception:  # noqa: BLE001 — best-effort; caller falls back to schema label
        return None


def _derive_strata(dossier) -> tuple[dict, list[str]]:
    """The campaign/target stratum for clampedness + conditioning (B2/F1). Shared
    with the gate bridge (A3) — see fermdocs.analysis.clampedness.derive_strata."""
    from fermdocs.analysis.clampedness import derive_strata
    return derive_strata(dossier)


def _discover_bundle_levers(reader, dossier, user_question, *, fallback_objective: str):
    """The experiment's own levers + their RANKING effect on the FREE objective.

    Returns ``(levers, effects, objective_name)``:
      - ``levers``: discovered levers ([] = ran-none -> trends; None = couldn't run).
      - ``effects``: {lever: {delta, direction, norm_effect (observational),
        ranking_effect (B2: conditioned on the target stratum when one exists,
        else == norm_effect), confounded, ...}}.
      - ``objective_name``: the resolved objective (F1) — a free rate when the
        schema objective is clamped, else the channel; fallback when unresolved.
    Never raises."""
    try:
        import pandas as pd

        from fermdocs.analysis.cross_run import lever_effects, ranking_effects
        from fermdocs.analysis.objective import resolve_objective_free
        from fermdocs_optimize.lever_discovery import discover_levers

        obs_path = reader.dir / "characterization" / "observations.csv"
        if not obs_path.exists():
            return None, {}, fallback_objective
        obs = pd.read_csv(obs_path)

        strata, conditioning = _derive_strata(dossier)
        obj = resolve_objective_free(obs, user_question=user_question, strata=strata or None)
        if obj is None:
            objective_name, base_channel, outcomes = fallback_objective, fallback_objective, None
        else:
            objective_name, base_channel = obj.name, obj.base_channel
            outcomes = obj.outcome_per_run(obs) or None

        levers = discover_levers(dossier, obs, objective=base_channel)
        effects = lever_effects(dossier, obs, objective=objective_name, outcomes=outcomes)
        # B2: redirect the ranking signal to the conditioned effect on the free
        # objective. Confounded/underpowered levers score 0 (sink); real ones keep
        # their magnitude. With no target stratum this == the unconditioned effect.
        ranking = ranking_effects(dossier, obs, objective=objective_name,
                                  outcomes=outcomes, conditioning=conditioning or None)
        for lev, e in effects.items():
            base = float(e.get("norm_effect") or 0.0)
            e["ranking_effect"] = float(ranking.get(lev, base))
        return levers, effects, objective_name
    except Exception:  # noqa: BLE001 — discovery is best-effort; fall back to static
        return None, {}, fallback_objective


def _build_within_run_pool(lever_effects: dict, objective: str) -> list:
    """Turn the per-lever cross-run effects into citeable WithinRunAssociationRefs
    the specialists see in their view. Sorted by effect size (strongest first)."""
    from fermdocs.analysis.cross_run import is_weak_effect
    from fermdocs_hypothesis.schema import WithinRunAssociationRef

    pool: list[WithinRunAssociationRef] = []
    for lever, eff in (lever_effects or {}).items():
        delta = eff.get("delta")
        if delta is None:
            continue
        n = int(eff.get("n", 0))
        best = eff.get("best_setting")
        if eff.get("confounded"):
            strength = (f" CONFOUNDED ({eff.get('confounded_with')}): the effect is not "
                        "separable / not attributable — do NOT act on this lever alone.")
        elif is_weak_effect(eff.get("norm_effect")):
            strength = (" WEAK/preliminary: the effect is a small fraction of the "
                        "run-to-run scatter, likely within noise on a confounded set — "
                        "treat as a lead to test, NOT a validated effect.")
        else:
            strength = ""
        summary = (f"Across {n} runs, {lever} associates with {delta:+g} peak titer "
                   f"(best observed: {best}); observational, not proven causal.{strength}")
        pool.append(WithinRunAssociationRef(
            assoc_id=f"WRA-{lever}", lever=lever, summary=summary,
            delta=float(delta), direction=str(eff.get("direction", "")), n=n,
            norm_effect=float(eff.get("norm_effect") or 0.0),
            objective=objective, best_setting=best))
    pool.sort(key=lambda a: a.norm_effect, reverse=True)
    return pool


@dataclass
class OptimizeLoadedBundle:
    """Everything the optimization debate needs from a bundle."""

    hyp_input: HypothesisInput
    diagnosis: DiagnosisOutput | None  # None when the run was never diagnosed
    diagnosis_id: uuid.UUID            # synthesized when no diagnosis exists
    characterization: CharacterizationOutput
    findings_pool: list[FindingRef]
    narratives_pool: list[NarrativeRef]
    trajectories_pool: list[TrajectoryViewRef]
    priors_pool: list[ResolvedPriorRef]
    analyses_pool: list[AnalysisRef]
    bundle_dir: Path
    # Within-experiment lever->objective associations (opportunity debate). Read
    # by LiveHooks.contribute_facet via getattr; the hypothesis stage has none.
    within_run_pool: list = field(default_factory=list)


def load_optimization_bundle(
    bundle_dir: str | Path,
    *,
    objective: str | None = None,
    max_trend_topics: int = 8,
) -> OptimizeLoadedBundle:
    reader = BundleReader(bundle_dir)

    char = CharacterizationOutput.model_validate_json(reader.get_characterization_json())
    if reader.has_narrative_observations() and not char.narrative_observations:
        import json
        char_dict = json.loads(reader.get_characterization_json())
        char_dict["narrative_observations"] = json.loads(reader.get_narrative_observations_json())
        char = CharacterizationOutput.model_validate(char_dict)

    # Diagnosis is optional: load it only for the analyses pool if present.
    diagnosis: DiagnosisOutput | None = None
    if reader.has_diagnosis():
        try:
            diagnosis = DiagnosisOutput.model_validate_json(reader.get_diagnosis_json())
        except Exception:  # noqa: BLE001 — a malformed diagnosis never blocks optimization
            diagnosis = None
    diagnosis_id = diagnosis.meta.diagnosis_id if diagnosis is not None else uuid.uuid4()

    dossier = reader.get_dossier()
    organism, process_family = _extract_organism_and_family(dossier)
    user_question = _load_user_question(reader.dir)

    # Resolve the objective from THIS bundle's data (+ the user's question), not a
    # fixed LABS species. Caller override wins; else resolve from measured channels;
    # else fall back to the golden-schema objective label (de-LABS, 2026-06-16).
    fallback_objective = objective or _resolve_bundle_objective(reader, user_question) \
        or _golden_objective_label()

    # Discover THIS experiment's own levers (run_conditions metadata + varying
    # observation channels) so the debate argues the real levers. There is no LABS
    # knob fallback (de-LABS): no levers -> trends carry the debate. B2/F1: the
    # resolved objective is the FREE variable (productivity when titer is clamped),
    # and lever ranking uses the conditioned effect on it.
    discovered, lever_effects, objective = _discover_bundle_levers(
        reader, dossier, user_question, fallback_objective=fallback_objective)
    seed_topics = extract_opportunity_topics(
        char, objective_species=objective,
        discovered_levers=discovered, lever_effects=lever_effects,
        max_trend_topics=max_trend_topics)

    hyp_input = HypothesisInput(
        diagnosis=diagnosis,  # Any; None is fine — nothing seeds from it
        characterization=char,
        bundle_path=str(reader.dir),
        seed_topics=seed_topics,
        organism=organism,
        process_family=process_family,
        user_question=user_question,
    )
    return OptimizeLoadedBundle(
        hyp_input=hyp_input,
        diagnosis=diagnosis,
        diagnosis_id=diagnosis_id,
        characterization=char,
        findings_pool=_build_findings_pool(char),
        narratives_pool=_build_narratives_pool(char),
        trajectories_pool=_build_trajectories_pool(char),
        priors_pool=_build_priors_pool(organism, process_family),
        analyses_pool=_build_analyses_pool(diagnosis) if diagnosis is not None else [],
        bundle_dir=Path(bundle_dir),
        within_run_pool=_build_within_run_pool(lever_effects, objective),
    )
