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

from fermdocs_optimize_debate.levers import DEFAULT_LEVERS, KnobLever
from fermdocs_optimize_debate.topics import extract_opportunity_topics


def _discover_bundle_levers(reader, dossier) -> tuple[list | None, dict]:
    """The experiment's own levers + their cross-run effect on the objective.

    Returns ``(levers, lever_effects)``:
      - ``levers``: discovered levers (None → caller falls back to static LABS
        levers, i.e. discovery did not run; an empty-after-filter list still
        counts as "ran", handled in topics.py).
      - ``lever_effects``: {lever_name: {delta, direction, n, norm_effect, ...}}
        from the within-run cross-run engine, used to RANK design-factor topics
        by how much they actually moved titer. ``{}`` when there aren't enough
        runs / no objective (topics then fall back to neutral ordering).
    Never raises."""
    try:
        import pandas as pd

        from fermdocs.analysis.cross_run import lever_effects
        from fermdocs_optimize.lever_discovery import discover_levers

        obs_path = reader.dir / "characterization" / "observations.csv"
        if not obs_path.exists():
            return None, {}
        obs = pd.read_csv(obs_path)
        levers = discover_levers(dossier, obs)
        effects = lever_effects(dossier, obs)  # objective defaults to product_g_l
        return (levers or None), effects
    except Exception:  # noqa: BLE001 — discovery is best-effort; fall back to static
        return None, {}


def _build_within_run_pool(lever_effects: dict) -> list:
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
            objective="product_g_l", best_setting=best))
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
    objective_species: str = "P",
    levers: tuple[KnobLever, ...] = DEFAULT_LEVERS,
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

    # Discover THIS experiment's own levers (run_conditions metadata + varying
    # observation channels) so the debate argues the real levers, not a fixed
    # LABS knob list. Falls back to the static `levers` tuple only when nothing
    # was discovered (no metadata + no varying channel).
    discovered, lever_effects = _discover_bundle_levers(reader, dossier)
    seed_topics = extract_opportunity_topics(
        char, objective_species=objective_species, levers=levers,
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
        within_run_pool=_build_within_run_pool(lever_effects),
    )
