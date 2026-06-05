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
from dataclasses import dataclass
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


def load_optimization_bundle(
    bundle_dir: str | Path,
    *,
    objective_species: str = "P",
    levers: tuple[KnobLever, ...] = DEFAULT_LEVERS,
    max_trend_topics: int = 4,
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

    organism, process_family = _extract_organism_and_family(reader.get_dossier())
    user_question = _load_user_question(reader.dir)

    seed_topics = extract_opportunity_topics(
        char, objective_species=objective_species, levers=levers,
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
    )
