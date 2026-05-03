"""Load a bundle directory into a HypothesisInput.

Plan ref: plans/2026-05-03-hypothesis-debate-v0.md §9.

Reads:
  - <bundle>/diagnosis/diagnosis.json       (DiagnosisOutput)
  - <bundle>/characterization/characterization.json
  - <bundle>/characterization/narrative_observations.json (optional)
  - <bundle>/dossier.json                   (for organism/process_family)

Produces:
  - HypothesisInput with seed_topics already extracted
  - Reference pools the projector consumes (FindingRef, NarrativeRef,
    TrajectoryViewRef, ResolvedPriorRef) — built lazily by `build_reference_pools`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from fermdocs.bundle.reader import BundleReader
from fermdocs.domain.process_priors import (
    ProcessPriors,
    cached_priors,
    resolve_priors,
)
from fermdocs_characterize.schema import CharacterizationOutput
from fermdocs_diagnose.schema import DiagnosisOutput
from fermdocs_hypothesis.schema import (
    FindingRef,
    HypothesisInput,
    NarrativeRef,
    ResolvedPriorRef,
    TrajectoryViewRef,
)
from fermdocs_hypothesis.seed_topic_extractor import extract_seed_topics


@dataclass
class LoadedBundle:
    """Container for everything Stage 2 needs from a bundle."""

    hyp_input: HypothesisInput
    diagnosis: DiagnosisOutput
    characterization: CharacterizationOutput
    findings_pool: list[FindingRef]
    narratives_pool: list[NarrativeRef]
    trajectories_pool: list[TrajectoryViewRef]
    priors_pool: list[ResolvedPriorRef]
    bundle_dir: Path


def load_bundle(bundle_dir: str | Path) -> LoadedBundle:
    """Load all artifacts from a bundle directory."""
    reader = BundleReader(bundle_dir)
    diagnosis_json = reader.get_diagnosis_json()
    diagnosis = DiagnosisOutput.model_validate_json(diagnosis_json)

    char_json = reader.get_characterization_json()
    characterization = CharacterizationOutput.model_validate_json(char_json)

    # Load narrative observations into characterization if not already present.
    if reader.has_narrative_observations() and not characterization.narrative_observations:
        narr_raw = json.loads(reader.get_narrative_observations_json())
        # Re-build characterization with narratives attached.
        char_dict = json.loads(char_json)
        char_dict["narrative_observations"] = narr_raw
        characterization = CharacterizationOutput.model_validate(char_dict)

    dossier = reader.get_dossier()
    organism, process_family = _extract_organism_and_family(dossier)

    seed_topics = extract_seed_topics(diagnosis)

    findings_pool = _build_findings_pool(characterization)
    narratives_pool = _build_narratives_pool(characterization)
    trajectories_pool = _build_trajectories_pool(characterization)
    priors_pool = _build_priors_pool(organism, process_family)

    hyp_input = HypothesisInput(
        diagnosis=diagnosis,
        characterization=characterization,
        bundle_path=str(reader.dir),
        seed_topics=seed_topics,
        organism=organism,
        process_family=process_family,
    )
    return LoadedBundle(
        hyp_input=hyp_input,
        diagnosis=diagnosis,
        characterization=characterization,
        findings_pool=findings_pool,
        narratives_pool=narratives_pool,
        trajectories_pool=trajectories_pool,
        priors_pool=priors_pool,
        bundle_dir=Path(bundle_dir),
    )


def _extract_organism_and_family(dossier: dict) -> tuple[str | None, str | None]:
    process = (dossier.get("experiment") or {}).get("process") or {}
    observed = process.get("observed") or {}
    registered = process.get("registered") or {}
    organism = (observed.get("organism") or "").strip() or None
    process_family = (registered.get("process_family") or registered.get("name") or "").strip() or None
    return organism, process_family


def _build_findings_pool(char: CharacterizationOutput) -> list[FindingRef]:
    return [
        FindingRef(
            finding_id=f.finding_id,
            summary=f.summary,
            variables_involved=list(f.variables_involved),
        )
        for f in char.findings
    ]


def _build_narratives_pool(char: CharacterizationOutput) -> list[NarrativeRef]:
    out: list[NarrativeRef] = []
    for n in char.narrative_observations:
        out.append(
            NarrativeRef(
                narrative_id=n.narrative_id,
                tag=n.tag.value if hasattr(n.tag, "value") else str(n.tag),
                summary=n.text,
                run_id=n.run_id,
            )
        )
    return out


def _build_trajectories_pool(char: CharacterizationOutput) -> list[TrajectoryViewRef]:
    return [
        TrajectoryViewRef(
            run_id=t.run_id,
            variable=t.variable,
            note=f"unit={t.unit}, quality={t.quality:.2f}",
        )
        for t in char.trajectories
    ]


def _build_priors_pool(
    organism: str | None, process_family: str | None
) -> list[ResolvedPriorRef]:
    if not organism:
        return []
    try:
        priors: ProcessPriors = cached_priors()
    except Exception:
        return []
    resolved = resolve_priors(priors, organism=organism, process_family=process_family)
    return [
        ResolvedPriorRef(
            organism=r.organism,
            process_family=r.process_family,
            variable=r.variable,
            range_low=r.range[0],
            range_high=r.range[1],
            typical=r.typical,
            source=r.source,
        )
        for r in resolved
    ]
