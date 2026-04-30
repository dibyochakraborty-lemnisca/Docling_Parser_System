"""CharacterizationPipeline: plan-and-execute, deterministic-only (v1).

    build_summary  →  build_trajectories  →  range_violation_generator
                                                  ↓
                                          assign_finding_ids (sorted)
                                                  ↓
                                          build_deviations
                                                  ↓
                                          build_timeline (uses findings)
                                                  ↓
                                          build_open_questions (findings + traj)
                                                  ↓
                                          build_facts_graph (empty in v1)
                                                  ↓
                                          assemble + validate

IDs are stable across re-runs of the same input when `characterization_id` and
`generation_timestamp` are pinned. The fixture tests pin both.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from fermdocs_characterize import CHARACTERIZATION_VERSION, SCHEMA_VERSION
from fermdocs_characterize.builders.expected_vs_observed import build_deviations
from fermdocs_characterize.builders.facts_graph import build_facts_graph
from fermdocs_characterize.builders.open_questions import build_open_questions
from fermdocs_characterize.builders.timeline import build_timeline
from fermdocs_characterize.candidates.range_violation import (
    CandidateFinding,
    find_range_violations,
)
from fermdocs_characterize.schema import (
    CharacterizationOutput,
    Finding,
    Meta,
)
from fermdocs_characterize.specs import DictSpecsProvider, SpecsProvider
from fermdocs_characterize.validators.output_validator import (
    ValidationError,
    validate_output,
)
from fermdocs_characterize.views.summary import build_summary
from fermdocs_characterize.views.trajectories import build_trajectories


class CharacterizationPipeline:
    """v1 pipeline: deterministic generators only, single pass.

    `specs_provider` is injected so production can swap DictSpecsProvider for
    an ingestion-backed setpoint table without touching the pipeline.
    """

    def __init__(
        self,
        specs_provider: SpecsProvider | None = None,
        *,
        validate: bool = True,
        current_schema_version: str = SCHEMA_VERSION,
        current_process_priors_version: str | None = None,
    ) -> None:
        self._specs_provider = specs_provider
        self._validate = validate
        self._current_schema_version = current_schema_version
        self._current_process_priors_version = current_process_priors_version

    def run(
        self,
        dossier: dict[str, Any],
        *,
        characterization_id: UUID | None = None,
        generation_timestamp: datetime | None = None,
        supersedes: UUID | None = None,
    ) -> CharacterizationOutput:
        specs = self._specs_provider or DictSpecsProvider.from_dossier(dossier)

        char_id = characterization_id or uuid4()
        gen_ts = generation_timestamp or datetime.utcnow()

        # 1. Views
        summary = build_summary(dossier, specs)
        trajectories = build_trajectories(summary, dossier)

        # 2. Trajectory grid hint (used by caveats and open_questions)
        grid_hint = dossier.get("_trajectory_grid")
        dt_hours = grid_hint.get("dt_hours") if isinstance(grid_hint, dict) else None

        # 3. Candidates (v1: range_violation only)
        candidates: list[CandidateFinding] = find_range_violations(
            summary, trajectories, dt_hours=dt_hours
        )

        # 4. Sort candidates and assign namespaced finding IDs
        candidates.sort(key=lambda c: c.sort_key)
        findings: list[Finding] = []
        for i, c in enumerate(candidates, start=1):
            findings.append(
                Finding(
                    finding_id=f"{char_id}:F-{i:04d}",
                    type=c.type,
                    severity=c.severity,
                    summary=c.summary,
                    confidence=c.confidence,
                    extracted_via=c.extracted_via,
                    caveats=c.caveats,
                    competing_explanations=c.competing_explanations,
                    evidence_strength=c.evidence_strength,
                    evidence_observation_ids=c.evidence_observation_ids,
                    variables_involved=c.variables_involved,
                    time_window=c.time_window,
                    run_ids=c.run_ids,
                    statistics=c.statistics,
                )
            )

        # 5. Other artifacts
        deviations = build_deviations(summary)
        timeline = build_timeline(findings)
        open_questions = build_open_questions(findings, trajectories, dt_hours=dt_hours)
        facts_graph = build_facts_graph(summary)

        # 6. Assemble
        experiment_id = (dossier.get("experiment") or {}).get("experiment_id")
        source_dossier_ids = [experiment_id] if experiment_id else []
        meta = Meta(
            schema_version=self._current_schema_version,
            characterization_version=CHARACTERIZATION_VERSION,
            process_priors_version=None,
            characterization_id=char_id,
            generation_timestamp=gen_ts,
            supersedes=supersedes,
            source_dossier_ids=source_dossier_ids,
        )
        output = CharacterizationOutput(
            meta=meta,
            findings=findings,
            timeline=timeline,
            expected_vs_observed=deviations,
            trajectories=trajectories,
            facts_graph=facts_graph,
            kinetic_estimates=[],
            open_questions=open_questions,
        )

        # 7. Validate
        if self._validate:
            errors = validate_output(
                output,
                dossiers={experiment_id: dossier} if experiment_id else None,
                current_schema_version=self._current_schema_version,
                current_process_priors_version=self._current_process_priors_version,
            )
            if errors:
                raise ValidationError(errors)

        return output
