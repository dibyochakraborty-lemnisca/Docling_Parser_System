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

import logging
import re
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from fermdocs_characterize import CHARACTERIZATION_VERSION, SCHEMA_VERSION
from fermdocs_characterize.agents.catalog_runner import (
    MetricCatalogRunner,
    _BundleView,
)
from fermdocs_characterize.agents.trajectory_analyzer import (
    TrajectoryAnalyzerAgent,
)
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
    NarrativeObservation,
)
from fermdocs.domain.golden_schema import load_schema
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
        trajectory_analyzer: TrajectoryAnalyzerAgent | None = None,
    ) -> None:
        # `trajectory_analyzer` is the LLM-driven pattern discovery stage
        # (May 2026). When None, the pipeline runs purely deterministic
        # — preserves backward compat for tests + fixture-based runs.
        # When provided, the analyzer runs after spec checks and appends
        # FindingType.TRAJECTORY_PATTERN findings to the output.
        self._specs_provider = specs_provider
        self._validate = validate
        self._current_schema_version = current_schema_version
        self._current_process_priors_version = current_process_priors_version
        self._trajectory_analyzer = trajectory_analyzer

    def run(
        self,
        dossier: dict[str, Any],
        *,
        characterization_id: UUID | None = None,
        generation_timestamp: datetime | None = None,
        supersedes: UUID | None = None,
    ) -> CharacterizationOutput:
        # Specs precedence:
        #   1. Explicitly injected provider wins (production / tests).
        #   2. Otherwise merge schema defaults with dossier `_specs` overrides
        #      via from_schema_with_overrides. This is what real ingestion
        #      dossiers want: golden_schema.yaml carries default
        #      nominal/std_dev per variable; the dossier may override any
        #      field per run.
        if self._specs_provider is not None:
            specs = self._specs_provider
        else:
            try:
                schema = load_schema()
                specs = DictSpecsProvider.from_schema_with_overrides(schema, dossier)
            except Exception:
                # Schema unavailable (offline tests, missing file): fall back
                # to dossier-only specs so existing fixture-based tests keep
                # working.
                specs = DictSpecsProvider.from_dossier(dossier)

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

        # 4b. LLM-driven trajectory pattern analysis (May 2026 architecture
        # shift). Optional — when no analyzer is wired, pipeline stays
        # purely deterministic. When wired, the analyzer reads the same
        # trajectories + spec findings, runs execute_python over a tmp
        # observations.csv, and emits FindingType.TRAJECTORY_PATTERN
        # findings that get IDs after the spec findings.
        # Pre-step: deterministic catalog runner. Iterates every
        # (metric, run) pair and emits Findings deterministically. This
        # fixes the IndPenSim multi-run asymmetry bug where the LLM
        # analyzer would compute B10/A8/etc on RUN-1 only and skip
        # RUN-2 silently. Plan ref:
        # plans/2026-05-07-characterize-determinism.md commit 1.
        observed = (
            (dossier.get("experiment") or {})
            .get("process", {})
            .get("observed", {})
        )
        organism = (observed.get("organism") or "").strip() or None
        process_family = (
            observed.get("process_family_hint") or ""
        ).strip() or None

        # `catalog_findings` is visible to the LLM analyzer below so it
        # can render an [ALREADY COMPUTED] block (A1 fix). Empty list on
        # back-compat / no-trajectories runs.
        catalog_findings: list[Finding] = []
        if trajectories:
            try:
                catalog_bundle = _BundleView(
                    characterization_id=str(char_id),
                    run_ids=sorted({t.run_id for t in trajectories}),
                    trajectories=trajectories,
                    organism=organism,
                    process_family=process_family,
                )
                catalog_runner = MetricCatalogRunner()
                raw_catalog_findings = catalog_runner.compute_all(catalog_bundle)
                # Re-namespace IDs to follow the bundle convention; keep
                # the renamed list as `catalog_findings` so the analyzer
                # gets the same-id view downstream agents will see.
                for i, cf in enumerate(
                    raw_catalog_findings, start=len(findings) + 1
                ):
                    renamed = cf.model_copy(
                        update={"finding_id": f"{char_id}:F-{i:04d}"}
                    )
                    findings.append(renamed)
                    catalog_findings.append(renamed)
            except RuntimeError as exc:
                # Pre-flight import failure (A2) is fatal: don't fall back
                # silently. Re-raise so characterize aborts loud.
                raise
            except Exception as exc:
                # Other failures (e.g. trajectory edge-cases) are
                # advisory; log and continue to the LLM analyzer.
                _log.warning(
                    "catalog runner raised %s; falling through to LLM analyzer",
                    exc.__class__.__name__,
                )

        if self._trajectory_analyzer is not None and trajectories:
            try:
                # Surface the dossier's identity layer to the analyzer so
                # Tier C metric calls (mu_max_reference_vs_observed,
                # qs_from_verduyn_yields, overflow_threshold) can pass the
                # organism string to process_priors lookup. Without this the
                # analyzer hardcodes organism=None and every Tier C metric
                # data-gaps even when priors registry has the entry.
                analyzer_result = self._trajectory_analyzer.analyze(
                    char_id=char_id,
                    trajectories=trajectories,
                    spec_findings=findings,
                    starting_index=len(findings) + 1,
                    organism=organism,
                    process_family=process_family,
                    catalog_findings=catalog_findings,
                )
                findings.extend(analyzer_result.findings)
            except Exception as exc:
                # Analyzer is advisory; never block the deterministic
                # spec-finding pipeline on a Gemini outage or sandbox error.
                _log.warning(
                    "trajectory_analyzer raised %s; skipping pattern findings",
                    exc.__class__.__name__,
                )

        # 5. Other artifacts
        deviations = build_deviations(summary)
        timeline = build_timeline(findings)
        open_questions = build_open_questions(findings, trajectories, dt_hours=dt_hours)
        facts_graph = build_facts_graph(summary)

        # 5b. Plan B Stage 2.2: materialize narrative observations from the
        # dossier. The dossier-side extractor minted narrative_ids under a
        # transient UUID; we renamespace each to char_id so
        # CharacterizationOutput's namespace validator accepts them. Bad
        # entries are dropped with a warning — extraction is additive.
        narrative_observations: list[NarrativeObservation] = _materialize_narratives(
            dossier.get("narrative_observations") or [],
            char_id=char_id,
        )

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
            narrative_observations=narrative_observations,
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


_log = logging.getLogger(__name__)
_NARRATIVE_ID_TAIL_RE = re.compile(r":(N-\d{4,})$")


def _materialize_narratives(
    raw_items: list[Any],
    *,
    char_id: UUID,
) -> list[NarrativeObservation]:
    """Coerce dossier-side narrative dicts into NarrativeObservation models.

    Plan B Stage 2.2: the dossier-side extractor namespaced narrative_ids
    under a transient UUID to keep them unique within the dossier. We
    renamespace each to the active char_id so the CharacterizationOutput
    namespace validator accepts them. Malformed entries are dropped with
    a warning — extraction is additive, never blocks the pipeline.
    """
    out: list[NarrativeObservation] = []
    for i, raw in enumerate(raw_items, start=1):
        if not isinstance(raw, dict):
            _log.info("narrative_observations[%d] is not a dict; skipping", i)
            continue
        # Replace the namespace prefix with the active char_id. Tail
        # (e.g. 'N-0001') is preserved if shaped right; else assigned
        # by position.
        original_id = str(raw.get("narrative_id", ""))
        m = _NARRATIVE_ID_TAIL_RE.search(original_id)
        tail = m.group(1) if m else f"N-{i:04d}"
        clone = dict(raw)
        clone["narrative_id"] = f"{char_id}:{tail}"
        try:
            out.append(NarrativeObservation.model_validate(clone))
        except Exception as exc:
            _log.info(
                "narrative_observations[%d] failed validation (%s); skipping",
                i,
                exc.__class__.__name__,
            )
    return out
