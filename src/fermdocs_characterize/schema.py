"""Characterization Agent contract.

See Execution.md for the full plan and the rationale behind every locked decision
(stable IDs, closed vocabularies, split confidence calibration, immutable outputs,
globally-unique finding IDs, forward-compatible kinetic_estimates, etc).

v1 populates only a subset of this schema, but the full shape is committed up front
so downstream agents (Diagnosis, Orchestrator, Simulation, Critic) can be written
against the final contract today.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# -----------------------------------------------------------------------------
# Closed vocabularies
#
# These enums ARE the spec. Adding a value requires a code change AND eval
# coverage. Validators reject anything not in these lists. See
# vocabularies/finding_types.md, edge_vocabulary.md, decision_types.md for prose.
# -----------------------------------------------------------------------------


class FindingType(str, Enum):
    RANGE_VIOLATION = "range_violation"  # v1
    # v2a:
    COHORT_OUTLIER = "cohort_outlier"
    # v2b:
    MASS_BALANCE_ERROR = "mass_balance_error"
    PROCESS_RELATIONSHIP_VIOLATION = "process_relationship_violation"
    CONTRADICTS = "contradicts"
    # v3+:
    PRECEDES_WITH_LAG = "precedes_with_lag"
    KINETIC_ANOMALY = "kinetic_anomaly"


class Severity(str, Enum):
    INFO = "info"
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"


class NodeType(str, Enum):
    SAMPLE = "sample"
    MEASUREMENT = "measurement"
    CONDITION = "condition"
    SOURCE = "source"


class EdgeType(str, Enum):
    MEASURED = "measured"  # Sample --measured--> Measurement
    UNDER_CONDITION = "under_condition"  # Sample --under_condition--> Condition
    AT_TIME = "at_time"  # Measurement --at_time--> (timestamp)
    DERIVED_FROM = "derived_from"  # Sample --derived_from--> Source


class DecisionType(str, Enum):
    """Closed list of decision categories an OpenQuestion can request.

    Used by the Orchestrator to route open questions to the right debate sub-agent.
    Adding a value requires a code change so routing stays deterministic.
    """

    CAUSAL_ATTRIBUTION = "causal_attribution"
    ANOMALY_CLASSIFICATION = "anomaly_classification"
    EVIDENCE_REQUEST = "evidence_request"


class ExtractedVia(str, Enum):
    """How a finding's confidence was calibrated. Drives downstream interpretation.

    See Execution.md §3: confidence is split between statistical (p-value/effect/n)
    and LLM-judged (≤0.85 cap). Consumers compare confidences only within the same
    `extracted_via` class.
    """

    DETERMINISTIC = "deterministic"  # Threshold-based, no LLM
    STATISTICAL = "statistical"  # Cohort z-scores, fits, p-values
    LLM_JUDGED = "llm_judged"  # Confidence capped at 0.85


class Tier(str, Enum):
    """Trust tier for a finding's evidence chain.

    A: direct measurement violation (range_violation against measured nominal+std_dev)
    B: derived from measurements (rates, yields, ratios)
    C: modeled / process-priors-derived (back-calculated from priors)

    Hypothesis-stage agents weight evidence by tier; downgrade C in early debate.
    """

    A = "A"
    B = "B"
    C = "C"


class NarrativeTag(str, Enum):
    """Closed vocabulary for narrative-extracted observations.

    Adding a tag is a code change + eval. Each tag covers one kind of
    operator/scientist statement in a fermentation report. The diagnosis
    agent weights tags differently — closure_event and observation are
    almost always actionable; protocol_note rarely is.
    """

    CLOSURE_EVENT = "closure_event"      # "terminated at 82h, white cells observed"
    DEVIATION = "deviation"              # "DO dropped to 20% during late phase"
    INTERVENTION = "intervention"        # "200 mL IPM added at 24h"
    OBSERVATION = "observation"          # "white cells visible during centrifugation"
    CONCLUSION = "conclusion"            # "yield 30% below target"
    PROTOCOL_NOTE = "protocol_note"      # method-section detail, low-action


# -----------------------------------------------------------------------------
# Provenance and evidence
# -----------------------------------------------------------------------------


class EvidenceStrength(BaseModel):
    """How strong the evidence behind a finding is. The Critic uses this to
    judge attack surface; the Orchestrator uses it for vote weighting.
    """

    n_observations: int = Field(
        ge=0, description="Number of underlying ingestion observations cited."
    )
    n_independent_runs: int = Field(
        ge=0,
        description="How many distinct runs the evidence spans. 1 means within-run only.",
    )
    statistical_power: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional power estimate for stat-grounded findings.",
    )


class TimeWindow(BaseModel):
    """A time interval the finding pertains to. Both bounds optional for
    findings that aren't time-localized (e.g. metadata issues).
    """

    start: float | None = Field(
        default=None, description="Start time in hours from run start (or absolute, see meta)."
    )
    end: float | None = Field(default=None)


# -----------------------------------------------------------------------------
# Trajectories
#
# One Trajectory per (run_id, variable). Regular grid + masks. Generic shape
# that any time-series consumer can use; happens to be JEPA-friendly.
# -----------------------------------------------------------------------------


class DataQuality(BaseModel):
    pct_missing: float = Field(ge=0.0, le=1.0)
    pct_imputed: float = Field(ge=0.0, le=1.0)
    pct_real: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _sums_to_one(self) -> DataQuality:
        total = self.pct_missing + self.pct_imputed + self.pct_real
        if not (0.999 <= total <= 1.001):
            raise ValueError(f"data_quality fractions must sum to 1.0, got {total}")
        return self


class Trajectory(BaseModel):
    """A regular-grid time series for one variable in one run.

    `values` and `imputation_flags` have the same length as `time_grid`.
    `values[i] is None` is allowed (truly missing); `imputation_flags[i] is True`
    means the value was synthesized by the imputer.
    """

    trajectory_id: str = Field(pattern=r"^T-\d{4,}$")
    run_id: str
    variable: str = Field(description="Canonical golden-column name.")
    time_grid: list[float] = Field(
        description="Uniform-Δt grid in hours from run start. Empty list allowed for"
        " non-time-resolved data."
    )
    values: list[float | None]
    imputation_flags: list[bool]
    imputation_method: str | None = Field(
        default=None,
        description="e.g. 'linear', 'carry_forward', None if no imputation.",
    )
    source_observation_ids: list[str] = Field(
        description="Ingestion observation IDs whose values fed this trajectory."
    )
    unit: str = Field(description="Canonical unit, e.g. 'g/L', 'h^-1', 'K'.")
    quality: float = Field(
        ge=0.0, le=1.0, description="Fraction of grid points that are real (not imputed/missing)."
    )
    data_quality: DataQuality

    @model_validator(mode="after")
    def _aligned_lengths(self) -> Trajectory:
        n = len(self.time_grid)
        if len(self.values) != n or len(self.imputation_flags) != n:
            raise ValueError(
                "time_grid, values, and imputation_flags must have the same length"
            )
        return self


# -----------------------------------------------------------------------------
# Findings
#
# Flat list. Stable IDs (`<characterization_id>:F-NNNN`). Calibrated uncertainty.
# Every finding cites real ingestion observation IDs. v1 only emits range_violation.
# -----------------------------------------------------------------------------


class Finding(BaseModel):
    finding_id: str = Field(
        description=(
            "Globally unique: '<characterization_id>:F-NNNN'. Never renumbered."
        ),
    )
    type: FindingType
    severity: Severity
    tier: Tier = Field(
        default=Tier.A,
        description="Trust tier (A=measured, B=derived, C=modeled). Default A so"
        " pre-tier findings backfill cleanly. range_violation always emits A.",
    )
    summary: str = Field(min_length=1, description="One-line description for human readers.")
    confidence: float = Field(ge=0.0, le=1.0)
    extracted_via: ExtractedVia
    caveats: list[str] = Field(default_factory=list)
    competing_explanations: list[str] = Field(default_factory=list)
    evidence_strength: EvidenceStrength
    evidence_observation_ids: list[str] = Field(
        min_length=1,
        description="Must resolve through ingestion's supersession chain.",
    )
    variables_involved: list[str] = Field(default_factory=list)
    time_window: TimeWindow | None = None
    run_ids: list[str] = Field(default_factory=list)
    statistics: dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form numeric context (z-score, threshold, p-value, etc)."
        " Schema-less by intent; consumers query by key.",
    )

    @field_validator("finding_id")
    @classmethod
    def _validate_id_shape(cls, v: str) -> str:
        # '<uuid>:F-NNNN' shape; full validation against meta.characterization_id
        # happens in the output validator (cross-field check).
        if ":F-" not in v:
            raise ValueError(f"finding_id must contain ':F-', got {v!r}")
        return v

    @model_validator(mode="after")
    def _confidence_cap_for_llm(self) -> Finding:
        if self.extracted_via == ExtractedVia.LLM_JUDGED and self.confidence > 0.85:
            raise ValueError(
                f"LLM-judged findings must have confidence ≤ 0.85, got {self.confidence}"
            )
        return self


# -----------------------------------------------------------------------------
# Narrative observations
#
# Pre-structured prose insights extracted from the source document during
# ingestion. The diagnosis agent treats these as direct operator/scientist
# statements (different evidence class than range_violation findings, which
# are deterministic detections over numerics).
#
# A document with no extractable prose insights produces an empty list and
# downstream behavior is identical to today. Plan B Stage 1 lands the
# schema; Stage 2 wires the extractor.
# -----------------------------------------------------------------------------


class NarrativeSourceLocator(BaseModel):
    """Where in the source document a narrative observation came from.

    All fields optional because real reports vary — some have page numbers,
    some have section names, some have neither. The locator is for human
    audit, not runtime routing.
    """

    page: int | None = Field(default=None, ge=1, description="1-indexed PDF page.")
    section: str | None = Field(default=None, description="e.g. 'Results', 'Procedure'.")
    paragraph_index: int | None = Field(default=None, ge=0)
    char_offset: int | None = Field(default=None, ge=0)


class NarrativeObservation(BaseModel):
    """A prose-derived insight from the source document.

    These bypass the deterministic finding pipeline — they are pre-structured
    statements from operators, scientists, or report authors. The agent
    treats them as direct evidence rather than something to be re-inferred
    from numbers.
    """

    narrative_id: str = Field(
        description=(
            "Globally unique: '<characterization_id>:N-NNNN'. Never renumbered."
        ),
    )
    tag: NarrativeTag
    text: str = Field(
        min_length=1,
        description="Verbatim or near-verbatim quote from the source. No paraphrase.",
    )
    source_locator: NarrativeSourceLocator = Field(default_factory=NarrativeSourceLocator)
    run_id: str | None = Field(
        default=None, description="When attributable to a single run."
    )
    time_h: float | None = Field(
        default=None, description="When attributable to a time point."
    )
    affected_variables: list[str] = Field(default_factory=list)
    confidence: float = Field(
        ge=0.0,
        le=0.85,
        description="Extractor's confidence in the tag + content. Capped at 0.85"
        " (LLM-judged, same posture as Finding.confidence).",
    )
    extraction_model: str = Field(
        min_length=1,
        description="Model+version that emitted this (e.g. 'gemini-3.1-pro-preview').",
    )

    @field_validator("narrative_id")
    @classmethod
    def _validate_id_shape(cls, v: str) -> str:
        if ":N-" not in v:
            raise ValueError(f"narrative_id must contain ':N-', got {v!r}")
        return v


# -----------------------------------------------------------------------------
# Timeline
# -----------------------------------------------------------------------------


class TimelineEvent(BaseModel):
    event_id: str = Field(pattern=r"^E-\d{4,}$")
    run_id: str
    time: float = Field(description="Hours from run start.")
    finding_id: str | None = Field(
        default=None,
        description="Optional pointer to a Finding this event materializes."
        " Events can also be raw observations promoted to the timeline.",
    )
    summary: str
    severity: Severity
    lag_to_next_seconds: float | None = Field(
        default=None, description="Seconds to next event in the same run; None if last."
    )


# -----------------------------------------------------------------------------
# Expected vs observed
# -----------------------------------------------------------------------------


class Deviation(BaseModel):
    deviation_id: str = Field(pattern=r"^D-\d{4,}$")
    run_id: str
    variable: str
    time: float | None = None
    expected: float
    expected_std_dev: float | None = None
    observed: float
    residual: float = Field(description="observed - expected")
    sigmas: float | None = Field(
        default=None, description="residual / expected_std_dev when stddev is known."
    )
    source_observation_ids: list[str] = Field(min_length=1)


# -----------------------------------------------------------------------------
# Facts graph (structural only)
# -----------------------------------------------------------------------------


class Node(BaseModel):
    node_id: str = Field(pattern=r"^N-\d{4,}$")
    type: NodeType
    label: str
    attributes: dict[str, Any] = Field(default_factory=dict)


class Edge(BaseModel):
    edge_id: str = Field(pattern=r"^G-\d{4,}$")
    type: EdgeType
    source_node_id: str
    target_node_id: str
    attributes: dict[str, Any] = Field(default_factory=dict)


class FactsGraph(BaseModel):
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)

    @model_validator(mode="after")
    def _no_dangling_edges(self) -> FactsGraph:
        node_ids = {n.node_id for n in self.nodes}
        for e in self.edges:
            if e.source_node_id not in node_ids or e.target_node_id not in node_ids:
                raise ValueError(
                    f"edge {e.edge_id} references nonexistent node"
                    f" ({e.source_node_id} -> {e.target_node_id})"
                )
        return self


# -----------------------------------------------------------------------------
# Open questions
# -----------------------------------------------------------------------------


class OpenQuestion(BaseModel):
    """Things characterization noticed but couldn't decide. The Orchestrator's
    debate loop is responsible for resolving these. Closed `decision_type`
    enables deterministic routing to the right sub-agent.
    """

    question_id: str = Field(pattern=r"^Q-\d{4,}$")
    question_text: str = Field(min_length=1)
    decision_type: DecisionType
    relevant_finding_ids: list[str] = Field(default_factory=list)
    relevant_run_ids: list[str] = Field(default_factory=list)
    would_resolve_with: list[str] = Field(
        default_factory=list,
        description="Concrete evidence types that would close this question, e.g."
        " 'media_batch_metadata', 'replicate_run_at_30C', 'OUR_trace'.",
    )


# -----------------------------------------------------------------------------
# Kinetic estimates (forward-compatible; v3 populates)
# -----------------------------------------------------------------------------


class KineticFit(BaseModel):
    fit_id: str = Field(pattern=r"^K-\d{4,}$")
    run_id: str
    parameter: str = Field(description="e.g. 'mu_x', 'mu_p', 'kla', 'Yxs', 'Ks'.")
    estimate: float
    confidence_interval: tuple[float, float] | None = None
    fit_window: TimeWindow
    r_squared: float | None = Field(default=None, ge=0.0, le=1.0)
    method: str = Field(description="e.g. 'windowed_ols', 'monod_nonlinear'.")
    evidence_observation_ids: list[str] = Field(min_length=1)


# -----------------------------------------------------------------------------
# Meta
# -----------------------------------------------------------------------------


class Meta(BaseModel):
    """Versioning, provenance, and supersession chain for the output.

    Outputs older than the current schema_version or process_priors_version are
    flagged invalid by the validator and must be regenerated rather than patched.
    """

    model_config = ConfigDict(frozen=True)

    schema_version: str = Field(
        description="Version of the CharacterizationOutput schema this output conforms to."
    )
    characterization_version: str = Field(
        description="Version of the characterization agent code that produced this output."
    )
    process_priors_version: str | None = Field(
        default=None,
        description="Version of process_priors.yaml used (v2b+). None when no priors used.",
    )
    characterization_id: UUID = Field(
        default_factory=uuid4,
        description="UUID. Used to namespace finding/event/deviation IDs across outputs.",
    )
    generation_timestamp: datetime
    supersedes: UUID | None = Field(
        default=None,
        description="characterization_id of the previous output for the same dossier(s)."
        " Outputs are immutable; re-running creates a new output.",
    )
    source_dossier_ids: list[str] = Field(
        min_length=1,
        description="Identifies which ingestion dossier(s) this output was built from.",
    )


# -----------------------------------------------------------------------------
# CharacterizationOutput
# -----------------------------------------------------------------------------


class CharacterizationOutput(BaseModel):
    """Single artifact every downstream agent reads.

    Read it under different lenses by querying the flat lists by ID, run, time,
    severity, or variable. The graph is reserved for structural data; insights
    live in `findings`.

    Immutable: re-running characterization on the same dossier produces a new
    CharacterizationOutput with `meta.supersedes` pointing at the previous one.
    """

    meta: Meta
    findings: list[Finding] = Field(default_factory=list)
    timeline: list[TimelineEvent] = Field(default_factory=list)
    expected_vs_observed: list[Deviation] = Field(default_factory=list)
    trajectories: list[Trajectory] = Field(default_factory=list)
    facts_graph: FactsGraph = Field(default_factory=FactsGraph)
    kinetic_estimates: list[KineticFit] = Field(
        default_factory=list,
        description="Empty in v1–v2; populated in v3.",
    )
    open_questions: list[OpenQuestion] = Field(default_factory=list)
    narrative_observations: list[NarrativeObservation] = Field(
        default_factory=list,
        description=(
            "Prose insights extracted from the source document during ingestion."
            " Empty list when no document text was available or extraction was"
            " disabled. See plans/2026-05-03-narrative-insight-extraction.md."
        ),
    )

    @model_validator(mode="after")
    def _finding_ids_namespaced(self) -> CharacterizationOutput:
        prefix = f"{self.meta.characterization_id}:F-"
        for f in self.findings:
            if not f.finding_id.startswith(prefix):
                raise ValueError(
                    f"finding_id {f.finding_id!r} not namespaced to characterization"
                    f" {self.meta.characterization_id} (expected prefix {prefix!r})"
                )
        return self

    @model_validator(mode="after")
    def _narrative_ids_namespaced(self) -> CharacterizationOutput:
        prefix = f"{self.meta.characterization_id}:N-"
        for n in self.narrative_observations:
            if not n.narrative_id.startswith(prefix):
                raise ValueError(
                    f"narrative_id {n.narrative_id!r} not namespaced to"
                    f" characterization {self.meta.characterization_id}"
                    f" (expected prefix {prefix!r})"
                )
        return self

    @model_validator(mode="after")
    def _no_duplicate_ids(self) -> CharacterizationOutput:
        seen: dict[str, str] = {}
        for collection_name, items, attr in (
            ("findings", self.findings, "finding_id"),
            ("timeline", self.timeline, "event_id"),
            ("expected_vs_observed", self.expected_vs_observed, "deviation_id"),
            ("trajectories", self.trajectories, "trajectory_id"),
            ("open_questions", self.open_questions, "question_id"),
            ("kinetic_estimates", self.kinetic_estimates, "fit_id"),
            ("facts_graph_nodes", self.facts_graph.nodes, "node_id"),
            ("facts_graph_edges", self.facts_graph.edges, "edge_id"),
            ("narrative_observations", self.narrative_observations, "narrative_id"),
        ):
            for item in items:
                key = getattr(item, attr)
                if key in seen:
                    raise ValueError(
                        f"duplicate id {key!r} in {collection_name}"
                        f" (also seen in {seen[key]})"
                    )
                seen[key] = collection_name
        return self

    @model_validator(mode="after")
    def _finding_xref_resolves(self) -> CharacterizationOutput:
        """Each TimelineEvent.finding_id, when set, must point at a real finding.
        Each OpenQuestion.relevant_finding_ids must all resolve.
        """
        finding_ids = {f.finding_id for f in self.findings}
        for ev in self.timeline:
            if ev.finding_id is not None and ev.finding_id not in finding_ids:
                raise ValueError(
                    f"timeline event {ev.event_id} cites missing finding_id {ev.finding_id!r}"
                )
        for q in self.open_questions:
            for fid in q.relevant_finding_ids:
                if fid not in finding_ids:
                    raise ValueError(
                        f"open_question {q.question_id} cites missing finding_id {fid!r}"
                    )
        return self
