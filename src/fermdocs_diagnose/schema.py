"""DiagnosisOutput contract — what the diagnosis agent emits.

Plan ref: plans/2026-05-02-diagnosis-agent.md §3.

Design rules:
  - Claims share a BaseClaim spine, kind-specific fields live on subclasses.
  - Closed-vocabulary enums (ConfidenceBasis) over free strings.
  - Citation discipline: every claim cites ≥1 finding (or trajectory for trends);
    enforced cross-output by validators.py against the upstream
    CharacterizationOutput.
  - Confidence cap at 0.85 matches identity_extractor convention for LLM output.
  - `domain_tags` is free-form (list[str]) in v1 — the closed vocabulary lands
    at the hypothesis-agent PR when orchestrator filtering needs it. Suggested
    values pre-listed in the prompt: growth, metabolism, environmental,
    data_quality, process_control, yield.
  - `re_run_from` deliberately absent from OpenQuestion — diagnosis cannot
    judge whether a user answer would change reasoning vs interpretation; the
    hypothesis stage owns that routing field.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from fermdocs_characterize.schema import Severity, TimeWindow

CLAIM_ID_RE = re.compile(r"^D-[FTA]-\d{4,}$")
QUESTION_ID_RE = re.compile(r"^D-Q-\d{4,}$")
LLM_CONFIDENCE_CAP = 0.85


class ConfidenceBasis(str, Enum):
    """How a claim's confidence was grounded.

    SCHEMA_ONLY: only nominal/std_dev from the golden schema; safe under
                 UNKNOWN_PROCESS / UNKNOWN_ORGANISM flags.
    PROCESS_PRIORS: registered-process priors invoked. Validator soft-downgrades
                    to SCHEMA_ONLY when an UNKNOWN flag is present.
    CROSS_RUN: pattern that requires multiple runs to assert.
    """

    SCHEMA_ONLY = "schema_only"
    PROCESS_PRIORS = "process_priors"
    CROSS_RUN = "cross_run"


class TrajectoryRef(BaseModel):
    """Typed (run_id, variable) reference. Replaces tuple[str, str] for stable
    JSON shape and downstream-agent referential integrity.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    variable: str


class DiagnosisMeta(BaseModel):
    """Versioning + provenance for a DiagnosisOutput. Frozen once emitted."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = Field(description="DiagnosisOutput schema version, e.g. '1.0'.")
    diagnosis_version: str = Field(description="Diagnosis-agent code version.")
    diagnosis_id: UUID
    supersedes_characterization_id: UUID = Field(
        description="UUID of the CharacterizationOutput this diagnosis was built from."
    )
    generation_timestamp: datetime
    model: str = Field(description="LLM model id, e.g. 'claude-opus-4-7'.")
    provider: Literal["anthropic", "gemini"]
    error: str | None = Field(
        default=None,
        description=(
            "Set when the agent could not produce a usable output (e.g."
            " 'llm_output_unparseable'). When non-None, claim lists are empty."
        ),
    )


class BaseClaim(BaseModel):
    """Shared spine for every diagnosis claim.

    Hard validators on this base catch the universal contract; subclasses add
    kind-specific fields (severity for failures, direction for trends, etc).
    Cross-output citation integrity (do these finding_ids exist in the upstream
    CharacterizationOutput?) lives in validators.py because BaseClaim has no
    handle to the upstream artifact.
    """

    claim_id: str
    summary: str = Field(min_length=1)
    cited_finding_ids: list[str] = Field(default_factory=list)
    affected_variables: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=LLM_CONFIDENCE_CAP)
    confidence_basis: ConfidenceBasis
    domain_tags: list[str] = Field(default_factory=list)
    provenance_downgraded: bool = Field(
        default=False,
        description=(
            "Set by validators.py when confidence_basis was downgraded from"
            " process_priors to schema_only because an UNKNOWN_* flag was"
            " present. Auditing field; not LLM-controlled."
        ),
    )

    @field_validator("claim_id")
    @classmethod
    def _claim_id_shape(cls, v: str) -> str:
        if not CLAIM_ID_RE.match(v):
            raise ValueError(
                f"claim_id must match D-{{F|T|A}}-NNNN, got {v!r}"
            )
        return v


class FailureClaim(BaseClaim):
    """An observable thing went wrong. Always cites ≥1 finding."""

    severity: Severity
    time_window: TimeWindow | None = None

    @field_validator("claim_id")
    @classmethod
    def _failure_prefix(cls, v: str) -> str:
        if not v.startswith("D-F-"):
            raise ValueError(f"FailureClaim id must start with D-F-, got {v!r}")
        return v

    @model_validator(mode="after")
    def _has_citation(self) -> FailureClaim:
        if not self.cited_finding_ids:
            raise ValueError(
                f"{self.claim_id}: FailureClaim must cite ≥1 finding_id"
            )
        return self


class TrendClaim(BaseClaim):
    """An observed pattern over time. May cite findings or trajectories or both;
    must cite at least one of either.
    """

    direction: Literal["increasing", "decreasing", "plateau", "oscillating"]
    cited_trajectories: list[TrajectoryRef] = Field(default_factory=list)
    time_window: TimeWindow | None = None

    @field_validator("claim_id")
    @classmethod
    def _trend_prefix(cls, v: str) -> str:
        if not v.startswith("D-T-"):
            raise ValueError(f"TrendClaim id must start with D-T-, got {v!r}")
        return v

    @model_validator(mode="after")
    def _has_citation(self) -> TrendClaim:
        if not self.cited_finding_ids and not self.cited_trajectories:
            raise ValueError(
                f"{self.claim_id}: TrendClaim must cite ≥1 finding or trajectory"
            )
        return self


class AnalysisClaim(BaseClaim):
    """Interpretive but observational. `kind` enumerates allowed angles to keep
    the LLM from drifting into causal territory.
    """

    kind: Literal[
        "cross_run_observation",
        "data_quality_caveat",
        "spec_alignment",
        "phase_characterization",
    ]

    @field_validator("claim_id")
    @classmethod
    def _analysis_prefix(cls, v: str) -> str:
        if not v.startswith("D-A-"):
            raise ValueError(f"AnalysisClaim id must start with D-A-, got {v!r}")
        return v

    @model_validator(mode="after")
    def _has_citation(self) -> AnalysisClaim:
        if not self.cited_finding_ids:
            raise ValueError(
                f"{self.claim_id}: AnalysisClaim must cite ≥1 finding_id"
            )
        return self


class OpenQuestion(BaseModel):
    """Diagnosis-stage open questions are always data-gap questions.

    `re_run_from` is deliberately absent. See plan §3 / §10: hypothesis stage
    owns that routing field.
    """

    question_id: str
    question: str = Field(min_length=1)
    why_it_matters: str = Field(min_length=1)
    cited_finding_ids: list[str] = Field(min_length=1)
    answer_format_hint: Literal["yes_no", "free_text", "numeric", "categorical"]
    domain_tags: list[str] = Field(default_factory=list)

    @field_validator("question_id")
    @classmethod
    def _question_id_shape(cls, v: str) -> str:
        if not QUESTION_ID_RE.match(v):
            raise ValueError(
                f"question_id must match D-Q-NNNN, got {v!r}"
            )
        return v


class DiagnosisOutput(BaseModel):
    """The artifact the hypothesis stage will read.

    Hard rejections (whole-output level) live here; soft enforcement
    (provenance downgrade, forbidden-phrase warnings) and cross-output citation
    integrity live in validators.py.
    """

    meta: DiagnosisMeta
    failures: list[FailureClaim] = Field(default_factory=list)
    trends: list[TrendClaim] = Field(default_factory=list)
    analysis: list[AnalysisClaim] = Field(default_factory=list)
    open_questions: list[OpenQuestion] = Field(default_factory=list)
    narrative: str | None = Field(
        default=None,
        description="Optional human-readable rollup, <500 words. Generated by"
        " the agent as a final pass; downstream agents read structured fields.",
    )

    @model_validator(mode="after")
    def _no_duplicate_ids(self) -> DiagnosisOutput:
        seen: dict[str, str] = {}
        for collection_name, items, attr in (
            ("failures", self.failures, "claim_id"),
            ("trends", self.trends, "claim_id"),
            ("analysis", self.analysis, "claim_id"),
            ("open_questions", self.open_questions, "question_id"),
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
    def _error_implies_empty(self) -> DiagnosisOutput:
        if self.meta.error is not None:
            non_empty = [
                name
                for name, lst in (
                    ("failures", self.failures),
                    ("trends", self.trends),
                    ("analysis", self.analysis),
                    ("open_questions", self.open_questions),
                )
                if lst
            ]
            if non_empty:
                raise ValueError(
                    f"meta.error is set but claim lists are non-empty: {non_empty}"
                )
        return self
