"""Pydantic output contract for the recommendation stage.

A recommendation is the bake-off result over three brewtwin model families.
The agent fits + scores the models in the sandbox; the deterministic rubric
(rubric.py) decides the winner or an honest refusal. The schema enforces that
the two stay consistent — a refusal carries no interventions, a confident
verdict names a real model — so an inconsistent LLM payload cannot slip through.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

RECOMMENDATION_SCHEMA_VERSION = "1.0.0"
RECOMMENDATION_AGENT_VERSION = "0.1.0"

ModelType = Literal["mechanistic", "surrogate", "hybrid"]
RecommendedModel = Literal["mechanistic", "surrogate", "hybrid", "none"]


class RecommendationMeta(BaseModel):
    schema_version: str = RECOMMENDATION_SCHEMA_VERSION
    agent_version: str = RECOMMENDATION_AGENT_VERSION
    recommendation_id: uuid.UUID
    run_id: str | None = None
    generation_timestamp: datetime
    model: str
    provider: str
    error: str | None = None


class CandidateReport(BaseModel):
    """One attempted model family + its scored fit (or why it was disqualified)."""

    model_type: ModelType
    attempted: bool
    disqualified: bool = False
    disqualification_reason: str | None = None
    # Rubric-derived selection scalars (filled by the engine, not the LLM).
    selection_r2: float | None = None
    selection_rmse: float | None = None
    good_fit: bool | None = None
    good_fit_reason: str | None = None
    plausible: bool | None = None
    offending_params: list[str] | None = None
    stalled: bool | None = None
    eligible_species: list[str] | None = None
    fitted_parameters: dict[str, Any] | None = None
    report: dict[str, Any] | None = Field(
        None, description="The raw build_report dict from brewtwin_metrics."
    )


class Intervention(BaseModel):
    """A counterfactual the selected model was simulated under."""

    intervention_id: str | None = None
    description: str
    knob: str | None = None  # control/initial-condition that was changed
    objective_metric: str | None = None  # e.g. "P.final_value"
    baseline_value: float | None = None
    predicted_value: float | None = None
    delta: float | None = None
    in_coverage: bool | None = None  # within the observed/validated regime?
    caveat: str | None = None
    rationale: str | None = None


class RecommendationOutput(BaseModel):
    meta: RecommendationMeta
    recommended_model: RecommendedModel
    confident: bool
    refusal_reason: str | None = None
    selection_rationale: str
    candidates: list[CandidateReport] = Field(default_factory=list)
    interventions: list[Intervention] = Field(default_factory=list)
    grounding_hyp_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _coherent(self) -> "RecommendationOutput":
        is_none = self.recommended_model == "none"
        # confident iff a real model is recommended
        if self.confident == is_none:
            self.confident = not is_none
        # a refusal carries no interventions and must give a reason
        if is_none:
            self.interventions = []
            if not self.refusal_reason:
                self.refusal_reason = "stage_error"
        else:
            # a confident recommendation does not carry a refusal reason
            self.refusal_reason = None
        return self
