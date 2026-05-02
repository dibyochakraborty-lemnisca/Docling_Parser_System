"""Diagnosis agent: first LLM-authored stage in the multi-agent pipeline.

Reads CharacterizationOutput + AgentContext, emits an observational
DiagnosisOutput (failures, trends, analysis, open_questions). Does not
speculate on causes — that belongs to the hypothesis stage.

See plans/2026-05-02-diagnosis-agent.md for the locked design.
"""

from fermdocs_diagnose.agent import DiagnosisAgent, DiagnosisLLMClient
from fermdocs_diagnose.schema import (
    AnalysisClaim,
    BaseClaim,
    ConfidenceBasis,
    DiagnosisMeta,
    DiagnosisOutput,
    FailureClaim,
    OpenQuestion,
    TrajectoryRef,
    TrendClaim,
)

__all__ = [
    "AnalysisClaim",
    "BaseClaim",
    "ConfidenceBasis",
    "DiagnosisAgent",
    "DiagnosisLLMClient",
    "DiagnosisMeta",
    "DiagnosisOutput",
    "FailureClaim",
    "OpenQuestion",
    "TrajectoryRef",
    "TrendClaim",
]
