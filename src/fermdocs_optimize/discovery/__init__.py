"""Equation discovery: the agent proposes the ODE structure, the oracle judges it,
the agent revises, repeat. Unlike the fixed mechanistic model (where only the 7
parameters are fit), here the *equations themselves* are what learns.
"""
from fermdocs_optimize.discovery.candidate_model import CandidateModel
from fermdocs_optimize.discovery.expr import ExprError, compile_spec
from fermdocs_optimize.discovery.loop import discover_model, discover_model_from_data
from fermdocs_optimize.discovery.proposers import (
    LLMSpecProposer,
    TemplateProposer,
)
from fermdocs_optimize.discovery.spec import (
    DiscoveryReport,
    DiscoveryRound,
    ModelSpec,
    ParamSpec,
)

__all__ = [
    "CandidateModel", "ExprError", "compile_spec",
    "discover_model", "discover_model_from_data",
    "LLMSpecProposer", "TemplateProposer",
    "DiscoveryReport", "DiscoveryRound", "ModelSpec", "ParamSpec",
]
