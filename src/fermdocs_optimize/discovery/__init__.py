"""Equation discovery: the agent proposes the ODE structure, the oracle judges it,
the agent revises, repeat. Unlike the fixed mechanistic model (where only the 7
parameters are fit), here the *equations themselves* are what learns.

Imports are LAZY (PEP 562 __getattr__) so that the data-native path
(``general_mech``/``expr``/``spec``) can be imported without pulling in the
Gen-1/LABS scaffolding (``candidate_model`` -> ``models.mechanistic`` +
``schema.Candidate``, ``loop`` -> the LABS active-learning loop). De-LABS,
2026-06-16: the data optimizer must not transitively import the benchmark stack.
"""
from fermdocs_optimize.discovery.expr import ExprError, compile_spec
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

# Lazy: candidate_model/loop bridge to the Gen-1/LABS stack; proposers is shared.
# Resolved on first attribute access so a bare `import ...discovery.general_mech`
# stays clean of LABS.
_LAZY = {
    "CandidateModel": "fermdocs_optimize.discovery.candidate_model",
    "discover_model": "fermdocs_optimize.discovery.loop",
    "discover_model_from_data": "fermdocs_optimize.discovery.loop",
    "LLMSpecProposer": "fermdocs_optimize.discovery.proposers",
    "TemplateProposer": "fermdocs_optimize.discovery.proposers",
}


def __getattr__(name):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    return getattr(importlib.import_module(target), name)
