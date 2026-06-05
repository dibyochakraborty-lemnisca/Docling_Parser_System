"""Phase B: optimization specs + hooks wiring (no LLM, no network).

Verifies the narrative reframe lives in the specialist specs, the domain
expertise is preserved per persona, and OptimizeHooks reuses the engine while
swapping only the three specialists.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from fermdocs_optimize_debate.hooks import OptimizeHooks
from fermdocs_optimize_debate.specs import (
    OPT_KINETICS_SPEC,
    OPT_MASS_TRANSFER_SPEC,
    OPT_METABOLIC_SPEC,
)

_ALL = (OPT_KINETICS_SPEC, OPT_MASS_TRANSFER_SPEC, OPT_METABOLIC_SPEC)


@pytest.mark.parametrize("spec", _ALL)
def test_specs_are_optimization_framed(spec):
    ident = spec["system_identity"].lower()
    assert "optimization" in ident and "headroom" in ident
    assert "fault" in ident  # explicitly says NOT a fault hunt
    # speaks the knob vocabulary so facets map to the optimizer's box
    assert "malt_frac" in spec["system_identity"]
    joined = " ".join(spec["invariants"]).lower()
    assert "lever" in joined and "affected_variables" in joined
    assert "oracle" in joined  # honesty: direction not magnitude


def test_domain_expertise_preserved_per_persona():
    assert "μ" in OPT_KINETICS_SPEC["system_identity"]
    assert "kLa" in OPT_MASS_TRANSFER_SPEC["system_identity"]
    assert "maltose" in OPT_METABOLIC_SPEC["system_identity"].lower()
    assert OPT_KINETICS_SPEC["role"] == "kinetics"
    assert OPT_MASS_TRANSFER_SPEC["role"] == "mass_transfer"
    assert OPT_METABOLIC_SPEC["role"] == "metabolic"


def _fake_bundle():
    return SimpleNamespace(
        hyp_input=SimpleNamespace(user_question=None, followup_context=None, seed_topics=[]),
        characterization=SimpleNamespace(findings=[], narrative_observations=[]),
        findings_pool=[], narratives_pool=[], trajectories_pool=[],
        priors_pool=[], analyses_pool=[],
    )


def test_hooks_swap_specialists_and_reuse_engine():
    hooks = OptimizeHooks(_fake_bundle())
    # the three specialists carry the optimization specs
    assert hooks._specialists["kinetics"]._spec is OPT_KINETICS_SPEC
    assert hooks._specialists["mass_transfer"]._spec is OPT_MASS_TRANSFER_SPEC
    assert hooks._specialists["metabolic"]._spec is OPT_METABOLIC_SPEC
    # the rest of the engine is reused (constructed by the base class)
    assert hooks._synthesizer is not None
    assert hooks._critic is not None
    assert hooks._judge is not None
    assert hooks._orchestrator is not None
    # run.py wires against these hooks (import check + signature)
    from fermdocs_optimize_debate.run import run_optimization_debate
    assert callable(run_optimization_debate)
