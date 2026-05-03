"""Tests that lock in the synthesizer's evidence-discipline guardrails.

Carotenoid debug: H-001, H-002, H-003 all rejected by the critic for
the same reason — synthesizer kept inferring CAUSAL ABSENCE from
DOCUMENTARY ABSENCE. Examples of the failure mode:

  - "independent of explicit physical transport triggers"
  - "pointing to an intrinsic biological trigger rather than external
    physical failure"
  - "driven by severe metabolic stress rather than macroscopic
    physical control failures"

The narratives don't say mass-transfer issues *didn't happen*. They
just don't mention them — and closure summaries omit details
routinely. The critic correctly rejected each iteration; the
synthesizer kept rephrasing the same overreach.

This commit added explicit guardrails to the synthesizer's prompt:
  1. Documented absence != ruled out
  2. Causal claims require causal evidence
  3. Pure observational hypotheses are valid (no mechanism required)

These tests assert the new wording is present so future edits can't
silently regress.
"""

from __future__ import annotations

from fermdocs_hypothesis.agents.synthesizer import (
    SYNTHESIZER_INVARIANTS,
    SYNTHESIZER_SYSTEM,
)


def test_system_prompt_distinguishes_documented_absence_from_ruled_out():
    """The carotenoid bug: synthesizer treated 'no documented X' as
    'X did not happen'. The new wording must explicitly call out the
    distinction."""
    import re

    flat = re.sub(r"\s+", " ", SYNTHESIZER_SYSTEM)
    assert "DOCUMENTED ABSENCE" in flat
    assert "RULED OUT" in flat
    assert "documentary absence" in flat or "absence of mention" in flat


def test_system_prompt_blocks_causal_overreach_phrases():
    """Three exact phrase patterns the critic flagged across H-001..H-003.
    The prompt must name them explicitly so the model recognizes them
    as red flags. Match against whitespace-normalized text so the
    assertions survive prompt re-flowing."""
    import re

    flat = re.sub(r"\s+", " ", SYNTHESIZER_SYSTEM)
    # The three failure phrases the critic kept catching
    assert "rather than X" in flat
    assert "independent of X" in flat
    assert "ruling out X" in flat
    # And the safer alternatives the model should reach for instead
    assert "associated with" in flat
    assert "coincides with" in flat or "observed alongside" in flat


def test_system_prompt_permits_observation_only_hypotheses():
    """Critical for breaking the rejection loop — the synthesizer needs
    to know that a hypothesis without a proposed mechanism is a
    legitimate, accept-able output."""
    import re

    flat = re.sub(r"\s+", " ", SYNTHESIZER_SYSTEM)
    assert "PURE OBSERVATIONAL HYPOTHESES ARE VALID" in flat
    # The model needs to know small grounded > big speculative
    assert "small accepted hypothesis beats a large rejected one" in flat


def test_invariants_call_out_documented_absence_specifically():
    """Prompt body has the long-form explanation; invariants are the
    quick-reference list at the top. Both must mention the rule."""
    flat = " ".join(SYNTHESIZER_INVARIANTS)
    assert "Documented absence" in flat or "documented absence" in flat
    assert "absence of mention" in flat or "proof of absence" in flat


def test_invariants_call_out_causal_evidence_requirement():
    flat = " ".join(SYNTHESIZER_INVARIANTS)
    assert "Causal language" in flat or "causal evidence" in flat.lower()
    # Specific overreach verbs must appear so the model recognizes them
    assert "driven by" in flat
    assert "due to" in flat


def test_no_reasoning_capability_removed():
    """Per design constraint: prompt edits must not strip the synthesizer's
    capability to reason — only redirect it. Verify the original
    instructions to integrate facets, surface tension, and not paper
    over disagreements remain intact."""
    import re

    flat = re.sub(r"\s+", " ", SYNTHESIZER_SYSTEM)
    assert "integrates the angles" in flat
    assert "Do not average" in flat
    assert "do not paper over disagreements" in flat
    assert "surface that tension" in flat
