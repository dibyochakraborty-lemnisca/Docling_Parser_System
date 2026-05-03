"""Tests that lock in the diagnose bundle prompt's UNKNOWN_PROCESS handling.

The prompt's rule 4 was previously a one-liner that biased the agent
toward emitting only data_quality_caveats under UNKNOWN_PROCESS flags.
Observed effect on the carotenoid PDF: diagnose ignored 8 narrative
observations (including 6 closure_events with cell death / pigment loss)
and emitted only "the registry doesn't match" caveats. Hypothesis stage
then debated the registry mismatch as the central question and rejected
both hypotheses.

The reworded rule 4 keeps schema_only confidence on numerical claims
under UNKNOWN_PROCESS but explicitly preserves narrative observations
as primary evidence regardless of the flag. This test asserts the
reworded text is present so future edits don't accidentally regress
to the old behavior.

This is a "behavioral guardrail" test: it doesn't replace eval testing
with real LLM calls (we have those in the live carotenoid eval), it
just catches accidental prompt-regression in code review.
"""

from __future__ import annotations

from fermdocs_diagnose.agent import _BUNDLE_SYSTEM_PROMPT


def test_rule4_distinguishes_trajectory_vs_narrative_grounding():
    """Rule 4(a) covers numerical claims; rule 4(b) keeps narrative
    primary. If either subclause is removed, regression risk is high."""
    assert "TRAJECTORY-GROUNDED claims" in _BUNDLE_SYSTEM_PROMPT
    assert "NARRATIVE OBSERVATIONS REMAIN PRIMARY EVIDENCE" in _BUNDLE_SYSTEM_PROMPT


def test_rule4_explicitly_disallows_registry_mismatch_as_headline():
    """The carotenoid bug: agents made 'organism doesn't match registry'
    the headline finding. Rule 4(c) tells them not to."""
    msg = _BUNDLE_SYSTEM_PROMPT
    assert "registry mismatch" in msg
    assert "headline finding" in msg
    assert "routing signal" in msg


def test_rule7_explicitly_calls_out_caveat_only_emit_as_violation():
    """Rule 7's specific case: empty failures + closure_event narrative
    obs is a contract violation, not a valid output."""
    msg = _BUNDLE_SYSTEM_PROMPT
    assert "data_quality_caveat" in msg
    assert "closure_event" in msg
    assert "contract violation" in msg.lower() or "CONTRACT VIOLATION" in msg


def test_rule7_explicitly_blocks_data_quality_to_spec_alignment_dodge():
    """Carotenoid follow-up: agent learned to dodge the
    'don't emit only data_quality_caveat' rule by relabeling the same
    meta-claim under kind=spec_alignment. Rule 7 must explicitly name
    BOTH meta kinds and the dodge pattern."""
    msg = _BUNDLE_SYSTEM_PROMPT
    assert "spec_alignment" in msg
    assert "META" in msg
    # The explicit "don't dodge" warning must be present
    assert "dodge" in msg
    # The required action must be stated
    assert "non-meta claim" in msg


def test_narrative_evidence_still_marked_primary():
    """The existing 'NARRATIVE EVIDENCE' section remains intact — it
    shouldn't have been removed by the rule 4 rewrite."""
    msg = _BUNDLE_SYSTEM_PROMPT
    assert "NARRATIVE EVIDENCE" in msg
    assert "narrative WINS for the OBSERVATION" in msg


def test_grounding_hierarchy_preserved():
    """The three-tier confidence_basis hierarchy (process_priors,
    cross_run, schema_only) must remain — rule 4 changes shouldn't
    have collapsed it."""
    msg = _BUNDLE_SYSTEM_PROMPT
    assert "process_priors" in msg
    assert "cross_run" in msg
    assert "schema_only" in msg
    assert "GROUNDING HIERARCHY" in msg
