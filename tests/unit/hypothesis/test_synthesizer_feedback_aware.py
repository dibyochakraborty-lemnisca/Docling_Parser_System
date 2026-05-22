"""Lock in the feedback-loop wording in synthesizer/critic/judge prompts.

The carotenoid loop (H-001/2/3 all rejected for "documented absence ≠
ruled out") is exactly what these prompts now address: the synthesizer
sees its prior attempts and is told to address each critic_reason
explicitly. The critic is told not to re-raise objections that have
been addressed. The judge is told to be consistent across retries.

These tests assert the wording is present so prompt re-flowing or
future edits can't silently regress the feedback loop.
"""

from __future__ import annotations

import re

from fermdocs_hypothesis.agents.critic import CRITIC_INVARIANTS, CRITIC_SYSTEM
from fermdocs_hypothesis.agents.judge import JUDGE_INVARIANTS
from fermdocs_hypothesis.agents.synthesizer import (
    SYNTHESIZER_INVARIANTS,
    SYNTHESIZER_SYSTEM,
)


def _flat(s: str) -> str:
    return re.sub(r"\s+", " ", s)


# ---------- synthesizer ----------


def test_synthesizer_system_addresses_previous_attempts():
    flat = _flat(SYNTHESIZER_SYSTEM)
    assert "previous_attempts" in flat
    # Critical rule: address each prior reason explicitly
    assert "address" in flat.lower() and "critic_reason" in flat
    # Anti-paraphrase guard
    assert "paraphrase" in flat.lower() or "rephras" in flat.lower()


def test_synthesizer_system_references_cross_topic_lessons():
    flat = _flat(SYNTHESIZER_SYSTEM)
    assert "cross_topic_lessons" in flat
    assert "standing rule" in flat.lower() or "standing rules" in flat.lower()


def test_synthesizer_system_offers_narrowing_as_escape():
    """The escape valve from the rejection loop: narrow the hypothesis
    when you can't address a reason without speculating."""
    flat = _flat(SYNTHESIZER_SYSTEM)
    assert "NARROW" in flat or "Narrow" in flat or "narrow" in flat.lower()


def test_synthesizer_invariants_include_feedback_rule():
    flat = " ".join(SYNTHESIZER_INVARIANTS)
    assert "previous_attempts" in flat
    assert "cross_topic_lessons" in flat


# ---------- critic ----------


def test_critic_system_warns_against_repeating_addressed_objections():
    flat = _flat(CRITIC_SYSTEM)
    # Must mention that previously-flagged reasons may now be fixed
    assert "previous_attempts" in flat
    # Anti-loop rule: don't re-raise an addressed objection
    assert "loop" in flat.lower() or "addressed" in flat.lower()


def test_critic_invariants_call_out_iterative_narrowing():
    flat = " ".join(CRITIC_INVARIANTS)
    assert "previous_attempts" in flat
    # Iterative narrowing should be welcomed, not punished
    assert "narrow" in flat.lower() or "iterative" in flat.lower()


# ---------- judge ----------


def test_judge_invariants_demand_consistency_across_retries():
    flat = " ".join(JUDGE_INVARIANTS)
    assert "previous_attempts" in flat
    # Consistency: if you ruled valid before but synth narrowed, rule invalid now
    assert "consistency" in flat.lower() or "consistent" in flat.lower()
