"""Diagnose-stage user_question wiring (PR-A commit 5).

Two narrow surfaces:
  1. AgentContext.user_question is rendered into the JSON blob the
     agent reads as its prompt prefix (when present, key 'user_question'
     appears at the top; when None, key is absent — preserves byte-
     identical legacy blob shape).
  2. The diagnose system prompt teaches the agent how to read it: the
     emit-≥1-claim contract, shape-specific routing, summary mention rule.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import UUID

from fermdocs.domain.user_question import UserQuestion
from fermdocs_characterize.agent_context import (
    AgentContext,
    serialize_for_agent,
)
from fermdocs_characterize.schema import CharacterizationOutput, Meta
from fermdocs_diagnose.agent import _BUNDLE_SYSTEM_PROMPT


CHAR_ID = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")


def _empty_output() -> CharacterizationOutput:
    return CharacterizationOutput(
        meta=Meta(
            schema_version="1.0",
            characterization_version="0.1.0",
            characterization_id=CHAR_ID,
            generation_timestamp=datetime(2026, 5, 5, tzinfo=timezone.utc),
            source_dossier_ids=["dossier-test"],
        ),
        findings=[],
        narrative_observations=[],
        trajectories=[],
    )


# ---------- blob shape ----------


def test_blob_omits_user_question_key_when_none() -> None:
    """Back-compat: legacy bundles produce identical blob shape — no
    'user_question' key in the JSON, prompt prefix cache stays warm."""
    ctx = AgentContext(process={}, schema_version="1.0", user_question=None)
    rendered = serialize_for_agent(ctx, _empty_output())
    blob = json.loads(rendered)
    assert "user_question" not in blob


def test_blob_renders_user_question_at_top_when_present() -> None:
    """When the user typed a question at run start, the blob carries it
    as the FIRST key so the agent reads it before everything else."""
    q = UserQuestion(
        text="Why did RUN-0002 plateau early?",
        shape="scoping",
        affected_runs=["RUN-0002"],
        affected_variables=["biomass_g_l"],
    )
    ctx = AgentContext(process={}, schema_version="1.0", user_question=q)
    rendered = serialize_for_agent(ctx, _empty_output())
    blob = json.loads(rendered)
    assert "user_question" in blob
    assert blob["user_question"]["text"] == "Why did RUN-0002 plateau early?"
    assert blob["user_question"]["shape"] == "scoping"
    assert blob["user_question"]["affected_runs"] == ["RUN-0002"]
    # Stable top-level ordering: user_question is the first key.
    assert list(blob.keys())[0] == "user_question"


def test_blob_carries_all_user_question_fields() -> None:
    q = UserQuestion(
        text="Compare BATCH-04 to BATCH-05",
        shape="comparative",
        affected_runs=["RUN-0004", "RUN-0005"],
        affected_variables=["wcw_g_l"],
        raised_by="user",
    )
    ctx = AgentContext(process={}, schema_version="1.0", user_question=q)
    rendered = serialize_for_agent(ctx, _empty_output())
    blob = json.loads(rendered)
    uq = blob["user_question"]
    assert uq["text"] == "Compare BATCH-04 to BATCH-05"
    assert uq["shape"] == "comparative"
    assert sorted(uq["affected_runs"]) == ["RUN-0004", "RUN-0005"]
    assert uq["affected_variables"] == ["wcw_g_l"]
    assert uq["raised_by"] == "user"


# ---------- prompt content ----------


def test_diagnose_prompt_mentions_user_question_section() -> None:
    assert "USER QUESTION" in _BUNDLE_SYSTEM_PROMPT
    assert "user_question" in _BUNDLE_SYSTEM_PROMPT


def test_diagnose_prompt_states_emit_at_least_one_claim_contract() -> None:
    """The contract: when user_question is non-null, agent MUST emit at
    least one non-meta claim or open_question addressing it."""
    assert "MUST include" in _BUNDLE_SYSTEM_PROMPT
    assert "CONTRACT VIOLATION" in _BUNDLE_SYSTEM_PROMPT
    # Specifically, the contract is anchored on the user_question presence.
    assert (
        "Empty output while a\n  user_question is present is a CONTRACT VIOLATION"
        in _BUNDLE_SYSTEM_PROMPT
    )


def test_diagnose_prompt_covers_all_four_shapes() -> None:
    for shape in ("scoping", "mechanistic", "comparative", "open"):
        assert shape in _BUNDLE_SYSTEM_PROMPT, f"prompt missing guidance for shape={shape}"


def test_diagnose_prompt_tells_agent_to_mention_question_in_summary() -> None:
    """UI surfaces 'we addressed your question' by grep'ing the claim
    summary for the question text. Prompt must instruct the agent to
    write the question-addressing claim's summary that way."""
    # Looser substring — we just need the agent told to put question
    # text or run/variable in the summary.
    assert "summary" in _BUNDLE_SYSTEM_PROMPT
    assert "question text" in _BUNDLE_SYSTEM_PROMPT


def test_diagnose_prompt_legacy_back_compat() -> None:
    """The user-question section starts with 'when AgentContext.user_question
    is non-null' — that's the gate. Legacy runs (user_question=None) skip
    the section entirely. This test just confirms the gate phrasing is
    present so legacy runs aren't accidentally rule-bound."""
    assert "non-null" in _BUNDLE_SYSTEM_PROMPT
