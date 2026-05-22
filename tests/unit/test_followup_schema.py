"""Schema additions for PR-A2 (hitl-followup, drive posture).

Plan ref: plans/2026-05-05-hitl-followup.md commit 1.

Covers:
  - FinalHypothesis.parent_hypothesis_ids field default + population
  - HypothesisFull does NOT carry parent_hypothesis_ids (mid-debate has no
    consumer that needs it; only the final-output projection does)
  - Run.followups + Run.followup_index defaults
  - FollowupResult dataclass shape
  - RunStore.add_followup append behavior
  - Run.bundle_followup_eligible computed property
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from apps.api.fermdocs_api.state import (
    FollowupResult,
    Run,
    RunStatus,
    RunStore,
)
from fermdocs_hypothesis.schema import FinalHypothesis, HypothesisFull


# ---------- FinalHypothesis.parent_hypothesis_ids ----------


def _final_hyp_kwargs(**overrides):
    base = dict(
        hyp_id="H-0001",
        summary="placeholder",
        facet_ids=["FCT-0001"],
        cited_finding_ids=["F-0001"],
        confidence=0.6,
        confidence_basis="schema_only",
        critic_flag="green",
        judge_ruled_criticism_valid=False,
    )
    base.update(overrides)
    return base


def test_final_hypothesis_parent_ids_default_empty() -> None:
    h = FinalHypothesis(**_final_hyp_kwargs())
    assert h.parent_hypothesis_ids == []


def test_final_hypothesis_parent_ids_populated() -> None:
    h = FinalHypothesis(
        **_final_hyp_kwargs(parent_hypothesis_ids=["H-0003", "H-0005"])
    )
    assert h.parent_hypothesis_ids == ["H-0003", "H-0005"]


def test_final_hypothesis_parent_ids_frozen() -> None:
    """Schema is frozen; can't mutate parent list after construction."""
    h = FinalHypothesis(**_final_hyp_kwargs(parent_hypothesis_ids=["H-0003"]))
    with pytest.raises((TypeError, ValueError)):
        h.parent_hypothesis_ids = ["H-0099"]  # type: ignore[misc]


def test_hypothesis_full_does_not_have_parent_ids() -> None:
    """Mid-debate, no agent needs to read parent_hypothesis_ids — it lives
    only on FinalHypothesis. Pinning that decision so we'd notice if
    someone copy-pasted the field over."""
    assert "parent_hypothesis_ids" not in HypothesisFull.model_fields


# ---------- Run + FollowupResult ----------


def test_run_followups_default_empty() -> None:
    r = Run(run_id="r1", upload_id="u1")
    assert r.followups == []
    assert r.followup_index == 0


def test_followup_result_shape() -> None:
    fr = FollowupResult(
        followup_index=1,
        user_question_text="What about RUN-0002?",
        output={"meta": "stub"},  # HypothesisOutput-shaped, loose Any
    )
    assert fr.followup_index == 1
    assert fr.user_question_text == "What about RUN-0002?"
    assert isinstance(fr.created_at, datetime)
    assert fr.created_at.tzinfo is timezone.utc


def test_run_store_add_followup_appends(tmp_path: Path) -> None:
    store = RunStore(uploads_root=tmp_path / "u", runs_root=tmp_path / "r")
    upload = store.add_upload(files=[("t.csv", "text/csv", b"x")])
    run = store.create_run(upload.upload_id)

    fr1 = FollowupResult(followup_index=1, user_question_text="q1", output=None)
    fr2 = FollowupResult(followup_index=2, user_question_text="q2", output=None)
    store.add_followup(run.run_id, fr1)
    store.add_followup(run.run_id, fr2)

    assert [f.followup_index for f in run.followups] == [1, 2]
    assert [f.user_question_text for f in run.followups] == ["q1", "q2"]


def test_run_store_add_followup_unknown_run_raises(tmp_path: Path) -> None:
    store = RunStore(uploads_root=tmp_path / "u", runs_root=tmp_path / "r")
    fr = FollowupResult(followup_index=1, user_question_text="q", output=None)
    with pytest.raises(KeyError):
        store.add_followup("nonexistent", fr)


# ---------- bundle_followup_eligible ----------


def test_bundle_followup_eligible_false_when_no_bundle() -> None:
    r = Run(run_id="r1", upload_id="u1", status=RunStatus.DONE)
    assert r.bundle_followup_eligible is False


def test_bundle_followup_eligible_false_when_bundle_path_missing(
    tmp_path: Path,
) -> None:
    r = Run(
        run_id="r1",
        upload_id="u1",
        status=RunStatus.DONE,
        bundle_dir=tmp_path / "does-not-exist",
    )
    assert r.bundle_followup_eligible is False


def test_bundle_followup_eligible_true_when_bundle_present(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    r = Run(
        run_id="r1",
        upload_id="u1",
        status=RunStatus.DONE,
        bundle_dir=bundle,
    )
    assert r.bundle_followup_eligible is True
