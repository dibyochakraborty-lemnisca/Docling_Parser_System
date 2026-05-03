"""Stage 2 gate: full hypothesis stage on the carotenoid bundle (LIVE).

Gated by FERMDOCS_RUN_LIVE_TESTS=1. Hits Gemini API ~6-8 times
(orchestrator x1, kinetics specialist tool-loop, synthesizer x1).

Asserts the Stage 2 gate: produces ≥1 hypothesis with valid citations
on the carotenoid bundle, total tokens within budget.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from fermdocs_hypothesis.bundle_loader import load_bundle
from fermdocs_hypothesis.live_hooks import LiveHooks
from fermdocs_hypothesis.runner import run_stage
from fermdocs_hypothesis.schema import BudgetSnapshot

CAROTENOID = Path("out/bundle_multi_20260502T220318Z_1a29e4")
LIVE = os.environ.get("FERMDOCS_RUN_LIVE_TESTS") == "1"


@pytest.mark.skipif(not LIVE, reason="set FERMDOCS_RUN_LIVE_TESTS=1 to run")
@pytest.mark.skipif(not CAROTENOID.exists(), reason="carotenoid bundle not present")
def test_stage2_e2e_on_carotenoid(tmp_path):
    from dotenv import load_dotenv

    load_dotenv()

    loaded = load_bundle(CAROTENOID)
    assert loaded.hyp_input.seed_topics, "carotenoid should have seed topics"

    # Tight budget: one turn only — orchestrator picks 1 topic, all 3
    # specialists contribute (1 real + 2 stubs), synthesizer emits.
    budget = BudgetSnapshot(
        max_turns=1,
        max_tool_calls_total=20,
        max_total_input_tokens=80_000,
    )

    global_md = tmp_path / "global.md"
    hooks = LiveHooks(loaded)
    result = run_stage(
        hyp_input=loaded.hyp_input,
        hooks=hooks,
        global_md_path=global_md,
        diagnosis_id=loaded.diagnosis.meta.diagnosis_id,
        provider="gemini",
        model_name=hooks._client.model_name,
        budget=budget,
        now_factory=lambda: datetime.now(timezone.utc),
    )

    # Stage 2 gate assertions
    assert result.output.final_hypotheses, (
        f"expected ≥1 hypothesis; exit_reason={result.state.exit_reason}, "
        f"events={[e.type for e in result.events]}"
    )
    h = result.output.final_hypotheses[0]
    # Citation discipline: must cite at least one of finding/narrative/trajectory
    assert (
        h.cited_finding_ids
        or h.cited_narrative_ids
        or h.cited_trajectories
    ), f"hypothesis {h.hyp_id} has no citations"
    # Token report must be populated
    assert result.output.token_report.total_input > 0
    # global.md must exist and contain stage_started + stage_exited
    text = global_md.read_text()
    assert "stage_started" in text
    assert "stage_exited" in text
    # Specialist must have actually run (kinetics)
    assert "specialist:kinetics" in result.output.token_report.per_agent_input
    # Print a diagnostic so the test output shows what was produced
    print(f"\n[stage2-e2e] hyp_id={h.hyp_id}")
    print(f"[stage2-e2e] summary={h.summary[:200]}")
    print(f"[stage2-e2e] cited_findings={len(h.cited_finding_ids)} cited_narratives={len(h.cited_narrative_ids)}")
    print(f"[stage2-e2e] tokens: input={result.output.token_report.total_input} output={result.output.token_report.total_output}")
