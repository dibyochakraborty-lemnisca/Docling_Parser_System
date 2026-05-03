"""Stage 3 gate: full hypothesis stage on carotenoid + IndPenSim (LIVE).

Gated by FERMDOCS_RUN_LIVE_TESTS=1.

Asserts the Stage 3 gate (per plans/2026-05-03-hypothesis-debate-v0.md §11):
  - End-to-end runs on BOTH bundles
  - Final hypothesis cites at least one finding/narrative/trajectory
  - Critic+judge cycle exercised at least once across the runs
  - Token report within max_total_input_tokens
  - global.md persisted with critic + judge events
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from fermdocs_hypothesis.bundle_loader import load_bundle
from fermdocs_hypothesis.events import CritiqueFiledEvent, JudgeRulingEvent
from fermdocs_hypothesis.live_hooks import LiveHooks
from fermdocs_hypothesis.runner import run_stage
from fermdocs_hypothesis.schema import BudgetSnapshot

CAROTENOID = Path("out/bundle_multi_20260502T220318Z_1a29e4")
INDPENSIM = Path("out/bundle_indpensim")
LIVE = os.environ.get("FERMDOCS_RUN_LIVE_TESTS") == "1"


def _run(bundle_dir: Path, tmp_path: Path):
    from dotenv import load_dotenv

    load_dotenv()
    loaded = load_bundle(bundle_dir)
    assert loaded.hyp_input.seed_topics, f"{bundle_dir} should have seed topics"

    # Use the production defaults (max_turns=10, max_critic_cycles_per_topic=3)
    # but cap input tokens so a runaway retry can't blow our wallet during tests.
    budget = BudgetSnapshot(
        max_total_input_tokens=200_000,
        max_tool_calls_total=80,
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
        validate=True,
        now_factory=lambda: datetime.now(timezone.utc),
    )
    return result, global_md


@pytest.mark.skipif(not LIVE, reason="set FERMDOCS_RUN_LIVE_TESTS=1 to run")
@pytest.mark.skipif(not CAROTENOID.exists(), reason="carotenoid bundle not present")
def test_stage3_e2e_carotenoid(tmp_path):
    result, global_md = _run(CAROTENOID, tmp_path)
    # Final OR rejected hypothesis exists (critic may have nuked the only one)
    total = len(result.output.final_hypotheses) + len(result.output.rejected_hypotheses)
    assert total >= 1, (
        f"expected ≥1 hypothesis (final or rejected); exit={result.state.exit_reason}, "
        f"events={[e.type for e in result.events]}"
    )

    # Critic and judge events must exist in the log
    critique_evs = [e for e in result.events if isinstance(e, CritiqueFiledEvent)]
    judge_evs = [e for e in result.events if isinstance(e, JudgeRulingEvent)]
    assert critique_evs, "critic must have filed at least one critique"
    assert judge_evs, "judge must have ruled at least once"

    # Token budget respected
    assert result.output.token_report.total_input <= 200_000

    # global.md persisted
    text = global_md.read_text()
    assert "critique_filed" in text
    assert "judge_ruling" in text

    print(f"\n[stage3-carotenoid] exit={result.state.exit_reason}")
    print(f"[stage3-carotenoid] final={len(result.output.final_hypotheses)} rejected={len(result.output.rejected_hypotheses)}")
    print(f"[stage3-carotenoid] critic_flag={critique_evs[0].flag} judge_valid={judge_evs[0].criticism_valid}")
    print(f"[stage3-carotenoid] tokens: in={result.output.token_report.total_input} out={result.output.token_report.total_output}")
    if result.output.final_hypotheses:
        h = result.output.final_hypotheses[0]
        print(f"[stage3-carotenoid] H={h.hyp_id}: {h.summary[:200]}")


@pytest.mark.skipif(not LIVE, reason="set FERMDOCS_RUN_LIVE_TESTS=1 to run")
@pytest.mark.skipif(not INDPENSIM.exists(), reason="indpensim bundle not present (run scripts/build_indpensim_bundle.py)")
def test_stage3_e2e_indpensim(tmp_path):
    result, global_md = _run(INDPENSIM, tmp_path)
    total = len(result.output.final_hypotheses) + len(result.output.rejected_hypotheses)
    assert total >= 1
    critique_evs = [e for e in result.events if isinstance(e, CritiqueFiledEvent)]
    judge_evs = [e for e in result.events if isinstance(e, JudgeRulingEvent)]
    assert critique_evs
    assert judge_evs
    assert result.output.token_report.total_input <= 200_000

    print(f"\n[stage3-indpensim] exit={result.state.exit_reason}")
    print(f"[stage3-indpensim] final={len(result.output.final_hypotheses)} rejected={len(result.output.rejected_hypotheses)}")
    print(f"[stage3-indpensim] critic_flag={critique_evs[0].flag} judge_valid={judge_evs[0].criticism_valid}")
    print(f"[stage3-indpensim] tokens: in={result.output.token_report.total_input} out={result.output.token_report.total_output}")
    if result.output.final_hypotheses:
        h = result.output.final_hypotheses[0]
        print(f"[stage3-indpensim] H={h.hyp_id}: {h.summary[:200]}")
