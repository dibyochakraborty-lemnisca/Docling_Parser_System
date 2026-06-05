"""Workflow-selector plumbing: state, request model, and the optimization
result assembly's model log. No network — exercises the wiring, not a live run.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from fermdocs_api.main import CreateRunRequest
from fermdocs_api.state import RunStore, WorkflowKind


def _store() -> RunStore:
    d = Path(tempfile.mkdtemp())
    return RunStore(uploads_root=d / "u", runs_root=d / "r")


def test_create_run_defaults_to_diagnostic():
    run = _store().create_run("up-1")
    assert run.workflow == WorkflowKind.DIAGNOSTIC


def test_create_run_stores_optimization_workflow():
    run = _store().create_run("up-1", workflow=WorkflowKind.OPTIMIZATION)
    assert run.workflow == WorkflowKind.OPTIMIZATION
    assert run.optimization_output is None  # not run yet


def test_request_model_defaults_and_parses_workflow():
    assert CreateRunRequest(upload_id="x").workflow == WorkflowKind.DIAGNOSTIC
    body = CreateRunRequest(upload_id="x", workflow="optimization")
    assert body.workflow == WorkflowKind.OPTIMIZATION


def test_simulator_gate_is_off_without_config(monkeypatch):
    """The closed-loop optimizer stays gated off unless a real simulator is
    configured — the honest default (we never optimize against a fake oracle)."""
    from fermdocs_api.runner_pipeline import _optimizer_simulator_available

    monkeypatch.delenv("FERMDOCS_OPTIMIZE_MECH_PARAMS", raising=False)
    assert _optimizer_simulator_available(Path("/tmp")) is False


def test_model_card_is_shown_in_debate_only_path():
    """Even with no simulator, the optimization result carries the governing
    equations so the UI can show how the agent uses the model."""
    from types import SimpleNamespace

    from fermdocs_api.runner_pipeline import _assemble_optimization_output

    debate = SimpleNamespace(output=SimpleNamespace(
        final_hypotheses=[], debate_summary="levers found"))
    out = _assemble_optimization_output(Path("/tmp"), debate, "run-1")
    assert out["simulator_available"] is False
    kinds = [e["kind"] for e in out["model_log"]]
    assert "equations" in kinds  # the model card is present
    assert any("mu_max" in eq for e in out["model_log"]
               if e["kind"] == "equations" for eq in e["equations"])
