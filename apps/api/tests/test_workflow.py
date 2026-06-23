"""Workflow-selector plumbing: state, request model, and the optimization
result assembly's model log. No network — exercises the wiring, not a live run.
"""
from __future__ import annotations

import json
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


def test_data_oracle_off_without_observations(monkeypatch):
    """The data optimizer is available only when the bundle has real observations.
    LABS is never consulted on the API path (de-LABS) — even with LABS env vars set,
    a bundle with no observations.csv yields debate-only, never a LABS substitute."""
    from fermdocs_api.runner_pipeline import _data_oracle_available

    monkeypatch.setenv("FERMDOCS_OPTIMIZE_ORACLE", "labs")
    monkeypatch.setenv("FERMDOCS_OPTIMIZE_TRAIN", "/some/labs/train.csv")
    assert _data_oracle_available(Path("/tmp")) is False


def test_debate_only_path_shows_no_labs_model_card():
    """With no observations, the result is debate-only: it shows the debated levers
    and a plain note — NOT a LABS mechanistic model card (no mu_max / X,S,P,M,V)."""
    from types import SimpleNamespace

    from fermdocs_api.runner_pipeline import _assemble_optimization_output

    debate = SimpleNamespace(output=SimpleNamespace(
        final_hypotheses=[], debate_summary="levers found"))
    out = _assemble_optimization_output(Path("/tmp"), debate, "run-1")
    assert out["simulator_available"] is False
    blob = json.dumps(out["model_log"])
    assert "mu_max" not in blob and '"X"' not in blob  # no LABS equations leak
    assert any(e.get("kind") == "note" for e in out["model_log"])
