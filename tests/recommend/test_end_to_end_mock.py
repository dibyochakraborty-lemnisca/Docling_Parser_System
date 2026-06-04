import pytest
from pathlib import Path
import asyncio
import os

from fermdocs_api.state import RunStore, Run, RunStatus
from fermdocs_api.runner_pipeline import _try_recommendation

def test_try_recommendation(tmp_path, monkeypatch):
    # Use the fake provider to bypass the API key
    monkeypatch.setenv("FERMDOCS_RECOMMEND_PROVIDER", "fake")
    store = RunStore(uploads_root=tmp_path / "uploads", runs_root=tmp_path / "runs")
    run = store.create_run(upload_id="up_123")
    run.status = RunStatus.DONE
    
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "meta.json").write_text('{"bundle_schema_version": "1.0", "golden_schema_version": "1.0", "bundle_id": "test", "run_ids": ["RUN-001"], "model_labels": {}, "flags": {}, "pipeline_version": "0.1", "created_at": "2024-01-01T00:00:00Z"}')
    
    async def run_test():
        await _try_recommendation(store, run, bundle_dir)
        
    asyncio.run(run_test())
    
    assert run.status == RunStatus.DONE
    assert (run.recommend_dir / "recommendation.json").exists()
    
    import json
    data = json.loads((run.recommend_dir / "recommendation.json").read_text())
    assert data["recommended_model"] == "none"
    assert data["confident"] is False
    assert data["refusal_reason"] == "stage_error"
