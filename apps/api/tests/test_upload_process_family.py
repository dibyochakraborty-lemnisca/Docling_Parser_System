"""POST /api/uploads accepts process_family form field.

Branch: upload-process-family-ui.

Verifies the API contract the web UI dropdown depends on:
  - process_family form field is optional
  - empty / "auto-detect" / "unknown" normalise to None on the server
  - a valid canonical name is stored on the Upload record and surfaces
    in the response body
"""

from __future__ import annotations

import importlib
import io
import zipfile

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("FERMDOCS_API_ROOT", str(tmp_path))
    import fermdocs_api.main as main_mod
    importlib.reload(main_mod)
    return main_mod.app


def _csv() -> tuple[str, bytes, str]:
    return ("data.csv", b"time,biomass\n0,1.0\n", "text/csv")


def test_upload_without_process_family_defaults_to_none(app):
    client = TestClient(app)
    r = client.post(
        "/api/uploads",
        files={"files": _csv()},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["process_family"] is None


def test_upload_with_canonical_family_is_persisted(app):
    client = TestClient(app)
    r = client.post(
        "/api/uploads",
        files={"files": _csv()},
        data={"process_family": "penicillin_fedbatch"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["process_family"] == "penicillin_fedbatch"


@pytest.mark.parametrize("sentinel", [
    "",
    "auto-detect",
    "auto",
    "unknown",
])
def test_upload_with_sentinel_normalises_to_none(app, sentinel):
    client = TestClient(app)
    r = client.post(
        "/api/uploads",
        files={"files": _csv()},
        data={"process_family": sentinel},
    )
    assert r.status_code == 200
    assert r.json()["process_family"] is None


@pytest.mark.parametrize("family", [
    "penicillin_fedbatch",
    "yeast_intracellular_product_fedbatch",
    "yeast_aerobic_fedbatch",
    "ecoli_recombinant_protein",
    "melanin_batch",
])
def test_all_dropdown_options_round_trip(app, family):
    """Every option in PROCESS_FAMILY_OPTIONS (except auto-detect)
    persists through the API. Catches mismatch between frontend enum
    and backend acceptance."""
    client = TestClient(app)
    r = client.post(
        "/api/uploads",
        files={"files": _csv()},
        data={"process_family": family},
    )
    assert r.status_code == 200
    assert r.json()["process_family"] == family
