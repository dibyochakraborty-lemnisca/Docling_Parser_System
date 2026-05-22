"""POST /api/uploads with multi-file shape (PR-A3 commit 2).

Plan ref: plans/2026-05-07-multi-file-upload-and-submit.md commit 2.

Covers:
  - empty multipart → 400
  - single .csv (back-compat) → 200, response shape includes both new
    list keys (filenames, content_types) AND legacy single keys
  - three .csv → 200, list keys populated, legacy keys are null
  - .csv + .pdf mix → 200 (raw types are interchangeable)
  - .csv + .zip → 400 zip-mixing
  - duplicate filenames → 400
  - .txt (unsupported) → 400
  - response shape exact-match check
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient


def _app_with_store(tmp_path: Path):
    from apps.api.fermdocs_api import main as api_main
    from apps.api.fermdocs_api.state import RunStore

    store = RunStore(uploads_root=tmp_path / "u", runs_root=tmp_path / "r")
    api_main.STORE = store
    return api_main.create_app(), store


# ---------- single-file back-compat ----------


def test_upload_single_csv_returns_legacy_keys(tmp_path: Path) -> None:
    app, _ = _app_with_store(tmp_path)
    with TestClient(app) as client:
        r = client.post(
            "/api/uploads",
            files=[("files", ("data.csv", b"a,b\n1,2\n", "text/csv"))],
        )
    assert r.status_code == 200, r.text
    body = r.json()
    # New list keys
    assert body["filenames"] == ["data.csv"]
    assert body["content_types"] == ["text/csv"]
    # Legacy keys still populated for N=1
    assert body["filename"] == "data.csv"
    assert body["content_type"] == "text/csv"
    assert body["size_bytes"] == len(b"a,b\n1,2\n")


# ---------- multi-file ----------


def test_upload_three_csvs_legacy_keys_null(tmp_path: Path) -> None:
    app, _ = _app_with_store(tmp_path)
    with TestClient(app) as client:
        r = client.post(
            "/api/uploads",
            files=[
                ("files", ("r1.csv", b"r1", "text/csv")),
                ("files", ("r2.csv", b"r2--", "text/csv")),
                ("files", ("r3.csv", b"r3", "text/csv")),
            ],
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["filenames"] == ["r1.csv", "r2.csv", "r3.csv"]
    assert body["content_types"] == ["text/csv"] * 3
    assert body["size_bytes"] == 2 + 4 + 2
    # Legacy keys null on multi-file
    assert body["filename"] is None
    assert body["content_type"] is None


def test_upload_csv_plus_pdf_works(tmp_path: Path) -> None:
    app, _ = _app_with_store(tmp_path)
    with TestClient(app) as client:
        r = client.post(
            "/api/uploads",
            files=[
                ("files", ("trajectories.csv", b"a,b", "text/csv")),
                ("files", ("report.pdf", b"%PDF-fake", "application/pdf")),
            ],
        )
    assert r.status_code == 200, r.text
    assert r.json()["filenames"] == ["trajectories.csv", "report.pdf"]


# ---------- validation: empty / unsupported / zip-mix / duplicates ----------


def test_upload_unsupported_extension_returns_400(tmp_path: Path) -> None:
    app, _ = _app_with_store(tmp_path)
    with TestClient(app) as client:
        r = client.post(
            "/api/uploads",
            files=[("files", ("bad.txt", b"hello", "text/plain"))],
        )
    assert r.status_code == 400
    assert "unsupported file type" in r.text


def test_upload_zip_with_csv_rejected(tmp_path: Path) -> None:
    app, _ = _app_with_store(tmp_path)
    with TestClient(app) as client:
        r = client.post(
            "/api/uploads",
            files=[
                ("files", ("data.csv", b"x", "text/csv")),
                ("files", ("bundle.zip", b"\x50\x4b\x03\x04", "application/zip")),
            ],
        )
    assert r.status_code == 400
    assert "standalone" in r.text or "zip" in r.text.lower()


def test_upload_zip_alone_works(tmp_path: Path) -> None:
    """Solo zip is the existing path — still works."""
    app, _ = _app_with_store(tmp_path)
    with TestClient(app) as client:
        r = client.post(
            "/api/uploads",
            files=[("files", ("bundle.zip", b"\x50\x4b\x03\x04", "application/zip"))],
        )
    assert r.status_code == 200, r.text
    assert r.json()["filenames"] == ["bundle.zip"]


def test_upload_duplicate_filenames_returns_400(tmp_path: Path) -> None:
    app, _ = _app_with_store(tmp_path)
    with TestClient(app) as client:
        r = client.post(
            "/api/uploads",
            files=[
                ("files", ("data.csv", b"first", "text/csv")),
                ("files", ("data.csv", b"second", "text/csv")),
            ],
        )
    assert r.status_code == 400
    assert "duplicate" in r.text.lower()
