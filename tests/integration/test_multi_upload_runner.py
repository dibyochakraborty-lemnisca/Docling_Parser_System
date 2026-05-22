"""_prepare_bundle_dir branching for N=1 and N=3 (PR-A3 commit 2).

Plan ref: plans/2026-05-07-multi-file-upload-and-submit.md commit 2.

Regression-critical: the N=1 .zip and N=1 raw paths existed before
multi-file. We must not break them. The N=3 raw path is the new wing.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch, AsyncMock
import zipfile

import pytest

from apps.api.fermdocs_api import runner_pipeline
from apps.api.fermdocs_api.state import RunStore, RunStatus


def _arun(coro):
    return asyncio.run(coro)


def _store_with_run(tmp_path: Path):
    store = RunStore(uploads_root=tmp_path / "u", runs_root=tmp_path / "r")
    return store


# ---------- regression: N=1 .zip ----------


def test_prepare_bundle_n1_zip_routes_to_unzip(tmp_path: Path) -> None:
    """REGRESSION: N=1 .zip must still go through _unzip_bundle, not
    _build_bundle_from_raw. This was the existing behavior and we must
    preserve it after introducing list-based paths."""
    store = _store_with_run(tmp_path)

    # Build a real-ish bundle zip with a meta.json so _find_bundle_root works
    src_bundle = tmp_path / "src_bundle"
    src_bundle.mkdir()
    (src_bundle / "meta.json").write_text("{}")
    zip_bytes_buf = tmp_path / "build.zip"
    with zipfile.ZipFile(zip_bytes_buf, "w") as zf:
        zf.write(src_bundle / "meta.json", arcname="meta.json")
    upload = store.add_upload(
        files=[("bundle.zip", "application/zip", zip_bytes_buf.read_bytes())]
    )
    run = store.create_run(upload.upload_id)

    sentinel = tmp_path / "sentinel"

    with patch.object(
        runner_pipeline, "_unzip_bundle", return_value=sentinel
    ) as unzip_spy, patch.object(
        runner_pipeline, "_build_bundle_from_raw", new=AsyncMock()
    ) as build_spy:
        result = _arun(
            runner_pipeline._prepare_bundle_dir(
                upload=upload, store=store, run=run
            )
        )

    assert result == sentinel
    assert unzip_spy.call_count == 1
    assert build_spy.call_count == 0


# ---------- regression: N=1 raw csv ----------


def test_prepare_bundle_n1_csv_routes_to_build_from_raw(tmp_path: Path) -> None:
    """REGRESSION: N=1 .csv must still call _build_bundle_from_raw."""
    store = _store_with_run(tmp_path)
    upload = store.add_upload(files=[("d.csv", "text/csv", b"x")])
    run = store.create_run(upload.upload_id)

    sentinel = tmp_path / "build_sentinel"

    with patch.object(
        runner_pipeline, "_unzip_bundle"
    ) as unzip_spy, patch.object(
        runner_pipeline,
        "_build_bundle_from_raw",
        new=AsyncMock(return_value=sentinel),
    ) as build_spy:
        result = _arun(
            runner_pipeline._prepare_bundle_dir(
                upload=upload, store=store, run=run
            )
        )

    assert result == sentinel
    assert unzip_spy.call_count == 0
    assert build_spy.call_count == 1


# ---------- new: N=3 csv ----------


def test_prepare_bundle_n3_csv_routes_to_build_from_raw(tmp_path: Path) -> None:
    """N>1 raw upload routes through _build_bundle_from_raw with all
    paths in upload.paths. ingest expansion is verified separately."""
    store = _store_with_run(tmp_path)
    upload = store.add_upload(
        files=[
            ("r1.csv", "text/csv", b"a"),
            ("r2.csv", "text/csv", b"b"),
            ("r3.csv", "text/csv", b"c"),
        ]
    )
    run = store.create_run(upload.upload_id)

    sentinel = tmp_path / "multi_sentinel"

    with patch.object(
        runner_pipeline, "_unzip_bundle"
    ) as unzip_spy, patch.object(
        runner_pipeline,
        "_build_bundle_from_raw",
        new=AsyncMock(return_value=sentinel),
    ) as build_spy:
        result = _arun(
            runner_pipeline._prepare_bundle_dir(
                upload=upload, store=store, run=run
            )
        )

    assert result == sentinel
    assert unzip_spy.call_count == 0
    assert build_spy.call_count == 1
    # Same upload object passed in, so paths are intact.
    kwargs = build_spy.call_args.kwargs
    assert kwargs["upload"].paths == upload.paths
    assert len(kwargs["upload"].paths) == 3


# ---------- defense-in-depth: N>1 with a zip raises ----------


def test_prepare_bundle_multi_with_zip_raises(tmp_path: Path) -> None:
    """The API endpoint blocks zip+other at the boundary, but if a caller
    bypasses that path (programmatic use, tests), _prepare_bundle_dir
    must still refuse."""
    store = _store_with_run(tmp_path)
    # add_upload itself doesn't enforce extension rules — that's the API's
    # job — so we can construct a multi-file upload with a zip directly.
    upload = store.add_upload(
        files=[
            ("a.csv", "text/csv", b"x"),
            ("b.zip", "application/zip", b"\x50\x4b\x03\x04"),
        ]
    )
    run = store.create_run(upload.upload_id)

    with pytest.raises(ValueError, match="unsupported|zip"):
        _arun(
            runner_pipeline._prepare_bundle_dir(
                upload=upload, store=store, run=run
            )
        )


# ---------- new: N=3 ingest CLI gets three --files args ----------


def test_build_bundle_from_raw_n3_passes_three_files_args(tmp_path: Path) -> None:
    """The fermdocs ingest CLI is the only place where multi-file
    semantics actually land. Verify the subprocess invocation expands
    one --files <path> per upload file. We patch _run_subprocess to
    capture the cmd without running anything."""
    import os

    # _build_bundle_from_raw needs DATABASE_URL set or it raises early.
    os.environ["DATABASE_URL"] = "postgres://stub"
    try:
        store = _store_with_run(tmp_path)
        upload = store.add_upload(
            files=[
                ("a.csv", "text/csv", b"x"),
                ("b.csv", "text/csv", b"y"),
                ("c.csv", "text/csv", b"z"),
            ]
        )
        run = store.create_run(upload.upload_id)

        captured_cmds = []

        async def _capture(cmd, cwd=None):
            captured_cmds.append(cmd)
            # Stop after the first subprocess call (the ingest one) —
            # raise so we don't have to mock every downstream stage.
            raise RuntimeError("stop after ingest")

        with patch.object(
            runner_pipeline, "_run_subprocess", side_effect=_capture
        ):
            with pytest.raises(RuntimeError, match="stop after ingest"):
                _arun(
                    runner_pipeline._build_bundle_from_raw(
                        upload=upload, store=store, run=run
                    )
                )

        assert len(captured_cmds) == 1
        cmd = captured_cmds[0]
        # Find every --files arg pair and assert there are 3, in order
        files_args = [
            cmd[i + 1] for i, tok in enumerate(cmd) if tok == "--files"
        ]
        assert len(files_args) == 3
        assert all(p.endswith(".csv") for p in files_args)
        assert files_args[0].endswith("a.csv")
        assert files_args[1].endswith("b.csv")
        assert files_args[2].endswith("c.csv")
    finally:
        os.environ.pop("DATABASE_URL", None)
