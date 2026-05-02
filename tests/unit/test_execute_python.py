"""Stage 2: execute_python sandbox tests.

Covers: success path, timeout, non-zero exit, stdout overflow with truncation,
trace fidelity (record + spillover), import-from-codebase works (cwd is project root).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from fermdocs_diagnose.audit.trace_writer import SPILL_THRESHOLD_BYTES, TraceWriter
from fermdocs_diagnose.tools_bundle.execute_python import (
    PYTHON_MAX_OUTPUT_BYTES,
    execute_python,
)


def test_success_returns_stdout(tmp_path: Path) -> None:
    res = execute_python('print("hello world")')
    assert res.returncode == 0
    assert res.timed_out is False
    assert res.stdout.strip() == "hello world"
    assert res.stderr == ""


def test_timeout_kills_subprocess() -> None:
    res = execute_python("import time; time.sleep(5)", timeout=1)
    assert res.timed_out is True
    assert res.returncode == -1
    assert "timed out" in res.to_agent_text().lower()


def test_nonzero_exit_returns_stderr() -> None:
    res = execute_python('import sys; sys.stderr.write("boom\\n"); sys.exit(2)')
    assert res.returncode == 2
    assert "boom" in res.stderr
    # to_agent_text should surface stderr
    assert "boom" in res.to_agent_text()


def test_stdout_overflow_is_truncated() -> None:
    # 200KB of output, agent-facing cap is 50KB.
    code = (
        "data = 'x' * (200 * 1024)\n"
        "print(data)\n"
    )
    res = execute_python(code)
    assert res.returncode == 0
    encoded = res.stdout.encode("utf-8")
    # truncation marker is appended; total stays close to but above the cap
    # (cap is on the prefix; marker adds a small tail).
    assert len(encoded) <= PYTHON_MAX_OUTPUT_BYTES + 200
    assert "truncated at 50KB" in res.stdout


def test_trace_writer_captures_full_record(tmp_path: Path) -> None:
    tw = TraceWriter(tmp_path / "python_trace.jsonl")
    res = execute_python("print(7 + 8)", trace_writer=tw)
    assert "15" in res.stdout
    lines = (tmp_path / "python_trace.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["kind"] == "python_call"
    assert record["returncode"] == 0
    assert "print(7 + 8)" in record["code"]
    assert "15" in record["stdout"]


def test_trace_writer_spills_oversized_records(tmp_path: Path) -> None:
    tw = TraceWriter(tmp_path / "python_trace.jsonl")
    # Generate stdout that, combined with the code, exceeds the spill threshold.
    big = SPILL_THRESHOLD_BYTES + 5_000
    code = f'print("y" * {big})'
    execute_python(code, trace_writer=tw)
    lines = (tmp_path / "python_trace.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    pointer = json.loads(lines[0])
    assert "spilled_to" in pointer
    spill_dir = tmp_path / "python_trace_spill"
    assert spill_dir.is_dir()
    spilled_files = list(spill_dir.glob("*.json"))
    assert len(spilled_files) == 1
    full = json.loads(spilled_files[0].read_text())
    assert full["kind"] == "python_call"
    assert "yyyy" in full["stdout"]


def test_codebase_import_works_from_sandbox() -> None:
    """cwd=project root, so `from fermdocs_characterize...` should resolve."""
    res = execute_python(
        "from fermdocs_characterize.schema import Tier\n"
        "print(Tier.A.value)\n"
    )
    assert res.returncode == 0, f"stderr={res.stderr!r}"
    assert res.stdout.strip() == "A"
