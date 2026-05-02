"""Sandboxed Python execution.

Subprocess-isolated pandas/numpy/scipy for the diagnosis agent. cwd is pinned
to the project root so `from fermdocs...` imports resolve (matches reference
repo). 50KB cap on the agent-facing return string; full fidelity preserved
in the trace via TraceWriter.

Plan ref: §4 of plans/2026-05-02-execute-python-default.md.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

from fermdocs_diagnose.audit.trace_writer import TraceWriter

PYTHON_DEFAULT_TIMEOUT = 120  # seconds
PYTHON_MAX_OUTPUT_BYTES = 50_000  # agent-facing cap
PYTHON_STREAM_TRACE_CAP = 1_000_000  # per-stream cap inside trace records (1MB)
PYTHON_RLIMIT_AS_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB virtual memory cap

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
"""Two levels up from src/fermdocs_diagnose/tools_bundle/ → repo root."""


@dataclass(frozen=True)
class ExecutePythonResult:
    """What the tool returns to the agent. The string form is what's appended
    to the conversation; the structured fields are recorded for diagnostics.
    """

    stdout: str  # post-cap, agent-facing
    stderr: str  # post-cap, agent-facing
    returncode: int
    timed_out: bool
    duration_ms: int

    def to_agent_text(self) -> str:
        """The literal text the agent sees as the tool result."""
        if self.timed_out:
            return f"Error: Code execution timed out after {self.duration_ms} ms."
        if self.returncode != 0:
            body = self.stderr or f"Process exited with code {self.returncode}"
            return body
        return self.stdout


def _truncate(s: str, cap: int, marker: str) -> str:
    if len(s.encode("utf-8")) <= cap:
        return s
    # Truncate by byte budget but keep utf-8 valid by trimming codepoints.
    encoded = s.encode("utf-8")[:cap]
    # back off until we have valid utf-8
    while encoded:
        try:
            decoded = encoded.decode("utf-8")
            return decoded + marker
        except UnicodeDecodeError:
            encoded = encoded[:-1]
    return marker


def _preexec_set_rlimit() -> None:
    """Run in the child process before exec to apply the virtual memory cap.

    Skipped on platforms without `resource` (e.g. Windows). On Linux/macOS
    this caps RLIMIT_AS so a runaway pandas allocation triggers MemoryError
    instead of OOM-killing other processes.
    """
    try:
        import resource

        resource.setrlimit(
            resource.RLIMIT_AS,
            (PYTHON_RLIMIT_AS_BYTES, PYTHON_RLIMIT_AS_BYTES),
        )
    except (ImportError, ValueError, OSError):
        # Some sandboxed envs forbid setrlimit; we'd rather run uncapped than fail.
        pass


def execute_python(
    code: str,
    *,
    timeout: int = PYTHON_DEFAULT_TIMEOUT,
    trace_writer: TraceWriter | None = None,
    cwd: Path | None = None,
) -> ExecutePythonResult:
    """Execute `code` in a fresh Python subprocess. Capture stdout/stderr.

    Args:
        code: Python source. Use `print()` for output — only stdout is shown.
        timeout: Wall-clock seconds before the subprocess is killed.
        trace_writer: optional sink for the full-fidelity trace record.
        cwd: override (default: project root).

    Returns:
        ExecutePythonResult with truncation already applied to stdout/stderr.
    """
    wrapped = textwrap.dedent(code)
    work_dir = str(cwd or _PROJECT_ROOT)
    t0 = time.time()
    timed_out = False
    raw_stdout = ""
    raw_stderr = ""
    returncode = 0

    try:
        proc = subprocess.run(
            [sys.executable, "-c", wrapped],
            cwd=work_dir,
            capture_output=True,
            timeout=timeout,
            text=True,
            preexec_fn=_preexec_set_rlimit if os.name == "posix" else None,
            check=False,
        )
        raw_stdout = proc.stdout or ""
        raw_stderr = proc.stderr or ""
        returncode = proc.returncode
    except subprocess.TimeoutExpired as e:
        timed_out = True
        returncode = -1
        raw_stdout = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        raw_stderr = e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")

    duration_ms = int((time.time() - t0) * 1000)

    agent_stdout = _truncate(raw_stdout, PYTHON_MAX_OUTPUT_BYTES, "\n... (output truncated at 50KB)")
    agent_stderr = _truncate(raw_stderr, PYTHON_MAX_OUTPUT_BYTES, "\n... (stderr truncated at 50KB)")

    if trace_writer is not None:
        trace_writer.write(
            {
                "kind": "python_call",
                "duration_ms": duration_ms,
                "returncode": returncode,
                "timed_out": timed_out,
                "code": wrapped,
                "stdout": _truncate(raw_stdout, PYTHON_STREAM_TRACE_CAP, "\n... (stream capped at 1MB in trace)"),
                "stderr": _truncate(raw_stderr, PYTHON_STREAM_TRACE_CAP, "\n... (stream capped at 1MB in trace)"),
            }
        )

    return ExecutePythonResult(
        stdout=agent_stdout,
        stderr=agent_stderr,
        returncode=returncode,
        timed_out=timed_out,
        duration_ms=duration_ms,
    )
