"""Bundle-aware tools for the diagnosis ReAct loop.

Stage 2 surface (plans/2026-05-02-execute-python-default.md §3):
  - execute_python(code, timeout=120) — sandboxed pandas/numpy/scipy
  - list_runs() — bundle's run_ids
  - get_meta() — bundle metadata
  - get_findings(...) — filtered access to deterministic findings
  - get_specs(variable) — schema spec for a variable
  - get_timecourse(run_id, variable, ...) — bundle trajectory slice
  - submit_diagnosis(payload) — synthetic terminator

Tools are closure-curried over a BundleReader / BundleWriter via
`make_diagnosis_tools(reader, writer)`. The agent sees clean signatures; the
factory holds the bundle handles.
"""

from fermdocs_diagnose.tools_bundle.execute_python import (
    PYTHON_DEFAULT_TIMEOUT,
    PYTHON_MAX_OUTPUT_BYTES,
    PYTHON_RLIMIT_AS_BYTES,
    ExecutePythonResult,
    execute_python,
)
from fermdocs_diagnose.tools_bundle.factory import (
    DiagnosisToolBundle,
    ToolError,
    make_diagnosis_tools,
)

__all__ = [
    "DiagnosisToolBundle",
    "ExecutePythonResult",
    "PYTHON_DEFAULT_TIMEOUT",
    "PYTHON_MAX_OUTPUT_BYTES",
    "PYTHON_RLIMIT_AS_BYTES",
    "ToolError",
    "execute_python",
    "make_diagnosis_tools",
]
