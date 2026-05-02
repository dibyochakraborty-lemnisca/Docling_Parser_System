"""Diagnose audit subsystem.

Owns the WRITE side of `bundle/audit/`. Nothing in this package is read at
runtime — see `scripts/check_audit_invariant.py`.

Public:
- TraceWriter — append-only jsonl with spillover for >100KB records
"""

from fermdocs_diagnose.audit.trace_writer import (
    SPILL_THRESHOLD_BYTES,
    TraceRecord,
    TraceWriter,
)

__all__ = ["SPILL_THRESHOLD_BYTES", "TraceRecord", "TraceWriter"]
