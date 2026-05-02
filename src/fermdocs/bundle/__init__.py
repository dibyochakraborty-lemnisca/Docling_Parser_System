"""File-based stage handoff between characterize and diagnose.

A bundle is a directory under `out/` that holds the inputs and outputs of one
end-to-end run. Stages read from and write to it; nothing is passed in memory.

See plans/2026-05-02-execute-python-default.md §2 for the layout and invariants.

Public API:
- BundleMeta — Pydantic model for `meta.json`
- BundleWriter — atomic temp+rename writer
- BundleReader — cached read-only accessor
- BundleSchemaMismatch / GoldenSchemaMajorMismatch — versioning errors

Audit invariant: nothing under `bundle/audit/` is read at runtime. Enforced by
CI guard (see scripts/check_audit_invariant.py).
"""

from fermdocs.bundle.meta import (
    BUNDLE_SCHEMA_VERSION,
    BundleMeta,
    BundleSchemaMismatch,
    GoldenSchemaMajorMismatch,
)
from fermdocs.bundle.reader import BundleReader, BundleNotReady
from fermdocs.bundle.writer import BundleWriter

__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "BundleMeta",
    "BundleNotReady",
    "BundleReader",
    "BundleSchemaMismatch",
    "BundleWriter",
    "GoldenSchemaMajorMismatch",
]
