"""Cross-run comparative intervention engine — back-compat shim.

The engine now lives in ``fermdocs.analysis.cross_run`` so both the recommend
stage and the optimize debate can import it without creating a
``debate -> recommend`` package dependency. This module re-exports the public
surface so existing imports (``from fermdocs_recommend import cross_run``;
``cross_run.analyze(...)``) keep working unchanged.
"""

from __future__ import annotations

from fermdocs.analysis.cross_run import (  # noqa: F401
    DEFAULT_OBJECTIVE,
    MIN_EFFECT_FRAC,
    MIN_RUNS,
    analyze,
    lever_effects,
    run_outcomes,
)

__all__ = [
    "DEFAULT_OBJECTIVE",
    "MIN_EFFECT_FRAC",
    "MIN_RUNS",
    "analyze",
    "lever_effects",
    "run_outcomes",
]
