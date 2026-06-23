"""LABS benchmark backend — OPT-IN, synthetic-data only.

This package holds the LABS process simulator: a synthetic ground-truth oracle
for the lactic-acid (LABS) model, used for benchmarking the optimizer against a
known answer. It is NEVER used on the API run path — real uploaded data always
wins there (de-LABS decision, 2026-06-16). It is reachable only via the
standalone optimize CLIs on synthetic input.

The data-native optimizer (``fermdocs_optimize.data_equation`` /
``fermdocs_optimize.discovery.general_mech`` / ``fermdocs_optimize.lever_discovery``)
and the optimization debate (``fermdocs_optimize_debate``) MUST NOT import anything
from this package. The import-guard test enforces that boundary.
"""
from fermdocs_optimize.benchmark.labs_simulator import LABSSimulator

__all__ = ["LABSSimulator"]
