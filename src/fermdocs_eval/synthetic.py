"""Synthetic test fixtures for E2 (critic axes precision/recall).

Each fixture is a labeled hypothesis with a deliberate defect on one axis (or
"clean"). The critic runs over the hypothesis + a stub dossier context; we
compare fired axes against the label.

This module only owns the FIXTURE TEXTS and labels. Wiring into the critic
loop lives in suites/e2.py.
"""

from __future__ import annotations

from dataclasses import dataclass

# 7 axes the critic uses today. Keep in sync with
# src/fermdocs_hypothesis/agents/critic.py axis names.
CRITIC_AXES = [
    "trajectory-axis",
    "robustness-axis",
    "tool-gap-axis",
    "memory-axis",
    "metadata-axis",
    "actionability-axis",
    "question-axis",
]


@dataclass(frozen=True)
class SyntheticHypothesis:
    fixture_id: str  # stable ID like "e2-clean-01"
    labeled_axis: str  # one of CRITIC_AXES or "clean"
    text: str  # the hypothesis the critic sees
    dossier_hint: str  # ground-truth dossier snippet that should make the defect detectable


# Fixtures are intentionally short and obvious for v1. We can expand once the
# harness is wired and we see how the critic actually responds. Keep 5 per
# axis + 5 clean = 40 total. Authoring is iterative — start with stubs.
FIXTURES: list[SyntheticHypothesis] = [
    # Filled in during E2 build-out. Keep this list empty so an
    # accidental run-all picks up the zero-fixture state and reports it.
]
