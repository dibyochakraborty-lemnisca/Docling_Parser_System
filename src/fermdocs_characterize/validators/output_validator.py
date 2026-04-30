"""Output validator: cross-cutting checks beyond Pydantic.

Pydantic enforces shape, ID format, enum membership, finding-id namespacing,
dangling-edge prevention, and confidence cap. This validator additionally:

- Verifies schema_version is current
- Verifies process_priors_version is current when set (v2b+)
- Verifies every evidence_observation_id resolves in the source dossier(s),
  following ingestion's supersession chain
- Returns a list of error strings; empty list = OK
"""

from __future__ import annotations

from typing import Any

from fermdocs_characterize import SCHEMA_VERSION
from fermdocs_characterize.schema import CharacterizationOutput


def validate_output(
    output: CharacterizationOutput,
    *,
    dossiers: dict[str, dict[str, Any]] | None = None,
    current_schema_version: str = SCHEMA_VERSION,
    current_process_priors_version: str | None = None,
) -> list[str]:
    """Run cross-cutting validation. Returns a list of error strings.

    `dossiers` is a mapping from `experiment_id` to the dossier dict. When
    provided, every evidence_observation_id in the output is checked against
    the union of observations in those dossiers.
    """
    errors: list[str] = []

    if output.meta.schema_version != current_schema_version:
        errors.append(
            f"schema_version {output.meta.schema_version!r} is not current"
            f" ({current_schema_version!r}); regenerate this output."
        )

    if (
        current_process_priors_version is not None
        and output.meta.process_priors_version is not None
        and output.meta.process_priors_version != current_process_priors_version
    ):
        errors.append(
            f"process_priors_version {output.meta.process_priors_version!r} is not"
            f" current ({current_process_priors_version!r}); regenerate this output."
        )

    if dossiers is not None:
        all_ids: set[str] = set()
        # Map old observation_id → latest observation_id via supersession chain.
        # In v1 dossiers don't carry supersession on golden_columns, so the chain
        # is trivial. Future: walk superseded_by until None.
        for d in dossiers.values():
            for col_data in d.get("golden_columns", {}).values():
                if not isinstance(col_data, dict):
                    continue
                for obs in col_data.get("observations", []):
                    obs_id = obs.get("observation_id")
                    if obs_id:
                        all_ids.add(str(obs_id))

        def _check(label: str, entity_id: str, obs_ids: list[str]) -> None:
            for oid in obs_ids:
                if oid not in all_ids:
                    errors.append(
                        f"{label} {entity_id} cites unknown observation_id {oid!r}"
                    )

        for f in output.findings:
            _check("finding", f.finding_id, f.evidence_observation_ids)
        for d in output.expected_vs_observed:
            _check("deviation", d.deviation_id, d.source_observation_ids)
        for t in output.trajectories:
            _check("trajectory", t.trajectory_id, t.source_observation_ids)
        for k in output.kinetic_estimates:
            _check("kinetic_estimate", k.fit_id, k.evidence_observation_ids)

    return errors


class ValidationError(Exception):
    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors
