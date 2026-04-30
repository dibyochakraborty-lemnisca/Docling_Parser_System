# Characterization fixtures (synthetic)

Three synthetic dossiers + expected `CharacterizationOutput` JSONs. Each fixture exercises a specific scenario the v1 pipeline must handle correctly.

| Fixture | Purpose |
|---|---|
| `01_boundary` | Single run, values placed at exactly ±2σ / ±3σ / ±5σ to test severity threshold sharpness. |
| `02_missing_data` | Single run with sparse measurements and large imputation gaps. Tests `Trajectory.data_quality` population. |
| `03_multi_run` | Four runs at the same condition, deliberately introducing range violations in only one. Tests multi-run handling without cohort logic. |

## Conventions

- Each fixture has a fixed `characterization_id` UUID so finding IDs are deterministic.
- Each fixture has a fixed `generation_timestamp` for byte-stable comparison.
- Real fermdocs ingestion does not yet emit `run_id` and `timestamp_h` in the observation locator. Fixtures include these as forward-compatible fields; characterization reads them best-effort.
- The expected output populates `findings`, `timeline`, `expected_vs_observed`, `trajectories`, `open_questions`. The `facts_graph` and `kinetic_estimates` are tested separately and may be empty in expected outputs.

## Severity / confidence rubric (v1, deterministic)

| |σ| range | Severity | Confidence (`extracted_via=deterministic`) |
|---|---|---|
| < 2.0 | (no finding emitted, but Deviation recorded) | n/a |
| [2.0, 3.0) | `minor` | 0.85 |
| [3.0, 5.0) | `major` | 0.95 |
| ≥ 5.0 | `critical` | 0.99 |

`info` severity is reserved for non-violation observations the pipeline still wants on the timeline. In v1 we do not emit info-severity findings; deviations within ±2σ are recorded in `expected_vs_observed` only.
