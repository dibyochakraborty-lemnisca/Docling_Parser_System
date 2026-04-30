# Finding Types (closed)

This is the closed list of `FindingType` values the characterization agent may emit. Validators reject any finding whose type is not on this list. Adding a new type requires (a) a code change in `schema.py`, (b) a candidate generator that emits it, and (c) eval coverage proving precision and recall.

Version column indicates the earliest version that emits the type.

| Type | Version | Emitted by | Means |
|---|---|---|---|
| `range_violation` | v1 | `candidates/range_violation.py` | An observed value falls outside `nominal ± k·σ` for a controlled or specified variable. Severity tied to k. |
| `cohort_outlier` | v2a | `candidates/cohort_outlier.py` | A run's trajectory feature (final titer, peak biomass, time-to-peak, etc.) is z-score outside the cohort distribution at the same nominal condition. Requires N≥2 runs in cohort. |
| `mass_balance_error` | v2b | `candidates/mass_balance.py` | Observed (volume, weight) pair is inconsistent with the prior `density` from `process_priors.yaml` beyond tolerance. Indicates a sampling/feed/measurement error. |
| `process_relationship_violation` | v2b | `candidates/process_relations.py` | A known fermentation relationship is violated: OUR/OTR balance, pH/NH3 coupling, Monod-substrate kinetics, T-mu Arrhenius, PAA depletion vs `mu_p`. Each check has its own statistic. |
| `contradicts` | v2b | `candidates/process_relations.py` | Two ingestion observations of the same canonical column at the same `(run_id, time)` disagree beyond instrument tolerance. Surfaced from cross-source contradictions. |
| `precedes_with_lag` | v3 | `candidates/causal_cascade.py` | Two findings/events on the timeline appear in a known plausible causal relationship within a short lag window. Promoted by LLM judge for plausible cascades only. |
| `kinetic_anomaly` | v3 | `candidates/kinetic_estimator.py` | A fitted kinetic parameter (`mu_x`, `mu_p`, `kla`) deviates from the nominal/specified value beyond CI overlap. Distinct from `range_violation` because it uses fitted, not raw, values. |

## Severity rubric

Applied uniformly across types. The severity *of a finding* is independent of its type.

| Severity | Meaning |
|---|---|
| `info` | Observation worth recording but not a deviation. Most v1 findings inside ±2σ fall here. |
| `minor` | Deviation present but small. ±2σ to ±3σ for `range_violation`. |
| `major` | Clear deviation. ±3σ to ±5σ for `range_violation`. Likely a real problem. |
| `critical` | Severe deviation. >±5σ for `range_violation`, or any finding the Diagnosis Agent would treat as a primary failure point. |

## Confidence calibration by `extracted_via`

| `extracted_via` | How confidence is computed | Cap |
|---|---|---|
| `deterministic` | Function of severity. info=0.6, minor=0.85, major=0.95, critical=0.99. No LLM. | 1.0 |
| `statistical` | Function of p-value, effect size, and n. | 1.0 |
| `llm_judged` | LLM emits a score; consumer must respect the cap. | 0.85 |

Downstream agents must compare confidence values only **within the same `extracted_via` class**. A `deterministic` 0.95 and an `llm_judged` 0.80 are not on the same scale.
