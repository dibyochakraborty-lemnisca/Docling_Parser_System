# Decision Types (closed) — OpenQuestion routing

`OpenQuestion.decision_type` is a closed enum so the Orchestrator can route open questions to the right debate sub-agent deterministically. Adding a new decision type requires a code change AND a corresponding sub-agent route.

| DecisionType | When characterization emits this | Who resolves it |
|---|---|---|
| `causal_attribution` | Multiple findings could explain an observed deviation; characterization can't choose which is causal without more evidence or domain reasoning. | Domain expert sub-agents in the debate loop. |
| `anomaly_classification` | A pattern is anomalous but characterization can't classify whether it's instrument failure, process failure, or biological variability. | Orchestrator routes to the most relevant domain (e.g. instrumentation vs microbiology). |
| `evidence_request` | Characterization would settle a question if it had specific additional evidence (a missing measurement, a replicate run, an operator note). The `would_resolve_with` field names what's needed. | Orchestrator decides whether to schedule re-ingestion, request data, or proceed without. |

## Why closed

Deterministic routing means each `decision_type` has a known handler in the Orchestrator. An open enum forces the Orchestrator to NLP the `question_text`, which is a context-engineering smell: routing should be explicit. Novelty in question structure is handled at code-change time, not at runtime.

## How `would_resolve_with` is used

Free-text list of evidence names. The Orchestrator may match these against tools/agents that produce such evidence. Examples that real fermentation pipelines produce:

- `media_batch_metadata`
- `replicate_run_at_30C`
- `OUR_trace`
- `operator_logbook_entry_18h`
- `qPCR_contamination_check`

Not a closed list — these are domain hints, not routing keys.
