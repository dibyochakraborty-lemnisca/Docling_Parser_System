# Diagnosis evals

Plan ref: `plans/2026-05-02-diagnosis-agent.md` §11.

These evals score the diagnosis agent's **plumbing** (citation integrity,
provenance downgrade, forbidden-phrase soft check, deterministic claim_id
assignment, question-quality contract) against scripted LLM responses.

They do **not** call a live LLM. Real-LLM evals belong in a separate
`pytest.mark.live` tier (not built yet) that runs before model upgrades or
prompt changes.

## Layout

Each fixture reuses the upstream characterization fixture in
`evals/characterize/fixtures/<name>/dossier.json`. The diagnosis fixture only
adds:

```
evals/diagnosis/<name>/
  scripted_llm_response.json   # what a "good" LLM would emit, single shot
  expected_claims.yaml         # what we score against
```

The runner:

1. Loads `evals/characterize/fixtures/<name>/dossier.json`.
2. Runs the characterization pipeline to produce a `CharacterizationOutput`.
3. Wires a scripted client that returns `scripted_llm_response.json`.
4. Calls `DiagnosisAgent.diagnose(...)`.
5. Scores the resulting `DiagnosisOutput` against `expected_claims.yaml`.

## Score axes (from plan §11)

- **Claim recall:** % of expected claims matched (keyword + cited finding).
- **Citation integrity:** 100% of cited finding IDs must resolve. Hard pass/fail.
- **Forbidden-phrase rate:** soft warning, not pass/fail.
- **Honesty under UNKNOWN flags:** hard pass/fail (no `process_priors` claims
  survive when `UNKNOWN_PROCESS` or `UNKNOWN_ORGANISM` is in the flag set).
- **Question quality (`04_unknown_everything`):** ≥3 open questions, each
  citing a specific finding, each with non-empty `why_it_matters`.

## Fixtures

| Fixture | Purpose |
|---------|---------|
| `01_boundary` | Happy-path. Recognized process, clean data, well-formed claims. |
| `02_missing_data` | Sparse dossier; tests question-emission and data_quality_caveat analysis. |
| `03_multi_run` | Cross-run patterns; tests `cross_run` confidence_basis + MIXED_RUNS flag. |
| `04_unknown_everything` | Adversarial: organism + process unknown. Asserts no `process_priors` survives, ≥3 open questions. |
