# Head-to-head eval — 10 questions for IndPenSim bundle

Bundle: `out/bundle_indpensim`
Organism: *Penicillium chrysogenum* (industrial fed-batch penicillin fermentation)
Runs in bundle: RUN-0001, RUN-0002
Primary variables instrumented: `biomass_g_l`, `paa_mg_l` (phenylacetic acid precursor), `nh3_mg_l`, `substrate_g_l`, `dissolved_o2_mg_l`, `temperature_k`, `volume_l`, `weight_kg`
Finding distribution: 86 critical, 10 major, 10 minor — heavily biomass + PAA deviation patterns

## Design intent

10 questions across 5 categories, 2 questions each, ordered by increasing
difficulty for a single-shot baseline. The agent should pull ahead on causal +
actionable + uncertainty questions, where the synthesizer/critic loop and
tool-grounded reasoning matter most.

## The questions

| # | shape | question |
|---|---|---|
| Q1 | descriptive | What were the most severe deviations observed in this fermentation bundle, and which variables did they involve? |
| Q2 | descriptive | Summarize how RUN-0001 and RUN-0002 compare in terms of overall process health. |
| Q3 | causal | What is the most likely cause of the sustained super-nominal biomass values observed in both runs after 130 hours? |
| Q4 | causal | The PAA (phenylacetic acid) concentration shows persistent deviations from nominal. What process mechanism best explains this, and how is it likely connected to the biomass anomaly? |
| Q5 | mechanistic | Walk through the metabolic mechanism that ties together substrate utilization, biomass growth, and PAA precursor feeding in this Penicillium chrysogenum fed-batch. Which findings in the bundle support or contradict your account? |
| Q6 | comparative | Compare the temporal profiles of biomass and PAA across RUN-0001 and RUN-0002. Are the deviations synchronized? What does that tell us? |
| Q7 | actionable | If you had to recommend three specific parameter changes for the next batch to reduce critical-severity findings, what would they be and what evidence supports each? |
| Q8 | actionable | What is the single highest-value piece of additional data you would collect on the next run, and which decision would it enable? |
| Q9 | uncertainty | What in this bundle is genuinely ambiguous — i.e., findings where the evidence does not uniquely determine a root cause? Be specific about which findings, and what additional evidence would resolve them. |
| Q10 | uncertainty | The "nominal 0.5 ± 0.05 g/L" reference for biomass produces enormous sigma deviations (400+σ). Is this a real process anomaly, a schema-spec mismatch, or both? Defend your reading with citations. |

## Why these specifically

- **Q1, Q2**: easy wins for either system. Baselines and agents should both
  produce reasonable answers. A tie here is fine; a large gap suggests format
  bias in the judge.
- **Q3, Q4**: causal explanation requires connecting multiple findings. The
  agent's specialists + critic should produce better-grounded claims.
- **Q5**: mechanistic walk-through is the kind of thing a single Gemini call
  often does well (it has the biochemistry), but the agent should better tie
  claims to specific findings in the bundle.
- **Q6**: comparative — forces both to engage with the cross-run structure.
- **Q7, Q8**: actionable recommendation is what the
  actionability-axis rule was designed for. Agent should win.
- **Q9, Q10**: uncertainty / honesty. Q10 specifically probes whether the
  system will commit to a strong claim or honestly acknowledge the schema
  problem. The agent's diagnose stage explicitly tags these (`schema_only`
  confidence_basis) so it has a structural advantage.

## Open notes

- All 10 questions use the same bundle. Cost: 10 × (1 pipeline run + 1
  baseline call + 3 judge calls) ≈ 10 × 5 calls = ~50 API calls.
- For the pipeline runs we re-use the existing ingested indpensim bundle —
  no re-ingest needed since we're not testing memory.
- Each question is independent. We do NOT carry memory across questions for
  this eval. Memory eval is out of scope per the reset decision.
