# Head-to-head — fermdocs agent vs single-shot Gemini

**N questions**: 8. **N judge rows**: 24. **Errors**: 1.
**Bundle**: indpensim (P. chrysogenum, industrial fed-batch).
**Treatment**: full fermdocs pipeline (synthesizer + 3 specialists + critic + judge, all gemini-3.1-pro-preview).
**Baseline**: single gemini-3.1-pro-preview call with bundle JSON + question.
**Judge**: gemini-3.1-pro-preview (separate call), 3 seeds, counterbalanced A/B.

## Headline: preference rate

- **Treatment win rate**: **0%** (0/24) — 95% bootstrap CI [0%, 0%]
- Baseline wins: 24 | ties: 0

## Per-axis score means (1-10)

| axis | treatment | baseline | Δ (T−B) | per-judge wins/losses/ties |
| --- | --- | --- | --- | --- |
| specificity | 7.00 ± 1.14 | 9.83 ± 0.38 | -2.83 | 0W / 23L / 1T |
| grounding | 7.67 ± 0.82 | 9.50 ± 0.51 | -1.83 | 0W / 24L / 0T |
| actionability | 3.58 ± 2.41 | 5.17 ± 3.75 | -1.58 | 0W / 14L / 10T |
| honesty | 7.75 ± 1.45 | 9.54 ± 0.59 | -1.79 | 0W / 20L / 4T |

## Per-question winners

| qid | treatment | baseline | tie | n_judges |
| --- | --- | --- | --- | --- |
| q1 | 0 | 3 | 0 | 3 |
| q10 | 0 | 3 | 0 | 3 |
| q2 | 0 | 3 | 0 | 3 |
| q3 | 0 | 3 | 0 | 3 |
| q5 | 0 | 3 | 0 | 3 |
| q7 | 0 | 3 | 0 | 3 |
| q8 | 0 | 3 | 0 | 3 |
| q9 | 0 | 3 | 0 | 3 |

## Judge rationales

### q1
- **seed s0** (baseline): Answer B provides greater detail including exact timestamps, setpoints, and UUID finding identifiers, though both answers correctly identify schema artifacts and lack concrete next steps.
- **seed s1** (baseline): Answer A provides a much more comprehensive and detailed response with explicit identifiers, whereas Answer B suffers from duplicated text and fewer specifics, though both completely fail to provide actionable next steps.
- **seed s2** (baseline): Answer B provides much greater specificity with exact timestamps, values, and UUIDs, and avoids Answer A's glaring textual repetition, though both answers fail to provide concrete next steps.

### q10
- **seed s0** (baseline): Answer B provides a superior, non-repetitive defense using precise schema paths and intermediate timepoints to prove the mismatch, though both fail to include explicit next steps.
- **seed s1** (baseline): Answer A provides a much more coherent explanation of the evaluation engine artifacts with direct citations to the system diagnosis, whereas Answer B is highly repetitive and introduces ungrounded external values.
- **seed s2** (baseline): Answer B provides superior specificity by quoting exact schema hints, intermediate timepoints, and diagnosis IDs, though both answers fail to conclude with concrete next steps.

### q2
- **seed s0** (baseline): Answer B provides superior specificity with detailed references to both diagnostic and finding IDs, avoids Answer A's repetitive structure, and demonstrates high honesty by thoroughly contextualizing process deviations as static schema artifacts.
- **seed s1** (baseline): Answer A provides comprehensive finding identifiers and astutely recognizes schema artifacts, whereas Answer B is highly repetitive and lacks any actionable direction for an engineer.
- **seed s2** (baseline): Answer B provides a well-structured, highly specific, and honest assessment by correctly identifying schema artifacts, whereas Answer A is repetitive and explicitly dismisses the context of those artifacts without providing next steps.

### q3
- **seed s0** (baseline): Answer B is substantially more specific by citing precise finding identifiers, run IDs, and exact measurements, though both unfortunately fail to provide actionable next steps.
- **seed s1** (baseline): Answer A provides superior specificity by citing exact finding IDs, run IDs, and precise numerical data to prove the schema artifact, though neither answer ends with explicit actionable next steps.
- **seed s2** (baseline): Answer B provides significantly greater specificity by citing exact finding IDs, run IDs, and diagnosis references, although both answers fail to conclude with concrete next steps.

### q5
- **seed s0** (baseline): Answer B provides a much more detailed and precise biological explanation, utilizing exact numerical bounds, long-form finding IDs, and diagnostic claims, whereas both lack concrete next steps.
- **seed s1** (baseline): Answer A provides a much more detailed explanation of the metabolic mechanism, leveraging full finding UUIDs, exact data values, and verbatim quotes to excellently expose the schema artifacts.
- **seed s2** (baseline): Answer B provides a much more detailed breakdown of the metabolic mechanism and uses precise UUIDs, diagnostic claim IDs, and standard deviation metrics to comprehensively explain the schema artifacts.

### q7
- **seed s0** (baseline): Answer B is vastly superior in specificity by citing exact run IDs, numeric values, time windows, and diagnostic IDs, and it provides three highly actionable, clearly structured recommendations.
- **seed s1** (baseline): Answer A provides exactly three highly specific, actionable schema parameter changes supported by exact numerical evidence and finding IDs, whereas Answer B offers only a generalized summary.
- **seed s2** (baseline): Answer B provides exceptionally high specificity with exact numerical values, run IDs, and diagnostic identifiers, while structuring exactly three highly actionable schema recommendations that honestly address the root cause.

### q8
- **seed s0** (baseline): Answer B astutely identifies severe schema artifacts causing false positive alarms, using highly specific finding IDs and statistical deviations to justify implementing dynamic trajectories over adding new physical sensors.
- **seed s1** (baseline): Answer A insightfully identifies the underlying schema artifact causing false alarms and provides highly specific, actionable advice to fix the evaluation logic, whereas Answer B gives a standard sensor recommendation that ignores the fundamental data-evaluation flaw.
- **seed s2** (baseline): Answer B astutely identifies a critical schema artifact causing massive false-positive alarms and supports its actionable recommendation with precise finding IDs, whereas Answer A completely misses the data limitation.

### q9
- **seed s0** (baseline): Answer B provides exceptionally specific references to finding IDs, explicitly addresses schema artifacts as a major source of ambiguity, and suggests concrete operational checks to decouple biological phenomena from equipment issues.
- **seed s1** (baseline): Answer A provides exceptional specificity, expertly identifies schema artifacts causing ambiguity, and clearly lists actionable next steps, whereas Answer B is repetitive and less comprehensive.
- **seed s2** (baseline): Answer B provides exceptionally detailed references to specific findings and values, accurately identifies schema artifacts, and offers highly concrete, actionable next steps to resolve the ambiguities.

## Errors

- `q4-treatment`: JSONDecodeError: Invalid \uXXXX escape: line 3 column 15576 (char 15609)

## Notes / limitations

- LLM-as-judge: same model family judges both outputs. Mitigated by counterbalanced A/B order (treatment is A on even seeds, B on odd).
- N=10 questions, all on a single bundle (indpensim). This is a case-study eval, not a benchmark. We do not claim cross-bundle generalization.
- Memory was held off for this eval (hermetic StubBackend per question). Memory-specific evals are out of scope for this report.
- Bootstrap CIs assume i.i.d. trials; with 3 judge seeds per question the same question's judges are correlated, so the reported CI is a lower bound on the true variance.