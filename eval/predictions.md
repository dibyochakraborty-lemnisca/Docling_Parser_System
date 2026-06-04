# E3a Pre-registration — Ablation predictions

**Date:** 2026-05-19
**Bundle:** indpensim (P. chrysogenum, industrial fed-batch)
**Committed before any ablation runs.**

## Prediction table

"Winner" = higher citation_coverage + finding_class_coverage + rubric_score.
"tie" = no meaningful difference expected.

| question | full vs baseline | full vs no_critic | full vs single_spec | full vs no_memory |
|----------|-----------------|-------------------|--------------------|--------------------|
| Q3 (super-nominal biomass cause) | full | full | full | tie |
| Q4 (PAA + biomass mechanism) | full | full | full | tie |
| Q5 (metabolic mechanism walkthrough) | full | tie | full | tie |
| Q7 (3 parameter recommendations) | full | full | tie | tie |
| Q9 (genuine ambiguity) | full | full | full | tie |

## Rationale per prediction

### full vs baseline
Full wins everywhere. The baseline has no structured citations, no debate,
no specialist decomposition. Citation coverage will be ~0 for baseline
(free-text, no finding IDs). Finding-class coverage will be lower because
the baseline doesn't systematically traverse the diagnosis.

### full vs no_critic
- **Q3, Q4, Q9**: critic catches over-stated claims and forces grounding.
  Without it, synthesizer's first draft likely contains ungrounded claims.
  Full wins on citation coverage.
- **Q5**: metabolic walkthrough is a knowledge-display question. Critic
  adds little — the synthesizer's first draft is probably close to final.
  Tie expected.
- **Q7**: recommendations need actionability. Critic forces specificity
  on the evidence behind each recommendation. Full wins.

### full vs single_spec (kinetics only)
- **Q3, Q4, Q9**: these questions touch mass_transfer or metabolic
  variables. Single specialist misses those facets entirely. Full wins
  on finding-class coverage.
- **Q5**: metabolic walkthrough — metabolic specialist is the primary
  contributor. Without it, kinetics alone covers less. Full wins.
- **Q7**: recommendations span multiple specialist domains. But kinetics
  alone can still produce 3 recommendations from its domain. Tie expected.

### full vs no_memory
All ties. This is the indpensim bundle, an unknown_process family. There
are no prior lessons in memory to retrieve. Memory ablation should produce
identical results to full system. If it doesn't, that's a bug.

## What would falsify these predictions

1. **full vs no_critic tie on Q3/Q4/Q9**: would mean the critic isn't
   contributing meaningfully — its rejections aren't improving grounding.
   Finding worth reporting.
2. **full vs single_spec tie on Q5**: would mean the metabolic specialist
   isn't contributing unique facets on its home-turf question. Concerning.
3. **full vs no_memory non-tie**: would mean there's a code path where
   memory presence/absence affects behavior even with no stored lessons.
   Bug, not a finding.
4. **baseline beats full on any question**: would mean the pipeline is
   actively worse than a single LLM call. Same finding as E3-original
   but now isolated to specific configs.
