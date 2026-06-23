# TODOS

## Remove `process_priors.yaml` hardcoded domain values
- **What:** Eliminate the last baked-in domain constants (nominal/spec/expected values) in [src/fermdocs/schema/process_priors.yaml](src/fermdocs/schema/process_priors.yaml); judge data against its own distribution (data-relative).
- **Why:** Standing rule "no hardcoded domain values anywhere" — these constants make the system wrong for any experiment that doesn't match LABS-era assumptions.
- **Pros:** Fully data-relative judging; removes the last LABS-era assumption file.
- **Cons:** Need a data-relative replacement for every prior currently read from the file; touches diagnose/characterize consumers.
- **Context:** Consumed by `fermdocs_diagnose/*`, `fermdocs/domain/process_priors.py`, `fermdocs/domain/process_families.py`. Must NOT become the source for `resolve_objective` (de-LABS PR explicitly excludes it).
- **Depends on:** none. Independent of the de-LABS PR.

## Full chart-grounding fix (synthesizer renderable-variable awareness)
- **What:** Feed the synthesizer the complete set of renderable `(run_id, variable)` pairs from the bundle, and the rule that a lever (per-run metadata scalar) is not a plottable trajectory. Fixes silently-dropped chart specs.
- **Why:** ~half of optimize runs render 0 charts because the synthesizer cites variable names that don't exist (`P`, `biomass_g_l`) or cites levers in `scatter_correlation`. The chart builder returns None and the spec is silently dropped.
- **Pros:** Charts actually render; specialists stop guessing canonical names.
- **Cons:** Adds fields to `SynthesizerView` + prompt changes; an LLM eval should confirm spec quality doesn't regress.
- **Context:** Diagnosed 2026-06-16. `SynthesizerView` (fermdocs_hypothesis/schema.py) carries no variable list; `_render_charts_into_finals` (runner.py:932) drops mismatches. The de-LABS PR Phase 1 fixes only the objective name (`product_g_l` not `P`), not the full variable set or the lever-as-chart-variable bug.
- **Depends on:** lands cleaner after de-LABS Phase 1 (objective threading).

## Exotic non-objective units exposed by the #0a unit-layer fix (deferred)
- **What:** Density-aware / non-pint unit resolution for channels NOT on the objective/cross-run path: Color (Hazen ↔ ICUMSA), specific gravity, Brix, FAN (ppm). #0a fixed the concentration channels (%w/w/g·kg⁻¹ → g/L) that feed the optimizer; these others are exposed to the same "convert-if-recognized / silent-pass-through-if-not" class but don't block 1→3.
- **Why:** Same defect class as #0; left unfixed they silently mis-store on any sheet whose unit string varies. Tracked so they're not forgotten, deferred so #0 doesn't balloon into a units-framework project before the productivity-objective work.
- **Context:** After #0a, these either convert via pint or now refuse legibly (no more silent pass-through), so they fail loud rather than wrong — acceptable interim. Fix = extend the resolver per unit family.
- **Depends on:** #0a (done). Independent of 1→3.

## 0b: general per-run density extraction (gated — ticket, not a row reader)
- **What:** Add `density` to the golden-schema vocabulary with its own unit resolution (specific gravity = dimensionless ~1.05; g/mL; g/cm³ incl. `g / cm3` with a space) so the EXISTING general extractor surfaces a labeled "Density" field; then the #0a converter reproduces the sheets' own g/L (praaj B474→108.67, B541→164.65) and the clampedness check re-runs on real units.
- **STOP condition:** if praaj's density is only reachable by reading a specific block/row (its density lives in a summary block), do NOT build a layout reader — ticket it here and stop. A row/block-shape conditional is the layer→reader tripwire.
- **Acceptance (two terminal states, both acceptable for the investigation):** (a) density extracts generally → live praaj reproduces g/L, oracle passes; or (b) it's layout-trapped → ticketed, praaj continues to refuse legibly. A ticket is a pass for 0b-the-investigation, a fail for 0b-the-oracle — keep distinct.
- **Note:** specific-gravity density is legitimately dimensionless and must NOT be caught by #0a's dimensionless-refusal guard (that guard is scoped to g/L conversion *targets* only — see converter.py comment).
- **Depends on:** #0a (done).
