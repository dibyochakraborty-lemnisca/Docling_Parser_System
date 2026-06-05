# Skill: optimize-titer

The recipe for driving one optimization. You normally need **one** call to
`run_optimization_loop` — it runs every round internally. Re-run only to try a
different model/proposer or a longer budget when the first result is unconverged
or low-confidence.

## Steps

1. `get_experiment()` — read the seed data: number of batches, species present,
   the baseline peak titer (best P already in the data), and the observed knob
   ranges. The baseline is what you must beat.
2. `get_box()` — read the feasible search box (per-knob lb/ub). Every proposal is
   clamped to it.
3. `run_optimization_loop(objective_species="P", model="mechanistic",
   proposer="optimize", max_rounds=6, proposals_per_round=4,
   delta_titer_threshold=2.0)` — run the closed loop. Read the returned summary:
   - `best_achieved_titer` + `improvement` over baseline
   - `trajectory` — achieved peak P per round (should rise, then plateau)
   - per-round `fit_target_r2` (model fit on P) and `model_vs_oracle_r2`
     (did the model predict the simulated proposals?)
   - `convergence` — `delta_below_threshold` (good) or `max_rounds` (may need more)
4. Decide if you are satisfied:
   - Converged with a real improvement and the model agreed with the oracle in
     the last round → submit, confident.
   - `max_rounds` with the trajectory still climbing → re-run with more rounds.
   - Improvement is tiny or `model_vs_oracle_r2` stayed low → still submit, but
     flag low confidence in your rationale (the point is a lead, not a verified
     optimum). Do not dress up a weak result as a strong one.
5. `submit_optimization(payload_json="...")` — a JSON string with:
   - `rationale`: 2-4 sentences a bioprocess engineer can act on — the proposed
     operating point, the achieved vs baseline titer, how convergence went, and
     any caveat.
   - `confidence_note`: one honest line on how much to trust the optimum.
   The authoritative numbers (best point, achieved titer, trajectory) are taken
   from the loop result you produced — you cannot override them here.

## Budget discipline

`run_optimization_loop` is the expensive call (it shells out to the oracle for
every proposal of every round). Prefer one well-configured run over many small
ones. Always reach `submit_optimization` before the step budget runs out.
