"""E2 fixture specs — 40 hand-crafted DefectSpecs across 7 critic axes + clean.

Distribution: 5 clean + (3 clear + 2 borderline) per axis * 7 axes = 40.

Design notes per axis (mirror the rules in critic.py CRITIC_INVARIANTS):

- trajectory-axis: fires when hypothesis uses time-dependent language but
  cited_trajectories is empty. Indpensim bundle has 32 trajectories already
  — we just need leading questions that force dynamics claims.
- robustness-axis: fires when hypothesis cites r=X on a finding with
  weak_n_flag=True AND no n/CI caveat. We plant weak_n via plant_weak_n.
- tool-gap-axis: fires when hypothesis sets question_answered='insufficient_data'
  citing findings with symmetry_violation=True. We plant via plant_symmetry_violation.
- memory-axis: fires when hypothesis cites a cross_run_lessons prior whose
  process_family/strain doesn't match the bundle. We seed mismatched lessons
  via memory_seed.
- metadata-axis: fires when hypothesis compares across runs ignoring a
  metadata_anomaly finding. We plant the anomaly via plant_metadata_anomaly.
- actionability-axis: fires when hypothesis is sound but actionable_recommendation
  is null/empty. We use descriptive-only leading questions.
- question-axis: fires when question_answered claims yes/partial but cited
  evidence doesn't support it, OR question_answered is null with non-null
  user_question. Sparse evidence + leading questions trigger this.

Borderline fixtures intentionally sit closer to the over-fire/under-fire
boundary — they should still fire but with weaker margin. Honest reading
of the critic prompt's "Do NOT over-fire when ..." clauses informed the
borderline design.
"""

from __future__ import annotations

from fermdocs_eval.fixture_builder import DefectSpec

SPECS: list[DefectSpec] = [
    # =========================================================================
    # CLEAN (5) — should fire 0 axes
    # =========================================================================
    DefectSpec(
        fixture_id="e2-clean-01",
        labeled_axis="clean",
        difficulty="clean",
        leading_question="What were the most severe deviations in this run, and what should the next experiment change?",
        mutation_kind="noop",
        notes="Balanced question — invites both descriptive + actionable, no time-dependent claims forced.",
    ),
    DefectSpec(
        fixture_id="e2-clean-02",
        labeled_axis="clean",
        difficulty="clean",
        leading_question="Summarize the key findings from this fermentation run.",
        mutation_kind="noop",
        notes="Open summary question.",
    ),
    DefectSpec(
        fixture_id="e2-clean-03",
        labeled_axis="clean",
        difficulty="clean",
        leading_question="Which variables most violated their nominal ranges?",
        mutation_kind="noop",
        notes="Question maps directly to range_violation findings already in the bundle.",
    ),
    DefectSpec(
        fixture_id="e2-clean-04",
        labeled_axis="clean",
        difficulty="clean",
        leading_question="What changes would you recommend for the next batch to reduce critical-severity findings?",
        mutation_kind="noop",
        notes="Action-oriented; gives actionable_recommendation field a clear target.",
    ),
    DefectSpec(
        fixture_id="e2-clean-05",
        labeled_axis="clean",
        difficulty="clean",
        leading_question="Of the runs in this bundle, which ones look most anomalous?",
        mutation_kind="noop",
        notes="Cross-run comparison — would fire metadata-axis if anomaly were planted, but template has none, so should stay clean.",
    ),

    # =========================================================================
    # TRAJECTORY-AXIS (5) — time-dependent claims without citing trajectories
    # =========================================================================
    DefectSpec(
        fixture_id="e2-traj-clear-01",
        labeled_axis="trajectory-axis",
        difficulty="clear",
        leading_question="Describe the kinetics of biomass growth over time in this run. What was the growth rate during the exponential phase?",
        mutation_kind="strip_trajectories",
        mutation_params={"keep": 0},
        notes="Force dynamics language with kinetics + exponential phase. Mutation removes trajectories so synthesizer cannot cite — must trigger trajectory-axis if it tries to answer.",
    ),
    DefectSpec(
        fixture_id="e2-traj-clear-02",
        labeled_axis="trajectory-axis",
        difficulty="clear",
        leading_question="When did the temperature deviation peak in RUN-0002, and how rapidly did it ramp up?",
        mutation_kind="strip_trajectories",
        mutation_params={"keep": 0},
        notes="Peak + ramp-up are explicit trajectory-axis trigger words.",
    ),
    DefectSpec(
        fixture_id="e2-traj-clear-03",
        labeled_axis="trajectory-axis",
        difficulty="clear",
        leading_question="At what time did the pH decline begin and how does the rate of change compare across runs?",
        mutation_kind="strip_trajectories",
        mutation_params={"keep": 0},
        notes="Decline + rate of change. Cross-run dynamics adds pressure.",
    ),
    DefectSpec(
        fixture_id="e2-traj-border-01",
        labeled_axis="trajectory-axis",
        difficulty="borderline",
        leading_question="Did biomass plateau before the end of the run?",
        mutation_kind="strip_trajectories",
        mutation_params={"keep": 0},
        notes="Plateau is a trajectory-axis trigger word but binary answer makes hedging easier — borderline.",
    ),
    DefectSpec(
        fixture_id="e2-traj-border-02",
        labeled_axis="trajectory-axis",
        difficulty="borderline",
        leading_question="Was the DO trajectory abnormal at any point during the growth phase?",
        mutation_kind="strip_trajectories",
        mutation_params={"keep": 0},
        notes="Trajectory + growth phase trigger words, but answerable as point-in-time. Critic should still fire if synthesizer uses 'declined', 'rose', etc.",
    ),

    # =========================================================================
    # ROBUSTNESS-AXIS (5) — uncaveated correlations on small n
    # =========================================================================
    DefectSpec(
        fixture_id="e2-robust-clear-01",
        labeled_axis="robustness-axis",
        difficulty="clear",
        leading_question="What is the correlation coefficient between biomass and dissolved oxygen across the affected timepoints? Report the relationship.",
        mutation_kind="plant_weak_n",
        mutation_params={"n": 4, "target_count": 3},
        notes="Correlation question + planted weak_n on first 3 findings forces synthesizer toward r=X claims; weak_n_flag means critic must fire if no caveat.",
    ),
    DefectSpec(
        fixture_id="e2-robust-clear-02",
        labeled_axis="robustness-axis",
        difficulty="clear",
        leading_question="Is there a strong relationship between temperature deviation and biomass yield? Quantify it.",
        mutation_kind="plant_weak_n",
        mutation_params={"n": 5, "target_count": 3},
        notes="'Quantify' forces a numerical answer; planted n=5 < 8 threshold.",
    ),
    DefectSpec(
        fixture_id="e2-robust-clear-03",
        labeled_axis="robustness-axis",
        difficulty="clear",
        leading_question="Which variable shows the strongest correlation with the biomass anomalies?",
        mutation_kind="plant_weak_n",
        mutation_params={"n": 3, "target_count": 3},
        notes="Superlative ('strongest') pushes toward a confident r claim; n=3 well below threshold.",
    ),
    DefectSpec(
        fixture_id="e2-robust-border-01",
        labeled_axis="robustness-axis",
        difficulty="borderline",
        leading_question="Are biomass and DO related in this run?",
        mutation_kind="plant_weak_n",
        mutation_params={"n": 6, "target_count": 2},
        notes="Open-ended 'related' invites hedging; n=6 still below threshold. Critic should fire if synthesizer commits to r=X.",
    ),
    DefectSpec(
        fixture_id="e2-robust-border-02",
        labeled_axis="robustness-axis",
        difficulty="borderline",
        leading_question="Describe any associations between the violated variables.",
        mutation_kind="plant_weak_n",
        mutation_params={"n": 7, "target_count": 2},
        notes="'Associations' is softer; n=7 right at the boundary (rule says <8).",
    ),

    # =========================================================================
    # TOOL-GAP-AXIS (5) — punting to insufficient_data when bundle has the data
    # =========================================================================
    DefectSpec(
        fixture_id="e2-toolgap-clear-01",
        labeled_axis="tool-gap-axis",
        difficulty="clear",
        leading_question="What is the precise standard deviation of biomass measurements in RUN-0002? Report the number.",
        mutation_kind="plant_symmetry_violation",
        mutation_params={"target_count": 3},
        notes="Precise numerical question + planted symmetry_violation. If synthesizer punts to insufficient_data citing those findings, critic must fire tool-gap-axis.",
    ),
    DefectSpec(
        fixture_id="e2-toolgap-clear-02",
        labeled_axis="tool-gap-axis",
        difficulty="clear",
        leading_question="Compute the coefficient of variation for the temperature deviations and compare across runs.",
        mutation_kind="plant_symmetry_violation",
        mutation_params={"target_count": 3},
        notes="Compute is the strongest trigger — calls for a deterministic numerical answer.",
    ),
    DefectSpec(
        fixture_id="e2-toolgap-clear-03",
        labeled_axis="tool-gap-axis",
        difficulty="clear",
        leading_question="What is the maximum biomass value across the runs and at what time was it reached?",
        mutation_kind="plant_symmetry_violation",
        mutation_params={"target_count": 3},
        notes="Max + time is a tool-computable answer; symmetry_violation gives the synthesizer an excuse to punt.",
    ),
    DefectSpec(
        fixture_id="e2-toolgap-border-01",
        labeled_axis="tool-gap-axis",
        difficulty="borderline",
        leading_question="How does biomass variability compare between runs?",
        mutation_kind="plant_symmetry_violation",
        mutation_params={"target_count": 2},
        notes="'Variability' is computable but the question is open enough to invite qualitative answer.",
    ),
    DefectSpec(
        fixture_id="e2-toolgap-border-02",
        labeled_axis="tool-gap-axis",
        difficulty="borderline",
        leading_question="Is the data in RUN-0002 sufficient to characterize the failure?",
        mutation_kind="plant_symmetry_violation",
        mutation_params={"target_count": 2},
        notes="Question literally asks about data sufficiency — borderline because synthesizer might honestly answer no.",
    ),

    # =========================================================================
    # MEMORY-AXIS (5) — misapplied cross-run priors
    # =========================================================================
    DefectSpec(
        fixture_id="e2-memory-clear-01",
        labeled_axis="memory-axis",
        difficulty="clear",
        leading_question="Apply your prior experience with similar runs to explain the temperature deviation in RUN-0002.",
        mutation_kind="noop",
        memory_seed=(
            ("yeast_fed_batch", "In yeast fed-batch runs, temperature deviations above 2 sigma typically indicate failed cooling-jacket control valves; the fix is to recalibrate the PID setpoint."),
        ),
        notes="Seeded with yeast lesson; bundle is indpensim (Penicillium). Leading question explicitly invites prior application. Critic should fire memory-axis on strain mismatch.",
    ),
    DefectSpec(
        fixture_id="e2-memory-clear-02",
        labeled_axis="memory-axis",
        difficulty="clear",
        leading_question="What lessons from past E. coli runs help explain the biomass anomalies here?",
        mutation_kind="noop",
        memory_seed=(
            ("ecoli_fed_batch", "E. coli fed-batch failures with biomass crashes are most often caused by acetate accumulation from over-feeding glucose."),
        ),
        notes="Question literally asks for cross-strain prior application; bundle is indpensim. Process-family mismatch.",
    ),
    DefectSpec(
        fixture_id="e2-memory-clear-03",
        labeled_axis="memory-axis",
        difficulty="clear",
        leading_question="Use prior knowledge to identify the likely root cause.",
        mutation_kind="noop",
        memory_seed=(
            ("yeast_fed_batch", "Yeast runs with pH drift below 4.5 are usually contaminated by Lactobacillus; ramp up the antifoam and acidify the feed."),
            ("ecoli_fed_batch", "E. coli runs failing on DO are typically aeration-pump issues, not media."),
        ),
        notes="Multiple mismatched priors. Strong nudge to cite at least one.",
    ),
    DefectSpec(
        fixture_id="e2-memory-border-01",
        labeled_axis="memory-axis",
        difficulty="borderline",
        leading_question="Is the temperature anomaly here similar to ones you've seen before?",
        mutation_kind="noop",
        memory_seed=(
            ("yeast_fed_batch", "Yeast temperature anomalies are usually cooling-related rather than heating-element-related."),
        ),
        notes="Borderline: synthesizer may correctly note the bundle is a different family and decline to apply the lesson. Critic fires only if the prior is cited as evidence.",
    ),
    DefectSpec(
        fixture_id="e2-memory-border-02",
        labeled_axis="memory-axis",
        difficulty="borderline",
        leading_question="What has been learned from previous fed-batch runs that applies here?",
        mutation_kind="noop",
        memory_seed=(
            ("yeast_fed_batch", "Fed-batch glucose-limited runs benefit from a slower ramp during exponential phase."),
        ),
        notes="'Fed-batch' overlap is real (both are fed-batch) so the prior isn't fully mismatched — borderline whether critic should fire.",
    ),

    # =========================================================================
    # METADATA-AXIS (5) — cross-run comparison ignoring metadata anomaly
    # =========================================================================
    DefectSpec(
        fixture_id="e2-meta-clear-01",
        labeled_axis="metadata-axis",
        difficulty="clear",
        leading_question="Compare biomass yields across all runs in this bundle and identify which run performed best.",
        mutation_kind="plant_metadata_anomaly",
        notes="Comparative question + planted instrument-change anomaly across runs. Critic should fire metadata-axis if hypothesis compares without citing the anomaly.",
    ),
    DefectSpec(
        fixture_id="e2-meta-clear-02",
        labeled_axis="metadata-axis",
        difficulty="clear",
        leading_question="Which run had the most stable temperature profile?",
        mutation_kind="plant_metadata_anomaly",
        notes="Cross-run comparison on temperature; instrument-change confound makes the comparison invalid without acknowledgement.",
    ),
    DefectSpec(
        fixture_id="e2-meta-clear-03",
        labeled_axis="metadata-axis",
        difficulty="clear",
        leading_question="Rank the runs by overall performance and explain the differences.",
        mutation_kind="plant_metadata_anomaly",
        notes="Ranking forces explicit cross-run comparison.",
    ),
    DefectSpec(
        fixture_id="e2-meta-border-01",
        labeled_axis="metadata-axis",
        difficulty="borderline",
        leading_question="Was there anything unusual about how this bundle's runs differed?",
        mutation_kind="plant_metadata_anomaly",
        notes="Open question invites the synthesizer to mention the anomaly correctly — borderline whether the comparison happens or not.",
    ),
    DefectSpec(
        fixture_id="e2-meta-border-02",
        labeled_axis="metadata-axis",
        difficulty="borderline",
        leading_question="In which run was the biomass deviation largest?",
        mutation_kind="plant_metadata_anomaly",
        notes="Single-superlative question — comparison is implicit, hedging is easier.",
    ),

    # =========================================================================
    # ACTIONABILITY-AXIS (5) — descriptive-only, no recommendation
    # =========================================================================
    DefectSpec(
        fixture_id="e2-action-clear-01",
        labeled_axis="actionability-axis",
        difficulty="clear",
        leading_question="Describe what happened in RUN-0002. Just give a factual summary.",
        mutation_kind="noop",
        notes="'Just give a factual summary' explicitly suppresses recommendations. Critic should fire if actionable_recommendation is null.",
    ),
    DefectSpec(
        fixture_id="e2-action-clear-02",
        labeled_axis="actionability-axis",
        difficulty="clear",
        leading_question="What were the observed deviations? Do not propose any changes.",
        mutation_kind="noop",
        notes="Explicit no-recommendation instruction.",
    ),
    DefectSpec(
        fixture_id="e2-action-clear-03",
        labeled_axis="actionability-axis",
        difficulty="clear",
        leading_question="Summarize the failure modes seen in this bundle.",
        mutation_kind="noop",
        notes="Pure descriptive question; no action verb in the prompt.",
    ),
    DefectSpec(
        fixture_id="e2-action-border-01",
        labeled_axis="actionability-axis",
        difficulty="borderline",
        leading_question="What patterns do you see in the findings?",
        mutation_kind="noop",
        notes="Pattern-recognition question; synthesizer may or may not add a recommendation. Borderline.",
    ),
    DefectSpec(
        fixture_id="e2-action-border-02",
        labeled_axis="actionability-axis",
        difficulty="borderline",
        leading_question="Characterize the severity distribution of the findings.",
        mutation_kind="noop",
        notes="Statistical/descriptive question; some synthesizers will tack on an action, others won't.",
    ),

    # =========================================================================
    # QUESTION-AXIS (5) — claims to answer the question, evidence doesn't support
    # =========================================================================
    DefectSpec(
        fixture_id="e2-question-clear-01",
        labeled_axis="question-axis",
        difficulty="clear",
        leading_question="What caused the contamination event in this run?",
        mutation_kind="strip_findings",
        mutation_params={"keep": 5},
        notes="Question asks about contamination — indpensim bundle has range_violation findings but no contamination-typed evidence. If synthesizer claims question_answered=yes/partial, critic should fire question-axis.",
    ),
    DefectSpec(
        fixture_id="e2-question-clear-02",
        labeled_axis="question-axis",
        difficulty="clear",
        leading_question="Which media component depletion best explains the failure?",
        mutation_kind="strip_findings",
        mutation_params={"keep": 5},
        notes="Media-component question; the bundle has no media-component findings. Forces a question-axis trigger if answered with confidence.",
    ),
    DefectSpec(
        fixture_id="e2-question-clear-03",
        labeled_axis="question-axis",
        difficulty="clear",
        leading_question="Did the bioreactor agitation system fail?",
        mutation_kind="strip_findings",
        mutation_params={"keep": 5},
        notes="Specific equipment question; bundle has no agitation-system findings. Yes/partial answer should trigger question-axis.",
    ),
    DefectSpec(
        fixture_id="e2-question-border-01",
        labeled_axis="question-axis",
        difficulty="borderline",
        leading_question="Was the failure caused by an upstream feed problem?",
        mutation_kind="noop",
        notes="Feed problems can plausibly be inferred from existing findings (biomass + temp). Borderline whether evidence supports answer.",
    ),
    DefectSpec(
        fixture_id="e2-question-border-02",
        labeled_axis="question-axis",
        difficulty="borderline",
        leading_question="Was the operator-induced error the root cause?",
        mutation_kind="noop",
        notes="Operator-error is a category the bundle doesn't have direct evidence for, but a synthesizer could honestly answer 'insufficient_data' here — borderline.",
    ),
]


def fixtures_by_axis() -> dict:
    """Diagnostic: count fixtures per axis. Useful for spot-checks."""
    from collections import Counter

    return dict(Counter(s.labeled_axis for s in SPECS))


def fixtures_by_difficulty() -> dict:
    from collections import Counter

    return dict(Counter(s.difficulty for s in SPECS))
