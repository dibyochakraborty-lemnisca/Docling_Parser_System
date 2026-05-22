# fermdocs Hypothesis Stage — Synap Use Case

**Instance:** `fermdocs-dev`
**Owner:** Lemnisca Bio
**Use case version:** v1 (Phase 1 of memory-layer plan, 2026-05-10)

---

## Agent Objective

fermdocs is an agentic system that turns raw fermentation experiment reports (PDFs, CSVs, scientist narratives) into structured, evidence-grounded hypotheses about what happened in each batch and why one batch outperformed another.

The system runs in four stages: **ingest** (parse + map raw columns to canonical names), **characterize** (compute deterministic metrics + extract narrative observations), **diagnose** (build a fault-tree from the characterization), and **hypothesis** (a multi-agent debate between domain specialists — kinetics, mass transfer, metabolic — moderated by a synthesizer, critic, and judge).

The hypothesis stage is currently a **single-shot reasoner over one bundle**. Every new fermentation report enters cold. It cannot remember that this strain showed a stress-response signature in three prior runs, that a specific reviewer pushed back on a hypothesis class six months ago and was right, or that "carotenoid yield drop after substrate switch" has been hypothesized fourteen times in this deployment with the synthesizer converging on a specific mechanism each time.

**Synap is the substrate that lets specialists become learned priors over the deployment, not just static prompts.** The agent uses Synap to remember and retrieve distilled lessons from prior runs and inject them into the synthesizer / critic context at the start of each new run. Bundle evidence remains ground truth; memory is prior context.

---

## Target Users

Three distinct roles interact with the agent. Synap memories surface differently for each.

**Process scientists (primary).** Bench/pilot-scale fermentation engineers analyzing 1–6 batch experiments. They upload a report and ask "why did Batch 4 underperform?" or "compare RUN-0001 and RUN-0002 yields." They read the agent's hypotheses, follow up with corrections ("no, this wasn't oxygen limitation, the impeller had degraded"), and accept or reject conclusions.

**Process development leads.** Senior scientists who oversee strain campaigns spanning months. They care about cross-batch patterns and want the agent to remember what was concluded about a strain across the campaign. The "system should know we already concluded X about this strain" expectation.

**Bioprocess data scientists / MLOps engineers (Lemnisca internal).** Tune the agent, debug failed runs, calibrate the eval harness. They read raw memory records via the admin endpoint and write the prompts that the agent uses.

---

## Task Examples

Three representative tasks that exercise Synap's role.

**Task 1 — First run on a new strain (cold start).**
User: "Why did Batch 4 underperform on this 6-batch yeast carotenoid campaign?"
Expected agent behavior: characterize each run, run hypothesis debate, emit final hypotheses. Synap retrieval returns zero memories (no prior history on this `process_family`). Synap write at clean exit: persist 1–3 lesson digests describing recurring rejection patterns surfaced during this run's debate. Memory store ends with N records where N=0 before, ≥1 after.

**Task 2 — Re-run on a known process family (warm).**
User: "New 4-batch run on the same yeast carotenoid process. Anything anomalous?"
Expected agent behavior: at view-build time, query Synap for top-5 lessons matching this `process_family`. Lessons are injected into specialists, synthesizer, and critic prompts as `[CROSS-RUN LESSONS]`. Synthesizer must surface relevant priors *and* explicit contradictions when bundle evidence overrides them. The system catches "we previously rejected hypotheses that extended single-batch evidence to multi-batch claims" and the synthesizer narrows accordingly on this run.

**Task 3 — Operator correction becomes durable.**
User submits a HITL follow-up: "The agent claimed oxygen limitation on RUN-0003 but the impeller was actually degraded between RUN-0002 and RUN-0003 — confirmed in the maintenance log."
Expected agent behavior: persist this correction as a high-weight Synap memory tagged to the relevant `process_family` and `affected_variables` (DO, agitation_rpm). On the next run with a similar DO-crash signature, retrieval surfaces this correction *before* the synthesizer concludes "oxygen limitation," letting the agent propose impeller integrity as a competing explanation.

---

## Behavioral Guidelines

**Do:**
- Retrieve memories at the start of each run and inject them as `[CROSS-RUN LESSONS]` context for synthesizer + critic.
- Treat memories as **priors**, not ground truth. Bundle evidence always overrides memory when they conflict.
- Persist a memory only when the run reaches a clean exit (`consensus_reached` or `no_topics_left`). Failed or budget-exhausted runs do **not** write.
- Tag every memory with `process_family`, `organism`, `run_id`, `hyp_id`, `lesson_id`, and a small `tags` list. These are how retrieval filters work.
- Surface contradictions explicitly. If a retrieved prior contradicts the current bundle, the synthesizer must name that tension in its hypothesis, not silently override.
- Only persist `lessons_summarizer` agent output (Phase 1). Future phases extend to ratified hypotheses, rejected hypotheses, and human corrections — those use distinct `MemoryKind` values.

**Don't:**
- Don't write memory inline during a run. Buffer in `<bundle>/lesson_buffer.json` and persist on clean exit only.
- Don't allow cross-strain retrieval without an explicit, separate API call. `fetch(kind="lesson", process_family=None)` must raise.
- Don't let the LLM author memory content directly. Only the deterministic `lessons_summarizer` output is persisted.
- Don't surface stale memories without provenance. Every retrieved record must trace back to a specific `(run_id, lesson_id, generation_timestamp)`.
- Don't retrieve more than 5 memories per run. The synthesizer prompt budget is finite.

---

## Role Descriptions

Mapping fermdocs domain concepts to Synap's scope chain.

- **Client = Lemnisca Bio.** The deployment owner. One Client per Synap instance. World-scope memories are Synap-managed shared ontology; we don't write to it.
- **Customer = Tenant.** Each commercial customer (Lemnisca-internal vs. Acme Biotech vs. Globex Bio) is a Synap Customer. Memories never cross Customer boundaries. For `fermdocs-dev`, Customer is `lemnisca-internal`. For `fermdocs-prod`, real customers will be added per onboarding.
- **User = process_family.** Unusual mapping but intentional: our retrieval primary key is `process_family` (a closed-vocabulary value validated against our family registry: `yeast_intracellular_product_fedbatch`, `penicillin_fedbatch`, `ecoli_recombinant_protein`, etc.). Treating each family as a Synap User gives us tight retrieval scoping without entity-resolution work in Phase 1.
- Lesson `metadata` carries: `run_id`, `hyp_id`, `lesson_id`, `organism`, `tags`, `generation_timestamp`, `source_event_offset`.

---

## Compliance & Data Sensitivity

**Data classification:** internal / regulated. Fermentation experiment data is proprietary and may include strain identities, recipes, and process IPR.

**Regulatory frameworks:** Lemnisca customers operate under FDA, EMA, and various national bioprocess GMP regimes. Memory stored about a customer's experiments is treated as their data, retrievable only within their Customer scope.

**Data residency:** Phase 1 uses Synap's hosted infrastructure. Customers that require data residency in a specific jurisdiction will be migrated to a self-hosted Postgres adapter (planned alternative backend in our `MemoryBackend` Protocol).

**PII handling:** memory content is agent-generated lesson digests, not raw experiment data and not user PII. Operator names mentioned in narrative observations *can* end up in lesson text; we mitigate by filtering operator names in the `lessons_summarizer` prompt before persistence.

**Retention:** memories accumulate indefinitely in Phase 1. A future TTL / decay policy is filed as a follow-up TODO. Customers can request full deletion of their Customer scope at offboarding.

**Audit:** every retrieved memory carries `provenance.{run_id, hyp_id, lesson_id, generation_timestamp, source_event_offset}` so any agent-cited prior traces back to a specific source event in our `global.md` audit log.

---

## Memory Priorities

What types of information matter most for the agent to remember, in priority order:

1. **Recurring rejection patterns** (highest priority). The critic's reasons-for-rejection across many runs — "we keep rejecting hypotheses that conflate within-run and cross-run evidence." These are the most information-dense signal because they prevent re-learning.
2. **Operator corrections** (Phase 2 — Tier 5). Senior-scientist HITL corrections are the closest thing we have to labelled training data. Highest weight when present.
3. **Ratified mechanism conclusions** (Phase 2 — Tier 2). What the system concluded about a strain's behavior post-judge-approval. "Yeast under SRL-glucose feed shows pigment-loss after 144h" is a durable prior.
4. **Recurring synthesis patterns** (Phase 1 — what we ship). Lesson digests from `lessons_summarizer`. "On this family, evidence symmetry across runs is the most common rejection axis."
5. **Strain-conditional KPI distributions** (Phase 4 — separate tier). Per-`process_family` empirical distributions of μ_max, peak titer, etc. Used by the deterministic `finding_validator`, not by retrieval.

Phase 1 only writes memories of type (4). Tiers 2/3/5 land in later phases.

---

## Additional Context

**Architecture boundary.** fermdocs maintains a strict separation: bundles are per-run state (filesystem JSON), Synap is across-run state. Within-run debate state lives in `global.md` event logs, never in Synap. This keeps audit trails locally inspectable and Synap retrieval scoped to *priors*, not *current run state*.

**Memory backend abstraction.** fermdocs implements a `MemoryBackend` Protocol with multiple adapters: `NoopBackend` (default, off — for tests and dev), `StubBackend` (in-memory dict — for unit tests), and `SynapBackend` (this instance — for dev runs and validation). A `PostgresBackend` is planned as the alternative for customers requiring self-hosted data residency. Synap is one swappable adapter, not the only one.

**Production vs. dev split.** This instance (`fermdocs-dev`) is for non-production runs: developer laptops, eval harness experiments, debugging. A separate `fermdocs-prod` instance will hold memories from real customer runs in production. Dev never reads from prod; prod never reads from dev.

**Memory-axis critic rule.** The critic agent has a `[memory-axis]` rejection rule that fires when a hypothesis cites a prior lesson but the cited evidence is from a different run / strain / family. This is the structural defense against the synthesizer trusting memory over bundle evidence. Synap retrievals must carry enough provenance for this rule to evaluate.

**Eval gate.** Phase 1 ships only after a small eval harness shows: (a) `NoopBackend` path is byte-identical to today's behavior on existing fixture bundles, and (b) the `[memory-axis]` rule fires reliably on planted bad-citation cases. This is an explicit pre-merge gate, not a post-deploy hope.

**Out of scope for Synap (Phase 1):** trajectory-shape similarity (DTW), strain-conditional KPI priors, ratified-hypothesis retrieval, human-correction memory. All listed in our deferral table with explicit triggers.

---

## Versioning

This use-case markdown is committed alongside the implementation plan in `plans/synap_setup/fermdocs-dev-usecase.md`. Updates require a new revision header and a Synap dashboard re-upload. The plan reference is `plans/2026-05-10-memory-layer.md`.
