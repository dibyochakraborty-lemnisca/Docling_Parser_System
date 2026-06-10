// Thin client wrapping the FastAPI backend. Same-origin via next.config.js
// rewrites; production deployment will need to set FERMDOCS_API_BASE.

const BASE = "/api";

export type RunStatus =
  | "pending"
  | "ingesting"
  | "characterizing"
  | "diagnosing"
  | "hypothesizing"
  | "recommending"
  | "debating_opportunities"
  | "optimizing"
  | "paused"
  | "resuming"
  | "done"
  | "failed";

// Which workflow the user picked on the input card. "diagnostic" is the
// fault-finding pipeline (ingest→characterize→diagnose→hypothesize→recommend);
// "optimization" runs the opportunity debate (and, where a process simulator
// exists, the closed-loop optimizer).
export type WorkflowKind = "diagnostic" | "optimization";

export interface RunSummary {
  run_id: string;
  upload_id: string;
  status: RunStatus;
  created_at: string;
  error: string | null;
  workflow?: WorkflowKind;
}

export interface FollowupResultDTO {
  followup_index: number;
  user_question_text: string;
  output: HypothesisOutput | null;
  created_at: string;
}

export interface Intervention {
  intervention_id: string | null;
  description: string;
  knob: string | null;
  objective_metric: string | null;
  baseline_value: number | null;
  predicted_value: number | null;
  delta: number | null;
  in_coverage: boolean | null;
  caveat: string | null;
  rationale: string | null;
}

export interface CandidateReport {
  model_type: string;
  attempted: boolean;
  disqualified: boolean;
  disqualification_reason: string | null;
  selection_r2: number | null;
  selection_rmse: number | null;
  good_fit: boolean | null;
  good_fit_reason: string | null;
  plausible: boolean | null;
  offending_params: string[] | null;
  stalled: boolean | null;
  eligible_species: string[] | null;
  report: Record<string, unknown> | null;
}

export interface RecommendationOutput {
  meta: Record<string, unknown>;
  recommended_model: string;
  confident: boolean;
  refusal_reason: string | null;
  selection_rationale: string;
  candidates: CandidateReport[];
  interventions: Intervention[];
  grounding_hyp_ids: string[];
}

// One governing-equation / fit log entry — "how the agent is using the models".
// Streamed during an optimization run and stored on the result.
export interface ModelLogEntry {
  kind: "equations" | "fit" | "propose" | "simulate" | "note";
  title: string;
  detail: string;
  // present on "fit" entries: the model's own fitted parameters + fit quality
  fitted_params?: Record<string, number> | null;
  r2_by_species?: Record<string, number> | null;
  method?: string | null;
}

export interface OptimizationLever {
  lever_id: string;
  summary: string;
  knobs: string[];
  affected_variables: string[];
  actionable_recommendation: string | null;
  confidence: number;
  supporting_specialists: string[];
}

// One active-learning cycle: the agent proposed a setpoint, predicted its titer,
// and the simulator measured the truth there.
export interface OptimizationCycle {
  cycle: number;
  predicted: number; // model's predicted titer at its optimum
  oracle_verified: number; // what the simulator actually measured there
  error: number; // |predicted - measured|
  converged: boolean;
  box_expansions?: number; // times the search box was widened to chase an out-of-range optimum
}

// The active-learning optimization result, surfaced for the UI.
export interface ActiveOptimization {
  proposer: string; // "llm" | "template"
  converged: boolean;
  cycles: number;
  recommended_knobs: Record<string, number>;
  oracle_verified_titer: number; // the number to trust
  model_predicted_titer: number; // the model's guess (for context only)
  final_error: number;
  baseline_titer: number;
  improvement: number;
  knobs_on_boundary: Record<string, string>;
  batches_added: number; // simulator-verified experiments folded into the data
  total_batches: number;
  iterations: OptimizationCycle[];
  equations: string[]; // the model the agent built (collapsed in the UI)
  model_name: string;
  oracle_evals: number;
}

export interface OptimizationOutput {
  meta: Record<string, unknown>;
  confident: boolean;
  refusal_reason: string | null;
  selection_rationale: string;
  best_candidate: Record<string, number> | null;
  best_achieved_titer: number | null;
  baseline_titer: number | null;
  improvement: number | null;
  levers: OptimizationLever[];
  model_log: ModelLogEntry[];
  simulator_available: boolean;
  optimization?: ActiveOptimization | null;
}

export interface RunDetail extends RunSummary {
  bundle_dir: string | null;
  hypothesis_dir: string | null;
  recommend_dir: string | null;
  global_md: string | null;
  output: HypothesisOutput | null;
  recommendation_output: RecommendationOutput | null;
  optimization_output: OptimizationOutput | null;
  // PR-A2 drive posture: follow-up question history + eligibility flag.
  // Legacy runs (pre-PR-A2) return [] / 0 / true-when-bundle-present.
  followups?: FollowupResultDTO[];
  followup_index?: number;
  bundle_followup_eligible?: boolean;
}

export interface OpenQuestion {
  qid: string;
  question: string;
  raised_by: string;
  tags: string[];
  resolved: boolean;
  resolution: string | null;
}

export interface FinalHypothesis {
  hyp_id: string;
  summary: string;
  facet_ids: string[];
  cited_finding_ids: string[];
  cited_narrative_ids: string[];
  cited_trajectories: { run_id: string; variable: string }[];
  affected_variables: string[];
  confidence: number;
  confidence_basis: string;
  critic_flag: "red" | "green";
  judge_ruled_criticism_valid: boolean;
  // PR-A user-question fields. Null on legacy runs.
  question_answered?: "yes" | "partial" | "insufficient_data" | null;
  question_response_summary?: string | null;
  // Commit 4 of rigour-and-actionability: concrete next-batch parameter
  // change. Null on red-flagged or legacy runs.
  actionable_recommendation?: string | null;
  // charts-and-pdf-export branch: synthesizer-emitted chart intents.
  // The LLM picks kind/runs/variables/story; the backend renders Plotly
  // JSON which the frontend feeds into react-plotly.js.
  chart_specs?: ChartSpec[];
  plotly_charts?: PlotlyFigure[];
}

export type ChartKind =
  | "time_series_overlay"
  | "scatter_correlation"
  | "faceted_time_series";

export interface ChartAnnotation {
  text: string;
  time_h?: number | null;
  run_id?: string | null;
}

export interface ChartSpec {
  kind: ChartKind;
  title: string;
  rationale: string;
  runs: string[];
  variables: string[];
  highlight_runs: string[];
  annotations: ChartAnnotation[];
}

export interface PlotlyFigure {
  data: unknown[];
  layout: Record<string, unknown>;
  spec: ChartSpec;
}

export interface RejectedHypothesis {
  hyp_id: string;
  summary: string;
  rejection_reason: string;
  critic_reasons: string[];
  judge_rationale: string;
}

export interface HypothesisOutput {
  meta: {
    hypothesis_id: string;
    model: string;
    provider: string;
    budget_used: Record<string, number>;
  };
  final_hypotheses: FinalHypothesis[];
  rejected_hypotheses: RejectedHypothesis[];
  open_questions: OpenQuestion[];
  debate_summary: string;
  token_report: {
    total_input: number;
    total_output: number;
    per_agent_input: Record<string, number>;
    per_agent_output: Record<string, number>;
  };
}

export interface UploadResponse {
  upload_id: string;
  filenames: string[];
  content_types: string[];
  size_bytes: number;
  // Operator-supplied process family from the upload dropdown
  // (upload-process-family-ui branch). null = auto-detect.
  process_family: string | null;
  // Legacy single-file keys, populated when N=1, null on N>1.
  // New code reads filenames/content_types instead.
  filename: string | null;
  content_type: string | null;
}

// Closed enum of process_family values, mirrors process_families.yaml.
// "auto-detect" is the sentinel for "let the LLM classify"; the API
// normalises it to null before persistence so downstream sees None.
export const PROCESS_FAMILY_OPTIONS: ReadonlyArray<{
  value: string;
  label: string;
  description: string;
}> = [
  {
    value: "auto-detect",
    label: "Auto-detect (LLM)",
    description: "Let the model classify from the source documents. Falls back to Unknown if no narrative is available (CSV-only).",
  },
  {
    value: "penicillin_fedbatch",
    label: "Penicillin fed-batch",
    description: "Penicillium chrysogenum / P. rubens fed-batch with PAA precursor feed.",
  },
  {
    value: "yeast_intracellular_product_fedbatch",
    label: "Yeast — intracellular product (carotenoid, lipid, terpenoid)",
    description: "Yeast fed-batch producing an intracellular product like β-carotene, lipid, or sterol.",
  },
  {
    value: "yeast_aerobic_fedbatch",
    label: "Yeast — aerobic fed-batch (biomass / extracellular)",
    description: "Yeast fed-batch for biomass or extracellular product. Use when no intracellular product is the focus.",
  },
  {
    value: "ecoli_recombinant_protein",
    label: "E. coli — recombinant protein",
    description: "E. coli expressing a recombinant protein, induced or constitutive.",
  },
  {
    value: "melanin_batch",
    label: "Melanin (batch)",
    description: "Microbial melanin production, batch mode.",
  },
];

export async function uploadFiles(
  files: File[],
  processFamily?: string | null,
): Promise<UploadResponse> {
  if (files.length === 0) {
    throw new Error("uploadFiles called with empty list");
  }
  const fd = new FormData();
  for (const f of files) {
    fd.append("files", f);
  }
  if (processFamily && processFamily !== "auto-detect") {
    fd.append("process_family", processFamily);
  }
  const r = await fetch(`${BASE}/uploads`, { method: "POST", body: fd });
  if (!r.ok) {
    let detail = `${r.status}`;
    try {
      const text = await r.text();
      detail = `${r.status}: ${text}`;
    } catch {}
    throw new Error(`uploadFiles failed: ${detail}`);
  }
  return r.json();
}

export async function createRun(
  uploadId: string,
  userQuestion?: string,
  workflow: WorkflowKind = "diagnostic",
): Promise<{ run_id: string; user_question?: string | null; workflow?: WorkflowKind }> {
  const body: Record<string, unknown> = { upload_id: uploadId, workflow };
  if (userQuestion && userQuestion.trim()) {
    body.user_question = userQuestion.trim();
  }
  const r = await fetch(`${BASE}/runs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    let detail = `${r.status}`;
    try {
      const text = await r.text();
      detail = `${r.status}: ${text}`;
    } catch {}
    throw new Error(`createRun failed: ${detail}`);
  }
  return r.json();
}

export async function listRuns(): Promise<{ runs: RunSummary[] }> {
  const r = await fetch(`${BASE}/runs`, { cache: "no-store" });
  if (!r.ok) throw new Error(`listRuns failed: ${r.status}`);
  return r.json();
}

export async function getRun(runId: string): Promise<RunDetail> {
  const r = await fetch(`${BASE}/runs/${runId}`, { cache: "no-store" });
  if (!r.ok) throw new Error(`getRun failed: ${r.status}`);
  return r.json();
}

export async function submitFollowup(
  runId: string,
  question: string,
): Promise<{
  run_id: string;
  status: string;
  anticipated_followup_index: number;
}> {
  const r = await fetch(`${BASE}/runs/${runId}/followup`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question: question.trim() }),
  });
  if (!r.ok) {
    let detail = `${r.status}`;
    try {
      const text = await r.text();
      detail = `${r.status}: ${text}`;
    } catch {}
    throw new Error(`submitFollowup failed: ${detail}`);
  }
  return r.json();
}

export async function submitAnswers(
  runId: string,
  answers: { qid: string; resolution: string }[],
): Promise<{ status: string }> {
  const r = await fetch(`${BASE}/runs/${runId}/answers`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ answers }),
  });
  if (!r.ok) throw new Error(`submitAnswers failed: ${r.status}`);
  return r.json();
}

export function eventStreamUrl(runId: string): string {
  if (typeof window === "undefined") return "";
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}${BASE}/runs/${runId}/events`;
}
