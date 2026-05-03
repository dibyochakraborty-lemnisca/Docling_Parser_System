// Thin client wrapping the FastAPI backend. Same-origin via next.config.js
// rewrites; production deployment will need to set FERMDOCS_API_BASE.

const BASE = "/api";

export type RunStatus =
  | "pending"
  | "ingesting"
  | "characterizing"
  | "diagnosing"
  | "hypothesizing"
  | "paused"
  | "resuming"
  | "done"
  | "failed";

export interface RunSummary {
  run_id: string;
  upload_id: string;
  status: RunStatus;
  created_at: string;
  error: string | null;
}

export interface RunDetail extends RunSummary {
  bundle_dir: string | null;
  hypothesis_dir: string | null;
  global_md: string | null;
  output: HypothesisOutput | null;
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
  filename: string;
  size_bytes: number;
  content_type: string;
}

export async function uploadFile(file: File): Promise<UploadResponse> {
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch(`${BASE}/uploads`, { method: "POST", body: fd });
  if (!r.ok) throw new Error(`upload failed: ${r.status}`);
  return r.json();
}

export async function createRun(uploadId: string): Promise<{ run_id: string }> {
  const r = await fetch(`${BASE}/runs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ upload_id: uploadId }),
  });
  if (!r.ok) {
    let detail = `${r.status}`;
    try {
      const body = await r.text();
      detail = `${r.status}: ${body}`;
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
