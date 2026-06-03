"use client";

import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import {
  eventStreamUrl,
  getRun,
  submitAnswers,
  submitFollowup,
  type FollowupResultDTO,
  type RunDetail,
} from "@/lib/api";
import { Timeline } from "@/components/Timeline";
import { HypothesisCharts } from "@/components/HypothesisChart";

export default function RunPage({ params }: { params: { id: string } }) {
  const runId = params.id;
  const [run, setRun] = useState<RunDetail | null>(null);
  const [events, setEvents] = useState<any[]>([]);
  const [statusMessages, setStatusMessages] = useState<
    { ts: string; status: string; message?: string }[]
  >([]);
  const [submitting, setSubmitting] = useState(false);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  // PR-A2: follow-up textarea state
  const [followupQuestion, setFollowupQuestion] = useState("");
  const [followupSubmitting, setFollowupSubmitting] = useState(false);
  const [followupError, setFollowupError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // Poll run status
  useEffect(() => {
    let alive = true;
    async function poll() {
      try {
        const r = await getRun(runId);
        if (alive) setRun(r);
      } catch {}
    }
    poll();
    const id = setInterval(poll, 2000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [runId]);

  // WebSocket subscription for live events
  useEffect(() => {
    const url = eventStreamUrl(runId);
    if (!url) return;
    const ws = new WebSocket(url);
    wsRef.current = ws;
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === "event") {
          setEvents((prev) => [...prev, msg.event]);
        } else if (msg.type === "status") {
          setStatusMessages((prev) => [
            ...prev,
            {
              ts: new Date().toISOString(),
              status: msg.status,
              message: msg.message,
            },
          ]);
        }
      } catch {}
    };
    ws.onerror = () => {};
    return () => {
      try {
        ws.close();
      } catch {}
    };
  }, [runId]);

  async function onSubmitFollowup() {
    const q = followupQuestion.trim();
    if (!q) return;
    setFollowupSubmitting(true);
    setFollowupError(null);
    try {
      await submitFollowup(runId, q);
      setFollowupQuestion("");
      // Immediate refresh so the badge can flip to "Running follow-up #N"
      // ahead of the next 2-second poll.
      try {
        const r = await getRun(runId);
        setRun(r);
      } catch {}
    } catch (e: any) {
      setFollowupError(e?.message ?? "Submission failed");
    } finally {
      setFollowupSubmitting(false);
    }
  }

  async function onSubmit() {
    if (!run?.output) return;
    const payload = run.output.open_questions
      .filter((q) => !q.resolved && (answers[q.qid] ?? "").trim())
      .map((q) => ({ qid: q.qid, resolution: answers[q.qid].trim() }));
    if (payload.length === 0) return;
    setSubmitting(true);
    try {
      await submitAnswers(runId, payload);
      setAnswers({});
    } finally {
      setSubmitting(false);
    }
  }

  if (!run) {
    return <p className="text-sm text-muted-foreground">Loading run…</p>;
  }

  const unresolved = (run.output?.open_questions ?? []).filter((q) => !q.resolved);
  const followups = run.followups ?? [];
  const followupIndex = run.followup_index ?? 0;
  const followupEligible = run.bundle_followup_eligible ?? false;
  const isRunningFollowup =
    run.status === "hypothesizing" && followupIndex > 0;
  const statusLabel = isRunningFollowup
    ? `Running follow-up #${followupIndex}`
    : run.status;

  // Sticky follow-up bar reserves vertical space; pb keeps content above it.
  const showFollowupBar = run.status === "done" && followupEligible;

  return (
    <div className={`space-y-8 ${showFollowupBar ? "pb-40" : ""}`}>
      <header className="flex items-center justify-between print:mb-4">
        <div>
          <h1 className="text-2xl font-semibold">Run {run.run_id.slice(0, 8)}</h1>
          <p className="text-sm text-muted-foreground">{run.run_id}</p>
        </div>
        <div className="flex items-center gap-2 print:hidden">
          <Badge>{statusLabel}</Badge>
          {run.status === "done" && (
            <Button
              size="sm"
              variant="outline"
              onClick={() => window.print()}
              title="Use your browser's PDF target in the print dialog to save as PDF"
            >
              Download PDF
            </Button>
          )}
        </div>
      </header>

      {run.error && (
        <Card className="border-destructive">
          <CardContent className="pt-6">
            <p className="text-destructive font-medium">Run failed</p>
            <pre className="text-xs mt-2 whitespace-pre-wrap">{run.error}</pre>
          </CardContent>
        </Card>
      )}

      {/* Recommendation Panel */}
      {run.recommendation_output && (
        <section className="space-y-3">
          <h2 className="text-lg font-semibold">Recommendation</h2>
          <Card className={run.recommendation_output.confident ? "border-primary" : ""}>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="font-mono text-sm">
                  Model: {run.recommendation_output.recommended_model}
                </CardTitle>
                <div className="flex items-center gap-2">
                  <Badge variant={run.recommendation_output.confident ? "success" : "secondary"}>
                    {run.recommendation_output.confident ? "Confident" : "Refused"}
                  </Badge>
                  {!run.recommendation_output.confident && (
                    <Badge variant="destructive">{run.recommendation_output.refusal_reason}</Badge>
                  )}
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <p className="text-sm leading-relaxed mb-4">
                {run.recommendation_output.selection_rationale}
              </p>
              
              {run.recommendation_output.candidates.length > 0 && (
                <div className="mt-4 space-y-2">
                  <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Candidate Bake-off</h3>
                  <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
                    {run.recommendation_output.candidates.map((c, i) => (
                      <div key={i} className="rounded-md border p-3 text-xs">
                        <div className="font-semibold mb-1 flex items-center justify-between">
                          <span>{c.model_type}</span>
                          {!c.attempted ? (
                            <Badge variant="outline" className="text-[10px]">Skipped</Badge>
                          ) : c.disqualified ? (
                            <Badge variant="destructive" className="text-[10px]">Disqualified</Badge>
                          ) : (
                            <Badge variant="secondary" className="text-[10px]">Attempted</Badge>
                          )}
                        </div>
                        {c.attempted && !c.disqualified && (
                          <div className="space-y-1 mt-2 text-muted-foreground">
                            <div>R² (worst): {c.selection_r2 !== null ? c.selection_r2.toFixed(3) : "N/A"}</div>
                            <div>RMSE (worst): {c.selection_rmse !== null ? c.selection_rmse.toFixed(3) : "N/A"}</div>
                            {c.good_fit !== null && (
                              <div className={c.good_fit ? "text-green-600" : "text-amber-600"}>
                                Good fit: {c.good_fit ? "Yes" : "No"}
                              </div>
                            )}
                            {c.plausible !== null && (
                              <div className={c.plausible ? "text-green-600" : "text-amber-600"}>
                                Plausible: {c.plausible ? "Yes" : "No"}
                              </div>
                            )}
                            {c.offending_params && c.offending_params.length > 0 && (
                              <div className="text-destructive mt-1">
                                Offending: {c.offending_params.join(", ")}
                              </div>
                            )}
                          </div>
                        )}
                        {c.disqualified && c.disqualification_reason && (
                          <div className="text-destructive mt-2">{c.disqualification_reason}</div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {run.recommendation_output.interventions.length > 0 && (
                <div className="mt-4 space-y-2">
                  <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Interventions</h3>
                  <ul className="space-y-2">
                    {run.recommendation_output.interventions.map((inv, i) => (
                      <li key={i} className="text-sm border-l-2 pl-3 border-muted">
                        <div className="font-medium">{inv.description}</div>
                        {inv.predicted_value !== null && inv.baseline_value !== null && (
                          <div className="text-xs mt-1">
                            <span className="font-mono">{inv.objective_metric ?? "objective"}</span>:{" "}
                            {inv.baseline_value} → <strong>{inv.predicted_value}</strong>
                            {inv.delta !== null && (
                              <span className={inv.delta >= 0 ? "text-green-600" : "text-destructive"}>
                                {" "}({inv.delta >= 0 ? "+" : ""}{inv.delta})
                              </span>
                            )}
                            {inv.in_coverage === false && (
                              <Badge variant="outline" className="ml-2 text-[10px]">extrapolation</Badge>
                            )}
                          </div>
                        )}
                        {inv.rationale && <p className="text-muted-foreground text-xs mt-1">{inv.rationale}</p>}
                        {inv.caveat && <p className="text-amber-600 text-xs mt-1">{inv.caveat}</p>}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </CardContent>
          </Card>
        </section>
      )}

      {/* Final hypotheses */}
      {run.output && run.output.final_hypotheses.length > 0 && (
        <section className="space-y-3">
          <h2 className="text-lg font-semibold">Final hypotheses</h2>
          {run.output.final_hypotheses.map((h) => (
            <Card key={h.hyp_id}>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="font-mono text-sm">{h.hyp_id}</CardTitle>
                  <div className="flex items-center gap-2">
                    <Badge variant="outline">{h.confidence_basis}</Badge>
                    <Badge variant="success">conf {h.confidence.toFixed(2)}</Badge>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                {h.question_answered && (
                  <div className="mb-3 rounded-md border bg-accent/40 px-3 py-2">
                    <div className="flex items-center gap-2 text-xs">
                      <span className="font-medium">Your question:</span>
                      <Badge
                        variant={
                          h.question_answered === "yes"
                            ? "success"
                            : h.question_answered === "partial"
                            ? "warning"
                            : "secondary"
                        }
                      >
                        {h.question_answered === "yes"
                          ? "Answered"
                          : h.question_answered === "partial"
                          ? "Partially answered"
                          : "Insufficient data"}
                      </Badge>
                    </div>
                    {h.question_response_summary && (
                      <p className="mt-2 text-sm leading-relaxed">
                        {h.question_response_summary}
                      </p>
                    )}
                  </div>
                )}
                <p className="text-sm leading-relaxed">{h.summary}</p>
                {h.actionable_recommendation && (
                  <div className="mt-3 rounded-md border-l-4 border-l-primary bg-primary/5 px-3 py-2">
                    <div className="text-xs font-medium uppercase tracking-wide text-primary">
                      Recommendation
                    </div>
                    <p className="mt-1 text-sm leading-relaxed">
                      {h.actionable_recommendation}
                    </p>
                  </div>
                )}
                <HypothesisCharts figures={h.plotly_charts} />
                <div className="mt-3 flex flex-wrap gap-1">
                  {h.affected_variables.map((v) => (
                    <Badge key={v} variant="secondary" className="font-mono text-xs">
                      {v}
                    </Badge>
                  ))}
                </div>
                <div className="mt-3 text-xs text-muted-foreground">
                  cites {h.cited_finding_ids.length} findings,{" "}
                  {h.cited_narrative_ids.length} narratives,{" "}
                  {h.cited_trajectories.length} trajectories
                </div>
              </CardContent>
            </Card>
          ))}
        </section>
      )}

      {/* Follow-up cards (PR-A2 drive posture) */}
      {followups.length > 0 && (
        <section className="space-y-4">
          <div className="border-t pt-4" />
          <h2 className="text-lg font-semibold">Follow-ups</h2>
          {followups.map((f: FollowupResultDTO) => (
            <div key={f.followup_index} className="space-y-2">
              <div className="text-sm">
                <span className="font-medium">Follow-up #{f.followup_index}:</span>{" "}
                <span className="italic text-muted-foreground">
                  {f.user_question_text}
                </span>
              </div>
              {f.output?.final_hypotheses?.map((h) => (
                <Card key={h.hyp_id}>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <CardTitle className="font-mono text-sm">{h.hyp_id}</CardTitle>
                      <div className="flex items-center gap-2">
                        <Badge variant="outline">{h.confidence_basis}</Badge>
                        <Badge variant="success">conf {h.confidence.toFixed(2)}</Badge>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent>
                    {h.question_answered && (
                      <div className="mb-3 rounded-md border bg-accent/40 px-3 py-2">
                        <div className="flex items-center gap-2 text-xs">
                          <span className="font-medium">Your follow-up:</span>
                          <Badge
                            variant={
                              h.question_answered === "yes"
                                ? "success"
                                : h.question_answered === "partial"
                                ? "warning"
                                : "secondary"
                            }
                          >
                            {h.question_answered === "yes"
                              ? "Answered"
                              : h.question_answered === "partial"
                              ? "Partially answered"
                              : "Insufficient data"}
                          </Badge>
                        </div>
                        {h.question_response_summary && (
                          <p className="mt-2 text-sm leading-relaxed">
                            {h.question_response_summary}
                          </p>
                        )}
                      </div>
                    )}
                    <p className="text-sm leading-relaxed">{h.summary}</p>
                    {h.actionable_recommendation && (
                      <div className="mt-3 rounded-md border-l-4 border-l-primary bg-primary/5 px-3 py-2">
                        <div className="text-xs font-medium uppercase tracking-wide text-primary">
                          Recommendation
                        </div>
                        <p className="mt-1 text-sm leading-relaxed">
                          {h.actionable_recommendation}
                        </p>
                      </div>
                    )}
                    <HypothesisCharts figures={h.plotly_charts} />
                  </CardContent>
                </Card>
              ))}
              {(f.output?.final_hypotheses?.length ?? 0) === 0 && (
                <p className="text-xs text-muted-foreground italic">
                  No final hypothesis was emitted for this follow-up.
                </p>
              )}
            </div>
          ))}
        </section>
      )}

      {/* Open questions form */}
      {unresolved.length > 0 && (
        <section>
          <Card>
            <CardHeader>
              <CardTitle>Answer open questions</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {unresolved.map((q) => (
                <div key={q.qid} className="space-y-2">
                  <label className="block">
                    <div className="text-xs text-muted-foreground font-mono">
                      {q.qid} · raised by {q.raised_by}
                    </div>
                    <div className="text-sm">{q.question}</div>
                  </label>
                  <Textarea
                    placeholder="Your answer (leave empty to skip)"
                    value={answers[q.qid] ?? ""}
                    onChange={(e) =>
                      setAnswers((prev) => ({ ...prev, [q.qid]: e.target.value }))
                    }
                  />
                </div>
              ))}
              <div className="pt-2">
                <Button onClick={onSubmit} disabled={submitting}>
                  {submitting ? "Submitting…" : "Submit and resume"}
                </Button>
              </div>
            </CardContent>
          </Card>
        </section>
      )}

      {/* Rejected hypotheses (collapsed) */}
      {run.output && run.output.rejected_hypotheses.length > 0 && (
        <section className="space-y-2">
          <h2 className="text-lg font-semibold">
            Rejected hypotheses ({run.output.rejected_hypotheses.length})
          </h2>
          <ul className="space-y-2">
            {run.output.rejected_hypotheses.map((r) => (
              <li
                key={r.hyp_id}
                className="rounded-md border bg-card px-4 py-3 text-sm"
              >
                <div className="font-mono text-xs text-muted-foreground">
                  {r.hyp_id}
                </div>
                <div className="mt-1 whitespace-pre-wrap">{r.summary}</div>
                <div className="mt-2 text-xs text-destructive">
                  Rejected: {r.rejection_reason}
                </div>
              </li>
            ))}
          </ul>
        </section>
      )}

      {/* Pipeline progress (per-stage status messages) */}
      {statusMessages.length > 0 && (
        <section>
          <h2 className="text-lg font-semibold mb-3">Pipeline progress</h2>
          <Card>
            <CardContent className="pt-6">
              <ul className="space-y-1 text-sm">
                {statusMessages.map((s, i) => (
                  <li key={i} className="flex items-baseline gap-3">
                    <span className="text-xs text-muted-foreground tabular-nums">
                      {new Date(s.ts).toLocaleTimeString()}
                    </span>
                    <Badge variant="secondary">{s.status}</Badge>
                    {s.message && (
                      <span className="text-muted-foreground">{s.message}</span>
                    )}
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>
        </section>
      )}

      {/* Live debate timeline */}
      <section>
        <h2 className="text-lg font-semibold mb-3">Debate timeline</h2>
        <Timeline events={events} />
      </section>

      {/* Sticky follow-up bar (PR-A2 drive posture). Always pinned to the
          viewport bottom while run is DONE and bundle is on disk. Disappears
          during a follow-up run (status flips to hypothesizing) and when
          the bundle has been GC'd. */}
      {showFollowupBar && (
        <div className="fixed bottom-0 left-0 right-0 z-30 border-t bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/80">
          <div className="mx-auto max-w-3xl px-4 py-3">
            <div className="flex items-end gap-2">
              <div className="flex-1">
                <Textarea
                  placeholder="Ask a follow-up — bundle is frozen, no re-ingest."
                  value={followupQuestion}
                  onChange={(e) => setFollowupQuestion(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
                      e.preventDefault();
                      onSubmitFollowup();
                    }
                  }}
                  maxLength={2000}
                  rows={2}
                  className="resize-none"
                />
                <div className="mt-1 flex items-center justify-between text-xs text-muted-foreground">
                  <span>
                    {followupError ? (
                      <span className="text-destructive">{followupError}</span>
                    ) : (
                      <>⌘/Ctrl+Enter to submit</>
                    )}
                  </span>
                  <span>{followupQuestion.length}/2000</span>
                </div>
              </div>
              <Button
                onClick={onSubmitFollowup}
                disabled={followupSubmitting || !followupQuestion.trim()}
              >
                {followupSubmitting ? "Submitting…" : "Send"}
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Token report */}
      {run.output?.token_report && (
        <section>
          <h2 className="text-lg font-semibold mb-3">Token report</h2>
          <Card>
            <CardContent className="pt-6">
              <div className="text-sm">
                Total: {run.output.token_report.total_input.toLocaleString()} in /{" "}
                {run.output.token_report.total_output.toLocaleString()} out
              </div>
              <table className="mt-4 w-full text-sm">
                <thead>
                  <tr className="text-muted-foreground text-xs uppercase">
                    <th className="text-left py-1">Agent</th>
                    <th className="text-right py-1">Input</th>
                    <th className="text-right py-1">Output</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.keys(run.output.token_report.per_agent_input)
                    .sort()
                    .map((agent) => (
                      <tr key={agent} className="border-t">
                        <td className="py-1 font-mono text-xs">{agent}</td>
                        <td className="py-1 text-right">
                          {run.output!.token_report.per_agent_input[agent].toLocaleString()}
                        </td>
                        <td className="py-1 text-right">
                          {run.output!.token_report.per_agent_output[agent].toLocaleString()}
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </CardContent>
          </Card>
        </section>
      )}
    </div>
  );
}
