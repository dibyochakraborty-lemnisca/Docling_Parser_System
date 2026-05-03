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
  type RunDetail,
} from "@/lib/api";
import { Timeline } from "@/components/Timeline";

export default function RunPage({ params }: { params: { id: string } }) {
  const runId = params.id;
  const [run, setRun] = useState<RunDetail | null>(null);
  const [events, setEvents] = useState<any[]>([]);
  const [statusMessages, setStatusMessages] = useState<
    { ts: string; status: string; message?: string }[]
  >([]);
  const [submitting, setSubmitting] = useState(false);
  const [answers, setAnswers] = useState<Record<string, string>>({});
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

  return (
    <div className="space-y-8">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Run {run.run_id.slice(0, 8)}</h1>
          <p className="text-sm text-muted-foreground">{run.run_id}</p>
        </div>
        <Badge>{run.status}</Badge>
      </header>

      {run.error && (
        <Card className="border-destructive">
          <CardContent className="pt-6">
            <p className="text-destructive font-medium">Run failed</p>
            <pre className="text-xs mt-2 whitespace-pre-wrap">{run.error}</pre>
          </CardContent>
        </Card>
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
                <p className="text-sm leading-relaxed">{h.summary}</p>
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
