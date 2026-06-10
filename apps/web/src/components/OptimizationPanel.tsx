"use client";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { ActiveOptimization, OptimizationOutput } from "@/lib/api";

// User-facing view of the optimization run. The plan is the simulator-verified
// recommendation; the refinement card tells the active-learning story (the agent
// built a model, tested its guess against the simulator, learned, repeated). The
// model's own equations are tucked behind a collapsed "details" so internals
// don't dominate the page.
export function OptimizationPanel({ opt }: { opt: OptimizationOutput }) {
  const a = opt.optimization ?? null;
  return (
    <section className="space-y-3">
      <h2 className="text-heading-sm">Optimization</h2>

      <Card className={opt.confident ? "border-primary" : ""}>
        <CardHeader>
          <div className="flex items-center justify-between gap-2">
            <CardTitle className="text-sm">Recommended operating point</CardTitle>
            <Badge variant={opt.simulator_available ? "success" : "secondary"}>
              {opt.simulator_available ? "Simulator-verified" : "Debate-only (no simulator)"}
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {a ? (
            <Recommendation a={a} />
          ) : (
            <p className="text-sm text-ink-muted whitespace-pre-wrap">
              {opt.selection_rationale ||
                "No process simulator is configured, so the agent reported the debated levers below as the optimization plan."}
            </p>
          )}
        </CardContent>
      </Card>

      {a && <RefinementCard a={a} />}

      {opt.levers.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Debated levers ({opt.levers.length})</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {opt.levers.map((lev) => (
              <div key={lev.lever_id} className="rounded-md border border-rule bg-surface-1 p-3">
                <div className="flex items-center justify-between gap-2">
                  <span className="font-mono text-xs text-ink-muted">{lev.lever_id}</span>
                  <div className="flex items-center gap-1">
                    {lev.knobs.map((k) => (
                      <Badge key={k} variant="outline" className="text-[10px]">{k}</Badge>
                    ))}
                    <Badge variant="secondary" className="text-[10px]">conf {lev.confidence.toFixed(2)}</Badge>
                  </div>
                </div>
                <p className="mt-1 text-sm">{lev.summary}</p>
                {lev.actionable_recommendation && (
                  <p className="mt-1 text-xs text-accent">→ {lev.actionable_recommendation}</p>
                )}
              </div>
            ))}
          </CardContent>
        </Card>
      )}
    </section>
  );
}

// The plan: knobs to run + the verified titer + improvement + box-edge guidance.
function Recommendation({ a }: { a: ActiveOptimization }) {
  const edges = Object.keys(a.knobs_on_boundary);
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-2">
        {Object.entries(a.recommended_knobs).map(([k, v]) => (
          <span key={k} className="rounded-md border border-rule bg-surface-1 px-2 py-1 font-mono text-xs">
            {k} = {v}
            {a.knobs_on_boundary[k] && (
              <span className="ml-1 text-accent">({a.knobs_on_boundary[k]} edge)</span>
            )}
          </span>
        ))}
      </div>

      <div className="grid grid-cols-3 gap-3 text-sm">
        <Metric label="Baseline (best in data)" value={fmt(a.baseline_titer)} />
        <Metric label="Recommended (verified)" value={fmt(a.oracle_verified_titer)} accent />
        <Metric
          label="Improvement"
          value={`${a.improvement >= 0 ? "+" : ""}${fmt(a.improvement)}`}
        />
      </div>

      {edges.length > 0 && (
        <p className="text-[11px] text-ink-faint">
          {edges.length === 4 ? "All four" : `${edges.length}`} recommended settings sit at the edge of
          what your data and limits allow — the true optimum likely lies beyond them. Run a few experiments
          past these settings to push higher.
        </p>
      )}
    </div>
  );
}

// The active-learning story: how the agent reached the recommendation, in plain
// terms. Each cycle = the agent's guess vs what the simulator measured there.
function RefinementCard({ a }: { a: ActiveOptimization }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm">How the agent reached this</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="text-sm text-ink-muted">
          The agent built a model from your data, proposed a best setpoint, and checked it against the
          simulator — repeating {a.cycles === 1 ? "once" : `${a.cycles} times`}, learning from each gap.
        </p>

        <div className="space-y-1">
          {a.iterations.map((it) => (
            <div
              key={it.cycle}
              className="flex items-center gap-2 rounded-md border border-rule bg-surface-1 px-2 py-1.5 text-xs"
            >
              <span className="font-mono text-ink-faint">cycle {it.cycle + 1}</span>
              <span className="flex-1">
                model guessed <span className="font-mono">{fmt(it.predicted)}</span>, simulator measured{" "}
                <span className="font-mono text-accent">{fmt(it.oracle_verified)}</span>
                {(it.box_expansions ?? 0) > 0 && (
                  <span className="ml-1 text-ink-faint">
                    · searched beyond the starting range
                    {(it.box_expansions ?? 0) > 1 ? ` (widened ${it.box_expansions}×)` : ""}
                  </span>
                )}
              </span>
              {it.converged ? (
                <Badge variant="success" className="text-[10px]">matched</Badge>
              ) : (
                <span className="font-mono text-ink-faint">gap {fmt(it.error)}</span>
              )}
            </div>
          ))}
        </div>

        {a.converged ? (
          <p className="text-xs text-accent">
            ✓ The model’s prediction matched the simulator within {fmt(a.final_error)} — high confidence in the setpoint.
          </p>
        ) : (
          <p className="text-xs text-ink-faint">
            The recommendation is simulator-verified, but the model still {a.model_predicted_titer < a.oracle_verified_titer ? "under" : "over"}-predicts
            by ~{fmt(a.final_error)}. The setpoint is trustworthy; the model would sharpen with more experiments near the optimum.
          </p>
        )}

        {a.batches_added > 0 && (
          <p className="rounded-md border border-rule bg-surface-1 px-2 py-1.5 text-[11px] text-ink-muted">
            ↻ Added {a.batches_added} simulator-verified experiment{a.batches_added === 1 ? "" : "s"} to your
            dataset (now {a.total_batches} batches) — the system gets sharper next run.
          </p>
        )}

        {a.equations.length > 0 && (
          <details className="rounded-md border border-rule bg-surface-1 text-xs">
            <summary className="cursor-pointer px-2 py-1.5 text-ink-muted">
              View the model the agent built ({a.proposer === "llm" ? "LLM-written" : "structural search"})
            </summary>
            <pre className="overflow-x-auto border-t border-rule/60 bg-surface-0 p-2 font-mono text-[11px] leading-relaxed text-ink">
              {a.equations.join("\n")}
            </pre>
          </details>
        )}
      </CardContent>
    </Card>
  );
}

function Metric({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className="rounded-md border border-rule bg-surface-1 px-3 py-2">
      <div className="text-[11px] text-ink-faint">{label}</div>
      <div className={`font-mono text-sm ${accent ? "text-accent" : "text-ink"}`}>{value}</div>
    </div>
  );
}

function fmt(v: number | null | undefined): string {
  return v == null ? "—" : `${v.toFixed(1)} g/L`;
}
