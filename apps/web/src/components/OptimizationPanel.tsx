"use client";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { EquationDiscovery, ModelLogEntry, OptimizationOutput } from "@/lib/api";

// Renders the optimization workflow result: the debated levers (the prior), the
// closed-loop optimum when a simulator ran (the oracle-verified posterior), and
// the model log — the governing equations and per-round fits, i.e. exactly how
// the agent is using the model.
export function OptimizationPanel({ opt }: { opt: OptimizationOutput }) {
  return (
    <section className="space-y-3">
      <h2 className="text-heading-sm">Optimization</h2>

      <Card className={opt.confident ? "border-primary" : ""}>
        <CardHeader>
          <div className="flex items-center justify-between gap-2">
            <CardTitle className="text-sm">Optimization plan</CardTitle>
            <div className="flex items-center gap-2">
              <Badge variant={opt.simulator_available ? "success" : "secondary"}>
                {opt.simulator_available ? "Simulator-verified" : "Debate-only (no simulator)"}
              </Badge>
              {!opt.confident && opt.refusal_reason && (
                <Badge variant="destructive">{opt.refusal_reason}</Badge>
              )}
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {opt.discovery ? (
            // The plan IS gemini's discovered-model recommendation.
            <GeminiRecommendation d={opt.discovery} baseline={opt.baseline_titer} />
          ) : (
            // Fallback only when discovery did not run (no LLM / disabled).
            <>
              {opt.selection_rationale && (
                <p className="text-sm text-ink-muted whitespace-pre-wrap">
                  {opt.selection_rationale}
                </p>
              )}
              {opt.best_achieved_titer != null && (
                <div className="grid grid-cols-3 gap-3 text-sm">
                  <Metric label="Baseline" value={fmt(opt.baseline_titer)} />
                  <Metric label="Achieved" value={fmt(opt.best_achieved_titer)} accent />
                  <Metric
                    label="Improvement"
                    value={opt.improvement != null ? `${opt.improvement >= 0 ? "+" : ""}${fmt(opt.improvement)}` : "—"}
                  />
                </div>
              )}
              {opt.best_candidate && (
                <div>
                  <p className="text-xs font-medium text-ink-muted mb-1">
                    Best operating point (oracle-verified knobs)
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(opt.best_candidate).map(([k, v]) => (
                      <span key={k} className="rounded-md border border-rule bg-surface-1 px-2 py-1 font-mono text-xs">
                        {k} = {typeof v === "number" ? v.toPrecision(4) : String(v)}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </CardContent>
      </Card>

      {/* Debated levers — the prior over the knobs. */}
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
                {lev.supporting_specialists.length > 0 && (
                  <p className="mt-1 text-[11px] text-ink-faint">
                    via {lev.supporting_specialists.join(", ")}
                  </p>
                )}
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {/* Equation discovery — agent writes the ODE, oracle refines it, scipy
          searches the equation, oracle verifies the setpoint. */}
      {opt.discovery && <DiscoveryCard d={opt.discovery} />}

      {/* Governing equations only — per-round fit logs are intentionally omitted. */}
      {opt.model_log.some((e) => e.kind === "equations") && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Model — governing equations</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {opt.model_log
              .filter((e) => e.kind === "equations" || e.kind === "note")
              .map((entry, i) => (
                <ModelLogItem key={i} entry={entry} />
              ))}
          </CardContent>
        </Card>
      )}
    </section>
  );
}

// The optimization plan itself: what the gemini-discovered model recommends —
// an actionable setpoint, the oracle-verified titer (the number to trust), and a
// box-edge note when the optimum is pinned to the limits.
function GeminiRecommendation({ d, baseline }: { d: EquationDiscovery; baseline: number | null }) {
  const who = d.proposer === "llm" ? "Gemini's discovered model" : "The discovered model";
  const edges = Object.keys(d.knobs_on_boundary);
  return (
    <div className="space-y-4">
      <p className="text-sm text-ink-muted">{who} recommends running these knobs:</p>
      <div className="flex flex-wrap gap-2">
        {Object.entries(d.predicted_knobs).map(([k, v]) => (
          <span key={k} className="rounded-md border border-rule bg-surface-1 px-2 py-1 font-mono text-xs">
            {k} = {v}
            {d.knobs_on_boundary[k] && (
              <span className="ml-1 text-accent">({d.knobs_on_boundary[k]} edge)</span>
            )}
          </span>
        ))}
      </div>
      <div className="grid grid-cols-3 gap-3 text-sm">
        <Metric label="Baseline" value={fmt(baseline)} />
        <Metric label="Oracle-verified" value={fmt(d.oracle_verified_titer)} accent />
        <Metric
          label="Capture of box max"
          value={d.capture_pct != null ? `${d.capture_pct}%` : "—"}
        />
      </div>
      <p className="text-xs text-ink-faint">
        The model itself predicted {fmt(d.predicted_optimum_titer)} — trust the oracle-verified value.
        {d.oracle_true_max != null && <> Box maximum is {fmt(d.oracle_true_max)}.</>}
      </p>
      {edges.length > 0 && (
        <p className="text-[11px] text-ink-faint">
          All {edges.length === 4 ? "four" : edges.length} recommended knobs sit on the box edge — the true
          optimum likely lies outside these limits. Widen var_params to push higher.
        </p>
      )}
    </div>
  );
}

function DiscoveryCard({ d }: { d: EquationDiscovery }) {
  const bestRound = d.rounds.reduce(
    (lo, r) => (r.compile_ok && r.oracle_peak_rmse < lo ? r.oracle_peak_rmse : lo),
    Infinity,
  );
  return (
    <Card className="border-accent/40">
      <CardHeader>
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="text-sm">
            Equation discovery — agent builds the model, oracle judges it
          </CardTitle>
          <Badge variant="outline" className="text-[10px] uppercase">
            {d.proposer === "llm" ? "LLM-written" : "structural search"}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Iterative improvement vs the oracle */}
        <div>
          <p className="text-xs font-medium text-ink-muted mb-1">
            Iterative refinement vs oracle truth (lower RMSE = closer to ground truth)
          </p>
          <div className="space-y-1">
            {d.rounds.map((r) => (
              <details
                key={r.round_index}
                className={`rounded-md border text-xs ${
                  r.oracle_peak_rmse === bestRound
                    ? "border-accent bg-accent-glow/10"
                    : "border-rule bg-surface-1"
                }`}
              >
                <summary className="flex cursor-pointer items-center gap-2 px-2 py-1 list-none">
                  <span className="font-mono text-ink-faint">r{r.round_index}</span>
                  <span className="flex-1 truncate">{r.name}</span>
                  {r.oracle_peak_rmse === bestRound && (
                    <Badge variant="success" className="text-[10px]">best</Badge>
                  )}
                  {!r.compile_ok && (
                    <Badge variant="destructive" className="text-[10px]">compile error</Badge>
                  )}
                  <span className="font-mono text-ink-muted">
                    RMSE {r.oracle_peak_rmse.toFixed(2)} g/L
                  </span>
                  <span className="font-mono text-accent">R² {r.oracle_peak_r2.toFixed(2)}</span>
                </summary>
                <div className="border-t border-rule/60 px-2 py-2 space-y-2">
                  {r.notes && <p className="text-[11px] text-ink-muted italic">{r.notes}</p>}
                  {r.error && <p className="text-[11px] text-destructive">{r.error}</p>}
                  {r.equations.length > 0 && (
                    <pre className="overflow-x-auto rounded bg-surface-0 p-2 font-mono text-[11px] leading-relaxed text-ink">
                      {r.equations.join("\n")}
                    </pre>
                  )}
                  {Object.keys(r.fitted_params).length > 0 && (
                    <div className="flex flex-wrap gap-1.5">
                      {Object.entries(r.fitted_params).map(([k, v]) => (
                        <span key={k} className="rounded border border-rule px-1.5 py-0.5 font-mono text-[10px]">
                          {k}={v}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </details>
            ))}
          </div>
          <p className="mt-1 text-[11px] text-ink-faint">Click a round to see the equations gemini wrote.</p>
        </div>

        {/* The best discovered equation */}
        <div>
          <p className="text-xs font-medium text-ink-muted mb-1">
            Best discovered equation: <span className="font-mono">{d.best_name}</span>
            {" "}(oracle peak RMSE {d.best_oracle_peak_rmse.toFixed(2)} g/L, R²{" "}
            {d.best_oracle_peak_r2.toFixed(2)})
          </p>
          <pre className="overflow-x-auto rounded bg-surface-0 p-2 font-mono text-[11px] leading-relaxed text-ink">
            {d.best_equations.join("\n")}
          </pre>
          {Object.keys(d.best_fitted_params).length > 0 && (
            <div className="mt-2 flex flex-wrap gap-1.5">
              {Object.entries(d.best_fitted_params).map(([k, v]) => (
                <span key={k} className="rounded border border-rule px-1.5 py-0.5 font-mono text-[10px]">
                  {k}={v}
                </span>
              ))}
            </div>
          )}
        </div>

        {/* scipy search on the equation, verified on the oracle */}
        <div>
          <p className="text-xs font-medium text-ink-muted mb-1">
            Optimum via {d.search_method} on the equation ({d.search_evals} evals), verified on oracle
          </p>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <Metric label="Equation predicted" value={fmt(d.predicted_optimum_titer)} />
            <Metric label="Oracle verified" value={fmt(d.oracle_verified_titer)} accent />
            <Metric label="Oracle true max" value={fmt(d.oracle_true_max)} />
            <Metric
              label="Capture"
              value={d.capture_pct != null ? `${d.capture_pct}%` : "—"}
            />
          </div>
          <div className="mt-2 flex flex-wrap gap-2">
            {Object.entries(d.predicted_knobs).map(([k, v]) => (
              <span key={k} className="rounded-md border border-rule bg-surface-1 px-2 py-1 font-mono text-xs">
                {k} = {v}
                {d.knobs_on_boundary[k] && (
                  <span className="ml-1 text-accent">({d.knobs_on_boundary[k]} edge)</span>
                )}
              </span>
            ))}
          </div>
          {Object.keys(d.knobs_on_boundary).length > 0 && (
            <p className="mt-1.5 text-[11px] text-ink-faint">
              Knobs on the box edge — the true optimum may lie outside these limits; widen var_params to test.
            </p>
          )}
          {d.appended_to_training && (
            <p className="mt-2 rounded-md border border-rule bg-surface-1 px-2 py-1.5 text-[11px] text-ink-muted">
              ↻ Active learning: this setpoint was simulated on LABS and folded into the
              training data as batch {d.appended_to_training.batch_id} (oracle titer{" "}
              {d.appended_to_training.peak_titer.toFixed(1)} g/L). Training set now{" "}
              {d.appended_to_training.total_batches} batches — the next run learns from it.
            </p>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

function ModelLogItem({ entry }: { entry: ModelLogEntry }) {
  return (
    <div className="rounded-md border border-rule bg-surface-1 p-3">
      <div className="flex items-center gap-2">
        <Badge variant="outline" className="text-[10px] uppercase">{entry.kind}</Badge>
        <span className="text-sm font-medium">{entry.title}</span>
      </div>
      {entry.detail && <p className="mt-1 text-xs text-ink-muted">{entry.detail}</p>}

      {/* equations block (model card) */}
      {(entry as any).equations && (
        <pre className="mt-2 overflow-x-auto rounded bg-surface-0 p-2 font-mono text-[11px] leading-relaxed text-ink">
          {((entry as any).equations as string[]).join("\n")}
        </pre>
      )}
      {(entry as any).method && (
        <p className="mt-2 text-[11px] text-ink-faint">{(entry as any).method}</p>
      )}

      {/* fit block: params + R² */}
      {entry.fitted_params && Object.keys(entry.fitted_params).length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {Object.entries(entry.fitted_params).map(([k, v]) => (
            <span key={k} className="rounded border border-rule px-1.5 py-0.5 font-mono text-[10px]">
              {k}={typeof v === "number" ? v.toPrecision(3) : String(v)}
            </span>
          ))}
        </div>
      )}
      {entry.r2_by_species && Object.keys(entry.r2_by_species).length > 0 && (
        <div className="mt-1.5 flex flex-wrap gap-1.5">
          {Object.entries(entry.r2_by_species).map(([k, v]) => (
            <span key={k} className="rounded bg-accent-glow/20 px-1.5 py-0.5 font-mono text-[10px] text-accent">
              R²({k})={typeof v === "number" ? v.toFixed(2) : String(v)}
            </span>
          ))}
        </div>
      )}
    </div>
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
