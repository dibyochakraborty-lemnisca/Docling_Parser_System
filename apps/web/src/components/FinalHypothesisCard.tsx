"use client";

// Final-hypothesis card. When the hypothesis carries charts, the layout
// splits two-up on wide screens — reasoning text on the left, the related
// Plotly figures on the right — instead of stacking them vertically. Falls
// back to a single column when there are no charts.

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { HypothesisCharts } from "@/components/HypothesisChart";
import type { FinalHypothesis } from "@/lib/api";

function answeredVariant(a: FinalHypothesis["question_answered"]) {
  return a === "yes" ? "success" : a === "partial" ? "warning" : "secondary";
}

function answeredLabel(a: FinalHypothesis["question_answered"]) {
  return a === "yes"
    ? "Answered"
    : a === "partial"
    ? "Partially answered"
    : "Insufficient data";
}

export function FinalHypothesisCard({
  h,
  questionLabel = "Your question",
  showCitations = true,
}: {
  h: FinalHypothesis;
  questionLabel?: string;
  showCitations?: boolean;
}) {
  const hasCharts = (h.plotly_charts?.length ?? 0) > 0;

  const reasoning = (
    <div className="space-y-3">
      {h.question_answered && (
        <div className="rounded-md border border-accent/30 bg-accent-soft px-3 py-2">
          <div className="flex items-center gap-2 text-xs">
            <span className="font-medium">{questionLabel}:</span>
            <Badge variant={answeredVariant(h.question_answered)}>
              {answeredLabel(h.question_answered)}
            </Badge>
          </div>
          {h.question_response_summary && (
            <p className="mt-2 text-sm leading-relaxed">{h.question_response_summary}</p>
          )}
        </div>
      )}
      <p className="text-sm leading-relaxed">{h.summary}</p>
      {h.actionable_recommendation && (
        <div className="rounded-md border-l-4 border-l-primary bg-primary/5 px-3 py-2">
          <div className="text-xs font-medium uppercase tracking-wide text-primary">
            Recommendation
          </div>
          <p className="mt-1 text-sm leading-relaxed">{h.actionable_recommendation}</p>
        </div>
      )}
      {showCitations && (
        <>
          {h.affected_variables.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {h.affected_variables.map((v) => (
                <Badge key={v} variant="secondary" className="font-mono text-xs">
                  {v}
                </Badge>
              ))}
            </div>
          )}
          <div className="text-xs text-muted-foreground">
            cites {h.cited_finding_ids.length} findings,{" "}
            {h.cited_narrative_ids.length} narratives,{" "}
            {h.cited_trajectories.length} trajectories
          </div>
        </>
      )}
    </div>
  );

  return (
    <Card>
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
        {hasCharts ? (
          <div className="grid gap-6 lg:grid-cols-2 lg:items-start">
            {reasoning}
            <div className="lg:sticky lg:top-24">
              <HypothesisCharts figures={h.plotly_charts} />
            </div>
          </div>
        ) : (
          reasoning
        )}
      </CardContent>
    </Card>
  );
}
