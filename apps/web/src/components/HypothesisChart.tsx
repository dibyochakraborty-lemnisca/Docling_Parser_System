"use client";

// Plotly figure renderer for hypothesis cards.
//
// react-plotly.js needs window/document at import time, so we dynamic-
// import it client-side only and ship a no-SSR loader. The fallback is
// a small block while the bundle hydrates.

import dynamic from "next/dynamic";
import type { PlotlyFigure } from "@/lib/api";

const Plot = dynamic(() => import("react-plotly.js"), {
  ssr: false,
  loading: () => (
    <div className="flex h-64 items-center justify-center rounded-md border bg-muted/30 text-xs text-muted-foreground">
      loading chart…
    </div>
  ),
});

const BASE_LAYOUT = {
  autosize: true,
  font: { family: "system-ui, sans-serif", size: 12 },
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
};

const BASE_CONFIG = {
  displaylogo: false,
  responsive: true,
  modeBarButtonsToRemove: [
    "sendDataToCloud",
    "toggleSpikelines",
    "hoverCompareCartesian",
    "hoverClosestCartesian",
  ],
};

export function HypothesisChart({ figure }: { figure: PlotlyFigure }) {
  const layout = {
    ...BASE_LAYOUT,
    ...(figure.layout as Record<string, unknown>),
  };
  return (
    <div className="mt-3 rounded-md border bg-card p-3">
      <Plot
        data={figure.data as Plotly.Data[]}
        layout={layout as Plotly.Layout}
        config={BASE_CONFIG as Plotly.Config}
        style={{ width: "100%", height: "360px" }}
        useResizeHandler
      />
      {figure.spec.rationale ? (
        <p className="mt-2 text-xs text-muted-foreground">
          <span className="font-medium text-foreground">Why this chart:</span>{" "}
          {figure.spec.rationale}
        </p>
      ) : null}
    </div>
  );
}

export function HypothesisCharts({ figures }: { figures: PlotlyFigure[] | undefined }) {
  if (!figures || figures.length === 0) return null;
  return (
    <div className="mt-2 space-y-3">
      {figures.map((fig, idx) => (
        <HypothesisChart key={idx} figure={fig} />
      ))}
    </div>
  );
}
