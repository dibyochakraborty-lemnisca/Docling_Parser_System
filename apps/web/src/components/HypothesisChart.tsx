"use client";

// Plotly figure renderer for hypothesis cards.
//
// react-plotly.js needs window/document at import time, so we dynamic-
// import it client-side only and ship a no-SSR loader. Charts are
// styled to the Lemnisca instrument panel: transparent paper, faint
// white gridlines, off-white labels, teal-led colorway.

import dynamic from "next/dynamic";
import type { PlotlyFigure } from "@/lib/api";

const Plot = dynamic(() => import("react-plotly.js"), {
  ssr: false,
  loading: () => (
    <div className="flex h-64 items-center justify-center rounded-md border border-rule bg-surface-2/40 font-ui text-ui-xs text-ink-muted">
      loading chart…
    </div>
  ),
});

// Lemnisca palette for data: teal signal first, then a restrained ramp.
const COLORWAY = ["#38afd8", "#5bc7ec", "#3fbfa6", "#e3a552", "#c7cbce", "#1c7c9c"];

const AXIS = {
  gridcolor: "rgba(255,255,255,0.07)",
  zerolinecolor: "rgba(255,255,255,0.16)",
  linecolor: "rgba(255,255,255,0.18)",
  tickcolor: "rgba(255,255,255,0.18)",
  tickfont: { color: "#92989c", size: 11 },
  titlefont: { color: "#c7cbce", size: 12 },
};

const BASE_LAYOUT = {
  autosize: true,
  font: { family: '"Helvetica Neue", Helvetica, Inter, system-ui, sans-serif', size: 12, color: "#c7cbce" },
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  colorway: COLORWAY,
  legend: { font: { color: "#c7cbce", size: 11 } },
  margin: { t: 28, r: 16, b: 40, l: 48 },
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
  const incoming = (figure.layout ?? {}) as Record<string, any>;
  // Deep-merge axes so figure-supplied titles/ranges survive while the
  // dark instrument colors apply as defaults.
  const layout = {
    ...BASE_LAYOUT,
    ...incoming,
    font: { ...BASE_LAYOUT.font, ...(incoming.font ?? {}) },
    legend: { ...BASE_LAYOUT.legend, ...(incoming.legend ?? {}) },
    xaxis: { ...AXIS, ...(incoming.xaxis ?? {}) },
    yaxis: { ...AXIS, ...(incoming.yaxis ?? {}) },
  };
  return (
    <div className="mt-3 rounded-md border border-rule bg-surface-1 p-3">
      <Plot
        data={figure.data as Plotly.Data[]}
        layout={layout as unknown as Plotly.Layout}
        config={BASE_CONFIG as Plotly.Config}
        style={{ width: "100%", height: "360px" }}
        useResizeHandler
      />
      {figure.spec.rationale ? (
        <p className="mt-2 font-ui text-ui-xs text-ink-muted">
          <span className="font-medium text-ink">Why this chart:</span>{" "}
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
