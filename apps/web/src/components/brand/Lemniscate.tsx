"use client";

// The namesake motif — an animated lemniscate of Bernoulli (∞) with
// light tracers orbiting the curve, over a soft teal glow. Lemnisca's
// signature hero backdrop. Drawn programmatically; respects
// prefers-reduced-motion (renders static when reduced).

import { useEffect, useRef } from "react";

export function Lemniscate({ className }: { className?: string }) {
  const gref = useRef<SVGGElement | null>(null);

  useEffect(() => {
    const g = gref.current;
    if (!g) return;
    const NS = "http://www.w3.org/2000/svg";
    const cx = 450, cy = 260, a = 360;
    const path = (aa: number) => {
      const s = 260, p: string[] = [];
      for (let i = 0; i <= s; i++) {
        const t = (i / s) * Math.PI * 2;
        const d = 1 + Math.sin(t) ** 2;
        p.push(
          (cx + (aa * Math.cos(t)) / d).toFixed(1) +
            " " +
            (cy + (aa * Math.sin(t) * Math.cos(t)) / d).toFixed(1),
        );
      }
      return "M " + p.join(" L ") + " Z";
    };
    const mk = (n: string, at: Record<string, string | number>) => {
      const e = document.createElementNS(NS, n);
      for (const k in at) e.setAttribute(k, String(at[k]));
      g.appendChild(e);
      return e;
    };
    g.innerHTML = "";
    mk("path", { d: path(a + 34), fill: "none", stroke: "rgba(255,255,255,0.05)", "stroke-width": 1 });
    mk("path", { d: path(a - 48), fill: "none", stroke: "rgba(56,175,216,0.10)", "stroke-width": 1 });
    mk("path", { d: path(a), fill: "none", stroke: "rgba(56,175,216,0.30)", "stroke-width": 1.4, "stroke-linecap": "round" });
    const main = mk("path", { d: path(a), fill: "none", stroke: "url(#loopGrad)", "stroke-width": 2.8, "stroke-linecap": "round" }) as SVGPathElement;
    const L = main.getTotalLength();
    const tracers: { halo: Element; dot: Element; off: number }[] = [];
    for (let i = 0; i < 3; i++) {
      const halo = mk("circle", { r: 7, fill: "url(#tracerGrad)" });
      const dot = mk("circle", { r: 2.4, fill: "#eaf9ff" });
      tracers.push({ halo, dot, off: i / 3 });
    }
    const place = () =>
      tracers.forEach((tr) => {
        const pt = main.getPointAtLength(tr.off * L);
        tr.halo.setAttribute("cx", String(pt.x));
        tr.halo.setAttribute("cy", String(pt.y));
        tr.dot.setAttribute("cx", String(pt.x));
        tr.dot.setAttribute("cy", String(pt.y));
      });

    const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    if (reduce) {
      tracers.forEach((tr, i) => (tr.off = 0.12 + i / 3));
      place();
      return;
    }
    let raf = 0;
    const tick = () => {
      tracers.forEach((tr) => (tr.off = (tr.off + 0.0016) % 1));
      place();
      raf = requestAnimationFrame(tick);
    };
    tick();
    return () => cancelAnimationFrame(raf);
  }, []);

  return (
    <svg
      className={className}
      viewBox="0 0 900 520"
      preserveAspectRatio="xMidYMid meet"
      aria-hidden="true"
    >
      <defs>
        <linearGradient id="loopGrad" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0" stopColor="#38afd8" stopOpacity="0" />
          <stop offset="0.5" stopColor="#5bc7ec" stopOpacity="0.85" />
          <stop offset="1" stopColor="#38afd8" stopOpacity="0" />
        </linearGradient>
        <radialGradient id="tracerGrad">
          <stop offset="0" stopColor="#cdefff" />
          <stop offset="55%" stopColor="#5bc7ec" stopOpacity="0.7" />
          <stop offset="100%" stopColor="#38afd8" stopOpacity="0" />
        </radialGradient>
      </defs>
      <g ref={gref} />
    </svg>
  );
}
