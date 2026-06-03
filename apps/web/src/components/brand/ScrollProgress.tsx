"use client";

// The pinned 2px teal scroll-progress bar — a Lemnisca signature. Glows
// faintly; tracks document scroll.

import { useEffect, useState } from "react";

export function ScrollProgress() {
  const [pct, setPct] = useState(0);
  useEffect(() => {
    const onScroll = () => {
      const h = document.documentElement;
      const max = h.scrollHeight - h.clientHeight;
      setPct(max > 0 ? (h.scrollTop / max) * 100 : 0);
    };
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => {
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
    };
  }, []);
  return (
    <div
      className="scroll-progress fixed left-0 top-0 z-[100] h-0.5 bg-accent shadow-glow transition-[width] duration-100"
      style={{ width: `${pct}%` }}
      aria-hidden="true"
    />
  );
}
