"use client";

// Top-right "Runs" dropdown for the homepage. Replaces the old full-width
// "Recent runs" section: a single mono button that opens a panel listing
// recent runs to navigate into. Closes on outside-click / Escape.

import { useEffect, useRef, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { formatRelative } from "@/lib/utils";
import type { RunStatus, RunSummary } from "@/lib/api";

const STATUS_VARIANT: Record<
  RunStatus,
  "default" | "secondary" | "success" | "warning" | "destructive"
> = {
  pending: "secondary",
  ingesting: "secondary",
  characterizing: "secondary",
  diagnosing: "secondary",
  hypothesizing: "secondary",
  paused: "warning",
  resuming: "secondary",
  done: "success",
  failed: "destructive",
};

export function RunsMenu({
  runs,
  onRefresh,
}: {
  runs: RunSummary[];
  onRefresh: () => void;
}) {
  const [open, setOpen] = useState(false);
  const wrapRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!open) return;
    function onDown(e: MouseEvent) {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div ref={wrapRef} className="relative">
      <button
        type="button"
        onClick={() => {
          if (!open) onRefresh();
          setOpen((v) => !v);
        }}
        aria-haspopup="menu"
        aria-expanded={open}
        className="inline-flex items-center gap-2 rounded-full border border-rule bg-surface-1 px-4 py-2 font-ui text-ui-xs uppercase tracking-[0.12em] text-ink-muted transition-colors hover:border-accent-deep hover:text-ink"
      >
        Runs
        <span className="rounded-full bg-surface-2 px-1.5 py-0.5 text-[10px] tabular-nums text-ink-faint">
          {runs.length}
        </span>
        <span
          aria-hidden="true"
          className={`text-[9px] transition-transform ${open ? "rotate-180" : ""}`}
        >
          ▾
        </span>
      </button>

      {open && (
        <div
          role="menu"
          className="animate-fade-up absolute right-0 z-50 mt-2 w-[min(92vw,30rem)] overflow-hidden rounded-lg border border-rule bg-surface-1 shadow-glow-soft"
        >
          <div className="flex items-center justify-between border-b border-rule-soft px-4 py-2.5">
            <p className="kicker">Recent runs</p>
            <button
              onClick={onRefresh}
              className="font-ui text-ui-xs uppercase tracking-[0.1em] text-ink-muted transition-colors hover:text-accent"
            >
              refresh
            </button>
          </div>
          {runs.length === 0 ? (
            <p className="px-4 py-6 text-center text-sm text-ink-faint">
              No runs yet. Upload a bundle to start.
            </p>
          ) : (
            <ul className="max-h-[min(60vh,28rem)] divide-y divide-rule-soft overflow-y-auto">
              {runs.map((r) => (
                <li key={r.run_id}>
                  <a
                    href={`/runs/${r.run_id}`}
                    className="block px-4 py-3 transition-colors hover:bg-surface-2"
                  >
                    <div className="flex items-center justify-between gap-3">
                      <div className="min-w-0">
                        <div className="font-ui text-ui-sm text-ink">
                          {r.run_id.slice(0, 8)}
                        </div>
                        <div className="font-ui text-ui-xs text-ink-muted">
                          started {formatRelative(r.created_at)}
                        </div>
                      </div>
                      <Badge variant={STATUS_VARIANT[r.status]}>{r.status}</Badge>
                    </div>
                    {r.error && (
                      <div className="mt-1.5 text-xs text-destructive">
                        {r.error}
                      </div>
                    )}
                  </a>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
