"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  PROCESS_FAMILY_OPTIONS,
  createRun,
  listRuns,
  uploadFiles,
  type RunStatus,
  type RunSummary,
} from "@/lib/api";
import { formatRelative } from "@/lib/utils";

const STATUS_VARIANT: Record<RunStatus, "default" | "secondary" | "success" | "warning" | "destructive"> = {
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

// Allowed file extensions, matched server-side in apps/api/fermdocs_api/main.py.
// Keep these in sync with the API's ALLOWED_SUFFIXES set.
const ALLOWED_EXTS = [".csv", ".xlsx", ".pdf", ".zip"] as const;

function extOf(name: string): string {
  const i = name.lastIndexOf(".");
  return i >= 0 ? name.slice(i).toLowerCase() : "";
}

function validateTray(files: File[]): string | null {
  if (files.length === 0) return null; // empty is fine — just disables Submit
  const exts = files.map((f) => extOf(f.name));
  for (let i = 0; i < files.length; i++) {
    if (!ALLOWED_EXTS.includes(exts[i] as typeof ALLOWED_EXTS[number])) {
      return `${files[i].name}: unsupported file type. Allowed: .csv .xlsx .pdf .zip`;
    }
  }
  const zipCount = exts.filter((e) => e === ".zip").length;
  if (zipCount > 0 && files.length > 1) {
    return "Zip uploads must be standalone — cannot mix .zip with other files.";
  }
  const seen = new Set<string>();
  for (const f of files) {
    if (seen.has(f.name)) {
      return `Duplicate filename: ${f.name}. Rename one or remove it.`;
    }
    seen.add(f.name);
  }
  return null;
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function Home() {
  const router = useRouter();
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [userQuestion, setUserQuestion] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  // Operator-supplied process family (upload-process-family-ui). Default
  // "auto-detect" runs the LLM identity extractor as before; any
  // specific pick short-circuits that and writes a manifest so the
  // dossier carries the canonical name immediately. Critical for
  // CSV-only uploads where the LLM extractor has nothing to read.
  const [processFamily, setProcessFamily] = useState<string>("auto-detect");

  async function refreshRuns() {
    try {
      const r = await listRuns();
      setRuns(r.runs);
    } catch (e) {
      // backend may be down — silent in v0.5b
    }
  }

  useEffect(() => {
    refreshRuns();
    const id = setInterval(refreshRuns, 3000);
    return () => clearInterval(id);
  }, []);

  // Live validation; recomputed every render off the tray contents.
  // Cheap (small N), so no memoization.
  const validationError = validateTray(files);

  function onAddFiles(e: React.ChangeEvent<HTMLInputElement>) {
    const picked = Array.from(e.target.files ?? []);
    if (picked.length === 0) return;
    // Append to the tray. Reset the input so the same file can be re-picked
    // after a remove.
    setFiles((prev) => [...prev, ...picked]);
    e.target.value = "";
    setError(null);
  }

  function onRemoveFile(idx: number) {
    setFiles((prev) => prev.filter((_, i) => i !== idx));
    setError(null);
  }

  async function onSubmit() {
    if (files.length === 0 || validationError || submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      const up = await uploadFiles(files, processFamily);
      // Empty question = legacy run. Trim before sending so a textarea
      // full of whitespace doesn't trip the "non-empty" gate downstream.
      const trimmed = userQuestion.trim();
      const run = await createRun(up.upload_id, trimmed || undefined);
      router.push(`/runs/${run.run_id}`);
    } catch (e: any) {
      setError(String(e.message ?? e));
      setSubmitting(false);
    }
  }

  return (
    <div className="space-y-8">
      <Card>
        <CardHeader>
          <CardTitle>Upload</CardTitle>
          <CardDescription>
            Upload a <code>.csv</code>, <code>.xlsx</code>, or <code>.pdf</code> —
            the full pipeline runs (ingest → characterize → diagnose → hypothesize).
            Or upload a <code>.zip</code> of an existing diagnose bundle to jump
            straight to the hypothesis stage. The system will ask you to answer
            any open questions it raises.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <label
              htmlFor="user-question"
              className="block text-sm font-medium mb-1"
            >
              Your question (optional)
            </label>
            <textarea
              id="user-question"
              value={userQuestion}
              onChange={(e) => setUserQuestion(e.target.value)}
              placeholder="e.g. Why did pigment loss happen earlier in BATCH-04 than BATCH-05?"
              maxLength={2000}
              rows={3}
              disabled={submitting}
              className="w-full rounded-md border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring resize-none"
            />
            <p className="mt-1 text-xs text-muted-foreground">
              Leave empty to run as today. When provided, the system biases
              every stage toward addressing your question and reports
              whether it could answer.
            </p>
          </div>
          {/* Process family selector (upload-process-family-ui).
              Closed enum from process_families.yaml. Auto-detect runs the
              existing LLM identity path (works on PDFs / mixed uploads).
              An explicit pick short-circuits the LLM and writes a
              manifest — required for CSV-only uploads where there's no
              narrative for the model to read. */}
          <div>
            <label
              htmlFor="process-family"
              className="block text-sm font-medium mb-1"
            >
              Process family
            </label>
            <select
              id="process-family"
              value={processFamily}
              onChange={(e) => setProcessFamily(e.target.value)}
              disabled={submitting}
              className="w-full rounded-md border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
            >
              {PROCESS_FAMILY_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
            <p className="mt-1 text-xs text-muted-foreground">
              {PROCESS_FAMILY_OPTIONS.find((o) => o.value === processFamily)?.description}
            </p>
          </div>
          {/* File tray (PR-A3, frontend-redesign): add files one at a time,
              review before submitting. Submit button gates the run start —
              picking a file no longer fires the pipeline. */}
          <div className="space-y-3">
            <div className="flex items-center gap-3">
              <input
                type="file"
                accept=".csv,.xlsx,.pdf,.zip"
                onChange={onAddFiles}
                disabled={submitting}
                className="hidden"
                id="upload-file"
                multiple
              />
              <Button asChild disabled={submitting} variant="secondary">
                <label htmlFor="upload-file" className="cursor-pointer">
                  {files.length === 0 ? "Add file" : "Add another file"}
                </label>
              </Button>
              <span className="text-xs text-muted-foreground">
                .csv / .xlsx / .pdf / .zip
              </span>
            </div>

            {files.length > 0 && (
              <ul className="divide-y rounded-md border bg-card text-sm">
                {files.map((f, idx) => (
                  <li
                    key={`${f.name}-${idx}`}
                    className="flex items-center justify-between px-3 py-2"
                  >
                    <div className="min-w-0 flex-1">
                      <div className="truncate font-mono text-xs">
                        {f.name}
                      </div>
                      <div className="text-xs text-muted-foreground">
                        {formatSize(f.size)}
                      </div>
                    </div>
                    <button
                      type="button"
                      onClick={() => onRemoveFile(idx)}
                      disabled={submitting}
                      className="text-muted-foreground hover:text-destructive disabled:opacity-50"
                      aria-label={`Remove ${f.name}`}
                    >
                      ✕
                    </button>
                  </li>
                ))}
              </ul>
            )}

            {validationError && (
              <p className="text-sm text-destructive">{validationError}</p>
            )}
            {error && !validationError && (
              <p className="text-sm text-destructive">{error}</p>
            )}

            <div className="flex items-center gap-3 pt-1">
              <Button
                onClick={onSubmit}
                disabled={
                  submitting || files.length === 0 || validationError !== null
                }
              >
                {submitting ? "Uploading…" : "Submit"}
              </Button>
              {!submitting && files.length === 0 && (
                <span className="text-xs text-muted-foreground">
                  Add at least one file to enable Submit.
                </span>
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      <section>
        <div className="flex items-baseline justify-between mb-3">
          <h2 className="text-lg font-semibold">Recent runs</h2>
          <button
            onClick={refreshRuns}
            className="text-xs text-muted-foreground hover:text-foreground"
          >
            refresh
          </button>
        </div>
        {runs.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            No runs yet. Upload a bundle above to start.
          </p>
        ) : (
          <ul className="space-y-2">
            {runs.map((r) => (
              <li key={r.run_id}>
                <a
                  href={`/runs/${r.run_id}`}
                  className="block rounded-md border bg-card px-4 py-3 hover:bg-accent transition-colors"
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="font-mono text-sm">{r.run_id.slice(0, 8)}</div>
                      <div className="text-xs text-muted-foreground">
                        started {formatRelative(r.created_at)}
                      </div>
                    </div>
                    <Badge variant={STATUS_VARIANT[r.status]}>{r.status}</Badge>
                  </div>
                  {r.error && (
                    <div className="mt-2 text-xs text-destructive">{r.error}</div>
                  )}
                </a>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
