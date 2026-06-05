"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Lemniscate } from "@/components/brand/Lemniscate";
import { RunsMenu } from "@/components/RunsMenu";
import {
  PROCESS_FAMILY_OPTIONS,
  createRun,
  listRuns,
  uploadFiles,
  type RunSummary,
  type WorkflowKind,
} from "@/lib/api";

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
  // Which workflow the run uses. Fault-finding is the existing diagnose →
  // hypothesize → recommend pipeline; optimization runs the opportunity debate
  // (and the closed-loop optimizer where a process simulator exists).
  const [workflow, setWorkflow] = useState<WorkflowKind>("diagnostic");

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
      const run = await createRun(up.upload_id, trimmed || undefined, workflow);
      router.push(`/runs/${run.run_id}`);
    } catch (e: any) {
      setError(String(e.message ?? e));
      setSubmitting(false);
    }
  }

  return (
    <div className="space-y-8">
      {/* Top bar — recent runs live behind a top-right dropdown now. */}
      <div className="flex items-center justify-end">
        <RunsMenu runs={runs} onRefresh={refreshRuns} />
      </div>

      <div className="grid items-start gap-10 lg:grid-cols-[1fr_minmax(0,30rem)] lg:gap-14">
        {/* LEFT — hero copy over the animated lemniscate motif. */}
        <section className="relative isolate flex flex-col">
          <p className="kicker kicker-accent">Fermentation hypothesis &amp; recommendation engine</p>
          <h1 className="mt-4 max-w-[18ch] text-display-lg">
            From raw batch data to tested hypotheses and{" "}
            <span className="serif-accent text-accent">model-backed recommendations.</span>
          </h1>
          <p className="mt-5 max-w-prose text-body-lg text-ink-muted">
            Upload a run bundle and the multi-agent pipeline ingests, characterizes,
            diagnoses, and debates its way to ranked, evidence-cited hypotheses —
            then fits models to recommend the change for your next run.
          </p>
          <div className="relative mt-10 lg:mt-14">
            <div className="pointer-events-none absolute left-1/2 top-1/2 h-[300px] w-[560px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-accent-glow blur-3xl" />
            <Lemniscate className="relative w-full max-w-[560px]" />
          </div>
        </section>

        {/* RIGHT — upload well. */}
        <Card>
        <CardHeader>
          <CardTitle>Upload</CardTitle>
          <CardDescription>
            Upload a <code>.csv</code>, <code>.xlsx</code>, or <code>.pdf</code> —
            the full pipeline runs (ingest → characterize → diagnose → hypothesize
            → recommend). Or upload a <code>.zip</code> of an existing diagnose
            bundle to jump straight to the hypothesis stage. The system will ask
            you to answer any open questions it raises.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Workflow selector — two checkboxes, mutually exclusive. Picks
              which pipeline the run uses. */}
          <div>
            <span className="block text-sm font-medium mb-2">Workflow</span>
            <div className="grid gap-2 sm:grid-cols-2">
              {([
                {
                  value: "diagnostic" as WorkflowKind,
                  title: "Fault finding",
                  desc: "Diagnose what went wrong and recommend a fix.",
                },
                {
                  value: "optimization" as WorkflowKind,
                  title: "Optimization",
                  desc: "Find levers to push the target variable higher — even on a healthy run.",
                },
              ]).map((opt) => {
                const checked = workflow === opt.value;
                return (
                  <button
                    type="button"
                    key={opt.value}
                    role="checkbox"
                    aria-checked={checked}
                    disabled={submitting}
                    onClick={() => setWorkflow(opt.value)}
                    className={`flex items-start gap-3 rounded-md border px-3 py-2 text-left transition-colors disabled:opacity-50 ${
                      checked
                        ? "border-accent-deep bg-accent-glow/20 shadow-glow-soft"
                        : "border-rule bg-surface-1 hover:border-accent-deep/60"
                    }`}
                  >
                    <span
                      aria-hidden
                      className={`mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded-[4px] border text-[10px] font-bold ${
                        checked
                          ? "border-accent-deep bg-accent text-surface-0"
                          : "border-rule text-transparent"
                      }`}
                    >
                      ✓
                    </span>
                    <span className="min-w-0">
                      <span className="block text-ui-base font-medium text-ink">
                        {opt.title}
                      </span>
                      <span className="block text-xs text-ink-muted">{opt.desc}</span>
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
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
              className="w-full resize-none rounded-md border border-rule bg-surface-1 px-3 py-2 text-ui-base text-ink placeholder:text-ink-faint transition-colors focus:border-accent-deep focus:shadow-glow-soft focus:outline-none"
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
              className="w-full rounded-md border border-rule bg-surface-1 px-3 py-2 text-ui-base text-ink transition-colors focus:border-accent-deep focus:shadow-glow-soft focus:outline-none"
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
              <ul className="divide-y divide-rule rounded-md border border-rule bg-surface-1 text-sm">
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
      </div>
    </div>
  );
}
