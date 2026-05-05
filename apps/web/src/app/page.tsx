"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  createRun,
  listRuns,
  uploadFile,
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

export default function Home() {
  const router = useRouter();
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [userQuestion, setUserQuestion] = useState("");

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

  async function onFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      const up = await uploadFile(file);
      // Empty question = legacy run. Trim before sending so a textarea
      // full of whitespace doesn't trip the "non-empty" gate downstream.
      const trimmed = userQuestion.trim();
      const run = await createRun(up.upload_id, trimmed || undefined);
      router.push(`/runs/${run.run_id}`);
    } catch (e: any) {
      setError(String(e.message ?? e));
    } finally {
      setUploading(false);
      e.target.value = "";
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
              disabled={uploading}
              className="w-full rounded-md border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring resize-none"
            />
            <p className="mt-1 text-xs text-muted-foreground">
              Leave empty to run as today. When provided, the system biases
              every stage toward addressing your question and reports
              whether it could answer.
            </p>
          </div>
          <label className="inline-flex items-center gap-3">
            <input
              type="file"
              accept=".csv,.xlsx,.pdf,.zip"
              onChange={onFileChange}
              disabled={uploading}
              className="hidden"
              id="upload-file"
            />
            <Button asChild disabled={uploading}>
              <label htmlFor="upload-file" className="cursor-pointer">
                {uploading ? "Uploading…" : "Choose file"}
              </label>
            </Button>
            <span className="text-xs text-muted-foreground">
              .csv / .xlsx / .pdf / .zip
            </span>
            {error && (
              <span className="text-sm text-destructive">{error}</span>
            )}
          </label>
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
