# Multi-file upload + explicit Submit gate

**Branch:** `frontend-redesign` (off `hitl-followup`)
**Status:** Plan locked — about to start commit 1.
**Companion plans:**
- `plans/2026-05-04-user-question-and-hitl.md` — PR-A (bias posture).
- `plans/2026-05-05-hitl-followup.md` — PR-A2 (drive posture follow-ups).

**Estimated cost:** ~600 LOC across 3 commits + 12 tests.

---

## Why

Two current bugs from a UX standpoint:

1. **Picking a file fires the pipeline immediately.** `apps/web/src/app/page.tsx`
   `onFileChange` calls `uploadFile` then `createRun` the moment a file lands
   in the input. The user can't review their selection or change their mind
   before the run starts. Multi-minute pipelines kicked off by an accidental
   click is bad.
2. **Single-file constraint forces the user to pre-zip.** Real research
   workflows usually have N CSVs (one per run) sitting in a folder. Today
   the user must zip them up first. The ingest CLI already accepts
   `--files <one or many>` (verified at `src/fermdocs/cli.py:108`), so the
   bottleneck is the API layer, not the pipeline.

The fix has two halves: **(a)** an explicit Submit step (file picked ≠ run
started), **(b)** a tray UI that accumulates files one at a time before
Submit fires the API.

---

## User flow target

```
1. User lands on home page — sees: question textarea, file tray (empty),
   greyed-out Submit button.
2. User types question (optional).
3. User clicks "Add file" — OS picker, picks one CSV. Tray shows 1 row.
4. User clicks "Add file" again, picks another CSV. Tray shows 2 rows.
5. User accidentally added the wrong file — clicks ✕ on that row. Tray = 1.
6. User clicks Submit. Submit becomes "Uploading…" disabled.
7. Backend writes all files atomically, creates one Upload + one Run.
8. Frontend navigates to `/runs/<id>`. Pipeline starts.
9. (If N=1 .zip) treated exactly as today — no behavior change.
```

Four validation rules at submit time:

- Tray must be non-empty (Submit disabled when zero files).
- No duplicate filenames in the tray (frontend block + backend 400).
- If a `.zip` is in the tray, it must be the only file (mixing zips with raw
  data files has no coherent semantics: zips are pre-built bundles).
- All other files must be one of `.csv` / `.xlsx` / `.pdf`.

---

## Architecture in one diagram

```
               ┌─────────────────────────────────────┐
               │ FRONTEND apps/web/src/app/page.tsx          │
               │  state.files: File[]  (tray accumulator)    │
               │  + question textarea  + Submit button       │
               └──────────┬─────────────────────────────┘
                          │ POST /api/uploads (multipart, N files)
                          ▼
          ┌─────────────────────────────────────────┐
          │ apps/api/fermdocs_api/main.py                       │
          │  validates: non-empty, no dupes, zip-or-many        │
          │  atomic write: tmpdir → final dir on full success   │
          │  returns {upload_id, filenames: list[str], ...}     │
          └────────┬────────────────────────────────────┘
                   │ (frontend) POST /api/runs {upload_id, question}
                   ▼
          ┌─────────────────────────────────────────┐
          │ apps/api/fermdocs_api/runner_pipeline.py            │
          │  _prepare_bundle_dir branches on len(upload.paths): │
          │   1 file × .zip       → _unzip_bundle (today)        │
          │   1 file × raw        → _build_bundle_from_raw([p])  │
          │   N files × raw       → _build_bundle_from_raw(paths)│
          └─────────────────────────────────────────┘
                   │
                   ▼
          fermdocs ingest --files a.csv --files b.csv --files c.csv
          (already supports `multiple=True` per src/fermdocs/cli.py:108)
```

---

## Locked decisions (4)

From plan-eng-review on 2026-05-07:

### D1: Duplicate filenames → reject (A1=A)
If the user picks two files named `data.csv` from different folders, both
the frontend and the backend reject the upload. Auto-suffixing or per-file
UUID dirs are clever but surprising; explicit error is honest.

### D2: Atomic multi-file upload (A2=A)
`POST /api/uploads` writes all files into a `TemporaryDirectory()` first,
only moves them to `uploads/<upload_id>/` when every write succeeds. On
partial failure (disk, network), the tmpdir is cleaned up and no Upload
record exists. Today's single-file path also goes through this — the
logic is uniform regardless of N.

### D3: Submit-time UX honesty (A3=A)
No idempotency keys. Submit button disables on click + shows
"Uploading…". If the user navigates away mid-upload, the upload is
cancelled (browser default). If they come back and submit again, two
uploads happen — acceptable risk for v1, tracked as a TODO if it ever bites.

### D4: Big-bang rename (Q1=A)
`Upload.path: Path` → `Upload.paths: list[Path]` and
`Upload.filename: str` → `Upload.filenames: list[str]`. All callers fixed
in the same commit. No alias for back-compat — there are only ~3 call
sites and a back-compat shim invites drift. Single-file uploads pass
`paths=[one_path]`.

---

## Schema changes

### Modified: `apps/api/fermdocs_api/state.py:Upload`

```python
@dataclass
class Upload:
    upload_id: str
    filenames: list[str]              # was: filename: str
    paths: list[Path]                 # was: path: Path
    content_types: list[str]          # was: content_type: str
    size_bytes: int                   # sum of all file sizes
    created_at: datetime = ...
```

N=1 invariant: when the user uploads one file, all three lists have len 1.

### Modified: `RunStore.add_upload`

```python
def add_upload(
    self, *, files: list[tuple[str, str, bytes]]  # (filename, content_type, content)
) -> Upload:
    ...
```

Fails fast on duplicate filenames in the input list (raises
`ValueError`, surfaced as 400 by the route handler). Atomic write via
`tempfile.TemporaryDirectory()` then `shutil.move()` on success.

### Modified: `POST /api/uploads` route

```python
@app.post("/api/uploads")
async def upload(files: list[UploadFile] = File(...)) -> dict:
    if not files:
        raise HTTPException(400, "at least one file required")
    if len(files) > 1 and any(f.filename.endswith(".zip") for f in files):
        raise HTTPException(400, "zip uploads must be standalone")
    # ... read all into bytes, call store.add_upload(files=...), 400 on duplicate
    return {"upload_id": ..., "filenames": [...], "size_bytes": ...}
```

FastAPI's multi-file shape is `files: list[UploadFile] = File(...)` — the
frontend sends `multipart/form-data` with multiple `files=` parts.

---

## File-by-file plan

### Commit 1 — Backend dataclass + RunStore (~140 LOC)
- `apps/api/fermdocs_api/state.py`:
  - Rename `Upload.path` → `paths: list[Path]`, `Upload.filename` →
    `filenames: list[str]`, `Upload.content_type` →
    `content_types: list[str]`. `size_bytes` becomes the sum.
  - `RunStore.add_upload(files: list[tuple[str, str, bytes]])`. Atomic
    write via `tempfile.TemporaryDirectory()` then `shutil.move()` to the
    final `uploads/<upload_id>/` dir on full success.
  - Reject duplicate filenames with `ValueError("duplicate filename: ...")`.
  - Reject empty list with `ValueError("at least one file required")`.
- `tests/integration/test_upload_state.py` (~120 LOC, 6 tests): empty list
  raises, single file works, three csvs work, duplicate filenames raise,
  atomic rollback (mock `shutil.move` to fail mid-list, assert no
  partial upload visible), `size_bytes` sums correctly.

### Commit 2 — API endpoint + runner_pipeline (~180 LOC)
- `apps/api/fermdocs_api/main.py`:
  - `POST /api/uploads` accepts `list[UploadFile] = File(...)`.
  - Validates: non-empty (else 400), zip-only-when-alone (else 400),
    extensions in {csv, xlsx, pdf, zip} (else 400). Calls
    `store.add_upload`, catches `ValueError` → 400.
  - Response shape: `{upload_id, filenames: list[str], size_bytes,
    content_types: list[str]}`.
- `apps/api/fermdocs_api/runner_pipeline.py`:
  - `_prepare_bundle_dir` branches on `len(upload.paths)`:
    - 1 file, suffix=`.zip` → `_unzip_bundle(path)` (unchanged).
    - 1 file, suffix in raw → `_build_bundle_from_raw(paths=[p])`.
    - N files, all raw → `_build_bundle_from_raw(paths=[...])`.
  - `_build_bundle_from_raw(*, upload, store, run, paths: list[Path])`
    expands the `--files` arg as `--files p1 --files p2 ...`. Other
    paths (`work_root`, `dossier_path`) derived from `upload.paths[0]`
    parent (which is `uploads/<upload_id>/`). The classifier path is
    unchanged — it sees the post-characterize bundle.
- `tests/integration/test_multi_upload_api.py` (~140 LOC, 6 tests):
  empty list 400, single csv 200, three csvs 200, csv+zip 400, duplicate
  filename 400, response shape correct.
- `tests/integration/test_multi_upload_runner.py` (~80 LOC, 3 tests):
  N=1 csv routes through single-file path (regression — mock subprocess,
  assert one `--files` arg), N=1 zip routes through unzip (regression),
  N=3 csv passes three `--files` args to ingest.

### Commit 3 — Frontend tray + Submit gate (~280 LOC)
- `apps/web/src/lib/api.ts`:
  - `uploadFile(file: File)` → `uploadFiles(files: File[])`. Body uses
    `FormData.append("files", file)` per file. Response type updated to
    `{upload_id, filenames: string[], size_bytes, content_types: string[]}`.
- `apps/web/src/app/page.tsx`:
  - State: `files: File[]`, `question: string`, `error: string | null`,
    `submitting: boolean`.
  - Hidden file input + visible "Add file" button. `onChange` appends
    to `files` (does NOT replace). After append, validates: duplicate
    filenames (by `.name`), zip-mixed-with-others, unknown extension.
    On error: shows inline message, leaves tray unchanged.
  - Tray: each row shows filename + size + ✕ button. ✕ removes that
    file from the array.
  - Submit button: enabled when `files.length >= 1 && !submitting && no
    validation error`. On click: `await uploadFiles(files)` →
    `await createRun(upload_id, question.trim() || undefined)` →
    `router.push('/runs/' + run.run_id)`. On error: stops, shows error,
    leaves tray intact so user can retry.
  - Question textarea kept above the tray (current position).

  Frontend coverage is type-checked via `tsc --noEmit` and exercised
  manually — no Vitest/Playwright suite exists in the repo today, so
  formal frontend tests are out of scope. The backend tests cover the
  contract.

---

## Tests at every step

Mirror PR-A2's discipline:

- **State layer:** dataclass shape, atomic write, duplicate rejection,
  empty rejection, size sum.
- **API layer:** multipart with 0/1/N files, zip-mixing rejection,
  extension rejection, response shape.
- **Runner:** N=1 zip path unchanged (REGRESSION), N=1 raw path
  unchanged (REGRESSION), N=3 raw path adds three `--files` args.
- **Frontend:** tsc clean. Manual smoke: pick 1, pick 3, remove 1, pick
  zip-after-csv (rejected), pick csv-after-zip (rejected), submit empty
  (button disabled), submit with question → navigates to run page.

Target: ~15 backend tests, full suite finishing >1185 (today: 1171).

---

## What this PR explicitly does NOT do

- **Drag-and-drop.** Plain file input is fine; can add later if anyone
  asks.
- **Per-file upload progress.** Per-batch "Uploading…" is enough for v1.
  N is small (user picks files manually, not bulk-imports).
- **Resume interrupted uploads.** Files are tiny (CSVs); if upload
  fails, user retries.
- **Idempotency keys for double-submit.** UX-only block (Submit disabled
  + visible progress) handles the common case. Real idempotency is a
  TODO if it bites.
- **`.zip` mixed with raw files in one run.** Validated as 400 — zips
  are pre-built bundles and skip ingest entirely. Mixing them with raw
  files has no coherent meaning.
- **Backend retry on partial multi-file failure.** Atomic write means
  partial = nothing. User retries the whole batch.
- **Persistent upload tray across page reloads.** Refresh = empty tray.
  Acceptable for v1.

---

## Risks & mitigations

1. **Renaming `Upload.path` breaks tests we don't see.** Existing call
   sites: `runner_pipeline._prepare_bundle_dir` (`upload.path.suffix`),
   `runner_pipeline._build_bundle_from_raw` (`upload.path.parent`,
   `upload.path.resolve()`, `upload.filename`). Both rewritten in commit
   2. Mitigation: full test suite gate after commit 2; tsc on frontend.
2. **`fermdocs ingest --files` semantics with multiple files.**
   Verified the CLI accepts `multiple=True`. What ingest *does* with
   multiple files (concatenate? separate-runs?) is the ingest layer's
   problem, not ours. If ingest does the wrong thing on multi-file, that
   surfaces as a separate bug.
3. **Frontend: appending to file input doesn't preserve user's prior
   selection.** `<input type="file">` is single-shot; we don't bind its
   `value` to state. The tray IS the state of truth; the input is a
   one-shot picker each click. No issue.
4. **Zip-rejection error text shown twice.** Frontend pre-blocks; if
   somehow the request reaches the backend with the same violation,
   backend 400. Mitigation: frontend shows its own error and never
   sends. Backend 400 is defense-in-depth, not user-visible normally.
5. **Atomic upload tmpdir on a different filesystem from final dir.**
   `shutil.move()` falls back to copy-then-delete across filesystems
   automatically. Slower for large files but correct. Tracked.

---

## Cost summary

| Component | LOC |
|---|---|
| Upload + RunStore.add_upload | ~80 |
| API endpoint + validation | ~70 |
| runner_pipeline branching | ~110 |
| Frontend tray + Submit | ~200 |
| Backend tests | ~340 |
| **Total** | **~800 LOC, 3 commits, ~15 new tests** |

---

## Resume checklist post-compact

If we hit `/compact` mid-build:

1. `git branch --show-current` — confirm `frontend-redesign`.
2. `git log --oneline frontend-redesign ^hitl-followup` — see how many
   of the 3 commits done.
3. Read this file `plans/2026-05-07-multi-file-upload-and-submit.md`.
4. Check the most recent commit's diff to see what's mid-flight.
5. Continue from the next commit; full-suite + tsc gate between every
   commit.

Branch state at plan-writing time: `frontend-redesign` is at `a0ce0c3`
(same as `hitl-followup`). Zero commits yet.
