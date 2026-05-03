"""Build a bundle dir from the existing flat IndPenSim artifacts in out/.

One-shot helper: takes out/dossier.json + out/characterization.json +
out/diagnosis.json and packages them into a proper bundle dir at
out/bundle_indpensim/ via BundleWriter so BundleReader can consume them.

This avoids re-running characterize+diagnose (and the API spend) just to
get the bundle structure for Stage 3 e2e tests.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fermdocs.bundle.writer import BundleWriter

OUT = Path("out")
TARGET_NAME = "bundle_indpensim"


def main() -> int:
    dossier_p = OUT / "dossier.json"
    char_p = OUT / "characterization.json"
    diag_p = OUT / "diagnosis.json"
    for p in (dossier_p, char_p, diag_p):
        if not p.exists():
            print(f"missing: {p}", file=sys.stderr)
            return 2

    dossier = json.loads(dossier_p.read_text())
    char_text = char_p.read_text()
    char = json.loads(char_text)
    diag_text = diag_p.read_text()

    run_ids = sorted({t.get("run_id") for t in char.get("trajectories", []) if t.get("run_id")})
    if not run_ids:
        # Fall back to dossier
        run_ids = list((dossier.get("experiment") or {}).get("runs") or ["RUN-0001"])
    print(f"run_ids={run_ids}")

    final_dir = OUT / TARGET_NAME
    if final_dir.exists():
        print(f"target already exists: {final_dir}; aborting (delete it first if you want to rebuild)")
        return 1

    # BundleWriter generates its own bundle_id; we'll rename after finalize so the
    # path is the predictable out/bundle_indpensim/.
    writer = BundleWriter.create(
        out_root=OUT,
        run_ids=run_ids,
        golden_schema_version="2.0",
        pipeline_version="0.1.0",
        model_labels={"characterization": "deterministic/v1"},
    )
    writer.write_dossier(dossier)
    writer.write_characterization(char_text)
    writer.write_diagnosis(diag_text)
    finalized = writer.finalize()

    # Rename to predictable name
    finalized.rename(final_dir)
    print(f"✓ bundle written: {final_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
