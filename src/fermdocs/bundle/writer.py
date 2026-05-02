"""Atomic bundle writer.

Writes to `out/.bundle_<...>.tmp/`, populates artifacts, then renames to the
final `out/bundle_<...>/`. `meta.json` is the LAST file written before rename
so its presence is the bundle's "ready" signal. A crash mid-write leaves the
temp dir; BundleReader will not see it.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fermdocs.bundle.meta import BUNDLE_SCHEMA_VERSION, BundleMeta


def _utc_iso_compact(dt: datetime) -> str:
    """20260502T143012Z."""
    return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def make_bundle_id(run_ids: list[str], created_at: datetime | None = None) -> str:
    """Build a collision-resistant bundle id.

    Format:
        bundle_<run_id>_<utc_iso_compact>_<short_uuid>      (single run)
        bundle_multi_<utc_iso_compact>_<short_uuid>          (multi-run)
    """
    ts = _utc_iso_compact(created_at or datetime.now(timezone.utc))
    short = uuid.uuid4().hex[:6]
    if len(run_ids) == 1:
        return f"bundle_{run_ids[0]}_{ts}_{short}"
    return f"bundle_multi_{ts}_{short}"


class BundleWriter:
    """Build a bundle directory atomically.

    Usage:
        writer = BundleWriter.create(out_root, run_ids=["RUN-0001"], ...)
        writer.write_dossier(dossier_dict)
        writer.write_characterization(char_output_json_str)
        bundle_path = writer.finalize()  # atomic temp -> final

    Once `finalize()` is called, the writer is closed. Subsequent writes raise.
    Crash before `finalize()` leaves `.bundle_*.tmp/` for manual cleanup.
    """

    def __init__(
        self,
        *,
        out_root: Path,
        meta: BundleMeta,
        temp_dir: Path,
        final_dir: Path,
    ) -> None:
        self._out_root = out_root
        self._meta = meta
        self._temp_dir = temp_dir
        self._final_dir = final_dir
        self._closed = False

    @classmethod
    def create(
        cls,
        out_root: str | Path,
        *,
        run_ids: list[str],
        golden_schema_version: str,
        pipeline_version: str,
        model_labels: dict[str, str] | None = None,
        flags: dict[str, bool] | None = None,
        created_at: datetime | None = None,
    ) -> BundleWriter:
        if not run_ids:
            raise ValueError("run_ids must be non-empty")
        out_root_p = Path(out_root)
        out_root_p.mkdir(parents=True, exist_ok=True)
        ts = created_at or datetime.now(timezone.utc)
        bundle_id = make_bundle_id(run_ids, created_at=ts)
        temp_dir = out_root_p / f".{bundle_id}.tmp"
        final_dir = out_root_p / bundle_id
        if final_dir.exists():
            raise FileExistsError(f"bundle already exists: {final_dir}")
        # exist_ok=False — collision means uuid clash, abort loudly
        temp_dir.mkdir(exist_ok=False)
        for sub in ("characterization", "diagnosis", "audit", "audit/python_calls"):
            (temp_dir / sub).mkdir(parents=True, exist_ok=True)
        meta = BundleMeta(
            bundle_schema_version=BUNDLE_SCHEMA_VERSION,
            golden_schema_version=golden_schema_version,
            pipeline_version=pipeline_version,
            created_at=ts,
            bundle_id=bundle_id,
            run_ids=run_ids,
            model_labels=model_labels or {},
            flags=flags or {},
        )
        return cls(out_root=out_root_p, meta=meta, temp_dir=temp_dir, final_dir=final_dir)

    @property
    def bundle_id(self) -> str:
        return self._meta.bundle_id

    @property
    def temp_dir(self) -> Path:
        return self._temp_dir

    @property
    def final_dir(self) -> Path:
        return self._final_dir

    @property
    def meta(self) -> BundleMeta:
        return self._meta

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("bundle writer is closed; finalize() was already called")

    def write_dossier(self, dossier: dict | str) -> Path:
        """Persist the ingestion dossier as `dossier.json` at the bundle root."""
        self._check_open()
        path = self._temp_dir / "dossier.json"
        if isinstance(dossier, str):
            path.write_text(dossier)
        else:
            path.write_text(json.dumps(dossier, indent=2, default=str))
        return path

    def write_characterization(self, characterization_json: str) -> Path:
        """Persist serialized CharacterizationOutput as
        `characterization/characterization.json`.

        Caller passes pre-serialized JSON (e.g. `output.model_dump_json(indent=2)`)
        so we don't need to import the characterization schema here.
        """
        self._check_open()
        path = self._temp_dir / "characterization" / "characterization.json"
        path.write_text(characterization_json)
        return path

    def write_narrative_observations(self, observations_json: str) -> Path:
        """Persist NarrativeObservation list at
        `characterization/narrative_observations.json`.

        Caller passes pre-serialized JSON. Optional file — bundles produced
        without prose extraction skip this entirely. BundleReader's
        get_narrative_observations_json() returns "[]" when missing.
        """
        self._check_open()
        path = self._temp_dir / "characterization" / "narrative_observations.json"
        path.write_text(observations_json)
        return path

    def write_observations_csv(self, rows: list[dict]) -> Path:
        """Persist a long-format observations CSV at
        `characterization/observations.csv`.

        Each row is one (run_id, variable, time_h, value, imputed, unit) point.
        This is what the diagnosis agent loads via pd.read_csv() inside
        execute_python — keeps the sandbox dependency-light (no parquet engine
        required) while giving the agent a pandas-shaped view of the data.

        Columns (in order): run_id, variable, time_h, value, imputed, unit.
        """
        import csv

        self._check_open()
        path = self._temp_dir / "characterization" / "observations.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["run_id", "variable", "time_h", "value", "imputed", "unit"],
            )
            writer.writeheader()
            for r in rows:
                writer.writerow(
                    {
                        "run_id": r.get("run_id", ""),
                        "variable": r.get("variable", ""),
                        "time_h": r.get("time_h", ""),
                        "value": "" if r.get("value") is None else r["value"],
                        "imputed": int(bool(r.get("imputed", False))),
                        "unit": r.get("unit", ""),
                    }
                )
        return path

    def write_diagnosis(self, diagnosis_json: str) -> Path:
        """Persist serialized DiagnosisOutput as `diagnosis/diagnosis.json`."""
        self._check_open()
        path = self._temp_dir / "diagnosis" / "diagnosis.json"
        path.write_text(diagnosis_json)
        return path

    def write_diagnosis_summary_md(self, markdown: str) -> Path:
        self._check_open()
        path = self._temp_dir / "diagnosis" / "summary.md"
        path.write_text(markdown)
        return path

    def update_flags(self, **flags: bool) -> None:
        """Adjust flags before finalize (e.g. budget_exhausted=True)."""
        self._check_open()
        merged = {**self._meta.flags, **flags}
        self._meta = self._meta.model_copy(update={"flags": merged})

    def finalize(self) -> Path:
        """Write meta.json (LAST) and atomically rename temp→final.

        Returns the final bundle path. Closes the writer.
        """
        self._check_open()
        meta_path = self._temp_dir / "meta.json"
        meta_path.write_text(self._meta.model_dump_json(indent=2))
        # POSIX rename within the same directory is atomic.
        os.rename(self._temp_dir, self._final_dir)
        self._closed = True
        return self._final_dir

    def abort(self) -> None:
        """Discard the temp directory. For tests / explicit cleanup."""
        if self._closed:
            return
        if self._temp_dir.exists():
            shutil.rmtree(self._temp_dir)
        self._closed = True
