"""Stage 1 bundle infrastructure tests.

Covers:
  - meta.json round-trip + schema-version enforcement
  - atomic temp→final rename (half-written bundles invisible)
  - bundle id format + collision behavior
  - golden-schema major mismatch fails, minor mismatch warns
  - tier='A' default-backfill on Finding
  - audit-invariant CI guard exits 0 on clean tree
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from fermdocs.bundle import (
    BUNDLE_SCHEMA_VERSION,
    BundleNotReady,
    BundleReader,
    BundleSchemaMismatch,
    BundleWriter,
    GoldenSchemaMajorMismatch,
)
from fermdocs.bundle.meta import BundleMeta
from fermdocs.bundle.writer import make_bundle_id


# ---------------------------------------------------------------------------
# meta + bundle id
# ---------------------------------------------------------------------------


def test_bundle_id_single_run_has_run_id_and_uuid_suffix() -> None:
    bid = make_bundle_id(["RUN-0001"], created_at=datetime(2026, 5, 2, 14, 30, 12, tzinfo=timezone.utc))
    assert bid.startswith("bundle_RUN-0001_20260502T143012Z_")
    assert len(bid.rsplit("_", 1)[-1]) == 6


def test_bundle_id_multi_run_uses_multi_token() -> None:
    bid = make_bundle_id(["RUN-0001", "RUN-0002"], created_at=datetime(2026, 5, 2, 0, 0, 0, tzinfo=timezone.utc))
    assert bid.startswith("bundle_multi_20260502T000000Z_")


def test_bundle_meta_roundtrip(tmp_path: Path) -> None:
    meta = BundleMeta(
        bundle_schema_version=BUNDLE_SCHEMA_VERSION,
        golden_schema_version="2.0",
        pipeline_version="0.1.0",
        created_at=datetime(2026, 5, 2, tzinfo=timezone.utc),
        bundle_id="bundle_RUN-0001_20260502T000000Z_abcdef",
        run_ids=["RUN-0001"],
    )
    data = json.loads(meta.model_dump_json())
    assert data["bundle_schema_version"] == "1.0"
    assert data["run_ids"] == ["RUN-0001"]


# ---------------------------------------------------------------------------
# Writer / Reader happy path
# ---------------------------------------------------------------------------


def test_writer_observations_csv_columns(tmp_path: Path) -> None:
    writer = BundleWriter.create(
        tmp_path,
        run_ids=["RUN-A"],
        golden_schema_version="2.0",
        pipeline_version="0.1.0",
    )
    writer.write_dossier({"x": 1})
    writer.write_characterization("{}")
    writer.write_observations_csv(
        [
            {"run_id": "RUN-A", "variable": "biomass_g_l", "time_h": 0.0, "value": 1.5, "imputed": False, "unit": "g/L"},
            {"run_id": "RUN-A", "variable": "biomass_g_l", "time_h": 1.0, "value": None, "imputed": True, "unit": "g/L"},
        ]
    )
    bundle_path = writer.finalize()
    csv_path = bundle_path / "characterization" / "observations.csv"
    assert csv_path.exists()
    text = csv_path.read_text().splitlines()
    assert text[0] == "run_id,variable,time_h,value,imputed,unit"
    # Missing value renders as empty cell, imputed as 0/1
    assert text[1] == "RUN-A,biomass_g_l,0.0,1.5,0,g/L"
    assert text[2] == "RUN-A,biomass_g_l,1.0,,1,g/L"


def test_writer_finalize_creates_readable_bundle(tmp_path: Path) -> None:
    writer = BundleWriter.create(
        tmp_path,
        run_ids=["RUN-0001"],
        golden_schema_version="2.0",
        pipeline_version="0.1.0",
    )
    writer.write_dossier({"experiment": {"experiment_id": "EXP-1"}})
    writer.write_characterization('{"meta": {"schema_version": "1.0"}}')
    bundle_path = writer.finalize()

    assert bundle_path.exists()
    assert (bundle_path / "meta.json").exists()
    assert (bundle_path / "dossier.json").exists()
    assert (bundle_path / "characterization" / "characterization.json").exists()
    # subdirs created up front
    assert (bundle_path / "audit" / "python_calls").is_dir()

    reader = BundleReader(bundle_path)
    assert reader.meta.run_ids == ["RUN-0001"]
    assert reader.get_dossier()["experiment"]["experiment_id"] == "EXP-1"
    assert "schema_version" in reader.get_characterization_json()


def test_writer_temp_dir_invisible_until_finalize(tmp_path: Path) -> None:
    writer = BundleWriter.create(
        tmp_path,
        run_ids=["RUN-0001"],
        golden_schema_version="2.0",
        pipeline_version="0.1.0",
    )
    writer.write_dossier({"x": 1})
    # Before finalize, the temp dir exists but the final dir does not.
    assert writer.temp_dir.exists()
    assert not writer.final_dir.exists()
    # A hostile reader pointed at the temp dir refuses (no meta.json).
    with pytest.raises(BundleNotReady):
        BundleReader(writer.temp_dir)
    writer.finalize()
    assert writer.final_dir.exists()


def test_writer_abort_removes_temp(tmp_path: Path) -> None:
    writer = BundleWriter.create(
        tmp_path,
        run_ids=["RUN-0001"],
        golden_schema_version="2.0",
        pipeline_version="0.1.0",
    )
    temp = writer.temp_dir
    assert temp.exists()
    writer.abort()
    assert not temp.exists()


def test_writer_finalize_twice_raises(tmp_path: Path) -> None:
    writer = BundleWriter.create(
        tmp_path,
        run_ids=["RUN-0001"],
        golden_schema_version="2.0",
        pipeline_version="0.1.0",
    )
    writer.finalize()
    with pytest.raises(RuntimeError):
        writer.write_dossier({"x": 1})


# ---------------------------------------------------------------------------
# Reader version enforcement
# ---------------------------------------------------------------------------


def _build_bundle(tmp_path: Path, *, golden_version: str, run_ids: list[str] | None = None) -> Path:
    writer = BundleWriter.create(
        tmp_path,
        run_ids=run_ids or ["RUN-0001"],
        golden_schema_version=golden_version,
        pipeline_version="0.1.0",
    )
    writer.write_dossier({"experiment": {"experiment_id": "EXP-1"}})
    writer.write_characterization("{}")
    return writer.finalize()


def test_reader_rejects_bundle_schema_mismatch(tmp_path: Path) -> None:
    bundle_path = _build_bundle(tmp_path, golden_version="2.0")
    meta_path = bundle_path / "meta.json"
    data = json.loads(meta_path.read_text())
    data["bundle_schema_version"] = "9.9"
    # Write back, but pydantic frozen=True + Literal["1.0"] means we can't
    # round-trip via the model. Write raw JSON.
    meta_path.write_text(json.dumps(data, indent=2))

    with pytest.raises(BundleSchemaMismatch):
        BundleReader(bundle_path)


def test_reader_rejects_golden_major_mismatch(tmp_path: Path) -> None:
    bundle_path = _build_bundle(tmp_path, golden_version="1.5")
    with pytest.raises(GoldenSchemaMajorMismatch):
        BundleReader(bundle_path, current_golden_schema_version="2.0")


def test_reader_warns_on_golden_minor_mismatch(tmp_path: Path) -> None:
    bundle_path = _build_bundle(tmp_path, golden_version="2.1")
    with pytest.warns(UserWarning, match="minor mismatch"):
        BundleReader(bundle_path, current_golden_schema_version="2.0")


def test_reader_no_check_when_current_golden_unspecified(tmp_path: Path) -> None:
    bundle_path = _build_bundle(tmp_path, golden_version="9.9")
    # No current_golden_schema_version → no enforcement
    reader = BundleReader(bundle_path)
    assert reader.meta.golden_schema_version == "9.9"


def test_reader_rejects_missing_meta(tmp_path: Path) -> None:
    bundle_path = _build_bundle(tmp_path, golden_version="2.0")
    (bundle_path / "meta.json").unlink()
    with pytest.raises(BundleNotReady):
        BundleReader(bundle_path)


# ---------------------------------------------------------------------------
# Tier field
# ---------------------------------------------------------------------------


def test_finding_tier_defaults_to_a() -> None:
    from fermdocs_characterize.schema import (
        EvidenceStrength,
        ExtractedVia,
        Finding,
        FindingType,
        Severity,
        Tier,
    )

    f = Finding(
        finding_id="00000000-0000-0000-0000-000000000000:F-0001",
        type=FindingType.RANGE_VIOLATION,
        severity=Severity.MINOR,
        summary="x",
        confidence=0.9,
        extracted_via=ExtractedVia.DETERMINISTIC,
        evidence_strength=EvidenceStrength(n_observations=1, n_independent_runs=1),
        evidence_observation_ids=["O-1"],
    )
    assert f.tier == Tier.A
    assert f.model_dump()["tier"] == "A"


def test_finding_tier_b_explicit() -> None:
    from fermdocs_characterize.schema import (
        EvidenceStrength,
        ExtractedVia,
        Finding,
        FindingType,
        Severity,
        Tier,
    )

    f = Finding(
        finding_id="00000000-0000-0000-0000-000000000000:F-0001",
        type=FindingType.RANGE_VIOLATION,
        severity=Severity.MAJOR,
        tier=Tier.B,
        summary="x",
        confidence=0.9,
        extracted_via=ExtractedVia.STATISTICAL,
        evidence_strength=EvidenceStrength(n_observations=10, n_independent_runs=2),
        evidence_observation_ids=["O-1"],
    )
    assert f.tier == Tier.B


# ---------------------------------------------------------------------------
# CI guard smoke
# ---------------------------------------------------------------------------


def test_audit_invariant_guard_passes_on_clean_tree() -> None:
    repo = Path(__file__).resolve().parents[2]
    script = repo / "scripts" / "check_audit_invariant.py"
    assert script.exists(), "audit guard script missing"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"audit guard failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )


def test_audit_invariant_guard_flags_offending_module(tmp_path: Path) -> None:
    """Plant a deliberately bad file under a scanned root and confirm the guard catches it.

    We copy the guard script into a temp tree alongside a synthetic offender so
    we don't have to mutate the real repo.
    """
    repo = Path(__file__).resolve().parents[2]
    script_src = (repo / "scripts" / "check_audit_invariant.py").read_text()
    fake_root = tmp_path / "repo"
    (fake_root / "scripts").mkdir(parents=True)
    (fake_root / "src" / "fermdocs_diagnose").mkdir(parents=True)
    # Patch SCAN_ROOTS + REPO_ROOT to point at fake_root
    patched = script_src.replace(
        "REPO_ROOT = Path(__file__).resolve().parent.parent",
        f"REPO_ROOT = Path(r'{fake_root}')",
    )
    (fake_root / "scripts" / "check_audit_invariant.py").write_text(patched)
    (fake_root / "src" / "fermdocs_diagnose" / "bad.py").write_text(
        'data = open("out/bundle_x/audit/diagnosis_trace.json").read()\n'
    )
    result = subprocess.run(
        [sys.executable, str(fake_root / "scripts" / "check_audit_invariant.py")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "audit-invariant violations" in result.stdout
    assert "bad.py" in result.stdout
