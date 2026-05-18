from __future__ import annotations

from pathlib import Path

import pytest

from fermdocs_eval.fixture_builder import DefectSpec, build_fixture, list_mutators

TEMPLATE = Path("out/bundle_indpensim")
REPO_ROOT = Path(__file__).resolve().parents[3]


def _template_path() -> Path:
    p = REPO_ROOT / TEMPLATE
    return p


@pytest.fixture
def template_dir() -> Path:
    p = _template_path()
    if not p.exists():
        pytest.skip(f"template bundle not present at {p}")
    return p


def test_mutators_registered() -> None:
    names = list_mutators()
    # Sanity: each axis-relevant mutator is wired.
    for required in ("noop", "strip_findings", "strip_trajectories", "drop_narratives"):
        assert required in names


def test_clean_fixture_roundtrips(template_dir: Path, tmp_path: Path) -> None:
    spec = DefectSpec(
        fixture_id="e2-clean-smoke",
        labeled_axis="clean",
        difficulty="clean",
        leading_question="What happened in this run? Summarize key findings.",
        mutation_kind="noop",
        notes="smoke test",
    )
    out = build_fixture(spec, template_dir=template_dir, out_root=tmp_path)
    assert (out / "diagnosis" / "diagnosis.json").exists()
    assert (out / "characterization" / "characterization.json").exists()
    assert (out / "dossier.json").exists()
    assert (out / "defect_spec.json").exists()


def test_strip_findings_keeps_only_first_n(template_dir: Path, tmp_path: Path) -> None:
    spec = DefectSpec(
        fixture_id="e2-robustness-smoke",
        labeled_axis="robustness-axis",
        difficulty="clear",
        leading_question="What single anomaly best explains the failure?",
        mutation_kind="strip_findings",
        mutation_params={"keep": 3},
        notes="smoke test: keep 3 findings, drop rest",
    )
    out = build_fixture(spec, template_dir=template_dir, out_root=tmp_path)

    # Re-read the written characterization and confirm the trim took.
    import json
    with (out / "characterization" / "characterization.json").open() as fh:
        char = json.load(fh)
    if "findings" in char:
        assert len(char["findings"]) <= 3
