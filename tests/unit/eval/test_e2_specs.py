from __future__ import annotations

from pathlib import Path

import pytest

from fermdocs_eval.fixture_builder import build_fixture
from fermdocs_eval.fixtures.e2_specs import SPECS, fixtures_by_axis, fixtures_by_difficulty

REPO_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = REPO_ROOT / "out" / "bundle_indpensim"


def test_total_count_is_40() -> None:
    assert len(SPECS) == 40, f"expected 40 fixtures, got {len(SPECS)}"


def test_ids_are_unique() -> None:
    ids = [s.fixture_id for s in SPECS]
    assert len(ids) == len(set(ids)), "duplicate fixture_ids"


def test_distribution_is_5_per_axis_plus_5_clean() -> None:
    counts = fixtures_by_axis()
    assert counts["clean"] == 5
    for axis in (
        "trajectory-axis",
        "robustness-axis",
        "tool-gap-axis",
        "memory-axis",
        "metadata-axis",
        "actionability-axis",
        "question-axis",
    ):
        assert counts.get(axis) == 5, f"expected 5 fixtures on {axis}, got {counts.get(axis, 0)}"


def test_difficulty_split() -> None:
    counts = fixtures_by_difficulty()
    # 5 clean + (3 clear + 2 borderline) * 7 axes = 5 + 21 + 14
    assert counts["clean"] == 5
    assert counts["clear"] == 21
    assert counts["borderline"] == 14


def test_memory_axis_specs_have_memory_seed() -> None:
    for spec in SPECS:
        if spec.labeled_axis == "memory-axis":
            assert spec.memory_seed, f"{spec.fixture_id} on memory-axis missing memory_seed"


def test_all_specs_have_leading_question() -> None:
    for s in SPECS:
        assert s.leading_question.strip(), f"{s.fixture_id} has empty leading_question"


@pytest.mark.skipif(not TEMPLATE.exists(), reason="indpensim template not present")
def test_all_fixtures_build_successfully(tmp_path: Path) -> None:
    """Round-trip every spec through build_fixture. Schema-validates each.

    This is the load-bearing test — if any fixture fails to build, the batch
    will fail at runtime. Better to catch here.
    """
    failures: list[tuple[str, str]] = []
    for spec in SPECS:
        try:
            build_fixture(spec, template_dir=TEMPLATE, out_root=tmp_path)
        except Exception as exc:  # noqa: BLE001
            failures.append((spec.fixture_id, f"{type(exc).__name__}: {str(exc)[:200]}"))
    assert not failures, "fixtures failed to build:\n" + "\n".join(f"  {fid}: {err}" for fid, err in failures)
