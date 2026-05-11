"""Tests for the RunIdResolver strategy chain.

Each strategy is tested in isolation, then together as a chain. Critical
generalization cases:

  - IndPenSim regression: Batch_ref column is always 0, Batch_ref.1 has
    real values. The old header-match heuristic picked the dead column.
    The new ColumnStrategy must pick the live one.
  - Yeast-batch case: a CSV with no run_id column at all. FilenameStrategy
    extracts from "yeast_run_3.csv". If the filename is opaque,
    SyntheticStrategy synthesizes one run_id per file.
  - String run ids: "EXP_2024-03-15_RunB" stays verbatim, no normalization.
"""

from __future__ import annotations

import pytest

from fermdocs.parsing.run_id_resolver import (
    ColumnStrategy,
    FilenameStrategy,
    ManifestStrategy,
    RunIdResolver,
    RunIdResolution,
    SyntheticStrategy,
)


# ---------- ManifestStrategy ----------


def test_manifest_strategy_takes_value_verbatim():
    s = ManifestStrategy()
    r = s.resolve(
        headers=["x"],
        rows=[["1"]],
        filename="ignored.csv",
        manifest_run_id="EXP-OPERATOR-1",
    )
    assert r is not None
    assert r.value == "EXP-OPERATOR-1"
    assert r.strategy == "manifest"
    assert r.confidence == 1.0


def test_manifest_strategy_returns_none_when_no_manifest():
    s = ManifestStrategy()
    assert s.resolve(
        headers=["x"], rows=[["1"]], filename=None, manifest_run_id=None
    ) is None


# ---------- ColumnStrategy: the IndPenSim regression ----------


def test_column_strategy_picks_varying_over_constant_column():
    """The IndPenSim regression: Batch_ref is always 0 (useless),
    Batch_ref.1 has values [1, 1, 2, 2] (real run-id signal). The
    old code picked the first match by name; the new strategy must
    pick the column with actual variability.
    """
    headers = ["Time (h)", "Batch_ref", "Batch_ref.1", "X"]
    rows = [
        [0.0, 0, 1, 0.5],
        [10.0, 0, 1, 5.0],
        [0.0, 0, 2, 0.5],
        [10.0, 0, 2, 4.8],
    ]
    s = ColumnStrategy()
    r = s.resolve(headers=headers, rows=rows, filename=None, manifest_run_id=None)
    assert r is not None
    assert r.column_idx == 2  # Batch_ref.1
    assert r.strategy == "column"
    assert "Batch_ref.1" in r.rationale


def test_column_strategy_picks_non_run_named_column_when_only_signal_there():
    """Header-hint is additive, not gating. A column with strong block
    structure should win even with a non-canonical header.
    """
    headers = ["Time (h)", "Trial_Number"]
    rows = [
        [0.0, "A"],
        [10.0, "A"],
        [0.0, "B"],
        [10.0, "B"],
    ]
    s = ColumnStrategy()
    r = s.resolve(headers=headers, rows=rows, filename=None, manifest_run_id=None)
    assert r is not None
    assert r.column_idx == 1


def test_column_strategy_rejects_single_value_column():
    """A column where every row has the same value carries no run-id
    information.
    """
    headers = ["Batch_ref"]
    rows = [[0], [0], [0], [0]]
    s = ColumnStrategy()
    assert s.resolve(
        headers=headers, rows=rows, filename=None, manifest_run_id=None
    ) is None


def test_column_strategy_rejects_too_few_rows():
    s = ColumnStrategy()
    assert s.resolve(
        headers=["Batch_ref"], rows=[["1"]], filename=None, manifest_run_id=None
    ) is None


def test_column_strategy_handles_string_ids():
    headers = ["run_label"]
    rows = [["expA"], ["expA"], ["expB"], ["expB"]]
    s = ColumnStrategy()
    r = s.resolve(headers=headers, rows=rows, filename=None, manifest_run_id=None)
    assert r is not None
    assert r.column_idx == 0


def test_column_strategy_returns_none_when_no_columns():
    s = ColumnStrategy()
    assert s.resolve(headers=[], rows=[], filename=None, manifest_run_id=None) is None


# ---------- ColumnStrategy: per-row-identifier guard (carotenoid regression) ----------


def test_column_strategy_rejects_time_column_as_run_id():
    """Carotenoid PDF regression: a single-fermentation timecourse table
    with a TIME column whose values [0, 6, 12, 18, 24, ...] are integer-shaped
    and all distinct. The old scorer accepted it (ratio = 1.0 ≤ 1.5),
    producing 21 fake one-point "runs" instead of one trajectory.

    The per-row-identifier guard must reject it. Without a header hint and
    with N distinct values across N rows, this is not a run-id.
    """
    headers = ["TIME", "OD", "WCW", "DO"]
    rows = [
        [0, 0.8, 3.2, 100.0],
        [6, 1.5, 6.4, 95.0],
        [12, 4.2, 18.0, 80.0],
        [18, 12.0, 51.0, 60.0],
        [24, 28.0, 119.0, 45.0],
        [48, 80.0, 340.0, 35.0],
        [72, 180.0, 765.0, 30.0],
        [96, 250.0, 1062.0, 26.0],
        [108, 254.0, 1082.0, 25.5],
        [120, 296.0, 1257.0, 95.1],
    ]
    s = ColumnStrategy()
    r = s.resolve(headers=headers, rows=rows, filename=None, manifest_run_id=None)
    assert r is None, (
        "TIME column was picked as run-id; per-row-identifier guard failed."
        " Each row would become its own 'run', destroying trajectory structure."
    )


def test_column_strategy_rejects_monotonic_distinct_column():
    """A row-index / sequence column (1, 2, 3, 4, ... — strictly increasing,
    all distinct) is never a run-id. The monotonic-and-all-distinct guard
    catches this even when the integer-shape check would pass.
    """
    headers = ["row_num", "biomass"]
    rows = [[1, 0.5], [2, 1.2], [3, 4.5], [4, 12.0], [5, 28.0], [6, 60.0]]
    s = ColumnStrategy()
    r = s.resolve(headers=headers, rows=rows, filename=None, manifest_run_id=None)
    assert r is None


def test_column_strategy_keeps_summary_table_when_header_hints():
    """Case 3 preservation: a summary table with one row per run, where
    the run-id column has a header hint. Even though every value is
    distinct (5 runs, 5 rows → ratio 1.0), the header_hint vouches for
    the column and the per-row-identifier guard does NOT fire.
    """
    headers = ["Batch_ref", "FinalYield", "FinalTime"]
    rows = [
        ["B1", 12.5, 96.0],
        ["B2", 14.2, 108.0],
        ["B3", 11.8, 96.0],
        ["B4", 13.9, 102.0],
    ]
    s = ColumnStrategy()
    r = s.resolve(headers=headers, rows=rows, filename=None, manifest_run_id=None)
    assert r is not None
    assert r.column_idx == 0


def test_column_strategy_rejects_garbled_string_id_column():
    """Carotenoid feed-rate table regression: Docling extracted time-range
    header strings (e.g. '103→ 120', '14 →20') into the column data, and
    the old scorer happily picked that as the run-id (46 fake runs of
    garbled strings). With most rows distinct and no header hint, the
    per-row-identifier guard must reject.
    """
    headers = ["interval", "feed_rate"]
    rows = [
        ["14 →20", 1.2],
        ["20 →26", 1.5],
        ["26 →30", 2.0],
        ["30 →75", 3.0],
        ["75 →91", 5.0],
        ["91 →103", 8.0],
        ["103→ 120", 15.0],
    ]
    s = ColumnStrategy()
    r = s.resolve(headers=headers, rows=rows, filename=None, manifest_run_id=None)
    assert r is None


def test_resolver_chain_falls_through_to_synthetic_on_carotenoid_shape():
    """End-to-end chain check: a wide single-fermentation table with no
    explicit run-id column should resolve to a single synthetic run-id
    (one fermentation per file), not 21 one-point runs.
    """
    resolver = RunIdResolver()
    headers = ["TIME", "OD", "WCW", "DO"]
    # Floats with decimals so _looks_like_id_column rejects them as
    # measurements (matches what a real PDF table looks like).
    rows = [
        [t, round(t * 2.51 + 0.3, 2), round(t * 9.87 + 0.5, 2), round(99.5 - t * 0.55, 2)]
        for t in range(0, 121, 6)
    ]
    r = resolver.resolve(
        headers=headers,
        rows=rows,
        filename="Carotenoid_batch_report.pdf",
        manifest_run_id=None,
    )
    # FilenameStrategy matches "batch" prefix → RUN-NNNN; if filename had
    # no token, it'd fall through to synthetic. Either way: single run-id.
    assert r.strategy in ("filename", "synthetic"), (
        f"Expected fall-through to filename/synthetic, got {r.strategy} "
        f"(column_idx={r.column_idx}, rationale={r.rationale!r})"
    )
    assert r.value.startswith("RUN-")


# ---------- FilenameStrategy ----------


@pytest.mark.parametrize(
    "filename,expected_run",
    [
        ("run_3.csv", "RUN-0003"),
        ("batch007.csv", "RUN-0007"),
        ("yeast_run_12.csv", "RUN-0012"),
        ("data_R5.csv", "RUN-0005"),
        ("experiment_42.csv", "RUN-0042"),
        ("trial-9.tsv", "RUN-0009"),
    ],
)
def test_filename_strategy_extracts_numeric_id(filename, expected_run):
    s = FilenameStrategy()
    r = s.resolve(headers=[], rows=[], filename=filename, manifest_run_id=None)
    assert r is not None
    assert r.value == expected_run
    assert r.strategy == "filename"


@pytest.mark.parametrize("filename", ["data.csv", "fermentation.tsv", None])
def test_filename_strategy_returns_none_for_opaque_filenames(filename):
    s = FilenameStrategy()
    assert s.resolve(
        headers=[], rows=[], filename=filename, manifest_run_id=None
    ) is None


# ---------- SyntheticStrategy ----------


def test_synthetic_strategy_uses_filename_stem():
    s = SyntheticStrategy()
    r = s.resolve(
        headers=[],
        rows=[],
        filename="my_yeast_data.csv",
        manifest_run_id=None,
    )
    assert r is not None
    assert r.value == "RUN-FROM-my_yeast_data"
    assert r.strategy == "synthetic"
    assert r.confidence < 0.5  # placeholder, not real signal


def test_synthetic_strategy_falls_back_when_no_filename():
    s = SyntheticStrategy()
    r = s.resolve(headers=[], rows=[], filename=None, manifest_run_id=None)
    assert r is not None
    assert r.value == "RUN-UNKNOWN"


def test_synthetic_strategy_strips_unsafe_filename_chars():
    s = SyntheticStrategy()
    r = s.resolve(
        headers=[], rows=[], filename="weird @#$ name.csv", manifest_run_id=None
    )
    assert r is not None
    # Spaces and special chars become underscores; alnum/underscore preserved.
    assert r.value.startswith("RUN-FROM-")
    assert "@" not in r.value


# ---------- Resolver chain ----------


def test_resolver_chain_manifest_wins_over_column():
    """When the operator supplies a manifest run_id, even a perfectly
    good run_id column should be ignored.
    """
    resolver = RunIdResolver()
    headers = ["Batch_ref"]
    rows = [["1"], ["2"], ["3"]]
    r = resolver.resolve(
        headers=headers,
        rows=rows,
        filename="data.csv",
        manifest_run_id="OPERATOR-RUN-42",
    )
    assert r.strategy == "manifest"
    assert r.value == "OPERATOR-RUN-42"


def test_resolver_chain_column_wins_when_no_manifest():
    resolver = RunIdResolver()
    headers = ["Batch_ref"]
    rows = [["1"], ["1"], ["2"], ["2"]]
    r = resolver.resolve(
        headers=headers, rows=rows, filename="run_99.csv", manifest_run_id=None
    )
    # Column has clear signal, so it beats filename.
    assert r.strategy == "column"
    assert r.column_idx == 0


def test_resolver_chain_falls_through_to_filename():
    """No manifest, no run_id column with signal → filename strategy fires."""
    resolver = RunIdResolver()
    headers = ["Time (h)", "X"]
    rows = [[0.0, 0.5], [10.0, 5.0]]  # too few + no run column
    r = resolver.resolve(
        headers=headers, rows=rows, filename="batch_007.csv", manifest_run_id=None
    )
    assert r.strategy == "filename"
    assert r.value == "RUN-0007"


def test_resolver_chain_falls_through_to_synthetic():
    """No manifest, no signal column, no recognizable filename → synthetic."""
    resolver = RunIdResolver()
    r = resolver.resolve(
        headers=["x"],
        rows=[["a"]],
        filename="opaque_data.csv",
        manifest_run_id=None,
    )
    assert r.strategy == "synthetic"
    assert r.confidence < 0.5
    assert r.value == "RUN-FROM-opaque_data"


def test_resolver_chain_always_returns_a_resolution():
    """Synthetic is the catch-all; the chain must never return None even
    on completely empty input.
    """
    resolver = RunIdResolver()
    r = resolver.resolve(
        headers=[], rows=[], filename=None, manifest_run_id=None
    )
    assert isinstance(r, RunIdResolution)
    # Confidence is low but the chain produced something.
    assert r.confidence > 0


def test_resolver_chain_custom_strategies():
    """Production swaps can pass a custom chain — useful for tests that
    want to assert one strategy in isolation.
    """
    resolver = RunIdResolver(strategies=[FilenameStrategy(), SyntheticStrategy()])
    r = resolver.resolve(
        headers=["Batch_ref"],
        rows=[["1"], ["2"], ["3"]],  # would normally win via ColumnStrategy
        filename="run_5.csv",
        manifest_run_id=None,
    )
    assert r.strategy == "filename"
    assert r.value == "RUN-0005"


# ---------- Generalization regressions ----------


def test_continuous_float_column_rejected_as_runid():
    """REGRESSION: an offline-measurement column like PAA_offline has 40
    unique floats spread across 1130 rows. Without shape filtering the
    resolver would pick it as a run-id column. This test guards that
    realistic measurement values are not mistaken for ids.
    """
    headers = ["Time (h)", "PAA_offline"]
    # 40 distinct float values, each unique, spread irregularly
    paa_values = [1019.34, 5202.81, 634.47, 717.62, 736.54, 748.25, 755.50, 764.86,
                  787.97, 798.56, 856.46, 877.86, 891.01, 952.34, 967.56, 996.03,
                  998.81, 1029.20, 1047.03, 1095.10, 1166.08, 1206.76, 1305.97,
                  1380.54, 1397.16, 1417.06, 1454.51, 1506.07, 1515.20, 1553.92,
                  1678.73, 1845.77, 2023.61, 2282.01, 2764.67, 3510.27, 4225.43,
                  4831.17, 5026.63, 1019.34]
    rows = [[t * 0.2, str(paa_values[i % len(paa_values)])] for i, t in enumerate(range(1130))]
    s = ColumnStrategy()
    r = s.resolve(headers=headers, rows=rows, filename=None, manifest_run_id=None)
    assert r is None, (
        f"PAA_offline must not be picked as run-id column; got {r}"
    )


def test_indpensim_batch_ref_dot_one_picked_over_constant_batch_ref():
    """REGRESSION: IndPenSim has TWO Batch_ref columns. Batch_ref=0
    everywhere (useless), Batch_ref.1 = [1, 2] (real run id). Resolver
    must skip the constant column and pick the varying one.
    """
    headers = ["Time (h)", "Batch_ref", "Batch_ref.1", "X"]
    # 1130 rows of batch 1, then 1130 rows of batch 2 — like IndPenSim
    rows = []
    for i in range(1130):
        rows.append([i * 0.2, 0, 1, 0.5 + i * 0.001])
    for i in range(1130):
        rows.append([i * 0.2, 0, 2, 0.5 + i * 0.001])

    resolver = RunIdResolver()
    r = resolver.resolve(headers=headers, rows=rows, filename="IndPenSim.csv", manifest_run_id=None)
    assert r.strategy == "column"
    assert r.column_idx == 2  # Batch_ref.1, not Batch_ref


def test_yeast_batch_no_run_column_resolves_via_filename():
    """A yeast-batch fixture: single batch, no run_id column, filename
    encodes the run number. This is the generalization case Wave 1c
    targets.
    """
    resolver = RunIdResolver()
    headers = ["Time (h)", "biomass_g_l", "pH", "DO2"]
    rows = [[0.0, 0.5, 5.5, 100.0], [2.0, 0.8, 5.4, 95.0]]
    r = resolver.resolve(
        headers=headers,
        rows=rows,
        filename="yeast_batch_run_3.csv",
        manifest_run_id=None,
    )
    assert r.strategy == "filename"
    assert r.value == "RUN-0003"
