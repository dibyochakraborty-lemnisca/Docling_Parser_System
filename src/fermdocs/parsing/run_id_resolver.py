"""Run-id resolution: a chain of strategies, not a single heuristic.

Plan ref: generalization Wave 1a.

Real-world fermentation files vary wildly in how they identify runs:

  - IndPenSim has Batch_ref (always 0, useless) AND Batch_ref.1 (real values)
  - Many small-lab CSVs have no run column at all — one file = one run
  - Some have string identifiers like "EXP_2024-03-15_RunB"
  - Multiple CSVs uploaded together, each is one run
  - Some have run_id buried in narrative metadata

A single column-name heuristic can't cover this. We resolve via a chain:

    1. ManifestStrategy   — operator-supplied run mapping wins
    2. ColumnStrategy     — find a column whose values vary meaningfully
    3. FilenameStrategy   — extract from filename (run_3.csv, batch_007.csv)
    4. SyntheticStrategy  — fallback: synthesize one run per file

Each strategy returns a RunIdResolution carrying the chosen value, the
strategy name, a confidence, and a rationale. Downstream agents can read
.strategy and decide whether to trust the value.

Adding a new format = adding a new Strategy class, not patching the
existing one. No silent failure modes: the chain always produces a
resolution, even if it's a synthetic fallback at confidence 0.3.

Generalization rule applied: this is the same pattern we should use for
time-column detection, phase detection, and other auto-detection
features. See memory: this is project-wide convention.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


# -----------------------------------------------------------------------------
# Result type
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class RunIdResolution:
    """One resolved run_id with provenance.

    Per-row resolution: when a Resolver returns a single resolution covering
    the whole table, every row gets the same run_id. When a Resolver returns
    a *vector* (one resolution per row, e.g. ColumnStrategy), each row gets
    its own.

    `column_idx` is set only by ColumnStrategy — it tells the pipeline which
    column to read per row. None means "use .value verbatim for every row".
    """

    value: str
    strategy: str
    confidence: float  # [0.0, 1.0]
    rationale: str
    column_idx: int | None = None


class RunIdStrategy(Protocol):
    """A strategy returns a RunIdResolution if it can, None otherwise."""

    name: str

    def resolve(
        self,
        *,
        headers: list[str],
        rows: list[list[Any]],
        filename: str | None,
        manifest_run_id: str | None,
    ) -> RunIdResolution | None: ...


# -----------------------------------------------------------------------------
# Strategies — each one is independent, ordered by precedence
# -----------------------------------------------------------------------------


class ManifestStrategy:
    """Operator-supplied run_id wins over any auto-detection.

    The dossier builder threads `manifest_run_id` through when the user
    invokes ingest with a manifest binding the file to a specific run.
    """

    name = "manifest"

    def resolve(
        self,
        *,
        headers: list[str],
        rows: list[list[Any]],
        filename: str | None,
        manifest_run_id: str | None,
    ) -> RunIdResolution | None:
        if not manifest_run_id:
            return None
        return RunIdResolution(
            value=manifest_run_id,
            strategy=self.name,
            confidence=1.0,
            rationale="operator-supplied run_id from manifest",
        )


# Headers we'll consider as "this might be a run/batch column".
_RUN_HEADER_TOKENS = (
    "batch_ref",
    "batch",
    "batch_id",
    "run",
    "run_id",
    "run_ref",
    "experiment_run",
    "trial",
)


class ColumnStrategy:
    """Pick a column whose values look like real run identifiers.

    Scoring per candidate column:

      - **Variability**: must have ≥2 distinct non-null values.
        A column that's always 0 (IndPenSim's Batch_ref) is worthless.
      - **Block constancy**: in a sorted-by-time table, run_id should
        be constant within consecutive row blocks (one run runs to
        completion before the next starts). Random-looking values
        with no block structure score lower.
      - **Header hint**: matching one of the run/batch header tokens
        boosts the score. We don't *require* the header match — if a
        column scores high on variability + constancy alone, it wins.

    Returns the highest-scoring candidate or None.
    """

    name = "column"

    def resolve(
        self,
        *,
        headers: list[str],
        rows: list[list[Any]],
        filename: str | None,
        manifest_run_id: str | None,
    ) -> RunIdResolution | None:
        if not rows or not headers:
            return None

        best: tuple[float, int, str] | None = None  # (score, col_idx, rationale)
        for col_idx, header in enumerate(headers):
            if col_idx >= len(rows[0]):
                continue
            values = [row[col_idx] for row in rows if col_idx < len(row)]
            score, rationale = self._score_column(header, values)
            if score <= 0:
                continue
            if best is None or score > best[0]:
                best = (score, col_idx, rationale)

        if best is None:
            return None

        score, col_idx, rationale = best
        # Confidence scales with score, capped at 0.95 (reserve 1.0 for manifest).
        confidence = min(0.95, 0.5 + score / 10.0)
        return RunIdResolution(
            value="",  # per-row; pipeline reads column_idx
            strategy=self.name,
            confidence=confidence,
            rationale=rationale,
            column_idx=col_idx,
        )

    # Below this row count we don't have enough data to trust a column-based
    # decision — fall through to filename / synthetic. Tables with 4+ rows
    # cover the realistic minimum (a 2-batch experiment has at least 2 rows
    # per batch, and we need to see a transition to call it a run column).
    MIN_ROWS_FOR_COLUMN_DECISION = 4

    def _score_column(
        self, header: str, values: list[Any]
    ) -> tuple[float, str]:
        normalized = [_coerce_run_id_value(v) for v in values]
        non_null = [v for v in normalized if v is not None]
        if len(non_null) < self.MIN_ROWS_FOR_COLUMN_DECISION:
            return 0.0, f"only {len(non_null)} non-null rows; need at least {self.MIN_ROWS_FOR_COLUMN_DECISION}"
        distinct = set(non_null)
        if len(distinct) < 2:
            # Single-value column (IndPenSim's Batch_ref=0) — useless.
            return 0.0, f"only one distinct value: {next(iter(distinct))!r}"

        # Header-hint check up front. Without it, we require either an
        # integer-looking shape or strong block structure to avoid mistaking
        # a measurement column (biomass, pH, PAA offline) for a run-id column.
        norm_header = (header or "").strip().lower()
        # Strip the pandas dedup suffix (.1, .2) so "Batch_ref.1" matches.
        norm_for_match = re.sub(r"\.\d+$", "", norm_header)
        header_hints = any(token in norm_for_match for token in _RUN_HEADER_TOKENS)

        # Shape-based filter: real run-ids are integers, short strings, or
        # consistent string identifiers. Continuous-looking floats (PAA
        # concentration, pH, biomass) are measurements, never run-ids.
        # We mark a column "id-shaped" if every non-null value is either:
        #   - the canonical RUN-NNNN form (numeric id we already normalized),
        #   - a short string (≤ 20 chars), AND
        #   - all same length (or close to it — id schemes are uniform)
        id_shaped = _looks_like_id_column(non_null)

        # Variability bonus, capped — a column with 1000 distinct values
        # is more likely a noisy measurement than a run id.
        if len(distinct) > 100:
            return 0.0, (
                f"column {header!r} has {len(distinct)} distinct values; too"
                f" many for a run-id, looks like a measurement"
            )

        score = 0.0
        rationale_bits: list[str] = []

        if len(distinct) <= 50:
            score += float(len(distinct))
            rationale_bits.append(f"{len(distinct)} distinct values")

        # Block-constancy bonus. Count transitions; fewer transitions per
        # distinct value = more block-like (run_id stays constant for many
        # consecutive rows).
        transitions = sum(
            1 for a, b in zip(non_null, non_null[1:]) if a != b
        )
        if transitions == 0:
            return 0.0, "no transitions"
        ratio = transitions / max(1, len(distinct))
        block_score = max(0.0, 5.0 - ratio)
        score += block_score
        rationale_bits.append(
            f"{transitions} transitions across {len(non_null)} rows"
        )

        # Gating: without a header hint, demand BOTH id-shape AND strong
        # block structure (ratio ≤ 1.5). This rejects measurement columns
        # whose values happen to repeat by coincidence.
        if not header_hints:
            if not id_shaped:
                return 0.0, (
                    f"column {header!r} has no run-id header hint and values"
                    f" don't look id-shaped (continuous floats / mixed lengths);"
                    f" likely a measurement"
                )
            if ratio > 1.5:
                return 0.0, (
                    f"column {header!r} has no run-id header hint and weak"
                    f" block structure (ratio {ratio:.2f}); likely a measurement"
                )

        if header_hints:
            score += 3.0
            rationale_bits.append(f"header {header!r} matches run-id pattern")

        return score, f"column {header!r}: " + ", ".join(rationale_bits)


# Patterns for filename-based run-id extraction. Order matters: the first
# match wins. We accept word-prefix tokens ("run", "batch", "exp", "trial")
# followed by an optional separator ("-" / "_" / nothing) and digits.
# `\b` boundaries are insufficient because `_` is a word character.
_FILENAME_RUN_PATTERNS = (
    re.compile(
        r"(?:^|[^A-Za-z0-9])"
        r"(?:run|batch|exp|experiment|trial)"
        r"[-_]?"
        r"(\d+)"
        r"(?:$|[^A-Za-z0-9])",
        re.IGNORECASE,
    ),
    re.compile(r"(?:^|_)R(\d+)(?:$|_|\.)"),
    re.compile(r"(?:^|_)B(\d+)(?:$|_|\.)"),
)


class FilenameStrategy:
    """Extract a numeric run/batch id from the filename.

    Examples that match:
      run_3.csv          -> RUN-0003
      batch007.csv       -> RUN-0007
      EXP_2024_run4.csv  -> RUN-0004 (last match wins)
      data_R5.csv        -> RUN-0005
    """

    name = "filename"

    def resolve(
        self,
        *,
        headers: list[str],
        rows: list[list[Any]],
        filename: str | None,
        manifest_run_id: str | None,
    ) -> RunIdResolution | None:
        if not filename:
            return None
        stem = Path(filename).stem
        for pattern in _FILENAME_RUN_PATTERNS:
            match = pattern.search(stem)
            if match:
                run_num = int(match.group(1))
                return RunIdResolution(
                    value=f"RUN-{run_num:04d}",
                    strategy=self.name,
                    confidence=0.7,
                    rationale=(
                        f"filename {filename!r} matched pattern {pattern.pattern!r}"
                        f" -> RUN-{run_num:04d}"
                    ),
                )
        return None


class SyntheticStrategy:
    """Last-resort fallback: synthesize one run_id per file.

    Always succeeds. Confidence is low so downstream consumers know the
    run_id is a placeholder, not a real identifier from the data. The
    diagnosis agent can still reason about within-file dynamics; cohort
    comparison just isn't meaningful at confidence 0.3.
    """

    name = "synthetic"

    def resolve(
        self,
        *,
        headers: list[str],
        rows: list[list[Any]],
        filename: str | None,
        manifest_run_id: str | None,
    ) -> RunIdResolution | None:
        if filename:
            stem = Path(filename).stem
            # Make a stable, readable identifier from the filename.
            safe = re.sub(r"[^A-Za-z0-9_-]", "_", stem)[:40] or "FILE"
            return RunIdResolution(
                value=f"RUN-FROM-{safe}",
                strategy=self.name,
                confidence=0.3,
                rationale=(
                    "no run_id column or recognizable filename pattern;"
                    " synthesized one run_id from filename"
                ),
            )
        return RunIdResolution(
            value="RUN-UNKNOWN",
            strategy=self.name,
            confidence=0.2,
            rationale="no filename, no run_id column; using RUN-UNKNOWN",
        )


# -----------------------------------------------------------------------------
# Resolver chain
# -----------------------------------------------------------------------------


class RunIdResolver:
    """Chain of strategies, first hit wins.

    Default strategy order (precedence):
      1. ManifestStrategy
      2. ColumnStrategy
      3. FilenameStrategy
      4. SyntheticStrategy (always succeeds)

    Tests / production can swap the chain by passing a custom list.
    """

    def __init__(self, strategies: list[RunIdStrategy] | None = None) -> None:
        self._strategies: list[RunIdStrategy] = strategies or [
            ManifestStrategy(),
            ColumnStrategy(),
            FilenameStrategy(),
            SyntheticStrategy(),
        ]

    def resolve(
        self,
        *,
        headers: list[str],
        rows: list[list[Any]],
        filename: str | None = None,
        manifest_run_id: str | None = None,
    ) -> RunIdResolution:
        for strategy in self._strategies:
            result = strategy.resolve(
                headers=headers,
                rows=rows,
                filename=filename,
                manifest_run_id=manifest_run_id,
            )
            if result is not None:
                return result
        # SyntheticStrategy never returns None, so we never reach here, but
        # keep this defensive return for type-checker satisfaction.
        return RunIdResolution(
            value="RUN-UNKNOWN",
            strategy="fallback",
            confidence=0.0,
            rationale="all strategies returned None (should not happen)",
        )


# -----------------------------------------------------------------------------
# Value coercion (kept module-private; pipeline.py has its own copy too)
# -----------------------------------------------------------------------------


def _looks_like_id_column(non_null_values: list[str]) -> bool:
    """Heuristic: do these values look like real run identifiers?

    A real run-id column has values that are:
      - All RUN-NNNN form (already normalized from int input), OR
      - All short, non-decimal alphanumeric strings of similar length
        (e.g. "expA", "B7", "trial-1", "RUN-0042")

    Continuous-looking decimal floats are measurements, never run-ids.
    PAA_offline (1019.34, 5202.81, 634.47) and X_offline (24.41, 24.72)
    must be rejected here. The filter is intentionally strict — we'd
    rather miss an unconventional run-id and fall through to filename /
    synthetic than mistake a measurement for a run-id.
    """
    if not non_null_values:
        return False
    distinct = list(set(non_null_values))

    # Reject anything that smells like a continuous float. If most values
    # contain decimal points AND parse as non-integer floats, it's a
    # measurement.
    decimal_count = 0
    for v in distinct:
        if "." not in v:
            continue
        try:
            f = float(v)
            # _coerce_run_id_value() already normalized integer-valued floats
            # to RUN-NNNN form, so anything still containing "." here that
            # parses as a float is a real decimal value.
            if not f.is_integer():
                decimal_count += 1
        except (TypeError, ValueError):
            pass
    if decimal_count >= max(1, len(distinct) // 2):
        return False

    # Pattern 1: every distinct value is RUN-NNNN form (from numeric input).
    if all(v.startswith("RUN-") for v in distinct):
        return True

    # Pattern 2: short alphanumeric strings of similar length.
    if all(len(v) <= 20 for v in distinct):
        lengths = [len(v) for v in distinct]
        if max(lengths) - min(lengths) <= 4:
            return True

    return False


def _coerce_run_id_value(value: Any) -> str | None:
    """Normalize a raw column cell into a run_id string, or None.

    Handles: numeric ids, string ids, NaN sentinels, blanks. Mirrors the
    coercion the pipeline already does at observation-write time.
    """
    if value is None or value == "":
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        f = float(s)
        if math.isnan(f):
            return None
        if f.is_integer():
            return f"RUN-{int(f):04d}"
    except (TypeError, ValueError):
        pass
    return s
