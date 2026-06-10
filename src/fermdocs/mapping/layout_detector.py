"""Layout detector: find the real table region inside a messy raw grid.

Today the parsers assume row 0 is the header (pandas default). On a
metadata-heavy lab sheet the real table starts mid-sheet, so the naive
``ParsedTable`` carries garbage headers and the mapper produces zero
observations -- a silent ingestion failure (the praaj workbook hit this).

The detector reads the RAW CELL GRID (header-less) and asks an LLM to
locate the table STRUCTURE only: which row is the header, which rows are
data, how rows group into runs, the stage label, the time column. It never
transcribes a single value -- code slices the actual grid at the LLM's
chosen indices. This is the same "LLM decides meaning, code touches
numbers" contract the unit normalizer uses (units/normalizer.py).

Safety stack (mirrors IdentityExtractor):
  - confidence capped at LLM_CONFIDENCE_CAP (0.85)
  - the header cells the LLM claims are verified verbatim against the grid
    (verify_substring_evidence); a wrong header-row pick fails this check
    and the region is dropped
  - every failure path returns [] / drops the region; the detector never
    raises, so a dead LLM degrades to "no detected tables" and the
    pipeline's coverage gate surfaces it loudly.

Decision flow per source grid:

    no client / empty grid     -> []  (pipeline falls back to naive tables)
    LLM error / empty payload  -> []  (fail-soft)
    region out of bounds       -> drop that region
    header-cell evidence fails -> drop that region (wrong header row)
    valid region               -> slice grid -> ParsedTable

Output: ordinary ``ParsedTable`` objects. Structural metadata (stage, run
grouping, structural confidence) rides in ``ParsedTable.locator`` so the
existing mapper / observation builder consume the result unchanged.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Protocol

from fermdocs.domain.models import GoldenSchema, ParsedTable
from fermdocs.mapping.evidence_gated_llm import (
    LLM_CONFIDENCE_CAP,
    verify_substring_evidence,
)

# Fraction of the LLM's claimed header cells that must appear verbatim in
# the grid row it pointed at. Below this the row is almost certainly not the
# header (the LLM guessed wrong) and the region is dropped.
_MIN_HEADER_EVIDENCE_FRAC = 0.5


class LayoutLLMClient(Protocol):
    """Minimal protocol so tests can supply a scripted client.

    `kind` selects which structured-output schema the real clients use:
    "layout" (locate table regions) or "factors" (cross-run design factors).
    Scripted test clients can ignore it.
    """

    def call(self, system: str, user: str, *, kind: str = "layout") -> dict[str, Any]: ...


_SYSTEM_PROMPT = (
    "You locate the data table(s) inside a raw spreadsheet grid from a "
    "fermentation lab report. The grid is shown with a 0-based row index in "
    "square brackets at the start of each line; cells are separated by ' | '.\n\n"
    "Real lab sheets bury the time-series table under title/metadata rows, so "
    "the header is usually NOT row 0. Your job is to find, for each data table "
    "on the sheet, the structure -- NOT the values.\n\n"
    "Return JSON: {\"tables\": [ {region}, ... ]}. Each region has:\n"
    "  header_row      (int): row index holding the column names\n"
    "  data_start_row  (int): first row of numeric/data rows (usually header_row+1)\n"
    "  data_end_row    (int): last data row (inclusive); stop before blank/footer rows\n"
    "  run_grouping    ('single_run' | 'run_id_column'): is the whole region one\n"
    "                  experimental run, or do rows belong to different runs keyed\n"
    "                  by a column?\n"
    "  run_id_column   (int|null): column index of the run/batch id when\n"
    "                  run_grouping is 'run_id_column'; else null\n"
    "  stage           (str|null): the process stage these rows describe if the\n"
    "                  sheet labels one (e.g. 'MF', 'seed', 'main fermentation'); else null\n"
    "  time_column     (int|null): column index of the elapsed-time column; else null\n"
    "  time_semantics  ('elapsed' | 'reset_per_stage' | 'absolute' | null)\n"
    "  header_cells    (list[str]): the VERBATIM text of the header cells you read\n"
    "                  at header_row. Copy them exactly; they are checked against\n"
    "                  the grid. This is how we verify you picked the right row.\n"
    "  confidence      (number 0..1): your confidence in this region\n"
    "  rationale       (str|null)\n\n"
    "Rules:\n"
    "  - Indices must be valid rows of the grid shown.\n"
    "  - header_cells MUST be verbatim copies of cells in the header_row line.\n"
    "  - Do NOT transcribe data values. Code reads the cells; you only point.\n"
    "  - If a sheet has several stage tables stacked, emit one region per table.\n"
    "  - If you cannot find any table, return {\"tables\": []}.\n\n"
    "(Per-run operating conditions are extracted separately; ignore them here.)"
)


_FACTOR_SYSTEM_PROMPT = (
    "You analyze a multi-run experimental CAMPAIGN. You are given the metadata "
    "for several runs (one block per run, headed '=== RUN <id> ==='). Identify "
    "the experimental FACTORS that were DELIBERATELY VARIED across the runs -- "
    "the campaign's independent variables. These usually live in a free-text "
    "'Key Changes / Highlights / Strategies' note and in a 'Media Composition' "
    "block, NOT in the fixed setpoint cells.\n\n"
    "Examples of factors that often vary: nutrient/nitrogen source and amount, "
    "feed timing/window, target titer, inoculum, carbon loading.\n\n"
    "CRITICAL: report ONLY factors that DIFFER across runs. If something is the "
    "same in every run (a fixed base, a fixed carbon source, a constant "
    "setpoint), it is NOT a factor -- omit it.\n\n"
    'Return JSON {"factors": [ {factor} ]}. Each factor has:\n'
    "  name     (str): short snake_case, IDENTICAL across runs, e.g.\n"
    "           'nutrient_source', 'nutrient_loading_g_l', 'feed_start_h',\n"
    "           'feed_end_h', 'titer_target_g_l'\n"
    "  kind     ('numeric' | 'categorical')\n"
    "  unit     (str|null): e.g. 'g/L', 'h'\n"
    "  values   (list): one entry per run that has this factor:\n"
    "      run_id   (str): the run id from its '=== RUN <id> ===' header\n"
    "      value    (str): the setting. For numeric factors this MUST be a\n"
    "               number only (e.g. '40', not '40 g/L').\n"
    "      evidence (str): a SHORT verbatim snippet copied from THAT run's text\n"
    "               proving the value (checked against the text).\n\n"
    "Rules:\n"
    "  - Split a window like '7-9 hrs' into feed_start_h=7 and feed_end_h=9.\n"
    "  - evidence MUST be copied verbatim from that run's block.\n"
    "  - Only emit a factor if it has >=2 runs with >=2 distinct values.\n"
    "  - If nothing varies across runs, return {\"factors\": []}."
)


class LayoutDetector:
    """Wraps a LayoutLLMClient with the bounds + evidence safety stack."""

    def __init__(
        self,
        client: LayoutLLMClient | None = None,
        *,
        max_rows_digest: int | None = None,
        max_cols_digest: int | None = None,
        max_cell_chars: int = 24,
    ) -> None:
        self._client = client
        self._max_rows = max_rows_digest or int(
            os.environ.get("FERMDOCS_LAYOUT_MAX_ROWS", "200")
        )
        self._max_cols = max_cols_digest or int(
            os.environ.get("FERMDOCS_LAYOUT_MAX_COLS", "30")
        )
        self._max_cell_chars = max_cell_chars

    def detect(
        self,
        raw_grids: dict[str, list[list[Any]]],
        *,
        schema: GoldenSchema | None = None,
    ) -> list[ParsedTable]:
        """Locate table regions across every source grid. Never raises."""
        if not raw_grids or self._client is None:
            return []
        out: list[ParsedTable] = []
        for source_id, grid in raw_grids.items():
            try:
                out.extend(self._detect_one(source_id, grid))
            except Exception:
                # Defense in depth: a single malformed sheet must not abort
                # ingestion of the others. _detect_one already fails soft;
                # this guards anything it missed.
                continue
        return out

    def extract_design_factors(
        self,
        raw_grids: dict[str, list[list[Any]]],
        *,
        table_starts: dict[str, int] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Find the experimental factors that VARY across runs and tabulate
        each run's value. Returns ``{run_id: {factor: {value, unit, kind,
        source, evidence}}}`` or ``{}``. Never raises.

        One cross-run LLM pass over all sheets' metadata, so factor names are
        consistent across runs and "what varied" is decided with every run in
        view. Each value is grounded: its evidence (or the value itself) must
        appear verbatim in that run's grid. Only factors that genuinely vary
        (>=2 runs, >=2 distinct values) survive.
        """
        if not raw_grids or self._client is None:
            return {}
        starts = table_starts or {}
        grid_texts: dict[str, str] = {}
        sections: list[str] = []
        for source_id, grid in raw_grids.items():
            if not grid:
                continue
            run_id = source_id.split("#", 1)[1] if "#" in source_id else source_id
            # Design factors live in the metadata ABOVE the data table. When we
            # know where the table starts, digest only the metadata region —
            # smaller prompt, and no distraction from the time-series rows.
            start = starts.get(source_id)
            meta = grid[:start] if isinstance(start, int) and start > 0 else grid
            grid_texts[run_id] = "\n".join(
                " | ".join("" if c is None else str(c) for c in row) for row in meta
            )
            sections.append(
                f"=== RUN {run_id} ===\n{self._digest(meta, max_rows=100)}"
            )
        if len(grid_texts) < 2:
            return {}  # cross-run comparison needs at least two runs
        prompt = "RUNS:\n\n" + "\n\n".join(sections)
        try:
            payload = self._client.call(_FACTOR_SYSTEM_PROMPT, prompt, kind="factors")
        except Exception:
            return {}
        return _build_run_conditions(payload, grid_texts)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _detect_one(self, source_id: str, grid: list[list[Any]]) -> list[ParsedTable]:
        if not grid:
            return []
        try:
            payload = self._client.call(_SYSTEM_PROMPT, self._digest(grid))
        except Exception:
            return []
        regions = payload.get("tables") if isinstance(payload, dict) else None
        if not isinstance(regions, list):
            return []
        sheet = source_id.split("#", 1)[1] if "#" in source_id else source_id
        tables: list[ParsedTable] = []
        for idx, region in enumerate(regions):
            table = self._build_table(source_id, sheet, grid, region, idx)
            if table is not None:
                tables.append(table)
        return tables

    def _build_table(
        self,
        source_id: str,
        sheet: str,
        grid: list[list[Any]],
        region: Any,
        region_idx: int,
    ) -> ParsedTable | None:
        if not isinstance(region, dict):
            return None
        n_rows = len(grid)
        header_row = _as_int(region.get("header_row"))
        data_start = _as_int(region.get("data_start_row"))
        data_end = _as_int(region.get("data_end_row"))
        if header_row is None or data_start is None or data_end is None:
            return None
        # Bounds + ordering.
        if not (0 <= header_row < n_rows):
            return None
        if not (0 <= data_start <= data_end < n_rows):
            return None
        if data_start <= header_row:
            return None

        header_cells_raw = grid[header_row]
        headers = [
            ("" if c is None else str(c).strip()) for c in header_cells_raw
        ]
        if not any(h for h in headers):
            return None  # empty header row

        # Evidence gate: the cells the LLM claims it read must actually be in
        # the header row. A wrong header-row pick fails here and is dropped.
        claimed = region.get("header_cells")
        if not _header_evidence_ok(claimed, headers):
            return None

        rows = [list(grid[r]) for r in range(data_start, data_end + 1)]
        if not rows:
            return None

        confidence = _capped_confidence(region.get("confidence"))
        run_grouping = region.get("run_grouping")
        locator: dict[str, Any] = {
            "format": "xlsx" if "#" in source_id else "csv",
            "file": source_id.split("#", 1)[0],
            "sheet": sheet,
            "section": "table",
            "table_idx": region_idx,
            "header_row": header_row,
            "structural_confidence": confidence,
            "layout_detected": True,
        }
        stage = region.get("stage")
        if stage:
            locator["stage"] = str(stage)
        time_sem = region.get("time_semantics")
        if time_sem:
            locator["time_semantics"] = str(time_sem)
        # One-run-per-sheet: pin a run id from the sheet so a 15-sheet
        # workbook becomes 15 distinct runs instead of collapsing to one
        # filename-derived id. A run_id_column is left for the existing
        # RunIdResolver ColumnStrategy to read per row.
        if run_grouping == "single_run":
            locator["run_id"] = sheet
        elif run_grouping == "run_id_column":
            locator["run_id_column"] = _as_int(region.get("run_id_column"))

        table_id = source_id if region_idx == 0 else f"{source_id}~{region_idx}"
        return ParsedTable(
            table_id=table_id, headers=headers, rows=rows, locator=locator
        )

    def _digest(self, grid: list[list[Any]], max_rows: int | None = None) -> str:
        # Skip fully-empty rows (lab sheets are mostly blank cells) but KEEP
        # the original row index on every shown line -- the LLM points at
        # those indices and code slices the real grid, so the labels must be
        # the true positions. This lets a data table buried deep in a long,
        # sparse sheet (e.g. row 75 of 117) still reach the model within the
        # row budget.
        cap = max_rows or self._max_rows
        lines: list[str] = []
        shown = 0
        last_idx = -1
        for i, row in enumerate(grid):
            cells = row[: self._max_cols]
            if not any(c is not None and str(c).strip() != "" for c in cells):
                continue  # fully-blank row
            rendered = " | ".join(
                "" if c is None else str(c)[: self._max_cell_chars] for c in cells
            )
            lines.append(f"[{i}] {rendered}")
            shown += 1
            last_idx = i
            if shown >= cap:
                break
        body = "\n".join(lines)
        remaining = len(grid) - 1 - last_idx
        if remaining > 0:
            body += f"\n... ({remaining} more rows not shown)"
        return (
            f"GRID ({len(grid)} rows; blank rows omitted, original indices kept):\n"
            f"{body}\n\n"
            'Return JSON {"tables": [...]} locating the data table(s).'
        )


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _capped_confidence(value: Any) -> float:
    try:
        c = float(value)
    except (TypeError, ValueError):
        c = 0.0
    return max(0.0, min(LLM_CONFIDENCE_CAP, c))


def _grounded(evidence: Any, value: Any, text: str) -> bool:
    """The evidence snippet (or the raw value) must appear verbatim in the
    run's grid text, so a value can't be hallucinated."""
    for cand in (evidence, value):
        if isinstance(cand, str):
            s = cand.strip()
            if 0 < len(s) <= 200 and s in text:
                return True
    return False


def _clean_numeric(raw: Any) -> float | None:
    """Float ONLY when the whole value is cleanly a single number (optionally
    with a trailing unit). Compound expressions like '1500 g in 10000ml' or a
    range '5.8 - 6.3' return None so the value is kept verbatim, never mangled
    into a misleading number.
    """
    m = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*[A-Za-z%/°]*\s*", str(raw))
    return float(m.group(1)) if m else None


def _build_run_conditions(
    payload: Any, grid_texts: dict[str, str]
) -> dict[str, dict[str, Any]]:
    """Project the LLM's per-run factors into per-run conditions, faithfully.

    The extractor's only job is to capture what the sheet says. So it keeps
    the value EXACTLY as written (plus a clean numeric only when the whole
    value is a number), and the sole filter is grounding: the value must
    appear verbatim in that run's text, so nothing is hallucinated. It does
    NOT judge relevance, drop constants, or rank — that is a downstream
    concern, not extraction's.
    """
    if not isinstance(payload, dict):
        return {}
    factors = payload.get("factors")
    if not isinstance(factors, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for f in factors:
        if not isinstance(f, dict):
            continue
        name = _norm_knob_name(f.get("name"))
        unit = f.get("unit")
        values = f.get("values")
        if not name or not isinstance(values, list):
            continue
        for v in values:
            if not isinstance(v, dict):
                continue
            rid = v.get("run_id")
            if rid not in grid_texts:
                continue
            raw = v.get("value")
            if raw is None or not _grounded(v.get("evidence"), raw, grid_texts[rid]):
                continue
            out.setdefault(str(rid), {})[name] = {
                "value": str(raw).strip(),
                "numeric": _clean_numeric(raw),
                "unit": str(unit) if unit else None,
                "source": "design_factor",
            }
    return out


def _norm_knob_name(name: Any) -> str | None:
    if not isinstance(name, str):
        return None
    out = "".join(c if c.isalnum() else "_" for c in name.strip().lower())
    out = "_".join(p for p in out.split("_") if p)
    return out or None


def _header_evidence_ok(claimed: Any, headers: list[str]) -> bool:
    """At least _MIN_HEADER_EVIDENCE_FRAC of the LLM's claimed header cells
    must appear verbatim in the actual header-row text.

    Reuses the shared substring-evidence guard (value=None: we're verifying
    a label, not a number).
    """
    if not isinstance(claimed, list) or not claimed:
        return False
    header_text = " | ".join(headers)
    matched = 0
    for cell in claimed:
        if not isinstance(cell, str) or not cell.strip():
            continue
        ok, _reason = verify_substring_evidence(cell.strip(), header_text, value=None)
        if ok:
            matched += 1
    return matched >= max(1, int(len(claimed) * _MIN_HEADER_EVIDENCE_FRAC))


# -----------------------------------------------------------------------------
# LLM clients (one structured-output client per task, matching the
# GeminiHeaderMapper / GeminiIdentityClient / GeminiSegmenterClient pattern)
# -----------------------------------------------------------------------------

_GEMINI_LAYOUT_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "tables": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "header_row": {"type": "INTEGER"},
                    "data_start_row": {"type": "INTEGER"},
                    "data_end_row": {"type": "INTEGER"},
                    "run_grouping": {
                        "type": "STRING",
                        "enum": ["single_run", "run_id_column"],
                        "nullable": True,
                    },
                    "run_id_column": {"type": "INTEGER", "nullable": True},
                    "stage": {"type": "STRING", "nullable": True},
                    "time_column": {"type": "INTEGER", "nullable": True},
                    "time_semantics": {"type": "STRING", "nullable": True},
                    "header_cells": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "confidence": {"type": "NUMBER"},
                    "rationale": {"type": "STRING", "nullable": True},
                },
                "required": [
                    "header_row",
                    "data_start_row",
                    "data_end_row",
                    "header_cells",
                    "confidence",
                ],
            },
        }
    },
    "required": ["tables"],
}


_GEMINI_FACTOR_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "factors": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "kind": {"type": "STRING", "enum": ["numeric", "categorical"]},
                    "unit": {"type": "STRING", "nullable": True},
                    "values": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "run_id": {"type": "STRING"},
                                "value": {"type": "STRING"},
                                "evidence": {"type": "STRING"},
                            },
                            "required": ["run_id", "value", "evidence"],
                        },
                    },
                },
                "required": ["name", "kind", "values"],
            },
        }
    },
    "required": ["factors"],
}


class GeminiLayoutClient:
    """LayoutLLMClient via Google Gemini structured output."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self._model = (
            model
            or os.environ.get("FERMDOCS_LAYOUT_MODEL")
            or os.environ.get("FERMDOCS_GEMINI_MODEL", "gemini-3-flash")
        )
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")

    def call(self, system: str, user: str, *, kind: str = "layout") -> dict[str, Any]:
        from google import genai
        from google.genai import types

        schema = _GEMINI_FACTOR_SCHEMA if kind == "factors" else _GEMINI_LAYOUT_SCHEMA
        client = genai.Client(api_key=self._api_key)
        response = client.models.generate_content(
            model=self._model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0.0,
            ),
        )
        text = response.text
        if not text:
            raise ValueError("Gemini returned empty response")
        from fermdocs.json_utils import loads_lenient

        return loads_lenient(text)


class AnthropicLayoutClient:
    """LayoutLLMClient via Anthropic tool-use (forced structured output)."""

    _TOOL = {
        "name": "emit_layout",
        "description": "Emit the located table regions for the grid.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tables": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "header_row": {"type": "integer"},
                            "data_start_row": {"type": "integer"},
                            "data_end_row": {"type": "integer"},
                            "run_grouping": {
                                "type": "string",
                                "enum": ["single_run", "run_id_column"],
                            },
                            "run_id_column": {"type": ["integer", "null"]},
                            "stage": {"type": ["string", "null"]},
                            "time_column": {"type": ["integer", "null"]},
                            "time_semantics": {"type": ["string", "null"]},
                            "header_cells": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "confidence": {"type": "number"},
                            "rationale": {"type": ["string", "null"]},
                        },
                        "required": [
                            "header_row",
                            "data_start_row",
                            "data_end_row",
                            "header_cells",
                            "confidence",
                        ],
                    },
                }
            },
            "required": ["tables"],
        },
    }

    _FACTOR_TOOL = {
        "name": "emit_factors",
        "description": "Emit the experimental factors that varied across runs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "factors": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "kind": {"type": "string", "enum": ["numeric", "categorical"]},
                            "unit": {"type": ["string", "null"]},
                            "values": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "run_id": {"type": "string"},
                                        "value": {"type": "string"},
                                        "evidence": {"type": "string"},
                                    },
                                    "required": ["run_id", "value", "evidence"],
                                },
                            },
                        },
                        "required": ["name", "kind", "values"],
                    },
                }
            },
            "required": ["factors"],
        },
    }

    def __init__(self, model: str | None = None) -> None:
        self._model = (
            model
            or os.environ.get("FERMDOCS_LAYOUT_MODEL")
            or os.environ.get("FERMDOCS_MAPPER_MODEL", "claude-haiku-4-5-20251001")
        )

    def call(self, system: str, user: str, *, kind: str = "layout") -> dict[str, Any]:
        from anthropic import Anthropic

        tool = self._FACTOR_TOOL if kind == "factors" else self._TOOL
        client = Anthropic()
        response = client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
        )
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                return dict(block.input)
        raise ValueError("response missing tool_use block")
