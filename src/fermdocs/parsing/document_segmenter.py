"""LLM-driven document segmentation for multi-batch PDFs.

Plan ref: docs/design/2026-05-03-pdf-document-segmentation.md

Multi-batch PDFs (e.g. carotenoid: 6 batches in one file) lose section context
before the run-id resolver runs. The resolver then invents per-row fake
run-ids from the time column, destroying trajectory structure.

This module fixes the upstream loss. The segmenter reads the document outline
(headings + first-line previews of text blocks + table positions, NOT table
values) and asks an LLM to identify experimental runs and assign each table
to a run. The output is a DocumentMap that the ingestion pipeline injects as
manifest_run_id per table — from the resolver's perspective it looks like an
operator-supplied manifest.

CSV / Excel paths never run the segmenter. The caller (IngestionPipeline)
must guard with a parser-type check before calling segment().

Failure modes (LLM down, malformed JSON, contradictory map, etc.) all fall
back to None, and the pipeline falls through to the existing column-heuristic
chain. The segmenter never raises into the pipeline.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol

from fermdocs.domain.models import (
    DocumentMap,
    NarrativeBlock,
    NarrativeBlockType,
    ParseResult,
    ParsedTable,
    RunSegment,
    RunSegmentSource,
)

_log = logging.getLogger(__name__)


class SegmenterLLMClient(Protocol):
    """Minimal protocol matching IdentityLLMClient — keeps the contract narrow.

    Implementations live alongside the gemini/anthropic identity clients.
    Tests supply a scripted stub. The segmenter validates output structurally
    regardless of what the client returns.
    """

    def call(self, system: str, user: str) -> dict[str, Any]: ...


# -----------------------------------------------------------------------------
# Outline construction — what the LLM sees
# -----------------------------------------------------------------------------


# First-line preview length per text block. Long enough to identify
# "Batch closure: Cultivation completed at 96h" and similar boundary
# phrases; short enough to keep prompt bounded for big PDFs.
_TEXT_PREVIEW_CHARS = 200


def build_outline(parse_result: ParseResult) -> str:
    """Render the parsed document as a structural outline for the LLM.

    Includes: every narrative block (heading / paragraph / list item) with
    a truncated text preview, every table with its headers, all interleaved
    in document order via locator paragraph_idx / table_idx fields.

    Excludes: table cell values. The LLM is making a structural decision
    (which tables belong to which run); it doesn't need the data.
    """
    items: list[tuple[int, int, str]] = []  # (page, idx, line)
    for block in parse_result.narrative_blocks:
        page = int(block.locator.get("page") or 0)
        idx = int(block.locator.get("paragraph_idx") or 0)
        kind = _block_kind_label(block.type)
        preview = block.text.strip().replace("\n", " ")[:_TEXT_PREVIEW_CHARS]
        items.append((page, idx, f"[{kind:<11} p{page}] {preview!r}"))

    for table in parse_result.tables:
        page = int(table.locator.get("page") or 0)
        table_idx = int(table.locator.get("table_idx") or 0)
        # Use a large idx offset so tables sort after narrative on same page.
        # Tables are typically rendered after the heading that introduces them,
        # which is what we want.
        sort_key = 1_000_000 + table_idx
        headers_preview = ", ".join(
            (h or "").strip()[:30] for h in table.headers[:6]
        )
        items.append(
            (
                page,
                sort_key,
                f"[TABLE idx={table_idx:<3} p{page}] headers: [{headers_preview}]",
            )
        )

    items.sort(key=lambda x: (x[0], x[1]))
    return "\n".join(line for _, _, line in items)


def _block_kind_label(block_type: NarrativeBlockType) -> str:
    return {
        NarrativeBlockType.HEADING: "HEADER",
        NarrativeBlockType.PARAGRAPH: "TEXT",
        NarrativeBlockType.LIST_ITEM: "LIST",
        NarrativeBlockType.CAPTION: "CAPTION",
        NarrativeBlockType.OTHER: "OTHER",
    }.get(block_type, "OTHER")


# -----------------------------------------------------------------------------
# Prompt
# -----------------------------------------------------------------------------


SYSTEM_PROMPT = (
    "You are a fermentation-data engineer reading a parsed PDF outline.\n"
    "Your job: identify experimental runs (one fermentation = one vessel,\n"
    "one timecourse) and assign each TABLE to the run it belongs to.\n"
    "\n"
    "Rules:\n"
    "  - A run typically corresponds to a `BATCH-NN REPORT` heading or a\n"
    "    similar section boundary (e.g., 'Experiment 3', 'Trial 2').\n"
    "  - Each TABLE belongs to AT MOST ONE run.\n"
    "  - Skip composition tables (Component | Concentration), feed-plan\n"
    "    tables (Segment | Batch hours), and supplement-additions tables.\n"
    "    They are NOT measurement data — list them in unassigned_table_indices.\n"
    "  - Assign only timecourse measurement tables (Time | OD | WCW | etc.)\n"
    "    to runs.\n"
    "  - If the document has no clear run boundaries, return runs=[] and\n"
    "    list all measurement tables in unassigned_table_indices. Do NOT\n"
    "    invent run boundaries.\n"
    "  - run_id format: 'RUN-NNNN' (zero-padded 4-digit). Number runs in\n"
    "    document order starting from 0001.\n"
    "  - Provide a one-sentence rationale per run citing the boundary signal.\n"
)


def build_user_prompt(parse_result: ParseResult, file_id: str) -> str:
    outline = build_outline(parse_result)
    n_tables = len(parse_result.tables)
    return (
        f"Document file_id: {file_id}\n"
        f"Total measurement-candidate tables: {n_tables}\n"
        f"(Feed-plan tables already filtered out upstream.)\n"
        f"\n"
        f"Outline:\n"
        f"{outline}\n"
        f"\n"
        f"Return DocumentMap JSON. table_indices values must be drawn from the\n"
        f"set of TABLE idx values shown above (0..{n_tables - 1 if n_tables else 0})."
    )


# -----------------------------------------------------------------------------
# Segmenter
# -----------------------------------------------------------------------------


class DocumentSegmenter:
    """Orchestrates the LLM call, schema validation, and fallback.

    Construction: pass a client (SegmenterLLMClient) and the model name +
    provider strings used in DocumentMap provenance fields. Tests pass a
    scripted client; production wires GeminiSegmenterClient via the factory.

    `segment(parse_result, file_id)` returns a DocumentMap on success, or
    None on any failure (LLM down, malformed output, contradictory map,
    table_idx out of range, etc.). Callers must handle None — the
    pipeline falls through to existing column heuristics.
    """

    def __init__(
        self,
        client: SegmenterLLMClient | None,
        *,
        model_name: str,
        provider: str,
    ) -> None:
        self._client = client
        self._model_name = model_name
        self._provider = provider

    def segment(
        self,
        parse_result: ParseResult,
        *,
        file_id: str,
        manifest_run_id: str | None = None,
    ) -> DocumentMap | None:
        """Segment a parsed PDF into runs.

        `manifest_run_id`, when supplied, indicates the operator pinned a
        single run-id for the entire file. The segmenter still runs (its
        output is recorded for inspection) but if the LLM disagrees with
        the manifest — i.e. detects multiple distinct runs — a loud WARN
        is emitted naming both. Per design doc: manifest wins, loudly.
        Pipeline still applies the manifest to every table; the WARN is
        the operator's signal that the manifest may be wrong.
        """
        if self._client is None:
            _log.info("segmenter: no LLM client configured; skipping")
            return None
        if not parse_result.tables:
            # Nothing to segment. Return an empty map so callers can persist
            # provenance ("we ran, found nothing"). Validation accepts this.
            return DocumentMap(
                file_id=file_id,
                runs=[],
                unassigned_table_indices=[],
                overall_confidence=0.0,
                llm_model=self._model_name,
                llm_provider=self._provider,
            )

        n_tables = len(parse_result.tables)
        valid_indices = {
            int(t.locator.get("table_idx", -1)) for t in parse_result.tables
        }

        try:
            raw = self._client.call(
                system=SYSTEM_PROMPT,
                user=build_user_prompt(parse_result, file_id),
            )
        except Exception as e:  # noqa: BLE001 - LLM client may raise anything
            _log.warning("segmenter: LLM call failed: %s", e)
            return None

        try:
            doc_map = self._parse_response(raw, file_id=file_id)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as e:
            _log.warning("segmenter: response validation failed: %s", e)
            return None

        # Range check: the LLM may have hallucinated a table_idx that doesn't
        # exist in the parsed document. Reject the whole map rather than
        # silently dropping bad indices, because the model's run boundaries
        # may be wrong if it saw phantom tables.
        used = {idx for run in doc_map.runs for idx in run.table_indices}
        used |= set(doc_map.unassigned_table_indices)
        unknown = used - valid_indices
        if unknown:
            _log.warning(
                "segmenter: LLM referenced unknown table_idx values %s "
                "(valid range: %d tables); falling back",
                sorted(unknown),
                n_tables,
            )
            return None

        # Manifest disagreement check: if the operator pinned a single
        # run-id but the LLM detected multiple runs, log loudly. Manifest
        # still wins (downstream pipeline applies it); this is the
        # operator's signal that the pinning may be wrong.
        if manifest_run_id is not None and len(doc_map.runs) > 1:
            display_names = [r.display_name for r in doc_map.runs]
            _log.warning(
                "segmenter: manifest pinned 1 run (%r) for file %s, but"
                " segmenter detected %d runs: %s. Manifest wins. If the"
                " segmenter is right, omit the manifest from this ingest.",
                manifest_run_id,
                file_id,
                len(doc_map.runs),
                display_names,
            )

        return doc_map

    def _parse_response(
        self, raw: dict[str, Any], *, file_id: str
    ) -> DocumentMap:
        """Coerce the LLM dict into a validated DocumentMap.

        We force-set file_id, llm_model, llm_provider from the segmenter's
        own state — the LLM doesn't need to (and shouldn't) echo these.
        Schema validation in DocumentMap catches the rest (duplicate
        table_idx, contradictory assignment, etc.).
        """
        runs_raw = raw.get("runs") or []
        runs: list[RunSegment] = []
        for r in runs_raw:
            source_str = (r.get("source_signal") or "inferred").lower()
            try:
                source = RunSegmentSource(source_str)
            except ValueError:
                source = RunSegmentSource.INFERRED
            runs.append(
                RunSegment(
                    run_id=r["run_id"],
                    display_name=r.get("display_name") or r["run_id"],
                    table_indices=[int(i) for i in r.get("table_indices", [])],
                    source_signal=source,
                    confidence=float(r.get("confidence", 0.5)),
                    rationale=r.get("rationale") or "",
                )
            )

        return DocumentMap(
            file_id=file_id,
            runs=runs,
            unassigned_table_indices=[
                int(i) for i in raw.get("unassigned_table_indices", [])
            ],
            overall_confidence=float(raw.get("overall_confidence", 0.5)),
            llm_model=self._model_name,
            llm_provider=self._provider,
        )
