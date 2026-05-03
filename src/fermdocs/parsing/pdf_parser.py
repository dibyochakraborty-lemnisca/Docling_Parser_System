from __future__ import annotations

from pathlib import Path
from typing import Any

from fermdocs.domain.models import (
    NarrativeBlock,
    NarrativeBlockType,
    ParsedTable,
    ParseResult,
)
from fermdocs.parsing.base import FileParser

_MIN_NARRATIVE_LEN = 20  # skip very short blocks (page numbers, single-word headers)


class DoclingPdfParser(FileParser):
    """Extract tables AND narrative blocks from a PDF via Docling.

    Docling is heavy (pulls ML deps). Imports stay inside .parse() so the rest
    of the package works without `fermdocs[pdf]` installed.

    Both tables and narrative blocks (paragraphs, headings, list items, captions)
    are returned. The pipeline always captures narrative blocks into residual JSONB
    (Tier 1). Optional LLM extraction over them is opt-in (Tier 2).
    """

    def __init__(self, converter: Any | None = None) -> None:
        self._converter = converter

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".pdf"

    def parse(self, path: Path) -> ParseResult:
        converter = self._converter or _build_default_converter()
        result = converter.convert(str(path))
        document = result.document
        tables, feed_plan_tables = self._extract_tables(document, path)
        narrative_blocks = self._extract_narrative(document, path)
        return ParseResult(
            tables=tables,
            narrative_blocks=narrative_blocks,
            feed_plan_tables=feed_plan_tables,
        )

    def _extract_tables(
        self, document: Any, path: Path
    ) -> tuple[list[ParsedTable], list[ParsedTable]]:
        """Walk Docling tables, route feed-plan tables out of the observation stream.

        Returns (measurement_tables, feed_plan_tables). Feed-plan tables are
        operator-supplied feeding schedules (Segment | Batch hours | Feed rate)
        that look like measurements but aren't — they're planned setpoints. Left
        in the observation stream they pollute feed_rate_l_per_h with
        impossible-looking values (e.g. 15 mL/h appearing as 15 L/h after unit
        normalization) and trigger false diagnose alarms.
        """
        measurement_tables: list[ParsedTable] = []
        feed_plan_tables: list[ParsedTable] = []
        for table_idx, table in enumerate(getattr(document, "tables", []) or []):
            headers, rows = _table_to_grid(table, document)
            if not headers:
                continue
            page_no = _page_no_for(table)
            locator: dict[str, Any] = {
                "format": "pdf",
                "file": path.name,
                "page": page_no,
                "table_idx": table_idx,
                "section": "table",
            }
            parsed = ParsedTable(
                table_id=f"{path.name}#p{page_no}#t{table_idx}",
                headers=headers,
                rows=rows,
                locator=locator,
            )
            if _is_feed_plan_table(headers):
                feed_plan_tables.append(parsed)
            else:
                measurement_tables.append(parsed)
        return measurement_tables, feed_plan_tables

    def _extract_narrative(self, document: Any, path: Path) -> list[NarrativeBlock]:
        """Walk Docling's text items. Skip tables and very-short artifacts."""
        blocks: list[NarrativeBlock] = []
        text_items = getattr(document, "texts", None) or []
        for idx, item in enumerate(text_items):
            text = (getattr(item, "text", "") or "").strip()
            if len(text) < _MIN_NARRATIVE_LEN:
                continue
            label = getattr(item, "label", None)
            block_type = _classify_label(label)
            page_no = _page_no_for(item)
            blocks.append(
                NarrativeBlock(
                    text=text,
                    type=block_type,
                    locator={
                        "format": "pdf",
                        "file": path.name,
                        "page": page_no,
                        "section": "narrative",
                        "paragraph_idx": idx,
                    },
                )
            )
        return blocks


def _build_default_converter() -> Any:
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as e:  # pragma: no cover - exercised at runtime only
        raise RuntimeError(
            "Docling is not installed. Install with: pip install -e '.[pdf]'"
        ) from e
    import os

    options = PdfPipelineOptions(
        do_ocr=os.environ.get("FERMDOCS_PDF_OCR", "false").lower() == "true",
        do_table_structure=True,
    )
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def _table_to_grid(table: Any, document: Any | None = None) -> tuple[list[str], list[list[Any]]]:
    """Pull headers + rows from a Docling TableItem."""
    df = None
    export = getattr(table, "export_to_dataframe", None)
    if callable(export):
        try:
            df = export(document) if document is not None else export()
        except TypeError:
            try:
                df = export()
            except Exception:
                df = None
        except Exception:
            df = None
    if df is not None and not df.empty:
        headers = [str(c).strip() for c in df.columns]
        rows = [
            [_normalize(v) for v in row]
            for row in df.itertuples(index=False, name=None)
        ]
        return headers, rows

    grid = getattr(getattr(table, "data", None), "grid", None)
    if not grid:
        return [], []
    headers = [_normalize(getattr(c, "text", "")) or "" for c in grid[0]]
    rows = [[_normalize(getattr(c, "text", "")) for c in row] for row in grid[1:]]
    return headers, rows


def _classify_label(label: Any) -> NarrativeBlockType:
    if label is None:
        return NarrativeBlockType.OTHER
    s = str(label).lower()
    if "head" in s or "title" in s:
        return NarrativeBlockType.HEADING
    if "list" in s or "item" in s:
        return NarrativeBlockType.LIST_ITEM
    if "caption" in s:
        return NarrativeBlockType.CAPTION
    if "text" in s or "paragraph" in s:
        return NarrativeBlockType.PARAGRAPH
    return NarrativeBlockType.OTHER


def _page_no_for(item: Any) -> int | None:
    prov = getattr(item, "prov", None)
    if not prov:
        return None
    first = prov[0] if isinstance(prov, list) else prov
    return getattr(first, "page_no", None) or getattr(first, "page", None)


def _normalize(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        return s if s != "" else None
    return v


def _is_feed_plan_table(headers: list[str]) -> bool:
    """True iff headers match the operator-supplied feeding-schedule shape.

    Pattern observed in fermentation reports (e.g. carotenoid BATCH-04 page 10):

        Segment | Batch hours (start → end) | Duration (h) | Feed rate (mL/h) | Feed volume (mL)

    These tables describe planned setpoints, not measurements. Routing them
    into the observation stream produces fake feed_rate_l_per_h values that
    diagnose flags as physically impossible.

    Detection rule: a "Segment" header AND at least one of "Batch hours" or
    "Feed rate". Conservative — we'd rather miss a real feed-plan table and
    have it flow through as observations (current behavior) than misclassify
    a real measurement table and lose data.
    """
    norm = [(h or "").strip().lower() for h in headers]
    has_segment = any("segment" in h for h in norm)
    has_batch_hours = any("batch hour" in h for h in norm)
    has_feed_rate = any("feed rate" in h for h in norm)
    return has_segment and (has_batch_hours or has_feed_rate)
