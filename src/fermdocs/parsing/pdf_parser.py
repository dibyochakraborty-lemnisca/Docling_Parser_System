from __future__ import annotations

from pathlib import Path
from typing import Any

from fermdocs.domain.models import ParsedTable
from fermdocs.parsing.base import FileParser


class DoclingPdfParser(FileParser):
    """Extract tables from a PDF via Docling.

    Docling is heavy (pulls ML deps). Imports stay inside .parse() so the rest
    of the package works without `fermdocs[pdf]` installed.
    """

    def __init__(self, converter: Any | None = None) -> None:
        self._converter = converter

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".pdf"

    def parse(self, path: Path) -> list[ParsedTable]:
        converter = self._converter or _build_default_converter()
        result = converter.convert(str(path))
        document = result.document
        tables: list[ParsedTable] = []
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
            }
            tables.append(
                ParsedTable(
                    table_id=f"{path.name}#p{page_no}#t{table_idx}",
                    headers=headers,
                    rows=rows,
                    locator=locator,
                )
            )
        return tables


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
    """Pull headers + rows from a Docling TableItem.

    Strategy: prefer DataFrame export (handles cell merging cleanly), fall back
    to manual grid construction from table.data.grid if export isn't available.
    """
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
    rows = [
        [_normalize(getattr(c, "text", "")) for c in row]
        for row in grid[1:]
    ]
    return headers, rows


def _page_no_for(table: Any) -> int | None:
    prov = getattr(table, "prov", None)
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

