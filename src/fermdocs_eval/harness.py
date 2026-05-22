"""Shared harness types and JSONL writer for eval runs.

Every eval row that lands in `eval/results/<suite>.jsonl` is an `EvalRun`
plus suite-specific payload. The shape is intentionally narrow so each
suite can layer its own fields on top via the `payload` dict.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class RunStatus(str, Enum):
    OK = "ok"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class EvalRun:
    """One row of an eval suite — a single trial."""

    suite: str  # "e1" | "e2" | "e3"
    trial_id: str  # stable identifier within the suite
    status: RunStatus
    started_at: str  # ISO-8601 UTC
    finished_at: str  # ISO-8601 UTC
    payload: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, run: EvalRun) -> None:
    """Append one EvalRun to a JSONL file (creates parents if needed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    row = asdict(run)
    row["status"] = run.status.value
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read all rows from a results JSONL file. Empty list if missing."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
