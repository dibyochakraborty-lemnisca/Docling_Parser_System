"""Append-only trace writer with spillover for oversized records.

Two callers share this writer:
  - execute_python writes `kind="python_call"` records to
    audit/python_trace.jsonl
  - The ReAct loop writes `kind="tool_call"|"llm_response"|"tool_result"`
    records to audit/diagnosis_trace.jsonl

Records over SPILL_THRESHOLD_BYTES are written to
`<jsonl_parent>/python_calls/<seq:06d>.json` and the jsonl line becomes a
pointer. Either way, the FULL record is preserved on disk — the audit/
invariant is "never read at runtime", not "truncate to fit".
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

SPILL_THRESHOLD_BYTES = 100_000


class TraceRecord(dict):
    """Marker subtype so callers can self-document without forcing a schema.

    Records are free-form JSON-able dicts. By convention every record has:
      - seq (int, monotonic per writer)
      - ts (float, unix epoch)
      - kind (str)
    Callers add whatever else makes sense for their kind.
    """


class TraceWriter:
    """Write-only sink for trace records.

    Thread-safe (single Lock) but not multi-process safe — each process should
    own its own writer. Spilled records live next to the jsonl file in a
    `<basename>_spill/` subdir; pointer entries reference them by relative path.
    """

    def __init__(self, jsonl_path: str | Path) -> None:
        self._path = Path(jsonl_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._spill_dir = self._path.parent / f"{self._path.stem}_spill"
        self._seq = 0
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def spill_dir(self) -> Path:
        return self._spill_dir

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def write(self, record: dict[str, Any]) -> int:
        """Append `record` to the jsonl. Returns the assigned seq.

        Adds `seq` and `ts` if absent. Never raises on I/O errors during
        trace I/O — audit must not break the user's run. Errors are swallowed
        silently (matches the reference repo's posture).
        """
        with self._lock:
            try:
                seq = record.get("seq")
                if not isinstance(seq, int):
                    seq = self._next_seq()
                else:
                    self._seq = max(self._seq, seq)
                payload = {
                    "seq": seq,
                    "ts": record.get("ts", time.time()),
                    **{k: v for k, v in record.items() if k not in ("seq", "ts")},
                }
                serialized = json.dumps(payload, default=str)
                if len(serialized.encode("utf-8")) > SPILL_THRESHOLD_BYTES:
                    self._spill_dir.mkdir(parents=True, exist_ok=True)
                    spill_path = self._spill_dir / f"{seq:06d}.json"
                    spill_path.write_text(serialized)
                    pointer = {
                        "seq": seq,
                        "ts": payload["ts"],
                        "kind": payload.get("kind"),
                        "spilled_to": f"{self._spill_dir.name}/{spill_path.name}",
                        "serialized_bytes": len(serialized.encode("utf-8")),
                    }
                    with open(self._path, "a") as f:
                        f.write(json.dumps(pointer) + "\n")
                else:
                    with open(self._path, "a") as f:
                        f.write(serialized + "\n")
                return seq
            except OSError:
                return -1
