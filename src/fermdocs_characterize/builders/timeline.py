"""Timeline builder: one TimelineEvent per Finding with time_window.

Sort: time asc, severity desc, run_id, variable. lag_to_next_seconds is
computed within each run only (cross-run lags don't make physical sense).

IDs assigned in sort order: E-0001, E-0002, …
"""

from __future__ import annotations

from fermdocs_characterize.schema import Finding, Severity, TimelineEvent

_SEVERITY_RANK = {
    Severity.CRITICAL: 4,
    Severity.MAJOR: 3,
    Severity.MINOR: 2,
    Severity.INFO: 1,
}


def build_timeline(findings: list[Finding]) -> list[TimelineEvent]:
    events_data: list[tuple[float, str, str, Severity, Finding]] = []
    for f in findings:
        if f.time_window is None or f.time_window.start is None:
            continue
        run_id = f.run_ids[0] if f.run_ids else ""
        variable = f.variables_involved[0] if f.variables_involved else ""
        events_data.append((f.time_window.start, run_id, variable, f.severity, f))

    events_data.sort(
        key=lambda x: (x[0], -_SEVERITY_RANK[x[3]], x[1], x[2])
    )

    out: list[TimelineEvent] = []
    next_by_run: dict[str, float | None] = {}
    # First pass: collect events ordered, then compute lags per run.
    raw_events: list[TimelineEvent] = []
    for i, (t, run_id, _variable, severity, f) in enumerate(events_data, start=1):
        raw_events.append(
            TimelineEvent(
                event_id=f"E-{i:04d}",
                run_id=run_id,
                time=t,
                finding_id=f.finding_id,
                summary=f.summary,
                severity=severity,
                lag_to_next_seconds=None,
            )
        )

    # For each run, compute lag from event[i] to event[i+1] in seconds.
    by_run_indices: dict[str, list[int]] = {}
    for idx, ev in enumerate(raw_events):
        by_run_indices.setdefault(ev.run_id, []).append(idx)

    out_list: list[TimelineEvent] = list(raw_events)
    for run_id, indices in by_run_indices.items():
        for j in range(len(indices) - 1):
            cur_i = indices[j]
            nxt_i = indices[j + 1]
            lag_h = out_list[nxt_i].time - out_list[cur_i].time
            if lag_h <= 0:
                continue
            lag_s = round(lag_h * 3600, 6)
            out_list[cur_i] = out_list[cur_i].model_copy(update={"lag_to_next_seconds": lag_s})
        # last event in run keeps lag_to_next_seconds=None

    out = out_list
    _ = next_by_run  # silence unused
    return out
