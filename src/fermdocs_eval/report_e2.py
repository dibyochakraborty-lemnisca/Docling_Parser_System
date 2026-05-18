"""Render E2 results JSONL into eval/e2_critic_axes.md for paper integration.

Pure I/O + tables — all math lives in metrics.py. Run after the batch:
    python -m fermdocs_eval.report_e2

Output: eval/e2_critic_axes.md + figures inline as markdown tables.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from fermdocs_eval.harness import read_jsonl
from fermdocs_eval.metrics import (
    catch_rate,
    confusion_matrix,
    over_fire_rate,
    per_axis_precision_recall,
    tag_accuracy,
)
from fermdocs_eval.synthetic import CRITIC_AXES

RESULTS = Path("eval/results/e2.jsonl")
OUT_MD = Path("eval/e2_critic_axes.md")


def _rows() -> list[dict]:
    raw = read_jsonl(RESULTS)
    # Flatten payload up to row level so metrics functions can read directly.
    out = []
    for r in raw:
        if r.get("status") != "ok":
            continue
        payload = r.get("payload") or {}
        out.append(
            {
                "trial_id": r["trial_id"],
                "labeled_axis": payload.get("labeled_axis"),
                "difficulty": payload.get("difficulty"),
                "fired_axes": payload.get("fired_axes") or [],
                "n_critiques_filed": payload.get("n_critiques_filed", 0),
                "n_final_hypotheses": payload.get("n_final_hypotheses", 0),
            }
        )
    return out


def _md_table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def render() -> None:
    rows = _rows()
    if not rows:
        OUT_MD.write_text(
            "# E2 — Critic Axes\n\nNo results yet. Run "
            "`python -m fermdocs_eval.cli e2` first.\n"
        )
        return

    total = len(rows)
    by_axis = Counter(r["labeled_axis"] for r in rows)
    by_difficulty = Counter(r["difficulty"] for r in rows)

    catch = catch_rate(rows)
    tag = tag_accuracy(rows)
    per_axis = per_axis_precision_recall(rows, CRITIC_AXES)
    overfire = over_fire_rate(rows)
    matrix = confusion_matrix(rows, CRITIC_AXES)

    error_rows = [
        r for r in read_jsonl(RESULTS) if r.get("status") == "error"
    ]

    lines: list[str] = []
    lines.append("# E2 — Critic Axes Evaluation")
    lines.append("")
    lines.append(
        f"**N**: {total} successful fixtures "
        f"(defect: {sum(1 for r in rows if r['labeled_axis'] != 'clean')}, "
        f"clean: {by_axis.get('clean', 0)}); "
        f"{len(error_rows)} fixtures errored."
    )
    lines.append("")
    lines.append("**Bundle template**: indpensim (P. chrysogenum, penicillin_fedbatch)")
    lines.append("**Pipeline model**: gemini-3.1-pro-preview (all agents)")
    lines.append("**Memory backend**: hermetic StubBackend (per-fixture)")
    lines.append("")
    lines.append("## Headline metrics")
    lines.append("")
    lines.append(
        f"- **Catch rate** (any axis fired on defect fixture): "
        f"**{catch['catch_rate']:.0%}** ({catch['n_caught']}/{catch['n_defect']})"
    )
    lines.append(
        f"- **False positive rate** (any axis fired on clean fixture): "
        f"**{catch['false_positive_rate']:.0%}** "
        f"({catch['n_false_positive']}/{catch['n_clean']})"
    )
    lines.append(
        f"- **Tag accuracy** (labeled axis among fired axes | caught): "
        f"**{tag['tag_accuracy']:.0%}** ({tag['n_correct_tag']}/{tag['n_caught']})"
    )
    lines.append(
        f"- **Strict per-axis recall** "
        f"(catch_rate * tag_accuracy): "
        f"**{catch['catch_rate'] * tag['tag_accuracy']:.0%}**"
    )
    lines.append("")
    lines.append("## Per-axis precision/recall")
    lines.append("")
    pr_rows = []
    for axis in CRITIC_AXES:
        pa = per_axis[axis]
        pr_rows.append(
            [
                axis,
                pa["labeled_count"],
                pa["tp"],
                pa["fp"],
                pa["fn"],
                f"{pa['precision']:.2f}",
                f"{pa['recall']:.2f}",
            ]
        )
    lines.append(
        _md_table(
            ["axis", "n_labeled", "tp", "fp", "fn", "precision", "recall"],
            pr_rows,
        )
    )
    lines.append("")
    lines.append("## Confusion matrix")
    lines.append("")
    lines.append("Rows = labeled axis; columns = fired axis (`none` = critic green-flagged).")
    lines.append("")
    cm_headers = ["labeled"] + CRITIC_AXES + ["none"]
    cm_rows = []
    for labeled in ["clean"] + CRITIC_AXES:
        row = [labeled]
        for fired in CRITIC_AXES + ["none"]:
            row.append(matrix.get(labeled, {}).get(fired, 0))
        cm_rows.append(row)
    lines.append(_md_table(cm_headers, cm_rows))
    lines.append("")
    lines.append("## Catch rate by difficulty")
    lines.append("")
    diff_rows = []
    for diff in ("clear", "borderline"):
        sub = [r for r in rows if r["difficulty"] == diff and r["labeled_axis"] != "clean"]
        if sub:
            caught = sum(1 for r in sub if r["fired_axes"])
            diff_rows.append([diff, len(sub), caught, f"{caught / len(sub):.0%}"])
    lines.append(_md_table(["difficulty", "n", "caught", "catch_rate"], diff_rows))
    lines.append("")
    lines.append("## Errors")
    lines.append("")
    if error_rows:
        for e in error_rows[:10]:
            err = (e.get("error") or "")[:200]
            lines.append(f"- `{e['trial_id']}`: {err}")
        if len(error_rows) > 10:
            lines.append(f"- ...and {len(error_rows) - 10} more")
    else:
        lines.append("None.")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Axis tag enforcement: critic prompt was tightened on 2026-05-18 "
        "to require an `[axis-name]` prefix on every red-flag reason "
        "(including `[general-axis]` as fallback)."
    )
    lines.append(
        "- Tag-accuracy < 100% on caught defects reflects axis-overlap: "
        "a fixture that planted trajectory-axis can correctly fire "
        "`[question-axis]` when the leading question itself demands "
        "dynamics."
    )
    lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines))
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    render()
