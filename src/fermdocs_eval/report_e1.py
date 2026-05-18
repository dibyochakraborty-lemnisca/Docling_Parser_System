"""Render E1 results JSONL into eval/e1_memory_mechanism.md."""

from __future__ import annotations

from pathlib import Path

from fermdocs_eval.harness import read_jsonl

RESULTS = Path("eval/results/e1.jsonl")
OUT_MD = Path("eval/e1_memory_mechanism.md")


def _by_bundle(rows: list[dict]) -> dict[str, dict[str, dict]]:
    """Group rows as bundle_name -> { 'cold': row, 'warm': row }."""
    out: dict[str, dict[str, dict]] = {}
    for r in rows:
        tid = r["trial_id"]
        if tid.endswith("-cold"):
            bundle = tid[: -len("-cold")]
            out.setdefault(bundle, {})["cold"] = r
        elif tid.endswith("-warm"):
            bundle = tid[: -len("-warm")]
            out.setdefault(bundle, {})["warm"] = r
    return out


def _md_table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def render() -> None:
    rows = [r for r in read_jsonl(RESULTS) if r.get("status") == "ok"]
    if not rows:
        OUT_MD.write_text("# E1 — Memory Mechanism\n\nNo results yet.\n")
        return

    grouped = _by_bundle(rows)

    lines: list[str] = []
    lines.append("# E1 — Memory Mechanism Demo (cold/warm same-bundle)")
    lines.append("")
    lines.append(
        f"**N bundles**: {len(grouped)}. **Backend**: hermetic StubBackend "
        f"per bundle. **Model**: gemini-3.1-pro-preview."
    )
    lines.append("")
    lines.append("## Per-bundle comparison")
    lines.append("")
    table_rows = []
    for bundle, pair in grouped.items():
        cold = (pair.get("cold") or {}).get("payload", {})
        warm = (pair.get("warm") or {}).get("payload", {})
        cold_spec = (cold.get("specificity") or {}).get("score", "—")
        warm_spec = (warm.get("specificity") or {}).get("score", "—")
        table_rows.append([
            bundle,
            cold.get("n_critiques", 0),
            warm.get("n_critiques", 0),
            cold.get("n_final_hypotheses", 0),
            warm.get("n_final_hypotheses", 0),
            cold.get("memory_records_post_cold", "—"),
            warm.get("priors_visible_at_warm_start", "—"),
            len(cold.get("fired_axes", [])),
            len(warm.get("fired_axes", [])),
            cold_spec,
            warm_spec,
        ])
    lines.append(_md_table(
        [
            "bundle",
            "cold n_critiques", "warm n_critiques",
            "cold n_hyp", "warm n_hyp",
            "lessons after cold", "priors visible at warm",
            "cold axes fired", "warm axes fired",
            "cold specificity", "warm specificity",
        ],
        table_rows,
    ))
    lines.append("")
    lines.append("## Per-bundle axis deltas")
    lines.append("")
    for bundle, pair in grouped.items():
        cold_axes = set((pair.get("cold") or {}).get("payload", {}).get("fired_axes", []))
        warm_axes = set((pair.get("warm") or {}).get("payload", {}).get("fired_axes", []))
        prevented = cold_axes - warm_axes
        new_warm = warm_axes - cold_axes
        persistent = cold_axes & warm_axes
        lines.append(f"### {bundle}")
        lines.append(f"- prevented (cold fired, warm did not): `{sorted(prevented) or 'none'}`")
        lines.append(f"- persistent (both fired): `{sorted(persistent) or 'none'}`")
        lines.append(f"- new in warm (only warm fired): `{sorted(new_warm) or 'none'}`")
        lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- This is a mechanism demo: we show the memory loop closes "
        "end-to-end on the same bundle. We do NOT claim cross-bundle "
        "generalization."
    )
    lines.append(
        "- Specificity is LLM-judged on a 1–5 scale; the judge prompt is "
        "checked into eval/prompts/."
    )
    lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines))
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    render()
