"""Render E3 results JSONL into eval/e3_case_studies.md."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from fermdocs_eval.harness import read_jsonl

RESULTS = Path("eval/results/e3.jsonl")
OUT_MD = Path("eval/e3_case_studies.md")


def _md_table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def render() -> None:
    rows = [r for r in read_jsonl(RESULTS) if r.get("status") == "ok"]
    if not rows:
        OUT_MD.write_text("# E3 — Case Studies\n\nNo results yet.\n")
        return

    # Group judge rows by bundle.
    judges_by_bundle: dict[str, list[dict]] = {}
    for r in rows:
        tid = r["trial_id"]
        if "-judge-" in tid:
            bundle = tid.split("-judge-")[0]
            judges_by_bundle.setdefault(bundle, []).append(r["payload"])

    lines: list[str] = []
    lines.append("# E3 — Case Studies vs Single-Shot Baseline")
    lines.append("")
    lines.append("**N bundles**: 2 (yeast, indpensim). **Baseline model**: gemini-3.1-pro-preview.")
    lines.append("**Judge**: gemini-3-pro (different from baseline to reduce same-model bias).")
    lines.append("**Order**: counterbalanced — treatment is A on even seeds, B on odd.")
    lines.append("")
    lines.append("## Per-bundle judge votes (3 seeds)")
    lines.append("")
    table_rows = []
    for bundle, judges in judges_by_bundle.items():
        votes = Counter(j.get("treatment_won") for j in judges)
        table_rows.append([
            bundle,
            votes.get("treatment", 0),
            votes.get("baseline", 0),
            votes.get("tie", 0),
            len(judges),
        ])
    lines.append(_md_table(
        ["bundle", "treatment_wins", "baseline_wins", "ties", "n_judges"],
        table_rows,
    ))
    lines.append("")
    lines.append("## Judge rationales")
    lines.append("")
    for bundle, judges in judges_by_bundle.items():
        lines.append(f"### {bundle}")
        for j in judges:
            seed = j.get("seed", "?")
            won = j.get("treatment_won", "?")
            rat = (j.get("rationale") or "").strip()
            lines.append(f"- **seed {seed}** (treatment_won={won}): {rat}")
        lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- N=2 bundles is a case study, not a benchmark. We report per-bundle "
        "outcomes and judge agreement, NOT an aggregate preference rate."
    )
    lines.append(
        "- Counterbalanced order (treatment as A vs B alternates) controls "
        "for position bias in LLM judges."
    )
    lines.append(
        "- LLM-as-judge has known length and same-model biases. Mitigations: "
        "different judge model than treatment/baseline, structured JSON output, "
        "and the rationales are inspectable in the appendix."
    )
    lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines))
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    render()
