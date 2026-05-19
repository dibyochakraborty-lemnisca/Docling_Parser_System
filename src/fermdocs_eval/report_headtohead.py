"""Render head-to-head results JSONL into eval/headtohead_report.md.

Pure I/O over the JSONL produced by suites/headtohead.py. All math
lives in metrics.py.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from fermdocs_eval.harness import read_jsonl
from fermdocs_eval.judges import JUDGE_AXES
from fermdocs_eval.metrics import (
    bootstrap_ci,
    per_axis_delta,
    per_axis_means,
    preference_rate,
)

RESULTS = Path("eval/results/headtohead.jsonl")
OUT_MD = Path("eval/headtohead_report.md")


def _md_table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _judges_only(rows: list[dict]) -> list[dict]:
    return [
        r.get("payload") or {}
        for r in rows
        if r.get("status") == "ok" and r.get("trial_id", "").rfind("-judge-") != -1
    ]


def render() -> None:
    raw = read_jsonl(RESULTS)
    if not raw:
        OUT_MD.parent.mkdir(parents=True, exist_ok=True)
        OUT_MD.write_text("# Head-to-head — Agent vs Single-Shot\n\nNo results yet.\n")
        return

    judges = _judges_only(raw)
    errors = [r for r in raw if r.get("status") == "error"]

    # Per-question grouping for the per-question table.
    by_q: dict[str, list[dict]] = {}
    for j in judges:
        qid = j.get("qid", "?")
        by_q.setdefault(qid, []).append(j)

    lines: list[str] = []
    lines.append("# Head-to-head — fermdocs agent vs single-shot Gemini")
    lines.append("")
    lines.append(
        "**N questions**: {n_q}. **N judge rows**: {n_j}. **Errors**: {n_err}.".format(
            n_q=len(by_q), n_j=len(judges), n_err=len(errors)
        )
    )
    lines.append("**Bundle**: indpensim (P. chrysogenum, industrial fed-batch).")
    lines.append("**Treatment**: full fermdocs pipeline (synthesizer + 3 specialists + critic + judge, all gemini-3.1-pro-preview).")
    lines.append("**Baseline**: single gemini-3.1-pro-preview call with bundle JSON + question.")
    lines.append("**Judge**: gemini-3.1-pro-preview (separate call), 3 seeds, counterbalanced A/B.")
    lines.append("")

    # === Headline preference ===
    pref = preference_rate(judges, treatment="A")
    lo, hi = bootstrap_ci(judges, treatment="A", n_resamples=2000, seed=0)
    lines.append("## Headline: preference rate")
    lines.append("")
    lines.append(
        f"- **Treatment win rate**: **{pref['rate']:.0%}** "
        f"({pref['treatment_wins']}/{pref['n']}) — 95% bootstrap CI [{lo:.0%}, {hi:.0%}]"
    )
    lines.append(
        f"- Baseline wins: {pref['baseline_wins']} | ties: {pref['ties']}"
    )
    lines.append("")

    # === Per-axis means ===
    t_means = per_axis_means(judges, axes=JUDGE_AXES, role="treatment")
    b_means = per_axis_means(judges, axes=JUDGE_AXES, role="baseline")
    deltas = per_axis_delta(judges, axes=JUDGE_AXES)

    lines.append("## Per-axis score means (1-10)")
    lines.append("")
    rows = []
    for axis in JUDGE_AXES:
        t = t_means[axis]
        b = b_means[axis]
        d = deltas[axis]
        rows.append([
            axis,
            f"{t['mean']:.2f} ± {t['stdev']:.2f}",
            f"{b['mean']:.2f} ± {b['stdev']:.2f}",
            f"{d['mean_delta']:+.2f}",
            f"{d['wins']}W / {d['losses']}L / {d['ties']}T",
        ])
    lines.append(_md_table(
        ["axis", "treatment", "baseline", "Δ (T−B)", "per-judge wins/losses/ties"],
        rows,
    ))
    lines.append("")

    # === Per-question table ===
    lines.append("## Per-question winners")
    lines.append("")
    pq_rows = []
    for qid in sorted(by_q.keys()):
        votes = Counter(j.get("treatment_won") for j in by_q[qid])
        pq_rows.append([
            qid,
            votes.get("treatment", 0),
            votes.get("baseline", 0),
            votes.get("tie", 0),
            len(by_q[qid]),
        ])
    lines.append(_md_table(
        ["qid", "treatment", "baseline", "tie", "n_judges"],
        pq_rows,
    ))
    lines.append("")

    # === Judge rationales (per question, all seeds) ===
    lines.append("## Judge rationales")
    lines.append("")
    for qid in sorted(by_q.keys()):
        lines.append(f"### {qid}")
        for j in by_q[qid]:
            seed = j.get("seed", "?")
            won = j.get("treatment_won", "?")
            r = (j.get("rationale") or "").strip()
            lines.append(f"- **seed {seed}** ({won}): {r}")
        lines.append("")

    # === Errors ===
    if errors:
        lines.append("## Errors")
        lines.append("")
        for e in errors[:20]:
            err = (e.get("error") or "")[:200]
            lines.append(f"- `{e.get('trial_id', '?')}`: {err}")
        if len(errors) > 20:
            lines.append(f"- ...and {len(errors) - 20} more")
        lines.append("")

    lines.append("## Notes / limitations")
    lines.append("")
    lines.append(
        "- LLM-as-judge: same model family judges both outputs. Mitigated by"
        " counterbalanced A/B order (treatment is A on even seeds, B on odd)."
    )
    lines.append(
        "- N=10 questions, all on a single bundle (indpensim). This is a"
        " case-study eval, not a benchmark. We do not claim cross-bundle"
        " generalization."
    )
    lines.append(
        "- Memory was held off for this eval (hermetic StubBackend per"
        " question). Memory-specific evals are out of scope for this report."
    )
    lines.append(
        "- Bootstrap CIs assume i.i.d. trials; with 3 judge seeds per question"
        " the same question's judges are correlated, so the reported CI is a"
        " lower bound on the true variance."
    )

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines))
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    render()
