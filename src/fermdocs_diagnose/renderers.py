"""Markdown renderers for DiagnosisOutput.

Plan ref: plans/2026-05-02-diagnosis-agent.md §8.

Generated on demand. The structured DiagnosisOutput is the contract; markdown
is the human-readable view. Downstream agents read structured fields, not
these strings.

All renderers are pure functions over DiagnosisOutput. No I/O. The CLI is
responsible for writing them to disk.
"""

from __future__ import annotations

from fermdocs_diagnose.schema import (
    AnalysisClaim,
    BaseClaim,
    DiagnosisOutput,
    FailureClaim,
    OpenQuestion,
    TrendClaim,
)

_CITATION_LIMIT = 5  # truncate long citation lists for readability


def render_failures_md(output: DiagnosisOutput) -> str:
    if not output.failures:
        return "# Failures\n\n_No failures emitted._\n"
    lines = ["# Failures", ""]
    for claim in output.failures:
        lines.extend(_failure_block(claim))
        lines.append("")
    return "\n".join(lines)


def render_trends_md(output: DiagnosisOutput) -> str:
    if not output.trends:
        return "# Trends\n\n_No trends emitted._\n"
    lines = ["# Trends", ""]
    for claim in output.trends:
        lines.extend(_trend_block(claim))
        lines.append("")
    return "\n".join(lines)


def render_analysis_md(output: DiagnosisOutput) -> str:
    if not output.analysis:
        return "# Analysis\n\n_No analysis claims emitted._\n"
    lines = ["# Analysis", ""]
    for claim in output.analysis:
        lines.extend(_analysis_block(claim))
        lines.append("")
    return "\n".join(lines)


def render_questions_md(output: DiagnosisOutput) -> str:
    if not output.open_questions:
        return "# Open questions\n\n_No open questions._\n"
    lines = ["# Open questions", ""]
    for q in output.open_questions:
        lines.extend(_question_block(q))
        lines.append("")
    return "\n".join(lines)


def render_diagnosis_md(output: DiagnosisOutput) -> str:
    """Combined report: meta header + narrative + each section."""
    lines = [
        "# Diagnosis Report",
        "",
        f"- diagnosis_id: `{output.meta.diagnosis_id}`",
        f"- supersedes characterization: `{output.meta.supersedes_characterization_id}`",
        f"- generated: {output.meta.generation_timestamp.isoformat()}",
        f"- model: `{output.meta.model}` ({output.meta.provider})",
    ]
    if output.meta.error:
        lines.append(f"- **error:** `{output.meta.error}`")
    lines.append("")

    if output.narrative:
        lines.extend(["## Narrative", "", output.narrative, ""])

    lines.append(render_failures_md(output))
    lines.append(render_trends_md(output))
    lines.append(render_analysis_md(output))
    lines.append(render_questions_md(output))
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------


def _failure_block(claim: FailureClaim) -> list[str]:
    header = f"## {claim.claim_id} [{claim.severity.value}]"
    return [
        header,
        "",
        claim.summary,
        "",
        *_meta_lines(claim),
        f"- time_window: {_format_window(claim.time_window)}",
    ]


def _trend_block(claim: TrendClaim) -> list[str]:
    return [
        f"## {claim.claim_id} [{claim.direction}]",
        "",
        claim.summary,
        "",
        *_meta_lines(claim),
        f"- trajectories: "
        + (
            ", ".join(
                f"({t.run_id}, {t.variable})"
                for t in claim.cited_trajectories[:_CITATION_LIMIT]
            )
            or "—"
        ),
        f"- time_window: {_format_window(claim.time_window)}",
    ]


def _analysis_block(claim: AnalysisClaim) -> list[str]:
    return [
        f"## {claim.claim_id} [{claim.kind}]",
        "",
        claim.summary,
        "",
        *_meta_lines(claim),
    ]


def _question_block(q: OpenQuestion) -> list[str]:
    return [
        f"## {q.question_id}",
        "",
        f"**{q.question}**",
        "",
        f"- why_it_matters: {q.why_it_matters}",
        f"- cites: {_format_citations(q.cited_finding_ids)}",
        f"- answer_format: {q.answer_format_hint}",
        f"- domain_tags: {', '.join(q.domain_tags) if q.domain_tags else '—'}",
    ]


def _meta_lines(claim: BaseClaim) -> list[str]:
    lines = [
        f"- cites: {_format_citations(claim.cited_finding_ids)}",
        f"- variables: {', '.join(claim.affected_variables) or '—'}",
        f"- confidence: {claim.confidence:.2f} ({claim.confidence_basis.value})",
        f"- domain_tags: {', '.join(claim.domain_tags) if claim.domain_tags else '—'}",
    ]
    if claim.provenance_downgraded:
        lines.append(
            "- _provenance_downgraded: process_priors → schema_only under UNKNOWN flag_"
        )
    return lines


def _format_citations(ids: list[str]) -> str:
    if not ids:
        return "—"
    head = ids[:_CITATION_LIMIT]
    suffix = "" if len(ids) <= _CITATION_LIMIT else f" (+{len(ids) - _CITATION_LIMIT} more)"
    return ", ".join(f"`{i}`" for i in head) + suffix


def _format_window(window) -> str:
    if window is None:
        return "—"
    return f"[{window.start}, {window.end}] h"
