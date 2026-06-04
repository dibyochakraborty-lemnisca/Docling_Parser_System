"use client";

// Debate stream — renders the hypothesis-stage event log as a conversation
// between the specialist agents. Each contribution is a dialog bubble next
// to the speaker's colored avatar; per-agent token spend accumulates on the
// left rail. Synthesis lands as a neutral card, verdicts as status banners,
// and orchestration events as quiet centered breadcrumbs.

import { agentFor, type AgentIdentity } from "@/lib/agents";

interface Event {
  type: string;
  ts: string;
  turn: number;
  [k: string]: any;
}

function timeOf(ts: string): string {
  return new Date(ts).toLocaleTimeString();
}

// ---- avatar + token rail -------------------------------------------------

function Avatar({ agent, size = 36 }: { agent: AgentIdentity; size?: number }) {
  return (
    <span
      className="flex shrink-0 items-center justify-center rounded-full font-ui font-semibold"
      style={{
        width: size,
        height: size,
        fontSize: size * 0.34,
        color: agent.color,
        backgroundColor: agent.color + "1f", // ~12% tint
        border: `1px solid ${agent.color}66`,
        boxShadow: `0 0 14px ${agent.color}33`,
      }}
      title={agent.label}
      aria-hidden="true"
    >
      {agent.short}
    </span>
  );
}

function Rail({ agent, tokens }: { agent: AgentIdentity; tokens: number }) {
  return (
    <div className="flex w-16 shrink-0 flex-col items-center gap-1">
      <Avatar agent={agent} />
      {tokens > 0 && (
        <span className="font-ui text-[10px] tabular-nums text-ink-faint" title="tokens spent by this agent so far">
          {tokens >= 1000 ? `${(tokens / 1000).toFixed(1)}k` : tokens}
        </span>
      )}
    </div>
  );
}

// ---- per-event body ------------------------------------------------------

function eventBody(ev: Event): React.ReactNode {
  switch (ev.type) {
    case "facet_contributed":
    case "hypothesis_synthesized":
    case "topic_selected":
      return ev.summary ? <span className="whitespace-pre-wrap">{ev.summary}</span> : null;
    case "critique_filed":
      return ev.reasons?.length > 0 ? (
        <ul className="list-disc space-y-1 pl-4">
          {ev.reasons.map((r: string, i: number) => (
            <li key={i} className="whitespace-pre-wrap">{r}</li>
          ))}
        </ul>
      ) : null;
    case "judge_ruling":
      return ev.rationale ? <span className="whitespace-pre-wrap">{ev.rationale}</span> : null;
    case "question_added":
      return ev.question ? <span>{ev.question}</span> : null;
    case "question_resolved":
      return ev.resolution ? <span>{ev.resolution}</span> : null;
    default:
      return null;
  }
}

// ---- row renderers -------------------------------------------------------

function Bubble({
  agent,
  tokens,
  title,
  body,
  ts,
  tone,
}: {
  agent: AgentIdentity;
  tokens: number;
  title: string;
  body: React.ReactNode;
  ts: string;
  tone?: string; // override accent (e.g. critic red, judge neutral)
}) {
  const c = tone ?? agent.color;
  return (
    <li className="flex items-start gap-3">
      <Rail agent={agent} tokens={tokens} />
      <div className="relative min-w-0 flex-1">
        {/* dialog tail pointing back at the avatar */}
        <span
          className="absolute -left-[7px] top-4 h-3 w-3 rotate-45 border-b border-l"
          style={{ backgroundColor: c + "14", borderColor: c + "55" }}
          aria-hidden="true"
        />
        <div
          className="rounded-xl border px-4 py-3"
          style={{ backgroundColor: c + "14", borderColor: c + "44" }}
        >
          <div className="flex items-center justify-between gap-3">
            <span className="font-ui text-ui-xs uppercase tracking-[0.1em]" style={{ color: c }}>
              {agent.label}
              <span className="ml-2 normal-case tracking-normal text-ink-faint">{title}</span>
            </span>
            <span className="font-ui text-ui-xs tabular-nums text-ink-faint">{timeOf(ts)}</span>
          </div>
          {body && <div className="mt-2 text-sm text-ink-secondary">{body}</div>}
        </div>
      </div>
    </li>
  );
}

function SynthesisRow({
  agent,
  tokens,
  ev,
}: {
  agent: AgentIdentity;
  tokens: number;
  ev: Event;
}) {
  return (
    <li className="flex items-start gap-3">
      <Rail agent={agent} tokens={tokens} />
      <div className="min-w-0 flex-1 rounded-xl border border-rule bg-surface-2 px-4 py-3">
        <div className="flex items-center justify-between gap-3">
          <span className="font-ui text-ui-xs uppercase tracking-[0.1em] text-ink-muted">
            Synthesized
            <span className="ml-2 font-mono normal-case tracking-normal text-ink-faint">{ev.hyp_id}</span>
          </span>
          <span className="font-ui text-ui-xs tabular-nums text-ink-faint">{timeOf(ev.ts)}</span>
        </div>
        {ev.summary && (
          <p className="mt-2 whitespace-pre-wrap text-sm text-ink-secondary">{ev.summary}</p>
        )}
      </div>
    </li>
  );
}

function VerdictRow({ ev }: { ev: Event }) {
  const accepted = ev.type === "hypothesis_accepted";
  const color = accepted ? "var(--color-ok)" : "var(--color-error)";
  return (
    <li className="flex justify-center py-1">
      <span
        className="inline-flex items-center gap-2 rounded-full border px-3 py-1 font-ui text-ui-xs uppercase tracking-[0.12em]"
        style={{ color, borderColor: "currentColor" }}
      >
        <span aria-hidden="true">{accepted ? "✓" : "✕"}</span>
        {accepted ? "Accepted" : "Rejected"}
        <span className="font-mono normal-case tracking-normal opacity-70">{ev.hyp_id}</span>
      </span>
    </li>
  );
}

function MarkerRow({ ev }: { ev: Event }) {
  const text =
    ev.type === "topic_selected"
      ? `Topic ${ev.topic_id}`
      : ev.type === "stage_started"
      ? "Debate started"
      : ev.type === "stage_paused"
      ? "Paused"
      : ev.type === "stage_exited"
      ? `Exited — ${ev.reason}`
      : ev.type === "question_added"
      ? `Open question ${ev.qid}`
      : ev.type === "question_resolved"
      ? `Resolved ${ev.qid}`
      : ev.type === "human_input_received"
      ? `Human input (${ev.input_type})`
      : ev.type.replace(/_/g, " ");
  const body = eventBody(ev);
  return (
    <li className="py-1">
      <div className="flex items-center gap-3">
        <span className="h-px flex-1 bg-rule" />
        <span className="font-ui text-ui-xs uppercase tracking-[0.14em] text-ink-faint">{text}</span>
        <span className="h-px flex-1 bg-rule" />
      </div>
      {body && (
        <p className="mx-auto mt-2 max-w-2xl text-center text-xs text-ink-muted">{body}</p>
      )}
    </li>
  );
}

// ---- timeline ------------------------------------------------------------

export function Timeline({ events }: { events: Event[] }) {
  if (events.length === 0) {
    return (
      <p className="text-sm text-ink-muted">
        Waiting for events… (the debate runs in the background; events stream
        in as they happen)
      </p>
    );
  }

  // Accumulate per-agent token spend across the stream. tokens_used events
  // carry the spend; we fold them into a running tally instead of rendering
  // them as their own rows, then snapshot the tally onto each visible row.
  const tally: Record<string, number> = {};

  return (
    <ol className="space-y-4">
      {events.map((ev, i) => {
        if (ev.type === "tokens_used") {
          const key = ev.agent ?? "system";
          tally[key] = (tally[key] ?? 0) + (ev.input ?? 0) + (ev.output ?? 0);
          return null; // folded into the rail, no standalone row
        }

        switch (ev.type) {
          case "facet_contributed": {
            const agent = agentFor(ev.specialist);
            return (
              <Bubble
                key={i}
                agent={agent}
                tokens={tally[agent.key] ?? 0}
                title="facet"
                body={eventBody(ev)}
                ts={ev.ts}
              />
            );
          }
          case "critique_filed": {
            const agent = agentFor("critic");
            return (
              <Bubble
                key={i}
                agent={agent}
                tokens={tally.critic ?? 0}
                title={`flag ${String(ev.flag ?? "").toUpperCase()} · ${ev.hyp_id}`}
                body={eventBody(ev)}
                ts={ev.ts}
              />
            );
          }
          case "judge_ruling": {
            const agent = agentFor("judge");
            return (
              <Bubble
                key={i}
                agent={agent}
                tokens={tally.judge ?? 0}
                title={`criticism ${ev.criticism_valid ? "valid" : "invalid"}`}
                body={eventBody(ev)}
                ts={ev.ts}
                tone={ev.criticism_valid ? "var(--color-warn)" : agent.color}
              />
            );
          }
          case "hypothesis_synthesized":
            return (
              <SynthesisRow
                key={i}
                agent={agentFor("synthesizer")}
                tokens={tally.synthesizer ?? 0}
                ev={ev}
              />
            );
          case "hypothesis_accepted":
          case "hypothesis_rejected":
            return <VerdictRow key={i} ev={ev} />;
          default:
            return <MarkerRow key={i} ev={ev} />;
        }
      })}
    </ol>
  );
}
