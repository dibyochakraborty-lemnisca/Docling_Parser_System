"use client";

import { Badge } from "@/components/ui/badge";

interface Event {
  type: string;
  ts: string;
  turn: number;
  [k: string]: any;
}

const TYPE_COLOR: Record<string, "default" | "secondary" | "success" | "warning" | "destructive"> = {
  stage_started: "secondary",
  topic_selected: "secondary",
  facet_contributed: "default",
  hypothesis_synthesized: "default",
  critique_filed: "warning",
  judge_ruling: "warning",
  hypothesis_accepted: "success",
  hypothesis_rejected: "destructive",
  question_added: "secondary",
  question_resolved: "success",
  human_input_received: "secondary",
  stage_paused: "warning",
  stage_exited: "secondary",
  tokens_used: "outline" as any,
};

function eventLabel(ev: Event): string {
  switch (ev.type) {
    case "stage_started":
      return "Stage started";
    case "topic_selected":
      return `Topic ${ev.topic_id}`;
    case "facet_contributed":
      return `Facet (${ev.specialist})`;
    case "hypothesis_synthesized":
      return `Synthesized ${ev.hyp_id}`;
    case "critique_filed":
      return `Critique ${ev.flag.toUpperCase()} on ${ev.hyp_id}`;
    case "judge_ruling":
      return `Judge: criticism ${ev.criticism_valid ? "valid" : "invalid"}`;
    case "hypothesis_accepted":
      return `Accepted ${ev.hyp_id}`;
    case "hypothesis_rejected":
      return `Rejected ${ev.hyp_id}`;
    case "question_added":
      return `Question ${ev.qid}`;
    case "question_resolved":
      return `Resolved ${ev.qid}`;
    case "human_input_received":
      return `Human input (${ev.input_type})`;
    case "stage_paused":
      return "Stage paused";
    case "stage_exited":
      return `Stage exited (${ev.reason})`;
    case "tokens_used":
      return `Tokens (${ev.agent})`;
    default:
      return ev.type;
  }
}

function eventBody(ev: Event): React.ReactNode {
  switch (ev.type) {
    case "topic_selected":
      // Full text — operators need to read the topic to evaluate the
      // debate. Truncating here was hiding agent reasoning. Cards
      // grow vertically instead; that's the right trade-off for a
      // research / debug UI. Use whitespace-pre-wrap so newlines and
      // long lines render readably.
      return (
        ev.summary && (
          <span className="whitespace-pre-wrap">{ev.summary}</span>
        )
      );
    case "facet_contributed":
    case "hypothesis_synthesized":
      return (
        ev.summary && (
          <span className="whitespace-pre-wrap">{ev.summary}</span>
        )
      );
    case "critique_filed":
      return ev.reasons?.length > 0 ? (
        <ul className="list-disc pl-4 space-y-1">
          {ev.reasons.map((r: string, i: number) => (
            <li key={i} className="whitespace-pre-wrap">
              {r}
            </li>
          ))}
        </ul>
      ) : null;
    case "judge_ruling":
      return (
        ev.rationale && (
          <span className="whitespace-pre-wrap">{ev.rationale}</span>
        )
      );
    case "question_added":
      return ev.question && <span>{ev.question}</span>;
    case "question_resolved":
      return ev.resolution && <span>{ev.resolution}</span>;
    case "tokens_used":
      return (
        <span className="text-xs text-muted-foreground font-mono">
          {ev.input.toLocaleString()} in / {ev.output.toLocaleString()} out
        </span>
      );
    default:
      return null;
  }
}

export function Timeline({ events }: { events: Event[] }) {
  if (events.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        Waiting for events… (the debate runs in the background; events stream
        in as they happen)
      </p>
    );
  }
  return (
    <ol className="space-y-3 border-l pl-6 ml-2">
      {events.map((ev, i) => (
        <li key={i} className="relative">
          <span className="absolute -left-[31px] top-2 h-2 w-2 rounded-full bg-foreground" />
          <div className="rounded-md border bg-card px-4 py-3">
            <div className="flex items-center justify-between gap-4">
              <div className="flex items-center gap-2">
                <Badge variant={TYPE_COLOR[ev.type] ?? "outline"}>
                  {eventLabel(ev)}
                </Badge>
                <span className="text-xs text-muted-foreground">turn {ev.turn}</span>
              </div>
              <span className="text-xs text-muted-foreground tabular-nums">
                {new Date(ev.ts).toLocaleTimeString()}
              </span>
            </div>
            <div className="mt-2 text-sm">{eventBody(ev)}</div>
          </div>
        </li>
      ))}
    </ol>
  );
}
