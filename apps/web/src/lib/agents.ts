// Agent identities for the debate stream. Each specialist / role gets a
// stable color + two-letter monogram so the runs-page timeline reads like
// a conversation between distinct participants. Keys match the backend:
// specialists from runner.py (kinetics / mass_transfer / metabolic) and the
// synthesizer / critic / judge roles.

export type AgentRole =
  | "specialist"
  | "synthesizer"
  | "critic"
  | "judge"
  | "system";

export interface AgentIdentity {
  key: string;
  label: string;
  /** Two-letter monogram for the avatar. */
  short: string;
  /** Signature hex — used for the avatar fill + bubble accent. */
  color: string;
  role: AgentRole;
}

const AGENTS: Record<string, AgentIdentity> = {
  mass_transfer: { key: "mass_transfer", label: "Mass Transfer", short: "MT", color: "#38afd8", role: "specialist" },
  kinetics: { key: "kinetics", label: "Kinetics", short: "KN", color: "#e3a552", role: "specialist" },
  metabolic: { key: "metabolic", label: "Metabolic", short: "MB", color: "#3fbfa6", role: "specialist" },
  synthesizer: { key: "synthesizer", label: "Synthesizer", short: "SY", color: "#8b9bf4", role: "synthesizer" },
  critic: { key: "critic", label: "Critic", short: "CR", color: "#e5484d", role: "critic" },
  judge: { key: "judge", label: "Judge", short: "JG", color: "#c7cbce", role: "judge" },
};

/** Resolve an agent key to its identity, falling back to a neutral system
 *  participant for orchestrator / unknown keys. */
export function agentFor(key?: string | null): AgentIdentity {
  if (key && AGENTS[key]) return AGENTS[key];
  const k = (key ?? "system").trim();
  return {
    key: k || "system",
    label: k ? k.replace(/_/g, " ") : "System",
    short: (k || "SY").replace(/[^a-zA-Z]/g, "").slice(0, 2).toUpperCase() || "SY",
    color: "#92989c",
    role: "system",
  };
}
