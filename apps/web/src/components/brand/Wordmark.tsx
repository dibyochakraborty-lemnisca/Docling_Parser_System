// Lemnisca-style wordmark: a prominent teal lemniscate (∞) mark + the
// "Lemnisca FASSO" name in grotesk. Server-safe (no client hooks).

export function Wordmark({ stage }: { stage?: string }) {
  return (
    <span className="inline-flex items-baseline gap-2.5">
      <span
        className="font-display text-3xl leading-none text-accent"
        style={{ textShadow: "0 0 14px var(--color-accent-glow)" }}
        aria-hidden="true"
      >
        ∞
      </span>
      <span className="text-xl font-semibold tracking-tight text-ink">
        Lemnisca <span className="text-accent">FASSO</span>
      </span>
      {stage && (
        <span className="kicker ml-1 hidden sm:inline" style={{ letterSpacing: "0.16em" }}>
          {stage}
        </span>
      )}
    </span>
  );
}
