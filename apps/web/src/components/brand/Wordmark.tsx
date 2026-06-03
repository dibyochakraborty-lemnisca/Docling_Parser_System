// Lemnisca-style wordmark: lowercase grotesk + the teal lemniscate (∞)
// mark as the brand signal. Server-safe (no client hooks).

export function Wordmark({ stage }: { stage?: string }) {
  return (
    <span className="inline-flex items-baseline gap-2">
      <span className="font-display text-accent text-lg leading-none" aria-hidden="true">
        ∞
      </span>
      <span className="text-base font-normal tracking-tight text-ink">fermdocs</span>
      {stage && (
        <span className="kicker ml-1 hidden sm:inline" style={{ letterSpacing: "0.16em" }}>
          {stage}
        </span>
      )}
    </span>
  );
}
