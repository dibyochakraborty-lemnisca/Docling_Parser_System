import type { Metadata } from "next";
import { Fraunces, Hanken_Grotesk } from "next/font/google";
import "./globals.css";

// Editorial Scientific type stack (Phase 1 of frontend-redesign).
// Plan ref: plans/2026-05-11-frontend-redesign-editorial.md
//
// Fraunces is a variable-axis serif: handles display headlines AND body
// text via its optical-size axis. One family doing two jobs is more
// refined than three loosely-related fonts. Used by Stripe Press, The
// Marshall Project.
//
// Hanken Grotesk is the open-source counterpart to Söhne — quiet UI
// chrome that doesn't compete with the serif. Variable weight.
//
// `display: "swap"` so first paint isn't blocked. Both are subset to
// latin so we don't ship glyph ranges we don't use.

const fraunces = Fraunces({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-display",
  // Fraunces' axes:
  //   wght: weight (300-900)
  //   opsz: optical-size (9-144) — bigger sizes get more contrast
  //   SOFT: softness (0-100) — 0 is austere, 100 is warm
  axes: ["opsz", "SOFT"],
});

const hanken = Hanken_Grotesk({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-ui",
});

export const metadata: Metadata = {
  title: "fermdocs — hypothesis stage",
  description: "Multi-agent fermentation-hypothesis debate viewer",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${fraunces.variable} ${hanken.variable}`}>
      <body className="min-h-screen bg-paper text-ink antialiased">
        <header className="border-b border-rule">
          <div className="mx-auto max-w-[1100px] px-8 py-5 flex items-center justify-between">
            <a href="/" className="font-display text-lg font-semibold tracking-tight">
              fermdocs
              <span className="ml-3 font-ui text-sm font-normal text-ink-muted">
                hypothesis stage
              </span>
            </a>
            <nav className="font-ui text-sm text-ink-secondary">
              <a className="hover:text-accent transition-colors" href="/">
                Runs
              </a>
            </nav>
          </div>
        </header>
        <main className="mx-auto max-w-[1100px] px-8 py-12">{children}</main>
      </body>
    </html>
  );
}
