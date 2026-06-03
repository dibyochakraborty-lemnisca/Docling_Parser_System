import type { Metadata } from "next";
import { Newsreader, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { ScrollProgress } from "@/components/brand/ScrollProgress";
import { Wordmark } from "@/components/brand/Wordmark";

// LEMNISCA type stack.
//   Body + headings: Helvetica Neue (system grotesk, set in globals.css).
//   Newsreader (--font-display): sparing serif italic accents.
//   JetBrains Mono (--font-ui): every label, eyebrow, tag, date, stat.
// `display: "swap"` so first paint isn't blocked; latin subset only.

const newsreader = Newsreader({
  subsets: ["latin"],
  display: "swap",
  style: ["italic", "normal"],
  variable: "--font-display",
  // Newsreader has no Next metric-override table; skip the size-adjust
  // fallback to avoid the build warning (it's a sparing accent face).
  adjustFontFallback: false,
});

const jetbrainsMono = JetBrains_Mono({
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
    <html lang="en" className={`${newsreader.variable} ${jetbrainsMono.variable}`}>
      <body className="min-h-screen bg-bg text-ink antialiased">
        <ScrollProgress />
        <header className="sticky top-0 z-50 border-b border-rule-soft bg-bg/55 backdrop-blur-[14px] print:hidden">
          <div className="mx-auto flex max-w-content items-center justify-between gap-5 px-8 py-4">
            <a href="/" aria-label="fermdocs home" className="transition-opacity hover:opacity-90">
              <Wordmark stage="hypothesis stage" />
            </a>
            <nav className="font-ui text-ui-xs uppercase tracking-[0.12em] text-ink-muted">
              <a className="relative py-1 transition-colors hover:text-ink" href="/">
                Runs
              </a>
            </nav>
          </div>
        </header>
        <main className="mx-auto max-w-content px-8 py-12">{children}</main>
        <footer className="mt-16 border-t border-rule-soft px-8 py-10 text-center print:hidden">
          <p className="kicker">fermdocs · fermentation hypothesis engine</p>
        </footer>
      </body>
    </html>
  );
}
