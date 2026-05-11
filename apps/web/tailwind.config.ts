import type { Config } from "tailwindcss";

/**
 * Editorial Scientific Tailwind theme (Phase 1 of frontend-redesign).
 * Plan ref: plans/2026-05-11-frontend-redesign-editorial.md
 *
 * Two token families exposed:
 *   1. Editorial-native tokens (paper, ink, rule, accent, ...) read
 *      directly from the CSS variables in globals.css. These are what
 *      new components should use.
 *   2. Shadcn-compatibility aliases (background, foreground, primary,
 *      ...) for components that haven't been re-treated yet. They map
 *      to the same underlying CSS variables but via Tailwind's
 *      hsl(var(--token)) idiom. Will be pruned as components are
 *      migrated in Phase 2.
 *
 * Dark mode removed — light-only per the locked design decision.
 */
const config: Config = {
  // No darkMode array — `dark:` variants are now compiled out.
  content: [
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      // ---------- Editorial tokens ----------
      colors: {
        // Paper and ink
        paper: "var(--color-paper)",
        "paper-elevated": "var(--color-paper-elevated)",
        ink: "var(--color-ink)",
        "ink-secondary": "var(--color-ink-secondary)",
        "ink-muted": "var(--color-ink-muted)",
        rule: "var(--color-rule)",
        "rule-strong": "var(--color-rule-strong)",

        // The accent
        accent: {
          DEFAULT: "var(--color-accent)",
          soft: "var(--color-accent-soft)",
          ink: "var(--color-accent-ink)",
        },

        // Functional / semantic
        warn: "var(--color-warn)",
        error: "var(--color-error)",

        // ---------- Shadcn-compat aliases (deprecated; prune in Phase 2) ----------
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
      },

      // ---------- Editorial typography ----------
      fontFamily: {
        display: ["var(--font-display)", "Georgia", "serif"],
        body: ["var(--font-body)", "Georgia", "serif"],
        ui: ["var(--font-ui)", "system-ui", "sans-serif"],
      },

      // Type scale tuned for editorial reading. Generous line-heights;
      // sizes step up in golden-ratio-ish increments.
      fontSize: {
        // UI / chrome / metadata
        "ui-xs": ["0.6875rem", { lineHeight: "1.25", letterSpacing: "0.04em" }],   // 11px
        "ui-sm": ["0.8125rem", { lineHeight: "1.35" }],                            // 13px
        "ui-base": ["0.9375rem", { lineHeight: "1.5" }],                           // 15px
        // Body prose
        "body": ["1.125rem", { lineHeight: "1.65" }],                              // 18px
        "body-lg": ["1.25rem", { lineHeight: "1.6" }],                             // 20px
        // Display
        "display-sm": ["1.75rem", { lineHeight: "1.2", letterSpacing: "-0.01em" }],  // 28px
        "display-md": ["2.5rem", { lineHeight: "1.15", letterSpacing: "-0.012em" }], // 40px
        "display-lg": ["3.5rem", { lineHeight: "1.05", letterSpacing: "-0.015em" }], // 56px
        "display-xl": ["4.5rem", { lineHeight: "1.0", letterSpacing: "-0.02em" }],   // 72px
      },

      // ---------- Editorial spacing ----------
      // The "asymmetric two-column grid" lives here. Body col 680px,
      // sidebar 240px, gap 56px. Phase 2 layouts read these.
      maxWidth: {
        "content": "1100px",
        "prose": "680px",
        "sidebar": "240px",
      },
      gridTemplateColumns: {
        "editorial": "minmax(0, 680px) 240px",
      },
      gap: {
        "column": "3.5rem",  // 56px between body and sidebar
      },

      borderRadius: {
        // Tighter than shadcn default — editorial designs are squared.
        lg: "var(--radius)",
        md: "calc(var(--radius) - 1px)",
        sm: "calc(var(--radius) - 2px)",
      },
    },
  },
  plugins: [],
};

export default config;
