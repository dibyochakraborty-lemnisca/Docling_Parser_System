import type { Config } from "tailwindcss";

/**
 * LEMNISCA Tailwind theme (frontend Lemnisca-design-refactor).
 *
 * Two token families:
 *   1. Lemnisca-native tokens (bg, surface-1..3, ink ramp, rule, accent,
 *      ok/warn/error) read direct CSS vars from globals.css. New chrome
 *      uses these.
 *   2. shadcn-compat aliases (background/foreground/primary/…) consumed
 *      as hsl(var(--x) / <alpha-value>) so untouched components inherit
 *      the dark theme AND opacity modifiers (bg-primary/5, …) work.
 *
 * Black instrument panel, one teal signal. Dark-only.
 */
const config: Config = {
  content: [
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // ---------- Lemnisca-native ----------
        bg: "var(--color-bg)",
        "surface-1": "var(--color-surface-1)",
        "surface-2": "var(--color-surface-2)",
        "surface-3": "var(--color-surface-3)",

        // editorial-name aliases (layout/components still use these)
        paper: "var(--color-bg)",
        "paper-elevated": "var(--color-surface-1)",

        ink: {
          DEFAULT: "var(--color-ink)",
          secondary: "var(--color-ink-secondary)",
          muted: "var(--color-ink-muted)",
          faint: "var(--color-faint)",
        },

        rule: {
          DEFAULT: "var(--color-rule)",
          soft: "var(--color-rule-soft)",
          strong: "var(--color-rule-strong)",
        },
        "rule-strong": "var(--color-rule-strong)",

        // The signature teal (alpha-aware DEFAULT for /opacity modifiers)
        accent: {
          DEFAULT: "rgb(var(--accent-rgb) / <alpha-value>)",
          bright: "var(--color-accent-bright)",
          deep: "var(--color-accent-deep)",
          soft: "var(--color-accent-soft)",
          glow: "var(--color-accent-glow)",
          ink: "var(--color-accent-ink)",
        },

        // Functional / semantic (alpha-aware for tinted fills/borders)
        ok: "rgb(var(--ok-rgb) / <alpha-value>)",
        warn: "rgb(var(--warn-rgb) / <alpha-value>)",
        error: "rgb(var(--error-rgb) / <alpha-value>)",

        // ---------- shadcn-compat (alpha-aware) ----------
        border: "hsl(var(--border) / <alpha-value>)",
        input: "hsl(var(--input) / <alpha-value>)",
        ring: "hsl(var(--ring) / <alpha-value>)",
        background: "hsl(var(--background) / <alpha-value>)",
        foreground: "hsl(var(--foreground) / <alpha-value>)",
        primary: {
          DEFAULT: "hsl(var(--primary) / <alpha-value>)",
          foreground: "hsl(var(--primary-foreground) / <alpha-value>)",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary) / <alpha-value>)",
          foreground: "hsl(var(--secondary-foreground) / <alpha-value>)",
        },
        muted: {
          DEFAULT: "hsl(var(--muted) / <alpha-value>)",
          foreground: "hsl(var(--muted-foreground) / <alpha-value>)",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive) / <alpha-value>)",
          foreground: "hsl(var(--destructive-foreground) / <alpha-value>)",
        },
        card: {
          DEFAULT: "hsl(var(--card) / <alpha-value>)",
          foreground: "hsl(var(--card-foreground) / <alpha-value>)",
        },
      },

      // ---------- Typography ----------
      fontFamily: {
        // grotesk does body + headings; display (Newsreader) is a sparing
        // italic accent; ui (JetBrains Mono) is every label/tag/date.
        sans: ["var(--font-grotesk)"],
        display: ["var(--font-display)", "Georgia", "serif"],
        body: ["var(--font-grotesk)"],
        ui: ["var(--font-ui)", "ui-monospace", "monospace"],
      },

      fontSize: {
        "ui-xs": ["0.6875rem", { lineHeight: "1.4", letterSpacing: "0.06em" }], // 11px
        "ui-sm": ["0.8125rem", { lineHeight: "1.4" }],                          // 13px
        "ui-base": ["0.9375rem", { lineHeight: "1.5" }],                        // 15px
        body: ["1rem", { lineHeight: "1.6" }],
        "body-lg": ["1.125rem", { lineHeight: "1.6" }],
        "display-sm": ["1.75rem", { lineHeight: "1.1", letterSpacing: "-0.015em" }],
        "display-md": ["2.5rem", { lineHeight: "1.05", letterSpacing: "-0.02em" }],
        "display-lg": ["3.5rem", { lineHeight: "1.0", letterSpacing: "-0.03em" }],
        "display-xl": ["clamp(2.5rem,6vw,5rem)", { lineHeight: "0.98", letterSpacing: "-0.03em" }],
      },

      maxWidth: {
        content: "1100px",
        prose: "680px",
        sidebar: "240px",
      },

      borderRadius: {
        lg: "var(--radius)",                  // 14px
        md: "calc(var(--radius) - 4px)",      // 10px
        sm: "calc(var(--radius) - 6px)",      // 8px
        xl: "20px",
      },

      boxShadow: {
        // The only "shadow" Lemnisca uses is the teal glow.
        glow: "0 0 10px var(--color-accent)",
        "glow-soft": "0 0 0 2px var(--color-accent-glow)",
      },

      keyframes: {
        "fade-up": {
          from: { opacity: "0", transform: "translateY(16px)" },
          to: { opacity: "1", transform: "none" },
        },
      },
      animation: {
        "fade-up": "fade-up 0.6s cubic-bezier(.2,.7,.2,1) both",
      },
    },
  },
  plugins: [],
};

export default config;
