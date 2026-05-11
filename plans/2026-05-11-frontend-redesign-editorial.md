# Frontend Redesign — Editorial Scientific

**Date:** 2026-05-11
**Branch:** `frontend-styling` (off `upload-process-family-ui`)
**Status:** Plan only — no code yet. Awaiting approval on the design doc + mockups before implementation.

## Direction

**Editorial Scientific** — the visual register of Quanta Magazine, Nature features, Aeon. Long-form prose treated with care. Generous whitespace. A single accent that earns its placement. Restrained chrome. Typography carries the design.

Confirmed decisions from D-001:

| Decision | Choice |
|---|---|
| Font licensing | Free path (no paid commercial fonts) |
| Accent color | Forest green (#1B4D3E) |
| Theme | Light-only (drop dark mode for v1) |
| Scope | Full redesign — every surface |
| Follow-up bar | Stays sticky as today |
| Process | Plan mode first; mockups before implementation |

## Why this direction is correct for fermdocs

The product is an evidence-and-argument tool. Scientists already read papers; fermdocs hypotheses *are* papers, condensed. The current shadcn/utility aesthetic looks like internal tooling — which makes credible scientific content feel low-effort. Editorial styling does two things simultaneously:

1. **Signals rigor.** A scientist will read a hypothesis differently when it's typeset like a Nature methods box vs. a Linear ticket.
2. **Slows the reader down.** Hypothesis content rewards careful reading. The current density encourages skim. Editorial layout enforces pause.

The risk is over-promising — looking polished but reading hollow. The mitigation is that we already shipped real content infrastructure (`[memory-axis]` critic, structured citations, actionable recommendations, charts with CIs). The visual layer should match what's already there.

## Typography stack

**Two fonts, one variable family + one grotesk.** Less is more.

### Fraunces — display + body
- Variable optical-size axis: small sizes get a sturdier text-grade, display sizes get more flair (sharper terminals, more contrast)
- Free, OFL, hosted on Google Fonts
- Used by Stripe Press, The Marshall Project, Vox features
- Character: serious but not stuffy. The "softie" axis lets us tune from austere to warm
- One file serves both `font-family: var(--font-display)` for headlines and `var(--font-body)` for paragraphs — axis-controlled

### Hanken Grotesk — UI chrome
- Open-source grotesk explicitly designed as a free Söhne alternative
- Variable weight; supports 100–900
- Used for: nav, buttons, labels, badges, metadata sidebars, form controls
- Character: quiet, well-spaced, doesn't compete with Fraunces

**Why not three fonts:** A display serif + body serif + UI sans is the textbook editorial trio. Fraunces' optical-size axis collapses display+body into one family, which is *more* refined, not less. The skill's "pair a distinctive display font with a refined body font" rule is satisfied — they just happen to share a family.

**What about pulling Söhne / Tiempos later:** the CSS variables are designed so swapping is one line. `--font-display`, `--font-body`, `--font-ui` are tokens. If Lemnisca wants the paid stack later, change three values.

## Color tokens

Single accent. Forest green for emphasis. Everything else is paper-and-ink.

```css
:root {
  /* Paper and ink — the foundation */
  --color-bg:           #FBFAF7;  /* off-white, warm */
  --color-bg-elevated:  #FFFFFF;  /* card surfaces, slight contrast against bg */
  --color-ink:          #0F1B2D;  /* deep navy, body text */
  --color-ink-secondary:#3D4A5C;  /* metadata, captions */
  --color-ink-muted:    #6B7280;  /* footnotes, byline meta */
  --color-rule:         #E5E2DA;  /* hairline dividers, table borders */
  --color-rule-strong:  #BFB9AB;  /* heavier rules below section headers */

  /* The accent — used sparingly */
  --color-accent:       #1B4D3E;  /* forest, the editorial accent */
  --color-accent-soft:  #E8EFE8;  /* tinted bg for accent callouts (recommendation pull-quote) */
  --color-accent-ink:   #FBFAF7;  /* text ON accent surface */

  /* Functional — only when absolutely needed */
  --color-warn:         #B45309;  /* amber, used only for [memory-axis], [robustness-axis] critic flags */
  --color-error:        #991B1B;  /* burgundy, used only for failed runs / rejected hypotheses */
}
```

**Rules of use:**
- Forest accent appears at most 3× per screen: (a) the active nav link or page kicker, (b) the recommendation pull-quote background, (c) primary CTA. Anywhere else is a violation.
- Warn/error are functional only. They're not part of the editorial palette; they appear on critic flags and failed-state badges. Editorial design tolerates one or two functional colors as long as they're treated as semantic, not decorative.
- All chrome borders are `--color-rule` hairlines. No drop shadows. Magazines don't use shadows.

## Layout system

**Asymmetric two-column grid** is the visual signature.

```
┌──────────────────────────────────────────────────────────┐
│  KICKER · KICKER · KICKER                                │
│                                                          │
│  Display headline of the run                             │
│  spanning the full content width                         │
│  ──────────────────────────────────────────────────────  │
│                                                          │
│  ┌──────────────────────────────┐  ┌──────────────────┐ │
│  │                              │  │  METADATA        │ │
│  │  Body column — 680px on      │  │                  │ │
│  │  desktop, 100% on mobile     │  │  Confidence 0.85 │ │
│  │                              │  │  Basis cross_run │ │
│  │  Hypothesis prose lives      │  │  Findings 9      │ │
│  │  here. Drop cap on first     │  │  Trajectories 3  │ │
│  │  letter. Long form,          │  │                  │ │
│  │  comfortable reading.        │  │  ────────        │ │
│  │                              │  │                  │ │
│  │  [chart inline, full         │  │  Affected runs   │ │
│  │   column width, labeled      │  │  RUN-0001        │ │
│  │   endpoints not legend]      │  │  RUN-0002        │ │
│  │                              │  │                  │ │
│  │  "Pull-quote treatment       │  │  ────────        │ │
│  │   for the recommendation     │  │                  │ │
│  │   gets oversized italic      │  │  Specialists     │ │
│  │   forest accent."            │  │  kinetics        │ │
│  │  — RECOMMENDED               │  │  metabolic       │ │
│  │                              │  │  mass_transfer   │ │
│  └──────────────────────────────┘  └──────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

**Grid specifics:**
- Total page max-width: 1100px, centered, generous side gutters
- Content column: 680px (golden-ratio comfortable for serif text at 18px/30px)
- Metadata sidebar: 240px
- Gap between columns: 56px
- Mobile (<768px): sidebar collapses to a labeled metadata strip *above* the content, separated by a hairline rule

**Vertical rhythm:**
- Hairline rule = 1px `--color-rule`. Used between sections.
- Section heading lockup: small kicker label (Hanken Grotesk, 11px, tracked +0.1em, uppercase) above a Fraunces display heading
- Body paragraphs separated by space, not indents — magazine convention, not academic-paper convention

## Surface-by-surface plan

### 1. Home / upload page (`/`)

**Current:** Card with form fields, dropdown, file tray, run list below.

**Editorial:**
```
        ┌───────────────────────────────────────┐
        │   fermdocs                            │  ← nav: Hanken, 14px, accent on hover
        ├───────────────────────────────────────┤
        │                                       │
        │   AN EDITORIAL HYPOTHESIS-ENGINE      │  ← kicker
        │   FOR FERMENTATION DATA               │
        │                                       │
        │   What did this run                   │  ← display headline, Fraunces 56px
        │   actually tell us?                   │     warm italic, ragged-right
        │                                       │
        │   Upload a CSV, XLSX, or PDF of your  │  ← body intro, Fraunces 20px
        │   fermentation experiment, name the   │
        │   process family, and the system will │
        │   produce evidence-grounded hypotheses│
        │   with cited findings, charts, and    │
        │   recommended next batches.           │
        │                                       │
        │   ────────────────────────────────    │
        │                                       │
        │   YOUR QUESTION (OPTIONAL)            │  ← small kicker for form
        │   [textarea, hairline border]         │
        │                                       │
        │   PROCESS FAMILY                      │
        │   [dropdown, native styled, hairline] │
        │                                       │
        │   FILES                               │
        │   + Add file                          │  ← link-style, forest on hover
        │                                       │
        │   [SUBMIT FOR ANALYSIS]               │  ← forest pill button
        │                                       │
        │   ────────────────────────────────    │
        │                                       │
        │   RECENT RUNS                         │  ← kicker
        │                                       │
        │   2026-05-11 · DONE                   │
        │   Why did RUN-0002 outperform on      │  ← run summary as a headline link
        │   penicillin titer?                   │
        │   penicillin_fedbatch · 6h ago        │  ← byline
        │                                       │
        │   ────                                │
        │                                       │
        │   2026-05-10 · DONE                   │
        │   Yeast carotenoid pigment-loss…      │
        │                                       │
        └───────────────────────────────────────┘
```

The form fields lose their boxed `<Card>` chrome and read like a magazine submission form. The run list becomes a Table of Contents.

### 2. Run page (`/runs/[id]`)

This is the heart of the redesign. The current run page has three sections (timeline, hypotheses, follow-up). All three get re-treated.

**The masthead:**
```
fermdocs                                       Download PDF
─────────────────────────────────────────────

RUN · f7d69118                                  ← kicker
penicillin_fedbatch · started 06:24, ended 06:51

Why did RUN-0002 outperform on                  ← user question as
penicillin titer? What process levers              display headline
explain the difference?
─────────────────────────────────────────────
```

**Hypothesis cards** (the key element) — see mockup variants below.

**Rejected hypotheses** become a smaller subsection at the end, treated as "Considered and dismissed" — small kicker, italic Fraunces text, no chart, no metadata sidebar. Reads as a footnote in the magazine sense.

**Timeline / debate panel** moves to a *collapsed* state by default, shown as a single line: `Debate · 14 turns · 6 lessons distilled · [Show timeline]`. Click expands into a side-drawer rather than the inline accordion we have today.

**Follow-up bar** stays sticky bottom (per your decision) but visually restyled: hairline top border, off-white bg, Hanken UI font, forest send button. Less prominent than today.

### 3. Hypothesis card — the centerpiece

Two mockup variants below. Both share the same "kicker → display headline → body → pull-quote → metadata sidebar" skeleton. Differences are in tone and density.

### 4. Charts (editorial treatment)

Currently Plotly with legends and default styling. Editorial version:

- **Labels at endpoints, not in a legend.** Each line's run_id label sits at the right edge of the trace, in the trace's color. Plotly supports this via `annotations` on the layout.
- **Single accent for highlighted runs**, hairline grey for non-highlighted.
- **Hairline grid lines** (`--color-rule`), no major/minor distinction. Tick labels in Hanken, 11px.
- **Title style:** Fraunces display, left-aligned, sentence case (not Title Case).
- **Rationale below the chart** in Fraunces italic 14px — already in our schema as `chart_spec.rationale`, just retreat the styling.
- **No background color on the chart panel.** Pure off-white, just the data.

Concrete change required in `chart_builder.py`: a new `editorial=True` rendering path that swaps the default Plotly theme. ~40 LOC.

### 5. Empty / loading / error states

Editorial designs require this. Three concrete cases:

**No runs yet** (homepage with empty run list):
```
NOTHING YET                          ← kicker

The first run you submit will        ← Fraunces 24px italic
appear here, with every subsequent
run added below it like entries
in a lab notebook.
```

**Run loading / processing:**
```
ANALYSIS IN PROGRESS                 ← kicker

The system is currently:             ← Fraunces 20px

Characterizing trajectories…         ← updates live via WebSocket,
                                        in Hanken sans, with a subtle
                                        animated dot at the end
```

**Failed run:**
```
RUN FAILED                           ← kicker, in error burgundy

The pipeline exited early at the     ← Fraunces 18px
diagnosis stage. The most likely
cause was Gemini API rate-limiting
at 06:34:12. No data was persisted.
```

No spinners. No skeleton placeholders. The loading state is *content* about what's happening.

### 6. The recommendation pull-quote

```
                        ────────────────
                                        ▎
                                        ▎  "Repeat with PAA feed
                                        ▎   profile from RUN-0002,
                                        ▎   particularly the
                                        ▎   60–96h window — that's
                                        ▎   where the divergence
                                        ▎   emerges."
                                        ▎
                                        ▎  — RECOMMENDED
                        ────────────────
```

- Indented from the body column, ~24px left
- Fraunces italic, 24px (vs 18px body)
- Forest left rule (3px solid `--color-accent`)
- "RECOMMENDED" attribution in Hanken Grotesk 11px tracked +0.15em, forest color
- Off-white background, slightly tinted with `--color-accent-soft` (#E8EFE8 — a 5% forest tint)
- Generous vertical space above and below (48px)

This is the visual high point of every hypothesis. It's the magazine "pull-quote" treatment doing the work of "actionable recommendation." Pulls the eye, gives the click-target for next action.

### 7. The drop cap

First paragraph of each accepted hypothesis gets a Fraunces drop cap:
- 64px display, italic, forest color
- `float: left`, `line-height: 1`, `margin-right: 8px`, `margin-top: 4px`
- Drops two body lines

Only on accepted hypotheses, not rejected. Drop caps are an editorial signal of "this matters; read this fully."

## Motion philosophy

Editorial designs are quiet. **No animations on first paint** except for typeface render. Specifically NO:
- Skeleton shimmers
- Fade-ins on scroll
- Animated gradients
- Bouncing micro-interactions

What IS allowed:
- A single staggered reveal on page load: kicker fades in at 0ms, headline at 80ms, body at 160ms. CSS only, total duration 240ms. Done. No further animation on that section.
- Hover state on links: forest color underline slides in from the left (CSS pseudo-element width transition, 120ms)
- The follow-up sticky bar has a subtle entrance: slides up 12px on mount, fades from 0 to 1 opacity, 240ms ease-out. Once.
- Chart hover: tooltip slides in 6px, 80ms, no shadow

That's it. No floating cards, no gradient meshes, no noise textures (the skill suggests these for *maximalist* designs — Editorial is the opposite extreme).

## Print CSS

Editorial designs print beautifully by default. The PDF export button you already shipped will produce dramatically better artifacts after this redesign. Specific print rules:

- `@page { margin: 2cm; size: A4; }`
- The sidebar metadata moves to a smaller font and appears in-flow under the body on print (no two-column print)
- Drop caps render fine
- Pull-quotes render fine
- Charts: Plotly's static export gets editorial styling automatically via the same `editorial=True` flag

Already-shipped print CSS gets reduced from ~30 lines to ~12 because the editorial design *is* print-friendly by default.

## Implementation phases

Three phases. Each independently shippable.

### Phase 1 — Type, color, layout tokens (small but foundational)
- Install Fraunces + Hanken Grotesk via `next/font` (free Google Fonts; no separate licensing step)
- Replace `tailwind.config.ts` color tokens with the editorial palette
- Replace `globals.css` typography defaults (sizes, line-heights, font-feature-settings for the optical-size axis)
- Add CSS variables for `--font-display`, `--font-body`, `--font-ui`, accent, paper colors
- Drop dark mode — remove the `.dark` class branches in globals.css and any `dark:` Tailwind variants in components
- ~2 hours. Outcome: type and color change everywhere; layouts don't change yet. Lets us A/B the look against the current design before further commitment.

### Phase 2 — Homepage + Run page layout overhaul
- New homepage masthead, form treatment, run list as TOC
- Run page masthead + new hypothesis card layout (kicker → headline → body → pull-quote → sidebar)
- Drop cap on accepted hypotheses
- Recommendation pull-quote with forest left rule
- Empty/loading/error states
- Follow-up bar restyled (sticky, hairline border, forest send button)
- ~1 day. Outcome: the redesign is visible end-to-end on every primary surface.

### Phase 3 — Editorial chart styling + polish
- New `editorial=True` flag on `chart_builder.py`
- Endpoint labels instead of legends
- Hairline grid, forest highlight, no panel bg
- Final type tuning (kerning pairs, optical-size axis values per heading level)
- Print CSS revisited
- ~half a day. Outcome: charts match the editorial register.

**Total scope:** ~2 days of focused work for full redesign.

## Mockup variants for the hypothesis card

Two flavors, both inside the Editorial Scientific direction. Showing both because they tune two different dials.

### Variant A — Quanta-flavored: clinical-clean, restrained

```
══════════════════════════════════════════════════════════
KINETICS  ·  ACCEPTED  ·  CROSS-RUN PRIOR APPLIED
══════════════════════════════════════════════════════════

R U N - 0 0 0 2  o u t p e r f o r m e d
b e c a u s e  P A A  p r e c u r s o r
f l o w  w a s  s u s t a i n e d  l o n g e r.

A cross-run lesson from a prior batch on the same process
family flagged that PAA depletion mid-fed-batch coincides
with stalled penicillin synthesis. The current bundle confirms
this pattern: RUN-0002's PAA trajectory remained above the
toxicity threshold throughout the productive window, while
RUN-0001 dropped below at hour 72.

[ chart: PAA over time, RUN-0001 grey, RUN-0002 forest,
  hairline grid, endpoint labels in trace colors ]

The data converges across kinetic, metabolic, and mass-transfer
specialists. No competing explanation survived the critic.

                        ┃
                        ┃  "Repeat with PAA feed
                        ┃   profile from RUN-0002,
                        ┃   particularly the 60–96h
                        ┃   window — that's where
                        ┃   the divergence emerges."
                        ┃
                        ┃  — RECOMMENDED

══════════════════════════════════════════════════════════
```

**Sidebar metadata** sits right of the body column:
```
CONFIDENCE         0.85
BASIS              cross_run
FINDINGS           9 cited
TRAJECTORIES       3 cited
SPECIALISTS        kinetics · metabolic · mass_transfer
AFFECTED           RUN-0001 · RUN-0002
```

**Character:** clinical. Wide letter-spacing on the headline. Restrained. Reads like a Quanta science feature.

### Variant B — Nature-flavored: warmer, more literary

```
══════════════════════════════════════════════════════════

  Hypothesis 1                            ⌗  H-0003
─────────────────────────────────────────────────────────

  RUN-0002 outperformed because PAA
  precursor flow was sustained longer.

  ╲╲ A cross-run lesson from a prior batch on
  ╲╲ the same process family flagged that PAA
  depletion mid-fed-batch coincides with stalled
  penicillin synthesis. The current bundle confirms
  this pattern: RUN-0002's PAA trajectory remained
  above the toxicity threshold throughout the
  productive window, while RUN-0001 dropped below
  at hour 72.

  [ chart: PAA over time, editorial styling ]

  The data converges across kinetic, metabolic, and
  mass-transfer specialists. No competing explanation
  survived the critic.

    ─── RECOMMENDED ───────────────────────────

       Repeat with PAA feed profile from
       RUN-0002, particularly the 60–96h
       window — that's where the divergence
       emerges.

    ───────────────────────────────────────────

  KINETICS · CROSS-RUN PRIOR · CONF 0.85 · 9 FINDINGS

══════════════════════════════════════════════════════════
```

**Sidebar:** same metadata, but presented as a "byline" footer line under the body rather than a right-column sidebar.

**Character:** warmer. Drop cap is the literal first letter ("A"). Headline is sentence case, not all-caps tracked. The byline below works like a magazine article's tag line. Recommendation pull-quote is treated as an inset block, not floated.

### Comparison

| Dimension | Variant A — Quanta | Variant B — Nature |
|---|---|---|
| Headline case | All-caps, tracked (more "newsroom") | Sentence case (more "long-form essay") |
| Metadata placement | Right sidebar (asymmetric grid) | Byline at bottom (linear flow) |
| Drop cap | None | Yes (initial "A") |
| Density | Sparser, more whitespace per card | Denser, more text-per-screen |
| Mobile behavior | Sidebar collapses above body | Already linear, no change |
| Visual rhythm | Strong horizontal rules every 200px | Softer transitions |
| Which use case wins | Quick scan / executive summary | Sit-down read / detailed analysis |

**My recommendation: Variant A.** Two reasons. (1) The asymmetric sidebar grid is a *stronger* visual signature — anyone who sees the run page will remember it. Variant B reads like a normal article, which is correct for magazines but loses the editorial-tool angle. (2) The metadata sidebar is genuinely useful on a run page — you want confidence/basis/citations visible without scrolling through the body. Variant B's bottom byline buries them.

But Variant B is more *humane* for long hypotheses. If our hypotheses end up averaging 400+ words each, B's softer rhythm scales better than A's stark sidebar.

## Open decisions before implementation

1. **Variant A or B** for hypothesis cards? My push is A.
2. **Phase order — small phase 1 first as a test, or commit straight to all three?** I'd vote phased: do phase 1, look at it, then decide.
3. **Drop the timeline / debate transcript entirely from the run page** vs. move it to a side-drawer? I'm proposing side-drawer. You may not want it at all.
4. **Keep the chart panel on the run page** or remove charts from the editorial run page entirely and put them inside the pull-quote-flow? I propose keep, inline in the body column at 100% of column width.
5. **Should the run list on the homepage support filtering by `process_family`?** Useful in editorial flow ("recent runs in penicillin_fedbatch"). Not in current scope.

## Failure modes to flag

- **Fraunces is heavy.** It's a variable font, ~150KB woff2. Acceptable, but worth noting; we'll subset and lazy-load.
- **Drop caps don't reflow well on narrow viewports.** Mobile (<480px) drops the drop cap and uses standard first paragraph styling.
- **Magazine-style typography requires good content.** A hypothesis with vague language will look *worse* in editorial layout than in utility layout because the layout amplifies the prose. This is fine — it's the right pressure on the synthesizer prompts. If we see a wave of "RUN-0002 had higher titer" empty headlines, we tighten synthesizer invariants. Don't redesign back to utility.

---

## Phase 1 PR plan (the only thing this plan commits to)

Single commit on `frontend-styling`:

1. Add `next/font/google` imports for Fraunces + Hanken Grotesk in `apps/web/src/app/layout.tsx`
2. Rewrite `apps/web/src/app/globals.css` color tokens to the editorial palette, drop dark mode declarations
3. Rewrite `apps/web/tailwind.config.ts` color tokens (typed Tailwind tokens that read from CSS variables)
4. Add `--font-display`, `--font-body`, `--font-ui` CSS variables wired to the next/font instances
5. No layout changes. No component restructuring. Just the typographic foundation.

Outcome: when you open the existing UI after Phase 1 lands, every page looks dramatically different *just from type and color* — but nothing breaks because layouts haven't changed. Lets us look at it side-by-side with the old, decide if we want to keep going.

If Phase 1 lands and feels wrong, we revert one commit. Cheap to try.

If Phase 1 lands and feels right, Phase 2 (layouts) is the next plan-mode iteration.
