# How We Keep the Multi-Agent System Safe (and Trustworthy)

*A plain-language explainer. No jargon required.*

---

## The one-sentence version

The system is built to **refuse rather than bluff**. When the data supports an
answer, it gives one and shows its work. When the data does not, it says so
instead of inventing something. Almost everything below is in service of that
one rule.

---

## Why this matters

Our pipeline is a team of AI "agents" that hand work to each other:

> read the lab data → describe what happened → diagnose problems → propose
> hypotheses (a debate between specialist agents) → recommend changes → optimize

Each agent is powered by a large language model (LLM). LLMs are powerful but
they have one well-known failure mode: when unsure, they **make things up**
confidently. A founder's fair worry is: *"How do I know the system isn't just
generating plausible-sounding nonsense?"*

The answer is that we never let a single AI's word be the final say. Every claim
passes through deterministic (non-AI, plain-code) checks before anyone sees it.
Think of the LLM as a smart analyst, and the checks as a compliance desk the
analyst cannot talk its way past.

---

## How we keep it safe right now

These are live in the system today. Each is a separate layer, so if one misses
something, the next can still catch it ("defense in depth").

### 1. The math is computed, not guessed

The important numbers (growth rates, yields, oxygen margins, peak titer, and so
on) are calculated by **plain, audited code**, not by the LLM. The LLM is told:
"the math is already done and verified, use these numbers, do not recompute
them." So the agent reasons *about* facts; it does not get to invent the facts.

### 2. Every claim must point to real evidence

An agent cannot make a claim without citing the specific data point or finding
it came from. If an agent cites something that doesn't exist (a made-up
reference), that claim is **automatically dropped or rejected** before it
reaches the output. This single rule kills the most common form of AI
hallucination: confident statements backed by nothing.

### 3. Impossible numbers are blocked

A separate checker knows basic physical limits (you can't have a yield above
100%, a fraction above 1, and so on). Any computed value that breaks a hard
physical rule is flagged and set aside with a clear reason, instead of flowing
downstream and poisoning later steps. Recently we added a **data-relative**
version for the optimizer: if a model predicts a titer many times higher than
anything ever actually observed in the data, it's rejected as implausible
(this caught a case where corrupted units produced a nonsense "1809 g/L"
recommendation).

Important nuance: we deliberately **do not hardcode "expected" values** (like
"pH should be 6.5"). Those bias the system toward what we assume instead of what
the data shows. We judge data against *its own* distribution, not against baked-in
guesses.

### 4. Claims can't contradict the data ("claim guard")

This is a recent, deterministic guard that catches an agent asserting the
opposite of what the data plainly says. Real examples it blocks:

- Saying *"no dissolved-oxygen data available"* when that channel **is** present
  and populated.
- Calling DO = 0 an *"oxygen bottleneck"* in an **anaerobic** process, where
  zero oxygen is the normal, intended operating state, not a problem.
- Inventing a *"reactor scale confound"* when every run used the **same** reactor
  (a changing volume number was just the broth being filled, not a different
  vessel).
- Reporting a rate *"at t=0"* — a rate needs two timepoints, so this is
  physically meaningless.

If an agent says any of these, the statement is rejected and replaced with the
correction, with the original kept for audit.

### 5. Confidence is capped, and refusal is allowed

No agent is allowed to claim near-total certainty; confidence is capped. And the
system is explicitly permitted to **refuse**: if the data is too sparse or too
confounded to support a model, the optimizer says *"I can't responsibly answer
this"* rather than forcing a weak answer. We treat an honest "no" as a correct
output, not a failure.

### 6. It degrades gracefully instead of crashing

LLMs occasionally return broken or cut-off responses. Instead of crashing the
whole run (which used to happen), we now **salvage** what we can from a partial
response and keep going. One flaky reply from one agent no longer takes down the
entire analysis.

### 7. Specialists stay in their lane

The debate specialists (kinetics, mass-transfer, metabolism) each carry written
rules: stay in your domain, cite your evidence, defer when you have nothing
useful to add. When a specialist genuinely has no relevant data, it says so and
steps back rather than padding the debate with filler.

### 8. The test wall: ~1,400 automated checks

This is a big one and easy to undervalue. The codebase ships with **~1,392 unit
tests** (and **~1,628 tests in total**, across **178 test files**). Every safety
mechanism above has tests that prove it works — including tests written from the
*actual* bad outputs we've seen, so the same mistake can't silently come back.

Why this matters for trust:

- **Nothing ships unchecked.** Every change runs against the full suite. If a
  change breaks a safety rule, a test goes red immediately, before it ever
  reaches a user.
- **Regressions are caught automatically.** When we fix a real-world failure, we
  add a test that reproduces it. That failure is now permanently fenced off.
- **It's living documentation.** The tests describe, in executable form, exactly
  what each agent is and isn't allowed to do — and they're verified on every run,
  unlike a written doc that can drift out of date.

A thousand-plus tests is not box-ticking; it's the difference between "we
believe it's safe" and "we can demonstrate it's safe, on demand, in seconds."

---

## The next step: explicit agent contracts (and the trap to avoid)

There's an open request to give every agent an explicit **contract**: a written
statement of what it may do and must not do, enforced at runtime. This is a good
idea and the natural next layer. But *how* we build it matters enormously,
because the obvious version would actually **hurt** the product.

### Why over-enforcing hampers

1. **Over-blocking.** A guard that blocks output will sometimes block *good*
   output. We saw this the same week we built the claim guard: it briefly flagged
   the system's own *corrected* sentence until we taught it to recognize the
   correction. Scale that across a dozen agents and the system gets quieter and
   dumber — suppressing valid insights, which is worse than the rare bad one.

2. **It fights the goal.** The whole point of contracts is to *trust agents as
   they get smarter over time*. But if the contract is a strict **permission
   list** ("this agent may ONLY do X, Y, Z"), then every time an agent
   legitimately learns a new useful trick, the system breaks until someone edits
   the contract. That's a permanent tax on every improvement — the opposite of
   what we want.

3. **Maintenance drag.** Strict, exhaustive contracts have to be kept perfectly
   in sync with reality forever, or they cause false alarms.

### The version that helps instead of hampers

- **List what's forbidden, not what's permitted.** Hard-enforce a short, stable
  set of "never do this" rules (don't fabricate evidence, don't output impossible
  numbers, don't contradict the data). *Describe* capabilities for humans, but
  don't lock them down. Forbidden things rarely change; allowed things change all
  the time. This is what makes it safe for agents to keep evolving.

- **Flag by default, block only the clear cases.** For anything borderline, lower
  the confidence or attach a caveat instead of deleting the output. Reserve hard
  blocking for the handful of rules that are never acceptable.

- **Watch before you block ("shadow mode").** Turn the new checks on in
  observe-only mode first: log what *would* have been blocked, measure how often
  it's a false alarm on real runs, then switch on blocking only for the rules
  that prove clean. This is the single best protection against the system
  silencing good work.

- **Reuse, don't reinvent.** The enforcer would be a thin organizer over the
  guards we already run and trust — not a new layer of AI judgment.

Built this way, agent contracts add safety and auditability **without** slowing
down the agents or the team.

---

## Why you can trust it today

- It **shows its work**: every claim points to the data behind it.
- It **can't make up the numbers**: the math is computed by audited code.
- It **can't state the impossible**, or contradict its own data: guards block it.
- It **says "I don't know"** when that's the honest answer.
- It **keeps running** when one agent stumbles.
- And **~1,400 automated tests prove all of the above on every change.**

Trust here isn't "the AI is always right." It's "the AI is never the only thing
standing between a guess and your screen." As the agents get better, that safety
net stays in place — and the contract work above will make it explicit, agent by
agent, without putting the brakes on progress.

---

*Owner: engineering. Status: safety layers 1–8 are live and tested today;
explicit per-agent contracts are planned along the "forbidden-not-permitted,
flag-first, shadow-mode" lines described above.*
