# Authoring guide — adding a hypothesis-stage specialist

> Status: hypothesis stage v0 (Stage 4)
> Plan ref: [plans/2026-05-03-hypothesis-debate-v0.md](../plans/2026-05-03-hypothesis-debate-v0.md)

The hypothesis stage runs three specialist personas by default:
**kinetics**, **mass_transfer**, **metabolic**. Each is a thin wrapper
over the shared [`SpecialistAgent`](../src/fermdocs_hypothesis/agents/specialist_base.py)
class — only the **persona spec** differs.

This guide documents how to add a fourth specialist (e.g.
`product_quality`, `regulatory`, `economics`) without touching the
runner, projector, or schema.

---

## 1. The `SPECIALIST_SPEC` contract

A specialist is a Python module under
`src/fermdocs_hypothesis/agents/` exporting two things:

```python
SPECIALIST_SPEC: dict[str, Any]
def build_<role>_specialist(client, tools) -> SpecialistAgent
```

The `SPECIALIST_SPEC` dict has six required keys:

| key | type | purpose |
|---|---|---|
| `role` | `str` | unique role name; used as the `SpecialistRole` literal |
| `system_identity` | `str` | the LLM's persona — who they are, what they do, what they don't touch |
| `invariants` | `tuple[str, ...]` | hard rules surfaced near the prompt's salience anchors (top + bottom) |
| `task_spec` | `str` | one paragraph of "what you do this turn" |
| `tool_hints` | `tuple[ToolHint, ...]` | one-line purpose per tool name |
| `recap` | `str` | the closing recency anchor: output schema + hard rules restated |

See [`specialist_kinetics.py`](../src/fermdocs_hypothesis/agents/specialist_kinetics.py)
for the canonical example.

---

## 2. The four invariants every specialist needs

Copy these verbatim into your `invariants` tuple. They aren't optional —
they encode the design rules the rest of the system relies on:

```python
"invariants": (
    "Stay in the <YOUR DOMAIN> domain. Don't reach into others' areas.",
    "Every facet must cite ≥1 finding, narrative, or trajectory.",
    "Confidence ≤ 0.85; if evidence is thin, drop confidence and call it out.",
    "If you used a process_priors lookup, set confidence_basis='process_priors'.",
    "READ relevant_analyses FIRST: if a diagnose-layer analysis already"
    " explains the topic's findings as data-quality / spec-config / known"
    " artifact, frame your facet to honor that — do NOT re-derive a"
    " process anomaly the analysis already explained away.",
),
```

The last invariant is **load-bearing**. Without it specialists can
re-derive false anomalies for findings the diagnose stage already
explained as data-quality issues (see the IndPenSim postmortem in
plans §15).

---

## 3. Adding a new specialist — step-by-step

Suppose we want a **`product_quality`** specialist focused on titer,
purity, impurities, batch-to-batch consistency.

### Step 1 — write the persona file

`src/fermdocs_hypothesis/agents/specialist_product_quality.py`:

```python
"""Product-quality specialist persona."""

from typing import Any

from fermdocs_hypothesis.agents.specialist_base import SpecialistAgent
from fermdocs_hypothesis.llm_clients import GeminiHypothesisClient
from fermdocs_hypothesis.prompts import ToolHint
from fermdocs_hypothesis.tools_bundle.factory import (
    GET_NARRATIVE_OBSERVATIONS, GET_PRIORS, HypothesisToolBundle, QUERY_BUNDLE,
)


SPECIALIST_SPEC: dict[str, Any] = {
    "role": "product_quality",
    "system_identity": """\
You are the Product-Quality specialist in a fermentation-hypothesis debate.

Your domain: titer (g/L of product), purity, impurity profile, batch-to-batch
consistency, downstream-processability hints (viscosity, filtration speed).

You do NOT opine on intrinsic kinetics (kinetics' domain), DO/kLa/agitation
(mass_transfer's domain), or pathway-level metabolism (metabolic's domain).
Focus on the product surface.

You are observational. Cite evidence. Never make causal claims you cannot
ground in the cited evidence.\
""",
    "invariants": (
        "Stay in the product-quality domain.",
        "Every facet must cite ≥1 finding, narrative, or trajectory.",
        "Confidence ≤ 0.85; if evidence is thin, drop confidence and call it out.",
        "If you used a process_priors lookup, set confidence_basis='process_priors'.",
        "READ relevant_analyses FIRST: if a diagnose-layer analysis already"
        " explains the topic's findings as data-quality / spec-config / known"
        " artifact, frame your facet to honor that — do NOT re-derive a"
        " process anomaly the analysis already explained away.",
    ),
    "task_spec": """\
Read the view, optionally call tools to fetch more data, then contribute
ONE facet on the current_topic from the product-quality angle.

Tool budget: up to 6 tool calls before you must contribute_facet.
""",
    "tool_hints": (
        ToolHint(name=QUERY_BUNDLE, purpose="search findings/narratives/trajectories"),
        ToolHint(name=GET_PRIORS, purpose="organism-aware bounds for product/titer vars"),
        ToolHint(name=GET_NARRATIVE_OBSERVATIONS, purpose="filter by run_id/tag/variable"),
        ToolHint(name="contribute_facet", purpose="TERMINAL: emit your facet"),
    ),
    "recap": """\
Output one JSON action.
When tool_call: {"action":"tool_call","tool":"<name>","args":{...}}
When done: {"action":"contribute_facet","summary":..., "cited_finding_ids":[...],
"cited_narrative_ids":[...], "cited_trajectories":[{"run_id":..., "variable":...}],
"affected_variables":[...], "confidence":<0..0.85>, "confidence_basis":"schema_only"|"process_priors"|"cross_run"}

Hard rules:
  - Stay in product-quality domain.
  - At least one citation field must be non-empty.
  - confidence_basis='process_priors' requires that you actually called get_priors first.\
""",
}


def build_product_quality_specialist(
    client: GeminiHypothesisClient,
    tools: HypothesisToolBundle,
) -> SpecialistAgent:
    return SpecialistAgent(
        client=client, spec=SPECIALIST_SPEC, tools=tools, role="product_quality"
    )
```

### Step 2 — extend the `SpecialistRole` literal

[`schema.py`](../src/fermdocs_hypothesis/schema.py):

```python
SpecialistRole = Literal[
    "kinetics", "mass_transfer", "metabolic", "product_quality"
]
```

### Step 3 — register in the runner's `SPECIALIST_ORDER`

[`runner.py`](../src/fermdocs_hypothesis/runner.py):

```python
SPECIALIST_ORDER: tuple[SpecialistRole, ...] = (
    "kinetics", "mass_transfer", "metabolic", "product_quality",
)
```

Order matters for **deterministic facet ordering**: tests rely on
specialists contributing in this order.

### Step 4 — register in `LiveHooks`

[`live_hooks.py`](../src/fermdocs_hypothesis/live_hooks.py):

```python
from fermdocs_hypothesis.agents.specialist_product_quality import (
    build_product_quality_specialist,
)

# in LiveHooks.__init__:
self._product_quality = build_product_quality_specialist(self._client, self._tools)
self._specialists: dict[SpecialistRole, SpecialistAgent] = {
    "kinetics": self._kinetics,
    "mass_transfer": self._mass_transfer,
    "metabolic": self._metabolic,
    "product_quality": self._product_quality,
}
```

### Step 5 — extend the projector's domain-tag filter

[`state.py`](../src/fermdocs_hypothesis/state.py)`::specialist_domain_tags`:

```python
if role == "product_quality":
    return {
        "titer", "product", "purity", "impurity", "yield_p",
        "viscosity", "color",
    }
```

The `SpecialistView` projector uses this set to filter findings,
narratives, and priors to the specialist's domain. Be generous — it's
better to over-include than miss a relevant finding.

### Step 6 — add a unit test

`tests/unit/hypothesis/test_specialist_product_quality.py`:

```python
from fermdocs_hypothesis.agents.specialist_product_quality import SPECIALIST_SPEC

def test_persona_has_required_keys():
    required = {"role", "system_identity", "invariants", "task_spec", "tool_hints", "recap"}
    assert required <= set(SPECIALIST_SPEC.keys())

def test_persona_role_matches_specialist_role_literal():
    assert SPECIALIST_SPEC["role"] == "product_quality"
```

Domain-shape tests (e.g. "facet from this specialist actually addresses
product variables") belong in your live-eval suite, not unit tests.

### Step 7 — done

Run the offline suite to confirm nothing regressed:

```bash
python -m pytest tests/unit/hypothesis/ -q
```

Then the live e2e if you want to see your new specialist in action:

```bash
FERMDOCS_RUN_LIVE_TESTS=1 \
  python -m pytest tests/unit/hypothesis/test_e2e_stage3_live.py -v
```

---

## 4. Things to watch for

- **Domain bleed.** The biggest failure mode is a specialist
  commenting outside its lane. The system prompt's "you do NOT opine
  on X, Y, Z" line is doing real work — be explicit.
- **Token budget.** Each new specialist adds one full Gemini call per
  topic (plus tool-loop overhead). Budget-wise, 4 specialists × 1
  topic costs ~25-35k tokens. The plan-§7 budget defaults
  (`max_total_input_tokens=200_000`) tolerate up to ~5 specialists
  before you start hitting caps in heavy debates.
- **Role conflicts.** If two specialists' domains overlap, the
  synthesizer has to figure out which framing wins. The current 3
  cover the main fermentation surfaces cleanly. A 4th should fill a
  genuine gap, not slice an existing domain finer.

## 5. What the runner does NOT need to change

- `runner.py` step function (other than `SPECIALIST_ORDER`)
- `projector.py` view builders
- `validators.py`
- Any agent other than `live_hooks.py`'s constructor

This is the v0 architectural payoff: **specialists are persona files,
not new code paths.**
