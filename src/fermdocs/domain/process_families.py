"""Process-family routing: which trajectory variable is THE product,
which are precursors, which are overflow byproducts.

Plan ref: plans/2026-05-07-characterize-determinism.md commit 2 (D2).

The Tier P (product) catalog metrics need to know WHICH trajectory
variable is the product for a given process family. For penicillin
fed-batch, that's `penicillin_g_l`. For E. coli recombinant protein,
`recombinant_protein_g_l`. For melanin batch, `melanin_g_l`. Hardcoding
that in catalog code would require a release per new process family;
putting it in YAML lets operators add families without code changes.

The loader is parallel to (but separate from) process_priors.py because
the two YAMLs answer different questions:
  - process_priors.yaml: 'what's the typical mu_max range for this
    organism in this process family?'
  - process_families.yaml: 'when bundle says process_family=penicillin
    fedbatch, which variable is the product?'

Missing process family → falls through to the `unknown` entry, which
routes to no product-KPI metrics. That's correct behavior for bundles
where we can't identify the process; the catalog runner emits no Tier P
findings rather than guessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_PATH = (
    Path(__file__).parent.parent / "schema" / "process_families.yaml"
)

# Family name reserved for 'we don't know what this is' bundles.
# When a dossier's process_family_hint is None or doesn't match a
# registered family, the loader returns this entry: empty product /
# precursor / byproduct lists. The catalog runner sees empty lists
# and skips Tier P metrics gracefully.
UNKNOWN_FAMILY_NAME = "unknown"


@dataclass(frozen=True)
class ProcessFamilyConfig:
    """One row of process_families.yaml resolved to a typed shape.

    `product_variable`: canonical golden-schema name of the EXCRETED
        product trajectory (g/L) that the run aims to maximize. Used
        by P1-P4. None when no excreted product applies.
    `intracellular_product_variable`: canonical golden-schema name of
        the INTRACELLULAR product trajectory in mg per g dry cell
        weight (mg/g DCW). Used by P_INTRACELLULAR_PRODUCT_YIELD for
        carotenoid, lipid, terpenoid, intracellular-protein
        processes. None when product is excreted (Tier P routes via
        product_variable instead) or no product applies.
    `precursor_variables`: variables that are CONSUMED into the product,
        not byproducts. PAA in penicillin fermentations: feeding it goes
        into the product, residual unconsumed PAA is waste. Yield is
        backwards from a byproduct: lower residual = better utilization.
    `overflow_byproducts`: byproducts that signal overflow metabolism
        (acetate for E. coli, ethanol for yeast). Used by B6 byproduct
        yield. Distinct from precursors.
    """

    name: str
    product_variable: str | None
    precursor_variables: tuple[str, ...]
    overflow_byproducts: tuple[str, ...]
    description: str | None = None
    intracellular_product_variable: str | None = None

    @property
    def is_unknown(self) -> bool:
        return self.name == UNKNOWN_FAMILY_NAME


def _config_from_dict(name: str, data: dict[str, Any]) -> ProcessFamilyConfig:
    return ProcessFamilyConfig(
        name=name,
        product_variable=(data.get("product_variable") or None),
        precursor_variables=tuple(data.get("precursor_variables") or ()),
        overflow_byproducts=tuple(data.get("overflow_byproducts") or ()),
        description=data.get("description"),
        intracellular_product_variable=(
            data.get("intracellular_product_variable") or None
        ),
    )


_UNKNOWN_FAMILY = ProcessFamilyConfig(
    name=UNKNOWN_FAMILY_NAME,
    product_variable=None,
    precursor_variables=(),
    overflow_byproducts=(),
    description="Catch-all for bundles without a recognized process family.",
)


@lru_cache(maxsize=4)
def load_process_families(
    path: str | None = None,
) -> dict[str, ProcessFamilyConfig]:
    """Load process_families.yaml into a dict keyed by family name.

    Always includes `UNKNOWN_FAMILY_NAME` even if the YAML omits it,
    so callers can use `lookup_family()` without guarding for None.

    Cached because YAML I/O on every catalog-runner adapter call would
    pile up; the file is tiny but reads serialize.
    """
    target = Path(path) if path else _DEFAULT_PATH
    out: dict[str, ProcessFamilyConfig] = {UNKNOWN_FAMILY_NAME: _UNKNOWN_FAMILY}
    if not target.exists():
        # Missing file is a soft case: callers fall through to unknown.
        # Tests can pass an empty/temporary path safely.
        return out
    with open(target) as f:
        data = yaml.safe_load(f) or {}
    families = data.get("process_families") or {}
    if not isinstance(families, dict):
        raise ValueError(
            f"process_families.yaml at {target}: top-level"
            f" 'process_families' must be a mapping; got {type(families).__name__}"
        )
    for name, entry in families.items():
        if not isinstance(entry, dict):
            continue
        out[name] = _config_from_dict(name, entry)
    return out


def lookup_family(
    process_family: str | None,
    *,
    path: str | None = None,
) -> ProcessFamilyConfig:
    """Resolve a dossier's `process_family_hint` to a ProcessFamilyConfig.

    Returns the `unknown` entry when:
      - process_family is None or empty/whitespace
      - the YAML has no entry by that name

    Empty / unknown family is a quiet fall-through, not an error —
    bundles for new process families should still characterize, just
    without the product-KPI metrics that the catalog can't route.
    """
    families = load_process_families(path=path)
    if process_family is None:
        return families[UNKNOWN_FAMILY_NAME]
    key = process_family.strip()
    if not key:
        return families[UNKNOWN_FAMILY_NAME]
    return families.get(key, families[UNKNOWN_FAMILY_NAME])
