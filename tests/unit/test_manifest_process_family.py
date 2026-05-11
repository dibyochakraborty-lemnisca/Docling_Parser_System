"""load_process_manifest writes RegisteredProcess.process_family (D2 + UI plumbing).

Branch: upload-process-family-ui.

Verifies the manifest path the upload dropdown depends on: when an
operator supplies a closed-vocab process_family in their manifest, it
lands in RegisteredProcess.process_family (the field downstream
agents — memory layer, catalog runner — actually read), not just in
the human-readable ObservedFacts.process_family_hint.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from fermdocs.domain.models import IdentityProvenance
from fermdocs.dossier import load_process_manifest


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "manifest.yaml"
    p.write_text(textwrap.dedent(body))
    return p


def test_manifest_with_canonical_process_family_populates_registered(tmp_path):
    p = _write(tmp_path, """
        organism: Penicillium chrysogenum
        product: penicillin
        process_family: penicillin_fedbatch
        rationale: operator-supplied
    """)
    pi = load_process_manifest(p)
    assert pi.registered.process_family == "penicillin_fedbatch"
    assert pi.registered.provenance == IdentityProvenance.MANIFEST


def test_manifest_off_whitelist_family_is_rejected(tmp_path):
    """Operators can typo. The closed-vocab guard returns None rather
    than poisoning the dossier with an unrecognized string."""
    p = _write(tmp_path, """
        process_family: penicilin_fedbatch  # typo: single 'l'
    """)
    pi = load_process_manifest(p)
    assert pi.registered.process_family is None


def test_manifest_unknown_pick_is_normalised_to_none(tmp_path):
    """Explicit 'unknown' is treated identically to None so downstream
    memory writes / catalog routing short-circuit cleanly."""
    p = _write(tmp_path, """
        process_family: unknown
    """)
    pi = load_process_manifest(p)
    assert pi.registered.process_family is None


def test_manifest_missing_process_family_leaves_registered_none(tmp_path):
    """Legacy manifests (no process_family field) work unchanged —
    Registered.process_family stays None, downstream memory no-ops."""
    p = _write(tmp_path, """
        organism: S. cerevisiae
        process_family_hint: fed-batch  # the old free-text path
    """)
    pi = load_process_manifest(p)
    assert pi.registered.process_family is None
    # The free-text hint still lands in observed for human readers.
    assert pi.observed.process_family_hint == "fed-batch"


def test_manifest_with_both_canonical_and_hint(tmp_path):
    """Manifest supplies both. Canonical drives routing; hint stays as
    the human-readable label."""
    p = _write(tmp_path, """
        process_family: yeast_intracellular_product_fedbatch
        process_family_hint: yeast carotenoid fed-batch
    """)
    pi = load_process_manifest(p)
    assert pi.registered.process_family == "yeast_intracellular_product_fedbatch"
    assert pi.observed.process_family_hint == "yeast carotenoid fed-batch"


def test_manifest_canonical_only_surfaces_as_hint_too(tmp_path):
    """Operator picked the canonical name from a dropdown and didn't
    write a separate hint. We surface the canonical name into the
    hint field too so the dossier reads consistently for humans."""
    p = _write(tmp_path, """
        process_family: ecoli_recombinant_protein
    """)
    pi = load_process_manifest(p)
    assert pi.registered.process_family == "ecoli_recombinant_protein"
    assert pi.observed.process_family_hint == "ecoli_recombinant_protein"


@pytest.mark.parametrize("family", [
    "penicillin_fedbatch",
    "yeast_intracellular_product_fedbatch",
    "yeast_aerobic_fedbatch",
    "ecoli_recombinant_protein",
    "melanin_batch",
])
def test_all_closed_enum_values_accepted(tmp_path, family):
    p = _write(tmp_path, f"""
        process_family: {family}
    """)
    pi = load_process_manifest(p)
    assert pi.registered.process_family == family
