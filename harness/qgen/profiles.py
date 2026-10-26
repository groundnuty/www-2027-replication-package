"""Question-set profiles by graph id.

``rhea-142`` is None: the original single-tier Rhea generation, whose output
stays byte-identical. ``uniprot-sp-2026_03`` is the two-tier UniProt set,
``text2sparql-ck26`` the two-tier corporate set. ``a15`` is the private graph's two-tier
set (audit rules on; IRIs from a private vocabulary file, outputs only to private files).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from harness.qgen import a15, corporate, uniprot
from harness.qgen import templates as rhea
from harness.qgen.base import Profile

PROFILES: dict[str, Profile | None] = {
    "rhea-142": None,
    uniprot.GRAPH_ID: uniprot.PROFILE,
    # a15's IRIs come from a private vocabulary read on first use (harness.qgen.a15)
    a15.GRAPH_ID: a15.PROFILE,
    corporate.GRAPH_ID: corporate.PROFILE,
}
TEMPLATE_SETS: dict[str, tuple] = {
    "rhea-142": rhea.TEMPLATES,
    uniprot.GRAPH_ID: uniprot.TEMPLATES,
    a15.GRAPH_ID: a15.TEMPLATES,
    corporate.GRAPH_ID: corporate.TEMPLATES,
}


def with_knobs(profile: Profile, overrides: dict[str, Any]) -> Profile:
    """The profile with some difficulty knobs changed (unknown knobs are refused)."""
    from dataclasses import replace

    unknown = set(overrides) - set(profile.knobs)
    if unknown:
        raise ValueError(f"unknown knobs {sorted(unknown)}; known: {sorted(profile.knobs)}")
    return replace(profile, knobs={**profile.knobs, **overrides})


REPO = Path(__file__).resolve().parent.parent.parent


def expected_counts(graph: str, root: Path = REPO) -> dict[str, int] | None:
    """Expected triples per named graph of the pinned store, or None when none is recorded.

    Rhea uses stores/rhea-142/expected_counts.json; other stores their
    descriptor stores/<graph>/store.json ("graphs").
    """
    legacy = root / "stores" / graph / "expected_counts.json"
    if legacy.exists():
        return json.loads(legacy.read_text())
    descriptor = root / "stores" / graph / "store.json"
    if descriptor.exists():
        return dict(json.loads(descriptor.read_text())["graphs"])
    return None


def store_mismatches(actual: dict[str, int], expected: dict[str, int]) -> list[str]:
    """Differences between the endpoint's triples per graph and the pinned store's (empty when it matches)."""
    out = []
    for g in sorted(set(actual) | set(expected)):
        if actual.get(g) != expected.get(g):
            out.append(f"{g}: expected {expected.get(g)}, endpoint has {actual.get(g)}")
    return out
