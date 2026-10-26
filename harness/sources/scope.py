"""The rendering and profiling scope, identical for every source.

Check contract v2, rule 16: a prompt renders the source's own axioms first,
then the terms of the shipped vocabularies (ChEBI on Rhea), "restricted to the
terms the data uses and their ancestors", so a large imported ontology cannot
crowd out a graph's own axioms; the checkers still use the full closure. The
used terms are read once from the store (``UsedTerms``) and the shipped terms
once from the shipped graphs; a source's scope is all its own terms plus its
shipped terms that the data uses, closed upwards in that source's own
hierarchy. The source itself is never cut.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from harness.sources.closure import Hierarchy
from harness.sources.model import OntologySource

SCOPE_RULE = (
    "rule 33: every class of the source that is not a shared-hierarchy class, and every property of the source; "
    "of its shared-hierarchy classes, those among the classes the data uses (an IRI object of a data-graph triple "
    "that is a class of the shared hierarchy) and their ancestors in the shared hierarchy. Disjointness pairs are "
    "in scope when both classes are; existentials when their class is."
)


@dataclass(frozen=True)
class UsedTerms:
    classes: frozenset[str]
    properties: frozenset[str]

    def to_json(self) -> dict:
        return {"classes": sorted(self.classes), "properties": sorted(self.properties), "digest": self.digest()}

    @staticmethod
    def from_json(obj: dict) -> UsedTerms:
        out = UsedTerms(frozenset(obj["classes"]), frozenset(obj["properties"]))
        if "digest" in obj and obj["digest"] != out.digest():
            raise ValueError("used-terms digest does not match its content")
        return out

    def digest(self) -> str:
        text = json.dumps([sorted(self.classes), sorted(self.properties)], separators=(",", ":"))
        return hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True)
class Scope:
    classes: frozenset[str]
    properties: frozenset[str]
    used_digest: str | None  # None: the whole source (no restriction)


def class_hierarchy(src: OntologySource) -> Hierarchy:
    return Hierarchy({iri: c.parents for iri, c in src.classes.items()})


def property_hierarchy(src: OntologySource) -> Hierarchy:
    return Hierarchy({iri: p.parents for iri, p in src.properties.items()})


def compute_scope(src: OntologySource, used: UsedTerms | None, shared_terms: frozenset[str] = frozenset(),
                  shared_classes: frozenset[str] | None = None) -> Scope:
    """Rule 33: every own term (not a shared-hierarchy class) and every property of the source; of its
    shared-hierarchy classes, those in the shared hierarchy's scope (``shared_classes``: the classes the data
    uses and their ancestors), or, without it, those the data uses, closed upwards in the source."""
    if used is None:
        return Scope(frozenset(src.classes), frozenset(src.properties), None)
    ch = class_hierarchy(src)
    if shared_classes is not None:
        classes = {c for c in src.classes if c not in shared_terms or c in shared_classes}
        classes = ch.upward_closure(c for c in classes if c not in shared_terms) | (classes & set(shared_terms))
    else:
        classes = ch.upward_closure(c for c in src.classes if c not in shared_terms or c in used.classes)
    ph = property_hierarchy(src)
    props = ph.upward_closure(src.properties)
    return Scope(
        frozenset(c for c in classes if c in src.classes),
        frozenset(p for p in props if p in src.properties),
        used.digest(),
    )
