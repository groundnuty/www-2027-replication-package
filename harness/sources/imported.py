"""Imported ontology files in the prompt (check contract rule 16 as amended, change-log entry 101).

An ontology file a curated source imports from another publisher is treated like a shipped vocabulary in the prompt:
its statements render after the source's own four layers, restricted to the terms the source's own axioms or the data
use, and their ancestors. For the checker nothing changes: every axiom stays part of the curated treatment.

Which statements are imported is decided from the source's per-axiom origin (harness.sources.origin) and the owner's
namespaces, an explicit input recorded in the render manifest (never a constant naming a graph):

* a part (file or graph) is another publisher's when it declares at least one class or property and none in the
  owner's namespaces;
* an axiom is imported when every part it comes from is another publisher's; one that any of the owner's parts states
  is the owner's, and so is one the origin record does not list;
* a term's labels and comments go with its declaration.

A source without an origin record, or rendered without the owner's namespaces, renders exactly as before.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass

from harness.sources.model import OntologySource

RULE = ("check contract rule 16 as amended (change-log entry 101): statements from an ontology file of another "
        "publisher (per-axiom origin: a part declaring terms only outside the owner's namespaces) render after the "
        "source's own four layers, restricted to the terms the source's own axioms or the data use, and their "
        "ancestors; the checker keeps every axiom")


@dataclass(frozen=True)
class Imported:
    """The owner's namespaces, the source's parts from another publisher, and the terms the data uses."""

    owner_namespaces: tuple[str, ...]
    parts: frozenset[str]
    used: frozenset[str]

    def __post_init__(self):
        if not self.owner_namespaces or not all(isinstance(n, str) and n for n in self.owner_namespaces):
            raise ValueError("the owner's namespaces must be given (at least one non-empty namespace)")

    def is_imported(self, parts: Iterable[str] | None) -> bool:
        ps = set(parts or ())
        return bool(ps) and ps <= self.parts


def namespaces_sha256(owner_namespaces: Iterable[str]) -> str:
    """The one digest of a set of owner's namespaces: what the fit sizes and the render guard compare, the namespaces
    themselves being private where they name a private graph's vocabulary."""
    return hashlib.sha256(json.dumps(sorted(owner_namespaces)).encode()).hexdigest()


def imported_parts(src: OntologySource, owner_namespaces: tuple[str, ...]) -> frozenset[str]:
    """The parts of ``src``'s origin record that declare at least one class or property and none in the owner's
    namespaces; empty without an origin record."""
    origin = src.origin
    if not origin:
        return frozenset()
    declared: dict[str, set[str]] = {p: set() for p in origin.get("parts", [])}
    for kind in ("classes", "properties"):
        for term, parts in (origin.get(kind) or {}).items():
            for p in parts:
                declared.setdefault(p, set()).add(term)
    return frozenset(p for p, terms in declared.items()
                     if terms and not any(t.startswith(ns) for t in terms for ns in owner_namespaces))


class Split:
    """Which of a source's axioms are imported, read from its origin record (``Imported``)."""

    def __init__(self, src: OntologySource, imported: Imported):
        self.src, self.imp = src, imported
        self.origin = src.origin or {}

    def _parts(self, kind: str, *keys: str):
        node = self.origin.get(kind) or {}
        for k in keys:
            if not isinstance(node, dict):
                return None
            node = node.get(k)
            if node is None:
                return None
        return node

    def imported_class(self, c: str) -> bool:
        return self.imp.is_imported(self._parts("classes", c))

    def imported_property(self, p: str) -> bool:
        return self.imp.is_imported(self._parts("properties", p))

    def imported_edge(self, c: str, d: str) -> bool:
        return self.imp.is_imported(self._parts("subclass", c, d))

    def imported_pair(self, a: str, b: str) -> bool:
        x, y = sorted((a, b))
        return self.imp.is_imported(self._parts("disjoint", x, y))

    def imported_expr(self, side: str, p: str, members: Iterable[str]) -> bool:
        return self.imp.is_imported(self._parts(side, p, "|".join(sorted(members))))

    def imported_datatype(self, p: str, dt: str) -> bool:
        return self.imp.is_imported(self._parts("datatype_range", p, dt))

    def imported_range_values(self, p: str) -> bool:
        return self.imp.is_imported(self._parts("range_values", p))

    def imported_existential(self, c: str, p: str, f: str) -> bool:
        return self.imp.is_imported(self._parts("existentials", c, p, f))

    def own_terms(self) -> set[str]:
        """Every term the source's own (not imported) axioms name."""
        s, out = self.src, set()
        for c, info in s.classes.items():
            if not self.imported_class(c):
                out.add(c)
            out |= {p for p in info.parents if not self.imported_edge(c, p)} | ({c} if any(
                not self.imported_edge(c, p) for p in info.parents) else set())
        for p, info in s.properties.items():
            if not self.imported_property(p):
                out.add(p)
                out |= set(info.parents)
            for side in ("domain", "range"):
                for e in getattr(info, side):
                    if not self.imported_expr(side, p, e.members):
                        out |= {p, *e.members}
            if any(not self.imported_datatype(p, d) for d in info.datatype_range):
                out.add(p)
            if info.range_values and not self.imported_range_values(p):
                out.add(p)
        for a, b in s.disjoint:
            if not self.imported_pair(a, b):
                out |= {a, b}
        for e in s.existentials:  # not rendered (no layer), but the terms the owner's own axioms use
            if not self.imported_existential(e.cls, e.prop, e.filler):
                out |= {e.cls, e.prop, e.filler}
        return out
