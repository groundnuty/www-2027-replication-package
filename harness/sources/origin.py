"""Per-axiom origin (design ruling in the design log Oct 08:04): which part of its input each axiom of a curated source
comes from, and the origin of a checker flag, from the axioms it rests on.

A part is a published file when the source is built from files (``curated.build_curated_from_files``) and a store
graph when it is read from a store (``curated.build_curated``). Each part is read alone too, by the same reader, and an
axiom of the source comes from every part whose own reading has it, and from every file holding a triple the reader
joined into it (``traced_joins``). A source built from files is read as one graph, so the reader can join triples of
two files on a node they share by IRI: an existential's restriction (C rdfs:subClassOf R, R owl:onProperty p,
R owl:someValuesFrom D) and an owl:AllDisjointClasses node with its members list; the classes these name are such axioms
too. Its other joins go through a blank node (a union, an intersection, an enumeration, and their lists as Turtle and
RDF/XML write them), which never spans two files, and a store's graphs are never joined (every reader pattern sits in
one GRAPH ?g). An axiom no part has alone and no
trace explains comes from the parts that declare its subject term, and failing that from every part.

The axioms are the frozen fragment's: a declared class or property, a subclass edge, a disjointness pair, a domain or
range (a named class or a union, by its members), a datatype range, an enumerated range and an existential. The record
(``OntologySource.origin``) maps each to the names of its parts, in the parts' order:

    {"by": "file" | "graph", "parts": [...], "classes": {C: [...]}, "properties": {p: [...]},
     "subclass": {C: {D: [...]}}, "disjoint": {A: {B: [...]}} (A < B), "domain": {p: {"M1|M2": [...]}},
     "range": {p: {"M1|M2": [...]}}, "datatype_range": {p: {dt: [...]}}, "range_values": {p: [...]},
     "existentials": {C: {p: {D: [...]}}}}

A flag's origin (``FlagOrigin.of``) lists the parts of the axioms it rests on, read from the terms it names (its key
and its evidence): the domain, range, datatype and enumerated ranges of the properties it names, and the subclass edges
up from the classes it names (in the source's own hierarchy); for a disjointness flag also the disjointness pairs
between the ancestors of two named classes, and for a provenance flag the existentials of a named class or of a named
property towards a named class. A flag that rests on none of these has the parts that declare its named terms, and
one that names no term of the source (a term the source lacks) every part. A check that reads no source
(``outcomes.INDEPENDENT_CHECKS``) has an empty origin. Verdicts never read the origin.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from harness.sources.model import ClassInfo, Existential, OntologySource, PropertyInfo

KINDS = ("classes", "properties", "subclass", "disjoint", "domain", "range", "datatype_range", "range_values",
         "existentials")


def expr_key(members: Iterable[str]) -> str:
    """A domain or range expression's key: its members, sorted, joined by "|" (no IRI holds one)."""
    return "|".join(sorted(members))


def axioms(classes: Mapping[str, ClassInfo], properties: Mapping[str, PropertyInfo],
           disjoint: Iterable[tuple[str, str]], existentials: Iterable[Existential]) -> dict[str, set[tuple]]:
    """The axioms of one reading, by kind, each as a tuple key."""
    out: dict[str, set[tuple]] = {k: set() for k in KINDS}
    for c, info in classes.items():
        out["classes"].add((c,))
        out["subclass"] |= {(c, p) for p in info.parents}
    for p, info in properties.items():
        out["properties"].add((p,))
        out["domain"] |= {(p, expr_key(e.members)) for e in info.domain}
        out["range"] |= {(p, expr_key(e.members)) for e in info.range}
        out["datatype_range"] |= {(p, d) for d in info.datatype_range}
        if info.range_values:
            out["range_values"].add((p,))
    out["disjoint"] |= {tuple(sorted(pair)) for pair in disjoint}
    out["existentials"] |= {(e.cls, e.prop, e.filler) for e in existentials}
    return out


def source_axioms(src: Any) -> dict[str, set[tuple]]:
    """``axioms`` of a source or an ``owl_reader.OwlFragment``."""
    return axioms(src.classes, src.properties, src.disjoint, src.existentials)


def record(combined: Any, parts: Mapping[str, dict[str, set[tuple]]], by: str,
           traced: Mapping[str, Mapping[tuple, Iterable[str]]] | None = None) -> dict:
    """The origin record of ``combined`` (a source or fragment) from each part's own axioms (``axioms``) and the parts
    holding the triples of its joins (``traced``, by kind and axiom key, from ``traced_joins``), by part name in the
    parts' order."""
    names = list(parts)
    traced = traced or {}

    def of(kind: str, key: tuple, subjects: tuple[str, ...]) -> list[str]:
        joined = set(traced.get(kind, {}).get(key, ()))
        found = [n for n in names if key in parts[n][kind] or n in joined]
        if not found:  # only the parts together give it, through no traced join: the parts that declare its subject
            found = [n for n in names if any((t,) in parts[n]["classes"] or (t,) in parts[n]["properties"]
                                             for t in subjects)]
        return found or names

    out: dict[str, Any] = {"by": by, "parts": names, **{k: {} for k in KINDS}}
    for c, info in sorted(combined.classes.items()):
        out["classes"][c] = of("classes", (c,), (c,))
        if info.parents:
            out["subclass"][c] = {p: of("subclass", (c, p), (c,)) for p in sorted(info.parents)}
    for p, info in sorted(combined.properties.items()):
        out["properties"][p] = of("properties", (p,), (p,))
        for kind, exprs in (("domain", info.domain), ("range", info.range)):
            if exprs:
                out[kind][p] = {expr_key(e.members): of(kind, (p, expr_key(e.members)), (p,)) for e in exprs}
        if info.datatype_range:
            out["datatype_range"][p] = {d: of("datatype_range", (p, d), (p,)) for d in info.datatype_range}
        if info.range_values:
            out["range_values"][p] = of("range_values", (p,), (p,))
    for a, b in sorted(tuple(sorted(x)) for x in combined.disjoint):
        out["disjoint"].setdefault(a, {})[b] = of("disjoint", (a, b), (a, b))
    for e in sorted(combined.existentials, key=lambda x: (x.cls, x.prop, x.filler)):
        out["existentials"].setdefault(e.cls, {}).setdefault(e.prop, {})[e.filler] = of(
            "existentials", (e.cls, e.prop, e.filler), (e.cls,))
    return out


def traced_joins(combined: Any, merged: Any, files: Mapping[str, Any]) -> dict[str, dict[tuple, set[str]]]:
    """The files holding each triple ``owl_reader.read_owl_fragment`` joined into an axiom of ``combined`` (a source
    read from ``merged``, the rdflib graph of every file's triples; ``files``: each file's own graph, by name), as
    ``record`` takes them: for each existential C -> p some D, the triples C rdfs:subClassOf R, R owl:onProperty p and
    R owl:someValuesFrom D of every R; for each disjointness pair, the typing, owl:members and list triples of every
    owl:AllDisjointClasses node listing both; for each class, those of the existentials and pairs that name it. A
    triple of two files is in both."""
    from rdflib import RDF, URIRef
    from rdflib.namespace import OWL, RDFS

    def holding(triples: Iterable[tuple]) -> set[str]:
        triples = list(triples)
        return {n for n, g in files.items() if any(t in g for t in triples)}

    out: dict[str, dict[tuple, set[str]]] = {"existentials": {}, "disjoint": {}, "classes": {}}

    def add(kind: str, key: tuple, names: set[str]) -> None:
        out[kind].setdefault(key, set()).update(names)

    for e in combined.existentials:
        c, p, f = URIRef(e.cls), URIRef(e.prop), URIRef(e.filler)
        names = holding(t for r in merged.objects(c, RDFS.subClassOf)
                        if (r, OWL.onProperty, p) in merged and (r, OWL.someValuesFrom, f) in merged
                        for t in ((c, RDFS.subClassOf, r), (r, OWL.onProperty, p), (r, OWL.someValuesFrom, f)))
        add("existentials", (e.cls, e.prop, e.filler), names)
        add("classes", (e.cls,), names)
        add("classes", (e.filler,), names)
    pairs = {tuple(sorted(x)) for x in combined.disjoint}
    for x in set(merged.subjects(RDF.type, OWL.AllDisjointClasses)):
        used, members = [(x, RDF.type, OWL.AllDisjointClasses)], set()
        for head in merged.objects(x, OWL.members):
            used.append((x, OWL.members, head))
            seen, stack = {head}, [head]
            while stack:  # the list, as the reader's rdf:rest*/rdf:first path walks it
                node = stack.pop()
                for m in merged.objects(node, RDF.first):
                    used.append((node, RDF.first, m))
                    members.add(str(m) if isinstance(m, URIRef) else None)
                for nxt in merged.objects(node, RDF.rest):
                    used.append((node, RDF.rest, nxt))
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
        if None in members:  # a member that is no IRI: the reader drops the node, which states no pair
            continue
        listed = sorted(pair for pair in pairs if pair[0] in members and pair[1] in members)
        if listed:
            names = holding(used)
            for a, b in listed:
                add("disjoint", (a, b), names)
                add("classes", (a,), names)
                add("classes", (b,), names)
    return out


def single_part(src: OntologySource, name: str, by: str) -> dict:
    """The record of a source read from one part: every axiom comes from it."""
    return record(src, {name: source_axioms(src)}, by)


_SPLIT = re.compile(r"[\s|<>\"]+")


def named_terms(*values: Any) -> set[str]:
    """Every token of the strings in ``values`` (dict keys and values, lists, recursively), split on white space, "|",
    angle brackets and quotes. A token ending in a colon also stands for itself without that one colon: prose after a
    term, as in a provenance requirement "<node> <p> some <D>: flag"; the token itself is kept, since an IRI may end in
    a colon (only a token that is a term of the source counts)."""
    out: set[str] = set()
    stack = list(values)
    while stack:
        v = stack.pop()
        if isinstance(v, str):
            for t in _SPLIT.split(v):
                if t:
                    out.add(t)
                    if len(t) > 1 and t.endswith(":"):
                        out.add(t[:-1])
        elif isinstance(v, Mapping):
            stack.extend(v.keys())
            stack.extend(v.values())
        elif isinstance(v, (list, tuple, set, frozenset)):
            stack.extend(v)
    return out


class FlagOrigin:
    """The origin of each flag on one source (``of``), its hierarchy's up-closures kept between flags."""

    def __init__(self, src: OntologySource):
        self.src = src
        self.rec = src.origin or {}
        self.parts: list[str] = list(self.rec.get("parts", []))
        self._up: dict[str, frozenset[str]] = {}

    def up(self, c: str) -> frozenset[str]:
        """``c`` and its ancestors in the source's own hierarchy."""
        if c not in self._up:
            seen, stack = {c}, [c]
            while stack:
                for p in self.src.classes[stack.pop()].parents:
                    if p not in seen and p in self.src.classes:
                        seen.add(p)
                        stack.append(p)
            self._up[c] = frozenset(seen)
        return self._up[c]

    def of(self, key: str, value: Any, check_type: str, independent: bool = False) -> list[str]:
        """The origin of a flag of ``check_type`` (``outcomes.CHECKS``) whose key and evidence are ``key`` and
        ``value``; ``independent``: a check that reads no source."""
        if independent:
            return []
        src, rec = self.src, self.rec
        terms = named_terms(key, value)
        classes = sorted(t for t in terms if t in src.classes)
        props = sorted(t for t in terms if t in src.properties)
        found: set[str] = set()
        for p in props:
            for kind in ("domain", "range", "datatype_range"):
                for names in rec.get(kind, {}).get(p, {}).values():
                    found.update(names)
            found.update(rec.get("range_values", {}).get(p, []))
        ups = {c: self.up(c) for c in classes}
        for above in ups.values():
            for a in above:
                for names in rec.get("subclass", {}).get(a, {}).values():
                    found.update(names)
        if check_type == "disjointness":
            for a, b in (tuple(sorted(x)) for x in src.disjoint):
                if any(a in ups[c] and b in ups[d] for c in classes for d in classes if c != d):
                    found.update(rec.get("disjoint", {}).get(a, {}).get(b, []))
        if check_type == "provenance":
            named = set(classes)
            for e in src.existentials:
                if e.cls in named or (e.prop in props and e.filler in named):
                    found.update(rec.get("existentials", {}).get(e.cls, {}).get(e.prop, {}).get(e.filler, []))
        if not found:  # no axiom but the named terms' declarations, or not even those
            for t in classes:
                found.update(rec.get("classes", {}).get(t, []))
            for t in props:
                found.update(rec.get("properties", {}).get(t, []))
        if not found:
            return list(self.parts)
        return [n for n in self.parts if n in found]
