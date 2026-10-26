"""The conformance labeller: does a mutation violate what a source's axioms say?

This is the fault bank's second, independent reading of the sources (issue E1.3, criterion 4).
It must not share code with the check compiler that is later scored against its labels, so it
imports nothing from ``harness`` at all: it reads the ontology-source-v1 JSON record, the shared
hierarchy JSON, and the ``site`` a mutation record carries (the mutated term and, for each node it
touches, the classes the query asserts, the other patterns it is used in, and for constants the
classes the store gives it). Its class closure is a deliberate second implementation beside the
compiler's (check contract rule 35's "single implementation" governs the checks, not this
labeller); comparing the two on a sample is the point.

Readings, all from the source's own axioms plus the shared hierarchy (rule 16), written from the
contract text (s01-check-contract.md), never from the compiler:

    vocabulary     a predicate must be a property of the source; a class must be a class of the
                   source or of the shared hierarchy (rule 25); data-node constants are exempt
                   (rule 32); in a class position a term is a data node when (a) the source does not
                   declare it as a class, (b) in the store it is a membership target or an entity
                   modelled as a class, and the subject of a data-graph triple whose predicate is
                   outside RDF, RDFS and OWL, and (c) it is never a predicate (rule 32 as amended,
                   change-log entry 26), read from the store facts the mutation record carries; any
                   other term there, an IRI absent from the store included, is a vocabulary term
                   (rule 37); the terms defined in the published RDF, RDFS, OWL 2 and XSD
                   vocabularies are exempt for every source (rule 36), read from the one shared term
                   list ``data/vocab/w3c-terms.json``, whose hash is checked
    store classes  (a constant's classes in the data) are evidence, read closed-world: a node whose
                   known classes are all outside a required domain or range violates it
    query classes  (asserted by the query) and the domain or range of the node's other patterns are
                   requirements: two requirements conflict when no class of the source lies under
                   both (no common subclass, or explicit disjointness)
    datatype range (rule 11; range checks dispatch on the range's target): a swapped-in predicate
                   whose range is a datatype makes its object a literal, which cannot be a subject,
                   carry a class, or fill a class range elsewhere; a literal cannot be the subject of
                   the swapped-in predicate; a literal constant must have a declared datatype (after
                   the scorer's normalisation, on the literal and on the range: the whole numeric
                   family, integer and its subtypes, decimal, float and double, is one datatype,
                   change-log entry 93; any other datatype compares exactly), and cannot fill a class
                   range
    census         (rule 20, a census record with ``position_support``; clarified by change-log
                   entry 30)
                   support counts upwards: a class C is supported at a position when a class listed
                   there is C or under C; a class below a listed class is not supported by it
                   the census's entries are alternatives (a union), never rule 2's intersection: its
                   domain or range groups are read as one group, and a class and a datatype listed
                   at one position are either-or (a literal of a listed datatype, or a resource)
                   join: an untyped node at two (predicate, position) pairs violates the census when
                   no class or datatype is supported at both; a position with nothing listed is
                   unknown when its untyped fillers reach the threshold, and has no observed class
                   otherwise
                   class pattern: a typed node's classes must be supported at each position it fills
                   single pattern: a literal membership pattern ``?x m C`` is empty when C's direct
                   count is 0 (the threshold does not apply)

The difference reading (design ruling in the design log): every check above runs on each node the
mutation touches, at each of its positions, once with the mutated term and once with the gold's
(``site["original"]``). A mutation violates a source when the mutated query raises a violation the
gold query does not, matched by check type, position and binding:

    check type     vocabulary, domain, range, class, census_support, census_pattern
    position       the mutated pattern or class ("site:subject", "site:object", "site:class"; the same
                   position in both queries), another pattern ("<predicate> <subject|object>"),
                   another class assertion ("class <C>"), or a census join ("<position> | <position>")
    binding        the node's term; a swapped or invented constant, predicate or class is its own
                   binding, so its flags are always the mutation's

A conflict between two positions is flagged at each of them, so a class swap on ``?x a C`` that
breaks the domain of another pattern of ?x counts there. Flags the gold raises too are dropped and
kept as ``discounted``. Without an original term, the gold side is the node without the mutated
pattern.

Labels per source: violates, conforms (an applicable axiom involving the mutated position was
evaluated, and the mutation adds no violation), undetermined (such an axiom applies but the node has
nothing to test it with), no_axiom, not_applicable (aggregation).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

THING = frozenset({"http://www.w3.org/2002/07/owl#Thing", "http://www.w3.org/2000/01/rdf-schema#Resource"})
XSD = "http://www.w3.org/2001/XMLSchema#"
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
OWL = "http://www.w3.org/2002/07/owl#"
# rule 36: the one term list shared with the compiler, as data (feat/checker 48d7781 has its builder)
W3C_TERMS_PATH = Path(__file__).resolve().parents[2] / "data" / "vocab" / "w3c-terms.json"
W3C_TERMS_SHA256 = "790efdc207dde6849f97db3c6b0bac7bb1e25dfb61ffd853097f5ceb46c579b8"


def load_w3c_terms(path: Path = W3C_TERMS_PATH) -> frozenset[str]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != W3C_TERMS_SHA256:
        raise ValueError(f"{path}: sha256 {digest} is not the shared W3C term list {W3C_TERMS_SHA256}")
    return frozenset(json.loads(raw)["terms"])


W3C_TERMS = load_w3c_terms()


def is_data_node(term: str, facts: dict, source: dict) -> bool:
    """Rule 32's data-node test, from its amended text (change-log entry 26). ``facts`` are store
    facts about ``term``: membership_target (an object of rdf:type, or of rdfs:subClassOf inside the
    data graph), modelled_as_class (the subject of rdfs:subClassOf to an IRI inside the data graph),
    data_subject (the subject of a data-graph triple whose predicate is outside the RDF, RDFS and OWL
    namespaces) and used_as_predicate."""
    declared = term in source.get("classes", {})  # (a)
    in_store = (facts.get("membership_target") or facts.get("modelled_as_class")) and facts.get("data_subject")  # (b)
    return not declared and bool(in_store) and not facts.get("used_as_predicate")  # (c)
RDF_TYPE = RDF + "type"
RDFS_SUBCLASSOF = RDFS + "subClassOf"
MEMBERSHIP = {"rdf:type": RDF_TYPE, "rdfs:subClassOf": RDFS_SUBCLASSOF, RDF_TYPE: RDF_TYPE, RDFS_SUBCLASSOF: RDFS_SUBCLASSOF}
# rule 11 as amended (change-log entry 93): the scorer's numeric family counts as one datatype, on the literal and
# on the range, because the engine does not keep the stored numeric datatype (stored doubles come back as decimals).
# Written here from the contract text, not taken from the checker (this module imports nothing from the harness).
INTEGER_TYPES = frozenset(XSD + t for t in (
    "integer", "int", "long", "short", "byte", "nonNegativeInteger", "positiveInteger", "negativeInteger",
    "nonPositiveInteger", "unsignedInt", "unsignedLong", "unsignedShort", "unsignedByte"))
NUMERIC_TYPES = INTEGER_TYPES | {XSD + "decimal", XSD + "float", XSD + "double"}
NUMERIC = XSD + "decimal"  # the family's one name; every integer is a decimal
_INTEGER = re.compile(r"^[+-]?\d+$")
_DECIMAL = re.compile(r"^[+-]?(\d+\.\d*|\.\d+)$")


def normalise_datatype(dt: str) -> str:
    """Every numeric datatype is ``NUMERIC``; any other compares exactly."""
    return NUMERIC if dt in NUMERIC_TYPES else dt


def _is_datatype(term: str) -> bool:
    return term.startswith(XSD) or term == RDF + "langString"


def literal_datatype(term: str) -> str:
    """The datatype of a query literal written as ``lit:<text>`` in a site record."""
    text = term[4:]
    if text in ("true", "false"):
        return XSD + "boolean"
    if _INTEGER.match(text):
        return XSD + "integer"
    if _DECIMAL.match(text):
        return XSD + "decimal"
    if text[:1] in "+-.0123456789":
        return XSD + "double"
    quote = text[:3] if text[:3] in ('"' * 3, "'" * 3) else text[:1]
    end = text.rfind(quote)
    rest = text[end + len(quote):] if end >= len(quote) else ""
    if rest.startswith("^^"):
        return rest[2:].strip("<>")
    if rest.startswith("@"):
        return RDF + "langString"
    return XSD + "string"


@dataclass(frozen=True)
class Pos:
    """A position of a node: a pattern it fills (``predicate`` at ``position``) or a class the query
    asserts of it. The mutated pattern or class has the same id in the mutated and the gold query."""

    id: str
    predicate: str | None = None
    position: str | None = None  # subject or object
    cls: str | None = None
    site: bool = False


def positions(node: dict, extra: Pos | None) -> list[Pos]:
    out = [Pos(f"{u['predicate']} {u['position']}", u["predicate"], u["position"])
           for u in node.get("uses", []) if u["position"] in ("subject", "object")]
    out += [Pos(f"class {c}", cls=c) for c in node.get("classes", [])]
    if extra is not None:
        out.append(extra)
    return list({p.id: p for p in out}.values())


@dataclass
class Checks:
    """Violations by (check type, position, binding), and whether a check involving the mutated
    position was evaluated (tested) or had nothing to test against (untested)."""

    violations: dict[tuple[str, str, str], list[dict]] = field(default_factory=dict)
    tested: bool = False
    untested: bool = False

    def met(self, involved: bool) -> None:
        self.tested |= involved

    def lacking(self, involved: bool) -> None:
        self.untested |= involved

    def flag(self, key: tuple[str, str, str], involved: bool, detail: dict) -> None:
        self.tested |= involved
        self.violations.setdefault(key, []).append({"constraint": key[0], "position": key[1], "binding": key[2], **detail})


@dataclass
class Labeller:
    source: dict
    shared: dict
    _parents: dict[str, set[str]] = field(default_factory=dict, repr=False)
    _children: dict[str, set[str]] | None = field(default=None, repr=False)
    _anc: dict[str, frozenset[str]] = field(default_factory=dict, repr=False)
    _desc: dict[str, frozenset[str]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for c, ps in self.shared.get("parents", {}).items():
            self._parents.setdefault(c, set()).update(ps)
        for c, info in self.source.get("classes", {}).items():
            self._parents.setdefault(c, set()).update(info.get("parents", []))
        self.disjoint = [tuple(p) for p in self.source.get("disjoint", [])]
        self.census = self.source.get("position_support") is not None

    # -- closure ----------------------------------------------------------------------------------
    def ancestors(self, c: str) -> frozenset[str]:
        """c and every class above it."""
        if c not in self._anc:
            seen, stack = {c}, [c]
            while stack:
                for p in self._parents.get(stack.pop(), ()):
                    if p not in seen:
                        seen.add(p)
                        stack.append(p)
            self._anc[c] = frozenset(seen)
        return self._anc[c]

    def descendants(self, c: str) -> frozenset[str]:
        """c and every class below it."""
        if self._children is None:
            self._children = {}
            for child, ps in self._parents.items():
                for p in ps:
                    self._children.setdefault(p, set()).add(child)
        if c not in self._desc:
            seen, stack = {c}, [c]
            while stack:
                for ch in self._children.get(stack.pop(), ()):
                    if ch not in seen:
                        seen.add(ch)
                        stack.append(ch)
            self._desc[c] = frozenset(seen)
        return self._desc[c]

    def is_disjoint(self, a: str, b: str) -> bool:
        aa, bb = self.ancestors(a), self.ancestors(b)
        return any((x in aa and y in bb) or (y in aa and x in bb) for x, y in self.disjoint)

    def compatible(self, a: str, b: str) -> bool:
        """Some class lies under both, and they are not declared disjoint."""
        if self.is_disjoint(a, b):
            return False
        if b in self.ancestors(a) or a in self.ancestors(b):
            return True
        return bool(self.descendants(a) & self.descendants(b))

    # -- the source's axioms --------------------------------------------------------------------------
    def groups(self, predicate: str, position: str) -> list[frozenset[str]]:
        info = self.source.get("properties", {}).get(predicate)
        if not info:
            return []
        key = "domain" if position == "subject" else "range"
        out = []
        for g in info.get(key, []):
            members = frozenset(g.get("members", []))
            if members and not (members & THING):  # rule 1: owl:Thing is NA
                out.append(members)
        if self.census and len(out) > 1:  # observed entries are alternatives (rule 20, entry 30)
            return [frozenset().union(*out)]
        return out

    def in_vocabulary(self, term: str, kind: str) -> bool:
        if term in W3C_TERMS:  # rule 36: defined terms only, not whole namespaces
            return True
        if kind == "predicate":
            return term in self.source.get("properties", {})
        return term in self.source.get("classes", {}) or term in self.shared.get("parents", {}) \
            or any(term in ps for ps in self.shared.get("parents", {}).values())

    def listed(self, predicate: str, position: str) -> list[str]:
        """Census only: the classes listed at (predicate, position)."""
        ps = self.source.get("position_support") or {}
        return [c for c in ps.get(predicate, {}).get(position, {}) if not _is_datatype(c)]

    def listed_datatypes(self, predicate: str, position: str) -> list[str]:
        """Census only: the literal datatypes listed at (predicate, position), normalised."""
        ps = self.source.get("position_support") or {}
        return sorted({normalise_datatype(c) for c in ps.get(predicate, {}).get(position, {}) if _is_datatype(c)})

    def supports(self, c: str, listed: list[str]) -> bool:
        """Rule 20 as clarified (entry 30): a listed class at or below ``c`` supports it."""
        return any(c in self.ancestors(x) for x in listed)

    def untyped_at_threshold(self, predicate: str, position: str) -> bool:
        n = (self.source.get("untyped_fillers") or {}).get(predicate, {}).get(position, 0)
        return n >= (self.source.get("support_threshold") or 0) and n > 0

    def datatype_range(self, predicate: str) -> frozenset[str] | None:
        """The declared datatype range (rule 11), normalised; None when the range is not a datatype
        (a class range, or no range at all)."""
        info = self.source.get("properties", {}).get(predicate)
        if not info:
            return None
        dts = set(info.get("datatype_range", []))
        classes = False
        for g in info.get("range", []):
            for m in g.get("members", []):
                if m.startswith(XSD) or m == RDF + "langString":
                    dts.add(m)
                else:
                    classes = True
        if not dts or (classes and not self.census):  # in an axiom, a class beside a datatype is rule 2's intersection
            return None
        return frozenset(normalise_datatype(d) for d in dts)

    def either_or(self, predicate: str) -> bool:
        """A census range listing a class and a datatype: the object is a literal of a listed
        datatype or a resource, either fits (rule 20, entry 30)."""
        return self.census and self.datatype_range(predicate) is not None \
            and any(not _is_datatype(m) for g in self.groups(predicate, "object") for m in g)

    def class_range(self, predicate: str) -> bool:
        if self.either_or(predicate):
            return False
        return any(not m.startswith(XSD) for g in self.groups(predicate, "object") for m in g)

    def check_position(self, predicate: str, position: str) -> tuple[str, list[str]]:
        """Census side of a join: "ok" with the supported classes, "unknown" for untyped fillers at
        the threshold, or "none" when no class is observed there."""
        listed = self.listed(predicate, position)
        datatypes = self.listed_datatypes(predicate, position)
        if listed or datatypes:  # alternatives: every class some listed class supports, and every listed datatype
            out: set[str] = set(datatypes)
            for c in listed:
                out |= self.ancestors(c)
            return "ok", sorted(out)
        return ("unknown" if self.untyped_at_threshold(predicate, position) else "none"), []

    def implications(self, p: Pos) -> list[frozenset[str]]:
        """The class requirements a position puts on its node: its class, or its domain or range."""
        return [frozenset([p.cls])] if p.cls else self.groups(p.predicate, p.position)

    # -- the checks on one node, by (check type, position, binding) ---------------------------------------
    def node_checks(self, node: dict, extra: Pos | None, evidence: list[str], out: Checks) -> None:
        """Every check on a node the mutation touches. ``extra`` is the mutated pattern's position on
        it (None: the node as if that pattern were absent)."""
        term = node["term"]
        ps = positions(node, extra)
        literal = term.startswith("lit:")
        if not literal:
            self._compatibility(term, ps, evidence, out)
            if self.census:
                self._census(term, ps, out)
        self._datatypes(term, ps, literal, not term.startswith(("?", "_:", "lit:")), out)

    def _compatibility(self, term: str, ps: list[Pos], evidence: list[str], out: Checks) -> None:
        """A position's requirement against the node's other requirements (no common subclass, or
        declared disjointness) and, for a constant, against its store classes read closed-world."""
        for u in ps:
            groups = self.implications(u)
            constraint = "class" if u.cls else ("domain" if u.position == "subject" else "range")
            key = (constraint, u.id, term)
            others = [(v, o) for v in ps if v.id != u.id for o in self.implications(v)]
            ev = [] if u.cls else evidence
            for g in groups:
                if not ev and not others:
                    if not u.cls:  # an axiom with nothing to test it against
                        out.lacking(u.site)
                    continue
                if ev:
                    if any(self.ancestors(e) & g for e in ev):
                        out.met(u.site)
                    else:
                        basis = "disjoint" if all(self.is_disjoint(e, x) for e in ev for x in g) else "closed_world"
                        out.flag(key, u.site, {"basis": basis, "required": sorted(g), "found": sorted(ev)})
                for v, o in others:
                    involved = u.site or v.site
                    if any(self.compatible(a, b) for a in o for b in g):
                        out.met(involved)
                    else:
                        basis = "disjoint" if all(self.is_disjoint(a, b) for a in o for b in g) else "no_common_subclass"
                        out.flag(key, involved, {"basis": basis, "required": sorted(g), "against": sorted(o), "at": v.id})

    def _census(self, term: str, ps: list[Pos], out: Checks) -> None:
        """Rule 20 (entry 30): a typed node's classes must be supported at each position it fills;
        an untyped node needs a class or datatype supported at both positions of each join."""
        uses = [p for p in ps if p.predicate]
        classes = [p for p in ps if p.cls]
        if classes:
            for u in uses:
                key = ("census_support", u.id, term)
                state, _ = self.check_position(u.predicate, u.position)
                if state == "unknown":
                    out.lacking(u.site or any(c.site for c in classes))
                elif state == "none":
                    out.flag(key, u.site or any(c.site for c in classes), {"basis": "no_observed_class", "pair": [u.id]})
                else:
                    listed = self.listed(u.predicate, u.position)
                    for c in classes:
                        if self.supports(c.cls, listed):
                            out.met(u.site or c.site)
                        else:
                            out.flag(key, u.site or c.site, {"basis": "class_pattern", "pair": [u.id, c.id]})
            return
        if len(uses) < 2:  # a join needs two positions
            return
        states = {u.id: self.check_position(u.predicate, u.position) for u in uses}
        for u in uses:
            if states[u.id][0] == "none":
                out.flag(("census_support", u.id, term), u.site, {"basis": "no_observed_class", "pair": [u.id]})
        for i, u in enumerate(uses):
            for v in uses[i + 1:]:
                (su, mine), (sv, theirs) = states[u.id], states[v.id]
                involved = u.site or v.site
                if "none" in (su, sv):
                    continue
                if "unknown" in (su, sv):
                    out.lacking(involved)
                    continue
                pair = sorted([u.id, v.id])
                if set(mine) & set(theirs):
                    out.met(involved)
                else:
                    out.flag(("census_support", " | ".join(pair), term), involved, {"basis": "no_supported_class", "pair": pair})

    def _datatypes(self, term: str, ps: list[Pos], literal: bool, iri: bool, out: Checks) -> None:
        """Rule 11 at each position: a literal cannot be a subject or fill a class range; a datatype
        range takes a literal of a listed datatype, never a resource."""
        def makes_literal(v: Pos) -> bool:
            return v.position == "object" and self.datatype_range(v.predicate) is not None and not self.either_or(v.predicate)

        def makes_resource(v: Pos) -> bool:
            return v.cls is not None or v.position == "subject" or (v.position == "object" and self.class_range(v.predicate))

        for u in ps:
            if not u.predicate:
                continue
            lits = [v for v in ps if v.id != u.id and makes_literal(v)]
            because = (["literal constant"] if literal else []) + [v.id for v in lits]
            involved = u.site or any(v.site for v in lits)
            if u.position == "subject":
                if because:
                    out.flag(("domain", u.id, term), involved, {"basis": "literal_in_subject_position", "because": because})
                continue
            key = ("range", u.id, term)
            dts = self.datatype_range(u.predicate)
            if dts is not None:
                if literal:
                    if normalise_datatype(literal_datatype(term)) in dts:
                        out.met(u.site)
                    else:
                        out.flag(key, u.site, {"basis": "wrong_datatype", "required": sorted(dts),
                                               "found": literal_datatype(term)})
                elif not self.either_or(u.predicate):  # a census range listing a class takes a resource too
                    res = [v for v in ps if v.id != u.id and makes_resource(v)]
                    subjects = [v for v in res if v.position == "subject"]
                    if subjects:
                        out.flag(key, u.site or any(v.site for v in subjects),
                                 {"basis": "literal_in_resource_position", "because": [v.id for v in subjects]})
                    elif res or iri:
                        out.flag(key, u.site or any(v.site for v in res), {"basis": "resource_where_literal_required",
                                                                           "because": [v.id for v in res] + (["IRI constant"] if iri else [])})
                    else:
                        out.lacking(u.site)
            elif self.class_range(u.predicate) and because:
                out.flag(key, involved, {"basis": "literal_where_class_required", "because": because})

    def _single_pattern(self, term: str, cls: str, membership: dict | None, out: Checks) -> None:
        """Rule 20's single pattern for the mutated class: ``?x m C`` is empty when C's direct count is 0."""
        if not self.census or not membership or membership.get("modifier"):
            return
        pred = MEMBERSHIP.get(membership["predicate"], membership["predicate"])
        if pred == MEMBERSHIP.get(self.source.get("membership", ""), ""):
            direct = (self.source.get("type_support") or {}).get(cls, 0)
        else:
            direct = sum(p.get("support", 0) for p in self.source.get("patterns", [])
                         if p.get("predicate") == pred and p.get("object_class") == cls)
        if direct:
            out.met(True)
        else:
            out.flag(("census_pattern", "site:class", term), True, {"basis": "census_empty_pattern", "class": cls})

    def _constant(self, site: dict) -> Checks:
        """A swapped, invented or stale constant is its own binding, so every flag on it is the
        mutation's: rules 32 and 37 in a class position, and its store classes read closed-world
        against each position's domain or range (rule 5)."""
        out = Checks()
        node, term = site["node"], site["term"]
        if "store_facts" in node and any(u["position"] == "membership_object" for u in node.get("uses", [])) \
                and not is_data_node(term, node["store_facts"], self.source) and not self.in_vocabulary(term, "class"):
            out.flag(("vocabulary", "site:class-position", term), True,
                     {"basis": "absent", "term": term, "rule": "32/37: not a data node, in a class position"})
        for u in node.get("uses", []):
            if u["position"] in ("subject", "object"):  # each position alone: the constant's classes are the evidence
                self._compatibility(term, [Pos(f"{u['predicate']} {u['position']}", u["predicate"], u["position"], site=True)],
                                    node.get("store_classes", []), out)
        return out

    def label(self, site: dict) -> dict:
        """The difference reading (design ruling in the design log): the mutation violates the source when
        the mutated query raises a violation the gold query does not, matched by check type, position
        and binding. Everything outside the nodes the mutation touches is the same in both queries."""
        kind = site.get("kind")
        if kind == "aggregation":
            return {"label": "not_applicable", "violations": [], "discounted": []}
        term = site["term"]
        if kind == "constant":
            mutated, gold = self._constant(site), Checks()
        else:
            mutated, gold = Checks(), Checks()
            original = site.get("original")
            nodes = [(site["subject"], "subject"), (site["object"], "object")] if kind == "predicate" else [(site["node"], None)]
            for node, position in nodes:
                evidence = node.get("store_classes") or [] if kind == "predicate" else []  # a class swap reads the query
                for state, t in ((mutated, term), (gold, original)):
                    extra = None if t is None else (Pos(f"site:{position}", t, position, site=True) if position
                                                    else Pos("site:class", cls=t, site=True))
                    self.node_checks(node, extra, evidence, state)
                    if kind == "class" and t is not None:
                        self._single_pattern(node["term"], t, site.get("membership"), state)
            class_term = kind == "class" and not ("store_facts" in site and is_data_node(term, site["store_facts"], self.source))
            if (kind == "predicate" or class_term) and not self.in_vocabulary(term, kind):
                mutated.flag(("vocabulary", f"site:{kind}", term), True, {"basis": "absent", "term": term})
        violations = [v for key, vs in sorted(mutated.violations.items()) if key not in gold.violations for v in vs]
        discounted = [{"constraint": k[0], "position": k[1], "binding": k[2]} for k in sorted(mutated.violations)
                      if k in gold.violations]
        if violations:
            label = "violates"
        elif mutated.tested:
            label = "conforms"
        elif mutated.untested:
            label = "undetermined"
        else:
            label = "no_axiom"
        return {"label": label, "violations": violations, "discounted": discounted}


def label_all(site: dict, labellers: dict[str, Labeller]) -> dict[str, dict]:
    return {name: lab.label(site) for name, lab in sorted(labellers.items())}
