"""The subclass closure of one source, computed once and reused by every check.

Only the frozen OWL fragment is read (§7.2): named classes and their direct
parents, domain and range as a named class or a union of named classes,
disjointness between named classes. Inheritance through rdfs:subPropertyOf is
not applied (S0.1 ruling 3).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from itertools import chain, product

from harness.checks.numbers import NUMERIC_DATATYPES
from harness.sources.model import ClassExpr, OntologySource

RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
OWL = "http://www.w3.org/2002/07/owl#"
XSD = "http://www.w3.org/2001/XMLSchema#"
RDF_TYPE = RDF + "type"
RDFS_SUBCLASSOF = RDFS + "subClassOf"

# A domain or range of owl:Thing or rdfs:Resource says nothing (ruling 1);
# rdfs:Literal says nothing about a datatype.
VACUOUS_CLASSES = frozenset({OWL + "Thing", RDFS + "Resource"})
VACUOUS_DATATYPES = frozenset({RDFS + "Literal"})

# Types every node of a schema-level graph carries; they are not memberships.
META_TYPES = frozenset({
    RDFS + "Class", OWL + "Class", RDFS + "Resource", OWL + "Thing", OWL + "NamedIndividual",
    RDF + "Property", OWL + "ObjectProperty", OWL + "DatatypeProperty", OWL + "AnnotationProperty",
    RDFS + "Datatype", OWL + "Ontology", OWL + "Restriction",
})

# Vocabulary that belongs to RDF, RDFS, OWL and XSD rather than to any source.
BUILTIN_NAMESPACES = (RDF, RDFS, OWL, XSD)

def accepted_datatypes(datatype: str) -> frozenset[str]:
    """The literal datatypes that satisfy a datatype range.

    Rule 11 as amended (change-log entry 93): the scorer's numeric family (xsd:integer and its 12 derived types,
    xsd:decimal, xsd:float, xsd:double; ``numbers.NUMERIC_DATATYPES`` is the checker's one copy of it) counts as one
    datatype on both sides, so a literal of any member satisfies a range naming any member. The store's result
    datatypes within the family are not reliable (QLever serialises stored xsd:double values as xsd:decimal). Every
    other datatype compares exactly; a plain literal has datatype xsd:string in RDF 1.1, so it satisfies xsd:string.
    """
    if datatype in NUMERIC_DATATYPES:
        return NUMERIC_DATATYPES
    return frozenset({datatype})


def is_builtin(iri: str) -> bool:
    return iri.startswith(BUILTIN_NAMESPACES)


def is_datatype(iri: str) -> bool:
    return iri.startswith(XSD) or iri == RDFS + "Literal"


@dataclass
class Requirement:
    """What the source requires of the node on one side of a predicate."""

    exprs: list[ClassExpr] = field(default_factory=list)  # all must hold (ruling 2)
    datatypes: list[str] = field(default_factory=list)  # declared datatype ranges, all must hold (any_of: one)
    enums: list[list[dict]] = field(default_factory=list)  # owl:oneOf data ranges (ruling 14)
    vacuous: bool = False  # an owl:Thing / rdfs:Resource / rdfs:Literal declaration was dropped (ruling 1)
    unsupported: bool = False  # a conjunct outside the fragment was left out (rule 13: NA, unsupported_axiom)
    # A census's entries at a position are alternatives (rule 20, change-log entry 30): then
    # ``exprs`` holds at most one union and a value fits when it fits it or any of ``datatypes``.
    any_of: bool = False

    @property
    def any(self) -> bool:
        return bool(self.exprs or self.datatypes or self.enums)

    def show(self) -> list[str]:
        if self.any_of:
            return ["|".join([*(m for e in self.exprs for m in e.members), *self.datatypes])] if self.any else []
        out = ["|".join(e.members) for e in self.exprs] + list(self.datatypes)
        out += ["oneOf(" + ",".join(v["lexical"] for v in en) + ")" for en in self.enums]
        return out + (["<unsupported conjunct, not checked>"] if self.unsupported else [])


class Closure:
    def __init__(self, source: OntologySource):
        self.source = source
        parents: dict[str, set[str]] = {}
        for iri, info in source.classes.items():
            parents.setdefault(iri, set()).update(info.parents)
        mentioned: set[str] = set(parents)
        for ps in parents.values():
            mentioned.update(ps)
        for prop in source.properties.values():
            for expr in (*prop.domain, *prop.range):
                mentioned.update(m for m in expr.members if not is_datatype(m))
        for a, b in source.disjoint:
            mentioned.update((a, b))
        for ex in source.existentials:
            mentioned.update((ex.cls, ex.filler))
        for pat in source.patterns:
            mentioned.add(pat.subject_class)
            if pat.object_class:
                mentioned.add(pat.object_class)
        # A term of the RDF, RDFS or OWL namespaces is never a named class: a census keeps literal
        # rdf:type patterns on meta classes (owl:Class, owl:Restriction, ...) for rule 20's counts only.
        self.classes: frozenset[str] = frozenset(
            c for c in mentioned if c not in VACUOUS_CLASSES and not c.startswith((RDF, RDFS, OWL)))

        self.parents: dict[str, set[str]] = parents
        # Rule 32's data nodes, which the compiler reads from the store (none without it): entity constants, never
        # classes; by rule 40 a data node in a class position implies no class requirement.
        self.data_nodes: frozenset[str] = frozenset()
        self._anc: dict[str, frozenset[str]] = {}
        for c in self.classes:
            seen = {c}
            stack = [c]
            while stack:
                for p in parents.get(stack.pop(), ()):
                    if p not in seen:
                        seen.add(p)
                        stack.append(p)
            self._anc[c] = frozenset(seen)
        self._desc: dict[str, set[str]] = {c: set() for c in self.classes}
        for c, ancs in self._anc.items():
            for a in ancs:
                self._desc.setdefault(a, set()).add(c)

        self.disjoint_pairs: frozenset[frozenset[str]] = frozenset(frozenset(p) for p in source.disjoint if p[0] != p[1])
        self._in_axiom: frozenset[str] = frozenset(c for p in self.disjoint_pairs for c in p)

        self.properties = source.properties
        # Vocabulary existence counts every term the source declares or describes (change-log entry 94): a term with
        # only a label is no class (``source.described``), but it is known vocabulary.
        self.vocabulary: frozenset[str] = frozenset(
            set(self.classes) | set(source.properties) | {p.predicate for p in source.patterns}
            | set(source.described or ())
        )

    # -- hierarchy ---------------------------------------------------------

    def ancestors(self, c: str) -> frozenset[str]:
        """Reflexive, transitive ancestors of a class known to the source."""
        return self._anc.get(c, frozenset({c}))

    def under(self, t: str, c: str) -> bool:
        return c in self.ancestors(t)

    def known(self, c: str) -> bool:
        return c in self.classes

    def merged(self, store_parents, store_children=None, store_classes=None, data_nodes=frozenset(),
               **search) -> "MergedClosure":
        """Rule 16: one closure from the store's subclass triples plus this
        source's own subclass axioms."""
        return MergedClosure(self, store_parents, store_children, store_classes, data_nodes, **search)

    def is_data_node(self, c: str) -> bool:
        return c in self.data_nodes

    def strictly_above(self, nodes: Iterable[str]) -> set[str]:
        """The union, over ``nodes``, of each node's ancestors other than itself."""
        out: set[str] = set()
        for n in nodes:
            out.update(a for a in self.ancestors(n) if a != n)
        return out

    def falls_under(self, types: Iterable[str], expr: ClassExpr) -> bool:
        """Some type falls under some member of the expression."""
        return any(self.under(t, m) for t in types for m in expr.members)

    # -- disjointness ------------------------------------------------------

    def disjoint(self, a: str, b: str) -> bool:
        """Explicitly disjoint, by an own or an inherited axiom."""
        if not self.disjoint_pairs:
            return False
        anc_a, anc_b = self.ancestors(a), self.ancestors(b)
        return any(frozenset((x, y)) in self.disjoint_pairs for x in anc_a for y in anc_b if x != y)

    def touched(self, c: str) -> bool:
        """Some disjointness axiom applies to the class, its own or inherited."""
        return bool(self.ancestors(c) & self._in_axiom)

    def exprs_disjoint(self, exprs: list[ClassExpr], cap: int = 4096) -> bool | None:
        """True when every choice of one member per expression contains an
        explicitly disjoint pair; None when there are too many choices."""
        choices = 1
        for e in exprs:
            choices *= len(e.members)
        if choices > cap:
            return None
        for pick in product(*(e.members for e in exprs)):
            if not any(self.disjoint(pick[i], pick[j]) for i in range(len(pick)) for j in range(i + 1, len(pick))):
                return False
        return True

    def common_named_subclass(self, exprs: list[ClassExpr]) -> str | None:
        """A named class of the closure under some member of every expression
        (for a union, at least one member must meet the other classes), or None."""
        common: set[str] | None = None
        for e in exprs:
            below: set[str] = set()
            for m in e.members:
                below |= self._desc.get(m, {m} if m in self.classes else set())
            common = below if common is None else common & below
            if not common:
                return None
        return min(common) if common else None

    def disjoint_with_all(self, types: Iterable[str], expr: ClassExpr) -> bool:
        """Every member of the expression is disjoint with some asserted type;
        then open-world semantics cannot accept the node either."""
        types = list(types)
        return all(any(self.disjoint(t, m) for t in types) for m in expr.members)

    # -- domain and range --------------------------------------------------

    def unsupported_axiom(self, predicate: str, side: str) -> bool:
        """Rule 3 and 13: the source states something here the compiler does not
        compile (an inherited declaration through rdfs:subPropertyOf, or a
        conjunct outside the fragment)."""
        prop = self.properties.get(predicate)
        if prop is None:
            return False
        return bool(prop.parents) or bool(getattr(prop, "unsupported_domain" if side == "subject" else "unsupported_range", False))

    def requirement(self, predicate: str, side: str) -> Requirement:
        """What the source requires of the node on one side of a predicate.
        rdfs:subPropertyOf is not followed (rule 3)."""
        prop = self.properties.get(predicate)
        if prop is None:
            return Requirement()
        exprs = prop.domain if side == "subject" else prop.range
        datatypes = list(prop.datatype_range) if side == "object" else []
        kept: list[ClassExpr] = []
        vacuous = False
        unsupported_dt = False
        for e in exprs:
            if any(m in VACUOUS_CLASSES for m in e.members):
                vacuous = True  # a union with owl:Thing is owl:Thing
                continue
            if any(is_datatype(m) for m in e.members):
                # rule 11: a range dispatches on its target, whatever the property's
                # kind; a single XSD datatype is a datatype range, a union with one
                # is outside the fragment (NA, unsupported_axiom)
                if len(e.members) == 1:
                    datatypes.append(e.members[0])
                else:
                    unsupported_dt = True
                continue
            kept.append(e)
        dts = [d for d in datatypes if d not in VACUOUS_DATATYPES]
        vacuous = vacuous or len(dts) < len(datatypes)
        values = getattr(prop, "range_values", []) if side == "object" else []
        unsupported = getattr(prop, "unsupported_domain" if side == "subject" else "unsupported_range", False)
        if self.source.kind == "census":
            # Rule 20 (change-log entry 30): the census is a set of allowed patterns, so the
            # classes and datatypes listed at a position are alternatives; rule 2's intersection
            # is for axioms (curated and generated sources). A vacuous alternative allows anything.
            if vacuous:
                return Requirement(vacuous=True)
            members = tuple(dict.fromkeys(m for e in kept for m in e.members))
            return Requirement([ClassExpr(members)] if members else [], dts, [list(values)] if values else [], False,
                               bool(unsupported) or unsupported_dt, any_of=True)
        return Requirement(kept, dts, [list(values)] if values else [], vacuous, bool(unsupported) or unsupported_dt)


# The memory cap of one no-common-subclass search: the hierarchy nodes it may hold at once, in its walks, in its
# candidate list and in the walk testing one candidate. A search that would hold more is cut, and its check is unknown
# with reason "bound" (static.py), never a pass or a flag. Deterministic, so a replay gives the same verdict on any
# machine, unlike a cap on the process's measured memory. A held node costs a set entry and a stack slot (the IRIs
# are the hierarchy's own strings): 71 bytes per node by tracemalloc on the largest search of the 756 UniProt
# calibration outputs ([up:Taxon], [up:Enzyme|up:Taxon]: 2,827,544 nodes, 201 MB), so the cap is about 285 MB.
# Every other recorded UniProt argument held at most 2,417 nodes.
COMMON_SUBCLASS_NODE_CAP = 4_000_000
_MEMO_ENTRIES = 4096  # searches remembered per closure; a pure function of its argument, so a cleared memo is safe
_MISSING = object()


class SearchCapExceeded(Exception):
    """A no-common-subclass search would hold more hierarchy nodes than one call may (the memory cap)."""

    def __init__(self, cap: int):
        super().__init__(f"the no-common-subclass search would hold more than {cap} hierarchy nodes")
        self.cap = cap


def _inverse(edges) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for c, ps in edges.items():
        for p in ps:
            out.setdefault(p, set()).add(c)
    return out


def _walk(stack: list[str], seen: set[str], step) -> Iterator[None]:
    """Extend ``seen`` with every node reachable from ``stack`` through ``step``, yielding once per node added."""
    while stack:
        for y in step(stack.pop()):
            if y not in seen:
                seen.add(y)
                stack.append(y)
                yield


class MergedClosure(Closure):
    """A source's closure extended with subclass edges read from the store (rule 16).

    A class outside the source is placed when its ancestors reach a source
    class; ``known`` then holds for it. Disjointness still comes only from the
    source's axioms.

    The common-subclass search reads the store's edges in both directions. ``store_parents_inverse`` (node ->
    nodes whose store parents hold it) and ``store_children_inverse`` (node -> nodes whose store children hold it)
    are the exact inverses of ``store_parents`` and of the children map; the compiler hands in the shared
    hierarchy's own (``closure_children`` and ``parents``), and they are derived here when not given.
    """

    def __init__(self, base: Closure, store_parents, store_children=None, store_classes=None,  # noqa: D107
                 data_nodes=frozenset(), *, store_parents_inverse=None, store_children_inverse=None,
                 search_cap: int = COMMON_SUBCLASS_NODE_CAP):
        self.__dict__.update(base.__dict__)
        self.data_nodes = frozenset(data_nodes)
        # Rule 40: a data node never enters a disjointness pair, also where it is reached as an ancestor (External reviews
        # of a929c39, item 1). Membership still runs through it (rule 35), but an axiom naming it pairs no classes.
        self.disjoint_pairs = frozenset(p for p in base.disjoint_pairs if not p & self.data_nodes)
        self._in_axiom = frozenset(c for p in self.disjoint_pairs for c in p)
        self._base = base
        self._store = store_parents
        self._store_children = store_children
        self._store_classes = store_classes  # None: every node of the store's hierarchy counts as a class
        self._manc: dict[str, frozenset[str]] = {}
        self._store_up_children = store_parents_inverse
        # Without a children map, the search derives one from the store's parents, whose inverse they then are.
        self._store_down_parents = store_parents if store_children is None else store_children_inverse
        self._base_children = _inverse(base.parents)
        self.search_cap = search_cap
        self._memo: dict = {}

    def _walk_up(self, c: str) -> frozenset[str]:
        """Reflexive ancestors through the source's axioms and the store's edges, never through a vacuous class."""
        seen = {c}
        stack = [c]
        while stack:
            x = stack.pop()
            for p in set(self._base.parents.get(x, ())) | set(self._store.get(x, ())):
                if p not in seen and p not in VACUOUS_CLASSES:
                    seen.add(p)
                    stack.append(p)
        return frozenset(seen)

    def ancestors(self, c: str) -> frozenset[str]:
        if c in self._manc:
            return self._manc[c]
        self._manc[c] = self._walk_up(c)
        return self._manc[c]

    def _ancestry_within(self, c: str, limit: int) -> set[str]:
        """The same set as ``_walk_up(c)``, built without a set per node, and cut (SearchCapExceeded) once it would
        hold more than ``limit`` nodes; whether it is cut depends on the set's size alone."""
        seen = {c}
        stack = [c]
        if len(seen) > limit:
            raise SearchCapExceeded(self.search_cap)
        while stack:
            x = stack.pop()
            for p in chain(self._base.parents.get(x, ()), self._store.get(x, ())):
                if p not in seen and p not in VACUOUS_CLASSES:
                    seen.add(p)
                    stack.append(p)
                    if len(seen) > limit:
                        raise SearchCapExceeded(self.search_cap)
        return seen

    def known(self, c: str) -> bool:
        return bool(self.ancestors(c) & self.classes)

    def strictly_above(self, nodes: Iterable[str]) -> set[str]:
        """The union, over ``nodes``, of each node's ancestors other than itself (``ancestors(n) - {n}``), without
        listing or caching each one's ancestry: on UniProt a census position is held by millions of hierarchy nodes.

        ``ancestors(n) - {n}`` is the set of p != n reached from n by one or more upward steps into non-vacuous
        parents, so the union is the set of nodes p reached that way from some node n != p. One upward propagation
        from all the nodes at once keeps, for every node reached, up to two of the nodes it was reached from: with two
        (distinct), one of them is not p; with one, it is the only one. A node's record grows at most twice, so each
        node is expanded at most twice."""
        base_parents, store = self._base.parents, self._store
        origins: dict[str, tuple[str, ...]] = {}
        stack: list[str] = []
        for n in nodes:
            for p in chain(base_parents.get(n, ()), store.get(n, ())):
                if p in VACUOUS_CLASSES:
                    continue
                cur = origins.get(p)
                if cur is None:
                    origins[p] = (n,)
                    stack.append(p)
                elif len(cur) == 1 and cur[0] != n:
                    origins[p] = (cur[0], n)
                    stack.append(p)
        while stack:
            x = stack.pop()
            ox = origins[x]
            for p in chain(base_parents.get(x, ()), store.get(x, ())):
                if p in VACUOUS_CLASSES:
                    continue
                cur = new = origins.get(p)
                for o in ox:
                    if new is None:
                        new = (o,)
                    elif len(new) == 1 and new[0] != o:
                        new = (new[0], o)
                if new != cur:
                    origins[p] = new
                    stack.append(p)
        return {p for p, o in origins.items() if len(o) == 2 or o[0] != p}

    def _children(self):
        children = self._store_children
        if children is None:
            children = {}
            for c, ps in self._store.items():
                for p in ps:
                    children.setdefault(p, set()).add(c)
            self._store_children = children
        return children

    def common_named_subclass(self, exprs: list[ClassExpr]) -> str | None:
        """As for the source's closure: the smallest named class (in code-point order) that the downward search
        from the first expression's members reaches, through the source's axioms and the shared hierarchy's subclass
        edges, and whose merged ancestry meets every expression. None when there is none. Raises SearchCapExceeded
        when the search would hold more than ``search_cap`` nodes (remembered like an answer).

        The answer is a pure function of the argument once the closure is built, so it is remembered."""
        key = tuple(tuple(e.members) for e in exprs)
        hit = self._memo.get(key, _MISSING)
        if hit is _MISSING:
            try:
                hit = self._common_named_subclass(exprs)
            except SearchCapExceeded as exc:
                hit = exc
            if len(self._memo) >= _MEMO_ENTRIES:
                self._memo.clear()
            self._memo[key] = hit
        if isinstance(hit, SearchCapExceeded):
            raise SearchCapExceeded(hit.cap)
        return hit

    def _common_named_subclass(self, exprs: list[ClassExpr]) -> str | None:
        """The answer of the original search, which listed every node the downward search reaches (on UniProt, the
        2.8 million taxa under up:Taxon) and tested them in order, without listing them.

        Every qualifying node has some member of each expression among its merged ancestors, so it lies in that
        expression's upward-closed set: the members and every node below them through the inverse of the edges
        ``ancestors`` follows (a superset, as vacuous classes are not skipped here). The downward search's own set
        contains the qualifying nodes too. All these sets are walked in turn, one node each, and the first one to be
        complete is the candidate list. Each candidate, in code-point order, must pass the original tests: not a data
        node, a named class, an ancestry meeting every expression, and, when the list is not the downward search's,
        being reached by it (an upward walk over the exact inverse of its edges). So the candidates that pass are
        exactly the nodes the original search accepted, and the first is its answer. Candidate ancestries are not
        cached: that cache lasts as long as the compiler.

        Which walk finishes first changes the work, never the answer, so walks that cannot finish first are left
        out: an expression whose members include another kept expression's (its set includes that one's), or the
        first expression's (its set includes the downward search's, whose edges ``ancestors`` also follows)."""
        if not exprs:
            return None
        children = self._children()
        base_desc, base_children = self._base._desc, self._base_children
        if self._store_up_children is None:
            self._store_up_children = _inverse(self._store)
        up_children = self._store_up_children

        def down(d: str):
            return chain(base_desc.get(d, ()), children.get(d, ()))

        def up(p: str):
            return chain(base_children.get(p, ()), up_children.get(p, ()))

        first = frozenset(exprs[0].members)
        kept: list[frozenset[str]] = []
        for e in exprs:
            ms = frozenset(e.members)
            if first <= ms or any(k <= ms for k in kept):
                continue
            kept = [k for k in kept if not ms <= k] + [ms]
        walks = []
        for members, step in [(exprs[0].members, down), *((sorted(k), up) for k in kept)]:
            seen = set(members)
            walks.append((seen, _walk(list(seen), seen, step)))
        cap = self.search_cap
        held = sum(len(seen) for seen, _ in walks)
        winner = None
        while winner is None:
            if held > cap:
                raise SearchCapExceeded(cap)
            for i, (_, walk) in enumerate(walks):
                if next(walk, _MISSING) is _MISSING:
                    winner = i
                    break
                held += 1
                if held > cap:
                    raise SearchCapExceeded(cap)
        candidates = sorted(walks[winner][0])
        del walks
        left = cap - len(candidates)  # what the candidate list leaves for the walks testing one candidate
        for d in candidates:
            if d in self.data_nodes:
                continue  # a data node is no named class, even where the source mentions it (rule 32)
            if d not in self.classes and self._store_classes is not None and d not in self._store_classes:
                continue  # a data node is no named class
            anc = self._manc.get(d)
            if anc is None:
                anc = self._ancestry_within(d, left)
            elif len(anc) > left:  # the same accounting as a walk, so the cut never depends on what was cached
                raise SearchCapExceeded(self.search_cap)
            if not all(any(m in anc for m in e.members) for e in exprs):
                continue
            if winner == 0 or self._reached_downwards(d, first, left):
                return d
        return None

    def _reached_downwards(self, d: str, first: frozenset[str], limit: int) -> bool:
        """Whether the downward search from ``first`` reaches ``d``: an upward walk from ``d`` over the inverse of the
        search's two steps. A node is in a source class's descendant set exactly when it is a source class with that
        class among its source ancestors; it is in a node's children exactly when that node is among its inverse.
        The walk is completed, never stopped at the first hit, so whether it is cut (SearchCapExceeded, over ``limit``
        nodes) depends on the set's size alone, not on the order sets are iterated in (PYTHONHASHSEED)."""
        down_parents = self._store_down_parents
        if down_parents is None:
            down_parents = self._store_down_parents = _inverse(self._children())
        base_anc = self._base._anc
        seen = {d}
        stack = [d]
        if len(seen) > limit:
            raise SearchCapExceeded(self.search_cap)
        while stack:
            x = stack.pop()
            for y in chain(base_anc.get(x, ()), down_parents.get(x, ())):
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
                    if len(seen) > limit:
                        raise SearchCapExceeded(self.search_cap)
        return not first.isdisjoint(seen)
