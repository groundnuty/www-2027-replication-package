"""The census of a store read through one graph, by plain aggregate queries (design log, ruling (a) on the census plan;
design log).

The definition is the census's own (``harness.sources.census``, check contract v2, rules 20, 33 and 35), and so is the
record: the counts below fill the same intermediate values ``census.build_censuses`` fills, and ``census._assemble``
turns them into the census at each threshold. Only the queries differ:

* DISTINCT counts become COUNT(*): over one graph, a set of triples, a pattern's solutions are distinct already;
* the per-signature counts (each triple's set of direct classes, rolled up through the shared hierarchy) become
  per-class join counts, ``?node rdf:type ?d`` grouped by ``?d``, of which the listed classes are kept;
* the untyped count gets its own count (FILTER NOT EXISTS), since the total minus the per-class sum is wrong whenever
  a filler has several classes. "Has a listed class" is tested as "has an IRI rdf:type object outside the graph's
  other rdf:type objects", which the complete rdf:type grouping gives; the two are the same test;
* the datatypes of the literal objects are read only where the predicate has literal objects (its total minus its
  non-literal objects).

A rewrite that is exact only under a property of the store is used only after that property has been verified, on the
layout or by a read-only query. The verification is recorded in ``provenance["plan"]["store_properties"]``, and the plan
refuses to run when a property does not hold, because its counts would not be the definition's.

Every grouped answer is shown complete before it is used, because a cut answer may lack a whole group, flagged or not
(External review of c00244b, item 2): the counts of all its groups must add up exactly to a separate COUNT(*) of the same
pattern (the rdf:type, rdfs:subClassOf and predicate groupings, each position's per-class grouping), or to the literal
objects (the datatype grouping); the named-restriction list must have as many rows as a COUNT(DISTINCT) of the same
pattern. An answer the client marks ``possibly_partial`` (ruling (d)) is used only once an exact identity with the
plan's other answers confirms it: the sums above, the literal split, typed + untyped = total, or the object total
equal to the subject total; where its identity goes unchecked (an ASK, or a cross-check count that did not answer) it
is unknown.

Every per-predicate position is checked by invariants before it is used: subject total = non-literal objects + literal
objects (all datatypes, an unbound one included, where read); filler with a listed class (FILTER EXISTS) + without one
(FILTER NOT EXISTS) = total (where the FILTER EXISTS count does not answer, the check is listed as unchecked and the
typed count is the total minus the untyped one); the largest per-class count <= the typed count <= the per-class sum.
A query that does not answer (time-out, cut, refused or possibly partial answer, transport error), a grouped answer not
shown complete or a failed invariant makes that position ``unknown``: its listing is empty, its untyped count is None
(never 0), no domain or range is derived from it, and the record says ``complete: false``. A global query that does
not answer, or a global grouping not shown complete, stops the build.

The datatype grouping alone is different (design ruling in the design log Oct 05:24): where it does not answer, or its
groups add up to fewer than the literal objects (a group missing, as from a cut answer, flagged or not; External review of
defd8f3, item 2), only the position's datatypes are unknown. Its classes are counted as usual, each by its own exact
queries; an object total marked possibly partial, which the split then does not confirm, still needs typed + untyped.
The census records the position's datatype as "unknown" wherever its literal objects reach the threshold (below it no
datatype can), and there the literal objects as a whole stand for the datatypes in the derivation, with no datatype
range (``census._derive_properties``). Groups adding up to more than the literal objects are no cut: the object total
is then as suspect as the grouping, and the failed split makes the object position unknown. A position with no filler
at all (no non-literal object) needs no query: it lists no class and has no untyped filler.
"""

from __future__ import annotations

from collections.abc import Callable

from harness.checks.hierarchy import is_data_node
from harness.sources import census
from harness.sources import hierarchy_adapter as ha
from harness.sources.closure import Hierarchy
from harness.sources.membership import CensusInput
from harness.sources.model import OntologySource
from harness.sources.vocab import PREFIXES, RDF_TYPE, RDFS_SUBCLASSOF, StoreLayout, is_meta, non_meta, ref, values
from harness.sparql import SparqlError
from harness.terms import IRI

PLAN = "census plan for one graph (design log, ruling (a)): COUNT(*) for DISTINCT counts, per-class join counts for the " \
       "per-signature roll-up, FILTER NOT EXISTS for the untyped count"
# Errors that leave a quantity unknown; any other error (a parse error, a closed gate) stops the build.
UNKNOWN_KINDS = frozenset({"timeout", "too_large", "transport", "http", "suspect"})
POSITIONS = ("subject", "object")


class PlanRefused(ValueError):
    """A store property the plan rests on does not hold, or a global count did not answer."""


def check_layout(layout: StoreLayout) -> None:
    """Refuse a layout the plan's rewrite is not exact for."""
    problems = []
    if layout.census_graphs != (layout.data_graph,):
        problems.append("the census input is not exactly the data graph")
    if layout.shipped_graphs or layout.excluded_graphs:
        problems.append("the layout has shipped or excluded graphs")
    if layout.exclude_schema_triples:
        problems.append("the schema's triples sit in the data graph")
    if layout.membership_predicate != RDF_TYPE:
        problems.append("the membership convention is not rdf:type")
    if problems:
        raise PlanRefused("store property P1 does not hold: " + "; ".join(problems))


class Queries:
    """The plan's query texts, all over the default graph (the client names the one graph).

    ``other_types``: the IRI rdf:type objects of the graph that are not listed (meta classes, data nodes, named
    restrictions), from the complete rdf:type grouping. A filler has a listed class exactly when it has an IRI rdf:type
    object outside ``other_types``, so the typed and untyped counts test that, one range lookup per filler, rather than
    one lookup per listed class; per-class counts group every rdf:type object and keep the listed ones."""

    def __init__(self, listed, other_types=()):
        self.listed = sorted(listed)
        self.other_types = sorted(other_types)

    ask_named_restrictions = PREFIXES + "ASK { ?r owl:onProperty ?x FILTER isIRI(?r) }"
    type_objects = PREFIXES + "SELECT ?c (COUNT(*) AS ?n) WHERE { ?y rdf:type ?c } GROUP BY ?c"
    type_total = PREFIXES + "SELECT (COUNT(*) AS ?n) WHERE { ?y rdf:type ?c }"
    subclass_objects = PREFIXES + "SELECT ?c (COUNT(*) AS ?n) WHERE { ?y rdfs:subClassOf ?c } GROUP BY ?c"
    subclass_total = PREFIXES + "SELECT (COUNT(*) AS ?n) WHERE { ?y rdfs:subClassOf ?c }"
    predicates = PREFIXES + "SELECT ?p (COUNT(*) AS ?n) WHERE { ?s ?p ?o } GROUP BY ?p"
    store_count = PREFIXES + "SELECT (COUNT(*) AS ?n) WHERE { ?s ?p ?o }"

    @staticmethod
    def restrictions(candidates) -> str:
        return PREFIXES + f"SELECT DISTINCT ?c WHERE {{ {values('c', candidates)} ?c owl:onProperty ?x }}"

    @staticmethod
    def restrictions_count(candidates) -> str:
        return PREFIXES + f"SELECT (COUNT(DISTINCT ?c) AS ?n) WHERE {{ {values('c', candidates)} ?c owl:onProperty ?x }}"

    @staticmethod
    def data_subject(c: str) -> str:
        return PREFIXES + f"ASK {{ {ref(c)} ?dp ?dv FILTER({non_meta('dp')}) }}"

    @staticmethod
    def datatypes(p: str) -> str:
        return PREFIXES + (f"SELECT ?dt (COUNT(*) AS ?n) WHERE {{ ?s {ref(p)} ?o FILTER isLiteral(?o) "
                           "BIND(DATATYPE(?o) AS ?dt) } GROUP BY ?dt")

    @staticmethod
    def non_literal_objects(p: str) -> str:
        return PREFIXES + f"SELECT (COUNT(*) AS ?n) WHERE {{ ?s {ref(p)} ?o FILTER(!isLiteral(?o)) }}"

    @staticmethod
    def _filler(position: str) -> tuple[str, str]:
        return ("s", "") if position == "subject" else ("o", "FILTER(!isLiteral(?o)) ")

    def per_class(self, p: str, position: str) -> str:
        """Per rdf:type object of the filler (a literal object has none, so no literal filter is needed)."""
        node = "s" if position == "subject" else "o"
        return PREFIXES + f"SELECT ?d (COUNT(*) AS ?n) WHERE {{ ?s {ref(p)} ?o . ?{node} rdf:type ?d }} GROUP BY ?d"

    def per_class_total(self, p: str, position: str) -> str:
        """The per-class grouping's pattern counted whole: what its groups must add up to."""
        node = "s" if position == "subject" else "o"
        return PREFIXES + f"SELECT (COUNT(*) AS ?n) WHERE {{ ?s {ref(p)} ?o . ?{node} rdf:type ?d }}"

    def _has_listed(self, node: str) -> str:
        others = f" && ?d NOT IN ({', '.join(ref(x) for x in self.other_types)})" if self.other_types else ""
        return f"{{ ?{node} rdf:type ?d FILTER(isIRI(?d){others}) }}"

    def untyped(self, p: str, position: str) -> str:
        node, lit = self._filler(position)
        return PREFIXES + (f"SELECT (COUNT(*) AS ?n) WHERE {{ ?s {ref(p)} ?o . {lit}"
                           f"FILTER NOT EXISTS {self._has_listed(node)} }}")

    def typed(self, p: str, position: str) -> str:
        node, lit = self._filler(position)
        return PREFIXES + (f"SELECT (COUNT(*) AS ?n) WHERE {{ ?s {ref(p)} ?o . {lit}"
                           f"FILTER EXISTS {self._has_listed(node)} }}")

    def templates(self) -> dict[str, str]:
        p = "urn:www2027:predicate"
        out = {"named_restrictions": self.ask_named_restrictions, "type_objects": self.type_objects,
               "type_total": self.type_total, "subclass_objects": self.subclass_objects,
               "subclass_total": self.subclass_total, "predicates": self.predicates, "store_count": self.store_count,
               "data_subject": self.data_subject("urn:www2027:candidate"),
               "restrictions": self.restrictions(["urn:www2027:candidate"]),
               "restrictions_count": self.restrictions_count(["urn:www2027:candidate"]),
               "datatypes": self.datatypes(p), "non_literal_objects": self.non_literal_objects(p)}
        for position in POSITIONS:
            out[f"per_class_{position}"] = self.per_class(p, position)
            out[f"per_class_total_{position}"] = self.per_class_total(p, position)
            out[f"untyped_{position}"] = self.untyped(p, position)
            out[f"typed_{position}"] = self.typed(p, position)
        return {k: v + "  # run once per predicate, the predicate in place of urn:www2027:predicate"
                if p in v else v for k, v in out.items()}


class _Unknown(Exception):
    def __init__(self, kind: str):
        super().__init__(kind)
        self.kind = kind


def _send(client, query: str, partial: set[str] | None = None, name: str = ""):
    """The answer, if whole. One the client marks possibly partial (ruling (d)) is noted under ``name`` in ``partial``,
    to be confirmed by an exact identity with other answers before it is used; with no ``partial`` to note it in, no
    identity can confirm it, and it is unknown."""
    try:
        result = client.query(query)
    except SparqlError as exc:
        if exc.kind in UNKNOWN_KINDS:
            raise _Unknown(exc.kind) from None
        raise
    if result.truncated:
        raise _Unknown("truncated")
    if getattr(result, "possibly_partial", False):
        if partial is None:
            raise _Unknown("possibly_partial")
        partial.add(name)
    return result


def _rows(client, query: str, partial: set[str] | None = None, name: str = "") -> list:
    """Rows of an aggregation; rows with an unbound first cell are rdflib's answer to an empty GROUP BY, except where
    the grouping key itself may be unbound (``keep_unbound``)."""
    return list(_send(client, query, partial, name).rows)


def _scalar(client, query: str, partial: set[str] | None = None, name: str = "") -> int:
    rows = _rows(client, query, partial, name)
    if len(rows) != 1 or rows[0][0] is None:
        raise _Unknown("no_single_count")
    return int(rows[0][0].lexical)


def _grouped(client, query: str, keep_unbound: bool = False, partial: set[str] | None = None,
             name: str = "") -> list[tuple]:
    """(key, count) pairs of a GROUP BY; an unbound key (a literal with no datatype) is kept only when asked."""
    out = []
    for r in _rows(client, query, partial, name):
        if r[1] is None:
            continue  # rdflib's single row for an empty GROUP BY
        if r[0] is None and not keep_unbound:
            continue
        out.append((r[0], int(r[1].lexical)))
    return out


def _grouped_complete(client, query: str, total: str, keep_unbound: bool = False) -> list[tuple]:
    """``_grouped``, once its groups (an unbound key's included) add up exactly to ``total``, a COUNT(*) of the same
    pattern; otherwise the answer may lack a group, and it is unknown. The exact sum also confirms either answer if
    the client marked it possibly partial."""
    seen: set[str] = set()
    rows = _grouped(client, query, keep_unbound=True, partial=seen, name="groups")
    if sum(n for _k, n in rows) != _scalar(client, total, seen, "total"):
        raise _Unknown("grouped_incomplete")
    return [(k, n) for k, n in rows if k is not None or keep_unbound]


def _restrictions(client, candidates) -> set[str]:
    """The named restrictions among ``candidates``, once the list has as many rows as their separate count (which
    also confirms either answer if possibly partial)."""
    seen: set[str] = set()
    rows = _rows(client, Queries.restrictions(candidates), seen, "list")
    if len(rows) != _scalar(client, Queries.restrictions_count(candidates), seen, "count"):
        raise _Unknown("list_incomplete")
    return {r[0].value for r in rows if isinstance(r[0], IRI)}


def _global(client, query: str, what: str):
    try:
        return query(client) if callable(query) else _send(client, query)
    except _Unknown as exc:
        raise PlanRefused(f"the {what} query did not answer ({exc.kind}); the census cannot be built") from None


def build_censuses_onegraph(client, layout: StoreLayout, thresholds=(census.DEFAULT_THRESHOLD,),
                            log: Callable[[str], None] | None = None) -> dict[int, OntologySource]:
    """The census at each threshold from one count of the store, by the plan above; the plan's own record (store
    properties, invariants, unknown quantities, query counts) is in each census's ``provenance["plan"]``."""
    if not thresholds or any(t < 1 for t in thresholds):
        raise ValueError("support_threshold must be at least 1")
    check_layout(layout)
    say = log or (lambda _m: None)
    record: dict = {"name": PLAN, "store_properties": {"P1_one_graph": True}, "unknown": [], "datatype_unknown": [],
                    "invariants": {"checked": 0, "failed": [], "unchecked": []}, "queries_by_family": {},
                    "position_totals": {}}

    def count_family(name: str) -> None:
        record["queries_by_family"][name] = record["queries_by_family"].get(name, 0) + 1

    count_family("named_restrictions")
    named = bool(_global(client, Queries.ask_named_restrictions, "restriction").boolean)
    cin = CensusInput(layout, named_restrictions=named)
    count_family("type_objects")
    count_family("type_total")
    types = _global(client, lambda c: _grouped_complete(c, Queries.type_objects, Queries.type_total),
                    "rdf:type grouping")
    count_family("subclass_objects")
    count_family("subclass_total")
    subs = _global(client, lambda c: _grouped_complete(c, Queries.subclass_objects, Queries.subclass_total),
                   "rdfs:subClassOf grouping")
    count_family("predicates")
    count_family("store_count")
    preds = _global(client, lambda c: _grouped_complete(c, Queries.predicates, Queries.store_count),
                    "predicate grouping")

    iri_subclass_objects = [c for c, _n in subs if isinstance(c, IRI)]
    record["store_properties"]["P2_no_subclass_membership"] = not iri_subclass_objects
    if iri_subclass_objects:
        raise PlanRefused(f"store property P2 does not hold: {len(iri_subclass_objects)} rdfs:subClassOf objects are "
                          "IRIs, so per-class join counts would not equal the census's roll-up")

    # The listing (hierarchy_adapter.fetch_listing on one graph with rdf:type as the convention).
    prim = {c.value for c, _n in types if isinstance(c, IRI) and not is_meta(c.value)}
    sec: set[str] = set()  # the IRI rdfs:subClassOf objects: none after the check above
    if named and prim:
        count_family("restrictions")
        count_family("restrictions_count")
        prim -= _global(client, lambda c: _restrictions(c, sorted(prim)), "restriction")
    candidates = prim | sec
    data_subjects = set()
    for c in sorted(candidates):
        count_family("data_subject")
        if _global(client, Queries.data_subject(c), "data-node").boolean:
            data_subjects.add(c)
    predicates = {p.value for p, _n in preds if isinstance(p, IRI)}
    entity_classes: set[str] = set()  # subjects of rdfs:subClassOf to an IRI: none after the check above
    data = {c for c in candidates if is_data_node(c, targets=candidates, data_subjects=data_subjects,
                                                  predicates=predicates, entity_classes=entity_classes)}
    sec_only = sorted(sec - prim - data)
    listing = ha.Listing({c for c in prim - data if not is_meta(c)}, len(prim), len(sec), data & candidates,
                         len(sec_only), len(entity_classes & candidates))
    listed = sorted(listing.listed)
    shared = Hierarchy({c: [] for c in listed})  # no hierarchy graphs, as the layout check requires
    q = Queries(listed, {c.value for c, _n in types if isinstance(c, IRI)} - set(listed))

    counts = {p.value: n for p, n in preds if isinstance(p, IRI)}
    untyped: dict[str, dict[str, int]] = {}
    counted: dict[str, dict[str, dict[str, int]]] = {}
    unknown: set[tuple[str, str]] = set()
    literals_unknown: dict[str, int] = {}  # predicate -> its literal objects, whose datatypes did not answer

    def invariant(ok: bool, p: str, position: str, name: str) -> None:
        record["invariants"]["checked"] += 1
        if not ok:
            record["invariants"]["failed"].append({"predicate": p, "position": position, "invariant": name})
            raise _Unknown(f"invariant:{name}")

    def mark(p: str, position: str, why: str) -> None:
        unknown.add((p, position))
        record["unknown"].append({"predicate": p, "position": position, "why": why})

    listed_set = set(listed)
    work = sorted(p for p in counts if p not in census.MEMBERSHIP_PREDICATES)
    for i, p in enumerate(work, 1):
        if i == 1 or i % 10 == 0 or i == len(work):
            say(f"  census positions: predicate {i}/{len(work)}")
        subject_total = counts[p]
        if not subject_total:
            continue
        untyped[p] = {"subject": 0, "object": 0}
        counted[p] = {"subject": {}, "object": {}}
        totals = {"subject": subject_total}
        # This predicate's answers the client marked possibly partial, each until an exact identity confirms it.
        pp: set[str] = set()
        try:
            count_family("non_literal_objects")
            totals["object"] = _scalar(client, q.non_literal_objects(p), pp, "object total")
            literal = subject_total - totals["object"]
            invariant(literal >= 0, p, "object", "non_literal_within_total")
        except _Unknown as exc:
            mark(p, "object", exc.kind)
        else:
            if not literal:  # the object total equals the (confirmed) subject total
                pp.discard("object total")
            else:  # the datatypes are read only where there are literal objects
                try:
                    count_family("datatypes")
                    dts = _grouped(client, q.datatypes(p), keep_unbound=True, partial=pp, name="datatypes")
                    split = sum(n for _dt, n in dts)
                    if split < literal:  # groups missing, as from a cut answer: the datatypes alone are unknown
                        record["invariants"]["checked"] += 1
                        raise _Unknown("grouped_incomplete")
                    # more than the literal objects is no cut: the object total is as suspect as the grouping
                    invariant(split == literal, p, "object", "literal_split")
                except _Unknown as exc:
                    if exc.kind.startswith("invariant:"):  # the answers disagree: the whole position is unknown
                        mark(p, "object", exc.kind)
                    else:  # the datatypes alone are unknown; the position's classes are counted as usual (the object
                        # total, unconfirmed by the split, still needs typed + untyped if marked possibly partial)
                        pp.discard("datatypes")
                        literals_unknown[p] = literal
                        record["datatype_unknown"].append({"predicate": p, "literal_objects": literal,
                                                           "why": exc.kind})
                else:
                    pp.difference_update({"datatypes", "object total"})  # the literal split confirms both
                    for dt, n in dts:
                        if isinstance(dt, IRI):
                            counted[p]["object"][dt.value] = n
        record["position_totals"][p] = dict(totals)
        for position in POSITIONS:
            if (p, position) in unknown:
                continue
            total = totals[position]
            mine = {f"{position} {x}" for x in ("total", "per class", "per-class total", "untyped", "typed")}
            if not listed or not total:  # no listed class, or no filler at all: every filler (if any) is untyped
                if pp & mine:
                    mark(p, position, "possibly_partial")
                    continue
                untyped[p][position] = total
                continue
            try:
                count_family(f"per_class_{position}")
                grouped = _grouped(client, q.per_class(p, position), partial=pp, name=f"{position} per class")
                count_family(f"per_class_total_{position}")
                join = _scalar(client, q.per_class_total(p, position), pp, f"{position} per-class total")
                invariant(sum(n for _d, n in grouped) == join, p, position, "per_class_complete")
                pp.difference_update({f"{position} per class", f"{position} per-class total"})
                per_class = {d.value: n for d, n in grouped if isinstance(d, IRI) and d.value in listed_set}
                count_family(f"untyped_{position}")
                none = _scalar(client, q.untyped(p, position), pp, f"{position} untyped")
                invariant(0 <= none <= total, p, position, "untyped_within_total")
                count_family(f"typed_{position}")
                try:
                    some = _scalar(client, q.typed(p, position), pp, f"{position} typed")
                except _Unknown as exc:  # the cross-check count only: the untyped count stands, the check is noted
                    some = total - none
                    record["invariants"]["unchecked"].append(
                        {"predicate": p, "position": position, "invariant": "typed_plus_untyped", "why": exc.kind})
                else:
                    invariant(some + none == total, p, position, "typed_plus_untyped")
                    pp.difference_update({f"{position} untyped", f"{position} typed", f"{position} total"})
                invariant(max(per_class.values(), default=0) <= some <= sum(per_class.values()), p, position,
                          "per_class_bounds")
                if pp & mine:  # an answer marked possibly partial that no identity confirmed
                    raise _Unknown("possibly_partial")
            except _Unknown as exc:
                mark(p, position, exc.kind)
                continue
            untyped[p][position] = none
            counted[p][position].update(per_class)

    type_counts = {c.value: n for c, n in types if isinstance(c, IRI)}
    members = {c: type_counts.get(c, 0) for c in listed}  # a node's classes are its rdf:type objects (checked above)
    other_counts = [(c.value, n) for c, n in subs if isinstance(c, IRI)]
    patterns = census.literal_patterns(type_counts, other_counts, listing, RDF_TYPE, RDFS_SUBCLASSOF)
    untyped = dict(sorted(untyped.items()))
    counted = dict(sorted(counted.items()))
    class_terms = set(listed) | set(shared.parents)
    full_entries = sum(len(v) for d in counted.values() for v in d.values())
    record["unknown"].sort(key=lambda u: (u["predicate"], u["position"]))
    record["complete"] = not unknown
    queries = q.templates()
    out = {}
    for t in thresholds:
        src = census._assemble(t, layout, cin, listing, listed, shared, frozenset(), queries, untyped, counted,
                               class_terms, full_entries, {}, type_counts, members, patterns, RDF_TYPE,
                               RDFS_SUBCLASSOF, literals_unknown)
        _apply_unknown(src, unknown)
        src.provenance["builder"] = "harness.sources.census_onegraph (record assembled by harness.sources.census)"
        src.provenance["plan"] = record
        out[t] = src
    return out


def _apply_unknown(src: OntologySource, unknown: set[tuple[str, str]]) -> None:
    """An unknown position lists nothing, has no untyped count, and gives no domain or range."""
    for p, position in sorted(unknown):
        src.position_support.setdefault(p, {"subject": {}, "object": {}})[position] = {}
        src.untyped_fillers.setdefault(p, {"subject": None, "object": None})[position] = None
        info = src.properties.get(p)
        if info is None:
            continue
        if position == "subject":
            info.domain = []
        else:
            info.range = []
            info.datatype_range = []
            info.kind = "unknown"
