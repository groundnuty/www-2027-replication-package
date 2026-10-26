"""The data-derived source: a census of the classes and predicates the data uses directly.

Built by SPARQL aggregation (GROUP BY) against the store. This is the one census rule shared by the
builder (#6, #13) and the check compiler (#8): check contract v2, rules 20, 33 and 35. Membership and
the shared hierarchy go through ``hierarchy_adapter`` (the compiler's ``harness/checks/hierarchy.py``
once it lands).

Input (rule 20)
    The census graphs, excluding the graph or file of any source under comparison: UniProt's ``core``
    by a FROM/FROM NAMED dataset clause; on Rhea, the schema's own triples in the Rhea graph.

What is listed (rule 33)
    Every class the data uses directly under the graph's membership convention, never dropped:
    the targets of the primary membership predicate that are not data nodes, plus the targets of the
    secondary one that are named classes (``hierarchy_adapter.fetch_listing``). Nodes reached only
    through the shared hierarchy (ChEBI's ancestors on Rhea; taxonomy, GO, ... on UniProt) are shared
    evidence and listed by no source. The census asserts no subclass axioms of its own: subclass edges
    inside the data graph are membership statements, and the hierarchy is the shared one (rule 35).

The census record (field names fixed in the design log)
    ``classes``           every listed class (``parents`` is empty)
    ``type_support``      class -> direct count: distinct subjects of the literal triple
                          ``?x <membership predicate> C`` (on Rhea rdfs:subClassOf in the data graph)
    ``class_support``     class -> membership count: distinct nodes with C among their classes, counted
                          up the shared hierarchy
    ``position_support``  predicate -> {"subject": {class: n}, "object": {class or datatype: n}}: per position,
                          the most specific classes (and datatypes) with support >= threshold, a class of the
                          shared hierarchy included when it meets the threshold only through its subclasses
                          (Dog 3 + Cat 3 under Mammal lists Mammal 6); their ancestors in the shared hierarchy
                          are implied (rule 33, change-log entries 12 and 14)
    ``untyped_fillers``   predicate -> {"subject": n, "object": n}, every position
    ``patterns``          literal patterns only: ``Pattern(owl:Thing, P, C, None, n, "literal")`` for the
                          secondary membership predicate P (rdf:type on Rhea), and for the graph's own
                          convention when C is an RDF, RDFS or OWL meta class (``?x rdf:type owl:Class`` on
                          UniProt; rule 35), the direct counts that rule 20's single-pattern check needs
                          beside ``type_support`` (F23, G8b)
    ``properties``        per predicate: kind, domain and range derived below

    Position counts are in triples: distinct triples ``s p o`` of the census input whose filler at that
    position has the class among its classes (or, for an object literal, that datatype), each node counted
    once per class and a multi-typed node under each. A node's classes are its direct classes (rule 35:
    rdf:type objects, and data-graph subclass objects followed through data nodes) and their ancestors in
    the shared hierarchy, closed over the union of the hierarchy graphs. A filler with no class (citations,
    EC numbers, OWL blank nodes, ChEBI classes as objects of rh:chebi) is untyped.

Domain and range
    The candidates at a position are the classes with support >= threshold there that the data uses
    directly (listed classes) or that the position lists (its most specific classes). domain(p) = the
    subject candidates, minimised to those with no candidate ancestor in the shared hierarchy; none if
    the untyped subjects reach the threshold. range(p) likewise from the object position. Literal objects
    at or over the threshold make p a datatype property with ``datatype_range`` = those datatypes
    ("any of"); literal and IRI objects both at or over the threshold give kind "unknown" and no range.

Rule 20's checks, for the compiler: a single pattern ``?x P C`` is empty only when its direct count
is 0 (``direct_count``); a join is ``census_low_support`` when no class meets the threshold at both
positions (``supported_classes``: the classes the position lists and their shared ancestors), and
``untyped_fillers`` at or over the threshold make it unknown.
"""

from __future__ import annotations

from urllib.parse import unquote

from harness.sources import hierarchy_adapter as ha
from harness.sources.closure import Hierarchy
from harness.sources.membership import CensusInput, excluded_triple_counts, fetch_census_input
from harness.sources.model import ClassExpr, ClassInfo, OntologySource, Pattern, PropertyInfo
from harness.sources.vocab import (
    PREFIXES, RDF_TYPE, RDFS_SUBCLASSOF, UNTYPED, StoreLayout, is_meta, membership_label, ref, values,
)
from harness.terms import IRI

DEFAULT_THRESHOLD = 5
MEMBERSHIP_PREDICATES = (RDF_TYPE, RDFS_SUBCLASSOF)
ANY_SUBJECT = "http://www.w3.org/2002/07/owl#Thing"  # subject_class of a literal pattern

DERIVATION_RULE = (
    "The candidates at a position are the classes with support >= threshold there that the data uses directly "
    "(listed classes) or that position_support lists at that position. domain(p) = p's subject candidates, "
    "minimised to those with no candidate ancestor in the shared hierarchy; none if p's untyped subjects reach "
    "the threshold. range(p) likewise from the object position; none if the untyped objects reach the threshold. "
    "Literal objects at or over the threshold: datatype property, datatype_range = those datatypes. Literal "
    "and IRI objects both at or over the threshold: kind unknown, no range."
)


# A filler's set of direct classes, as one string: each class IRI percent-encoded (ENCODE_FOR_URI leaves no space, nor
# any other character but the unreserved ones), joined by single spaces; ``signature`` reads the classes back exactly
# (External review of c00244b, item 1: an IRI may hold a character str.split() splits on, U+00A0 among them).
SIGNATURE = '(GROUP_CONCAT(DISTINCT ENCODE_FOR_URI(STR(?d)); separator=" ") AS ?sig)'


def signature(lexical: str) -> tuple[str, ...]:
    """The classes a ``SIGNATURE`` value stands for, sorted: split on the separator only, each part decoded."""
    return tuple(sorted(unquote(part) for part in lexical.split(" ")))


class _Q:
    """The census queries, all over the census input. None of them follows the shared hierarchy: it is
    closed offline (``hierarchy_adapter.fetch_shared_parents``) and applied to the counts in Python, from
    counts per set of direct classes (``_roll_up``)."""

    def __init__(self, cin: CensusInput, predicate: str | None = None, drop: frozenset[str] = frozenset()):
        """With ``predicate``, the position queries count that predicate only (the census runs them once per
        predicate, so no query scans every triple of a large store; counts are exact, since every triple
        has one predicate)."""
        self.cin = cin
        self.layout = cin.layout
        self.drop = drop
        self.ds = cin.dataset()
        only = f"VALUES ?p {{ {ref(predicate)} }} " if predicate else ""
        self.triples = (
            f"{only}GRAPH ?g {{ ?s ?p ?o }} {cin.keep('s', 'g')} "
            "FILTER(?p != rdf:type && ?p != rdfs:subClassOf) "
        )

    def _membership(self, node: str, var: str, classes: list[str], inner: str = "") -> str:
        """The node's direct classes among ``classes``, with ``inner`` and the class list joined inside each branch
        of the membership union (``hierarchy_adapter.direct_membership_branches``)."""
        v = values(var, classes or [NO_CLASS])
        a, b = ha.direct_membership_branches(node, var, self.layout, self.cin.keep_subject(node), "m", self.drop)
        return f"{{ {inner} {a} {v} }} UNION {{ {inner} {b} {v} }}"

    def _filler(self, position: str) -> tuple[str, str]:
        return ("s", "") if position == "subject" else ("o", "FILTER(!isLiteral(?o)) ")

    def position_total(self, position: str) -> str:
        """Distinct triples per predicate (the untyped count is the total minus the typed ones)."""
        _, lit = self._filler(position)
        return PREFIXES + (
            f"SELECT ?p (COUNT(*) AS ?n) {self.ds} WHERE {{ {{ SELECT DISTINCT ?s ?p ?o WHERE {{ {self.triples} {lit}}} }} }} "
            "GROUP BY ?p"
        )

    def position_signatures(self, position: str, classes: list[str]) -> str:
        """Distinct triples per (predicate, set of the filler's direct classes among ``classes``): a triple whose
        filler has no such class has no row. Rolled up through the shared hierarchy in Python, set-wise, so a
        filler is counted once per class; the typed count is the sum of the rows."""
        f, lit = self._filler(position)
        return PREFIXES + (
            f"SELECT ?p ?sig (COUNT(*) AS ?n) {self.ds} WHERE {{ {{ SELECT ?s ?p ?o "
            f"{SIGNATURE} WHERE {{ "
            f"{self._membership(f, 'd', classes, self.triples + lit)} }} GROUP BY ?s ?p ?o }} }} GROUP BY ?p ?sig"
        )

    def object_datatypes(self) -> str:
        return PREFIXES + (
            f"SELECT ?p ?dt (COUNT(*) AS ?n) {self.ds} WHERE {{ {{ SELECT DISTINCT ?s ?p ?o WHERE {{ "
            f"{self.triples} FILTER(isLiteral(?o)) }} }} BIND(DATATYPE(?o) AS ?dt) }} GROUP BY ?p ?dt"
        )

    def class_signatures(self, classes: list[str]) -> str:
        """Distinct nodes per set of their direct classes among ``classes`` (rolled up like the positions)."""
        return PREFIXES + (
            f"SELECT ?sig (COUNT(*) AS ?n) {self.ds} WHERE {{ {{ SELECT ?x {SIGNATURE} "
            f"WHERE {{ {self._membership('x', 'd', classes)} }} GROUP BY ?x }} }} GROUP BY ?sig"
        )

    def direct(self, predicate: str) -> str:
        if predicate == RDFS_SUBCLASSOF:  # membership subclass edges live in the data graph only
            body = f"GRAPH {ref(self.layout.data_graph)} {{ ?x rdfs:subClassOf ?c }} {self.cin.keep_subject('x')}"
        else:
            body = f"GRAPH ?g {{ ?x rdf:type ?c }} {self.cin.keep('x', 'g')}"
        return PREFIXES + f"SELECT ?c (COUNT(DISTINCT ?x) AS ?n) {self.ds} WHERE {{ {body} FILTER isIRI(?c) }} GROUP BY ?c"


def _roll_up(sig_counts, shared: Hierarchy) -> dict[str, int]:
    """Counts per class of the shared hierarchy from counts per signature (a set of direct shared classes):
    each signature adds its count once to every class in the union of its members' ancestors-or-self."""
    out: dict[str, int] = {}
    for sig, n in sig_counts:
        closed = set()
        for d in sig:
            closed |= shared.ancestors_or_self(d) if d in shared.parents else {d}
        for c in closed:
            out[c] = out.get(c, 0) + n
    return out


NO_CLASS = "urn:www2027:no-listed-class"  # stands for the empty class list, so every filler counts as untyped


def _rows(client, query: str):
    """Rows of an aggregation, refusing a truncated result (External review 2). Rows with an unbound first cell
    are rdflib's answer to an empty GROUP BY; QLever returns none."""
    result = client.query(query)
    if result.truncated:
        raise RuntimeError(f"a census query returned a truncated result ({result.total_rows} rows); refusing it")
    return [r for r in result.rows if r[0] is not None]


def build_census(
    client,
    layout: StoreLayout,
    support_threshold: int = DEFAULT_THRESHOLD,
    cin: CensusInput | None = None,
    listing: ha.Listing | None = None,
    shared: Hierarchy | None = None,
    log=None,
) -> OntologySource:
    return build_censuses(client, layout, (support_threshold,), cin, listing, shared, log)[support_threshold]


def build_censuses(
    client,
    layout: StoreLayout,
    thresholds=(DEFAULT_THRESHOLD,),
    cin: CensusInput | None = None,
    listing: ha.Listing | None = None,
    shared: Hierarchy | None = None,
    log=None,
) -> dict[int, OntologySource]:
    """The census at each threshold, from one count of the store: the threshold decides only which (predicate,
    position, class) entries are listed and the domains and ranges derived from them (rules 20 and 33); every
    count, the listing and the literal patterns are the same at every threshold."""
    if not thresholds or any(t < 1 for t in thresholds):
        raise ValueError("support_threshold must be at least 1")
    cin = cin or fetch_census_input(client, layout)
    listing = listing or ha.fetch_listing(client, layout, cin.keep_subject("y"))
    listed = sorted(listing.listed)
    shared = shared if shared is not None else ha.fetch_shared_parents(client, layout, listed)
    drop = ha.fetch_dropped(client, layout, cin.schema_subjects)
    q = _Q(cin, drop=drop)
    queries: dict[str, str] = {}

    untyped: dict[str, dict[str, int]] = {}
    counted: dict[str, dict[str, dict[str, int]]] = {}  # every class with support, before pruning

    def signatures(rows, at: int) -> list[tuple[tuple[str, ...], int]]:
        return [(signature(r[at].lexical), int(r[at + 1].lexical)) for r in rows]

    predicates = sorted(ha.fetch_used_predicates(client, layout) - set(MEMBERSHIP_PREDICATES))
    for name, query in (("object_datatypes", q.object_datatypes()),
                        ("position_total", q.position_total("subject")),
                        ("position_signatures", q.position_signatures("subject", listed))):
        queries[name] = query + "  # run once per predicate and position, with VALUES ?p"
    import time as _time

    t0 = _time.monotonic()
    for i_pred, pred in enumerate(predicates, 1):
        if log and (i_pred == 1 or i_pred % 10 == 0 or i_pred == len(predicates)):
            log(f"  census positions: predicate {i_pred}/{len(predicates)} ({_time.monotonic() - t0:.0f}s) {pred}")
        qp = _Q(cin, pred, drop)
        subject_total = sum(int(r[1].lexical) for r in _rows(client, qp.position_total("subject")))
        if not subject_total:  # used only by excluded triples (the schema's own): not census input
            continue
        untyped[pred] = {"subject": 0, "object": 0}
        counted[pred] = {"subject": {}, "object": {}}
        for p, dt, n in _rows(client, qp.object_datatypes()):
            if isinstance(dt, IRI):
                counted[p.value]["object"][dt.value] = int(n.lexical)
        for position in ("subject", "object"):
            total = subject_total if position == "subject" else sum(
                int(r[1].lexical) for r in _rows(client, qp.position_total(position)))
            sigs = signatures(_rows(client, qp.position_signatures(position, listed)), 1) if listed else []
            untyped[pred][position] = total - sum(n for _sig, n in sigs)
            counted[pred][position].update(_roll_up(sigs, shared))
    convention = layout.membership_predicate
    other = ha.other_membership_predicate(layout)
    members: dict[str, int] = {}
    if listed:
        queries["class_signatures"] = q.class_signatures(listed)
        rolled = _roll_up(signatures(_rows(client, queries["class_signatures"]), 0), shared)
        members = {c: rolled.get(c, 0) for c in listed}
    queries["type_support"] = q.direct(convention)
    queries["literal_other"] = q.direct(other)
    direct = {r[0].value: int(r[1].lexical) for r in _rows(client, queries["type_support"]) if isinstance(r[0], IRI)}
    other_counts = [(r[0].value, int(r[1].lexical)) for r in _rows(client, queries["literal_other"]) if isinstance(r[0], IRI)]
    patterns = literal_patterns(direct, other_counts, listing, convention, other)

    untyped = dict(sorted(untyped.items()))
    counted = dict(sorted(counted.items()))
    class_terms = set(listed) | set(shared.parents)
    full_entries = sum(len(v) for d in counted.values() for v in d.values())
    excluded = excluded_triple_counts(client, cin)
    return {t: _assemble(t, layout, cin, listing, listed, shared, drop, queries, untyped, counted, class_terms,
                         full_entries, excluded, direct, members, patterns, convention, other)
            for t in thresholds}


def literal_patterns(direct: dict[str, int], other_counts, listing: ha.Listing, convention: str, other: str) -> list[Pattern]:
    """The literal patterns from the direct counts: ``direct`` (class -> distinct subjects of ``?x <convention> C``) and
    ``other_counts`` ((class, distinct subjects of ``?x <other> C``) for IRI classes)."""
    patterns = []
    for c, n in other_counts:
        if other == RDF_TYPE or c in listing.listed:  # literal subClassOf only for listed classes
            patterns.append(Pattern(ANY_SUBJECT, other, c, None, n, "literal"))
    # Rule 35: a literal membership triple whose object is an RDF, RDFS or OWL meta class (``?x rdf:type owl:Class``
    # on UniProt) keeps its count for rule 20's single-pattern check, as a literal pattern; the meta class is never
    # listed as a class. Other unlisted targets are data nodes (rule 32): entity constants, with no count.
    for c, n in sorted(direct.items()):
        if is_meta(c) and c not in listing.listed:
            patterns.append(Pattern(ANY_SUBJECT, convention, c, None, n, "literal"))
    patterns.sort(key=lambda x: (x.predicate, x.object_class or ""))
    return patterns


def _assemble(support_threshold, layout, cin, listing, listed, shared, drop, queries, untyped, counted, class_terms,
              full_entries, excluded, direct, members, patterns, convention, other,
              literals_unknown: dict[str, int] | None = None) -> OntologySource:
    """The census record at one threshold, from the counts (``build_censuses``). ``literals_unknown``: predicate -> its
    literal objects, where their datatypes could not be counted (``counted`` then holds none of them)."""
    literals_unknown = literals_unknown or {}
    classes = {c: ClassInfo(c) for c in listed}
    positions = prune_positions(counted, support_threshold, shared, class_terms)
    unknown_dt = {p: {"object": "unknown"} for p, n in sorted(literals_unknown.items()) if n >= support_threshold}
    properties = _derive_properties(counted, positions, untyped, support_threshold, shared, set(listed), class_terms,
                                    {p: literals_unknown[p] for p in unknown_dt})
    kept_entries = sum(len(v) for d in positions.values() for v in d.values())
    untyped = {p: dict(v) for p, v in untyped.items()}

    return OntologySource(
        name=f"census-t{support_threshold}",
        kind="census",
        graph=layout.name,
        membership=membership_label(layout),
        classes=classes,
        properties=properties,
        patterns=list(patterns),
        support_threshold=support_threshold,
        type_support={c: direct.get(c, 0) for c in listed},
        class_support={c: members.get(c, 0) for c in listed},
        untyped_fillers=untyped,
        position_support=positions,
        position_datatype=unknown_dt or None,
        provenance={
            "builder": "harness.sources.census",
            "contract": "check contract v2, rules 20, 33 and 35 (science-agent 066225a)",
            "definition": ha.DEFINITION,
            "layout": {"name": layout.name, "data_graph": layout.data_graph, "graphs": list(layout.graphs)},
            "input": {**cin.describe(), "excluded_schema_triples": excluded,
                      "excluded_schema_triples_total": sum(excluded.values())},
            "membership_reading": cin.reading(),
            "membership_convention": convention,
            "hierarchy_graphs": list(ha.hierarchy_graphs(layout)),
            "dropped_endpoints": sorted(drop),
            "listing": {"listed": len(listed), "primary_targets": listing.primary_targets,
                        "secondary_targets": listing.secondary_targets, "data_nodes": len(listing.data_nodes),
                        "secondary_not_named": listing.not_named,
                        "entity_class_targets": listing.entity_class_targets,
                        "data_node_test": "rule 32 as amended (change-log entry 26), harness/checks/hierarchy.is_data_node; "
                                          "data_nodes counts the membership targets that are data nodes",
                        "rule": "rule 33: classes the data uses directly; nodes reached only through the shared "
                                "hierarchy are not listed"},
            "untyped_sentinel": UNTYPED,
            "support": "distinct triples of the census input whose filler at the position has the class among its "
                       "classes (its direct classes and their ancestors in the shared hierarchy, each counted once); "
                       "untyped fillers have no class",
            "type_support": f"distinct subjects of the literal triple ?x <{convention}> C",
            "class_support": "distinct nodes with C among their classes (direct, or up the shared hierarchy)",
            "literal_patterns": f"Pattern(owl:Thing, <{other}>, C, None, distinct subjects, reading='literal'); and "
                                f"Pattern(owl:Thing, <{convention}>, M, ...) for an RDF, RDFS or OWL meta class M, "
                                "which is never listed as a class (rule 35)",
            "domain_range_rule": DERIVATION_RULE,
            "position_rule": "rule 33 as clarified (change-log entries 12 and 14): each (predicate, position) lists its "
                             "most specific classes (and datatypes) with support >= threshold, support counted up the "
                             "shared hierarchy, so a shared-hierarchy class is listed when it meets the threshold only "
                             "through its subclasses; their ancestors in the shared hierarchy are implied, since support "
                             "only grows upwards; entries below the threshold are not listed; untyped-filler counts are "
                             "listed for every position",
            "counts": {
                "position_entries_counted": full_entries,
                "position_entries_listed": kept_entries,
                "classes": len(classes),
                "data_nodes": len(listing.data_nodes),
                "literal_patterns": len(patterns),
                "predicates": len(properties),
                **({"positions_datatype_unknown": len(unknown_dt)} if literals_unknown else {}),
            },
            "queries": dict(queries),
        },
    )


def _derive_properties(counted, positions, untyped, t: int, shared: Hierarchy, listed, class_terms,
                       literals_unknown: dict[str, int] | None = None) -> dict[str, PropertyInfo]:
    """``DERIVATION_RULE``: ``counted`` holds every class's support (before pruning), ``positions`` the pruned
    record, ``listed`` the classes the data uses directly, ``class_terms`` every class (listed or shared).
    ``literals_unknown``: predicate -> its literal objects, at or over ``t``, whose datatypes are unknown; there the
    literal objects as a whole stand for the datatypes at or over the threshold, and no datatype range is given."""
    literals_unknown = literals_unknown or {}
    out: dict[str, PropertyInfo] = {}
    for p in sorted(counted):
        info = PropertyInfo(p)

        def candidates(position: str) -> set[str]:
            over = {c for c, n in counted[p][position].items() if n >= t}
            return (over & listed) | (set(positions[p][position]) & class_terms)

        subj = candidates("subject")
        if subj and untyped[p]["subject"] < t:
            info.domain = [ClassExpr(tuple(shared.minimal_general(subj)))]
        cls = candidates("object")
        dts = sorted(d for d, n in counted[p]["object"].items() if n >= t and d not in class_terms)
        literal = bool(dts) or literals_unknown.get(p, 0) >= t
        iri_allowed = bool(cls) or untyped[p]["object"] >= t
        if literal and not iri_allowed:
            info.kind = "datatype"
            info.datatype_range = dts
        elif iri_allowed and not literal:
            info.kind = "object"
            if cls and untyped[p]["object"] < t:
                info.range = [ClassExpr(tuple(shared.minimal_general(cls)))]
        out[p] = info
    return out


def mixed_positions(src: OntologySource) -> list[str]:
    """The predicates whose object position has literal and IRI objects both at or over the threshold, where
    ``DERIVATION_RULE`` gives kind unknown and no range (change-log entry 97), read from the census record. Kind unknown
    also means that neither reaches the threshold; such a position lists nothing at its object and its untyped objects
    stay below the threshold, while a mixed one lists a datatype or class there, or has its untyped objects at it (a
    position whose literal datatypes are unknown, ``position_datatype``, lists no datatype)."""
    t = src.support_threshold
    if t is None:
        raise ValueError(f"{src.name} is not a census: it has no support threshold")
    positions = src.position_support or {}
    untyped = src.untyped_fillers or {}
    return sorted(p for p, info in src.properties.items() if info.kind == "unknown" and (
        positions.get(p, {}).get("object") or untyped.get(p, {}).get("object", 0) >= t))


def prune_positions(positions, t: int, shared: Hierarchy, listed) -> dict:
    """Rule 33 (clarified): per (predicate, position), the classes with support >= t that have no other such
    class below them in the shared hierarchy, and the datatypes with support >= t. Classes on one cycle of the
    shared hierarchy (A below B and B below A) have the same support and imply each other; the cycle keeps one
    representative, its smallest IRI, so a supported cycle is never pruned whole."""
    out: dict[str, dict[str, dict[str, int]]] = {}
    for p, d in sorted(positions.items()):
        out[p] = {}
        for position, counts in d.items():
            over = {c: n for c, n in counts.items() if n >= t}
            cls = {c for c in over if c in listed}
            implied = set()
            for c in cls:
                if c in shared.parents:
                    eq = shared.equivalents(c)
                    implied |= shared.ancestors(c) - eq
                    if any(e < c for e in eq & cls):
                        implied.add(c)
            out[p][position] = dict(sorted((c, n) for c, n in over.items() if c not in implied))
    return out


def supported_classes(src: OntologySource, shared: Hierarchy, predicate: str, position: str) -> set[str]:
    """The classes meeting the threshold at a position: the listed ones and all their shared ancestors."""
    top = set((src.position_support or {}).get(predicate, {}).get(position, {}))
    out = set(top)
    for c in top:
        if c in shared.parents:
            out |= shared.ancestors(c)
    return out


def direct_count(src: OntologySource, predicate: str, cls: str) -> int:
    """Rule 20's direct count of the single pattern ``?x predicate cls`` (0 if never asserted): ``type_support``
    for a listed class under the graph's convention, else the literal pattern (the other predicate, or a meta
    class under the convention, rule 35)."""
    convention = RDFS_SUBCLASSOF if src.membership == "rdfs:subClassOf" else RDF_TYPE
    if predicate == convention and cls in (src.type_support or {}):
        return src.type_support[cls]
    return next((x.support for x in src.patterns
                 if x.reading == "literal" and x.predicate == predicate and x.object_class == cls), 0)


def literal_type_support(src: OntologySource, cls: str) -> int:
    """Distinct subjects of the literal pattern ``?x rdf:type cls`` (0 if never asserted)."""
    return direct_count(src, RDF_TYPE, cls)
