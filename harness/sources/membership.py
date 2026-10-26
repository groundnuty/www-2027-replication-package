"""The membership reading and the census input, read from the store.

``MEMBERSHIP_READING`` (in ``vocab``) states the reading in words; this module
is its only implementation on the store side.

The census input (check contract v2, rule 20) is the store **excluding the
graph or file of any source under comparison**: the layout's
``excluded_graphs`` (``core`` on UniProt) are left out of every census query
by a ``FROM``/``FROM NAMED`` dataset clause, so subclass paths never cross
them; on Rhea, whose schema triples sit inside the data graph, the data-graph
triples whose subject is a schema term are filtered out (``CensusInput.keep``).
A property path cannot skip single triples, so on Rhea the builder checks that
no excluded rdfs:subClassOf edge leads to a census class
(``ObservedHierarchy.check_excluded_edges``); it fails loudly otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from harness.sources.closure import Hierarchy
from harness.sources.scope import UsedTerms
from harness.sources.vocab import (
    MEMBERSHIP_READING_BASE,
    NAMED_RESTRICTIONS_READING,
    PREFIXES,
    StoreLayout,
    is_meta,
    non_meta,
    not_data_node,
    ref,
    values,
)
from harness.terms import IRI


def membership_path(node: str, cls: str) -> str:
    """A pattern binding ``?cls`` to every class of ``?node`` (its asserted classes and their
    ancestors). Distinct per (node, class) only inside a DISTINCT subquery."""
    return f"?{node} (rdf:type|rdfs:subClassOf)/rdfs:subClassOf* ?{cls} ."


@dataclass(frozen=True)
class CensusInput:
    layout: StoreLayout
    schema_subjects: frozenset[str] = frozenset()  # data-graph subjects whose triples are the schema's own
    named_restrictions: bool = False  # the census graphs publish restrictions as IRIs (UniProt's GO)

    def not_restriction(self, var: str) -> str:
        """A FILTER dropping named restrictions (empty where the store has none, so queries stay as they were)."""
        if not self.named_restrictions:
            return ""
        return f"FILTER NOT EXISTS {{ GRAPH ?rg_{var} {{ ?{var} owl:onProperty ?rx_{var} }} }} "

    def reading(self) -> str:
        return MEMBERSHIP_READING_BASE + self.layout.reading_note + (NAMED_RESTRICTIONS_READING if self.named_restrictions else "")

    def dataset(self) -> str:
        """FROM/FROM NAMED clauses: the default graph (for paths) and the named graphs (for
        GRAPH ?g) are both the census graphs."""
        gs = self.layout.census_graphs
        return " ".join(f"FROM {ref(g)}" for g in gs) + " " + " ".join(f"FROM NAMED {ref(g)}" for g in gs)

    def keep(self, s: str = "s", g: str = "g") -> str:
        """A FILTER dropping the schema's own triples from ``GRAPH ?g { ?s ... }``."""
        if not self.schema_subjects:
            return ""
        listed = ", ".join(ref(x) for x in sorted(self.schema_subjects))
        return f"FILTER(!(?{g} = {ref(self.layout.data_graph)} && ?{s} IN ({listed})))"

    def keep_subject(self, s: str = "s") -> str:
        """The same exclusion for a pattern inside ``GRAPH <data graph>``."""
        if not self.schema_subjects:
            return ""
        listed = ", ".join(ref(x) for x in sorted(self.schema_subjects))
        return f"FILTER(?{s} NOT IN ({listed}))"

    def describe(self) -> dict:
        return {
            "census_graphs": list(self.layout.census_graphs),
            "excluded_graphs": list(self.layout.excluded_graphs),
            "excluded_schema_subjects": sorted(self.schema_subjects),
            "rule": "check contract v2, rule 20: the census is built from the data excluding the graph or file "
                    "of any source under comparison",
        }


def fetch_census_input(client, layout: StoreLayout) -> CensusInput:
    named = client.query(PREFIXES + (
        f"ASK {{ {values('g', layout.census_graphs)} GRAPH ?g {{ ?r owl:onProperty ?x }} FILTER isIRI(?r) }}"
    )).boolean
    if not layout.exclude_schema_triples:
        return CensusInput(layout, named_restrictions=bool(named))
    q = PREFIXES + (
        f"SELECT DISTINCT ?s WHERE {{ GRAPH {ref(layout.data_graph)} {{ ?s ?p ?o }} FILTER isIRI(?s) "
        f"{not_data_node('s', layout)} }}"
    )
    return CensusInput(layout, frozenset(r[0].value for r in client.query(q).rows if isinstance(r[0], IRI)), bool(named))


def excluded_triple_counts(client, cin: CensusInput) -> dict[str, int]:
    """The schema's own triples left out of the census, by predicate (for the report)."""
    if not cin.schema_subjects:
        return {}
    q = PREFIXES + (
        f"SELECT ?p (COUNT(*) AS ?n) WHERE {{ GRAPH {ref(cin.layout.data_graph)} {{ ?s ?p ?o }} "
        f"{values('s', cin.schema_subjects)} }} GROUP BY ?p"
    )
    return {r[0].value: int(r[1].lexical) for r in client.query(q).rows if r[0] is not None}


def _targets_subquery(cin: CensusInput) -> str:
    return (
        "{ SELECT DISTINCT ?c WHERE { "
        f"GRAPH ?tg {{ ?y rdf:type|rdfs:subClassOf ?c }} {cin.keep('y', 'tg')} "
        f"FILTER(isIRI(?c) && {non_meta('c')}) {cin.not_restriction('c')}}} }}"
    )


@dataclass
class ObservedHierarchy:
    """Classes observed in the census input and their nearest observed superclasses."""

    parents: dict[str, list[str]]  # class -> direct superclasses that are classes
    data_nodes: set[str] = field(default_factory=set)  # membership targets that are data nodes
    n_targets: int = 0
    raw_edges: dict[str, set[str]] = field(default_factory=dict)  # every kept subClassOf edge from a target
    excluded_edges: list[tuple[str, str]] = field(default_factory=list)  # the schema's own subClassOf edges

    def hierarchy(self) -> Hierarchy:
        return Hierarchy(self.parents)

    def check_excluded_edges(self) -> None:
        """No excluded edge may lead, through any edge, to a census class (a subclass path in a
        census query could otherwise cross it)."""
        edges: dict[str, set[str]] = {k: set(v) for k, v in self.raw_edges.items()}
        for s, o in self.excluded_edges:
            edges.setdefault(s, set()).add(o)
        for s, o in self.excluded_edges:
            seen, stack = set(), [o]
            while stack:
                n = stack.pop()
                if n in seen:
                    continue
                seen.add(n)
                if n in self.parents:
                    raise ValueError(f"excluded edge {s} -> {o} reaches census class {n}; exclude by graph instead")
                stack.extend(edges.get(n, ()))


def fetch_observed_hierarchy(client, cin: CensusInput) -> ObservedHierarchy:
    layout = cin.layout
    ds = cin.dataset()
    targets_q = PREFIXES + f"SELECT ?c {ds} WHERE {{ {_targets_subquery(cin)} }}"
    targets = {r[0].value for r in client.query(targets_q).rows if isinstance(r[0], IRI)}

    data_q = PREFIXES + (
        f"SELECT ?c {ds} WHERE {{ {_targets_subquery(cin)} "
        f"FILTER EXISTS {{ GRAPH {ref(layout.data_graph)} {{ ?c ?dp ?dv . FILTER({non_meta('dp')}) }} }} "
        f"FILTER NOT EXISTS {{ GRAPH ?ug {{ ?us ?c ?uo }} }} }}"
    )
    data_nodes = {r[0].value for r in client.query(data_q).rows if isinstance(r[0], IRI)}

    edges_q = PREFIXES + (
        f"SELECT DISTINCT ?c ?p {ds} WHERE {{ {_targets_subquery(cin)} "
        f"GRAPH ?eg {{ ?c rdfs:subClassOf ?p }} {cin.keep('c', 'eg')} "
        f"FILTER(isIRI(?p) && {non_meta('p')}) {cin.not_restriction('p')}}}"
    )
    raw: dict[str, set[str]] = {}
    for c, p in client.query(edges_q).rows:
        raw.setdefault(c.value, set()).add(p.value)

    excluded: list[tuple[str, str]] = []
    if cin.schema_subjects:
        ex_q = PREFIXES + (
            f"SELECT ?s ?o WHERE {{ GRAPH {ref(layout.data_graph)} {{ ?s rdfs:subClassOf ?o }} "
            f"{values('s', cin.schema_subjects)} FILTER isIRI(?o) }}"
        )
        excluded = sorted((r[0].value, r[1].value) for r in client.query(ex_q).rows)

    classes = {t for t in targets if t not in data_nodes and not is_meta(t)}
    # A class's parents are its nearest superclasses that are classes, looking
    # through data nodes (a data node in the chain is traversed, not listed).
    parents: dict[str, list[str]] = {}
    for c in classes:
        found: set[str] = set()
        seen: set[str] = set()
        stack = list(raw.get(c, ()))
        while stack:
            n = stack.pop()
            if n in seen or n == c:
                continue
            seen.add(n)
            if n in classes:
                found.add(n)
            else:
                stack.extend(raw.get(n, ()))
        parents[c] = sorted(found)
    return ObservedHierarchy(parents, data_nodes & targets, len(targets), raw, excluded)


def fetch_used_terms(client, cin: CensusInput, candidates=None, chunk: int = 2000) -> UsedTerms:
    """The terms the data uses: every IRI object of a data-graph triple (the schema's own triples
    excluded) that is not a data node, and every predicate used in the census graphs.

    With ``candidates`` (the terms of the sources), only those are tested, in chunks: the same set
    where it matters (scope is always intersected with a source's terms), without listing every
    object of a very large data graph (UniProt's has 252 million triples)."""
    layout = cin.layout
    if candidates is None:
        cls_q = PREFIXES + (
            f"SELECT DISTINCT ?o WHERE {{ GRAPH ?g {{ ?s ?p ?o }} VALUES ?g {{ {ref(layout.data_graph)} }} {cin.keep('s', 'g')} "
            f"FILTER isIRI(?o) {not_data_node('o', layout)} }}"
        )
        classes = {r[0].value for r in client.query(cls_q).rows if isinstance(r[0], IRI)}
    else:
        classes = set()
        todo = sorted(set(candidates))
        for i in range(0, len(todo), chunk):
            q = PREFIXES + (
                f"SELECT DISTINCT ?o WHERE {{ {values('o', todo[i:i + chunk])} GRAPH ?g {{ ?s ?p ?o }} "
                f"VALUES ?g {{ {ref(layout.data_graph)} }} {cin.keep('s', 'g')} {not_data_node('o', layout)} }}"
            )
            classes |= {r[0].value for r in client.query(q).rows if isinstance(r[0], IRI)}
    prop_q = PREFIXES + f"SELECT DISTINCT ?p {cin.dataset()} WHERE {{ GRAPH ?g {{ ?s ?p ?o }} {cin.keep('s', 'g')} }}"
    props = {r[0].value for r in client.query(prop_q).rows if isinstance(r[0], IRI)}
    return UsedTerms(frozenset(classes), frozenset(props))


def fetch_shipped_terms(client, layout: StoreLayout, candidates=None, chunk: int = 2000) -> frozenset[str]:
    """IRIs that are subjects in a shipped-vocabulary graph (ChEBI's classes and properties on Rhea);
    with ``candidates``, only those are tested, in chunks (UniProt's taxonomy has 3 million subjects)."""
    if not layout.shipped_graphs:
        return frozenset()
    if candidates is None:
        q = PREFIXES + (
            f"SELECT DISTINCT ?s WHERE {{ {values('g', layout.shipped_graphs)} GRAPH ?g {{ ?s ?p ?o }} FILTER isIRI(?s) }}"
        )
        return frozenset(r[0].value for r in client.query(q).rows if isinstance(r[0], IRI))
    out: set[str] = set()
    todo = sorted(set(candidates))
    for i in range(0, len(todo), chunk):
        q = PREFIXES + (
            f"SELECT DISTINCT ?s WHERE {{ {values('s', todo[i:i + chunk])} {values('g', layout.shipped_graphs)} "
            "GRAPH ?g { ?s ?p ?o } }"
        )
        out |= {r[0].value for r in client.query(q).rows if isinstance(r[0], IRI)}
    return frozenset(out)


def undeclared_terms(client, layout: StoreLayout, declared_classes, declared_properties) -> dict:
    """Predicates used in each census graph, and classes used as rdf:type objects there, that the
    curated assembly does not declare (neither its own ontology nor a shipped vocabulary graph types
    them as a property or class). Reported, never patched (rule 15)."""
    # One graph at a time: a single GROUP BY over every graph sorts hundreds of millions of rows.
    preds: dict[str, dict[str, int]] = {}
    types: dict[str, dict[str, int]] = {}
    for g in layout.census_graphs:
        for p, n in client.query(PREFIXES + f"SELECT ?p (COUNT(*) AS ?n) WHERE {{ GRAPH {ref(g)} {{ ?s ?p ?o }} }} GROUP BY ?p").rows:
            if isinstance(p, IRI):
                preds.setdefault(g, {})[p.value] = int(n.lexical)
        for t, n in client.query(PREFIXES + f"SELECT ?t (COUNT(*) AS ?n) WHERE {{ GRAPH {ref(g)} {{ ?s rdf:type ?t }} }} GROUP BY ?t").rows:
            if isinstance(t, IRI):
                types.setdefault(g, {})[t.value] = int(n.lexical)
    vocab = tuple(dict.fromkeys(layout.curated_graphs + layout.curated_restricted_graphs + layout.shipped_graphs))
    all_p = sorted({p for d in preds.values() for p in d} - set(declared_properties))
    all_t = sorted({t for d in types.values() for t in d} - set(declared_classes))
    typed_p: set[str] = set()
    typed_c: set[str] = set()
    for i in range(0, len(all_p), 500):
        typed_p |= {r[0].value for r in client.query(PREFIXES + (
            f"SELECT DISTINCT ?p WHERE {{ {values('p', all_p[i:i + 500])} {values('vg', vocab)} GRAPH ?vg {{ ?p rdf:type ?k }} "
            "VALUES ?k { rdf:Property owl:ObjectProperty owl:DatatypeProperty owl:AnnotationProperty } }")).rows}
    for i in range(0, len(all_t), 500):
        typed_c |= {r[0].value for r in client.query(PREFIXES + (
            f"SELECT DISTINCT ?t WHERE {{ {values('t', all_t[i:i + 500])} {values('vg', vocab)} GRAPH ?vg {{ "
            "{ ?t rdf:type ?k VALUES ?k { owl:Class rdfs:Class } } UNION { ?t rdfs:subClassOf ?z } } }")).rows}
    out_p = {g: dict(sorted((p, n) for p, n in d.items() if p not in declared_properties and p not in typed_p))
             for g, d in sorted(preds.items())}
    out_t = {g: dict(sorted((t, n) for t, n in d.items() if t not in declared_classes and t not in typed_c))
             for g, d in sorted(types.items())}
    return {
        "rule": "predicates of each census graph, and rdf:type objects there, that neither the curated source nor a "
                "type declaration in its graphs (own ontology and shipped vocabularies) declares; meta-vocabulary "
                "(RDF, RDFS, OWL) is listed too, since the curated ontology does not declare it either",
        "vocabulary_graphs": list(vocab),
        "predicates": {g: d for g, d in out_p.items() if d},
        "classes": {g: d for g, d in out_t.items() if d},
    }
