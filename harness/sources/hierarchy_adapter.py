"""The one adapter between the source builders and the membership and shared-hierarchy definition.

Check contract v2, rules 32, 33 and 35 (science-agent 066225a). The compiler is implementing the same
definition in ``harness/checks/hierarchy.py`` (feat/checker); when that lands, the functions here
become thin calls into it. Until then this module is the builders' only implementation, and no other
builder module states the definition.

Rule 35 (with change-log entries 12 and 13)
    * Shared hierarchy: subclass edges only from graphs other than the data graph and other than the
      sources under comparison, chosen by role (``hierarchy_graphs``: on Rhea the chebi graph; on
      UniProt every census graph but the data graph, chebi and rhea included).
    * Membership: a node's classes are its rdf:type objects (outside RDF, RDFS and OWL, in the census
      graphs) plus its rdfs:subClassOf objects inside the data graph followed transitively through data
      nodes, excluding a compared source's own edges (Rhea's 12 schema subclass edges), and then their
      ancestors in the shared hierarchy. Restrictions, blank or named, are never classes.
Rule 32 (data nodes, parts (b) and (c), as amended by change-log entry 26): a membership target, or an
    entity modelled as a class (the subject of rdfs:subClassOf to an IRI inside the data graph), that is
    the subject of a data-graph triple whose predicate is outside RDF, RDFS and OWL, and that is never used
    as a predicate; the test is the shared module's ``is_data_node``. Only membership targets can be
    listed, so the second clause of (b) never changes the listing: a target is a data node under the
    amended test exactly when it was under the old one. Part (a), "the source does not declare it", is
    applied per source by its consumer.
Rule 33 (what the census lists): every class the data uses directly under the graph's membership
    convention: the targets of the primary predicate (``StoreLayout.membership_predicate``; on Rhea the
    data-graph rdfs:subClassOf, on UniProt rdf:type) that are not data nodes, plus the targets of the
    secondary one that are named classes (present in the shared hierarchy; UniProt's ligand nodes are
    subclasses of ChEBI classes, while its 40,204 entry-local ``<acc>#molecule_*`` targets are not
    named classes and are not listed). Nodes reached only through the shared hierarchy are not listed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from harness.checks.hierarchy import is_data_node as shared_is_data_node
from harness.sources.closure import Hierarchy
from harness.sources.vocab import PREFIXES, RDF_TYPE, RDFS_SUBCLASSOF, StoreLayout, is_meta, non_meta, ref, values
from harness.terms import IRI

DEFINITION = ("check contract v2, rules 32 (amended, change-log entry 26), 33 and 35 with change-log entries 12, 13 "
              "and 14; "
              "harness/sources/hierarchy_adapter.py, data-node test from harness/checks/hierarchy.py")


def _query(client, query: str):
    """Every adapter query, refusing a truncated result (External review 2)."""
    result = client.query(query)
    if result.truncated:
        raise RuntimeError(f"an adapter query returned a truncated result ({result.total_rows} rows); refusing it")
    return result


def not_restriction(var: str) -> str:
    """A FILTER dropping named restrictions (an IRI that is the subject of owl:onProperty in any graph of the
    query's dataset): they are never classes (UniProt's GO publishes its restrictions as IRIs)."""
    return f"FILTER NOT EXISTS {{ GRAPH ?rg_{var} {{ ?{var} owl:onProperty ?rx_{var} }} }}"


def hierarchy_graphs(layout: StoreLayout) -> tuple[str, ...]:
    """Rule 35: the graphs whose subclass edges form the shared hierarchy."""
    return tuple(g for g in layout.census_graphs if g != layout.data_graph)


def hierarchy_dataset(layout: StoreLayout) -> str:
    """A dataset whose default graph is the union of the hierarchy graphs (for paths across them), with the
    same graphs as named graphs (for restriction lookups)."""
    gs = hierarchy_graphs(layout)
    return " ".join(f"FROM {ref(g)}" for g in gs) + " " + " ".join(f"FROM NAMED {ref(g)}" for g in gs)


def _hg(layout: StoreLayout, var: str = "hg") -> str:
    return values(var, hierarchy_graphs(layout))


def direct_membership_branches(node: str, cls: str, layout: StoreLayout, keep: str = "", suffix: str = "",
                               drop: frozenset[str] = frozenset()) -> tuple[str, str]:
    """The two branches of ``direct_membership``: the node's rdf:type objects, and its data-graph rdfs:subClassOf
    objects followed transitively. A caller that joins the membership with other patterns repeats those patterns
    inside each branch (a join distributes over a union), so the store joins them first: QLever otherwise
    evaluates the union unbound (on UniProt, 62 s against 1.7 s for one predicate)."""
    t = f"tg{suffix}"
    dropped = f"FILTER(?{cls} NOT IN ({', '.join(ref(x) for x in sorted(drop))}))" if drop else ""
    return (
        f"GRAPH ?{t} {{ ?{node} rdf:type ?{cls} }} FILTER({non_meta(cls)})",
        f"GRAPH {ref(layout.data_graph)} {{ ?{node} rdfs:subClassOf+ ?{cls} }} {keep} FILTER isIRI(?{cls}) {dropped} "
        f"{not_restriction(cls)}",
    )


def direct_membership(node: str, cls: str, layout: StoreLayout, keep: str = "", suffix: str = "",
                      drop: frozenset[str] = frozenset(), inner: str = "") -> str:
    """A pattern binding ``?cls`` to the node's classes before the shared hierarchy (rule 35, change-log entry 13):
    its rdf:type objects, and its rdfs:subClassOf objects inside the data graph followed transitively (through data
    nodes). ``keep`` drops the node itself when it is a schema subject; ``drop`` lists the objects reachable only
    through a compared source's own edges (Rhea's 12 schema edges), which a property path cannot skip edge by edge
    (``check_dropped`` proves the endpoint filter exact). ``inner`` is joined inside each branch (see
    ``direct_membership_branches``)."""
    a, b = direct_membership_branches(node, cls, layout, keep, suffix, drop)
    return f"{{ {{ {inner} {a} }} UNION {{ {inner} {b} }} }}"


def fetch_dropped(client, layout: StoreLayout, schema_subjects) -> frozenset[str]:
    """Objects of the compared source's own subclass edges in the data graph (Rhea's 12 schema edges), checked to be
    reachable only through those edges: then dropping them as endpoints equals skipping the edges."""
    if not schema_subjects:
        return frozenset()
    rows = _query(client, PREFIXES + (
        f"SELECT DISTINCT ?o WHERE {{ {values('s', schema_subjects)} GRAPH {ref(layout.data_graph)} {{ ?s rdfs:subClassOf ?o }} "
        "FILTER isIRI(?o) }")).rows
    drop = frozenset(r[0].value for r in rows if isinstance(r[0], IRI))
    if drop:
        other = _query(client, PREFIXES + (
            f"SELECT DISTINCT ?o WHERE {{ {values('o', drop)} GRAPH {ref(layout.data_graph)} {{ ?x rdfs:subClassOf ?o }} "
            f"FILTER(?x NOT IN ({', '.join(ref(x) for x in sorted(schema_subjects))})) }}")).rows
        also = sorted(r[0].value for r in other if isinstance(r[0], IRI))
        if also:
            raise ValueError(f"{also} are reached by data edges too; an endpoint filter would not be exact")
        below = _query(client, PREFIXES + (
            f"SELECT DISTINCT ?o WHERE {{ {values('o', drop)} GRAPH {ref(layout.data_graph)} {{ ?o rdfs:subClassOf ?y }} }}")).rows
        cont = sorted(r[0].value for r in below if isinstance(r[0], IRI) and r[0].value not in schema_subjects)
        if cont:
            raise ValueError(f"paths continue in the data graph beyond the dropped objects {cont}")
    return drop


@dataclass
class Listing:
    """What the census lists (rule 33), with the data nodes found on the way (rule 32 (b)(c))."""

    listed: set[str]
    primary_targets: int
    secondary_targets: int
    data_nodes: set[str] = field(default_factory=set)  # membership targets that are data nodes
    not_named: int = 0  # secondary targets that are not named classes (not listed)
    entity_class_targets: int = 0  # targets that are also entities modelled as classes (rule 32 (b), entry 26)


def fetch_listing(client, layout: StoreLayout, keep: str = "") -> Listing:
    """``keep``: a FILTER on ``?y`` dropping the schema's own data-graph triples (CensusInput.keep_subject)."""
    primary = layout.membership_predicate
    if primary == RDFS_SUBCLASSOF:
        prim_pat = f"GRAPH {ref(layout.data_graph)} {{ ?y rdfs:subClassOf ?c }} {keep}"
        sec_pat = "GRAPH ?tg { ?y rdf:type ?c }"
    else:
        prim_pat = "GRAPH ?tg { ?y rdf:type ?c }"
        sec_pat = f"GRAPH {ref(layout.data_graph)} {{ ?y rdfs:subClassOf ?c }} {keep}"
    gs = layout.census_graphs
    ds = " ".join(f"FROM NAMED {ref(g)}" for g in gs)

    def targets(pattern: str) -> set[str]:
        q = PREFIXES + f"SELECT DISTINCT ?c {ds} WHERE {{ {pattern} FILTER(isIRI(?c) && {non_meta('c')}) {not_restriction('c')} }}"
        return {r[0].value for r in _query(client, q).rows if isinstance(r[0], IRI)}

    prim, sec = targets(prim_pat), targets(sec_pat)
    candidates = prim | sec
    # Rule 32 (b) and (c), through the shared module's test: (b) the subject of a data-graph triple whose
    # predicate is outside RDF, RDFS and OWL (a join on the bound candidates, never an EXISTS over the store);
    # (c) never a predicate in any graph read.
    data_subjects: set[str] = set()
    todo = sorted(candidates)
    for i in range(0, len(todo), 2000):
        q = PREFIXES + (
            f"SELECT DISTINCT ?c WHERE {{ {values('c', todo[i:i + 2000])} GRAPH {ref(layout.data_graph)} {{ ?c ?dp ?dv }} "
            f"FILTER({non_meta('dp')}) }}"
        )
        data_subjects |= {r[0].value for r in _query(client, q).rows if isinstance(r[0], IRI)}
    # Rule 32 (b)'s second clause (change-log entry 26): the candidates that are subjects of a data-graph
    # rdfs:subClassOf to an IRI. Only candidates are ever listed, and every candidate is a membership target, so
    # the clause cannot change which candidates are data nodes; it is passed so the test is the shared one.
    entity_classes: set[str] = set()
    for i in range(0, len(todo), 2000):
        q = PREFIXES + (
            f"SELECT DISTINCT ?c WHERE {{ {values('c', todo[i:i + 2000])} GRAPH {ref(layout.data_graph)} "
            "{ ?c rdfs:subClassOf ?k } FILTER isIRI(?k) }"
        )
        entity_classes |= {r[0].value for r in _query(client, q).rows if isinstance(r[0], IRI)}
    predicates = fetch_used_predicates(client, layout)
    data = {c for c in candidates
            if shared_is_data_node(c, targets=candidates, data_subjects=data_subjects, predicates=predicates,
                                   entity_classes=entity_classes)}
    named_sec = set()
    sec_only = sorted(sec - prim - data)
    hg = _hg(layout)
    for i in range(0, len(sec_only) if hierarchy_graphs(layout) else 0, 2000):
        q = PREFIXES + (
            f"SELECT DISTINCT ?c WHERE {{ {values('c', sec_only[i:i + 2000])} {hg} "
            "GRAPH ?hg { { ?c rdfs:subClassOf ?x } UNION { ?x rdfs:subClassOf ?c } } }"
        )
        named_sec |= {r[0].value for r in _query(client, q).rows if isinstance(r[0], IRI)}
    listed = {c for c in (prim - data) | named_sec if not is_meta(c)}
    return Listing(listed, len(prim), len(sec), data & candidates, len(sec_only) - len(named_sec),
                   len(entity_classes & candidates))


def fetch_shared_parents(client, layout: StoreLayout, seeds, chunk: int = 2000) -> Hierarchy:
    """The shared hierarchy above ``seeds`` (the seeds and all their ancestors in the hierarchy graphs),
    with every published subclass edge among them. Restrictions are left out."""
    hg = _hg(layout)
    closure = set(seeds)
    if not hierarchy_graphs(layout):
        return Hierarchy({c: [] for c in closure})
    todo = sorted(set(seeds))
    for i in range(0, len(todo), chunk):
        # The path runs over the union of the hierarchy graphs (the default graph of this dataset), so a chain may
        # cross graphs (External review of b0c3920, item 2); named restrictions are looked up in the named graphs.
        q = PREFIXES + (
            f"SELECT DISTINCT ?a {hierarchy_dataset(layout)} WHERE {{ {values('c', todo[i:i + chunk])} "
            f"?c rdfs:subClassOf+ ?a FILTER(isIRI(?a) && {non_meta('a')}) {not_restriction('a')} }}"
        )
        closure |= {r[0].value for r in _query(client, q).rows if isinstance(r[0], IRI)}
    parents: dict[str, set[str]] = {c: set() for c in closure}
    allc = sorted(closure)
    for i in range(0, len(allc), chunk):
        q = PREFIXES + (
            f"SELECT ?c ?p WHERE {{ {values('c', allc[i:i + chunk])} {hg} GRAPH ?hg {{ ?c rdfs:subClassOf ?p }} "
            "FILTER isIRI(?p) }"
        )
        for c, p in _query(client, q).rows:
            if p.value in closure and p.value != c.value:
                parents[c.value].add(p.value)
    return Hierarchy({c: sorted(ps) for c, ps in parents.items()})


def fetch_shared_terms(client, layout: StoreLayout, candidates, chunk: int = 2000) -> frozenset[str]:
    """Which candidates are classes of the shared hierarchy (subject or object of a hierarchy edge)."""
    hg = _hg(layout)
    out: set[str] = set()
    todo = sorted(set(candidates)) if hierarchy_graphs(layout) else []
    for i in range(0, len(todo), chunk):
        q = PREFIXES + (
            f"SELECT DISTINCT ?c WHERE {{ {values('c', todo[i:i + chunk])} {hg} "
            "GRAPH ?hg { { ?c rdfs:subClassOf ?x } UNION { ?x rdfs:subClassOf ?c } } }"
        )
        out |= {r[0].value for r in _query(client, q).rows if isinstance(r[0], IRI)}
    return frozenset(out)


def other_membership_predicate(layout: StoreLayout) -> str:
    return RDF_TYPE if layout.membership_predicate == RDFS_SUBCLASSOF else RDFS_SUBCLASSOF


def fetch_used_shared(client, layout: StoreLayout, keep_subject: str = "") -> frozenset[str]:
    """The shared-hierarchy classes the data uses: IRI objects of data-graph triples that are subjects of a
    subclass edge in a hierarchy graph (one query per hierarchy graph)."""
    if not hierarchy_graphs(layout):
        return frozenset()
    # One scan of the data graph's IRI objects, joined with the subjects of the hierarchy graphs' edges.
    q = PREFIXES + (
        f"SELECT DISTINCT ?c WHERE {{ {{ SELECT DISTINCT ?c WHERE {{ GRAPH {ref(layout.data_graph)} {{ ?s ?p ?c }} "
        f"{keep_subject} FILTER isIRI(?c) }} }} {_hg(layout)} GRAPH ?hg {{ ?c rdfs:subClassOf ?x }} {not_restriction('c')} }}"
    )
    return frozenset(r[0].value for r in _query(client, q).rows if isinstance(r[0], IRI))


def fetch_used_predicates(client, layout: StoreLayout) -> frozenset[str]:
    """Every predicate used in the census graphs (one query per graph)."""
    out: set[str] = set()
    for g in layout.census_graphs:
        q = PREFIXES + f"SELECT DISTINCT ?p WHERE {{ GRAPH {ref(g)} {{ ?s ?p ?o }} }}"
        out |= {r[0].value for r in _query(client, q).rows if isinstance(r[0], IRI)}
    return frozenset(out)
