"""Membership and the shared hierarchy: the single implementation (S0.1 check
contract v2 rule 16; amendment 2.1 rules 25, 27, 29; rules 32 and 35).

Graph roles (rule 35). A store has one data graph, the graphs of the sources
under comparison (excluded, rule 29: a required setting with no default, e.g.
``core`` on UniProt, possibly an empty list), and hierarchy graphs (every other
graph read: ChEBI on Rhea; taxonomy, GO, keywords, enzymes, locations,
pathways, ... on UniProt).

Membership (rule 35, the same rule on every store). A node's classes are its
rdf:type objects outside the RDF, RDFS and OWL namespaces, plus its
rdfs:subClassOf objects inside the data graph, followed transitively through
the data graph's subclass edges (through data nodes, as RDFS entails: a Rhea
participant under its compound under a ChEBI class), then upwards through the
shared hierarchy and the evaluated source's own subclass axioms (rule 16).
Data-graph subclass edges are membership statements, never hierarchy: the
primary convention on Rhea (compounds and reactions), a secondary one on
UniProt (binding-site ligand nodes ``<acc>#SIP...``). Data-graph edges that
belong to a compared source are left out, as rule 20 leaves them out of the
census: on Rhea the curated schema's own subclass edges (rh:SmallMolecule
rdfs:subClassOf rh:Compound, ...), found mechanically as the data-graph edges
whose subject is a schema term, an IRI with no data-graph triple whose
predicate is outside RDF, RDFS and OWL (``GraphRoles.exclude_schema_subjects``,
the census builder's ``exclude_schema_triples``). A source that does not
declare such an edge does not inherit it.

The shared hierarchy is built only from subclass edges in the hierarchy graphs,
precomputed offline once per store together with the data graph's membership
edges; neither counts against an output's retrieval budget (rule 27). Where a
hierarchy graph materialises its closure (UniProt's taxonomy) the builder loads
direct parents only and reads rdfs:subClassOf transitively (the transitive
completion, rule 35's ruling (a)); a seeded sample of at least 1,000 nodes
compares the recomputed closure with the materialised one, and the result, with
the count of nodes that get several direct parents, goes to the build log. The
build fails only if the recomputed closure misses a materialised ancestor.

Named classes (rules 32 and 35): a named class is one declared in a source or
present in the shared hierarchy. Membership targets that appear only in the data
graph (UniProt's entry-local ``<acc>#...`` nodes) are not named classes.

Data nodes (rule 32). The test is mechanical and the same for every source:

A term is a data node when (a) the source does not declare it as a class, and
(b) in the store it is a membership target (an object of rdf:type, or of
rdfs:subClassOf inside the data graph) or an entity modelled as a class (the
subject of rdfs:subClassOf to an IRI inside the data graph), and it is the subject of at
least one triple in the data graph whose predicate is outside the RDF, RDFS and
OWL namespaces, and (c) it is never used as a predicate in any graph read.

(The "entity modelled as a class" clause, change-log entry 24, covers Rhea's
obsolete reactions: subclasses of rh:Reaction with data triples that nothing is
a subclass of. A schema term such as rh:SmallMolecule has no data triple and
stays out; an entity typed only by rdf:type, such as a UniProt protein, is a
data node only if it is a membership target.)

This module computes (b) and (c), once per store, as ``data_nodes``; the
compiler applies (a) per source, so a class the source declares is never a data
node however the store uses it. A data node in a class position is an entity
constant (rule 5), not vocabulary, and never a named class in the
common-subclass search.

Public API (stable; the census builder imports it):

    is_meta(iri) -> bool
    membership_targets(data_triples, type_triples=()) -> set[str]
    is_data_node(term, *, targets, data_subjects, predicates, declared=frozenset(), entity_classes=frozenset()) -> bool
    classify(data_triples, type_triples=(), hierarchy_edges=(), predicates=None, declared=()) -> Classification
    membership_query(nodes, roles) -> str
    membership_closure(types, membership_parents, hierarchy_parents, source_parents=None) -> set[str]
    GraphRoles(data_graph, hierarchy_graphs, excluded_graphs, closure_graphs=(), *, exclude_schema_subjects: bool)
    SharedHierarchy(roles, parents, data_nodes=frozenset(), provenance={}, membership_parents={}, positions=None)
    SharedHierarchy.positions -> Occupancy | None   # predicate positions hierarchy nodes hold, per hierarchy graph
    SharedHierarchy.edge_graphs -> dict | None   # the graphs of the edges of nodes not placed by their positions alone
    SharedHierarchy.edge_in(child, parent, graphs) / SharedHierarchy.descendants_within(cls, graphs)   # per-graph edges
    Occupancy.held(node, pred, side, graphs) / Occupancy.occupants(pred, side, graphs)   # rule 20, entries 49 and 61
    SharedHierarchy.closure_parents() -> dict[str, tuple[str, ...]]   # hierarchy plus membership edges
    SharedHierarchy.build(client, roles) -> SharedHierarchy
    SharedHierarchy.build_streamed(endpoint, roles, out_path, log_path, timeout_s=280.0, sample=1000, seed=20271018)
    SharedHierarchy.save(path) / SharedHierarchy.load(path)
"""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import io
import json
import random
import resource
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path

import httpx

from harness.checks.closure import OWL, RDF, RDFS
from harness.terms import IRI

META = (RDF, RDFS, OWL)
RDF_TYPE = RDF + "type"
RDFS_SUBCLASSOF = RDFS + "subClassOf"
# v4 (change-log entry 72): a hierarchy that records its positions per graph and which graph holds each edge. A v3
# file still loads, for curated and generated sources; a census refuses it, so a reader never judges census
# support from a hierarchy that cannot tell graphs apart, and a reader older than v4 refuses a v4 file.
SCHEMA_VERSION = "shared-hierarchy-v4"
LEGACY_VERSION = "shared-hierarchy-v3"


def is_meta(iri: str) -> bool:
    """In the RDF, RDFS or OWL namespace."""
    return iri.startswith(META)


# -- the pure part: classify a node set from given triples (testable without a store) ---------


@dataclass(frozen=True)
class Classification:
    targets: frozenset[str]  # membership targets
    data_nodes: frozenset[str]  # parts (a), (b) and (c) of the data-node test
    named_classes: frozenset[str]  # declared, or present in the shared hierarchy


def membership_targets(
    data_triples: Iterable[tuple[str, str, str]], type_triples: Iterable[tuple[str, str]] = ()
) -> set[str]:
    """Objects of rdf:type (outside RDF, RDFS and OWL) in any graph read, and of
    rdfs:subClassOf inside the data graph. ``data_triples``: (s, p, o) of the data
    graph; ``type_triples``: (s, class) rdf:type statements of the other graphs read."""
    out = set()
    for _, p, o in data_triples:
        if (p == RDF_TYPE and not is_meta(o)) or p == RDFS_SUBCLASSOF:
            out.add(o)
    out.update(o for _, o in type_triples if not is_meta(o))
    return out


def is_data_node(
    term: str,
    *,
    targets: set[str] | frozenset[str],
    data_subjects: set[str] | frozenset[str],
    predicates: set[str] | frozenset[str],
    declared: Iterable[str] = frozenset(),
    entity_classes: set[str] | frozenset[str] = frozenset(),
) -> bool:
    """The data-node test (module docstring): (a) not declared by the source,
    (b) a membership target, or an entity modelled as a class (``entity_classes``:
    subjects of rdfs:subClassOf inside the data graph), that is the subject of a
    data-graph triple with a predicate outside RDF, RDFS and OWL
    (``data_subjects``), (c) never a predicate in any graph read (``predicates``)."""
    return (term not in set(declared) and (term in targets or term in entity_classes)
            and term in data_subjects and term not in predicates)


def classify(
    data_triples: Iterable[tuple[str, str, str]],
    type_triples: Iterable[tuple[str, str]] = (),
    hierarchy_edges: Iterable[tuple[str, str]] = (),
    predicates: Iterable[str] | None = None,
    declared: Iterable[str] = (),
) -> Classification:
    """Membership targets, data nodes and named classes of a store given as
    triples: the data graph's triples, the other graphs' rdf:type statements, the
    hierarchy graphs' subclass edges (child, parent), and the predicates used in
    every graph read (default: the data graph's)."""
    data_triples = list(data_triples)
    declared = set(declared)
    targets = membership_targets(data_triples, type_triples)
    subjects = {s for s, p, _ in data_triples if not is_meta(p)}
    preds = set(predicates) if predicates is not None else {p for _, p, _ in data_triples}
    entity_classes = {s for s, p, o in data_triples if p == RDFS_SUBCLASSOF and not o.startswith("_:")}
    data_nodes = {t for t in targets | entity_classes
                  if is_data_node(t, targets=targets, data_subjects=subjects, predicates=preds, declared=declared,
                                  entity_classes=entity_classes)}
    in_hierarchy = {c for edge in hierarchy_edges for c in edge}
    return Classification(frozenset(targets), frozenset(data_nodes), frozenset((in_hierarchy - data_nodes) | declared))


def membership_closure(
    types: Iterable[str],
    membership_parents: dict[str, Iterable[str]],
    hierarchy_parents: dict[str, Iterable[str]],
    source_parents: dict[str, Iterable[str]] | None = None,
) -> set[str]:
    """Every class of a node (rule 35): its membership objects (``types``),
    followed transitively through the data graph's membership edges (schema
    edges of a compared source already left out), the shared hierarchy and the
    evaluated source's own subclass axioms. Reflexive on ``types``."""
    out: set[str] = set()
    stack = list(types)
    while stack:
        x = stack.pop()
        if x in out:
            continue
        out.add(x)
        stack.extend(membership_parents.get(x, ()))
        stack.extend(hierarchy_parents.get(x, ()))
        if source_parents is not None:
            stack.extend(source_parents.get(x, ()))
    return out


# -- graph roles and the store queries -------------------------------------------------------


@dataclass(frozen=True)
class GraphRoles:
    data_graph: str
    hierarchy_graphs: tuple[str, ...]
    excluded_graphs: tuple[str, ...]  # rule 29: required, possibly empty
    closure_graphs: tuple[str, ...] = ()  # hierarchy graphs that materialise their closure (load direct parents)
    # Required, keyword-only, no default (like rule 29's excluded graphs): True when the compared source's
    # schema triples sit inside the data graph (Rhea), False when it is a separate graph (UniProt's core).
    exclude_schema_subjects: bool = field(kw_only=True)

    def __post_init__(self) -> None:
        if self.excluded_graphs is None or isinstance(self.excluded_graphs, str):
            raise ValueError("rule 29: the graphs excluded from the shared hierarchy must be given as a list (it may be empty)")
        if not isinstance(self.exclude_schema_subjects, bool):
            raise ValueError("exclude_schema_subjects must be stated explicitly as True (the compared source's schema sits "
                             "inside the data graph, Rhea) or False (it is a separate graph, UniProt)")
        object.__setattr__(self, "excluded_graphs", tuple(self.excluded_graphs))
        object.__setattr__(self, "hierarchy_graphs",
                           tuple(g for g in self.hierarchy_graphs if g not in self.excluded_graphs and g != self.data_graph))
        object.__setattr__(self, "closure_graphs", tuple(self.closure_graphs))

    @property
    def read(self) -> tuple[str, ...]:
        return (self.data_graph, *self.hierarchy_graphs)


def _values(var: str, iris: Iterable[str]) -> str:
    return f"VALUES ?{var} {{ {' '.join(f'<{i}>' for i in iris)} }}"


def _nonmeta(var: str) -> str:
    return " && ".join(f'!STRSTARTS(STR(?{var}), "{ns}")' for ns in META)


def membership_query(nodes: Iterable[str], roles: GraphRoles) -> str:
    """The membership triples of some nodes (rule 35): their rdf:type objects in
    any graph read and their rdfs:subClassOf objects in the data graph, as rows
    (?s ?t ?p)."""
    return (f"SELECT ?s ?t ?p WHERE {{ {_values('s', nodes)} "
            f"{{ {_values('g', roles.read)} GRAPH ?g {{ ?s <{RDF_TYPE}> ?t }} BIND(<{RDF_TYPE}> AS ?p) }} UNION "
            f"{{ GRAPH <{roles.data_graph}> {{ ?s <{RDFS_SUBCLASSOF}> ?t }} BIND(<{RDFS_SUBCLASSOF}> AS ?p) }} }}")


def _queries(roles: GraphRoles) -> dict[str, str]:
    d = roles.data_graph
    targets_body = (f"{{ {_values('g', roles.read)} GRAPH ?g {{ ?y <{RDF_TYPE}> ?c }} FILTER(isIRI(?c) && {_nonmeta('c')}) }} "
                    f"UNION {{ GRAPH <{d}> {{ ?y <{RDFS_SUBCLASSOF}> ?c }} FILTER(isIRI(?c)) }}")
    return {
        "targets": f"SELECT DISTINCT ?c WHERE {{ {targets_body} }}",
        "data_nodes": (f"SELECT DISTINCT ?c WHERE {{ {{ SELECT DISTINCT ?c WHERE {{ {targets_body} }} }} "
                       f"GRAPH <{d}> {{ ?c ?dp ?dv }} FILTER({_nonmeta('dp')}) "
                       f"FILTER NOT EXISTS {{ {_values('ug', roles.read)} GRAPH ?ug {{ ?us ?c ?uo }} }} }}"),
        # (b)'s second clause, a query of its own (a UNION with the first runs past QLever's cap on UniProt)
        "entity_class_data_nodes": (
            f"SELECT DISTINCT ?c WHERE {{ {{ SELECT DISTINCT ?c WHERE {{ GRAPH <{d}> {{ ?c <{RDFS_SUBCLASSOF}> ?k }} "
            f"FILTER(isIRI(?c) && isIRI(?k)) }} }} GRAPH <{d}> {{ ?c ?dp ?dv }} FILTER({_nonmeta('dp')}) "
            f"FILTER NOT EXISTS {{ {_values('ug', roles.read)} GRAPH ?ug {{ ?us ?c ?uo }} }} }}"),
    }


def _membership_edges_query(roles: GraphRoles, schema_only: bool = False) -> str:
    """The data graph's subclass edges; with ``schema_only``, those whose subject
    is a schema term (no data-graph triple with a predicate outside RDF, RDFS and OWL)."""
    d = roles.data_graph
    schema = (f"FILTER NOT EXISTS {{ GRAPH <{d}> {{ ?c ?dp ?dv }} FILTER({_nonmeta('dp')}) }}" if schema_only else "")
    return (f"SELECT DISTINCT ?c ?p WHERE {{ GRAPH <{d}> {{ ?c <{RDFS_SUBCLASSOF}> ?p }} "
            f"FILTER(isIRI(?c) && isIRI(?p) && ?c != ?p) {schema} }}")


def _edges_query(graph: str) -> str:
    return (f"SELECT DISTINCT ?c ?p WHERE {{ GRAPH <{graph}> {{ ?c <{RDFS_SUBCLASSOF}> ?p }} "
            f"FILTER(isIRI(?c) && isIRI(?p) && ?c != ?p && {_nonmeta('p')}) }}")


def _direct_edges_query(graph: str) -> str:
    """Direct parents inside a graph whose subclass closure is materialised and
    forms a tree: counting only ancestors that are themselves nodes of the graph
    (subjects of its subclass triples), the ancestor with exactly one ancestor fewer."""
    sub = RDFS_SUBCLASSOF

    def internal(v: str) -> str:
        return f"FILTER EXISTS {{ GRAPH <{graph}> {{ ?{v} <{sub}> ?w_{v} }} }}"

    return (f"SELECT ?c ?p WHERE {{ "
            f"{{ SELECT ?c (COUNT(?x) AS ?nc) WHERE {{ GRAPH <{graph}> {{ ?c <{sub}> ?x }} {internal('x')} }} GROUP BY ?c }} "
            f"GRAPH <{graph}> {{ ?c <{sub}> ?p }} {internal('p')} "
            f"OPTIONAL {{ SELECT ?p (COUNT(?y) AS ?np0) WHERE {{ GRAPH <{graph}> {{ ?p <{sub}> ?y }} {internal('y')} }} GROUP BY ?p }} "
            f"BIND(COALESCE(?np0, 0) AS ?np) FILTER(?np = ?nc - 1) }}")


def _external_edges_query(graph: str) -> str:
    """Subclass edges from a closure graph's nodes to objects outside it (UniProt's
    top taxa to up:Taxon)."""
    sub = RDFS_SUBCLASSOF
    return (f"SELECT ?c ?p WHERE {{ GRAPH <{graph}> {{ ?c <{sub}> ?p }} FILTER(isIRI(?p)) "
            f"FILTER NOT EXISTS {{ GRAPH <{graph}> {{ ?p <{sub}> ?w }} }} }}")


def _occupancy_query(graph: str, side: str, roles: GraphRoles) -> str:
    """The predicates each shared-hierarchy node (a node of a subclass edge in a hierarchy graph) holds in ``graph``,
    as the subject (``side`` "subject") or the IRI object ("object") of a triple, as rows (?n ?p)."""
    sub = RDFS_SUBCLASSOF
    nodes = (f"{{ SELECT DISTINCT ?n WHERE {{ {_values('hg', roles.hierarchy_graphs)} GRAPH ?hg {{ "
             f"{{ ?n <{sub}> ?x FILTER(isIRI(?x) && {_nonmeta('x')}) }} UNION {{ ?y <{sub}> ?n FILTER(isIRI(?y)) }} }} "
             f"FILTER(isIRI(?n) && {_nonmeta('n')}) }} }}")
    triple = "?n ?p ?o" if side == "subject" else "?s ?p ?n"
    return f"SELECT DISTINCT ?n ?p WHERE {{ {nodes} GRAPH <{graph}> {{ {triple} }} }}"


def _graph_predicates_query(graph: str) -> str:
    return f"SELECT DISTINCT ?p WHERE {{ GRAPH <{graph}> {{ ?s ?p ?o }} }}"


def _position_query(graph: str, pred: str, side: str) -> str:
    """The IRIs at one predicate position in ``graph``: subjects (``side`` "subject") or IRI objects ("object")."""
    triple = f"?n <{pred}> ?o" if side == "subject" else f"?s <{pred}> ?n"
    return f"SELECT DISTINCT ?n WHERE {{ GRAPH <{graph}> {{ {triple} }} FILTER(isIRI(?n)) }}"


def _is_iri(term: str) -> bool:
    """An IRI as the builders keep a node: the TSV stream writes a blank node as ``_:<label>`` (``build`` keeps IRIs
    only)."""
    return not term.startswith("_:")


def _query_nodes(parents: dict[str, Iterable[str]]) -> frozenset[str]:
    """The hierarchy's nodes that the edges the builders keep show to be in ``_occupancy_query``'s node set, as the
    query reads a subclass triple: an IRI parent of an IRI child, and an IRI child of an IRI parent outside RDF, RDFS
    and OWL. Every kept edge is a subclass triple of a hierarchy graph. An edge with a blank node at either end shows
    nothing (external reviewer 6 and 6.1 reviews of 0763b60: a closure graph's external edge from ``_:b``). The edges cannot decide
    the rest, such as a child whose only kept parents are such terms or blank nodes: the query may still select it
    through a triple the builders do not keep as an edge, a reflexive one or another superclass (``_eligible_query``)."""
    out = {p for c, ps in parents.items() if _is_iri(c) for p in ps if _is_iri(p)}
    out.update(c for c, ps in parents.items() if _is_iri(c) and any(_is_iri(p) and not is_meta(p) for p in ps))
    return frozenset(out)


def _eligible_query(nodes: Iterable[str], roles: GraphRoles) -> str:
    """Which of ``nodes`` ``_occupancy_query``'s node set holds, read from the hierarchy graphs themselves: the subject
    of a subclass triple with an IRI object outside RDF, RDFS and OWL, or the object of one with an IRI subject
    (reflexive triples included, as the query reads them)."""
    sub = RDFS_SUBCLASSOF
    return (f"SELECT DISTINCT ?n WHERE {{ {_values('n', nodes)} {_values('hg', roles.hierarchy_graphs)} GRAPH ?hg {{ "
            f"{{ ?n <{sub}> ?x FILTER(isIRI(?x) && {_nonmeta('x')}) }} UNION {{ ?y <{sub}> ?n FILTER(isIRI(?y)) }} }} }}")


def _occupancy_rows(fetch, roles: GraphRoles, keep, note=None, *, parents: dict[str, Iterable[str]]):
    """The rows of ``_occupancy_query`` for every hierarchy graph and side, as (node, graph, predicate, side), for the
    nodes in ``keep`` (the hierarchy's classes) that the query's node set holds (``_query_nodes(parents)``, and for
    the nodes the kept edges cannot decide, ``_eligible_query``), read
    one predicate position at a time: the graph's predicates, then for each the IRIs at its subject and object
    position, kept when they are such nodes outside RDF, RDFS and OWL. The single query per graph and side joins
    every hierarchy node with every triple of the graph and runs past QLever's cap on UniProt's taxonomy (52 million
    materialised subclass triples); one position is one index scan. ``fetch(query)`` yields rows as tuples of
    strings."""
    selected = set(_query_nodes(parents))
    undecided = sorted(n for n in keep if n not in selected and _is_iri(n) and not is_meta(n))
    for i in range(0, len(undecided), 500):  # the few nodes the kept edges cannot decide: ask the graphs
        selected.update(row[0] for row in fetch(_eligible_query(undecided[i:i + 500], roles)))
    keep = {n for n in keep if n in selected}
    for g in roles.hierarchy_graphs:
        preds = sorted({row[0] for row in fetch(_graph_predicates_query(g))})
        for side in ("subject", "object"):
            n_rows = 0
            for pred in preds:
                for row in fetch(_position_query(g, pred, side)):
                    if row[0] in keep and not is_meta(row[0]):
                        n_rows += 1
                        yield row[0], g, pred, side
            if note is not None:
                note(f"graph {g}: {n_rows} (node, predicate) pairs with hierarchy nodes as the {side}, "
                     f"over {len(preds)} predicates, one position per query")


class _EdgeGraphs:
    """Records, while the hierarchy graphs' edges are read graph by graph, the graphs each subclass edge comes from:
    in full for a node with edges in two or more graphs; for the others the graph they came from (``first``), checked
    against the node's positions once they are read (``record``)."""

    def __init__(self, graphs: tuple[str, ...]):
        self.graphs = graphs
        self.first: dict[str, int] = {}
        self.multi: dict[str, dict[str, set[int]]] = {}

    def add(self, c: str, p: str, gi: int, before: Iterable[str]) -> None:
        """Edge c -> p read from graph number ``gi``; ``before``: c's parents read until now."""
        f = self.first.setdefault(c, gi)
        if f != gi and c not in self.multi:
            self.multi[c] = {q: {f} for q in before}
        if c in self.multi:
            self.multi[c].setdefault(p, set()).add(gi)

    def record(self, parents: dict[str, tuple[str, ...]], positions: Occupancy) -> dict[str, dict[str, tuple[str, ...]]]:
        """The explicit record: every node with edges in several graphs, and every node whose one graph is not the
        single hierarchy graph in which its positions show it as the subject of rdfs:subClassOf (``edge_in`` reads
        the graph from the positions for every node left out)."""
        out = {c: {p: tuple(self.graphs[i] for i in sorted(gs)) for p, gs in sorted(m.items())}
               for c, m in self.multi.items()}
        for c, f in self.first.items():
            if c in out:
                continue
            if _subject_graphs(positions, c) != (self.graphs[f],):
                out[c] = {p: (self.graphs[f],) for p in parents.get(c, ())}
        return dict(sorted(out.items()))


def _subject_graphs(positions: Occupancy, node: str) -> tuple[str, ...]:
    """The hierarchy graphs in which ``node`` is the subject of an rdfs:subClassOf triple, from its positions."""
    i = positions.nodes.get(node)
    if i is None:
        return ()
    cache = positions.__dict__.setdefault("_subject_graphs", {})
    if i not in cache:
        cache[i] = tuple(sorted(g for g, p, side in positions.signatures[i] if p == RDFS_SUBCLASSOF and side == "subject"))
    return cache[i]


@dataclass(frozen=True)
class Occupancy:
    """Which shared-hierarchy nodes each hierarchy graph holds at each predicate position: the evidence rule 20 needs
    to judge census support against what the store would match, the data graph and the hierarchy graphs (change-log
    entries 49 and 61). ``signatures[i]`` is a set of (graph, predicate, side), side "subject" or "object";
    ``nodes`` maps a hierarchy node to its signature's index (nodes share a few signatures, ChEBI's 204,585 classes
    among them). Only the nodes of the hierarchy's subclass edges are recorded: the data graph is the census's."""

    signatures: tuple[frozenset[tuple[str, str, str]], ...] = ()
    nodes: dict[str, int] = field(default_factory=dict)

    @staticmethod
    def build(rows: Iterable[tuple[str, str, str, str]], keep=None) -> Occupancy:
        """From (node, graph, predicate, side) rows; ``keep``: the nodes to record (all when None)."""
        b = _OccupancyBuilder()
        for n, g, p, side in rows:
            if keep is None or n in keep:
                b.add(n, (g, p, side))
        return b.done()

    def held(self, node: str, pred: str, side: str, graphs: Iterable[str]) -> tuple[str, ...]:
        """The graphs among ``graphs`` in which ``node`` holds the position."""
        i = self.nodes.get(node)
        if i is None:
            return ()
        sig = self.signatures[i]
        return tuple(g for g in graphs if (g, pred, side) in sig)

    def occupants(self, pred: str, side: str, graphs: Iterable[str]) -> frozenset[str]:
        """The nodes holding the position in any of ``graphs``."""
        key = (pred, side, tuple(sorted(graphs)))
        cache = self.__dict__.setdefault("_occupants", {})
        if key not in cache:
            idx = {i for i, sig in enumerate(self.signatures) if any((g, pred, side) in sig for g in key[2])}
            cache[key] = frozenset(n for n, i in self.nodes.items() if i in idx) if idx else frozenset()
        return cache[key]

    def to_json(self) -> dict:
        return {"signatures": [sorted(list(x) for x in sig) for sig in self.signatures],
                "nodes": dict(sorted(self.nodes.items()))}

    @staticmethod
    def from_json(obj: dict) -> Occupancy:
        return Occupancy(tuple(frozenset(tuple(x) for x in sig) for sig in obj["signatures"]), dict(obj["nodes"]))


class _OccupancyBuilder:
    """Interns signatures as rows arrive, so a large hierarchy holds one integer per node."""

    def __init__(self) -> None:
        self.index: dict[frozenset, int] = {frozenset(): 0}
        self.sigs: list[frozenset] = [frozenset()]
        self.step: dict[tuple[int, tuple], int] = {}
        self.nodes: dict[str, int] = {}

    def add(self, node: str, item: tuple[str, str, str]) -> None:
        cur = self.nodes.get(node, 0)
        nxt = self.step.get((cur, item))
        if nxt is None:
            sig = self.sigs[cur] | {item}
            nxt = self.index.get(sig)
            if nxt is None:
                nxt = self.index[sig] = len(self.sigs)
                self.sigs.append(sig)
            self.step[(cur, item)] = nxt
        self.nodes[node] = nxt

    def done(self) -> Occupancy:
        # the signatures in use, renumbered in a deterministic order
        used = sorted(set(self.nodes.values()), key=lambda i: sorted(self.sigs[i]))
        new = {old: k for k, old in enumerate(used)}
        return Occupancy(tuple(self.sigs[i] for i in used), {n: new[i] for n, i in sorted(self.nodes.items())})


class _UnionMap:
    """Two node -> set maps read as their union, through ``get`` only (how the closure's searches read edges). A key in
    both gives an iterable over both sets, possibly naming a node twice, and allocates nothing in proportion to them
    (a search counts the nodes it holds as it reads them)."""

    def __init__(self, a: dict[str, set[str]], b: dict[str, set[str]]):
        self._a, self._b = a, b

    def get(self, key: str, default=()):
        a, b = self._a.get(key), self._b.get(key)
        if a is None:
            return default if b is None else b
        return a if b is None else chain(a, b)


@dataclass(frozen=True)
class SharedHierarchy:
    roles: GraphRoles
    parents: dict[str, tuple[str, ...]]  # direct subclass edges of the hierarchy graphs
    data_nodes: frozenset[str] = frozenset()  # parts (b) and (c) of the data-node test
    provenance: dict = field(default_factory=dict)
    membership_parents: dict[str, tuple[str, ...]] = field(default_factory=dict)  # data-graph subclass edges, minus a
    #   compared source's schema edges: membership, followed transitively, never named classes
    # The hierarchy graphs' predicate positions held by hierarchy nodes (change-log entries 49 and 61). None: not
    # recorded (a hierarchy built before entry 61), which a census source refuses; an empty Occupancy: none held.
    positions: Occupancy | None = None
    # Which hierarchy graphs hold each subclass edge, for the nodes the positions do not place: a node with edges in
    # several hierarchy graphs, or whose one edge graph is not the single graph where its positions show it as the
    # subject of rdfs:subClassOf. Every other node's edges are in that graph. A GRAPH-confined pattern sees only its
    # graph's edges (External reviews of 7e81a1e). None: not recorded (a hierarchy built before), which a census refuses
    # when there are several hierarchy graphs; with one, every edge is in it.
    edge_graphs: dict[str, dict[str, tuple[str, ...]]] | None = None
    # The schema version a file declared (from_json); None for a hierarchy built or made in memory, whose version
    # follows its content (``version``).
    schema_version: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.roles, GraphRoles):
            raise ValueError("rule 29: a shared hierarchy needs its graph roles, with the excluded graphs set explicitly")

    @staticmethod
    def empty(roles: GraphRoles) -> SharedHierarchy:
        """No hierarchy graph content: no edges, and no positions held (recorded, empty)."""
        return SharedHierarchy(roles, {}, positions=Occupancy(), edge_graphs={})

    @property
    def classes(self) -> frozenset[str]:
        """Classes present in the shared hierarchy (named classes, rule 35)."""
        cache = self.__dict__.get("_classes")
        if cache is None:
            cache = frozenset(self.parents) | frozenset(p for ps in self.parents.values() for p in ps)
            object.__setattr__(self, "_classes", cache)
        return cache

    def closure_parents(self) -> dict[str, tuple[str, ...]]:
        """The edges membership follows upwards: the shared hierarchy plus the data
        graph's membership edges (rule 35)."""
        cache = self.__dict__.get("_closure_parents")
        if cache is None:
            if not self.membership_parents:
                cache = self.parents
            else:
                cache = dict(self.parents)
                for c, ps in self.membership_parents.items():
                    cache[c] = tuple(dict.fromkeys((*cache.get(c, ()), *ps)))
            object.__setattr__(self, "_closure_parents", cache)
        return cache

    @property
    def version(self) -> str:
        """The schema version: the one a file declared, else v4 when the positions and the edge record are both
        recorded, v3 otherwise."""
        if self.schema_version is not None:
            return self.schema_version
        return SCHEMA_VERSION if self.positions is not None and self.edge_graphs is not None else LEGACY_VERSION

    def to_v4(self) -> SharedHierarchy:
        """The same hierarchy as v4, carried over unchanged, when it can be done without the store: its positions
        recorded, and its edge record too, or a single hierarchy graph, which holds every edge (an empty record)."""
        if self.positions is None:
            raise ValueError("this hierarchy records no positions: it cannot become v4 without the store; rebuild it")
        edges = self.edge_graphs
        if edges is None:
            if len(self.roles.hierarchy_graphs) > 1:
                raise ValueError("this hierarchy has several hierarchy graphs and no edge record: it cannot become v4 "
                                 "without the store; rebuild it")
            edges = {}
        return dataclasses.replace(self, edge_graphs=edges, schema_version=SCHEMA_VERSION)

    @property
    def edge_graphs_recorded(self) -> bool:
        """Whether every edge's hierarchy graphs are known: recorded, or a single hierarchy graph holds them all."""
        return self.edge_graphs is not None or len(self.roles.hierarchy_graphs) <= 1

    def edge_in(self, child: str, parent: str, graphs: Iterable[str]) -> bool:
        """Whether one of ``graphs`` holds the subclass edge child -> parent."""
        if parent not in self.parents.get(child, ()):
            return False
        gs = set(graphs)
        if self.edge_graphs is not None and child in self.edge_graphs:
            return any(g in gs for g in self.edge_graphs[child].get(parent, ()))
        if len(self.roles.hierarchy_graphs) <= 1:
            return bool(gs & set(self.roles.hierarchy_graphs))
        if self.positions is None or self.edge_graphs is None:
            raise ValueError("the shared hierarchy does not record which hierarchy graph holds each edge; rebuild it")
        return any(g in gs for g in _subject_graphs(self.positions, child))

    def descendants_within(self, cls: str, graphs: Iterable[str]) -> Iterable[str]:
        """The nodes strictly under ``cls`` through subclass edges held in ``graphs`` (breadth first, each once), as
        a store matching only those graphs would reach them."""
        gs = tuple(graphs)
        children = self.children()
        seen, todo = {cls}, [cls]
        while todo:
            nxt = []
            for p in todo:
                for c in sorted(children.get(p, ())):
                    if c not in seen and self.edge_in(c, p, gs):
                        seen.add(c)
                        nxt.append(c)
                        yield c
            todo = nxt

    def children(self) -> dict[str, set[str]]:
        cache = self.__dict__.get("_children")
        if cache is None:
            cache = {}
            for c, ps in self.parents.items():
                for p in ps:
                    cache.setdefault(p, set()).add(c)
            object.__setattr__(self, "_children", cache)
        return cache

    def closure_children(self):
        """The inverse of ``closure_parents``, read through ``get``: the hierarchy's children plus the data graph's
        membership edges read downwards. Without membership edges it is ``children`` itself (``closure_parents`` is
        then ``parents``). Computed once, shared by every compiler on this hierarchy."""
        cache = self.__dict__.get("_closure_children")
        if cache is None:
            if not self.membership_parents:
                cache = self.children()
            else:
                extra: dict[str, set[str]] = {}
                for c, ps in self.membership_parents.items():
                    for p in ps:
                        extra.setdefault(p, set()).add(c)
                cache = _UnionMap(self.children(), extra)
            object.__setattr__(self, "_closure_children", cache)
        return cache

    # -- storage ------------------------------------------------------------------------------

    def to_json(self) -> dict:
        r = self.roles
        return {
            "schema_version": self.version,
            "roles": {"data_graph": r.data_graph, "hierarchy_graphs": list(r.hierarchy_graphs),
                      "excluded_graphs": list(r.excluded_graphs), "closure_graphs": list(r.closure_graphs),
                      "exclude_schema_subjects": r.exclude_schema_subjects},
            "parents": {c: list(ps) for c, ps in sorted(self.parents.items())},
            "membership_parents": {c: list(ps) for c, ps in sorted(self.membership_parents.items())},
            "data_nodes": sorted(self.data_nodes),
            "provenance": self.provenance,
            **({"positions": self.positions.to_json()} if self.positions is not None else {}),
            **({"edge_graphs": {c: {p: list(gs) for p, gs in m.items()} for c, m in sorted(self.edge_graphs.items())}}
               if self.edge_graphs is not None else {}),
        }

    @staticmethod
    def from_json(obj: dict) -> SharedHierarchy:
        version = obj.get("schema_version")
        if version not in (SCHEMA_VERSION, LEGACY_VERSION):
            raise ValueError(f"unsupported shared-hierarchy version {version!r}")
        if version == SCHEMA_VERSION and not ("positions" in obj and "edge_graphs" in obj):
            raise ValueError(f"a {SCHEMA_VERSION} hierarchy records its positions and its edge record; this one lacks "
                             f"{[k for k in ('positions', 'edge_graphs') if k not in obj]}")
        r = obj["roles"]
        if "exclude_schema_subjects" not in r:
            raise ValueError("the saved graph roles do not state exclude_schema_subjects")
        roles = GraphRoles(r["data_graph"], tuple(r["hierarchy_graphs"]), tuple(r["excluded_graphs"]),
                           tuple(r.get("closure_graphs", ())), exclude_schema_subjects=r["exclude_schema_subjects"])
        return SharedHierarchy(roles, {c: tuple(ps) for c, ps in obj["parents"].items()},
                               frozenset(obj.get("data_nodes", ())), obj.get("provenance", {}),
                               {c: tuple(ps) for c, ps in obj["membership_parents"].items()},
                               Occupancy.from_json(obj["positions"]) if "positions" in obj else None,
                               {c: {p: tuple(gs) for p, gs in m.items()} for c, m in obj["edge_graphs"].items()}
                               if "edge_graphs" in obj else None, version)

    def save(self, path: Path) -> None:
        """Gzipped JSON, written edge list by edge list (the same document as
        ``to_json``, without holding a second copy of the hierarchy in memory).
        The gzip header carries no time stamp or name, so equal content saves to
        equal bytes and the file's sha256 names its content."""
        r = self.roles
        roles = {"data_graph": r.data_graph, "hierarchy_graphs": list(r.hierarchy_graphs),
                 "excluded_graphs": list(r.excluded_graphs), "closure_graphs": list(r.closure_graphs),
                 "exclude_schema_subjects": r.exclude_schema_subjects}
        with open(path, "wb") as fh, gzip.GzipFile(filename="", fileobj=fh, mode="wb", mtime=0) as gz, \
                io.TextIOWrapper(gz, encoding="utf-8") as f:
            f.write(f'{{"schema_version": {json.dumps(self.version)}, "roles": {json.dumps(roles)}')
            for name, edges in (("parents", self.parents), ("membership_parents", self.membership_parents)):
                f.write(f', {json.dumps(name)}: {{')
                for i, c in enumerate(sorted(edges)):
                    f.write(("," if i else "") + json.dumps(c) + ":" + json.dumps(list(edges[c])))
                f.write("}")
            f.write(f', "data_nodes": {json.dumps(sorted(self.data_nodes))}, "provenance": {json.dumps(self.provenance)}')
            if self.positions is not None:
                sigs = self.positions.to_json()["signatures"]
                f.write(f', "positions": {{"signatures": {json.dumps(sigs)}, "nodes": {{')
                for i, (n, k) in enumerate(sorted(self.positions.nodes.items())):
                    f.write(("," if i else "") + json.dumps(n) + ":" + str(k))
                f.write("}}")
            if self.edge_graphs is not None:
                f.write(', "edge_graphs": ' + json.dumps({c: {p: list(gs) for p, gs in m.items()}
                                                          for c, m in sorted(self.edge_graphs.items())}))
            f.write("}")

    @staticmethod
    def load(path: Path) -> SharedHierarchy:
        with gzip.open(path, "rt") as f:
            return SharedHierarchy.from_json(json.load(f))

    # -- building from a store ------------------------------------------------------------------

    @staticmethod
    def build(client, roles: GraphRoles) -> SharedHierarchy:
        """Read the hierarchy graphs' subclass edges (direct parents for closure
        graphs) and the data nodes' parts (b) and (c), with a SparqlClient. For
        stores whose hierarchy fits in a JSON response; see ``build_streamed``.
        A response cut by the client's row cap is refused, never used."""
        client = _Complete(client)
        parents: dict[str, set[str]] = {}
        edge_graphs = _EdgeGraphs(roles.hierarchy_graphs)
        for gi, g in enumerate(roles.hierarchy_graphs):
            q = _direct_edges_query(g) if g in roles.closure_graphs else _edges_query(g)
            for c, p in client.query(q).rows:
                if isinstance(c, IRI) and isinstance(p, IRI):
                    edge_graphs.add(c.value, p.value, gi, parents.get(c.value, ()))
                    parents.setdefault(c.value, set()).add(p.value)
        qs = _queries(roles)
        data_nodes = {r[0].value for q in ("data_nodes", "entity_class_data_nodes")
                      for r in client.query(qs[q]).rows if isinstance(r[0], IRI)}
        targets = {r[0].value for r in client.query(qs["targets"]).rows if isinstance(r[0], IRI)}
        membership: dict[str, set[str]] = {}
        for c, p in client.query(_membership_edges_query(roles)).rows:
            if isinstance(c, IRI) and isinstance(p, IRI):
                membership.setdefault(c.value, set()).add(p.value)
        excluded: list[tuple[str, str]] = []
        if roles.exclude_schema_subjects:
            for c, p in client.query(_membership_edges_query(roles, schema_only=True)).rows:
                if isinstance(c, IRI) and isinstance(p, IRI):
                    excluded.append((c.value, p.value))
                    membership.get(c.value, set()).discard(p.value)
        h = SharedHierarchy(roles, {c: tuple(sorted(ps)) for c, ps in parents.items()}, frozenset(data_nodes),
                            membership_parents={c: tuple(sorted(ps)) for c, ps in membership.items() if ps})
        # entries 49 and 61: the positions hierarchy nodes hold in each hierarchy graph, one position per query

        def fetch(q: str):
            return [tuple(x.value for x in row) for row in client.query(q).rows if all(isinstance(x, IRI) for x in row)]

        object.__setattr__(h, "positions", Occupancy.build(_occupancy_rows(fetch, roles, h.classes, parents=h.parents)))
        object.__setattr__(h, "edge_graphs", edge_graphs.record(h.parents, h.positions))
        object.__setattr__(h, "provenance", {
            "edges": sum(len(ps) for ps in h.parents.values()), "classes": len(h.classes),
            "targets": len(targets), "data_nodes": len(data_nodes),
            "membership_edges": sum(len(ps) for ps in h.membership_parents.values()),
            "excluded_schema_edges": sorted(excluded),
            "position_nodes": len(h.positions.nodes), "position_signatures": len(h.positions.signatures),
            "edge_graph_records": len(h.edge_graphs)})
        return h

    @staticmethod
    def build_streamed(
        endpoint: str,
        roles: GraphRoles,
        out_path: Path,
        log_path: Path,
        timeout_s: float = 280.0,
        sample: int = 1000,
        seed: int = 20271018,
    ) -> SharedHierarchy:
        """Build a large store's shared hierarchy one query at a time, streaming
        TSV, and save it gzipped. For each closure graph, load direct parents only
        and check on a seeded sample of ``sample`` nodes that the recomputed closure
        equals the materialised one; the check and its result go to ``log_path``."""
        if timeout_s > 290:
            raise ValueError("QLever caps a query at 300 s: keep the client time-out at or below 290 s")
        t0 = time.monotonic()
        log = open(log_path, "a")

        def note(msg: str) -> None:
            log.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}\n")
            log.flush()

        note(f"build {endpoint} roles={roles}")
        parents: dict[str, list[str]] = {}
        intern: dict[str, str] = {}
        internal: dict[str, set[str]] = {}  # closure graph -> its nodes
        edge_graphs = _EdgeGraphs(roles.hierarchy_graphs)
        for gi, g in enumerate(roles.hierarchy_graphs):
            q = _direct_edges_query(g) if g in roles.closure_graphs else _edges_query(g)
            n = 0
            nodes: set[str] = set()
            for row in _stream_tsv(endpoint, q, timeout_s):
                c, p = intern.setdefault(row[0], row[0]), intern.setdefault(row[1], row[1])
                edge_graphs.add(c, p, gi, parents.get(c, ()))
                parents.setdefault(c, []).append(p)
                nodes.update((c, p))
                n += 1
            note(f"graph {g}: {n} {'direct ' if g in roles.closure_graphs else ''}edges")
            if g in roles.closure_graphs:
                # edges to objects outside the graph: kept for the graph's top nodes only
                ext = list(_stream_tsv(endpoint, _external_edges_query(g), timeout_s))
                kept = [(c, p) for c, p in ext if not any(q in nodes for q in parents.get(c, ()))]
                for c, p in kept:
                    c, p = intern.setdefault(c, c), intern.setdefault(p, p)
                    edge_graphs.add(c, p, gi, parents.get(c, ()))
                    parents.setdefault(c, []).append(p)
                note(f"graph {g}: {len(ext)} edges to objects outside the graph ({sorted({p for _, p in ext})[:5]}), "
                     f"{len(kept)} kept as the top nodes' parents; the materialised closure of their descendants omits them")
                internal[g] = nodes | {c for c, _ in kept}
        closure_checks = {}
        for g in roles.closure_graphs:
            ok, checked, mismatches, missing = _check_closure(endpoint, g, parents, sample, seed, timeout_s, internal[g])
            multi = sum(1 for c, ps in parents.items() if c in internal[g] and len(set(ps)) > 1)
            note(f"closure check {g}: sample={checked} seed={seed} compared within the graph's own nodes; "
                 f"recomputed equals materialised on {checked - len(mismatches)}, differs on {len(mismatches)} "
                 f"(materialised closure not transitive there; the transitive completion is used, rule 35 ruling (a)) "
                 f"{mismatches[:10]}; recomputed misses a materialised ancestor on {len(missing)}; "
                 f"nodes with several direct parents: {multi}")
            closure_checks[g] = {"sample": checked, "seed": seed, "differs": len(mismatches),
                                 "misses_materialised": len(missing), "several_direct_parents": multi}
            if missing:
                raise RuntimeError(f"the recomputed closure misses materialised ancestors in {g}: {missing[:5]}")
        del internal  # the closure check is done
        qs = _queries(roles)
        data_nodes = {row[0] for q in ("data_nodes", "entity_class_data_nodes")
                      for row in _stream_tsv(endpoint, qs[q], timeout_s)}
        targets = sum(1 for _ in _stream_tsv(endpoint, qs["targets"], timeout_s))
        membership: dict[str, set[str]] = {}
        for c, p in _stream_tsv(endpoint, _membership_edges_query(roles), timeout_s):
            membership.setdefault(intern.setdefault(c, c), set()).add(intern.setdefault(p, p))
        excluded = []
        if roles.exclude_schema_subjects:
            for c, p in _stream_tsv(endpoint, _membership_edges_query(roles, schema_only=True), timeout_s):
                excluded.append((c, p))
                membership.get(c, set()).discard(p)
        note(f"data graph {roles.data_graph}: {sum(len(v) for v in membership.values())} membership edges, "
             f"{len(excluded)} schema edges of a compared source left out {excluded[:12]}")
        h = SharedHierarchy(roles, {c: tuple(sorted(set(ps))) for c, ps in parents.items()}, frozenset(data_nodes),
                            membership_parents={c: tuple(sorted(ps)) for c, ps in membership.items() if ps})
        # entries 49 and 61: the positions hierarchy nodes hold in each hierarchy graph, one position per query
        b = _OccupancyBuilder()
        for n, g, p, side in _occupancy_rows(lambda q: _stream_tsv(endpoint, q, timeout_s), roles, h.classes, note,
                                             parents=h.parents):
            b.add(intern.setdefault(n, n), (g, p, side))
        object.__setattr__(h, "positions", b.done())
        object.__setattr__(h, "edge_graphs", edge_graphs.record(h.parents, h.positions))
        del edge_graphs
        note(f"edge graphs: {len(h.edge_graphs)} nodes recorded explicitly (edges in several hierarchy graphs, or "
             "not placed by their positions)")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        prov = {"endpoint": endpoint, "edges": sum(len(ps) for ps in h.parents.values()), "classes": len(h.classes),
                "targets": targets, "data_nodes": len(data_nodes),
                "membership_edges": sum(len(ps) for ps in h.membership_parents.values()),
                "excluded_schema_edges": sorted(excluded), "closure_checks": closure_checks,
                "position_nodes": len(h.positions.nodes), "position_signatures": len(h.positions.signatures),
                "edge_graph_records": len(h.edge_graphs),
                "seconds": round(time.monotonic() - t0, 1), "max_rss_mb": round(rss / 1024)}
        object.__setattr__(h, "provenance", prov)
        h.save(out_path)
        digest = hashlib.sha256(Path(out_path).read_bytes()).hexdigest()
        note(f"done {prov} sha256={digest}")
        log.close()
        return h


class _Complete:
    """A client wrapper that refuses truncated responses. A response marked possibly partial arrived before the
    client's cut and is complete (change-log entry 57, design log), so it is used."""

    def __init__(self, client):
        self._client = client

    def query(self, q: str):
        r = self._client.query(q)
        if getattr(r, "truncated", False):
            raise RuntimeError("a response was cut by the client's row cap; the shared hierarchy would be incomplete")
        return r


def _stream_tsv(endpoint: str, query: str, timeout_s: float):
    """Rows of a SELECT, streamed as TSV (IRIs without angle brackets)."""
    with httpx.stream("POST", endpoint, content=query.encode(), params={"timeout": f"{timeout_s:g}s"},
                      headers={"Content-Type": "application/sparql-query", "Accept": "text/tab-separated-values"},
                      timeout=timeout_s + 10) as resp:
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.read()[:500]!r}")
        lines = resp.iter_lines()
        next(lines, None)  # header
        for line in lines:
            if line:
                yield tuple(cell[1:-1] if cell.startswith("<") else cell for cell in line.split("\t"))


def _check_closure(endpoint, graph, parents, sample, seed, timeout_s, internal):
    """Compare, on a seeded sample of the graph's nodes, the closure recomputed from
    direct parents with the closure materialised in ``graph``, both restricted to
    the graph's own nodes (``internal``)."""
    nodes = sorted(c for c in parents if c in internal)
    rng = random.Random(seed)
    picked = rng.sample(nodes, min(sample, len(nodes)))
    memo: dict[str, frozenset[str]] = {}

    def ancestors(n: str) -> frozenset[str]:
        if n in memo:
            return memo[n]
        out, stack = set(), list(parents.get(n, ()))
        while stack:
            x = stack.pop()
            if x not in out:
                out.add(x)
                stack.extend(parents.get(x, ()))
        memo[n] = frozenset(out)
        return memo[n]

    mismatches, missing = [], []
    for i in range(0, len(picked), 200):
        chunk = picked[i : i + 200]
        q = (f"SELECT ?c ?p WHERE {{ {_values('c', chunk)} GRAPH <{graph}> {{ ?c <{RDFS_SUBCLASSOF}> ?p }} "
             f"FILTER(isIRI(?p)) }}")
        mat: dict[str, set[str]] = {c: set() for c in chunk}
        for c, p in _stream_tsv(endpoint, q, timeout_s):
            mat.setdefault(c, set()).add(p)
        for c in chunk:
            rec, m = ancestors(c) & internal, frozenset(mat[c]) & internal
            if rec != m:
                mismatches.append(c)
            if not m <= rec:
                missing.append(c)
    return not mismatches, len(picked), mismatches, missing
