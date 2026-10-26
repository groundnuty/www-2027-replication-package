"""One reader of the frozen OWL fragment, for the curated and the generated sources alike.

It runs SPARQL, so the same queries read the curated ontology in the store
(Rhea's schema and ChEBI) and a generated Turtle ontology loaded into an
in-memory ``LocalSparqlClient``: the two are read on equal terms (proposal §7.3).

What it keeps (proposal §7.2, "the frozen fragment"): named classes and their
named superclasses; properties with their kind and super-properties; domain and
range as a named class or a union of named classes; datatype ranges;
disjointness between named classes (owl:disjointWith and owl:AllDisjointClasses);
existential restrictions ``C rdfs:subClassOf [owl:onProperty p; owl:someValuesFrom D]``
with named ``C``, ``p``, ``D``; labels and comments. Everything else in the OWL
namespace is counted in ``unsupported`` by construct and never translated.

What counts as a class: an IRI typed owl:Class or rdfs:Class; the subject or
named object of rdfs:subClassOf; a named domain, range, union member,
disjointness member or existential filler. A term with only a label or comment
is described, not a class (design ruling in the design log Oct 23:02, the kind of
change-log entry 64's defect): it is kept in ``described`` with its labels and
comments, which the labels layer renders. Classes marked
owl:deprecated are left out (counted in ``excluded``): a deprecated identifier
is stale, and the vocabulary check should be able to say so. IRIs in the RDF,
RDFS and OWL namespaces (owl:Thing) are never classes or parents.

With a ``StoreLayout``, triples whose subject is a data node are skipped, so in
the Rhea graph only the schema's own triples are read (the store also types
every compound and reaction rdfs:Class).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from harness.sources.model import ClassExpr, ClassInfo, Existential, PropertyInfo
from harness.sources.vocab import (
    COMMENT_PROPERTIES,
    LABEL_PROPERTIES,
    OWL,
    PREFIXES,
    RDF,
    RDFS,
    XSD,
    StoreLayout,
    is_meta,
    not_data_node,
    values,
)
from harness.terms import IRI, BNode

PROPERTY_TYPES = {
    RDF + "Property": "unknown",
    OWL + "ObjectProperty": "object",
    OWL + "DatatypeProperty": "datatype",
    OWL + "AnnotationProperty": "annotation",
}
_KIND_RANK = {"object": 3, "datatype": 3, "annotation": 2, "unknown": 1}
DATATYPE_NAMESPACES = (XSD,)
DATATYPE_IRIS = {RDFS + "Literal", RDF + "langString", RDF + "PlainLiteral", RDF + "XMLLiteral", RDF + "HTML", RDF + "JSON"}

# OWL predicates and types that are declarations, ontology headers or the
# reification of annotated axioms (counted once, as owl:Axiom), not constructs.
# owl:onProperty is part of every restriction, which is counted once by its
# other predicate (owl:allValuesFrom, owl:hasValue, a cardinality, ...).
_SKIP_PREDICATES = {
    OWL + p for p in (
        "onProperty", "annotatedSource", "annotatedProperty", "annotatedTarget", "imports", "versionIRI", "versionInfo",
        "priorVersion", "backwardCompatibleWith", "incompatibleWith", "deprecated",
    )
}
_SKIP_TYPES = {
    OWL + t for t in (
        "Class", "ObjectProperty", "DatatypeProperty", "AnnotationProperty", "Ontology", "Restriction",
        "AllDisjointClasses", "Thing",
    )
}


def is_datatype(iri: str) -> bool:
    return iri.startswith(DATATYPE_NAMESPACES) or iri in DATATYPE_IRIS


def qname(iri: str) -> str:
    for prefix, ns in (("owl", OWL), ("rdfs", RDFS), ("rdf", RDF)):
        if iri.startswith(ns):
            return f"{prefix}:{iri[len(ns):]}"
    return iri


@dataclass
class OwlFragment:
    classes: dict[str, ClassInfo] = field(default_factory=dict)
    properties: dict[str, PropertyInfo] = field(default_factory=dict)
    disjoint: list[tuple[str, str]] = field(default_factory=list)
    existentials: list[Existential] = field(default_factory=list)
    unsupported: dict[str, int] = field(default_factory=dict)
    excluded: dict[str, int] = field(default_factory=dict)
    # The classes typed owl:Class or rdfs:Class (rule 32 (a)'s declarations); ``classes`` also holds the IRIs that
    # only stand in a class position (a superclass, a domain or range, a disjointness or restriction member).
    declared: list[str] = field(default_factory=list)
    # Terms with a label or comment that are no class, property, ontology header, individual or restriction: their
    # labels and comments ({"labels": [...], "comments": [...]}), by IRI.
    described: dict[str, dict[str, list[str]]] = field(default_factory=dict)


def read_owl_fragment(client, graphs: Sequence[str], layout: StoreLayout | None = None) -> OwlFragment:
    g = values("g", graphs)

    def run(body: str, select: str = "*", tail: str = "") -> list[tuple]:
        return client.query(PREFIXES + f"SELECT {select} WHERE {{ {body} }} {tail}").rows

    def dn(var: str) -> str:
        return not_data_node(var, layout)

    deprecated = {
        r[0].value for r in run(f'{g} GRAPH ?g {{ ?c owl:deprecated ?d }} FILTER(isIRI(?c) && STR(?d) = "true")', "DISTINCT ?c")
    }
    typed_classes = {
        r[0].value for r in run(
            f"{g} GRAPH ?g {{ ?c rdf:type ?t }} VALUES ?t {{ owl:Class rdfs:Class }} FILTER isIRI(?c) {dn('c')}",
            "DISTINCT ?c",
        )
    }
    headers = {r[0].value for r in run(f"{g} GRAPH ?g {{ ?o rdf:type owl:Ontology }} FILTER isIRI(?o)", "DISTINCT ?o")}
    # Individuals: IRIs typed owl:NamedIndividual or as members of a class outside RDF, RDFS and OWL, and not declared
    # classes. A label or comment does not make one a class (design ruling in the design log: UniProt core's 44 named
    # individuals, up:API among them, were read as classes); only a declaration or an axiom using it as one does.
    individuals = {
        r[0].value for r in run(
            f"{g} GRAPH ?g {{ ?c rdf:type ?t }} FILTER(isIRI(?c) && isIRI(?t) && (?t = owl:NamedIndividual || "
            f"({_non_meta('t')}))) {dn('c')}",
            "DISTINCT ?c",
        )
    } - typed_classes

    # A restriction is read as a restriction whether it is a blank node or a named node (UniProt's GO
    # publishes its restrictions as IRIs such as GO_0000015#part-of-parent-5829); it is never a class.
    restriction_iris = {
        r[0].value for r in run(f"{g} GRAPH ?g {{ ?r owl:onProperty ?x }} FILTER isIRI(?r)", "DISTINCT ?r")
    }
    parents: dict[str, set[str]] = defaultdict(set)
    for c, p in run(f"{g} GRAPH ?g {{ ?c rdfs:subClassOf ?p }} FILTER(isIRI(?c) && isIRI(?p)) {dn('c')}", "DISTINCT ?c ?p"):
        if p.value not in restriction_iris and c.value not in restriction_iris:
            parents[c.value].add(p.value)

    existential_rows = run(
        f"{g} GRAPH ?g {{ ?c rdfs:subClassOf ?r . ?r owl:onProperty ?prop . ?r owl:someValuesFrom ?f }} "
        f"FILTER(isIRI(?c) && isIRI(?prop) && isIRI(?f)) {dn('c')}",
        "DISTINCT ?c ?r ?prop ?f",
    )
    restrictions_used = {_node_key(r[1]) for r in existential_rows}

    # Properties.
    kinds: dict[str, str] = {}
    for p, t in run(
        f"{g} GRAPH ?g {{ ?p rdf:type ?t }} {values('t', PROPERTY_TYPES)} FILTER isIRI(?p) {dn('p')}", "DISTINCT ?p ?t"
    ):
        k = PROPERTY_TYPES[t.value]
        cur = kinds.get(p.value)
        # The most specific declared kind wins; a tie between object and datatype resolves to the
        # alphabetically first, so the result does not depend on row order.
        if cur is None or _KIND_RANK[k] > _KIND_RANK[cur] or (_KIND_RANK[k] == _KIND_RANK[cur] and k < cur):
            kinds[p.value] = k
    super_props: dict[str, set[str]] = defaultdict(set)
    for p, q in run(f"{g} GRAPH ?g {{ ?p rdfs:subPropertyOf ?q }} FILTER(isIRI(?p) && isIRI(?q)) {dn('p')}", "DISTINCT ?p ?q"):
        super_props[p.value].add(q.value)

    named_dr: dict[tuple[str, str], set[str]] = defaultdict(set)  # (prop, "domain"|"range") -> named classes/datatypes
    for p, k, d in run(
        f"{g} GRAPH ?g {{ ?p ?k ?d }} VALUES ?k {{ rdfs:domain rdfs:range }} FILTER(isIRI(?p) && isIRI(?d)) {dn('p')}",
        "DISTINCT ?p ?k ?d",
    ):
        named_dr[(p.value, k.value.rsplit("#", 1)[1])].add(d.value)
    union_members: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    union_bad: set[tuple[str, str, str]] = set()
    for p, k, u, m in run(
        f"{g} GRAPH ?g {{ ?p ?k ?u . ?u owl:unionOf ?l . ?l rdf:rest*/rdf:first ?m }} "
        f"VALUES ?k {{ rdfs:domain rdfs:range }} FILTER(isIRI(?p) && isBlank(?u)) {dn('p')}",
        "DISTINCT ?p ?k ?u ?m",
    ):
        key = (p.value, k.value.rsplit("#", 1)[1], u.label)
        if isinstance(m, IRI) and not is_meta(m.value):
            union_members[key].add(m.value)
        else:
            union_bad.add(key)
    # Rule 13 (check contract v2): an intersection of named classes and unions is split into one
    # declaration per conjunct; one that involves a restriction stays unsupported.
    inter: dict[tuple[str, str, str], dict[str, set[str] | None]] = defaultdict(dict)
    inter_bad: set[tuple[str, str, str]] = set()
    for p, k, u, m, mm in run(
        f"{g} GRAPH ?g {{ ?p ?k ?u . ?u owl:intersectionOf ?l . ?l rdf:rest*/rdf:first ?m "
        f"OPTIONAL {{ ?m owl:unionOf ?l2 . ?l2 rdf:rest*/rdf:first ?mm }} }} "
        f"VALUES ?k {{ rdfs:domain rdfs:range }} FILTER(isIRI(?p) && isBlank(?u)) {dn('p')}",
        "DISTINCT ?p ?k ?u ?m ?mm",
    ):
        key = (p.value, k.value.rsplit("#", 1)[1], u.label)
        if isinstance(m, IRI) and not is_meta(m.value):
            inter[key][m.value] = None
        elif isinstance(m, BNode) and isinstance(mm, IRI) and not is_meta(mm.value):
            got = inter[key].setdefault("_:" + m.label, set())
            got.add(mm.value)
        else:
            inter_bad.add(key)
    # A mixed intersection keeps its supported conjuncts and marks its side unsupported (rule 13; the
    # compiler reads PropertyInfo.unsupported_domain / unsupported_range).
    inter_ok = {k: v for k, v in inter.items() if v}
    unsupported_sides: set[tuple[str, str]] = {(k[0], k[1]) for k in inter_bad}
    # Rule 14: an enumerated data range (owl:oneOf over literals) as range_values.
    one_of: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    one_of_keys: set[tuple[str, str, str]] = set()
    for p, u, lit in run(
        f"{g} GRAPH ?g {{ ?p rdfs:range ?u . ?u owl:oneOf ?l . ?l rdf:rest*/rdf:first ?lit }} "
        f"FILTER(isIRI(?p) && isBlank(?u) && isLiteral(?lit)) {dn('p')}",
        "DISTINCT ?p ?u ?lit",
    ):
        one_of_keys.add((p.value, "range", u.label))
        one_of[(p.value, "range")].append({"lexical": lit.lexical, "datatype": lit.datatype or XSD + "string"})
    complex_dr: dict[str, int] = defaultdict(int)
    for p, k, u in run(
        f"{g} GRAPH ?g {{ ?p ?k ?u }} VALUES ?k {{ rdfs:domain rdfs:range }} FILTER(isIRI(?p) && isBlank(?u)) {dn('p')}",
        "DISTINCT ?p ?k ?u",
    ):
        key = (p.value, k.value.rsplit("#", 1)[1], u.label)
        if key in one_of_keys:
            continue
        if (key not in union_members or key in union_bad) and key not in inter:
            complex_dr[f"rdfs:{key[1]} (class expression)"] += 1
            unsupported_sides.add((key[0], key[1]))
        elif key in union_bad:
            unsupported_sides.add((key[0], key[1]))

    disjoint_pairs: set[tuple[str, str]] = set()
    n_disjoint_with = 0
    for a, b in run(f"{g} GRAPH ?g {{ ?a owl:disjointWith ?b }} FILTER(isIRI(?a) && isIRI(?b)) {dn('a')}", "DISTINCT ?a ?b"):
        n_disjoint_with += 1
        if a.value != b.value:
            disjoint_pairs.add(tuple(sorted((a.value, b.value))))
    all_disjoint: dict[str, set[str]] = defaultdict(set)
    bad_all_disjoint: set[str] = set()
    for x, m in run(
        f"{g} GRAPH ?g {{ ?x rdf:type owl:AllDisjointClasses . ?x owl:members ?l . ?l rdf:rest*/rdf:first ?m }}",
        "DISTINCT ?x ?m",
    ):
        key = x.label if isinstance(x, BNode) else x.value
        if isinstance(m, IRI):
            all_disjoint[key].add(m.value)
        else:
            bad_all_disjoint.add(key)
    for key, members in all_disjoint.items():
        if key in bad_all_disjoint:
            continue
        ms = sorted(members)
        for i, a in enumerate(ms):
            for b in ms[i + 1:]:
                disjoint_pairs.add((a, b))

    labels: dict[str, set[str]] = defaultdict(set)
    for x, lit in run(
        f"{g} GRAPH ?g {{ ?x ?lp ?l }} {values('lp', LABEL_PROPERTIES)} FILTER(isIRI(?x) && isLiteral(?l)) {dn('x')}",
        "DISTINCT ?x ?l",
    ):
        labels[x.value].add(lit.lexical)
    comments: dict[str, set[str]] = defaultdict(set)
    for x, lit in run(
        f"{g} GRAPH ?g {{ ?x ?cp ?l }} {values('cp', COMMENT_PROPERTIES)} FILTER(isIRI(?x) && isLiteral(?l)) {dn('x')}",
        "DISTINCT ?x ?l",
    ):
        comments[x.value].add(lit.lexical)

    # Which IRIs are properties and which are classes.
    prop_iris = set(kinds) | set(super_props) | {q for qs in super_props.values() for q in qs} | {p for p, _ in named_dr}
    prop_iris |= {p for p, _, _ in union_members} | {p for p, _, _ in inter_ok}
    prop_iris = {p for p in prop_iris if not is_meta(p)}

    class_iris = set(typed_classes) | set(parents) | {p for ps in parents.values() for p in ps}
    for ds in named_dr.values():
        class_iris |= {d for d in ds if not is_datatype(d)}
    for key, ms in union_members.items():
        if key not in union_bad:
            class_iris |= {m for m in ms if not is_datatype(m)}
    for conj in inter_ok.values():
        for name, members in conj.items():
            class_iris |= {m for m in (members or {name}) if not is_datatype(m)}
    for a, b in disjoint_pairs:
        class_iris |= {a, b}
    for c, _, _, f in existential_rows:
        class_iris |= {c.value, f.value}
    class_iris = {c for c in class_iris if not is_meta(c) and c not in headers and c not in restriction_iris}
    n_deprecated = len(class_iris & deprecated)
    class_iris -= deprecated
    # A label or comment alone makes no class (design ruling in the design log Oct 23:02); the term is described.
    described = sorted((set(labels) | set(comments)) - prop_iris - headers - individuals - class_iris - restriction_iris
                       - deprecated)

    frag = OwlFragment()
    frag.declared = sorted(set(typed_classes) & class_iris)
    frag.described = {d: {"labels": sorted(labels.get(d, ())), "comments": sorted(comments.get(d, ()))}
                      for d in described if not is_meta(d)}
    for c in sorted(class_iris):
        frag.classes[c] = ClassInfo(
            c,
            parents=sorted(p for p in parents.get(c, ()) if p in class_iris and p != c),
            labels=sorted(labels.get(c, ())),
            comments=sorted(comments.get(c, ())),
        )

    unsupported: dict[str, int] = defaultdict(int)
    for p in sorted(prop_iris):
        info = PropertyInfo(p, kind=kinds.get(p, "unknown"), parents=sorted(q for q in super_props.get(p, ()) if q != p))
        for side in ("domain", "range"):
            named = sorted(named_dr.get((p, side), ()))
            for d in named:
                if side == "range" and is_datatype(d):
                    info.datatype_range.append(d)
                elif d not in deprecated:
                    getattr(info, side).append(ClassExpr.named(d))
            for key in sorted(k for k in union_members if k[0] == p and k[1] == side and k not in union_bad):
                ms = sorted(union_members[key])
                if all(is_datatype(m) for m in ms):
                    unsupported[f"rdfs:{side} (union of datatypes)"] += 1
                    unsupported_sides.add((p, side))
                elif any(is_datatype(m) for m in ms):
                    unsupported[f"rdfs:{side} (class expression)"] += 1
                    unsupported_sides.add((p, side))
                else:
                    getattr(info, side).append(ClassExpr(tuple(ms)))
            for key in sorted(k for k in inter_ok if k[0] == p and k[1] == side):
                conjuncts = sorted((sorted(members) if members else [name]) for name, members in inter_ok[key].items())
                for ms in sorted(conjuncts, key=lambda m: (len(m) > 1, m)):
                    if side == "range" and all(is_datatype(m) for m in ms):
                        info.datatype_range.extend(m for m in ms if m not in info.datatype_range)
                    else:
                        getattr(info, side).append(ClassExpr(tuple(ms)))
        info.unsupported_domain = (p, "domain") in unsupported_sides
        info.unsupported_range = (p, "range") in unsupported_sides
        info.range_values = sorted(one_of.get((p, "range"), []), key=lambda v: (v["datatype"], v["lexical"]))
        info.labels = sorted(labels.get(p, ()))
        info.comments = sorted(comments.get(p, ()))
        frag.properties[p] = info

    frag.disjoint = sorted(p for p in disjoint_pairs if p[0] in class_iris and p[1] in class_iris)
    frag.existentials = sorted(
        {Existential(c.value, prop.value, f.value) for c, _, prop, f in existential_rows if c.value in class_iris},
        key=lambda e: (e.cls, e.prop, e.filler),
    )

    # Unsupported constructs: every OWL-namespace predicate and type not read above.
    n_unions_used = sum(1 for k in union_members if k not in union_bad) + sum(
        1 for conj in inter_ok.values() for members in conj.values() if members)
    handled = {
        OWL + "someValuesFrom": len(restrictions_used),
        OWL + "unionOf": n_unions_used,
        OWL + "disjointWith": n_disjoint_with,
        OWL + "members": len(all_disjoint) - len(bad_all_disjoint & set(all_disjoint)),
        OWL + "intersectionOf": len(inter_ok) - len(set(inter_ok) & inter_bad),
        OWL + "oneOf": len(one_of_keys),
    }
    for p, n in run(
        f"{g} GRAPH ?g {{ ?s ?p ?o }} FILTER STRSTARTS(STR(?p), \"{OWL}\") {dn('s')}",
        "?p (COUNT(*) AS ?n)",
        "GROUP BY ?p",
    ):
        if p is None or p.value in _SKIP_PREDICATES:  # rdflib returns one unbound row for an empty group
            continue
        rest = int(n.lexical) - handled.get(p.value, 0)
        if rest > 0:
            unsupported[qname(p.value)] += rest
    for t, n in run(
        f"{g} GRAPH ?g {{ ?s rdf:type ?t }} FILTER STRSTARTS(STR(?t), \"{OWL}\") {dn('s')}",
        "?t (COUNT(DISTINCT ?s) AS ?n)",
        "GROUP BY ?t",
    ):
        if t is None or t.value in _SKIP_TYPES:
            continue
        unsupported[qname(t.value)] += int(n.lexical)
    for k, n in complex_dr.items():
        unsupported[k] += n
    frag.unsupported = dict(sorted(unsupported.items()))
    frag.excluded = {"owl:deprecated classes": n_deprecated} if n_deprecated else {}
    return frag


def read_vocabulary_restricted(client, graph: str, data_graph: str, chunk: int = 2000) -> tuple[OwlFragment, dict]:
    """A shipped vocabulary restricted to the terms the data uses plus their ancestors (rule 16).

    Used terms: IRI objects of data-graph triples that are typed in the vocabulary graph and that the graph declares a
    class (owl:Class, rdfs:Class) or uses as one (the subject or object of an rdfs:subClassOf edge to an IRI): an IRI
    the graph types only as a member of one of its classes is an instance, never a class (design ruling in the design log:
    UniProt's enzymes graph types the 6,946 citations and 634 catalytic activities the data cites). Ancestors:
    ``rdfs:subClassOf*`` inside the vocabulary graph. For those terms the same fragment is read as by
    ``read_owl_fragment`` (named superclasses, labels, comments, existentials), as published; other OWL
    constructs on them are counted in ``unsupported``. ``declared``: the terms typed owl:Class or rdfs:Class.
    """
    used_q = PREFIXES + (
        f"SELECT DISTINCT ?c WHERE {{ GRAPH <{data_graph}> {{ ?s ?p ?c }} FILTER isIRI(?c) GRAPH <{graph}> {{ ?c a ?t }} "
        f"{_not_deprecated_or_restriction('c', graph)} }}"
    )
    typed_used = sorted({r[0].value for r in client.query(used_q).rows if isinstance(r[0], IRI)})

    def chunks(items):
        items = sorted(items)
        for i in range(0, len(items), chunk):
            yield items[i:i + chunk]

    def class_like(terms) -> tuple[set[str], set[str]]:
        """(declared, declared or in a subclass edge) among ``terms``, in the vocabulary graph."""
        declared, like = set(), set()
        for part in chunks(terms):
            v = values("c", part)
            declared |= {r[0].value for r in client.query(PREFIXES + (
                f"SELECT DISTINCT ?c WHERE {{ {v} GRAPH <{graph}> {{ ?c a ?t }} VALUES ?t {{ owl:Class rdfs:Class }} }}")).rows}
            like |= {r[0].value for r in client.query(PREFIXES + (
                f"SELECT DISTINCT ?c WHERE {{ {v} GRAPH <{graph}> {{ {{ ?c rdfs:subClassOf ?x FILTER isIRI(?x) }} UNION "
                f"{{ ?y rdfs:subClassOf ?c FILTER isIRI(?y) }} }} }}")).rows}
        return declared, declared | like

    _, like = class_like(typed_used)
    used = [c for c in typed_used if c in like]

    closure: set[str] = set(used)
    for part in chunks(used):
        q = PREFIXES + (
            f"SELECT DISTINCT ?a WHERE {{ {values('c', part)} GRAPH <{graph}> {{ ?c rdfs:subClassOf* ?a }} "
            f"FILTER(isIRI(?a) && {_non_meta('a')}) {_not_deprecated_or_restriction('a', graph)} }}"
        )
        closure |= {r[0].value for r in client.query(q).rows if isinstance(r[0], IRI)}

    parents: dict[str, set[str]] = defaultdict(set)
    labels: dict[str, set[str]] = defaultdict(set)
    comments: dict[str, set[str]] = defaultdict(set)
    existential_rows: set[tuple[str, str, str]] = set()
    restrictions: set[str] = set()
    owl_counts: dict[str, int] = defaultdict(int)
    for part in chunks(closure):
        v = values("c", part)
        for c, p in client.query(PREFIXES + (
            f"SELECT ?c ?p WHERE {{ {v} GRAPH <{graph}> {{ ?c rdfs:subClassOf ?p }} FILTER isIRI(?p) }}")).rows:
            parents[c.value].add(p.value)
        for c, lit in client.query(PREFIXES + (
            f"SELECT ?c ?l WHERE {{ {v} GRAPH <{graph}> {{ ?c ?lp ?l }} {values('lp', LABEL_PROPERTIES)} FILTER isLiteral(?l) }}")).rows:
            labels[c.value].add(lit.lexical)
        for c, lit in client.query(PREFIXES + (
            f"SELECT ?c ?l WHERE {{ {v} GRAPH <{graph}> {{ ?c ?cp ?l }} {values('cp', COMMENT_PROPERTIES)} FILTER isLiteral(?l) }}")).rows:
            comments[c.value].add(lit.lexical)
        for c, r, prop, f in client.query(PREFIXES + (
            f"SELECT ?c ?r ?prop ?f WHERE {{ {v} GRAPH <{graph}> {{ ?c rdfs:subClassOf ?r . ?r owl:onProperty ?prop ; "
            "owl:someValuesFrom ?f } FILTER(isIRI(?prop) && isIRI(?f)) }")).rows:
            existential_rows.add((c.value, prop.value, f.value))
            restrictions.add(_node_key(r))
        for p, n in client.query(PREFIXES + (
            f"SELECT ?p (COUNT(*) AS ?n) WHERE {{ {v} GRAPH <{graph}> {{ {{ ?c ?p ?o }} UNION "
            f"{{ ?c rdfs:subClassOf ?r . ?r owl:onProperty ?x . ?r ?p ?o }} }} FILTER STRSTARTS(STR(?p), \"{OWL}\") }} GROUP BY ?p")).rows:
            if p is not None:
                owl_counts[p.value] += int(n.lexical)

    frag = OwlFragment()
    for c in sorted(closure):
        frag.classes[c] = ClassInfo(
            c, parents=sorted(p for p in parents.get(c, ()) if p in closure and p != c),
            labels=sorted(labels.get(c, ())), comments=sorted(comments.get(c, ())),
        )
    frag.declared = sorted(class_like(closure)[0])
    frag.existentials = sorted({Existential(*e) for e in existential_rows if e[0] in closure},
                               key=lambda e: (e.cls, e.prop, e.filler))
    handled = {OWL + "someValuesFrom": len(restrictions)}
    unsupported: dict[str, int] = defaultdict(int)
    for p, n in owl_counts.items():
        if p in _SKIP_PREDICATES:
            continue
        rest = n - handled.get(p, 0)
        if rest > 0:
            unsupported[qname(p)] += rest
    frag.unsupported = dict(sorted(unsupported.items()))
    info = {"graph": graph, "used_terms": len(used), "with_ancestors": len(closure),
            "typed_instances_left_out": len(typed_used) - len(used),
            "rule": "IRI objects of data-graph triples typed in this graph that it declares a class or uses as one (in an "
                    "rdfs:subClassOf edge), plus their rdfs:subClassOf ancestors in it; instances are left out"}
    return frag, info


def _non_meta(var: str) -> str:
    from harness.sources.vocab import non_meta

    return non_meta(var)


def _node_key(cell) -> str:
    return "_:" + cell.label if isinstance(cell, BNode) else cell.value


def _not_deprecated_or_restriction(var: str, graph: str) -> str:
    """FILTERs dropping deprecated classes and named restrictions of a vocabulary graph (Gemini review #16)."""
    return (f"FILTER NOT EXISTS {{ GRAPH <{graph}> {{ ?{var} owl:onProperty ?rx_{var} }} }} "
            f"FILTER NOT EXISTS {{ GRAPH <{graph}> {{ ?{var} owl:deprecated ?dep_{var} }} "
            f'FILTER(STR(?dep_{var}) = "true" || STR(?dep_{var}) = "1") }}')
