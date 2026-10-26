"""Profile of an OWL ontology's axioms (UniProt ``core.owl``) and of how the data uses them.

``profile_ontology`` counts what the checker can use: named classes, properties
by kind, rdfs:domain and rdfs:range split into named classes and anonymous
class expressions, disjointness, restrictions, subclass and subproperty axioms
and property characteristics. Counts are of RDF triples (rdflib). For RDF/XML
input, ``raw_xml_counts`` also counts the XML elements, because a reader who
greps the file counts elements, and the two differ where the file repeats a
statement.

``predicate_usage`` (needs the store) counts triples per graph and predicate,
and ``data_usage`` joins that with the profile: for every predicate the data
uses, whether core.owl declares it and whether it has a declared domain and
range.
"""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from harness.sib import CORE_GRAPH, LOADED_GRAPHS
from harness.sparql import SparqlClient, SparqlError

RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS_NS = "http://www.w3.org/2000/01/rdf-schema#"
OWL_NS = "http://www.w3.org/2002/07/owl#"
XSD_NS = "http://www.w3.org/2001/XMLSchema#"
UP_NS = "http://purl.uniprot.org/core/"

PREFIXES = {
    "up": UP_NS,
    "rdf": RDF_NS,
    "rdfs": RDFS_NS,
    "owl": OWL_NS,
    "xsd": XSD_NS,
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "foaf": "http://xmlns.com/foaf/0.1/",
    "faldo": "http://biohackathon.org/resource/faldo#",
    "dcterms": "http://purl.org/dc/terms/",
    "bibo": "http://purl.org/ontology/bibo/",
    "rh": "http://rdf.rhea-db.org/",
    "obo": "http://purl.obolibrary.org/obo/",
    "schema": "http://schema.org/",
    "void": "http://rdfs.org/ns/void#",
    "dc": "http://purl.org/dc/elements/1.1/",
    # Text2SPARQL 2026 corporate graph (E6.0)
    "pv": "http://ld.company.org/prod-vocab/",
    "prodi": "http://ld.company.org/prod-instances/",
    "dbo": "http://dbpedia.org/ontology/",
    "wgs84": "http://www.w3.org/2003/01/geo/wgs84_pos#",
    "vann": "http://purl.org/vocab/vann/",
    "cc": "http://creativecommons.org/ns#",
    "ecc": "https://ns.eccenca.com/",
}

PROPERTY_KINDS = {
    "owl:ObjectProperty": OWL_NS + "ObjectProperty",
    "owl:DatatypeProperty": OWL_NS + "DatatypeProperty",
    "owl:AnnotationProperty": OWL_NS + "AnnotationProperty",
    "rdf:Property": RDF_NS + "Property",
}
CHARACTERISTICS = (
    "FunctionalProperty",
    "InverseFunctionalProperty",
    "TransitiveProperty",
    "SymmetricProperty",
    "AsymmetricProperty",
    "ReflexiveProperty",
    "IrreflexiveProperty",
)
RESTRICTION_KINDS = (
    "someValuesFrom",
    "allValuesFrom",
    "hasValue",
    "hasSelf",
    "cardinality",
    "minCardinality",
    "maxCardinality",
    "qualifiedCardinality",
    "minQualifiedCardinality",
    "maxQualifiedCardinality",
)
EXPRESSION_KINDS = ("unionOf", "intersectionOf", "complementOf", "oneOf", "restriction", "other")
LITERAL_TYPES = {RDFS_NS + "Literal", RDF_NS + "langString", RDF_NS + "PlainLiteral", RDFS_NS + "Datatype", RDF_NS + "XMLLiteral"}


def curie(iri: str) -> str:
    for p, ns in PREFIXES.items():
        if iri.startswith(ns) and len(iri) > len(ns):
            return f"{p}:{iri[len(ns):]}"
    return f"<{iri}>"


def _is_datatype(iri: str) -> bool:
    return iri.startswith(XSD_NS) or iri in LITERAL_TYPES


def load_graph(path: str | Path):
    from rdflib import Graph

    path = Path(path)
    fmt = "xml" if path.suffix in (".owl", ".rdf", ".xml") else None
    g = Graph()
    g.parse(str(path), format=fmt)
    return g


# ---------------------------------------------------------------------------
# Class expressions


def describe_class(g, node, depth: int = 0) -> dict[str, Any]:
    """A JSON description of a class: a named IRI or an anonymous expression."""
    from rdflib import RDF, BNode, Namespace, URIRef
    from rdflib.collection import Collection

    owl = Namespace(OWL_NS)
    if isinstance(node, URIRef):
        return {"named": str(node)}
    if not isinstance(node, BNode) or depth > 8:
        return {"kind": "other", "value": str(node)}
    for kind in ("unionOf", "intersectionOf", "oneOf"):
        lst = g.value(node, owl[kind])
        if lst is not None:
            members = list(Collection(g, lst))
            if kind == "oneOf":  # individuals (IRIs) or, as a data range, literals (N3 form)
                out = {"kind": kind, "members": [str(m) if isinstance(m, URIRef) else m.n3(g.namespace_manager) for m in members]}
                if any(not isinstance(m, URIRef) for m in members):
                    out["data_range"] = True
                return out
            return {"kind": kind, "members": [describe_class(g, m, depth + 1) for m in members]}
    comp = g.value(node, owl.complementOf)
    if comp is not None:
        return {"kind": "complementOf", "of": describe_class(g, comp, depth + 1)}
    if (node, RDF.type, owl.Restriction) in g or (node, owl.onProperty, None) in g:
        return describe_restriction(g, node, depth)
    from rdflib import RDFS

    on_dt = g.value(node, owl.onDatatype)
    if on_dt is not None or (node, RDF.type, RDFS.Datatype) in g:
        return {"kind": "datatype_restriction", "on_datatype": str(on_dt) if on_dt is not None else None, "data_range": True}

    return {
        "kind": "other",
        "predicates": sorted(curie(str(p)) for p in set(g.predicates(node)) if p != RDF.type),
        "sub_class_of": [describe_class(g, o, depth + 1) for o in g.objects(node, RDFS.subClassOf)],
    }


def describe_restriction(g, node, depth: int = 0) -> dict[str, Any]:
    from rdflib import Namespace

    owl = Namespace(OWL_NS)
    out: dict[str, Any] = {"kind": "restriction", "on_property": str(g.value(node, owl.onProperty))}
    constraints = {}
    for k in RESTRICTION_KINDS:
        v = g.value(node, owl[k])
        if v is not None:
            constraints[k] = describe_class(g, v, depth + 1) if k in ("someValuesFrom", "allValuesFrom") else str(v)
    on_class = g.value(node, owl.onClass)
    if on_class is not None:
        out["on_class"] = describe_class(g, on_class, depth + 1)
    out["constraints"] = constraints
    return out


def _kind(desc: dict[str, Any]) -> str:
    return "named" if "named" in desc else desc["kind"]


def category(desc: dict[str, Any]) -> str:
    """named_class, datatype (XSD or rdfs:Literal), class_expression or data_range."""
    if "named" in desc:
        return "datatype" if _is_datatype(desc["named"]) else "named_class"
    return "data_range" if desc.get("data_range") else "class_expression"


def _union_members(desc: dict[str, Any]) -> list[dict]:
    return desc.get("members", []) if desc.get("kind") == "unionOf" else []


def _short(desc: dict[str, Any]) -> str:
    """A compact Manchester-like rendering of a class description."""
    if "named" in desc:
        return curie(desc["named"])
    k = desc["kind"]
    if k in ("unionOf", "intersectionOf"):
        sep = " or " if k == "unionOf" else " and "
        return "(" + sep.join(_short(m) for m in desc["members"]) + ")"
    if k == "oneOf":
        return "{" + ", ".join(curie(m) if m.startswith(("http://", "https://")) else m for m in desc["members"]) + "}"
    if k == "complementOf":
        return "not " + _short(desc["of"])
    if k == "datatype_restriction":
        return f"(data range on {curie(desc['on_datatype']) if desc['on_datatype'] else 'rdfs:Datatype'})"
    if k == "restriction":
        parts = []
        for c, v in desc["constraints"].items():
            parts.append(f"{c} {_short(v) if isinstance(v, dict) else v}")
        if "on_class" in desc:
            parts.append(f"onClass {_short(desc['on_class'])}")
        return f"[{curie(desc['on_property'])} {'; '.join(parts) or 'no constraint'}]"
    subs = desc.get("sub_class_of") or []
    if subs:
        return "(anonymous class subClassOf " + " and ".join(_short(s) for s in subs) + ")"
    return "(anonymous class)"


# ---------------------------------------------------------------------------
# Profile


def _axis(g, prop, declared_props: set[str], props_detail: dict[str, dict], key: str) -> dict[str, Any]:
    triples = list(g.subject_objects(prop))
    by_kind = Counter()
    by_category = Counter()
    union_sizes = Counter()
    unions_all_named = 0
    per_subject = defaultdict(list)
    for s, o in triples:
        desc = describe_class(g, o)
        per_subject[str(s)].append(desc)
        props_detail.setdefault(str(s), {"property": str(s)}).setdefault(key, []).append(desc)
        by_category[category(desc)] += 1
        if category(desc) == "class_expression":
            by_kind[_kind(desc)] += 1
        members = _union_members(desc)
        if members:
            union_sizes[len(members)] += 1
            unions_all_named += all("named" in m for m in members)
    anonymous = {k: by_kind.get(k, 0) for k in EXPRESSION_KINDS}
    return {
        "triples": len(triples),
        "properties": len(per_subject),
        "named_class": by_category["named_class"],
        "datatype": by_category["datatype"],
        "class_expression": by_category["class_expression"],
        "class_expression_by_kind": anonymous,
        "data_range": by_category["data_range"],
        "union_members": {
            "unions": sum(union_sizes.values()),
            "members_total": sum(k * v for k, v in union_sizes.items()),
            "size_distribution": {str(k): union_sizes[k] for k in sorted(union_sizes)},
            "unions_of_named_classes_only": unions_all_named,
        },
        "properties_with_multiple": [
            {"property": p, "values": [_short(d) for d in ds]} for p, ds in sorted(per_subject.items()) if len(ds) > 1
        ],
        "on_undeclared_properties": sorted(p for p in per_subject if p not in declared_props),
    }


def profile_ontology(g, source: str | Path | None = None) -> dict[str, Any]:
    from rdflib import RDF, RDFS, BNode, Literal, Namespace, URIRef

    owl = Namespace(OWL_NS)
    out: dict[str, Any] = {}
    onto = next(iter(g.subjects(RDF.type, owl.Ontology)), None)
    out["source"] = {
        "file": Path(source).name if source else None,
        "sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest() if source else None,
        "ontology": str(onto) if onto is not None else None,
        "version_info": str(g.value(onto, owl.versionInfo)) if onto is not None else None,
        "imports": sorted(str(o) for o in g.objects(onto, owl.imports)) if onto is not None else [],
        "triples": len(g),
    }

    def deprecated(s) -> bool:
        return any(str(v).lower() == "true" for v in g.objects(s, owl.deprecated) if isinstance(v, Literal))

    named_classes = {s for t in (owl.Class, RDFS.Class) for s in g.subjects(RDF.type, t) if isinstance(s, URIRef)}
    anon_class_nodes = {s for s in g.subjects(RDF.type, owl.Class) if isinstance(s, BNode)}

    kinds_of: dict[str, list[str]] = defaultdict(list)
    for label, iri in PROPERTY_KINDS.items():
        for s in g.subjects(RDF.type, URIRef(iri)):
            if isinstance(s, URIRef):
                kinds_of[str(s)].append(label)
    declared_props = set(kinds_of)
    characteristics = {c: sorted(str(s) for s in g.subjects(RDF.type, owl[c])) for c in CHARACTERISTICS}

    props_detail: dict[str, dict] = {}
    domain = _axis(g, RDFS.domain, declared_props, props_detail, "domains")
    range_ = _axis(g, RDFS.range, declared_props, props_detail, "ranges")

    # Disjointness
    from rdflib.collection import Collection

    dw_nodes = list(g.subject_objects(owl.disjointWith))
    dw = [(str(a), str(b)) for a, b in dw_nodes]
    adc_members = []
    for x in g.subjects(RDF.type, owl.AllDisjointClasses):
        lst = g.value(x, owl.members)
        adc_members.append(len(list(Collection(g, lst))) if lst is not None else 0)
    du = []
    for s, lst in g.subject_objects(owl.disjointUnionOf):
        du.append({"class": str(s), "members": [str(m) for m in Collection(g, lst)]})
    out_disjoint = {
        "declarations": len(dw) + len(adc_members) + len(du),
        "disjoint_with_pairs": len(dw),
        "disjoint_with": [[a, b] for a, b in sorted(dw)],
        "disjoint_with_involving_anonymous": sum(isinstance(a, BNode) or isinstance(b, BNode) for a, b in dw_nodes),
        "all_disjoint_classes": len(adc_members),
        "all_disjoint_classes_sizes": adc_members,
        "disjoint_union_of": du,
    }

    # Restrictions: every node with owl:onProperty, typed or not
    restriction_nodes = set(g.subjects(owl.onProperty, None)) | set(g.subjects(RDF.type, owl.Restriction))
    r_kinds = Counter()
    on_prop = Counter()
    on_class_no_card = 0
    used_in = Counter()
    for r in restriction_nodes:
        present = [k for k in RESTRICTION_KINDS if (r, owl[k], None) in g]
        for k in present:
            r_kinds[k] += 1
        if not present:
            r_kinds["none"] += 1
        if (r, owl.onClass, None) in g and not any(k.lower().endswith("qualifiedcardinality") for k in present):
            on_class_no_card += 1
        for p in g.objects(r, owl.onProperty):
            on_prop[str(p)] += 1
        for s, p in g.subject_predicates(r):
            if p == RDFS.subClassOf:
                used_in["superclass_of_named" if isinstance(s, URIRef) else "superclass_of_anonymous"] += 1
            elif p == owl.equivalentClass:
                used_in["equivalent_class"] += 1
            elif p in (RDFS.domain, RDFS.range):
                used_in["domain_or_range"] += 1
            elif p == RDF.first:
                used_in["member_of_class_expression"] += 1
            elif p in (owl.someValuesFrom, owl.allValuesFrom):
                used_in["nested_in_restriction"] += 1
            else:
                used_in[curie(str(p))] += 1
    out_restr = {
        "total": len(restriction_nodes),
        "typed_owl_restriction": len(set(g.subjects(RDF.type, owl.Restriction))),
        "by_kind": {k: r_kinds.get(k, 0) for k in (*RESTRICTION_KINDS, "none")},
        "on_class_without_qualified_cardinality": on_class_no_card,
        "on_property": {p: on_prop[p] for p in sorted(on_prop, key=lambda k: (-on_prop[k], k))},
        "used_in": dict(sorted(used_in.items())),
    }

    def split(prop) -> dict[str, Any]:
        pairs = list(g.subject_objects(prop))
        nn = sum(isinstance(s, URIRef) and isinstance(o, URIRef) for s, o in pairs)
        to_anon = Counter(_kind(describe_class(g, o)) for s, o in pairs if isinstance(s, URIRef) and isinstance(o, BNode))
        return {
            "total": len(pairs),
            "named_to_named": nn,
            "named_to_anonymous": dict(to_anon),
            "anonymous_subject": sum(isinstance(s, BNode) for s, _ in pairs),
        }

    sub_props = defaultdict(list)
    for s, o in g.subject_objects(RDFS.subPropertyOf):
        sub_props[str(s)].append(str(o))

    for p, kinds in kinds_of.items():
        props_detail.setdefault(p, {"property": p})
    for p, d in props_detail.items():
        d["kinds"] = sorted(kinds_of.get(p, []))
        d["super_properties"] = sorted(sub_props.get(p, []))
        d["characteristics"] = [c for c in CHARACTERISTICS if p in characteristics[c]]
        d["deprecated"] = deprecated(URIRef(p))
        d.setdefault("domains", [])
        d.setdefault("ranges", [])

    # Anomalies a checker author should know about
    obj_dt_range, dt_class_range = [], []
    for p, d in props_detail.items():
        for r in d["ranges"]:
            if "named" not in r:
                continue
            if "owl:ObjectProperty" in d["kinds"] and _is_datatype(r["named"]):
                obj_dt_range.append({"property": p, "range": r["named"]})
            if "owl:DatatypeProperty" in d["kinds"] and not _is_datatype(r["named"]):
                dt_class_range.append({"property": p, "range": r["named"]})
    referenced = set()
    for d in props_detail.values():
        for desc in d["domains"] + d["ranges"]:
            stack = [desc]
            while stack:
                x = stack.pop()
                if "named" in x:
                    referenced.add(x["named"])
                else:
                    stack.extend(m for m in x.get("members", []) if isinstance(m, dict))
                    stack.extend(x.get("sub_class_of", []))
    for s, o in g.subject_objects(RDFS.subClassOf):
        if isinstance(o, URIRef):
            referenced.add(str(o))
    undeclared = sorted(
        c for c in referenced if URIRef(c) not in named_classes and not _is_datatype(c) and c != OWL_NS + "Thing"
    )

    out["classes"] = {
        "named": len(named_classes),
        "named_deprecated": sum(deprecated(c) for c in named_classes),
        "anonymous_class_nodes": len(anon_class_nodes),
        "referenced_but_not_declared": undeclared,
    }
    out["properties"] = {
        "by_kind": {label: sum(label in k for k in kinds_of.values()) for label in PROPERTY_KINDS},
        "distinct": len(kinds_of),
        "with_more_than_one_kind": sorted(p for p, k in kinds_of.items() if len(set(k)) > 1),
        "deprecated": sum(deprecated(URIRef(p)) for p in kinds_of),
        "characteristics": {c: len(v) for c, v in characteristics.items()},
        "inverse_of_pairs": len(list(g.subject_objects(owl.inverseOf))),
        "property_chain_axioms": len(list(g.subject_objects(owl.propertyChainAxiom))),
        "equivalent_property": len(list(g.subject_objects(owl.equivalentProperty))),
        "property_disjoint_with": len(list(g.subject_objects(owl.propertyDisjointWith))),
    }
    out["domain"] = domain
    out["range"] = range_
    out["disjointness"] = out_disjoint
    out["restrictions"] = out_restr
    out["sub_class_of"] = split(RDFS.subClassOf)
    out["sub_property_of"] = {"total": len(list(g.subject_objects(RDFS.subPropertyOf)))}
    out["equivalent_class"] = split(owl.equivalentClass)
    out["anomalies"] = {
        "object_property_with_datatype_range": obj_dt_range,
        "datatype_property_with_class_range": dt_class_range,
        "properties_with_multiple_domains": domain["properties_with_multiple"],
        "properties_with_multiple_ranges": range_["properties_with_multiple"],
        "domain_or_range_on_undeclared_property": sorted(
            set(domain["on_undeclared_properties"]) | set(range_["on_undeclared_properties"])
        ),
    }
    out["properties_detail"] = [props_detail[p] for p in sorted(props_detail)]
    return out


# ---------------------------------------------------------------------------
# Raw XML counts (what a grep of the file sees, minus comments)

_RAW_TAGS = (
    "rdfs:domain",
    "rdfs:range",
    "rdfs:subClassOf",
    "rdfs:subPropertyOf",
    "owl:Class",
    "owl:ObjectProperty",
    "owl:DatatypeProperty",
    "owl:AnnotationProperty",
    "rdf:Property",
    "owl:disjointWith",
    "owl:AllDisjointClasses",
    "owl:disjointUnionOf",
    "owl:unionOf",
    "owl:intersectionOf",
    "owl:complementOf",
    "owl:oneOf",
    "owl:Restriction",
    "owl:onProperty",
    "owl:onClass",
    *(f"owl:{k}" for k in RESTRICTION_KINDS),
    "owl:equivalentClass",
    "owl:inverseOf",
)
_NS = {"rdf": RDF_NS, "rdfs": RDFS_NS, "owl": OWL_NS}


def _clark(qname: str) -> str:
    p, local = qname.split(":")
    return "{" + _NS[p] + "}" + local


def raw_xml_counts(path: str | Path) -> dict[str, Any]:
    """Element counts of an RDF/XML file (comments are not elements)."""
    root = ET.parse(str(path)).getroot()
    tags = Counter(e.tag for e in root.iter())
    about = _clark("rdf:about")
    resource = _clark("rdf:resource")
    out: dict[str, Any] = {"elements": {q: tags.get(_clark(q), 0) for q in _RAW_TAGS}}
    for q in ("rdfs:domain", "rdfs:range"):
        split = Counter()
        for e in root.iter(_clark(q)):
            if resource in e.attrib:
                split["rdf_resource"] += 1
            elif len(e) == 1 and about in e[0].attrib:
                split["nested_named_class"] += 1
            else:
                split["nested_anonymous"] += 1
        out[q] = dict(split)
    type_res = Counter(
        e.attrib.get(resource, "") for e in root.iter(_clark("rdf:type")) if e.attrib.get(resource, "").startswith(OWL_NS)
    )
    out["rdf_type_owl"] = {curie(k): v for k, v in sorted(type_res.items())}
    seen = Counter(e.attrib[about] for e in root if about in e.attrib)
    out["top_level_about_repeated"] = sorted(k for k, n in seen.items() if n > 1)
    return out


# ---------------------------------------------------------------------------
# Markdown


def _pct(a: int, b: int) -> str:
    return "–" if not b else f"{100 * a / b:.1f}%"


def profile_markdown(p: dict[str, Any], raw: dict[str, Any] | None = None, ontology_name: str = "core.owl") -> str:
    src = p["source"]
    L = [
        f"# Profile of `{src['file'] or 'ontology'}`",
        "",
        f"Ontology `{src['ontology']}`, versionInfo `{src['version_info']}`, {src['triples']} triples, sha256 `{src['sha256']}`.",
        "Counts are of RDF triples (rdflib)" + ("; the raw XML element counts are in the last section." if raw else "."),
        "",
        "## Classes and properties",
        "",
        "| Item | Count |",
        "|---|---|",
        f"| Named classes (owl:Class or rdfs:Class) | {p['classes']['named']} |",
        f"| of which deprecated | {p['classes']['named_deprecated']} |",
        f"| Anonymous owl:Class nodes (expressions) | {p['classes']['anonymous_class_nodes']} |",
    ]
    for k, v in p["properties"]["by_kind"].items():
        L.append(f"| {k} | {v} |")
    pr = p["properties"]
    L += [
        f"| Distinct declared properties | {pr['distinct']} |",
        f"| of which deprecated | {pr['deprecated']} |",
        f"| rdfs:subClassOf triples | {p['sub_class_of']['total']} (named→named {p['sub_class_of']['named_to_named']}, named→anonymous {sum(p['sub_class_of']['named_to_anonymous'].values())} {p['sub_class_of']['named_to_anonymous']}) |",
        f"| owl:equivalentClass triples | {p['equivalent_class']['total']} (named→named {p['equivalent_class']['named_to_named']}, named→anonymous {p['equivalent_class']['named_to_anonymous']}) |",
        f"| rdfs:subPropertyOf triples | {p['sub_property_of']['total']} |",
        f"| owl:inverseOf pairs | {pr['inverse_of_pairs']} |",
    ]
    for c, n in pr["characteristics"].items():
        L.append(f"| owl:{c} | {n} |")
    L += [
        f"| owl:propertyChainAxiom | {pr['property_chain_axioms']} |",
        f"| owl:equivalentProperty | {pr['equivalent_property']} |",
        "",
        "## Domain and range",
        "",
        "| | rdfs:domain | rdfs:range |",
        "|---|---|---|",
    ]
    d, r = p["domain"], p["range"]
    rows = [
        ("Triples", d["triples"], r["triples"]),
        ("Properties with a declaration", d["properties"], r["properties"]),
        ("Named class", d["named_class"], r["named_class"]),
        ("Datatype (XSD or rdfs:Literal)", d["datatype"], r["datatype"]),
        ("Anonymous class expression", d["class_expression"], r["class_expression"]),
    ]
    rows += [(f"– {k}", d["class_expression_by_kind"][k], r["class_expression_by_kind"][k]) for k in EXPRESSION_KINDS]
    rows += [("Anonymous data range (oneOf literals, datatype restriction)", d["data_range"], r["data_range"])]
    rows += [
        ("Union members, total", d["union_members"]["members_total"], r["union_members"]["members_total"]),
        ("Union size distribution (size: count)", d["union_members"]["size_distribution"], r["union_members"]["size_distribution"]),
        ("Unions of named classes only", d["union_members"]["unions_of_named_classes_only"], r["union_members"]["unions_of_named_classes_only"]),
        ("Properties with more than one declaration", len(d["properties_with_multiple"]), len(r["properties_with_multiple"])),
    ]
    for label, a, b in rows:
        L.append(f"| {label} | {a} | {b} |")
    L += ["", "Anonymous domain and range expressions:", ""]
    for det in p["properties_detail"]:
        for axis in ("domains", "ranges"):
            for desc in det[axis]:
                if "named" not in desc:
                    L.append(f"- {curie(det['property'])} {axis[:-1]}: {_short(desc)}")
    dj = p["disjointness"]
    L += [
        "",
        "## Disjointness",
        "",
        f"{dj['declarations']} declarations: {dj['disjoint_with_pairs']} owl:disjointWith, "
        f"{dj['all_disjoint_classes']} owl:AllDisjointClasses, {len(dj['disjoint_union_of'])} owl:disjointUnionOf.",
        "",
    ]
    L += [f"- {curie(a)} disjointWith {curie(b)}" for a, b in dj["disjoint_with"]]
    rs = p["restrictions"]
    L += [
        "",
        "## Restrictions",
        "",
        f"{rs['total']} restriction nodes (nodes with owl:onProperty; {rs['typed_owl_restriction']} typed owl:Restriction). "
        f"onClass without a qualified cardinality: {rs['on_class_without_qualified_cardinality']}.",
        "",
        "| Kind | Count |",
        "|---|---|",
    ]
    L += [f"| {k} | {n} |" for k, n in rs["by_kind"].items() if n]
    L += ["", f"Used in: {rs['used_in']}", "", "On property (count): " + ", ".join(f"{curie(k)} {v}" for k, v in rs["on_property"].items())]
    an = p["anomalies"]
    L += ["", "## Anomalies", ""]
    L.append(f"- Object properties with a datatype range: {', '.join(curie(x['property']) + ' → ' + curie(x['range']) for x in an['object_property_with_datatype_range']) or 'none'}")
    L.append(f"- Datatype properties with a class range: {', '.join(curie(x['property']) + ' → ' + curie(x['range']) for x in an['datatype_property_with_class_range']) or 'none'}")
    L.append("- Properties with more than one domain (read as an intersection): " + ("; ".join(f"{curie(x['property'])}: {', '.join(x['values'])}" for x in an["properties_with_multiple_domains"]) or "none"))
    L.append("- Properties with more than one range (read as an intersection): " + ("; ".join(f"{curie(x['property'])}: {', '.join(x['values'])}" for x in an["properties_with_multiple_ranges"]) or "none"))
    L.append(f"- Domain or range on a property {ontology_name} does not declare: {', '.join(curie(x) for x in an['domain_or_range_on_undeclared_property']) or 'none'}")
    L.append(f"- Classes referenced in domain, range or subClassOf but not declared: {', '.join(curie(x) for x in p['classes']['referenced_but_not_declared']) or 'none'}")
    if raw:
        L += ["", "## Raw RDF/XML elements versus triples", ""]
        L += ["| Element | XML elements | Triples |", "|---|---|---|"]
        pairs = [
            ("rdfs:domain", d["triples"]),
            ("rdfs:range", r["triples"]),
            ("owl:disjointWith", dj["disjoint_with_pairs"]),
            ("owl:Restriction", rs["typed_owl_restriction"]),
            ("owl:onProperty", rs["total"]),
            ("owl:allValuesFrom", rs["by_kind"]["allValuesFrom"]),
            ("owl:someValuesFrom", rs["by_kind"]["someValuesFrom"]),
            ("rdfs:subClassOf", p["sub_class_of"]["total"]),
            ("rdfs:subPropertyOf", p["sub_property_of"]["total"]),
            ("owl:equivalentClass", p["equivalent_class"]["total"]),
            ("owl:inverseOf", pr["inverse_of_pairs"]),
            ("owl:unionOf", None),
        ]
        for q, t in pairs:
            L.append(f"| {q} | {raw['elements'].get(q, 0)} | {'' if t is None else t} |")
        L += [
            "",
            f"- rdfs:domain elements: {raw['rdfs:domain']}; rdfs:range elements: {raw['rdfs:range']} "
            "(nested_named_class is an inline `rdf:Description rdf:about`, a named class, not an expression).",
            f"- rdf:type elements naming OWL types: {raw['rdf_type_owl']}",
            f"- Top-level rdf:about repeated (the same IRI declared in two blocks): {', '.join(curie(x) for x in raw['top_level_about_repeated']) or 'none'}",
        ]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Data usage (needs the store)

def usage_query(graphs: tuple[str, ...] | list[str]) -> str:
    """One GROUP BY over the requested named graphs only."""
    values = " ".join(f"<{g}>" for g in graphs)
    return f"SELECT ?g ?p (COUNT(*) AS ?n) WHERE {{ VALUES ?g {{ {values} }} GRAPH ?g {{ ?s ?p ?o }} }} GROUP BY ?g ?p"


def graph_totals_query(graphs: tuple[str, ...] | list[str]) -> str:
    """Triples per requested graph, counted independently of the per-predicate GROUP BY."""
    values = " ".join(f"<{g}>" for g in graphs)
    return f"SELECT ?g (COUNT(*) AS ?n) WHERE {{ VALUES ?g {{ {values} }} GRAPH ?g {{ ?s ?p ?o }} }} GROUP BY ?g"


def per_graph_usage_query(graph: str) -> str:
    return f"SELECT ?p (COUNT(*) AS ?n) WHERE {{ GRAPH <{graph}> {{ ?s ?p ?o }} }} GROUP BY ?p"


_QUERY_ERRORS = (SparqlError, ValueError, KeyError, TypeError, AttributeError)


def predicate_usage(client: SparqlClient, graphs: tuple[str, ...] | list[str] = LOADED_GRAPHS) -> dict[str, Any]:
    """Triples per (graph, predicate) over ``graphs``: one GROUP BY, or one query per graph if that fails.

    ``complete`` is False when any requested graph failed or returned no
    triples (``missing_graphs``), when a graph's per-predicate counts do not
    sum to its independently counted triples (``total_mismatches``), or when
    those totals could not be read (``totals_error``). Shares computed from an
    incomplete result must not be read as the store's.
    """
    graphs = list(graphs)
    errors: list[dict[str, str]] = []
    counts: list[dict[str, Any]] = []
    method = "group_by_graph"
    try:
        res = client.query(usage_query(graphs))
        for g, p, n in res.rows:
            counts.append({"graph": g.value, "predicate": p.value, "triples": int(n.lexical)})
    except _QUERY_ERRORS as exc:
        errors.append({"graph": "*", "error": str(exc)[:500]})
        method, counts = "per_graph", []
        for graph in graphs:
            try:
                res = client.query(per_graph_usage_query(graph))
                for p, n in res.rows:
                    counts.append({"graph": graph, "predicate": p.value, "triples": int(n.lexical)})
            except _QUERY_ERRORS as exc2:
                errors.append({"graph": graph, "error": str(exc2)[:500]})
    seen = {c["graph"] for c in counts if c["triples"] > 0}
    missing = [g for g in graphs if g not in seen]
    sums: dict[str, int] = {}
    for c in counts:
        sums[c["graph"]] = sums.get(c["graph"], 0) + c["triples"]
    totals: dict[str, int] = {}
    totals_error = None
    try:
        res = client.query(graph_totals_query(graphs))
        totals = {g.value: int(n.lexical) for g, n in res.rows}
    except _QUERY_ERRORS as exc:
        totals_error = str(exc)[:500]
    mismatches = [] if totals_error else [
        {"graph": g, "predicate_sum": sums.get(g, 0), "triples": totals.get(g, 0)}
        for g in graphs
        if g not in missing and sums.get(g, 0) != totals.get(g, 0)
    ]
    return {
        "method": method,
        "graphs_requested": graphs,
        "complete": not missing and not mismatches and totals_error is None,
        "missing_graphs": missing,
        "graph_totals": totals,
        "total_mismatches": mismatches,
        "totals_error": totals_error,
        "counts": counts,
        "errors": errors,
    }


def _ancestors(p: str, supers: dict[str, list[str]]) -> set[str]:
    seen, stack = set(), list(supers.get(p, []))
    while stack:
        q = stack.pop()
        if q not in seen:
            seen.add(q)
            stack.extend(supers.get(q, []))
    return seen


def data_usage(
    profile: dict[str, Any],
    usage: dict[str, Any],
    ontology_graph: str = CORE_GRAPH,
    namespace: str = UP_NS,
    namespace_scope: str = "up_namespace",
) -> dict[str, Any]:
    """Join predicate usage with the profile.

    Denominators leave out ``ontology_graph`` (the ontology itself). ``namespace``
    is the ontology's own namespace: it picks the ``namespace_scope`` coverage
    row and the declared-but-unused list. The defaults are UniProt's. A domain or
    range declared on a super-property (rdfs:subPropertyOf) is reported
    separately as "inherited"; the direct columns do not count it.
    """
    detail = {d["property"]: d for d in profile["properties_detail"]}
    supers = {p: d["super_properties"] for p, d in detail.items()}
    per_pred: dict[str, dict[str, Any]] = {}
    for c in usage["counts"]:
        e = per_pred.setdefault(c["predicate"], {"predicate": c["predicate"], "triples": 0, "per_graph": {}})
        e["per_graph"][c["graph"]] = e["per_graph"].get(c["graph"], 0) + c["triples"]
        if c["graph"] != ontology_graph:
            e["triples"] += c["triples"]

    def axis(p: str, key: str) -> dict[str, Any]:
        own = detail.get(p, {}).get(key, [])
        inherited = [s for a in sorted(_ancestors(p, supers)) for s in detail.get(a, {}).get(key, [])]
        return {
            "declared": bool(own),
            "named": sorted(d["named"] for d in own if category(d) == "named_class"),
            "datatypes": sorted(d["named"] for d in own if category(d) == "datatype"),
            "anonymous": [_short(d) for d in own if category(d) == "class_expression"],
            "data_ranges": [_short(d) for d in own if category(d) == "data_range"],
            "inherited": bool(inherited),
        }

    for p, e in per_pred.items():
        d = detail.get(p)
        e["declared_in_ontology"] = d is not None and bool(d.get("kinds"))
        e["kinds"] = d["kinds"] if d else []
        e["domain"] = axis(p, "domains")
        e["range"] = axis(p, "ranges")

    def coverage(entries: list[dict]) -> dict[str, Any]:
        total = sum(e["triples"] for e in entries)

        def share(pred) -> dict[str, Any]:
            n = sum(e["triples"] for e in entries if pred(e))
            return {"triples": n, "share": (n / total) if total else None}

        return {
            "predicates": len(entries),
            "triples": total,
            "declared_in_ontology": share(lambda e: e["declared_in_ontology"]),
            "domain_declared": share(lambda e: e["domain"]["declared"]),
            "domain_declared_or_inherited": share(lambda e: e["domain"]["declared"] or e["domain"]["inherited"]),
            "domain_named_class_only": share(lambda e: e["domain"]["declared"] and not e["domain"]["anonymous"]),
            "domain_class_expression": share(lambda e: bool(e["domain"]["anonymous"])),
            "range_declared": share(lambda e: e["range"]["declared"]),
            "range_declared_or_inherited": share(lambda e: e["range"]["declared"] or e["range"]["inherited"]),
            "range_class": share(lambda e: bool(e["range"]["named"] or e["range"]["anonymous"])),
            "range_named_class_only": share(lambda e: bool(e["range"]["named"]) and not e["range"]["anonymous"]),
            "range_class_expression": share(lambda e: bool(e["range"]["anonymous"])),
            "range_datatype_or_data_range": share(lambda e: bool(e["range"]["datatypes"] or e["range"]["data_ranges"])),
        }

    data_entries = [e for e in per_pred.values() if e["triples"] > 0]
    entries = sorted(per_pred.values(), key=lambda e: (-e["triples"], e["predicate"]))
    return {
        "method": usage["method"],
        "complete": usage.get("complete", False),
        "missing_graphs": usage.get("missing_graphs", []),
        "total_mismatches": usage.get("total_mismatches", []),
        "totals_error": usage.get("totals_error"),
        "graphs_requested": usage.get("graphs_requested", []),
        "errors": usage["errors"],
        "ontology_graph_excluded": ontology_graph,
        "graphs": sorted({c["graph"] for c in usage["counts"]}),
        "coverage": {
            "all_predicates": coverage(data_entries),
            "all_except_rdf_type": coverage([e for e in data_entries if e["predicate"] != RDF_NS + "type"]),
            namespace_scope: coverage([e for e in data_entries if e["predicate"].startswith(namespace)]),
        },
        "declared_but_unused": sorted(
            p for p, d in detail.items() if d.get("kinds") and p not in per_pred and p.startswith(namespace)
        ),
        "predicates": entries,
    }


def usage_markdown(u: dict[str, Any], ontology_name: str = "core.owl") -> str:
    def fmt(s: dict) -> str:
        return "–" if s["share"] is None else f"{100 * s['share']:.1f}%"

    L = [f"# Predicate usage in the store versus {ontology_name}", ""]
    if not u["complete"]:
        why = []
        if u["missing_graphs"]:
            why.append("graphs that failed or returned no triples: " + ", ".join(u["missing_graphs"]))
        if u.get("total_mismatches"):
            why.append("graphs whose per-predicate counts do not sum to their triple count: " + ", ".join(
                f"{m['graph']} ({m['predicate_sum']:,} vs {m['triples']:,})" for m in u["total_mismatches"]))
        if u.get("totals_error"):
            why.append(f"per-graph triple totals could not be read ({u['totals_error'][:120]})")
        L += ["**PARTIAL: the counts and shares below are not the store's; " + "; ".join(why) + ".**", ""]
    L += [
        f"Method: `{u['method']}`; graphs seen: {len(u['graphs'])}; errors: {len(u['errors'])}. "
        f"Triple counts leave out the ontology graph `{u['ontology_graph_excluded']}`. "
        "\"Inherited\" counts a domain or range declared on a super-property.",
        "",
        "| Scope | Predicates | Triples | Declared | Domain | Domain incl. inherited | Domain is a class expression "
        "| Range | Range incl. inherited | Range is a class | Range is a datatype |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for scope, c in u["coverage"].items():
        L.append(
            f"| {scope} | {c['predicates']} | {c['triples']:,} | {fmt(c['declared_in_ontology'])} | {fmt(c['domain_declared'])} | "
            f"{fmt(c['domain_declared_or_inherited'])} | {fmt(c['domain_class_expression'])} | {fmt(c['range_declared'])} | "
            f"{fmt(c['range_declared_or_inherited'])} | {fmt(c['range_class'])} | {fmt(c['range_datatype_or_data_range'])} |"
        )
    L += ["", "| Predicate | Triples | Graphs | Declared as | Domain | Range |", "|---|---|---|---|---|---|"]

    def ax(a: dict) -> str:
        if not a["declared"]:
            return "inherited" if a["inherited"] else "–"
        parts = [curie(n) for n in a["named"] + a["datatypes"]] + a["anonymous"] + a["data_ranges"]
        return "; ".join(parts)

    for e in u["predicates"]:
        graphs = ", ".join(curie(g) if not g.startswith("http://sparql.uniprot.org/") else g.rsplit("/", 1)[-1] for g in sorted(e["per_graph"]))
        L.append(
            f"| {curie(e['predicate'])} | {e['triples']:,} | {graphs} | {', '.join(e['kinds']) or '–'} | {ax(e['domain'])} | {ax(e['range'])} |"
        )
    if u["declared_but_unused"]:
        L += ["", f"Declared in {ontology_name} but unused in the data ({len(u['declared_but_unused'])}): " + ", ".join(curie(p) for p in u["declared_but_unused"])]
    return "\n".join(L) + "\n"
