"""The curated source: the graph's own schema plus the ontologies shipped with it.

For Rhea that is Rhea's RDF schema (the schema triples of the Rhea graph) and
ChEBI (the ChEBI graph), read by the one OWL reader (``owl_reader``) that also
reads generated ontologies. The source keeps the full ontology, because the
checkers use the full closure; rendering and profiling restrict it to the terms
the data uses plus their ancestors (``scope``), and that restriction is
recorded in provenance, not applied here. The subclass closure is computed
offline from the parent links (``closure.Hierarchy``); the store does no
reasoning.

In the Rhea graph every compound and reaction is typed rdfs:Class, so triples
whose subject is a data node are skipped (``vocab.not_data_node``): what is
left is the schema's own 21 classes and 57 properties.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from harness.sources import origin
from harness.sources.closure import Hierarchy
from harness.sources.model import OntologySource
from harness.sources.origin import source_axioms
from harness.sources.owl_reader import read_owl_fragment, read_vocabulary_restricted
from harness.sources.scope import SCOPE_RULE
from harness.sources.vocab import MEMBERSHIP_READING_BASE, StoreLayout, membership_label


def build_curated(client, layout: StoreLayout) -> OntologySource:
    whole = layout.curated_graphs or layout.graphs
    # The data-node filter is needed only where the schema sits inside the data graph (Rhea).
    data_filter = layout if layout.data_graph in whole else None
    frag = read_owl_fragment(client, whole, data_filter)
    # Per-axiom origin (design ruling in the design log Oct 08:04): each graph read alone too, by the same reader
    # (harness.sources.origin); one graph is its own reading.
    parts = {whole[0]: source_axioms(frag)} if len(whole) == 1 else {
        g: source_axioms(read_owl_fragment(client, [g], data_filter)) for g in whole}
    restricted = []
    declared = set(frag.declared)
    for g in layout.curated_restricted_graphs:
        part, info = read_vocabulary_restricted(client, g, layout.data_graph)
        parts[g] = source_axioms(part)
        restricted.append(info)
        declared |= set(part.declared)
        for c, ci in part.classes.items():
            if c not in frag.classes:
                frag.classes[c] = ci
        frag.existentials = sorted(set(frag.existentials) | set(part.existentials), key=lambda e: (e.cls, e.prop, e.filler))
        for k, n in part.unsupported.items():
            frag.unsupported[k] = frag.unsupported.get(k, 0) + n
    frag.unsupported = dict(sorted(frag.unsupported.items()))
    hier = Hierarchy({c: i.parents for c, i in frag.classes.items()})
    # Rule 32 (a): the classes the source declares (owl:Class / rdfs:Class typings), as the generated reader records
    # them (external reviewer 6 review of a929c39, item 2); left out (None, "every listed class is declared") when they are all.
    declared &= set(frag.classes)
    declared_classes = None if declared == set(frag.classes) else sorted(declared)
    n_dr = sum(len(p.domain) + len(p.range) for p in frag.properties.values())
    src = OntologySource(
        name="curated",
        kind="curated",
        graph=layout.name,
        membership=membership_label(layout),
        classes=frag.classes,
        properties=frag.properties,
        disjoint=frag.disjoint,
        existentials=frag.existentials,
        unsupported=frag.unsupported,
        declared_classes=declared_classes,
        described=frag.described or None,
        provenance={
            "builder": "harness.sources.curated",
            "layout": {"name": layout.name, "data_graph": layout.data_graph, "graphs": list(layout.graphs)},
            "read": {
                "whole": list(whole),
                "whole_reader": "harness.sources.owl_reader.read_owl_fragment (skipping triples whose subject is a "
                                "data node where the schema sits in the data graph; an individual, typed "
                                "owl:NamedIndividual or as a member of a class, is no class unless declared one or "
                                "used as one by an axiom)",
                "restricted": restricted,
                "restricted_reader": "harness.sources.owl_reader.read_vocabulary_restricted (terms the data uses plus "
                                     "their ancestors, check contract v2 rule 16)",
            },
            "membership_reading": MEMBERSHIP_READING_BASE + layout.reading_note,
            "excluded": frag.excluded,
            "closure": "computed offline from ClassInfo.parents (harness.sources.closure.Hierarchy); no reasoning in the store",
            "render_and_profile_scope": SCOPE_RULE,
            "counts": {
                "classes": len(frag.classes),
                "root_classes": len(hier.roots(frag.classes)),
                "subclass_axioms": sum(len(c.parents) for c in frag.classes.values()),
                "properties": len(frag.properties),
                "domain_range_declarations": n_dr,
                "disjoint_pairs": len(frag.disjoint),
                "existentials": len(frag.existentials),
                "classes_by_namespace": _by_namespace(frag.classes),
                "properties_by_namespace": _by_namespace(frag.properties),
            },
        },
    )
    src.origin = origin.record(src, parts, "graph")
    return src


_OBO_ID = re.compile(r"^(http://purl\.obolibrary\.org/obo/[A-Za-z]+_)\d+$")


def _by_namespace(iris) -> dict[str, int]:
    """Counts by namespace; OBO identifiers (obo/CHEBI_15377) count under their ontology prefix."""
    out: dict[str, int] = {}
    for iri in iris:
        m = _OBO_ID.match(iri)
        ns = m.group(1) if m else iri[: max(iri.rfind("#"), iri.rfind("/")) + 1]
        out[ns] = out.get(ns, 0) + 1
    return dict(sorted(out.items()))


# The one graph the published files are read into (the name is the build's own; no store holds it).
CURATED_FILES_GRAPH = "urn:www2027:curated-files"
_NO_DATA_GRAPH = "urn:www2027:no-data-graph"


def directory_digest(directory) -> str:
    """sha256 of ``LC_ALL=C sha256sum * | LC_ALL=C sort`` run inside ``directory``: one '<sha256>  <name>' line per
    non-hidden file, sorted bytewise, then hashed (the recipe an ontology release's version digest is minted by)."""
    lines = []
    for path in Path(directory).iterdir():
        if path.name.startswith("."):
            continue  # the shell's * does not match hidden files
        if not path.is_file() or "\n" in path.name or "\\" in path.name:
            raise ValueError("the ontology directory must hold plain files only")
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n".encode())
    return hashlib.sha256(b"".join(sorted(lines))).hexdigest()


def build_curated_from_files(name: str, files, base: str | None = None) -> OntologySource:
    """The curated source from published RDF files read whole (rule 15, as published): every file is parsed in its
    stated syntax into one graph of an in-memory store, which ``build_curated`` reads like any curated graph, the
    class rule of change-log entry 94 included (a class only if declared or used as one; a term only labelled or
    commented is described vocabulary). ``files``: (path, rdflib format) pairs.

    Each file is parsed with its own document base, so a relative IRI means in each file what it means where that file
    is published (External review of c00244b, item 7: two files' <#A> are two terms): the file's own @base or xml:base
    where it states one, else ``base`` (the directory's published location) followed by the file's name, else the
    file's own location (its file: URI, which then makes such IRIs depend on where the files sit). The base each file
    was given is recorded."""
    from rdflib import Dataset, Graph, URIRef

    from harness.sources.local import LocalSparqlClient

    ds = Dataset(default_union=True)
    graph = ds.graph(URIRef(CURATED_FILES_GRAPH))
    read, alone = [], {}
    for path, fmt in files:
        path = Path(path)
        if path.name in alone:
            raise ValueError(f"two files are named {path.name}: a file's name is its origin")
        raw = path.read_bytes()
        doc_base = base + path.name if base is not None else path.resolve().as_uri()
        part = Graph()
        part.parse(data=raw, format=fmt, publicID=doc_base)
        for triple in part:
            graph.add(triple)
        alone[path.name] = part
        read.append({"file": path.name, "sha256": hashlib.sha256(raw).hexdigest(), "format": fmt,
                     "triples": len(part), "base": doc_base})
    layout = StoreLayout(name=name, data_graph=_NO_DATA_GRAPH, graphs=(CURATED_FILES_GRAPH,),
                         curated_graphs=(CURATED_FILES_GRAPH,))
    src = build_curated(LocalSparqlClient(ds), layout)
    # Per-axiom origin (design ruling in the design log Oct 08:04): each file read alone too, as its own store
    parts = {}
    for file_name, part in alone.items():
        one = Dataset(default_union=True)
        g1 = one.graph(URIRef(CURATED_FILES_GRAPH))
        for triple in part:
            g1.add(triple)
        parts[file_name] = source_axioms(build_curated(LocalSparqlClient(one), layout))
    # An axiom the reader joined from several files' triples is in no file's own reading: it comes from the files
    # holding those triples (review of fc05950, item 2).
    src.origin = origin.record(src, parts, "file", origin.traced_joins(src, graph, alone))
    src.provenance["read"]["files"] = read
    src.provenance["read"]["graph_triples"] = len(graph)
    src.provenance["read"]["from_files"] = ("each published file parsed in its stated syntax into one in-memory graph "
                                            "(harness.sources.curated.build_curated_from_files); no store, no data graph")
    return src


# Design ruling in the design log: the curated source as published stays primary; "curated+doc", used only
# offline by the checker, adds declarations of the terms the release's documentation describes but
# its RDF schema omits, and nothing else (no domain, range or other axiom).
RHEA_142_DOC = {
    "file": "rhea_rdf_documentation.pdf",
    "release": "Rhea release 142",
    "url": "https://ftp.expasy.org/databases/rhea/rdf/rhea_rdf_documentation.pdf",
    "sha256": "70bc1bd0247aac309781958b78d8eafdd4010f3e5f42d0d9e9abbad5663729b5",
    "pages_note": "1-based PDF pages, extracted with pypdf by the lead",
}
DOC_ADDITIONS = {
    "rhea-142": {
        "http://rdf.rhea-db.org/compound": [5, 6, 11, 15, 19, 21, 23],
        "http://rdf.rhea-db.org/chebi": [5, 6, 14, 15, 18, 19, 23],
        "http://rdf.rhea-db.org/isTransport": [5, 8, 10, 22, 23, 29, 32],
        "http://rdf.rhea-db.org/isChemicallyBalanced": [5, 8, 29, 32],
    },
}
DOC_SOURCES = {"rhea-142": RHEA_142_DOC}


@dataclass(frozen=True)
class PublishedFile:
    """The published file a store's curated graph was loaded from: the committed copy (path from the repository
    root), its sha256, where it was published and under which licence."""

    path: str
    sha256: str
    url: str
    release: str
    license: str
    fmt: str = "turtle"


# Rule 15 (every source as published): where the curated source is one published file loaded whole into its own
# graph, the build proves the store graph is that file, triple for triple, before it reads the curated source.
PUBLISHED = {
    "text2sparql-ck26": PublishedFile(
        path="stores/text2sparql-ck26/prod-vocab.ttl",
        sha256="9f2d168beae66a89d29e9b938efd9ebd9a9e8a1a70436884f9899538208427e8",
        url="https://raw.githubusercontent.com/AKSW/text2sparql.aksw.org/935cb5a210717a7596a6028740136ac842db7ede/"
            "docs/participation/data/prod-vocab.ttl",
        release="Text2SPARQL 2026 corporate KG: AKSW/text2sparql.aksw.org, branch 2026, commit "
                "935cb5a210717a7596a6028740136ac842db7ede (stores/text2sparql-ck26/inputs.tsv)",
        license="CC BY 4.0 (the ontology's own cc:license; the repository is CC BY-SA 4.0), by eccenca GmbH",
    ),
}


class NotAsPublished(ValueError):
    """A curated graph that is not its published file, or a committed copy that is not the published file."""


def _rdflib_term(term):
    from rdflib import BNode as RBNode
    from rdflib import Literal as RLiteral
    from rdflib import URIRef

    from harness.terms import IRI, BNode

    if isinstance(term, IRI):
        return URIRef(term.value)
    if isinstance(term, BNode):
        return RBNode(term.label)
    if term.lang:
        return RLiteral(term.lexical, lang=term.lang)
    return RLiteral(term.lexical, datatype=URIRef(term.datatype) if term.datatype else None)


def as_published(client, layout: StoreLayout, root: Path) -> dict:
    """The provenance of a curated graph loaded from one published file (``PUBLISHED``), refused unless the committed
    copy has the published sha256 and the store's graph holds exactly its triples (isomorphic: equal up to blank-node
    labels). Literals compare by lexical form, datatype and language tag."""
    from rdflib import Graph
    from rdflib.compare import graph_diff, isomorphic, to_isomorphic

    pub = PUBLISHED[layout.name]
    (graph,) = layout.curated_graphs  # one published file, one graph
    raw = (Path(root) / pub.path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != pub.sha256:
        raise NotAsPublished(f"{pub.path} has sha256 {sha}, not the published file's {pub.sha256}")
    published = Graph()
    published.parse(data=raw.decode("utf-8"), format=pub.fmt)
    result = client.query(f"SELECT ?s ?p ?o WHERE {{ GRAPH <{graph}> {{ ?s ?p ?o }} }}")
    if result.truncated:
        raise NotAsPublished(f"the store's {graph} could not be read whole ({result.total_rows} rows)")
    stored = Graph()
    for s, p, o in result.rows:
        stored.add((_rdflib_term(s), _rdflib_term(p), _rdflib_term(o)))
    if not isomorphic(published, stored):
        _, only_file, only_store = graph_diff(to_isomorphic(published), to_isomorphic(stored))
        raise NotAsPublished(f"the store's {graph} is not {pub.path}: {len(only_file)} triples only in the file "
                             f"(e.g. {sorted(only_file)[:2]}), {len(only_store)} only in the store "
                             f"(e.g. {sorted(only_store)[:2]})")
    return {
        "rule": "check contract rule 15: the curated source as published; the store graph it is read from holds "
                "exactly the published file's triples (isomorphic, checked at build time)",
        "file": pub.path, "sha256": sha, "bytes": len(raw), "url": pub.url, "release": pub.release,
        "license": pub.license, "triples": len(published), "store_graph": graph, "store_triples": len(stored),
    }


def with_documented_terms(curated: OntologySource, layout_name: str) -> OntologySource:
    """The curated source plus declarations of exactly the documented-but-undeclared terms."""
    import copy

    from harness.sources.model import PropertyInfo

    additions = DOC_ADDITIONS[layout_name]
    clash = sorted(set(additions) & (set(curated.properties) | set(curated.classes)))
    if clash:
        raise ValueError(f"the curated source already declares {clash}")
    out = copy.deepcopy(curated)
    out.name = "curated+doc"
    for iri in sorted(additions):
        out.properties[iri] = PropertyInfo(iri)
    out.properties = dict(sorted(out.properties.items()))
    doc = DOC_SOURCES[layout_name]
    if out.origin is not None:  # the added declarations come from the documentation file
        out.origin["parts"] = [*out.origin["parts"], doc["file"]]
        out.origin["properties"] = dict(sorted({**out.origin["properties"],
                                                **{iri: [doc["file"]] for iri in additions}}.items()))
    out.provenance["variant"] = {
        "of": curated.name,
        "of_digest": curated.digest(),
        "use": "offline checker only; not rendered into prompts and not generator input",
        "ruling": "design ruling in the design log: the published curated source stays primary",
        "added_declarations": {iri: {**doc, "pages": pages} for iri, pages in sorted(additions.items())},
    }
    return out
