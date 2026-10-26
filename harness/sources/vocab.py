"""Namespaces, store layouts and SPARQL snippets shared by every source builder.

The one reading of the data that the census, the curated scope and the
generator's instance sample all use is defined here (see ``MEMBERSHIP_READING``),
so that no builder can drift from it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
OWL = "http://www.w3.org/2002/07/owl#"
XSD = "http://www.w3.org/2001/XMLSchema#"
SKOS = "http://www.w3.org/2004/02/skos/core#"
OBO = "http://purl.obolibrary.org/obo/"

RDF_TYPE = RDF + "type"
RDFS_SUBCLASSOF = RDFS + "subClassOf"
RDFS_LABEL = RDFS + "label"
RDFS_COMMENT = RDFS + "comment"

# The language's own vocabulary. A node's rdf:type in one of these namespaces
# (rdfs:Class, owl:Class, rdf:Property, owl:Restriction, owl:Axiom, ...) says
# what kind of term the node is, not which class of the domain it belongs to,
# so it is never membership. Every node of the Rhea graph is typed rdfs:Class;
# counting that as membership would make every pattern allowed.
META_NAMESPACES = (RDF, RDFS, OWL)

# The class of a node (or blank node) that has no class under the membership
# reading: citation and EC IRIs, rh:containsN subjects, OWL blank nodes. Used
# only in census patterns; it is never an ancestor of a real class, and the
# census derives no domain or range from it.
UNTYPED = RDFS + "Resource"

# Annotation properties read as labels and as comments, for every source alike.
LABEL_PROPERTIES = (RDFS_LABEL, SKOS + "prefLabel")
COMMENT_PROPERTIES = (RDFS_COMMENT, OBO + "IAO_0000115", SKOS + "definition")

MEMBERSHIP_READING_BASE = (
    "A node's asserted classes are the objects of its rdf:type triples, in any graph the census reads, that are not "
    "in the RDF, RDFS or OWL namespaces, plus the IRI objects of its rdfs:subClassOf triples in the data graph "
    "(blank-node restrictions excluded), followed transitively along the data graph's rdfs:subClassOf triples; the "
    "compared curated schema's own subclass triples are not followed. Its classes are the asserted classes and all "
    "their ancestors in the shared hierarchy: the rdfs:subClassOf triples of the census's other graphs, closed "
    "transitively over their union, without reasoning. Those triples relate classes to classes and give no node a "
    "class. A node with several asserted classes counts under each of them, and once per class however many paths "
    "lead there. A data node (an IRI that is the subject of a data-graph triple whose predicate is outside the RDF, "
    "RDFS and OWL namespaces, and that is never used as a predicate) is traversed in the chain but is never listed "
    "as a class"
)  # rule 35 (change-log entries 13 and 14); the pre-rule-35 wording named every graph's subclass triples
# The Rhea-specific end of the reading (a layout's ``reading_note``); other graphs end it with ".".
RHEA_READING_NOTE = ': in Rhea this keeps compounds (rh:Compound_N), reactions and participants out of the class vocabulary although the store models them as classes. On Rhea the data has no rdf:type outside the meta-vocabulary, so membership there is exactly rdfs:subClassOf.'
MEMBERSHIP_READING = MEMBERSHIP_READING_BASE + RHEA_READING_NOTE  # Rhea's reading, as drawn on 18 October


# Appended to the reading where a store publishes restrictions as named nodes (UniProt's GO).
NAMED_RESTRICTIONS_READING = (
    " Restrictions published as named nodes (an IRI that is the subject of owl:onProperty) are excluded "
    "likewise: they are never classes."
)


@dataclass(frozen=True)
class StoreLayout:
    """Which named graphs make up a store, and what role each plays.

    * ``data_graph``: the graph whose nodes the questions are about.
    * ``shipped_graphs``: vocabularies shipped in the store beside the data
      (ChEBI on Rhea; GO, taxonomy, ... on UniProt). Shared evidence for every
      source (check contract v2, rule 16); rendered after a source's own terms
      and only where the data uses them.
    * ``excluded_graphs``: graphs of a source under comparison (``core`` on
      UniProt). Never census input (rule 20) and never part of the shared
      subclass chain (rule 16).
    * ``exclude_schema_triples``: the curated schema's own triples sit inside
      the data graph (Rhea): the data-graph triples whose subject is a schema
      term (an IRI that is not a data node) are left out of the census input.
    * ``membership_predicate``: the graph's membership convention, which the
      census's direct count (``type_support``) reads literally.
    """

    name: str
    data_graph: str
    graphs: tuple[str, ...]
    membership_predicate: str = RDF_TYPE
    shipped_graphs: tuple[str, ...] = ()
    excluded_graphs: tuple[str, ...] = ()
    exclude_schema_triples: bool = False
    endpoint_hint: str | None = None
    # The curated source: graphs read whole (the curated ontology and, on Rhea, ChEBI), and shipped
    # vocabularies read restricted to the terms the data uses plus their ancestors (UniProt's six
    # hierarchies; taxonomy alone has 52 million subclass triples).
    curated_graphs: tuple[str, ...] = ()
    curated_restricted_graphs: tuple[str, ...] = ()
    reading_note: str = "."  # how the membership reading ends for this graph; names no other graph
    # The prompt render budget in o200k tokens, the same for every source on the graph (change-log entry 22: the
    # largest own content among curated and the generated draws, rounded up to the next 1,000); None where no number
    # has been ruled, and then a build must name one.
    render_budget: int | None = None
    # The rule that gives render_budget (harness.sources.render_guard.RULES), which the guard recomputes on the
    # current sources: "entry-22", "no-truncation", or "fit-bound" (entry 88), whose number only the committed fit
    # computation it names gives, as (path from the repository root, sha256).
    render_budget_rule: str | None = None
    render_budget_fit: tuple[str, str] | None = None
    # An opaque graph (rule 34): what is written about it (its profile) gives counts and names no term.
    opaque: bool = False
    # A graph whose name is private (a15): no graph IRI is written in the code; a build reads it from a private file
    # (scripts/build_prompts.py --graph-file) and refuses without one, and build_sources never reads such a store.
    private_graphs: bool = False
    # Arms beyond curated, census and draw 1 that the graph's prompts and fit must cover (a15, design log, 20 Oct
    # 12:54): "aligned-curated" (its own condition) and "generated-2", "generated-3" (rendered for inspection and
    # measured by the fit). A build or fit lacking one is refused, and an aligned source elsewhere is refused.
    extra_arms: tuple[str, ...] = ()
    # Rule 16 as amended (change-log entry 101): the prompt renders an imported ontology file's statements after the
    # source's own layers, imported files found by per-axiom origin and the owner's namespaces, which a build takes
    # as an explicit input (a private file) and records in the render manifest.
    imports_by_origin: bool = False

    @property
    def census_graphs(self) -> tuple[str, ...]:
        return tuple(g for g in self.graphs if g not in self.excluded_graphs)


# Change-log entry 88's fit computation for Rhea and opaque Rhea (scripts/fit_budget.py compute; design log, 20 Oct
# 18:16): the largest budget at which every model in P fits entry 60's rule by the strict reading, 27,000, with its
# sizes and models files committed beside it (results/rhea-fit-20261020).
RHEA_FIT = ("results/rhea-fit-20261020/rhea-fit.json", "05f17f276563c96362dbd96db6d3b24d60f9853a44f99fa21e31e1a80429589d")

LAYOUTS = {
    "rhea-142": StoreLayout(
        name="rhea-142",
        data_graph="https://sparql.rhea-db.org/rhea",
        graphs=("https://sparql.rhea-db.org/rhea", "http://sparql.rhea-db.org/chebi"),
        membership_predicate=RDFS_SUBCLASSOF,
        shipped_graphs=("http://sparql.rhea-db.org/chebi",),
        exclude_schema_triples=True,
        endpoint_hint="http://localhost:7019/",
        curated_graphs=("https://sparql.rhea-db.org/rhea", "http://sparql.rhea-db.org/chebi"),
        reading_note=RHEA_READING_NOTE,
        render_budget=27000,  # change-log entry 88 (design log, 20 Oct 18:16): fit-bound, one B with opaque Rhea
        render_budget_rule="fit-bound",
        render_budget_fit=RHEA_FIT,
    ),
    # Opaque Rhea (rule 34): the same recipe from the opaque store. Its name is neutral because the census
    # carries it into the generator's input, which must hold no original name.
    "opaque-rhea-142": StoreLayout(
        name="opaque-graph-1",
        data_graph="http://example.org/opaque/g0000001",
        graphs=("http://example.org/opaque/g0000001", "http://example.org/opaque/g0000000"),
        membership_predicate=RDFS_SUBCLASSOF,
        shipped_graphs=("http://example.org/opaque/g0000000",),
        exclude_schema_triples=True,
        endpoint_hint="http://localhost:7033/",
        curated_graphs=("http://example.org/opaque/g0000001", "http://example.org/opaque/g0000000"),
        render_budget=27000,  # change-log entry 88: Rhea's fit-bound budget
        render_budget_rule="fit-bound",
        render_budget_fit=RHEA_FIT,
        opaque=True,
    ),
    "uniprot-sp-2026_03": StoreLayout(
        name="uniprot-sp-2026_03",
        data_graph="http://sparql.uniprot.org/uniprot",
        graphs=tuple(f"http://sparql.uniprot.org/{g}" for g in (
            "uniprot", "taxonomy", "proteomes", "citations", "chebi", "go", "enzymes", "diseases", "journal",
            "pathways", "keywords", "locations", "tissues", "databases")) + (
            "https://sparql.rhea-db.org/rhea", "http://purl.uniprot.org/core"),
        membership_predicate=RDF_TYPE,
        shipped_graphs=tuple(f"http://sparql.uniprot.org/{g}" for g in (
            "go", "taxonomy", "enzymes", "keywords", "locations", "pathways")),
        excluded_graphs=("http://purl.uniprot.org/core",),
        endpoint_hint="http://localhost:7020/",
        curated_graphs=("http://purl.uniprot.org/core",),
        curated_restricted_graphs=tuple(f"http://sparql.uniprot.org/{g}" for g in (
            "go", "taxonomy", "enzymes", "keywords", "locations", "pathways")),
        # Design log, change-log entries 22, 56 and 60: entry 22's value, the largest own content among the curated
        # source (14,110) and the redraw's three natural draws (draw 2: 44,791), rounded up; B_fit does not bind.
        # Raised by design log, 19 Oct 15:49 (no budget may truncate a source) to the census's own content, 49,078,
        # rounded up.
        render_budget=50000,
        render_budget_rule="no-truncation",
    ),
    # The Text2SPARQL 2026 corporate KG (design log, 19 Oct): the instance graph is the data; the vocabulary graph
    # (prod-vocab.ttl, loaded whole and unchanged) is the curated source under comparison, so it is never census
    # input and never part of the shared hierarchy (rules 16, 20 and 35). No other graph holds subclass edges, so
    # the shared hierarchy is empty and membership is the literal rdf:type.
    "text2sparql-ck26": StoreLayout(
        name="text2sparql-ck26",
        data_graph="http://ld.company.org/prod-instances/",
        graphs=("http://ld.company.org/prod-instances/", "http://ld.company.org/prod-vocab/"),
        membership_predicate=RDF_TYPE,
        excluded_graphs=("http://ld.company.org/prod-vocab/",),
        endpoint_hint="http://localhost:7021/",
        curated_graphs=("http://ld.company.org/prod-vocab/",),
        render_budget=4000,  # change-log entry 94: no truncation (census 3,343 o200k; curated 2,001, draws to 2,665)
        render_budget_rule="no-truncation",
    ),
    # a15 (design log): its sources are built from files, and its graph is named only in private files. B is the
    # no-truncation value under rule 16 as amended (change-log entry 101), below the largest fitting budget.
    "a15": StoreLayout(name="a15", data_graph="", graphs=(), membership_predicate=RDF_TYPE, private_graphs=True,
                       extra_arms=("aligned-curated", "generated-2", "generated-3"), imports_by_origin=True,
                       render_budget=29000, render_budget_rule="no-truncation"),
}

PREFIXES = (
    f"PREFIX rdf: <{RDF}>\n"
    f"PREFIX rdfs: <{RDFS}>\n"
    f"PREFIX owl: <{OWL}>\n"
    f"PREFIX xsd: <{XSD}>\n"
)


def is_meta(iri: str) -> bool:
    return iri.startswith(META_NAMESPACES)


def ref(iri: str) -> str:
    if any(c in iri for c in "<>\"{}|^`\\ \n"):
        raise ValueError(f"not a safe IRI for a query: {iri!r}")
    return f"<{iri}>"


def values(var: str, iris: Iterable[str]) -> str:
    return f"VALUES ?{var} {{ " + " ".join(ref(i) for i in sorted(set(iris))) + " }"


def non_meta(var: str) -> str:
    """A FILTER expression: ``?var`` is outside the RDF, RDFS and OWL namespaces."""
    return " && ".join(f'!STRSTARTS(STR(?{var}), "{ns}")' for ns in META_NAMESPACES)


def not_data_node(var: str, layout: StoreLayout | None) -> str:
    """A FILTER that drops data nodes (see ``MEMBERSHIP_READING``); empty without a layout."""
    if layout is None:
        return ""
    graphs = values("dng", layout.graphs)
    return (
        f"FILTER NOT EXISTS {{ GRAPH {ref(layout.data_graph)} {{ ?{var} ?dnp ?dnv . "
        f"FILTER({non_meta('dnp')}) }} "
        f"FILTER NOT EXISTS {{ {graphs} GRAPH ?dng {{ ?dns ?{var} ?dno }} }} }}"
    )


def membership_label(layout: StoreLayout) -> str:
    """The OntologySource.membership value for a store: its membership convention."""
    return "rdfs:subClassOf" if layout.membership_predicate == RDFS_SUBCLASSOF else "rdf:type"
