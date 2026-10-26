"""Question templates for the UniProt Swiss-Prot subset (uniprot-sp-2026_03).

Two tiers per template: about a third **lookup** questions and two thirds
**compositional** ones (paths of 3-4 hops, grouping with HAVING or ORDER BY /
LIMIT, negation over joins, comparisons between two entities, intersections
and unions of two populations, distinct counts across a join). Calibration
round 2 raises each compositional variant one structural step (STEPS, chosen
by the delta knobs); ROUND_1_KNOBS reproduce the round-1 set.

How anchors are chosen: from the data, never from core.owl. Every candidate
query starts from instances and links that occur in the uniprot graph
(reviewed proteins, their organisms, annotations, keywords, GO terms,
catalysed Rhea reactions, structures) and keeps anchors by their counts in
the data (e.g. organisms with a given number of reviewed proteins, taxa whose
subtree holds at most 50 of them, keyword pairs that co-occur in one
organism). Nothing reads class or property axioms (domain, range,
disjointness, subproperty) from the core graph. Entities are named by a label
that is unique in the graph for their kind (checked per question), never by
an accession: a protein by its recommended name and its organism, a taxon by
its scientific name, a GO term, keyword or disease by its label, a Rhea
compound by its Rhea name.

Data facts the patterns rely on (checked on the store, recorded in the
manifest notes):

- The rdfs:subClassOf closure is written into the data for taxonomy,
  keywords, enzymes, pathways and locations, so plain rdfs:subClassOf already
  reaches every ancestor. The immediate parent of a taxon is given by
  ``?parent skos:narrowerTransitive ?child`` (one per taxon, despite the
  name). GO's rdfs:subClassOf holds the is-a closure; part-of parents appear
  only as restriction IRIs (``GO_x#part-of-parent-n``), so the GO patterns
  here follow is-a only.
- All entries in the subset are reviewed (up:reviewed true); the golds still
  state it, and every protein is typed up:Protein.
- up:catalyzedReaction points at Rhea master reactions, which the Rhea graph
  describes.
- QLever returns xsd:double values (structure resolution) typed xsd:decimal;
  the scorer compares numbers by value, so this does not change answers.
"""

from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from itertools import combinations
from typing import Any, Iterable

from harness.qgen.base import (
    Anchor,
    AnswerColumn,
    PoolRow,
    Profile,
    QuestionTemplate,
    StepCheck,
    StructuralStep,
    Variant,
    check_iri,
    usable_label,
)
from harness.qgen.readings import (
    Reading,
    compose,
    exclude_anchor,
    immediate_children_by_closure,
    immediate_children_via_link,
    membership_reading,
    restrict_subtree_to_anchor,
    scope_iri_to_closure,
    widen_answer_to_ancestors,
)
from harness.qgen.templates import _threshold_options

GRAPH_ID = "uniprot-sp-2026_03"
ID_PREFIX = "uniprotsp202603"

PREFIXES = (
    "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
    "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n"
    "PREFIX skos: <http://www.w3.org/2004/02/skos/core#>\n"
    "PREFIX up: <http://purl.uniprot.org/core/>\n"
    "PREFIX rh: <http://rdf.rhea-db.org/>\n"
)
GRAPHS = {
    "UNIPROT": "<http://sparql.uniprot.org/uniprot>",
    "TAXONOMY": "<http://sparql.uniprot.org/taxonomy>",
    "GO": "<http://sparql.uniprot.org/go>",
    "KEYWORDS": "<http://sparql.uniprot.org/keywords>",
    "DISEASES": "<http://sparql.uniprot.org/diseases>",
    "LOCATIONS": "<http://sparql.uniprot.org/locations>",
    "RHEA": "<https://sparql.rhea-db.org/rhea>",
}

# Difficulty knobs (recorded in the manifest; a given setting gives the same set).
KNOBS: dict[str, Any] = {
    # Organism-anchored populations: organisms with this many reviewed proteins.
    "org_min": 20,
    "org_max": 200,
    # Proteins named in lookup questions come from organisms of this size.
    "protein_org_min": 20,
    "protein_org_max": 49,
    # ORDER BY / LIMIT size and HAVING threshold.
    "top_k": 3,
    "having_min": 3,
    # At most this many pair options per anchor organism (pairs of proteins or keywords).
    "max_pair_options": 40,
    # Relative weights of variants within their tier (default 1 each), to shift the mix.
    "variant_weights": {},
    # Structural steps of the compositional tier (calibration round 2, design ruling on experiments#12):
    # 1 raises every compositional variant whose axis it is by one step (see STEPS below); 0 keeps the
    # round-1 form. The lookup tier has no steps.
    "path_hops_delta": 1,
    "grouping_constraints_delta": 1,
    "negation_depth_delta": 1,
    "populations_delta": 1,
    # Upper bound of the HAVING constraint the grouping step adds (one value for every grouping variant).
    "having_max": 10,
}
ROUND_DELTAS: dict[int, dict[str, int]] = {
    1: {"path_hops_delta": 0, "grouping_constraints_delta": 0, "negation_depth_delta": 0, "populations_delta": 0},
    2: {"path_hops_delta": 1, "grouping_constraints_delta": 1, "negation_depth_delta": 1, "populations_delta": 1},
}
# Round 1 (the frozen set b805acd9...) is these knobs; they reproduce it byte for byte.
ROUND_1_KNOBS: dict[str, Any] = {**KNOBS, **ROUND_DELTAS[1]}

IRI_COL, LIT_COL, NUM_COL = "iri", "literal", "number"
KW_PREFIX = "http://purl.uniprot.org/keywords/"
GO_PREFIX = "http://purl.obolibrary.org/obo/GO_"

# --------------------------------------------------------------------------
# Scope clauses and the alternative readings they rule out (harness.qgen.readings)

ORG_EXACT = "(recorded against that taxon exactly, not its strains or sub-taxa)"
KW_ITSELF = "the keyword itself; narrower keywords do not count"
KW_ASSIGNED = "only keywords assigned directly; broader keywords above them do not count"
LOC_NAMED = "only the locations named in the annotation; broader locations above them do not count"
LOC_COUNTED = "each location counts only proteins annotated with that location itself; narrower locations do not count"

R_ORGANISM_SUBTAXA = Reading(
    "organism_with_subtaxa",
    "the organism also covers its strains and sub-taxa",
    r"that taxon exactly|not its strains or sub-taxa",
    transform=scope_iri_to_closure("up:organism", "org", GRAPHS["TAXONOMY"], "scopeOrganism"),
)
R_TAXON_ONLY = Reading(
    "taxon_without_subtaxa",
    "the taxon means only itself, not the taxa below it",
    r"any taxon below it|all taxa below it",
    transform=restrict_subtree_to_anchor("taxon", GRAPHS["TAXONOMY"]),
)
R_NARROWER_KEYWORDS = Reading(
    "narrower_keywords",
    "a keyword also covers the narrower keywords below it",
    r"narrower keywords do not count",
    transform=compose(
        scope_iri_to_closure("up:classifiedWith", "kw_a", GRAPHS["KEYWORDS"], "scopeKeywordA"),
        scope_iri_to_closure("up:classifiedWith", "kw_b", GRAPHS["KEYWORDS"], "scopeKeywordB"),
    ),
)
R_BROADER_KEYWORDS = Reading(
    "broader_keywords",
    "an assigned keyword also implies the broader keywords above it",
    r"broader keywords above them do not count",
    transform=widen_answer_to_ancestors("up:classifiedWith", "keyword", GRAPHS["KEYWORDS"], "assignedKeyword"),
)
R_BROADER_LOCATIONS = Reading(
    "broader_locations",
    "a named location also implies the broader locations above it",
    r"broader locations above them do not count",
    transform=widen_answer_to_ancestors("up:cellularComponent", "location", GRAPHS["LOCATIONS"], "namedLocation"),
)
R_NARROWER_LOCATIONS = Reading(
    "narrower_locations",
    "a location's count also takes proteins annotated with narrower locations below it",
    r"narrower locations do not count",
    transform=widen_answer_to_ancestors("up:cellularComponent", "location", GRAPHS["LOCATIONS"], "namedLocation"),
)


def _go_with_part_of(sparql: str, params: dict) -> str | None:
    """Reading: "below a GO term" also follows part-of (one part-of step anywhere in the is-a chain)."""
    go = params.get("go")
    if not go:
        return None
    old = f"GRAPH {GRAPHS['GO']} {{ ?term rdfs:subClassOf* <{go}> . }}"
    if old not in sparql:
        return None
    # is-a from the annotated term up to a term that carries the part-of
    # restriction, then part-of, then is-a down to the anchor's subtree.
    new = (
        f"GRAPH {GRAPHS['GO']} {{ {{ ?term rdfs:subClassOf* <{go}> . }} UNION {{ ?term rdfs:subClassOf* ?isaAncestor . "
        "?isaAncestor rdfs:subClassOf ?partOfRestriction . "
        "?partOfRestriction <http://www.w3.org/2002/07/owl#onProperty> <http://purl.obolibrary.org/obo/BFO_0000050> ; "
        f"<http://www.w3.org/2002/07/owl#someValuesFrom> ?partOfParent . ?partOfParent rdfs:subClassOf* <{go}> . }} }}"
    )
    return sparql.replace(old, new)


R_GO_PART_OF = Reading("go_part_of", "below a GO term also by part-of, not only is-a", r"is-a hierarchy", transform=_go_with_part_of)

# The closure is materialised in plain rdfs:subClassOf, so the direct reading
# of "below" is spelled out: immediate children through the taxonomy's parent
# link (skos:narrowerTransitive), and through "no term in between" for GO. The
# anchor-inclusion reading leaves the anchor itself out.
_TAXON_IMMEDIATE = immediate_children_via_link("taxon", GRAPHS["TAXONOMY"])
_GO_IMMEDIATE = immediate_children_by_closure("go", GRAPHS["GO"])
_TAXON_WITHOUT_ANCHOR = exclude_anchor("taxon", GRAPHS["TAXONOMY"])
_GO_WITHOUT_ANCHOR = exclude_anchor("go", GRAPHS["GO"])


def _direct(clause: str, transform) -> Reading:
    return Reading("direct_vs_transitive", "only the anchor and its immediate children, not every depth", clause, transform=transform)


def _anchor_inclusion(clause: str, transform) -> Reading:
    return Reading("anchor_inclusion", "the anchor itself left out, only what lies below it", clause, transform=transform)

_ACTIVITY_LINK = re.compile(
    r"(\?\w+) a up:Catalytic_Activity_Annotation ; up:catalyticActivity \?activity \.\s*\?activity up:catalyzedReaction \?reaction \."
)


def _with_physiological_directions(sparql: str, params: dict) -> str | None:
    """Reading: a protein's reactions also include the direction-specific ones annotated as physiological."""
    out, n = _ACTIVITY_LINK.subn(
        lambda m: (
            f"{m.group(1)} a up:Catalytic_Activity_Annotation . "
            f"{{ {m.group(1)} up:catalyticActivity ?activity . ?activity up:catalyzedReaction ?reaction . }} "
            f"UNION {{ {m.group(1)} up:catalyzedPhysiologicalReaction ?reaction . }}"
        ),
        sparql,
    )
    return out if n else None


R_PHYSIOLOGICAL = Reading(
    "physiological_directions",
    "a protein's reactions also include the direction-specific forms annotated as its physiological direction",
    r"not their direction-specific forms",
    transform=_with_physiological_directions,
)
R_GENE_SYNONYMS = Reading(
    "gene_synonyms",
    "a gene's names also include its synonyms",
    r"primary name",
    transform=lambda sparql, params: (
        sparql.replace("skos:prefLabel ?geneName", "skos:prefLabel|skos:altLabel ?geneName")
        if "skos:prefLabel ?geneName" in sparql else None
    ),
)

# "Directly below" read as the whole subtree (the materialised rdfs:subClassOf closure).
_CHILDREN_ALTERNATIVE = """SELECT DISTINCT ?child WHERE {
  GRAPH $TAXONOMY { ?child rdfs:subClassOf <$taxon> . }
  MINUS {
    GRAPH $TAXONOMY { ?child rdfs:subClassOf <$taxon> . ?org rdfs:subClassOf* ?child . }
    GRAPH $UNIPROT { ?protein a up:Protein ; up:reviewed true ; up:organism ?org . }
  }
}"""
R_CHILD_WITHOUT_SUBTAXA = Reading(
    "child_without_subtaxa",
    "a child taxon counts as protein-less when no protein is recorded against it exactly, whatever lies below it",
    r"or against any taxon below them",
    pattern="""SELECT DISTINCT ?child WHERE {
  GRAPH $TAXONOMY { <$taxon> skos:narrowerTransitive ?child . }
  MINUS { GRAPH $UNIPROT { ?protein a up:Protein ; up:reviewed true ; up:organism ?child . } }
}""",
)
R_COUNTS_WITH_SUBTAXA = Reading(
    "counts_with_subtaxa",
    "each taxon's count also takes the proteins of the taxa below it",
    r"recorded directly against them|sub-taxa do not count",
    pattern="""SELECT ?org (COUNT(DISTINCT ?protein) AS ?proteins) WHERE {
  GRAPH $TAXONOMY { ?org rdfs:subClassOf* <$taxon> . ?below rdfs:subClassOf* ?org . }
  GRAPH $UNIPROT { ?protein a up:Protein ; up:reviewed true ; up:organism ?below . }
}
GROUP BY ?org
HAVING (COUNT(DISTINCT ?protein) >= $having_min)""",
)

PROTEIN_ANCHOR = Anchor(
    "protein",
    ("name", "organism"),
    "GRAPH $UNIPROT { ?e a up:Protein ; up:reviewed true ; up:recommendedName/up:fullName $name_lit ; up:organism ?eo . }\n"
    "GRAPH $TAXONOMY { ?eo rdfs:subClassOf* ?named . ?named up:scientificName $organism_lit . }",
    param="entity",
)


def _protein_anchor(prefix: str) -> Anchor:
    return Anchor(
        "protein",
        (f"{prefix}_name", "org_name"),
        f"GRAPH $UNIPROT {{ ?e a up:Protein ; up:reviewed true ; up:recommendedName/up:fullName ${prefix}_name_lit ; "
        "up:organism ?eo . }\nGRAPH $TAXONOMY { ?eo rdfs:subClassOf* ?named . ?named up:scientificName $org_name_lit . }",
        param=prefix,
    )


TAXON_ANCHOR = Anchor("taxon", ("taxon_name",), "GRAPH $TAXONOMY { ?e a up:Taxon ; up:scientificName $taxon_name_lit . }", "taxon")
ORG_ANCHOR = Anchor("taxon", ("org_name",), "GRAPH $TAXONOMY { ?e a up:Taxon ; up:scientificName $org_name_lit . }", "org")
GO_ANCHOR = Anchor("GO term", ("go_label",), "GRAPH $GO { ?e rdfs:label $go_label_lit . }", "go")
DISEASE_ANCHOR = Anchor("disease", ("disease_label",), "GRAPH $DISEASES { ?e skos:prefLabel $disease_label_lit . }", "disease")
COMPOUND_ANCHOR = Anchor(
    "Rhea compound", ("compound_name",), "GRAPH $RHEA { ?e rh:name $compound_name_lit ; rh:accession ?ea . }", "compound"
)


def _keyword_anchor(param: str) -> Anchor:
    return Anchor("keyword", (param,), f"GRAPH $KEYWORDS {{ ?e skos:prefLabel ${param}_lit . }}", param.removesuffix("_label"))


def uvariant(**kwargs: Any) -> Variant:
    """A Variant with the UniProt prefixes and graphs."""
    return Variant(prefixes=PREFIXES, graphs=GRAPHS, **kwargs)


# --------------------------------------------------------------------------
# Shared pieces of candidate queries

_SIZED_ORGANISMS = """
  {
    SELECT ?org WHERE { GRAPH $UNIPROT { ?sizedProtein a up:Protein ; up:reviewed true ; up:organism ?org . } }
    GROUP BY ?org
    HAVING (COUNT(?sizedProtein) >= $org_min && COUNT(?sizedProtein) <= $org_max)
  }
  GRAPH $TAXONOMY { ?org up:scientificName ?orgName . }
"""

_LOOKUP_PROTEINS = """
  {
    SELECT ?org WHERE { GRAPH $UNIPROT { ?sizedProtein a up:Protein ; up:reviewed true ; up:organism ?org . } }
    GROUP BY ?org
    HAVING (COUNT(?sizedProtein) >= $protein_org_min && COUNT(?sizedProtein) <= $protein_org_max)
  }
  GRAPH $TAXONOMY { ?org up:scientificName ?orgName . }
  GRAPH $UNIPROT { ?protein a up:Protein ; up:reviewed true ; up:organism ?org ; up:recommendedName/up:fullName ?name . }
"""

_PROTEIN_IN = 'GRAPH $UNIPROT { ?protein a up:Protein ; up:reviewed true ; up:organism <$org> . }'


# --------------------------------------------------------------------------
# Row -> parameter helpers


def _shape(row: PoolRow) -> dict[str, Any]:
    return {"_stratum": "multi" if int(row["n"]) > 1 else "single"} if row.get("n") not in (None, "") else {}


def _proteins(rows: list[PoolRow], knobs: Any = None) -> Iterable[dict[str, Any]]:
    for row in rows:
        if usable_label(row["name"]) and usable_label(row["orgName"]):
            yield {"entity": check_iri(row["protein"]), "name": row["name"], "organism": row["orgName"], **_shape(row)}


def _organisms(rows: list[PoolRow], knobs: Any = None) -> Iterable[dict[str, Any]]:
    for row in rows:
        if usable_label(row["orgName"]):
            yield {"org": check_iri(row["org"]), "org_name": row["orgName"], **_shape(row)}


def _taxa(rows: list[PoolRow], knobs: Any = None) -> Iterable[dict[str, Any]]:
    for row in rows:
        if usable_label(row["taxonName"]):
            yield {"taxon": check_iri(row["taxon"]), "taxon_name": row["taxonName"], **_shape(row)}


def _go_terms(rows: list[PoolRow], knobs: Any = None) -> Iterable[dict[str, Any]]:
    for row in rows:
        if usable_label(row["goLabel"]):
            yield {"go": check_iri(row["go"]), "go_label": row["goLabel"], **_shape(row)}


def _diseases(rows: list[PoolRow], knobs: Any = None) -> Iterable[dict[str, Any]]:
    for row in rows:
        if usable_label(row["diseaseLabel"]):
            yield {"disease": check_iri(row["disease"]), "disease_label": row["diseaseLabel"], **_shape(row)}


def _stable_order(items: Iterable[Any], salt: str) -> list[Any]:
    """A seed-independent but scrambled order, so a cap does not favour low identifiers."""
    return sorted(items, key=lambda x: hashlib.sha256(f"{salt}|{x}".encode()).hexdigest())


def _group(rows: list[PoolRow], key: str) -> "OrderedDict[str, list[PoolRow]]":
    out: "OrderedDict[str, list[PoolRow]]" = OrderedDict()
    for row in rows:
        out.setdefault(row[key], []).append(row)
    return out


def _taxon_go_pairs(rows: list[PoolRow], knobs: Any = None) -> Iterable[dict[str, Any]]:
    for row in rows:
        if usable_label(row["taxonName"]) and usable_label(row["goLabel"]):
            yield {
                "taxon": check_iri(row["taxon"]),
                "taxon_name": row["taxonName"],
                "go": check_iri(row["go"]),
                "go_label": row["goLabel"],
            }


def _keyword_pairs(rows: list[PoolRow], knobs: Any = None) -> Iterable[dict[str, Any]]:
    """Pairs of keywords used in one organism; one question per organism."""
    cap = int((knobs or KNOBS)["max_pair_options"])
    for org, group in _group(rows, "org").items():
        kws = {r["kw"]: r for r in group if usable_label(r["kwLabel"])}
        if len(kws) < 2 or not usable_label(group[0]["orgName"]):
            continue
        pairs = _stable_order(combinations(sorted(kws), 2), org)[:cap]
        for a, b in pairs:
            yield {
                "org": check_iri(org),
                "org_name": group[0]["orgName"],
                "kw_a": check_iri(a),
                "kw_a_label": kws[a]["kwLabel"],
                "kw_b": check_iri(b),
                "kw_b_label": kws[b]["kwLabel"],
            }


def _protein_pairs(rows: list[PoolRow], knobs: Any = None) -> Iterable[dict[str, Any]]:
    """Pairs of proteins of one organism; one question per organism."""
    cap = int((knobs or KNOBS)["max_pair_options"])
    for org, group in _group(rows, "org").items():
        prots = {r["protein"]: r for r in group if usable_label(r["name"])}
        if len(prots) < 2 or not usable_label(group[0]["orgName"]):
            continue
        for a, b in _stable_order(combinations(sorted(prots), 2), org)[:cap]:
            yield {
                "org": check_iri(org),
                "org_name": group[0]["orgName"],
                "a": check_iri(a),
                "a_name": prots[a]["name"],
                "b": check_iri(b),
                "b_name": prots[b]["name"],
            }


def _protein_pairs_differing(field: str):
    """Protein pairs whose ``field`` values differ (so a comparison has one answer)."""

    def expand(rows: list[PoolRow], knobs: Any = None) -> Iterable[dict[str, Any]]:
        values = {r["protein"]: r[field] for r in rows}
        for params in _protein_pairs(rows, knobs):
            if values[params["a"]] != values[params["b"]]:
                yield params

    return expand


RESOLUTION_COMPARISONS = (("<", "better (numerically lower) than {v} Å"), (">", "worse (numerically higher) than {v} Å"))
MASS_COMPARISONS = ((">", "above {v} Da"), ("<", "below {v} Da"))


def _options(values: list[float], comparisons: tuple, lexical: dict[float, str], max_rows: int = 50) -> list[dict[str, Any]]:
    """Thresholds at existing values that keep a proper, non-empty subset of at most ``max_rows``."""
    out = []
    for v in sorted(set(values)):
        for op, wording in comparisons:
            n = sum(1 for x in values if (x < v if op == "<" else x > v))
            if 1 <= n <= max_rows and n < len(values):
                out.append({"op": op, "threshold": lexical[v], "comparison": wording.format(v=lexical[v])})
    return out


def _structure_thresholds(rows: list[PoolRow], knobs: Any = None) -> Iterable[dict[str, Any]]:
    for protein, group in _group(rows, "protein").items():
        first = group[0]
        if not (usable_label(first["name"]) and usable_label(first["orgName"])):
            continue
        lexical = {float(r["resolution"]): r["resolution"] for r in group}
        values = [float(r["resolution"]) for r in group]
        base = {"entity": check_iri(protein), "name": first["name"], "organism": first["orgName"]}
        for option in _options(values, RESOLUTION_COMPARISONS, lexical):
            yield {**base, **option}


def _mass_thresholds(rows: list[PoolRow], knobs: Any = None) -> Iterable[dict[str, Any]]:
    """Per organism: heavier-than uses each protein's heaviest sequence, lighter-than its lightest."""
    for org, group in _group(rows, "org").items():
        if not usable_label(group[0]["orgName"]):
            continue
        heaviest = [int(r["maxMass"]) for r in group]
        lightest = [int(r["minMass"]) for r in group]
        base = {"org": check_iri(org), "org_name": group[0]["orgName"]}
        for option in _options(heaviest, MASS_COMPARISONS[:1], {v: str(v) for v in heaviest}):
            yield {**base, **option}
        for option in _options(lightest, MASS_COMPARISONS[1:], {v: str(v) for v in lightest}):
            yield {**base, **option}


def _reference_proteins(rows: list[PoolRow], knobs: Any = None) -> Iterable[dict[str, Any]]:
    """Per organism, reference proteins that some but not all others outweigh."""
    for org, group in _group(rows, "org").items():
        if not usable_label(group[0]["orgName"]):
            continue
        masses = sorted(int(r["maxMass"]) for r in group)
        for r in group:
            heavier = sum(1 for m in masses if m > int(r["maxMass"]))
            if 1 <= heavier <= 50 and usable_label(r["name"]):
                yield {"org": check_iri(org), "org_name": r["orgName"], "ref": check_iri(r["protein"]), "ref_name": r["name"]}


def _org_compounds(rows: list[PoolRow], knobs: Any = None) -> Iterable[dict[str, Any]]:
    for row in rows:
        if usable_label(row["orgName"]) and usable_label(row["compoundName"]):
            yield {
                "org": check_iri(row["org"]),
                "org_name": row["orgName"],
                "compound": check_iri(row["compound"]),
                "compound_name": row["compoundName"],
            }


# --------------------------------------------------------------------------
# 1. type_members


class TypeMembers(QuestionTemplate):
    id = "type_members"
    description = (
        "Members of a type: proteins of a taxon or of a GO term (at any depth below it), and in the compositional "
        "tier the intersection of two such populations or the union of two keyword populations."
    )
    membership_mode = "transitive"
    variants = (
        uvariant(
            name="taxon_proteins",
            description="Reviewed proteins of a taxon or any taxon below it (subtree of at most 50 proteins, 2+ organisms).",
            candidate_query="""
SELECT ?taxon ?taxonName WHERE {
  {
    SELECT ?taxon (COUNT(DISTINCT ?protein) AS ?n) (COUNT(DISTINCT ?org) AS ?orgs) WHERE {
      GRAPH $UNIPROT { ?protein a up:Protein ; up:reviewed true ; up:organism ?org . }
      GRAPH $TAXONOMY { ?org rdfs:subClassOf ?taxon . }
    }
    GROUP BY ?taxon
    HAVING (COUNT(DISTINCT ?protein) <= 50 && COUNT(DISTINCT ?org) >= 2)
  }
  GRAPH $TAXONOMY { ?taxon up:scientificName ?taxonName . }
}
ORDER BY ?taxon
""",
            gold_pattern="""SELECT DISTINCT ?protein WHERE {
  GRAPH $TAXONOMY { ?org rdfs:subClassOf* <$taxon> . }
  GRAPH $UNIPROT { ?protein a up:Protein ; up:reviewed true ; up:organism ?org . }
}""",
            question_pattern="Which reviewed proteins come from the taxon \"$taxon_name\" or from any taxon below it? List each protein once.",
            readings=(
                R_TAXON_ONLY,
                _direct(r"or from any taxon below it", _TAXON_IMMEDIATE),
                _anchor_inclusion(r'taxon "[^"]+" or from any taxon below it', _TAXON_WITHOUT_ANCHOR),
            ),
            answer_columns=(AnswerColumn("protein", IRI_COL),),
            key_param="taxon",
            expand=_taxa,
            anchors=(TAXON_ANCHOR,),
            difficulty={"hops": 2, "anchors": 1, "membership": "transitive"},
        ),
        uvariant(
            name="go_term_proteins",
            description="Reviewed proteins annotated with a GO term or any is-a descendant (at most 50).",
            candidate_query="""
SELECT ?go ?goLabel WHERE {
  {
    SELECT ?go (COUNT(DISTINCT ?protein) AS ?n) (COUNT(DISTINCT ?child) AS ?terms) WHERE {
      GRAPH $GO { ?child rdfs:subClassOf ?go . }
      GRAPH $UNIPROT { ?protein up:classifiedWith ?child . }
    }
    GROUP BY ?go
    HAVING (COUNT(DISTINCT ?protein) <= 50)
  }
  GRAPH $GO { ?go rdfs:label ?goLabel . }
}
ORDER BY ?go
""",
            gold_pattern="""SELECT DISTINCT ?protein WHERE {
  GRAPH $GO { ?term rdfs:subClassOf* <$go> . }
  GRAPH $UNIPROT { ?protein a up:Protein ; up:reviewed true ; up:classifiedWith ?term . }
}""",
            question_pattern="Which reviewed proteins are annotated with the Gene Ontology term \"$go_label\" or with any Gene Ontology term below it in the is-a hierarchy? List each protein once.",
            readings=(
                R_GO_PART_OF,
                _direct(r"any Gene Ontology term below it", _GO_IMMEDIATE),
                _anchor_inclusion(r'Gene Ontology term "[^"]+" or with any Gene Ontology term below it', _GO_WITHOUT_ANCHOR),
            ),
            answer_columns=(AnswerColumn("protein", IRI_COL),),
            key_param="go",
            expand=_go_terms,
            anchors=(GO_ANCHOR,),
            difficulty={"hops": 2, "anchors": 1, "membership": "transitive"},
        ),
        uvariant(
            name="taxon_go_intersection",
            tier="compositional",
            description=(
                "Reviewed proteins in a taxon's subtree that also carry a GO term or one of its is-a descendants "
                "(intersection of two transitive populations; each alone exceeds 50)."
            ),
            candidate_query="""
SELECT ?taxon ?taxonName ?go ?goLabel WHERE {
  {
    SELECT ?taxon ?go (COUNT(DISTINCT ?protein) AS ?n) WHERE {
      {
        SELECT ?taxon WHERE {
          GRAPH $UNIPROT { ?p a up:Protein ; up:organism ?o . }
          GRAPH $TAXONOMY { ?o rdfs:subClassOf ?taxon . ?taxon up:rank up:Taxonomic_Rank_Genus . }
        }
        GROUP BY ?taxon
        HAVING (COUNT(DISTINCT ?p) > 50 && COUNT(DISTINCT ?p) <= 1000)
      }
      GRAPH $TAXONOMY { ?org rdfs:subClassOf ?taxon . }
      GRAPH $UNIPROT { ?protein a up:Protein ; up:organism ?org ; up:classifiedWith ?go . }
      FILTER(STRSTARTS(STR(?go), "http://purl.obolibrary.org/obo/GO_"))
    }
    GROUP BY ?taxon ?go
    HAVING (COUNT(DISTINCT ?protein) >= 2 && COUNT(DISTINCT ?protein) <= 40)
  }
  GRAPH $TAXONOMY { ?taxon up:scientificName ?taxonName . }
  GRAPH $GO { ?go rdfs:label ?goLabel . }
}
ORDER BY ?taxon ?go
""",
            gold_pattern="""SELECT DISTINCT ?protein WHERE {
  GRAPH $TAXONOMY { ?org rdfs:subClassOf* <$taxon> . }
  GRAPH $GO { ?term rdfs:subClassOf* <$go> . }
  GRAPH $UNIPROT { ?protein a up:Protein ; up:reviewed true ; up:organism ?org ; up:classifiedWith ?term . }
}""",
            question_pattern="Which reviewed proteins come from the taxon \"$taxon_name\" (or any taxon below it) and are annotated with the Gene Ontology term \"$go_label\" (or any Gene Ontology term below it in the is-a hierarchy)? List each protein once.",
            readings=(
                R_TAXON_ONLY,
                R_GO_PART_OF,
                _direct(r"any taxon below it.*any Gene Ontology term below it", compose(_TAXON_IMMEDIATE, _GO_IMMEDIATE)),
                _anchor_inclusion(
                    r'taxon "[^"]+" \(or any taxon below it\).*Gene Ontology term "[^"]+" \(or any Gene Ontology term below it',
                    compose(_TAXON_WITHOUT_ANCHOR, _GO_WITHOUT_ANCHOR),
                ),
            ),
            answer_columns=(AnswerColumn("protein", IRI_COL),),
            key_param="taxon",
            expand=_taxon_go_pairs,
            anchors=(TAXON_ANCHOR, GO_ANCHOR),
            difficulty={"hops": 3, "anchors": 2, "set_operation": "intersection", "membership": "transitive"},
        ),
        uvariant(
            name="organism_keyword_union",
            tier="compositional",
            membership_mode="n.a.",
            description="Reviewed proteins of one organism carrying either of two keywords (union of two populations).",
            candidate_query="""
SELECT ?org ?orgName ?kw ?kwLabel WHERE {
""" + _SIZED_ORGANISMS + """
  GRAPH $UNIPROT { ?protein a up:Protein ; up:organism ?org ; up:classifiedWith ?kw . }
  FILTER(STRSTARTS(STR(?kw), "http://purl.uniprot.org/keywords/"))
  GRAPH $KEYWORDS { ?kw skos:prefLabel ?kwLabel . }
}
GROUP BY ?org ?orgName ?kw ?kwLabel
HAVING (COUNT(DISTINCT ?protein) >= 2 && COUNT(DISTINCT ?protein) <= 25)
ORDER BY ?org ?kw
""",
            gold_pattern="""SELECT DISTINCT ?protein WHERE {
  GRAPH $UNIPROT { ?protein a up:Protein ; up:reviewed true ; up:organism <$org> . }
  { GRAPH $UNIPROT { ?protein up:classifiedWith <$kw_a> . } }
  UNION
  { GRAPH $UNIPROT { ?protein up:classifiedWith <$kw_b> . } }
}""",
            question_pattern="Which reviewed proteins of the organism \"$org_name\" (recorded against that taxon exactly, not its strains or sub-taxa) carry the keyword annotation \"$kw_a_label\" or the keyword annotation \"$kw_b_label\" (or both; only the keywords themselves, narrower keywords do not count)? List each protein once.",
            readings=(R_ORGANISM_SUBTAXA, R_NARROWER_KEYWORDS,),
            answer_columns=(AnswerColumn("protein", IRI_COL),),
            key_param="org",
            expand=_keyword_pairs,
            anchors=(ORG_ANCHOR, _keyword_anchor("kw_a_label"), _keyword_anchor("kw_b_label")),
            difficulty={"hops": 1, "anchors": 3, "set_operation": "union"},
        ),
    )


# --------------------------------------------------------------------------
# 2. one_hop_attribute


class OneHopAttribute(QuestionTemplate):
    id = "one_hop_attribute"
    description = (
        "An attribute one hop from a protein (its keywords, its creation date); compositional: the keywords two "
        "proteins share, and which of two proteins entered UniProtKB first."
    )
    membership_mode = "n.a."
    variants = (
        uvariant(
            name="protein_keywords",
            description="UniProt keywords of a protein; proteins with one and with several keywords alternate.",
            candidate_query="""
SELECT ?protein ?name ?orgName (COUNT(DISTINCT ?kw) AS ?n) WHERE {
""" + _LOOKUP_PROTEINS + """
  GRAPH $UNIPROT { ?protein up:classifiedWith ?kw . }
  FILTER(STRSTARTS(STR(?kw), "http://purl.uniprot.org/keywords/"))
}
GROUP BY ?protein ?name ?orgName
ORDER BY ?protein
""",
            gold_pattern="""SELECT DISTINCT ?keyword WHERE {
  GRAPH $UNIPROT { <$entity> up:classifiedWith ?keyword . }
  FILTER(STRSTARTS(STR(?keyword), "http://purl.uniprot.org/keywords/"))
}""",
            question_pattern="Which keyword annotations does the protein \"$name\" of \"$organism\" carry (only keywords assigned directly; broader keywords above them do not count)? List each keyword once.",
            readings=(R_BROADER_KEYWORDS,),
            answer_columns=(AnswerColumn("keyword", IRI_COL),),
            key_param="entity",
            expand=_proteins,
            hops=1,
            anchors=(PROTEIN_ANCHOR,),
            difficulty={"hops": 1, "anchors": 1},
        ),
        uvariant(
            name="protein_created",
            description="The date a protein entry was created.",
            candidate_query="""
SELECT ?protein ?name ?orgName WHERE {
""" + _LOOKUP_PROTEINS + """
}
ORDER BY ?protein
""",
            gold_pattern="""SELECT ?created WHERE {
  GRAPH $UNIPROT { <$entity> up:created ?created . }
}""",
            question_pattern="On what date was the entry for the protein \"$name\" of \"$organism\" created?",
            answer_columns=(AnswerColumn("created", LIT_COL),),
            key_param="entity",
            expand=_proteins,
            hops=1,
            anchors=(PROTEIN_ANCHOR,),
            difficulty={"hops": 1, "anchors": 1},
        ),
        uvariant(
            name="shared_keywords",
            tier="compositional",
            description="Keywords that two proteins of one organism both carry (intersection of two attribute sets).",
            candidate_query="""
SELECT ?org ?orgName ?protein ?name WHERE {
""" + _SIZED_ORGANISMS + """
  GRAPH $UNIPROT {
    ?protein a up:Protein ; up:organism ?org ; up:recommendedName/up:fullName ?name ; up:classifiedWith ?kw .
  }
  FILTER(STRSTARTS(STR(?kw), "http://purl.uniprot.org/keywords/"))
}
GROUP BY ?org ?orgName ?protein ?name
HAVING (COUNT(DISTINCT ?kw) >= 3)
ORDER BY ?org ?protein
""",
            gold_pattern="""SELECT DISTINCT ?keyword WHERE {
  GRAPH $UNIPROT { <$a> up:classifiedWith ?keyword . <$b> up:classifiedWith ?keyword . }
  FILTER(STRSTARTS(STR(?keyword), "http://purl.uniprot.org/keywords/"))
}""",
            question_pattern="Which keyword annotations do the proteins \"$a_name\" and \"$b_name\", both of \"$org_name\", have in common (only keywords assigned directly; broader keywords above them do not count)? List each keyword once.",
            readings=(R_BROADER_KEYWORDS,),
            answer_columns=(AnswerColumn("keyword", IRI_COL),),
            key_param="org",
            expand=_protein_pairs,
            hops=1,
            anchors=(_protein_anchor("a"), _protein_anchor("b")),
            difficulty={"hops": 2, "anchors": 2, "set_operation": "intersection"},
        ),
        uvariant(
            name="earlier_entry",
            tier="compositional",
            description="Which of two proteins of one organism entered UniProtKB first (comparison of two entities).",
            candidate_query="""
SELECT ?org ?orgName ?protein ?name ?created WHERE {
""" + _SIZED_ORGANISMS + """
  GRAPH $UNIPROT { ?protein a up:Protein ; up:organism ?org ; up:recommendedName/up:fullName ?name ; up:created ?created . }
}
ORDER BY ?org ?protein
""",
            gold_pattern="""SELECT ?protein ?created WHERE {
  GRAPH $UNIPROT { VALUES ?protein { <$a> <$b> } ?protein a up:Protein ; up:reviewed true ; up:created ?created . }
}
ORDER BY ?created
LIMIT 1""",
            question_pattern="Which of the two proteins \"$a_name\" and \"$b_name\", both of \"$org_name\", has the earlier entry creation date? Give that protein with the date its entry was created.",
            answer_columns=(AnswerColumn("protein", IRI_COL), AnswerColumn("created", LIT_COL)),
            key_param="org",
            expand=_protein_pairs_differing("created"),
            hops=1,
            anchors=(_protein_anchor("a"), _protein_anchor("b")),
            difficulty={"hops": 1, "anchors": 2, "comparison": True, "aggregate": "order_limit"},
        ),
    )


# --------------------------------------------------------------------------
# 3. multi_hop_path


class MultiHopPath(QuestionTemplate):
    id = "multi_hop_path"
    description = (
        "Paths: protein -> gene -> name (2), protein -> annotation -> location -> component (3); compositional: "
        "organism -> protein -> annotation -> activity -> Rhea reaction (4), disease -> annotation -> protein -> "
        "gene -> name (4)."
    )
    membership_mode = "n.a."
    variants = (
        uvariant(
            name="protein_gene_names",
            description="Primary names of the genes encoding a protein; one and several alternate.",
            candidate_query="""
SELECT ?protein ?name ?orgName (COUNT(DISTINCT ?geneName) AS ?n) WHERE {
""" + _LOOKUP_PROTEINS + """
  GRAPH $UNIPROT { ?protein up:encodedBy ?gene . ?gene skos:prefLabel ?geneName . }
}
GROUP BY ?protein ?name ?orgName
ORDER BY ?protein
""",
            gold_pattern="""SELECT DISTINCT ?geneName WHERE {
  GRAPH $UNIPROT { <$entity> up:encodedBy ?gene . ?gene skos:prefLabel ?geneName . }
}""",
            question_pattern="Which genes encode the protein \"$name\" of \"$organism\"? Give the primary name of each gene, each name once.",
            readings=(R_GENE_SYNONYMS,),
            answer_columns=(AnswerColumn("gene_name", LIT_COL),),
            key_param="entity",
            expand=_proteins,
            hops=2,
            anchors=(PROTEIN_ANCHOR,),
            difficulty={"hops": 2, "anchors": 1},
        ),
        uvariant(
            name="protein_locations",
            description="Subcellular locations of a protein (protein -> annotation -> located-in -> component).",
            candidate_query="""
SELECT ?protein ?name ?orgName (COUNT(DISTINCT ?location) AS ?n) WHERE {
""" + _LOOKUP_PROTEINS + """
  GRAPH $UNIPROT {
    ?protein up:annotation ?annotation .
    ?annotation a up:Subcellular_Location_Annotation ; up:locatedIn ?located .
    ?located up:cellularComponent ?location .
  }
}
GROUP BY ?protein ?name ?orgName
ORDER BY ?protein
""",
            gold_pattern="""SELECT DISTINCT ?location WHERE {
  GRAPH $UNIPROT {
    <$entity> up:annotation ?annotation .
    ?annotation a up:Subcellular_Location_Annotation ; up:locatedIn ?located .
    ?located up:cellularComponent ?location .
  }
}""",
            question_pattern="In which subcellular locations is the protein \"$name\" of \"$organism\" found, according to its subcellular-location annotation (only the locations named in the annotation; broader locations above them do not count)? List each location once.",
            readings=(R_BROADER_LOCATIONS,),
            answer_columns=(AnswerColumn("location", IRI_COL),),
            key_param="entity",
            expand=_proteins,
            hops=3,
            anchors=(PROTEIN_ANCHOR,),
            difficulty={"hops": 3, "anchors": 1},
        ),
        uvariant(
            name="organism_reactions",
            tier="compositional",
            description="Rhea reactions catalysed by reviewed proteins of an organism (4 hops from the organism).",
            candidate_query="""
SELECT ?org ?orgName (COUNT(DISTINCT ?reaction) AS ?n) WHERE {
""" + _SIZED_ORGANISMS + """
  GRAPH $UNIPROT {
    ?protein a up:Protein ; up:organism ?org ; up:annotation ?annotation .
    ?annotation a up:Catalytic_Activity_Annotation ; up:catalyticActivity ?activity .
    ?activity up:catalyzedReaction ?reaction .
  }
}
GROUP BY ?org ?orgName
HAVING (COUNT(DISTINCT ?reaction) <= 50)
ORDER BY ?org
""",
            gold_pattern="""SELECT DISTINCT ?reaction WHERE {
  GRAPH $UNIPROT {
    ?protein a up:Protein ; up:reviewed true ; up:organism <$org> ; up:annotation ?annotation .
    ?annotation a up:Catalytic_Activity_Annotation ; up:catalyticActivity ?activity .
    ?activity up:catalyzedReaction ?reaction .
  }
}""",
            question_pattern="Which reactions are catalysed by reviewed proteins of the organism \"$org_name\" (recorded against that taxon exactly, not its strains or sub-taxa)? List each reaction once (the reactions as annotated, not their direction-specific forms).",
            readings=(R_ORGANISM_SUBTAXA, R_PHYSIOLOGICAL,),
            answer_columns=(AnswerColumn("reaction", IRI_COL),),
            key_param="org",
            expand=_organisms,
            hops=4,
            anchors=(ORG_ANCHOR,),
            difficulty={"hops": 4, "anchors": 1},
        ),
        uvariant(
            name="disease_genes",
            tier="compositional",
            description="Proteins linked to a disease with the primary names of their genes (4 hops from the disease).",
            candidate_query="""
SELECT ?disease ?diseaseLabel (COUNT(DISTINCT ?protein) AS ?n) WHERE {
  GRAPH $UNIPROT {
    ?protein a up:Protein ; up:annotation ?annotation ; up:encodedBy ?gene .
    ?annotation a up:Disease_Annotation ; up:disease ?disease .
    ?gene skos:prefLabel ?geneName .
  }
  GRAPH $DISEASES { ?disease skos:prefLabel ?diseaseLabel . }
}
GROUP BY ?disease ?diseaseLabel
HAVING (COUNT(DISTINCT ?protein) <= 50)
ORDER BY ?disease
""",
            gold_pattern="""SELECT DISTINCT ?protein ?geneName WHERE {
  GRAPH $UNIPROT {
    ?protein a up:Protein ; up:reviewed true ; up:annotation ?annotation ; up:encodedBy ?gene .
    ?annotation a up:Disease_Annotation ; up:disease <$disease> .
    ?gene skos:prefLabel ?geneName .
  }
}""",
            question_pattern="Which reviewed proteins are linked to the disease \"$disease_label\"? List each protein with the primary name of a gene that encodes it.",
            readings=(R_GENE_SYNONYMS,),
            answer_columns=(AnswerColumn("protein", IRI_COL), AnswerColumn("gene_name", LIT_COL)),
            key_param="disease",
            expand=_diseases,
            hops=4,
            anchors=(DISEASE_ANCHOR,),
            difficulty={"hops": 4, "anchors": 1},
        ),
    )


# --------------------------------------------------------------------------
# 4. aggregation


class Aggregation(QuestionTemplate):
    id = "aggregation"
    description = (
        "Counts: reviewed proteins of an organism, transmembrane regions of a protein; compositional: top-k "
        "keywords of an organism (ORDER BY / LIMIT, alphabetical tie-break), organisms of a taxon with at least n "
        "proteins (GROUP BY / HAVING), distinct reactions across a join."
    )
    membership_mode = "n.a."
    variants = (
        uvariant(
            name="organism_protein_count",
            description="Number of reviewed proteins recorded against exactly one organism taxon.",
            candidate_query="""
SELECT ?org ?orgName WHERE {
  { SELECT DISTINCT ?org WHERE { GRAPH $UNIPROT { ?p a up:Protein ; up:organism ?org . } } }
  GRAPH $TAXONOMY { ?org up:scientificName ?orgName . }
}
ORDER BY ?org
""",
            gold_pattern="""SELECT (COUNT(DISTINCT ?protein) AS ?count) WHERE {
  GRAPH $UNIPROT { ?protein a up:Protein ; up:reviewed true ; up:organism <$org> . }
}""",
            question_pattern="How many reviewed proteins does the organism \"$org_name\" have (recorded against that taxon exactly, not its strains or sub-taxa)?",
            readings=(R_ORGANISM_SUBTAXA,),
            answer_columns=(AnswerColumn("count", NUM_COL),),
            key_param="org",
            expand=_organisms,
            anchors=(ORG_ANCHOR,),
            difficulty={"hops": 1, "anchors": 1, "aggregate": "count"},
        ),
        uvariant(
            name="protein_transmembrane_count",
            description="Number of transmembrane regions annotated on a protein.",
            candidate_query="""
SELECT DISTINCT ?protein ?name ?orgName WHERE {
""" + _LOOKUP_PROTEINS + """
  GRAPH $UNIPROT { ?protein up:annotation ?annotation . ?annotation a up:Transmembrane_Annotation . }
}
ORDER BY ?protein
""",
            gold_pattern="""SELECT (COUNT(DISTINCT ?annotation) AS ?count) WHERE {
  GRAPH $UNIPROT { <$entity> up:annotation ?annotation . ?annotation a up:Transmembrane_Annotation . }
}""",
            question_pattern="How many transmembrane regions are annotated on the protein \"$name\" of \"$organism\"?",
            answer_columns=(AnswerColumn("count", NUM_COL),),
            key_param="entity",
            expand=_proteins,
            anchors=(PROTEIN_ANCHOR,),
            difficulty={"hops": 2, "anchors": 1, "aggregate": "count"},
        ),
        uvariant(
            name="organism_top_keywords",
            tier="compositional",
            description="The k keywords on the most reviewed proteins of an organism, ties broken alphabetically.",
            candidate_query="""
SELECT ?org ?orgName WHERE {
""" + _SIZED_ORGANISMS + """
}
ORDER BY ?org
""",
            gold_pattern="""SELECT ?keyword ?proteins WHERE {
  {
    SELECT ?keyword ?label (COUNT(DISTINCT ?protein) AS ?proteins) WHERE {
      GRAPH $UNIPROT { ?protein a up:Protein ; up:reviewed true ; up:organism <$org> ; up:classifiedWith ?keyword . }
      GRAPH $KEYWORDS { ?keyword skos:prefLabel ?label . }
    }
    GROUP BY ?keyword ?label
  }
}
ORDER BY DESC(?proteins) ?label
LIMIT $top_k""",
            question_pattern="Which $top_k keyword annotations are carried by the most reviewed proteins of the organism \"$org_name\" (recorded against that taxon exactly, not its strains or sub-taxa)? Count only keywords assigned to the proteins directly; broader keywords above them do not count. List each keyword with its number of proteins; when keywords tie, take them in alphabetical order of their names.",
            readings=(R_ORGANISM_SUBTAXA, R_BROADER_KEYWORDS,),
            answer_columns=(AnswerColumn("keyword", IRI_COL), AnswerColumn("proteins", NUM_COL)),
            key_param="org",
            expand=_organisms,
            anchors=(ORG_ANCHOR,),
            knob_params=("top_k",),
            difficulty={"hops": 2, "anchors": 1, "aggregate": "group_order_limit", "tie_break": "label"},
        ),
        uvariant(
            name="taxon_organisms_having",
            tier="compositional",
            membership_mode="transitive",
            description="Organisms in a taxon's subtree with at least n reviewed proteins each (GROUP BY / HAVING).",
            candidate_query="""
SELECT ?taxon ?taxonName WHERE {
  {
    SELECT ?taxon WHERE {
      GRAPH $UNIPROT { ?protein a up:Protein ; up:organism ?org . }
      GRAPH $TAXONOMY { ?org rdfs:subClassOf ?taxon . ?taxon up:rank ?rank . }
      VALUES ?rank { up:Taxonomic_Rank_Genus up:Taxonomic_Rank_Family }
    }
    GROUP BY ?taxon
    HAVING (COUNT(DISTINCT ?org) >= 3 && COUNT(DISTINCT ?protein) <= 3000)
  }
  GRAPH $TAXONOMY { ?taxon up:scientificName ?taxonName . }
}
ORDER BY ?taxon
""",
            gold_pattern="""SELECT ?org (COUNT(DISTINCT ?protein) AS ?proteins) WHERE {
  GRAPH $TAXONOMY { ?org rdfs:subClassOf* <$taxon> . }
  GRAPH $UNIPROT { ?protein a up:Protein ; up:reviewed true ; up:organism ?org . }
}
GROUP BY ?org
HAVING (COUNT(DISTINCT ?protein) >= $having_min)""",
            question_pattern="Among the taxon \"$taxon_name\" and all taxa below it, which have at least $having_min reviewed proteins recorded directly against them (proteins of their own sub-taxa do not count)? List each taxon with its number of reviewed proteins.",
            readings=(
                _direct(r"and all taxa below it", _TAXON_IMMEDIATE),
                _anchor_inclusion(r'taxon "[^"]+" and all taxa below it', _TAXON_WITHOUT_ANCHOR),
                R_COUNTS_WITH_SUBTAXA,
            ),
            answer_columns=(AnswerColumn("taxon", IRI_COL), AnswerColumn("proteins", NUM_COL)),
            key_param="taxon",
            expand=_taxa,
            anchors=(TAXON_ANCHOR,),
            knob_params=("having_min",),
            difficulty={"hops": 2, "anchors": 1, "aggregate": "group_having", "membership": "transitive"},
        ),
        uvariant(
            name="organism_distinct_reactions",
            tier="compositional",
            description="Number of distinct Rhea reactions catalysed by reviewed proteins of an organism (distinct count across a join).",
            candidate_query="""
SELECT ?org ?orgName WHERE {
""" + _SIZED_ORGANISMS + """
  FILTER EXISTS {
    GRAPH $UNIPROT { ?p up:organism ?org ; up:annotation ?a . ?a up:catalyticActivity ?ca . ?ca up:catalyzedReaction ?r . }
  }
}
ORDER BY ?org
""",
            gold_pattern="""SELECT (COUNT(DISTINCT ?reaction) AS ?count) WHERE {
  GRAPH $UNIPROT {
    ?protein a up:Protein ; up:reviewed true ; up:organism <$org> ; up:annotation ?annotation .
    ?annotation a up:Catalytic_Activity_Annotation ; up:catalyticActivity ?activity .
    ?activity up:catalyzedReaction ?reaction .
  }
}""",
            question_pattern="How many different reactions are catalysed by the reviewed proteins of the organism \"$org_name\" (recorded against that taxon exactly, not its strains or sub-taxa)? Count each reaction once, however many proteins catalyse it (the reactions as annotated, not their direction-specific forms).",
            readings=(R_ORGANISM_SUBTAXA, R_PHYSIOLOGICAL,),
            answer_columns=(AnswerColumn("count", NUM_COL),),
            key_param="org",
            expand=_organisms,
            anchors=(ORG_ANCHOR,),
            difficulty={"hops": 4, "anchors": 1, "aggregate": "count_distinct_join"},
        ),
    )


# --------------------------------------------------------------------------
# 5. numeric_filter


class NumericFilter(QuestionTemplate):
    id = "numeric_filter"
    description = (
        "Numeric comparisons: structure resolution of a protein's structures, sequence masses of an organism's "
        "proteins; compositional: which of two proteins has the best-resolved structure, and proteins heavier than "
        "a reference protein."
    )
    membership_mode = "n.a."
    variants = (
        uvariant(
            name="structures_by_resolution",
            description="Experimental structures of a protein better or worse than a resolution threshold.",
            candidate_query="""
SELECT ?protein ?name ?orgName ?resolution WHERE {
  {
    SELECT ?protein WHERE {
      GRAPH $UNIPROT { ?protein a up:Protein ; rdfs:seeAlso ?s . ?s a up:Structure_Resource ; up:resolution ?r . }
    }
    GROUP BY ?protein
    HAVING (COUNT(DISTINCT ?s) >= 2 && COUNT(DISTINCT ?s) <= 60)
  }
  GRAPH $UNIPROT {
    ?protein up:recommendedName/up:fullName ?name ; up:organism ?org ; rdfs:seeAlso ?structure .
    ?structure a up:Structure_Resource ; up:resolution ?resolution .
  }
  GRAPH $TAXONOMY { ?org up:scientificName ?orgName . }
}
GROUP BY ?protein ?name ?orgName ?resolution
ORDER BY ?protein ?resolution
""",
            gold_pattern="""SELECT DISTINCT ?structure ?resolution WHERE {
  GRAPH $UNIPROT { <$entity> rdfs:seeAlso ?structure . ?structure a up:Structure_Resource ; up:resolution ?resolution . }
  FILTER(?resolution $op $threshold)
}""",
            question_pattern="Which experimental structures of the protein \"$name\" of \"$organism\" have a resolution $comparison? List each structure with its resolution.",
            answer_columns=(AnswerColumn("structure", IRI_COL), AnswerColumn("resolution", NUM_COL)),
            key_param="entity",
            expand=_structure_thresholds,
            anchors=(PROTEIN_ANCHOR,),
            difficulty={"hops": 2, "anchors": 1, "comparison": "threshold"},
        ),
        uvariant(
            name="organism_proteins_by_mass",
            description="Reviewed proteins of an organism with a sequence (any isoform) above or below a mass.",
            candidate_query="""
SELECT ?org ?orgName ?protein (MAX(?mass) AS ?maxMass) (MIN(?mass) AS ?minMass) WHERE {
""" + _SIZED_ORGANISMS + """
  GRAPH $UNIPROT { ?protein a up:Protein ; up:organism ?org ; up:sequence ?sequence . ?sequence up:mass ?mass . }
}
GROUP BY ?org ?orgName ?protein
ORDER BY ?org ?protein
""",
            gold_pattern="""SELECT DISTINCT ?protein WHERE {
  GRAPH $UNIPROT {
    ?protein a up:Protein ; up:reviewed true ; up:organism <$org> ; up:sequence ?sequence .
    ?sequence up:mass ?mass .
  }
  FILTER(?mass $op $threshold)
}""",
            question_pattern="Which reviewed proteins of the organism \"$org_name\" (recorded against that taxon exactly, not its strains or sub-taxa) have at least one sequence (any isoform) with a mass $comparison? List each protein once.",
            readings=(R_ORGANISM_SUBTAXA,),
            answer_columns=(AnswerColumn("protein", IRI_COL),),
            key_param="org",
            expand=_mass_thresholds,
            anchors=(ORG_ANCHOR,),
            difficulty={"hops": 2, "anchors": 1, "comparison": "threshold"},
        ),
        uvariant(
            name="better_resolved_protein",
            tier="compositional",
            description="Which of two proteins of one organism has the structure with the best resolution (comparison of two entities).",
            candidate_query="""
SELECT ?org ?orgName ?protein ?name (MIN(?resolution) AS ?best) WHERE {
  {
    SELECT ?org WHERE {
      GRAPH $UNIPROT { ?p a up:Protein ; up:organism ?org ; rdfs:seeAlso ?s . ?s up:resolution ?r . }
    }
    GROUP BY ?org
    HAVING (COUNT(DISTINCT ?p) >= 2 && COUNT(DISTINCT ?p) <= $org_max)
  }
  GRAPH $TAXONOMY { ?org up:scientificName ?orgName . }
  GRAPH $UNIPROT {
    ?protein a up:Protein ; up:organism ?org ; up:recommendedName/up:fullName ?name ; rdfs:seeAlso ?structure .
    ?structure a up:Structure_Resource ; up:resolution ?resolution .
  }
}
GROUP BY ?org ?orgName ?protein ?name
ORDER BY ?org ?protein
""",
            gold_pattern="""SELECT ?protein ?best WHERE {
  {
    SELECT ?protein (MIN(?resolution) AS ?best) WHERE {
      GRAPH $UNIPROT {
        VALUES ?protein { <$a> <$b> }
        ?protein a up:Protein ; up:reviewed true ; rdfs:seeAlso ?structure .
        ?structure a up:Structure_Resource ; up:resolution ?resolution .
      }
    }
    GROUP BY ?protein
  }
}
ORDER BY ?best
LIMIT 1""",
            question_pattern="Of the two proteins \"$a_name\" and \"$b_name\", both of \"$org_name\", which has the experimental structure with the best (numerically lowest) resolution? Give that protein with that resolution.",
            answer_columns=(AnswerColumn("protein", IRI_COL), AnswerColumn("resolution", NUM_COL)),
            key_param="org",
            expand=_protein_pairs_differing("best"),
            anchors=(_protein_anchor("a"), _protein_anchor("b")),
            difficulty={"hops": 2, "anchors": 2, "comparison": "two_entities", "aggregate": "min"},
        ),
        uvariant(
            name="heavier_than_reference",
            tier="compositional",
            description="Reviewed proteins of an organism with a sequence heavier than the heaviest sequence of a reference protein.",
            candidate_query="""
SELECT ?org ?orgName ?protein ?name (MAX(?mass) AS ?maxMass) WHERE {
""" + _SIZED_ORGANISMS + """
  GRAPH $UNIPROT {
    ?protein a up:Protein ; up:organism ?org ; up:recommendedName/up:fullName ?name ; up:sequence ?sequence .
    ?sequence up:mass ?mass .
  }
}
GROUP BY ?org ?orgName ?protein ?name
ORDER BY ?org ?protein
""",
            gold_pattern="""SELECT DISTINCT ?protein WHERE {
  {
    SELECT (MAX(?referenceMass) AS ?limit) WHERE {
      GRAPH $UNIPROT { <$ref> up:sequence ?referenceSequence . ?referenceSequence up:mass ?referenceMass . }
    }
  }
  GRAPH $UNIPROT {
    ?protein a up:Protein ; up:reviewed true ; up:organism <$org> ; up:sequence ?sequence .
    ?sequence up:mass ?mass .
  }
  FILTER(?mass > ?limit)
}""",
            question_pattern="Which reviewed proteins of the organism \"$org_name\" (recorded against that taxon exactly, not its strains or sub-taxa) have a sequence (any isoform) heavier than the heaviest sequence (any isoform) of its protein \"$ref_name\"? List each protein once.",
            readings=(R_ORGANISM_SUBTAXA,),
            answer_columns=(AnswerColumn("protein", IRI_COL),),
            key_param="org",
            expand=_reference_proteins,
            anchors=(ORG_ANCHOR, _protein_anchor("ref")),
            difficulty={"hops": 2, "anchors": 2, "comparison": "reference_entity", "aggregate": "max"},
        ),
    )


# --------------------------------------------------------------------------
# 6. absence_negation


class AbsenceNegation(QuestionTemplate):
    id = "absence_negation"
    description = (
        "Absence: immediate sub-taxa without reviewed proteins, proteins without GO annotation; compositional "
        "(negation over joins): proteins with a catalytic activity but no linked Rhea reaction, proteins with one "
        "keyword but not another."
    )
    membership_mode = "n.a."
    variants = (
        uvariant(
            name="taxon_children_without_proteins",
            membership_mode="direct",
            description=(
                "Immediate sub-taxa (skos:narrowerTransitive, the data's parent link) with no reviewed protein in "
                "their subtree."
            ),
            candidate_query="""
SELECT ?taxon ?taxonName WHERE {
  {
    SELECT ?taxon WHERE { GRAPH $TAXONOMY { ?taxon skos:narrowerTransitive ?child . } }
    GROUP BY ?taxon
    HAVING (COUNT(?child) >= 2 && COUNT(?child) <= 60)
  }
  {
    SELECT DISTINCT ?taxon WHERE {
      GRAPH $UNIPROT { ?protein a up:Protein ; up:organism ?org . }
      GRAPH $TAXONOMY { ?org rdfs:subClassOf ?taxon . ?taxon up:rank up:Taxonomic_Rank_Genus . }
    }
  }
  GRAPH $TAXONOMY { ?taxon up:scientificName ?taxonName . }
}
ORDER BY ?taxon
""",
            gold_pattern="""SELECT DISTINCT ?child WHERE {
  GRAPH $TAXONOMY { <$taxon> skos:narrowerTransitive ?child . }
  MINUS {
    GRAPH $TAXONOMY { <$taxon> skos:narrowerTransitive ?child . ?org rdfs:subClassOf* ?child . }
    GRAPH $UNIPROT { ?protein a up:Protein ; up:reviewed true ; up:organism ?org . }
  }
}""",
            membership_alternative=_CHILDREN_ALTERNATIVE,
            question_pattern="Which taxa directly below the taxon \"$taxon_name\" (its immediate children in the taxonomy, not deeper) have no reviewed protein recorded against them or against any taxon below them? List each taxon once.",
            readings=(membership_reading("direct", r"immediate children|not deeper", _CHILDREN_ALTERNATIVE), R_CHILD_WITHOUT_SUBTAXA,),
            answer_columns=(AnswerColumn("taxon", IRI_COL),),
            key_param="taxon",
            expand=_taxa,
            anchors=(TAXON_ANCHOR,),
            prefer=("direct_vs_transitive",),
            difficulty={"hops": 2, "anchors": 1, "negation": 1, "membership": "direct"},
        ),
        uvariant(
            name="proteins_without_go",
            description="Reviewed proteins of an organism with no GO annotation (organisms where some protein lacks one).",
            candidate_query="""
SELECT ?org ?orgName WHERE {
  {
    SELECT ?org WHERE {
""" + _SIZED_ORGANISMS + """
      GRAPH $UNIPROT {
        ?p a up:Protein ; up:organism ?org .
        OPTIONAL { ?p up:classifiedWith ?t . FILTER(STRSTARTS(STR(?t), "http://purl.obolibrary.org/obo/GO_")) BIND(?p AS ?withGo) }
      }
    }
    GROUP BY ?org
    HAVING (COUNT(DISTINCT ?p) > COUNT(DISTINCT ?withGo))
  }
  GRAPH $TAXONOMY { ?org up:scientificName ?orgName . }
}
ORDER BY ?org
""",
            gold_pattern="""SELECT DISTINCT ?protein WHERE {
  GRAPH $UNIPROT { ?protein a up:Protein ; up:reviewed true ; up:organism <$org> . }
  MINUS {
    GRAPH $UNIPROT {
      ?protein up:organism <$org> ; up:classifiedWith ?term .
      FILTER(STRSTARTS(STR(?term), "http://purl.obolibrary.org/obo/GO_"))
    }
  }
}""",
            question_pattern="Which reviewed proteins of the organism \"$org_name\" (recorded against that taxon exactly, not its strains or sub-taxa) have no Gene Ontology annotation at all? List each protein once.",
            readings=(R_ORGANISM_SUBTAXA,),
            answer_columns=(AnswerColumn("protein", IRI_COL),),
            key_param="org",
            expand=_organisms,
            anchors=(ORG_ANCHOR,),
            difficulty={"hops": 1, "anchors": 1, "negation": 1},
        ),
        uvariant(
            name="activity_without_reaction",
            tier="compositional",
            description="Reviewed proteins of an organism with a catalytic activity but no Rhea reaction in any of their activities.",
            candidate_query="""
SELECT ?org ?orgName WHERE {
  {
    SELECT ?org WHERE {
""" + _SIZED_ORGANISMS + """
      GRAPH $UNIPROT {
        ?p a up:Protein ; up:organism ?org ; up:annotation ?a . ?a a up:Catalytic_Activity_Annotation .
        OPTIONAL { ?a up:catalyticActivity ?ca . ?ca up:catalyzedReaction ?r . }
      }
      BIND(IF(BOUND(?r), ?p, ?noReaction) AS ?withReaction)
    }
    GROUP BY ?org
    HAVING (COUNT(DISTINCT ?p) > COUNT(DISTINCT ?withReaction))
  }
  GRAPH $TAXONOMY { ?org up:scientificName ?orgName . }
}
ORDER BY ?org
""",
            gold_pattern="""SELECT DISTINCT ?protein WHERE {
  GRAPH $UNIPROT {
    ?protein a up:Protein ; up:reviewed true ; up:organism <$org> ; up:annotation ?annotation .
    ?annotation a up:Catalytic_Activity_Annotation .
  }
  MINUS {
    GRAPH $UNIPROT {
      ?protein up:organism <$org> ; up:annotation ?other .
      ?other a up:Catalytic_Activity_Annotation ; up:catalyticActivity ?activity .
      ?activity up:catalyzedReaction ?reaction .
    }
  }
}""",
            question_pattern="Which reviewed proteins of the organism \"$org_name\" (recorded against that taxon exactly, not its strains or sub-taxa) have a catalytic activity annotated, yet every one of their catalytic activities is linked to no catalysed reaction (an EC number alone does not count)? List each protein once.",
            readings=(R_ORGANISM_SUBTAXA, R_PHYSIOLOGICAL,),
            answer_columns=(AnswerColumn("protein", IRI_COL),),
            key_param="org",
            expand=_organisms,
            anchors=(ORG_ANCHOR,),
            difficulty={"hops": 4, "anchors": 1, "negation": 1},
        ),
        uvariant(
            name="keyword_a_not_b",
            tier="compositional",
            description="Reviewed proteins of an organism with one keyword but not another (negation over a join, two anchors).",
            candidate_query="""
SELECT ?org ?orgName ?kw ?kwLabel WHERE {
""" + _SIZED_ORGANISMS + """
  GRAPH $UNIPROT { ?protein a up:Protein ; up:organism ?org ; up:classifiedWith ?kw . }
  FILTER(STRSTARTS(STR(?kw), "http://purl.uniprot.org/keywords/"))
  GRAPH $KEYWORDS { ?kw skos:prefLabel ?kwLabel . }
}
GROUP BY ?org ?orgName ?kw ?kwLabel
HAVING (COUNT(DISTINCT ?protein) >= 2 && COUNT(DISTINCT ?protein) <= 50)
ORDER BY ?org ?kw
""",
            gold_pattern="""SELECT DISTINCT ?protein WHERE {
  GRAPH $UNIPROT { ?protein a up:Protein ; up:reviewed true ; up:organism <$org> ; up:classifiedWith <$kw_a> . }
  FILTER NOT EXISTS { GRAPH $UNIPROT { ?protein up:classifiedWith <$kw_b> . } }
}""",
            question_pattern="Which reviewed proteins of the organism \"$org_name\" (recorded against that taxon exactly, not its strains or sub-taxa) carry the keyword annotation \"$kw_a_label\" but not the keyword annotation \"$kw_b_label\" (each keyword itself; narrower keywords do not count)? List each protein once.",
            readings=(R_ORGANISM_SUBTAXA, R_NARROWER_KEYWORDS,),
            answer_columns=(AnswerColumn("protein", IRI_COL),),
            key_param="org",
            expand=_keyword_pairs,
            anchors=(ORG_ANCHOR, _keyword_anchor("kw_a_label"), _keyword_anchor("kw_b_label")),
            difficulty={"hops": 1, "anchors": 3, "negation": 1, "set_operation": "difference"},
        ),
    )


# --------------------------------------------------------------------------
# 7. cross_graph_join


class CrossGraphJoin(QuestionTemplate):
    id = "cross_graph_join"
    description = (
        "Joins of the uniprot graph with Rhea, diseases and locations: reactions with their Rhea equations, "
        "diseases with their names; compositional: proteins catalysing reactions that involve a Rhea compound, "
        "subcellular locations counted per organism (GROUP BY / HAVING)."
    )
    membership_mode = "n.a."
    variants = (
        uvariant(
            name="protein_reaction_equations",
            description="Reactions a protein catalyses, with their equations from the Rhea graph.",
            candidate_query="""
SELECT ?protein ?name ?orgName (COUNT(DISTINCT ?reaction) AS ?n) WHERE {
""" + _LOOKUP_PROTEINS + """
  GRAPH $UNIPROT {
    ?protein up:annotation ?annotation .
    ?annotation a up:Catalytic_Activity_Annotation ; up:catalyticActivity ?activity .
    ?activity up:catalyzedReaction ?reaction .
  }
}
GROUP BY ?protein ?name ?orgName
ORDER BY ?protein
""",
            gold_pattern="""SELECT DISTINCT ?reaction ?equation WHERE {
  GRAPH $UNIPROT {
    <$entity> up:annotation ?annotation .
    ?annotation a up:Catalytic_Activity_Annotation ; up:catalyticActivity ?activity .
    ?activity up:catalyzedReaction ?reaction .
  }
  GRAPH $RHEA { ?reaction rdfs:label ?equation . }
}""",
            question_pattern="Which reactions does the protein \"$name\" of \"$organism\" catalyse (the reactions as annotated, not their direction-specific forms)? List each reaction with its equation as written in the reaction data.",
            readings=(R_PHYSIOLOGICAL,),
            answer_columns=(AnswerColumn("reaction", IRI_COL), AnswerColumn("equation", LIT_COL)),
            key_param="entity",
            expand=_proteins,
            anchors=(PROTEIN_ANCHOR,),
            difficulty={"hops": 4, "anchors": 1, "graphs": 2},
        ),
        uvariant(
            name="protein_disease_names",
            description="Diseases linked to a protein, with their names from the diseases graph.",
            candidate_query="""
SELECT ?protein ?name ?orgName (COUNT(DISTINCT ?disease) AS ?n) WHERE {
  GRAPH $UNIPROT {
    ?protein a up:Protein ; up:recommendedName/up:fullName ?name ; up:organism ?org ; up:annotation ?annotation .
    ?annotation a up:Disease_Annotation ; up:disease ?disease .
  }
  GRAPH $TAXONOMY { ?org up:scientificName ?orgName . }
}
GROUP BY ?protein ?name ?orgName
ORDER BY ?protein
""",
            gold_pattern="""SELECT DISTINCT ?disease ?diseaseName WHERE {
  GRAPH $UNIPROT { <$entity> up:annotation ?annotation . ?annotation a up:Disease_Annotation ; up:disease ?disease . }
  GRAPH $DISEASES { ?disease skos:prefLabel ?diseaseName . }
}""",
            question_pattern="Which diseases is the protein \"$name\" of \"$organism\" associated with? List each disease with its name.",
            answer_columns=(AnswerColumn("disease", IRI_COL), AnswerColumn("disease_name", LIT_COL)),
            key_param="entity",
            expand=_proteins,
            anchors=(PROTEIN_ANCHOR,),
            difficulty={"hops": 3, "anchors": 1, "graphs": 2},
        ),
        uvariant(
            name="proteins_reacting_with_compound",
            tier="compositional",
            description=(
                "Reviewed proteins of an organism catalysing a reaction in which a Rhea compound takes part "
                "(uniprot -> Rhea join, 6 hops)."
            ),
            candidate_query="""
SELECT ?org ?orgName ?compound ?compoundName WHERE {
  {
    SELECT ?org ?compound (COUNT(DISTINCT ?protein) AS ?n) WHERE {
""" + _SIZED_ORGANISMS + """
      GRAPH $UNIPROT { ?protein up:organism ?org ; up:annotation ?annotation . ?annotation up:catalyticActivity ?activity . ?activity up:catalyzedReaction ?reaction . }
      GRAPH $RHEA { ?reaction rh:side ?side . ?side rh:contains ?participant . ?participant rh:compound ?compound . }
    }
    GROUP BY ?org ?compound
    HAVING (COUNT(DISTINCT ?protein) >= 2 && COUNT(DISTINCT ?protein) <= 40)
  }
  GRAPH $TAXONOMY { ?org up:scientificName ?orgName . }
  GRAPH $RHEA { ?compound rh:name ?compoundName . }
}
ORDER BY ?org ?compound
""",
            gold_pattern="""SELECT DISTINCT ?protein WHERE {
  GRAPH $RHEA { ?reaction rh:side ?side . ?side rh:contains ?participant . ?participant rh:compound <$compound> . }
  GRAPH $UNIPROT {
    ?protein a up:Protein ; up:reviewed true ; up:organism <$org> ; up:annotation ?annotation .
    ?annotation a up:Catalytic_Activity_Annotation ; up:catalyticActivity ?activity .
    ?activity up:catalyzedReaction ?reaction .
  }
}""",
            question_pattern="Which reviewed proteins of the organism \"$org_name\" (recorded against that taxon exactly, not its strains or sub-taxa) catalyse a reaction in which the compound \"$compound_name\" (as named in the reaction data) takes part, on either side? List each protein once.",
            readings=(R_ORGANISM_SUBTAXA, R_PHYSIOLOGICAL,),
            answer_columns=(AnswerColumn("protein", IRI_COL),),
            key_param="org",
            expand=_org_compounds,
            anchors=(ORG_ANCHOR, COMPOUND_ANCHOR),
            difficulty={"hops": 6, "anchors": 2, "graphs": 2},
        ),
        uvariant(
            name="organism_location_counts",
            tier="compositional",
            description="Subcellular locations (by name) recorded for at least n reviewed proteins of an organism (GROUP BY / HAVING across graphs).",
            candidate_query="""
SELECT ?org ?orgName WHERE {
""" + _SIZED_ORGANISMS + """
  FILTER EXISTS { GRAPH $UNIPROT { ?p up:organism ?org ; up:annotation ?a . ?a a up:Subcellular_Location_Annotation . } }
}
ORDER BY ?org
""",
            gold_pattern="""SELECT ?locationName (COUNT(DISTINCT ?protein) AS ?proteins) WHERE {
  GRAPH $UNIPROT {
    ?protein a up:Protein ; up:reviewed true ; up:organism <$org> ; up:annotation ?annotation .
    ?annotation a up:Subcellular_Location_Annotation ; up:locatedIn ?located .
    ?located up:cellularComponent ?location .
  }
  GRAPH $LOCATIONS { ?location skos:prefLabel ?locationName . }
}
GROUP BY ?locationName
HAVING (COUNT(DISTINCT ?protein) >= $having_min)""",
            question_pattern="For the organism \"$org_name\" (recorded against that taxon exactly, not its strains or sub-taxa), which subcellular locations, according to the subcellular-location annotation, are recorded for at least $having_min of its reviewed proteins (each location counts only proteins annotated with that location itself; narrower locations do not count)? List each location by name with its number of proteins.",
            readings=(R_ORGANISM_SUBTAXA, R_NARROWER_LOCATIONS,),
            answer_columns=(AnswerColumn("location_name", LIT_COL), AnswerColumn("proteins", NUM_COL)),
            key_param="org",
            expand=_organisms,
            anchors=(ORG_ANCHOR,),
            knob_params=("having_min",),
            difficulty={"hops": 4, "anchors": 1, "graphs": 2, "aggregate": "group_having"},
        ),
    )


TEMPLATES: tuple[type[QuestionTemplate], ...] = (
    TypeMembers,
    OneHopAttribute,
    MultiHopPath,
    Aggregation,
    NumericFilter,
    AbsenceNegation,
    CrossGraphJoin,
)
TEMPLATES_BY_ID = {t.id: t for t in TEMPLATES}


# --------------------------------------------------------------------------
# Calibration round 2: one structural step per compositional variant
#
# Round 1 gave Sonnet 0.899 exact on the development split of the
# compositional tier, above the 50-80% band, so every compositional variant
# goes up one step along its structural axis (design ruling on
# experiments#12): one more population, one more HAVING constraint, one more
# nested negation, or one more edge. The axis is read from the round-1
# difficulty declaration by a fixed precedence (generate.STEP_RULES); nothing
# is tuned per template. The raised variants keep their names, population
# sizes and scope clauses; each checks on the store that its step matters
# (StepCheck). The lookup tier is untouched.


def _go_part_of(param: str, var: str):
    """Reading transform: "below the GO term in ``param``" also follows part-of (see _go_with_part_of)."""

    def transform(sparql: str, params: dict) -> str | None:
        go = params.get(param)
        if not go:
            return None
        old = f"GRAPH {GRAPHS['GO']} {{ ?{var} rdfs:subClassOf* <{go}> . }}"
        if old not in sparql:
            return None
        new = (
            f"GRAPH {GRAPHS['GO']} {{ {{ ?{var} rdfs:subClassOf* <{go}> . }} UNION {{ ?{var} rdfs:subClassOf* ?isaAncestor{var} . "
            f"?isaAncestor{var} rdfs:subClassOf ?partOfRestriction{var} . "
            f"?partOfRestriction{var} <http://www.w3.org/2002/07/owl#onProperty> <http://purl.obolibrary.org/obo/BFO_0000050> ; "
            f"<http://www.w3.org/2002/07/owl#someValuesFrom> ?partOfParent{var} . ?partOfParent{var} rdfs:subClassOf* <{go}> . }} }}"
        )
        return sparql.replace(old, new)

    return transform


GO_B_ANCHOR = Anchor("GO term", ("go_b_label",), "GRAPH $GO { ?e rdfs:label $go_b_label_lit . }", "go_b")
KW_ANCHOR = _keyword_anchor("kw_label")

R2_GO_PART_OF = Reading(
    "go_part_of", "below a GO term also by part-of, not only is-a", r"is-a hierarchy.*is-a hierarchy",
    transform=compose(_go_part_of("go", "term"), _go_part_of("go_b", "termB")),
)
R2_NARROWER_KEYWORDS = Reading(
    "narrower_keywords",
    "a keyword also covers the narrower keywords below it",
    r"narrower keywords do not count",
    transform=compose(
        scope_iri_to_closure("up:classifiedWith", "kw_a", GRAPHS["KEYWORDS"], "scopeKeywordA"),
        scope_iri_to_closure("up:classifiedWith", "kw_b", GRAPHS["KEYWORDS"], "scopeKeywordB"),
        scope_iri_to_closure("up:classifiedWith", "kw_c", GRAPHS["KEYWORDS"], "scopeKeywordC"),
    ),
)
R2_COUNTS_WITH_SUBTAXA = Reading(
    "counts_with_subtaxa",
    "each taxon's count also takes the proteins of the taxa below it",
    r"recorded directly against them|sub-taxa do not count",
    pattern="""SELECT ?org (COUNT(DISTINCT ?protein) AS ?proteins) WHERE {
  GRAPH $TAXONOMY { ?org rdfs:subClassOf* <$taxon> . ?below rdfs:subClassOf* ?org . }
  GRAPH $UNIPROT { ?protein a up:Protein ; up:reviewed true ; up:organism ?below . }
}
GROUP BY ?org
HAVING (COUNT(DISTINCT ?protein) >= $having_min && COUNT(DISTINCT ?protein) <= $having_max)""",
)
R2_SOME_ACTIVITY = Reading(
    "some_activity_linked",
    "a protein qualifies when some of its catalytic activities is linked to a reaction, not every one",
    # Exact polarity: "not every one ..." or "... linked to no catalysed reaction" must not match.
    r"(?<!not )\bevery one of their catalytic activities is linked to a catalysed reaction",
    pattern="""SELECT DISTINCT ?protein WHERE {
  GRAPH $UNIPROT {
    ?protein a up:Protein ; up:reviewed true ; up:organism <$org> ; up:annotation ?annotation .
    ?annotation a up:Catalytic_Activity_Annotation ; up:catalyticActivity ?activity .
    ?activity up:catalyzedReaction ?reaction .
  }
}""",
)
# "Every catalytic activity is linked to a reaction", quantified over activities; the protein must have at least
# one catalytic activity annotated (an annotation with no activity does not qualify it; confirmation review of
# 2d53607). $PHYSIOLOGICAL is empty in the
# gold; the physiological reading also counts an activity as linked when its annotation records a physiological
# (direction-specific) reaction.
_ACTIVITY_GOLD = """SELECT DISTINCT ?protein WHERE {
  GRAPH $UNIPROT {
    ?protein a up:Protein ; up:reviewed true ; up:organism <$org> ; up:annotation ?annotation .
    ?annotation a up:Catalytic_Activity_Annotation ; up:catalyticActivity ?annotatedActivity .
  }
  MINUS {
    GRAPH $UNIPROT { ?protein up:organism <$org> ; up:annotation ?other . ?other a up:Catalytic_Activity_Annotation ; up:catalyticActivity ?activity . }
    MINUS { GRAPH $UNIPROT { ?activity up:catalyzedReaction ?reaction . } }$PHYSIOLOGICAL
  }
}"""
R2_PHYSIOLOGICAL_ACTIVITY = Reading(
    "physiological_directions",
    "an activity also counts as linked when its annotation records a direction-specific (physiological) reaction",
    r"not their direction-specific forms",
    pattern=_ACTIVITY_GOLD.replace(
        "$PHYSIOLOGICAL", "\n    MINUS { GRAPH $UNIPROT { ?other up:catalyzedPhysiologicalReaction ?physiologicalReaction . } }"
    ),
)
# Chain-, peptide- and isoform-scoped location annotations (up:sequence on the annotation) are attached to the
# protein and counted by the gold; the reading drops them (study lead's audit: they decide 5 of 14 answers).
LOCATION_SCOPE = "(annotations on the protein's mature chains, peptides or isoforms count)"
R2_CHAIN_LOCATIONS = Reading(
    "chain_scoped_locations",
    "only annotations of the whole protein count, not those scoped to one of its chains, peptides or isoforms",
    r"mature chains, peptides or isoforms count\)",
    pattern="""SELECT ?locationName (COUNT(DISTINCT ?protein) AS ?proteins) WHERE {
  GRAPH $UNIPROT {
    ?protein a up:Protein ; up:reviewed true ; up:organism <$org> ; up:annotation ?annotation .
    ?annotation a up:Subcellular_Location_Annotation ; up:locatedIn ?located .
    ?located up:cellularComponent ?location .
  }
  FILTER NOT EXISTS { GRAPH $UNIPROT { ?annotation up:sequence ?scopedSequence . } }
  GRAPH $LOCATIONS { ?location skos:prefLabel ?locationName . }
}
GROUP BY ?locationName
HAVING (COUNT(DISTINCT ?protein) >= $having_min && COUNT(DISTINCT ?protein) <= $having_max)""",
)
R2_DISEASE_NARROWER_KEYWORDS = Reading(
    "narrower_keywords",
    "a keyword also covers the narrower keywords below it",
    r"narrower keywords do not count",
    transform=scope_iri_to_closure("rdfs:seeAlso", "kw", GRAPHS["KEYWORDS"], "scopeKeyword"),
)
R2_PROTEIN_KEYWORD = Reading(
    "protein_keyword",
    "the keyword is read as one assigned to the proteins (with a disease annotation), not one their diseases refer to",
    r"nor do keywords assigned to the proteins themselves",
    pattern="""SELECT DISTINCT ?protein ?geneName WHERE {
  GRAPH $UNIPROT {
    ?protein a up:Protein ; up:reviewed true ; up:classifiedWith <$kw> ; up:annotation ?annotation ; up:encodedBy ?gene .
    ?annotation a up:Disease_Annotation .
    ?gene skos:prefLabel ?geneName .
  }
}""",
)

# The hop step's subtree wording (study lead's audit, experiments#12): "the taxon X or ... any taxon below it
# at any depth". The exact reading (proteins recorded against the taxon itself, as an organism question means it)
# and the immediate-children reading are guarded by the depth clause; the anchor-left-out reading by naming it.
R2_TAXON_EXACT = Reading(
    "taxon_without_subtaxa",
    "the taxon means only itself (proteins recorded against it exactly, as for an organism), not its subtaxa",
    r"or of any taxon below it at any depth",
    transform=restrict_subtree_to_anchor("taxon", GRAPHS["TAXONOMY"]),
)
_SUBTREE_CLAUSES = (
    R2_TAXON_EXACT,
    _direct(r"any taxon below it at any depth", _TAXON_IMMEDIATE),
    _anchor_inclusion(r'taxon "[^"]+" or of any taxon below it', _TAXON_WITHOUT_ANCHOR),
)

# A taxon whose subtree (itself included) holds 2+ organisms with reviewed proteins and, in all, a number of
# reviewed proteins in the organism-size band (org_min..org_max): the hop step's anchor population.
_SIZED_SUBTREES = """
  {
    SELECT ?taxon WHERE {
      GRAPH $UNIPROT { ?sizedProtein a up:Protein ; up:reviewed true ; up:organism ?sizedOrg . }
      GRAPH $TAXONOMY { ?sizedOrg rdfs:subClassOf* ?taxon . }
    }
    GROUP BY ?taxon
    HAVING (COUNT(DISTINCT ?sizedOrg) >= 2 && COUNT(DISTINCT ?sizedProtein) >= $org_min && COUNT(DISTINCT ?sizedProtein) <= $org_max)
  }
"""

# Triples (three keywords or proteins of one anchor) come from the anchor's first TRIPLE_POOL items in a
# stable scrambled order, so the number of combinations stays bounded; pairs use every combination.
TRIPLE_POOL = 20


def _members(row: PoolRow, field_name: str) -> frozenset[str]:
    return frozenset((row.get(field_name) or "").split())


def _triples(keys: Iterable[str], salt: str) -> list[tuple[str, str, str]]:
    pool = sorted(_stable_order(keys, salt)[:TRIPLE_POOL])
    return list(combinations(pool, 3))


def _keyword_triples(ok):
    """Triples of keywords used in one organism whose protein sets pass ``ok(a, b, c)``; one question per organism."""

    def expand(rows: list[PoolRow], knobs: Any = None) -> Iterable[dict[str, Any]]:
        cap = int((knobs or KNOBS)["max_pair_options"])
        for org, group in _group(rows, "org").items():
            kws = {r["kw"]: r for r in group if usable_label(r["kwLabel"])}
            if len(kws) < 3 or not usable_label(group[0]["orgName"]):
                continue
            sets = {k: _members(r, "proteins") for k, r in kws.items()}
            valid = [t for t in _triples(kws, org) if ok(*(sets[k] for k in t))]
            for a, b, c in _stable_order(valid, org)[:cap]:
                yield {
                    "org": check_iri(org),
                    "org_name": group[0]["orgName"],
                    "kw_a": check_iri(a),
                    "kw_a_label": kws[a]["kwLabel"],
                    "kw_b": check_iri(b),
                    "kw_b_label": kws[b]["kwLabel"],
                    "kw_c": check_iri(c),
                    "kw_c_label": kws[c]["kwLabel"],
                }

    return expand


def _union_ok(a: frozenset, b: frozenset, c: frozenset) -> bool:
    """Each keyword brings a protein the other two do not (every population matters), at most 50 in all."""
    return len(a | b | c) <= 50 and bool(a - b - c) and bool(b - a - c) and bool(c - a - b)


def _difference_ok(a: frozenset, b: frozenset, c: frozenset) -> bool:
    """A minus B minus C is not empty, and each subtracted keyword removes a protein the other does not."""
    return bool(a - b - c) and bool((a & b) - c) and bool((a & c) - b)


def _protein_triples(ok, field_name: str):
    """Triples of proteins of one organism whose ``field_name`` values pass ``ok(a, b, c)``; one question per organism.

    The triple comes in IRI order; comparison questions reorder it after selection (generate.place_winner).
    """

    def expand(rows: list[PoolRow], knobs: Any = None) -> Iterable[dict[str, Any]]:
        cap = int((knobs or KNOBS)["max_pair_options"])
        for org, group in _group(rows, "org").items():
            prots = {r["protein"]: r for r in group if usable_label(r["name"])}
            if len(prots) < 3 or not usable_label(group[0]["orgName"]):
                continue
            valid = [t for t in _triples(prots, org) if ok(*(prots[p][field_name] for p in t))]
            for a, b, c in _stable_order(valid, org)[:cap]:
                yield {
                    "org": check_iri(org),
                    "org_name": group[0]["orgName"],
                    "a": check_iri(a),
                    "a_name": prots[a]["name"],
                    "b": check_iri(b),
                    "b_name": prots[b]["name"],
                    "c": check_iri(c),
                    "c_name": prots[c]["name"],
                }

    return expand


def _shared_ok(a: str, b: str, c: str) -> bool:
    """Keywords common to all three exist, and dropping any one protein adds a common keyword."""
    sa, sb, sc = (frozenset(x.split()) for x in (a, b, c))
    common = sa & sb & sc
    return bool(common) and (sa & sb) != common and (sa & sc) != common and (sb & sc) != common


def _distinct_dates(a: str, b: str, c: str) -> bool:
    return len({a, b, c}) == 3


def _distinct_numbers(a: str, b: str, c: str) -> bool:
    return len({float(a), float(b), float(c)}) == 3


def _reference_pairs(rows: list[PoolRow], knobs: Any = None) -> Iterable[dict[str, Any]]:
    """Per organism, pairs of round-1 reference proteins with different heaviest sequences."""
    cap = int((knobs or KNOBS)["max_pair_options"])
    for org, group in _group(rows, "org").items():
        if not usable_label(group[0]["orgName"]):
            continue
        masses = [int(r["maxMass"]) for r in group]
        refs = {
            r["protein"]: r for r in group
            if usable_label(r["name"]) and 1 <= sum(1 for m in masses if m > int(r["maxMass"])) <= 50
        }
        pairs = [(a, b) for a, b in combinations(sorted(refs), 2) if refs[a]["maxMass"] != refs[b]["maxMass"]]
        for a, b in _stable_order(pairs, org)[:cap]:
            yield {
                "org": check_iri(org),
                "org_name": group[0]["orgName"],
                "ref": check_iri(a),
                "ref_name": refs[a]["name"],
                "ref_b": check_iri(b),
                "ref_b_name": refs[b]["name"],
            }


def _taxon_go_triples(rows: list[PoolRow], knobs: Any = None) -> Iterable[dict[str, Any]]:
    """Per taxon, pairs of GO terms whose directly annotated proteins there overlap in 2+ and neither contains the other."""
    cap = int((knobs or KNOBS)["max_pair_options"])
    for taxon, group in _group(rows, "taxon").items():
        if not usable_label(group[0]["taxonName"]):
            continue
        gos = {r["go"]: r for r in group if usable_label(r["goLabel"])}
        sets = {g: _members(r, "proteins") for g, r in gos.items()}
        valid = [
            (a, b) for a, b in combinations(sorted(gos), 2)
            if len(sets[a] & sets[b]) >= 2 and not sets[a] <= sets[b] and not sets[b] <= sets[a]
        ]
        for a, b in _stable_order(valid, taxon)[:cap]:
            yield {
                "taxon": check_iri(taxon),
                "taxon_name": group[0]["taxonName"],
                "go": check_iri(a),
                "go_label": gos[a]["goLabel"],
                "go_b": check_iri(b),
                "go_b_label": gos[b]["goLabel"],
            }


def _taxon_compounds(rows: list[PoolRow], knobs: Any = None) -> Iterable[dict[str, Any]]:
    for row in rows:
        if usable_label(row["taxonName"]) and usable_label(row["compoundName"]):
            yield {
                "taxon": check_iri(row["taxon"]),
                "taxon_name": row["taxonName"],
                "compound": check_iri(row["compound"]),
                "compound_name": row["compoundName"],
            }


def _keywords(rows: list[PoolRow], knobs: Any = None) -> Iterable[dict[str, Any]]:
    for row in rows:
        if usable_label(row["kwLabel"]):
            yield {"kw": check_iri(row["kw"]), "kw_label": row["kwLabel"], **_shape(row)}


_ABC = (("a", "a_name"), ("b", "b_name"), ("c", "c_name"))


def _winner_in_rows(slots: tuple[tuple[str, ...], ...]):
    """Winner of a comparison whose answer row starts with the winning entity: its slot index."""

    def winner(params: dict, rows: list, step: Any) -> int:
        return [params[s[0]] for s in slots].index(rows[0][0]["value"])

    return winner


def _deciding_reference(params: dict, rows: list, step: Any) -> int:
    """The reference whose heaviest sequence sets the limit: the one whose round-1 form alone gives the gold rows.

    The step check's undo forms are the round-1 gold with ``ref`` alone, then with ``ref_b`` alone.
    """
    same = [i for i, changed in enumerate(step["undo_changed"]) if not changed]
    if len(same) != 1:
        raise ValueError(f"exactly one reference must decide, got {step['undo_changed']}")
    return same[0]


def _undo(variant: Variant, *renames: dict[str, str]) -> tuple[tuple[str, dict[str, str]], ...]:
    """Undo patterns: the round-1 gold under each renaming of its parameters."""
    return tuple((variant.gold_pattern, r) for r in renames)


_V = {(t.id, v.name): v for t in TEMPLATES for v in t.variants}

# --- populations: set operations (one more population of the last-named kind) ---

R2_TAXON_GO = uvariant(
    name="taxon_go_intersection",
    tier="compositional",
    description=(
        "Reviewed proteins in a taxon's subtree that also carry two GO terms, each or one of its is-a descendants "
        "(intersection of three transitive populations)."
    ),
    candidate_query="""
SELECT ?taxon ?taxonName ?go ?goLabel ?proteins WHERE {
  {
    SELECT ?taxon ?go (GROUP_CONCAT(DISTINCT STR(?protein); separator=" ") AS ?proteins) WHERE {
      {
        SELECT ?taxon WHERE {
          GRAPH $UNIPROT { ?p a up:Protein ; up:organism ?o . }
          GRAPH $TAXONOMY { ?o rdfs:subClassOf ?taxon . ?taxon up:rank up:Taxonomic_Rank_Genus . }
        }
        GROUP BY ?taxon
        HAVING (COUNT(DISTINCT ?p) > 50 && COUNT(DISTINCT ?p) <= 1000)
      }
      GRAPH $TAXONOMY { ?org rdfs:subClassOf ?taxon . }
      GRAPH $UNIPROT { ?protein a up:Protein ; up:organism ?org ; up:classifiedWith ?go . }
      FILTER(STRSTARTS(STR(?go), "http://purl.obolibrary.org/obo/GO_"))
    }
    GROUP BY ?taxon ?go
    HAVING (COUNT(DISTINCT ?protein) >= 2 && COUNT(DISTINCT ?protein) <= 40)
  }
  GRAPH $TAXONOMY { ?taxon up:scientificName ?taxonName . }
  GRAPH $GO { ?go rdfs:label ?goLabel . }
}
ORDER BY ?taxon ?go
""",
    gold_pattern="""SELECT DISTINCT ?protein WHERE {
  GRAPH $TAXONOMY { ?org rdfs:subClassOf* <$taxon> . }
  GRAPH $GO { ?term rdfs:subClassOf* <$go> . }
  GRAPH $GO { ?termB rdfs:subClassOf* <$go_b> . }
  GRAPH $UNIPROT { ?protein a up:Protein ; up:reviewed true ; up:organism ?org ; up:classifiedWith ?term ; up:classifiedWith ?termB . }
}""",
    question_pattern=(
        "Which reviewed proteins come from the taxon \"$taxon_name\" (or any taxon below it at any depth), are annotated with the Gene "
        "Ontology term \"$go_label\" (or any Gene Ontology term below it in the is-a hierarchy) and are also annotated with "
        "the Gene Ontology term \"$go_b_label\" (or any Gene Ontology term below it in the is-a hierarchy)? List each "
        "protein once."
    ),
    readings=(
        R_TAXON_ONLY,
        R2_GO_PART_OF,
        _direct(
            r"any taxon below it at any depth.*any Gene Ontology term below it.*any Gene Ontology term below it",
            compose(_TAXON_IMMEDIATE, _GO_IMMEDIATE, immediate_children_by_closure("go_b", GRAPHS["GO"])),
        ),
        _anchor_inclusion(
            r'taxon "[^"]+" \(or any taxon below it at any depth\).*Gene Ontology term "[^"]+" \(or any Gene Ontology term '
            r'below it.*Gene Ontology term "[^"]+" \(or any Gene Ontology term below it',
            compose(_TAXON_WITHOUT_ANCHOR, _GO_WITHOUT_ANCHOR, exclude_anchor("go_b", GRAPHS["GO"])),
        ),
    ),
    answer_columns=(AnswerColumn("protein", IRI_COL),),
    key_param="taxon",
    expand=_taxon_go_triples,
    anchors=(TAXON_ANCHOR, GO_ANCHOR, GO_B_ANCHOR),
    difficulty={"hops": 6, "anchors": 3, "set_operation": "intersection", "populations": 3, "membership": "transitive"},
    step_check=StepCheck(
        "every_population_matters",
        undo=_undo(_V[("type_members", "taxon_go_intersection")], {}, {"go": "go_b", "go_label": "go_b_label"}),
    ),
)

R2_KEYWORD_UNION = uvariant(
    name="organism_keyword_union",
    tier="compositional",
    membership_mode="n.a.",
    description="Reviewed proteins of one organism carrying any of three keywords (union of three populations).",
    candidate_query="""
SELECT ?org ?orgName ?kw ?kwLabel (GROUP_CONCAT(DISTINCT STR(?protein); separator=" ") AS ?proteins) WHERE {
""" + _SIZED_ORGANISMS + """
  GRAPH $UNIPROT { ?protein a up:Protein ; up:organism ?org ; up:classifiedWith ?kw . }
  FILTER(STRSTARTS(STR(?kw), "http://purl.uniprot.org/keywords/"))
  GRAPH $KEYWORDS { ?kw skos:prefLabel ?kwLabel . }
}
GROUP BY ?org ?orgName ?kw ?kwLabel
HAVING (COUNT(DISTINCT ?protein) >= 2 && COUNT(DISTINCT ?protein) <= 25)
ORDER BY ?org ?kw
""",
    gold_pattern="""SELECT DISTINCT ?protein WHERE {
  GRAPH $UNIPROT { ?protein a up:Protein ; up:reviewed true ; up:organism <$org> . }
  { GRAPH $UNIPROT { ?protein up:classifiedWith <$kw_a> . } }
  UNION
  { GRAPH $UNIPROT { ?protein up:classifiedWith <$kw_b> . } }
  UNION
  { GRAPH $UNIPROT { ?protein up:classifiedWith <$kw_c> . } }
}""",
    question_pattern=(
        "Which reviewed proteins of the organism \"$org_name\" (recorded against that taxon exactly, not its strains or "
        "sub-taxa) carry the keyword annotation \"$kw_a_label\", the keyword annotation \"$kw_b_label\" or the keyword "
        "annotation \"$kw_c_label\" (any of them; only the keywords themselves, narrower keywords do not count)? List "
        "each protein once."
    ),
    readings=(R_ORGANISM_SUBTAXA, R2_NARROWER_KEYWORDS),
    answer_columns=(AnswerColumn("protein", IRI_COL),),
    key_param="org",
    expand=_keyword_triples(_union_ok),
    anchors=(ORG_ANCHOR, _keyword_anchor("kw_a_label"), _keyword_anchor("kw_b_label"), _keyword_anchor("kw_c_label")),
    difficulty={"hops": 4, "anchors": 4, "set_operation": "union", "populations": 3},
    step_check=StepCheck(
        "every_population_matters",
        undo=_undo(
            _V[("type_members", "organism_keyword_union")],
            {},
            {"kw_b": "kw_c"},
            {"kw_a": "kw_b", "kw_b": "kw_c"},
        ),
    ),
)

R2_SHARED_KEYWORDS = uvariant(
    name="shared_keywords",
    tier="compositional",
    description="Keywords that three proteins of one organism all carry (intersection of three attribute sets).",
    candidate_query="""
SELECT ?org ?orgName ?protein ?name (GROUP_CONCAT(DISTINCT STR(?kw); separator=" ") AS ?keywords) WHERE {
""" + _SIZED_ORGANISMS + """
  GRAPH $UNIPROT {
    ?protein a up:Protein ; up:organism ?org ; up:recommendedName/up:fullName ?name ; up:classifiedWith ?kw .
  }
  FILTER(STRSTARTS(STR(?kw), "http://purl.uniprot.org/keywords/"))
}
GROUP BY ?org ?orgName ?protein ?name
HAVING (COUNT(DISTINCT ?kw) >= 3)
ORDER BY ?org ?protein
""",
    gold_pattern="""SELECT DISTINCT ?keyword WHERE {
  GRAPH $UNIPROT { <$a> up:classifiedWith ?keyword . <$b> up:classifiedWith ?keyword . <$c> up:classifiedWith ?keyword . }
  FILTER(STRSTARTS(STR(?keyword), "http://purl.uniprot.org/keywords/"))
}""",
    question_pattern=(
        "Which keyword annotations do the proteins \"$a_name\", \"$b_name\" and \"$c_name\", all of \"$org_name\", have "
        "in common (only keywords assigned directly; broader keywords above them do not count)? List each keyword once."
    ),
    readings=(R_BROADER_KEYWORDS,),
    answer_columns=(AnswerColumn("keyword", IRI_COL),),
    key_param="org",
    expand=_protein_triples(_shared_ok, "keywords"),
    hops=1,
    anchors=(_protein_anchor("a"), _protein_anchor("b"), _protein_anchor("c")),
    difficulty={"hops": 3, "anchors": 3, "set_operation": "intersection", "populations": 3},
    step_check=StepCheck(
        "every_population_matters",
        undo=_undo(
            _V[("one_hop_attribute", "shared_keywords")],
            {},
            {"b": "c", "b_name": "c_name"},
            {"a": "b", "a_name": "b_name", "b": "c", "b_name": "c_name"},
        ),
    ),
)

R2_KEYWORD_DIFFERENCE = uvariant(
    name="keyword_a_not_b",
    tier="compositional",
    description="Reviewed proteins of an organism with one keyword but neither of two others (difference of three populations).",
    candidate_query="""
SELECT ?org ?orgName ?kw ?kwLabel (GROUP_CONCAT(DISTINCT STR(?protein); separator=" ") AS ?proteins) WHERE {
""" + _SIZED_ORGANISMS + """
  GRAPH $UNIPROT { ?protein a up:Protein ; up:organism ?org ; up:classifiedWith ?kw . }
  FILTER(STRSTARTS(STR(?kw), "http://purl.uniprot.org/keywords/"))
  GRAPH $KEYWORDS { ?kw skos:prefLabel ?kwLabel . }
}
GROUP BY ?org ?orgName ?kw ?kwLabel
HAVING (COUNT(DISTINCT ?protein) >= 2 && COUNT(DISTINCT ?protein) <= 50)
ORDER BY ?org ?kw
""",
    gold_pattern="""SELECT DISTINCT ?protein WHERE {
  GRAPH $UNIPROT { ?protein a up:Protein ; up:reviewed true ; up:organism <$org> ; up:classifiedWith <$kw_a> . }
  FILTER NOT EXISTS { GRAPH $UNIPROT { ?protein up:classifiedWith <$kw_b> . } }
  FILTER NOT EXISTS { GRAPH $UNIPROT { ?protein up:classifiedWith <$kw_c> . } }
}""",
    question_pattern=(
        "Which reviewed proteins of the organism \"$org_name\" (recorded against that taxon exactly, not its strains or "
        "sub-taxa) carry the keyword annotation \"$kw_a_label\" but neither the keyword annotation \"$kw_b_label\" nor the "
        "keyword annotation \"$kw_c_label\" (each keyword itself; narrower keywords do not count)? List each protein once."
    ),
    readings=(R_ORGANISM_SUBTAXA, R2_NARROWER_KEYWORDS),
    answer_columns=(AnswerColumn("protein", IRI_COL),),
    key_param="org",
    expand=_keyword_triples(_difference_ok),
    anchors=(ORG_ANCHOR, _keyword_anchor("kw_a_label"), _keyword_anchor("kw_b_label"), _keyword_anchor("kw_c_label")),
    difficulty={"hops": 4, "anchors": 4, "negation": 1, "set_operation": "difference", "populations": 3},
    step_check=StepCheck(
        "every_population_matters",
        undo=_undo(_V[("absence_negation", "keyword_a_not_b")], {}, {"kw_b": "kw_c", "kw_b_label": "kw_c_label"}),
    ),
)

# --- populations: comparisons (one more entity compared) ---

R2_EARLIER_ENTRY = uvariant(
    name="earlier_entry",
    tier="compositional",
    description="Which of three proteins of one organism entered UniProtKB first (comparison of three entities).",
    candidate_query=_V[("one_hop_attribute", "earlier_entry")].candidate_query,
    gold_pattern="""SELECT ?protein ?created WHERE {
  GRAPH $UNIPROT { VALUES ?protein { <$a> <$b> <$c> } ?protein a up:Protein ; up:reviewed true ; up:created ?created . }
}
ORDER BY ?created
LIMIT 1""",
    question_pattern=(
        "Which of the three proteins \"$a_name\", \"$b_name\" and \"$c_name\", all of \"$org_name\", has the earliest "
        "entry creation date? Give that protein with the date its entry was created."
    ),
    answer_columns=(AnswerColumn("protein", IRI_COL), AnswerColumn("created", LIT_COL)),
    key_param="org",
    expand=_protein_triples(_distinct_dates, "created"),
    hops=1,
    anchors=(_protein_anchor("a"), _protein_anchor("b"), _protein_anchor("c")),
    difficulty={"hops": 1, "anchors": 3, "comparison": True, "aggregate": "order_limit", "populations": 3},
    slots=_ABC,
    winner=_winner_in_rows(_ABC),
    step_check=StepCheck(
        "winner_matters",
        undo=_undo(
            _V[("one_hop_attribute", "earlier_entry")],
            {},
            {"b": "c"},
            {"a": "b", "b": "c"},
        ),
        mode="some",
    ),
)

R2_BETTER_RESOLVED = uvariant(
    name="better_resolved_protein",
    tier="compositional",
    description="Which of three proteins of one organism has the structure with the best resolution (comparison of three entities).",
    candidate_query=_V[("numeric_filter", "better_resolved_protein")].candidate_query,
    gold_pattern="""SELECT ?protein ?best WHERE {
  {
    SELECT ?protein (MIN(?resolution) AS ?best) WHERE {
      GRAPH $UNIPROT {
        VALUES ?protein { <$a> <$b> <$c> }
        ?protein a up:Protein ; up:reviewed true ; rdfs:seeAlso ?structure .
        ?structure a up:Structure_Resource ; up:resolution ?resolution .
      }
    }
    GROUP BY ?protein
  }
}
ORDER BY ?best
LIMIT 1""",
    question_pattern=(
        "Of the three proteins \"$a_name\", \"$b_name\" and \"$c_name\", all of \"$org_name\", which has the experimental "
        "structure with the best (numerically lowest) resolution? Give that protein with that resolution."
    ),
    answer_columns=(AnswerColumn("protein", IRI_COL), AnswerColumn("resolution", NUM_COL)),
    key_param="org",
    expand=_protein_triples(_distinct_numbers, "best"),
    anchors=(_protein_anchor("a"), _protein_anchor("b"), _protein_anchor("c")),
    difficulty={"hops": 2, "anchors": 3, "comparison": "three_entities", "aggregate": "min", "populations": 3},
    slots=_ABC,
    winner=_winner_in_rows(_ABC),
    step_check=StepCheck(
        "winner_matters",
        undo=_undo(
            _V[("numeric_filter", "better_resolved_protein")],
            {},
            {"b": "c"},
            {"a": "b", "b": "c"},
        ),
        mode="some",
    ),
)

R2_HEAVIER = uvariant(
    name="heavier_than_reference",
    tier="compositional",
    description="Reviewed proteins of an organism with a sequence heavier than the heaviest sequence of each of two reference proteins.",
    candidate_query=_V[("numeric_filter", "heavier_than_reference")].candidate_query,
    gold_pattern="""SELECT DISTINCT ?protein WHERE {
  {
    SELECT (MAX(?referenceMass) AS ?limit) WHERE {
      GRAPH $UNIPROT { VALUES ?reference { <$ref> <$ref_b> } ?reference up:sequence ?referenceSequence . ?referenceSequence up:mass ?referenceMass . }
    }
  }
  GRAPH $UNIPROT {
    ?protein a up:Protein ; up:reviewed true ; up:organism <$org> ; up:sequence ?sequence .
    ?sequence up:mass ?mass .
  }
  FILTER(?mass > ?limit)
}""",
    question_pattern=(
        "Which reviewed proteins of the organism \"$org_name\" (recorded against that taxon exactly, not its strains or "
        "sub-taxa) have a sequence (any isoform) heavier than the heaviest sequence (any isoform) of each of its proteins "
        "\"$ref_name\" and \"$ref_b_name\"? List each protein once."
    ),
    readings=(R_ORGANISM_SUBTAXA,),
    answer_columns=(AnswerColumn("protein", IRI_COL),),
    key_param="org",
    expand=_reference_pairs,
    anchors=(ORG_ANCHOR, _protein_anchor("ref"), _protein_anchor("ref_b")),
    difficulty={"hops": 5, "anchors": 3, "comparison": "reference_entities", "aggregate": "max", "populations": 2},
    slots=(("ref", "ref_name"), ("ref_b", "ref_b_name")),
    winner=_deciding_reference,
    step_check=StepCheck(
        "deciding_reference_matters",
        undo=_undo(_V[("numeric_filter", "heavier_than_reference")], {}, {"ref": "ref_b", "ref_name": "ref_b_name"}),
        mode="some",
    ),
)

# --- grouping constraints (one more HAVING constraint: at most having_max) ---

R2_TOP_KEYWORDS = uvariant(
    name="organism_top_keywords",
    tier="compositional",
    description=(
        "The k keywords on the most reviewed proteins of an organism among keywords on at most m of them, ties broken "
        "alphabetically (GROUP BY with HAVING, ORDER BY / LIMIT)."
    ),
    candidate_query=_V[("aggregation", "organism_top_keywords")].candidate_query,
    gold_pattern="""SELECT ?keyword ?proteins WHERE {
  {
    SELECT ?keyword ?label (COUNT(DISTINCT ?protein) AS ?proteins) WHERE {
      GRAPH $UNIPROT { ?protein a up:Protein ; up:reviewed true ; up:organism <$org> ; up:classifiedWith ?keyword . }
      GRAPH $KEYWORDS { ?keyword skos:prefLabel ?label . }
    }
    GROUP BY ?keyword ?label
    HAVING (COUNT(DISTINCT ?protein) <= $having_max)
  }
}
ORDER BY DESC(?proteins) ?label
LIMIT $top_k""",
    question_pattern=(
        "For the organism \"$org_name\" (recorded against that taxon exactly, not its strains or sub-taxa), consider only "
        "keyword annotations that at most $having_max of its reviewed proteins carry; of those, which $top_k are carried "
        "by the most of its reviewed proteins? Count only keywords assigned to the proteins directly; broader keywords "
        "above them do not count. List each keyword with its number of proteins; when keywords tie, take them in "
        "alphabetical order of their names."
    ),
    readings=(R_ORGANISM_SUBTAXA, R_BROADER_KEYWORDS),
    answer_columns=(AnswerColumn("keyword", IRI_COL), AnswerColumn("proteins", NUM_COL)),
    key_param="org",
    expand=_organisms,
    anchors=(ORG_ANCHOR,),
    knob_params=("top_k", "having_max"),
    difficulty={"hops": 2, "anchors": 1, "aggregate": "group_having_order_limit", "tie_break": "label", "grouping_constraints": 2},
    step_check=StepCheck("undo_changes", undo=_undo(_V[("aggregation", "organism_top_keywords")], {})),
)

R2_TAXON_HAVING = uvariant(
    name="taxon_organisms_having",
    tier="compositional",
    membership_mode="transitive",
    description="Organisms in a taxon's subtree with between n and m reviewed proteins each (GROUP BY / HAVING with two constraints).",
    candidate_query=_V[("aggregation", "taxon_organisms_having")].candidate_query,
    gold_pattern="""SELECT ?org (COUNT(DISTINCT ?protein) AS ?proteins) WHERE {
  GRAPH $TAXONOMY { ?org rdfs:subClassOf* <$taxon> . }
  GRAPH $UNIPROT { ?protein a up:Protein ; up:reviewed true ; up:organism ?org . }
}
GROUP BY ?org
HAVING (COUNT(DISTINCT ?protein) >= $having_min && COUNT(DISTINCT ?protein) <= $having_max)""",
    question_pattern=(
        "Among the taxon \"$taxon_name\" and all taxa below it at any depth, which have at least $having_min and at most $having_max "
        "reviewed proteins recorded directly against them (proteins of their own sub-taxa do not count)? List each taxon "
        "with its number of reviewed proteins."
    ),
    readings=(
        _direct(r"and all taxa below it at any depth", _TAXON_IMMEDIATE),
        _anchor_inclusion(r'taxon "[^"]+" and all taxa below it', _TAXON_WITHOUT_ANCHOR),
        R2_COUNTS_WITH_SUBTAXA,
    ),
    answer_columns=(AnswerColumn("taxon", IRI_COL), AnswerColumn("proteins", NUM_COL)),
    key_param="taxon",
    expand=_taxa,
    anchors=(TAXON_ANCHOR,),
    knob_params=("having_min", "having_max"),
    difficulty={"hops": 2, "anchors": 1, "aggregate": "group_having", "membership": "transitive", "grouping_constraints": 2},
    step_check=StepCheck("undo_changes", undo=_undo(_V[("aggregation", "taxon_organisms_having")], {})),
)

R2_LOCATION_COUNTS = uvariant(
    name="organism_location_counts",
    tier="compositional",
    description=(
        "Subcellular locations (by name) recorded for between n and m reviewed proteins of an organism (GROUP BY / HAVING "
        "with two constraints, across graphs)."
    ),
    candidate_query=_V[("cross_graph_join", "organism_location_counts")].candidate_query,
    gold_pattern="""SELECT ?locationName (COUNT(DISTINCT ?protein) AS ?proteins) WHERE {
  GRAPH $UNIPROT {
    ?protein a up:Protein ; up:reviewed true ; up:organism <$org> ; up:annotation ?annotation .
    ?annotation a up:Subcellular_Location_Annotation ; up:locatedIn ?located .
    ?located up:cellularComponent ?location .
  }
  GRAPH $LOCATIONS { ?location skos:prefLabel ?locationName . }
}
GROUP BY ?locationName
HAVING (COUNT(DISTINCT ?protein) >= $having_min && COUNT(DISTINCT ?protein) <= $having_max)""",
    question_pattern=(
        "For the organism \"$org_name\" (recorded against that taxon exactly, not its strains or sub-taxa), which "
        "subcellular locations, according to the subcellular-location annotation, are recorded for at least $having_min "
        "and at most $having_max of its reviewed proteins (each location counts only proteins annotated with that "
        "location itself; narrower locations do not count) " + LOCATION_SCOPE + "? List each location by name with its "
        "number of proteins."
    ),
    readings=(R_ORGANISM_SUBTAXA, R_NARROWER_LOCATIONS, R2_CHAIN_LOCATIONS),
    answer_columns=(AnswerColumn("location_name", LIT_COL), AnswerColumn("proteins", NUM_COL)),
    key_param="org",
    expand=_organisms,
    anchors=(ORG_ANCHOR,),
    knob_params=("having_min", "having_max"),
    difficulty={"hops": 5, "anchors": 1, "graphs": 2, "aggregate": "group_having", "grouping_constraints": 2},
    step_check=StepCheck("undo_changes", undo=_undo(_V[("cross_graph_join", "organism_location_counts")], {})),
)

# --- negation depth (one more negation nested inside the existing one) ---

R2_ACTIVITY = uvariant(
    name="activity_without_reaction",
    tier="compositional",
    description=(
        "Reviewed proteins of an organism with a catalytic activity, every one of whose catalytic activities is linked to "
        "a Rhea reaction (NOT EXISTS an activity with NOT EXISTS a reaction: negation depth 2)."
    ),
    candidate_query="""
SELECT ?org ?orgName WHERE {
""" + _SIZED_ORGANISMS + """
  FILTER EXISTS {
    GRAPH $UNIPROT { ?p up:organism ?org ; up:annotation ?a . ?a up:catalyticActivity ?ca . ?ca up:catalyzedReaction ?r . }
  }
}
ORDER BY ?org
""",
    # Quantified over activities, not annotations (code reviews of 99e6fd4): a protein is out when any of its
    # catalytic activities has no catalysed reaction, even if another activity of the same annotation has one.
    gold_pattern=_ACTIVITY_GOLD.replace("$PHYSIOLOGICAL", ""),
    question_pattern=(
        "Which reviewed proteins of the organism \"$org_name\" (recorded against that taxon exactly, not its strains or "
        "sub-taxa) have a catalytic activity annotated, and every one of their catalytic activities is linked to a "
        "catalysed reaction (an EC number alone does not count; the reactions as annotated, not their direction-specific "
        "forms)? List each protein once."
    ),
    readings=(R_ORGANISM_SUBTAXA, R2_PHYSIOLOGICAL_ACTIVITY, R2_SOME_ACTIVITY),
    answer_columns=(AnswerColumn("protein", IRI_COL),),
    key_param="org",
    expand=_organisms,
    anchors=(ORG_ANCHOR,),
    difficulty={"hops": 5, "anchors": 1, "negation": 2},
    # The nested negation must matter: the answer differs from the round-1 form and from the positive reading
    # without it (some activity linked), which would otherwise answer the question without any negation.
    step_check=StepCheck(
        "undo_and_positive_reading_change",
        undo=(*_undo(_V[("absence_negation", "activity_without_reaction")], {}), (R2_SOME_ACTIVITY.pattern, {})),
    ),
)

# --- path hops (one more edge at the anchor end) ---

_TAXON_REACTIONS_BODY = """
  GRAPH $TAXONOMY { ?org rdfs:subClassOf* <$taxon> . }
  GRAPH $UNIPROT {
    ?protein a up:Protein ; up:reviewed true ; up:organism ?org ; up:annotation ?annotation .
    ?annotation a up:Catalytic_Activity_Annotation ; up:catalyticActivity ?activity .
    ?activity up:catalyzedReaction ?reaction .
  }
"""

R2_TAXON_REACTIONS = uvariant(
    name="organism_reactions",
    tier="compositional",
    description="Rhea reactions catalysed by reviewed proteins of a taxon or any taxon below it (5 hops from the taxon).",
    candidate_query="""
SELECT ?taxon ?taxonName (COUNT(DISTINCT ?reaction) AS ?n) WHERE {
""" + _SIZED_SUBTREES + """
  GRAPH $TAXONOMY { ?taxon up:scientificName ?taxonName . ?org rdfs:subClassOf* ?taxon . }
  GRAPH $UNIPROT {
    ?protein a up:Protein ; up:organism ?org ; up:annotation ?annotation .
    ?annotation a up:Catalytic_Activity_Annotation ; up:catalyticActivity ?activity .
    ?activity up:catalyzedReaction ?reaction .
  }
}
GROUP BY ?taxon ?taxonName
HAVING (COUNT(DISTINCT ?reaction) <= 50 && COUNT(DISTINCT ?org) >= 2)
ORDER BY ?taxon
""",
    gold_pattern="SELECT DISTINCT ?reaction WHERE {" + _TAXON_REACTIONS_BODY + "}",
    question_pattern=(
        "Which reactions are catalysed by reviewed proteins of the taxon \"$taxon_name\" or of any taxon below it at any "
        "depth? List each reaction once (the reactions as annotated, not their direction-specific forms)."
    ),
    readings=(*_SUBTREE_CLAUSES, R_PHYSIOLOGICAL),
    answer_columns=(AnswerColumn("reaction", IRI_COL),),
    key_param="taxon",
    expand=_taxa,
    hops=5,
    anchors=(TAXON_ANCHOR,),
    membership_mode="transitive",
    difficulty={"hops": 5, "anchors": 1, "membership": "transitive"},
    step_check=StepCheck("fan_out", fan_out="SELECT DISTINCT ?org ?reaction WHERE {" + _TAXON_REACTIONS_BODY + "}"),
)

R2_TAXON_DISTINCT_REACTIONS = uvariant(
    name="organism_distinct_reactions",
    tier="compositional",
    description=(
        "Number of distinct Rhea reactions catalysed by reviewed proteins of a taxon or any taxon below it (distinct "
        "count across a 5-hop join)."
    ),
    candidate_query="""
SELECT ?taxon ?taxonName WHERE {
  {
    SELECT ?taxon WHERE {
""" + _SIZED_SUBTREES + """
      GRAPH $TAXONOMY { ?org rdfs:subClassOf* ?taxon . }
      GRAPH $UNIPROT { ?p up:organism ?org ; up:annotation ?a . ?a up:catalyticActivity ?ca . ?ca up:catalyzedReaction ?r . }
    }
    GROUP BY ?taxon
    HAVING (COUNT(DISTINCT ?org) >= 2)
  }
  GRAPH $TAXONOMY { ?taxon up:scientificName ?taxonName . }
}
ORDER BY ?taxon
""",
    gold_pattern="SELECT (COUNT(DISTINCT ?reaction) AS ?count) WHERE {" + _TAXON_REACTIONS_BODY + "}",
    question_pattern=(
        "How many different reactions are catalysed by the reviewed proteins of the taxon \"$taxon_name\" or of any taxon "
        "below it at any depth? Count each reaction once, however many proteins catalyse it (the reactions as annotated, not their "
        "direction-specific forms)."
    ),
    readings=(*_SUBTREE_CLAUSES, R_PHYSIOLOGICAL),
    answer_columns=(AnswerColumn("count", NUM_COL),),
    key_param="taxon",
    expand=_taxa,
    anchors=(TAXON_ANCHOR,),
    membership_mode="transitive",
    difficulty={"hops": 5, "anchors": 1, "aggregate": "count_distinct_join", "membership": "transitive"},
    step_check=StepCheck("fan_out", fan_out="SELECT DISTINCT ?org ?reaction WHERE {" + _TAXON_REACTIONS_BODY + "}"),
)

_TAXON_COMPOUND_BODY = """
  GRAPH $RHEA { ?reaction rh:side ?side . ?side rh:contains ?participant . ?participant rh:compound <$compound> . }
  GRAPH $TAXONOMY { ?org rdfs:subClassOf* <$taxon> . }
  GRAPH $UNIPROT {
    ?protein a up:Protein ; up:reviewed true ; up:organism ?org ; up:annotation ?annotation .
    ?annotation a up:Catalytic_Activity_Annotation ; up:catalyticActivity ?activity .
    ?activity up:catalyzedReaction ?reaction .
  }
"""

R2_TAXON_COMPOUND = uvariant(
    name="proteins_reacting_with_compound",
    tier="compositional",
    description=(
        "Reviewed proteins of a taxon or any taxon below it catalysing a reaction in which a Rhea compound takes part "
        "(uniprot -> Rhea join, 8 hops)."
    ),
    candidate_query="""
SELECT ?taxon ?taxonName ?compound ?compoundName WHERE {
  {
    SELECT ?taxon ?compound WHERE {
""" + _SIZED_SUBTREES + """
      GRAPH $TAXONOMY { ?org rdfs:subClassOf* ?taxon . }
      GRAPH $UNIPROT { ?protein up:organism ?org ; up:annotation ?annotation . ?annotation up:catalyticActivity ?activity . ?activity up:catalyzedReaction ?reaction . }
      GRAPH $RHEA { ?reaction rh:side ?side . ?side rh:contains ?participant . ?participant rh:compound ?compound . }
    }
    GROUP BY ?taxon ?compound
    HAVING (COUNT(DISTINCT ?protein) >= 2 && COUNT(DISTINCT ?protein) <= 40 && COUNT(DISTINCT ?org) >= 2)
  }
  GRAPH $TAXONOMY { ?taxon up:scientificName ?taxonName . }
  GRAPH $RHEA { ?compound rh:name ?compoundName . }
}
ORDER BY ?taxon ?compound
""",
    gold_pattern="SELECT DISTINCT ?protein WHERE {" + _TAXON_COMPOUND_BODY + "}",
    question_pattern=(
        "Which reviewed proteins of the taxon \"$taxon_name\" or of any taxon below it at any depth catalyse a reaction "
        "in which the "
        "compound \"$compound_name\" (as named in the reaction data) takes part, on either side (the reactions as "
        "annotated, not their direction-specific forms)? List each protein once."
    ),
    readings=(*_SUBTREE_CLAUSES, R_PHYSIOLOGICAL),
    answer_columns=(AnswerColumn("protein", IRI_COL),),
    key_param="taxon",
    expand=_taxon_compounds,
    anchors=(TAXON_ANCHOR, COMPOUND_ANCHOR),
    membership_mode="transitive",
    difficulty={"hops": 8, "anchors": 2, "graphs": 3, "membership": "transitive"},
    step_check=StepCheck("fan_out", fan_out="SELECT DISTINCT ?org ?protein WHERE {" + _TAXON_COMPOUND_BODY + "}"),
)

_KEYWORD_DISEASE_BODY = """
  GRAPH $DISEASES { ?disease rdfs:seeAlso <$kw> . }
  GRAPH $UNIPROT {
    ?protein a up:Protein ; up:reviewed true ; up:annotation ?annotation ; up:encodedBy ?gene .
    ?annotation a up:Disease_Annotation ; up:disease ?disease .
    ?gene skos:prefLabel ?geneName .
  }
"""

R2_KEYWORD_DISEASE_GENES = uvariant(
    name="disease_genes",
    tier="compositional",
    description=(
        "Proteins linked to a disease whose disease entry refers to a keyword, with the primary names of their genes "
        "(5 hops from the keyword)."
    ),
    candidate_query="""
SELECT ?kw ?kwLabel (COUNT(DISTINCT ?protein) AS ?n) WHERE {
  GRAPH $DISEASES { ?disease rdfs:seeAlso ?kw . FILTER(STRSTARTS(STR(?kw), "http://purl.uniprot.org/keywords/")) }
  GRAPH $UNIPROT {
    ?protein a up:Protein ; up:annotation ?annotation ; up:encodedBy ?gene .
    ?annotation a up:Disease_Annotation ; up:disease ?disease .
    ?gene skos:prefLabel ?geneName .
  }
  GRAPH $KEYWORDS { ?kw skos:prefLabel ?kwLabel . }
}
GROUP BY ?kw ?kwLabel
HAVING (COUNT(DISTINCT ?protein) <= 50 && COUNT(DISTINCT ?disease) >= 2)
ORDER BY ?kw
""",
    gold_pattern="SELECT DISTINCT ?protein ?geneName WHERE {" + _KEYWORD_DISEASE_BODY + "}",
    question_pattern=(
        "Which reviewed proteins are linked to a disease whose disease entry refers to the keyword \"$kw_label\" (the "
        "keyword itself; narrower keywords do not count, nor do keywords assigned to the proteins themselves)? List each "
        "protein with the primary name of a gene that encodes it."
    ),
    readings=(R_GENE_SYNONYMS, R2_DISEASE_NARROWER_KEYWORDS, R2_PROTEIN_KEYWORD),
    answer_columns=(AnswerColumn("protein", IRI_COL), AnswerColumn("gene_name", LIT_COL)),
    key_param="kw",
    expand=_keywords,
    hops=5,
    anchors=(KW_ANCHOR,),
    difficulty={"hops": 5, "anchors": 1},
    step_check=StepCheck("fan_out", fan_out="SELECT DISTINCT ?disease ?protein ?geneName WHERE {" + _KEYWORD_DISEASE_BODY + "}"),
)


def _step(template: type[QuestionTemplate], raised: Variant, axis: str, construct: str, r1: int, hops: tuple[int, int], note: str = "") -> tuple[str, StructuralStep]:
    key = f"{template.id}/{raised.name}"
    if _V[(template.id, raised.name)].tier != "compositional":
        raise ValueError(f"{key}: only compositional variants take a step")
    return key, StructuralStep(axis, construct, r1, r1 + 1, raised, hops[0], hops[1], note)


# variant -> axis -> round-1 value -> round-2 value. Hops are recounted from the gold text (edges joining the
# anchors and the answer; generate.STEP_RULES); the round-1 declared value is noted where it differs.
STEPS: dict[str, StructuralStep] = dict(
    (
        _step(TypeMembers, R2_TAXON_GO, "populations", "set_operation=intersection", 2, (4, 6),
              "third population: a second GO term with its is-a subtree; round-1 declared hops 3"),
        _step(TypeMembers, R2_KEYWORD_UNION, "populations", "set_operation=union", 2, (3, 4),
              "third population: a third keyword; round-1 declared hops 1"),
        _step(OneHopAttribute, R2_SHARED_KEYWORDS, "populations", "set_operation=intersection", 2, (2, 3),
              "third population: a third protein's keywords"),
        _step(OneHopAttribute, R2_EARLIER_ENTRY, "populations", "comparison of two entities", 2, (1, 1),
              "comparison, not a set operation: one more protein compared; all three creation dates differ"),
        _step(MultiHopPath, R2_TAXON_REACTIONS, "path_hops", "path", 4, (4, 5),
              "anchor organism (exact) -> taxon and its subtree (rdfs:subClassOf*)"),
        _step(MultiHopPath, R2_KEYWORD_DISEASE_GENES, "path_hops", "path", 4, (4, 5),
              "anchor disease -> a keyword the disease entries refer to (rdfs:seeAlso)"),
        _step(Aggregation, R2_TOP_KEYWORDS, "grouping_constraints", "aggregate=group_order_limit", 1, (2, 2),
              "added HAVING: at most having_max proteins per keyword"),
        _step(Aggregation, R2_TAXON_HAVING, "grouping_constraints", "aggregate=group_having", 1, (2, 2),
              "added HAVING: at most having_max proteins per taxon"),
        _step(Aggregation, R2_TAXON_DISTINCT_REACTIONS, "path_hops", "aggregate=count_distinct_join (a path)", 4, (4, 5),
              "anchor organism (exact) -> taxon and its subtree (rdfs:subClassOf*)"),
        _step(NumericFilter, R2_BETTER_RESOLVED, "populations", "comparison of two entities", 2, (2, 2),
              "comparison, not a set operation: one more protein compared; all three best resolutions differ"),
        _step(NumericFilter, R2_HEAVIER, "populations", "comparison with a reference entity", 1, (5, 5),
              "comparison, not a set operation: one more reference protein (heavier than both); round-1 declared hops 2"),
        _step(AbsenceNegation, R2_ACTIVITY, "negation_depth", "negation", 1, (5, 5),
              "every activity linked: NOT EXISTS (activity, NOT EXISTS reaction); round-1 declared hops 4"),
        _step(AbsenceNegation, R2_KEYWORD_DIFFERENCE, "populations", "set_operation=difference (also negation)", 2, (3, 4),
              "third population: a second subtracted keyword; negation depth stays 1 (set operation takes precedence); "
              "round-1 declared hops 1"),
        _step(CrossGraphJoin, R2_TAXON_COMPOUND, "path_hops", "path (cross-graph join)", 7, (7, 8),
              "anchor organism (exact) -> taxon and its subtree (rdfs:subClassOf*); round-1 declared hops 6"),
        _step(CrossGraphJoin, R2_LOCATION_COUNTS, "grouping_constraints", "aggregate=group_having", 1, (5, 5),
              "added HAVING: at most having_max proteins per location; round-1 declared hops 4"),
    )
)

NOTES = {
    "anchors": (
        "Anchors come from instances and links in the uniprot graph and their counts (organisms by number of reviewed "
        "proteins, taxa by subtree size, keywords and GO terms by use, keyword pairs that co-occur in one organism), "
        "never from core.owl axioms. Each is named by a label unique for its kind (checked per question): protein = "
        "recommended name + organism, taxon = scientific name, GO term / keyword / disease = label, Rhea compound = "
        "Rhea name."
    ),
    "closure": (
        "rdfs:subClassOf is materialised (every ancestor asserted) in the taxonomy, keywords, enzymes, pathways and "
        "locations graphs; direct and transitive readings of rdfs:subClassOf therefore differ only by the taxon itself "
        "(zero-length path) there."
    ),
    "taxon_parent": (
        "The immediate parent of a taxon is ?parent skos:narrowerTransitive ?child (exactly one per taxon), despite "
        "the property's name; 'directly below' questions use it."
    ),
    "go": (
        "GO rdfs:subClassOf holds the is-a closure; part-of parents appear only as owl:Restriction IRIs "
        "(GO_x#part-of-parent-n) without a further rdfs:subClassOf, so GO questions follow is-a only."
    ),
    "nested_graph": (
        "QLever evaluates a GRAPH clause nested inside another GRAPH clause (directly or in a subquery) against the "
        "outer graph, unlike the SPARQL semantics; golds therefore never nest GRAPH clauses, and the alternative "
        "readings put their hierarchy patterns next to, not inside, the GRAPH group they extend."
    ),
    "datatypes": (
        "QLever returns xsd:double values (structure resolution) typed xsd:decimal, and COUNT and casts as xsd:int; "
        "the scorer compares numbers by value, so answers do not change."
    ),
    "reviewed": "Every entry in the subset is reviewed; golds still state up:reviewed true and up:Protein.",
    "rhea_links": "up:catalyzedReaction points at Rhea master reactions (14,163 distinct), all described in the Rhea graph.",
    "triples": (
        "Raised variants that name three keywords or proteins of one organism draw them from the organism's first "
        f"{TRIPLE_POOL} candidates in a stable scrambled order (hash of organism and IRI), then take at most "
        "max_pair_options valid triples; pairs (GO terms, reference proteins) use every combination."
    ),
    "disease_keywords": (
        "Disease entries refer to UniProt keywords with rdfs:seeAlso in the diseases graph (5,156 links); the raised "
        "disease question is anchored at such a keyword."
    ),
}

# The frozen round-1 file: its dev questions were the calibration pool, whose results were seen, so no round-2 test
# question may name an anchor that a round-1 dev question of the same variant named (design ruling, design log).
ROUND_1_QUESTIONS_SHA256 = "b805acd905b2fde99d4304e5e10c8c6050031477382a5059545d7c92a8f87ffe"
# Round 2 as the study lead audited it (99e6fd4); the shipped set's changes since, checked against git in tests.
AUDITED_ROUND_2_SHA256 = "d1278589be9670a172d3fd5f631dc567dcefaab6d529e774a4072517871e6991"
ROUND_2_REVISION = {
    "applies_to": {"seed": 20261018, "per_template": 43},
    "audited_questions_sha256": AUDITED_ROUND_2_SHA256,
    "audited_commit": "99e6fd4",
    "gold_sparql_changed": {
        "total": 51,
        "rows_unchanged": {
            "total": 43,
            "by_variant": {"activity_without_reaction": 15, "earlier_entry": 12, "better_resolved_protein": 11,
                           "heavier_than_reference": 5},
        },
        "redrawn": 8,
    },
    "note": (
        "Versus round 2 as audited (d1278589..., 99e6fd4), 51 questions have a different gold_sparql. 43 keep their "
        "gold rows: the 15 activity_without_reaction questions (the every-clause quantifies over activities, and the "
        "outer pattern requires an annotated activity) and the 28 comparison questions whose entities are now "
        "mentioned in a seeded order that balances the winner's slot (12 earlier_entry, 11 better_resolved_protein, "
        "5 heavier_than_reference; same entities and rows). The other 8 are test questions redrawn off round-1 dev "
        "anchors (prior_round_anchors.redrawn): new anchors, texts, gold queries and rows. Question texts and "
        "alternative readings also changed with the depth, chain-scope and activity wording, and every record "
        "carries generator_version 0.3.0."
    ),
}

PROFILE = Profile(
    graph_id=GRAPH_ID,
    id_prefix=ID_PREFIX,
    templates=TEMPLATES,
    store_identity={
        "engine": "QLever 9bebc60",
        "image": "qlever-9bebc60c51.sif",
        "store": "uniprot-sp-2026_03 (UniProtKB/Swiss-Prot 2026_03 subset with Rhea 142 and ChEBI)",
    },
    two_tier=True,
    lookup_share=1 / 3,
    knobs=KNOBS,
    expected_counts=None,
    notes=NOTES,
    forbid_database_names=True,
    global_answer_uniqueness=True,
    steps=STEPS,
    calibration_rounds=ROUND_DELTAS,
    # Round 1 was generated as 0.2.0 (the frozen file); round 2 is 0.3.0 (study lead's audit, experiments#12).
    round_versions={1: "0.2.0", 2: "0.3.0"},
    prior_dev_rounds={2: (1, ROUND_1_QUESTIONS_SHA256)},
    revisions={2: ROUND_2_REVISION},
)
