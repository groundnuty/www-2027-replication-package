"""The seven question templates (the paper's fixed list).

Conventions every template follows:

- Reactions are master (undirected) reactions, ``rdfs:subClassOf rh:Reaction``,
  with status Approved or Preliminary. Obsolete reactions have no label,
  equation or participants, so no candidate query can return them.
- A Rhea compound is named by its ``rh:name`` and its accession (CHEBI:...,
  GENERIC:... or POLYMER:...); a ChEBI class by its ``rdfs:label`` and CHEBI
  CURIE; a reaction by its accession and its label (the equation). Each
  question says which IRI to return, since CHEBI:15377 names both a Rhea
  compound and a ChEBI class.
- ChEBI hierarchy steps are scoped to the ChEBI named graph, so Rhea compounds,
  participants and reactive parts (which the Rhea graph places under ChEBI
  classes) never count as ChEBI subclasses.
- Participants are reached through ``rh:contains``, which the data
  materialises next to the coefficient-specific ``rh:containsN`` properties.
- A template's membership mode names the subclass relation its questions
  ask about: a "transitive" gold uses ``rdfs:subClassOf+`` for it, a "direct"
  gold plain ``rdfs:subClassOf``, an "n.a." gold neither. Separately, any gold
  that returns or counts reactions restricts them to master reactions with
  the type test ``?reaction rdfs:subClassOf rh:Reaction``. Every master
  reaction is asserted as a direct subclass of ``rh:Reaction`` (the direct
  and transitive forms select the same 18,907 reactions), so the type test
  does not depend on the membership mode.
- Where the data allows, a variant's candidates carry an answer shape
  ("single" or "multi" values) and sampling alternates between shapes.
- Hop counts run from the named entity to the answer node and do not count
  the label or accession used to name it.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Iterable

from harness.qgen.base import (
    AnswerColumn,
    EngineProbe,
    PoolRow,
    QuestionTemplate,
    Variant,
    check_iri,
    chebi_curie,
    usable_label,
)

IRI_COL = "iri"
LIT_COL = "literal"
NUM_COL = "number"

# Comparison operators and their English wording, in a fixed order.
COMPARISONS: tuple[tuple[str, str], ...] = (
    ("<", "less than {v}"),
    ("<=", "of at most {v}"),
    (">", "greater than {v}"),
    (">=", "of at least {v}"),
    ("=", "of exactly {v}"),
)
_COMPARE = {
    "<": lambda x, v: x < v,
    "<=": lambda x, v: x <= v,
    ">": lambda x, v: x > v,
    ">=": lambda x, v: x >= v,
    "=": lambda x, v: x == v,
}


# --------------------------------------------------------------------------
# Row -> parameter helpers


def _hint(row: PoolRow) -> dict[str, Any]:
    """A candidate query's optional ``hint`` column becomes the candidate's sampling priority."""
    return {"_priority": int(row["hint"])} if row.get("hint") not in (None, "", "0") else {}


def _chebi_classes(rows: list[PoolRow]) -> Iterable[dict[str, Any]]:
    for row in rows:
        if usable_label(row["label"]):
            yield {
                "entity": check_iri(row["class"]),
                "label": row["label"],
                "curie": chebi_curie(row["class"]),
                **_hint(row),
            }


def _reactions(rows: list[PoolRow]) -> Iterable[dict[str, Any]]:
    for row in rows:
        if usable_label(row["label"]):
            yield {"entity": check_iri(row["reaction"]), "accession": row["accession"], "label": row["label"]}


def _compounds(rows: list[PoolRow]) -> Iterable[dict[str, Any]]:
    for row in rows:
        if usable_label(row["name"]):
            yield {"entity": check_iri(row["compound"]), "accession": row["accession"], "name": row["name"]}


def _shape(n: Any) -> str:
    return "multi" if int(n) > 1 else "single"


def _reactions_by_shape(rows: list[PoolRow]) -> Iterable[dict[str, Any]]:
    """Reactions, with the number of answer values (column ``n``) as the shape."""
    for row, params in _with_rows(rows, _reactions):
        yield {**params, "_stratum": _shape(row["n"])}


def _compounds_by_shape(rows: list[PoolRow]) -> Iterable[dict[str, Any]]:
    for row, params in _with_rows(rows, _compounds):
        yield {**params, "_stratum": _shape(row["n"])}


def _min_charge_reactions(rows: list[PoolRow]) -> Iterable[dict[str, Any]]:
    """Reactions, shaped by whether some compound lacks an integer charge (count and MIN then differ in scope)."""
    for row, params in _with_rows(rows, _reactions):
        mixed = int(row["all"]) > int(row["withCharge"])
        yield {**params, "_stratum": "some_uncharged" if mixed else "all_charged"}


def _with_rows(rows: list[PoolRow], fn) -> Iterable[tuple[PoolRow, dict[str, Any]]]:
    """Pairs of (row, params) for the rows that ``fn`` keeps."""
    for row in rows:
        for params in fn([row]):
            yield row, params


def _comparison(op: str, value: int) -> dict[str, Any]:
    wording = dict(COMPARISONS)[op]
    return {"op": op, "threshold": int(value), "comparison": wording.format(v=int(value))}


def _threshold_options(counts: "OrderedDict[int, int]", min_rows: int = 1, max_rows: int = 50) -> list[dict[str, Any]]:
    """Every comparison whose result is a proper, non-empty subset of sizes in range.

    ``counts`` maps each value present to the number of entities with it.
    Proper subset: at least one entity with a value is filtered out, so the
    filter does some work.
    """
    total = sum(counts.values())
    options = []
    for value in counts:
        for op, _ in COMPARISONS:
            n = sum(c for x, c in counts.items() if _COMPARE[op](x, value))
            if min_rows <= n <= max_rows and n < total:
                options.append(_comparison(op, value))
    return options


def _reaction_participant_thresholds(rows: list[PoolRow]) -> Iterable[dict[str, Any]]:
    """Per reaction, every threshold that keeps a proper, non-empty subset of its compounds."""
    by_reaction: "OrderedDict[str, tuple[dict[str, Any], OrderedDict[int, int]]]" = OrderedDict()
    for r in rows:
        meta, counts = by_reaction.setdefault(r["reaction"], (r, OrderedDict()))
        counts[int(r["charge"])] = int(r["n"])
    for reaction, (meta, counts) in by_reaction.items():
        if len(counts) < 2:
            continue
        for base in _reactions([meta]):
            for option in _threshold_options(counts):
                yield {**base, **option}


def _chebi_children_thresholds(rows: list[PoolRow]) -> Iterable[dict[str, Any]]:
    by_class: "OrderedDict[str, tuple[PoolRow, OrderedDict[int, int]]]" = OrderedDict()
    for r in rows:
        first, counts = by_class.setdefault(r["class"], (r, OrderedDict()))
        counts[int(r["charge"])] = int(r["n"])
    for cls, (first, counts) in by_class.items():
        label = first["label"]
        if len(counts) < 2 or not usable_label(label):
            continue
        base = {"entity": check_iri(cls), "label": label, "curie": chebi_curie(cls), **_hint(first)}
        for option in _threshold_options(counts):
            yield {**base, **option}


# --------------------------------------------------------------------------
# Shared candidate queries

_MASTER_REACTIONS = """
SELECT ?reaction ?accession ?label WHERE {
  GRAPH $RHEA {
    ?reaction rdfs:subClassOf rh:Reaction ;
              rh:accession ?accession ;
              rdfs:label ?label .
  }
}
ORDER BY ?reaction
"""

_PARTICIPATING_COMPOUNDS = """
SELECT DISTINCT ?compound ?accession ?name WHERE {
  GRAPH $RHEA {
    ?participant rh:compound ?compound .
    ?compound rh:accession ?accession ;
              rh:name ?name .
  }
}
ORDER BY ?compound
"""

# ChEBI classes with at least one direct ChEBI subclass that a Rhea
# small-molecule compound links to. Hint: a grandchild is linked too, so the
# transitive reading of "direct subclass" may give a different answer.
_CHEBI_PARENTS_OF_RHEA_COMPOUNDS = """
SELECT DISTINCT ?class ?label ?hint WHERE {
  GRAPH $RHEA {
    ?compound rdfs:subClassOf rh:SmallMolecule ;
              rh:chebi ?child .
    ?participant rh:compound ?compound .
  }
  GRAPH $CHEBI {
    ?child rdfs:subClassOf ?class .
    ?class rdfs:label ?label .
  }
  OPTIONAL {
    SELECT DISTINCT ?class (1 AS ?hint) WHERE {
      GRAPH $RHEA {
        ?hintCompound rdfs:subClassOf rh:SmallMolecule ;
                      rh:chebi ?hintGrandchild .
        ?hintParticipant rh:compound ?hintCompound .
      }
      GRAPH $CHEBI { ?hintGrandchild rdfs:subClassOf ?hintChild . ?hintChild rdfs:subClassOf ?class . }
    }
  }
}
ORDER BY ?class
"""


# --------------------------------------------------------------------------
# 1. type_members


class TypeMembers(QuestionTemplate):
    id = "type_members"
    description = (
        "Instances of a type. Rhea and ChEBI model reactions and compounds as classes, so the members of a "
        "type are its subclasses; this template asks for all descendants at any depth (rdfs:subClassOf+). "
        "Only types whose descendants include at least one grandchild are used, so a direct-subclass answer "
        "is wrong."
    )
    membership_mode = "transitive"
    variants = (
        Variant(
            name="chebi_descendants",
            description="All ChEBI classes below a ChEBI class, at any depth, within the ChEBI graph.",
            candidate_query="""
SELECT ?class ?label ?hint WHERE {
  {
    SELECT ?class (COUNT(DISTINCT ?d) AS ?n) WHERE {
      GRAPH $CHEBI { ?d rdfs:subClassOf+ ?class . }
      FILTER(isIRI(?class) && isIRI(?d))
    }
    GROUP BY ?class
    HAVING (COUNT(DISTINCT ?d) <= 50)
  }
  {
    SELECT ?class (COUNT(DISTINCT ?c) AS ?k) WHERE {
      GRAPH $CHEBI { ?c rdfs:subClassOf ?class . }
      FILTER(isIRI(?class) && isIRI(?c))
    }
    GROUP BY ?class
  }
  FILTER(?n > ?k)
  GRAPH $CHEBI { ?class rdfs:label ?label . }
  OPTIONAL {
    SELECT DISTINCT ?class (1 AS ?hint) WHERE {
      GRAPH $RHEA { ?rheaIri rdfs:subClassOf ?below . }
      GRAPH $CHEBI { ?below rdfs:subClassOf* ?class . }
    }
  }
}
ORDER BY ?class
""",
            gold_pattern="""SELECT DISTINCT ?class WHERE {
  GRAPH $CHEBI { ?class rdfs:subClassOf+ <$entity> . }
  FILTER(isIRI(?class))
}
ORDER BY ?class""",
            question_pattern=(
                'Which ChEBI classes are descendants of the ChEBI class "$label" ($curie) at any depth of the '
                "ChEBI subclass hierarchy, that is, its subclasses, their subclasses, and so on? Do not include "
                "$curie itself, and count only ChEBI classes, not Rhea compounds. "
                "Return one row per class: the ChEBI class IRI."
            ),
            answer_columns=(AnswerColumn("class", IRI_COL),),
            key_param="entity",
            expand=_chebi_classes,
            # Rhea IRIs under the class make the union-graph reading differ.
            prefer=("graph_scope",),
        ),
        Variant(
            name="reaction_descendants",
            description="All master reactions below a master reaction in the Rhea reaction hierarchy, at any depth.",
            candidate_query="""
SELECT ?reaction ?accession ?label WHERE {
  {
    SELECT ?reaction (COUNT(DISTINCT ?d) AS ?n) WHERE {
      GRAPH $RHEA {
        ?reaction rdfs:subClassOf rh:Reaction .
        ?d rdfs:subClassOf+ ?reaction ; rdfs:subClassOf rh:Reaction .
      }
    }
    GROUP BY ?reaction
    HAVING (COUNT(DISTINCT ?d) <= 50)
  }
  {
    SELECT ?reaction (COUNT(DISTINCT ?c) AS ?k) WHERE {
      GRAPH $RHEA {
        ?reaction rdfs:subClassOf rh:Reaction .
        ?c rdfs:subClassOf ?reaction ; rdfs:subClassOf rh:Reaction .
      }
    }
    GROUP BY ?reaction
  }
  FILTER(?n > ?k)
  GRAPH $RHEA { ?reaction rh:accession ?accession ; rdfs:label ?label . }
}
ORDER BY ?reaction
""",
            gold_pattern="""SELECT DISTINCT ?reaction WHERE {
  GRAPH $RHEA {
    ?reaction rdfs:subClassOf+ <$entity> ;
              rdfs:subClassOf rh:Reaction .
  }
}
ORDER BY ?reaction""",
            question_pattern=(
                'Which Rhea reactions are descendants of the reaction $accession ("$label") at any depth of the '
                "Rhea reaction hierarchy, that is, its more specific subclass reactions, their subclasses, and so "
                "on? Do not include $accession itself, and count only undirected (master) reactions, not "
                "directional or bidirectional variants. Return one row per reaction: the reaction IRI."
            ),
            answer_columns=(AnswerColumn("reaction", IRI_COL),),
            key_param="entity",
            expand=_reactions,
        ),
    )


# --------------------------------------------------------------------------
# 2. one_hop_attribute


class OneHopAttribute(QuestionTemplate):
    id = "one_hop_attribute"
    description = (
        "An attribute one hop from the named entity: a reaction's EC numbers or citations (one or several "
        "values), a Rhea compound's formula, or a ChEBI class's InChIKey. (Neither graph holds masses, so mass "
        "is not asked.)"
    )
    membership_mode = "n.a."
    variants = (
        Variant(
            name="reaction_ec",
            description="EC numbers of a master reaction; reactions with one and with several EC numbers alternate.",
            candidate_query="""
SELECT ?reaction ?accession ?label (COUNT(DISTINCT ?ec) AS ?n) WHERE {
  GRAPH $RHEA {
    ?reaction rdfs:subClassOf rh:Reaction ;
              rh:ec ?ec ;
              rh:accession ?accession ;
              rdfs:label ?label .
  }
}
GROUP BY ?reaction ?accession ?label
ORDER BY ?reaction
""",
            gold_pattern="""SELECT DISTINCT ?ec WHERE {
  GRAPH $RHEA { <$entity> rh:ec ?ec . }
}
ORDER BY ?ec""",
            question_pattern=(
                'Which EC numbers does Rhea assign to the reaction $accession ("$label")? '
                "Return one row per EC number: the EC number IRI."
            ),
            answer_columns=(AnswerColumn("ec", IRI_COL),),
            key_param="entity",
            expand=_reactions_by_shape,
            hops=1,
        ),
        Variant(
            name="reaction_citations",
            description="Publications a master reaction cites; reactions with one and with several alternate.",
            candidate_query="""
SELECT ?reaction ?accession ?label (COUNT(DISTINCT ?citation) AS ?n) WHERE {
  GRAPH $RHEA {
    ?reaction rdfs:subClassOf rh:Reaction ;
              rh:citation ?citation ;
              rh:accession ?accession ;
              rdfs:label ?label .
  }
}
GROUP BY ?reaction ?accession ?label
ORDER BY ?reaction
""",
            gold_pattern="""SELECT DISTINCT ?citation WHERE {
  GRAPH $RHEA { <$entity> rh:citation ?citation . }
}
ORDER BY ?citation""",
            question_pattern=(
                'Which publications does Rhea cite for the reaction $accession ("$label")? '
                "Return one row per publication: the PubMed IRI."
            ),
            answer_columns=(AnswerColumn("citation", IRI_COL),),
            key_param="entity",
            expand=_reactions_by_shape,
            hops=1,
        ),
        Variant(
            name="compound_formula",
            description=(
                "Formula of a Rhea small-molecule compound. Compounds whose Rhea formula differs from the "
                "ChEBI formula of their class (9 in release 142) are left out, so the answer does not depend "
                "on which graph is read."
            ),
            candidate_query="""
SELECT DISTINCT ?compound ?accession ?name WHERE {
  GRAPH $RHEA {
    ?compound rdfs:subClassOf rh:SmallMolecule ;
              rh:accession ?accession ;
              rh:name ?name ;
              rh:formula ?formula ;
              rh:chebi ?chebi .
  }
  GRAPH $CHEBI { ?chebi chemrof:generalized_empirical_formula ?chebiFormula . }
  FILTER(STR(?formula) = STR(?chebiFormula))
}
ORDER BY ?compound
""",
            gold_pattern="""SELECT DISTINCT ?formula WHERE {
  GRAPH $RHEA { <$entity> rh:formula ?formula . }
}""",
            question_pattern=(
                'What chemical formula does Rhea record for the compound "$name" ($accession)? '
                "Return one row: the formula as a text value."
            ),
            answer_columns=(AnswerColumn("formula", LIT_COL),),
            key_param="entity",
            expand=_compounds,
            hops=1,
        ),
        Variant(
            name="chebi_inchikey",
            description="InChIKey of a ChEBI class that a Rhea small-molecule compound links to.",
            candidate_query="""
SELECT DISTINCT ?class ?label WHERE {
  GRAPH $RHEA { ?compound rdfs:subClassOf rh:SmallMolecule ; rh:chebi ?class . }
  GRAPH $CHEBI { ?class rdfs:label ?label ; chemrof:inchi_key_string ?key . }
}
ORDER BY ?class
""",
            gold_pattern="""SELECT DISTINCT ?inchikey WHERE {
  GRAPH $CHEBI { <$entity> chemrof:inchi_key_string ?inchikey . }
}""",
            question_pattern=(
                'What is the InChIKey of the ChEBI class "$label" ($curie), as recorded in ChEBI? '
                "Return one row: the InChIKey as a text value."
            ),
            answer_columns=(AnswerColumn("inchikey", LIT_COL),),
            key_param="entity",
            expand=_chebi_classes,
            hops=1,
        ),
    )


# --------------------------------------------------------------------------
# 3. multi_hop_path


class MultiHopPath(QuestionTemplate):
    id = "multi_hop_path"
    description = (
        "A path of two or three hops: reaction -> side -> participant -> compound (3), its inverse (3), "
        "reaction -> directional or bidirectional variant -> cross-reference (2), generic compound -> reactive "
        "part -> ChEBI class (2)."
    )
    membership_mode = "n.a."
    variants = (
        Variant(
            name="reaction_compounds",
            description="Compounds on either side of a master reaction (reaction -> side -> participant -> compound).",
            candidate_query=_MASTER_REACTIONS,
            gold_pattern="""SELECT DISTINCT ?compound WHERE {
  GRAPH $RHEA {
    <$entity> rh:side ?side .
    ?side rh:contains ?participant .
    ?participant rh:compound ?compound .
  }
}
ORDER BY ?compound""",
            question_pattern=(
                'Which compounds take part in the Rhea reaction $accession ("$label"), on either side of the '
                "reaction? Return one row per compound: the Rhea compound IRI (not the IRI of its ChEBI class)."
            ),
            answer_columns=(AnswerColumn("compound", IRI_COL),),
            key_param="entity",
            expand=_reactions,
            hops=3,
        ),
        Variant(
            name="compound_reactions",
            description="Master reactions a compound takes part in (compound <- participant <- side <- reaction).",
            candidate_query=_PARTICIPATING_COMPOUNDS,
            gold_pattern="""SELECT DISTINCT ?reaction WHERE {
  GRAPH $RHEA {
    ?participant rh:compound <$entity> .
    ?side rh:contains ?participant .
    ?reaction rh:side ?side ;
              rdfs:subClassOf rh:Reaction .
  }
}
ORDER BY ?reaction""",
            question_pattern=(
                'In which Rhea reactions does the compound "$name" ($accession) take part, on either side? '
                "Count only undirected (master) reactions, not their directional or bidirectional variants. "
                "Return one row per reaction: the reaction IRI."
            ),
            answer_columns=(AnswerColumn("reaction", IRI_COL),),
            key_param="entity",
            expand=_compounds,
            hops=3,
        ),
        Variant(
            name="variant_cross_references",
            description=(
                "Cross-references of the directional and bidirectional variants of a master reaction "
                "(reaction -> variant -> cross-reference); one and several alternate."
            ),
            candidate_query="""
SELECT ?reaction ?accession ?label (COUNT(*) AS ?n) WHERE {
  GRAPH $RHEA {
    ?reaction rdfs:subClassOf rh:Reaction ;
              rh:accession ?accession ;
              rdfs:label ?label .
    { ?reaction rh:directionalReaction ?variant . } UNION { ?reaction rh:bidirectionalReaction ?variant . }
    ?variant rdfs:seeAlso ?xref .
  }
}
GROUP BY ?reaction ?accession ?label
ORDER BY ?reaction
""",
            gold_pattern="""SELECT DISTINCT ?variant ?xref WHERE {
  GRAPH $RHEA {
    { <$entity> rh:directionalReaction ?variant . } UNION { <$entity> rh:bidirectionalReaction ?variant . }
    ?variant rdfs:seeAlso ?xref .
  }
}
ORDER BY ?variant ?xref""",
            question_pattern=(
                'Which cross-references to other databases do the directional (left-to-right, right-to-left) and '
                'bidirectional variants of the Rhea reaction $accession ("$label") carry? Return one row per '
                "variant and cross-reference: the variant reaction IRI, then the cross-reference IRI."
            ),
            answer_columns=(AnswerColumn("variant_reaction", IRI_COL), AnswerColumn("xref", IRI_COL)),
            key_param="entity",
            expand=_reactions_by_shape,
            hops=2,
        ),
        Variant(
            name="generic_reactive_parts",
            description=(
                "ChEBI classes of the reactive parts of a generic compound; generics with one and with several "
                "reactive-part classes alternate."
            ),
            candidate_query="""
SELECT ?compound ?accession ?name (COUNT(DISTINCT ?chebi) AS ?n) WHERE {
  GRAPH $RHEA {
    ?compound rh:reactivePart ?part ;
              rh:accession ?accession ;
              rh:name ?name .
    ?part rh:chebi ?chebi .
  }
}
GROUP BY ?compound ?accession ?name
ORDER BY ?compound
""",
            gold_pattern="""SELECT DISTINCT ?chebi WHERE {
  GRAPH $RHEA {
    <$entity> rh:reactivePart ?part .
    ?part rh:chebi ?chebi .
  }
}
ORDER BY ?chebi""",
            question_pattern=(
                'The Rhea generic compound "$name" ($accession) has one or more reactive parts. Which ChEBI '
                "classes do these reactive parts correspond to? Return one row per ChEBI class: the ChEBI class IRI."
            ),
            answer_columns=(AnswerColumn("chebi_class", IRI_COL),),
            key_param="entity",
            expand=_compounds_by_shape,
            hops=2,
        ),
    )


# --------------------------------------------------------------------------
# 4. aggregation


class Aggregation(QuestionTemplate):
    id = "aggregation"
    description = (
        "COUNT, optionally with one other aggregate (MIN), returning a single row. Integer results only, so "
        "the default tolerances apply."
    )
    membership_mode = "n.a."
    variants = (
        Variant(
            name="compound_reaction_count",
            description="Number of master reactions a compound takes part in.",
            candidate_query=_PARTICIPATING_COMPOUNDS,
            gold_pattern="""SELECT (COUNT(DISTINCT ?reaction) AS ?count) WHERE {
  GRAPH $RHEA {
    ?participant rh:compound <$entity> .
    ?side rh:contains ?participant .
    ?reaction rh:side ?side ;
              rdfs:subClassOf rh:Reaction .
  }
}""",
            question_pattern=(
                'In how many Rhea reactions does the compound "$name" ($accession) take part, on either side? '
                "Count each undirected (master) reaction once and do not count directional or bidirectional "
                "variants. Return one row: the number of reactions."
            ),
            answer_columns=(AnswerColumn("count", NUM_COL),),
            key_param="entity",
            expand=_compounds,
        ),
        Variant(
            name="reaction_count_min_charge",
            description=(
                "Number of distinct compounds in a master reaction (all compounds, charged or not) and the lowest "
                "integer charge among those that have one; reactions with and without uncharged (generic or "
                "polymer) compounds alternate."
            ),
            candidate_query="""
SELECT ?reaction ?accession ?label (COUNT(DISTINCT ?compound) AS ?all) (COUNT(DISTINCT ?charged) AS ?withCharge) WHERE {
  GRAPH $RHEA {
    ?reaction rdfs:subClassOf rh:Reaction ;
              rh:accession ?accession ;
              rdfs:label ?label ;
              rh:side ?side .
    ?side rh:contains ?participant .
    ?participant rh:compound ?compound .
    OPTIONAL { ?compound rh:charge ?c . FILTER(DATATYPE(?c) = xsd:int) BIND(?compound AS ?charged) }
  }
}
GROUP BY ?reaction ?accession ?label
HAVING (COUNT(DISTINCT ?charged) >= 1)
ORDER BY ?reaction
""",
            # Two subqueries: the count sees every compound, the MIN only integer charges.
            # (One query with OPTIONAL does not work on QLever: its MIN is unbound as soon
            # as one input is unbound.)
            gold_pattern="""SELECT ?count ?min_charge WHERE {
  {
    SELECT (COUNT(DISTINCT ?compound) AS ?count) WHERE {
      GRAPH $RHEA {
        <$entity> rh:side ?side .
        ?side rh:contains ?participant .
        ?participant rh:compound ?compound .
      }
    }
  }
  {
    SELECT (MIN(?charge) AS ?min_charge) WHERE {
      GRAPH $RHEA {
        <$entity> rh:side ?side .
        ?side rh:contains ?participant .
        ?participant rh:compound ?compound .
        ?compound rh:charge ?c .
        FILTER(DATATYPE(?c) = xsd:int)
      }
      BIND(xsd:integer(?c) AS ?charge)
    }
  }
}""",
            question_pattern=(
                'How many distinct compounds take part in the Rhea reaction $accession ("$label"), counting each '
                "compound once even if it appears on both sides of the reaction, and including compounds without a "
                "charge, and what is the lowest net charge among the "
                "compounds that have an integer charge recorded in Rhea (generic compounds have none, and polymers "
                "record a formula-like text instead)? Return one row: the number of compounds, then the lowest "
                "charge as an integer."
            ),
            answer_columns=(AnswerColumn("count", NUM_COL), AnswerColumn("min_charge", NUM_COL)),
            key_param="entity",
            expand=_min_charge_reactions,
            engine_probes=(
                EngineProbe(
                    reason="qlever_min_unbound",
                    description=(
                        "The natural single query (OPTIONAL charge, then COUNT and MIN) returns an unbound MIN on "
                        "QLever as soon as one compound has no integer charge; other engines skip unbound values."
                    ),
                    pattern="""SELECT (COUNT(DISTINCT ?compound) AS ?count) (MIN(?charge) AS ?min_charge) WHERE {
  GRAPH $RHEA {
    <$entity> rh:side ?side .
    ?side rh:contains ?participant .
    ?participant rh:compound ?compound .
    OPTIONAL { ?compound rh:charge ?c . FILTER(DATATYPE(?c) = xsd:int) }
  }
  BIND(xsd:integer(?c) AS ?charge)
}""",
                ),
            ),
        ),
    )


# --------------------------------------------------------------------------
# 5. numeric_filter


class NumericFilter(QuestionTemplate):
    id = "numeric_filter"
    description = (
        "A numeric comparison on charge, the only numeric chemical attribute in the store (there is no mass), "
        "over a population anchored to one entity: the small-molecule compounds of a reaction (direct "
        "subclasses of rh:SmallMolecule) or the direct ChEBI subclasses of a ChEBI class. Charges are cast with "
        "xsd:integer because ChEBI stores positive charges as strings such as \"+1\"."
    )
    membership_mode = "direct"
    variants = (
        Variant(
            name="reaction_participant_charge",
            description=(
                "Small-molecule compounds of one master reaction whose Rhea charge passes a threshold; each "
                "question has its own population (one reaction's compounds)."
            ),
            candidate_query="""
SELECT ?reaction ?accession ?label ?charge (COUNT(DISTINCT ?compound) AS ?n) WHERE {
  GRAPH $RHEA {
    ?reaction rdfs:subClassOf rh:Reaction ;
              rh:accession ?accession ;
              rdfs:label ?label ;
              rh:side ?side .
    ?side rh:contains ?participant .
    ?participant rh:compound ?compound .
    ?compound rdfs:subClassOf rh:SmallMolecule ;
              rh:charge ?c .
  }
  BIND(xsd:integer(?c) AS ?charge)
}
GROUP BY ?reaction ?accession ?label ?charge
ORDER BY ?reaction ?charge
""",
            gold_pattern="""SELECT DISTINCT ?compound ?charge WHERE {
  GRAPH $RHEA {
    <$entity> rh:side ?side .
    ?side rh:contains ?participant .
    ?participant rh:compound ?compound .
    ?compound rdfs:subClassOf rh:SmallMolecule ;
              rh:charge ?c .
  }
  BIND(xsd:integer(?c) AS ?charge)
  FILTER(?charge $op $threshold)
}
ORDER BY ?compound""",
            question_pattern=(
                'Which small-molecule compounds taking part in the Rhea reaction $accession ("$label"), on either '
                "side, have a net charge $comparison, as recorded in Rhea? Return one row per compound: the Rhea "
                "compound IRI and its charge as an integer."
            ),
            answer_columns=(AnswerColumn("compound", IRI_COL), AnswerColumn("charge", NUM_COL)),
            key_param="entity",
            expand=_reaction_participant_thresholds,
        ),
        Variant(
            name="chebi_children_charge",
            description="Direct ChEBI subclasses of a ChEBI class whose ChEBI charge passes a threshold.",
            candidate_query="""
SELECT ?class ?label ?hint ?charge (COUNT(DISTINCT ?child) AS ?n) WHERE {
  GRAPH $CHEBI {
    ?child rdfs:subClassOf ?class ;
           chemrof:charge ?c .
    ?class rdfs:label ?label .
  }
  BIND(xsd:integer(?c) AS ?charge)
  OPTIONAL {
    SELECT DISTINCT ?class (1 AS ?hint) WHERE {
      GRAPH $CHEBI { ?grandchild rdfs:subClassOf ?mid ; chemrof:charge ?gc . ?mid rdfs:subClassOf ?class . }
    }
  }
}
GROUP BY ?class ?label ?hint ?charge
ORDER BY ?class ?charge
""",
            gold_pattern="""SELECT DISTINCT ?class ?charge WHERE {
  GRAPH $CHEBI {
    ?class rdfs:subClassOf <$entity> ;
           chemrof:charge ?c .
  }
  BIND(xsd:integer(?c) AS ?charge)
  FILTER(?charge $op $threshold)
}
ORDER BY ?class""",
            question_pattern=(
                'Which ChEBI classes that are direct subclasses of the ChEBI class "$label" ($curie) have a '
                "charge $comparison, as recorded in ChEBI? Consider only direct subclasses (one level down), "
                "not deeper descendants. Return one row per class: the ChEBI class IRI and its charge as an "
                "integer."
            ),
            answer_columns=(AnswerColumn("class", IRI_COL), AnswerColumn("charge", NUM_COL)),
            key_param="entity",
            expand=_chebi_children_thresholds,
            prefer=("direct_vs_transitive",),
        ),
    )


# --------------------------------------------------------------------------
# 6. absence_negation


class AbsenceNegation(QuestionTemplate):
    id = "absence_negation"
    description = (
        "Absence and negation: direct subclasses that lack an attribute (FILTER NOT EXISTS, MINUS). "
        "Candidates have at least one direct subclass with the attribute and one without, so the gold is "
        "non-empty and the negation removes something."
    )
    membership_mode = "direct"
    variants = (
        Variant(
            name="reaction_children_without_ec",
            description="Direct sub-reactions of a master reaction that have no EC number (FILTER NOT EXISTS).",
            candidate_query="""
SELECT DISTINCT ?reaction ?accession ?label WHERE {
  GRAPH $RHEA {
    ?reaction rdfs:subClassOf rh:Reaction ;
              rh:accession ?accession ;
              rdfs:label ?label .
    ?with rdfs:subClassOf ?reaction ; rh:ec ?ec .
    ?without rdfs:subClassOf ?reaction ; rdfs:subClassOf rh:Reaction .
    FILTER NOT EXISTS { ?without rh:ec ?anyEc . }
  }
}
ORDER BY ?reaction
""",
            gold_pattern="""SELECT DISTINCT ?reaction WHERE {
  GRAPH $RHEA {
    ?reaction rdfs:subClassOf <$entity> ;
              rdfs:subClassOf rh:Reaction .
    FILTER NOT EXISTS { ?reaction rh:ec ?ec . }
  }
}
ORDER BY ?reaction""",
            question_pattern=(
                'Which Rhea reactions that are direct subclasses of the reaction $accession ("$label") in the '
                "Rhea reaction hierarchy have no EC number assigned? Consider only direct subclasses (one level "
                "down), and count only undirected (master) reactions, not directional or bidirectional variants. "
                "Return one row per reaction: the reaction IRI."
            ),
            answer_columns=(AnswerColumn("reaction", IRI_COL),),
            key_param="entity",
            expand=_reactions,
            prefer=("direct_vs_transitive",),
        ),
        Variant(
            name="chebi_children_without_formula",
            description="Direct ChEBI subclasses of a ChEBI class with no ChEBI formula (MINUS).",
            candidate_query="""
SELECT DISTINCT ?class ?label ?hint WHERE {
  GRAPH $CHEBI {
    ?class rdfs:label ?label .
    ?with rdfs:subClassOf ?class ; chemrof:generalized_empirical_formula ?formula .
    ?without rdfs:subClassOf ?class ; rdfs:label ?withoutLabel .
    FILTER NOT EXISTS { ?without chemrof:generalized_empirical_formula ?anyFormula . }
  }
  OPTIONAL { SELECT DISTINCT ?class (1 AS ?rheaBelow) WHERE { GRAPH $RHEA { ?rheaIri rdfs:subClassOf ?class . } } }
  OPTIONAL {
    SELECT DISTINCT ?class (1 AS ?deeper) WHERE {
      GRAPH $CHEBI {
        ?grandchild rdfs:subClassOf ?mid ; rdfs:label ?grandchildLabel .
        ?mid rdfs:subClassOf ?class .
        FILTER NOT EXISTS { ?grandchild chemrof:generalized_empirical_formula ?grandchildFormula . }
      }
    }
  }
  BIND(COALESCE(?rheaBelow, 0) + COALESCE(?deeper, 0) AS ?hint)
}
ORDER BY ?class
""",
            gold_pattern="""SELECT DISTINCT ?class WHERE {
  GRAPH $CHEBI { ?class rdfs:subClassOf <$entity> . }
  FILTER(isIRI(?class))
  MINUS { GRAPH $CHEBI { ?class chemrof:generalized_empirical_formula ?formula . } }
}
ORDER BY ?class""",
            question_pattern=(
                'Which ChEBI classes that are direct subclasses of the ChEBI class "$label" ($curie) have no '
                "chemical formula recorded in ChEBI? Consider only direct subclasses (one level down) that are "
                "ChEBI classes, not Rhea compounds. Return one row per class: the ChEBI class IRI."
            ),
            answer_columns=(AnswerColumn("class", IRI_COL),),
            key_param="entity",
            expand=_chebi_classes,
            prefer=("direct_vs_transitive", "graph_scope"),
        ),
    )


# --------------------------------------------------------------------------
# 7. cross_graph_join


class CrossGraphJoin(QuestionTemplate):
    id = "cross_graph_join"
    description = (
        "Joins Rhea data with the ChEBI graph: the ChEBI hierarchy one level down (direct rdfs:subClassOf in "
        "the ChEBI graph) and ChEBI labels, linked to Rhea compounds through rh:chebi."
    )
    membership_mode = "direct"
    variants = (
        Variant(
            name="reactions_with_chebi_child_participant",
            description="Master reactions with a small-molecule participant whose ChEBI class is a direct subclass of X.",
            candidate_query=_CHEBI_PARENTS_OF_RHEA_COMPOUNDS,
            gold_pattern="""SELECT DISTINCT ?reaction WHERE {
  GRAPH $CHEBI { ?chebi rdfs:subClassOf <$entity> . }
  GRAPH $RHEA {
    ?compound rh:chebi ?chebi ;
              rdfs:subClassOf rh:SmallMolecule .
    ?participant rh:compound ?compound .
    ?side rh:contains ?participant .
    ?reaction rh:side ?side ;
              rdfs:subClassOf rh:Reaction .
  }
}
ORDER BY ?reaction""",
            question_pattern=(
                "Which Rhea reactions have a small-molecule participant whose ChEBI class is a direct subclass of "
                'the ChEBI class "$label" ($curie)? Use the ChEBI hierarchy one level down from $curie (not '
                "$curie itself, not deeper), and count only undirected (master) reactions. "
                "Return one row per reaction: the reaction IRI."
            ),
            answer_columns=(AnswerColumn("reaction", IRI_COL),),
            key_param="entity",
            expand=_chebi_classes,
            prefer=("direct_vs_transitive",),
        ),
        Variant(
            name="compounds_in_chebi_children",
            description="Rhea small-molecule compounds linked to a direct ChEBI subclass of X, with the ChEBI label.",
            candidate_query=_CHEBI_PARENTS_OF_RHEA_COMPOUNDS,
            gold_pattern="""SELECT DISTINCT ?compound ?chebiLabel WHERE {
  GRAPH $CHEBI {
    ?chebi rdfs:subClassOf <$entity> ;
           rdfs:label ?chebiLabel .
  }
  GRAPH $RHEA {
    ?compound rh:chebi ?chebi ;
              rdfs:subClassOf rh:SmallMolecule .
  }
}
ORDER BY ?compound""",
            question_pattern=(
                "Which Rhea small-molecule compounds are linked to a ChEBI class that is a direct subclass of the "
                'ChEBI class "$label" ($curie) (one level down, not $curie itself)? Return one row per compound: '
                "the Rhea compound IRI and the label that ChEBI gives to the linked class."
            ),
            answer_columns=(AnswerColumn("compound", IRI_COL), AnswerColumn("chebi_label", LIT_COL)),
            key_param="entity",
            expand=_chebi_classes,
            prefer=("direct_vs_transitive",),
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
