"""Calibration round 2 of the Rhea set: every compositional variant one structural step up.

Round 1 gave Sonnet 0.926 exact match on the 63 development compositional
questions, above the 50-80% band, so every active compositional variant goes
one step up along one structural axis (pre-registration change-log entry 6, as
applied to UniProt in the design log, entry 23), mechanically and uniformly, never
tuned per template. The lookup tier is untouched.

The axis of a variant is read from its round-1 forms in a fixed order (see
STEP_RULES): a set operation -> one more population; a comparison -> one more
compared entity (comparisons count as populations); negation -> one more
negation nested inside a negation; grouping -> one more HAVING constraint
(at most ``Knobs.having_max``, one value for every grouping variant);
otherwise -> one more path hop. Every raised variant checks on the store that
its step matters for each question (``base.StepCheck``); a comparison that
asks for a winner (partners_with_lower_charge: the binding reference) also
puts it in each mention slot as evenly as each (variant, split) allows
(generate.balance_slots). citation_count_difference asks for none: like
round 1 it names its subject first and reads the other two as a set.

The knobs ``<axis>_delta`` (base.Knobs) choose the raised forms; with every
delta 0 (round 1) ``resolve_templates`` returns the round-1 templates
themselves, so the frozen round-1 file reproduces byte for byte.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from typing import Any, Iterable, Mapping

from harness.qgen_rhea import compositional as c1
from harness.qgen_rhea.base import (
    COMPOSITIONAL,
    LOOKUP,
    Anchor,
    AnswerColumn,
    EngineProbe,
    Knobs,
    PoolRow,
    QuestionTemplate,
    StepCheck,
    StructuralStep,
    TemplateError,
    Variant,
    check_iri,
)
from harness.qgen_rhea.compositional import (
    _N,
    _TAKES_PART,
    CLASSES_AS,
    COMPOUNDS_AS,
    IRI_COL,
    MAX_REACTION_LABEL_CHARS,
    NUM_COL,
    REACTIONS_ONCE,
    UNDIRECTED,
    _comp,
    _label_ok,
    _thresholds,
)
from harness.qgen_rhea.readings import (
    STATUS_NOTE,
    Reading,
    approved_only_reading,
    other_kind_reading,
    rhea_readings,
)
from harness.qgen_rhea.templates import TEMPLATES_BY_ID

EC_AS = "Give each EC number by its IRI."
# The rejection of a candidate that a retained round-1 bound excludes (base.Candidate.excluded).
ROUND_1_BOUND = "round_1_bound"
_RHEA_TAKES_PART = "GRAPH $RHEA {\n    ?compound rh:chebi ?class .\n    " + _TAKES_PART + "\n  }"


def _with_readings(variant: Variant, *extra: Reading) -> Variant:
    """The variant with the Rhea readings for its gold (as templates.py attaches them) plus its own.

    With two or more label anchors, the other-kind reading resolves each anchor on its own
    (other_kind_anchor:<i>) rather than all at once: resolving two homonym compounds through their class
    subtrees in one query times out on the store (QLever sorts the cross product), and a time-out fails the run.
    """
    base = rhea_readings(
        variant.membership_mode or "n.a.", variant.anchors, variant.side_clauses, variant.gold_pattern,
        variant.joint_sides, variant.joint_clause,
    )
    labelled = [a for a in variant.anchors if a.label_param]
    if len(labelled) > 1:
        base = tuple(r for r in base if r.name != "other_kind_anchor") + tuple(
            replace(other_kind_reading((a,)), name=f"other_kind_anchor:{i}") for i, a in enumerate(labelled)
        )
    return replace(variant, readings=base + tuple(extra))


def _r1(template: str, name: str) -> Variant:
    return TEMPLATES_BY_ID[template].variant(name)


# --------------------------------------------------------------------------
# populations: set operations (one more population)


def _class_triples(rows: list[PoolRow], knobs: Knobs) -> Iterable[dict[str, Any]]:
    """Triples of small classes that are direct parents of one class (rows: shared class, parent, label).

    Built here rather than by a three-way join in the store, which times out.
    Each triple is drawn once, in the order it first appears.
    """
    by_shared: "OrderedDict[str, list[tuple[str, str]]]" = OrderedDict()
    for r in rows:
        if _label_ok(r["label"]):
            by_shared.setdefault(r["shared"], []).append((check_iri(r["p"]), r["label"]))
    seen: set[str] = set()
    for parents in by_shared.values():
        parents = sorted(set(parents))
        for i, a in enumerate(parents):
            for j in range(i + 1, len(parents)):
                for k in range(j + 1, len(parents)):
                    b, c = parents[j], parents[k]
                    key = f"{a[0]}|{b[0]}|{c[0]}"
                    if key in seen:
                        continue
                    seen.add(key)
                    yield {
                        "key": key,
                        "entity_a": a[0], "label_a": a[1],
                        "entity_b": b[0], "label_b": b[1],
                        "entity_c": c[0], "label_c": c[1],
                    }


def _below_all(*names: str) -> str:
    lines = "\n".join(f"    ?class rdfs:subClassOf+ <$entity_{x}> ." for x in names)
    return f"SELECT DISTINCT ?class WHERE {{\n  GRAPH $CHEBI {{\n{lines}\n  }}\n  FILTER(isIRI(?class))\n}}\nORDER BY ?class"


COMMON_DESCENDANTS = _comp(
    name="common_descendants",
    description=(
        "ChEBI classes below all three of three ChEBI classes, at any depth (intersection of three transitive "
        "populations). The three are direct parents of one ChEBI class; the step check (dropping any one changes "
        "the answer) also rules out one being below another."
    ),
    candidate_query="""
SELECT ?shared ?p ?label WHERE {
  {
    SELECT ?shared WHERE { GRAPH $CHEBI { ?shared rdfs:subClassOf ?q . } FILTER(isIRI(?q)) }
    GROUP BY ?shared
    HAVING (COUNT(DISTINCT ?q) >= 3)
  }
  GRAPH $CHEBI { ?shared rdfs:subClassOf ?p . }
  FILTER(isIRI(?p))
  { SELECT ?p WHERE { GRAPH $CHEBI { ?d rdfs:subClassOf+ ?p . } } GROUP BY ?p HAVING (COUNT(DISTINCT ?d) <= 400) }
  GRAPH $CHEBI { ?p rdfs:label ?label . }
}
ORDER BY ?shared ?p ?label
""",
    gold_pattern=_below_all("a", "b", "c"),
    question_pattern=(
        "Which chemical classes are subclasses, at any depth, of all three of the chemical classes named "
        f'"$label_a", "$label_b" and "$label_c"? {CLASSES_AS}'
    ),
    answer_columns=(AnswerColumn("class", IRI_COL),),
    key_param="key",
    expand=_class_triples,
    anchors=tuple(Anchor("chemical_class", f"entity_{x}", f"label_{x}") for x in "abc"),
    membership_mode="transitive",
    forms=("intersection",),
    features={"populations": 3},
    population_pattern="""SELECT ?n WHERE {
  { SELECT (COUNT(DISTINCT ?ca) AS ?na) WHERE { GRAPH $CHEBI { ?ca rdfs:subClassOf+ <$entity_a> . } FILTER(isIRI(?ca)) } }
  { SELECT (COUNT(DISTINCT ?cb) AS ?nb) WHERE { GRAPH $CHEBI { ?cb rdfs:subClassOf+ <$entity_b> . } FILTER(isIRI(?cb)) } }
  { SELECT (COUNT(DISTINCT ?cc) AS ?nc) WHERE { GRAPH $CHEBI { ?cc rdfs:subClassOf+ <$entity_c> . } FILTER(isIRI(?cc)) } }
  BIND(IF(IF(?na < ?nb, ?na, ?nb) < ?nc, IF(?na < ?nb, ?na, ?nb), ?nc) AS ?n)
}""",
    prefer=("direct_vs_transitive", "graph_scope"),
    step_check=StepCheck(
        "every_population_matters",
        undo=(_below_all("b", "c"), _below_all("a", "c"), _below_all("a", "b")),
    ),
)


def _shared_pair(x: str, y: str) -> str:
    return f"""SELECT DISTINCT ?reaction WHERE {{
  GRAPH $RHEA {{
    ?reaction rdfs:subClassOf rh:Reaction ;
              rh:side ?side{x.upper()} , ?side{y.upper()} .
    ?side{x.upper()} rh:contains ?participant{x.upper()} .
    ?participant{x.upper()} rh:compound <$entity_{x}> .
    ?side{y.upper()} rh:contains ?participant{y.upper()} .
    ?participant{y.upper()} rh:compound <$entity_{y}> .
  }}
}}
ORDER BY ?reaction"""


# The round-1 variant shared_reactions_3 (inactive in round 1) is the raised form of shared_reactions_2.
SHARED_REACTIONS_3 = replace(
    _r1("multi_hop_path", "shared_reactions_3"),
    step_check=StepCheck(
        "every_population_matters",
        undo=(_shared_pair("b", "c"), _shared_pair("a", "c"), _shared_pair("a", "b")),
    ),
)


# --------------------------------------------------------------------------
# populations: comparisons (one more compared entity)


def _reaction_triples_same_ec(rows: list[PoolRow], knobs: Knobs) -> Iterable[dict[str, Any]]:
    """Triples of reactions that share an EC number and have pairwise different citation counts.

    Within an EC number, reactions in IRI order are taken three at a time
    (consecutive members), so the pool is deterministic and each triple is
    related. The most-cited one is the subject (slot a), as round 1 put the
    more-cited of its pair first; the other two follow in IRI order.
    """
    by_ec: "OrderedDict[str, list[PoolRow]]" = OrderedDict()
    for r in rows:
        by_ec.setdefault(r["ec"], []).append(r)
    seen: set[str] = set()
    for members in by_ec.values():
        members = sorted((m for m in members if _label_ok(m["label"], MAX_REACTION_LABEL_CHARS)), key=lambda m: m["reaction"])
        for x, y, z in zip(members, members[1:], members[2:]):
            if len({int(x["n"]), int(y["n"]), int(z["n"])}) < 3:
                continue
            key = "|".join(sorted((x["reaction"], y["reaction"], z["reaction"])))
            if key in seen:
                continue
            seen.add(key)
            subject = max((x, y, z), key=lambda m: int(m["n"]))
            out: dict[str, Any] = {"key": key}
            for slot, m in zip("abc", [subject, *(m for m in (x, y, z) if m is not subject)]):
                out[f"entity_{slot}"] = check_iri(m["reaction"])
                out[f"label_{slot}"] = m["label"]
            yield out


_CITATIONS_OF_A = "{ SELECT (COUNT(DISTINCT ?citationA) AS ?a) WHERE { GRAPH $RHEA { <$entity_a> rh:citation ?citationA . } } }"
# The publications of the other two reactions as one set: a publication cited for both counts once.
_CITATIONS_OF_B_AND_C = """{ SELECT (COUNT(DISTINCT ?citationBC) AS ?bc) WHERE {
      VALUES ?other { <$entity_b> <$entity_c> }
      GRAPH $RHEA { ?other rh:citation ?citationBC . }
    } }"""

R2_CITATIONS_SUMMED = Reading(
    "citations_summed",
    "the two other reactions' counts added, a publication cited for both of them counted twice",
    # Exact polarity: the shipped sentences word for word, from the start of the first (external review of 3ea978e: "It
    # is false that (a publication cited for both counts once)" passed; external review of 01eeb51: ", not minus" did).
    rf"(?:^|[.?!]\s)Give the number of distinct publications cited for the reaction «label_a»{_N}, minus the number of "
    rf"distinct publications cited for the reaction «label_b»{_N} or the reaction «label_c»{_N} "
    r"\(a publication cited for both counts once\)\. The result may be zero or negative\.(?:\s|$)",
    pattern=f"""SELECT ?difference WHERE {{
  {_CITATIONS_OF_A}
  {{ SELECT (COUNT(DISTINCT ?citationB) AS ?b) WHERE {{ GRAPH $RHEA {{ <$entity_b> rh:citation ?citationB . }} }} }}
  {{ SELECT (COUNT(DISTINCT ?citationC) AS ?c) WHERE {{ GRAPH $RHEA {{ <$entity_c> rh:citation ?citationC . }} }} }}
  BIND(?a - (?b + ?c) AS ?difference)
}}""",
)

CITATION_COUNT_DIFFERENCE = _comp(
    name="citation_count_difference",
    description=(
        "The number of distinct publications one master reaction cites minus the number the two others cite together "
        "(a publication cited for both counted once); the three share an EC number, cite different numbers of "
        "publications, and the first is the most-cited (comparison of one entity with two, one signed numeric answer)."
    ),
    candidate_query=c1.CITATION_COUNT_DIFFERENCE.candidate_query,
    # Round 1's directed difference A - B with one more reaction on B's side: A - |publications of B or C|. The
    # subject is fixed by the wording (the most-cited, as in round 1), and B and C are read as a set, so no mention
    # slot can carry the answer and none is balanced.
    gold_pattern=f"""SELECT ?difference WHERE {{
  {_CITATIONS_OF_A}
  {_CITATIONS_OF_B_AND_C}
  BIND(?a - ?bc AS ?difference)
}}""",
    # Worded as an explicit subtraction (design ruling in the design log).
    question_pattern=(
        'Give the number of distinct publications cited for the reaction "$label_a", minus the number of distinct '
        'publications cited for the reaction "$label_b" or the reaction "$label_c" (a publication cited for both '
        "counts once). The result may be zero or negative."
    ),
    answer_columns=(AnswerColumn("difference", NUM_COL),),
    key_param="key",
    expand=_reaction_triples_same_ec,
    anchors=tuple(Anchor("reaction", f"entity_{x}", f"label_{x}") for x in "abc"),
    membership_mode="n.a.",
    hops=1,
    forms=("comparison",),
    features={"compared": 3},
    single_row=True,
    # The three counts differ (as round 1's two did), and each of the two reactions on the compared side changes the
    # answer: without it the question is round 1's on the other two (code review of a0024f4: an added reaction whose
    # publications the other one already cites would change nothing).
    step_check=StepCheck(
        "distinct_values_and_each_compared_reaction_matters",
        distinct="""SELECT ?reaction ?value WHERE {
  { SELECT ?reaction (COUNT(DISTINCT ?citation) AS ?value) WHERE {
      VALUES ?reaction { <$entity_a> <$entity_b> <$entity_c> }
      GRAPH $RHEA { ?reaction rh:citation ?citation . }
    }
    GROUP BY ?reaction }
}""",
        compared=("entity_a", "entity_b", "entity_c"),
        undo=(
            c1.CITATION_COUNT_DIFFERENCE.gold_pattern,
            c1.CITATION_COUNT_DIFFERENCE.gold_pattern.replace("$entity_b", "$entity_c"),
        ),
    ),
)


def _reference_pairs(rows: list[PoolRow], knobs: Knobs) -> Iterable[dict[str, Any]]:
    """(compound, second reference) options; the compound is the key, so it is drawn once.

    The two references start in the order (compound, second reference); the
    mention order is set later, balancing the binding reference over the slots.
    """
    for r in rows:
        if not (_label_ok(r["name"]) and _label_ok(r["nameB"])):
            continue
        a, b = check_iri(r["compound"]), check_iri(r["b"])
        yield {
            "entity": a, "name": r["name"], "entity_b": b, "name_b": r["nameB"],
            "ref_1_entity": a, "ref_1": r["name"], "ref_2_entity": b, "ref_2": r["nameB"],
        }


_LOWER_THAN = """SELECT DISTINCT ?compound ?charge WHERE {
  GRAPH $RHEA {
    $REFERENCES
    ?anchorParticipant rh:compound <$entity> .
    ?anchorSide rh:contains ?anchorParticipant .
    ?reaction rh:side ?anchorSide ;
              rdfs:subClassOf rh:Reaction ;
              rh:side ?side .
    ?side rh:contains ?participant .
    ?participant rh:compound ?compound .
    ?compound rh:charge ?c .
  }
  BIND(xsd:integer(?c) AS ?charge)
  FILTER($TEST)
}
ORDER BY ?compound"""


def _lower_than(references: str, test: str) -> str:
    return _LOWER_THAN.replace("$REFERENCES", references).replace("$TEST", test)


_BOTH_REFERENCES = "<$entity> rh:charge ?anchorCharge .\n    <$entity_b> rh:charge ?referenceCharge ."

R2_REFERENCE_ANY = Reading(
    "reference_any",
    "lower than at least one of the two references (the higher one), not lower than each",
    r"(?<!not )\bhave a lower net charge than each of the compound entries named\b",
    pattern=_lower_than(_BOTH_REFERENCES, "?charge < xsd:integer(?anchorCharge) || ?charge < xsd:integer(?referenceCharge)"),
)

PARTNERS_WITH_LOWER_CHARGE = _comp(
    name="partners_with_lower_charge",
    description=(
        "Compounds that share a master reaction with a small molecule and have a lower integer charge than each of "
        "two references: the small molecule itself and another compound of its reactions (comparison with two "
        "reference entities across a 6-hop join, charges cast)."
    ),
    candidate_query=f"""
SELECT DISTINCT ?compound ?name ?b ?nameB WHERE {{
  {{
    SELECT ?compound ?name WHERE {{
      GRAPH $RHEA {{
        ?compound rdfs:subClassOf rh:SmallMolecule ;
                  rh:name ?name ;
                  rh:charge ?anyCharge .
        {_TAKES_PART}
      }}
    }}
    GROUP BY ?compound ?name
    HAVING (COUNT(DISTINCT ?reaction) >= 1 && COUNT(DISTINCT ?reaction) <= 6)
  }}
  GRAPH $RHEA {{
    ?compound rh:charge ?chargeA .
    ?pa rh:compound ?compound .
    ?sa rh:contains ?pa .
    ?r rh:side ?sa , ?sb ;
       rdfs:subClassOf rh:Reaction .
    ?sb rh:contains ?pb .
    ?pb rh:compound ?b .
    ?b rdfs:subClassOf rh:SmallMolecule ;
       rh:name ?nameB ;
       rh:charge ?chargeB .
  }}
  FILTER(?b != ?compound && DATATYPE(?chargeA) = xsd:int && DATATYPE(?chargeB) = xsd:int
         && xsd:integer(?chargeA) != xsd:integer(?chargeB))
}}
ORDER BY ?compound ?b
""",
    # The gold reads the references by entity, never by mention slot, so the mention order cannot change it.
    gold_pattern=_lower_than(
        _BOTH_REFERENCES, "?charge < xsd:integer(?anchorCharge) && ?charge < xsd:integer(?referenceCharge)"
    ),
    question_pattern=(
        'Take the reactions that the compound entry named "$name" takes part in, on either side '
        f"({UNDIRECTED}). Which compounds take part in them, on either side, and have a lower net charge than each "
        'of the compound entries named "$ref_1" and "$ref_2"? Consider only compounds whose charge is a whole '
        f"number. List each such compound with its charge. {COMPOUNDS_AS}"
    ),
    answer_columns=(AnswerColumn("compound", IRI_COL), AnswerColumn("charge", NUM_COL)),
    expand=_reference_pairs,
    side_clauses={
        "?anchorSide": rf"\bTake the reactions that the compound entry named «name»{_N} takes part in, on either side\b",
        "?side": r"\bWhich compounds take part in them, on either side, and have a lower net charge\b",
    },
    joint_sides=("?anchorSide", "?side"),
    anchors=(Anchor("compound", label_param="name"), Anchor("compound", "entity_b", "name_b")),
    hops=6,
    membership_mode="n.a.",
    forms=("comparison", "path"),
    features={"references": 2, "hops": 6},
    population_pattern=c1.PARTNERS_WITH_LOWER_CHARGE.population_pattern,
    slots=(("ref_1_entity", "ref_1"), ("ref_2_entity", "ref_2")),
    winner="min",
    # As for every comparison (the study lead's ruling for UniProt round 2, change-log entry 31): the two
    # charges differ (no ties) and the binding reference's mention slot is balanced; which reference binds is
    # not required ("the added entity" is undefined under redrawing) but recorded (binding_is_the_anchor).
    # Dropping the binding reference changes the rows (mode "some").
    step_check=StepCheck(
        "distinct_references",
        distinct="""SELECT ?reference ?value WHERE {
  VALUES ?reference { <$entity> <$entity_b> }
  GRAPH $RHEA { ?reference rh:charge ?c . }
  BIND(xsd:integer(?c) AS ?value)
}""",
        undo=(
            _lower_than("<$entity> rh:charge ?anchorCharge .", "?charge < xsd:integer(?anchorCharge)"),
            _lower_than("<$entity_b> rh:charge ?referenceCharge .", "?charge < xsd:integer(?referenceCharge)"),
        ),
        mode="some",
    ),
)


# --------------------------------------------------------------------------
# negation depth (one more negation nested inside a negation)

_PARTNER_ABSENT = """FILTER NOT EXISTS {
      ?reaction rh:side ?otherSide .
      ?otherSide rh:contains ?otherParticipant .
      ?otherParticipant rh:compound <$entity_b> .
    }"""
_REACTIONS_OF_A = """?participant rh:compound <$entity_a> .
    ?side rh:contains ?participant .
    ?reaction rh:side ?side ;
              rdfs:subClassOf rh:Reaction ."""


def _partner_gold(member_test: str, partner_absent: str = _PARTNER_ABSENT) -> str:
    return f"""SELECT DISTINCT ?reaction WHERE {{
  GRAPH $RHEA {{
    {_REACTIONS_OF_A}
    {partner_absent}
    {member_test}
  }}
}}
ORDER BY ?reaction"""


_MEMBER = """?reaction rh:side ?memberSide .
      ?memberSide rh:contains ?memberParticipant .
      ?memberParticipant rh:compound ?member ."""
# Quantified over the compounds of the reaction, not its participants or sides: a compound on both sides of a
# transport reaction is one compound, linked or not.
_EVERY_MEMBER_LINKED = f"""FILTER NOT EXISTS {{
      {_MEMBER}
      FILTER NOT EXISTS {{ ?member rh:chebi ?memberClass . }}
    }}"""
_SOME_MEMBER_LINKED = f"""FILTER EXISTS {{
      {_MEMBER}
      ?member rh:chebi ?memberClass .
    }}"""

R2_SOME_MEMBER_LINKED = Reading(
    "some_member_linked",
    "a reaction qualifies when some of its compounds is linked to a chemical class, not every one",
    # Exact polarity: "not every compound ...", or "... is not linked", must not match.
    r"(?<!not )\bevery compound taking part in them, on either side, is linked to a chemical class\b",
    pattern=_partner_gold(_SOME_MEMBER_LINKED),
)

REACTIONS_WITHOUT_PARTNER = _comp(
    name="reactions_without_partner",
    description=(
        "Master reactions of compound A in which compound B (one of the 30 compounds in most reactions) does not "
        "take part and every compound of which is linked to a ChEBI class (FILTER NOT EXISTS a compound with "
        "FILTER NOT EXISTS a class: negation nested twice)."
    ),
    candidate_query=c1.REACTIONS_WITHOUT_PARTNER.candidate_query,
    gold_pattern=_partner_gold(_EVERY_MEMBER_LINKED),
    question_pattern=(
        'Which reactions that the compound entry named "$name_a" takes part in, on either side, do not also involve '
        'the compound entry named "$name_b" on either side, and are such that every compound taking part in them, '
        "on either side, is linked to a chemical class? Polymers do not count as linked through the class of their "
        f"repeating unit, and generic compounds do not count as linked through their reactive parts. {REACTIONS_ONCE}"
    ),
    answer_columns=(AnswerColumn("reaction", IRI_COL),),
    key_param="entity_a",
    expand=c1._compound_without_partner,
    side_clauses={
        "?side": rf"\bWhich reactions that the compound entry named «name_a»{_N} takes part in, on either side, do not also involve\b",
        "?otherSide": rf"\bdo not also involve the compound entry named «name_b»{_N} on either side\b",
        "?memberSide": R2_SOME_MEMBER_LINKED.clause,
    },
    joint_sides=("?side", "?otherSide"),
    anchors=(Anchor("compound", "entity_a", "name_a"), Anchor("compound", "entity_b", "name_b")),
    membership_mode="n.a.",
    forms=("negation",),
    features={"negation_nesting": 2},
    population_pattern=c1.REACTIONS_WITHOUT_PARTNER.population_pattern,
    # The nested negation must matter: the answer differs from the round-1 form and from the positive reading
    # (some compound linked), which would answer the question without the nested negation. The round-1 negation
    # must still matter too (without it, other rows), so no clause of the text is idle.
    step_check=StepCheck(
        "undo_and_positive_reading_change",
        undo=(_partner_gold(""), _partner_gold(_SOME_MEMBER_LINKED), _partner_gold(_EVERY_MEMBER_LINKED, "")),
    ),
)


def _classes_with_qualifying_children(rows: list[PoolRow], knobs: Knobs) -> Iterable[dict[str, Any]]:
    # Round 1's bounds stay (code review of e892710): at least one used direct child (the query joins on it),
    # min_population direct children, min_rows..max_rows unused ones; the qualifying ones (k) are the round-2 answer.
    for r in rows:
        unused = int(r["n"]) - int(r["used"])
        if (_label_ok(r["label"]) and int(r["used"]) >= 1 and int(r["n"]) >= knobs.min_population
                and knobs.min_rows <= unused <= knobs.max_rows and knobs.min_rows <= int(r["k"]) <= knobs.max_rows):
            yield {"entity": check_iri(r["class"]), "label": r["label"]}


def _subclass_test(kind: str) -> str:
    """The test on the child's own direct subclasses: every one linked (gold) or some linked (positive reading)."""
    linked = """GRAPH $RHEA {
        ?subclassCompound rh:chebi ?subclass .
        ?subclassParticipant rh:compound ?subclassCompound .
        ?subclassSide rh:contains ?subclassParticipant .
        ?subclassReaction rh:side ?subclassSide ;
                          rdfs:subClassOf rh:Reaction .
      }"""
    if kind == "every":
        return f"""FILTER NOT EXISTS {{
    GRAPH $CHEBI {{ ?subclass rdfs:subClassOf ?class . }}
    FILTER(isIRI(?subclass))
    FILTER NOT EXISTS {{
      {linked}
    }}
  }}"""
    return f"""FILTER EXISTS {{
    GRAPH $CHEBI {{ ?subclass rdfs:subClassOf ?class . }}
    FILTER(isIRI(?subclass))
    {linked}
  }}"""


def _unused_gold(own_subclass: bool, test: str, unused: bool = True) -> str:
    grandchild = "\n    ?grandchild rdfs:subClassOf ?class ." if own_subclass else ""
    grandchild_iri = " && isIRI(?grandchild)" if own_subclass else ""
    not_linked = f"""FILTER NOT EXISTS {{
    GRAPH $RHEA {{
      ?compound rh:chebi ?class .
      {_TAKES_PART}
    }}
  }}""" if unused else ""
    return f"""SELECT DISTINCT ?class WHERE {{
  GRAPH $CHEBI {{
    ?class rdfs:subClassOf <$entity> .{grandchild}
  }}
  FILTER(isIRI(?class){grandchild_iri})
  {not_linked}
  {test}
}}
ORDER BY ?class"""


R2_SOME_SUBCLASS_LINKED = Reading(
    "some_subclass_linked",
    "a class qualifies when some of its direct subclasses is linked, not every one",
    r"(?<!not )\bevery one of which is linked to a compound that takes part in a reaction, on either side\b",
    pattern=_unused_gold(True, _subclass_test("some")),
)
R2_SUBCLASS_REQUIRED = Reading(
    "vacuous_universal",
    "a class without any direct subclass of its own counts too (every one of none is linked)",
    r"(?<!not )\bhave at least one direct subclass of their own\b",
    pattern=_unused_gold(False, _subclass_test("every")),
)

UNUSED_CHILD_CLASSES = _comp(
    name="unused_child_classes",
    description=(
        "Direct ChEBI subclasses of X that no compound taking part in a master reaction is linked to, that have a "
        "direct subclass, and every direct subclass of which is linked to such a compound (FILTER NOT EXISTS a "
        "subclass with FILTER NOT EXISTS a linked participating compound: negation nested twice)."
    ),
    candidate_query="""
SELECT ?class ?label ?n ?k ?used WHERE {
  {
    SELECT ?class (COUNT(DISTINCT ?child) AS ?k) WHERE {
      GRAPH $CHEBI { ?child rdfs:subClassOf ?class . ?grandchild rdfs:subClassOf ?child . }
      FILTER(isIRI(?class) && isIRI(?child) && isIRI(?grandchild))
      FILTER NOT EXISTS { GRAPH $RHEA { ?childCompound rh:chebi ?child . ?childParticipant rh:compound ?childCompound . } }
      FILTER NOT EXISTS {
        GRAPH $CHEBI { ?sub rdfs:subClassOf ?child . }
        FILTER(isIRI(?sub))
        FILTER NOT EXISTS { GRAPH $RHEA { ?subCompound rh:chebi ?sub . ?subParticipant rh:compound ?subCompound . } }
      }
    }
    GROUP BY ?class
    HAVING (COUNT(DISTINCT ?child) >= 2)
  }
  { SELECT ?class (COUNT(DISTINCT ?member) AS ?n) WHERE { GRAPH $CHEBI { ?member rdfs:subClassOf ?class . } FILTER(isIRI(?member)) } GROUP BY ?class }
  {
    SELECT ?class (COUNT(DISTINCT ?usedChild) AS ?used) WHERE {
      GRAPH $CHEBI { ?usedChild rdfs:subClassOf ?class . }
      FILTER(isIRI(?usedChild))
      GRAPH $RHEA { ?usedCompound rh:chebi ?usedChild . ?usedParticipant rh:compound ?usedCompound . }
    }
    GROUP BY ?class
  }
  GRAPH $CHEBI { ?class rdfs:label ?label . }
}
ORDER BY ?class
""",
    gold_pattern=_unused_gold(True, _subclass_test("every")),
    question_pattern=(
        'Which direct subclasses (one level down, not the class itself) of the chemical class named "$label" are not '
        f"linked to any compound that takes part in a reaction ({UNDIRECTED}), on either side, but have at least one "
        "direct subclass of their own, every one of which is linked to a compound that takes part in a reaction, on "
        "either side? Polymers do not count as linked through the class of their repeating unit, and generic "
        f"compounds do not count as linked through their reactive parts. {CLASSES_AS}"
    ),
    answer_columns=(AnswerColumn("class", IRI_COL),),
    expand=_classes_with_qualifying_children,
    side_clauses={
        "?side": r"\bare not linked to any compound that takes part in a reaction \(taking each reaction in its undirected form\), on either side\b",
        "?subclassSide": R2_SOME_SUBCLASS_LINKED.clause,
    },
    anchors=(Anchor("chemical_class"),),
    membership_mode="direct",
    forms=("negation",),
    features={"negation_nesting": 2},
    population_pattern=c1.UNUSED_CHILD_CLASSES.population_pattern,
    prefer=("direct_vs_transitive", "graph_scope"),
    engine_probes=(
        EngineProbe(
            reason="qlever_nested_graph",
            description=(
                "The natural query that writes the negated reaction-graph patterns inside the chemical-class GRAPH "
                "block. QLever 9bebc60 evaluates a GRAPH clause nested in a plain group against the outer graph; a "
                "question is tagged when the nested form gives other rows than the gold."
            ),
            pattern=f"""SELECT DISTINCT ?class WHERE {{
  GRAPH $CHEBI {{
    ?class rdfs:subClassOf <$entity> .
    ?grandchild rdfs:subClassOf ?class .
    FILTER(isIRI(?class) && isIRI(?grandchild))
    FILTER NOT EXISTS {{
      GRAPH $RHEA {{
        ?compound rh:chebi ?class .
        {_TAKES_PART}
      }}
    }}
    FILTER NOT EXISTS {{
      ?subclass rdfs:subClassOf ?class .
      FILTER NOT EXISTS {{
        GRAPH $RHEA {{
          ?subclassCompound rh:chebi ?subclass .
          ?subclassParticipant rh:compound ?subclassCompound .
          ?subclassSide rh:contains ?subclassParticipant .
          ?subclassReaction rh:side ?subclassSide ;
                            rdfs:subClassOf rh:Reaction .
        }}
      }}
    }}
  }}
}}""",
        ),
    ),
    step_check=StepCheck(
        "undo_and_positive_reading_change",
        undo=(c1.UNUSED_CHILD_CLASSES.gold_pattern, R2_SOME_SUBCLASS_LINKED.pattern, _unused_gold(True, _subclass_test("every"), False)),
    ),
)


# --------------------------------------------------------------------------
# grouping constraints (one more HAVING constraint: at most having_max)


def _reaction_size_bands(rows: list[PoolRow], knobs: Knobs) -> Iterable[dict[str, Any]]:
    """Per compound, lower thresholds on reaction size below having_max (the added upper bound).

    A round-1 upper threshold ("at most t") would make one of the two bounds
    redundant, so the raised form pairs having_max with a lower bound only.
    """
    by_compound: "OrderedDict[str, tuple[PoolRow, OrderedDict[int, int]]]" = OrderedDict()
    for r in rows:
        first, counts = by_compound.setdefault(r["compound"], (r, OrderedDict()))
        counts[int(r["size"])] = int(r["n"])
    for compound, (first, counts) in by_compound.items():
        total = sum(counts.values())
        if not _label_ok(first["name"]) or total < knobs.min_population or total > 60 or len(counts) < 2:
            continue
        for _, value, _ in _thresholds(counts, knobs, (">=",)):
            if value < knobs.having_max:
                yield {"entity": check_iri(compound), "name": first["name"], "threshold": value, "having_max": knobs.having_max}


def _size_gold(having: str) -> str:
    return c1.REACTIONS_BY_SIZE.gold_pattern.replace("HAVING (COUNT(DISTINCT ?compound) $op $threshold)", f"HAVING ({having})")


REACTIONS_BY_SIZE = _comp(
    name="reactions_by_size",
    description=(
        "Master reactions of a compound whose number of distinct compounds lies between a threshold and "
        "having_max, each with that number (GROUP BY reaction, two HAVING constraints on COUNT(DISTINCT compound))."
    ),
    candidate_query=c1.REACTIONS_BY_SIZE.candidate_query,
    gold_pattern=_size_gold("COUNT(DISTINCT ?compound) >= $threshold && COUNT(DISTINCT ?compound) <= $having_max"),
    question_pattern=(
        'Among the reactions in which the compound entry named "$name" takes part, on either side, which involve at '
        "least $threshold and at most $having_max different compounds in total, counting a compound once even if it "
        "appears on both sides? List each such reaction with its number of different compounds. Give each reaction "
        "once, whatever its direction, by the IRI of its undirected form."
    ),
    answer_columns=(AnswerColumn("reaction", IRI_COL), AnswerColumn("count", NUM_COL)),
    expand=_reaction_size_bands,
    side_clauses={
        "?anchorSide": rf"\bAmong the reactions in which the compound entry named «name»{_N} takes part, on either side, which involve\b",
        "?side": (
            r"\bwhich involve at least -?\d+ and at most -?\d+ different compounds in total, "
            r"counting a compound once even if it appears on both sides\?"
        ),
    },
    joint_sides=("?anchorSide", "?side"),
    anchors=(Anchor("compound", label_param="name"),),
    membership_mode="n.a.",
    forms=("grouping", "distinct_count"),
    features={"grouping": "having", "grouping_constraints": 2},
    population_pattern=c1.REACTIONS_BY_SIZE.population_pattern,
    # Both constraints matter: dropping either one gives other rows (the first undo is the round-1 form).
    step_check=StepCheck(
        "each_constraint_matters",
        undo=(_size_gold("COUNT(DISTINCT ?compound) >= $threshold"), _size_gold("COUNT(DISTINCT ?compound) <= $having_max")),
    ),
)


def _top_k_classes_capped(rows: list[PoolRow], knobs: Knobs) -> Iterable[dict[str, Any]]:
    """Classes whose compounds in at most having_max reactions have a strict gap after the k-th count.

    Round 1's bounds stay (code reviews of 3ea978e): at least max(min_population, k + 1) linked compounds, and a
    strict gap after the k-th count over all of them. A class that fails round 1's gap stays in the pool but is
    rejected when drawn (ROUND_1_BOUND), so the draw order of every other class is the one 3ea978e drew from.
    """
    by_class: "OrderedDict[str, list[PoolRow]]" = OrderedDict()
    for r in rows:
        by_class.setdefault(r["class"], []).append(r)
    k = knobs.top_k
    for cls, members in by_class.items():
        if not _label_ok(members[0]["label"]) or len(members) < max(knobs.min_population, k + 1):
            continue
        every = sorted((int(m["n"]) for m in members), reverse=True)
        counts = [n for n in every if n <= knobs.having_max]
        if len(counts) > k and counts[k - 1] > counts[k]:
            yield {
                "entity": check_iri(cls), "label": members[0]["label"], "k": k,
                "k_words": c1._NUMBER_WORDS.get(k, str(k)), "having_max": knobs.having_max,
                **({} if every[k - 1] > every[k] else {"_excluded": ROUND_1_BOUND}),
            }


TOP_COMPOUNDS_IN_CLASS = _comp(
    name="top_compounds_in_class",
    description=(
        "Among the compounds whose ChEBI class is a direct subclass of X and that take part in at most having_max "
        "master reactions, the k in the most, each with its count (GROUP BY with HAVING, ORDER BY count then name, "
        "LIMIT k); only classes with a strict gap after the k-th count among them."
    ),
    candidate_query=c1.TOP_COMPOUNDS_IN_CLASS.candidate_query,
    gold_pattern=c1.TOP_COMPOUNDS_IN_CLASS.gold_pattern.replace(
        "    GROUP BY ?compound\n", "    GROUP BY ?compound\n    HAVING (COUNT(DISTINCT ?reaction) <= $having_max)\n"
    ),
    question_pattern=(
        "Of the compounds whose chemical class is a direct subclass (one level down, not the class itself) of the "
        'chemical class named "$label", consider only those that take part in at most $having_max reactions, on '
        "either side, counting each reaction once in its undirected form; of those, which $k_words take part in the "
        "most reactions? List each with its number of reactions; if compounds are tied, take them in alphabetical "
        f"order of their names. Polymers do not count through the class of their repeating unit. {COMPOUNDS_AS}"
    ),
    answer_columns=(AnswerColumn("compound", IRI_COL), AnswerColumn("count", NUM_COL)),
    expand=_top_k_classes_capped,
    side_clauses={
        "?side": (
            r"\bconsider only those that take part in at most \d+ reactions, on either side, counting each reaction "
            r"once in its undirected form; of those, which (?:one|two|three|four|five|six|\d+) take part in the most reactions\?"
        )
    },
    anchors=(Anchor("chemical_class"),),
    membership_mode="direct",
    forms=("grouping", "distinct_count"),
    features={"grouping": "order_limit", "grouping_constraints": 2},
    population_pattern=c1.TOP_COMPOUNDS_IN_CLASS.population_pattern,
    prefer=("direct_vs_transitive",),
    step_check=StepCheck("undo_changes", undo=(c1.TOP_COMPOUNDS_IN_CLASS.gold_pattern,)),
)


# --------------------------------------------------------------------------
# path hops (one more edge; see STEP_RULES["hops"])




_CLASS_OR_BELOW = "{ GRAPH $CHEBI { ?class rdfs:subClassOf+ <$entity> . } } UNION { VALUES ?class { <$entity> } }"
_COMPOUNDS_OF_CLASS_BODY = f"""
  {_CLASS_OR_BELOW}
  GRAPH $RHEA {{
    ?compound rh:chebi ?class .
    {_TAKES_PART}
    ?reaction rh:ec ?ec .
  }}"""

COMPOUNDS_OF_CLASS = _comp(
    name="compounds_of_class",
    description=(
        "Rhea compounds whose ChEBI class is X or below X at any depth and that take part in a master reaction "
        "with an EC number (class -> compound -> participant -> side -> reaction -> EC, 6 hops)."
    ),
    candidate_query=c1.COMPOUNDS_OF_CLASS.candidate_query,
    gold_pattern=f"SELECT DISTINCT ?compound WHERE {{{_COMPOUNDS_OF_CLASS_BODY}\n}}\nORDER BY ?compound",
    question_pattern=(
        'Which compounds belong to the chemical class named "$label", either directly or through any of its subclasses '
        f"at any depth, and take part in at least one reaction that has an EC number ({UNDIRECTED}), on either side? "
        "Polymers do not count through the class of their repeating unit, and generic compounds do not count through "
        f"their reactive parts. {COMPOUNDS_AS}"
    ),
    answer_columns=(AnswerColumn("compound", IRI_COL),),
    expand=c1._classes_sized,
    side_clauses={
        "?side": r"\band take part in at least one reaction that has an EC number \(taking each reaction in its undirected form\), on either side\?"
    },
    anchors=(Anchor("chemical_class"),),
    membership_mode="transitive",
    hops=6,
    forms=("path",),
    features={"hops": 6},
    prefer=("direct_vs_transitive",),
    step_check=StepCheck(
        "fan_out_and_undo_changes",
        fan_out=f"SELECT DISTINCT ?reaction ?compound WHERE {{{_COMPOUNDS_OF_CLASS_BODY}\n}}",
        undo=(c1.COMPOUNDS_OF_CLASS.gold_pattern,),
    ),
)


def _compounds_with_ec_numbers(rows: list[PoolRow], knobs: Knobs) -> Iterable[dict[str, Any]]:
    # Round 1's bounds stay: min_population reactions, min_rows..max_rows of them with an EC number (w); k >= 2
    # distinct EC numbers is the hop's.
    for r in rows:
        if (_label_ok(r["name"]) and int(r["n"]) >= knobs.min_population and knobs.min_rows <= int(r["w"]) <= knobs.max_rows
                and int(r["k"]) >= 2):
            yield {"entity": check_iri(r["compound"]), "name": r["name"]}


_SHARED_EC_BODY = """
  GRAPH $RHEA {
    ?anchorParticipant rh:compound <$entity> .
    ?anchorSide rh:contains ?anchorParticipant .
    ?anchorReaction rh:side ?anchorSide ;
                    rdfs:subClassOf rh:Reaction ;
                    rh:ec ?ec .
    ?reaction rh:ec ?ec ;
              rdfs:subClassOf rh:Reaction .
  }"""
_OWN_REACTION = """
  FILTER NOT EXISTS {
    GRAPH $RHEA {
      ?reaction rh:side ?ownSide .
      ?ownSide rh:contains ?ownParticipant .
      ?ownParticipant rh:compound <$entity> .
    }
  }"""

R2_OWN_REACTIONS_EXCLUDED = Reading(
    "own_reactions_excluded",
    "only the other reactions with those EC numbers, not the ones the named compound takes part in",
    rf"\bwhether or not «name»{_N} takes part in it\b",
    pattern=f"SELECT DISTINCT ?reaction ?ec WHERE {{{_SHARED_EC_BODY}{_OWN_REACTION}\n}}\nORDER BY ?reaction ?ec",
)

R2_EVERY_EC_OF_LISTED = Reading(
    "every_ec_of_the_listed_reactions",
    "each listed reaction with each of its EC numbers, not only the shared one",
    r"(?<!not )\bwith that EC number\. Give one row per reaction and EC number\b",
    pattern=f"SELECT DISTINCT ?reaction ?otherEc WHERE {{{_SHARED_EC_BODY}\n  GRAPH $RHEA {{ ?reaction rh:ec ?otherEc . }}\n}}\nORDER BY ?reaction ?otherEc",
)

COMPOUND_REACTION_EC_NUMBERS = _comp(
    name="compound_reaction_ec_numbers",
    description=(
        "Each master reaction that has an EC number of a reaction a compound takes part in, with that EC number "
        "(compound -> participant -> side -> reaction -> EC -> reaction, 5 hops)."
    ),
    # Round 1's bounds on the compound stay (the EC edge is optional, so they count reactions of any kind): at most
    # 60 reactions, some with and some without an EC number, and in expand at least min_population reactions and
    # min_rows..max_rows with an EC number (code reviews of a0024f4 and e892710). COUNT(DISTINCT ?ec) >= 2 is the hop's.
    candidate_query=f"""
SELECT ?compound ?name (COUNT(DISTINCT ?ec) AS ?k) (COUNT(DISTINCT ?reaction) AS ?n) (COUNT(DISTINCT ?withEc) AS ?w) WHERE {{
  GRAPH $RHEA {{
    ?compound rh:name ?name .
    {_TAKES_PART}
    OPTIONAL {{ ?reaction rh:ec ?ec . BIND(?reaction AS ?withEc) }}
  }}
}}
GROUP BY ?compound ?name
HAVING (COUNT(DISTINCT ?ec) >= 2 && COUNT(DISTINCT ?withEc) < COUNT(DISTINCT ?reaction) && COUNT(DISTINCT ?reaction) <= 60)
ORDER BY ?compound
""",
    gold_pattern=f"SELECT DISTINCT ?reaction ?ec WHERE {{{_SHARED_EC_BODY}\n}}\nORDER BY ?reaction ?ec",
    question_pattern=(
        'For the compound entry named "$name", take the EC numbers of the reactions it takes part in (on either side), '
        'and list each reaction that has one of those EC numbers, whether or not "$name" takes part in it, with that '
        "EC number. Give one row per reaction and EC number: the reaction by the IRI of its undirected form, then the "
        "EC number by its IRI."
    ),
    answer_columns=(AnswerColumn("reaction", IRI_COL), AnswerColumn("ec", IRI_COL)),
    expand=_compounds_with_ec_numbers,
    side_clauses={"?anchorSide": rf"\bFor the compound entry named «name»{_N}, take the EC numbers of the reactions it takes part in \(on either side\)"},
    anchors=(Anchor("compound", label_param="name"),),
    membership_mode="n.a.",
    hops=5,
    forms=("path",),
    features={"hops": 5},
    step_check=StepCheck(
        "fan_out_and_undo_changes",
        fan_out=f"SELECT DISTINCT ?near ?reaction ?ec WHERE {{{_SHARED_EC_BODY}\n  BIND(?ec AS ?near)\n}}",
        undo=(c1.COMPOUND_REACTION_EC_NUMBERS.gold_pattern,),
    ),
)


def _reactions_with_ec_numbers(rows: list[PoolRow], knobs: Knobs) -> Iterable[dict[str, Any]]:
    for r in rows:
        if _label_ok(r["label"], MAX_REACTION_LABEL_CHARS) and int(r["k"]) >= 2:
            yield {"entity": check_iri(r["reaction"]), "label": r["label"]}


_EC_OF_REACTION_BODY = """
  GRAPH $RHEA {
    <$entity> rh:ec ?ec .
    ?reaction rh:ec ?ec ;
              rdfs:subClassOf rh:Reaction ;
              rh:side ?side .
    ?side rh:contains ?participant .
    ?participant rh:compound ?compound .
  }"""

R2_ANCHOR_REACTION_EXCLUDED = Reading(
    "anchor_reaction_excluded",
    "the reactions with those EC numbers other than the named reaction itself",
    rf"\bof the reaction «label»{_N}, including that reaction itself\b",
    pattern=f"SELECT DISTINCT ?compound WHERE {{{_EC_OF_REACTION_BODY}\n  FILTER(?reaction != <$entity>)\n}}\nORDER BY ?compound",
)

R2_ALL_EC_NUMBERS = Reading(
    "all_ec_numbers_required",
    "only the reactions that have every EC number of the named reaction, not any of them",
    r"(?<!not )\bin the reactions that have any of the EC numbers of the reaction\b",
    pattern="""SELECT DISTINCT ?compound WHERE {
  GRAPH $RHEA {
    <$entity> rh:ec ?ec .
    ?reaction rh:ec ?ec ;
              rdfs:subClassOf rh:Reaction ;
              rh:side ?side .
    ?side rh:contains ?participant .
    ?participant rh:compound ?compound .
    FILTER NOT EXISTS {
      <$entity> rh:ec ?requiredEc .
      FILTER NOT EXISTS { ?reaction rh:ec ?requiredEc . }
    }
  }
}
ORDER BY ?compound""",
)

EC_COMPOUNDS = _comp(
    name="ec_compounds",
    description=(
        "Compounds of the master reactions that carry an EC number of a named reaction with at least two EC numbers "
        "(reaction -> EC <- reaction -> side -> participant -> compound, 5 hops)."
    ),
    candidate_query="""
SELECT ?reaction ?label (COUNT(DISTINCT ?ec) AS ?k) WHERE {
  GRAPH $RHEA {
    ?reaction rdfs:subClassOf rh:Reaction ;
              rdfs:label ?label ;
              rh:ec ?ec .
  }
}
GROUP BY ?reaction ?label
HAVING (COUNT(DISTINCT ?ec) >= 2)
ORDER BY ?reaction
""",
    gold_pattern=f"SELECT DISTINCT ?compound WHERE {{{_EC_OF_REACTION_BODY}\n}}\nORDER BY ?compound",
    question_pattern=(
        "Which compounds take part, on either side, in the reactions that have any of the EC numbers of the reaction "
        f'"$label", including that reaction itself ({UNDIRECTED})? {COMPOUNDS_AS}'
    ),
    answer_columns=(AnswerColumn("compound", IRI_COL),),
    expand=_reactions_with_ec_numbers,
    side_clauses={"?side": r"\bWhich compounds take part, on either side, in the reactions that have any of the EC numbers of the reaction\b"},
    anchors=(Anchor("reaction"),),
    membership_mode="n.a.",
    hops=5,
    forms=("path",),
    features={"hops": 5},
    step_check=StepCheck("fan_out", fan_out=f"SELECT DISTINCT ?ec ?compound WHERE {{{_EC_OF_REACTION_BODY}\n}}"),
)


def _compounds_with_ec_reactions(low: int, high: int):
    def expand(rows: list[PoolRow], knobs: Knobs) -> Iterable[dict[str, Any]]:
        for r in rows:
            if _label_ok(r["name"]) and int(r["k"]) >= 2 and low <= int(r["n"]) <= high:
                yield {"entity": check_iri(r["compound"]), "name": r["name"]}

    return expand


# Per compound: its master reactions (n) and those with an EC number (k).
_EC_REACTIONS_PER_COMPOUND = f"""
SELECT ?compound ?name (COUNT(DISTINCT ?reaction) AS ?n) (COUNT(DISTINCT ?ecReaction) AS ?k) WHERE {{
  GRAPH $RHEA {{
    ?compound rh:name ?name .
    {_TAKES_PART}
    OPTIONAL {{ ?reaction rh:ec ?anyEc . BIND(?reaction AS ?ecReaction) }}
  }}
}}
GROUP BY ?compound ?name
ORDER BY ?compound
"""

_OPPOSITE_BODY = """
  GRAPH $RHEA {
    ?participant rh:compound <$entity> .
    ?side rh:contains ?participant .
    ?reaction rh:side ?side , ?otherSide ;
              rdfs:subClassOf rh:Reaction ;
              rh:ec ?ec .
    ?otherSide rh:contains ?otherParticipant .
    ?otherParticipant rh:compound ?other .
  }
  FILTER(?side != ?otherSide && ?other != <$entity>)"""

OPPOSITE_SIDE_COMPOUNDS = _comp(
    name="opposite_side_compounds",
    description=(
        "Compounds on the other side of a master reaction with an EC number from a compound, other than the compound "
        "itself (compound -> participant -> side -> reaction (-> EC) -> other side -> participant -> compound, 7 hops)."
    ),
    candidate_query=_EC_REACTIONS_PER_COMPOUND,
    gold_pattern=f"SELECT DISTINCT ?other WHERE {{{_OPPOSITE_BODY}\n}}\nORDER BY ?other",
    question_pattern=(
        'In the reactions that have an EC number and that the compound entry named "$name" takes part in, on either '
        f'side ({UNDIRECTED}), which compounds appear on the other side of the reaction from it? Do not include "$name" '
        f"itself. {COMPOUNDS_AS}"
    ),
    answer_columns=(AnswerColumn("compound", IRI_COL),),
    # Round 1's 2 to 6 reactions (code review of a0024f4: 12 was a ceiling the hop does not need); k >= 2 is the hop's.
    expand=_compounds_with_ec_reactions(2, 6),
    side_clauses={
        "?side": rf"\bthat the compound entry named «name»{_N} takes part in, on either side\b",
        "?otherSide": r"\bwhich compounds appear on the other side of the reaction from it\?",
    },
    anchors=(Anchor("compound", label_param="name"),),
    membership_mode="n.a.",
    hops=7,
    forms=("path",),
    features={"hops": 7},
    step_check=StepCheck(
        "fan_out_and_undo_changes",
        fan_out=f"SELECT DISTINCT ?reaction ?other WHERE {{{_OPPOSITE_BODY}\n}}",
        undo=(c1.OPPOSITE_SIDE_COMPOUNDS.gold_pattern,),
    ),
)

_PARTNER_BODY = """
  GRAPH $RHEA {
    ?anchorParticipant rh:compound <$entity> .
    ?anchorSide rh:contains ?anchorParticipant .
    ?reaction rh:side ?anchorSide ;
              rdfs:subClassOf rh:Reaction ;
              rh:ec ?ec ;
              rh:side ?side .
    ?side rh:contains ?participant .
    ?participant rh:compound ?other .
  }
  FILTER(?other != <$entity>)"""

PARTNER_COUNT = _comp(
    name="partner_count",
    description=(
        "Number of distinct compounds that share a master reaction with an EC number with a compound, the compound "
        "itself not counted (COUNT DISTINCT across a 7-hop join)."
    ),
    candidate_query=_EC_REACTIONS_PER_COMPOUND,
    gold_pattern=f"SELECT (COUNT(DISTINCT ?other) AS ?count) WHERE {{{_PARTNER_BODY}\n}}",
    question_pattern=(
        "How many different compounds take part, on either side, in at least one reaction that has an EC number "
        f'({UNDIRECTED}) and that the compound entry named "$name" also takes part in, on either side? Do not count '
        '"$name" itself. Give the number.'
    ),
    answer_columns=(AnswerColumn("count", NUM_COL),),
    expand=_compounds_with_ec_reactions(3, 40),
    side_clauses={
        "?anchorSide": rf"\bthat the compound entry named «name»{_N} also takes part in, on either side\?",
        "?side": r"\bHow many different compounds take part, on either side, in at least one reaction that has an EC number\b",
    },
    joint_sides=("?anchorSide", "?side"),
    anchors=(Anchor("compound", label_param="name"),),
    membership_mode="n.a.",
    hops=7,
    forms=("distinct_count", "path"),
    features={"hops": 7},
    single_row=True,
    step_check=StepCheck(
        "fan_out_and_undo_changes",
        fan_out=f"SELECT DISTINCT ?reaction ?other WHERE {{{_PARTNER_BODY}\n}}",
        undo=(c1.PARTNER_COUNT.gold_pattern,),
    ),
)


BELOW_SUBCLASS, BELOW_CLASS = "below_subclass", "below_class"


def _descendant_charge_thresholds(rows: list[PoolRow], knobs: Knobs) -> Iterable[dict[str, Any]]:
    """Round 1's thresholds on the classes below a direct subclass (the round-2 population), with round 1's bounds
    kept on the classes below the class itself (external review of 3ea978e).

    Rows carry a scope: below_subclass (the hop's population) or below_class (round 1's). Round 1 asks of the
    classes below the class at least min_population of them, two charges, and a threshold answer of
    min_rows..max_rows that is a proper subset; an option whose round-1 answer fails that stays in the pool but is
    rejected when drawn (ROUND_1_BOUND), so the options and their order are the ones 3ea978e drew from.
    """
    narrowed = [r for r in rows if r["scope"] == BELOW_SUBCLASS]
    original: dict[str, OrderedDict[int, int]] = {}
    for r in rows:
        if r["scope"] == BELOW_CLASS:
            original.setdefault(r["class"], OrderedDict())[int(r["charge"])] = int(r["n"])
    for option in c1._descendant_charge_thresholds(narrowed, knobs):
        counts = original.get(option["entity"], OrderedDict())
        total = sum(counts.values())
        n = sum(c for x, c in counts.items() if c1._COMPARE[option["op"]](x, option["threshold"]))
        if total < knobs.min_population or len(counts) < 2 or not (knobs.min_rows <= n <= knobs.max_rows and n < total):
            option = {**option, "_excluded": ROUND_1_BOUND}
        yield option


def _below_a_subclass(path: str, extra: str = "") -> str:
    return f"""SELECT DISTINCT ?class ?charge WHERE {{
  GRAPH $CHEBI {{
    ?subclass rdfs:subClassOf <$entity> .
    ?class {path} ?subclass ;
           chemrof:charge ?c .
  }}
  BIND(xsd:integer(?c) AS ?charge)
  FILTER(?charge $op $threshold){extra}
}}
ORDER BY ?class"""


R2_SUBCLASS_ITSELF = Reading(
    "subclass_itself",
    "the direct subclasses themselves counted among the classes below them",
    r"\(not the direct subclass itself\)",
    pattern=_below_a_subclass("rdfs:subClassOf*"),
)
R2_DIRECT_SUBCLASSES_EXCLUDED = Reading(
    "direct_subclasses_excluded",
    "a class that is itself a direct subclass of the named class left out even when it lies below another one",
    rf"\bA direct subclass of «label»{_N} counts too if it lies below another of its direct subclasses\b",
    pattern=_below_a_subclass(
        "rdfs:subClassOf+", "\n  FILTER NOT EXISTS { GRAPH $CHEBI { ?class rdfs:subClassOf <$entity> . } }"
    ),
)

DESCENDANTS_BY_CHARGE = _comp(
    name="descendants_by_charge",
    description=(
        "ChEBI classes below a direct subclass of X at any depth whose ChEBI charge passes a threshold, each with its "
        "charge (X <- direct subclass <- subClassOf+ class -> charge, 3 hops; numeric filter with the xsd:integer cast)."
    ),
    # Per class and charge: the classes below a direct subclass (the round-2 population) and, for round 1's bounds,
    # the classes below the class itself.
    candidate_query=f"""
SELECT ?class ?label ?scope ?charge (COUNT(DISTINCT ?d) AS ?n) WHERE {{
  {{
    SELECT ?class WHERE {{ GRAPH $CHEBI {{ ?member rdfs:subClassOf+ ?class . }} FILTER(isIRI(?class)) }}
    GROUP BY ?class
    HAVING (COUNT(DISTINCT ?member) >= 5 && COUNT(DISTINCT ?member) <= 200)
  }}
  {{
    GRAPH $CHEBI {{
      ?mid rdfs:subClassOf ?class .
      ?d rdfs:subClassOf+ ?mid ;
         chemrof:charge ?c .
    }}
    BIND("{BELOW_SUBCLASS}" AS ?scope)
  }}
  UNION
  {{
    GRAPH $CHEBI {{
      ?d rdfs:subClassOf+ ?class ;
         chemrof:charge ?c .
    }}
    BIND("{BELOW_CLASS}" AS ?scope)
  }}
  GRAPH $CHEBI {{ ?class rdfs:label ?label . }}
  BIND(xsd:integer(?c) AS ?charge)
}}
GROUP BY ?class ?label ?scope ?charge
ORDER BY ?class ?scope ?charge
""",
    gold_pattern=_below_a_subclass("rdfs:subClassOf+"),
    question_pattern=(
        'Which chemical classes lie below one of the direct subclasses (one level down) of the chemical class named '
        '"$label", at any depth below that direct subclass (not the direct subclass itself), and have a charge '
        '$comparison? A direct subclass of "$label" counts too if it lies below another of its direct subclasses. '
        f"List each such class with its charge. {CLASSES_AS}"
    ),
    answer_columns=(AnswerColumn("class", IRI_COL), AnswerColumn("charge", NUM_COL)),
    expand=_descendant_charge_thresholds,
    anchors=(Anchor("chemical_class"),),
    membership_mode="transitive",
    hops=3,
    forms=("path",),
    features={"hops": 3},
    population_pattern="""SELECT (COUNT(DISTINCT ?class) AS ?n) WHERE {
  GRAPH $CHEBI {
    ?subclass rdfs:subClassOf <$entity> .
    ?class rdfs:subClassOf+ ?subclass ;
           chemrof:charge ?c .
  }
}""",
    prefer=("direct_vs_transitive",),
    step_check=StepCheck(
        "fan_out_and_undo_changes",
        undo=(c1.DESCENDANTS_BY_CHARGE.gold_pattern,),
        fan_out="""SELECT DISTINCT ?subclass ?class ?charge WHERE {
  GRAPH $CHEBI {
    ?subclass rdfs:subClassOf <$entity> .
    ?class rdfs:subClassOf+ ?subclass ;
           chemrof:charge ?c .
  }
  BIND(xsd:integer(?c) AS ?charge)
  FILTER(?charge $op $threshold)
}""",
    ),
)

_MICROSPECIES_BODY = f"""
  GRAPH $CHEBI {{ <$entity> chemrof:has_major_microspecies_at_pH7_3 ?species . }}
  GRAPH $RHEA {{
    ?compound rh:chebi ?species .
    {_TAKES_PART}
    ?reaction rh:ec ?ec .
  }}"""


MICROSPECIES_REACTIONS = _comp(
    name="microspecies_reactions",
    description=(
        "EC numbers of the master reactions of the major microspecies at pH 7.3 of a ChEBI class (ChEBI class -> "
        "microspecies -> compound -> participant -> side -> reaction -> EC, 6 hops)."
    ),
    candidate_query=c1.MICROSPECIES_REACTIONS.candidate_query,
    gold_pattern=f"SELECT DISTINCT ?ec WHERE {{{_MICROSPECIES_BODY}\n}}\nORDER BY ?ec",
    question_pattern=(
        "Which EC numbers do the reactions have in which the major microspecies at pH 7.3 of the chemical class named "
        f'"$label" takes part, on either side ({UNDIRECTED})? Polymers do not count through the class of their '
        f"repeating unit, and generic compounds do not count through their reactive parts. {EC_AS}"
    ),
    answer_columns=(AnswerColumn("ec", IRI_COL),),
    # Round 1's 2..30 reactions of the microspecies (code review of e892710); at least 2 is also the hop's fan-out.
    expand=c1.MICROSPECIES_REACTIONS.expand,
    side_clauses={"?side": rf"\bof the chemical class named «label»{_N} takes part, on either side\b"},
    anchors=(Anchor("chemical_class"),),
    membership_mode="n.a.",
    hops=6,
    forms=("path",),
    features={"hops": 6},
    step_check=StepCheck("fan_out", fan_out=f"SELECT DISTINCT ?reaction ?ec WHERE {{{_MICROSPECIES_BODY}\n}}"),
)

_BELOW_CLASS_EC_BODY = f"""
  {_CLASS_OR_BELOW}
  GRAPH $RHEA {{
    ?compound rh:chebi ?class .
    {_TAKES_PART}
    ?reaction rh:ec ?ec .
  }}"""

REACTION_COUNT_BELOW_CLASS = _comp(
    name="reaction_count_below_class",
    description=(
        "Number of distinct EC numbers of the master reactions with a compound whose ChEBI class is X or below X at "
        "any depth (COUNT DISTINCT across a cross-graph join, 6 hops)."
    ),
    candidate_query=c1.REACTION_COUNT_BELOW_CLASS.candidate_query,
    gold_pattern=f"SELECT (COUNT(DISTINCT ?ec) AS ?count) WHERE {{{_BELOW_CLASS_EC_BODY}\n}}",
    question_pattern=(
        "How many different EC numbers do the reactions have that involve, on either side, a compound of the chemical "
        f'class named "$label" or of one of its subclasses at any depth ({UNDIRECTED})? Polymers do not count through '
        "the class of their repeating unit, and generic compounds do not count through their reactive parts. "
        "Give the number."
    ),
    answer_columns=(AnswerColumn("count", NUM_COL),),
    expand=c1._classes_named,
    side_clauses={"?side": r"\bthat involve, on either side, a compound of the chemical class named «label»"},
    anchors=(Anchor("chemical_class"),),
    membership_mode="transitive",
    hops=6,
    forms=("distinct_count",),
    features={"hops": 6},
    single_row=True,
    prefer=("direct_vs_transitive",),
    step_check=StepCheck("fan_out", fan_out=f"SELECT DISTINCT ?reaction ?ec WHERE {{{_BELOW_CLASS_EC_BODY}\n}}"),
)


# --------------------------------------------------------------------------
# The step table

STEP_RULES = {
    "axis_precedence": (
        "each active compositional variant is raised along exactly one axis, read from its round-1 forms in this "
        "order: a set operation (intersection) -> populations; a comparison of entities (a comparison with a "
        "reference entity included) -> populations, counting the compared entities; negation -> negation depth "
        "(nesting); grouping (GROUP BY with HAVING or ORDER BY / LIMIT) -> grouping constraints; otherwise (paths, "
        "distinct counts across a join) -> path hops. The variant's other constructs keep their round-1 values"
    ),
    "populations": (
        "one more population: a third class or compound in an intersection; a third reaction compared (on the "
        "compared side of round 1's directed difference, read with the other one as a set); a second reference "
        "compound for a threshold comparison"
    ),
    "negation_depth": (
        "negation depth is nesting depth (NOT EXISTS inside NOT EXISTS): the round-1 negation stays and a universal "
        "is added (NOT EXISTS an X with NOT EXISTS a link: every X linked)"
    ),
    "grouping_constraints": (
        "one more HAVING constraint on the grouped count: at most having_max (one value for every grouping variant); "
        "an ORDER BY / LIMIT or a HAVING counts as one constraint"
    ),
    "hops": (
        "path hops count the edges of the gold pattern that join the named entity, the answer and the reactions the "
        "answer is qualified by (as round 1 counted compounds_of_class: 5); a property path such as "
        "rdfs:subClassOf+ or the 'X or below' union counts as one edge, the master-reaction type test and the label "
        "that names the anchor do not count, an attribute that gives an answer column does. The hop step adds one "
        "edge that fans out: the reaction-EC number edge (rh:ec) wherever the path has a reaction - a reaction "
        "inside the path or qualifying the answer must have an EC number, an answer or counted reaction moves on to "
        "its EC numbers, an EC number at the answer end moves on to the reactions that have it, and an EC-number "
        "anchor moves back to a reaction that has it; in the one path without a reaction (descendants_by_charge) "
        "the class anchor moves up one direct-subclass edge. rh:chebi is not used: it links each class to at most "
        "one participating compound in this store, so an edge through it would fan out to nothing"
    ),
    "load_bearing": (
        "every raised question's step must matter, checked on the store (rejection 'step_not_load_bearing'): set "
        "operations - dropping any one population changes the gold rows; comparisons - the compared values are "
        "pairwise distinct (no ties), and the winner's (or the binding reference's) mention slot is balanced within "
        "each (variant, split) by a seeded permutation of the mention order; grouping - the round-1 form (the step "
        "undone) gives other rows, and for two HAVING bounds dropping either one does; negation - the answer differs "
        "from the round-1 form, from the positive reading (some X linked) and from the form without the round-1 negation; hops - answer items come through at "
        "least two entities at the near end of the added edge and no single one of them accounts for all, and where "
        "the hop restricts the same answer (an EC-number condition on a reaction of the path; the class anchor moved "
        "up one edge) the round-1 form on the same parameters also gives other rows (a hop that moves the answer on "
        "to other entities, or the anchor to another kind, changes the answer by construction)"
    ),
}

_HOPS_NOTE = "hop counted by the rule; "

# Round-1 pool bounds (design ruling in the design log, with the lead's reading of 3(i): keep round 1's bounds).
# Every round-1 bound of every raised variant stays on the quantity round 1 bounded, and the row check bounds the
# round-2 answer too (knobs.min_rows..max_rows on every compositional list gold); an anchor-type restriction applies
# to every anchor of the same role; a bound tied to a replaced anchor type is dropped, or replaced by an obvious
# analog on the new anchor; a bound the step itself changes is listed with the change. The table lists them all.
BOUND_ROLES = ("answer_size", "population", "eligibility", "anchor_type", "anchor_relation", "replaced_anchor", "comparison")
BOUNDS_RULE = (
    "round-1 pool bounds (design ruling in the design log; 3(i) read as: keep round 1's bounds): every round-1 bound of a "
    "raised variant (its candidate query, expand and population check) stays on the quantity round 1 bounded, and the "
    "generator's row check (knobs.min_rows..max_rows) bounds every compositional list answer of round 2 as well; an "
    "anchor-type restriction applies to every anchor of the same role; a bound tied to a replaced anchor type is "
    "dropped, unless an analogous bound on the new anchor is obvious, which then applies; a bound the structural step "
    "itself changes is listed with the change. A kept bound that the round-2 pool does not apply by itself rejects "
    "a candidate when it is drawn (rejection round_1_bound), so the draw order of the other candidates stays. "
    "Usable labels (base.usable_label) apply to every named entity in both rounds and are not listed"
)


def _bound(variant: str, bound: str, role: str, round_1: str, round_2: str, action: str, changed: bool = False,
           changed_3ea978e: bool = False) -> dict[str, Any]:
    return {"variant": variant, "bound": bound, "role": role, "round_1": round_1, "round_2": round_2, "action": action,
            "changed_after_e892710": changed, "changed_after_3ea978e": changed_3ea978e}


_SAME = "the same"
ROUND_1_BOUNDS: tuple[dict[str, Any], ...] = (
    _bound("type_members/common_descendants", "at most 400 classes below each compared class", "eligibility",
           "HAVING (COUNT(DISTINCT ?da) <= 400), the same for ?db", "HAVING (COUNT(DISTINCT ?d) <= 400) on each parent", "kept"),
    _bound("type_members/common_descendants", "compared classes neither below another", "anchor_relation",
           "MINUS: neither parent below the other", "not in the pool: the step check rejects it (every compared class "
           "must change the answer, and one below another adds nothing)", "covered_by_step_check"),
    _bound("type_members/common_descendants", "the compared classes are direct parents of one class", "anchor_relation",
           "?shared rdfs:subClassOf ?a , ?b (two direct parents of one class)",
           "three direct parents of one class", "kept", changed_3ea978e=True),
    _bound("type_members/common_descendants", "population: the smallest subtree at least min_population, the answer smaller",
           "population", "the generator's population check on the smaller of two subtrees",
           "the same check on the smallest of three", "kept"),
    _bound("type_members/compounds_of_class", "2..50 compounds below the class (query), 2..30 (expand)", "answer_size",
           "HAVING 2..50 and min_rows..max_rows on the compounds below the class", _SAME, "kept"),
    _bound("one_hop_attribute/compound_reaction_ec_numbers", "at most 60 reactions of any kind", "eligibility",
           "COUNT(DISTINCT ?reaction) <= 60", "the same, the EC edge optional so all reactions count", "kept"),
    _bound("one_hop_attribute/compound_reaction_ec_numbers", "at least min_population reactions", "population",
           "n >= min_population (and round 1's population check)", "n >= min_population in the pool", "kept", True),
    _bound("one_hop_attribute/compound_reaction_ec_numbers", "some reaction with and some without an EC number",
           "population", "1 <= COUNT(DISTINCT ?withEc) < COUNT(DISTINCT ?reaction)",
           "COUNT(DISTINCT ?withEc) < COUNT(DISTINCT ?reaction) with at least two EC numbers", "kept", True),
    _bound("one_hop_attribute/compound_reaction_ec_numbers", "2..30 reactions with an EC number", "answer_size",
           "min_rows <= k <= max_rows (k: reactions with an EC number)", "min_rows <= w <= max_rows, the same count",
           "kept", True),
    _bound("one_hop_attribute/citation_count_difference", "the compared counts differ", "comparison",
           "the pair's two counts differ", "the triple's three counts differ pairwise", "kept"),
    _bound("one_hop_attribute/citation_count_difference", "the compared reactions share an EC number", "anchor_relation",
           "two reactions next to each other in IRI order among the reactions of one EC number",
           "three reactions next to each other in IRI order among the reactions of one EC number", "kept",
           changed_3ea978e=True),
    _bound("one_hop_attribute/citation_count_difference", "the subject (named first) is the more cited", "comparison",
           "the more-cited reaction of the pair is named first", "the most-cited reaction of the three is named first",
           "kept", changed_3ea978e=True),
    _bound("one_hop_attribute/citation_count_difference", "every compared reaction cites a publication, labels of at "
           "most 150 characters", "eligibility", "the candidate query joins rh:citation; label at most "
           "MAX_REACTION_LABEL_CHARS", "the same query and label limit for all three", "kept", changed_3ea978e=True),
    _bound("multi_hop_path/ec_compounds", "the EC number is complete (it was named in the text)", "replaced_anchor",
           "every part of the EC number a digit or n...", "none: the anchor is a reaction, and no EC number is named",
           "dropped_anchor_replaced"),
    _bound("multi_hop_path/ec_compounds", "the EC number is carried by at least 2 reactions", "replaced_anchor",
           "HAVING (COUNT(DISTINCT ?reaction) >= 2) per EC number", "the analog on the reaction anchor: at least 2 EC "
           "numbers (the hop's fan-out needs it too)", "analog_applied"),
    _bound("multi_hop_path/ec_compounds", "2..30 compounds of the EC number", "replaced_anchor",
           "min_rows..max_rows on the compounds of the EC number", "the analog on the reaction anchor: the row check, "
           "2..30 compounds reached through its EC numbers", "analog_applied"),
    _bound("multi_hop_path/opposite_side_compounds", "2..6 reactions of the compound", "eligibility",
           "HAVING 2..6 reactions", "2..6 reactions (code review of a0024f4: it had been raised to 12)", "kept"),
    _bound("multi_hop_path/shared_reactions_2", "2..50 shared reactions (query), 2..30 (expand)", "answer_size",
           "HAVING 2..50 on the reactions both compounds take part in; min_rows..max_rows on them in expand",
           "the same on the reactions all three take part in (one builder for both; external review of 3ea978e: it "
           "had said per compound)", "kept", changed_3ea978e=True),
    _bound("multi_hop_path/shared_reactions_2", "population: the compound in fewest reactions at least min_population, "
           "the answer smaller", "population", "the generator's population check on the smaller reaction count of the "
           "two compounds", "the same check on the smallest of three", "kept", changed_3ea978e=True),
    _bound("aggregation/reactions_by_size", "min_population..60 reactions, two sizes at least", "population",
           "total >= min_population, total <= 60, at least two reaction sizes", _SAME, "kept"),
    _bound("aggregation/reactions_by_size", "thresholds >= or <= keeping a proper subset", "answer_size",
           "a threshold >= or <= that keeps a proper subset of min_rows..max_rows", "a >= threshold below having_max "
           "keeping a proper subset: the step adds HAVING <= having_max, so the threshold is the lower bound",
           "changed_by_step"),
    _bound("aggregation/top_compounds_in_class", "at least max(min_population, k + 1) linked compounds", "population",
           "len(members) >= max(min_population, k + 1)", _SAME, "kept", changed_3ea978e=True),
    _bound("aggregation/top_compounds_in_class", "a strict gap after the k-th count over all linked compounds",
           "eligibility", "counts[k-1] > counts[k] over every linked compound",
           "the same, rejected when drawn (round_1_bound; code reviews of 3ea978e: it had been checked among the "
           "compounds within having_max only); the step adds the same gap among the compounds within having_max",
           "kept", changed_3ea978e=True),
    _bound("aggregation/partner_count", "3..40 reactions of the compound", "eligibility", "3 <= n <= 40",
           "the same; at least 2 with an EC number is the hop's", "kept"),
    _bound("numeric_filter/partners_with_lower_charge", "the reference is a small molecule with a charge", "anchor_type",
           "the one reference (the anchor) is a small molecule with a charge",
           "both references are small molecules with an integer charge", "applied_to_every_reference", True),
    _bound("numeric_filter/partners_with_lower_charge", "1..6 reactions of the compound", "eligibility",
           "1 <= n <= 6 (expand)", "HAVING 1..6 (in the query)", "kept"),
    _bound("numeric_filter/partners_with_lower_charge", "population: at least min_population compounds with an "
           "integer charge in the compound's reactions, the answer smaller", "population",
           "the generator's population check (population_pattern on the compound)", "the same pattern on the same "
           "compound", "kept", changed_3ea978e=True),
    _bound("numeric_filter/descendants_by_charge", "5..200 classes below the class", "eligibility",
           "HAVING 5..200 classes below the class", _SAME, "kept"),
    _bound("numeric_filter/descendants_by_charge", "population: at least min_population classes with a charge below "
           "the class, two charges at least", "population",
           "sum(counts) >= min_population and len(counts) >= 2 over the classes below the class",
           "the same over the classes below the class, rejected when drawn (round_1_bound; external review of 3ea978e); "
           "the same again over the classes below a direct subclass, the population the hop moves the answer to",
           "kept", changed_3ea978e=True),
    _bound("numeric_filter/descendants_by_charge", "a threshold answer of 2..30 classes below the class, a proper subset",
           "answer_size", "the threshold keeps a proper subset of min_rows..max_rows of the classes below the class",
           "the same over the classes below the class, rejected when drawn (round_1_bound; external review of 3ea978e: "
           "only the narrowed answer had been bounded); the same again over the classes below a direct subclass "
           "(the round-2 answer)", "kept", changed_3ea978e=True),
    _bound("absence_negation/reactions_without_partner", "the partner among the 30 compounds in most reactions",
           "eligibility", "LIMIT 30 on compounds by reaction count", _SAME, "kept"),
    _bound("absence_negation/reactions_without_partner", "3..60 reactions of A, at least min_population", "population",
           "HAVING 3..60 reactions of A; n >= min_population", _SAME, "kept"),
    _bound("absence_negation/reactions_without_partner", "2..30 reactions of A without B (and at least 1 in the query)",
           "answer_size", "min_rows <= n - with <= max_rows; ?with < ?n", _SAME, "kept"),
    _bound("absence_negation/reactions_without_partner", "B takes part in a reaction of A and is not A", "anchor_relation",
           "the join on a shared reaction (?with >= 1); ?a != ?b", "the same query", "kept", changed_3ea978e=True),
    _bound("absence_negation/unused_child_classes", "at least min_population direct children", "population",
           "n >= min_population", _SAME, "kept"),
    _bound("absence_negation/unused_child_classes", "at least one used direct child", "population",
           "HAVING (COUNT(DISTINCT ?usedChild) >= 1)", "the same (code review of e892710)", "kept", True),
    _bound("absence_negation/unused_child_classes", "2..30 unused direct children", "answer_size",
           "min_rows <= n - used <= max_rows", "the same (code review of e892710), and min_rows..max_rows qualifying "
           "children (the round-2 answer)", "kept", True),
    _bound("cross_graph_join/microspecies_reactions", "2..30 reactions of the microspecies", "answer_size",
           "max(1, min_rows) <= n <= max_rows", "the same (code review of e892710); at least 2 is also the hop's", "kept", True),
    _bound("cross_graph_join/microspecies_reactions", "the microspecies is another class", "anchor_relation",
           "FILTER(?class != ?species)", "the same query", "kept", changed_3ea978e=True),
    _bound("cross_graph_join/reaction_count_below_class", "3..200 linked classes at or below the class", "eligibility",
           "HAVING 3..200", _SAME, "kept"),
)


def _with_notes(variant: Variant, extra: tuple[Reading, ...]) -> Variant:
    """The variant with its round-2 readings; a reading with a note_param also gives the variant its note."""
    variant = _with_readings(variant, *extra)
    notes = {r.note_param: STATUS_NOTE for r in extra if r.note_param == "any_status"}
    return replace(variant, notes={**variant.notes, **notes}) if notes else variant


def _step(template: str, round_1: str, raised: Variant, axis: str, construct: str, r1: int, note: str) -> tuple[str, StructuralStep]:
    key = f"{template}/{round_1}"
    if _r1(template, round_1).tier != COMPOSITIONAL:
        raise TemplateError(f"{key}: only compositional variants take a step")
    extra = _EXTRA.get(raised.name, ()) + tuple(
        approved_only_reading(*variables) for k, variables in STATUS_TARGETS.items() if k == f"{template}/{raised.name}"
    )
    return key, StructuralStep(axis, construct, r1, r1 + 1, _with_notes(raised, extra), note)


_EXTRA: dict[str, tuple[Reading, ...]] = {
    "citation_count_difference": (R2_CITATIONS_SUMMED,),
    "partners_with_lower_charge": (R2_REFERENCE_ANY,),
    "reactions_without_partner": (R2_SOME_MEMBER_LINKED,),
    "unused_child_classes": (R2_SOME_SUBCLASS_LINKED, R2_SUBCLASS_REQUIRED),
    "compound_reaction_ec_numbers": (R2_OWN_REACTIONS_EXCLUDED, R2_EVERY_EC_OF_LISTED),
    "ec_compounds": (R2_ANCHOR_REACTION_EXCLUDED, R2_ALL_EC_NUMBERS),
    "descendants_by_charge": (R2_SUBCLASS_ITSELF, R2_DIRECT_SUBCLASSES_EXCLUDED),
}

# The approved-only reading (study lead, design log) on every round-2 variant whose gold reaches reactions through a
# variable: listed, counted, or qualifying the answer ("compounds that take part in some reaction"); every such
# variable is restricted at once. A reaction the question names is an anchor and is not restricted, so variants that
# reach reactions only through named ones (reaction_ec, reaction_citations, reaction_compounds,
# reaction_count_min_charge, reaction_participant_charge, citation_count_difference) are not listed.
STATUS_TARGETS: dict[str, tuple[str, ...]] = {
    # lookup
    "type_members/reaction_descendants": ("?reaction",),
    "multi_hop_path/compound_reactions": ("?reaction",),
    "multi_hop_path/variant_cross_references": ("?variant",),
    "aggregation/compound_reaction_count": ("?reaction",),
    "absence_negation/reaction_children_without_ec": ("?reaction",),
    "cross_graph_join/reactions_with_chebi_child_participant": ("?reaction",),
    # compositional (raised forms)
    "type_members/compounds_of_class": ("?reaction",),
    "one_hop_attribute/compound_reaction_ec_numbers": ("?anchorReaction", "?reaction"),
    "multi_hop_path/ec_compounds": ("?reaction",),
    "multi_hop_path/opposite_side_compounds": ("?reaction",),
    "multi_hop_path/shared_reactions_3": ("?reaction",),
    "aggregation/reactions_by_size": ("?reaction",),
    "aggregation/top_compounds_in_class": ("?reaction",),
    "aggregation/partner_count": ("?reaction",),
    "numeric_filter/partners_with_lower_charge": ("?reaction",),
    "absence_negation/reactions_without_partner": ("?reaction",),
    "absence_negation/unused_child_classes": ("?reaction", "?subclassReaction"),
    "cross_graph_join/microspecies_reactions": ("?reaction",),
    "cross_graph_join/reaction_count_below_class": ("?reaction",),
}
# Answer-column names that named a database (study lead's audit, design log), renamed from round 2 on.
COLUMN_RENAMES: dict[str, dict[str, str]] = {
    "multi_hop_path/generic_reactive_parts": {"chebi_class": "class"},
    "cross_graph_join/compounds_in_chebi_children": {"chebi_label": "class_label"},
}
# The readings round 2 adds to the Rhea ones (harness.qgen_rhea.readings.RHEA_READING_NAMES).
R2_READING_NAMES = tuple(sorted({r.name for extra in _EXTRA.values() for r in extra} | {"approved_only"}))


def round_2_lookup(template: str, variant: Variant) -> Variant:
    """A lookup variant's round-2 form: neutral column names, and the approved-only reading where it applies."""
    key = f"{template}/{variant.name}"
    if variant.tier != LOOKUP:
        raise TemplateError(f"{key}: not a lookup variant")
    renames = COLUMN_RENAMES.get(key, {})
    if renames:
        variant = replace(variant, answer_columns=tuple(replace(c, name=renames.get(c.name, c.name)) for c in variant.answer_columns))
    if key in STATUS_TARGETS:
        variant = replace(
            variant, readings=variant.readings + (approved_only_reading(*STATUS_TARGETS[key]),),
            notes={**variant.notes, "any_status": STATUS_NOTE},
        )
    return variant

STEPS: dict[str, StructuralStep] = dict(
    (
        _step("type_members", "common_descendants", COMMON_DESCENDANTS, "populations", "set operation (intersection)", 2,
              "a third class whose subtree is intersected"),
        _step("type_members", "compounds_of_class", COMPOUNDS_OF_CLASS, "path_hops", "path", 5,
              "the qualifying reaction must have an EC number (reaction -> EC)"),
        _step("one_hop_attribute", "compound_reaction_ec_numbers", COMPOUND_REACTION_EC_NUMBERS, "path_hops", "path", 4,
              "the EC numbers move on to every reaction that has them (EC -> reaction)"),
        _step("one_hop_attribute", "citation_count_difference", CITATION_COUNT_DIFFERENCE, "populations",
              "comparison of two entities", 2,
              "a third reaction compared: round 1's directed difference with one more reaction on the compared "
              "side (the subject's publications minus those of the other two together); the three counts differ and "
              "each of the two changes the answer"),
        _step("multi_hop_path", "ec_compounds", EC_COMPOUNDS, "path_hops", "path", 4,
              "the EC-number anchor moves back to a named reaction with at least two EC numbers (reaction -> EC)"),
        _step("multi_hop_path", "opposite_side_compounds", OPPOSITE_SIDE_COMPOUNDS, "path_hops", "path", 6,
              "the reaction must have an EC number (reaction -> EC)"),
        _step("multi_hop_path", "shared_reactions_2", SHARED_REACTIONS_3, "populations", "set operation (intersection)", 2,
              "a third compound; the round-1 variant shared_reactions_3 (inactive in round 1) is the raised form"),
        _step("aggregation", "reactions_by_size", REACTIONS_BY_SIZE, "grouping_constraints", "grouping (HAVING)", 1,
              "added HAVING: at most having_max compounds per reaction; the round-1 threshold becomes the lower bound"),
        _step("aggregation", "top_compounds_in_class", TOP_COMPOUNDS_IN_CLASS, "grouping_constraints",
              "grouping (ORDER BY / LIMIT)", 1, "added HAVING: at most having_max reactions per compound"),
        _step("aggregation", "partner_count", PARTNER_COUNT, "path_hops", "distinct count across a join (a path)", 6,
              "the shared reaction must have an EC number (reaction -> EC)"),
        _step("numeric_filter", "partners_with_lower_charge", PARTNERS_WITH_LOWER_CHARGE, "populations",
              "comparison with a reference entity (also a path)", 1,
              "a second reference compound from the same reactions (lower than each); the two charges differ"),
        _step("numeric_filter", "descendants_by_charge", DESCENDANTS_BY_CHARGE, "path_hops", "path", 2,
              _HOPS_NOTE + "round-1 declared hops null; the class anchor moves up one direct-subclass edge"),
        _step("absence_negation", "reactions_without_partner", REACTIONS_WITHOUT_PARTNER, "negation_depth", "negation", 1,
              "every compound of the reaction linked to a class: NOT EXISTS (compound, NOT EXISTS class); the round-1 "
              "negation stays"),
        _step("absence_negation", "unused_child_classes", UNUSED_CHILD_CLASSES, "negation_depth", "negation", 1,
              "every direct subclass of the unused class linked: NOT EXISTS (subclass, NOT EXISTS linked compound); "
              "the round-1 negation stays"),
        _step("cross_graph_join", "microspecies_reactions", MICROSPECIES_REACTIONS, "path_hops", "path", 5,
              "the answer reactions move on to their EC numbers (reaction -> EC)"),
        _step("cross_graph_join", "reaction_count_below_class", REACTION_COUNT_BELOW_CLASS, "path_hops",
              "distinct count across a join (a path)", 5,
              _HOPS_NOTE + "round-1 declared hops null; the counted reactions move on to their EC numbers (reaction -> EC)"),
    )
)


def resolve_templates(templates: Iterable[type[QuestionTemplate]], knobs: Knobs) -> tuple[type[QuestionTemplate], ...]:
    """The templates with each stepped variant in the form its axis delta asks for.

    A round-1 setting (every delta 0) returns the templates themselves. With a
    delta, an active round-1 variant of that axis is replaced by its raised form,
    and an inactive variant the raised form supersedes (by name) is dropped.
    """
    templates = tuple(templates)
    if not any(knobs.delta(s.axis) for s in STEPS.values()):
        return templates
    out = []
    for t in templates:
        raised: dict[str, Variant] = {}
        for v in t.variants:
            step = STEPS.get(f"{t.id}/{v.name}")
            if step is not None and knobs.delta(step.axis) and knobs.allows(v.features):
                raised[v.name] = step.raised
        # From round 2 on, the lookup variants take their round-2 forms (round_2_lookup) as well.
        lookups = {
            v.name: round_2_lookup(t.id, v) for v in t.variants
            if v.tier == LOOKUP and (f"{t.id}/{v.name}" in STATUS_TARGETS or f"{t.id}/{v.name}" in COLUMN_RENAMES)
        }
        if not raised and not lookups:
            out.append(t)
            continue
        names = {r.name for r in raised.values()}
        variants = []
        for v in t.variants:
            if v.name in raised:
                variants.append(raised[v.name])
            elif v.name in lookups:
                variants.append(lookups[v.name])
            elif v.name not in names:
                variants.append(v)
        out.append(type(t.__name__, (t,), {"variants": tuple(variants)}))
    return tuple(out)


def step_of(template_id: str, variant_name: str) -> tuple[str, StructuralStep] | None:
    """The (round-1 key, step) whose raised form is this variant, or None."""
    for key, step in STEPS.items():
        if key.split("/")[0] == template_id and step.raised.name == variant_name:
            return key, step
    return None


def step_table() -> dict[str, Mapping[str, Any]]:
    return {key: step.to_json() for key, step in STEPS.items()}
