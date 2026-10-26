"""The compositional tier: questions that chain or combine several steps.

The ruling in the design log (2026-10-18 08:39) asks for about two thirds of the
questions to combine: paths of 3 or more hops, aggregation with grouping
(HAVING, or ORDER BY and LIMIT with a tie-break), negation over joins,
comparisons between two entities, intersections of two populations, and
counting distinct entities across a join. Every variant here names the forms
it has (``Variant.forms``) and its difficulty features (``Variant.features``),
which the generator's knobs (``base.Knobs``) allow or leave out.

Wording, against the lookup tier:

- entities are named by a label alone (no accession), and only when the label
  names exactly one entity in the store (checked after sampling);
- the output is described in plain words that fix the column order ("each
  reaction with each of its EC numbers") instead of column by column;
- no schema hints: "count each reaction once, in its undirected form" instead
  of "master reactions" (the word "undirected" fixes the scope the gold uses,
  which "whatever its direction" alone does not); one closing clause says which
  IRI to give;
- no database names: Rhea is "the reaction graph", ChEBI "the chemical class
  hierarchy".

The gold queries keep the lookup tier's explicit constraints: ChEBI hierarchy
steps scoped to the ChEBI graph, reactions restricted to master reactions with
``rdfs:subClassOf rh:Reaction``, participants through ``rh:contains``, charges
cast with ``xsd:integer``.

Compositional expand functions take the knobs as a second argument, so the
answer-size and population settings shape the candidate pools.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Iterable

from harness.qgen_rhea.base import (
    COMPOSITIONAL,
    Anchor,
    AnswerColumn,
    EngineProbe,
    Knobs,
    PoolRow,
    TemplateError,
    Variant,
    check_iri,
    usable_label,
)

IRI_COL = "iri"
NUM_COL = "number"

EC_IRI_PREFIX = "http://purl.uniprot.org/enzyme/"
# Reaction equations quoted in compositional questions stay readable.
MAX_REACTION_LABEL_CHARS = 150

REACTIONS_ONCE = "Count each reaction once, whatever its direction, and give it by the IRI of its undirected form."
# For questions whose answer is reached through reactions but is not a reaction.
UNDIRECTED = "taking each reaction in its undirected form"
# An optional homonym note after a label token, in side clauses (readings.neutral_text).
_N = r"(?: \(meaning [^)]*\))?"
COMPOUNDS_AS = "Give each compound by its compound IRI, not the IRI of its chemical class."
CLASSES_AS = "Give each class by its IRI."

# Comparison wording for thresholds on counts and charges.
_WORDING = {
    "<": "lower than {v}",
    "<=": "of at most {v}",
    ">": "greater than {v}",
    ">=": "of at least {v}",
    "=": "of exactly {v}",
}
_COUNT_WORDING = {
    "<": "fewer than {v}",
    "<=": "at most {v}",
    ">": "more than {v}",
    ">=": "at least {v}",
    "=": "exactly {v}",
}
_COMPARE = {
    "<": lambda x, v: x < v,
    "<=": lambda x, v: x <= v,
    ">": lambda x, v: x > v,
    ">=": lambda x, v: x >= v,
    "=": lambda x, v: x == v,
}
_NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}

# The participation chain from a compound to a master reaction, in the Rhea graph.
_TAKES_PART = """?participant rh:compound ?compound .
    ?side rh:contains ?participant .
    ?reaction rh:side ?side ;
              rdfs:subClassOf rh:Reaction ."""


def _comp(**kw: Any) -> Variant:
    return Variant(tier=COMPOSITIONAL, key_param=kw.pop("key_param", "entity"), **kw)


def _thresholds(counts: "OrderedDict[int, int]", knobs: Knobs, ops: Iterable[str]) -> list[tuple[str, int, int]]:
    """(op, value, rows) for every comparison keeping a proper subset of knob-sized rows."""
    total = sum(counts.values())
    out = []
    for value in counts:
        for op in ops:
            n = sum(c for x, c in counts.items() if _COMPARE[op](x, value))
            if knobs.min_rows <= n <= knobs.max_rows and n < total:
                out.append((op, value, n))
    return out


def _label_ok(label: Any, limit: int | None = None) -> bool:
    return isinstance(label, str) and usable_label(label) and (limit is None or len(label) <= limit)


# --------------------------------------------------------------------------
# type_members


def _class_pairs(rows: list[PoolRow], knobs: Knobs) -> Iterable[dict[str, Any]]:
    for r in rows:
        if _label_ok(r["labelA"]) and _label_ok(r["labelB"]):
            a, b = check_iri(r["a"]), check_iri(r["b"])
            yield {"key": f"{a}|{b}", "entity_a": a, "label_a": r["labelA"], "entity_b": b, "label_b": r["labelB"]}


def _classes_sized(rows: list[PoolRow], knobs: Knobs) -> Iterable[dict[str, Any]]:
    for r in rows:
        if _label_ok(r["label"]) and knobs.min_rows <= int(r["n"]) <= knobs.max_rows:
            yield {"entity": check_iri(r["class"]), "label": r["label"]}


COMMON_DESCENDANTS = _comp(
    name="common_descendants",
    description=(
        "ChEBI classes below both of two ChEBI classes, at any depth (intersection of two transitive populations). "
        "The pair are two parents of one ChEBI class, neither below the other."
    ),
    candidate_query="""
SELECT ?a ?labelA ?b ?labelB WHERE {
  {
    SELECT DISTINCT ?a ?labelA ?b WHERE {
      {
        SELECT ?a ?labelA WHERE { GRAPH $CHEBI { ?da rdfs:subClassOf+ ?a . ?a rdfs:label ?labelA . } }
        GROUP BY ?a ?labelA
        HAVING (COUNT(DISTINCT ?da) <= 400)
      }
      GRAPH $CHEBI { ?shared rdfs:subClassOf ?a , ?b . }
      FILTER(isIRI(?b) && STR(?a) < STR(?b))
    }
  }
  {
    SELECT ?b ?labelB WHERE { GRAPH $CHEBI { ?db rdfs:subClassOf+ ?b . ?b rdfs:label ?labelB . } }
    GROUP BY ?b ?labelB
    HAVING (COUNT(DISTINCT ?db) <= 400)
  }
  MINUS { GRAPH $CHEBI { ?a rdfs:subClassOf+ ?b . } }
  MINUS { GRAPH $CHEBI { ?b rdfs:subClassOf+ ?a . } }
}
ORDER BY ?a ?b
""",
    gold_pattern="""SELECT DISTINCT ?class WHERE {
  GRAPH $CHEBI {
    ?class rdfs:subClassOf+ <$entity_a> .
    ?class rdfs:subClassOf+ <$entity_b> .
  }
  FILTER(isIRI(?class))
}
ORDER BY ?class""",
    question_pattern=(
        'Which chemical classes are subclasses, at any depth, of both the chemical class named "$label_a" and '
        f'the chemical class named "$label_b"? {CLASSES_AS}'
    ),
    answer_columns=(AnswerColumn("class", IRI_COL),),
    key_param="key",
    expand=_class_pairs,
    anchors=(Anchor("chemical_class", "entity_a", "label_a"), Anchor("chemical_class", "entity_b", "label_b")),
    forms=("intersection",),
    features={"populations": 2},
    population_pattern="""SELECT ?n WHERE {
  { SELECT (COUNT(DISTINCT ?ca) AS ?na) WHERE { GRAPH $CHEBI { ?ca rdfs:subClassOf+ <$entity_a> . } FILTER(isIRI(?ca)) } }
  { SELECT (COUNT(DISTINCT ?cb) AS ?nb) WHERE { GRAPH $CHEBI { ?cb rdfs:subClassOf+ <$entity_b> . } FILTER(isIRI(?cb)) } }
  BIND(IF(?na < ?nb, ?na, ?nb) AS ?n)
}""",
    prefer=("direct_vs_transitive", "graph_scope"),
)

COMPOUNDS_OF_CLASS = _comp(
    name="compounds_of_class",
    description=(
        "Rhea compounds that take part in a master reaction and whose ChEBI class is X or below X at any depth "
        "(cross-graph; class -> compound -> participant -> side -> reaction)."
    ),
    candidate_query="""
SELECT ?class ?label (COUNT(DISTINCT ?compound) AS ?n) WHERE {
  GRAPH $RHEA {
    ?compound rh:chebi ?linked .
    ?participant rh:compound ?compound .
  }
  GRAPH $CHEBI {
    ?linked rdfs:subClassOf+ ?class .
    ?class rdfs:label ?label .
  }
  FILTER(isIRI(?class))
}
GROUP BY ?class ?label
HAVING (COUNT(DISTINCT ?compound) >= 2 && COUNT(DISTINCT ?compound) <= 50)
ORDER BY ?class
""",
    gold_pattern=f"""SELECT DISTINCT ?compound WHERE {{
  {{ GRAPH $CHEBI {{ ?class rdfs:subClassOf+ <$entity> . }} }} UNION {{ VALUES ?class {{ <$entity> }} }}
  GRAPH $RHEA {{
    ?compound rh:chebi ?class .
    {_TAKES_PART}
  }}
}}
ORDER BY ?compound""",
    question_pattern=(
        'Which compounds belong to the chemical class named "$label", either directly or through any of its subclasses '
        f"at any depth, and take part in at least one reaction ({UNDIRECTED}), on either side? Polymers "
        "do not count through the class of their repeating unit, and generic compounds do not count through their "
        f"reactive parts. {COMPOUNDS_AS}"
    ),
    answer_columns=(AnswerColumn("compound", IRI_COL),),
    expand=_classes_sized,
    side_clauses={"?side": r"\band take part in at least one reaction \(taking each reaction in its undirected form\), on either side\?"},
    anchors=(Anchor("chemical_class"),),
    hops=5,
    forms=("path",),
    features={"hops": 5},
    prefer=("direct_vs_transitive",),
)

TYPE_MEMBERS = (COMMON_DESCENDANTS, COMPOUNDS_OF_CLASS)


# --------------------------------------------------------------------------
# one_hop_attribute


def _compounds_named(rows: list[PoolRow], knobs: Knobs) -> Iterable[dict[str, Any]]:
    for r in rows:
        if _label_ok(r["name"]):
            yield {"entity": check_iri(r["compound"]), "name": r["name"]}


def _compounds_with_ec_reactions(rows: list[PoolRow], knobs: Knobs) -> Iterable[dict[str, Any]]:
    for r in rows:
        if _label_ok(r["name"]) and int(r["n"]) >= knobs.min_population and knobs.min_rows <= int(r["k"]) <= knobs.max_rows:
            yield {"entity": check_iri(r["compound"]), "name": r["name"]}


def _reaction_pairs_same_ec(rows: list[PoolRow], knobs: Knobs) -> Iterable[dict[str, Any]]:
    """Pairs of reactions that share an EC number and differ in citation count, more-cited first.

    Within an EC number, reactions in IRI order are paired with the next one,
    so the pool is deterministic and each pair is related.
    """
    by_ec: "OrderedDict[str, list[PoolRow]]" = OrderedDict()
    for r in rows:
        by_ec.setdefault(r["ec"], []).append(r)
    seen: set[str] = set()
    for ec, members in by_ec.items():
        members = [m for m in members if _label_ok(m["label"], MAX_REACTION_LABEL_CHARS)]
        members.sort(key=lambda m: m["reaction"])
        for x, y in zip(members, members[1:]):
            nx, ny = int(x["n"]), int(y["n"])
            if nx == ny:
                continue
            a, b = (x, y) if nx > ny else (y, x)
            key = "|".join(sorted((a["reaction"], b["reaction"])))
            if key in seen:
                continue
            seen.add(key)
            yield {
                "key": key,
                "entity_a": check_iri(a["reaction"]),
                "label_a": a["label"],
                "entity_b": check_iri(b["reaction"]),
                "label_b": b["label"],
            }


COMPOUND_REACTION_EC_NUMBERS = _comp(
    name="compound_reaction_ec_numbers",
    description=(
        "Each master reaction a compound takes part in, with each of its EC numbers (compound -> participant -> "
        "side -> reaction -> EC, 4 hops); reactions without an EC number are left out, and some exist."
    ),
    candidate_query=f"""
SELECT ?compound ?name (COUNT(DISTINCT ?reaction) AS ?n) (COUNT(DISTINCT ?withEc) AS ?k) WHERE {{
  GRAPH $RHEA {{
    ?compound rh:name ?name .
    {_TAKES_PART}
    OPTIONAL {{ ?reaction rh:ec ?anyEc . BIND(?reaction AS ?withEc) }}
  }}
}}
GROUP BY ?compound ?name
HAVING (COUNT(DISTINCT ?withEc) >= 1 && COUNT(DISTINCT ?withEc) < COUNT(DISTINCT ?reaction) && COUNT(DISTINCT ?reaction) <= 60)
ORDER BY ?compound
""",
    gold_pattern=f"""SELECT DISTINCT ?reaction ?ec WHERE {{
  GRAPH $RHEA {{
    ?participant rh:compound <$entity> .
    ?side rh:contains ?participant .
    ?reaction rh:side ?side ;
              rdfs:subClassOf rh:Reaction ;
              rh:ec ?ec .
  }}
}}
ORDER BY ?reaction ?ec""",
    question_pattern=(
        'For the compound entry named "$name", list each reaction it takes part in (on either side) with each of its EC '
        "numbers, leaving out reactions that have no EC number. Give one row per reaction and EC number: the "
        "reaction by the IRI of its undirected form, then the EC number by its IRI."
    ),
    answer_columns=(AnswerColumn("reaction", IRI_COL), AnswerColumn("ec", IRI_COL)),
    expand=_compounds_with_ec_reactions,
    side_clauses={"?side": rf"\bFor the compound entry named «name»{_N}, list each reaction it takes part in \(on either side\) with each of its EC numbers\b"},
    anchors=(Anchor("compound", label_param="name"),),
    hops=4,
    forms=("path",),
    features={"hops": 4},
    population_pattern=f"""SELECT (COUNT(DISTINCT ?reaction) AS ?n) WHERE {{
  GRAPH $RHEA {{
    ?participant rh:compound <$entity> .
    ?side rh:contains ?participant .
    ?reaction rh:side ?side ;
              rdfs:subClassOf rh:Reaction .
  }}
}}""",
)

CITATION_COUNT_DIFFERENCE = _comp(
    name="citation_count_difference",
    description=(
        "How many more publications one master reaction cites than another; the two share an EC number and "
        "their counts differ (comparison of two entities, one numeric answer)."
    ),
    candidate_query="""
SELECT ?ec ?reaction ?label (COUNT(DISTINCT ?citation) AS ?n) WHERE {
  GRAPH $RHEA {
    ?reaction rdfs:subClassOf rh:Reaction ;
              rh:ec ?ec ;
              rdfs:label ?label ;
              rh:citation ?citation .
  }
}
GROUP BY ?ec ?reaction ?label
ORDER BY ?ec ?reaction
""",
    gold_pattern="""SELECT ?difference WHERE {
  { SELECT (COUNT(DISTINCT ?citationA) AS ?a) WHERE { GRAPH $RHEA { <$entity_a> rh:citation ?citationA . } } }
  { SELECT (COUNT(DISTINCT ?citationB) AS ?b) WHERE { GRAPH $RHEA { <$entity_b> rh:citation ?citationB . } } }
  BIND(?a - ?b AS ?difference)
}""",
    question_pattern=(
        'How many more publications are cited for the reaction "$label_a" than for the reaction "$label_b"? '
        "Give the difference as a number."
    ),
    answer_columns=(AnswerColumn("difference", NUM_COL),),
    key_param="key",
    expand=_reaction_pairs_same_ec,
    anchors=(Anchor("reaction", "entity_a", "label_a"), Anchor("reaction", "entity_b", "label_b")),
    hops=1,
    forms=("comparison",),
    single_row=True,
)

ONE_HOP_ATTRIBUTE = (COMPOUND_REACTION_EC_NUMBERS, CITATION_COUNT_DIFFERENCE)


# --------------------------------------------------------------------------
# multi_hop_path


def _ec_numbers(rows: list[PoolRow], knobs: Knobs) -> Iterable[dict[str, Any]]:
    for r in rows:
        iri = check_iri(r["ec"])
        if not iri.startswith(EC_IRI_PREFIX):
            raise TemplateError(f"not an EC number IRI: {iri}")
        number = iri[len(EC_IRI_PREFIX):]
        if knobs.min_rows <= int(r["n"]) <= knobs.max_rows and all(p.isdigit() or p.startswith("n") for p in number.split(".")):
            yield {"entity": iri, "ec_number": number}


def _compound_sets(size: int):
    names = "abc"[:size]

    def expand(rows: list[PoolRow], knobs: Knobs) -> Iterable[dict[str, Any]]:
        for r in rows:
            if not knobs.min_rows <= int(r["n"]) <= knobs.max_rows:
                continue
            if not all(_label_ok(r[f"name{x.upper()}"]) for x in names):
                continue
            params: dict[str, Any] = {"key": "|".join(r[x] for x in names)}
            for x in names:
                params[f"entity_{x}"] = check_iri(r[x])
                params[f"name_{x}"] = r[f"name{x.upper()}"]
            yield params

    return expand


EC_COMPOUNDS = _comp(
    name="ec_compounds",
    description=(
        "Compounds of the master reactions that carry an EC number (EC <- reaction -> side -> participant -> "
        "compound, 4 hops); at least two reactions carry the EC number."
    ),
    candidate_query="""
SELECT ?ec (COUNT(DISTINCT ?reaction) AS ?r) (COUNT(DISTINCT ?compound) AS ?n) WHERE {
  GRAPH $RHEA {
    ?reaction rh:ec ?ec ;
              rdfs:subClassOf rh:Reaction ;
              rh:side ?side .
    ?side rh:contains ?participant .
    ?participant rh:compound ?compound .
  }
}
GROUP BY ?ec
HAVING (COUNT(DISTINCT ?reaction) >= 2)
ORDER BY ?ec
""",
    gold_pattern="""SELECT DISTINCT ?compound WHERE {
  GRAPH $RHEA {
    ?reaction rh:ec <$entity> ;
              rdfs:subClassOf rh:Reaction ;
              rh:side ?side .
    ?side rh:contains ?participant .
    ?participant rh:compound ?compound .
  }
}
ORDER BY ?compound""",
    question_pattern=(
        "Which compounds take part, on either side, in the reactions that have the EC number $ec_number "
        f"({UNDIRECTED})? {COMPOUNDS_AS}"
    ),
    answer_columns=(AnswerColumn("compound", IRI_COL),),
    expand=_ec_numbers,
    side_clauses={"?side": r"\bWhich compounds take part, on either side, in the reactions that have the EC number\b"},
    anchors=(Anchor("ec_number", label_param=None, id_param="ec_number"),),
    hops=4,
    forms=("path",),
    features={"hops": 4},
)

OPPOSITE_SIDE_COMPOUNDS = _comp(
    name="opposite_side_compounds",
    description=(
        "Compounds on the other side of a master reaction from a compound, other than the compound itself "
        "(compound -> participant -> side -> reaction -> other side -> participant -> compound, 6 hops)."
    ),
    candidate_query=f"""
SELECT ?compound ?name (COUNT(DISTINCT ?reaction) AS ?n) WHERE {{
  GRAPH $RHEA {{
    ?compound rh:name ?name .
    {_TAKES_PART}
  }}
}}
GROUP BY ?compound ?name
HAVING (COUNT(DISTINCT ?reaction) >= 2 && COUNT(DISTINCT ?reaction) <= 6)
ORDER BY ?compound
""",
    gold_pattern="""SELECT DISTINCT ?other WHERE {
  GRAPH $RHEA {
    ?participant rh:compound <$entity> .
    ?side rh:contains ?participant .
    ?reaction rh:side ?side , ?otherSide ;
              rdfs:subClassOf rh:Reaction .
    ?otherSide rh:contains ?otherParticipant .
    ?otherParticipant rh:compound ?other .
  }
  FILTER(?side != ?otherSide && ?other != <$entity>)
}
ORDER BY ?other""",
    question_pattern=(
        'In the reactions that the compound entry named "$name" takes part in, on either side '
        f'({UNDIRECTED}), which compounds appear on the other side of the reaction from it? Do not include "$name" '
        f"itself. {COMPOUNDS_AS}"
    ),
    answer_columns=(AnswerColumn("compound", IRI_COL),),
    expand=_compounds_named,
    side_clauses={
        "?side": rf"\bIn the reactions that the compound entry named «name»{_N} takes part in, on either side\b",
        "?otherSide": r"\bwhich compounds appear on the other side of the reaction from it\?",
    },
    anchors=(Anchor("compound", label_param="name"),),
    hops=6,
    forms=("path",),
    features={"hops": 6},
)


def _shared_reactions(size: int) -> Variant:
    names = "abc"[:size]
    side_patterns = "\n".join(
        f"    ?side{x.upper()} rh:contains ?participant{x.upper()} .\n"
        f"    ?participant{x.upper()} rh:compound <$entity_{x}> ."
        for x in names
    )
    sides = " , ".join(f"?side{x.upper()}" for x in names)
    pool_sides = " , ".join(f"?s{x}" for x in names)
    pool_parts = "\n".join(f"    ?s{x} rh:contains ?p{x} .\n    ?p{x} rh:compound ?{x} .\n    ?{x} rh:name ?name{x.upper()} ." for x in names)
    order = " && ".join(f"STR(?{x}) < STR(?{y})" for x, y in zip(names, names[1:]))
    quoted = [f'"$name_{x}"' for x in names]
    listing = " and ".join(quoted) if size == 2 else ", ".join(quoted[:-1]) + " and " + quoted[-1]
    lead = "both of the compound entries named" if size == 2 else f"all {_NUMBER_WORDS[size]} of the compound entries named"
    tokens = [f"«name_{x}»{_N}" for x in names]
    listed = " and ".join(tokens) if size == 2 else ", ".join(tokens[:-1]) + " and " + tokens[-1]
    shared_clause = (
        rf"\b{'do both' if size == 2 else 'do all ' + _NUMBER_WORDS[size]} of the compound entries named {listed} "
        r"take part, each on either side, not necessarily the same side\?"
    )
    pop = "\n".join(
        f"  {{ SELECT (COUNT(DISTINCT ?r{x}) AS ?n{x}) WHERE {{ GRAPH $RHEA {{ ?p{x} rh:compound <$entity_{x}> . "
        f"?s{x} rh:contains ?p{x} . ?r{x} rh:side ?s{x} ; rdfs:subClassOf rh:Reaction . }} }} }}"
        for x in names
    )
    smallest = f"?n{names[0]}"
    for x in names[1:]:
        smallest = f"IF({smallest} < ?n{x}, {smallest}, ?n{x})"
    return _comp(
        name=f"shared_reactions_{size}",
        description=(
            f"Master reactions in which all of {size} compounds take part (intersection of {size} populations, "
            "each compound -> participant -> side -> reaction)."
        ),
        candidate_query=f"""
SELECT {" ".join(f"?{x} ?name{x.upper()}" for x in names)} (COUNT(DISTINCT ?reaction) AS ?n) WHERE {{
  GRAPH $RHEA {{
    ?reaction rdfs:subClassOf rh:Reaction ;
              rh:side {pool_sides} .
{pool_parts}
  }}
  FILTER({order})
}}
GROUP BY {" ".join(f"?{x} ?name{x.upper()}" for x in names)}
HAVING (COUNT(DISTINCT ?reaction) >= 2 && COUNT(DISTINCT ?reaction) <= 50)
ORDER BY {" ".join(f"?{x}" for x in names)}
""",
        gold_pattern=f"""SELECT DISTINCT ?reaction WHERE {{
  GRAPH $RHEA {{
    ?reaction rdfs:subClassOf rh:Reaction ;
              rh:side {sides} .
{side_patterns}
  }}
}}
ORDER BY ?reaction""",
        question_pattern=(
            f"In which reactions do {lead} {listing} take part, each on either side, not necessarily the same side? "
            f"{REACTIONS_ONCE}"
        ),
        answer_columns=(AnswerColumn("reaction", IRI_COL),),
        key_param="key",
        expand=_compound_sets(size),
        anchors=tuple(Anchor("compound", f"entity_{x}", f"name_{x}") for x in names),
        side_clauses={f"?side{x.upper()}": shared_clause for x in names},
        joint_sides=tuple(f"?side{x.upper()}" for x in names),
        joint_clause=r"\btake part, each on either side, not necessarily the same side\?",
        hops=3,
        forms=("intersection",),
        features={"populations": size},
        population_pattern=f"SELECT ?n WHERE {{\n{pop}\n  BIND({smallest} AS ?n)\n}}",
    )


SHARED_REACTIONS_2 = _shared_reactions(2)
SHARED_REACTIONS_3 = _shared_reactions(3)

MULTI_HOP_PATH = (EC_COMPOUNDS, OPPOSITE_SIDE_COMPOUNDS, SHARED_REACTIONS_2, SHARED_REACTIONS_3)


# --------------------------------------------------------------------------
# aggregation


def _reaction_size_thresholds(rows: list[PoolRow], knobs: Knobs) -> Iterable[dict[str, Any]]:
    """Per compound, every HAVING threshold on reaction size that keeps a proper, knob-sized subset."""
    by_compound: "OrderedDict[str, tuple[PoolRow, OrderedDict[int, int]]]" = OrderedDict()
    for r in rows:
        first, counts = by_compound.setdefault(r["compound"], (r, OrderedDict()))
        counts[int(r["size"])] = int(r["n"])
    for compound, (first, counts) in by_compound.items():
        total = sum(counts.values())
        if not _label_ok(first["name"]) or total < knobs.min_population or total > 60 or len(counts) < 2:
            continue
        for op, value, _ in _thresholds(counts, knobs, (">=", "<=")):
            yield {
                "entity": check_iri(compound),
                "name": first["name"],
                "op": op,
                "threshold": value,
                "comparison": _COUNT_WORDING[op].format(v=value),
            }


def _top_k_classes(rows: list[PoolRow], knobs: Knobs) -> Iterable[dict[str, Any]]:
    """Classes whose linked compounds have a strict gap between the k-th and (k+1)-th reaction count."""
    by_class: "OrderedDict[str, list[PoolRow]]" = OrderedDict()
    for r in rows:
        by_class.setdefault(r["class"], []).append(r)
    k = knobs.top_k
    for cls, members in by_class.items():
        if not _label_ok(members[0]["label"]) or len(members) < max(knobs.min_population, k + 1):
            continue
        counts = sorted((int(m["n"]) for m in members), reverse=True)
        if counts[k - 1] > counts[k]:
            yield {"entity": check_iri(cls), "label": members[0]["label"], "k": k, "k_words": _NUMBER_WORDS.get(k, str(k))}


def _compounds_in_range(low: int, high: int):
    def expand(rows: list[PoolRow], knobs: Knobs) -> Iterable[dict[str, Any]]:
        for r in rows:
            if _label_ok(r["name"]) and low <= int(r["n"]) <= high:
                yield {"entity": check_iri(r["compound"]), "name": r["name"]}

    return expand


_REACTIONS_PER_COMPOUND = f"""
SELECT ?compound ?name (COUNT(DISTINCT ?reaction) AS ?n) WHERE {{
  GRAPH $RHEA {{
    ?compound rh:name ?name .
    {_TAKES_PART}
  }}
}}
GROUP BY ?compound ?name
ORDER BY ?compound
"""

REACTIONS_BY_SIZE = _comp(
    name="reactions_by_size",
    description=(
        "Master reactions of a compound whose number of distinct compounds passes a threshold, each with that "
        "number (GROUP BY reaction, HAVING on COUNT(DISTINCT compound))."
    ),
    candidate_query=f"""
SELECT ?compound ?name ?size (COUNT(?reaction) AS ?n) WHERE {{
  {{
    SELECT ?compound ?reaction (COUNT(DISTINCT ?member) AS ?size) WHERE {{
      GRAPH $RHEA {{
        {_TAKES_PART}
        ?reaction rh:side ?memberSide .
        ?memberSide rh:contains ?memberParticipant .
        ?memberParticipant rh:compound ?member .
      }}
    }}
    GROUP BY ?compound ?reaction
  }}
  GRAPH $RHEA {{ ?compound rh:name ?name . }}
}}
GROUP BY ?compound ?name ?size
ORDER BY ?compound ?size
""",
    gold_pattern="""SELECT ?reaction (COUNT(DISTINCT ?compound) AS ?count) WHERE {
  GRAPH $RHEA {
    ?anchorParticipant rh:compound <$entity> .
    ?anchorSide rh:contains ?anchorParticipant .
    ?reaction rh:side ?anchorSide ;
              rdfs:subClassOf rh:Reaction ;
              rh:side ?side .
    ?side rh:contains ?participant .
    ?participant rh:compound ?compound .
  }
}
GROUP BY ?reaction
HAVING (COUNT(DISTINCT ?compound) $op $threshold)
ORDER BY ?reaction""",
    question_pattern=(
        'Among the reactions in which the compound entry named "$name" takes part, on either side, which involve $comparison '
        "different compounds in total, counting a compound once even if it appears on both sides? List each such "
        "reaction with its number of different compounds. Give each reaction once, whatever its direction, by the "
        "IRI of its undirected form."
    ),
    answer_columns=(AnswerColumn("reaction", IRI_COL), AnswerColumn("count", NUM_COL)),
    expand=_reaction_size_thresholds,
    side_clauses={
        "?anchorSide": rf"\bAmong the reactions in which the compound entry named «name»{_N} takes part, on either side, which involve\b",
        "?side": (
            r"\bwhich involve (?:fewer than|at most|more than|at least|exactly) -?\d+ different compounds in total, "
            r"counting a compound once even if it appears on both sides\?"
        ),
    },
    joint_sides=('?anchorSide', '?side'),
    anchors=(Anchor("compound", label_param="name"),),
    forms=("grouping", "distinct_count"),
    features={"grouping": "having"},
    population_pattern=f"""SELECT (COUNT(DISTINCT ?reaction) AS ?n) WHERE {{
  GRAPH $RHEA {{
    ?participant rh:compound <$entity> .
    ?side rh:contains ?participant .
    ?reaction rh:side ?side ;
              rdfs:subClassOf rh:Reaction .
  }}
}}""",
)

TOP_COMPOUNDS_IN_CLASS = _comp(
    name="top_compounds_in_class",
    description=(
        "The k compounds whose ChEBI class is a direct subclass of X that take part in the most master reactions, "
        "each with its count (GROUP BY, ORDER BY count then name, LIMIT k); only classes with a strict gap after "
        "the k-th count, so the tie-break never decides the answer set."
    ),
    candidate_query=f"""
SELECT ?class ?label ?compound (COUNT(DISTINCT ?reaction) AS ?n) WHERE {{
  GRAPH $CHEBI {{
    ?child rdfs:subClassOf ?class .
    ?class rdfs:label ?label .
  }}
  GRAPH $RHEA {{
    ?compound rh:chebi ?child .
    {_TAKES_PART}
  }}
}}
GROUP BY ?class ?label ?compound
ORDER BY ?class ?compound
""",
    gold_pattern=f"""SELECT ?compound ?count WHERE {{
  {{
    SELECT ?compound (COUNT(DISTINCT ?reaction) AS ?count) WHERE {{
      GRAPH $CHEBI {{ ?class rdfs:subClassOf <$entity> . }}
      GRAPH $RHEA {{
        ?compound rh:chebi ?class .
        {_TAKES_PART}
      }}
    }}
    GROUP BY ?compound
  }}
  GRAPH $RHEA {{ ?compound rh:name ?name . }}
}}
ORDER BY DESC(?count) ?name
LIMIT $k""",
    question_pattern=(
        "Of the compounds whose chemical class is a direct subclass (one level down, not the class itself) of the "
        'chemical class named "$label", which $k_words take part in the most reactions, on either side? List each '
        "with its number of reactions, counting each reaction once in its undirected form; if compounds are tied, "
        f"take them in alphabetical order of their names. Polymers do not count through the class of their repeating unit. {COMPOUNDS_AS}"
    ),
    answer_columns=(AnswerColumn("compound", IRI_COL), AnswerColumn("count", NUM_COL)),
    expand=_top_k_classes,
    side_clauses={"?side": r"\bwhich (?:one|two|three|four|five|six|\d+) take part in the most reactions, on either side\?"},
    anchors=(Anchor("chemical_class"),),
    membership_mode="direct",
    forms=("grouping", "distinct_count"),
    features={"grouping": "order_limit"},
    population_pattern=f"""SELECT (COUNT(DISTINCT ?compound) AS ?n) WHERE {{
  GRAPH $CHEBI {{ ?class rdfs:subClassOf <$entity> . }}
  GRAPH $RHEA {{
    ?compound rh:chebi ?class .
    {_TAKES_PART}
  }}
}}""",
    prefer=("direct_vs_transitive",),
)

PARTNER_COUNT = _comp(
    name="partner_count",
    description=(
        "Number of distinct compounds that share a master reaction with a compound, the compound itself not "
        "counted (COUNT DISTINCT across a 6-hop join)."
    ),
    candidate_query=_REACTIONS_PER_COMPOUND,
    gold_pattern="""SELECT (COUNT(DISTINCT ?other) AS ?count) WHERE {
  GRAPH $RHEA {
    ?anchorParticipant rh:compound <$entity> .
    ?anchorSide rh:contains ?anchorParticipant .
    ?reaction rh:side ?anchorSide ;
              rdfs:subClassOf rh:Reaction ;
              rh:side ?side .
    ?side rh:contains ?participant .
    ?participant rh:compound ?other .
  }
  FILTER(?other != <$entity>)
}""",
    question_pattern=(
        f"How many different compounds take part, on either side, in at least one reaction ({UNDIRECTED}) that "
        'the compound entry named "$name" also takes part in, on either side? Do not count "$name" itself. '
        "Give the number."
    ),
    answer_columns=(AnswerColumn("count", NUM_COL),),
    expand=_compounds_in_range(3, 40),
    side_clauses={
        "?anchorSide": rf"\bthat the compound entry named «name»{_N} also takes part in, on either side\?",
        "?side": r"\bHow many different compounds take part, on either side, in at least one reaction\b",
    },
    joint_sides=('?anchorSide', '?side'),
    anchors=(Anchor("compound", label_param="name"),),
    hops=6,
    forms=("distinct_count", "path"),
    features={"hops": 6},
    single_row=True,
)

AGGREGATION = (REACTIONS_BY_SIZE, TOP_COMPOUNDS_IN_CLASS, PARTNER_COUNT)


# --------------------------------------------------------------------------
# numeric_filter


def _descendant_charge_thresholds(rows: list[PoolRow], knobs: Knobs) -> Iterable[dict[str, Any]]:
    """Per class, thresholds on descendants' charges; those that keep a positive charge come first.

    ChEBI stores positive charges as strings ("+1"), so a comparison without a
    cast misses exactly those.
    """
    by_class: "OrderedDict[str, tuple[PoolRow, OrderedDict[int, int]]]" = OrderedDict()
    for r in rows:
        first, counts = by_class.setdefault(r["class"], (r, OrderedDict()))
        counts[int(r["charge"])] = int(r["n"])
    for cls, (first, counts) in by_class.items():
        if not _label_ok(first["label"]) or sum(counts.values()) < knobs.min_population or len(counts) < 2:
            continue
        for op, value, _ in _thresholds(counts, knobs, ("<", "<=", ">", ">=", "=")):
            keeps_positive = any(_COMPARE[op](x, value) for x in counts if x > 0)
            yield {
                "entity": check_iri(cls),
                "label": first["label"],
                "op": op,
                "threshold": value,
                "comparison": _WORDING[op].format(v=f"+{value}" if value > 0 else value),
                **({"_priority": 1} if keeps_positive else {}),
            }


PARTNERS_WITH_LOWER_CHARGE = _comp(
    name="partners_with_lower_charge",
    description=(
        "Compounds that share a master reaction with a small molecule and have a lower integer charge than it "
        "(comparison with the named entity's own value, across a 6-hop join, charges cast)."
    ),
    candidate_query=f"""
SELECT ?compound ?name (COUNT(DISTINCT ?reaction) AS ?n) WHERE {{
  GRAPH $RHEA {{
    ?compound rdfs:subClassOf rh:SmallMolecule ;
              rh:name ?name ;
              rh:charge ?anyCharge .
    {_TAKES_PART}
  }}
}}
GROUP BY ?compound ?name
ORDER BY ?compound
""",
    gold_pattern="""SELECT DISTINCT ?compound ?charge WHERE {
  GRAPH $RHEA {
    <$entity> rh:charge ?anchorCharge .
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
  FILTER(?charge < xsd:integer(?anchorCharge))
}
ORDER BY ?compound""",
    question_pattern=(
        'Take the reactions that the compound entry named "$name" takes part in, on either side '
        f"({UNDIRECTED}). Which compounds take part in them, on either side, and have a lower net charge than "
        '"$name"? Consider only compounds whose charge is a whole number. List each such compound with its '
        f"charge. {COMPOUNDS_AS}"
    ),
    answer_columns=(AnswerColumn("compound", IRI_COL), AnswerColumn("charge", NUM_COL)),
    expand=_compounds_in_range(1, 6),
    side_clauses={
        "?anchorSide": rf"\bTake the reactions that the compound entry named «name»{_N} takes part in, on either side\b",
        "?side": r"\bWhich compounds take part in them, on either side, and have a lower net charge\b",
    },
    joint_sides=('?anchorSide', '?side'),
    anchors=(Anchor("compound", label_param="name"),),
    hops=6,
    membership_mode="n.a.",
    forms=("comparison", "path"),
    features={"hops": 6},
    population_pattern="""SELECT (COUNT(DISTINCT ?compound) AS ?n) WHERE {
  GRAPH $RHEA {
    ?anchorParticipant rh:compound <$entity> .
    ?anchorSide rh:contains ?anchorParticipant .
    ?reaction rh:side ?anchorSide ;
              rdfs:subClassOf rh:Reaction ;
              rh:side ?side .
    ?side rh:contains ?participant .
    ?participant rh:compound ?compound .
    ?compound rh:charge ?c .
    FILTER(DATATYPE(?c) = xsd:int)
  }
  FILTER(?compound != <$entity>)
}""",
)

DESCENDANTS_BY_CHARGE = _comp(
    name="descendants_by_charge",
    description=(
        "ChEBI classes below X at any depth whose ChEBI charge passes a threshold, each with its charge "
        "(transitive population, numeric filter with the xsd:integer cast)."
    ),
    candidate_query="""
SELECT ?class ?label ?charge (COUNT(DISTINCT ?d) AS ?n) WHERE {
  {
    SELECT ?class WHERE { GRAPH $CHEBI { ?member rdfs:subClassOf+ ?class . } FILTER(isIRI(?class)) }
    GROUP BY ?class
    HAVING (COUNT(DISTINCT ?member) >= 5 && COUNT(DISTINCT ?member) <= 200)
  }
  GRAPH $CHEBI {
    ?d rdfs:subClassOf+ ?class ;
       chemrof:charge ?c .
    ?class rdfs:label ?label .
  }
  BIND(xsd:integer(?c) AS ?charge)
}
GROUP BY ?class ?label ?charge
ORDER BY ?class ?charge
""",
    gold_pattern="""SELECT DISTINCT ?class ?charge WHERE {
  GRAPH $CHEBI {
    ?class rdfs:subClassOf+ <$entity> ;
           chemrof:charge ?c .
  }
  BIND(xsd:integer(?c) AS ?charge)
  FILTER(?charge $op $threshold)
}
ORDER BY ?class""",
    question_pattern=(
        'Which chemical classes lie below the chemical class named "$label" at any depth of the class hierarchy (not the class itself) and '
        f"have a charge $comparison? List each such class with its charge. {CLASSES_AS}"
    ),
    answer_columns=(AnswerColumn("class", IRI_COL), AnswerColumn("charge", NUM_COL)),
    expand=_descendant_charge_thresholds,
    anchors=(Anchor("chemical_class"),),
    membership_mode="transitive",
    forms=("path",),
    population_pattern="""SELECT (COUNT(DISTINCT ?class) AS ?n) WHERE {
  GRAPH $CHEBI {
    ?class rdfs:subClassOf+ <$entity> ;
           chemrof:charge ?c .
  }
}""",
    prefer=("direct_vs_transitive",),
)

NUMERIC_FILTER = (PARTNERS_WITH_LOWER_CHARGE, DESCENDANTS_BY_CHARGE)


# --------------------------------------------------------------------------
# absence_negation


def _compound_without_partner(rows: list[PoolRow], knobs: Knobs) -> Iterable[dict[str, Any]]:
    """(compound, common co-compound) options; the compound is the key, so each is drawn once."""
    for r in rows:
        without = int(r["n"]) - int(r["with"])
        if not (_label_ok(r["nameA"]) and _label_ok(r["nameB"])):
            continue
        if int(r["n"]) < knobs.min_population or not knobs.min_rows <= without <= knobs.max_rows:
            continue
        yield {
            "entity_a": check_iri(r["a"]),
            "name_a": r["nameA"],
            "entity_b": check_iri(r["b"]),
            "name_b": r["nameB"],
        }


def _compounds_some_without_ec(rows: list[PoolRow], knobs: Knobs) -> Iterable[dict[str, Any]]:
    for r in rows:
        without = int(r["n"]) - int(r["k"])
        if _label_ok(r["name"]) and int(r["n"]) >= knobs.min_population and knobs.min_rows <= without <= knobs.max_rows:
            yield {"entity": check_iri(r["compound"]), "name": r["name"]}


def _classes_with_unused_children(rows: list[PoolRow], knobs: Knobs) -> Iterable[dict[str, Any]]:
    for r in rows:
        unused = int(r["n"]) - int(r["used"])
        if _label_ok(r["label"]) and int(r["n"]) >= knobs.min_population and knobs.min_rows <= unused <= knobs.max_rows:
            yield {"entity": check_iri(r["class"]), "label": r["label"]}


_COMPOUND_REACTIONS_POPULATION = """SELECT (COUNT(DISTINCT ?reaction) AS ?n) WHERE {
  GRAPH $RHEA {
    ?participant rh:compound <$ENTITY> .
    ?side rh:contains ?participant .
    ?reaction rh:side ?side ;
              rdfs:subClassOf rh:Reaction .
  }
}"""

REACTIONS_WITHOUT_EC = _comp(
    name="reactions_without_ec",
    description="Master reactions of a compound that have no EC number (negation of one hop).",
    candidate_query=COMPOUND_REACTION_EC_NUMBERS.candidate_query,
    gold_pattern=f"""SELECT DISTINCT ?reaction WHERE {{
  GRAPH $RHEA {{
    ?participant rh:compound <$entity> .
    ?side rh:contains ?participant .
    ?reaction rh:side ?side ;
              rdfs:subClassOf rh:Reaction .
    FILTER NOT EXISTS {{ ?reaction rh:ec ?ec . }}
  }}
}}
ORDER BY ?reaction""",
    question_pattern=(
        f'Which reactions that the compound entry named "$name" takes part in, on either side, have no EC number? {REACTIONS_ONCE}'
    ),
    answer_columns=(AnswerColumn("reaction", IRI_COL),),
    expand=_compounds_some_without_ec,
    side_clauses={"?side": rf"\bWhich reactions that the compound entry named «name»{_N} takes part in, on either side, have no EC number\?"},
    anchors=(Anchor("compound", label_param="name"),),
    membership_mode="n.a.",
    forms=("negation",),
    features={"negation_depth": 1},
    population_pattern=_COMPOUND_REACTIONS_POPULATION.replace("$ENTITY", "$entity"),
)

REACTIONS_WITHOUT_PARTNER = _comp(
    name="reactions_without_partner",
    description=(
        "Master reactions of compound A in which compound B (one of the 30 compounds in most reactions) does not "
        "take part (FILTER NOT EXISTS over side -> participant -> compound, depth 3)."
    ),
    candidate_query="""
SELECT ?a ?nameA ?b ?nameB ?n ?with WHERE {
  {
    SELECT ?b WHERE {
      GRAPH $RHEA {
        ?pb rh:compound ?b .
        ?sb rh:contains ?pb .
        ?rb rh:side ?sb ;
            rdfs:subClassOf rh:Reaction .
      }
    }
    GROUP BY ?b
    ORDER BY DESC(COUNT(DISTINCT ?rb)) ?b
    LIMIT 30
  }
  {
    SELECT ?a ?b (COUNT(DISTINCT ?r) AS ?with) (SAMPLE(?total) AS ?n) WHERE {
      {
        SELECT ?a (COUNT(DISTINCT ?ra) AS ?total) WHERE {
          GRAPH $RHEA {
            ?pa rh:compound ?a .
            ?sa rh:contains ?pa .
            ?ra rh:side ?sa ;
                rdfs:subClassOf rh:Reaction .
          }
        }
        GROUP BY ?a
        HAVING (COUNT(DISTINCT ?ra) >= 3 && COUNT(DISTINCT ?ra) <= 60)
      }
      GRAPH $RHEA {
        ?p1 rh:compound ?a .
        ?s1 rh:contains ?p1 .
        ?r rh:side ?s1 , ?s2 ;
           rdfs:subClassOf rh:Reaction .
        ?s2 rh:contains ?p2 .
        ?p2 rh:compound ?b .
      }
    }
    GROUP BY ?a ?b
  }
  FILTER(?a != ?b && ?with < ?n)
  GRAPH $RHEA { ?a rh:name ?nameA . ?b rh:name ?nameB . }
}
ORDER BY ?a ?b
""",
    gold_pattern="""SELECT DISTINCT ?reaction WHERE {
  GRAPH $RHEA {
    ?participant rh:compound <$entity_a> .
    ?side rh:contains ?participant .
    ?reaction rh:side ?side ;
              rdfs:subClassOf rh:Reaction .
    FILTER NOT EXISTS {
      ?reaction rh:side ?otherSide .
      ?otherSide rh:contains ?otherParticipant .
      ?otherParticipant rh:compound <$entity_b> .
    }
  }
}
ORDER BY ?reaction""",
    question_pattern=(
        'Which reactions that the compound entry named "$name_a" takes part in, on either side, do not also involve '
        f'the compound entry named "$name_b" on either side? {REACTIONS_ONCE}'
    ),
    answer_columns=(AnswerColumn("reaction", IRI_COL),),
    key_param="entity_a",
    expand=_compound_without_partner,
    side_clauses={
        "?side": rf"\bWhich reactions that the compound entry named «name_a»{_N} takes part in, on either side, do not also involve\b",
        "?otherSide": rf"\bdo not also involve the compound entry named «name_b»{_N} on either side\?",
    },
    joint_sides=('?side', '?otherSide'),
    anchors=(Anchor("compound", "entity_a", "name_a"), Anchor("compound", "entity_b", "name_b")),
    membership_mode="n.a.",
    forms=("negation",),
    features={"negation_depth": 3},
    population_pattern=_COMPOUND_REACTIONS_POPULATION.replace("$ENTITY", "$entity_a"),
)

UNUSED_CHILD_CLASSES = _comp(
    name="unused_child_classes",
    description=(
        "Direct ChEBI subclasses of X that no compound taking part in a master reaction is linked to "
        "(FILTER NOT EXISTS over a cross-graph join of depth 4)."
    ),
    candidate_query="""
SELECT ?class ?label (COUNT(DISTINCT ?child) AS ?n) (COUNT(DISTINCT ?usedChild) AS ?used) WHERE {
  GRAPH $CHEBI {
    ?child rdfs:subClassOf ?class .
    ?class rdfs:label ?label .
  }
  FILTER(isIRI(?child))
  OPTIONAL {
    SELECT DISTINCT ?child (?child AS ?usedChild) WHERE {
      GRAPH $RHEA { ?compound rh:chebi ?child . ?participant rh:compound ?compound . }
    }
  }
}
GROUP BY ?class ?label
HAVING (COUNT(DISTINCT ?usedChild) >= 1)
ORDER BY ?class
""",
    gold_pattern=f"""SELECT DISTINCT ?class WHERE {{
  GRAPH $CHEBI {{ ?class rdfs:subClassOf <$entity> . }}
  FILTER(isIRI(?class))
  FILTER NOT EXISTS {{
    GRAPH $RHEA {{
      ?compound rh:chebi ?class .
      {_TAKES_PART}
    }}
  }}
}}
ORDER BY ?class""",
    question_pattern=(
        'Which direct subclasses (one level down, not the class itself) of the chemical class named "$label" are not '
        "linked to any compound "
        f"that takes part in a reaction ({UNDIRECTED}), on either side? Polymers do not count as linked "
        "through the class of their repeating unit, and generic compounds do not count as linked through their "
        f"reactive parts. {CLASSES_AS}"
    ),
    answer_columns=(AnswerColumn("class", IRI_COL),),
    expand=_classes_with_unused_children,
    side_clauses={"?side": r"\bare not linked to any compound that takes part in a reaction \(taking each reaction in its undirected form\), on either side\?"},
    anchors=(Anchor("chemical_class"),),
    membership_mode="direct",
    forms=("negation",),
    features={"negation_depth": 4},
    engine_probes=(
        EngineProbe(
            reason="qlever_nested_graph",
            description=(
                "The natural query that writes the negated reaction-graph pattern inside the chemical-class GRAPH "
                "block. QLever 9bebc60 evaluates a GRAPH clause nested in a plain group against the outer graph; "
                "a question is tagged when the nested negation also goes wrong (then every subclass comes back)."
            ),
            pattern=f"""SELECT DISTINCT ?class WHERE {{
  GRAPH $CHEBI {{
    ?class rdfs:subClassOf <$entity> .
    FILTER(isIRI(?class))
    FILTER NOT EXISTS {{
      GRAPH $RHEA {{
        ?compound rh:chebi ?class .
        {_TAKES_PART}
      }}
    }}
  }}
}}""",
        ),
    ),
    population_pattern="""SELECT (COUNT(DISTINCT ?class) AS ?n) WHERE {
  GRAPH $CHEBI { ?class rdfs:subClassOf <$entity> . }
  FILTER(isIRI(?class))
}""",
    prefer=("direct_vs_transitive", "graph_scope"),
)

ABSENCE_NEGATION = (REACTIONS_WITHOUT_EC, REACTIONS_WITHOUT_PARTNER, UNUSED_CHILD_CLASSES)


# --------------------------------------------------------------------------
# cross_graph_join


def _classes_named(rows: list[PoolRow], knobs: Knobs) -> Iterable[dict[str, Any]]:
    for r in rows:
        if _label_ok(r["label"]):
            yield {"entity": check_iri(r["class"]), "label": r["label"]}


def _classes_with_reaction_counts(low: int, high: int | None):
    def expand(rows: list[PoolRow], knobs: Knobs) -> Iterable[dict[str, Any]]:
        top = knobs.max_rows if high is None else high
        for r in rows:
            if _label_ok(r["label"]) and max(low, knobs.min_rows) <= int(r["n"]) <= top:
                yield {"entity": check_iri(r["class"]), "label": r["label"]}

    return expand


MICROSPECIES_REACTIONS = _comp(
    name="microspecies_reactions",
    description=(
        "Master reactions of the major microspecies at pH 7.3 of a ChEBI class (ChEBI class -> microspecies -> "
        "compound -> participant -> side -> reaction, 5 hops); the microspecies is another class."
    ),
    candidate_query=f"""
SELECT ?class ?label (COUNT(DISTINCT ?reaction) AS ?n) WHERE {{
  GRAPH $CHEBI {{
    ?class chemrof:has_major_microspecies_at_pH7_3 ?species ;
           rdfs:label ?label .
  }}
  FILTER(?class != ?species)
  GRAPH $RHEA {{
    ?compound rh:chebi ?species .
    {_TAKES_PART}
  }}
}}
GROUP BY ?class ?label
ORDER BY ?class
""",
    gold_pattern=f"""SELECT DISTINCT ?reaction WHERE {{
  GRAPH $CHEBI {{ <$entity> chemrof:has_major_microspecies_at_pH7_3 ?species . }}
  GRAPH $RHEA {{
    ?compound rh:chebi ?species .
    {_TAKES_PART}
  }}
}}
ORDER BY ?reaction""",
    question_pattern=(
        'In which reactions does the major microspecies at pH 7.3 of the chemical class named "$label" take part, on '
        f"either side? Polymers do not count through the class of their repeating unit. {REACTIONS_ONCE}"
    ),
    answer_columns=(AnswerColumn("reaction", IRI_COL),),
    expand=_classes_with_reaction_counts(1, None),
    side_clauses={"?side": rf"\bof the chemical class named «label»{_N} take part, on either side\?"},
    anchors=(Anchor("chemical_class"),),
    membership_mode="n.a.",
    hops=5,
    forms=("path",),
    features={"hops": 5},
)

REACTION_COUNT_BELOW_CLASS = _comp(
    name="reaction_count_below_class",
    description=(
        "Number of distinct master reactions with a compound whose ChEBI class is X or below X at any depth "
        "(COUNT DISTINCT across a cross-graph join)."
    ),
    candidate_query="""
SELECT ?class ?label (COUNT(DISTINCT ?linked) AS ?n) WHERE {
  GRAPH $RHEA {
    ?compound rh:chebi ?linked .
    ?participant rh:compound ?compound .
  }
  GRAPH $CHEBI {
    ?linked rdfs:subClassOf+ ?class .
    ?class rdfs:label ?label .
  }
  FILTER(isIRI(?class))
}
GROUP BY ?class ?label
HAVING (COUNT(DISTINCT ?linked) >= 3 && COUNT(DISTINCT ?linked) <= 200)
ORDER BY ?class
""",
    gold_pattern=f"""SELECT (COUNT(DISTINCT ?reaction) AS ?count) WHERE {{
  {{ GRAPH $CHEBI {{ ?class rdfs:subClassOf+ <$entity> . }} }} UNION {{ VALUES ?class {{ <$entity> }} }}
  GRAPH $RHEA {{
    ?compound rh:chebi ?class .
    {_TAKES_PART}
  }}
}}""",
    question_pattern=(
        'How many different reactions involve, on either side, a compound of the chemical class named "$label" or of '
        "one of its subclasses at any depth? Count each reaction once, in its undirected form. Polymers do not count through the class of their repeating unit. "
        "Give the number."
    ),
    answer_columns=(AnswerColumn("count", NUM_COL),),
    expand=_classes_named,
    side_clauses={"?side": r"\bHow many different reactions involve, on either side, a compound of the chemical class named «label»"},
    anchors=(Anchor("chemical_class"),),
    membership_mode="transitive",
    forms=("distinct_count",),
    single_row=True,
    prefer=("direct_vs_transitive",),
)

CROSS_GRAPH_JOIN = (MICROSPECIES_REACTIONS, REACTION_COUNT_BELOW_CLASS)
