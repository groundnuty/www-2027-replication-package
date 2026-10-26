"""Alternative-reading guard: a question must say which scope it means.

Many questions have a second defensible reading: an organism with or without
its strains and sub-taxa, a keyword or location with or without the narrower
terms below it, "directly below" against "at any depth", is-a against
part-of. The guard computes the gold query under each alternative reading,
runs it on the store, and if the answer changes, requires the question text
to carry the clause that fixes the scope. Otherwise generation fails.

Interface (shared by every graph's templates):

- A ``Reading`` has a ``name`` (counted in the manifest), a ``clause`` (a
  regular expression the question text must match when the reading changes
  the answer) and a way to build the alternative query: ``transform`` (a
  function of the rendered gold SPARQL and the question's parameters) or
  ``pattern`` (a gold-like ``string.Template`` rendered with the variant's
  prefixes, graphs and the parameters). Either returns None when the reading
  does not apply to this question.
- A variant lists its readings in ``Variant.readings``.
- ``check_readings(client, variant, params, gold_sparql, gold_rows, text)``
  returns ``{name: ReadingResult}``; ``enforce(qid, results)`` raises
  ``ReadingError`` for any reading that changes the answer while the text
  lacks its clause.
- A reading with a ``note_param`` puts its clause in the text only where it
  changes the answer: generation sets that parameter on such a question, and
  the variant's ``notes`` give the sentence the text then ends with.

The helpers below build transforms on gold SPARQL text: scoping an anchor
IRI to its closure, widening an answer variable to the terms above it, and
restricting a subtree pattern to the anchor itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from harness.sparql import SparqlClient, SparqlError
from harness.terms import row_to_json


class ReadingError(ValueError):
    """An alternative reading changes the answer and the question text does not rule it out."""


@dataclass(frozen=True)
class Reading:
    name: str
    description: str
    clause: str
    transform: Callable[[str, Mapping[str, Any]], str | None] | None = None
    pattern: str | None = None
    # The parameter generation sets when this reading changes the answer (the text then gets the variant's note).
    note_param: str | None = None

    def alternative(self, variant: Any, params: Mapping[str, Any], gold_sparql: str) -> str | None:
        if self.transform is not None:
            return self.transform(gold_sparql, params)
        if self.pattern is not None:
            from harness.qgen_rhea.base import _substitute

            return variant._prefixes() + _substitute(self.pattern, {**params, **variant._graphs()})
        return None

    def clause_present(self, text: str) -> bool:
        return re.search(self.clause, text, flags=re.IGNORECASE) is not None


@dataclass(frozen=True)
class ReadingResult:
    changed: bool | None  # None: the alternative query failed, so the change is unknown
    clause: bool

    def to_json(self) -> dict[str, Any]:
        return {"changed": self.changed, "clause": self.clause}


def _canonical(rows: Any) -> list:
    import json

    as_json = [row_to_json(r) for r in rows]
    return sorted(as_json, key=lambda r: json.dumps(r, sort_keys=True, ensure_ascii=False))


def check_readings(
    client: SparqlClient, variant: Any, params: Mapping[str, Any], gold_sparql: str, gold_rows: list, text: str
) -> dict[str, ReadingResult]:
    """Each applicable reading of the variant: does it change the answer, and does the text carry its clause?"""
    out: dict[str, ReadingResult] = {}
    for reading in getattr(variant, "readings", ()):
        changed = reading_changes(client, reading, variant, params, gold_sparql, gold_rows)
        if changed != "n.a.":
            out[reading.name] = ReadingResult(changed, reading.clause_present(text))
    return out


def reading_changes(
    client: SparqlClient, reading: Reading, variant: Any, params: Mapping[str, Any], gold_sparql: str, gold_rows: list
) -> bool | None | str:
    """Whether the reading changes the answer: True or False, None when its query fails, "n.a." when it does not apply."""
    alt = reading.alternative(variant, params, gold_sparql)
    if alt is None or alt == gold_sparql:
        return "n.a."
    try:
        return _canonical(client.query(alt).rows) != gold_rows
    except SparqlError:
        return None


def enforce(qid: str, results: Mapping[str, ReadingResult]) -> None:
    """Fail when a reading changes (or may change) the answer and the text lacks the clause."""
    missing = [name for name, r in results.items() if r.changed is not False and not r.clause]
    if missing:
        raise ReadingError(f"{qid}: alternative readings {missing} change the answer but the question does not rule them out")


# --------------------------------------------------------------------------
# Membership swaps (also used for the discrimination fields)


def membership_swapped(sparql: str, mode: str) -> str | None:
    """The gold with its membership relation swapped (direct <-> transitive); None for "n.a."

    The master-reaction type test (rdfs:subClassOf rh:Reaction) is not the
    relation a question asks about and is left alone.
    """
    if mode == "transitive":
        return sparql.replace("rdfs:subClassOf+", "rdfs:subClassOf").replace("rdfs:subClassOf*", "rdfs:subClassOf")
    if mode == "direct":
        return re.sub(r"rdfs:subClassOf(?![+*])(?=\s+(?!rh:Reaction\b))", "rdfs:subClassOf+", sparql)
    return None


def graph_unscoped(sparql: str) -> str | None:
    """The gold without its GRAPH clauses (so it reads the union graph); None if it has none."""
    out = re.sub(r"GRAPH\s+<[^>]*>\s*", "", sparql)
    return out if out != sparql else None


# --------------------------------------------------------------------------
# Transform helpers on gold SPARQL text

_STATEMENT_END = re.compile(r"\.(?=\s)")


def _matching_open(sparql: str, close: int) -> int:
    depth = 0
    for i in range(close - 1, -1, -1):
        if sparql[i] == "}":
            depth += 1
        elif sparql[i] == "{":
            if depth == 0:
                return i
            depth -= 1
    raise ValueError("unbalanced braces")


def _enclosing_group_end(sparql: str, position: int) -> int:
    depth = 0
    for i in range(position, len(sparql)):
        if sparql[i] == "{":
            depth += 1
        elif sparql[i] == "}":
            if depth == 0:
                return i
            depth -= 1
    raise ValueError("no enclosing group")


def _insert_after_statement(sparql: str, position: int, text: str) -> str:
    """Insert ``text`` (a GRAPH pattern) next to the triple statement that contains ``position``.

    When the statement sits directly in a ``GRAPH <g> { ... }`` group, the text
    goes right after that group, because QLever evaluates a GRAPH clause nested
    inside another GRAPH clause against the outer graph (an engine quirk; see
    the manifest notes). Golds therefore keep negation and union groups outside
    their GRAPH clauses. Otherwise the text goes after the statement's end.
    """
    try:
        close = _enclosing_group_end(sparql, position)
        opening = _matching_open(sparql, close)
    except ValueError:
        close = opening = -1
    if opening >= 0 and re.search(r"GRAPH\s+<[^>]*>\s*$", sparql[:opening]):
        return sparql[: close + 1] + " " + text + sparql[close + 1 :]
    m = _STATEMENT_END.search(sparql, position)
    if m is None:
        raise ValueError("no statement end after the pattern")
    return sparql[: m.end()] + " " + text + sparql[m.end() :]


def scope_iri_to_closure(predicate: str, iri_param: str, graph: str, var: str) -> Callable[[str, Mapping[str, Any]], str | None]:
    """Reading: ``predicate <anchor>`` also matches anything below the anchor (``rdfs:subClassOf*``).

    Every occurrence of ``predicate <anchor IRI>`` becomes ``predicate ?var``,
    and ``GRAPH graph { ?var rdfs:subClassOf* <anchor> . }`` is added after
    the statement, in the same group.
    """

    def transform(sparql: str, params: Mapping[str, Any]) -> str | None:
        iri = params.get(iri_param)
        if not iri:
            return None
        needle = re.compile(re.escape(predicate) + r"\s+<" + re.escape(iri) + ">")
        out, count = sparql, 0
        while True:
            m = needle.search(out)
            if m is None:
                break
            out = out[: m.start()] + f"{predicate} ?{var}" + out[m.end() :]
            out = _insert_after_statement(out, m.start(), f"GRAPH {graph} {{ ?{var} rdfs:subClassOf* <{iri}> . }}")
            count += 1
        return out if count else None

    return transform


def widen_answer_to_ancestors(predicate: str, answer_var: str, graph: str, var_prefix: str) -> Callable[[str, Mapping[str, Any]], str | None]:
    """Reading: an answer term also stands for every term above it (``rdfs:subClassOf*``).

    Every ``predicate ?answer`` becomes ``predicate ?<prefix><i>`` with
    ``GRAPH graph { ?<prefix><i> rdfs:subClassOf* ?answer . }`` after the statement.
    """

    def transform(sparql: str, params: Mapping[str, Any]) -> str | None:
        needle = re.compile(re.escape(predicate) + r"\s+\?" + re.escape(answer_var) + r"\b")
        out, i = sparql, 0
        while True:
            m = needle.search(out)
            if m is None:
                break
            i += 1
            var = f"{var_prefix}{i}"
            out = out[: m.start()] + f"{predicate} ?{var}" + out[m.end() :]
            out = _insert_after_statement(out, m.start(), f"GRAPH {graph} {{ ?{var} rdfs:subClassOf* ?{answer_var} . }}")
        return out if i else None

    return transform


def restrict_subtree_to_anchor(iri_param: str, graph: str) -> Callable[[str, Mapping[str, Any]], str | None]:
    """Reading: "the taxon X" means X itself, not the taxa below it.

    ``GRAPH graph { ?v rdfs:subClassOf* <X> . }`` becomes ``VALUES ?v { <X> }``.
    """

    def transform(sparql: str, params: Mapping[str, Any]) -> str | None:
        iri = params.get(iri_param)
        if not iri:
            return None
        pattern = re.compile(r"GRAPH\s+" + re.escape(graph) + r"\s*\{\s*\?(\w+)\s+rdfs:subClassOf\*\s+<" + re.escape(iri) + r">\s*\.\s*\}")
        out, count = pattern.subn(lambda m: f"VALUES ?{m.group(1)} {{ <{iri}> }}", sparql)
        return out if count else None

    return transform


def compose(*transforms: Callable[[str, Mapping[str, Any]], str | None]) -> Callable[[str, Mapping[str, Any]], str | None]:
    """Apply several transforms in turn; None if none of them applies."""

    def transform(sparql: str, params: Mapping[str, Any]) -> str | None:
        out, applied = sparql, False
        for t in transforms:
            nxt = t(out, params)
            if nxt is not None:
                out, applied = nxt, True
        return out if applied else None

    return transform


def membership_reading(mode: str, clause: str, alternative_pattern: str | None = None) -> Reading:
    """Reading: the other membership mode (direct against transitive) of a variant whose mode is ``mode``.

    ``alternative_pattern`` spells the other reading out when a swap of
    rdfs:subClassOf cannot (e.g. the immediate-parent link of a taxonomy).
    """
    if alternative_pattern is not None:
        return Reading(
            "direct_vs_transitive",
            "the other membership reading (immediate children against the whole subtree)",
            clause,
            pattern=alternative_pattern,
        )
    return Reading(
        "direct_vs_transitive",
        "the other membership reading (direct rdfs:subClassOf against the transitive path)",
        clause,
        transform=lambda sparql, params: membership_swapped(sparql, mode),
    )


# --------------------------------------------------------------------------
# Nested GRAPH clauses (QLever 9bebc60 quirk)


_IRI_AT = re.compile(r'<[^\s<>"{}|^`\\]*>')

# Lexical scan copied from harness/agent/engine_flags.py (feat/e0.1-skeleton, nested_graph): strings, IRIs
# and comments are removed, then GRAPH keywords (any case) and braces are tracked. Prefixed names
# (ex:graph) and variables (?graph) are single tokens and never count as the keyword.
_STRING = re.compile(
    r'"""(?:[^"\\]|\\.|"(?!""))*"""|\'\'\'(?:[^\'\\]|\\.|\'(?!\'\'))*\'\'\'|"(?:[^"\\\n]|\\.)*"|\'(?:[^\'\\\n]|\\.)*\'',
    re.DOTALL,
)
_IRI = re.compile(r'<[^<>"{}|^`\\\x00-\x20]*>')
_COMMENT = re.compile(r"#[^\n]*")
_TOKEN = re.compile(r"[A-Za-z_][\w.-]*:[\w.-]*|:[\w.-]*|[?$]\w+|[A-Za-z_]\w*|[{}]")


def _tokens(query: str) -> list[str]:
    text = _STRING.sub(" ", query)
    text = _IRI.sub(" <iri> ", text)
    text = _COMMENT.sub(" ", text)
    return _TOKEN.findall(text)


def nests_graph(sparql: str) -> bool:
    """Whether a GRAPH group opens inside another GRAPH group (directly, in a subquery or in a filter).

    QLever 9bebc60 evaluates such an inner GRAPH clause against the outer graph,
    not the one it names, so no gold or alternative query may nest them.
    """
    depth = 0
    graph_depths: list[int] = []  # brace depth at which each open GRAPH group started
    pending_graph = False  # saw GRAPH; its group opens at the next "{"
    for tok in _tokens(sparql):
        if tok == "{":
            depth += 1
            if pending_graph:
                if graph_depths:
                    return True
                graph_depths.append(depth)
                pending_graph = False
        elif tok == "}":
            if graph_depths and graph_depths[-1] == depth:
                graph_depths.pop()
            depth -= 1
        elif tok.upper() == "GRAPH":
            pending_graph = True
    return False


nested_graph = nests_graph  # the name harness/agent/engine_flags.py uses


# --------------------------------------------------------------------------
# Rhea readings: transforms of the rendered gold, attached to every Rhea variant
#
# - class_itself: the named class counted among its own subclasses, or left
#   out where the gold counts it ("X or its subclasses").
# - all_reaction_forms: directional and bidirectional reactions counted next
#   to the undirected (master) reaction they belong to.
# - one_side:<?side var>:<L|R>: one participation (a side variable of the gold,
#   for the named entity or for the answer or counted compounds, including one
#   inside a negation) limited to the left or to the right side; every side
#   variable is tried both ways.
# - opposite_side_only: compounds "together with" the named one limited to the
#   other side of the reaction.
# - per_participant: compounds counted per participant, so a compound on both
#   sides of a transport reaction counts twice.
# - non_distinct_count:<?var>: one COUNT(DISTINCT ?var) without DISTINCT.
#
# Clauses are matched on the question text with every quoted label replaced by
# a token for its parameter («name», «label_a», ...; neutral_text), so a label
# can never supply a clause, and each clause is the complete scope statement for
# its subject: a participation's clause is the phrase stating that participation
# (declared per variant in Variant.side_clauses), a count's clause names the
# counted entity, the class-itself clause refers to the named class.
# - anchor_included: the named entity not excluded from its own partners.
# - tie_break: the opposite order among tied values before a LIMIT.
# - polymer_class: a polymer counted under the class of its repeating unit
#   (rh:underlyingChebi) as if that were its class.
# - reactive_part_class: a participating generic compound counted under the
#   classes of its reactive parts.
# - joint_side:<same|different>: the named entities' participations all on the
#   same side, or not all on one side (Variant.joint_sides).

_TYPE_TEST = re.compile(r"rdfs:subClassOf rh:Reaction\b")
_SIDE = re.compile(r"\brh:side(?=\s)")
_SIDE_VARS = re.compile(r"\brh:side\s+(\?\w+(?:\s*,\s*\?\w+)*)")
_COUNT_DISTINCT = re.compile(r"COUNT\(DISTINCT (\?\w+)\)")
_SELF_UNION = re.compile(r"\{\s*(GRAPH <[^>]+> \{[^{}]*\})\s*\}\s*UNION\s*\{\s*VALUES \?\w+ \{ <[^>]+> \}\s*\}")
# The same "X itself or below" union in an unrendered gold pattern (graph and IRI still placeholders).
_SELF_UNION_PATTERN = re.compile(r"UNION\s*\{\s*VALUES\s+\?\w+\s*\{\s*<\$\w+>\s*\}\s*\}")
_TRANSITIVE_TO_IRI = re.compile(r"rdfs:subClassOf\+(?=\s+<)")
_DIRECT_TO_IRI = re.compile(r"rdfs:subClassOf(?![+*?])(?=\s+<)")
_ANCHOR_EXCLUSION_AND = re.compile(r"\s*&&\s*\?\w+ != <[^>]+>")
_ANCHOR_EXCLUSION = re.compile(r"\s*FILTER\(\?\w+ != <[^>]+>\)")
_TIE_ORDER = re.compile(r"ORDER BY DESC\((\?\w+)\) (\?\w+)")
_CHEBI_LINK = re.compile(r"\brh:chebi(?=\s+\?)")

REACTION_FORMS = "rh:Reaction, rh:DirectionalReaction, rh:BidirectionalReaction"

# What a question must say when a reading changes its answer (matched ignoring case, on neutral_text).
_NOTE = r"(?: \(meaning [^)]*\))?"
# Every clause is the exact template phrase that states the scope, with its quantifier and negation
# (External review rev-qgen-rhea3: an opposite instruction, "Count «name» itself", "either", "also involve",
# "on the left side only", must not pass). Patterns are matched ignoring case on neutral_text.
CLAUSES: dict[str, str] = {
    "transitive": r"(?<!not )\bat any depth\b",
    # The parenthesised statement that fixes the depth, not a bare "direct subclass" in a heading phrase.
    "direct": (
        r"\bdirect subclass(?:es)? \(one level down\b"
        r"|(?<!than )\bone level down from that class \(not the class itself, not deeper\)"
    ),
    # class_itself takes the polarity of the gold: excluded (descendants, direct subclasses) or included.
    "class_itself_excluded": r"\b(?:do not include|not) the (?:class|reaction) itself\b",
    "class_itself_included": (
        rf"«label\w*»{_NOTE}, either directly or through any of its subclasses\b"
        rf"|«label\w*»{_NOTE} or of one of its subclasses\b"
    ),
    # Only "undirected" names the scope the golds use (the undirected reaction); "whatever its direction"
    # does not rule out counting the directional forms too (External review of 355f04e, finding 2).
    "all_reaction_forms": (
        r"\b(?:of|in) its undirected form[).;,]"
        r"|\bCount each undirected \(master\) reaction once and do not count directional or bidirectional variants\b"
        r"|\bcount only undirected \(master\) reactions(?:, not (?:their )?directional or bidirectional variants\b|\.)"
    ),
    # Compounds counted once, not once per participant.
    "per_participant": (
        r"\bHow many (?:different|distinct) compounds\b"
        r"|\bdifferent compounds in total, counting a compound once even if it appears on both sides\b"
    ),
    # The golds leave the named entity out of its own partners.
    "anchor_included": rf"\bDo not (?:count|include) «name\w*»{_NOTE} itself\b",
    "tie_break": r"\bif compounds are tied, take them in alphabetical order of their names\b",
    "polymer_class": r"\bPolymers do not count (?:as linked )?through the class of their repeating unit\b",
    # A generic compound's reactive parts carry classes (rh:chebi) too; the golds do not count them.
    "reactive_part_class": r"\bGeneric compounds do not count (?:as linked )?through their reactive parts\b",
}
# other_kind_anchor: the note for the anchor's own kind, after its token.
HOMONYM_CLAUSES: dict[str, str] = {
    "compound": r"«\w+» \(meaning the compound entry, not the chemical class of the same name\)",
    "chemical_class": r"«\w+» \(meaning the chemical class, not the compound entry of the same name\)",
}
CLAUSES["other_kind_anchor"] = "|".join(HOMONYM_CLAUSES.values())


def homonym_clause(kinds: Any) -> str:
    """other_kind_anchor for anchors of these kinds: the own kind's note present, and no note naming the other kind.

    Every flagged anchor gets its note (checked by generate.check_label_anchors); this clause makes a note
    of the wrong polarity ("meaning the chemical class" for a compound entry) fail.
    """
    kinds = sorted(set(kinds))
    if len(kinds) != 1:
        return CLAUSES["other_kind_anchor"]
    own = kinds[0]
    other = next(k for k in HOMONYM_CLAUSES if k != own)
    return rf"(?s)^(?!.*{HOMONYM_CLAUSES[other]}).*{HOMONYM_CLAUSES[own]}"
# non_distinct_count: the clause names the counted entity, by the counted variable.
COUNTED_CLAUSES: dict[str, str] = {
    "reaction": (
        r"\bHow many different reactions\b|(?<!not )\bcounting each reaction once\b"
        r"|\bCount each undirected \(master\) reaction once\b"
    ),
    "compound": (
        r"\bHow many (?:different|distinct) compounds\b"
        r"|\bdifferent compounds in total, counting a compound once\b"
    ),
    # Round 1 asks "how many more"; round 2 counts distinct publications explicitly (design ruling in the design log).
    "citation": r"\bHow many more publications are cited\b|(?<!not )\bthe number of distinct publications cited\b",
    # EC numbers counted once each (calibration round 2, reaction_count_below_class).
    "ec": r"\bHow many different EC numbers\b",
}
RHEA_READING_NAMES = (
    "direct_vs_transitive",
    "class_itself",
    "all_reaction_forms",
    "one_side",
    "opposite_side_only",
    "per_participant",
    "non_distinct_count",
    "anchor_included",
    "tie_break",
    "polymer_class",
    "other_kind_anchor",
    "joint_side",
    "reactive_part_class",
)


def reading_base(name: str) -> str:
    """The reading kind of a reading name ("one_side:?sideA:L" -> "one_side")."""
    return name.split(":", 1)[0]


_QUOTED = re.compile(r'"[^"]*"')


def neutral_text(text: str, params: Mapping[str, Any], anchors: Any = ()) -> str:
    """The question text with each anchor's quoted label replaced by «its parameter» and any other quote by «quoted».

    Clauses are matched on this text, so a label cannot supply a clause, and a
    clause can refer to its anchor by token.
    """
    out = text
    named = [(a.label_param, params.get(a.label_param)) for a in anchors if getattr(a, "label_param", None)]
    for param, label in sorted(named, key=lambda x: -len(str(x[1] or ""))):
        if label:
            out = out.replace(f'"{label}"', f"«{param}»")
    return _QUOTED.sub("«quoted»", out)


def side_vars(sparql: str) -> list[str]:
    """The side variables a gold binds with rh:side, in order of first appearance."""
    out: list[str] = []
    for m in _SIDE_VARS.finditer(sparql):
        for var in re.split(r"\s*,\s*", m.group(1)):
            if var not in out:
                out.append(var)
    return out


def insert_in_group(sparql: str, pos: int, text: str) -> str:
    """``text`` inserted just before the closing brace of the group that contains ``pos``."""
    close = _enclosing_group_end(sparql, pos)
    return sparql[:close] + text + "\n  " + sparql[close:]


def _class_itself(mode: str) -> Callable[[str, Mapping[str, Any]], str | None]:
    def transform(sparql: str, params: Mapping[str, Any]) -> str | None:
        if _SELF_UNION.search(sparql):  # the gold counts the class itself: leave it out
            return _SELF_UNION.sub(r"\1", sparql)
        if mode == "transitive" and _TRANSITIVE_TO_IRI.search(sparql):
            return _TRANSITIVE_TO_IRI.sub("rdfs:subClassOf*", sparql)
        if mode == "direct" and _DIRECT_TO_IRI.search(sparql):
            return _DIRECT_TO_IRI.sub("rdfs:subClassOf?", sparql)
        return None

    return transform


def all_reaction_forms(sparql: str, params: Mapping[str, Any]) -> str | None:
    if not _TYPE_TEST.search(sparql):
        return None
    out = _SIDE.sub("(rh:side|rh:substrates|rh:products|rh:substratesOrProducts)", sparql)
    n = 0
    while m := _TYPE_TEST.search(out):
        n += 1
        var = f"?reactionForm{n}"
        out = out[: m.start()] + f"rdfs:subClassOf {var}" + out[m.end():]
        out = insert_in_group(out, m.start(), f"FILTER({var} IN ({REACTION_FORMS}))")
    return out


def _var_groups(sparql: str, var: str) -> list[int]:
    """Positions of one occurrence of ``var`` in each distinct innermost group that holds it."""
    seen: dict[int, int] = {}
    for m in re.finditer(re.escape(var) + r"\b", sparql):
        close = _enclosing_group_end(sparql, m.start())
        seen.setdefault(close, m.start())
    return list(seen.values())


def one_side(var: str, suffix: str) -> Callable[[str, Mapping[str, Any]], str | None]:
    """Reading: the participation bound to side variable ``var`` only on one side (suffix "_L" or "_R")."""

    def transform(sparql: str, params: Mapping[str, Any]) -> str | None:
        positions = _var_groups(sparql, var)
        if not positions:
            return None
        out = sparql
        for pos in sorted(positions, reverse=True):
            out = insert_in_group(out, pos, f'FILTER(STRENDS(STR({var}), "{suffix}"))')
        return out

    return transform


def opposite_side_only(sparql: str, params: Mapping[str, Any]) -> str | None:
    """Golds that pair the named compound's side (?anchorSide) with any side (?side) of the reaction."""
    if "?anchorSide" not in sparql:
        return None
    m = re.search(r"\brh:side\s+\?side\b", sparql)
    if not m:
        return None
    return insert_in_group(sparql, m.start(), "FILTER(?side != ?anchorSide)")


def per_participant(sparql: str, params: Mapping[str, Any]) -> str | None:
    out, changed = sparql, False
    for m in _COUNT_DISTINCT.finditer(sparql):
        p = re.search(r"(\?\w+) rh:compound " + re.escape(m.group(1)) + r"\b", sparql)
        if p:
            out = out.replace(m.group(0), f"COUNT(DISTINCT {p.group(1)})")
            changed = True
    return out if changed else None


def non_distinct_count(var: str) -> Callable[[str, Mapping[str, Any]], str | None]:
    """Reading: COUNT(DISTINCT var) counted without DISTINCT."""

    def transform(sparql: str, params: Mapping[str, Any]) -> str | None:
        needle = f"COUNT(DISTINCT {var})"
        return sparql.replace(needle, f"COUNT({var})") if needle in sparql else None

    return transform


def counted_entity(var: str) -> str:
    """What a counted variable counts: reaction, compound, citation or EC number (by its name in the golds)."""
    name = var.lstrip("?").lower()
    if name == "ec":
        return "ec"
    for entity in ("reaction", "citation"):
        if name.startswith(entity):
            return entity
    return "compound"


def anchor_included(sparql: str, params: Mapping[str, Any]) -> str | None:
    out = _ANCHOR_EXCLUSION.sub("", _ANCHOR_EXCLUSION_AND.sub("", sparql))
    return out if out != sparql else None


def tie_break(sparql: str, params: Mapping[str, Any]) -> str | None:
    if "LIMIT" not in sparql or not _TIE_ORDER.search(sparql):
        return None
    return _TIE_ORDER.sub(r"ORDER BY DESC(\1) DESC(\2)", sparql)


def polymer_class(sparql: str, params: Mapping[str, Any]) -> str | None:
    if not _CHEBI_LINK.search(sparql):
        return None
    return _CHEBI_LINK.sub("(rh:chebi|rh:underlyingChebi)", sparql)


_PARTICIPANT_CHEBI = re.compile(r"(\?\w+) rh:chebi (\?\w+) \.")


def reactive_part_class(sparql: str, params: Mapping[str, Any]) -> str | None:
    """Reading: a participating generic compound counts through the classes of its reactive parts.

    Applies where a gold links a participating compound to a class by ``?compound rh:chebi ?class``
    (the compound then also takes part through rh:compound).
    """
    out, applied = sparql, False
    for m in list(_PARTICIPANT_CHEBI.finditer(sparql)):
        compound, cls = m.group(1), m.group(2)
        if not re.search(r"rh:compound " + re.escape(compound) + r"\b", sparql):
            continue
        replacement = (
            f"{{ {compound} rh:chebi {cls} . }} UNION "
            f"{{ {compound} rh:reactivePart ?reactivePart{cls[1:]} . ?reactivePart{cls[1:]} rh:chebi {cls} . }}"
        )
        out = out.replace(m.group(0), replacement, 1)
        applied = True
    return out if applied else None


# approved_only (study lead's audit, design log): the store holds approved, preliminary and obsolete master
# reactions, and the golds take every status; a reader could list or count only the approved ones. The reading
# restricts the variable by which a variant lists or counts reactions, and the text gets STATUS_NOTE where that
# changes the answer (Reading.note_param).
STATUS_NOTE = "Consider reactions of any status: approved, preliminary or obsolete."
# The note as a sentence of its own, word for word: "Do not consider ...", "It is false that consider ..." or a
# sentence that goes on after "obsolete" does not pass.
CLAUSES["approved_only"] = r"(?:^|[.?!]\s)Consider reactions of any status: approved, preliminary or obsolete\.(?:\s|$)"


def restrict_to_approved(*variables: str) -> Callable[[str, Mapping[str, Any]], str | None]:
    """Reading: only approved reactions bound to the variables; a status triple per variable goes into every
    reaction-graph group that binds it directly (also inside a negation; not into a plain group within one)."""

    def transform(sparql: str, params: Mapping[str, Any]) -> str | None:
        from harness.qgen_rhea.base import RHEA_GRAPH

        groups = _brace_groups(sparql)
        closes: dict[int, list[str]] = {}
        for var in variables:
            for m in re.finditer(re.escape(var) + r"\b", sparql):
                inner = max((g for g in groups if g[0] < m.start() < g[1]), key=lambda g: g[0], default=None)
                in_rhea = inner is not None and inner[2] == "graph" and sparql[: inner[0]].rstrip().endswith(f"<{RHEA_GRAPH}>")
                if in_rhea and var not in closes.setdefault(inner[1], []):
                    closes[inner[1]].append(var)
        if not closes:
            return None
        out = sparql
        for close in sorted(closes, reverse=True):
            before = out[:close].rstrip()
            end = "" if before.endswith((".", "{", "}")) else " ."  # a triple may follow a group or a filter directly
            triples = "".join(f"\n    {var} rh:status rh:Approved ." for var in closes[close])
            out = f"{before}{end}{triples}\n  {out[close:]}"
        return out

    return transform


def approved_only_reading(*variables: str) -> Reading:
    return Reading(
        "approved_only", "only approved reactions listed, counted or qualifying the answer, not preliminary or obsolete ones",
        CLAUSES["approved_only"], transform=restrict_to_approved(*variables), note_param="any_status",
    )


def joint_side(variables: tuple[str, ...], same: bool) -> Callable[[str, Mapping[str, Any]], str | None]:
    """Reading: the named entities' participations all on the same side (same=True) or not all on one side."""

    def transform(sparql: str, params: Mapping[str, Any]) -> str | None:
        if len(variables) < 2 or not all(re.search(re.escape(v) + r"\b", sparql) for v in variables):
            return None
        test = " && ".join(f"{a} = {b}" for a, b in zip(variables, variables[1:]))
        condition = test if same else f"!({test})"
        # The filter goes where the last variable is bound (inside a negation, the outer ones are in scope).
        last = max(variables, key=lambda v: re.search(re.escape(v) + r"\b", sparql).start())
        pos = re.search(re.escape(last) + r"\b", sparql).start()
        return insert_in_group(sparql, pos, f"FILTER({condition})")

    return transform


# --------------------------------------------------------------------------
# other_kind_anchor: a label shared by a compound entry and a chemical class
#
# The question names a compound entry (or a chemical class) whose label is also
# the label of an entity of the other kind. The reading resolves the anchor
# through that entity: a compound anchor becomes the compound entries linked to
# the same-named class or any class below it ("the class and the compounds under
# it"); a class anchor becomes the classes of the same-named compound entries.
# The alternative replaces the anchor IRI by a variable bound by a resolution
# block. The block contains GRAPH clauses, so it goes into the innermost group
# around the anchor that is neither a GRAPH group nor inside one (and not past a
# subquery's WHERE, so the variable is in scope); "VALUES ?v { <anchor> }"
# becomes "BIND(?anchor AS ?v)" after the block.

_GROUP_HEAD = re.compile(
    r"(GRAPH\s+(?:<[^>]*>|[?$]\w+|[A-Za-z_][\w.-]*:[\w.-]*|:[\w.-]*)|VALUES\s+(?:[?$]\w+|\([^)]*\))|WHERE)\s*$",
    re.IGNORECASE,
)


def _brace_groups(sparql: str) -> list[tuple[int, int, str]]:
    """(open, close, kind) of every brace group; kind is graph, values, where or group. IRIs and strings are skipped."""
    groups, stack = [], []
    i = 0
    while i < len(sparql):
        c = sparql[i]
        if c == "<" and (m := _IRI_AT.match(sparql, i)):
            i = m.end()
            continue
        if c == '"':
            j = i + 1
            while j < len(sparql) and sparql[j] != '"':
                j += 2 if sparql[j] == "\\" else 1
            i = j + 1
            continue
        if c == "{":
            head = _GROUP_HEAD.search(sparql[max(0, i - 200):i])
            kind = "group"
            if head:
                word = head.group(1).split()[0].upper()
                kind = {"GRAPH": "graph", "VALUES": "values", "WHERE": "where"}.get(word, "group")
            stack.append((i, kind))
        elif c == "}" and stack:
            start, kind = stack.pop()
            groups.append((start, i, kind))
        i += 1
    return groups


def _home_group(groups: list[tuple[int, int, str]], pos: int) -> tuple[int, int, str] | None:
    """The group the resolution block for an anchor at ``pos`` goes into, or None if there is none."""
    path = sorted((g for g in groups if g[0] < pos < g[1]), key=lambda g: g[0], reverse=True)  # innermost first
    for i, g in enumerate(path):
        inside_graph = any(o[2] == "graph" for o in path[i:])
        if g[2] not in ("graph", "values") and not inside_graph:
            return g
        if g[2] == "where":
            return None
    return None


def _resolution_block(kind: str, label: str, var: str) -> str | None:
    lit = '"' + label.replace("\\", "\\\\").replace('"', '\\"') + '"'
    n = var.lstrip("?")
    if kind == "compound":
        return (
            f"GRAPH <http://sparql.rhea-db.org/chebi> {{ ?{n}Class rdfs:label {lit} . ?{n}Member rdfs:subClassOf* ?{n}Class . }} "
            f"GRAPH <https://sparql.rhea-db.org/rhea> {{ {var} rh:chebi ?{n}Member . }} "
        )
    if kind == "chemical_class":
        return f"GRAPH <https://sparql.rhea-db.org/rhea> {{ ?{n}Compound rh:name {lit} ; rh:chebi {var} . }} "
    return None


def resolve_through_other_kind(sparql: str, iri: str, kind: str, label: str, tag: str) -> str | None:
    """The gold with the anchor ``iri`` resolved through the same-labelled entity of the other kind."""
    groups = _brace_groups(sparql)
    needle = "<" + iri + ">"
    edits: list[tuple[int, int, str]] = []  # (position, length to delete, text to insert)
    homes: dict[int, str] = {}
    for m in re.finditer(re.escape(needle), sparql):
        home = None
        values = next((g for g in groups if g[2] == "values" and g[0] < m.start() < g[1]), None)
        if values is not None:
            clause = re.search(r"VALUES\s+(\?\w+)\s*$", sparql[:values[0]])
            if clause is None or sparql[values[0] + 1:values[1]].strip() != needle:
                return None
            home = _home_group(groups, clause.start())
        else:
            home = _home_group(groups, m.start())
        if home is None:
            return None
        var = homes.setdefault(home[0], f"?homonym{tag}{len(homes)}")
        if values is not None:
            edits.append((clause.start(), values[1] + 1 - clause.start(), f"BIND({var} AS {clause.group(1)})"))
        else:
            edits.append((m.start(), len(needle), var))
    if not edits:
        return None
    for open_pos, var in homes.items():
        block = _resolution_block(kind, label, var)
        if block is None:
            return None
        edits.append((open_pos + 1, 0, " " + block))
    out = sparql
    for pos, length, text in sorted(edits, key=lambda e: e[0], reverse=True):
        out = out[:pos] + text + out[pos + length:]
    return out


def other_kind_reading(anchors: Any) -> Reading:
    """The anchors that share their label with the other kind, resolved through that entity (all at once)."""

    def transform(sparql: str, params: Mapping[str, Any]) -> str | None:
        out, applied = sparql, False
        for i, anchor in enumerate(anchors):
            if not (anchor.label_param and params.get(anchor.homonym_param)):
                continue
            nxt = resolve_through_other_kind(out, params[anchor.entity_param], anchor.kind, params[anchor.label_param], str(i))
            if nxt is None:
                raise ValueError(f"cannot resolve {params[anchor.label_param]!r} through the other kind")
            out, applied = nxt, True
        return out if applied else None

    return Reading(
        "other_kind_anchor",
        "an anchor resolved through the entity of the other kind with the same label (compound entry against chemical class)",
        homonym_clause({a.kind for a in anchors if a.kind in HOMONYM_CLAUSES}),
        transform=transform,
    )


def rhea_readings(
    mode: str,
    anchors: Any = (),
    side_clauses: Mapping[str, str] | None = None,
    gold_pattern: str = "",
    joint_sides: tuple[str, ...] = (),
    joint_clause: str | None = None,
) -> tuple[Reading, ...]:
    """The readings a Rhea variant is checked against; each applies only where its gold has the scope choice.

    Every side variable of the gold (``gold_pattern``) needs its participation
    clause in ``side_clauses``; a variant that lacks one is refused.
    """
    side_clauses = dict(side_clauses or {})
    missing = [v for v in side_vars(gold_pattern) if v not in side_clauses]
    if missing:
        raise ValueError(f"side variables without a participation clause: {missing}")
    out = []
    if mode in ("direct", "transitive"):
        out.append(membership_reading(mode, CLAUSES[mode]))
    out += [
        Reading("class_itself", "the named class counted among its own subclasses, or left out where the gold counts it",
                CLAUSES["class_itself_included" if _SELF_UNION_PATTERN.search(gold_pattern) else "class_itself_excluded"],
                transform=_class_itself(mode)),
        Reading("all_reaction_forms", "directional and bidirectional reactions counted next to the undirected one",
                CLAUSES["all_reaction_forms"], transform=all_reaction_forms),
    ]
    for var in side_vars(gold_pattern):
        for suffix, side in (("_L", "L"), ("_R", "R")):
            out.append(Reading(f"one_side:{var}:{side}", f"the participation of {var} on the {side} side only",
                               side_clauses[var], transform=one_side(var, suffix)))
    if "?side" in side_clauses:
        out.append(Reading("opposite_side_only", "compounds together with the named one only from the other side",
                           side_clauses["?side"], transform=opposite_side_only))
    out.append(Reading("per_participant", "compounds counted per participant (a transported compound twice)",
                       CLAUSES["per_participant"], transform=per_participant))
    for var in dict.fromkeys(_COUNT_DISTINCT.findall(gold_pattern)):
        out.append(Reading(f"non_distinct_count:{var}", f"COUNT({var}) without DISTINCT",
                           COUNTED_CLAUSES[counted_entity(var)], transform=non_distinct_count(var)))
    out += [
        Reading("anchor_included", "the named entity among its own partners", CLAUSES["anchor_included"], transform=anchor_included),
        Reading("tie_break", "the other order among tied values before the LIMIT", CLAUSES["tie_break"], transform=tie_break),
        Reading("polymer_class", "a polymer under the class of its repeating unit", CLAUSES["polymer_class"], transform=polymer_class),
        Reading("reactive_part_class", "a generic compound under the classes of its reactive parts",
                CLAUSES["reactive_part_class"], transform=reactive_part_class),
    ]
    if joint_sides:
        if not joint_clause:
            # Each participation states its own side ("A takes part in, on either side ... B on either side"):
            # the joint readings are ruled out when every one of those statements is there. A single phrase
            # shared by several participations needs an explicit joint clause ("not necessarily the same side").
            clauses = list(dict.fromkeys(side_clauses[v] for v in joint_sides))
            if len(clauses) < len(joint_sides):
                raise ValueError(f"joint side variables {joint_sides} share a clause; give a joint clause")
            joint_clause = "(?s)^" + "".join(f"(?=.*?(?:{c}))" for c in clauses)
        for same, label in ((True, "same"), (False, "different")):
            out.append(Reading(f"joint_side:{label}", f"the named entities' participations on {label} sides",
                               joint_clause, transform=joint_side(tuple(joint_sides), same)))
    if any(a.label_param for a in anchors):
        out.append(other_kind_reading(tuple(anchors)))
    return tuple(out)
