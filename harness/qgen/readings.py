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
    # A declared reading must apply to every question of its variant (a
    # transform that finds nothing to change is a template error), unless
    # it is marked optional.
    optional: bool = False

    def alternative(self, variant: Any, params: Mapping[str, Any], gold_sparql: str) -> str | None:
        if self.transform is not None:
            return self.transform(gold_sparql, params)
        if self.pattern is not None:
            from harness.qgen.base import _substitute

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


# Lexical nested-GRAPH detector, copied from harness/agent/engine_flags.py
# (nested_graph) on feat/e0.1-skeleton: strings, IRIs and comments are removed,
# then GRAPH keywords (any case) and braces are tracked. Prefixed names
# (ex:graph) and variables (?graph, $graph) are single tokens, so they never
# count as the keyword, and any graph name form is accepted.
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
    """Whether a GRAPH group opens inside another GRAPH group, directly or in a subquery.

    QLever evaluates such an inner group against the outer graph, where SPARQL
    says the inner graph takes over.
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


def check_readings(
    client: SparqlClient, variant: Any, params: Mapping[str, Any], gold_sparql: str, gold_rows: list, text: str
) -> dict[str, ReadingResult]:
    """Each declared reading of the variant: does it change the answer, and does the text carry its clause?

    Fails (ReadingError) rather than passing silently when a non-optional
    reading yields no distinct query, when the alternative query nests GRAPH
    clauses, or when the alternative query fails on the store.
    """
    out: dict[str, ReadingResult] = {}
    for reading in getattr(variant, "readings", ()):
        alt = reading.alternative(variant, params, gold_sparql)
        if alt is None or alt == gold_sparql:
            if reading.optional:
                continue
            raise ReadingError(f"reading {reading.name} yields no alternative query for this question")
        if nests_graph(alt):
            raise ReadingError(f"reading {reading.name}: the alternative query nests GRAPH clauses")
        try:
            rows = client.query(alt).rows
        except SparqlError as exc:
            raise ReadingError(f"reading {reading.name}: the alternative query failed: {exc}") from exc
        out[reading.name] = ReadingResult(_canonical(rows) != gold_rows, reading.clause_present(text))
    return out


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


SKOS_NARROWER_TRANSITIVE = "<http://www.w3.org/2004/02/skos/core#narrowerTransitive>"


def _subtree_pattern(graph: str, iri: str) -> re.Pattern:
    return re.compile(r"GRAPH\s+" + re.escape(graph) + r"\s*\{\s*\?(\w+)\s+rdfs:subClassOf\*\s+<" + re.escape(iri) + r">\s*\.\s*\}")


def immediate_children_via_link(iri_param: str, graph: str, link: str = SKOS_NARROWER_TRANSITIVE) -> Callable[[str, Mapping[str, Any]], str | None]:
    """Reading: "below X" means X and its immediate children only, by the data's parent link.

    Needed where the rdfs:subClassOf closure is materialised (a plain
    rdfs:subClassOf already reaches every depth): ``GRAPH g { ?v
    rdfs:subClassOf* <X> . }`` becomes ``X`` itself or ``<X> link ?v``.
    """

    def transform(sparql: str, params: Mapping[str, Any]) -> str | None:
        iri = params.get(iri_param)
        if not iri:
            return None
        out, n = _subtree_pattern(graph, iri).subn(
            lambda m: f"GRAPH {graph} {{ {{ VALUES ?{m.group(1)} {{ <{iri}> }} }} UNION {{ <{iri}> {link} ?{m.group(1)} . }} }}",
            sparql,
        )
        return out if n else None

    return transform


def immediate_children_by_closure(iri_param: str, graph: str) -> Callable[[str, Mapping[str, Any]], str | None]:
    """Reading: "below X" means X and its immediate children only, in a materialised hierarchy without a parent link.

    An immediate child is a subclass of X with no other subclass of X between them.
    """

    def transform(sparql: str, params: Mapping[str, Any]) -> str | None:
        iri = params.get(iri_param)
        if not iri:
            return None

        def immediate(m: re.Match) -> str:
            v = m.group(1)
            return (
                f"GRAPH {graph} {{ {{ VALUES ?{v} {{ <{iri}> }} }} UNION {{ ?{v} rdfs:subClassOf <{iri}> . FILTER(?{v} != <{iri}>) "
                f"FILTER NOT EXISTS {{ ?{v} rdfs:subClassOf ?between . ?between rdfs:subClassOf <{iri}> . "
                f"FILTER(?between != ?{v} && ?between != <{iri}>) }} }} }}"
            )

        out, n = _subtree_pattern(graph, iri).subn(immediate, sparql)
        return out if n else None

    return transform


def exclude_anchor(iri_param: str, graph: str) -> Callable[[str, Mapping[str, Any]], str | None]:
    """Reading: "below X" leaves X itself out."""

    def transform(sparql: str, params: Mapping[str, Any]) -> str | None:
        iri = params.get(iri_param)
        if not iri:
            return None
        out, n = _subtree_pattern(graph, iri).subn(
            lambda m: f"GRAPH {graph} {{ ?{m.group(1)} rdfs:subClassOf* <{iri}> . FILTER(?{m.group(1)} != <{iri}>) }}",
            sparql,
        )
        return out if n else None

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
# Clause matching with labels masked (profiles with audit_rules; ported from feat/qgen)

_QUOTED = re.compile(r'"[^"]*"')


def neutral_text(text: str, params: Mapping[str, Any]) -> str:
    """The question text with each quoted parameter value replaced by «its parameter» and any other quote by «quoted».

    Clauses are matched on this text, so a label or a quoted value can never
    supply a clause, and a clause can refer to its anchor by token.
    """
    out = text
    named = [(k, v) for k, v in params.items() if isinstance(v, str) and v]
    for param, value in sorted(named, key=lambda kv: (-len(kv[1]), kv[0])):
        out = out.replace(f'"{value}"', f"«{param}»")
    return _QUOTED.sub("«quoted»", out)
