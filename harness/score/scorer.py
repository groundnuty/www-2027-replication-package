"""Scoring of an agent's answer rows against the gold rows of a question.

An answer and a gold result are tables of RDF terms (``harness.terms``). The
score is the size of a maximum one-to-one matching between answer rows and gold
rows, turned into precision, recall and F1. Two cells are equal, and so can be
matched, under the following rules (fixed by ``ScoringRules`` / RULES_VERSION).

Term equality
-------------
* An IRI equals an IRI only by exact string. It never equals a literal or a
  blank node.
* A blank node equals any blank node (labels are local to a store) and nothing
  else.
* An unbound cell (None) equals only an unbound cell.
* Literals fall into families. A literal of one family never equals a literal
  of another, whatever the lexical forms:

  numeric     xsd:integer, decimal, float, double and the 12 integer-derived
              types (int, long, short, byte, nonNegativeInteger,
              positiveInteger, negativeInteger, nonPositiveInteger,
              unsignedLong, unsignedInt, unsignedShort, unsignedByte). Compared
              by value, not by datatype, so "3"^^xsd:integer equals
              "3.0"^^xsd:decimal and "3.0E0"^^xsd:double. Two values are equal
              when ``math.isclose(a, b, rel_tol, abs_tol)`` holds (symmetric).
              Whitespace around the lexical form is ignored (as for boolean and
              temporal literals below). The lexical form must follow the XSD
              pattern of its own datatype ("3.0" is not a valid xsd:integer);
              the ranges of the derived integer types are not checked. A
              literal that does not parse equals only a literal of the same
              datatype with the identical lexical form. INF, -INF and NaN
              (xsd:float and xsd:double) are exact tokens: each equals itself,
              so NaN equals NaN, and none is close to a finite value. A value
              too large for a float (about 1e308) equals only the same exact
              value.
  boolean     xsd:boolean, lexical "true"/"1" and "false"/"0" (case-sensitive,
              as in XSD), compared by value.
  string      A plain literal and an xsd:string literal are the same; their
              lexical forms are compared exactly (case and whitespace count).
              A language-tagged literal equals a literal with the same lexical
              form and the same tag (tags compared case-insensitively), and
              also a plain / xsd:string literal with that lexical form, in both
              directions. Two different tags never match. (So "x"@en equals
              "x", and "x" equals "x"@fr, but "x"@en does not equal "x"@fr.)
  temporal    xsd:dateTime and xsd:date, same datatype required. When both
              parse, the values are compared: a value with a timezone is
              normalised to UTC, fractional seconds are kept exactly. A value
              with a timezone never equals one without. Values that do not
              parse (including hour 24 and years outside 0001-9999) equal only
              the identical lexical form of the same datatype.
  other       Any other datatype: the same datatype IRI and the identical
              lexical form.
* One exception to the family rule, decided per gold column: where every
  bound gold cell of a column is numeric, an untagged plain or xsd:string
  answer cell in that column whose lexical form is an optional sign, digits
  and an optional decimal part ("+1", "-2", "3.50") is read as a number and
  compared by value. Data sometimes stores numbers as strings (ChEBI writes
  positive charges as "+1"), and an agent returning the data's own value is
  right. Where the gold cell is a string, the strict rule stands ("5" does not
  equal 5). The number of cells read this way is given in ``notes``.
* scorer-rules-v5 (science ruling on issue 3, 21 Oct 02:48, change-log entry
  120) extends that exception, decided per gold column in the same way and
  under the same column permutation, with two more readings of an untagged
  plain or xsd:string answer cell. Each is a field of the rules, on by default;
  with all three v5 fields off the scorer is exactly scorer-rules-v4.
  temporal_strings  where every bound gold cell of a column is xsd:date (or
                    every one xsd:dateTime), a cell whose lexical form is a
                    valid form of that datatype (``_temporal``, the lexical
                    form exactly as written) is read with that datatype and
                    compared by value as above: "2019-05-08" equals the gold
                    "2019-05-08"^^xsd:date.
  anchor_names      against a gold of the "anchor_names" form (gold.py: the
                    names by which the question quotes its anchors, each name
                    of one anchor only and never a marked homonym), where every
                    bound gold cell of a column is an IRI, a cell whose lexical
                    form is exactly one of those names is read as that anchor's
                    IRI: "Prot C" equals the gold IRI of the anchor the
                    question calls "Prot C".
  After the readings, set semantics collapse rows that became equal (as for
  numeric strings). The numbers of cells read each way are given in ``notes``.

Rows and matching
-----------------
* Two rows match when they have the same arity and all cells are equal.
* An answer row whose arity differs from the gold arity never matches; the
  scorer then adds the note "arity mismatch" and a count. Gold rows must all
  have the same arity (a ragged gold table raises ValueError).
* extra_columns (scorer-rules-v5, change-log entry 120): when every answer row
  has the same arity n and n is larger than the gold arity g (the gold not
  empty), the answer is also scored on each choice of g of its columns, the
  chosen columns kept in their order, in lexicographic order of the choice.
  Each choice is scored exactly as a plain answer of arity g: its rows are
  projected onto the chosen columns, then set semantics collapse the rows that
  became equal (an entity listed once per label becomes one row, as it is in
  the gold), then the column rule, the readings above and the matching run as
  for any answer. The plain score (an arity mismatch) comes first; a choice
  replaces the best only by a strictly better (exact, F1, rows matched), so
  ties keep the plain score, then the earliest choice. The choice used is
  given in ``notes``. Every choice is searched; one whose columns no allowed
  permutation can line up with the gold's (by the values each column could
  hold) matches no row and is skipped, which changes no score. A ragged answer
  (rows of several arities) is scored only as before. That is scorer-rules-v5
  (extra_columns_by_kind off), kept for a v5 rules object only.
* extra_columns under scorer-rules-v6 (change-log entry 127; a pre-freeze
  cross-review found that v5 credits a wrong answer: gold (2, 3) is in the
  last two columns of an answer (9, 2, 3)): one choice of g columns is used
  only where exactly one order-kept choice matches the gold's column kinds,
  decided by kind, never by value. A cell's kind is its type: IRI, numeric
  literal (by its datatype's canonical form), any other literal, blank node;
  a column's kind is the one kind of its bound cells (cells of several kinds,
  or none bound, give no kind: such an answer column is never chosen, and such
  a gold column means no choice is made). Where the column rule may permute
  the chosen columns (column_order "any", g at most 4) the kinds match as a
  whole: the one choice exists exactly when the answer has, for each kind of
  the gold, as many columns of that kind as the gold, and it is those columns;
  otherwise the kinds must follow the gold's order and the one choice is the
  one order-kept subsequence of the answer's kinds equal to the gold's. The
  choice is scored as v5 scores a choice (project, set semantics, column rule,
  readings, matching) and replaces the plain score only by a strictly better
  (exact, F1, rows matched). With no choice, or several, the plain score
  stands, and a note says how many choices matched the gold's kinds.
* Where a date or anchor reading happens under some column permutation, the
  readings and set semantics can leave a different number of answer rows
  under each, so the permutation is chosen by the completed score: exact
  first, then F1, then rows matched, the first permutation on a tie.
  Elsewhere (no permutation reads a date or an anchor's name) it is chosen as
  before, the first permutation with the most rows matched.
* column_order "any": when the arities agree and are at most 4, the
  permutations of the answer's columns are tried and the one that matches the
  most rows is used. Since scorer-rules-v4 a permutation may move columns only
  between different value families (IRI, number, string, boolean, a temporal
  or other datatype): columns of the same family keep the question's stated
  order, because nothing in the values tells, say, a count from a charge. Ties go to the identity, then to the lexicographically
  smallest permutation. Gold column j is compared with answer column p[j]; p is
  reported as ``column_permutation``. Above 4 columns only the identity is used.
* Set semantics (multiset=False) first collapse rows that are equal in canonical
  form on each side. The canonical form compares numbers by exact decimal value
  (the tolerance is applied only when matching, so two answer rows that differ
  in the 8th digit are two rows and the extra one costs precision), strings by
  lexical form plus lower-cased tag ("x" and "x"@en stay two rows), and all
  blank nodes as one (rows that differ only in blank-node label collapse).
  Multiset semantics keep every row. ``n_answer_rows`` and ``n_gold_rows`` are
  the counts after this step; the number collapsed is given in ``notes``.
* Set semantics and language tags: deduplication keeps the tag, so "x"@en and
  "x"@fr are two answer rows even though each matches an untagged gold "x";
  untagged answers to a gold that lists one string under several tags collapse
  to one row. Both follow from comparing tags exactly when deduplicating and
  leniently when matching.
* The matching is a maximum bipartite matching, so tolerance, which is not
  transitive, cannot make the result depend on row order. Rows are bucketed by
  everything that is compared exactly; inside a bucket, rows are connected
  when their tolerance-compared parts (numbers, language tags) are compatible.

Score
-----
precision = matched / n_answer, recall = matched / n_gold, F1 their harmonic
mean (0 when both are 0). ``exact`` means matched == n_answer == n_gold.
Empty results: gold and answer both empty is exact with P = R = F1 = 1. Gold
empty and answer not: P = 0, R = 1 (nothing was missed), F1 = 0, not exact. Gold
not empty and answer empty: P = R = F1 = 0 (precision is not vacuously 1), not
exact. An abstained or failed attempt is wrong with P = R = F1 = 0, even when
the gold result is empty.

Gold forms (issue 10 rulings; ``gold.py``)
------------------------------------------
A question with no gold rule is scored against its gold rows as above. A gold
form stands for several plain gold tables; the answer is scored as above
against each of them, and the best score counts: exact first, then F1, then
rows matched, a tie going to the first table in the order below. So a plain
gold table scores exactly as before and RULES_VERSION is unchanged. A score
made against a gold form names it in its first note.

* tie-aware: the tables are the valid fills of the window, the required rows
  plus ``slots`` members of each tied group, each member at most once (under
  set semantics, members equal in canonical form count once). Fills come in
  order: each group's members combined in their listed order, the first group
  varying slowest. Every fill has len(required) + sum(slots) rows, so an answer
  is exact when it holds every required row and ``slots`` members of each group
  and nothing else; a tied member beyond the slots is an extra row. Each fill
  is a plain table, with its own numeric columns and column families: a member
  outside the fill never changes how the answer is read. A rule with more than
  ``MAX_TIE_FILLS`` fills is refused (no ruling needs more than nine). An
  abstained or failed attempt counts len(required) + sum(slots) gold rows.
* either-value: the tables are the alternatives, each as a whole, in their
  listed order; rows of different alternatives never combine. An abstained or
  failed attempt is reported against the first alternative.
"""

from __future__ import annotations

import math
import re
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation, localcontext
from itertools import combinations, permutations, product
from typing import Any, Literal, Sequence

from harness.terms import IRI, XSD, XSD_STRING, BNode, Cell, Row
from harness.terms import Literal as TermLiteral

from .gold import AnchorGold, EitherGold, TieAwareGold
from .rules import ScoringRules

# ---------------------------------------------------------------------------
# Canonical form of one cell
# ---------------------------------------------------------------------------
#
# A cell canonicalises to ``(key, soft, ident)``:
#   key    hashable, compared exactly; it starts with the family of the cell
#   soft   the part compared leniently: (float,) for a number, (tag-or-None,)
#          for a string, () for everything else
#   ident  the exact value of the soft part, used only to deduplicate in set mode
# Two cells are equal iff their keys are equal and their soft parts are compatible.

_Canon = tuple[tuple, tuple, tuple]

_INTEGER_TYPES = (
    "integer", "int", "long", "short", "byte", "nonNegativeInteger", "positiveInteger",
    "negativeInteger", "nonPositiveInteger", "unsignedLong", "unsignedInt", "unsignedShort",
    "unsignedByte",
)
_INTEGER = re.compile(r"[+-]?[0-9]+")
_DECIMAL = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)")
_FLOATING = re.compile(r"[+-]?(?:(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|INF)|NaN")
_NUMERIC_PATTERN = {XSD + name: _INTEGER for name in _INTEGER_TYPES}
_NUMERIC_PATTERN |= {XSD + "decimal": _DECIMAL, XSD + "float": _FLOATING, XSD + "double": _FLOATING}

XSD_BOOLEAN = XSD + "boolean"
XSD_DATETIME = XSD + "dateTime"
XSD_DATE = XSD + "date"
_TEMPORAL = re.compile(
    r"([0-9]{4})-([0-9]{2})-([0-9]{2})(?:T([0-9]{2}):([0-9]{2}):([0-9]{2})(\.[0-9]+)?)?"
    r"(Z|[+-][0-9]{2}:[0-9]{2})?"
)
_NO_SOFT: tuple = ()


def _number(text: str, pattern: re.Pattern) -> _Canon | None:
    if not pattern.fullmatch(text):
        return None
    try:
        value = Decimal(text)
    except InvalidOperation:
        return None
    if not value.is_finite():
        return ("numtok", str(value)), _NO_SOFT, _NO_SOFT
    as_float = float(value)
    if not math.isfinite(as_float):  # beyond the float range: equal only to the exact same value
        return ("numbig", value), _NO_SOFT, _NO_SOFT
    return ("num",), (value,), (value,)


def _temporal(text: str, datatype: str) -> _Canon | None:
    m = _TEMPORAL.fullmatch(text)
    if not m:
        return None
    year, month, day, hour, minute, second, fraction, zone = m.groups()
    if (hour is None) != (datatype == XSD_DATE):
        return None
    try:
        local = datetime(int(year), int(month), int(day), int(hour or 0), int(minute or 0), int(second or 0))
    except ValueError:
        return None
    # The UTC instant as whole seconds, so offsets at the edges of years 0001
    # and 9999 cannot overflow a datetime.
    instant = (local - datetime(1, 1, 1)) // timedelta(seconds=1)
    if zone and zone != "Z":
        offset = timedelta(hours=int(zone[1:3]), minutes=int(zone[4:6]))
        if int(zone[4:6]) > 59 or offset > timedelta(hours=14):
            return None
        instant -= (offset // timedelta(seconds=1)) * (1 if zone[0] == "+" else -1)
    return ("temporal", datatype, zone is not None, instant, Decimal(fraction or 0)), _NO_SOFT, _NO_SOFT


def _canon(cell: Cell) -> _Canon:
    if cell is None:
        return ("null",), _NO_SOFT, _NO_SOFT
    if isinstance(cell, IRI):
        return ("iri", cell.value), _NO_SOFT, _NO_SOFT
    if isinstance(cell, BNode):
        return ("bnode",), _NO_SOFT, _NO_SOFT
    if not isinstance(cell, TermLiteral):
        raise TypeError(f"not a term: {cell!r}")
    lang = cell.lang.lower() if cell.lang else None
    datatype = cell.datatype
    if lang is not None or datatype is None or datatype == XSD_STRING:
        return ("str", cell.lexical), (lang,), (lang,)
    text = cell.lexical.strip()  # XSD whiteSpace=collapse for every non-string datatype below
    found: _Canon | None = None
    if datatype in _NUMERIC_PATTERN:
        found = _number(text, _NUMERIC_PATTERN[datatype])
    elif datatype == XSD_BOOLEAN and text in ("true", "false", "1", "0"):
        found = ("bool", text in ("true", "1")), _NO_SOFT, _NO_SOFT
    elif datatype in (XSD_DATETIME, XSD_DATE):
        found = _temporal(text, datatype)
    return found or (("other", datatype, cell.lexical), _NO_SOFT, _NO_SOFT)


def _close(a: Decimal, b: Decimal, rules: ScoringRules) -> bool:
    """math.isclose semantics in exact decimal arithmetic, so no two distinct
    values become equal through binary rounding (e.g. 2**53 and 2**53 + 1)."""
    with localcontext() as ctx:
        ctx.prec = 200
        rel, tol = Decimal(repr(rules.rel_tol)), Decimal(repr(rules.abs_tol))
        return abs(a - b) <= max(rel * max(abs(a), abs(b)), tol)


def _soft_ok(x: tuple, y: tuple, rules: ScoringRules) -> bool:
    """Whether two soft parts with the same key layout are compatible."""
    for a, b in zip(x, y):
        if isinstance(a, Decimal):
            if not _close(a, b, rules):
                return False
        elif a is not None and b is not None and a != b:  # language tags: untagged matches any tag
            return False
    return True


def terms_equal(a: Cell, b: Cell, rules: ScoringRules = ScoringRules()) -> bool:
    """Equality of two cells under the term rules in the module docstring."""
    (ka, sa, _), (kb, sb, _) = _canon(a), _canon(b)
    return ka == kb and _soft_ok(sa, sb, rules)


# ---------------------------------------------------------------------------
# Maximum matching
# ---------------------------------------------------------------------------

_CRow = tuple[_Canon, ...]


def _dedup(rows: Sequence[_CRow]) -> list[_CRow]:
    """Rows equal in canonical form, kept once (set semantics)."""
    out, seen = [], set()
    for crow in rows:
        signature = tuple((c[0], c[2]) for c in crow)
        if signature not in seen:
            seen.add(signature)
            out.append(crow)
    return out


def _canon_rows(rows: Sequence[Row], rules: ScoringRules) -> tuple[list[_CRow], int]:
    """Canonical rows, deduplicated in set mode; also the number of rows collapsed."""
    out: list[_CRow] = []
    seen: set = set()
    for row in rows:
        crow = tuple(_canon(c) for c in row)
        if not rules.multiset:
            signature = tuple((c[0], c[2]) for c in crow)
            if signature in seen:
                continue
            seen.add(signature)
        out.append(crow)
    return out, len(rows) - len(out)


def _buckets(rows: Sequence[_CRow]) -> dict[tuple, Counter]:
    """Rows grouped by their exactly-compared part; each group counts its soft vectors."""
    out: dict[tuple, Counter] = defaultdict(Counter)
    for row in rows:
        out[tuple(c[0] for c in row)][tuple(x for c in row for x in c[1])] += 1
    return out


def _edges(gold: list, answer: list, rules: ScoringRules) -> list[list[int]]:
    """For each gold class, the answer classes with compatible soft vectors.

    Classes are (soft vector, count) pairs. Candidates come from a window on the
    numeric slot with the most distinct answer values, found by bisection; the
    window is a superset of the isclose neighbourhood, and every candidate is
    then checked on all slots, so the result equals the all-pairs comparison.
    """
    numeric = [i for i, x in enumerate(gold[0][0]) if isinstance(x, Decimal)]
    rel = rules.rel_tol
    if not numeric or rel >= 1:
        return [[j for j, (b, _) in enumerate(answer) if _soft_ok(a, b, rules)] for a, _ in gold]
    slot = max(numeric, key=lambda i: len({b[i] for b, _ in answer}))
    order = sorted(range(len(answer)), key=lambda j: answer[j][0][slot])
    values = [answer[j][0][slot] for j in order]
    adjacency = []
    rel_d, tol_d = Decimal(repr(rel)), Decimal(repr(rules.abs_tol))
    for a, _ in gold:
        centre = a[slot]
        # |x - c| <= max(rel * max(|x|, |c|), abs)  implies  |x - c| <= max(rel * |c| / (1 - rel), abs);
        # the small slack keeps the window a superset despite rounding in the division
        with localcontext() as ctx:
            ctx.prec = 200
            radius = max(rel_d * abs(centre) / (1 - rel_d), tol_d) * Decimal("1.000000001")
            lo, hi = centre - radius, centre + radius
        window = order[bisect_left(values, lo):bisect_right(values, hi)]
        adjacency.append(sorted(j for j in window if _soft_ok(a, answer[j][0], rules)))
    return adjacency


def _max_b_matching(supply: list[int], capacity: list[int], adjacency: list[list[int]]) -> int:
    """Largest number of units that can be routed from left nodes to right nodes.

    Left node u holds supply[u] interchangeable rows, right node v holds
    capacity[v]; a unit may go along any edge of adjacency[u]. This is bipartite
    matching with identical rows merged into one node, by augmenting paths
    (breadth-first, iterative). A left node that has no augmenting path stays
    without one, so each is tried once after a greedy start.
    """
    left, free = list(supply), list(capacity)
    into: list[dict[int, int]] = [{} for _ in capacity]  # into[v][u]: units routed u -> v
    for u, targets in enumerate(adjacency):
        for v in targets:
            units = min(left[u], free[v])
            if units:
                into[v][u] = into[v].get(u, 0) + units
                left[u] -= units
                free[v] -= units
    for root in range(len(adjacency)):
        while left[root]:
            reached_by: dict[int, int] = {}  # right node -> left node it was reached from
            entered_via: dict[int, int] = {}  # left node -> right node whose unit it gives up
            queue, seen, end = deque([root]), {root}, None
            while queue and end is None:
                u = queue.popleft()
                for v in adjacency[u]:
                    if v in reached_by:
                        continue
                    reached_by[v] = u
                    if free[v]:
                        end = v
                        break
                    for w in into[v]:
                        if w not in seen:
                            seen.add(w)
                            entered_via[w] = v
                            queue.append(w)
            if end is None:
                break
            units, v = min(left[root], free[end]), end  # bottleneck along the path found
            while True:
                u = reached_by[v]
                if u == root:
                    break
                v = entered_via[u]
                units = min(units, into[v][u])
            v = end
            while True:
                u = reached_by[v]
                into[v][u] = into[v].get(u, 0) + units
                if u == root:
                    break
                back = entered_via[u]
                into[back][u] -= units
                if not into[back][u]:
                    del into[back][u]
                v = back
            left[root] -= units
            free[end] -= units
    return sum(supply) - sum(left)


def _match_bucket(gold: Counter, answer: Counter, rules: ScoringRules) -> int:
    g, a = list(gold.items()), list(answer.items())
    if len(g) == 1 and len(a) == 1:  # the common case: one class on each side
        return min(g[0][1], a[0][1]) if _soft_ok(g[0][0], a[0][0], rules) else 0
    return _max_b_matching([n for _, n in g], [n for _, n in a], _edges(g, a, rules))


def _matched(gold: dict[tuple, Counter], answer: dict[tuple, Counter], rules: ScoringRules) -> int:
    return sum(_match_bucket(gold[k], answer[k], rules) for k in gold.keys() & answer.keys())


_NUMERIC_STRING = re.compile(r"[+-]?[0-9]+(?:\.[0-9]+)?")


def _numeric_columns(gold: Sequence[_CRow]) -> frozenset[int]:
    """Gold columns whose bound cells are all numeric."""
    cols = set()
    for j in range(len(gold[0]) if gold else 0):
        families = [r[j][0][0] for r in gold if r[j][0][0] != "null"]
        if families and all(f.startswith("num") for f in families):
            cols.add(j)
    return frozenset(cols)


def _coerce_numeric_strings(row: _CRow, numeric_cols: frozenset[int]) -> tuple[_CRow, int]:
    """Read an untagged string answer cell as a number where its gold column is numeric."""
    out, coerced = list(row), 0
    for j in numeric_cols:
        key, soft, _ = row[j]
        if key[0] == "str" and soft == (None,) and _NUMERIC_STRING.fullmatch(key[1]):
            number = _number(key[1], _DECIMAL)
            if number is not None:
                out[j], coerced = number, coerced + 1
    return tuple(out), coerced


def _temporal_columns(gold: Sequence[_CRow]) -> dict[int, str]:
    """scorer-rules-v5: gold columns whose bound cells are all temporal of one datatype -> that datatype."""
    cols = {}
    for j in range(len(gold[0]) if gold else 0):
        keys = [r[j][0] for r in gold if r[j][0][0] != "null"]
        if keys and all(k[0] == "temporal" for k in keys) and len({k[1] for k in keys}) == 1:
            cols[j] = keys[0][1]
    return cols


def _iri_columns(gold: Sequence[_CRow]) -> frozenset[int]:
    """Gold columns whose bound cells are all IRIs."""
    cols = set()
    for j in range(len(gold[0]) if gold else 0):
        kinds = [r[j][0][0] for r in gold if r[j][0][0] != "null"]
        if kinds and all(k == "iri" for k in kinds):
            cols.add(j)
    return frozenset(cols)


def _coerce_v5_strings(row: _CRow, temporal_cols: dict[int, str], iri_cols: frozenset[int],
                       anchors: dict[str, str]) -> tuple[_CRow, int, int]:
    """scorer-rules-v5: read an untagged string answer cell with its gold column's temporal datatype when it is a valid
    lexical form of it (temporal_strings), and as an anchor's IRI when it is exactly that anchor's quoted name and its
    gold column holds IRIs (anchor_names). Returns the row and the cells read each way."""
    out, dates, names = list(row), 0, 0
    for j, datatype in temporal_cols.items():
        key, soft, _ = row[j]
        if key[0] == "str" and soft == (None,):
            value = _temporal(key[1], datatype)
            if value is not None:
                out[j], dates = value, dates + 1
    for j in iri_cols:
        key, soft, _ = row[j]
        if key[0] == "str" and soft == (None,) and key[1] in anchors:
            out[j], names = (("iri", anchors[key[1]]), _NO_SOFT, _NO_SOFT), names + 1
    return tuple(out), dates, names


def _family(key: tuple) -> tuple:
    kind = key[0]
    if kind in ("num", "numtok", "numbig"):
        return ("num",)
    if kind == "other":
        return ("other", key[1])
    if kind == "temporal":
        return ("temporal", key[1])
    return (kind,)


def _column_families(gold: Sequence[_CRow]) -> list[frozenset]:
    """The value families each gold column holds (unbound cells ignored)."""
    if not gold:
        return []
    return [frozenset(_family(r[j][0]) for r in gold if r[j][0][0] != "null") for j in range(len(gold[0]))]


def _keeps_family_order(perm: tuple[int, ...], families: list[frozenset]) -> bool:
    """scorer-rules-v4: a permutation may move columns only between different
    value families; columns of the same family keep the question's order."""
    for i in range(len(perm)):
        for j in range(i + 1, len(perm)):
            if families[i] and families[i] == families[j] and perm[i] > perm[j]:
                return False
    return True


def _best_matching(
    gold: Sequence[_CRow], answer: Sequence[_CRow], arity: int, rules: ScoringRules, anchors: dict[str, str]
) -> tuple[int, list[int], tuple[int, int, int], int]:
    """Matched rows under the best column permutation allowed by the rules, with the number of string cells read as
    numbers, as dates and as anchors' IRIs under that permutation, and the answer rows left after set semantics."""
    gold_buckets = _buckets(gold)
    numeric_cols = _numeric_columns(gold)
    temporal_cols = _temporal_columns(gold) if rules.temporal_strings else {}
    iri_cols = _iri_columns(gold) if anchors and rules.anchor_names else frozenset()
    identity = tuple(range(arity))
    candidates = permutations(range(arity)) if rules.column_order == "any" and arity <= 4 else [identity]
    families = _column_families(gold)
    candidates = [p for p in candidates if _keeps_family_order(tuple(p), families)]
    bound = min(len(gold), len(answer))
    best, best_perm, best_coerced, best_n = -1, identity, (0, 0, 0), len(answer)
    v5_readings = bool(temporal_cols or iri_cols)
    tried: list[tuple] = []  # (permutation, rows matched, answer rows, readings), where a v5 reading can apply
    for perm in candidates:  # lexicographic, identity first; only a strict gain replaces the best
        rows = answer if perm == identity else [tuple(r[j] for j in perm) for r in answer]
        coerced, dates, names = 0, 0, 0
        if numeric_cols:
            pairs = [_coerce_numeric_strings(r, numeric_cols) for r in rows]
            rows, coerced = [r for r, _ in pairs], sum(n for _, n in pairs)
        if v5_readings:
            read = [_coerce_v5_strings(r, temporal_cols, iri_cols, anchors) for r in rows]
            rows, dates, names = [r for r, _, _ in read], sum(d for _, d, _ in read), sum(n for _, _, n in read)
        if (coerced or dates or names) and not rules.multiset:  # "+1" and 1 are now the same row
            rows = _dedup(rows)
        matched = _matched(gold_buckets, _buckets(rows), rules)
        if not v5_readings:  # the v4 choice: the most rows matched, the first permutation on a tie
            if matched > best:
                best, best_perm, best_coerced, best_n = matched, perm, (coerced, dates, names), len(rows)
            if best == bound:
                break
            continue
        tried.append((perm, matched, len(rows), (coerced, dates, names)))
    if not v5_readings:
        return best, list(best_perm), best_coerced, best_n
    if not any(counts[1] or counts[2] for _, _, _, counts in tried):
        # no permutation read a date or an anchor's name: the v4 choice, the first permutation with the most rows
        # matched (review of a18041b: v5 changes nothing where none of its readings happens)
        perm, matched, n_rows, counts = max(tried, key=lambda t: t[1])  # max keeps the first of equal values
        return matched, list(perm), counts, n_rows
    # scorer-rules-v5, where a date or anchor reading happens: the readings and set semantics can leave a different
    # number of answer rows under each permutation, so the completed scores are compared, exact first, then F1
    # (2 matched / (answer rows + gold rows)), then rows matched; the first permutation on a tie
    perm, matched, n_rows, counts = max(tried, key=lambda t: (t[1] == t[2] == len(gold), 2 * t[1] / (t[2] + len(gold)),
                                                              t[1]))
    return matched, list(perm), counts, n_rows


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------

Outcome = Literal["scored", "abstained", "failed"]


@dataclass
class Score:
    outcome: Outcome
    exact: bool
    precision: float
    recall: float
    f1: float
    n_answer_rows: int
    n_gold_rows: int
    n_matched: int
    gold_empty: bool
    column_permutation: list[int] | None
    notes: list[str]
    rules: dict

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> Score:
        return cls(**obj)


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def score_rows(answer: Sequence[Row], gold: Sequence[Row] | TieAwareGold | EitherGold | AnchorGold,
               rules: ScoringRules) -> Score:
    """Score submitted answer rows against gold rows, or a gold form (outcome "scored")."""
    anchors: dict[str, str] = {}
    if isinstance(gold, AnchorGold):
        anchors = dict(gold.anchors) if rules.anchor_names else {}
        gold = gold.gold
    if isinstance(gold, (TieAwareGold, EitherGold)):
        return _score_gold_form(answer, gold, rules, anchors)
    return _score_table(answer, gold, rules, anchors)


def _score_table(answer: Sequence[Row], gold: Sequence[Row], rules: ScoringRules, anchors: dict[str, str]) -> Score:
    """A plain gold table's score, with the extra-columns search (module doc)."""
    gold = [tuple(r) for r in gold]
    answer = [tuple(r) for r in answer]
    score = _score_plain(answer, gold, rules, anchors)
    widths = {len(r) for r in answer}
    if not (rules.extra_columns and gold and len(widths) == 1 and len(gold[0]) < min(widths)):
        return score
    n, g = len(answer[0]), len(gold[0])
    if rules.extra_columns_by_kind:  # scorer-rules-v6: the one choice by kind, or none
        return _score_kind_choice(answer, gold, rules, anchors, score)
    compatible = _column_compatibility(answer, gold, rules, anchors)
    perms = list(permutations(range(g))) if rules.column_order == "any" and g <= 4 else [tuple(range(g))]
    best, best_key = score, (score.exact, score.f1, score.n_matched)
    for cols in combinations(range(n), g):  # in lexicographic order; only a strict gain replaces the best
        # a choice no permutation can line up with the gold's columns matches no row, so it cannot beat the plain
        # score (no row matched either): it is skipped unscored, which changes no result
        if not any(all(compatible[cols[p[j]]][j] for j in range(g)) for p in perms):
            continue
        projected = _score_plain([tuple(r[i] for i in cols) for r in answer], gold, rules, anchors)
        key = (projected.exact, projected.f1, projected.n_matched)
        if key > best_key:
            projected.notes.insert(0, f"answer has {n} columns, gold {g}: scored on answer columns {list(cols)}")
            best, best_key = projected, key
    return best


def _kind(key: tuple) -> str:
    """A cell's kind for scorer-rules-v6's choice of columns, by its type alone: an IRI, a numeric literal (by the
    canonical form of its datatype), any other literal, or a blank node."""
    if key[0] in ("num", "numtok", "numbig"):
        return "numeric"
    if key[0] in ("iri", "bnode"):
        return key[0]
    return "literal"


def _column_kinds(rows: Sequence[Row]) -> list[str | None]:
    """Each column's kind: the one kind of its bound cells, else None (cells of several kinds, or none bound)."""
    out = []
    for j in range(len(rows[0])):
        kinds = {_kind(_canon(r[j])[0]) for r in rows if r[j] is not None}
        out.append(next(iter(kinds)) if len(kinds) == 1 else None)
    return out


def _kind_choices(answer_kinds: Sequence[str | None], gold_kinds: Sequence[str], rules: ScoringRules
                  ) -> tuple[int, tuple[int, ...] | None]:
    """(how many order-kept choices of len(gold_kinds) answer columns match the gold's column kinds, the choice when
    there is exactly one). Where the column rule may permute the chosen columns (column_order "any", at most four
    columns) the kinds match as a whole, each gold kind by as many answer columns; otherwise in the gold's order."""
    g = len(gold_kinds)
    if rules.column_order == "any" and g <= 4:
        need = Counter(gold_kinds)
        have = Counter(k for k in answer_kinds if k is not None)
        count = math.prod(math.comb(have[k], c) for k, c in need.items())
        choice = tuple(i for i, k in enumerate(answer_kinds) if k in need) if count == 1 else None
        return count, choice
    n = len(answer_kinds)
    # ways[j][i]: the order-kept choices matching gold_kinds[j:] within answer columns i: (counted exactly)
    ways = [[0] * (n + 1) for _ in range(g + 1)]
    for i in range(n + 1):
        ways[g][i] = 1
    for j in range(g - 1, -1, -1):
        for i in range(n - 1, -1, -1):
            ways[j][i] = ways[j][i + 1] + (ways[j + 1][i + 1] if answer_kinds[i] == gold_kinds[j] else 0)
    count = ways[0][0]
    if count != 1:
        return count, None
    choice, i = [], 0
    for j in range(g):
        while not (answer_kinds[i] == gold_kinds[j] and ways[j + 1][i + 1]):
            i += 1
        choice.append(i)
        i += 1
    return count, tuple(choice)


def _score_kind_choice(answer: list[Row], gold: list[Row], rules: ScoringRules, anchors: dict[str, str],
                       plain: Score) -> Score:
    """scorer-rules-v6 (change-log entry 127): an answer with more columns than the gold is scored on a choice of its
    columns only where exactly one order-kept choice matches the gold's column kinds; the choice is made by kind, never
    by value, and scored as any answer. With no such choice, or several, the plain score stands, with a note."""
    n, g = len(answer[0]), len(gold[0])
    gold_kinds = _column_kinds(gold)
    if None in gold_kinds:
        plain.notes.append(f"answer has {n} columns, gold {g}: a gold column holds cells of several kinds, so no "
                           f"choice is made by kind")
        return plain
    count, choice = _kind_choices(_column_kinds(answer), gold_kinds, rules)
    named = ", ".join(gold_kinds)
    if choice is None:
        plain.notes.append(f"answer has {n} columns, gold {g}: no choice of {g} columns matches the gold's column kinds "
                           f"({named})" if count == 0 else
                           f"answer has {n} columns, gold {g}: {count} choices of {g} columns match the gold's column "
                           f"kinds ({named}), so none is taken")
        return plain
    projected = _score_plain([tuple(r[i] for i in choice) for r in answer], gold, rules, anchors)
    if (projected.exact, projected.f1, projected.n_matched) <= (plain.exact, plain.f1, plain.n_matched):
        return plain  # no row of the choice matches: the plain score, listed first, stands
    projected.notes.insert(0, f"answer has {n} columns, gold {g}: scored on answer columns {list(choice)}, the one "
                              f"choice of the gold's column kinds")
    return projected


def _column_compatibility(answer: Sequence[Row], gold: Sequence[Row], rules: ScoringRules,
                          anchors: dict[str, str]) -> list[list[bool]]:
    """compatible[i][j]: whether some cell of answer column i could equal some cell of gold column j. Conservative
    (never False where a match is possible): an untagged string cell, which a reading may turn into a number, a date
    or an IRI, is compatible with every column, as is any column of a gold column holding only unbound cells."""
    gold_c = [tuple(_canon(c) for c in r) for r in gold]
    gold_families = _column_families(gold_c)
    out = []
    for i in range(len(answer[0])):
        families, wildcard = set(), False
        for r in answer:
            key, soft, _ = _canon(r[i])
            if key[0] == "str" and soft == (None,):
                wildcard = True
            families.add(_family(key))
        out.append([wildcard or not gold_families[j] or bool(families & gold_families[j]) or
                    ("null",) in families for j in range(len(gold[0]))])
    return out


def _score_plain(answer: list[Row], gold: list[Row], rules: ScoringRules, anchors: dict[str, str]) -> Score:
    """A plain table's score: set semantics (or multiset), then the best column permutation with the string readings,
    then the matching."""
    gold_rows, collapsed_gold = _canon_rows(gold, rules)
    answer_rows, collapsed_answer = _canon_rows(answer, rules)
    if len({len(r) for r in gold_rows}) > 1:
        raise ValueError("gold rows have different arities")
    notes: list[str] = []
    for side, n in (("gold", collapsed_gold), ("answer", collapsed_answer)):
        if n:
            notes.append(f"set semantics: {n} duplicate {side} row(s) collapsed")
    n_gold, n_answer, matched, permutation = len(gold_rows), len(answer_rows), 0, None
    if gold_rows and answer_rows:
        arity = len(gold_rows[0])
        usable = [r for r in answer_rows if len(r) == arity]
        if len(usable) < n_answer:
            notes.append("arity mismatch")
            notes.append(f"{n_answer - len(usable)} of {n_answer} answer row(s) have arity != {arity}")
        if usable:
            matched, permutation, (coerced, dates, names), n_usable = _best_matching(gold_rows, usable, arity, rules,
                                                                                    anchors)
            if permutation != list(range(arity)):
                notes.append(f"column permutation {permutation} applied")
            if coerced:
                notes.append(f"{coerced} numeric string cell(s) compared by value")
            if dates:
                notes.append(f"{dates} date string cell(s) read with the gold column's datatype")
            if names:
                notes.append(f"{names} anchor name(s) read as the anchor's IRI")
            if n_usable < len(usable):
                read = "numeric strings" if not (dates or names) else "strings as numbers, dates or anchors' IRIs"
                notes.append(f"set semantics: {len(usable) - n_usable} answer row(s) collapsed after reading {read}")
                n_answer -= len(usable) - n_usable
            if rules.column_order == "any" and arity > 4:
                notes.append("arity > 4: column order not searched")
    elif not gold_rows and answer_rows:
        notes.append("gold is empty, answer is not")
    elif gold_rows:
        notes.append("empty answer")
    if not gold_rows and not answer_rows:
        precision = recall = 1.0
    else:
        precision = matched / n_answer if n_answer else 0.0
        recall = matched / n_gold if n_gold else 1.0
    return Score(
        outcome="scored",
        exact=matched == n_answer == n_gold,
        precision=precision,
        recall=recall,
        f1=_f1(precision, recall),
        n_answer_rows=n_answer,
        n_gold_rows=n_gold,
        n_matched=matched,
        gold_empty=not gold_rows,
        column_permutation=permutation,
        notes=notes,
        rules=rules.to_json(),
    )


MAX_TIE_FILLS = 1000


def _signature(row: Row) -> tuple:
    """A row's canonical form as set semantics compares it (``_canon_rows``)."""
    return tuple((c[0], c[2]) for c in map(_canon, row))


def _distinct(rows: Sequence[Row], rules: ScoringRules) -> list[Row]:
    """The rows, those equal in canonical form kept once under set semantics."""
    out, seen = [], set()
    for row in rows:
        if rules.multiset or _signature(row) not in seen:
            seen.add(_signature(row))
            out.append(tuple(row))
    return out


def _tie_parts(gold: TieAwareGold, rules: ScoringRules) -> tuple[list[Row], list[tuple[int, list[Row]]], str]:
    """Required rows, (slots, distinct members) per group and the gold's note, after the checks a
    tie-aware gold must pass: one arity, enough distinct members for the slots, no row both required
    and tied or in two groups, at most MAX_TIE_FILLS fills."""
    required = _distinct(gold.required, rules)
    groups = [(g.slots, _distinct(g.members, rules)) for g in gold.groups]
    if len({len(r) for r in [*required, *(m for _, members in groups for m in members)]}) > 1:
        raise ValueError("gold rows have different arities")
    seen = {_signature(r) for r in required}
    fills = 1
    for slots, members in groups:
        if slots > len(members):
            raise ValueError(f"a tied group has {slots} slots but {len(members)} distinct member(s)")
        mine = {_signature(r) for r in members}
        if mine & seen:
            raise ValueError("a tied member is also a required row or a member of another group")
        seen |= mine
        fills *= math.comb(len(members), slots)
    if fills > MAX_TIE_FILLS:
        raise ValueError(f"the tie-aware gold has {fills} fills, more than MAX_TIE_FILLS = {MAX_TIE_FILLS}")
    tied = ", ".join(f"{slots} of {len(members)}" for slots, members in groups) or "none"
    return required, groups, f"tie-aware gold: {len(required)} required row(s), tied groups {tied}"


def _tables(gold: TieAwareGold | EitherGold, rules: ScoringRules) -> tuple[list[list[Row]], list[str]]:
    """The plain gold tables a gold form stands for, in order, with the note naming each."""
    if isinstance(gold, EitherGold):
        n = len(gold.alternatives)
        notes = [f"either-value gold: alternative {name!r} ({i + 1} of {n})" for i, (name, _) in enumerate(gold.alternatives)]
        return [list(rows) for _, rows in gold.alternatives], notes
    required, groups, note = _tie_parts(gold, rules)
    fills = product(*(combinations(members, slots) for slots, members in groups))
    tables = [[*required, *(r for chosen in fill for r in chosen)] for fill in fills]
    return tables, [note] * len(tables)


def check_gold(gold: TieAwareGold | EitherGold | AnchorGold, rules: ScoringRules) -> None:
    """Raise ValueError where scoring against the gold form under ``rules`` would (the same checks)."""
    if isinstance(gold, AnchorGold):
        if isinstance(gold.gold, (TieAwareGold, EitherGold)):
            check_gold(gold.gold, rules)
        elif len({len(r) for r in gold.gold}) > 1:
            raise ValueError("gold rows have different arities")
        return
    if isinstance(gold, TieAwareGold):
        _tie_parts(gold, rules)
        return
    for _, rows in gold.alternatives:
        if len({len(r) for r in rows}) > 1:
            raise ValueError("gold rows have different arities")


def _score_gold_form(answer: Sequence[Row], gold: TieAwareGold | EitherGold, rules: ScoringRules,
                     anchors: dict[str, str] | None = None) -> Score:
    tables, notes = _tables(gold, rules)
    scores = [_score_table(answer, table, rules, anchors or {}) for table in tables]
    best = max(range(len(scores)), key=lambda i: (scores[i].exact, scores[i].f1, scores[i].n_matched, -i))
    score = scores[best]
    score.notes = [notes[best], *score.notes]
    return score


def score_attempt(
    outcome: str, answer: Sequence[Row] | None, gold: Sequence[Row] | TieAwareGold | EitherGold | AnchorGold,
    rules: ScoringRules
) -> Score:
    """Score one attempt: "submitted" is scored, "abstained" and "failed" are wrong."""
    if outcome == "submitted":
        if answer is None:
            raise ValueError("a submitted attempt needs answer rows")
        return score_rows(answer, gold, rules)
    if outcome not in ("abstained", "failed"):
        raise ValueError(f"unknown outcome {outcome!r}; expected submitted, abstained or failed")
    if isinstance(gold, (TieAwareGold, EitherGold)):  # every table scores 0, so the first is reported
        tables, notes = _tables(gold, rules)
        score = score_attempt(outcome, None, tables[0], rules)
        score.notes = [notes[0], *score.notes]
        return score
    if isinstance(gold, AnchorGold):
        return score_attempt(outcome, answer, gold.gold, rules)
    gold_rows, _ = _canon_rows([tuple(r) for r in gold], rules)
    return Score(
        outcome=outcome,
        exact=False,
        precision=0.0,
        recall=0.0,
        f1=0.0,
        n_answer_rows=0,
        n_gold_rows=len(gold_rows),
        n_matched=0,
        gold_empty=not gold_rows,
        column_permutation=None,
        notes=[outcome],
        rules=rules.to_json(),
    )


def summarize(scores: Sequence[Score]) -> dict[str, Any]:
    """Aggregate attempts. Failures and abstentions count as wrong in the *_att figures.

    N_att counts every attempt, N_out only those that produced a scored output.
    ``accuracy_exact`` and ``mean_f1`` divide by N_att (an abstention or failure
    adds 0), the ``*_out`` variants by N_out. A ratio whose denominator is 0 is None.
    All scores must have been made under the same rules.
    """
    if len({repr(sorted(s.rules.items())) for s in scores}) > 1:
        raise ValueError("scores were made under different rules")
    scored = [s for s in scores if s.outcome == "scored"]

    def ratio(total: float, n: int) -> float | None:
        return total / n if n else None

    return {
        "N_att": len(scores),
        "N_out": len(scored),
        "n_abstained": sum(s.outcome == "abstained" for s in scores),
        "n_failed": sum(s.outcome == "failed" for s in scores),
        "n_exact": sum(s.exact for s in scores),
        "accuracy_exact": ratio(sum(s.exact for s in scores), len(scores)),
        "mean_f1": ratio(math.fsum(s.f1 for s in scores), len(scores)),
        "accuracy_exact_out": ratio(sum(s.exact for s in scored), len(scored)),
        "mean_f1_out": ratio(math.fsum(s.f1 for s in scored), len(scored)),
        "n_gold_empty": sum(s.gold_empty for s in scores),
    }
