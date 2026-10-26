"""Number traceability (S0.1 contract v2, rules 8 and 21): every number in the
answer appears in a result row the agent received in its trajectory, or equals
the row count of a result it received (0 included). Nothing else is accepted.

It does not depend on the ontology: it is reported as its own layer, never
charged to a source and never a reason to retry.

Numbers are read with the scorer's rules (harness/score/scorer.py on
feat/scorer, rules version scorer-rules-v1), copied here because the scorer is
not on this branch: the numeric family is xsd:integer, decimal, float, double
and the 12 integer-derived types, compared by value with
``math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)``; the lexical form must
follow its datatype's pattern. Unlike the scorer, a plain or xsd:string literal
whose whole lexical form is a number also counts as a number here, so that
dropping a datatype does not hide a number from the check.
"""

from __future__ import annotations

import math
import re
from itertools import permutations
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from harness.terms import XSD, XSD_STRING, Literal, Row

REL_TOL = 1e-6
ABS_TOL = 1e-9

_INTEGER_TYPES = (
    "integer", "int", "long", "short", "byte", "nonNegativeInteger", "positiveInteger",
    "negativeInteger", "nonPositiveInteger", "unsignedLong", "unsignedInt", "unsignedShort",
    "unsignedByte",
)
_INTEGER = re.compile(r"[+-]?[0-9]+")
_DECIMAL = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)")
_FLOATING = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
_PATTERN = {XSD + t: _INTEGER for t in _INTEGER_TYPES}
_PATTERN |= {XSD + "decimal": _DECIMAL, XSD + "float": _FLOATING, XSD + "double": _FLOATING}
# The scorer's numeric family, one copy for the checker: number traceability reads numbers in it (rule 21), and a
# datatype range counts it as one datatype (rule 11 as amended, change-log entry 93; closure.accepted_datatypes).
NUMERIC_DATATYPES = frozenset(_PATTERN)


def number_of(cell) -> float | None:
    """The finite numeric value of a cell, or None if it is not a number."""
    if not isinstance(cell, Literal) or cell.lang is not None:
        return None
    text = cell.lexical.strip()
    if cell.datatype is None or cell.datatype == XSD_STRING:
        pattern = _FLOATING
    else:
        pattern = _PATTERN.get(cell.datatype)
    if pattern is None or not pattern.fullmatch(text):
        return None
    try:
        value = float(Decimal(text))
    except (InvalidOperation, ValueError):
        return None
    return value if math.isfinite(value) else None


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=REL_TOL, abs_tol=ABS_TOL)


@dataclass
class Traceability:
    outcome: str  # "pass", "flag", "not_applicable"
    reason: str
    answer_numbers: list[float]
    untraced: list[float]
    via_row_count: list[float]


def trace(answer: Sequence[Row], row_sets: Sequence[Sequence[Row]]) -> Traceability:
    """Rule 21: a number passes only if it appears in a row the agent received
    (by value, with the scorer's tolerance) or equals the row count of a result
    it received (0 included). No other arithmetic (F24e). ``row_sets``: the
    rows of each query the agent received."""
    wanted: list[float] = []
    for row in answer:
        for cell in row:
            x = number_of(cell)
            if x is not None and not any(close(x, w) for w in wanted):
                wanted.append(x)
    if not wanted:
        return Traceability("not_applicable", "no_numbers", [], [], [])
    in_rows = [x for rows in row_sets for r in rows for x in (number_of(c) for c in r) if x is not None]
    counts = [float(len(rows)) for rows in row_sets]
    untraced, via_count = [], []
    for x in wanted:
        if any(close(x, y) for y in in_rows):
            continue
        if any(close(x, y) for y in counts):
            via_count.append(x)
            continue
        untraced.append(x)
    if untraced:
        return Traceability("flag", "number_not_in_rows", wanted, untraced, via_count)
    return Traceability("pass", "holds", wanted, [], via_count)


def _cell_key(cell):
    """The scorer's equality, approximately: numbers by value, a plain literal
    equal to an xsd:string one, any blank node equal to any blank node."""
    x = number_of(cell)
    if x is not None and isinstance(cell, Literal) and cell.datatype not in (None, XSD_STRING):
        return ("num", x)
    if isinstance(cell, Literal) and cell.lang is None and cell.datatype in (None, XSD_STRING):
        return ("str", cell.lexical)
    if cell is not None and type(cell).__name__ == "BNode":
        return ("bnode",)
    return ("term", cell)


def _cells_equal(a, b) -> bool:
    ka, kb = _cell_key(a), _cell_key(b)
    if ka[0] == "num" and kb[0] == "num":
        return close(ka[1], kb[1])
    return ka == kb


def row_matches(row: Row, result: Row) -> bool:
    """Same arity and, for at most four columns, some column order that makes
    every cell equal (the scorer's column_order "any")."""
    if len(row) != len(result):
        return False
    orders = permutations(range(len(row))) if len(row) <= 4 else [tuple(range(len(row)))]
    return any(all(_cells_equal(row[i], result[j]) for i, j in enumerate(order)) for order in orders)


def row_among(row: Row, rows: Sequence[Row]) -> bool:
    return any(row_matches(row, r) for r in rows)


def rows_contained(answer: Sequence[Row], rows: Sequence[Row]) -> bool:
    """Rule 22: the result rows contain all the submitted rows."""
    return all(row_among(a, rows) for a in answer)
