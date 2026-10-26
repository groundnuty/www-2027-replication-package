"""RDF terms as they cross the harness.

One JSON form is used everywhere a term is stored or exchanged: gold answers in
question files, rows the agent submits through ``submit_report``, and tool
results in run records. It mirrors SPARQL 1.1 Query Results JSON:

    {"type": "iri", "value": "http://rdf.rhea-db.org/10000"}
    {"type": "literal", "value": "42", "datatype": "http://www.w3.org/2001/XMLSchema#integer"}
    {"type": "literal", "value": "water", "lang": "en"}
    {"type": "bnode", "value": "b0"}
    null                                   (an unbound value)

A row is a list of such cells, in the order of the question's answer columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union

XSD = "http://www.w3.org/2001/XMLSchema#"
RDF_LANGSTRING = "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString"
XSD_STRING = XSD + "string"


@dataclass(frozen=True, slots=True)
class IRI:
    value: str


@dataclass(frozen=True, slots=True)
class Literal:
    lexical: str
    datatype: str | None = None
    lang: str | None = None


@dataclass(frozen=True, slots=True)
class BNode:
    label: str


Term = Union[IRI, Literal, BNode]
Cell = Union[Term, None]  # None is an unbound value
Row = tuple[Cell, ...]


class TermError(ValueError):
    """A cell that is not a valid term in the harness's JSON form."""


def term_from_json(obj: Any) -> Cell:
    """Parse one cell: a SPARQL results JSON binding, or None for unbound."""
    if obj is None:
        return None
    if not isinstance(obj, dict):
        raise TermError(f"cell must be an object or null, got {type(obj).__name__}")
    kind = obj.get("type")
    value = obj.get("value")
    if not isinstance(value, str):
        raise TermError(f"cell value must be a string, got {value!r}")
    if kind in ("iri", "uri"):
        return IRI(value)
    if kind in ("literal", "typed-literal"):
        lang = obj.get("xml:lang", obj.get("lang"))
        datatype = obj.get("datatype")
        if datatype == RDF_LANGSTRING:
            datatype = None
        if lang is not None and datatype is not None:
            raise TermError("a literal cannot have both a language tag and a datatype")
        return Literal(value, datatype, lang)
    if kind == "bnode":
        return BNode(value)
    raise TermError(f"unknown cell type {kind!r}")


def term_to_json(term: Cell) -> dict[str, str] | None:
    """Serialise one cell to the harness's JSON form."""
    if term is None:
        return None
    if isinstance(term, IRI):
        return {"type": "iri", "value": term.value}
    if isinstance(term, Literal):
        out = {"type": "literal", "value": term.lexical}
        if term.datatype is not None:
            out["datatype"] = term.datatype
        if term.lang is not None:
            out["lang"] = term.lang
        return out
    if isinstance(term, BNode):
        return {"type": "bnode", "value": term.label}
    raise TermError(f"not a term: {term!r}")


def row_from_json(cells: list[Any]) -> Row:
    if not isinstance(cells, list):
        raise TermError(f"row must be a list, got {type(cells).__name__}")
    return tuple(term_from_json(c) for c in cells)


def row_to_json(row: Row) -> list[dict[str, str] | None]:
    return [term_to_json(c) for c in row]


def rows_from_sparql_json(result: dict[str, Any]) -> tuple[list[str], list[Row]]:
    """Columns and rows of a SPARQL 1.1 SELECT result in JSON form.

    Unbound variables become None, so every row has one cell per column.
    """
    columns = list(result["head"].get("vars", []))
    rows = []
    for binding in result["results"]["bindings"]:
        rows.append(tuple(term_from_json(binding.get(v)) for v in columns))
    return columns, rows
