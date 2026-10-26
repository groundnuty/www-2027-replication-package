"""An in-memory stand-in for ``SparqlClient``, backed by rdflib.

It runs the exact query strings the builders send to the store, on an rdflib
``Dataset`` whose default graph is the union of its named graphs (as QLever's
is). Two uses: the tests run every builder against hand-made graphs, and the
generated source reads a returned Turtle ontology with the same OWL reader the
curated source uses, so both are read on equal terms.
"""

from __future__ import annotations

import json
import time

from rdflib import Dataset, URIRef

from harness.sparql import QueryResult, SparqlError, query_form
from harness.sparql_grammar import graph_query
from harness.terms import rows_from_sparql_json


class LocalSparqlClient:
    def __init__(self, dataset: Dataset | None = None):
        self.dataset = dataset if dataset is not None else Dataset(default_union=True)
        self.queries: list[str] = []  # every query run, for tests that inspect them

    @classmethod
    def from_graphs(cls, graphs: dict[str, str], fmt: str = "turtle") -> LocalSparqlClient:
        """``graphs`` maps a named-graph IRI to RDF text in ``fmt``."""
        ds = Dataset(default_union=True)
        for name, text in graphs.items():
            ds.graph(URIRef(name)).parse(data=text, format=fmt)
        return cls(ds)

    def settings(self) -> dict:
        return {"endpoint": "rdflib:in-memory", "timeout_s": None, "row_cap": None}

    def query(self, query: str) -> QueryResult:
        form = query_form(query)
        if form not in ("SELECT", "ASK"):
            raise SparqlError("only SELECT and ASK are supported", "not_read_only")
        self.queries.append(query)
        start = time.monotonic()
        try:
            res = graph_query(self.dataset, query)
        except Exception as exc:  # rdflib raises parser and evaluation errors of many types
            raise SparqlError(f"rdflib: {exc}", "parse") from exc
        raw = json.loads(res.serialize(format="json"))
        elapsed = time.monotonic() - start
        if form == "ASK":
            return QueryResult(form, [], [], bool(raw["boolean"]), 0, False, elapsed, raw)
        columns, rows = rows_from_sparql_json(raw)
        return QueryResult(form, columns, rows, None, len(rows), False, elapsed, raw)
