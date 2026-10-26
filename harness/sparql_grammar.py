"""rdflib's SPARQL grammar, used under one process-wide lock (dress rehearsal F2).

rdflib parses SPARQL with a pyparsing grammar that is shared by the whole process and is not thread-safe: pyparsing
learns each parse action's arity on the action's first calls and keeps what it learned, so first calls racing on
several threads leave actions that fail for every later parse, on any thread ("TypeError: <lambda>() missing 1
required positional argument"). The dress rehearsal's branch phase (13 worker threads) lost every online checker
verdict that way (unknown, parser_mismatch), and a warm-up on the main thread did not prevent it.

So every use of the grammar holds ``PARSER_LOCK``: parsing (``parse_query``), the algebra translation
(``translate_query``), both together (``prepare_query``), and a query string run on an in-memory rdflib graph
(``graph_query``, which parses it). Concurrent threads serialise there; they gain nothing from running it together,
since parsing is pure Python under the GIL. Nothing else in the harness or the scripts imports rdflib's SPARQL parser,
algebra or ``prepareQuery``, or passes a query string to an rdflib graph (tests/test_sparql_grammar_threads.py
checks every caller on the agent and checker paths). The lock is re-entrant, so a caller may hold it around several
uses (``harness.checks.query.analyse`` does: parse, translate and read the parse tree).
"""

from __future__ import annotations

import threading
from typing import Any

from rdflib.plugins.sparql import prepareQuery as _prepare_query
from rdflib.plugins.sparql.algebra import translateQuery as _translate_query
from rdflib.plugins.sparql.parser import parseQuery as _parse_query

PARSER_LOCK = threading.RLock()


def parse_query(text: str) -> Any:
    """rdflib's parse tree of ``text`` (``parseQuery``), under the lock."""
    with PARSER_LOCK:
        return _parse_query(text)


def translate_query(tree: Any, **kw: Any) -> Any:
    """rdflib's algebra of a parse tree (``translateQuery``), under the lock."""
    with PARSER_LOCK:
        return _translate_query(tree, **kw)


def prepare_query(text: str, **kw: Any) -> Any:
    """rdflib's prepared query (``prepareQuery``: parse and translate), under the lock."""
    with PARSER_LOCK:
        return _prepare_query(text, **kw)


def graph_query(graph: Any, query: Any, **kw: Any) -> Any:
    """``graph.query(query)`` on an in-memory rdflib graph or dataset, under the lock: a query string is parsed
    there. A SELECT's rows are read before the lock is released, so the result is complete when it is returned."""
    with PARSER_LOCK:
        result = graph.query(query, **kw)
        if result.type == "SELECT":
            result.bindings  # noqa: B018 - reading the property evaluates every row now
        return result
