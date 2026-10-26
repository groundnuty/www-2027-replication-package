"""Query shapes on which the pinned engine departs from SPARQL.

Design ruling (design log): such queries are recorded at run time and reported
as an engine-affected stratum beside the main score, so absolute accuracy does
not hide an engine defect.

* ``nested_graph``: a GRAPH group inside another GRAPH group, directly or in a
  subquery. QLever 9bebc60 evaluates the inner group against the outer graph,
  where SPARQL says the inner graph takes over (stores/OPERATIONS.md).

The scan is lexical, so it also works on queries rdflib cannot parse: strings,
IRIs and comments are removed, then GRAPH keywords and braces are tracked.
Prefixed names (``ex:graph``) and variables (``?graph``) are single tokens and
never count as the keyword.
"""

from __future__ import annotations

import re

_STRING = re.compile(r'"""(?:[^"\\]|\\.|"(?!""))*"""|\'\'\'(?:[^\'\\]|\\.|\'(?!\'\'))*\'\'\'|"(?:[^"\\\n]|\\.)*"|\'(?:[^\'\\\n]|\\.)*\'',
                     re.DOTALL)
_IRI = re.compile(r'<[^<>"{}|^`\\\x00-\x20]*>')
_COMMENT = re.compile(r"#[^\n]*")
_TOKEN = re.compile(r"[A-Za-z_][\w.-]*:[\w.-]*|:[\w.-]*|[?$]\w+|[A-Za-z_]\w*|[{}]")


def _tokens(query: str) -> list[str]:
    text = _STRING.sub(" ", query)
    text = _IRI.sub(" <iri> ", text)
    text = _COMMENT.sub(" ", text)
    return _TOKEN.findall(text)


def nested_graph(query: str) -> bool:
    """Whether a GRAPH group opens inside another GRAPH group."""
    depth = 0
    graph_depths: list[int] = []  # brace depth at which each open GRAPH group started
    pending_graph = False  # saw GRAPH; its group opens at the next "{"
    for tok in _tokens(query):
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


def engine_flags(query: str) -> list[str]:
    """The engine-affected shapes this query has, by name."""
    return ["nested_graph"] if nested_graph(query) else []
