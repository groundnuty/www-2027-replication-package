"""Refusing federated queries: the agent's SPARQL tool reaches only its own store.

A query is refused when it contains the SERVICE keyword (any form: an IRI or a
variable endpoint, SERVICE SILENT, any letter case, at any depth, subqueries
included). Every SERVICE clause, however nested, starts with that keyword token,
so a lexical scan finds every clause a parse of the query's algebra would find;
it needs no parser and cannot be defeated by a query the parser rejects.

The scan follows the SPARQL 1.1 grammar's lexical rules:

* codepoint escapes (``\\uXXXX``, ``\\UXXXXXXXX``) are decoded first, as the
  standard requires before parsing;
* string literals (all four quote forms), IRI references and comments are
  removed, so "service" inside them is ignored; an escaped character in a
  prefixed name (``ex:p\\#``, ``ex:a\\'``) stays part of the name and never
  opens a comment or a string;
* the keyword must stand alone: ``?service``, ``$service``, ``ex:service``,
  ``_:service``, ``@service`` and a prefix named ``service:`` are names, not the
  keyword.

Anything else spelled SERVICE is treated as the keyword. When in doubt the scan
refuses, which costs the model a rewrite, never a request to another endpoint.
"""

from __future__ import annotations

import re

_ESCAPE = re.compile(r"\\u([0-9A-Fa-f]{4})|\\U([0-9A-Fa-f]{8})")
_IRIREF = re.compile(r"<[^<>\"{}|^`\\\x00-\x20]*>")
_SERVICE = re.compile(r"(?<![?$:@\w\-])SERVICE(?![\w:\-])", re.IGNORECASE)


def _decode_escapes(query: str) -> str:
    def repl(m: re.Match) -> str:
        try:
            return chr(int(m.group(1) or m.group(2), 16))
        except (ValueError, OverflowError):
            return m.group(0)

    return _ESCAPE.sub(repl, query)


def strip_literals_iris_comments(query: str) -> str:
    """The query with strings, IRIs and comments replaced by neutral placeholders."""
    out: list[str] = []
    i, n = 0, len(query)
    while i < n:
        c = query[i]
        if query.startswith(('"""', "'''"), i):
            quote = query[i:i + 3]
            j = i + 3
            while j < n and not query.startswith(quote, j):
                j += 2 if query[j] == "\\" else 1
            out.append(' "" ')
            i = j + 3
        elif c in "\"'":
            j = i + 1
            while j < n and query[j] != c and query[j] != "\n":
                j += 2 if query[j] == "\\" else 1
            out.append(' "" ')
            i = j + 1
        elif c == "<" and (m := _IRIREF.match(query, i)):
            out.append(" <> ")
            i = m.end()
        elif c == "\\":
            # Outside strings and IRIs a backslash can only be a PN_LOCAL_ESC inside a
            # prefixed name (ex:p\# , ex:a\'): the escaped character is part of the
            # name and must not open a comment or a string.
            out.append(query[i:i + 2])
            i += 2
        elif c == "#":
            j = query.find("\n", i)
            i = n if j < 0 else j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def uses_service(query: str) -> bool:
    """Whether the query contains a SERVICE clause (see the module docstring)."""
    return _SERVICE.search(strip_literals_iris_comments(_decode_escapes(query))) is not None
