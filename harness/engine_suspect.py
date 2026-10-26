"""A static detector for a query shape on which a pinned store engine returned wrong answers (design log).

On one store a query of this shape came back short, deterministically and with
no flag, in under a second as well as in minutes:

* a grouped sub-SELECT (DISTINCT, REDUCED or GROUP BY) joined to the enclosing
  group pattern through a variable it projects, and
* in that enclosing pattern (as a triple, or inside FILTER EXISTS), an
  ``rdf:type`` test whose class is one of a configured set of classes.

Writing the same join without the grouped sub-SELECT, without the type test,
or with the type test only under FILTER NOT EXISTS or MINUS gave the full
answer. The detector flags the shape, not proven wrong answers: it does not
check the further conditions under which the fault may not appear, so it errs
towards flagging. Flagged outputs are reported as the stratum
``engine_suspect`` beside the main results.

The suspect classes are data, never code: a list of IRIs given by the caller,
or read from the private vocabulary file named by ``ENGINE_SUSPECT_VOCAB`` (a
JSON object whose "iri" maps placeholder keys to IRIs and whose
"suspect_class_keys" lists the keys of the suspect classes). With no suspect
classes the detector flags nothing, so other stores are unaffected.

The scan is lexical, so it works on queries rdflib cannot parse: comments and
strings are removed, prefixed names are expanded with the query's own PREFIX
declarations, and braces are tracked.
"""

from __future__ import annotations

import functools
import json
import os
import re
from collections.abc import Collection
from itertools import pairwise
from pathlib import Path

ENGINE_SUSPECT = "engine_suspect"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
# Some engines predefine the rdf: prefix, so a query may use rdf:type without declaring it.
_DEFAULT_PREFIXES = {"rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#"}
VOCAB_ENV = "ENGINE_SUSPECT_VOCAB"
VOCAB_KEYS = "suspect_class_keys"

_STRING = r'"""(?:[^"\\]|\\.|"(?!""))*"""|\'\'\'(?:[^\'\\]|\\.|\'(?!\'\'))*\'\'\'|"(?:[^"\\\n]|\\.)*"|\'(?:[^\'\\\n]|\\.)*\''
_IRI = r"<[^<>\"{}|^`\\\x00-\x20]*>"
# One left-to-right pass, so a '#' inside an IRI or a string is never taken for a comment.
_LEXEMES = re.compile(f"(?P<string>{_STRING})|(?P<iri>{_IRI})|(?P<comment>#[^\\n]*)", re.DOTALL)
_PREFIX = re.compile(r"PREFIX\s+([A-Za-z_][\w.-]*)?:\s*<([^<>\s]*)>", re.IGNORECASE)
_TOKEN = re.compile(r"<[^<>\"{}|^`\\\x00-\x20]*>|[A-Za-z_][\w.-]*:[\w.-]*|:[\w.-]*|[?$]\w+|[A-Za-z_]\w*|[{}().;,*]")
# The SPARQL keywords after which a group pattern is negated (FILTER NOT EXISTS, MINUS).
_NEGATING = ("MINUS",)


def _strip(query: str) -> str:
    """The query with strings emptied and comments removed; IRIs kept."""
    def keep(m: re.Match) -> str:
        return m.group("iri") if m.group("iri") else (' "" ' if m.group("string") else " ")
    return _LEXEMES.sub(keep, query)


def _tokens(query: str) -> list[str]:
    text = _strip(query)
    prefixes = {**_DEFAULT_PREFIXES, **{p or "": iri for p, iri in _PREFIX.findall(text)}}
    text = _PREFIX.sub(" ", text)
    out = []
    for tok in _TOKEN.findall(text):
        if ":" in tok and not tok.startswith(("<", "?", "$")):
            pfx, local = tok.split(":", 1)
            if pfx in prefixes:
                tok = f"<{prefixes[pfx]}{local}>"
        out.append(tok)
    return out


def _match_brace(tokens: list[str], open_index: int) -> int:
    depth = 0
    for i in range(open_index, len(tokens)):
        if tokens[i] == "{":
            depth += 1
        elif tokens[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return len(tokens) - 1


def _subselects(tokens: list[str]) -> list[tuple[int, int]]:
    """(start, end) token spans of every sub-SELECT: from the brace that opens it to the one that closes it."""
    spans = []
    for i, tok in enumerate(tokens):
        if tok.upper() == "SELECT" and i > 0 and tokens[i - 1] == "{":
            spans.append((i - 1, _match_brace(tokens, i - 1)))
    return spans


def _grouped(tokens: list[str], start: int, end: int) -> bool:
    if tokens[start + 2].upper() in ("DISTINCT", "REDUCED"):
        return True
    where = next((i for i in range(start + 2, end) if tokens[i] == "{"), end)
    group_end = _match_brace(tokens, where) if where < end else end
    tail = [t.upper() for t in tokens[group_end + 1:end]]
    return any(a == "GROUP" and b == "BY" for a, b in pairwise(tail))


def _projected(tokens: list[str], start: int, end: int) -> set[str]:
    where = next((i for i in range(start + 2, end) if tokens[i] == "{" or tokens[i].upper() == "WHERE"), end)
    head = tokens[start + 2:where]
    if "*" in head:
        return {t[1:] for t in tokens[start:end] if t[0] in "?$"}
    out, depth, after_as = set(), 0, False
    for tok in head:
        if tok == "(":
            depth += 1
        elif tok == ")":
            depth -= 1
        elif tok[0] in "?$" and (depth == 0 or after_as):
            out.add(tok[1:])
        after_as = tok.upper() == "AS"
    return out


def _negated_spans(tokens: list[str]) -> list[tuple[int, int]]:
    """Token spans of groups under FILTER NOT EXISTS or MINUS."""
    spans = []
    for i, tok in enumerate(tokens):
        up = tok.upper()
        negating = (up == "EXISTS" and i > 0 and tokens[i - 1].upper() == "NOT") or up in _NEGATING
        if negating and i + 1 < len(tokens) and tokens[i + 1] == "{":
            spans.append((i + 1, _match_brace(tokens, i + 1)))
    return spans


def _inside(i: int, spans: list[tuple[int, int]]) -> bool:
    return any(a <= i <= b for a, b in spans)


def _type_tests(tokens: list[str], classes: set[str], skip: list[tuple[int, int]]) -> bool:
    """An rdf:type test (``a`` or rdf:type) with one of the classes as its object, outside the skipped spans."""
    for i, tok in enumerate(tokens):
        if _inside(i, skip) or not (tok == "a" or tok == f"<{RDF_TYPE}>"):
            continue
        j = i + 1
        while j < len(tokens) and tokens[j] not in (".", ";", "}", "{"):
            if tokens[j].startswith("<") and tokens[j][1:-1] in classes:
                return True
            j += 1
    return False


def suspect(query: str, classes: Collection[str] | None = None) -> bool:
    """Whether the query has the shape (see the module docstring); ``classes`` default to the configured ones."""
    wanted = set(configured_suspect_classes() if classes is None else classes)
    if not wanted:
        return False
    tokens = _tokens(query)
    negated = _negated_spans(tokens)
    for start, end in _subselects(tokens):
        if not _grouped(tokens, start, end):
            continue
        projected = _projected(tokens, start, end)
        outside = [i for i in range(len(tokens)) if not start <= i <= end]
        first_group = next((i for i, t in enumerate(tokens) if t == "{"), len(tokens))
        # a projected variable used in the enclosing pattern (not merely in the outer projection)
        joined = any(tokens[i][0] in "?$" and tokens[i][1:] in projected for i in outside if i > first_group)
        if joined and _type_tests(tokens, wanted, [(start, end), *negated]):
            return True
    return False


def suspect_classes_from_vocab(data: dict) -> list[str]:
    """The suspect-class IRIs of a private vocabulary (see the module docstring).

    A vocabulary without "suspect_class_keys", or one listing a key that "iri" lacks, is refused, so a stale or
    mistyped file cannot switch the flag off unnoticed; an empty list is an explicit "none".
    """
    if VOCAB_KEYS not in data:
        raise ValueError(f'the vocabulary has no "{VOCAB_KEYS}" list')
    iris = data.get("iri", {})
    missing = [k for k in data[VOCAB_KEYS] if k not in iris]
    if missing:
        raise ValueError(f'{len(missing)} of the "{VOCAB_KEYS}" are not in the vocabulary\'s "iri"')
    return sorted(iris[k] for k in data[VOCAB_KEYS])


@functools.lru_cache(maxsize=1)
def _from_env(path: str) -> tuple[str, ...]:
    return tuple(suspect_classes_from_vocab(json.loads(Path(path).expanduser().read_text())))


def configured_suspect_classes() -> tuple[str, ...]:
    """Suspect classes from the vocabulary file named by ENGINE_SUSPECT_VOCAB, or none."""
    path = os.environ.get(VOCAB_ENV)
    return _from_env(path) if path else ()
