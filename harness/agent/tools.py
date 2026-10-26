"""The agent's three tools: specs, JSON schemas, wire renderings and execution.

Tool specs are provider-neutral and rendered for both wires (Anthropic
``input_schema``; OpenAI-compatible ``function.parameters``). The schemas are
plain JSON Schema with no ``strict`` mode: strict decoding would reject the
bare-string cells this module must refuse with an explanation.

What the model sees from a tool is a compact JSON string, and that exact
string is what a run record stores as ``result_shown``. Errors are shown as
``{"error": {"kind": ..., "message": ...}}`` on both wires; the Anthropic wire
additionally marks them ``is_error``.

A sparql_query result is at most ``max_result_chars`` characters (code points;
80000 in the frozen settings, design log, change-log entry 66): a longer SELECT
result keeps as many leading rows as fit and ends with a notice line
(``render_query_result``). Errors and ASK results are never cut. The cap governs
only what the model sees: the record keeps the full result, and the submission
is scored and checked uncut. With no cap (records before agent-run/15) the
result and the tool description are the earlier ones, byte for byte.
"""

from __future__ import annotations

import ipaddress
import json
import math
import re
import sys
import time
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

from harness.agent.errors import HarnessAbort, InternalToolError
from harness.agent.infra import StoreUnavailable
from harness.agent.sparql_guard import uses_service
from harness.sparql import QueryResult, SparqlError
from harness.terms import XSD, BNode, IRI, Literal, TermError, row_to_json, term_from_json, term_to_json

XSD_INTEGER = XSD + "integer"
XSD_DECIMAL = XSD + "decimal"
XSD_BOOLEAN = XSD + "boolean"

# Full SPARQL JSON kept per successful query in a run record. A larger row cap
# raises this so replay can always re-render what the model was shown.
RAW_ROW_CAP = 10_000

# Characters of one sparql_query result the model is shown (design log, change-log entry 66), and the smallest cap
# the loop accepts: a smaller one could not hold the notice.
MAX_RESULT_CHARS = 80_000
MIN_RESULT_CHARS = 100
# Ends the shown text where a first row too long to fit was cut, before the notice line.
ROW_CUT_MARKER = "[row 1 cut here]"
# The same where the header {"columns":[...],"rows":[ itself leaves no room for the row marker (not in the tool
# description: the longest header in the committed records is 87 characters).
HEADER_CUT_MARKER = "[header cut here]"

# How many problems a refused submission lists before "and N more".
MAX_REPORTED_PROBLEMS = 10

_SCHEME = re.compile(r"^([A-Za-z][A-Za-z0-9+.\-]*):")

# What counts as an absolute IRI in a submission (design ruling, design log, and the runner review).
# Under "registered_schemes" an IRI must start with a scheme registered with IANA (Permanent or
# Provisional; the snapshot in uri_schemes.json names its source and retrieval date) and ':',
# case-insensitive: http, https, urn, doi, did and the rest. Anything else of the form prefix:local,
# such as CHEBI:15377 or rh:Compound, is a prefixed name, refused with a tool error the model sees
# ("unknown scheme — write IRIs in full"), identically in every arm; a value with no scheme at all is
# "not absolute". "registered_schemes_v3" (agent-run/12) adds: an http or https IRI needs '//' and an
# authority with a non-empty host (after any userinfo, before any port; a port is digits), no IRI may
# hold a broken percent escape ('%' not followed by two hex digits), and none may hold whitespace;
# other registered schemes keep the generic scheme:rest form. A literal's datatype IRI gets exactly
# the same checks. Earlier rules stay for replay (replay.py): "registered_schemes_v2" (agent-run/11:
# '//' and any authority, escapes; no whitespace check on datatypes), "registered_schemes" (agent-run/10),
# "known_schemes" (agent-run/7 to /9, seven schemes) and "any_scheme" (before, any prefix).
URI_SCHEMES = json.loads((Path(__file__).with_name("uri_schemes.json")).read_text(encoding="utf-8"))
REGISTERED_IRI_SCHEMES = frozenset(URI_SCHEMES["schemes"])
KNOWN_IRI_SCHEMES = frozenset({"http", "https", "urn", "ftp", "mailto", "file", "tag"})
IRI_RULES = ("registered_schemes_v3", "registered_schemes_v2", "registered_schemes", "known_schemes", "any_scheme")
_HTTP_AUTHORITY = re.compile(r"^https?://[^/?#]+", re.IGNORECASE)  # agent-run/11's check, kept for replay
_AUTHORITY = re.compile(r"^https?://([^/?#]*)", re.IGNORECASE)
_HOST_PORT = re.compile(r"^(\[[^\]]*\]|[^:\[\]]*)(?::(.*))?$")
_BROKEN_ESCAPE = re.compile(r"%(?![0-9A-Fa-f]{2})")
NOT_ABSOLUTE = "is not absolute; write IRIs in full"
NOT_ABSOLUTE_DATATYPE = "is not an absolute IRI; write datatype IRIs in full"
UNKNOWN_SCHEME = "— write IRIs in full"
UNKNOWN_SCHEME_DATATYPE = "— write datatype IRIs in full"

FEDERATION_MESSAGE = (
    "the query uses SERVICE (a federated query), which this tool does not allow: it queries only its own "
    "knowledge graph. The query was not sent. Rewrite it without SERVICE."
)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict


_IRI_CELL = {
    "type": "object",
    "description": "An IRI, written in full.",
    "properties": {"type": {"const": "iri"}, "value": {"type": "string"}},
    "required": ["type", "value"],
}
_LITERAL_CELL = {
    "type": "object",
    "description": "A literal: a lexical form with an optional datatype IRI or language tag.",
    "properties": {
        "type": {"const": "literal"},
        "value": {"type": "string"},
        "datatype": {"type": "string"},
        "lang": {"type": "string"},
    },
    "required": ["type", "value"],
}

def _sparql_description(max_result_chars: int | None) -> str:
    """The sparql_query description. Without a character cap it is the text every record before agent-run/15
    ran with; with one, it states the cap, the notice and the row-cut marker, so the tools hash pins the cap."""
    if max_result_chars is None:
        return (
            "Run a read-only SPARQL 1.1 query (SELECT or ASK) against the knowledge graph. "
            "Returns JSON: the result columns, the rows (each cell an RDF term object or null "
            "for an unbound value), total_rows, and truncated (true when rows were cut to the row cap). "
            "An ASK query returns {\"boolean\": ...}."
        )
    n = max_result_chars
    return (
        "Run a read-only SPARQL 1.1 query (SELECT or ASK) against the knowledge graph. "
        "Returns JSON: the result columns, the rows (each cell an RDF term object or null "
        "for an unbound value), total_rows, and truncated (true when rows were cut to the row cap "
        "or to the character cap). "
        f"The character cap: a result longer than {n} characters keeps as many leading rows as fit, "
        f"and a line {cut_notice(n, 'K', 'M')} follows the JSON, "
        f"M being total_rows; a first row too long to fit is cut inside and marked {ROW_CUT_MARKER}. "
        "An ASK query returns {\"boolean\": ...}."
    )


def sparql_query_spec(max_result_chars: int | None) -> ToolSpec:
    return ToolSpec(
        name="sparql_query",
        description=_sparql_description(max_result_chars),
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The SPARQL query text."}},
            "required": ["query"],
        },
    )


def cut_notice(max_chars: int, shown_rows: int | str | None = None, total_rows: int | str | None = None) -> str:
    """The line that follows a cut result: with rows, how many of the result's total_rows are shown."""
    if shown_rows is None:
        return f"[result truncated at {max_chars} characters]"
    return f"[result truncated at {max_chars} characters: {shown_rows} of {total_rows} rows shown]"


SPARQL_QUERY = sparql_query_spec(MAX_RESULT_CHARS)

SUBMIT_REPORT = ToolSpec(
    name="submit_report",
    description=(
        "Submit the final answer as rows of RDF terms, one row per answer, with the columns in the "
        "order the question asks for. Each cell is an IRI object {\"type\": \"iri\", \"value\": ...}, "
        "a literal object {\"type\": \"literal\", \"value\": ..., \"datatype\"?: ..., \"lang\"?: ...}, "
        "or null. A JSON number or boolean is read as an xsd:integer, xsd:decimal or xsd:boolean "
        "literal. Plain strings are not accepted. A valid submission ends the attempt."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "description": "The answer rows. Every row has the same number of cells.",
                "items": {
                    "type": "array",
                    "items": {
                        "anyOf": [
                            _IRI_CELL,
                            _LITERAL_CELL,
                            {"type": "null"},
                            {"type": "number"},
                            {"type": "boolean"},
                        ]
                    },
                },
            }
        },
        "required": ["rows"],
    },
)

ABSTAIN = ToolSpec(
    name="abstain",
    description="End the attempt without an answer.",
    input_schema={
        "type": "object",
        "properties": {"reason": {"type": "string", "description": "Why no answer is given."}},
        "required": ["reason"],
    },
)



def agent_tools(max_result_chars: int | None) -> tuple[ToolSpec, ...]:
    """The agent's tools under a character cap (None: the specs of records before agent-run/15)."""
    return (sparql_query_spec(max_result_chars), SUBMIT_REPORT, ABSTAIN)


TOOLS: tuple[ToolSpec, ...] = agent_tools(MAX_RESULT_CHARS)  # the frozen settings' specs
TOOL_NAMES = tuple(t.name for t in TOOLS)


def anthropic_tools(specs: tuple[ToolSpec, ...] = TOOLS) -> list[dict]:
    return [{"name": s.name, "description": s.description, "input_schema": s.input_schema} for s in specs]


def openai_tools(specs: tuple[ToolSpec, ...] = TOOLS) -> list[dict]:
    return [
        {"type": "function", "function": {"name": s.name, "description": s.description, "parameters": s.input_schema}}
        for s in specs
    ]


# --- what the model sees ------------------------------------------------------


def compact_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def render_error(kind: str, message: str) -> str:
    return compact_json({"error": {"kind": kind, "message": message}})


def render_query_result(result: QueryResult, max_chars: int | None = None) -> str:
    """The text the model is shown for a query result. A SELECT result longer than ``max_chars`` characters keeps
    its leading rows that fit, as valid JSON with ``truncated`` true, followed by a line ``cut_notice(max_chars, K,
    M)``: K rows shown of M = ``total_rows`` (the rows the store returned; the row cap applies first). JSON,
    newline and notice together are at most ``max_chars``. When not even the first row fits, the text is cut
    at or inside row 1 (never after its end) and ``ROW_CUT_MARKER`` marks the cut, then the notice line with
    K = 0; when the header ``{"columns":[...],"rows":[`` leaves no room for that (a result with no rows
    included), it is cut inside the header and ``HEADER_CUT_MARKER`` marks the cut, then the same notice line.
    In both the JSON is not complete. Every case ends with the notice line."""
    if result.form == "ASK":
        return compact_json({"boolean": result.boolean})
    rows = [row_to_json(r) for r in result.rows]
    shown = {
        "columns": result.columns,
        "rows": rows,
        "total_rows": result.total_rows,
        "truncated": result.truncated,
    }
    # An answer that reached the endpoint's own row limit (design log; set only by a client configured with one, so
    # every other text is as before): its total_rows is a lower bound.
    exact = not getattr(result, "store_cut", False)
    if not exact:
        shown["total_rows_exact"] = False
    text = compact_json(shown)
    if max_chars is None or len(text) <= max_chars:
        return text
    return _cut_rows(text, result.columns, rows, result.total_rows, max_chars, exact)


def _cut_rows(text: str, columns: list, rows: list, total_rows: int, max_chars: int, exact: bool = True) -> str:
    head = '{"columns":' + compact_json(columns) + ',"rows":['
    tail = ('],"total_rows":' + compact_json(total_rows) + ',"truncated":true'
            + ("" if exact else ',"total_rows_exact":false') + "}")
    encoded = [compact_json(r) for r in rows]
    size, k = len(head) + len(tail), 0
    while k < len(encoded):
        grown = size + len(encoded[k]) + (1 if k else 0)  # the comma before every row but the first
        if grown + 1 + len(cut_notice(max_chars, k + 1, total_rows)) > max_chars:
            break
        size, k = grown, k + 1
    if k:
        return head + ",".join(encoded[:k]) + tail + "\n" + cut_notice(max_chars, k, total_rows)
    note = "\n" + cut_notice(max_chars, 0, total_rows)
    if encoded:
        first = head + encoded[0]
        # At or inside row 1, never after its last character: the marker always stands in a cut row.
        keep = min(max_chars - len(ROW_CUT_MARKER) - len(note), len(first) - 1)
        if keep >= len(head):
            return first[:keep] + ROW_CUT_MARKER + note
    # The header leaves no room for the row marker. Here the cut always falls inside the header: with a row, the
    # check above failed; with none, the whole text (header plus a tail of at least 35) is over the cap.
    return head[: max_chars - len(HEADER_CUT_MARKER) - len(note)] + HEADER_CUT_MARKER + note


def capped_raw(raw: dict, cap: int) -> tuple[dict, bool]:
    """The SPARQL JSON with at most ``cap`` bindings, and whether any were dropped."""
    results = raw.get("results")
    if not isinstance(results, dict) or not isinstance(results.get("bindings"), list):
        return raw, False
    bindings = results["bindings"]
    if len(bindings) <= cap:
        return raw, False
    return {**raw, "results": {**results, "bindings": bindings[:cap]}}, True


# --- submit_report ------------------------------------------------------------


def _decimal_lexical(x: float) -> str:
    text = format(Decimal(repr(x)), "f")
    return text if "." in text else text + ".0"


def is_absolute_iri(value: str, iri_rule: str = "registered_schemes_v3") -> bool:
    return _iri_problem(value, iri_rule, "IRI", NOT_ABSOLUTE, UNKNOWN_SCHEME) is None


def _iri_problem(value: str, iri_rule: str, what: str, not_absolute: str, unknown: str) -> str | None:
    if iri_rule not in IRI_RULES:
        raise ValueError(f"unknown IRI rule {iri_rule!r}; known: {', '.join(IRI_RULES)}")
    match = _SCHEME.match(value)
    if iri_rule == "any_scheme":
        return None if match else f"{what} {value[:80]!r} {not_absolute}"
    if iri_rule == "known_schemes":  # the wording of agent-run/7 to /9, kept exact for replay
        known = match and match.group(1).lower() in KNOWN_IRI_SCHEMES
        return None if known else f"{what} {value[:80]!r} {not_absolute}"
    if not match:
        return f"{what} {value[:80]!r} {not_absolute}"
    scheme = match.group(1).lower()
    if scheme not in REGISTERED_IRI_SCHEMES:
        return f"{what} {value[:80]!r}: unknown scheme {match.group(1)[:40]!r} {unknown}"
    if iri_rule == "registered_schemes":  # agent-run/10: the scheme alone
        return None
    if iri_rule == "registered_schemes_v2":  # agent-run/11, kept exact for replay
        if scheme in ("http", "https") and not _HTTP_AUTHORITY.match(value):
            return f"{what} {value[:80]!r}: an {scheme} IRI needs '//' and a host {unknown}"
        if _BROKEN_ESCAPE.search(value):
            return f"{what} {value[:80]!r}: invalid percent escape {unknown}"
        return None
    if scheme in ("http", "https"):
        problem = _http_authority_problem(value)
        if problem:
            return f"{what} {value[:80]!r}: an {scheme} IRI {problem} {unknown}"
    if _BROKEN_ESCAPE.search(value):
        return f"{what} {value[:80]!r}: invalid percent escape {unknown}"
    if any(c.isspace() for c in value):
        return f"{what} {value[:80]!r} contains whitespace {unknown}"
    return None


_IPV_FUTURE = re.compile(r"^v[0-9A-Fa-f]+\.[A-Za-z0-9._~!$&'()*+,;=:-]+$")


def _ip_literal(inner: str) -> bool:
    """RFC 3986 IP-literal (the text between the brackets): an IPv6 address or an IPvFuture."""
    if _IPV_FUTURE.match(inner):
        return True
    try:
        ipaddress.IPv6Address(inner)
    except ValueError:
        return False
    return True


def _http_authority_problem(value: str) -> str | None:
    """Why an http(s) IRI has no usable authority ``[userinfo@]host[:port]``, or None."""
    match = _AUTHORITY.match(value)
    if match is None:
        return "needs '//' and a host"
    host_port = match.group(1).rpartition("@")[2]  # userinfo never holds an unescaped '@'
    parts = _HOST_PORT.match(host_port)
    host = parts.group(1) if parts else ""
    if not host or host == "[]" or any(c.isspace() for c in host):
        return "needs '//' and a host"
    if host.startswith("[") and not _ip_literal(host[1:-1]):
        return "needs '//' and a host"
    port = parts.group(2)
    if port and not (port.isascii() and port.isdigit()):
        return f"has a port that is not a number ({port[:20]!r})"
    return None


# The fields of a cell object that must be strings when present and not null; ``type`` and ``value`` are checked by
# term_from_json. A field of another type is the agent's malformed input, returned to it as a problem, never a crash
# (confirmation review of 57e148c: a datatype list reached the IRI pattern and crashed the harness).
CELL_STRING_FIELDS = ("datatype", "lang", "xml:lang")
NOT_TEXT = "holds a character that is not valid Unicode text (an unpaired surrogate)"


def valid_text(text: str) -> bool:
    """Whether ``text`` is valid Unicode text: no unpaired surrogate, which JSON's \\u escapes can carry into a
    Python string and which UTF-8 cannot encode (the SPARQL client, the record writer and the provider all do)."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def parse_cell(value: Any, iri_rule: str = "registered_schemes_v3") -> tuple[dict | None, str | None]:
    """One submitted cell -> (cell in the harness JSON form, None) or (None, problem). Never raises on any JSON value:
    every field is type-checked before it is used. The checks run in the order they always did, and a field-type check
    only where the harness used to crash or keep the wrong type, so every input answered before keeps its message."""
    if value is None:
        return None, None
    if isinstance(value, bool):  # before int: True is an int in Python
        return term_to_json(Literal("true" if value else "false", XSD_BOOLEAN)), None
    if isinstance(value, int):
        try:
            lexical = str(value)
        except ValueError:  # longer than the interpreter writes out (sys.get_int_max_str_digits)
            return None, f"an integer of more than {sys.get_int_max_str_digits()} digits cannot be read as a literal"
        return term_to_json(Literal(lexical, XSD_INTEGER)), None
    if isinstance(value, float):
        if not math.isfinite(value):
            return None, f"{value!r} is not a finite number"
        return term_to_json(Literal(_decimal_lexical(value), XSD_DECIMAL)), None
    if isinstance(value, str):
        return None, (
            f"bare string {value[:80]!r} is ambiguous: write {{\"type\": \"iri\", \"value\": ...}} for an IRI "
            f"or {{\"type\": \"literal\", \"value\": ...}} for a literal"
        )
    if not isinstance(value, dict):
        return None, f"a cell must be an object, a number, a boolean or null, got {type(value).__name__}"
    try:
        term = term_from_json(value)
    except TermError as exc:
        return None, str(exc)
    if isinstance(term, BNode):
        return None, "a blank node is not an answer; give an IRI or a literal"
    if isinstance(term, IRI) and (problem := _iri_problem(term.value, iri_rule, "IRI", NOT_ABSOLUTE,
                                                           UNKNOWN_SCHEME)):
        return None, problem
    if isinstance(term, IRI) and any(c.isspace() for c in term.value):
        return None, f"IRI {term.value[:80]!r} contains whitespace"
    for key in CELL_STRING_FIELDS:
        if value.get(key) is not None and not isinstance(value[key], str):
            return None, f"{key} must be a string, got {type(value[key]).__name__}"
    for key in ("value", *CELL_STRING_FIELDS):
        if isinstance(value.get(key), str) and not valid_text(value[key]):
            return None, f"{key} {NOT_TEXT}"
    if isinstance(term, Literal) and term.datatype is not None and (problem := _iri_problem(
            term.datatype, iri_rule, "datatype", NOT_ABSOLUTE_DATATYPE, UNKNOWN_SCHEME_DATATYPE)):
        return None, problem
    return term_to_json(term), None


# How submit_report reads the strings of a cell object, per model (change-log entries 110 and 112; owner, design log, 20
# Oct 22:53, and the follow-up of 21 Oct 00:20, which completes the rule; scripts/run_pilot.py MODELS). None, every
# model's but Gemma's: exactly as before.
# "json_string_cells_v1" (Gemma's, from UniProt dev prompted on): any string of a cell object, a key or the string value
# of a key, that is itself one complete JSON string literal (it starts and ends with a double quote and json.loads
# reads it whole as a string) is decoded once, exactly as json.loads decodes it: the decoded cell is read as the strict
# tool reads any cell, so its decoded content is checked and scored as submitted. Gemma JSON-encodes every cell string
# one time too many: keys, the type value and the cell values alike (entry 112). A string that is no complete JSON
# string literal (inner unescaped quotes, trailing text, surrounding whitespace, single quotes, a malformed escape)
# stays as submitted; two keys decoded to the same cell key are refused (unknown keys stay ignored, as the strict tool
# ignores them); a cell that is no object, and the rows, are read as before. Every refused cell object names its
# unknown keys (as submitted) and lists the expected ones. The tools the model is shown are the same (no nudge).
# One thing more (the 00:20 follow-up): a cell whose type is exactly "number" (as submitted or once decoded) is read
# as a literal with its value unchanged, and a bare JSON number value becomes its JSON text exactly as submitted (1,
# 1.0, 1e3 and -0 are "1", "1.0", "1e3" and "-0"), taken from the call's JSON text (``arguments_raw``), since json.loads
# keeps no number's text. No datatype is chosen: the scorer's rule for an untagged numeric-looking string decides. NaN,
# Infinity, a boolean and null are no JSON numbers and stay as submitted; any other type ("integer") stays refused. A
# bare number whose JSON text is not in the call's text is a harness fault (ValueError), never guessed.
REPORT_KEY_RULES = ("json_string_cells_v1",)
REPORT_CELL_KEYS = ("type", "value", "datatype", "lang")  # the cell keys the tool's schema names (_IRI/_LITERAL_CELL)
_READ_CELL_KEYS = frozenset((*REPORT_CELL_KEYS, "xml:lang"))  # read by term_from_json: never an unknown key
_CELL_KEYS_NOTE = "the keys of a cell are " + ", ".join(REPORT_CELL_KEYS[:-1]) + " and " + REPORT_CELL_KEYS[-1]


def json_string_once(text: str) -> str | None:
    """``text`` decoded once when it is one complete JSON string literal (what json.loads returns for it), else None."""
    if len(text) < 2 or text[0] != '"' or text[-1] != '"':
        return None
    try:
        decoded = json.loads(text)
    except ValueError:  # no JSON at all, more than one value, or a malformed escape
        return None
    return decoded if isinstance(decoded, str) else None


class JsonNumberText(str):
    """A JSON number's text as submitted (json_number_texts), told apart from a JSON string of the same characters."""


def json_number_texts(raw: Any) -> Any:
    """``raw`` (a tool call's JSON text) read as json.loads reads it but with every JSON number kept as its text, as
    submitted (a JsonNumberText); NaN and Infinity, which json.loads reads too, stay floats. None when ``raw`` is no JSON
    text."""
    if not isinstance(raw, str):
        return None
    try:
        return json.loads(raw, parse_int=JsonNumberText, parse_float=JsonNumberText)
    except (ValueError, RecursionError):
        return None


def _json_number_text(number: int | float, texts: Any, key: str) -> int | float | str:
    """The JSON text of the bare number ``number`` submitted as a cell's ``key`` (``texts``: the cell as
    json_number_texts read it), or ``number`` itself when it is NaN or Infinity (no JSON number: kept as submitted)."""
    text = texts.get(key) if isinstance(texts, dict) else None
    if isinstance(text, float) and not math.isfinite(text) and not math.isfinite(number):
        return number
    try:
        parsed = json.loads(text) if isinstance(text, JsonNumberText) else None  # a number there, not a string
    except ValueError:
        parsed = None
    # the same number as json.loads parsed it, not only an equal one (1 and 1.0, 0.0 and -0.0 are other texts)
    if type(parsed) is not type(number) or repr(parsed) != repr(number):
        raise ValueError(f"json_string_cells_v1: the JSON text of the number {number!r} is not in the call's "
                         f"arguments_raw, so it cannot be read as submitted")
    return str(text)


def _named_keys(keys: list[str]) -> str:
    return ", ".join(repr(k[:80]) for k in keys)  # every one (the ruling), each cut as a bare string's is


def _parse_cell_json_strings(value: dict, iri_rule: str, texts: Any = None) -> tuple[dict | None, str | None]:
    """A cell object read under json_string_cells_v1 (REPORT_KEY_RULES); ``texts``: the same cell with its numbers as
    their JSON text (json_number_texts), read only for a bare number under a type "number"."""
    read: dict[str, list[str]] = {}  # the cell key -> the keys of the object decoded to it, in the object's order
    unknown: list[str] = []
    decoded: dict[str, Any] = {}
    for key, item in value.items():
        name = json_string_once(key)
        name = key if name is None else name
        if isinstance(item, str) and (text := json_string_once(item)) is not None:
            item = text
        if name in _READ_CELL_KEYS:
            read.setdefault(name, []).append(key)
        else:
            unknown.append(key)  # ignored by the reading, as the strict tool ignores it; named on a refusal
        decoded[name] = item
    note = ((f"unknown key{'s' if len(unknown) > 1 else ''} {_named_keys(unknown)}; " if unknown else "")
            + _CELL_KEYS_NOTE)
    clashes = [f"the keys {_named_keys(keys)} are each read as {name}" for name, keys in read.items() if len(keys) > 1]
    if clashes:
        return None, "; ".join(clashes) + "; " + note
    missing = [k for k in ("type", "value") if k not in read]
    if missing:
        return None, f"the cell has no {' or '.join(missing)} key; " + note
    if decoded["type"] == "number":  # the 00:20 follow-up: a literal, its value unchanged
        decoded["type"] = "literal"
        number = decoded["value"]
        if isinstance(number, (int, float)) and not isinstance(number, bool):
            decoded["value"] = _json_number_text(number, texts, read["value"][0])
    cell, problem = parse_cell(decoded, iri_rule)
    return (cell, None) if problem is None else (None, f"{problem}; {note}")


def _at(items: Any, k: int) -> Any:
    return items[k] if isinstance(items, list) and k < len(items) else None


def parse_submission(rows: Any, iri_rule: str = "registered_schemes_v3", report_key_rule: str | None = None,
                     number_texts: Any = None) -> tuple[list[list[dict | None]] | None, list[str]]:
    """Validate ``submit_report.rows`` -> (rows in JSON form, []) or (None, problems). ``report_key_rule``: how a cell
    object's keys are read (REPORT_KEY_RULES; None as before). ``number_texts``: under the rule, the same rows with
    their numbers as their JSON text (json_number_texts), never read without it."""
    if iri_rule not in IRI_RULES:
        raise ValueError(f"unknown IRI rule {iri_rule!r}; known: {', '.join(IRI_RULES)}")
    if report_key_rule is not None and report_key_rule not in REPORT_KEY_RULES:
        raise ValueError(f"unknown report-key rule {report_key_rule!r}; known: {', '.join(REPORT_KEY_RULES)}")
    if not isinstance(rows, list):
        return None, [f"rows must be an array of rows, got {type(rows).__name__}"]
    problems: list[str] = []
    out: list[list[dict | None]] = []
    arity: int | None = None
    for i, row in enumerate(rows, start=1):
        if not isinstance(row, list):
            problems.append(f"row {i}: a row must be an array of cells, got {type(row).__name__}")
            continue
        if not row:
            problems.append(f"row {i}: a row must have at least one cell")
            continue
        if arity is None:
            arity = len(row)
        elif len(row) != arity:
            problems.append(f"row {i}: has {len(row)} cells, row 1 has {arity}; every row needs the same number")
        cells = []
        for j, value in enumerate(row, start=1):
            if report_key_rule is not None and isinstance(value, dict):
                cell, problem = _parse_cell_json_strings(value, iri_rule, _at(_at(number_texts, i - 1), j - 1))
            else:
                cell, problem = parse_cell(value, iri_rule)
            if problem is not None:
                problems.append(f"row {i}, column {j}: {problem}")
            cells.append(cell)
        out.append(cells)
    if problems:
        return None, problems
    return out, []


def _problems_message(problems: list[str]) -> str:
    shown = problems[:MAX_REPORTED_PROBLEMS]
    more = len(problems) - len(shown)
    text = "the report was not accepted: " + "; ".join(shown)
    if more:
        text += f"; and {more} more"
    return text + ". Fix these and call submit_report again."


# --- execution ----------------------------------------------------------------


class SparqlLike(Protocol):
    def query(self, query: str) -> QueryResult: ...

    def settings(self) -> dict[str, Any]: ...


@dataclass
class ToolOutcome:
    shown: str
    is_error: bool = False
    error: dict | None = None
    terminal: str | None = None  # "submitted" | "abstained"
    final_rows: list | None = None
    abstain_reason: str | None = None
    result_raw: dict | None = None
    result_raw_truncated: bool = False
    result_meta: dict | None = None
    elapsed_s: float = 0.0
    client_called: bool = False
    refused: bool = False  # refused by the harness before reaching the SPARQL client
    infra_retries: list = field(default_factory=list)  # store-outage retries behind this call (infra.py)
    infra_s: float = 0.0
    # The client's own timing entries for this call (one per request it sent: wall time, client and server
    # time-outs, the server's execution time, the outcome), when the client keeps them (harness.sparql
    # VirtuosoClient.timings); None otherwise (agent-run/13).
    query_timings: list | None = None


def _timings(sparql: SparqlLike) -> list | None:
    """The client's timing list, looked up through wrappers (RetryingSparql's ``inner``)."""
    client = sparql
    for _ in range(4):
        timings = getattr(client, "timings", None)
        if isinstance(timings, list):
            return timings
        client = getattr(client, "inner", None)
        if client is None:
            return None
    return None


def _new_timings(sparql: SparqlLike, before: int | None) -> list | None:
    if hasattr(sparql, "last_query_timings"):  # a replay (RecordedSparql) serves the call's recorded timings
        recorded = sparql.last_query_timings
        return [dict(t) for t in recorded] if isinstance(recorded, list) else None
    timings = _timings(sparql)
    if timings is None or before is None:
        return None
    return [dict(t) for t in timings[before:]]


def _observe_call(sparql: SparqlLike, before: int | None, retries: list) -> None:
    """A client that counts what its requests met for its gate (VirtuosoClient.observe_call, a gate of several slots;
    harness/gate_step_down.py) is told of this call's, looked up through wrappers; any other client is left alone."""
    if before is None:
        return
    client = sparql
    for _ in range(4):
        observe = getattr(client, "observe_call", None)
        if callable(observe):
            observe(before, retries)
            return
        client = getattr(client, "inner", None)
        if client is None:
            return


def _elapsed(sparql: SparqlLike, measured: float) -> float:
    """The call's wall time: as measured, or as recorded when a replay serves the call (RecordedSparql), so a decision
    that reads it (the loop's ceiling, agent-run/20) replays the same."""
    recorded = getattr(sparql, "last_elapsed_s", None)
    return float(recorded) if isinstance(recorded, (int, float)) and not isinstance(recorded, bool) else measured


def _error(kind: str, message: str, elapsed: float = 0.0) -> ToolOutcome:
    return ToolOutcome(shown=render_error(kind, message), is_error=True,
                       error={"kind": kind, "message": message}, elapsed_s=elapsed)


def run_sparql(arguments: dict, sparql: SparqlLike, raw_row_cap: int,
               max_result_chars: int | None = MAX_RESULT_CHARS) -> ToolOutcome:
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        return _error("invalid_arguments", "sparql_query needs a non-empty 'query' string")
    if not valid_text(query):  # the client sends it as UTF-8: the agent's malformed input, not a crash of ours
        return _error("invalid_arguments", f"the query {NOT_TEXT}")
    if uses_service(query):
        out = _error("federation_refused", FEDERATION_MESSAGE)
        out.refused = True
        return out
    timings = _timings(sparql)
    before = len(timings) if timings is not None else None
    start = time.perf_counter()
    try:
        result = sparql.query(query)
    except SparqlError as exc:
        out = _error(exc.kind, str(exc), _elapsed(sparql, time.perf_counter() - start))  # never cut: the cap is for SELECT results
        out.client_called = True
        out.infra_retries, out.infra_s = _infra(sparql)
        out.query_timings = _new_timings(sparql, before)
        _observe_call(sparql, before, out.infra_retries)
        return out
    except StoreUnavailable as down:  # the requests behind it keep their timings in the record (agent-run/20)
        down.query_timings = _new_timings(sparql, before)
        _observe_call(sparql, before, down.retries)
        raise
    except (HarnessAbort, InternalToolError):
        raise
    except Exception as exc:
        raise InternalToolError(f"{type(exc).__name__}: {exc}", client_called=True) from exc
    elapsed = _elapsed(sparql, time.perf_counter() - start)
    try:
        raw, raw_truncated = capped_raw(result.raw, raw_row_cap)
        # A replay serves the stored, already capped raw result and says whether it was capped.
        raw_truncated = raw_truncated or bool(getattr(sparql, "last_raw_truncated", False))
        meta = {
            "form": result.form,
            "columns": list(result.columns),
            "boolean": result.boolean,
            "total_rows": result.total_rows,
            "truncated": result.truncated,
            "elapsed_s": result.elapsed_s,
        }
        if getattr(result, "store_cut", False):  # recorded only when set, so other records keep their exact form
            meta["store_cut"] = True
        retries, infra_s = _infra(sparql)
        shown = render_query_result(result, max_result_chars)
    except HarnessAbort:
        raise
    except Exception as exc:  # our own code failed after the store answered: the client was called (agent-run/16)
        raise InternalToolError(f"{type(exc).__name__}: {exc}", client_called=True) from exc
    _observe_call(sparql, before, retries)
    return ToolOutcome(shown=shown, result_raw=raw, result_raw_truncated=raw_truncated,
                       result_meta=meta, elapsed_s=elapsed, client_called=True, infra_retries=retries,
                       infra_s=infra_s, query_timings=_new_timings(sparql, before))


def _infra(sparql: SparqlLike) -> tuple[list, float]:
    return list(getattr(sparql, "last_retries", None) or []), float(getattr(sparql, "last_infra_s", 0.0) or 0.0)


def run_submit(arguments: dict, iri_rule: str = "registered_schemes_v3",
               report_key_rule: str | None = None, arguments_raw: str | None = None) -> ToolOutcome:
    """``arguments_raw``: the call's JSON text, which ``arguments`` was parsed from; read only under a report-key rule
    (a bare number's text, json_string_cells_v1)."""
    if "rows" not in arguments:
        return _error("invalid_report", "submit_report needs a 'rows' array. Call submit_report again.")
    texts = json_number_texts(arguments_raw) if report_key_rule is not None else None
    rows, problems = parse_submission(arguments["rows"], iri_rule, report_key_rule,
                                      texts.get("rows") if isinstance(texts, dict) else None)
    if problems:
        return _error("invalid_report", _problems_message(problems))
    n = len(rows)
    return ToolOutcome(shown=f"Report submitted: {n} row{'s' if n != 1 else ''}.",
                       terminal="submitted", final_rows=rows)


def run_abstain(arguments: dict) -> ToolOutcome:
    reason = arguments.get("reason")
    if reason is not None and not isinstance(reason, str):
        reason = compact_json(reason)
    return ToolOutcome(shown="Attempt ended without an answer.", terminal="abstained", abstain_reason=reason)


def execute_tool(name: str, arguments: dict, sparql: SparqlLike, *, raw_row_cap: int = RAW_ROW_CAP,
                 iri_rule: str = "registered_schemes_v3",
                 max_result_chars: int | None = MAX_RESULT_CHARS, report_key_rule: str | None = None,
                 arguments_raw: str | None = None) -> ToolOutcome:
    """Run one parsed tool call (``arguments_raw``: its JSON text, read by submit_report only under a report-key
    rule). SparqlError becomes a tool error for the model;
    StoreUnavailable (the store stayed down, see infra.py) propagates; any other
    exception from the SPARQL client is raised as InternalToolError."""
    if name == SPARQL_QUERY.name:
        return run_sparql(arguments, sparql, raw_row_cap, max_result_chars)
    if name == SUBMIT_REPORT.name:
        return run_submit(arguments, iri_rule, report_key_rule, arguments_raw)
    if name == ABSTAIN.name:
        return run_abstain(arguments)
    return _error("unknown_tool", f"there is no tool named {str(name)[:80]!r}; the tools are {', '.join(TOOL_NAMES)}")
