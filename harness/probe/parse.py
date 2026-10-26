"""Reading a probe's reply: the answer rows of probe (a), the query of probe (b).

Probe (a) is read under the record's answer format (``ProbeConfig.answer_format``), so a
record replays under the format it was made with; an unknown format is refused.

Answer format ``rows-json-v2`` (design ruling in the design log-10-19; new probes). The
answer is the first well-formed JSON object with a ``rows`` key anywhere in the reply's
text (thinking and any reasoning channel are not part of it); prose, fences and other JSON
around it are ignored. "First" is by start: an object is tried at every "{" in order, so
an object nested in an earlier one counts when the earlier one has no ``rows`` key, and an
outer object with ``rows`` wins over the objects inside it. Well-formed is strict JSON
(NaN and Infinity are not). Other keys beside ``rows`` are ignored. That object is the
answer: when its rows are not acceptable the reply is unparseable, and no later object is
tried. Its ``rows`` must be what ``submit_report`` accepts (tools.parse_submission, under
the agent's IRI rule), with two readings added before that check:

* a bare JSON string cell is a value: the IRI it spells when ``{"type": "iri", "value": s}``
  passes the IRI rule (an absolute IRI with a known scheme and no whitespace), else the
  plain literal ``s`` (so "CHEBI:15377" and "3" are literals; the scorer reads a numeric
  string as a number in a numeric gold column);
* a row that is not an array is a one-cell row (``{"rows": ["x"]}`` is ``{"rows": [["x"]]}``).

Nothing is repaired: no brackets added, no trailing comma dropped. ``form`` is "json" (the
reply is the object), "fenced_json" (the reply is one code fence around it) or
"embedded_json" (anything else around it).

Answer format ``rows-json-v1`` (the records made before the ruling). The reply's text
with surrounding whitespace removed must be either one JSON object, or one JSON object
inside a single code fence (a line "```" or "```json", the object, a closing line "```")
and nothing else. The object must have exactly one key, ``rows``, whose value
``submit_report`` would accept: an array of rows of equal arity, each cell an IRI object,
a literal object, a JSON number or boolean, or null.

Under either format nothing else is read: an unparseable answer is scored wrong and
counted separately, with the reasons kept.

Declined, rule ``declined-v1`` (a reporting stratum of probe (a), design ruling on issue
26; the outcome label and the score are unchanged, a decline is wrong): a reply not cut by
the output cap that is

* ``empty_rows``: a parsed answer with no rows (``{"rows": []}``, the explicit empty decline;
  the label stays ``answered``, and every gold set has at least one row);
* ``empty_reply``: no text at all (unparseable);
* ``phrase:<name>``: unparseable, holding no ``{`` (no JSON object was attempted), and matching
  one of ``DECLINE_PATTERNS`` (case-insensitive, searched in order; the first match names it).
  The rule is the same under both answer formats: under ``rows-json-v2`` a prose decline with
  a brace in it is still not a phrase decline.

Anything else unparseable is unparseable output, not a decline.

Probe (b), extraction rule ``first-sparql-fence-v1``: the body of the first fenced
block whose info string is ``sparql`` (any case); else the body of the first fenced
block with an empty info string; else the whole reply. The result is stripped of
surrounding whitespace; an empty result is "no query". The query text is never
compared with the gold query: only its executed rows are scored (run.py).
"""

from __future__ import annotations

import json
import re
from typing import Any

from harness.agent.tools import parse_cell, parse_submission

ANSWER_FORMAT = "rows-json-v2"  # new probes (design ruling in the design log)
ANSWER_FORMATS = ("rows-json-v1", "rows-json-v2")
QUERY_EXTRACTION = "first-sparql-fence-v1"
DECLINE_RULE = "declined-v1"

_APOS = "['\u2019]"  # a straight or a typographic apostrophe
_DECLINE_SOURCES = (
    ("dont_know", rf"\bI\s+(?:do\s+not|don{_APOS}t)\s+know\b"),
    ("cannot_answer", rf"\b(?:cannot|can\s+not|can{_APOS}t|unable\s+to|not\s+able\s+to)\s+"
                      r"(?:answer|determine|provide|give|know|tell|say|recall|identify|list)\b"),
    ("no_access", rf"\b(?:do\s+not|don{_APOS}t)\s+have\s+(?:access|enough|the\s+(?:data|information|knowledge)|"
                  r"reliable|specific)\b"),
    ("needs_query", r"\b(?:would|will)\s+need\s+to\s+(?:query|access|look\s+up|run)\b|"
                    r"\bwithout\s+(?:querying|accessing|access\s+to|being\s+able\s+to\s+query|running)\b"),
    ("not_sure", rf"\bI(?:\s+am|{_APOS}m)\s+not\s+(?:sure|certain|confident)\b"),
    ("decline", r"\b(?:decline|refuse)\b"),
)
DECLINE_PATTERNS: tuple[tuple[str, re.Pattern], ...] = tuple(
    (name, re.compile(pattern, re.IGNORECASE)) for name, pattern in _DECLINE_SOURCES)

_ANSWER_FENCE = re.compile(r"\A```(?:json)?[ \t]*\n(.*)\n[ \t]*```\Z", re.DOTALL | re.IGNORECASE)
# A fenced block: an opening line of three backticks and an info string, the body, a closing line.
_FENCE = re.compile(r"^```[ \t]*([^\n`]*)\n(.*?)^[ \t]*```[ \t]*$", re.DOTALL | re.MULTILINE)


def parse_answer(text: str, iri_rule: str, answer_format: str = ANSWER_FORMAT
                 ) -> tuple[list[list[dict | None]] | None, str | None, list[str]]:
    """The reply of probe (a) under ``answer_format`` -> (rows, form, []) or (None, form, problems)."""
    if answer_format == "rows-json-v2":
        return _parse_answer_v2(text, iri_rule)
    if answer_format == "rows-json-v1":
        return _parse_answer_v1(text, iri_rule)
    raise ValueError(f"unknown answer format {answer_format!r}; known: {', '.join(ANSWER_FORMATS)}")


def _strict_constant(name: str) -> None:
    raise ValueError(f"{name} is not JSON")


_STRICT_JSON = json.JSONDecoder(parse_constant=_strict_constant)
NO_ROWS_OBJECT = 'the reply holds no well-formed JSON object with a "rows" key'


def first_rows_object(text: str) -> tuple[dict, int, int] | None:
    """The first well-formed JSON object with a "rows" key in ``text``, with its start and end offsets."""
    start = text.find("{")
    while start != -1:
        try:
            obj, end = _STRICT_JSON.raw_decode(text, start)
        except ValueError:
            obj = None
        if isinstance(obj, dict) and "rows" in obj:
            return obj, start, end
        start = text.find("{", start + 1)
    return None


def _bare_string_cell(value: str, iri_rule: str) -> dict:
    iri = {"type": "iri", "value": value}
    return iri if parse_cell(iri, iri_rule)[1] is None else {"type": "literal", "value": value}


def _v2_rows(rows: Any, iri_rule: str) -> Any:
    """rows-json-v2's two readings, applied before submit_report's check: a row that is not an array
    is one cell, and a bare string cell is an IRI or a plain literal."""
    if not isinstance(rows, list):
        return rows
    return [[_bare_string_cell(c, iri_rule) if isinstance(c, str) else c
             for c in (row if isinstance(row, list) else [row])] for row in rows]


def _parse_answer_v2(text: str, iri_rule: str) -> tuple[list[list[dict | None]] | None, str | None, list[str]]:
    reply = (text or "").strip()
    if not reply:
        return None, None, ["the reply is empty"]
    found = first_rows_object(reply)
    if found is None:
        return None, None, [NO_ROWS_OBJECT]
    obj, start, end = found
    fenced = _ANSWER_FENCE.match(reply)
    if start == 0 and end == len(reply):
        form = "json"
    elif fenced and fenced.group(1).strip() == reply[start:end]:
        form = "fenced_json"
    else:
        form = "embedded_json"
    parsed, problems = parse_submission(_v2_rows(obj["rows"], iri_rule), iri_rule)
    return (parsed, form, []) if parsed is not None else (None, form, problems)


def _parse_answer_v1(text: str, iri_rule: str) -> tuple[list[list[dict | None]] | None, str | None, list[str]]:
    reply = (text or "").strip()
    if not reply:
        return None, None, ["the reply is empty"]
    form = "json"
    fenced = _ANSWER_FENCE.match(reply)
    if fenced:
        form, reply = "fenced_json", fenced.group(1).strip()
    try:
        obj = json.loads(reply)
    except ValueError as exc:
        pos = getattr(exc, "pos", None)
        where = f" at offset {pos}" if isinstance(pos, int) else ""
        return None, None, [f"the reply is not one JSON object: {getattr(exc, 'msg', exc)}{where}"]
    if not isinstance(obj, dict):
        return None, form, [f"the reply is JSON {type(obj).__name__}, not an object"]
    if set(obj) != {"rows"}:
        return None, form, [f"the object's keys are {sorted(obj)}, not exactly ['rows']"]
    rows, problems = parse_submission(obj["rows"], iri_rule)
    return (rows, form, []) if rows is not None else (None, form, problems)


def declined(text: str, rows: list | None, truncated: bool) -> str | None:
    """The ``declined-v1`` stratum of a probe (a) reply (None when it is not a decline)."""
    if truncated:
        return None
    if rows is not None:
        return "empty_rows" if not rows else None
    reply = (text or "").strip()
    if not reply:
        return "empty_reply"
    if "{" in reply:
        return None
    for name, pattern in DECLINE_PATTERNS:
        if pattern.search(reply):
            return f"phrase:{name}"
    return None


def extract_query(text: str) -> tuple[str, str]:
    """The reply of probe (b) -> (query text, source): source is "sparql_fence", "plain_fence",
    "reply" or "none" (nothing left: no query)."""
    fences = [(m.group(1).strip().lower(), m.group(2)) for m in _FENCE.finditer(text or "")]
    for wanted, source in (("sparql", "sparql_fence"), ("", "plain_fence")):
        for info, body in fences:
            if info == wanted:
                query = body.strip()
                return (query, source) if query else ("", "none")
    query = (text or "").strip()
    return (query, "reply") if query else ("", "none")
