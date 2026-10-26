"""Model routes: one provider-neutral ``step`` over two wires.

The agent loop keeps a neutral transcript, a list of turns:

    {"role": "user", "kind": "question" | "nudge" | "checker_feedback" | "generic_retry", "text": ...}
    {"role": "assistant", "turn": n, "text": ..., "tool_calls": [...],
     "native_format": "anthropic" | "openai", "native": {...}, ...}
    {"role": "tool", "results": [{"tool_call_id", "name", "content", "is_error"}]}

Each assistant turn carries the provider's own message (``native``). On the
Anthropic wire it is resent verbatim, so thinking blocks and their signatures
go back exactly as received and the history stays append-only. On the
OpenAI-compatible wire the message is rebuilt from the neutral fields, reusing
the argument strings exactly as the server sent them; any reasoning channel is
recorded but never sent back.

Usage is normalised to four disjoint counts, so a turn's whole prompt is the
sum of the first, third and fourth:

    input_tokens        prompt tokens not read from or written to a cache
    output_tokens       generated tokens (thinking included)
    cache_read_tokens   prompt tokens read from a cache
    cache_write_tokens  prompt tokens written to a cache

Anthropic reports these as disjoint counts already. OpenAI-compatible servers
report ``prompt_tokens`` with cached tokens included, so ``input_tokens`` is
``prompt_tokens - cached_tokens`` there. A count the provider did not report
is None, never 0.

Retries are done here, not in the SDK (the SDK clients are built with
``max_retries=0``), so every retry is counted and recorded. Transient classes
(rate limit, 408/409, 5xx and 529, time-outs, connection errors) are retried
with exponential backoff and jitter, honouring ``retry-after``; anything else
ends the step with a ``ProviderError`` of failure type ``provider_error`` (or
``context_overflow``). A transient error after the last retry is the provider
failing, not the agent: it ends the step with failure type ``infrastructure``,
which the attempt records as an infrastructure failure from the provider, never
scored and re-run by the pilot runner (the design log). The step never
runs past the wall budget the loop hands it. When the budget cuts a step short,
it ends ``budget_wall`` (the attempt is capped) only if the attempt's own
deadline cut off a call the provider was answering normally: the step's first
error is a client time-out on a request whose time-out had been clipped to the
deadline. A transient provider error, or a time-out the provider ran into with
its full request time, that leaves no budget to retry is the provider's failure:
``infrastructure``. A step's ``wall_s`` covers every request, retry and
backoff; ``latency_s`` only the request that succeeded.

Streaming (OpenAI-compatible route, ``streaming=True``; off by default, so an
existing route sends and records exactly what it did). The LLM gateway
gateway answers a request whose reply has not started within about 180 s with
a 504, and a non-streamed reply sends nothing until it is complete; a stream
starts at once (gateway probe, 19 Oct 2026). A streamed request is the
same request plus ``stream: true`` and ``stream_options`` ``{"include_usage":
true}`` (unless ``stream_include_usage=False``). The stream is read to its end inside
the retry loop: a stream that breaks, carries an error event or ends before
``[DONE]`` or without a finish reason is a transient provider error, retried
like a 5xx, and a partial reply never reaches the step. The chunks are added up into the reply a
non-streamed request returns and parsed by the same ``parse_openai_response``,
so the turn, its tool calls and its usage are recorded as without streaming.
``request_timeout_s`` stays the time of the whole request, and ``latency_s``
runs to the stream's end.

Stream guards (OpenAI-compatible route, streamed only; each off unless its
route turns it on; stream-integrity probe of 19 Oct 2026, design ruling on
design log). A stream that ends cleanly can still have lost whole events on the
server's side. ``guard_call_identity`` (G1, narrow) rejects a reply with a tool
call that has no id or no function name; a call whose arguments do not parse
is left to the loop, since that may be the model's own invalid JSON.
``guard_token_gap`` (G2) rejects a reply with a chunk whose cumulative
completion-token counter rose by at least ``TOKEN_GAP_MIN_RISE`` while the
chunk delivered fewer than ``TOKEN_GAP_MAX_CHARS_PER_TOKEN`` characters per
token (``token_gaps``). A rejected stream is the same transient connection
error a stream that did not finish is: retried within the step's ordinary
retry budget, never reaching the loop, and an infrastructure failure once the
retries are used up. Its evidence is kept in that retry's ``retry_errors``
entry (``stream_guard``), and a step that fails keeps every rejection's
evidence in its error's details (``stream_guard``).

Every step records ``first_byte_s``, for the request that succeeded: streamed,
the time from sending it to the first line of its body; non-streamed, the
reply arrives whole (the Gateway sends nothing until it is complete), so it is
``latency_s``, as the stream-integrity probe measured it.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Protocol
from urllib.parse import urlsplit, urlunsplit

from harness.agent.errors import HarnessAbort  # noqa: F401  (re-exported)
from harness.agent.tools import TOOLS, ToolSpec, anthropic_tools, openai_tools

NATIVE_ANTHROPIC = "anthropic"
NATIVE_OPENAI = "openai"

# Stop reasons meaning the reply was cut off by the output budget.
OUTPUT_LIMIT_STOPS = frozenset({"max_tokens", "length"})
# Stop reasons meaning the prompt no longer fits the model's context window.
CONTEXT_STOPS = frozenset({"model_context_window_exceeded"})
# Stop reasons meaning the provider declined to answer.
REFUSAL_STOPS = frozenset({"refusal", "content_filter"})

ERROR_MESSAGE_LIMIT = 2000


# --- results and failures -----------------------------------------------------


@dataclass
class StepResult:
    text: str
    tool_calls: list[dict]  # {"id", "name", "arguments": dict | None, "arguments_raw"?: str, "error"?: str}
    usage: dict
    usage_raw: dict
    latency_s: float  # the request that succeeded
    stop_reason: str | None
    response_id: str | None
    model: str | None  # the model the provider says served the reply
    native: dict
    native_format: str
    wall_s: float = 0.0  # the whole step, retries and backoff included
    retries: int = 0
    retry_errors: list[dict] = field(default_factory=list)
    leaked_tool_call: dict | None = None
    refusal: dict | None = None
    max_tokens: int | None = None  # the output budget sent with this request
    first_byte_s: float | None = None  # the request that succeeded: to its body's first line (module docstring)
    # The server's system_fingerprint (OpenAI wire: a non-streamed reply's field, a streamed one's from its chunks,
    # where the Gateway sends it on the closing usage chunk); None when the server sent none (agent-run/21)
    system_fingerprint: str | None = None
    # A route that echoes reasoning (OpenAICompatRoute ``echo_reasoning``): the sha256 of the messages it sent
    # (messages_sha256), which every replay rebuilds and checks; None otherwise (agent-run/22, recorded only when set)
    sent_messages_sha256: str | None = None


class StepFailure(Exception):
    """A step that ended the attempt, with the run record's failure type."""

    def __init__(self, failure_type: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.failure_type = failure_type
        self.message = message
        self.details = details or {}


class ProviderError(StepFailure):
    def __init__(self, failure_type: str, message: str, *, error_class: str, status: int | None = None,
                 code: str | None = None, retries: int = 0, transient: bool = False):
        details = {"error_class": error_class, "status": status, "code": code, "retries": retries,
                   "transient": transient}
        super().__init__(failure_type, message, details)
        self.error_class = error_class
        self.status = status
        self.code = code
        self.retries = retries
        self.transient = transient


class Provider(Protocol):
    model: str

    def step(self, system: str, transcript: list[dict], tools: tuple[ToolSpec, ...] = TOOLS, *,
             max_tokens: int | None = None, time_left_s: float | None = None) -> StepResult: ...

    def route_info(self) -> dict: ...


# --- neutral transcript -> wire -----------------------------------------------


def _arguments_text(call: dict) -> str:
    raw = call.get("arguments_raw")
    if isinstance(raw, str) and call.get("arguments") is not None:
        return raw
    return json.dumps(call.get("arguments") or {}, ensure_ascii=False)


def to_anthropic_messages(transcript: list[dict]) -> list[dict]:
    msgs: list[dict] = []
    for turn in transcript:
        role = turn["role"]
        if role == "user":
            if msgs and msgs[-1]["role"] == "user" and isinstance(msgs[-1]["content"], list):
                # a user turn after tool results (a checker cell's retry message) joins their message
                msgs[-1]["content"].append({"type": "text", "text": turn["text"]})
            else:
                msgs.append({"role": "user", "content": turn["text"]})
        elif role == "assistant":
            native = turn.get("native") or {}
            if turn.get("native_format") == NATIVE_ANTHROPIC and isinstance(native.get("content"), list):
                content = copy.deepcopy(native["content"])
            else:
                content = [{"type": "text", "text": turn["text"]}] if turn.get("text") else []
                content += [{"type": "tool_use", "id": c["id"], "name": c["name"], "input": c.get("arguments") or {}}
                            for c in turn.get("tool_calls", [])]
            if content:  # the API refuses an empty assistant message; a reply with no content is skipped
                msgs.append({"role": "assistant", "content": content})
        elif role == "tool":
            msgs.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": r["tool_call_id"], "content": r["content"],
                 **({"is_error": True} if r.get("is_error") else {})}
                for r in turn["results"]]})
        else:
            raise ValueError(f"unknown transcript role {role!r}")
    return msgs


def turn_calls(turn: dict) -> list[dict]:
    """The calls an assistant turn made: its parsed ones, or the one a reasoning-call rule took from its reasoning
    channel (loop.py; recorded as the turn's ``recovered_tool_call``), which the next request carries as the turn's
    call so that the tool's answer names it."""
    taken = turn.get("recovered_tool_call")
    if not turn.get("tool_calls") and isinstance(taken, dict) and taken.get("taken"):
        return [taken["call"]]
    return turn.get("tool_calls", [])


# The wire names under which an assistant turn's reasoning channel can be sent back (``echo_reasoning``): vLLM reads
# ``reasoning_content`` (and returns ``reasoning`` from later versions on); which one a deployment's template renders
# is measured, not assumed (results/gateway-mitigations-20261020, the echo gate probe).
ECHO_REASONING_FIELDS = ("reasoning_content", "reasoning")


def _echoed_reasoning(turn: dict) -> str | None:
    """An OpenAI-wire assistant turn's reasoning channel as the server returned it, or None. A turn whose calls were
    recovered from that channel sends only the text before them, so the calls are not sent twice."""
    if turn.get("native_format") != NATIVE_OPENAI:
        return None
    native = turn.get("native") or {}
    reasoning = native.get("reasoning_content") or native.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning:
        return None
    recovered = turn.get("recovered_tool_call")
    if isinstance(recovered, dict) and isinstance(recovered.get("offset"), int):
        reasoning = reasoning[: recovered["offset"]].rstrip()
    return reasoning or None


def messages_sha256(messages: list[dict]) -> str:
    """The sha256 of a request's messages as canonical JSON (sorted keys, no white space, UTF-8): what an echoing route
    records of each request it sends, and what a replay rebuilds from the record and compares (replay.RecordedRoute)."""
    text = json.dumps(messages, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def to_openai_messages(system: str, transcript: list[dict], echo_reasoning: str | None = None) -> list[dict]:
    """The transcript on the OpenAI wire. ``echo_reasoning`` (one of ECHO_REASONING_FIELDS; None: never) sends each
    assistant turn's reasoning channel back under that name, as the model's chat template expects for the turns of a
    tool-call chain (a per-model route setting; without it the channel is recorded, never sent)."""
    msgs: list[dict] = [{"role": "system", "content": system}]
    for turn in transcript:
        role = turn["role"]
        if role == "user":
            msgs.append({"role": "user", "content": turn["text"]})
        elif role == "assistant":
            calls = turn_calls(turn)
            m: dict = {"role": "assistant", "content": turn.get("text") or (None if calls else "")}
            if calls:
                m["tool_calls"] = [{"id": c["id"], "type": "function",
                                    "function": {"name": c["name"], "arguments": _arguments_text(c)}}
                                   for c in calls]
            if echo_reasoning is not None and (reasoning := _echoed_reasoning(turn)) is not None:
                m[echo_reasoning] = reasoning
            msgs.append(m)
        elif role == "tool":
            for r in turn["results"]:  # one role:tool message per result
                msgs.append({"role": "tool", "tool_call_id": r["tool_call_id"], "content": r["content"]})
        else:
            raise ValueError(f"unknown transcript role {role!r}")
    return msgs


# --- usage --------------------------------------------------------------------


def _count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def normalise_anthropic_usage(raw: dict | None) -> dict:
    raw = raw or {}
    return {
        "input_tokens": _count(raw.get("input_tokens")),
        "output_tokens": _count(raw.get("output_tokens")),
        "cache_read_tokens": _count(raw.get("cache_read_input_tokens")),
        "cache_write_tokens": _count(raw.get("cache_creation_input_tokens")),
    }


def normalise_openai_usage(raw: dict | None) -> dict:
    raw = raw or {}
    prompt = _count(raw.get("prompt_tokens"))
    details = raw.get("prompt_tokens_details")
    cached = _count(details.get("cached_tokens")) if isinstance(details, dict) else None
    uncached = prompt - cached if prompt is not None and cached is not None else prompt
    return {
        "input_tokens": uncached,
        "output_tokens": _count(raw.get("completion_tokens")),
        "cache_read_tokens": cached,
        "cache_write_tokens": None,  # the protocol has no cache-write count
    }


# --- tool calls as the wire delivers them ---------------------------------------

# Serving-stack special tokens that cannot appear in faithfully serialised arguments.
_SERVING_TOKENS = ('<|"|>', "<|tool_call")


def _strings(obj: Any):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)


NOT_UNICODE = "that is not valid Unicode text (an unpaired surrogate)"


def _all_text(obj: Any) -> bool:
    """Whether every string in a decoded JSON value is valid Unicode text. JSON's \\u escapes can carry an unpaired
    surrogate into a Python string, which UTF-8 cannot encode: such arguments could be neither run (the SPARQL client
    sends UTF-8) nor sent back to the model, so the call is malformed, the model's error (confirmation review of
    57e148c: malformed agent input is never a harness defect)."""
    for s in _strings(obj):
        try:
            s.encode("utf-8")
        except UnicodeEncodeError:
            return False
    return True


def parse_tool_arguments(raw: Any) -> tuple[dict | None, str | None]:
    """OpenAI ``function.arguments`` (a JSON string) -> (object, None) or (None, why not).

    Nothing is repaired: a payload that does not parse as a JSON object is
    reported, not coerced.
    """
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return {}, None
    if not isinstance(raw, str):
        return None, f"arguments arrived as {type(raw).__name__}, not a JSON string"
    try:
        decoded = json.loads(raw)
    except ValueError as exc:
        pos = getattr(exc, "pos", None)
        where = f" at offset {pos} of {len(raw)}" if isinstance(pos, int) else ""
        return None, f"arguments did not parse as JSON: {getattr(exc, 'msg', exc)}{where}"
    if not isinstance(decoded, dict):
        return None, f"arguments parsed as {type(decoded).__name__}, not an object"
    for s in _strings(decoded):
        for token in _SERVING_TOKENS:
            if token in s:
                return None, f"arguments carry the serving-stack token {token!r}"
    if not _all_text(decoded):
        return None, f"arguments hold a string {NOT_UNICODE}"
    return decoded, None


# A tool call written into a text channel instead of the tool-call field.
_LEAK_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("special_token", re.compile(r"<\|tool_call")),
    ("hermes_json", re.compile(r"<tool_call>\s*\{")),
    ("xml_function", re.compile(r"<tool_call>.*?<function=", re.S)),
    ("function_calls_xml", re.compile(r"<function_calls>.*?<invoke\b", re.S)),
)

# Shape "bare_json", checked after the envelopes above (so a call inside one keeps that envelope's shape): a JSON
# object written as text whose top-level "name" is one of the agent's tools and which has "parameters" or "arguments",
# its keys in any order. That is Llama 3.3's own call format, which the Gateway's tool parser returns as content when it
# cannot read it (stream-integrity probe, 19 Oct 2026: 10 of 191 non-streamed Llama replies); an object naming
# anything else is the model's JSON about the domain, not a call (confirmation review of 1b7b7ec). The keys are read
# tolerantly, since the calls written as text are often not JSON inside their values ([[5 - (3 + 2)]]). The scan is
# bounded: at most LEAK_SCAN_OBJECTS object starts, and LEAK_SCAN_CHARS characters stepped over, per channel.
TOOL_NAMES = frozenset(spec.name for spec in TOOLS)  # the frozen tool set's names (sparql_query, submit_report, ...)
LEAK_SCAN_OBJECTS = 1_000
LEAK_SCAN_CHARS = 1_000_000
_OBJECT_START = re.compile(r'\{\s*"')
_SPACE = re.compile(r"\s*")
_VALUE_STOP = re.compile(r'["{}\[\],]')


class _Budget:
    def __init__(self, chars: int):
        self.left = chars

    def spend(self, n: int) -> bool:
        self.left -= n
        return self.left > 0


def _skip_value(text: str, i: int, budget: _Budget) -> int | None:
    """The index of the ',' or '}' that ends the value starting at ``i`` (outside strings, at depth 0), or None when
    the text or the budget ends first. Brackets and braces are balanced; what lies between them need not be JSON."""
    depth = 0
    while True:
        m = _VALUE_STOP.search(text, i)
        if m is None or not budget.spend(m.start() + 1 - i):
            return None
        c, i = m.group(), m.start()
        if c == '"':
            try:
                _, end = json.decoder.scanstring(text, i + 1)
            except ValueError:
                return None
            if not budget.spend(end - i):
                return None
            i = end
            continue
        if c in "{[":
            depth += 1
        elif depth:
            depth -= 1 if c in "}]" else 0
        elif c in ",}":
            return i
        i += 1


def _top_level_keys(text: str, start: int, budget: _Budget) -> dict[str, str | None]:
    """The top-level keys of the object at ``text[start]`` (a '{'), each with its value when that is a string. Reading
    stops, keeping what it found, at the object's end, at anything that cannot continue an object, or when the budget
    runs out."""
    keys: dict[str, str | None] = {}
    i = start + 1
    while budget.left > 0:
        i = _SPACE.match(text, i).end()
        if i >= len(text) or text[i] != '"':
            break
        before = i
        try:
            key, i = json.decoder.scanstring(text, i + 1)
        except ValueError:
            break
        i = _SPACE.match(text, i).end()
        if i >= len(text) or text[i] != ":":
            break
        i = _SPACE.match(text, i + 1).end()
        value: str | None = None
        if i < len(text) and text[i] == '"':
            try:
                value, i = json.decoder.scanstring(text, i + 1)
            except ValueError:
                keys[key] = None
                break
            i = _SPACE.match(text, i).end()
        else:
            end = _skip_value(text, i, budget)
            if end is None:
                keys[key] = None
                break
            i = end
        keys[key] = value
        if not budget.spend(i - before) or i >= len(text) or text[i] != ",":
            break
        i += 1
    return keys


def _scan_json_tool_call(text: str, tool_names: frozenset[str] = TOOL_NAMES) -> tuple[int | None, bool]:
    """(the offset of the first JSON object in ``text`` that is a tool call written as text, or None; whether the scan
    ended within its bounds, so that None means there is none)."""
    budget = _Budget(LEAK_SCAN_CHARS)
    for n, m in enumerate(_OBJECT_START.finditer(text)):
        if n >= LEAK_SCAN_OBJECTS or budget.left <= 0:
            return None, False
        keys = _top_level_keys(text, m.start(), budget)
        if keys.get("name") in tool_names and ("parameters" in keys or "arguments" in keys):
            return m.start(), True
    return None, budget.left > 0


def find_json_tool_call(text: str, tool_names: frozenset[str] = TOOL_NAMES) -> int | None:
    """The offset of the first JSON object in ``text`` that is a tool call written as text (shape "bare_json"), or
    None, within the scan's bounds."""
    return _scan_json_tool_call(text, tool_names)[0]


def detect_leaked_tool_call(channels: dict[str, str | None]) -> dict | None:
    """The first tool-call envelope found in any text channel, or None. Detection only."""
    for channel, text in channels.items():
        if not isinstance(text, str) or not text:
            continue
        found = None
        for shape, pattern in _LEAK_PATTERNS:
            m = pattern.search(text)
            if m:
                found = (shape, m.start())
                break
        if found is None:
            offset = find_json_tool_call(text)
            found = None if offset is None else ("bare_json", offset)
        if found is not None:
            shape, offset = found
            return {"shape": shape, "channel": channel, "offset": offset, "excerpt": text[offset: offset + 300]}
    return None


def reasoning_text(native: dict | None) -> str | None:
    """An OpenAI-wire message's reasoning channel, under whichever name the server sent it (parse_openai_response
    reads it so too); None when it has none."""
    text = (native or {}).get("reasoning_content") or (native or {}).get("reasoning")
    return text if isinstance(text, str) and text else None


# A call written into the reasoning channel, run as if parsed under a model's reasoning-call rule (design ruling on
# design log, 21 Oct 00:09: Qwen3.6, exploratory, whose calls the Gateway's reasoning parser keeps when the model writes them
# before ending its thinking; a per-model setting, scripts/run_pilot.py MODELS). "reasoning_xml_call_v1": a turn with no
# parsed call whose reasoning holds exactly one well-formed call in the model's own format to a tool the request
# offered, with valid arguments, runs that call (loop.py); anything else is the leaked_tool_call failure as before.
# * Exactly one: the reasoning holds one "<tool_call>" and one "<function=", and no other call envelope (the special
#   token, the <function_calls> form, a bare JSON call); a drafted call and then the call are two. The bare JSON scan
#   is bounded (LEAK_SCAN_OBJECTS, LEAK_SCAN_CHARS): a reasoning it cannot scan to its end is not taken (scan_bound).
# * Well formed: <tool_call>, <function=NAME>, then only parameters <parameter=KEY>TEXT</parameter> with white space
#   between them, </function> and </tool_call>; a parameter's text holds no other tag of the format. As the server
#   reads the model's parsed calls (6,354 of Qwen3.6's 6,356 parsed queries in the UniProt dev natural records neither
#   start nor end with a newline, the other two end with one), one newline after the opening tag and one before the
#   closing tag are dropped.
# * Declared: NAME is exactly the name of a tool the request offered.
# * Valid arguments, by the tool's declared schema: every parameter declared, none twice, every required one present;
#   a string parameter is its text, an array parameter (submit_report's rows) what json.loads reads from its text; the
#   text "null" (the parser's null) is a value of neither; each value then matches the parameter's schema (the JSON
#   Schema the tool declares, _schema_problem: rows an array of rows, each an array of IRI or literal objects, null,
#   numbers or booleans, so a bare-string cell is invalid); every string valid Unicode text with no serving-stack token
#   (as parse_tool_arguments). What the tool then says of a valid query or report is its own, as for any call.
REASONING_CALL_RULES = ("reasoning_xml_call_v1",)
_XML_CALL = re.compile(r"<tool_call>\s*<function=([^>\n]*)>(.*?)</function>\s*</tool_call>", re.S)
_XML_PARAMETER = re.compile(r"\s*<parameter=([^>\n]*)>(.*?)</parameter>", re.S)
_XML_TAGS = ("<tool_call>", "</tool_call>", "<function=", "</function>", "<parameter=")
_OTHER_ENVELOPES = tuple(p for shape, p in _LEAK_PATTERNS if shape not in ("hermes_json", "xml_function"))


def _not_taken(reason: str, detail: str) -> dict:
    return {"taken": False, "reason": reason, "detail": detail}


_JSON_TYPES: dict[str, Callable[[Any], bool]] = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "number": lambda v: (isinstance(v, int) and not isinstance(v, bool)) or (isinstance(v, float) and math.isfinite(v)),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def _json_number(text: str) -> float:
    """A JSON float, refused when it is no finite number (json.loads reads 1e999 as infinity)."""
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"{text[:40]} is no finite number")
    return value


def _no_constant(name: str) -> Any:
    raise ValueError(f"{name} is no JSON value")  # json.loads would read NaN, Infinity and -Infinity


def _schema_problem(value: Any, schema: dict, where: str) -> str | None:
    """Where ``value`` first breaks ``schema``, or None: the part of JSON Schema the tools declare (tools.py: type,
    const, properties, required, items, anyOf; a key no "properties" names is allowed, as JSON Schema allows it). A
    schema word outside that part is not read past: the value is refused."""
    unread = set(schema) - {"type", "const", "properties", "required", "items", "anyOf", "description"}
    if unread:
        return f"{where}: the schema uses {sorted(unread)}, which the rule does not read"
    if "anyOf" in schema:
        if all(_schema_problem(value, option, where) is not None for option in schema["anyOf"]):
            return f"{where} is none of the forms the schema allows"
        return None
    kind = schema.get("type")
    if kind is not None and not (kind in _JSON_TYPES and _JSON_TYPES[kind](value)):
        return f"{where} is not a JSON {kind}"
    if "const" in schema and not (type(value) is type(schema["const"]) and value == schema["const"]):
        return f"{where} is not {schema['const']!r}"
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                return f"{where} has no {key!r}"
        for key, sub in schema.get("properties", {}).items():
            if key in value and (problem := _schema_problem(value[key], sub, f"{where}.{key}")) is not None:
                return problem
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            if (problem := _schema_problem(item, schema["items"], f"{where}[{i}]")) is not None:
                return problem
    return None


def reasoning_tool_call(text: str | None, tools: tuple[ToolSpec, ...]) -> dict:
    """Whether the reasoning ``text`` holds the one call "reasoning_xml_call_v1" runs (REASONING_CALL_RULES) of the
    offered ``tools``: {"taken": True, "name", "arguments", "start", "end"} (the call's span in ``text``), or {"taken":
    False, "reason", "detail"}, the reason one of no_call, several_calls, scan_bound, malformed_call, undeclared_tool
    and invalid_arguments."""
    if not isinstance(text, str) or "<tool_call>" not in text or "<function=" not in text:
        return _not_taken("no_call", "the reasoning holds no <tool_call> with a <function=")
    if text.count("<tool_call>") > 1 or text.count("<function=") > 1:
        return _not_taken("several_calls", f"the reasoning holds {text.count('<tool_call>')} <tool_call> and "
                                           f"{text.count('<function=')} <function= openers")
    m = _XML_CALL.search(text)
    if m is None:
        return _not_taken("malformed_call", "no <tool_call><function=...>...</function></tool_call> in the reasoning")
    rest = text[:m.start()] + text[m.end():]
    bare, scanned = _scan_json_tool_call(rest)
    if any(p.search(rest) for p in _OTHER_ENVELOPES) or bare is not None:
        return _not_taken("several_calls", "another call envelope stands beside the call")
    if not scanned:
        return _not_taken("scan_bound", "the reasoning around the call is longer than the scan for another call reads")
    name, body = m.group(1), m.group(2)
    texts: list[tuple[str, str]] = []
    at = 0
    while (p := _XML_PARAMETER.match(body, at)) is not None:
        key, value = p.group(1), p.group(2)
        if any(tag in value for tag in _XML_TAGS):
            return _not_taken("malformed_call", f"parameter {key[:80]!r} holds another tag of the call format")
        value = value[1:] if value.startswith("\n") else value
        texts.append((key, value[:-1] if value.endswith("\n") else value))
        at = p.end()
    if body[at:].strip():
        return _not_taken("malformed_call", "the call holds text that is no parameter")
    spec = next((t for t in tools if t.name == name), None)
    if spec is None:
        return _not_taken("undeclared_tool", f"{name[:80]!r} is not a tool the request offered")
    try:
        arguments = _reasoning_arguments(name, spec, texts)
    except RecursionError:  # nested deeper than Python's JSON reader and these checks walk (External review of e5a8ebd)
        return _not_taken("invalid_arguments", "the arguments are nested deeper than the harness reads")
    if isinstance(arguments, str):
        return _not_taken("invalid_arguments", arguments)
    return {"taken": True, "name": name, "arguments": arguments, "start": m.start(), "end": m.end()}


def _reasoning_arguments(name: str, spec: ToolSpec, texts: list[tuple[str, str]]) -> dict | str:
    """The arguments of a reasoning call's parameter texts, read and checked as reasoning_tool_call says, or why they
    are invalid."""
    properties = spec.input_schema.get("properties", {})
    arguments: dict[str, Any] = {}
    for key, value in texts:
        if key not in properties:
            return f"{name} has no parameter {key[:80]!r}"
        if key in arguments:
            return f"parameter {key!r} is given twice"
        kind = properties[key].get("type")
        if value.strip().lower() == "null":
            return f"parameter {key!r} is null"
        if kind == "string":
            arguments[key] = value
        elif kind == "array":
            # strict JSON: no NaN or infinity anywhere in it (External review of 3903ded); an integer longer than Python
            # reads (sys.get_int_max_str_digits) is refused here as parse_tool_arguments refuses it in a parsed call
            try:
                decoded = json.loads(value, parse_constant=_no_constant, parse_float=_json_number)
            except ValueError as exc:
                return f"parameter {key!r} is no JSON: {getattr(exc, 'msg', exc)}"
            if not isinstance(decoded, list):
                return f"parameter {key!r} is a JSON {type(decoded).__name__}, not an array"
            arguments[key] = decoded
        else:
            return f"parameter {key!r} has schema type {kind!r}, which the rule does not read"
    missing = [k for k in spec.input_schema.get("required", []) if k not in arguments]
    if missing:
        return f"{name} needs {', '.join(missing)}"
    if (problem := _schema_problem(arguments, spec.input_schema, "arguments")) is not None:
        return f"the arguments do not match {name}'s schema: {problem[:300]}"
    if not _all_text(arguments):
        return f"the arguments hold a string {NOT_UNICODE}"
    for s in _strings(arguments):
        for token in _SERVING_TOKENS:
            if token in s:
                return f"the arguments carry the serving-stack token {token!r}"
    return arguments


# --- provider errors ------------------------------------------------------------

# Context overflow (design ruling, design log): the conversation has outgrown the model's context window.
# The attempt ends ``capped:context`` (loop.py), an agent outcome like a turn cap: scored wrong, never
# re-run, never infrastructure or provider_error. It is read from the provider's error by this table,
# the one place the wording is kept: an HTTP 400 (or 413, a request body the server will not take)
# whose message matches a row, or whose error code is in CONTEXT_OVERFLOW_CODES, is an overflow. Each
# row is (id, an example of the provider's wording, pattern); the examples are tested against their row.
# A reply whose stop reason is in CONTEXT_STOPS is the same outcome reported by the model's turn.
CONTEXT_OVERFLOW_PATTERNS: tuple[tuple[str, str, re.Pattern], ...] = (
    ("prompt_too_long",  # Anthropic invalid_request_error
     "prompt is too long: 1000001 tokens > 1000000 maximum",
     re.compile(r"prompt is too long", re.IGNORECASE)),
    ("maximum_context_length",  # OpenAI and vLLM 400
     "This model's maximum context length is 131072 tokens. However, your request has 140211 input tokens. "
     "Please reduce the length of the input messages.",
     re.compile(r"maximum context length", re.IGNORECASE)),
    ("maximum_model_length",  # vLLM 400, older wording
     "The decoder prompt (length 40000) is longer than the maximum model length of 32768.",
     re.compile(r"maximum model length|max_model_len", re.IGNORECASE)),
    ("context_limit",  # Anthropic invalid_request_error: input plus max_tokens over the window
     "input length and `max_tokens` exceed context limit: 188240 + 21333 > 200000, decrease input length or "
     "`max_tokens` and try again",
     re.compile(r"context (?:limit|window|length)|context_length_exceeded", re.IGNORECASE)),
)
CONTEXT_OVERFLOW_CODES = frozenset({"context_length_exceeded"})  # OpenAI's error code
CONTEXT_OVERFLOW_STATUSES = frozenset({400, 413})


def context_overflow_match(status: int | None, code: str | None, message: str) -> str | None:
    """The id of the CONTEXT_OVERFLOW_PATTERNS row (or "code" / "http_413") that makes this provider
    error a context overflow, or None."""
    if status not in CONTEXT_OVERFLOW_STATUSES:
        return None
    if code in CONTEXT_OVERFLOW_CODES:
        return "code"
    for row_id, _, pattern in CONTEXT_OVERFLOW_PATTERNS:
        if pattern.search(message):
            return row_id
    return "http_413" if status == 413 else None


@dataclass(frozen=True)
class ErrorInfo:
    # rate_limit, server, timeout, connection, conflict, context_overflow, auth, not_found, bad_request
    error_class: str
    transient: bool
    status: int | None
    code: str | None
    message: str
    retry_after_s: float | None = None

    @property
    def failure_type(self) -> str:
        return "context_overflow" if self.error_class == "context_overflow" else "provider_error"


def _error_code(exc: BaseException) -> str | None:
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code:
        return code
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        for holder in (body, body.get("error")):
            if isinstance(holder, dict):
                found = holder.get("code") or holder.get("type")
                if isinstance(found, str) and found:
                    return found
    return None


def _retry_after(exc: BaseException) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None:
        return None
    try:
        ms = headers.get("retry-after-ms")
        if ms is not None:
            return float(ms) / 1000.0
        s = headers.get("retry-after")
        return float(s) if s is not None else None
    except (TypeError, ValueError):
        return None


def _is_sdk_exception(exc: BaseException) -> bool:
    return any((k.__module__ or "").split(".", 1)[0] in ("openai", "anthropic") for k in type(exc).__mro__)


def classify_exception(exc: BaseException) -> ErrorInfo | None:
    """Classify an SDK exception by status code and class, or None if it is not one."""
    if not _is_sdk_exception(exc):
        return None
    message = str(exc)[:ERROR_MESSAGE_LIMIT]
    mro = {k.__name__ for k in type(exc).__mro__}
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int) or isinstance(status, bool):
        status = None
    code = _error_code(exc)
    if status is None:
        if "APITimeoutError" in mro:
            return ErrorInfo("timeout", True, None, code, message)
        if "APIConnectionError" in mro:
            return ErrorInfo("connection", True, None, code, message)
        return None
    retry_after = _retry_after(exc)
    if status == 429:
        return ErrorInfo("rate_limit", True, status, code, message, retry_after)
    if status == 408:
        return ErrorInfo("timeout", True, status, code, message, retry_after)
    if status == 409:
        return ErrorInfo("conflict", True, status, code, message, retry_after)
    if status >= 500:
        return ErrorInfo("server", True, status, code, message, retry_after)
    if context_overflow_match(status, code, message) is not None:
        return ErrorInfo("context_overflow", False, status, code, message)
    if status in (401, 403):
        return ErrorInfo("auth", False, status, code, message)
    if status == 404:
        return ErrorInfo("not_found", False, status, code, message)
    return ErrorInfo("bad_request", False, status, code, message)


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 4
    base_delay_s: float = 2.0
    max_delay_s: float = 60.0
    jitter: float = 0.25  # +/- fraction of the delay

    def delay(self, retry: int, retry_after_s: float | None) -> float:
        base = min(self.base_delay_s * (2 ** retry), self.max_delay_s)
        if retry_after_s is not None and retry_after_s > 0:
            base = min(max(base, retry_after_s), self.max_delay_s)
        return max(0.0, base * (1 + random.uniform(-self.jitter, self.jitter)))


class _RetryingRoute:
    """Shared request/retry machinery for both routes."""

    provider: str
    model: str

    def __init__(self, *, request_timeout_s: float, retry: RetryPolicy,
                 sleep: Callable[[float], None], clock: Callable[[], float]):
        self.request_timeout_s = request_timeout_s
        self.retry = retry
        self._sleep = sleep
        self._clock = clock

    def _call(self, send: Callable[[float], Any], time_left_s: float | None,
              classify: Callable[[BaseException], ErrorInfo | None] = classify_exception):
        """-> (response, latency of the request that succeeded, retries, retry_errors, step wall).

        A stream guard's rejection (``_read_stream``) is one more transient error here, in the same retry
        budget. Its entry in retry_errors carries the guard's evidence (``stream_guard``), and a step that
        fails carries the evidence of every rejection in it, in order, in its error's details: the failed
        step reaches no turn's retry_errors, and the attempt counts its guards' firings from both."""
        start = self._clock()
        deadline = None if time_left_s is None else start + time_left_s
        retries = 0
        errors: list[dict] = []

        def failed(error: ProviderError) -> ProviderError:
            guards = [e["stream_guard"] for e in errors if "stream_guard" in e]
            if guards:
                error.details["stream_guard"] = guards
            return error

        while True:
            timeout = self.request_timeout_s
            clipped = False  # the request's time-out was cut to the attempt's deadline
            if deadline is not None:
                remaining = deadline - self._clock()
                if remaining <= 0:
                    if errors:  # the provider's errors and their backoff used the budget up
                        last = errors[-1]
                        raise failed(ProviderError(
                            "infrastructure", f"the provider failed ({last['error_class']}) and the wall-clock "
                            f"budget ran out while retrying: {last['message'][:300]}",
                            error_class=last["error_class"], status=last["status"], code=last["code"],
                            retries=retries, transient=True))
                    raise ProviderError("budget_wall", "the wall-clock budget ran out while retrying the provider",
                                        error_class="budget", retries=retries)
                clipped = remaining < timeout
                timeout = min(timeout, remaining)
            t0 = self._clock()
            try:
                response = send(timeout)
            except Exception as exc:
                info = classify(exc)
                if info is None:
                    raise
                entry = {"error_class": info.error_class, "status": info.status, "code": info.code,
                         "message": info.message[:500]}
                guard = getattr(exc, "stream_guard", None)  # a stream guard's rejection (_read_stream)
                if isinstance(guard, dict):
                    entry["stream_guard"] = guard
                errors.append(entry)
                # The attempt's own deadline cut off a call the provider was answering normally (a client
                # time-out on a request clipped to the deadline, no earlier error in this step): the agent
                # ran out of time, whatever the retry policy. Checked first, so max_retries cannot turn it
                # into an outage (runner review).
                if clipped and info.error_class == "timeout" and info.status is None and len(errors) == 1:
                    raise ProviderError("budget_wall", f"the wall-clock budget ran out while waiting for the "
                                        f"provider: {info.message[:300]}", error_class="budget",
                                        status=info.status, code=info.code, retries=retries) from exc
                if not info.transient or retries >= self.retry.max_retries:
                    # overload or an outage that outlasted the retries is the provider's failure, not the agent's
                    failure_type = "infrastructure" if info.transient else info.failure_type
                    raise failed(ProviderError(failure_type, info.message, error_class=info.error_class,
                                               status=info.status, code=info.code, retries=retries,
                                               transient=info.transient)) from exc
                wait = self.retry.delay(retries, info.retry_after_s)
                if deadline is not None and self._clock() + wait >= deadline:  # the provider failed
                    raise failed(ProviderError(
                        "infrastructure", f"the provider failed ({info.error_class}) and the wall-clock budget "
                        f"left no time to retry: {info.message[:300]}", error_class=info.error_class,
                        status=info.status, code=info.code, retries=retries, transient=True)) from exc
                self._sleep(wait)
                retries += 1
                continue
            latency = self._clock() - t0
            return response, latency, retries, errors, self._clock() - start


def _public_url(url: str | None) -> str | None:
    """The URL without credentials, query or fragment."""
    if not url:
        return url
    parts = urlsplit(url)
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def _sdk_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # noqa: BLE001 - provenance only
        return None


def _to_dict(obj: Any) -> dict:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        return to_dict(mode="json")
    return dict(obj)


# --- Anthropic ------------------------------------------------------------------


class AnthropicRoute(_RetryingRoute):
    """Anthropic Messages API with tools.

    Defaults chosen for ``claude-sonnet-5-5``: adaptive thinking and the effort
    level are sent explicitly so the record pins them; ``temperature`` is not
    sent unless given (the model rejects non-default sampling values, and the
    1.x SDK has no ``temperature`` argument, so a given value goes in
    ``extra_body``); ``tool_choice`` stays ``auto`` (forced tool choice is a 400
    on this model); no server-side refusal fallback is requested, because a
    fallback would switch the model under test mid-attempt. The base URL is
    passed explicitly so an ambient ``ANTHROPIC_BASE_URL`` cannot redirect the
    route without the record showing it.

    Prompt caching (``prompt_cache``, on by default) uses two breakpoints with
    the same lifetime (``cache_ttl``, "1h" by default, or "5m"): an explicit one
    on the system prompt, which renders after the tools, so tools + system form
    one shared prefix that every attempt with the same row cap and time-out
    reads, whatever its question; and top-level automatic caching for the
    growing conversation. Nothing that varies between attempts is placed before
    the first breakpoint: the question and everything after it are messages.
    ``auto_cache=False`` drops the automatic breakpoint, so only the static
    system prefix is cached (the closed-book probes: one call, the question is
    the item and is never cached); the route's settings then name one breakpoint.

    A request with no tools (the probes) sends neither ``tools`` nor
    ``tool_choice``; the agent always sends its three tools, unchanged.
    """

    provider = "anthropic"
    seedable = False  # sampling cannot be seeded (recorded in the route settings)
    DEFAULT_BASE_URL = "https://api.anthropic.com"

    def __init__(self, model: str = "claude-sonnet-5-5", *, max_tokens: int = 16000,
                 temperature: float | None = None, effort: str | None = "high",
                 thinking: dict | None = None, prompt_cache: bool = True, cache_ttl: str = "1h",
                 auto_cache: bool = True, base_url: str = DEFAULT_BASE_URL, api_key_env: str = "ANTHROPIC_API_KEY",
                 request_timeout_s: float = 600.0, retry: RetryPolicy = RetryPolicy(),
                 client: Any = None, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic):
        super().__init__(request_timeout_s=request_timeout_s, retry=retry, sleep=sleep, clock=clock)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.effort = effort
        self.thinking = dict(thinking) if thinking is not None else {"type": "adaptive"}
        if cache_ttl not in ("5m", "1h"):
            raise ValueError(f"cache_ttl must be '5m' or '1h', got {cache_ttl!r}")
        self.prompt_cache = prompt_cache
        self.cache_ttl = cache_ttl
        self.auto_cache = auto_cache
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.__api_key = os.environ.get(api_key_env) or None
        self._client = client

    def __repr__(self) -> str:
        return f"AnthropicRoute(model={self.model!r})"

    def _ensure_client(self):
        if self._client is None:
            if not self.__api_key:
                raise ProviderError("provider_error", f"{self.api_key_env} is not set", error_class="auth")
            import anthropic

            self._client = anthropic.Anthropic(api_key=self.__api_key, base_url=self.base_url,
                                               max_retries=0, timeout=self.request_timeout_s)
        return self._client

    def route_info(self) -> dict:
        return {
            "provider": self.provider,
            "base_url": _public_url(self.base_url),
            "model": self.model,
            "sdk": {"name": "anthropic", "version": _sdk_version("anthropic")},
            "settings": {
                "default_max_tokens": self.max_tokens,  # used only when the loop sends none
                "temperature": self.temperature,
                "effort": self.effort,
                "thinking": self.thinking,
                "prompt_cache": self.prompt_cache,
                "cache_ttl": self.cache_ttl if self.prompt_cache else None,
                "cache_breakpoints": self.cache_breakpoints(),
                "tool_choice": "auto",
                "server_side_fallbacks": False,
                "seedable": False,  # the Messages API takes no sampling seed: retries cannot be frozen here
                "streaming": False,
                "request_timeout_s": self.request_timeout_s,
                "retry": asdict(self.retry),
            },
        }

    def cache_breakpoints(self) -> list[str]:
        if not self.prompt_cache:
            return []
        return ["system", "automatic"] if self.auto_cache else ["system"]

    def cache_control(self) -> dict:
        return {"type": "ephemeral", "ttl": "1h"} if self.cache_ttl == "1h" else {"type": "ephemeral"}

    def system_blocks(self, system: str) -> str | list[dict]:
        """The system prompt, carrying the shared-prefix breakpoint when caching is on."""
        if not self.prompt_cache:
            return system
        return [{"type": "text", "text": system, "cache_control": self.cache_control()}]

    def request(self, system: str, transcript: list[dict], tools: tuple[ToolSpec, ...],
                max_tokens: int | None) -> dict:
        """The keyword arguments sent to ``messages.create`` (without the time-out)."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "system": self.system_blocks(system),
            "messages": to_anthropic_messages(transcript),
        }
        if tools:  # a tool-free probe (scripts/refusal_screen.py) sends no tool fields at all
            kwargs["tools"] = anthropic_tools(tools)
        if self.thinking:
            kwargs["thinking"] = self.thinking
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        if self.prompt_cache and self.auto_cache:
            kwargs["cache_control"] = self.cache_control()
        if self.temperature is not None:
            kwargs["extra_body"] = {"temperature": self.temperature}
        return kwargs

    def step(self, system: str, transcript: list[dict], tools: tuple[ToolSpec, ...] = TOOLS, *,
             max_tokens: int | None = None, time_left_s: float | None = None) -> StepResult:
        client = self._ensure_client()
        kwargs = self.request(system, transcript, tools, max_tokens)
        resp, latency, retries, errors, wall = self._call(
            lambda timeout: client.messages.create(**kwargs, timeout=timeout), time_left_s)
        step = parse_anthropic_response(resp, latency_s=latency, wall_s=wall, retries=retries, retry_errors=errors)
        step.max_tokens = kwargs["max_tokens"]
        step.first_byte_s = latency  # a non-streamed reply arrives whole
        return step


def parse_anthropic_response(resp: Any, *, latency_s: float = 0.0, wall_s: float = 0.0, retries: int = 0,
                             retry_errors: list[dict] | None = None) -> StepResult:
    blocks = [_to_dict(b) for b in (getattr(resp, "content", None) or [])]
    text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    calls = []
    for b in blocks:
        if b.get("type") != "tool_use":
            continue
        call = {"id": b.get("id"), "name": b.get("name"), "arguments": b.get("input")}
        if not isinstance(call["arguments"], dict):
            call["arguments"] = None
            call["error"] = "tool input is not an object"
        elif not _all_text(call["arguments"]):
            call["arguments"] = None
            call["error"] = f"tool input holds a string {NOT_UNICODE}"
        calls.append(call)
    stop_reason = getattr(resp, "stop_reason", None)
    refusal = None
    if stop_reason in REFUSAL_STOPS:
        details = _to_dict(getattr(resp, "stop_details", None))
        refusal = {"category": details.get("category"), "explanation": details.get("explanation")}
    usage_raw = _to_dict(getattr(resp, "usage", None))
    return StepResult(
        text=text,
        tool_calls=calls,
        usage=normalise_anthropic_usage(usage_raw),
        usage_raw=usage_raw,
        latency_s=latency_s,
        wall_s=wall_s,
        stop_reason=stop_reason,
        response_id=getattr(resp, "id", None),
        model=getattr(resp, "model", None),
        native={"role": "assistant", "content": blocks},
        native_format=NATIVE_ANTHROPIC,
        retries=retries,
        retry_errors=list(retry_errors or []),
        leaked_tool_call=None if calls else detect_leaked_tool_call({"text": text}),
        refusal=refusal,
    )


# --- streaming (OpenAI-compatible) ---------------------------------------------------

# Sent with a streamed request unless ``stream_include_usage=False``: the protocol's way to ask for the reply's
# usage at the end of a stream (the Gateway sends usage on every chunk without it).
STREAM_OPTIONS = {"include_usage": True}

# Fields a non-streamed Gateway (vLLM) reply carries as null and its stream leaves out (gateway probe,
# 19 Oct 2026, GLM-5.3-Flash, DeepSeek-V4-Flash-0731 and V4.1-Flash; every Qwen turn of the lookup pilot has
# them too): the assembled reply carries them as null too, so a streamed turn's native message and usage are
# recorded exactly as the non-streamed turn's. ``reasoning`` is null for a model that did not reason (V4
# Flash 0731) and a string otherwise; a non-streamed reply has ``tool_calls`` only when it made a call.
STREAM_NULL_MESSAGE_FIELDS = ("refusal", "annotations", "audio", "function_call", "reasoning")
STREAM_NULL_USAGE_FIELDS = ("prompt_tokens_details",)
# Response fields a non-streamed Gateway reply carries, null in every capture, that its stream never sends
# (gateway probe/out/*-whole-*, all five streamed models; Llama's older server also leaves out a null
# system_fingerprint, and does not send prompt_text, ec_transfer_params, metrics or routed_experts either way):
# a streamed response lacks them. Nothing is lost (they are null) and nothing is invented; the agent records
# never hold them (they keep the message, usage, id, model and stop reason).
STREAM_OMITTED_RESPONSE_FIELDS = ("service_tier", "prompt_logprobs", "kv_transfer_params", "ec_transfer_params",
                                  "metrics")
STREAM_OMITTED_CHOICE_FIELDS = ("routed_experts",)
# Joins the distinct system fingerprints of one streamed reply when its chunks carried several (a reply spliced from
# servers that differ): a value no single fingerprint equals, which the runner's served-version check takes for a
# change by itself (scripts/run_pilot.py ServedModels). vLLM's fingerprints ("vllm-<version>-tp<n>-<hash>") hold none.
MIXED_FINGERPRINTS = " | "


class StreamAssembler:
    """Adds ``chat.completion.chunk`` objects (as dicts) up into the ``chat.completion`` they stream.

    Only choice 0 is kept (the route asks for one). Its string fields (content,
    refusal, a reasoning channel under whatever name the server uses) are
    concatenated in arrival order and are None when no non-empty fragment came,
    as in a non-streamed reply. Tool calls are merged by their ``index``: id, type
    and name as first sent, the argument fragments concatenated exactly as sent.
    Usage is the last non-null value of each key (nested details too) over every
    chunk that carries usage: the Gateway reports cumulative usage on each chunk and
    its credits on the finishing one only. A field a delta sends as null is null in the message unless a value
    came (Llama's ``reasoning_content``), and a tool-call fragment with no id, name
    or arguments makes no call (Llama sends one with its finish reason).
    ``empty_tool_calls`` gives a reply without calls ``tool_calls: []``, for a server
    whose non-streamed message always carries the field (the Gateway's Llama 3.3).

    Every other field the stream carries is kept, at the top level and in choice 0
    (a closed-book probe record stores the whole response): its last non-null value,
    or null when it only ever came as null (``system_fingerprint``: its distinct
    non-null values joined by " | " in arrival order when there were several); top-level fields in the order the chunks
    first sent them, choice fields in the order of the finishing chunk. A field the
    stream never sends is not invented (``STREAM_OMITTED_*``).
    """

    def __init__(self, *, empty_tool_calls: bool = False) -> None:
        self.empty_tool_calls = empty_tool_calls
        self.nulls: list[str] = []  # fields a delta sent as null, in first-seen order
        self.head: dict[str, Any] = {}  # every top-level field but choices, usage and object
        self.choice_fields: dict[str, Any] = {}  # every choice-0 field but delta, index and finish_reason
        self.choice_order: list[str] = []  # the finishing chunk's field order
        self.role: str | None = None
        self.strings: dict[str, list[str]] = {}  # field -> fragments, fields in first-seen order
        self.others: dict[str, Any] = {}  # any other delta field: its latest non-null value
        self.calls: dict[int, dict] = {}
        self.finish_reason: str | None = None
        self.usage: dict[str, Any] = {}
        self.final_usage = False  # a usage-only chunk (no choices) came, as the protocol ends an asked-for stream
        self.chunks = 0  # chunks added so far; a call keeps the index of the chunk its first fragment came in
        self.fingerprints: list[str] = []  # the distinct non-null system_fingerprint values, in arrival order

    @staticmethod
    def _keep(fields: dict, key: str, value: Any) -> None:
        if key not in fields or value is not None:
            fields[key] = value

    @classmethod
    def _merge(cls, fields: dict, values: dict) -> None:
        """Each usage count (nested details too) at its last non-null value: a later null never replaces a
        reported count."""
        for key, value in values.items():
            if isinstance(value, dict) and isinstance(fields.get(key), dict):
                cls._merge(fields[key], value)
            else:
                cls._keep(fields, key, dict(value) if isinstance(value, dict) else value)

    def add(self, chunk: dict) -> None:
        self.chunks += 1
        for key, value in chunk.items():
            if key not in ("choices", "usage", "object"):
                self._keep(self.head, key, value)
        fingerprint = chunk.get("system_fingerprint")
        if fingerprint is not None and fingerprint not in self.fingerprints:
            self.fingerprints.append(fingerprint)
        if isinstance(chunk.get("usage"), dict):
            self._merge(self.usage, chunk["usage"])
            if not chunk.get("choices"):
                self.final_usage = True
        for choice in chunk.get("choices") or []:
            if choice.get("index", 0) != 0:
                continue
            for key, value in choice.items():
                if key not in ("delta", "index", "finish_reason"):
                    self._keep(self.choice_fields, key, value)
            if choice.get("finish_reason") is not None:
                self.choice_order = [k for k in choice if k not in ("delta", "index", "finish_reason")]
            for key, value in (choice.get("delta") or {}).items():
                if value is None:
                    if key not in self.nulls:
                        self.nulls.append(key)
                    continue
                if key == "role":
                    self.role = self.role or value
                elif key == "tool_calls":
                    self._add_calls(value)
                elif isinstance(value, str):
                    self.strings.setdefault(key, []).append(value)
                else:
                    self.others[key] = value
            if choice.get("finish_reason") is not None:
                self.finish_reason = choice["finish_reason"]

    def _add_calls(self, deltas: list) -> None:
        for delta in deltas:
            index = delta.get("index")
            if not isinstance(index, int):  # a server that sends whole calls without an index
                index = len(self.calls)
            call = self.calls.setdefault(index, {"id": None, "type": None, "name": None, "arguments": [],
                                                 "chunk": self.chunks - 1})
            function = delta.get("function") or {}
            for key, value in (("id", delta.get("id")), ("type", delta.get("type")), ("name", function.get("name"))):
                if call[key] is None and value:
                    call[key] = value
            if isinstance(function.get("arguments"), str):
                call["arguments"].append(function["arguments"])

    def _kept_calls(self) -> list[dict]:
        """The calls the reply holds, in index order: a fragment with no id, name or arguments makes none."""
        return [c for _, c in sorted(self.calls.items()) if c["id"] or c["name"] or "".join(c["arguments"])]

    def calls_without_identity(self) -> list[dict]:
        """The narrow G1's evidence: each call of the reply (by its position) that has no id or no function
        name, with the index of the chunk its first fragment came in. Its arguments are not looked at."""
        return [{"call_index": i, "missing": [k for k in ("id", "name") if not c[k]], "chunk_index": c["chunk"]}
                for i, c in enumerate(self._kept_calls()) if not c["id"] or not c["name"]]

    def result(self) -> dict | None:
        """The assembled reply, or None when choice 0 never finished (the reply did not arrive whole)."""
        if self.finish_reason is None:
            return None
        message: dict[str, Any] = {"role": self.role or "assistant", "content": None,
                                   **{name: None for name in STREAM_NULL_MESSAGE_FIELDS}}
        for key, parts in self.strings.items():
            message[key] = "".join(parts) or None
        message.update({k: v for k, v in self.others.items() if k not in message})
        message.update({k: None for k in self.nulls if k not in message and k not in ("role", "tool_calls")})
        calls = [{"id": c["id"], "type": c["type"] or "function",
                  "function": {"name": c["name"], "arguments": "".join(c["arguments"])}}
                 for c in self._kept_calls()]
        if calls or self.empty_tool_calls:
            message["tool_calls"] = calls
        usage = None
        if self.usage:
            usage = {**self.usage, **{k: None for k in STREAM_NULL_USAGE_FIELDS if k not in self.usage}}
        order = [*self.choice_order, *(k for k in self.choice_fields if k not in self.choice_order)]
        choice = {"index": 0, "message": message, "finish_reason": self.finish_reason,
                  **{k: self.choice_fields[k] for k in order if k in self.choice_fields}}
        head = dict(self.head)
        if len(self.fingerprints) > 1:  # a reply spliced from servers that differ: all kept, so no check passes it
            head["system_fingerprint"] = MIXED_FINGERPRINTS.join(str(f) for f in self.fingerprints)
        return {"id": head.get("id"), "choices": [choice], "created": head.get("created"),
                "model": head.get("model"), "object": "chat.completion", "usage": usage,
                **{k: v for k, v in head.items() if k not in ("id", "created", "model")}}


def assemble_stream(chunks: list[dict], *, empty_tool_calls: bool = False) -> dict | None:
    reply = StreamAssembler(empty_tool_calls=empty_tool_calls)
    for chunk in chunks:
        reply.add(chunk)
    return reply.result()


# --- stream guards (stream-integrity probe, 19 Oct 2026; design ruling in the design log) ---------------------

STREAM_GUARDS = ("call_identity", "token_gap")  # G1 (narrow) and G2, as their evidence names them
# G2, the token gap: the probe's rule (results/stream-integrity-20261019, gap-check.json and scripts/gap_check.py).
TOKEN_GAP_MIN_RISE = 30
TOKEN_GAP_MAX_CHARS_PER_TOKEN = 0.5


def chunk_characters(chunk: dict) -> int:
    """The characters a chunk delivers, as the probe counted them: every string field of choice 0's delta but
    ``role`` (content, reasoning under any name, refusal), and each tool-call fragment's name and arguments (not
    its id or type)."""
    choice = (chunk.get("choices") or [None])[0]
    delta = (choice.get("delta") if isinstance(choice, dict) else None) or {}
    chars = sum(len(v) for k, v in delta.items() if isinstance(v, str) and k != "role")
    for call in delta.get("tool_calls") or []:
        function = (call.get("function") if isinstance(call, dict) else None) or {}
        chars += sum(len(function[k]) for k in ("name", "arguments") if isinstance(function.get(k), str))
    return chars


class TokenGapCounter:
    """G2 over a stream's chunks as they arrive: each chunk whose cumulative ``usage.completion_tokens`` rose by at
    least ``min_rise`` over the last chunk that carried the count, while the chunks since that one (this one
    included) delivered fewer than ``max_chars_per_token`` characters (``chunk_characters``) per token of the rise:
    tokens generated whose events never came. The characters of a chunk without the count go to the next chunk
    that has one, so a stream that reports usage only now and then is judged on everything it delivered
    (confirmation review of 1b7b7ec; the Gateway reports it on every chunk, and there this is the probe's rule). As
    in the probe, a chunk without choices (the usage-only chunk that ends a stream) is neither checked nor moves
    the counter, and the first count is compared with nothing. Chunks are numbered from 0 in the order added, as
    ``StreamAssembler`` numbers them."""

    def __init__(self, min_rise: int = TOKEN_GAP_MIN_RISE, max_chars_per_token: float = TOKEN_GAP_MAX_CHARS_PER_TOKEN):
        self.min_rise = min_rise
        self.max_chars_per_token = max_chars_per_token
        self.chunks = 0
        self.gaps: list[dict] = []  # {"chunk_index", "token_rise", "chars_delivered"}, in stream order
        self._last: int | float | None = None
        self._chars = 0  # characters delivered since the last chunk that carried the count

    def add(self, chunk: dict) -> None:
        index = self.chunks
        self.chunks += 1
        if not chunk.get("choices"):
            return
        self._chars += chunk_characters(chunk)
        usage = chunk.get("usage")
        tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
        if not isinstance(tokens, (int, float)) or isinstance(tokens, bool):
            return
        if self._last is not None:
            rise = tokens - self._last
            if rise >= self.min_rise and self._chars < self.max_chars_per_token * rise:
                self.gaps.append({"chunk_index": index, "token_rise": rise, "chars_delivered": self._chars})
        self._last, self._chars = tokens, 0


def token_gaps(chunks: list[dict], **thresholds: Any) -> list[dict]:
    counter = TokenGapCounter(**thresholds)
    for chunk in chunks:
        counter.add(chunk)
    return counter.gaps


def _stream_guard_error(fired: dict, request: Any) -> Exception:
    """The transient connection error a stream that did not finish raises, carrying the guards' evidence
    (``stream_guard``: per guard that fired, its first finding and how many it made)."""
    import openai

    parts = []
    if "call_identity" in fired:
        e = fired["call_identity"]
        parts.append(f"call_identity: tool call {e['call_index']} has no {' and no '.join(e['missing'])} (its first "
                     f"fragment in chunk {e['chunk_index']}; {e['calls']} such call(s))")
    if "token_gap" in fired:
        e = fired["token_gap"]
        parts.append(f"token_gap: chunk {e['chunk_index']} raised the completion tokens by {e['token_rise']} and "
                     f"delivered {e['chars_delivered']} characters ({e['gaps']} such chunk(s))")
    exc = openai.APIConnectionError(message="the stream lost events (stream guard " + "; ".join(parts) + ")",
                                    request=request)
    exc.stream_guard = fired
    return exc


def classify_stream_exception(exc: BaseException) -> ErrorInfo | None:
    """``classify_exception``, and an error event inside a stream: the server had answered 200 and failed
    mid-reply (the SDK raises it as a bare ``APIError`` with no status), a transient server error."""
    info = classify_exception(exc)
    if info is None and _is_sdk_exception(exc) and "APIError" in {k.__name__ for k in type(exc).__mro__}:
        return ErrorInfo("server", True, None, _error_code(exc), str(exc)[:ERROR_MESSAGE_LIMIT])
    return info


# --- OpenAI-compatible ------------------------------------------------------------

GATEWAY_DEFAULT_BASE_URL = "https://gateway.example/api/v1"


class OpenAICompatRoute(_RetryingRoute):
    """Any OpenAI-compatible chat-completions endpoint with tools (default: HPCNet LLM Gateway).

    ``temperature`` is not sent unless given, so the server's default for the
    model applies; the record shows which. ``extra_body`` passes server-specific
    options (for example vLLM ``chat_template_kwargs``) and is recorded.
    ``parallel_tool_calls`` is not sent unless given: False asks for at most one
    tool call per turn, for a model whose chat template takes only one (Llama
    3.3: "This model only supports single tool-calls at once!"). When given it is
    recorded in the route's settings; left unset, the settings are as before.
    ``streaming`` (default False) sends the same request as a stream and adds the
    chunks up into the reply a non-streamed request returns (module docstring);
    the settings record it as ``streaming`` and, only when on, ``stream_options``.
    A streamed request also sends ``stream_options`` ``{"include_usage": true}``,
    the protocol's way to ask for the usage at the end of a stream (a conforming
    endpoint sends none without it; the Gateway sends usage on every chunk anyway).
    ``stream_include_usage=False`` leaves ``stream: true`` the only field it adds,
    for a server known to report usage unasked. A stream that reports no usage at
    all leaves the counts None (unreported), never 0. ``stream_empty_tool_calls`` (recorded only
    when on) adds ``tool_calls: []`` to a streamed reply without calls, for a server
    whose non-streamed message always has the field (the Gateway's Llama 3.3).
    ``guard_call_identity`` and ``guard_token_gap`` (default off, streamed routes only)
    turn on the stream guards G1 (narrow) and G2 (module docstring); the settings
    record them as ``stream_guards`` only when one is on, so the settings of a route
    without them are as before.
    ``echo_reasoning`` (one of ECHO_REASONING_FIELDS; default None) sends each earlier
    turn's reasoning channel back under that wire name (to_openai_messages): a per-model
    setting for the Gateway mitigation A/B (results/gateway-mitigations-20261020), recorded
    only when set. (A call written into the reasoning channel is the loop's to run, under
    a model's reasoning-call rule: loop.py, REASONING_CALL_RULES.)
    """

    provider = "openai-compatible"
    seedable = True  # vLLM and OpenAI chat completions take a sampling seed

    def __init__(self, model: str = "Qwen/Qwen3.8-27B", *, base_url: str | None = None,
                 api_key_env: str = "LLM_GATEWAY_API_KEY", temperature: float | None = None,
                 max_tokens: int = 16000, extra_body: dict | None = None,
                 parallel_tool_calls: bool | None = None, seed: int | None = None, streaming: bool = False,
                 stream_include_usage: bool = True, stream_empty_tool_calls: bool = False,
                 guard_call_identity: bool = False, guard_token_gap: bool = False,
                 echo_reasoning: str | None = None,
                 request_timeout_s: float = 600.0, retry: RetryPolicy = RetryPolicy(),
                 client: Any = None, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic):
        super().__init__(request_timeout_s=request_timeout_s, retry=retry, sleep=sleep, clock=clock)
        if (guard_call_identity or guard_token_gap) and not streaming:
            raise ValueError("the stream guards read a stream: guard_call_identity and guard_token_gap need "
                             "streaming=True")
        if echo_reasoning is not None and echo_reasoning not in ECHO_REASONING_FIELDS:
            raise ValueError(f"echo_reasoning must be one of {', '.join(ECHO_REASONING_FIELDS)}, got "
                             f"{echo_reasoning!r}")
        self.echo_reasoning = echo_reasoning
        self.model = model
        self.streaming = bool(streaming)
        self.stream_include_usage = bool(stream_include_usage)
        self.stream_empty_tool_calls = bool(stream_empty_tool_calls)
        self.guard_call_identity = bool(guard_call_identity)
        self.guard_token_gap = bool(guard_token_gap)
        self.base_url = base_url or os.environ.get("LLM_GATEWAY_BASE_URL") or GATEWAY_DEFAULT_BASE_URL
        self.api_key_env = api_key_env
        self.__api_key = (os.environ.get(api_key_env) or "").strip() or None
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_body = dict(extra_body) if extra_body else None
        self.parallel_tool_calls = parallel_tool_calls
        self.seed = seed
        self._client = client

    def with_seed(self, seed: int) -> OpenAICompatRoute:
        """A copy of this route that sends ``seed`` with every request (the same client)."""
        route = copy.copy(self)
        route.seed = seed
        return route

    _REBUILD_OPTIONS = ("model", "base_url", "api_key_env", "temperature", "max_tokens", "extra_body",
                        "parallel_tool_calls", "seed", "streaming", "stream_include_usage", "stream_empty_tool_calls",
                        "guard_call_identity", "guard_token_gap", "echo_reasoning",
                        "request_timeout_s", "retry")

    def with_options(self, **overrides: Any) -> OpenAICompatRoute:
        """A new route with this one's options and ``overrides``, built through the constructor so every check runs
        again (a non-streamed route must also turn the stream guards off). The key is read from the environment again;
        this route is left as it was."""
        unknown = sorted(set(overrides) - set(self._REBUILD_OPTIONS))
        if unknown:
            raise ValueError(f"unknown route option(s) {', '.join(unknown)}; known: {', '.join(self._REBUILD_OPTIONS)}")
        options = {name: getattr(self, name) for name in self._REBUILD_OPTIONS}
        options.update(overrides)
        # the route's own class rebuilds itself (so routes are still built only by the runner's table, which
        # tests/test_agent_streaming.py checks)
        return type(self)(**options, client=self._client, sleep=self._sleep, clock=self._clock)

    def __repr__(self) -> str:
        return f"OpenAICompatRoute(model={self.model!r})"

    def _ensure_client(self):
        if self._client is None:
            if not self.__api_key:
                raise ProviderError("provider_error", f"{self.api_key_env} is not set", error_class="auth")
            import openai

            self._client = openai.OpenAI(api_key=self.__api_key, base_url=self.base_url,
                                         max_retries=0, timeout=self.request_timeout_s)
        return self._client

    def route_info(self) -> dict:
        settings = {
            "default_max_tokens": self.max_tokens,  # used only when the loop sends none
            "temperature": self.temperature,
            "extra_body": self.extra_body,
            "tool_choice": "auto",
            "streaming": self.streaming,
            "request_timeout_s": self.request_timeout_s,
            "retry": asdict(self.retry),
        }
        if self.parallel_tool_calls is not None:  # recorded only when sent, so earlier settings stay identical
            settings["parallel_tool_calls"] = self.parallel_tool_calls
        if self.seed is not None:  # likewise: a sampling seed (a retry arm's, harness/checkloop/online.py)
            settings["seed"] = self.seed
        if self.streaming:  # likewise: a non-streamed route's settings are as before
            settings["stream_options"] = dict(STREAM_OPTIONS) if self.stream_include_usage else None
            if self.stream_empty_tool_calls:
                settings["stream_empty_tool_calls"] = True
            if self.guard_call_identity or self.guard_token_gap:  # recorded only when one is on
                settings["stream_guards"] = {
                    "call_identity": self.guard_call_identity,
                    "token_gap": {"min_rise": TOKEN_GAP_MIN_RISE, "max_chars_per_token": TOKEN_GAP_MAX_CHARS_PER_TOKEN}
                    if self.guard_token_gap else None,
                }
        if self.echo_reasoning is not None:  # likewise: recorded only when set
            settings["echo_reasoning"] = self.echo_reasoning
        return {
            "provider": self.provider,
            "base_url": _public_url(self.base_url),
            "model": self.model,
            "sdk": {"name": "openai", "version": _sdk_version("openai")},
            "settings": settings,
        }

    def request(self, system: str, transcript: list[dict], tools: tuple[ToolSpec, ...],
                max_tokens: int | None) -> dict:
        kwargs: dict[str, Any] = {"model": self.model,
                                  "messages": to_openai_messages(system, transcript, self.echo_reasoning)}
        if tools:  # a request with no tools (the probes) sends no tool_choice either: servers refuse it alone
            kwargs["tools"] = openai_tools(tools)
            kwargs["tool_choice"] = "auto"
        kwargs["max_tokens"] = max_tokens or self.max_tokens
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.parallel_tool_calls is not None and tools:
            kwargs["parallel_tool_calls"] = self.parallel_tool_calls
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        if self.streaming:  # the only fields a streamed request adds
            kwargs["stream"] = True
            if self.stream_include_usage:
                kwargs["stream_options"] = dict(STREAM_OPTIONS)
        return kwargs

    def step(self, system: str, transcript: list[dict], tools: tuple[ToolSpec, ...] = TOOLS, *,
             max_tokens: int | None = None, time_left_s: float | None = None) -> StepResult:
        if self.echo_reasoning is None:  # exactly as before: the client, then the request, nothing recorded of it
            client = self._ensure_client()
            return self._step(self.request(system, transcript, tools, max_tokens), time_left_s, None, client)
        kwargs = self.request(system, transcript, tools, max_tokens)
        # what the echoed history was, for the replay to rebuild (agent-run/22): on the turn, or on a failure's details;
        # taken before the client is made, so a missing key's failure carries it too
        sent = messages_sha256(kwargs["messages"])
        try:
            return self._step(kwargs, time_left_s, sent)
        except StepFailure as failure:
            if sent is not None:  # the attempt's last request has no turn: its failure carries what was sent
                failure.details = {**failure.details, "sent_messages_sha256": sent}
            raise
        except HarnessAbort:
            raise
        except Exception as exc:
            # a reply an echoing route could not read: the failure the loop records for any such reply (failed:other
            # at stage provider, its message), carrying what was sent (External review of 5138781)
            raise StepFailure("other", f"{type(exc).__name__}: {exc}", {"sent_messages_sha256": sent}) from exc

    def _step(self, kwargs: dict, time_left_s: float | None, sent: str | None, client: Any = None) -> StepResult:
        """One request (``kwargs``), its retries and the reading of its reply (step)."""
        client = client if client is not None else self._ensure_client()
        if self.streaming:
            (resp, first_byte), latency, retries, errors, wall = self._call(
                lambda timeout: self._read_stream(client, kwargs, timeout), time_left_s,
                classify=classify_stream_exception)
        else:
            resp, latency, retries, errors, wall = self._call(
                lambda timeout: client.chat.completions.create(**kwargs, timeout=timeout), time_left_s)
            first_byte = latency  # a non-streamed reply arrives whole
        step = parse_openai_response(resp, latency_s=latency, wall_s=wall, retries=retries, retry_errors=errors)
        step.max_tokens = kwargs["max_tokens"]
        step.first_byte_s = first_byte
        step.sent_messages_sha256 = sent
        return step

    def _read_stream(self, client: Any, kwargs: dict, timeout: float) -> Any:
        """One streamed request read to its end: the ``chat.completion`` its chunks add up to.

        The server-sent events are read here, not through the SDK's stream iterator,
        because that iterator ends the same way on ``[DONE]`` and on a body that
        simply stops: a reply counts only when ``[DONE]`` arrived and choice 0 had a
        finish reason. A stream that ends before either, carries an event that is
        not JSON, or breaks is a connection error, and so is one that asked for usage
        and reported some but not in the usage-only chunk that ends such a stream (a
        stream with no usage at all is accepted: its counts are unreported); an error
        event is the server's (``classify_stream_exception``). All are retried.

        ``timeout`` bounds the whole request. The SDK's time-out bounds each read of
        a stream, not the whole of it, so a watchdog cuts the connection when the
        time is up, however quiet the stream is, and the clock is checked as each
        event arrives and at the end: a reply not complete in time ends with the
        client time-out a non-streamed request would have hit (``_call`` then tells
        the attempt's deadline from the provider's failure as before).

        The stream guards the route turns on are judged once the reply is complete,
        after those checks (a stream that did not finish is that error, whatever
        else it lost); a rejected reply is a connection error carrying every
        firing guard's evidence (``_stream_guard_error``). -> (the reply, the time
        to the body's first line).
        """
        import httpx2
        import openai
        from openai.types.chat import ChatCompletion

        t0 = self._clock()
        reply = StreamAssembler(empty_tool_calls=self.stream_empty_tool_calls)
        gaps = TokenGapCounter() if self.guard_token_gap else None
        done = False
        first_byte: float | None = None

        def timed(lines):
            nonlocal first_byte
            for line in lines:
                if first_byte is None:
                    first_byte = self._clock() - t0
                yield line

        with client.chat.completions.with_streaming_response.create(**kwargs, timeout=timeout) as response:
            request = getattr(response, "http_request", None)
            watchdog = StreamWatchdog(response, max(0.0, timeout - (self._clock() - t0))).start()
            try:
                for data in sse_data(timed(response.iter_lines())):
                    if data == "[DONE]":
                        # marked complete only in time, judged by one clock reading under the watchdog's lock
                        if not watchdog.complete(lambda: self._clock() - t0 <= timeout):
                            raise openai.APITimeoutError(request=request)
                        done = True
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError as exc:  # a garbled event: the reply did not arrive intact
                        raise openai.APIConnectionError(
                            message=f"the stream carried an event that is not JSON: {exc}", request=request) from exc
                    if isinstance(chunk, dict) and chunk.get("error"):  # the server failed after its 200
                        error = chunk["error"]
                        message = error.get("message") if isinstance(error, dict) else None
                        raise openai.APIError(message if isinstance(message, str) and message else
                                              "An error occurred during streaming", request, body=error)
                    if isinstance(chunk, dict):
                        reply.add(chunk)
                        if gaps is not None:
                            gaps.add(chunk)
                    if self._clock() - t0 > timeout:
                        raise openai.APITimeoutError(request=request)
            except openai.APIError:
                raise
            except Exception as exc:
                if watchdog.state == "cut":  # the watchdog cut a stream still running at its time-out
                    raise openai.APITimeoutError(request=request) from exc
                if isinstance(exc, httpx2.TimeoutException):  # one read waited the whole time-out
                    raise openai.APITimeoutError(request=request) from exc
                if isinstance(exc, (httpx2.TransportError, httpx2.StreamError, OSError)):
                    raise openai.APIConnectionError(request=request) from exc
                raise
            finally:
                watch = watchdog.stop()  # before the response's own cleanup; the watchdog's thread is over after it
        if watch == "cut":
            raise openai.APITimeoutError(request=request)
        assembled = reply.result()
        if done and assembled is not None and self.stream_include_usage and reply.usage and not reply.final_usage:
            # usage came on the chunks but not the usage-only chunk that ends a stream that asked for it
            raise openai.APIConnectionError(message="the stream ended before the reply was complete (no final "
                                                    "usage chunk)", request=request)
        if not done or assembled is None:
            missing = ("no [DONE]" if assembled is not None else "no finish reason" if done
                       else "no [DONE] and no finish reason")
            raise openai.APIConnectionError(message=f"the stream ended before the reply was complete ({missing})",
                                            request=request)
        fired: dict[str, dict] = {}
        headless = reply.calls_without_identity() if self.guard_call_identity else []
        if headless:
            fired["call_identity"] = {**headless[0], "calls": len(headless)}
        if gaps is not None and gaps.gaps:
            fired["token_gap"] = {**gaps.gaps[0], "gaps": len(gaps.gaps)}
        if fired:
            raise _stream_guard_error(fired, request)
        return ChatCompletion.construct(**assembled), first_byte  # as the SDK builds a non-streamed reply


def sse_data(lines: Any):
    """The ``data`` of each server-sent event in ``lines`` (one event per blank-line-ended block; comments and
    other fields skipped; several data lines joined by newlines, as the SSE format has it)."""
    data: list[str] = []
    for line in lines:
        if line == "":
            if data:
                yield "\n".join(data)
                data = []
            continue
        if line.startswith(":"):
            continue
        name, _, value = line.partition(":")
        if name == "data":
            data.append(value.removeprefix(" "))
    if data:
        yield "\n".join(data)


def _watchdog_timer(seconds: float, callback: Callable[[], Any]) -> Any:
    import threading

    timer = threading.Timer(seconds, callback)
    timer.daemon = True
    timer.name = "stream-watchdog"
    return timer


class StreamWatchdog:
    """Cuts a streamed response still being read when its time is up, without a race against the reply.

    One lock guards the state: ``reading`` until the reader marks the reply complete
    (``complete``, as ``[DONE]`` arrives: ``done``, or ``late`` past the deadline) or
    stops reading (``stop``: ``stopped``), or ``cut`` when the timer fired first. The timer cuts only a stream
    still being read, so a reply complete before the deadline is never turned into a
    time-out by a timer firing during cleanup; ``stop`` cancels the timer and waits,
    without a limit, for a cut already under way, so nothing is left running once
    the request returns. The wait is safe because the cut cannot block: it only marks
    the state and shuts the socket down (``_cut_stream``).
    """

    def __init__(self, response: Any, seconds: float):
        import threading

        self._lock = threading.Lock()
        self.state = "reading"
        self._response = response
        self._timer = _watchdog_timer(seconds, self._fire)

    def start(self) -> StreamWatchdog:
        self._timer.start()
        return self

    def _fire(self) -> bool:
        with self._lock:
            if self.state != "reading":
                return False
            self.state = "cut"
        _cut_stream(self._response)
        return True

    def complete(self, in_time: Callable[[], bool] = lambda: True) -> bool:
        """Mark the reply complete if ``in_time()``, read under the lock; False when the time ran out first
        (the timer cut the stream, or the reading says the deadline had passed: ``late``). The mark and the
        reading are one step, so nothing measured afterwards can undo a reply complete in time."""
        with self._lock:
            if self.state == "reading":
                self.state = "done" if in_time() else "late"
            return self.state == "done"

    def stop(self) -> str:
        """Stop watching and wait for a cut under way; the final state."""
        with self._lock:
            if self.state == "reading":
                self.state = "stopped"
            state = self.state
        self._timer.cancel()
        if self._timer.is_alive():
            self._timer.join()
        return state


def _cut_stream(response: Any) -> None:
    """Shut a streamed response's socket down, so a read waiting on a quiet stream returns at once.

    Only ``shutdown``, which returns at once, is called: never ``close`` (the reader closes the response
    itself as it leaves), so the watchdog cannot block and the reader may wait for it without a limit. Where
    no socket is reachable nothing is done here; the reader still stops at its next read, by the clock or by
    the read's own time-out."""
    import contextlib
    import socket

    with contextlib.suppress(Exception):
        stream = (getattr(getattr(response, "http_response", None), "extensions", None) or {}).get("network_stream")
        sock = stream.get_extra_info("socket") if stream is not None else None
        if sock is not None:
            sock.shutdown(socket.SHUT_RDWR)


def parse_openai_response(resp: Any, *, latency_s: float = 0.0, wall_s: float = 0.0, retries: int = 0,
                          retry_errors: list[dict] | None = None) -> StepResult:
    choices = getattr(resp, "choices", None) or []
    choice = choices[0] if choices else None
    message = getattr(choice, "message", None)
    native = _to_dict(message)
    text = native.get("content") or ""
    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False)
    calls = []
    for tc in native.get("tool_calls") or []:
        function = tc.get("function") if isinstance(tc, dict) else None
        if not isinstance(function, dict):
            calls.append({"id": tc.get("id") if isinstance(tc, dict) else None, "name": None, "arguments": None,
                          "error": "the tool call has no function payload"})
            continue
        raw = function.get("arguments")
        arguments, problem = parse_tool_arguments(raw)
        call = {"id": tc.get("id"), "name": function.get("name"), "arguments": arguments,
                "arguments_raw": raw if isinstance(raw, str) else None}
        if not call["name"]:
            arguments, problem = None, problem or "the tool call has no function name"
            call["arguments"] = None
        if problem:
            call["error"] = problem
        calls.append(call)
    reasoning = native.get("reasoning_content") or native.get("reasoning")
    stop_reason = getattr(choice, "finish_reason", None)
    refusal = None
    if stop_reason in REFUSAL_STOPS or native.get("refusal"):
        refusal = {"category": stop_reason, "explanation": native.get("refusal")}
    usage_raw = _to_dict(getattr(resp, "usage", None))
    leaked = None
    if not calls:
        channels = {"content": text, "reasoning": reasoning if isinstance(reasoning, str) else None}
        leaked = detect_leaked_tool_call(channels)
    return StepResult(
        text=text,
        tool_calls=calls,
        usage=normalise_openai_usage(usage_raw),
        usage_raw=usage_raw,
        latency_s=latency_s,
        wall_s=wall_s,
        stop_reason=stop_reason,
        response_id=getattr(resp, "id", None),
        model=getattr(resp, "model", None),
        native=native,
        native_format=NATIVE_OPENAI,
        system_fingerprint=fingerprint if isinstance(fingerprint := getattr(resp, "system_fingerprint", None), str)
        else None,
        retries=retries,
        retry_errors=list(retry_errors or []),
        leaked_tool_call=leaked,
        refusal=refusal,
    )
