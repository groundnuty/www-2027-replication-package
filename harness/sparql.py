"""A small SPARQL 1.1 protocol client for read-only queries.

Every query in the harness goes through this client: gold-answer generation,
the agent's SPARQL tool and the store checks. It records how long each query
took and whether the row cap cut the result, so callers can log both.
"""

from __future__ import annotations

import hashlib
import math
import json
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import httpx

from harness.gate_step_down import call_outcomes
from harness.query_gate import GATE_POLL_S, GateClosed, Hold, QueryGate
from harness.terms import Row, TermError, rows_from_sparql_json

SPARQL_JSON = "application/sparql-results+json"
MAX_RESPONSE_BYTES = 256 * 2**20
# httpx errors raised before the request can have reached the store.
_NEVER_SENT = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout, httpx.UnsupportedProtocol)
# QLever 9bebc60's error body is a JSON object with sorted keys, so its message comes first:
# {\n    "exception": "...",\n    "query": "<the query echoed>", ...}. A body the client cut off
# is read only when it starts with that field.
_EXCEPTION_FIRST = re.compile(r'\A\s*\{\s*"exception"\s*:\s*"((?:[^"\\]|\\.)*)', re.DOTALL)
# QLever sends "200 OK" before a lazily computed result is complete. When the export then fails
# (its time-out included) it appends exactly this, then the error message, which ends the body,
# and closes the stream cleanly: a 200 whose body is not JSON (bytes as sent by 9bebc60 on
# UniProt 7020). The line breaks are raw bytes, which a JSON string can never hold, so a copy of
# the text inside a row does not match.
_EXPORT_ERROR = (b"\n !!!!>># An error has occurred while exporting the query result. Unfortunately due to "
                 b"limitations in the HTTP 1.1 protocol, there is no better way to report this than to append it "
                 b"to the incomplete result. The error message was:\n")
# What may follow the export-error text, to the end of the body: QLever's time-out message, in the
# forms 9bebc60 emits ("Operation timed out." in every captured export error; "Operation timed out.
# Last operation: <operation>" in its 429 bodies), one line, no markup.
_EXPORT_TIMED_OUT = re.compile(r"(Operation timed out\.?(?: Last operation: [^<>\r\n]*)?)\s*")

# The first keyword after the prologue (BASE/PREFIX declarations and comments).
_PROLOGUE = re.compile(
    r"""\A(?:\s+|\#[^\n]*\n?|BASE\s*<[^>]*>|PREFIX\s+[^\s:]*:\s*<[^>]*>)*""",
    re.IGNORECASE,
)
_READ_FORMS = ("SELECT", "ASK", "CONSTRUCT", "DESCRIBE")

# SPARQL Update operations. A query that holds one as a keyword is refused
# whatever its first word, so "SELECT ... ; INSERT DATA {...}" never reaches an
# endpoint that runs several operations from one request.
_UPDATE_KEYWORDS = ("INSERT", "DELETE", "LOAD", "CLEAR", "DROP", "CREATE", "ADD", "MOVE", "COPY", "WITH")
# One token at a time. Comments, strings, IRIs, variables, language tags and
# prefixed names (blank node labels included) hide keyword-shaped text; their
# character classes follow the SPARQL 1.1 grammar, so the scanner never reads
# more into a string, IRI or comment than an endpoint would. A comment ends at
# CR or LF. A bare word is letters only, as an endpoint's lexer reads a keyword,
# so "INSERT2" is INSERT followed by 2.
_ESC = r"""\\(?:[tbnrf\\"']|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8})"""
_TOKEN = re.compile(
    r"""
    (?P<space>\s+)
  | (?P<comment>\#[^\r\n]*)
  | (?P<string>'''(?:(?:'|'')?(?:[^'\\]|ESC))*'''
             | \"\"\"(?:(?:"|"")?(?:[^"\\]|ESC))*\"\"\"
             | '(?:[^'\\\n\r]|ESC)*'
             | "(?:[^"\\\n\r]|ESC)*")
  | (?P<iri><[^<>"{}|^`\\\x00-\x20]*>)
  | (?P<var>[?$][\w·̀-ͯ‿-⁀]*)
  | (?P<lang>@[A-Za-z]+(?:-[A-Za-z0-9]+)*)
  | (?P<pname>[\w\-.]*:[\w\-.:%\\]*)
  | (?P<word>[A-Za-z]+)
  | (?P<number>\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)
  | (?P<other>.)
    """.replace("ESC", _ESC),
    re.VERBOSE | re.DOTALL,
)
# SPARQL lets \u and \U escapes stand for any character before the grammar
# applies, so the text is checked both as written and with them decoded.
_UCHAR = re.compile(r"\\u([0-9A-Fa-f]{4})|\\U([0-9A-Fa-f]{8})")


def _decode_uchar(match: re.Match) -> str:
    code = int(match.group(1) or match.group(2), 16)
    return chr(code) if code <= 0x10FFFF else match.group(0)


def _scan(text: str) -> str | None:
    depth = 0
    after_semicolon = False
    for m in _TOKEN.finditer(text):
        kind, token = m.lastgroup, m.group()
        if kind in ("space", "comment"):
            continue
        if after_semicolon:
            return "more text after a top-level ';'"
        if kind == "word":
            upper = token.upper()
            # A lexer reads "DROPSILENT" as DROP SILENT, and no read keyword or
            # function starts with an update keyword, so a prefix is enough.
            for keyword in _UPDATE_KEYWORDS:
                if upper.startswith(keyword):
                    return f"update keyword {keyword}"
        elif kind == "other":
            if token in "'\"":
                return "unterminated or malformed string literal"
            if token in "{([":
                depth += 1
            elif token in "})]":
                depth -= 1
            elif token == ";" and depth <= 0:
                after_semicolon = True
    return None


def read_only_violation(query: str) -> str | None:
    """Why the text is not a single read-only query, or None if nothing is wrong.

    An update keyword counts as a token outside strings, IRIs and comments and
    not inside a prefixed name, variable or language tag; a ';' at the top level
    (outside braces, parentheses and brackets) may be followed only by
    whitespace and comments.
    """
    texts = [query]
    decoded = _UCHAR.sub(_decode_uchar, query)
    if decoded != query:
        texts.append(decoded)
    for text in texts:
        reason = _scan(text)
        if reason is not None:
            return reason
    return None


def query_form(query: str) -> str | None:
    """The query form (SELECT, ASK, ...) or None if it is not one read query."""
    if read_only_violation(query) is not None:
        return None
    rest = query[_PROLOGUE.match(query).end():]
    word = re.match(r"[A-Za-z]+", rest)
    if word and word.group(0).upper() in _READ_FORMS:
        return word.group(0).upper()
    return None


class SparqlError(RuntimeError):
    """The endpoint refused or failed the query.

    ``sent`` says whether the store can have received the request: False when it provably never
    left (no connection), True when it went out, None when not known. ``server_exception`` is the
    store's own error message (QLever's JSON ``exception``), when its error body has one.
    """

    def __init__(self, message: str, kind: str, status: int | None = None, *, sent: bool | None = None,
                 server_exception: str | None = None):
        super().__init__(message)
        self.kind = kind  # "parse", "timeout", "http", "transport", "not_read_only", "too_large"
        self.status = status
        self.sent = sent
        self.server_exception = server_exception


@dataclass
class QueryResult:
    form: str
    columns: list[str]
    rows: list[Row]
    boolean: bool | None
    total_rows: int
    truncated: bool
    elapsed_s: float
    raw: dict[str, Any] = field(repr=False, default_factory=dict)  # bindings cut to the row cap
    # Set by clients that know the server's limit (VirtuosoClient): the server time-out sent, the server's own
    # execution time when it reports one, and whether the answer may be partial (wall time over a threshold).
    server_timeout_ms: int | None = None
    server_exec_ms: int | None = None
    possibly_partial: bool = False
    # Set by a client configured with a row limit (VirtuosoClient ``result_cap``): the answer reached it, so an
    # endpoint may have cut it there without any flag. Such an answer is also ``truncated``, and ``total_rows`` is
    # only a lower bound.
    store_cut: bool = False


class SparqlClient:
    def __init__(
        self,
        endpoint: str,
        timeout_s: float = 60.0,
        row_cap: int | None = None,
        http: httpx.Client | None = None,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
    ):
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self.row_cap = row_cap
        self.max_response_bytes = max_response_bytes
        self._http = http or httpx.Client()
        self._clock = time.monotonic

    def settings(self) -> dict[str, Any]:
        """What a run record needs to know about this client."""
        return {"endpoint": self.endpoint, "timeout_s": self.timeout_s, "row_cap": self.row_cap,
                "max_response_bytes": self.max_response_bytes}

    def query(self, query: str) -> QueryResult:
        form = query_form(query)
        if form is None:
            reason = read_only_violation(query)
            raise SparqlError("only SELECT, ASK, CONSTRUCT and DESCRIBE queries are allowed"
                              + (f" ({reason})" if reason else ""), "not_read_only")
        if form in ("CONSTRUCT", "DESCRIBE"):
            raise SparqlError(f"{form} is not supported by this client", "not_read_only")
        start = self._clock()
        try:
            headers, body = self._fetch(query, start)
        except _NEVER_SENT as exc:  # no connection: the store never received the query
            raise SparqlError(f"transport error: {exc}", "transport", sent=False) from exc
        except httpx.TimeoutException as exc:
            raise SparqlError(f"query timed out after {self.timeout_s:g}s", "timeout", sent=True) from exc
        except httpx.HTTPError as exc:
            raise SparqlError(f"transport error: {exc}", "transport", sent=True) from exc
        elapsed = self._clock() - start
        self._check(headers, elapsed)
        try:
            raw = json.loads(body)
        except ValueError as exc:
            message = _export_timed_out(body)
            if message is not None:
                raise SparqlError(f"partial result (the server stopped the export: {message})",
                                  "timeout", 200) from exc
            raise SparqlError("HTTP 200 with a body that is not JSON", "http", 200) from exc
        if not isinstance(raw, dict):
            raise SparqlError("HTTP 200 with JSON that is not a SPARQL result", "http", 200)
        if form == "ASK":
            if not isinstance(raw.get("boolean"), bool):
                raise SparqlError("ASK result without a boolean", "http", 200)
            return QueryResult(form, [], [], raw["boolean"], 0, False, elapsed, raw)
        head, results = raw.get("head"), raw.get("results")
        if not (isinstance(head, dict) and isinstance(head.get("vars"), list)
                and isinstance(results, dict) and isinstance(results.get("bindings"), list)):
            raise SparqlError("SELECT result without head.vars and results.bindings", "http", 200)
        total = len(results["bindings"])
        truncated = self.row_cap is not None and total > self.row_cap
        if truncated:
            raw = {**raw, "results": {**results, "bindings": results["bindings"][: self.row_cap]}}
        try:
            columns, rows = rows_from_sparql_json(raw)
        except (TermError, AttributeError, TypeError) as exc:
            raise SparqlError(f"malformed SPARQL result: {exc}", "http", 200) from exc
        return QueryResult(form, columns, rows, None, total, truncated, elapsed, raw)

    def _fetch(self, query: str, start: float) -> tuple[httpx.Headers, bytes]:
        """POST the query and read the whole answer: its headers and body, once the status is 200.

        The store-specific part of ``query``: a subclass with another request form (VirtuosoClient) overrides it.
        """
        with self._http.stream(
            "POST",
            self.endpoint,
            content=query.encode("utf-8"),
            headers={"Content-Type": "application/sparql-query", "Accept": SPARQL_JSON},
            # QLever reads its own time-out from the request; the HTTP
            # time-out is a backstop a little above it.
            params={"timeout": _timeout_param(self.timeout_s)},
            timeout=self.timeout_s + 10,
        ) as resp:
            if resp.status_code != 200:
                raise _http_error(resp.status_code, _read_prefix(resp, 2000))
            # Virtuoso answers a query that hit its time-out with HTTP 200
            # and the rows found so far; only these headers say so.
            if _partial_result(resp.headers):
                raise SparqlError("partial result (server time-out)", "timeout", 200)
            return resp.headers, self._read_body(resp, start + self.timeout_s + 10)

    def _check(self, headers: httpx.Headers, elapsed: float) -> None:
        """Refuse a 200 answer that is not a complete result (no further check by default)."""

    def _read_body(self, resp: httpx.Response, deadline: float) -> bytes:
        """The whole body; past max_response_bytes it is refused, not kept."""
        too_large = SparqlError(f"response larger than {self.max_response_bytes} bytes", "too_large", 200)
        declared = resp.headers.get("Content-Length", "")
        if declared.isdigit() and int(declared) > self.max_response_bytes:
            raise too_large
        chunks, size = [], 0
        for chunk in resp.iter_bytes():
            size += len(chunk)
            if size > self.max_response_bytes:
                raise too_large
            if self._clock() > deadline:
                raise SparqlError(f"query timed out after {self.timeout_s:g}s", "timeout", sent=True)
            chunks.append(chunk)
        return b"".join(chunks)


def _read_prefix(resp: httpx.Response, limit: int) -> str:
    """At most `limit` bytes of a streamed body, for an error message."""
    chunks, size = [], 0
    for chunk in resp.iter_bytes():
        chunks.append(chunk)
        size += len(chunk)
        if size >= limit:
            break
    return b"".join(chunks)[:limit].decode("utf-8", errors="replace")


def _partial_result(headers: httpx.Headers) -> bool:
    if headers.get("X-SQL-State", "").strip().upper() == "S1TAT":
        return True
    message = headers.get("X-SQL-Message", "").lower()
    return any(word in message for word in ("timed out", "timeout", "partial", "incomplete"))


def _timeout_param(timeout_s: float) -> str:
    """QLever's `timeout` value: whole seconds as before ("60s"), anything else in
    milliseconds, since QLever 9bebc60 refuses a fractional "0.5s" with HTTP 500."""
    if float(timeout_s).is_integer():
        return f"{int(timeout_s)}s"
    return f"{max(1, round(timeout_s * 1000))}ms"


def _export_timed_out(body: bytes) -> str | None:
    """QLever's time-out message when a streamed body ends with QLever's export-error text and that
    message, else None. Only called once the body failed to parse as JSON. The last copy of the text
    counts, and what follows it must be the message and nothing else (no markup, no further text);
    any other body is the generic not-JSON error."""
    at = body.rfind(_EXPORT_ERROR)
    if at < 0:
        return None
    match = _EXPORT_TIMED_OUT.fullmatch(body[at + len(_EXPORT_ERROR):].decode("utf-8", errors="replace"))
    return match.group(1) if match else None


def _says_timeout(text: str) -> bool:
    lowered = text.lower()
    return "timeout" in lowered or "timed out" in lowered or "cancelled" in lowered


def server_exception(body: str) -> str | None:
    """The store's own error message: the ``exception`` string of a body that is QLever's JSON error
    object, either whole (a JSON object with a top-level "exception" string) or cut off by the
    client (it starts with that field, QLever's order). None for any other body, a page, plain text
    or other JSON, wherever an "exception" text appears in it."""
    try:
        data = json.loads(body)
    except ValueError:
        match = _EXCEPTION_FIRST.match(body)
        if match is None:
            return None
        try:
            return json.loads(f'"{match.group(1)}"')
        except ValueError:  # cut off inside an escape
            return match.group(1)
    if isinstance(data, dict) and isinstance(data.get("exception"), str):
        return data["exception"]
    return None


# Virtuoso stops a query at its `timeout` parameter (milliseconds); the client always sends one, and never
# more than this.
VIRTUOSO_MAX_TIMEOUT_MS = 280_000
# The server time-out is this many times the client's (design ruling (b), design log).
SERVER_TIMEOUT_FACTOR = 2


class VirtuosoClient(SparqlClient):
    """A Virtuoso endpoint read through one default graph, with HTTP digest auth.

    Every query names the graph (default-graph-uri) and carries Virtuoso's own
    time-out. Partial answers are made impossible inside the client's window
    (design ruling (b), design log): the client gives up at ``timeout_s`` (a
    total deadline, not a per-read one) and the server time-out is twice that,
    so the server's anytime cut, which returns part of the rows with HTTP 200
    and no flag, can only fire after the client has stopped waiting. A client
    time-out above half the 280 s server maximum is refused, because the server
    time-out could then not be twice it. A store's own ``server_timeout_s`` (its
    description's setting, harness/gated_store.py) replaces twice the client's;
    it must still be after the client's time-out, which keeps the same order:
    the server's cut is measured from when the server received the request,
    which is after the client's deadline started, so it can only come after the
    client has stopped waiting.

    The wall-time guard stays as a second line: a response at or after
    ``wall_timeout_share`` of the client time-out is a time-out whatever its
    flags. So are HTTP 200 with ``X-SQL-State: S1TAT`` and HTTP 206 (partial
    rows). Every time-out raises the same error, so a caller (an agent) sees
    one kind of time-out. A returned result whose wall time exceeds
    ``possibly_partial_s`` is marked ``possibly_partial`` (ruling (d)).
    ``timings`` keeps one entry per query: wall time, client and server
    time-outs, the server's execution time when reported, and the outcome.
    ``settings()`` holds a digest of the graph IRI, not the IRI, because the
    graph's name can be private.

    With a ``gate`` (``harness.query_gate``), one query at a time holds across
    processes, at the lowest send layer (``_post``, which every request takes):
    the gate's lock is held for the whole request, and its reservation, set
    again immediately before the request goes out, runs for the server time-out
    from the send; the gate is released once the whole answer has arrived, and
    after a client time-out, a transport error, an answer cut or refused
    mid-read, an HTTP 206, a flagged answer or an answer slower than ``slow_s``
    the client first waits, still holding it, until the server time-out has
    passed. A closed gate refuses the query unsent (SparqlError kind "gate").
    With ``gate_wait_s`` the client waits for the gate instead (``QueryGate.
    wait_acquire``), up to that long, and only a gate still closed then is
    refused unsent. The time-out rules (``wall_timeout_share``,
    ``possibly_partial_s``, ``slow_s``) and ``wall_s`` count from the send, so
    a wait for the gate never makes an answer a time-out. Each ``timings``
    entry of a gated client adds ``gate_wait_s`` (the wait for the gate) and
    ``server_wait_s`` (the wait for the server time-out after the request),
    and ``last_waits_s`` is their sum for the latest query, which the agent's
    store wrapper counts as infrastructure time (harness/agent/infra.py).
    ``named_graphs`` are sent as named-graph-uri, for queries that name the
    graph inside GRAPH.

    A client keeps the state of its query in flight (the gate's hold, the
    timing entry), so one thread at a time may use it: a second query entered
    while one runs is refused (RuntimeError). Concurrent work takes one client
    each (the agent runner makes one per attempt).

    ``result_cap`` is an endpoint's own limit on the rows of a SELECT answer,
    where it has one (private configuration, read from a file by
    ``from_files``): an endpoint may cut a longer answer at its limit without
    any flag, so an answer with that many rows is marked ``store_cut`` and
    ``truncated`` (design log), never taken as complete.
    """

    TIMEOUT_MESSAGE = "query reached the time-out; no answer is returned"

    def __init__(
        self,
        endpoint: str,
        default_graph: str,
        timeout_s: float = 60.0,
        row_cap: int | None = None,
        http: httpx.Client | None = None,
        wall_timeout_share: float = 0.95,
        possibly_partial_s: float = 30.0,
        gate: QueryGate | None = None,
        named_graphs: tuple[str, ...] = (),
        slow_s: float | None = None,
        result_cap: int | None = None,
        gate_wait_s: float | None = None,
        gate_poll_s: float = GATE_POLL_S,
        http_factory: Callable[[], httpx.Client] | None = None,
        gate_monitor: Any = None,
        server_timeout_s: float | None = None,
        refuse_from_s: float | None = None,
    ):
        # a store's own server time-out (harness/gated_store.py), sent in whole milliseconds rounded UP, so the value
        # sent is never before the one set; it must still be after the client's time-out
        self._server_timeout_ms = None if server_timeout_s is None else math.ceil(round(server_timeout_s * 1000, 6))
        if self._server_timeout_ms is not None:
            if not (timeout_s * 1000 < self._server_timeout_ms <= VIRTUOSO_MAX_TIMEOUT_MS):
                raise ValueError(
                    f"a server time-out of {server_timeout_s:g} s must be after the client's {timeout_s:g} s and within "
                    f"the store's {VIRTUOSO_MAX_TIMEOUT_MS // 1000} s limit")
        # a store's own refusal time: an answer from this many seconds after the send is a time-out (compared as given,
        # not through a share of the client time-out, which rounding could move)
        if refuse_from_s is not None and not 0 < refuse_from_s < timeout_s:
            raise ValueError(f"answers are refused from a time before the client's {timeout_s:g} s time-out")
        self.refuse_from_s = refuse_from_s
        if self._server_timeout_ms is None and SERVER_TIMEOUT_FACTOR * timeout_s * 1000 > VIRTUOSO_MAX_TIMEOUT_MS:
            raise ValueError(
                f"a client time-out of {timeout_s:g} s needs a server time-out of "
                f"{SERVER_TIMEOUT_FACTOR * timeout_s:g} s, above the store's {VIRTUOSO_MAX_TIMEOUT_MS // 1000} s limit"
            )
        if http is None and http_factory is not None:
            http = http_factory()
        super().__init__(endpoint, timeout_s=timeout_s, row_cap=row_cap, http=http)
        self._http_factory = http_factory  # a new HTTP client after the deadline broke an exchange off
        self.default_graph = default_graph
        self.wall_timeout_share = wall_timeout_share
        self.possibly_partial_s = possibly_partial_s
        self.gate = gate
        self.named_graphs = tuple(named_graphs)
        self.slow_s = possibly_partial_s if slow_s is None else slow_s
        if result_cap is not None and result_cap < 1:
            raise ValueError("the store's row limit must be a positive number of rows")
        self.result_cap = result_cap
        if gate_wait_s is not None and (gate is None or gate_wait_s < 0):
            raise ValueError("a wait for the gate needs a gate and must not be negative")
        self.gate_wait_s = gate_wait_s
        self.gate_poll_s = gate_poll_s
        # a gate of several slots: told what each tool call's requests met (observe_call; harness/gate_step_down.py)
        if gate_monitor is not None and gate is None:
            raise ValueError("a gate monitor needs a gate")
        self.gate_monitor = gate_monitor
        self._sent_epochs: list[int | None] = []  # per ``timings`` entry: the gate's epoch its hold was taken in
        self._hold_epoch: int | None = None
        self._pending_hold: Hold | None = None
        self.timings: list[dict[str, Any]] = []
        self._exec_ms: int | None = None
        self._in_use = threading.Lock()
        self._sent_at: float | None = None  # when the latest request went out (client clock)
        self._queued_s = 0.0  # from asking for the gate to the send (the wait for the gate and building the request)
        self._gate_wait_s = 0.0  # the wait for the gate alone
        self._deadline: float | None = None  # the latest request's client deadline, from its first send
        self._exchange_sent = 0  # requests sent so far in the latest exchange
        self.last_waits_s = 0.0
        # Every request the HTTP client sends under a hold renews the reservation from its own send: an authenticating
        # client sends the query again after the store's challenge, and the server's time-out runs from that request.
        self._renew: Callable[[], Any] | None = None
        self._hook(self._http)

    def _hook(self, http: httpx.Client) -> None:
        hooks = http.event_hooks
        http.event_hooks = {**hooks, "request": [*hooks.get("request", []), self._on_request]}

    def _new_http(self) -> None:
        """Replace an HTTP client the deadline closed to break an exchange off."""
        old = self._http
        self._http = self._http_factory() if self._http_factory is not None else httpx.Client(auth=old.auth)
        self._hook(self._http)

    @classmethod
    def from_files(cls, endpoint: str, user_file, password_file, graph_file, result_cap_file=None,
                   **kw: Any) -> VirtuosoClient:
        """Credentials and graph IRI from private files (read here, never printed); a row limit too, from
        ``result_cap_file``, where given. ``name_the_graph`` sends the graph as the one named graph as well."""
        user = Path(user_file).expanduser().read_text().strip()
        password = Path(password_file).expanduser().read_text().strip()
        graph = Path(graph_file).expanduser().read_text().split()[0]
        if kw.pop("name_the_graph", False):
            kw["named_graphs"] = (graph,)
        if result_cap_file is not None:
            kw["result_cap"] = read_result_cap(result_cap_file)
        return cls(endpoint, graph, http_factory=lambda: httpx.Client(auth=httpx.DigestAuth(user, password)), **kw)

    @property
    def server_timeout_ms(self) -> int:
        if self._server_timeout_ms is not None:
            return self._server_timeout_ms
        return int(SERVER_TIMEOUT_FACTOR * self.timeout_s * 1000)

    def settings(self) -> dict[str, Any]:
        digest = hashlib.sha256(self.default_graph.encode()).hexdigest()[:16]
        out = {**super().settings(), "default_graph_sha256": digest, "server_timeout_ms": self.server_timeout_ms,
               "wall_timeout_share": self.wall_timeout_share, "possibly_partial_s": self.possibly_partial_s}
        if self.refuse_from_s is not None:  # only when set, so a client without it keeps its settings as before
            out["refuse_from_s"] = self.refuse_from_s
        if self.named_graphs:
            out["named_graphs_sha256"] = [hashlib.sha256(g.encode()).hexdigest()[:16] for g in self.named_graphs]
        if self.gate is not None:
            out.update(gate=True, slow_s=self.slow_s)
        if self.gate_wait_s is not None:
            out["gate_wait_s"] = self.gate_wait_s
        if self.result_cap is not None:
            out["result_cap_set"] = True  # the limit itself is private configuration
        return out

    def _fetch(self, query: str, start: float) -> tuple[httpx.Headers, bytes]:
        resp = self._post(query)
        if resp.status_code != 200:
            raise _virtuoso_http_error(resp.status_code, resp.text[:2000])
        return resp.headers, resp.content

    def _on_request(self, request: httpx.Request) -> None:
        deadline = self._deadline
        if deadline is not None:
            if self._exchange_sent:  # a later request of the exchange (an authenticating client's second one) gets
                left = deadline - self._clock()  # only what is left of the query's one client time-out
                if left <= 0:
                    raise httpx.ReadTimeout("the client deadline passed before the request was sent", request=request)
                request.extensions["timeout"] = httpx.Timeout(left).as_dict()
            self._exchange_sent += 1
        renew = self._renew
        if renew is not None:
            renew()
        if deadline is not None or renew is not None:
            request.extensions["trace"] = self._trace

    def _trace(self, name: str, info: dict) -> None:
        """httpcore's trace, at the moment the request's bytes go out (after connecting, however long that took): no
        request goes out past the client deadline, and the reservation runs from this send."""
        if not name.endswith("send_request_headers.started"):
            return
        deadline = self._deadline
        if deadline is not None and self._clock() >= deadline:
            raise httpx.ReadTimeout("the client deadline passed before the request was sent")
        renew = self._renew
        if renew is not None:
            renew()

    def _since_send(self, elapsed: float) -> float:
        """A time measured from before the gate, taken from the send instead."""
        return elapsed - self._queued_s

    def _post(self, query: str) -> httpx.Response:
        self._sent_at, self._queued_s, self._gate_wait_s = None, 0.0, 0.0
        asked = self._clock()

        def sent() -> None:
            self._sent_at = self._clock()
            self._queued_s = self._sent_at - asked

        if self.gate is None:
            return self._send(query, sent)
        try:
            if self.gate_wait_s is None:
                hold = self.gate.acquire(self.server_timeout_ms / 1000)
            else:
                hold = self.gate.wait_acquire(self.server_timeout_ms / 1000, max_wait_s=self.gate_wait_s,
                                              poll_s=self.gate_poll_s)
        except GateClosed as exc:
            self._queued_s = self._gate_wait_s = self._clock() - asked
            raise SparqlError(f"not sent: {exc}", "gate", sent=False) from None
        # from here to its release or its handing on to _settle, nothing may leave the hold behind
        try:
            self._hold_epoch = hold.epoch
            self._gate_wait_s = self._clock() - asked

            def sending() -> None:
                hold.sending()
                sent()

            self._renew = hold.sending
            try:
                resp = self._send(query, sending)
            finally:
                self._renew = None
            wait = (self._clock() - self._sent_at >= self.slow_s or resp.status_code == 206
                    or _partial_result(resp.headers))  # a slow or cut answer: wait the server time-out out all the same
        except BaseException:
            if self._sent_at is None:  # nothing went out: the store runs nothing of it
                hold.release()
            else:  # no whole answer, or not yet judged: the server may still be running the query
                self._pending_hold = hold
            raise
        if wait:
            self._pending_hold = hold
        else:
            hold.release()
        return resp

    def _settle(self) -> float:
        """The wait ``_post`` set, taken once the caller has measured the query (so the wait is not its wall time);
        returns how long it took."""
        hold = self._pending_hold
        if hold is None:
            return 0.0
        try:
            start = self._clock()
            hold.wait_out()
        except BaseException:  # interrupted: the lock goes, the reservation stays (Hold.wait_out does the same)
            hold._unlock()
            raise
        finally:
            self._pending_hold = None
        return self._clock() - start

    def _exclusive(self) -> None:
        if not self._in_use.acquire(blocking=False):
            raise RuntimeError("a VirtuosoClient is used by one thread at a time: make one client per attempt")

    def close(self) -> None:
        """Close the client's own HTTP connections."""
        self._http.close()

    def _send(self, query: str, sending: Callable[[], None] | None = None) -> httpx.Response:
        data: dict[str, Any] = {"query": query, "default-graph-uri": self.default_graph,
                                "timeout": str(self.server_timeout_ms)}
        if self.named_graphs:
            data["named-graph-uri"] = list(self.named_graphs)
        request = self._http.build_request(
            "POST",
            self.endpoint,
            data=data,
            headers={"Accept": SPARQL_JSON},
            timeout=httpx.Timeout(self.timeout_s),
        )
        # httpx's time-outs bound each operation, not the whole exchange: one timer, from the send, breaks it off at
        # the deadline whatever part runs then (connecting, an authenticating exchange, the headers: it closes the HTTP
        # client, replaced after; the body: it closes the response), and the body is checked against it chunk by chunk.
        if sending is not None:
            sending()  # the gate's reservation runs from here
        deadline = self._clock() + self.timeout_s
        guard, live = threading.Lock(), {"resp": None, "fired": False, "closed_http": False}
        http = self._http

        def expire() -> None:
            with guard:
                live["fired"] = True
                resp = live["resp"]
                live["closed_http"] = resp is None
            if resp is not None:
                resp.close()
            else:
                http.close()

        timer = threading.Timer(max(0.0, deadline - self._clock()), expire)
        timer.start()
        resp = None
        try:
            self._deadline, self._exchange_sent = deadline, 0  # bounds every request of the exchange (_on_request)
            try:
                resp = self._http.send(request, stream=True)
            except (httpx.TransportError, httpx.StreamError, RuntimeError) as exc:
                if live["fired"] or self._clock() >= deadline:
                    raise httpx.ReadTimeout("the client deadline passed before the answer came",
                                            request=request) from exc
                raise
            finally:
                self._deadline = None
            with guard:
                live["resp"] = resp
                fired = live["fired"]
            if fired:  # the deadline passed as the headers came
                raise httpx.ReadTimeout("the client deadline passed before the answer came", request=request)
            body = self._read_answer(resp, request, deadline)
        finally:
            timer.cancel()
            timer.join()  # a deadline that fired has finished closing before the client is judged
            if resp is not None:
                resp.close()
            if (live["closed_http"] or http.is_closed) and http is self._http:
                self._new_http()
        # the body is already decoded, so the encoding headers no longer describe it
        headers = [(k, v) for k, v in resp.headers.multi_items() if k.lower() not in ("content-encoding", "content-length")]
        return httpx.Response(resp.status_code, headers=headers, content=body, request=request)

    def _read_answer(self, resp: httpx.Response, request: httpx.Request, deadline: float) -> bytes:
        try:
            chunks, size = [], 0
            for chunk in resp.iter_bytes():
                chunks.append(chunk)
                size += len(chunk)
                if size > self.max_response_bytes:  # the reviewed client's size rule holds here too
                    raise SparqlError(f"response larger than {self.max_response_bytes} bytes", "too_large", 200)
                if self._clock() >= deadline:
                    raise httpx.ReadTimeout("the client deadline passed while reading the answer", request=request)
            return b"".join(chunks)
        except httpx.TimeoutException:
            raise
        except (httpx.TransportError, httpx.StreamError) as exc:  # the timer closed the stream, or the link failed
            if self._clock() >= deadline:
                raise httpx.ReadTimeout("the client deadline passed while reading the answer", request=request) from exc
            raise httpx.ReadError(f"the answer could not be read: {exc}", request=request) from exc

    # Virtuoso's count of current client connections; readable without admin rights (design log, load record).
    CONNECTIONS_QUERY = "SELECT (bif:sys_stat('st_cli_n_current_connections') AS ?n) WHERE { }"

    def active_connections(self) -> int | None:
        """The store's current client connections, this probe included; None when it does not say.

        Sent like any query (graph, time-outs) but kept out of ``timings``, which describe the caller's queries.
        """
        self._exclusive()
        try:
            resp = self._post(self.CONNECTIONS_QUERY)
            if resp.status_code != 200:
                return None
            return int(resp.json()["results"]["bindings"][0]["n"]["value"])
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError):
            return None
        finally:
            try:
                self._settle()
            finally:
                self._in_use.release()

    def _timeout(self, reason: str, status: int | None) -> SparqlError:
        self.timings[-1]["outcome"] = f"timeout:{reason}"
        return SparqlError(self.TIMEOUT_MESSAGE, "timeout", status)

    def _check(self, headers: httpx.Headers, elapsed: float) -> None:
        elapsed = self._since_send(elapsed)
        exec_ms = headers.get("X-Exec-Milliseconds")
        self._exec_ms = int(exec_ms) if exec_ms and exec_ms.isdigit() else None
        self.timings[-1].update(wall_s=round(elapsed, 3), server_exec_ms=self._exec_ms)
        if _partial_result(headers):  # X-SQL-State S1TAT, or a time-out message
            raise self._timeout("flagged", 200)
        refuse_at = self.refuse_from_s if self.refuse_from_s is not None else self.wall_timeout_share * self.timeout_s
        if elapsed >= refuse_at:
            raise self._timeout("wall_time", 200)  # _check runs only on a 200 answer

    def query(self, query: str) -> QueryResult:
        self._exclusive()
        try:
            return self._query(query)
        finally:
            self._in_use.release()

    def observe_call(self, before: int, retries: list | None) -> None:
        """Tell the gate monitor, where there is one, what the requests of one tool call met: the entries of
        ``timings`` from ``before`` on, with the call's ``infra_retries`` (gate_step_down.call_outcomes)."""
        if self.gate_monitor is None:
            return
        entries, epochs = self.timings[before:], self._sent_epochs[before:]
        for i, timed_out in call_outcomes(entries, retries):
            if i < len(epochs) and epochs[i] is not None:
                self.gate_monitor.observe(epochs[i], timed_out)

    def _query(self, query: str) -> QueryResult:
        self.last_waits_s = 0.0
        self._hold_epoch = None
        self._sent_at, self._queued_s, self._gate_wait_s = None, 0.0, 0.0
        self.timings.append({"timeout_s": self.timeout_s, "server_timeout_ms": self.server_timeout_ms,
                             "wall_s": None, "server_exec_ms": None, "outcome": None})
        entry = self.timings[-1]
        start = self._clock()
        try:
            result = super().query(query)
        except SparqlError as exc:
            if entry["wall_s"] is None:  # from the send; nothing was sent when the gate stayed closed
                entry["wall_s"] = round(self._clock() - self._sent_at, 3) if self._sent_at is not None else (
                    0.0 if exc.kind == "gate" else round(self._clock() - start, 3))
            if exc.status == 206:  # Virtuoso also answers an interrupted query with HTTP 206 and partial rows
                raise self._timeout("http_206", 206) from None
            if entry["outcome"] is None and exc.kind == "timeout":  # the client gave up first
                raise self._timeout("client", exc.status) from None
            if entry["outcome"] is None:
                entry["outcome"] = f"error:{exc.kind}"
            raise
        finally:
            self._sent_epochs.append(self._hold_epoch)
            waited_out = self._settle()
            if self.gate is not None:
                entry.update(gate_wait_s=round(self._gate_wait_s, 3), server_wait_s=round(waited_out, 3))
                self.last_waits_s = self._gate_wait_s + waited_out
        elapsed = self._since_send(result.elapsed_s)
        possibly_partial = elapsed > self.possibly_partial_s
        self.timings[-1].update(outcome="ok", rows=result.total_rows, possibly_partial=possibly_partial)
        result = replace(result, elapsed_s=elapsed, server_timeout_ms=self.server_timeout_ms,
                         server_exec_ms=self._exec_ms, possibly_partial=possibly_partial)
        if reaches_cap(result, self.result_cap):
            self.timings[-1]["store_cut"] = True
            result = mark_store_cut(result)
        return result


def read_result_cap(path) -> int:
    """A row limit from a private file holding one positive integer."""
    text = Path(path).expanduser().read_text().strip()
    if not text.isdigit() or int(text) < 1:
        raise ValueError("the row-limit file must hold one positive integer")
    return int(text)


def reaches_cap(result: QueryResult, result_cap: int | None) -> bool:
    """A SELECT answer with as many rows as the configured limit: an endpoint may cut an answer there with no flag."""
    return result_cap is not None and result.form == "SELECT" and result.total_rows >= result_cap


def mark_store_cut(result: QueryResult) -> QueryResult:
    return replace(result, truncated=True, store_cut=True)


# Virtuoso's error body is plain text whose first line is its own, "Virtuoso <SQLSTATE> Error <code>: <message>",
# followed by the query it echoes. Like QLever's JSON "exception" for the QLever client, only that line is the
# store's message: the echoed query (its "?timeout", a comment) never decides the kind.
_VIRTUOSO_ERROR = re.compile(r"\AVirtuoso (?P<state>[0-9A-Z]{5}) Error (?P<rest>[^\n]*)")


def _virtuoso_http_error(status: int, text: str) -> SparqlError:
    """The VirtuosoClient's reading of an HTTP error: SQLSTATE 37000 (a syntax error) is a parse error; S1T00, or a
    first line that says the transaction timed out, is a time-out; any other Virtuoso line is "http". A body
    without that first line is read by its status alone, as the QLever reader does (400 parse, 504 time-out)."""
    match = _VIRTUOSO_ERROR.match(text)
    message = match.group(0) if match else None
    if status == 400 or (match is not None and match.group("state") == "37000"):
        kind = "parse"
    elif match is None:
        kind = "timeout" if status == 504 else "http"
    elif match.group("state") == "S1T00" or _says_timeout(match.group("rest")):
        kind = "timeout"
    else:
        kind = "http"
    return SparqlError(f"HTTP {status}: {text}", kind, status, sent=True, server_exception=message)


def _http_error(status: int, text: str) -> SparqlError:
    # The kind comes from the store's own message only, QLever's JSON "exception" field: the body
    # also echoes the query, whose text ("?timeout", a comment) must not decide it. A body without
    # that field (a gateway's page, plain text) is never read: its kind comes from the status alone.
    exception = server_exception(text)
    if status == 400:
        kind = "parse"
    elif exception is None:
        kind = "timeout" if status == 504 else "http"
    elif _says_timeout(exception):
        kind = "timeout"
    elif "parse" in exception.lower() or "syntax" in exception.lower():
        kind = "parse"
    else:
        kind = "http"
    return SparqlError(f"HTTP {status}: {text}", kind, status, sent=True, server_exception=exception)
