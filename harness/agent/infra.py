"""Store outages: told apart from query errors, waited out, never blamed on the agent.

Every failure of the agent's query falls in one of three classes (``failure_class``):

* **query**: the model's to see as a tool error. Refused by the client (not read-only, too
  large), a 4xx (a parse error; the store's own time-out, QLever's 429), or an HTTP 200 the
  client could not use (a streamed result cut by the store's cap, a body that is not JSON).
* **store**: an outage in which the store cannot be computing the query. The connection was
  refused or never opened (the client's ``sent`` is False), a 503 (the store is not taking
  queries), or a 5xx whose message names the store's own condition and would fail any query
  alike (``STORE_5XX``: the index still loading; a request parameter the store refuses), as an
  exact phrase at the start of QLever's ``exception`` field or of a plain-text body.
* **ambiguous**: anything else that may have reached the store, whose cause the failure does
  not show: no answer before the HTTP backstop time-out, a connection lost after the request
  went out (or not known to have stayed local), a gateway's 502 or 504, a 500 without a
  message, and every other 5xx, QLever's known ones included: "Waited for a result from another
  thread which then failed" (the request joined an identical query's computation, which failed
  for a reason not given), "Tried to allocate ..." (memory, which depends on the load), an
  engine "Assertion ... failed", and any message not listed (the conservative default).

A query error is raised at once. Otherwise the agent's query is sent again at most once, and
never while the store may still be computing it:

1. After an ambiguous failure nothing is sent, not even a probe, until the server's maximum
   time-out (``server_max_s``: 300 s unless the store client declares another; QLever refuses
   more without its access token, and the stores run with that default) plus a margin has
   passed since the query was sent. The request's own cap would not do: QLever keeps
   computing a query whose client has gone, and Sort-type operators overrun their cap (by up
   to 9 s in the calibration records, and by more than 275 s past a 10 s cap in a measured
   case whose computation ended about 297 s after it was sent; www2027-reviews/qlever-timeouts).
   This wait is infrastructure time, outside the outage limit. After a store failure the
   outage backoff starts at once.
2. The store is probed with a cheap query, not the agent's, so a query that crashed the store
   is not re-sent into it while it restarts. Any error on a probe means the store is still
   down; probes back off (5 s doubling to 60 s) until one is answered or the outage limit
   (failed requests plus waits) is passed, which raises ``StoreUnavailable``.
3. The query is sent again, once. An answer is used. A query error reaches the model. No
   answer again (the backstop) or a 504 again is the query's own time-out, a ``timeout`` tool
   error. A 5xx with the store's own message (QLever's ``exception`` field) again is shown by
   its kind (a second joined failure, assertion or allocation failure is an ``http`` tool
   error: the query fails alike on a store that answers probes); a 5xx without one is never
   read for a kind (harness/sparql.py), so it is not scored. Any other failure (a lost or refused connection, a 503, a
   bare 500 or 502) ends the attempt as an infrastructure failure flagged
   ``suspected_query``: the query may be what breaks the store, but on a shared store another
   attempt's query (or an unstable restart) can do the same, so it is never blamed on this
   agent. Nothing is sent a third time.

``StoreUnavailable`` ends the attempt with failure type ``infrastructure``, which is never
scored. The time spent on store retries is recorded per call (``infra_retries``,
``infra_s``) and is not charged to the attempt's wall cap.

A store client that waits for a shared query gate (harness/query_gate.py, the
VirtuosoClient's ``gate_wait_s``) reports, per query, how long it waited for the gate and for
the server time-out after the request (``last_waits_s``). Those waits are infrastructure time
too (design ruling, design log, 20 Oct): added to ``infra_s`` for every request whose time is not
already counted there (an answer or a query error; a failed request and a probe count whole).
A gate that stays closed past the client's bound (SparqlError kind "gate": nothing was sent)
ends the attempt at once as ``StoreUnavailable``: no probe and no re-send, since either would
wait for the same gate, and never ``suspected_query``.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable

from harness.sparql import QueryResult, SparqlError, server_exception

PROBE_QUERY = "SELECT ?s WHERE { ?s ?p ?o } LIMIT 1"
BASE_DELAY_S = 5.0
MAX_DELAY_S = 60.0
DEFAULT_LIMIT_S = 600.0
SERVER_MAX_S = 300.0  # the stores' maximum query time-out (QLever 9bebc60, stores/OPERATIONS.md)
AMBIGUOUS_MARGIN_S = 15.0
# 5xx messages that name the store's own condition: any query fails alike, so it is an outage. Each
# is an exact phrase at the start of the store's message (QLever's "exception" field) or of a
# plain-text body, so a phrase inside an echoed query, a page or another message does not count.
STORE_5XX = (re.compile(r"the index is still being loaded\b", re.IGNORECASE),
             # QLever refuses a request parameter (e.g. "0.5s" as a time-out)
             re.compile(r"failed to convert string '[^']*' to duration type\b", re.IGNORECASE))
_QUERY_KINDS = ("not_read_only", "too_large")


class StoreUnavailable(Exception):
    """The store stayed unavailable past the retry limit, or failed again on the
    agent's query right after answering a probe (``suspected_query``)."""

    def __init__(self, message: str, retries: list[dict], infra_s: float, *, suspected_query: bool = False):
        super().__init__(message)
        self.message = message
        self.retries = retries
        self.infra_s = infra_s
        self.suspected_query = suspected_query
        self.query_timings: list | None = None  # the store client's own timing of the requests, where it keeps one


def _message(exc: SparqlError) -> str | None:
    """The store's own message for an HTTP error: the client's reading, else read from the text."""
    if exc.server_exception is not None:
        return exc.server_exception
    text = str(exc)
    prefix = f"HTTP {exc.status}: "
    return server_exception(text[len(prefix):]) if text.startswith(prefix) else None


def _plain_body(exc: SparqlError) -> str | None:
    """The body of an HTTP error that is plain text (not JSON, not markup), else None."""
    text = str(exc)
    prefix = f"HTTP {exc.status}: "
    if not text.startswith(prefix):
        return None
    body = text[len(prefix):].strip()
    try:
        json.loads(body)
        return None
    except ValueError:
        return None if body.startswith("<") else body


def _names_store_condition(exc: SparqlError) -> bool:
    for text in (_message(exc), _plain_body(exc) if _message(exc) is None else None):
        if text is not None and any(phrase.match(text.strip()) for phrase in STORE_5XX):
            return True
    return False


def failure_class(exc: SparqlError) -> str:
    """"query", "store" or "ambiguous" (module docstring)."""
    if exc.kind in _QUERY_KINDS:
        return "query"
    if exc.kind == "transport":
        return "store" if exc.sent is False else "ambiguous"
    if exc.status is None:
        return "ambiguous" if exc.kind == "timeout" else "query"
    if exc.status < 500:
        return "query"
    if exc.status == 503 or _names_store_condition(exc):
        return "store"
    return "ambiguous"


def _failure(kind_of_query: str, exc: SparqlError, request_s: float, wait_s: float) -> dict:
    return {"query": kind_of_query, "kind": exc.kind, "status": exc.status, "message": str(exc)[:300],
            "request_s": round(request_s, 3), "wait_s": wait_s}


class RetryingSparql:
    """Wraps a SPARQL client: store errors are retried, query errors pass through.

    After each ``query`` call, ``last_retries`` lists the failed requests and
    probes (with their durations and the waits after them) and ``last_infra_s``
    is their total time.
    """

    replays_infra_retries = False

    def __init__(self, inner: Any, *, limit_s: float = DEFAULT_LIMIT_S, base_delay_s: float = BASE_DELAY_S,
                 max_delay_s: float = MAX_DELAY_S, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic, server_max_s: float | None = None,
                 ambiguous_margin_s: float = AMBIGUOUS_MARGIN_S):
        self.inner = inner
        self.limit_s = limit_s
        # how long the store may still compute a query its client gave up on: given, declared by the
        # store client, or the stores' hard maximum
        self.server_max_s = float(server_max_s if server_max_s is not None
                                  else getattr(inner, "server_max_s", SERVER_MAX_S))
        self.ambiguous_margin_s = ambiguous_margin_s
        self.base_delay_s = base_delay_s
        self.max_delay_s = max_delay_s
        self._sleep = sleep
        self._clock = clock
        self.last_retries: list[dict] = []
        self.last_infra_s = 0.0

    def settings(self) -> dict[str, Any]:
        return self.inner.settings()

    def _send(self, query: str) -> tuple[QueryResult | None, SparqlError | None, float]:
        start = self._clock()
        try:
            return self.inner.query(query), None, self._clock() - start
        except SparqlError as exc:
            return None, exc, self._clock() - start

    def _waits(self) -> float:
        """The inner client's waits for a query gate behind its latest query (0 for a client without one)."""
        return float(getattr(self.inner, "last_waits_s", 0.0) or 0.0)

    def _gate_closed(self, who: str, exc: SparqlError, request_s: float, retries: list[dict],
                     infra_s: float) -> StoreUnavailable:
        retries.append(_failure(who, exc, request_s, 0.0))
        infra_s += request_s
        self._finish(retries, infra_s)
        return StoreUnavailable(f"the query gate stayed closed; the query was not sent ({str(exc)[:300]})", retries,
                                round(infra_s, 3))

    def query(self, query: str) -> QueryResult:
        retries: list[dict] = []
        result, exc, request_s = self._send(query)
        if exc is None:
            self._finish(retries, self._waits())
            return result
        if exc.kind == "gate":
            raise self._gate_closed("agent", exc, request_s, retries, 0.0) from exc
        first = failure_class(exc)
        if first == "query":
            self._finish(retries, self._waits())
            raise exc
        infra_s = request_s
        delay = self.base_delay_s
        waited_s = 0.0  # the wait after an ambiguous failure: infrastructure time, outside the outage limit
        if first == "ambiguous":  # the store may still be computing it: wait out its maximum from the send
            waited_s = max(0.0, self.server_max_s + self.ambiguous_margin_s - request_s)
            retries.append(_failure("agent", exc, request_s, round(waited_s, 3)))
            self._sleep(waited_s)
            infra_s += waited_s
            sent, failed = PROBE_QUERY, None
        else:
            sent, failed = None, exc
        while True:  # probe until one is answered
            if failed is None:
                _, failed, request_s = self._send(sent)
                if failed is None:
                    retries.append({"query": "probe", "ok": True, "request_s": round(request_s, 3)})
                    infra_s += request_s
                    break
                if failed.kind == "gate":
                    raise self._gate_closed("probe", failed, request_s, retries, infra_s) from failed
                infra_s += request_s
            who = "probe" if sent == PROBE_QUERY else "agent"
            outage_s = infra_s - waited_s
            if outage_s + delay > self.limit_s:
                retries.append(_failure(who, failed, request_s, 0.0))
                self._finish(retries, infra_s)
                raise StoreUnavailable(
                    f"the SPARQL store stayed unavailable for {outage_s:.0f} s (retry limit {self.limit_s:g} s); "
                    f"last error: {str(failed)[:300]}", retries, round(infra_s, 3)) from failed
            retries.append(_failure(who, failed, request_s, delay))
            self._sleep(delay)
            infra_s += delay
            delay = min(delay * 2, self.max_delay_s)
            sent, failed = PROBE_QUERY, None
        result, exc, request_s = self._send(query)  # the one re-send
        if exc is None:
            self._finish(retries, infra_s + self._waits())
            return result
        if exc.kind == "gate":
            raise self._gate_closed("agent", exc, request_s, retries, infra_s) from exc
        second = failure_class(exc)
        if second == "query":
            self._finish(retries, infra_s + self._waits())
            raise exc
        retries.append(_failure("agent", exc, request_s, 0.0))
        infra_s += request_s
        self._finish(retries, infra_s)
        if second == "ambiguous" and (exc.kind == "timeout" or exc.status == 504):
            raise SparqlError(f"the query did not finish in time ({exc})", "timeout", exc.status) from exc
        if second == "ambiguous" and exc.status is not None and _message(exc) is not None:
            raise exc  # the store's own error again, on a store that answers probes: shown by its kind
        raise StoreUnavailable(
            f"the SPARQL store failed on this query right after answering a probe query ({str(exc)[:300]}); "
            f"it was not sent again (the query may be what breaks the store)",
            retries, round(infra_s, 3), suspected_query=True) from exc

    def _finish(self, retries: list[dict], infra_s: float) -> None:
        self.last_retries = retries
        self.last_infra_s = round(infra_s, 3)


def wait_for_store(sparql: Any, *, limit_s: float, sleep: Callable[[float], None] = time.sleep,
                   base_delay_s: float = BASE_DELAY_S, max_delay_s: float = MAX_DELAY_S) -> tuple[bool, float]:
    """Probe until the store answers without error -> (answered, seconds waited).

    Waits with the same backoff as the retries, and gives up when another wait
    would pass ``limit_s``. Only SparqlError counts as the store being down.
    """
    waited = 0.0
    delay = base_delay_s
    while True:
        try:
            sparql.query(PROBE_QUERY)
            return True, waited
        except SparqlError:
            pass
        if waited + delay > limit_s:
            return False, waited
        sleep(delay)
        waited += delay
        delay = min(delay * 2, max_delay_s)
