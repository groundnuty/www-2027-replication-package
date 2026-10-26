"""What the checks read from the store, kept so that a checker decision can be
recomputed offline with no store.

The compiler reads the store through ``client.query(text)`` only (constant
types, the footprint re-run and its membership look-ups, provenance), and
writes those queries deterministically from the query, the answer and the
source. ``RecordingClient`` memoises every response by query text (one client
per output, shared by every source's compiler, so a query two sources both ask
goes to the store once) and keeps the responses as a log; ``ReplayStoreClient``
serves that log back by query text. A replay that asks for a query the log does
not hold, or leaves a logged query unused, has diverged: the compiler turns a
client exception into an ``unknown`` result, so the divergence is kept on the
client and the replay checks it afterwards, never through the result alone.
"""

from __future__ import annotations

from typing import Any

from harness.sparql import QueryResult, SparqlError
from harness.terms import row_from_json, row_to_json

STORE_LOG_VERSION = "checker-store-log/1"

# The checker's own store client (science, design log, change-log entry 125): the registered checker is the offline one,
# whose reads only rule 39's retrieval budget bounds (every query it sends carries a LIMIT of the budget left plus one),
# never the agents' row cap; so the online path (scripts/run_pilot.py), the offline replay (scripts/checker_replay.py)
# and the bank replay (scripts/bank_replay.py) all build the client the checks read through from these, and the
# agents' client keeps the run's row cap and time-out. Before entry 125 the online path read through the agents'
# client: a footprint response over its row cap was cut, and the checks reading it were unknown with reason "bound"
# where the offline replay completed them. A record keeps the responses it read (cut or not), so its decision
# recomputes from its log whichever client made it.
CHECKER_ROW_CAP: int | None = None
CHECKER_TIMEOUT_S = 280.0  # below the stores' 300 s server time-out; the replays' --timeout-s default


def result_to_json(r: QueryResult) -> dict[str, Any]:
    """The parts of a result the compiler reads (no timings, no raw JSON)."""
    if r.form == "ASK":
        return {"form": "ASK", "boolean": r.boolean}
    return {"form": r.form, "columns": list(r.columns), "rows": [row_to_json(row) for row in r.rows],
            "total_rows": r.total_rows, "truncated": r.truncated}


def result_from_json(obj: dict[str, Any]) -> QueryResult:
    if obj["form"] == "ASK":
        return QueryResult("ASK", [], [], obj["boolean"], 0, False, 0.0)
    return QueryResult(obj["form"], list(obj["columns"]), [row_from_json(r) for r in obj["rows"]], None,
                       obj["total_rows"], obj["truncated"], 0.0)


class RecordingClient:
    """Wraps a store client; answers repeated queries from memory and logs each
    distinct query once, with its response or its error."""

    def __init__(self, client: Any):
        self.client = client
        self.entries: list[dict[str, Any]] = []
        self._by_query: dict[str, dict[str, Any]] = {}

    def query(self, query: str) -> QueryResult:
        entry = self._by_query.get(query)
        if entry is None:
            try:
                entry = {"query": query, "result": result_to_json(self.client.query(query))}
            except SparqlError as exc:
                entry = {"query": query, "error": {"kind": exc.kind, "message": str(exc)[:500]}}
            except Exception as exc:  # noqa: BLE001 - the compiler reads any failure as a failed query
                entry = {"query": query, "error": {"kind": "client", "message": f"{type(exc).__name__}: {exc}"[:500]}}
            self._by_query[query] = entry
            self.entries.append(entry)
        if "error" in entry:
            raise SparqlError(entry["error"]["message"], entry["error"]["kind"])
        return result_from_json(entry["result"])

    def log(self) -> dict[str, Any]:
        return {"version": STORE_LOG_VERSION, "entries": list(self.entries)}


class ReplayStoreClient:
    """Serves a recorded store log by query text; records any divergence."""

    def __init__(self, log: dict[str, Any] | None):
        if log is not None and log.get("version") != STORE_LOG_VERSION:
            raise ValueError(f"unsupported store log {log.get('version')!r}")
        self._by_query = {e["query"]: e for e in (log or {}).get("entries", [])}
        self.used: set[str] = set()
        self.missing: list[str] = []

    def query(self, query: str) -> QueryResult:
        entry = self._by_query.get(query)
        if entry is None:
            self.missing.append(query)
            raise SparqlError("the replay asked for a query the store log does not hold", "replay")
        self.used.add(query)
        if "error" in entry:
            raise SparqlError(entry["error"]["message"], entry["error"]["kind"])
        return result_from_json(entry["result"])

    @property
    def unused(self) -> list[str]:
        return [q for q in self._by_query if q not in self.used]

    def divergence(self) -> str | None:
        if self.missing:
            return f"{len(self.missing)} query(ies) not in the store log, first: {self.missing[0][:200]}"
        if self.unused:
            return f"{len(self.unused)} logged query(ies) left unused, first: {self.unused[0][:200]}"
        return None
