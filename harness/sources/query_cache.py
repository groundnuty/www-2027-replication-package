"""Cached answers for a long build over a store we do not own, so an interrupted build resumes without resending.

Every answer and every time-out is appended to a private JSONL file (mode 600: the answers are the store's content),
keyed by the sha256 of the client's settings and the query text, so an answer read through another dataset or other
time-outs is never reused. A cached answer is returned as it came; a cached time-out is raised again without sending,
unless the build asks to retry time-outs. Other errors are not cached, so a resumed build sends those queries again.

With ``result_cap`` (a configured row limit, private configuration), every answer returned, cached or sent, that
reaches the limit is marked cut (``harness.sparql.mark_store_cut``): the check changes no answer, so it is not part of
the cache key, and an answer cached before it was in place is checked as well.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from harness.engine_suspect import suspect
from harness.sparql import QueryResult, SparqlError, mark_store_cut, reaches_cap
from harness.terms import rows_from_sparql_json


class SuspectGuard:
    """Refuses, unsent, a query of the shape ``harness.engine_suspect`` flags, with ``classes`` as the suspect classes
    (give every class the store types with, a superset of the known ones); the refusal is a SparqlError of kind
    "suspect", which a build records as unknown."""

    def __init__(self, inner, classes):
        self.inner = inner
        self.classes = frozenset(classes)
        self.refused = 0

    def settings(self) -> dict:
        return self.inner.settings()

    def query(self, query: str) -> QueryResult:
        if suspect(query, self.classes):
            self.refused += 1
            raise SparqlError("not sent: the query has the engine-suspect shape", "suspect", sent=False)
        return self.inner.query(query)


class CachingClient:
    def __init__(self, inner, path, key_settings: dict | None = None, retry_timeouts: bool = False,
                 result_cap: int | None = None):
        self.inner = inner
        self.path = Path(path)
        self.retry_timeouts = retry_timeouts
        self.result_cap = result_cap
        self._base = json.dumps(key_settings if key_settings is not None else inner.settings(), sort_keys=True)
        self.entries: dict[str, dict] = {}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as fh:
                for line in fh:
                    rec = json.loads(line)
                    self.entries[rec["key"]] = rec
        self.stats = {"sent": 0, "answered": 0, "timeouts": 0, "errors": 0, "cache_hits": 0,
                      "cached_timeouts_raised": 0, "possibly_partial": 0, "sent_seconds": 0.0, "store_cut": 0}

    def settings(self) -> dict:
        return self.inner.settings()

    def key(self, query: str) -> str:
        return hashlib.sha256(f"{self._base}\n{query}".encode()).hexdigest()

    def _append(self, rec: dict) -> None:
        self.entries[rec["key"]] = rec
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n")
        os.chmod(self.path, 0o600)

    @staticmethod
    def _result(rec: dict) -> QueryResult:
        raw = rec["raw"]
        if rec["form"] == "ASK":
            return QueryResult("ASK", [], [], bool(raw["boolean"]), 0, False, rec["elapsed_s"], raw,
                               possibly_partial=rec["possibly_partial"])
        columns, rows = rows_from_sparql_json(raw)
        return QueryResult(rec["form"], columns, rows, None, rec["total_rows"], rec["truncated"], rec["elapsed_s"], raw,
                           possibly_partial=rec["possibly_partial"])

    def query(self, query: str) -> QueryResult:
        result = self._query(query)
        if reaches_cap(result, self.result_cap):
            self.stats["store_cut"] += 1
            result = result if result.store_cut else mark_store_cut(result)
        return result

    def _query(self, query: str) -> QueryResult:
        key = self.key(query)
        rec = self.entries.get(key)
        if rec is not None and (rec["outcome"] == "ok" or not self.retry_timeouts):
            self.stats["cache_hits"] += 1
            if rec["outcome"] == "timeout":
                self.stats["cached_timeouts_raised"] += 1
                raise SparqlError("a cached time-out (not sent again)", "timeout", sent=False)
            return self._result(rec)
        query_sha = hashlib.sha256(query.encode()).hexdigest()
        t0 = time.monotonic()
        try:
            result = self.inner.query(query)
        except SparqlError as exc:
            if exc.sent is not False:
                self.stats["sent"] += 1
                self.stats["sent_seconds"] += time.monotonic() - t0
            if exc.kind == "timeout":
                self.stats["timeouts"] += 1
                self._append({"key": key, "query_sha256": query_sha, "outcome": "timeout",
                              "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            else:
                self.stats["errors"] += 1
            raise
        self.stats["sent"] += 1
        self.stats["answered"] += 1
        self.stats["sent_seconds"] += time.monotonic() - t0
        self.stats["possibly_partial"] += bool(result.possibly_partial)
        self._append({"key": key, "query_sha256": query_sha, "outcome": "ok", "form": result.form, "raw": result.raw,
                      "total_rows": result.total_rows, "truncated": result.truncated,
                      "elapsed_s": round(result.elapsed_s, 3), "possibly_partial": bool(result.possibly_partial),
                      "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        return result
