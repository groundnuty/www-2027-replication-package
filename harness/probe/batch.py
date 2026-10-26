"""Probe calls to Claude through the Message Batches API (AGENTS.md: batch single-shot sweeps).

A batch request's ``params`` are exactly the keyword arguments the route would send
synchronously (run.build_request: the route's ``request`` with no tools; the 1-hour
breakpoint on the static system prefix, none on the question). Each request gets a
``custom_id`` that is a digest of the run's nonce, its attempt and its try (``custom_id``: 1 to
64 of ``[A-Za-z0-9_-]``; the nonce, 64 random bits made once per run directory and kept in its
manifest, keeps two runs of the same attempts in one account from sharing an id), and the runner
keeps the map from custom id to attempt on disk
before it waits (``BatchState``), so a resumed run polls the batch it already paid
for instead of submitting again. Results come back in any order and are matched by
custom id only. Each result becomes a ``Reply`` (run.py), so a batch result goes
through the same parsing, scoring and record as a synchronous call:

    succeeded                       the message is the response (call mode "batch")
    errored, invalid_request_error  a provider error, read by the route's table: a context
                                    overflow is capped:context, anything else provider_error
    errored, authentication/permission/billing/not_found
                                    provider_error of class auth or not_found (the runner
                                    treats it as a misconfigured route and stops the model)
    errored, rate_limit/api/overloaded/timeout, or any other type
                                    infrastructure from the provider: never scored, resubmitted
    expired, canceled               infrastructure from the provider: resubmitted

Creating a batch is never retried (review of 05649ea): a create whose response is lost may
have made the batch, and a second create would pay for every request twice. ``submit``
calls ``create`` once; the runner writes a marker with the requests before it, and on any
failure leaves the marker and stops the model. ``find_submitted`` then settles the marker
(the runner's ``--reconcile-batches``): it lists the batches created since the marker was
written (less a margin for clock skew) that hold as many requests as the marker, waits for
each to end (a batch still running is waited for, never duplicated), and returns those
whose results carry exactly the marker's custom ids, which are unique per run, attempt and
try (review of 81f7af2: one shared id is not enough). A listing that finds nothing is not
proof that no batch was made, since a new batch may not be listed yet: the runner settles
such a marker only past a grace period and after a second listing (scripts/run_probes.py). The
retrieve, results and list calls, which change nothing, go through the route's retry
machinery (providers.py) and are retried and counted like any other call. The Batch API bills every token at half
the list price (cost.py prices records of call mode "batch" so). The order in which
the API processes a batch's requests cannot be set: requests are submitted in the
seeded order and the record keeps that position.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timedelta
from typing import Any, Callable

from harness.agent.providers import context_overflow_match
from harness.probe.run import Reply

CUSTOM_ID_PREFIX = "cbp-"
TRANSIENT_ERROR_TYPES = frozenset({"rate_limit_error", "api_error", "overloaded_error", "timeout_error"})
_CONFIG_ERROR_CLASSES = {"authentication_error": ("auth", 401), "permission_error": ("auth", 403),
                         "billing_error": ("auth", 402), "not_found_error": ("not_found", 404)}


def custom_id(attempt: tuple | list, try_number: int, run_nonce: str) -> str:
    """A batch custom id for one try of an attempt in the run of ``run_nonce``: 4 + 40 characters of [a-z0-9-]."""
    digest = hashlib.sha256(json.dumps([run_nonce, list(attempt), try_number],
                                       separators=(",", ":")).encode()).hexdigest()
    return CUSTOM_ID_PREFIX + digest[:40]


def batch_params(request: dict) -> dict:
    """A request's keyword arguments as batch ``params`` (``extra_body`` fields are body fields)."""
    params = {k: v for k, v in request.items() if k not in ("extra_body", "timeout")}
    params.update(request.get("extra_body") or {})
    return params


def _get(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def reply_from_result(result: Any, request: dict, call: dict) -> Reply:
    """One batch result (a MessageBatchIndividualResponse's ``result``) -> a Reply."""
    from harness.agent.providers import _to_dict

    kind = _get(result, "type")
    call = {**call, "result_type": kind, "latency_s": 0.0, "wall_s": call.get("wall_s") or 0.0, "retries": 0,
            "retry_errors": []}
    if kind == "succeeded":
        return Reply(request, call, _to_dict(_get(result, "message")), None)
    if kind == "errored":
        error = _get(_get(result, "error"), "error")
        etype, message = _get(error, "type") or "unknown", _get(error, "message") or ""
        if etype == "invalid_request_error":
            overflow = context_overflow_match(400, None, message) is not None
            failure_type, error_class = ("context_overflow", "context_overflow") if overflow else \
                ("provider_error", "bad_request")
            return Reply(request, call, None, {"failure_type": failure_type, "message": message, "details": {
                "error_class": error_class, "status": 400, "code": etype, "retries": 0, "transient": False}})
        if etype in _CONFIG_ERROR_CLASSES:
            error_class, status = _CONFIG_ERROR_CLASSES[etype]
            return Reply(request, call, None, {"failure_type": "provider_error", "message": message, "details": {
                "error_class": error_class, "status": status, "code": etype, "retries": 0, "transient": False}})
        return Reply(request, call, None, {"failure_type": "infrastructure", "message": f"batch request errored "
                                           f"({etype}): {message}", "details": {
            "error_class": "batch_" + etype, "status": None, "code": etype, "retries": 0, "transient": True}})
    return Reply(request, call, None, {"failure_type": "infrastructure", "message": f"batch request {kind}: not "
                                       f"processed", "details": {"error_class": f"batch_{kind}", "status": None,
                                                                  "code": kind, "retries": 0, "transient": True}})


class AnthropicBatches:
    """Create, wait for and read one Message Batch through an AnthropicRoute's client and retries."""

    def __init__(self, route: Any, *, poll_s: float = 60.0, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic):
        self.route = route
        self.poll_s = poll_s
        self._sleep = sleep
        self._clock = clock

    def _client(self):
        return self.route._ensure_client()

    def _retrying(self, fn: Callable[[], Any]) -> Any:
        response, *_ = self.route._call(lambda timeout: fn(), None)
        return response

    def submit(self, requests: list[tuple[str, dict]]) -> str:
        """``requests``: (custom id, request kwargs) pairs; returns the batch id."""
        ids = [cid for cid, _ in requests]
        if len(set(ids)) != len(ids):
            raise ValueError("custom ids must be unique within a batch")
        payload = [{"custom_id": cid, "params": batch_params(req)} for cid, req in requests]
        batch = self._client().messages.batches.create(requests=payload)  # once: a retry could submit twice
        return _get(batch, "id")

    def find_submitted(self, custom_ids: list[str] | set[str], since: str, *, slack_s: float = 600.0,
                       say: Callable[[str], None] = lambda msg: None) -> list[str]:
        """The ids of the batches created since ``since`` (an ISO time, less ``slack_s``) whose results hold
        exactly ``custom_ids``; a candidate still running is waited for. Candidates: as many requests as
        ``custom_ids``. The API lists batches newest first, so the listing stops at the first older one."""
        wanted = set(custom_ids)
        start = datetime.fromisoformat(since) - timedelta(seconds=slack_s)
        candidates = []
        for batch in self._retrying(lambda: self._client().messages.batches.list(limit=100)):
            created = _get(batch, "created_at")
            created = datetime.fromisoformat(created) if isinstance(created, str) else created
            if created < start:
                break
            counts = _get(batch, "request_counts")
            total = sum(_get(counts, k) or 0 for k in ("processing", "succeeded", "errored", "canceled", "expired"))
            if total == len(wanted):
                candidates.append(_get(batch, "id"))
        found = []
        for batch_id in candidates:
            self.wait(batch_id, say)
            if set(self.results(batch_id)) == wanted:  # every id, and no other: never another run's batch
                found.append(batch_id)
        return found

    def wait(self, batch_id: str, say: Callable[[str], None] = lambda msg: None) -> dict:
        """Poll until the batch has ended; returns {ended_at, request_counts}."""
        while True:
            batch = self._retrying(lambda: self._client().messages.batches.retrieve(batch_id))
            if _get(batch, "processing_status") == "ended":
                ended = _get(batch, "ended_at")
                counts = _get(batch, "request_counts")
                return {"ended_at": ended.isoformat() if hasattr(ended, "isoformat") else ended,
                        "request_counts": counts.to_dict() if hasattr(counts, "to_dict") else counts}
            say(f"batch {batch_id}: {_get(batch, 'processing_status')}; next poll in {self.poll_s:g} s")
            self._sleep(self.poll_s)

    def results(self, batch_id: str) -> dict[str, Any]:
        """custom id -> result, whatever order they arrive in."""
        out: dict[str, Any] = {}
        for item in self._retrying(lambda: list(self._client().messages.batches.results(batch_id))):
            cid = _get(item, "custom_id")
            if cid in out:
                raise ValueError(f"batch {batch_id} returned custom id {cid} twice")
            out[cid] = _get(item, "result")
        return out

