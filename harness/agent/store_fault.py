"""The store time-out rule of the test runs (change-log entry 149, item 2; science, design log, 21 Oct 18:10): the same
hold-and-confirm shape as the model rule (harness/agent/infra_fault.py, entry 148), per STORE COPY, so a store's
time-outs, which the agent sees as failed queries, can never by themselves authorise re-running the attempts.

* Copy: the store copy an OUT reads, as the launch step recorded it (``e.json``'s ``stores.by_out``: each launch's
  replica, its OUT being ``<runs dir>/<launch name without tier1-/tier2->``; ``copies_from_e_record``), or a file
  the operator writes from it, ``{"outs": {"<OUT path>": "<copy id>"}}`` (``load_copies``). One copy per (model,
  OUT), so every query of an OUT's records is on its copy.
* A query: each request a ``sparql_query`` tool call sent to the store. A call with the client's timing list
  (``query_timings``, the gated client's) counts its requests by gate_step_down.call_outcomes (the agent's query and
  its one re-send; never a probe or a request the gate kept back) and a request timed out when its outcome starts with
  "timeout:". A call without one (the plain client's) that reached the store (``client_called``) is one request, timed
  out when its error kind is "timeout" (harness/sparql.py's SparqlError kinds). A refused call sent nothing.
* Its time: the call's turn start (the record's ``started_at`` plus that turn's ``tokens.per_turn`` ``started_s``),
  plus the turn's ``wall_s`` (the model's reply, after which the calls run), plus the ``elapsed_s`` of the turn's
  earlier calls (they run one after another); it ends ``elapsed_s`` later. A branch record counts its live retry only,
  on its parent's continued clock (infra_fault.record_events' reading). A call whose turn has no timed entry is left
  out and counted (``untimed_calls``).
* Trigger, a HOLD only: in an aligned 15-minute UTC bin (infra_fault.bin_start), a copy's share of timed-out queries
  (by send time) REACHES its trigger share (>=, exact fractions) with at least STORE_MIN_TIMEOUTS time-outs. The
  trigger share is STORE_FACTOR (10) times the copy's dev time-out share, or, for an a15 copy, A15_FACTOR (2) times the
  copy's qualification rate (its rate sample); a zero reference leaves the time-outs alone to trigger (5 in a bin). A
  copy with no frozen reference is never evaluated: it is reported as unmonitored.
* Confirmation: the STORE PROBE (scripts/store_probe.py): the copy's fixed, hashed set of STORE_PROBE_N dev gold
  queries that completed at its identity check, sent one at a time; it FAILS when more than STORE_PROBE_MAX_TIMEOUTS
  time out, or when any request still errors after STORE_PROBE_RESENDS re-sends (a store that cannot answer is not
  healthy). A probe that cannot run at all decides nothing and the copy stays held.
* Confirmed: the window runs from the first triggering bin's start until the probe passes again, and every attempt
  with a query on that copy whose time overlaps the window (``attempts_with_query_in_window``), in every arm, is an
  infrastructure failure, re-run under section 4.3. Not confirmed: the attempts stand, scored as usual.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from harness.agent import infra_fault as fault
from harness.gate_step_down import call_outcomes

STORE_FACTOR = Fraction(10)
A15_FACTOR = Fraction(2)  # an a15 copy: twice its qualification rate (entry 149)
STORE_MIN_TIMEOUTS = 5
STORE_PROBE_N = 20
STORE_PROBE_MAX_TIMEOUTS = 2
STORE_PROBE_RESENDS = 3  # section 4.3's re-runs, for a request that errors (a time-out is the probe's measure)
STORE_PROBE_SET_SCHEMA = "store-probe-set/1"
COPIES_SCHEMA = "store-copies/1"
QUERY_TOOL = "sparql_query"
STORE_RULE = {
    "rule": "change-log entry 149, item 2 (science, design log, 21 Oct 18:10)", "bin_s": fault.BIN_S,
    "bins": "aligned to UTC :00, :15, :30, :45", "factor": str(STORE_FACTOR), "a15_factor": str(A15_FACTOR),
    "min_timeouts": STORE_MIN_TIMEOUTS, "trigger": "share of timed-out queries in the bin >= factor x reference share",
    "probe": {"n": STORE_PROBE_N, "max_timeouts": STORE_PROBE_MAX_TIMEOUTS, "resends": STORE_PROBE_RESENDS,
              "fails": f"more than {STORE_PROBE_MAX_TIMEOUTS} of {STORE_PROBE_N} time out, or a request still errors "
                       f"after {STORE_PROBE_RESENDS} re-sends"},
    "window": "first triggering bin's start until the probe passes; every attempt with a query overlapping it, "
              "every arm"}
PROBE_SEED_BASE = "20261021"
# the approved copy ids and the rule later copies are named by: '<graph>-<port>',
# '<graph>-<host tag>-<port>', '<graph>-<host>-<n>'; an a15 copy is named by its site
COPY_IDS = ("example-copy-52", "example-copy-46", "example-copy-53", "example-copy-47", "example-copy-38", "example-copy-12",
            "example-copy-9", "a15-example-1", "a15-example-2", "a15-example-3")
COPY_GRAPHS = ("uniprot", "rhea", "opaque-rhea", "corporate")
COPY_ID_RULE = re.compile(rf"(?:{'|'.join(COPY_GRAPHS)})(?:-[a-z][a-z0-9]*)?-[0-9]{{4,5}}"
                          rf"|(?:{'|'.join(COPY_GRAPHS)}|a15)-example-[0-9]+|a15-example-[1-3]")
STORE_PROBE_PROCEDURE_VERSION = "store-probe-procedure/1"
STORE_PROBE_PROCEDURE = {
    "version": STORE_PROBE_PROCEDURE_VERSION,
    "rule": "change-log entry 149, item 2 (science, design log, 21 Oct 18:10 and 19:52)",
    "identity_check_gold_run": ("every dev gold query of the copy's graph, each sent once, one at a time, at the "
                                "copy's time-out (scripts/store_probe.py golds); for an a15 copy its qualification "
                                "gold re-run"),
    "set": (f"the dev gold queries that completed (outcome 'answered') at that run, ordered by "
            f"sha256('{PROBE_SEED_BASE}|<copy id>|<query id>') (seed '{PROBE_SEED_BASE}|<copy id>'), the first "
            f"{STORE_PROBE_N}; the set file holds each query's id and the sha256 of its gold text"),
    "fails": STORE_RULE["probe"]["fails"],
    "binding": ("the set's sha256 is recorded in the copy's qualification record before its first test attempt; a "
                "run refuses a set or a query text whose sha256 differs"),
    "reference": ("a copy of a non-a15 store: its store's dev time-out share x 10; an a15 copy: its qualification "
                  "rate sample (timed out of sample) x 2"),
    "copy_ids": list(COPY_IDS),
    "copy_id_rule": ("'<graph>-<port>', '<graph>-<host tag>-<port>' or '<graph>-example-<n>', graph one of "
                     + ", ".join(COPY_GRAPHS) + "; a15 copies 'a15-<site>' (example-1, example-2, example-3)"),
}


def probe_seed(copy: str) -> str:
    """A copy's probe seed (STORE_PROBE_PROCEDURE): each copy draws its own set."""
    return f"{PROBE_SEED_BASE}|{check_copy_id(copy)}"


_PORTED_COPY = re.compile(rf"(?:{'|'.join(COPY_GRAPHS)})(?:-[a-z][a-z0-9]*)?-(?P<port>[0-9]{{4,5}})")
_SITEB_COPY = re.compile(rf"(?:{'|'.join(COPY_GRAPHS)}|a15)-example-[0-9]+")


# Each approved non-a15 copy's endpoint, registered from records made before and apart from its gold run: the store
# agent's copy record (the copies.json of the store agent's reviews: the reference copy and the copies on two
# nodes) and the dev-grid store descriptions (data/runs/dev-grid/store-*.json). An identity-check gold
# run reads exactly this (host, port), or it is refused: a port alone does not name a replica. A later copy is added
# here before its gold run. HOST_ALIASES: one server's names.
COPY_ENDPOINTS = {
    "example-copy-52": ("192.0.2.105", 7022), "example-copy-46": ("192.0.2.105", 7024),
    "example-copy-53": ("192.0.2.101", 7026), "example-copy-47": ("192.0.2.101", 7028),
    "example-copy-38": ("192.0.2.105", 7039), "example-copy-12": ("192.0.2.101", 7033),
    "example-copy-9": ("192.0.2.101", 7041),
    "example-copy-13": ("192.0.2.105", 7045),
    "example-copy-39": ("192.0.2.105", 7043),
}
HOST_ALIASES = {"node-1.cluster-a.example": "192.0.2.105", "node-1": "192.0.2.105",
                "node-2.cluster-a.example": "192.0.2.101", "node-2": "192.0.2.101"}


def check_copy_endpoint(copy: str, endpoint: str | None) -> None:
    """ValueError unless ``endpoint`` is the copy's registered (host, port) (COPY_ENDPOINTS); a copy with no registered
    endpoint is refused, never assumed."""
    want = COPY_ENDPOINTS.get(copy)
    if want is None:
        raise ValueError(f"copy {copy} has no registered endpoint (store_fault.COPY_ENDPOINTS)")
    parts = urlsplit(endpoint or "")
    got = (HOST_ALIASES.get(parts.hostname or "", parts.hostname), parts.port)
    if got != want:
        raise ValueError(f"copy {copy} is the store at {want[0]}:{want[1]}, not {got[0]}:{got[1]}")


def copy_port(copy: str) -> int | None:
    """The port a copy id ending in '-<port>' names (its identity-check gold run must read that port); None for an id
    that names none (an a15 copy, '<graph>-example-<n>')."""
    if _SITEB_COPY.fullmatch(copy or ""):
        return None
    m = _PORTED_COPY.fullmatch(copy or "")
    return int(m["port"]) if m else None


def check_copy_id(copy: str) -> str:
    """``copy`` if it is an approved id or follows the naming rule, else ValueError."""
    if copy in COPY_IDS or COPY_ID_RULE.fullmatch(copy or ""):
        return copy
    raise ValueError(f"{copy!r} is not an approved copy id ({STORE_PROBE_PROCEDURE['copy_id_rule']})")


@dataclass(frozen=True)
class StoreQueryEvents:
    """What the store rule reads of one record."""

    run_id: str
    model: str  # the runner's name (the OUT's manifest routes)
    copy: str
    arm: str
    started_at: datetime
    last_activity_at: datetime
    queries: tuple[tuple[datetime, bool], ...]  # each request: (sent, timed out)
    query_ends: tuple[datetime, ...] = ()  # each request's end (sent + its call's elapsed_s), in the same order
    untimed_calls: int = 0
    branch: bool = False


def call_requests(call: dict) -> list[bool]:
    """Whether each store request of one tool call timed out (module docstring); [] for a call that sent none."""
    if call.get("name") != QUERY_TOOL:
        return []
    timings = call.get("query_timings")
    if isinstance(timings, list):
        return [timed_out for _, timed_out in call_outcomes(timings, call.get("infra_retries"))]
    if not call.get("client_called"):
        return []
    return [((call.get("error") or {}).get("kind")) == "timeout"]


def store_record_events(rec: dict, name_of: Mapping[str, str], copy: str) -> StoreQueryEvents | str:
    """The store rule's events of one record on ``copy``, or why it is left out: "infrastructure", "no_retry",
    "unmapped" or "untimed" (infra_fault.record_events' reasons, read the same way)."""
    outcome = rec.get("outcome") or {}
    if outcome.get("failure_type") == "infrastructure":
        return "infrastructure"
    checker = rec.get("checker")
    if checker is not None and not checker.get("retry"):
        return "no_retry"
    name = name_of.get(rec.get("model"))
    if name is None:
        return "unmapped"
    start = fault.parse_time(rec["started_at"])
    per_turn = {t.get("turn"): t for t in (rec.get("tokens") or {}).get("per_turn") or []}
    replayed = 0
    if checker is not None:  # the parent's turns (and their calls) replay without reaching the store
        replayed = ((checker.get("initial") or {}).get("outcome") or {}).get("turns") or 0
        ended, wall = rec.get("ended_at"), rec.get("wall_s")
        if not (isinstance(ended, str) and isinstance(wall, (int, float))):
            return "untimed"
        end = fault.parse_time(ended)
        at_clock = lambda s: end - timedelta(seconds=wall - s)  # noqa: E731
    else:
        at_clock = lambda s: start + timedelta(seconds=s)  # noqa: E731
    queries: list[tuple[datetime, bool]] = []
    ends: list[datetime] = []
    untimed = 0
    spent: dict[int, float] = {}  # turn -> elapsed_s of its calls so far
    for call in sorted(rec.get("tool_calls") or [], key=lambda c: (c.get("turn") or 0, c.get("index") or 0)):
        turn = call.get("turn") or 0
        if turn <= replayed:
            continue
        requests = call_requests(call)
        elapsed = call.get("elapsed_s") if isinstance(call.get("elapsed_s"), (int, float)) else 0.0
        entry = per_turn.get(turn) or {}
        s = entry.get("started_s")
        if not isinstance(s, (int, float)):
            untimed += bool(requests)
            continue
        reply_s = entry.get("wall_s") if isinstance(entry.get("wall_s"), (int, float)) else 0.0
        sent = at_clock(s + reply_s + spent.get(turn, 0.0))
        spent[turn] = spent.get(turn, 0.0) + elapsed
        for timed_out in requests:
            queries.append((sent, timed_out))
            ends.append(sent + timedelta(seconds=elapsed))
    ended_at = rec.get("ended_at")
    last = max([*ends, start, *([fault.parse_time(ended_at)] if isinstance(ended_at, str) else [])])
    arm = (rec.get("load") or {}).get("arm") or rec.get("condition") or "none"
    return StoreQueryEvents(run_id=rec["run_id"], model=name, copy=copy,
                            arm=fault.BRANCH_PREFIX + arm if checker is not None else arm, started_at=start,
                            last_activity_at=last, queries=tuple(queries), query_ends=tuple(ends),
                            untimed_calls=untimed, branch=checker is not None)


def out_store_events(out: Path, copy: str, cache: dict[Path, StoreQueryEvents | str] | None = None
                     ) -> tuple[list[StoreQueryEvents], Counter]:
    """Every record of OUT/records/ as StoreQueryEvents on ``copy``, and how many were left out by reason (a record
    never changes once written: ``cache`` spares reading it again; a path gone is dropped from it)."""
    names = fault.route_names(out)
    files = fault.record_files(out)
    if cache is not None:
        for gone in {p for p in cache if p.parent == Path(out) / "records"} - set(files):
            del cache[gone]
    evs, left_out = [], Counter()
    for path in files:
        e = cache.get(path) if cache is not None else None
        if e is None:
            e = store_record_events(json.loads(path.read_text(encoding="utf-8")), names, copy)
            if cache is not None:
                cache[path] = e
        if isinstance(e, str):
            left_out[e] += 1
        else:
            if e.untimed_calls:
                left_out["untimed_calls"] += e.untimed_calls
            evs.append(e)
    return evs, left_out


# --- which copy each OUT reads --------------------------------------------------------------------


def _launch_out(runs_dir: str, name: str) -> str:
    """The launch step's OUT of a launch name (freeze_day.run_out_of)."""
    return f"{runs_dir}/{name.replace('tier1-', '', 1).replace('tier2-', '', 1)}"


def copies_from_e_record(e_doc: dict, runs_dir: str) -> dict[str, str]:
    """OUT path -> copy id from the launch step's record (e.json ``stores.by_out``: launch name -> {"replica",
    "endpoint", "store_json_sha256", ...}); the replica's name is the copy id. An OUT whose store has no replica name
    is refused: a copy must be named to be monitored and probed."""
    by_out = ((e_doc.get("stores") or {}).get("by_out")) or {}
    out: dict[str, str] = {}
    for name, rec in sorted(by_out.items()):
        replica = (rec or {}).get("replica")
        if not isinstance(replica, str) or not replica:
            raise ValueError(f"launch {name}: its store names no replica (copy id)")
        out[_launch_out(runs_dir, name)] = replica
    return out


def load_copies(path: Path) -> dict[str, str]:
    """OUT path -> copy id from the operator's file ({"outs": {...}} or a flat {OUT: copy})."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    outs = doc.get("outs", doc) if isinstance(doc, dict) else None
    if not isinstance(outs, dict) or not outs or not all(isinstance(k, str) and isinstance(v, str) and v
                                                         for k, v in outs.items()):
        raise ValueError(f"{path}: expected {{\"outs\": {{OUT path: copy id}}}}")
    return {k: v for k, v in outs.items() if k != "schema"}


# --- the trigger ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class StoreRef:
    """A copy's frozen reference: its time-out counts (dev, or an a15 copy's qualification rate sample) and factor."""

    copy: str
    timeouts: int
    queries: int
    factor: Fraction = STORE_FACTOR
    source: Any = None

    def __post_init__(self) -> None:
        if not (0 <= self.timeouts <= self.queries and self.queries >= 1):
            raise ValueError(f"copy {self.copy}: reference {self.timeouts}/{self.queries} is not a share")

    @property
    def share(self) -> Fraction:
        return Fraction(self.timeouts, self.queries)

    @property
    def trigger_share(self) -> Fraction:
        return self.factor * self.share


def refs_from_doc(copies: Mapping[str, Mapping[str, Any]]) -> dict[str, StoreRef]:
    """The frozen references, {copy: {"timeouts", "queries", "factor", "source"}}; a copy whose counts are not
    filled in yet ("[FILL]") is left out (unmonitored until it is)."""
    refs = {}
    for copy, c in copies.items():
        if not isinstance(c.get("timeouts"), int) or not isinstance(c.get("queries"), int):
            continue
        refs[copy] = StoreRef(copy, c["timeouts"], c["queries"], Fraction(str(c.get("factor", STORE_FACTOR))),
                              c.get("source"))
    return refs


def store_triggers(timeouts: int, queries: int, ref: StoreRef) -> bool:
    """The rule: at least STORE_MIN_TIMEOUTS time-outs, and a share that reaches the trigger share (exact)."""
    return timeouts >= STORE_MIN_TIMEOUTS and Fraction(timeouts, queries) >= ref.trigger_share


@dataclass(frozen=True)
class StoreBinResult:
    copy: str
    start: datetime
    timeouts: int
    queries: int
    share: float
    threshold: float
    triggered: bool

    def describe(self) -> dict[str, Any]:
        return {"copy": self.copy, "bin": self.start.isoformat(timespec="milliseconds"), "timeouts": self.timeouts,
                "queries": self.queries, "share": round(self.share, 6), "threshold": round(self.threshold, 6)}


def evaluate_store(evs: Iterable[StoreQueryEvents], refs: Mapping[str, StoreRef],
                   since: Mapping[str, datetime] | None = None) -> list[StoreBinResult]:
    """Every (copy, bin) with queries on a copy with a reference, judged by the rule; per copy only queries sent at or
    after ``since[copy]`` count (after a resume the old bins are not a new trigger)."""
    since = since or {}
    cells: dict[tuple[str, datetime], list[int]] = {}
    for e in evs:
        if e.copy not in refs:
            continue
        for sent, timed_out in e.queries:
            if e.copy in since and sent < since[e.copy]:
                continue
            c = cells.setdefault((e.copy, fault.bin_start(sent)), [0, 0])
            c[0] += timed_out
            c[1] += 1
    out = []
    for (copy, start), (k, n) in sorted(cells.items()):
        ref = refs[copy]
        out.append(StoreBinResult(copy, start, k, n, k / n, float(ref.trigger_share), store_triggers(k, n, ref)))
    return out


def first_store_triggers(results: Iterable[StoreBinResult]) -> dict[str, StoreBinResult]:
    """Per copy, its earliest triggering bin."""
    first: dict[str, StoreBinResult] = {}
    for r in sorted(results, key=lambda r: r.start):
        if r.triggered and r.copy not in first:
            first[r.copy] = r
    return first


def unmonitored_copies(evs: Iterable[StoreQueryEvents], refs: Mapping[str, StoreRef]) -> dict[str, dict[str, int]]:
    """Per copy with queries but no frozen reference, its attempts and queries: the rule cannot see a fault there."""
    out: dict[str, dict[str, int]] = {}
    for e in evs:
        if e.copy not in refs and e.queries:
            c = out.setdefault(e.copy, {"attempts": 0, "queries": 0})
            c["attempts"] += 1
            c["queries"] += len(e.queries)
    return out


def attempts_with_query_in_window(evs: Iterable[StoreQueryEvents], copy: str, start: datetime,
                                  end: datetime | None) -> list[str]:
    """The run ids of every attempt with a query on ``copy`` overlapping the window [start, end) (end None: open), in
    every arm and whatever its outcome: a query sent before the start that was still running at it counts."""
    hit = set()
    for e in evs:
        if e.copy != copy:
            continue
        for (sent, _), done in zip(e.queries, e.query_ends or [q[0] for q in e.queries]):
            # a query is [sent, done): it overlaps when it was sent before the end and finished after the start (an
            # instant query, done == sent, when it was sent in the window)
            if (end is None or sent < end) and (max(done, sent) > start or start <= sent):
                hit.add(e.run_id)
                break
    return sorted(hit)


# --- the store probe ------------------------------------------------------------------------------


def pick_probe_queries(gold_results: Iterable[Mapping[str, Any]], seed: str, n: int = STORE_PROBE_N,
                       split: str = "dev") -> list[str]:
    """The probe's ``n`` query ids: the ``split`` gold queries that completed ("answered") at the copy's identity check,
    in the order of sha256('<seed>|<id>') (outcome-blind beyond completing; a fixed seed fixes the set). Refused
    (ValueError) with fewer than ``n``."""
    done = sorted({str(r["id"]) for r in gold_results if r.get("split") == split and r.get("outcome") == "answered"},
                  key=lambda q: hashlib.sha256(f"{seed}|{q}".encode()).hexdigest())
    if len(done) < n:
        raise ValueError(f"only {len(done)} {split} gold queries completed at the identity check; the probe needs {n}")
    return done[:n]


def probe_set_doc(copy: str, store: str, seed: str, ids: list[str], sparql_by_id: Mapping[str, str],
                  source: Mapping[str, Any]) -> dict:
    """A copy's probe set: the ids in order and each gold query's sha256 (the text stays in the questions file, which a
    run reads and checks against it)."""
    return {"schema": STORE_PROBE_SET_SCHEMA, "copy": copy, "store": store, "seed": seed, "n": len(ids),
            "rule": STORE_RULE["probe"], "source": dict(source),
            "queries": [{"id": q, "sparql_sha256": hashlib.sha256(sparql_by_id[q].encode("utf-8")).hexdigest()}
                        for q in ids]}


def probe_set_bytes(doc: dict) -> bytes:
    """The probe set's file content, whose sha256 the frozen config binds."""
    return (json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def load_store_probe_set(path: Path, expected_sha256: str) -> dict:
    """The probe set at ``path``, refused (ValueError) unless its bytes have the bound sha256 and it is a set of
    STORE_PROBE_N distinct queries."""
    raw = Path(path).read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != expected_sha256:
        raise ValueError(f"{path}: sha256 {got}, not the bound {expected_sha256}")
    doc = json.loads(raw)
    ids = [q.get("id") for q in doc.get("queries") or []]
    if doc.get("schema") != STORE_PROBE_SET_SCHEMA or len(ids) != STORE_PROBE_N or len(set(ids)) != len(ids):
        raise ValueError(f"{path}: not a {STORE_PROBE_SET_SCHEMA} set of {STORE_PROBE_N} distinct queries")
    return doc


@dataclass(frozen=True)
class ProbeRow:
    """One probe request's result: "ok", "timeout" or "error" (an error still there after the re-sends)."""

    i: int
    outcome: str
    tries: int = 1
    wall_s: float | None = None
    error_kinds: tuple[str, ...] = field(default_factory=tuple)


def judge_store_probe(rows: Iterable[ProbeRow | Mapping[str, Any]], n: int = STORE_PROBE_N) -> dict[str, Any]:
    """The probe's verdict: it passes when all ``n`` requests were sent, at most STORE_PROBE_MAX_TIMEOUTS timed out
    and none still errored after its re-sends."""
    outcomes = [r.outcome if isinstance(r, ProbeRow) else r.get("outcome") for r in rows]
    k = outcomes.count("timeout")
    errors = outcomes.count("error")
    ok = outcomes.count("ok")
    passed = len(outcomes) == n and ok + k + errors == n and k <= STORE_PROBE_MAX_TIMEOUTS and errors == 0
    reasons = ([f"{len(outcomes)} requests, {n} expected"] if len(outcomes) != n else []) + (
        [f"{k} of {n} timed out (> {STORE_PROBE_MAX_TIMEOUTS})"] if k > STORE_PROBE_MAX_TIMEOUTS else []) + (
        [f"{errors} request(s) still failing after {STORE_PROBE_RESENDS} re-sends"] if errors else [])
    if not passed and not reasons:
        reasons = ["a request with no outcome"]
    return {"n": len(outcomes), "expected": n, "ok": ok, "timeouts": k, "errors": errors, "pass": passed,
            "verdict": "pass" if passed else "fail", "reasons": reasons, "rule": STORE_RULE["probe"]["fails"]}
