"""A store read through a shared query gate: its private description, its clients and its preflight.

The store description (the runner's ``--store-json``) of such a store is private: besides the public fields every
description has (``name``, ``engine``, ``version``, ...) it names, under ``private_files``, the PATHS of the private
files the client reads, never their contents:

    {"name": "<graph>", "engine": "Virtuoso", "endpoint": "<url>",
     "private_files": {"user": <path>, "password": <path>, "graph": <path>, "row_limit": <path>, "gate": <path>}}

``user`` and ``password`` hold the HTTP digest credentials, ``graph`` the one data graph IRI, ``row_limit`` the
endpoint's own row limit (one positive integer) and ``gate`` the gate file (harness/query_gate.py) every client of the
store shares. Every path, the description's own included, is outside the repository and readable by its owner only (a
gate file not yet created: its directory). A description with ``private_files`` must name a Virtuoso engine, and a
Virtuoso description must have them: such a store is never read with the plain client. A record keeps only the
sanitised description (harness/agent/metadata.py drops ``private_files``). The repository holds a fake example
(stores/a15/store-description.example.json) and stores/a15/write_store_description.py, which writes the real one.

``client_factory`` makes one VirtuosoClient per call (the runner calls it per try, so no two attempts share a client's
state): the data graph sent as default-graph-uri and named-graph-uri on every request, the client time-out and the
server time-out of twice it in milliseconds, the row limit, and one shared ``QueryGate`` that each request waits for
(up to ``gate_wait_s``).

A description may also set ``query_gate``, the gate's size and its step-down rule (harness/gate_step_down.py): how many
queries may run at once (``slots``), the lower sizes it steps down to, and the time-out baseline the rule compares with.
Without it the gate has one slot, as before. It is a run setting, not part of what the store is: ``public_description``
leaves it out of what a run fingerprints and records, and the runner records it per start instead.

A description may also set ``query_timeout``, its clients' time-outs (a per-store setting, change-log entry 113):
``{"client_s": C, "refuse_from_s": R, "server_s": S}`` with 0 < R < C < S <= 280: the client gives up at C seconds,
takes no answer that arrives from R seconds on, and sends S as the server's time-out, so the server's cut still comes
only after the client has stopped waiting. Without it: 60, 57 and 120, as before. The run's ``--sparql-timeout`` must
be C; the client's settings (and so a run's fingerprint and the prompt's time-out) carry the values.
"""

from __future__ import annotations

import fcntl
import json
import os
import stat
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.gate_step_down import GateSetting, parse_setting
from harness.query_gate import QueryGate, slot_path
from harness.sparql import SparqlError, VirtuosoClient, read_result_cap

PRIVATE_FILES = ("user", "password", "graph", "row_limit", "gate")
GATE_SETTING_KEY = "query_gate"
TIMEOUT_SETTING_KEY = "query_timeout"
TIMEOUT_KEYS = frozenset({"client_s", "refuse_from_s", "server_s"})
MAX_SERVER_TIMEOUT_S = 280.0
RUNNER_LOCK_SUFFIX = ".runner.lock"  # held by the one runner that steps a gate of several slots down
# The longest a query waits for the gate before its attempt fails infrastructure. One query holds the gate for at
# most its client time-out plus the wait-out of the server time-out (60 + 65 s), and an agent query that times out
# holds it twice with a probe between (about 260 s; harness/agent/infra.py), so this covers about six sessions queued
# ahead of it.
GATE_WAIT_S = 1800.0
# A phase whose whole time, waits included, reaches this many times its wall cap while the agent's own time is under
# its cap ends failed:infrastructure (design rulings, design log, 20 Oct).
TOTAL_CAP_FACTOR = 3.0
COUNT_QUERY = "SELECT (COUNT(*) AS ?n) WHERE { ?s ?p ?o }"  # scoped to the data graph by default-graph-uri
# The client time-out of every query (design log rules): the server's is twice it, 120 s.
CLIENT_TIMEOUT_S = 60.0


class DescriptionError(ValueError):
    """The store description, or a private file it names, cannot be used."""


@dataclass(frozen=True)
class QueryTimeout:
    client_s: float
    refuse_from_s: float
    server_s: float

    def to_json(self) -> dict:
        return {"client_s": self.client_s, "refuse_from_s": self.refuse_from_s, "server_s": self.server_s}


def parse_timeout(obj: Any) -> QueryTimeout:
    """A description's ``query_timeout``, checked (DescriptionError otherwise)."""
    def number(v: Any) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and v not in (float("inf"),)

    if not isinstance(obj, dict) or set(obj) != TIMEOUT_KEYS or not all(number(obj[k]) for k in TIMEOUT_KEYS):
        raise DescriptionError(f"{TIMEOUT_SETTING_KEY} must give exactly {', '.join(sorted(TIMEOUT_KEYS))} as numbers")
    t = QueryTimeout(float(obj["client_s"]), float(obj["refuse_from_s"]), float(obj["server_s"]))
    if not 0 < t.refuse_from_s < t.client_s < t.server_s <= MAX_SERVER_TIMEOUT_S:
        raise DescriptionError(f"{TIMEOUT_SETTING_KEY} needs 0 < refuse_from_s < client_s < server_s <= "
                               f"{MAX_SERVER_TIMEOUT_S:g}")
    return t


@dataclass(frozen=True)
class GatedStore:
    endpoint: str | None
    files: dict[str, Path]
    gate: GateSetting = GateSetting(1)
    timeout: QueryTimeout | None = None  # None: the clients' time-outs as before the setting (60, 57, 120)

    @property
    def slots(self) -> int:
        return self.gate.slots


def is_virtuoso(store: dict) -> bool:
    engine = store.get("engine")
    return isinstance(engine, str) and engine.strip().lower().startswith("virtuoso")


def gated_store(store: dict) -> GatedStore | None:
    """The private part of a description of a store read through a query gate; None for any other store (the plain
    client's). Checks the shape only; ``check_private`` checks the files."""
    files = store.get("private_files")
    if files is None:
        for key in (GATE_SETTING_KEY, TIMEOUT_SETTING_KEY):
            if key in store:
                raise DescriptionError(f"{key} is for a store read through a query gate (private_files)")
        if is_virtuoso(store):
            raise DescriptionError("a Virtuoso store needs private_files (its credentials, graph, row limit and "
                                   "gate files): it is never read with the plain client")
        return None
    if not is_virtuoso(store):
        raise DescriptionError("private_files are for a Virtuoso store read through a query gate")
    if not isinstance(files, dict) or set(files) != set(PRIVATE_FILES) or not all(
            isinstance(v, str) and v for v in files.values()):
        raise DescriptionError(f"private_files must name exactly {', '.join(PRIVATE_FILES)}, each by its path")
    endpoint = store.get("endpoint")
    if endpoint is not None and not (isinstance(endpoint, str) and endpoint.startswith(("http://", "https://"))):
        raise DescriptionError("the description's endpoint must be an http(s) URL")
    setting = GateSetting(1)
    if GATE_SETTING_KEY in store:
        try:
            setting = parse_setting(store[GATE_SETTING_KEY])
        except ValueError as exc:
            raise DescriptionError(str(exc)) from None
    timeout = parse_timeout(store[TIMEOUT_SETTING_KEY]) if TIMEOUT_SETTING_KEY in store else None
    return GatedStore(endpoint, {k: Path(files[k]).expanduser() for k in PRIVATE_FILES}, setting, timeout)


def client_timeout_s(gs: GatedStore) -> float:
    """The client time-out a run on the store uses (its ``--sparql-timeout``)."""
    return gs.timeout.client_s if gs.timeout is not None else CLIENT_TIMEOUT_S


def public_description(store: dict) -> dict:
    """The description without its gate and time-out settings: what a run fingerprints and its records keep of the
    store (the time-outs are in the run's client settings instead)."""
    return {k: v for k, v in store.items() if k not in (GATE_SETTING_KEY, TIMEOUT_SETTING_KEY)}


def owner_only(path: Path) -> bool:
    st = path.stat()
    return st.st_uid == os.getuid() and not st.st_mode & (stat.S_IRWXG | stat.S_IRWXO)


def check_private_path(path: Path, repo_root: Path, what: str, *, may_be_missing: bool = False) -> None:
    """``path`` is outside the repository and readable by its owner only; a file that ``may_be_missing`` (the gate,
    created at its first query) and is missing needs an owner-only directory."""
    path = Path(path).expanduser()
    if path.resolve().is_relative_to(Path(repo_root).resolve()):
        raise DescriptionError(f"{what} is inside the repository; private files live outside it")
    if not path.exists():
        if may_be_missing and path.parent.is_dir() and owner_only(path.parent):
            return
        raise DescriptionError(f"{what} does not exist" + (" (nor an owner-only directory for it)"
                                                          if may_be_missing else ""))
    if not owner_only(path):
        raise DescriptionError(f"{what} must be readable by its owner only (mode 600)")


def check_private(gs: GatedStore, repo_root: Path) -> None:
    """Every private file is where it should be and private; the graph file names one IRI and the row-limit file holds
    one positive integer. Nothing read is printed or returned."""
    for key in PRIVATE_FILES:
        check_private_path(gs.files[key], repo_root, f"private_files.{key}", may_be_missing=key == "gate")
    gate = gs.files["gate"]
    lock = Path(f"{gate}.lock")
    if lock.exists() and not owner_only(lock):
        raise DescriptionError("private_files.gate's lock file must be readable by its owner only")
    for slot in range(1, gs.slots):  # a gate of several slots: each slot's reservation and lock file, once made
        for path in (Path(slot_path(str(gate), slot)), Path(slot_path(str(gate), slot) + ".lock")):
            if path.exists() and not owner_only(path):
                raise DescriptionError(f"private_files.gate's slot {slot} files must be readable by their owner only")
    if len(gs.files["graph"].read_text(encoding="utf-8").split()) != 1:
        raise DescriptionError("private_files.graph must hold exactly one IRI")
    try:
        read_result_cap(gs.files["row_limit"])
    except ValueError:
        raise DescriptionError("private_files.row_limit must hold one positive integer") from None


def client_factory(gs: GatedStore, *, endpoint: str, timeout_s: float, row_cap: int | None,
                   gate_wait_s: float = GATE_WAIT_S, gate: QueryGate | None = None, monitor: Any = None,
                   ) -> tuple[Callable[[Any], VirtuosoClient], QueryGate]:
    """A factory of new clients of the store, all on one gate, and that gate (whose ``stats`` give the busy share).
    ``monitor``: the step-down rule of a gate of several slots (gate_step_down.TimeoutStepDown), told by each client
    what its tool calls' requests met."""
    gate = gate if gate is not None else QueryGate(gs.files["gate"])
    extra = {"gate_monitor": monitor} if monitor is not None else {}
    if gs.timeout is not None:  # the store's own time-outs; without them the client's defaults, as before
        if timeout_s != gs.timeout.client_s:
            raise DescriptionError(f"the store's clients time out at {gs.timeout.client_s:g} s, not {timeout_s:g} s")
        extra.update(server_timeout_s=gs.timeout.server_s, refuse_from_s=gs.timeout.refuse_from_s,
                     wall_timeout_share=gs.timeout.refuse_from_s / gs.timeout.client_s)  # the share: recorded only

    def make(_args: Any = None) -> VirtuosoClient:
        return VirtuosoClient.from_files(endpoint, gs.files["user"], gs.files["password"], gs.files["graph"],
                                         gs.files["row_limit"], name_the_graph=True, timeout_s=timeout_s,
                                         row_cap=row_cap, gate=gate, gate_wait_s=gate_wait_s, **extra)

    return make, gate


def lock_gate_runner(gs: GatedStore) -> int:
    """Take the gate's runner lock (``<gate>.runner.lock``, owner-only) for the life of a run that steps the gate down,
    so one runner at a time does; its descriptor, which the caller closes at the end. DescriptionError when another
    process holds it."""
    path = Path(f"{gs.files['gate']}{RUNNER_LOCK_SUFFIX}")
    if path.exists() and not owner_only(path):
        raise DescriptionError("private_files.gate's runner lock must be readable by its owner only")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise DescriptionError("another runner holds this store's gate of several slots: one runner at a time steps "
                               "it down") from None
    except BaseException:
        os.close(fd)
        raise
    return fd


def close_client(client: Any) -> None:
    """Close a client made for one try (a client without ``close``, the shared plain one, is left open)."""
    close = getattr(client, "close", None)
    if callable(close):
        close()


def append_private(path: Path, line: dict) -> None:
    """Append one JSON line to a private log, created mode 600."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(line, sort_keys=True) + "\n")
    os.chmod(path, 0o600)


def preflight_count(make: Callable[[Any], VirtuosoClient], log: Path, *, now: Callable[[], str]) -> bool:
    """Read the store's triple count (the data graph's, one query through the client and its gate) and append it, with
    the query's timings, to the private ``log``; nothing of it is returned or printed. True when the count was read."""
    client = make(None)
    started = time.time()
    entry: dict[str, Any] = {"at": now(), "query": "count", "client": client.settings()}
    try:
        result = client.query(COUNT_QUERY)
        entry.update(ok=True, count=int(result.rows[0][0].lexical))
    except SparqlError as exc:  # kind and status only: an error body can echo the store's content
        entry.update(ok=False, error={"kind": exc.kind, "status": exc.status})
    except (IndexError, ValueError, AttributeError, TypeError) as exc:
        entry.update(ok=False, error={"kind": "unreadable_answer", "exception": type(exc).__name__})
    finally:
        close_client(client)
    entry["timings"] = client.timings[-1] if client.timings else None
    entry["elapsed_s"] = round(time.time() - started, 3)
    append_private(log, entry)
    return bool(entry["ok"])
