"""Change-log entry 189 item 4: each turn's and each attempt's deployment, from infrastructure fields only (the design
approved in the design log:05Z).

- A fronted turn (its record's route has ``front_url``: Gemma's Tier 1 after its switch) joins its response id to the
  e167 front's provenance log (results/e167-front/join.py's reader, imported from the code root): its backend is its
  deployment. Fingerprints are a counted cross-check only (gemma-1 and gemma-3 share one with the router).
- A pre-front turn of a model with admitted endpoints (OUT/admitted-fingerprints.json) is attributed by the sha256 of
  its record's pinned route URL (results/e167-admission/admit.py's url_sha256) against the admitted backends' URLs.
- A turn of an unfronted route (Anthropic, the Gateway, SiteA GLM, the SiteB stacks) is attributed to (route-URL hash,
  served fingerprint); a router is one deployment.
- An attempt is served by d only if all its own turns were; otherwise it is ``mixed``, or ``unattributed`` when an own
  turn has no deployment (unjoined, ambiguous, no id, an unknown URL), or ``no own turns``. A branch record's own turns
  exclude its parent's replayed initial phase (run_pilot.live_turns: turns after checker.initial.outcome.turns).
- A model's reference deployment is its earliest admitted one (its route's first-turn deployment when it has no
  admitted endpoints), and it must have served the OUT's first turn of the model (OUT/served-models.json: the first
  turn naming a served model or carrying a fingerprint), or the tool refuses (ReferenceRefused); so it does when the
  tries that may hold that turn disagree on its deployment, or its fingerprint is not the one the file implies.

Blinding (entry 186): ``read_record`` reads only the fields in PRE_CALL_FIELDS, plus SEALED_FIELDS when ``sealed`` is
given, which only the post-call tool does (an arm, a branch's parent and its replayed length). Never an outcome, score,
cap, looping or checker verdict field; never a runner log (runs.jsonl)."""

from __future__ import annotations

import collections
import dataclasses
import gzip
import importlib.util
import json
import sys
import zlib
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

DAMAGED = (OSError, ValueError, EOFError, zlib.error)
RECORD_DIRS = ("records", "aborted", "infrastructure", "harness")  # join.py's: the runner's record and set-aside dirs
ATTEMPT_DIR = "records"  # the analysis reads its attempts (and a grid's branch records) from records/ only
SET_ASIDE_DIRS = ("infrastructure", "harness")
SIDECAR_SUFFIX = ".setaside.json"  # harness/'s set-aside decisions: never read

# the record fields each mode may read (paths; [*] is every list element), checked by the tests with a tracking dict
PRE_CALL_FIELDS = ("run_id", "model", "question_id", "repetition", "started_at", "route.base_url", "route.front_url",
                   "transcript[*].role", "transcript[*].turn", "transcript[*].response_id",
                   "tokens.per_turn[*].turn", "tokens.per_turn[*].system_fingerprint",
                   "tokens.per_turn[*].served_model")
SEALED_FIELDS = ("condition", "load.arm", "checker.branched_from", "checker.initial.outcome.turns")

MIXED = "mixed"
UNATTRIBUTED = "unattributed"
NO_OWN_TURNS = "no own turns"
NOT_DEPLOYMENTS = (MIXED, UNATTRIBUTED, NO_OWN_TURNS)


class AttributionError(ValueError):
    pass


class ReferenceRefused(AttributionError):
    """The earliest admitted deployment did not serve the OUT's first turn of the model, or that turn is not found."""


# --- the code root's modules (the frozen deploy's join.py and admit.py, never a copy) --------------------------------


def _load(name: str, path: Path) -> ModuleType:
    if not path.is_file():
        raise AttributionError(f"the code root has no {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclasses.dataclass(frozen=True)
class CodeRoot:
    root: Path
    join: ModuleType
    admit: ModuleType

    @classmethod
    def load(cls, root: Path) -> CodeRoot:
        root = Path(root).resolve()
        return cls(root, _load("e189_join", root / "results" / "e167-front" / "join.py"),
                   _load("e189_admit", root / "results" / "e167-admission" / "admit.py"))

    def url_sha256(self, url: str) -> str | None:
        try:
            return self.admit.url_sha256(url)
        except Exception:  # noqa: BLE001 - admit.Failed: not an http(s) URL with a host
            return None


# --- records, infrastructure fields only ------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Turn:
    turn: Any  # the transcript turn's number (the per-turn tokens' "turn")
    response_id: Any
    fingerprint: str | None  # tokens.per_turn's system_fingerprint at this turn number (None: none sent, or no entry)
    served_model: str | None = None  # tokens.per_turn's served_model at this turn number


@dataclasses.dataclass(frozen=True)
class RecordInfo:
    name: str  # <dir>/<file>
    run_id: Any
    model: Any  # the route model
    question_id: Any
    repetition: Any
    started_at: Any
    base_url: Any
    front_url: Any
    turns: tuple[Turn, ...]  # every assistant turn, in order
    sealed: bool = False
    cohort: Any = None  # condition (sealed mode)
    cell: Any = None  # load.arm (sealed mode)
    branched_from: Any = None  # checker.branched_from (sealed mode)
    initial_turns: Any = None  # checker.initial.outcome.turns (sealed mode, branch records only)

    @property
    def directory(self) -> str:
        return self.name.split("/", 1)[0]

    @property
    def fronted(self) -> bool:
        return bool(self.front_url)


def _get(d: Any, key: str) -> Any:
    return d.get(key) if isinstance(d, dict) else None


def read_record(data: Any, name: str, *, sealed: bool = False) -> RecordInfo:
    """The record's infrastructure fields (PRE_CALL_FIELDS; with ``sealed`` also SEALED_FIELDS), by ``get`` only."""
    route = _get(data, "route")
    per_turn = _get(_get(data, "tokens"), "per_turn")
    served: dict[Any, tuple[Any, Any]] = {}  # turn number -> (fingerprint, served model)
    for t in per_turn if isinstance(per_turn, list) else []:
        served.setdefault(_get(t, "turn"), (_get(t, "system_fingerprint"), _get(t, "served_model")))
    transcript = _get(data, "transcript")
    turns = tuple(Turn(_get(t, "turn"), _get(t, "response_id"), *served.get(_get(t, "turn"), (None, None)))
                  for t in (transcript if isinstance(transcript, list) else []) if _get(t, "role") == "assistant")
    info = {"name": name, "run_id": _get(data, "run_id"), "model": _get(data, "model"),
            "question_id": _get(data, "question_id"), "repetition": _get(data, "repetition"),
            "started_at": _get(data, "started_at"), "base_url": _get(route, "base_url"),
            "front_url": _get(route, "front_url"), "turns": turns}
    if sealed:
        checker = _get(data, "checker")
        branched = _get(checker, "branched_from")
        info.update(sealed=True, cohort=_get(data, "condition"), cell=_get(_get(data, "load"), "arm"),
                    branched_from=branched,
                    initial_turns=_get(_get(_get(checker, "initial"), "outcome"), "turns") if branched else None)
    return RecordInfo(**info)


def _read_json(path: Path) -> Any:
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.name.endswith(".gz") else raw)


def record_paths(out: Path, dirs: Iterable[str] = RECORD_DIRS) -> list[Path]:
    return [p for sub in dirs for p in sorted((Path(out) / sub).glob("*"))
            if p.is_file() and (p.name.endswith(".json") or p.name.endswith(".json.gz"))
            and not p.name.removesuffix(".gz").endswith(SIDECAR_SUFFIX)]


def read_out(out: Path, *, dirs: Iterable[str] = RECORD_DIRS, sealed: bool = False) -> tuple[list[RecordInfo], list[str]]:
    """(the records' infrastructure fields, the names of record files that are not readable JSON)."""
    rows, unreadable = [], []
    for path in record_paths(out, dirs):
        name = f"{path.parent.name}/{path.name}"
        try:
            data = _read_json(path)
        except DAMAGED:
            unreadable.append(name)
            continue
        rows.append(read_record(data, name, sealed=sealed))
    return rows, unreadable


def own_turns(rec: RecordInfo) -> tuple[Turn, ...]:
    """The turns this attempt asked the model for: every turn, but in a branch record only those after its parent's
    replayed initial phase (run_pilot.live_turns)."""
    if not rec.branched_from:
        return rec.turns
    if not rec.sealed:
        raise AttributionError(f"{rec.name}: a branch record's own turns need the sealed fields (post-call only)")
    n = rec.initial_turns
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise AttributionError(f"{rec.name}: a branch record without its parent's replayed length")
    return tuple(t for t in rec.turns if isinstance(t.turn, int) and not isinstance(t.turn, bool) and t.turn > n)


# --- an OUT's infrastructure files --------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Admitted:
    backend: str
    at: str
    url_sha256: str | None
    fingerprints: frozenset[str]


@dataclasses.dataclass
class OutInfo:
    """An OUT's route models (runner name -> route model), admitted endpoints by route model, and first served turns."""

    path: Path
    route_models: dict[str, str]
    admitted: dict[str, list[Admitted]]  # route model -> its admitted endpoints
    first_turns: dict[str, dict]  # route model -> served-models.json's entry

    @classmethod
    def read(cls, out: Path, mixed_separator: str = " | ") -> OutInfo:
        out = Path(out)
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        routes = ((manifest.get("fingerprint") or {}).get("routes")) or {}
        route_models = {name: r.get("model") for name, r in routes.items() if isinstance(r, dict)}
        admitted: dict[str, list[Admitted]] = {}
        path = out / "admitted-fingerprints.json"
        if path.exists():
            doc = json.loads(path.read_text(encoding="utf-8"))
            for name, entry in (doc.get("models") or {}).items():
                if name not in route_models:
                    raise AttributionError(f"{path}: model {name!r} is not one of the OUT's routes")
                rows = []
                for a in entry.get("admitted") or []:
                    fp = a.get("system_fingerprint")
                    if not a.get("backend") or not a.get("admitted_at") or not isinstance(fp, str) or not fp:
                        raise AttributionError(f"{path}: {name}: an admitted entry without a backend, a time or a "
                                               f"fingerprint")
                    rows.append(Admitted(a["backend"], a["admitted_at"], a.get("backend_url_sha256"),
                                         frozenset(fp.split(mixed_separator))))
                admitted[route_models[name]] = rows
        first: dict[str, dict] = {}
        served = out / "served-models.json"
        if served.exists():
            for name, entry in (json.loads(served.read_text(encoding="utf-8")).get("models") or {}).items():
                first[entry.get("route_model") or route_models.get(name)] = entry
        return cls(out, route_models, admitted, first)


# --- attribution ----------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Attribution:
    deployment: str | None  # None: no deployment (see ``how``)
    how: str  # joined | pre_front_url | route | unjoined | ambiguous | without_id | inherited | url_unmatched |
    # url_ambiguous | url_missing
    fingerprint_mismatch: bool = False  # a joined or URL-attributed turn whose fingerprint its backend never admitted


def route_deployment(url_sha256: str | None, fingerprint: str | None) -> str:
    return f"url:{url_sha256[:16] if url_sha256 else '-'}|fp:{fingerprint if fingerprint is not None else 'none'}"


class Attributor:
    """Each turn's deployment for one OUT (``info``), with the front's provenance log (``log``: join.read_log's
    result, or None when no record of the OUT went through a front)."""

    def __init__(self, code: CodeRoot, info: OutInfo, log: Mapping[str, Any] | None,
                 pre_front_ids: Iterable[str] = ()):
        self.code, self.info, self.log = code, info, log
        self.pre_front_ids = set(pre_front_ids)
        self.backend_fps: dict[str, dict[str, frozenset[str]]] = {}
        for model, rows in info.admitted.items():
            fps: dict[str, set[str]] = collections.defaultdict(set)
            for a in rows:
                fps[a.backend] |= a.fingerprints
            self.backend_fps[model] = {b: frozenset(v) for b, v in fps.items()}

    def _mismatch(self, model: Any, backend: str, fp: str | None) -> bool:
        admitted = self.backend_fps.get(model, {}).get(backend)
        return admitted is not None and fp not in admitted

    def turn(self, rec: RecordInfo, t: Turn) -> Attribution:
        if rec.fronted:
            if self.log is None:
                raise AttributionError(f"{rec.name}: a fronted record, and no provenance log was given")
            rid = self.code.join.id_key(t.response_id)
            if rid is None:
                return Attribution(None, "without_id")
            found = self.log["backends"].get(rid, set())
            if len(found) == 1:
                (backend,) = found
                return Attribution(backend, "joined", self._mismatch(rec.model, backend, t.fingerprint))
            if found:
                return Attribution(None, "ambiguous")
            return Attribution(None, "inherited" if rid in self.pre_front_ids else "unjoined")
        sha = self.code.url_sha256(rec.base_url) if isinstance(rec.base_url, str) else None
        admitted = self.info.admitted.get(rec.model)
        if admitted:
            if sha is None:
                return Attribution(None, "url_missing")
            backends = {a.backend for a in admitted if a.url_sha256 == sha}
            if len(backends) == 1:
                (backend,) = backends
                return Attribution(backend, "pre_front_url", self._mismatch(rec.model, backend, t.fingerprint))
            return Attribution(None, "url_ambiguous" if backends else "url_unmatched")
        if sha is None:
            return Attribution(None, "url_missing")
        return Attribution(route_deployment(sha, t.fingerprint), "route")


@dataclasses.dataclass(frozen=True)
class AttemptAttribution:
    served_by: str  # a deployment, or one of NOT_DEPLOYMENTS
    own: tuple[Attribution, ...]


def attempt(rec: RecordInfo, attributor: Attributor) -> AttemptAttribution:
    own = tuple(attributor.turn(rec, t) for t in own_turns(rec))
    if not own:
        return AttemptAttribution(NO_OWN_TURNS, own)
    if any(a.deployment is None for a in own):
        return AttemptAttribution(UNATTRIBUTED, own)
    deps = {a.deployment for a in own}
    return AttemptAttribution(deps.pop() if len(deps) == 1 else MIXED, own)


def pre_front_ids(code: CodeRoot, records: Iterable[RecordInfo]) -> set[str]:
    """The response ids of records without front_url (join.py's "inherited" test)."""
    return {k for r in records if not r.fronted for k in (code.join.id_key(t.response_id) for t in r.turns)
            if k is not None}


def read_log(code: CodeRoot, paths: Iterable[Path]) -> dict[str, Any] | None:
    paths = list(paths)
    return code.join.read_log(paths) if paths else None


# --- the reference deployment ----------------------------------------------------------------------------------------


def _when(stamp: Any) -> datetime | None:
    try:
        when = datetime.fromisoformat(stamp) if isinstance(stamp, str) else None
    except ValueError:
        return None
    return when.replace(tzinfo=UTC) if when is not None and when.tzinfo is None else when


def first_turns(model: Any, records: Iterable[RecordInfo], entry: Mapping[str, Any],
                mixed_separator: str = " | ") -> list[tuple[RecordInfo, Turn]]:
    """The candidates for the OUT's first served turn of ``model``: served-models.json's entry is written at the first
    turn that names a served model or carries one fingerprint (run_pilot ServedModels.check), with that turn's question
    and repetition, at ``at``. Each record of that attempt (set-aside tries included, branch records not) started by
    ``at`` gives its first such turn; several tries can, and the caller refuses them unless they agree."""
    at = _when(entry.get("at"))
    found = []
    for r in records:
        if r.model != model or r.question_id != entry.get("question_id") or r.repetition != entry.get("repetition") \
                or r.branched_from or at is None or (start := _when(r.started_at)) is None or start > at:
            continue
        turn = next((t for t in r.turns if t.served_model or (t.fingerprint and mixed_separator not in t.fingerprint)),
                    None)
        if turn is not None:
            found.append((r, turn))
    if not found:
        raise ReferenceRefused(f"{model}: no record holds the OUT's first served turn ({entry.get('question_id')} "
                               f"rep {entry.get('repetition')} at {entry.get('at')})")
    return sorted(found, key=lambda rt: rt[0].name)


def _first_fingerprint_problem(turn: Turn, entry: Mapping[str, Any]) -> str | None:
    """Whether the first turn's fingerprint is one served-models.json allows (run_pilot ServedModels.check and
    record_deployment_changes): the model's first fingerprinted turn sets the baseline, so the first turn carries the
    baseline or none. The baseline is the entry's fingerprint, unless a recorded deployment change moved it (then the
    earliest change's from_system_fingerprint, and the entry's system_fingerprint_question_id is the change's); without
    a change, system_fingerprint_question_id means a later turn set it, so the first turn carried none."""
    changes = sorted((c for c in entry.get("deployment_changes") or [] if isinstance(c, dict)),
                     key=lambda c: str(c.get("at")))
    if changes:
        allowed = {None, changes[0].get("from_system_fingerprint")}
    elif "system_fingerprint" in entry and "system_fingerprint_question_id" not in entry:
        allowed = {entry["system_fingerprint"]}
    else:
        allowed = {None}
    if turn.fingerprint not in allowed:
        return (f"the first turn's fingerprint {turn.fingerprint!r} is not "
                f"{' or '.join(repr(a) for a in sorted(allowed, key=str))} (served-models.json)")
    return None


def reference(model: Any, info: OutInfo, records: Iterable[RecordInfo], attributor: Attributor) -> str:
    """The model's reference deployment: its earliest admitted endpoint (its route's first-turn deployment without
    admitted endpoints), which must have served the OUT's first turn of the model; refused when the candidates for
    that turn (first_turns) disagree on its deployment or fingerprint, or when it has none."""
    entry = info.first_turns.get(model)
    if entry is None:
        raise ReferenceRefused(f"{model}: the OUT records no first served turn (served-models.json)")
    candidates = first_turns(model, list(records), entry, attributor.code.admit.MIXED)
    deployments = set()
    for rec, turn in candidates:
        first = attributor.turn(rec, turn)
        if first.deployment is None:
            raise ReferenceRefused(f"{model}: the OUT's first turn ({rec.name}) has no deployment ({first.how})")
        if (problem := _first_fingerprint_problem(turn, entry)) is not None:
            raise ReferenceRefused(f"{model}: {rec.name}: {problem}")
        deployments.add(first.deployment)
    if len(deployments) != 1:
        raise ReferenceRefused(f"{model}: the tries that may hold the OUT's first turn ({len(candidates)}) were served "
                               f"by different deployments, so which served it is ambiguous")
    (first_deployment,) = deployments
    admitted = info.admitted.get(model)
    if admitted:
        earliest = min(a.at for a in admitted)
        backends = {a.backend for a in admitted if a.at == earliest}
        if len(backends) != 1:
            raise ReferenceRefused(f"{model}: several endpoints were admitted first, at {earliest}")
        (ref,) = backends
    else:
        ref = first_deployment
    if first_deployment != ref:
        raise ReferenceRefused(f"{model}: the earliest admitted deployment {ref!r} did not serve the OUT's first turn "
                               f"({first_deployment!r} did)")
    return ref


# --- the pre-call totals-only check ----------------------------------------------------------------------------------

TOTAL_FIELDS = ("records", "unreadable_records", "turns", "front_records", "front_turns", "joined", "unjoined",
                "ambiguous", "without_id", "inherited", "pre_front_records", "pre_front_turns",
                "pre_front_url_admitted", "pre_front_url_unmatched", "pre_front_url_ambiguous", "pre_front_url_missing",
                "route_records", "route_turns", "route_url_hashed", "route_url_missing", "route_fingerprint_none",
                "fingerprint_cross_check_mismatch")


def totals(code: CodeRoot, out: Path, logs: Iterable[Path] = (), parents: Iterable[Path] = ()) -> dict[str, Any]:
    """One OUT's attribution totals over every record (set-aside ones included), phase and arm: no per-arm, per-phase
    or per-backend figure. Fronted turns as join.py counts them (its own report is attached for the cross-check),
    pre-front turns by URL hash against the admitted endpoints, unfronted routes' turns by (URL hash, fingerprint);
    plus each model's reference check (ok, or why it refuses)."""
    out = Path(out)
    info = OutInfo.read(out, code.admit.MIXED)
    records, unreadable = read_out(out)
    parent_records = [r for p in parents for r in read_out(Path(p))[0]]
    logs = list(logs)
    attributor = Attributor(code, info, read_log(code, logs), pre_front_ids(code, [*records, *parent_records]))
    c: collections.Counter = collections.Counter(records=len(records) + len(unreadable),
                                                 unreadable_records=len(unreadable))
    deployments = set()
    none_seen: dict[Any, set[bool]] = collections.defaultdict(set)  # route model -> {fingerprint is None}
    for rec in records:
        c["turns"] += len(rec.turns)
        kind = "front" if rec.fronted else "pre_front" if info.admitted.get(rec.model) else "route"
        c[f"{kind}_records"] += 1
        c[f"{kind}_turns"] += len(rec.turns)
        for t in rec.turns:
            a = attributor.turn(rec, t)
            c["fingerprint_cross_check_mismatch"] += a.fingerprint_mismatch
            if kind == "front":
                c[a.how] += 1
            elif kind == "pre_front":
                c[{"pre_front_url": "pre_front_url_admitted"}.get(a.how, f"pre_front_{a.how}")] += 1
            else:
                c["route_url_hashed" if a.deployment else "route_url_missing"] += 1
                c["route_fingerprint_none"] += t.fingerprint is None
                none_seen[rec.model].add(t.fingerprint is None)
                if a.deployment:
                    deployments.add(a.deployment)
    report: dict[str, Any] = {"out": str(out), **{k: c[k] for k in TOTAL_FIELDS},
                              "route_deployments": len(deployments),
                              # a route model with turns both with and without a fingerprint: its null-fingerprint
                              # turns are a deployment of their own (route_deployment), so they would show as such
                              "route_models_with_and_without_fingerprint": sum(len(v) == 2 for v in none_seen.values())}
    refs = {}
    for model in sorted({r.model for r in records if isinstance(r.model, str)}):
        try:
            reference(model, info, records, attributor)
            refs[model] = "ok"
        except ReferenceRefused as exc:
            refs[model] = f"refused: {exc}"
    report["reference_check"] = refs
    if c["front_records"]:  # join.py's own totals on the same OUT, for the cross-check (it must agree)
        j, clean = code.join.join(logs, [out], [Path(p) for p in parents], balance=False)
        report["join_py"] = {"clean": clean, **{k: j["outs"][0][k] for k in (
            "records", "front_records", "turns", "joined", "unjoined", "inherited", "ambiguous", "without_id",
            "pre_front_records", "pre_front_turns", "unreadable_records")}}
    report["clean"] = not (c["unjoined"] or c["ambiguous"] or c["without_id"] or c["unreadable_records"]
                           or c["pre_front_url_unmatched"] or c["pre_front_url_ambiguous"] or c["pre_front_url_missing"]
                           or c["route_url_missing"]) and all(v == "ok" for v in refs.values())
    return report
