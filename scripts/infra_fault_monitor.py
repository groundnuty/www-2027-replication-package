#!/usr/bin/env python3
"""The infrastructure-fault monitor of the test runs (change-log entries 129, 148 and 149, and the amendment of 18:21;
science, design log, 21 Oct; pre-registration section 5), in hold-only mode with the frozen fault probe. The trigger and
its definitions are harness/agent/infra_fault.py's, the probe harness/agent/fault_probe.py's; the frozen rates, probe
sets and thresholds are the fault-rule file's (data/runs/infra-fault/fault-rule.json), loaded only when its sha256 is
the frozen config's pin (data/freeze/frozen-config.json ``fault_rule``).

    uv run python scripts/infra_fault_monitor.py scan --state DIR [--every S] [--probe-cmd TEMPLATE] \\
        [--probe-every S] [--rule FILE --pin FILE] OUT [OUT ...]
    uv run python scripts/infra_fault_monitor.py confirm --state DIR --model NAME --by provider|fingerprint \\
        --note TEXT [--at ISO] OUT [OUT ...]
    uv run python scripts/infra_fault_monitor.py window --state DIR --model NAME OUT [OUT ...]

``scan`` reads every record of the OUTs (read-only: it writes only the hold files and DIR) and pools each model's
attempts across them, since a fault is the provider's.

1. TRIGGER, HOLD ONLY: a model whose share of call-less turns or of attempts without a submission, in a clock-aligned
   15-minute bin of UTC, passes ten times its frozen dev rate with at least five events is HELD: OUT/holds/<model>.json
   in every OUT (atomically; the runner, scripts/run_pilot.py, reads it before each claim of the model and starts no
   attempt of it, those in flight go on), the hold logged to DIR/infra-faults.jsonl, an ALERT line. Nothing is
   relabelled. A hold stays, across runner restarts and a hold file removed by hand (the next scan writes it again),
   until the confirmation decides.
2. THE PROBE, AT ONCE: the same scan starts the model's frozen fault probe on its route (``--probe-cmd``, a template
   with {model}, {dir} and {manifest}: the first OUT whose manifest routes the model; without it the scan prints the
   command to run by hand into that directory). Each later scan reads the directory's summary.json.
3. CONFIRMED: the probe fails, or its replies show a served fingerprint other than the recorded ones (the served one
   and those admitted beside it, change-log entry 167; fingerprint_change), or an operator records a
   provider-confirmed restart or incident or a fingerprint change (``confirm``). The window runs from the first
   triggering bin's start (moved earlier when late records show an earlier triggering bin) until the probe passes
   again: the model stays held and its probe runs again every ``--probe-every`` seconds (default 900); the first
   passing probe closes the window at its end and releases the model. Every attempt of the model with any model turn
   in the window, in every arm, is set aside as an infrastructure failure and re-run (``window`` prints the commands;
   entry 149).
4. NOT CONFIRMED: the first probe passes with no confirmatory event: the attempts stand, scored as usual, the hold is
   recorded and the model resumes (its hold files removed); from the probe's end it is watched again on attempts
   started since (the old bins are not a new trigger). An operator's ``confirm`` that arrives later (a provider's
   incident report) turns that hold into a closed window, from its start to the end of the probe that passed.
A probe that ends without a summary (it crashed, a key is missing) decides nothing: it is alerted and started again
after ``--probe-every``. ``scan`` exits 3 while any model is held, else 0; ``--every S`` scans again every S seconds.

The hold, not an alert alone, is kept because GLM's fault began at 05:45Z with the owner asleep; the probe, not the
test attempts, decides the window, because a trigger on attempts without a submission is a scored outcome and a hard
arm could otherwise open a window and have its own failures re-run (the freeze-day review, entry 148).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.agent import infra_fault as fault  # noqa: E402
from harness.agent import store_fault as sf  # noqa: E402
from harness.agent.providers import MIXED_FINGERPRINTS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = "infra-faults.jsonl"
PROBES_DIR = "probes"
PROBE_EVERY_S = 900.0
ADMITTED_FILE = "admitted-fingerprints.json"  # scripts/run_pilot.py's (change-log entry 167)
ADMITTED_SCHEMA = "admitted-fingerprints/1"
# the monitor's own interpreter (the deploy's .venv), so the probe runs the same code and environment
DEFAULT_PROBE_CMD = (shlex.quote(sys.executable)
                     + " scripts/fault_probe.py run --model {model} --manifest {manifest} --out {dir}")
CONFIRM_BY = ("provider", "fingerprint")
# DIR/watched-copies.json: every OUT -> copy the monitor has watched (its own record, never the operator's input)
STORE_COPIES_FILE = "watched-copies.json"
STORE_PREFIX = "store:"  # a store copy's holds are logged under the subject "store:<copy>"
STORE_PROBE_HINT = ("uv run python scripts/store_probe.py run --copy {copy} --set <its probe set> --sha256 <its bound "
                    "sha256> --store-json <its store description> --questions <its questions> --out {dir}")


def now() -> datetime:
    return datetime.now(UTC)


def stamp(t: datetime | None) -> str | None:
    return None if t is None else t.isoformat(timespec="milliseconds")


def _write_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=1, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def read_log(state: Path) -> list[dict]:
    path = state / LOG_FILE
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def append_log(state: Path, entry: dict) -> None:
    state.mkdir(parents=True, exist_ok=True)
    with (state / LOG_FILE).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def episodes(log: list[dict], model: str) -> tuple[list[dict], str | None]:
    """The model's holds in order and the time it is watched from (its last resume; None before any). A hold is
    {"start", "end", "since", "state", "confirmed_by", "probe", "last_probe"}; its state is "held" (the probe pending),
    "confirmed" (the window open), "closed" (a confirmed window, ended by a passing probe: "end") or "not_confirmed"
    (the probe passed with nothing confirmatory: the attempts stand; "passed_at", the probe's end)."""
    eps: list[dict] = []
    since = None
    for e in log:
        if e.get("model") != model:
            continue
        cur = eps[-1] if eps and eps[-1]["state"] in ("held", "confirmed") else None
        kind = e["kind"]
        if kind == "hold":
            if cur is None:
                eps.append({"start": e["window_start"], "end": None, "since": since, "state": "held", "hold": e,
                            "confirmed_by": [], "probe": None, "last_probe": None})
        elif kind == "window_start" and eps:  # late records of an earlier triggering bin moved the start
            eps[-1]["start"] = e["window_start"]
        elif kind == "probe_start" and cur is not None:
            cur["probe"] = e
        elif kind == "probe_result" and cur is not None:
            cur.update(probe=None, last_probe=e)
        elif kind == "confirmed" and eps:
            ep = eps[-1]
            ep["confirmed_by"].append({k: e.get(k) for k in ("by", "note", "at", "evidence")})
            if ep["state"] == "held":
                ep["state"] = "confirmed"
            elif ep["state"] == "not_confirmed":  # a provider's report after the probe passed: the window closes there
                ep.update(state="closed", end=ep["passed_at"])
        elif kind == "closed" and cur is not None:
            cur.update(state="closed", end=e["end"])
            since = e["end"]
        elif kind == "not_confirmed" and cur is not None:
            cur.update(state="not_confirmed", passed_at=e["end"])
            since = e["end"]
    return eps, since


def model_state(log: list[dict], model: str) -> dict[str, Any]:
    eps, since = episodes(log, model)
    if eps and eps[-1]["state"] in ("held", "confirmed"):
        return {**eps[-1], "window_start": eps[-1]["start"]}
    return {"state": "watching", "since": since}


def windows(log: list[dict], model: str) -> list[dict]:
    """The model's confirmed windows: start, end (None while open) and what confirmed them."""
    return [{"start": ep["start"], "end": ep["end"], "confirmed_by": ep["confirmed_by"]}
            for ep in episodes(log, model)[0] if ep["state"] in ("confirmed", "closed")]


def hold_all(outs: list[Path], model: str, entry: dict, state: str) -> None:
    for out in outs:
        _write_atomic(fault.hold_path(out, model), {
            "schema": "infra-fault-hold/2", "model": model, "window_start": entry["window_start"], "at": entry["at"],
            "state": state, "rule": fault.RULE["rule"], "triggers": entry.get("triggers"),
            "release_with": "the fault probe passing (scripts/infra_fault_monitor.py); never by deleting this file"})


def release_all(outs: list[Path], model: str) -> None:
    for out in outs:
        fault.hold_path(out, model).unlink(missing_ok=True)


def manifest_of(outs: list[Path], model: str) -> Path | None:
    for out in outs:
        try:
            if model in fault.route_names(out).values():
                return Path(out) / "manifest.json"
        except (OSError, ValueError):
            continue
    return None


def start_probe(args: argparse.Namespace, model: str, say=print) -> dict:
    """Start the subject's probe into a new directory under DIR/probes: a model's fault probe (``--probe-cmd``), a store
    copy's store probe (``--store-probe-cmd``, subject "store:<copy>"); without a command, print the one to run by
    hand into that directory. The entry the log records."""
    t = now()
    pdir = Path(args.state) / PROBES_DIR / f"{model.replace(':', '-')}-{t.strftime('%Y%m%dT%H%M%S%fZ')}"
    if model.startswith(STORE_PREFIX):
        given = getattr(args, "store_probe_cmd", None)
        template = given or STORE_PROBE_HINT
        fields = {"copy": model.removeprefix(STORE_PREFIX), "dir": str(pdir)}
    else:
        given = args.probe_cmd
        template = given or DEFAULT_PROBE_CMD
        fields = {"model": model, "dir": str(pdir), "manifest": str(manifest_of(args.outs, model))}
    cmd = [part.format(**fields) for part in shlex.split(template)]
    entry = {"kind": "probe_start", "model": model, "at": stamp(t), "dir": str(pdir), "cmd": cmd, "pid": None}
    if given:
        pdir.parent.mkdir(parents=True, exist_ok=True)
        log_path = pdir.parent / f"{pdir.name}.log"
        with log_path.open("w", encoding="utf-8") as fh:
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT), start_new_session=True)
        entry["pid"] = proc.pid
        say(f"{model}: fault probe started ({pdir}; log {log_path})")
    else:
        say(f"{model}: run its fault probe now, by hand: {shlex.join(cmd)}")
    return entry


def alive(pid: int | None) -> bool:
    if pid is None:
        return True  # started by hand: wait for its summary
    with contextlib.suppress(ChildProcessError):
        if os.waitpid(pid, os.WNOHANG)[0] == pid:  # our own child, reaped: it has ended
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_probe(entry: dict) -> dict | None:
    """The probe's summary.json once written; {"verdict": "error"} when its process ended without one; else None."""
    path = Path(entry["dir"]) / "summary.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None  # being written
    if not alive(entry.get("pid")):
        if path.exists():  # written as it ended
            return read_probe(entry)
        return {"verdict": "error", "pass": None, "reasons": ["the probe ended without a summary"]}
    return None


def refine_starts(args: argparse.Namespace, base: fault.Baselines, evs: list, log: list[dict], models: list[str],
                  say=print) -> None:
    """A window's start is its earliest triggering bin, and records arrive late (an attempt is recorded when it ends):
    each model's latest hold that is held, confirmed or closed moves its start earlier when a triggering bin before it
    appears among the attempts started since the time it was watched from. Logged, its hold files rewritten while it
    is held, and alerted when the window is confirmed: its set-aside must be run again from the new start."""
    for m in models:
        eps, _ = episodes(log, m)
        if not eps or eps[-1]["state"] == "not_confirmed":
            continue
        ep = eps[-1]
        since = {m: fault.parse_time(ep["since"])} if ep["since"] else None
        results = [r for r in fault.evaluate(evs, base, since) if r.model == m]
        first = fault.first_triggers(results).get(m)
        if first is None or not first.start < fault.parse_time(ep["start"]):
            continue
        entry = {"kind": "window_start", "model": m, "at": stamp(now()), "window_start": stamp(first.start),
                 "was": ep["start"],
                 "triggers": [r.describe() for r in results if r.start == first.start and r.triggered]}
        append_log(args.state, entry)
        log.append(entry)
        if ep["state"] in ("held", "confirmed"):
            hold_all(args.outs, m, {**ep["hold"], "window_start": entry["window_start"]}, ep["state"])
        if ep["state"] in ("confirmed", "closed"):
            say(f"ALERT infra-fault: {m}'s confirmed window now starts at {entry['window_start']} (was {ep['start']}): "
                f"late records show an earlier triggering bin; run its set-aside again (window --model {m})")
        else:
            say(f"{m}: its hold now dates from {entry['window_start']} (was {ep['start']}): late records")


def load_rule(args: argparse.Namespace, memo: dict, say=print) -> fault.Baselines:
    """The frozen rule's rates, read again on every pass; a rule that cannot be read now keeps the one read before."""
    try:
        doc = fault.load_rule(args.rule, args.pin)
        memo["baselines"], memo["doc"] = fault.Baselines.from_rule(doc), doc
        args.rule_doc, args.rule_sha = doc, fault.rule_sha256(args.rule)
    except (OSError, ValueError, KeyError) as exc:
        if "baselines" not in memo:
            raise
        say(f"warning: {args.rule} cannot be read now ({exc}); keeping the rule read before")
    return memo["baselines"]


def _log(args: argparse.Namespace, log: list[dict], entry: dict) -> dict:
    append_log(args.state, entry)
    log.append(entry)
    return entry


def read_admitted(path: Path) -> dict[str, list[str]]:
    """Each model's admitted system fingerprints in ``path`` (OUT/admitted-fingerprints.json, change-log entry 167),
    sorted; {} without the file; ValueError when it is malformed anywhere: a copy of scripts/run_pilot.py read_admitted,
    which the runner's check reads, and of scripts/fault_probe.py's (tests/test_e167_admitted_fingerprints.py holds the
    three to one result)."""
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return {}
    doc = json.loads(raw)
    if not isinstance(doc, dict) or doc.get("schema") != ADMITTED_SCHEMA or not isinstance(doc.get("models"), dict):
        raise ValueError(f"not an {ADMITTED_SCHEMA} file with a models object")
    out = {}
    for model, entry in doc["models"].items():
        admitted = entry.get("admitted") if isinstance(entry, dict) else None
        if not isinstance(admitted, list):
            raise ValueError(f"{model}: no admitted list")  # noqa: TRY004 - the file's content
        fingerprints = set()
        for item in admitted:
            fp_ = item.get("system_fingerprint") if isinstance(item, dict) else None
            if not isinstance(fp_, str) or not fp_ or MIXED_FINGERPRINTS in fp_:
                raise ValueError(f"{model}: an admitted entry without one system fingerprint")
            fingerprints.add(fp_)
        out[model] = sorted(fingerprints)
    return out


def served_fingerprint(manifest: Path | None, model: str) -> str | None:
    """The model's served fingerprint as its OUT recorded it (served-models.json beside the manifest), or None."""
    path = Path(manifest).parent / "served-models.json" if manifest is not None else None
    if path is None or not path.exists():
        return None
    entry = (json.loads(path.read_text(encoding="utf-8")).get("models") or {}).get(model) or {}
    fp_ = entry.get("system_fingerprint")
    return fp_ if isinstance(fp_, str) and fp_ else None


def manifest_route(manifest: Path | None, model: str) -> dict:
    """The model's route in the OUT's manifest (fingerprint.routes), {} without one."""
    if manifest is None:
        return {}
    routes = (json.loads(Path(manifest).read_text(encoding="utf-8")).get("fingerprint") or {}).get("routes") or {}
    return routes.get(model) if isinstance(routes.get(model), dict) else {}


def recorded_fingerprints(manifest: Path | None, model: str) -> list[str]:
    """The model's fingerprints as its OUT recorded them, as the probe reads them (scripts/fault_probe.py
    recorded_fingerprints): the served one and those admitted beside it (admitted-fingerprints.json beside the
    manifest, change-log entry 167; a malformed file adds none, with a warning)."""
    if manifest is None:
        return []
    served = served_fingerprint(manifest, model)
    found = [served] if served is not None else []
    path = Path(manifest).parent / ADMITTED_FILE
    try:
        admitted = read_admitted(path).get(model, [])
    except (OSError, ValueError) as exc:
        print(f"warning: {path} is ignored ({type(exc).__name__}: {exc})", file=sys.stderr)
        admitted = []
    return found + [f for f in admitted if f not in found]


def summary_problem(args: argparse.Namespace, subject: str, summary: dict) -> str | None:
    """Why a probe's summary cannot decide (it then decides nothing and the subject stays held), or None: it must be
    this subject's, from its bound probe set (and, for a model, the pinned rule), of all 20 requests, with a verdict
    consistent with its pass flag and reasons."""
    doc = getattr(args, "rule_doc", None) or {}
    verdict, passed = summary.get("verdict"), summary.get("pass")
    if verdict not in ("pass", "fail") or passed is not (verdict == "pass") or (passed and summary.get("reasons")):
        return f"an inconsistent verdict ({verdict!r}, pass {passed!r})"
    if subject.startswith(STORE_PREFIX):
        copy = subject.removeprefix(STORE_PREFIX)
        ps = (((doc.get("stores") or {}).get("copies") or {}).get(copy) or {}).get("probe_set")
        if summary.get("copy") != copy:
            return f"the summary is store copy {summary.get('copy')!r}'s, not {copy}'s"
        if not isinstance(ps, dict) or summary.get("set_sha256") != ps.get("sha256"):
            return "its probe set is not the copy's bound one"
        n, k, errors, ok = (summary.get(x) for x in ("n", "timeouts", "errors", "ok"))
        if n != sf.STORE_PROBE_N:
            return f"{n} requests, {sf.STORE_PROBE_N} expected"
        if not all(isinstance(x, int) for x in (k, errors, ok)) or ok + k + errors != n:
            return "its counts do not add up to its requests"
        if passed != (k <= sf.STORE_PROBE_MAX_TIMEOUTS and errors == 0):
            return f"its verdict contradicts its counts ({k} time-outs, {errors} errors)"
        return None
    entry = (doc.get("probe_sets") or {}).get(subject) or {}
    if summary.get("model") != subject:
        return f"the summary is {summary.get('model')!r}'s, not {subject}'s"
    if summary.get("probe_set_sha256") != entry.get("sha256"):
        return "its probe set is not the model's bound one"
    if summary.get("rule_sha256") != getattr(args, "rule_sha", None):
        return "it ran on another fault rule than the pinned one"
    counts, n = summary.get("counts") or {}, (doc.get("probe") or {}).get("n", 20)
    if counts.get("n") != n:
        return f"{counts.get('n')} requests, not the probe's"
    names = ("replies", "errors", "no_call", "low_coverage", "missing")
    if not all(isinstance(counts.get(x), int) for x in names) or counts["replies"] + counts["errors"] != n:
        return "its counts do not add up to its requests"
    limits = (doc.get("probe") or {}).get("thresholds", {}).get(subject) or {}
    within = all(limits.get(c, {}).get("k") is None or counts[c] <= limits[c]["k"] for c in ("no_call", "low_coverage"))
    if passed != (counts["errors"] == 0 and counts["missing"] == 0 and within):
        return "its verdict contradicts its counts and the pinned thresholds"
    manifest = manifest_of(args.outs, subject)
    # the route the OUT's manifest pins for the model (the probe sends through a backend's listener under it, change-log
    # entry 167, and checks its route against it), from that manifest when it names one
    route = summary.get("route") if isinstance(summary.get("route"), dict) else {}
    pinned = manifest_route(manifest, subject)
    if (route.get("model"), str(route.get("base_url") or "").rstrip("/")) != (
            pinned.get("model"), str(pinned.get("base_url") or "").rstrip("/")):
        return "its route is not the model's route in the OUT's manifest"
    if summary.get("manifest") is not None and (manifest is None
                                                or Path(summary["manifest"]).resolve() != Path(manifest).resolve()):
        return "it read another manifest than the OUT's that routes the model"
    # the OUT's served fingerprint and only fingerprints it recorded: an admission appended while the probe ran
    # (entry 167; the file only grows) leaves the probe's set a part of the OUT's (its change is then judged again
    # against the OUT's set, current_change)
    expected = set(summary.get("expected_fingerprints") or [])
    served, recorded = served_fingerprint(manifest, subject), set(recorded_fingerprints(manifest, subject))
    if not expected <= recorded or (served is not None and served not in expected) or (recorded and not expected):
        return "its expected fingerprints are not the ones the OUT recorded"
    change = bool(expected) and any(f not in expected for f in summary.get("fingerprints") or [])
    if bool(summary.get("fingerprint_change")) != change:
        return "its fingerprint change contradicts its fingerprints"
    return None


def current_change(args: argparse.Namespace, model: str, summary: dict) -> bool | None:
    """Whether a probe's replies show a fingerprint other than the OUT's recorded ones NOW (the served one and those
    admitted beside it, change-log entry 167): what confirms a fault, so a fingerprint admitted while the probe ran is
    no change; None when the OUT records none (a route that reports none), as the probe reports it."""
    recorded = set(recorded_fingerprints(manifest_of(args.outs, model), model))
    if not recorded:
        return None
    return any(f not in recorded for f in summary.get("fingerprints") or [])


def step_probe(args: argparse.Namespace, log: list[dict], m: str, s: dict, say=print, release=None) -> str:
    """A held model's probe: read the running one's result and act on it, or start one when due. The model's state
    after this step ("held", "confirmed", or "watching" once released)."""
    t = now()
    release = release or (lambda: release_all(args.outs, m))
    state, last = s["state"], s["last_probe"]
    if s["probe"] is not None:
        result = read_probe(s["probe"])
        if result is None:
            return state
        if result.get("verdict") in ("pass", "fail"):
            problem = summary_problem(args, m, result)
            if problem:
                result = {"verdict": "invalid", "pass": None, "reasons": [problem]}
        decided = {}
        if result.get("verdict") in ("pass", "fail"):  # the change that decides: against the OUT's fingerprints now
            decided = {"fingerprint_change": current_change(args, m, result),
                       "probe_fingerprint_change": result.get("fingerprint_change")}
        entry = _log(args, log, {
            "kind": "probe_result", "model": m, "at": stamp(t), "dir": s["probe"]["dir"],
            "verdict": result.get("verdict"), "pass": result.get("pass"), "reasons": result.get("reasons"),
            "counts": result.get("counts"), "fingerprints": result.get("fingerprints"),
            "fingerprint_change": result.get("fingerprint_change"), "started_at": result.get("started_at"),
            "ended_at": result.get("ended_at"), "probe_url": result.get("probe_url"), **decided})
        last = entry
        if entry["verdict"] not in ("pass", "fail"):
            say(f"ALERT infra-fault: {m}'s probe gave no usable verdict ({'; '.join(entry['reasons'] or [])}; "
                f"{s['probe']['dir']}); it decides nothing; started again after {args.probe_every:.0f} s")
            return _due(args, log, m, state, last, t, say)
        if entry["fingerprint_change"]:
            _log(args, log, {"kind": "confirmed", "model": m, "at": stamp(t), "by": "fingerprint",
                             "evidence": {"dir": entry["dir"], "fingerprints": entry["fingerprints"]}})
        if entry["pass"] is False:
            _log(args, log, {"kind": "confirmed", "model": m, "at": stamp(t), "by": "probe",
                             "evidence": {"dir": entry["dir"], "reasons": entry["reasons"]}})
        confirmed = s["state"] == "confirmed" or entry["pass"] is False or bool(entry["fingerprint_change"])
        if entry["pass"] and not entry["fingerprint_change"]:
            end = entry["ended_at"] or entry["at"]
            if confirmed:
                _log(args, log, {"kind": "closed", "model": m, "at": stamp(t), "end": end, "dir": entry["dir"]})
                release()
                say(f"{m}: its fault probe passes again: the confirmed window {s['window_start']} .. {end} is "
                    f"closed and the model resumes; set the window aside with: window --model {m}")
            else:
                _log(args, log, {"kind": "not_confirmed", "model": m, "at": stamp(t), "end": end,
                                 "dir": entry["dir"]})
                release()
                say(f"{m}: its hold from {s['window_start']} is NOT confirmed (the fault probe passed, nothing "
                    f"confirmatory): its attempts stand, scored as usual; the model resumes")
            return "watching"
        if state == "held":
            why = "; ".join(entry["reasons"] or ["a served-fingerprint change"])
            say(f"ALERT infra-fault: {m}'s fault is CONFIRMED ({why}): the window is open from {s['window_start']} "
                f"until its probe passes again; the model stays held")
        state = "confirmed"
    return _due(args, log, m, state, last, t, say)


def _due(args: argparse.Namespace, log: list[dict], m: str, state: str, last: dict | None, t: datetime,
         say=print) -> str:
    """Start the held model's next probe once ``--probe-every`` has passed since the last result (at once if none)."""
    if last is None or fault.parse_time(last["at"]) + timedelta(seconds=args.probe_every) <= t:
        _log(args, log, start_probe(args, m, say))
    return state


def check_store_copies(args: argparse.Namespace, doc: dict) -> dict[Path, str]:
    """The watched OUTs' copies, refused (before any hold is written or probe started) when a watched copy has no
    frozen reference share or probe set in the rule: a copy the rule cannot judge or confirm is not left unmonitored.
    The mapping is recorded in the state directory (watched-copies.json), where the set-aside reads which copy an
    OUT is on."""
    copies = store_outs(args)
    if not copies:
        return copies
    path = Path(args.state) / STORE_COPIES_FILE
    if Path(args.store_copies).resolve() == path.resolve():
        raise ValueError(f"--store-copies is the monitor's own record {path}: give the launch step's mapping")
    recorded = json.loads(path.read_text(encoding="utf-8")).get("outs", {}) if path.exists() else {}
    now_map = {str(Path(o).resolve()): c for o, c in copies.items()}
    moved = sorted(o for o, c in now_map.items() if o in recorded and recorded[o] != c)
    if moved:  # one copy per OUT for its whole run (science, 17:28): a moved OUT would mix its windows' attempts
        raise ValueError(f"OUTs {moved} are on another copy than the monitor recorded ({path}): refused")
    store_copies = (doc.get("stores") or {}).get("copies") or {}
    refs = sf.refs_from_doc(store_copies)
    incomplete = [c for c in sorted(set(copies.values()))
                  if c not in refs or not isinstance((store_copies.get(c) or {}).get("probe_set"), dict)]
    if incomplete:
        raise ValueError(f"store copies {incomplete} are watched but their reference share or probe set is not frozen "
                         f"in {args.rule}: fill and pin them before the monitor starts")
    _write_atomic(path, {"outs": {**recorded, **now_map}})
    return copies


def scan_once(args: argparse.Namespace, cache: dict, say=print, memo: dict | None = None) -> int:
    base = load_rule(args, memo if memo is not None else {}, say)
    check_store_copies(args, (memo or {}).get("doc") or getattr(args, "rule_doc", {}) or {})
    log = read_log(args.state)
    evs = []
    for out in args.outs:
        got, left_out = fault.out_events(out, cache)
        evs += got
        if left_out:
            say(f"{out}: left out of the rule: {dict(sorted(left_out.items()))}")
    models = sorted({e.model for e in evs}
                    | {e["model"] for e in log if "model" in e and not e["model"].startswith(STORE_PREFIX)})
    refine_starts(args, base, evs, log, models, say)
    states = {m: model_state(log, m) for m in models}
    since = {m: fault.parse_time(s["since"]) for m, s in states.items() if s["state"] == "watching" and s["since"]}
    for m, arms in sorted(fault.unmonitored(evs, base, since).items()):
        for arm, counts in sorted(arms.items()):
            why = (f"no post-fix branch baseline (branch section {base.branch_status})"
                   if arm.startswith(fault.BRANCH_PREFIX) else "no frozen dev rate of the model")
            say(f"unmonitored: {m} {arm} ({counts['attempts']} attempts, {counts['turns']} turns): {why}; the rule "
                f"cannot see a fault there")
    results = fault.evaluate(evs, base, since)
    first = fault.first_triggers(results)
    held = []
    for m in models:
        s = states[m]
        if s["state"] in ("held", "confirmed"):
            hold_all(args.outs, m, {**s["hold"], "window_start": s["window_start"]}, s["state"])
            after = step_probe(args, log, m, s, say)
            if after != "watching":
                held.append(m)
            if after == "confirmed" and s["state"] == "held":  # the hold files say what the probe decided
                hold_all(args.outs, m, {**s["hold"], "window_start": s["window_start"]}, after)
        elif m in first:
            start = first[m].start
            entry = _log(args, log, {
                "kind": "hold", "model": m, "at": stamp(now()), "window_start": stamp(start),
                "triggers": [r.describe() for r in results if r.model == m and r.start == start and r.triggered]})
            hold_all(args.outs, m, entry, "held")
            held.append(m)
            said = "; ".join(f"{t['metric']} {t['events']}/{t['denominator']} = {t['share']:.3f} > {t['threshold']:.3f}"
                             for t in entry["triggers"])
            say(f"ALERT infra-fault: {m} HELD in {len(args.outs)} OUT(s) from the bin {stamp(start)} ({said}); "
                f"nothing is relabelled; its fault probe decides")
            _log(args, log, start_probe(args, m, say))
    held += scan_stores(args, memo if memo is not None else {}, log, say)
    return 3 if held else 0


# --- store copies (entry 149, item 2) ------------------------------------------------------------------


def store_outs(args: argparse.Namespace) -> dict[Path, str]:
    """The watched OUTs' store copies (``--store-copies``: the operator's file from the launch step's e.json)."""
    if not getattr(args, "store_copies", None):
        return {}
    named = {Path(k).resolve(): v for k, v in sf.load_copies(args.store_copies).items()}
    return {out: named[Path(out).resolve()] for out in args.outs if Path(out).resolve() in named}


def hold_store(args: argparse.Namespace, copy: str, entry: dict, state: str) -> None:
    """Hold every model of every watched OUT on the copy (the runner reads OUT/holds/<model>.store-<copy>.json)."""
    for out, c in store_outs(args).items():
        if c != copy:
            continue
        for name in sorted(fault.route_names(out).values()):
            _write_atomic(fault.store_hold_path(out, name, copy), {
                "schema": "store-fault-hold/1", "model": name, "store_copy": copy, "state": state,
                "window_start": entry["window_start"], "at": entry["at"], "rule": sf.STORE_RULE["rule"],
                "triggers": entry.get("triggers"),
                "release_with": "the store probe passing (scripts/infra_fault_monitor.py); never by deleting this file"})


def release_store(args: argparse.Namespace, copy: str) -> None:
    for out, c in store_outs(args).items():
        if c == copy:
            for path in (Path(out) / fault.HOLDS_DIR).glob(f"*.store-{copy}.json"):
                path.unlink(missing_ok=True)


def refine_store_starts(args: argparse.Namespace, refs: dict, evs: list, log: list[dict], subjects: list[str],
                        say=print) -> None:
    """refine_starts for store copies: late records (a query is recorded when its attempt ends) that show an earlier
    triggering bin move the latest hold's start earlier; logged, hold files rewritten while held, alerted."""
    for subj in subjects:
        eps, _ = episodes(log, subj)
        if not eps or eps[-1]["state"] == "not_confirmed":
            continue
        ep, copy = eps[-1], subj.removeprefix(STORE_PREFIX)
        since = {copy: fault.parse_time(ep["since"])} if ep["since"] else None
        first = sf.first_store_triggers(sf.evaluate_store(evs, refs, since)).get(copy)
        if first is None or not first.start < fault.parse_time(ep["start"]):
            continue
        entry = _log(args, log, {"kind": "window_start", "model": subj, "at": stamp(now()),
                                 "window_start": stamp(first.start), "was": ep["start"]})
        if ep["state"] in ("held", "confirmed"):
            hold_store(args, copy, {**ep["hold"], "window_start": entry["window_start"]}, ep["state"])
        say(f"ALERT store-fault: {subj}'s hold now starts at {entry['window_start']} (was {ep['start']}): late "
            f"records show an earlier triggering bin" + ("; run its set-aside again" if ep["state"] != "held" else ""))


def scan_stores(args: argparse.Namespace, memo: dict, log: list[dict], say=print) -> list[str]:
    """The store time-out rule over the watched OUTs' copies, with the same hold, probe and confirmation as a model's:
    a copy whose bin reaches its trigger share with at least five time-outs is held (every model of its OUTs) and its
    store probe starts; a failing probe confirms the window until the probe passes again; a passing first probe leaves
    the attempts standing. The held subjects ("store:<copy>")."""
    copies = store_outs(args)
    if not copies:
        return []
    refs = sf.refs_from_doc(((memo.get("doc") or {}).get("stores") or {}).get("copies") or {})
    cache = memo.setdefault("store_cache", {})
    evs = []
    for out, copy in copies.items():
        got, left_out = sf.out_store_events(out, copy, cache)
        evs += got
        if left_out:
            say(f"{out}: left out of the store rule: {dict(sorted(left_out.items()))}")
    subjects = sorted({STORE_PREFIX + c for c in copies.values()})
    refine_store_starts(args, refs, evs, log, subjects, say)
    states = {subj: model_state(log, subj) for subj in subjects}
    since = {subj.removeprefix(STORE_PREFIX): fault.parse_time(s["since"]) for subj, s in states.items()
             if s["state"] == "watching" and s["since"]}
    for copy, counts in sorted(sf.unmonitored_copies(evs, refs).items()):
        say(f"unmonitored: store copy {copy} ({counts}): no frozen reference share; the rule cannot see its time-outs")
    results = sf.evaluate_store(evs, refs, since)
    first = sf.first_store_triggers(results)
    held = []
    for subj in subjects:
        copy, s = subj.removeprefix(STORE_PREFIX), states[subj]
        if s["state"] in ("held", "confirmed"):
            hold_store(args, copy, {**s["hold"], "window_start": s["window_start"]}, s["state"])
            after = step_probe(args, log, subj, s, say, release=lambda c=copy: release_store(args, c))
            if after != "watching":
                held.append(subj)
            if after == "confirmed" and s["state"] == "held":
                hold_store(args, copy, {**s["hold"], "window_start": s["window_start"]}, after)
        elif copy in first:
            start = first[copy].start
            trig = [{"bin": stamp(r.start), "timeouts": r.timeouts, "queries": r.queries, "share": round(r.share, 6),
                     "threshold": round(r.threshold, 6)} for r in results if r.copy == copy and r.start == start]
            entry = _log(args, log, {"kind": "hold", "model": subj, "at": stamp(now()), "window_start": stamp(start),
                                     "triggers": trig})
            hold_store(args, copy, entry, "held")
            held.append(subj)
            say(f"ALERT store-fault: store copy {copy} HELD from the bin {stamp(start)} ({trig[0]['timeouts']}/"
                f"{trig[0]['queries']} time-outs, share {trig[0]['share']:.3f} >= {trig[0]['threshold']:.3f}); every "
                f"model of its OUTs is held; nothing is relabelled; its store probe decides")
            _log(args, log, start_probe(args, subj, say))
    return held


def cmd_scan(args: argparse.Namespace) -> int:
    cache: dict = {}
    memo: dict = {}
    while True:
        status = scan_once(args, cache, memo=memo)
        if not args.every:
            return status
        sys.stdout.flush()
        time.sleep(args.every)


def cmd_confirm(args: argparse.Namespace) -> int:
    """An operator's confirmatory event: a provider-confirmed restart or incident, or a served-fingerprint change seen
    outside the probe. It confirms the model's latest hold (held: its window opens; already resumed unconfirmed: the
    window closes at the end of the probe that passed)."""
    subject = args.model or STORE_PREFIX + args.copy
    log = read_log(args.state)
    eps, _ = episodes(log, subject)
    if not eps:
        print(f"refused: {subject} has no hold to confirm", file=sys.stderr)
        return 2
    try:
        at = fault.parse_time(args.at) if args.at else now()
    except ValueError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    append_log(args.state, {"kind": "confirmed", "model": subject, "at": stamp(now()), "by": args.by,
                            "note": args.note, "evidence": {"event_at": stamp(at)}})
    after = episodes(read_log(args.state), subject)[0][-1]
    print(f"{subject}: its hold from {after['start']} is confirmed ({args.by}: {args.note}); window "
          f"{after['start']} .. {after['end'] or 'open until its probe passes'}")
    return 0


def cmd_window(args: argparse.Namespace) -> int:
    if args.copy:
        return store_window(args)
    log = read_log(args.state)
    for ep in episodes(log, args.model)[0]:
        if ep["state"] == "not_confirmed":
            print(f"{args.model}: hold from {ep['start']}: NOT confirmed (probe passed {ep['passed_at']}): its "
                  f"attempts stand; nothing to set aside")
        elif ep["state"] == "held":
            print(f"{args.model}: hold from {ep['start']}: not decided yet (its fault probe is pending)")
    found = windows(log, args.model)
    if not found:
        print(f"{args.model}: no confirmed window")
        return 0
    for w in found:
        end = w["end"] or "open"
        by = ", ".join(c["by"] for c in w["confirmed_by"])
        print(f"{args.model}: window {w['start']} .. {end} (confirmed by {by})")
        for out in args.outs:
            route = {v: k for k, v in fault.route_names(out).items()}.get(args.model)
            if route is None:
                print(f"  {out}: no route of {args.model} in its manifest")
                continue
            defect = (f"provider fault on {args.model}, confirmed by {by}: attempts with a model turn in the window "
                      f"{w['start']} .. {end} (change-log entries 129, 148 and 149; every arm, any outcome)")
            print("  uv run python " + " ".join([
                "scripts/relabel_harness_failure.py", shlex.quote(str(out)), "--window-model", shlex.quote(route),
                "--window-start", w["start"], "--window-end", end, "--defect", shlex.quote(defect),
                "--ruling", "RULING", "--monitor-state", shlex.quote(str(args.state))]))
    return 0


def store_window(args: argparse.Namespace) -> int:
    subject = STORE_PREFIX + args.copy
    log = read_log(args.state)
    for ep in episodes(log, subject)[0]:
        if ep["state"] == "not_confirmed":
            print(f"{subject}: hold from {ep['start']}: NOT confirmed (store probe passed {ep['passed_at']}): its "
                  f"attempts stand; nothing to set aside")
        elif ep["state"] == "held":
            print(f"{subject}: hold from {ep['start']}: not decided yet (its store probe is pending)")
    found = windows(log, subject)
    if not found:
        print(f"{subject}: no confirmed window")
        return 0
    outs = [out for out, c in store_outs(args).items() if c == args.copy]
    for w in found:
        end = w["end"] or "open"
        by = ", ".join(c["by"] for c in w["confirmed_by"])
        print(f"{subject}: window {w['start']} .. {end} (confirmed by {by})")
        if not outs:
            print("  no watched OUT on this copy (--store-copies)")
        for out in outs:
            defect = (f"store time-outs on copy {args.copy}, confirmed by {by}: attempts with a query on it in the "
                      f"window {w['start']} .. {end} (change-log entry 149, item 2; every model and arm, any outcome)")
            print("  uv run python " + " ".join([
                "scripts/relabel_harness_failure.py", shlex.quote(str(out)), "--window-store", shlex.quote(args.copy),
                "--window-start", w["start"], "--window-end", end, "--defect", shlex.quote(defect),
                "--ruling", "RULING", "--monitor-state", shlex.quote(str(args.state))]))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    scan = sub.add_parser("scan", help="evaluate the trigger over the OUTs' records; hold; probe; confirm or resume")
    scan.add_argument("--rule", type=Path, default=ROOT / "data/runs/infra-fault/fault-rule.json")
    scan.add_argument("--pin", type=Path, default=ROOT / "data/freeze/frozen-config.json")
    scan.add_argument("--every", type=float, default=None, help="scan again every this many seconds")
    scan.add_argument("--probe-cmd", default=None,
                      help=f"the probe command template ({{model}}, {{dir}}, {{manifest}}), e.g. {DEFAULT_PROBE_CMD!r}")
    scan.add_argument("--store-probe-cmd", default=None,
                      help="the store probe command template ({copy}, {dir}), e.g. a wrapper that runs "
                           "scripts/store_probe.py run for the copy")
    scan.add_argument("--probe-every", type=float, default=PROBE_EVERY_S,
                      help="while a window is confirmed (or a probe gave no verdict), probe again after this")
    confirm = sub.add_parser("confirm", help="record a provider-confirmed incident or a fingerprint change")
    who = confirm.add_mutually_exclusive_group(required=True)
    who.add_argument("--model")
    who.add_argument("--copy", help="a store copy (entry 149, item 2)")
    confirm.add_argument("--by", choices=CONFIRM_BY, required=True)
    confirm.add_argument("--note", required=True, help="the operator note: who confirmed what, when, where")
    confirm.add_argument("--at", default=None, help="when the event happened (ISO with its zone; default now)")
    window = sub.add_parser("window", help="print the model's holds, windows and each OUT's set-aside command")
    which = window.add_mutually_exclusive_group(required=True)
    which.add_argument("--model")
    which.add_argument("--copy", help="a store copy (entry 149, item 2)")
    for p in (scan, confirm, window):
        p.add_argument("--state", type=Path, required=True, help="the monitor's directory (its infra-faults.jsonl)")
        p.add_argument("--store-copies", type=Path, default=None,
                       help="the watched OUTs' store copies, {\"outs\": {OUT: copy}}, written from the launch step's "
                            "e.json (store_fault.copies_from_e_record): the store time-out rule runs on them")
        p.add_argument("outs", type=Path, nargs="+", metavar="OUT")
    args = ap.parse_args(argv)
    return {"scan": cmd_scan, "confirm": cmd_confirm, "window": cmd_window}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
