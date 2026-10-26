#!/usr/bin/env python3
"""Starts the C1a choice and the power run the moment a graph's dev-grid prompted phase is complete (design log ruling
2; the freeze path: prompted, then choose-c1a and the power rerun, then the freeze).

**Complete** is the runner's own rule for the initial phases (scripts/run_pilot.py ``require_initial_complete``),
applied to the prompted phase alone:
- OUT's manifest (``pilot-grid-manifest/1``) pins the prompted phase, and its prompted arms hold every planned cell;
- every planned prompted attempt of the graph has a record in OUT/records/ whose outcome did not fail infrastructure.
  The planned attempts are the grid manifest's attempts file (checked against the manifest's ``attempts_sha256``),
  restricted to the graph and the phase. A record's attempt key is the runner's (``run_pilot.attempt_key`` with
  ``run_pilot.record_arm``): arm, question, route model, repetition. Two records with one key are refused;
- the phase is not HALTED (OUT's replay check, ``replay_check.halt_message`` for phase prompted). A halted phase waits,
  however many records it has: its records are re-run under the fixed code first;
- the runner's own order of the phase (OUT/order-prompted.json, model names mapped to route models) holds no attempt
  outside the planned set: another grid manifest than the one given is refused.
- **Fallback** (science, design log; FALLBACK_RULE): the models named in ``--fallback ROUTE-MODEL[=ISO-CUTOFF]`` are excused
  their remainder (a planned prompted model, given, never inferred from outcomes); every other model must be complete.
  Their completed planned attempts are counted in the state and in done.json, and dev-rates applies the rule (their
  completed prompted records, before the cutoff if given, plus their natural rate, stated).
Nothing in OUT is written: the record facts read are cached in RESULTS by file name, size and mtime.

**When complete, once** (RESULTS/done.json marks it; a failed step writes RESULTS/failed.json and stops, and the
watcher then refuses to start again until that file is moved away), each step under nice -n 19 from this code copy:
1. ``power_weighted.py dev-rates``: C1's dev rates from OUT (RESULTS/dev-rates.json), under the frozen scorer
   (scorer-rules-v6 since change-log entry 127): the dev OUT's v4- or v5-scored records of the C1 arms are re-scored
   under the frozen rules in memory (power_weighted.rescore_frozen; entries 120, 121 and 127, science in the design log);
2. ``power_weighted.py plan --agent-only`` and ``run --jobs 4 --no-aggregate`` in RESULTS/power: the C1 cells and, when
   the file is there, C5's entry-77 counts (the bank cells are the ``--bank-power`` OUT's, which done.json names);
3. ``power_weighted.py aggregate`` on the registered calibration of both families (step 3's final calibration, or
   before its addendum step 3's selection levels with the validation pending, change-log entry 131; the review of
   1e31f1f: never the one-sided calibration). While the config does not bind step 3's selection it exits 3: RESULTS/
   waiting-step3.json is written (with the completed phase's status) and every later look retries the aggregation
   alone (state "waiting for step 3", exit 6 with ``--once``), until a config that binds step 3 is given (a watcher
   redeployed from the commit that binds it, on the same RESULTS);
4. ``calibrate.py choose-c1a`` on a copy of the config in RESULTS (record_c1a_choice writes once and never changes a
   recorded choice) with the two weighted-Holm rejection powers at the 3-point margin (the first-level powers are
   recorded beside them in done.json); the repository's config is not touched.
   The pooled C1a is eligible only if entry 81 keeps all seven models of P (prereg.C1A_ELIGIBILITY, design log 18:50): a
   model whose dev capped:context rate differs between the curated and census arms by more than 2 points makes the
   two-family C1a enter, recorded with the dropped models and the rule. Only missing entry-81 rates (or c1a_choice's
   refusal: a two-family power of 0 with a model dropped) write RESULTS/needs-ruling.json and record no choice.
``--dry-run`` checks completeness for real and, when complete, writes the commands it would run to
RESULTS/dry-run.json instead.

    nohup nice -n 19 .venv/bin/python scripts/watch_prompted.py --grid-out OUT --grid-manifest M/manifest.json \\
        --graph uniprot-sp-2026_03 --results RESULTS --bank-power BANK [--c5-inputs F] &
"""

from __future__ import annotations

import argparse
import datetime
import fcntl
import gzip
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.agent.replay_check import CHECK_FILE, current_records_sha256, halt_message
from harness.analysis.digest import code_digest

PHASE = "prompted"
GRID_OUT_SCHEMA = "pilot-grid-manifest/1"
WAITING, HALTED, COMPLETE, STEP3 = "waiting", "halted", "complete", "waiting for step 3"
EXIT = {COMPLETE: 0, WAITING: 3, HALTED: 4, "failed": 5, STEP3: 6}
PENDING = "waiting-step3.json"  # the C1 cells simulated; the aggregation and the choice wait for step 3's calibration
STEP3_PENDING = 3  # power_weighted.py aggregate's exit code while the config does not bind step 3
CACHE = "watch-cache.json"


def _run_pilot():
    module = sys.modules.get("run_pilot")
    if module is None:
        spec = importlib.util.spec_from_file_location("run_pilot", ROOT / "scripts" / "run_pilot.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["run_pilot"] = module
        spec.loader.exec_module(module)
    return module


class WatchError(RuntimeError):
    """An input the watcher refuses (not a wait: someone must look)."""


def route_models(grid_manifest: Path) -> dict[str, str]:
    """The grid's model names and the route models its records carry."""
    m = json.loads(Path(grid_manifest).read_text(encoding="utf-8"))
    return {x["name"]: x["route_model"] for x in m["spec"]["models"]}


def planned_attempts(grid_manifest: Path, graph: str, phase: str = PHASE) -> list[tuple[str, str, str, int]]:
    """The phase's planned attempts of the graph as runner keys (cell, question, route model, repetition), from the
    grid manifest's attempts file, refused unless it is the manifest's by sha256."""
    m = json.loads(Path(grid_manifest).read_text(encoding="utf-8"))
    schema = _run_pilot()._grid_builder().MANIFEST_SCHEMA
    if m.get("schema") != schema:
        raise WatchError(f"{grid_manifest}: schema {m.get('schema')!r}, not {schema!r}")
    raw = gzip.decompress((Path(grid_manifest).parent / m["attempts_file"]).read_bytes())
    if hashlib.sha256(raw).hexdigest() != m.get("attempts_sha256"):
        raise WatchError(f"{m['attempts_file']}: not the manifest's attempts (attempts_sha256)")
    route = {x["name"]: x["route_model"] for x in m["spec"]["models"]}
    rows = [json.loads(x) for x in raw.decode("utf-8").splitlines() if x.strip()]
    planned = [(r["cell"], r["question_id"], route[r["model"]], int(r["repetition"])) for r in rows
               if r["graph"] == graph and r["phase"] == phase]
    if not planned:
        raise WatchError(f"{grid_manifest}: no {phase} attempt planned for graph {graph}")
    return planned


def record_facts(out: Path, cache_path: Path | None) -> dict[tuple, dict[str, Any]]:
    """Each record's runner key with its file and outcome's failure type, every record read once and then cached by
    (name, size, mtime); refused when two records share a key (as run_pilot.record_paths refuses)."""
    rp = _run_pilot()
    cache = {}
    if cache_path is not None and cache_path.is_file():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except ValueError:
            cache = {}
    fresh, facts, dupes = {}, {}, []
    for path in sorted((Path(out) / "records").glob("*.json")):
        st = path.stat()
        stamp = [st.st_size, st.st_mtime_ns]
        hit = cache.get(path.name)
        if hit and hit["stamp"] == stamp and "ended_at" in hit:
            entry = hit
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
            key = rp.attempt_key(data["question_id"], data["model"], data["repetition"], rp.record_arm(data))
            entry = {"stamp": stamp, "key": list(key), "failure_type": (data.get("outcome") or {}).get("failure_type"),
                     "ended_at": data.get("ended_at")}
        fresh[path.name] = entry
        key = tuple(entry["key"])
        if key in facts:
            dupes.append(f"{key}: {facts[key]['file']} and {path.name}")
        facts[key] = {"file": path.name, "failure_type": entry["failure_type"], "ended_at": entry["ended_at"]}
    if cache_path is not None:
        tmp = cache_path.with_name(cache_path.name + ".tmp")
        tmp.write_text(json.dumps(fresh), encoding="utf-8")
        tmp.replace(cache_path)
    if dupes:
        raise WatchError("records share an attempt key:\n  " + "\n  ".join(dupes[:20]))
    return facts


FALLBACK_RULE = ("science's fallback rule (design log: Gemma 20 Oct 19:33Z, DeepSeek V4.1 21 Oct 03:22Z, GLM 21 Oct "
                 "08:10Z): a model named in --fallback, never one inferred from outcomes, is excused its incomplete "
                 "prompted remainder; its C1 planning input is its completed prompted records (before its cutoff, if "
                 "given) plus its natural rate, stated (power_weighted.py dev-rates --fallback); every other model must "
                 "be complete")


def fallback_models(args) -> dict[str, str | None]:
    """The explicit fallback list (``--fallback ROUTE-MODEL[=ISO-CUTOFF] ...``): model -> cutoff or None. Refused: an
    empty or repeated model, a cutoff that is not an ISO time."""
    out: dict[str, str | None] = {}
    for item in getattr(args, "fallback", None) or []:
        model, sep, cutoff = item.partition("=")
        if not model or model in out:
            raise WatchError(f"--fallback {item!r}: an empty or repeated model")
        if sep:
            try:
                zoned = datetime.datetime.fromisoformat(cutoff).tzinfo is not None
            except ValueError:
                zoned = False
            if not zoned:
                raise WatchError(f"--fallback {item!r}: its cutoff is not an ISO time with a time zone (e.g. Z)")
        out[model] = cutoff if sep else None
    return out


def _utc(value: str) -> datetime.datetime:
    t = datetime.datetime.fromisoformat(value)
    if t.tzinfo is None:
        raise WatchError(f"{value!r}: a record's end time without a time zone")
    return t.astimezone(datetime.UTC)


def _completed_before(fact: dict[str, Any], cutoff: str | None) -> bool:
    """A completed record: not failed infrastructure, and with a cutoff ended before it (UTC instants)."""
    if fact["failure_type"] == "infrastructure":
        return False
    if cutoff is None:
        return True
    end = fact.get("ended_at")
    return end is not None and _utc(end) < _utc(cutoff)


def prompted_status(out: Path, grid_manifest: Path, graph: str, cache_path: Path | None = None,
                    fallback: dict[str, str | None] | None = None) -> dict[str, Any]:
    """The prompted phase's state in OUT: waiting (with what is missing), halted, or complete. A model in ``fallback``
    (FALLBACK_RULE: given, never inferred) is excused its remainder, and its completed planned attempts are counted
    (``fallback`` in the state); every other model's planned attempts must all have records."""
    out = Path(out)
    fallback = fallback or {}
    planned = planned_attempts(grid_manifest, graph)
    stray = sorted(set(fallback) - {k[2] for k in planned})
    if stray:
        raise WatchError(f"--fallback {stray}: not a model of the planned prompted phase")
    path = out / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    if manifest.get("schema") != GRID_OUT_SCHEMA:
        raise WatchError(f"{path}: not a grid OUT's manifest ({GRID_OUT_SCHEMA})")
    phases = (manifest.get("fingerprint") or {}).get("phases") or {}
    if PHASE not in phases:
        return {"state": WAITING, "why": "the prompted phase has not started in OUT", "planned": len(planned)}
    arms = set((phases[PHASE] or {}).get("arms") or {})
    unpinned = sorted({c for c, *_ in planned} - arms)
    if unpinned:
        raise WatchError(f"{path}: the prompted phase pins {sorted(arms)}, not the planned cells {unpinned}")
    order_file = out / f"order-{PHASE}.json"
    if order_file.is_file():
        route = route_models(grid_manifest)
        ordered = {(c, q, route.get(m, m), int(r)) for m, c, q, r in
                   json.loads(order_file.read_text(encoding="utf-8"))["order"]}
        stray = sorted(ordered - set(planned))
        if stray:
            raise WatchError(f"{order_file}: {len(stray)} attempts outside the planned set (first {list(stray[0])}): "
                             f"the phase runs another grid manifest than {grid_manifest}")
    check = out / CHECK_FILE
    if check.is_file():
        halted = halt_message(out, json.loads(check.read_text(encoding="utf-8")), PHASE)
        if halted:
            return {"state": HALTED, "why": halted, "planned": len(planned)}
    facts = record_facts(out, cache_path)
    required = [k for k in planned if k[2] not in fallback]
    missing = [k for k in required if k not in facts]
    infra = [k for k in required if k in facts and facts[k]["failure_type"] == "infrastructure"]
    excused = {"fallback": {m: {"planned": sum(k[2] == m for k in planned),
                                "completed": sum(k[2] == m and k in facts and _completed_before(facts[k], c)
                                                 for k in planned), "cutoff": c}
                            for m, c in fallback.items()}} if fallback else {}
    if missing or infra:
        return {"state": WAITING, "why": f"{len(missing)} planned attempts without a record, {len(infra)} failed "
                                         f"infrastructure", "planned": len(planned),
                "missing": [list(k) for k in missing[:5]], "infrastructure": [list(k) for k in infra[:5]], **excused}
    return {"state": COMPLETE, "planned": len(planned), "records": len(facts), **excused}


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


PATH_ARGS = ("grid_out", "grid_manifest", "results", "config", "bank_power", "c5_inputs")


def absolute(args) -> argparse.Namespace:
    """Every path argument resolved against the watcher's own working directory: the steps run with cwd /tmp, where a
    relative path would name another file (the review of e479fc6)."""
    return argparse.Namespace(**{**vars(args), **{k: Path(v).resolve() for k in PATH_ARGS
                                                    if (v := getattr(args, k, None)) is not None}})


def commands(args) -> list[list[str]]:
    """The steps' commands, each under nice -n 19 with this copy's interpreter, every one with the config the choice is
    recorded against (its P is the simulated panel) and absolute paths."""
    args = absolute(args)
    py = [sys.executable]
    nice = ["nice", "-n", "19"]
    pw = [*nice, *py, str(ROOT / "scripts" / "power_weighted.py")]
    res, config = Path(args.results), ["--config", str(args.config)]
    rates = [*pw, "dev-rates", "--grid-out", str(args.grid_out), "--out", str(res / "dev-rates.json"), *config]
    if getattr(args, "fallback", None):  # the explicit list, as given
        rates += ["--fallback", *args.fallback]
    plan = [*pw, "plan", "--out", str(res / "power"), "--dev-rates", str(res / "dev-rates.json"), "--agent-only",
            *config]
    if args.c5_inputs and Path(args.c5_inputs).is_file():
        plan += ["--c5-inputs", str(args.c5_inputs)]
    run = [*pw, "run", "--out", str(res / "power"), "--jobs", str(args.jobs), "--no-aggregate", *config]
    return [rates, plan, run]


def aggregate_command(args) -> list[str]:
    """The aggregation on the final calibration (power_weighted.py aggregate; exit STEP3_PENDING while the config does
    not bind step 3), with this copy's interpreter and config."""
    args = absolute(args)
    return ["nice", "-n", "19", sys.executable, str(ROOT / "scripts" / "power_weighted.py"), "aggregate", "--out",
            str(Path(args.results) / "power"), "--config", str(args.config)]


def _replay_current(out: Path) -> bool | None:
    """Whether OUT's stored replay check was computed on the records as they are now (None: no check stored)."""
    path = Path(out) / CHECK_FILE
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("records_sha256") == current_records_sha256(Path(out))


def fire(args, status: dict[str, Any], runner: Callable[[list[str]], int | None]) -> dict[str, Any] | None:
    """The steps, in order, then ``finish``."""
    for cmd in commands(args):
        runner(cmd)
    return finish(args, status, runner)


def finish(args, status: dict[str, Any], runner: Callable[[list[str]], int | None]) -> dict[str, Any] | None:
    """The aggregation on the final calibration of both families (the review of 1e31f1f: never the one-sided one); while
    the config does not bind step 3 it exits STEP3_PENDING, and the pending marker (PENDING) is written (once, with the
    completed phase's status) and None returned: each later look retries this alone. Then the C1a choice on the config
    copy (unless entry 81 needs a ruling) and the done marker."""
    res = Path(args.results)
    if runner(aggregate_command(args)) == STEP3_PENDING:
        if not (res / PENDING).is_file():
            (res / PENDING).write_text(json.dumps({
                "since": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"), "status": status,
                "why": "the C1 cells are simulated; the registered powers and the C1a choice wait for calibration step "
                       "3's final calibration (entry 106): power_weighted.py aggregate runs once --config binds it"},
                indent=1) + "\n", encoding="utf-8")
        return None
    power = json.loads((res / "power" / "power.json").read_text(encoding="utf-8"))
    c1a = power.get("c1a_powers")
    if not c1a:
        raise WatchError(f"{res / 'power' / 'power.json'}: no C1a powers (the C1 cells were not simulated)")
    config = res / "prereg-config.json"
    if c1a.get("needs_ruling"):  # entry 81's rates missing, or c1a_choice refused (a two-family power of 0)
        (res / "needs-ruling.json").write_text(json.dumps({"why": c1a["needs_ruling"], "entry81": power.get("entry81"),
                                                           "c1a_comparison": power.get("c1a_comparison")}, indent=1)
                                               + "\n", encoding="utf-8")
        recorded = {}
    else:
        # the pooled C1a is eligible only if entry 81 kept all seven models (prereg.C1A_ELIGIBILITY): the dropped
        # models go to choose-c1a, which records the two-family C1a and why when there are any
        if not config.exists():
            shutil.copy2(args.config, config)
        runner(["nice", "-n", "19", sys.executable, str(ROOT / "scripts" / "calibrate.py"), "choose-c1a", "--config",
                str(config), "--power-pooled", repr(c1a["pooled C1a"]), "--power-two-family",
                repr(c1a["two-family"]), "--entry81-dropped", *c1a["entry81_dropped"]])
        recorded = json.loads(config.read_text(encoding="utf-8"))
    done = {"fired_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            "commit": (ROOT / "COMMIT").read_text(encoding="utf-8").strip() if (ROOT / "COMMIT").is_file() else None,
            "code_digest": code_digest(), "grid_out": str(args.grid_out), "graph": args.graph, "status": status,
            "replay_check_current": _replay_current(Path(args.grid_out)),
            "dev_rates_sha256": _sha(res / "dev-rates.json"), "power_sha256": _sha(res / "power" / "power.json"),
            "bank_power": str(args.bank_power) if args.bank_power else None,
            "calibration": power.get("calibration"),
            "fallback": ({"rule": FALLBACK_RULE, "models": status.get("fallback"),
                          "dev_rates": _json_or_none(res / "dev-rates.json").get("fallback")}
                         if status.get("fallback") else None),
            "c1a_powers": c1a, "c1a_comparison": power.get("c1a_comparison"), "needs_ruling": c1a.get("needs_ruling"),
            "powers": {"weighted_holm": {k: c1a.get(k) for k in ("pooled C1a", "two-family")},
                       "first_level": c1a.get("first_level")},
            "c1a_family": recorded.get("c1a_family"), "c1a_power": recorded.get("c1a_power"),
            "c1a_eligibility": recorded.get("c1a_eligibility"), "entry81_dropped": c1a.get("entry81_dropped"),
            "config_copy": str(config) if recorded else None,
            "config_copy_sha256": _sha(config) if recorded else None,
            "c5_inputs": (str(args.c5_inputs), _sha(args.c5_inputs))
            if args.c5_inputs and Path(args.c5_inputs).is_file() else None}
    tmp = res / "done.json.tmp"
    tmp.write_text(json.dumps(done, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(res / "done.json")
    return done


def _json_or_none(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _subprocess(cmd: list[str]) -> int:
    """A step's exit code: 0, or STEP3_PENDING from the aggregation; any other failure raises."""
    code = subprocess.run(cmd, check=False, cwd="/tmp").returncode
    if code == STEP3_PENDING and cmd[5:6] == ["aggregate"]:
        return code
    if code:
        raise subprocess.CalledProcessError(code, cmd)
    return 0


def _markers(res: Path) -> str | None:
    if (res / "done.json").is_file():
        print(f"{res / 'done.json'}: already fired; nothing to do")
        return COMPLETE
    if (res / "failed.json").is_file():
        raise WatchError(f"{res / 'failed.json'}: an earlier firing failed; look at it, then move it away to retry")
    return None


def check_once(args, runner: Callable[[list[str]], int | None] = _subprocess) -> str:
    """One look: the state, firing (once) when complete. A done marker makes it a no-op, a failure marker a refusal;
    both are read again once the lock is held, since another watcher may have fired or failed and released the lock
    in between (the review of e479fc6)."""
    args = absolute(args)
    res = Path(args.results)
    res.mkdir(parents=True, exist_ok=True)
    early = _markers(res)
    if early:
        return early
    with open(res / ".lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise WatchError(f"{res}: another watcher holds the lock") from None
        held = _markers(res)
        if held:
            return held
        if (res / PENDING).is_file():  # simulated on the complete phase: only the aggregation and the choice are left
            stamp = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
            status = json.loads((res / PENDING).read_text(encoding="utf-8"))["status"]
            try:
                done = None if args.dry_run else finish(args, status, runner)
            except Exception as exc:
                (res / "failed.json").write_text(json.dumps({"at": stamp, "error": repr(exc), "status": status},
                                                            indent=1) + "\n", encoding="utf-8")
                raise
            print(f"{stamp} {STEP3 if done is None else 'fired'}: {res / PENDING}", flush=True)
            return STEP3 if done is None else COMPLETE
        status = prompted_status(Path(args.grid_out), Path(args.grid_manifest), args.graph, res / CACHE,
                                 fallback_models(args))
        stamp = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
        print(f"{stamp} {status['state']}: {status.get('why', '')} (planned {status['planned']})", flush=True)
        if status["state"] != COMPLETE:
            return status["state"]
        if args.dry_run:
            (res / "dry-run.json").write_text(json.dumps({"status": status, "commands": [
                *commands(args), aggregate_command(args)]}, indent=1) + "\n", encoding="utf-8")
            print(f"{res / 'dry-run.json'}: the commands it would run")
            return COMPLETE
        try:
            done = fire(args, status, runner)
        except Exception as exc:
            (res / "failed.json").write_text(json.dumps({"at": stamp, "error": repr(exc), "status": status}, indent=1)
                                             + "\n", encoding="utf-8")
            raise
        if done is None:
            print(f"simulated; the aggregation and the C1a choice wait for step 3: {res / PENDING}", flush=True)
            return STEP3
        print(f"fired: C1a family {done['c1a_family'] or 'not recorded: ' + str(done['needs_ruling'])} from powers "
              f"{done['c1a_powers']}; {res / 'done.json'}")
        return COMPLETE


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--grid-out", type=Path, required=True, help="the graph's grid OUT (read only)")
    ap.add_argument("--grid-manifest", type=Path, required=True, help="the grid manifest.json the phase runs")
    ap.add_argument("--graph", required=True)
    ap.add_argument("--results", type=Path, required=True, help="the fixed results directory")
    ap.add_argument("--config", type=Path, default=ROOT / "data" / "analysis" / "prereg-config.json")
    ap.add_argument("--bank-power", type=Path, default=None, help="the power_weighted.py OUT of the bank cells (named)")
    ap.add_argument("--c5-inputs", type=Path, default=None, help="C5's entry-77 counts, used if present at firing")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--fallback", nargs="*", default=[], metavar="ROUTE-MODEL[=ISO-CUTOFF]",
                    help="models excused their incomplete prompted remainder by science's fallback rule (given, never "
                         "inferred); their C1 input is their completed records plus their natural rate")
    ap.add_argument("--interval", type=float, default=300.0, help="seconds between looks")
    ap.add_argument("--once", action="store_true",
                    help="look once and exit (0 fired or done, 3 waiting, 4 halted, 6 waiting for step 3)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if args.jobs > 4:
        ap.error("--jobs: at most 4 processes on the shared spoke")
    while True:
        try:
            state = check_once(args)
        except (WatchError, subprocess.CalledProcessError) as exc:
            print(f"refused: {exc}", file=sys.stderr, flush=True)
            return EXIT["failed"]
        if state == COMPLETE or args.once:
            return EXIT[state]
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
