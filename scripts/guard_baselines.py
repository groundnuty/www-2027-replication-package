"""The guard rails' baselines from dev records (science, the design log Oct 14:34), per model and per phase,
for run_pilot.py's --latency-baseline-s MODEL=S and --wallcap-baseline MODEL=R. Read-only on the OUTs.

    python3 -I scripts/guard_baselines.py --source uniprot-natural=OUT@natural --source uniprot-prompted=OUT@prompted \
        [--source NAME=OUT@PHASE ...] [--as-of ISO] [--since MODEL=ISO] [--until MODEL=ISO] [--min-attempts 50] \
        --out baselines.json

A source is one OUT's records of one phase (natural: the condition "none"; prompted: any other condition; branch: a
checker arm's record branched from a parent). Every value is computed from the records that ended by --as-of (default:
now), so a live OUT gives a reproducible answer; each OUT's file list is digested in the output.

* The latency baseline is the calm-period p50 of the guard's own measure: the per-turn call time (the request with
  its retries and backoff, the record's per-turn wall_s, else its latency_s, over the turns the try asked the model
  for: run_pilot's _call_times and live_turns), over every turn of every try (records/ and infrastructure/), as the
  guard's latency window reads it. Not the first byte: the guard compares its window's median of the call time with
  twice the baseline, and a streamed Gateway turn's first byte is a fraction of its call time, so a first-byte
  baseline would fire the trigger at normal speed. A turn is left out when its request starts in a sync minute (the
  Gateway's front-end sync: minutes :06-:14 and :36-:44 of every hour, and :52-:58 of even UTC hours) or in an
  all-model stall bin. The p50 with the turn's whole span checked instead, and the p50 over every turn, are given
  beside it.
* The stall bins are the stall tools' (runs/dev-natural-20261020/spec/stalls.py, lead, 20 Oct): 2-minute UTC
  bins in which 3 or more models' first-byte p90 (xs sorted, xs[min(n - 1, int(0.9 n))]) exceeds 15 s, counting a
  model with at least 3 turns there; a turn is placed at its request's start. They are recomputed here from every
  live turn of every record and infrastructure try of every OUT given (re-implemented, not imported: stalls.py
  appends its caches to the OUT's logs/). Live turns, unlike stalls.py's every stored turn: a branch record's
  replayed initial turns are its parent's, timed in the parent's run (stalls.py never read a branch record).
* An attempt that failed infrastructure on every try has its final try written both to infrastructure/ and to
  records/ (run_pilot); it is counted once, as the guard counts it (attempt_stats).
* The wall-cap rate is the share of the model's agent outcomes (records/, not a final infrastructure failure) that
  ended capped by the wall clock (outcome.cap "wall"), as the guard's wall-cap window counts them; the baseline is
  that rate with a floor of 1%. It is not restricted to calm minutes: a 1,800 s attempt always spans a sync minute,
  so a calm filter would drop every wall cap. The rate over the attempts that met no stall bin is given beside it.
* The model's current deployment: its records that started at or after its latest recorded deployment change (the
  OUTs' MODEL_CHANGED.json, or a later --since MODEL=ISO) and before --until MODEL=ISO (a provider fault). The
  records left must carry at most one system fingerprint, and that the recorded change's, or the tool refuses.
  Records written before the fingerprint was recorded (no fingerprint) count as the deployment of their time.

A model with fewer than --min-attempts attempts (the latency trigger's own window is 50 attempts) gets no value, and
the output says why. Exit status: 0 written; 2 refused (two deployments mixed, a malformed input).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = "guard-baselines/1"
BIN_S = 120
STALL_FIRST_BYTE_S = 15.0
STALL_MIN_MODELS = 3
STALL_MIN_TURNS = 3
SYNC_MINUTES = ((6, 14), (36, 44))  # every hour
EVEN_HOUR_SYNC_MINUTES = (52, 58)  # even UTC hours only
WALLCAP_FLOOR = 0.01
MIN_ATTEMPTS = 50
PHASES = ("natural", "prompted", "branch")
DEFINITIONS = {
    "call_time": "per-turn wall_s, else latency_s (run_pilot _call_times), over the live turns of every try "
                 "(records/ and infrastructure/)",
    "sync_minute": "minutes :06-:14 and :36-:44 of every UTC hour, :52-:58 of even UTC hours",
    "stall_bin": f"{BIN_S} s UTC bins with {STALL_MIN_MODELS}+ models whose first-byte p90 exceeds "
                 f"{STALL_FIRST_BYTE_S:g} s, {STALL_MIN_TURNS}+ turns each (stalls.py), over every OUT given",
    "calm_turn": "a turn whose request starts outside the sync minutes and the stall bins",
    "calm_span_turn": "a turn whose whole span [start, start + call time] meets no sync minute and no stall bin",
    "wallcap_rate": "agent outcomes (records/, not a final infrastructure failure) with outcome.cap 'wall', over "
                    "those outcomes; not calm-filtered",
    "wallcap_baseline": f"max(wallcap_rate, {WALLCAP_FLOOR})",
    "deployment": "records started at or after the latest MODEL_CHANGED.json change (or a later --since) and before "
                  "--until; one system fingerprint at most",
}


class Refused(Exception):
    pass


def parse_ts(text: str) -> float:
    return datetime.fromisoformat(text).timestamp()


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sync_minute(ts: float) -> bool:
    t = datetime.fromtimestamp(ts, tz=UTC)
    if any(lo <= t.minute <= hi for lo, hi in SYNC_MINUTES):
        return True
    lo, hi = EVEN_HOUR_SYNC_MINUTES
    return t.hour % 2 == 0 and lo <= t.minute <= hi


def phase_of(rec: dict) -> str:
    if (rec.get("checker") or {}).get("branched_from"):
        return "branch"
    return "natural" if rec.get("condition") == "none" else "prompted"


def live_turns(rec: dict) -> list[dict]:
    """run_pilot.live_turns: a branch record's replayed initial phase is its parent's."""
    per_turn = (rec.get("tokens") or {}).get("per_turn") or []
    checker = rec.get("checker")
    if checker and checker.get("branched_from"):
        initial_turns = checker["initial"]["outcome"]["turns"]
        return [t for t in per_turn if t["turn"] > initial_turns]
    return per_turn


def turn_offset(rec: dict) -> float:
    """The wall-clock time a record's turn offsets count from (run_pilot.calm_turn_offset): a branch record's offsets
    run on from its replayed parent's, so its live turns are anchored at the retry's first request as stamped
    (load.started_at), else at started_at less the parent's replayed time (wall_s less ended_at - started_at)."""
    t0 = parse_ts(rec["started_at"])
    live = live_turns(rec)
    if not ((rec.get("checker") or {}).get("branched_from") and live):
        return t0
    stamped = (rec.get("load") or {}).get("started_at")
    if stamped:
        return parse_ts(stamped) - float(live[0]["started_s"])
    return t0 - (float(rec["wall_s"]) - (parse_ts(rec["ended_at"]) - t0))


def call_times(rec: dict) -> list[tuple[float, float]]:
    """(the request's start, its call time) per live turn, the call time as run_pilot._call_times reads it."""
    t0 = turn_offset(rec)
    return [(t0 + float(t["started_s"]), float(t.get("wall_s") or t["latency_s"])) for t in live_turns(rec)
            if isinstance(t.get("wall_s") or t.get("latency_s"), (int, float))]


def compact(rec: dict, kind: str, file: str) -> dict:
    """What this tool reads from one record or infrastructure try."""
    t0 = turn_offset(rec)
    turns = [[t0 + float(t["started_s"]), t.get("first_byte_s")] for t in live_turns(rec)]
    fps = sorted({str(t["system_fingerprint"]) for t in (rec.get("tokens") or {}).get("per_turn") or []
                  if t.get("system_fingerprint")})
    outcome = rec.get("outcome") or {}
    return {"kind": kind, "file": file, "run_id": rec.get("run_id"), "model": rec["model"], "phase": phase_of(rec),
            "started": parse_ts(rec["started_at"]),
            "ended": parse_ts(rec.get("ended_at") or rec["started_at"]), "cap": outcome.get("cap"),
            "final_infra": outcome.get("failure_type") == "infrastructure", "fingerprints": fps,
            "calls": call_times(rec), "first_bytes": [(s, fb) for s, fb in turns if isinstance(fb, (int, float))]}


def p90(xs: list[float]) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(0.9 * len(xs)))]


def stall_bins(rows: list[dict]) -> set[int]:
    bins: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for start, fb in row["first_bytes"]:
            bins[int(start // BIN_S)][row["model"]].append(float(fb))
    found = set()
    for b, models in bins.items():
        over = sum(1 for xs in models.values() if len(xs) >= STALL_MIN_TURNS and p90(xs) > STALL_FIRST_BYTE_S)
        if over >= STALL_MIN_MODELS:
            found.add(b)
    return found


def span_meets(start: float, end: float, stalls: set[int]) -> bool:
    """Whether [start, end] meets a sync minute or a stall bin (every minute and bin it touches is checked)."""
    if any(b in stalls for b in range(int(start // BIN_S), int(end // BIN_S) + 1)):
        return True
    return any(sync_minute(m * 60) for m in range(int(start // 60), int(end // 60) + 1))


def median(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def measure(sources: dict[str, tuple[str, str]], loaded: dict[str, list[dict]], *, names: dict[str, str],
            min_attempts: int = MIN_ATTEMPTS, changes: dict[str, tuple[float, str]] | None = None,
            since: dict[str, float] | None = None, until: dict[str, float] | None = None,
            floor: float = WALLCAP_FLOOR, all_turns: set[tuple[str, str]] | None = None) -> dict:
    """``sources``: name -> (OUT key of ``loaded``, phase); ``loaded``: OUT key -> compact rows; ``names``: route
    model -> the runner's model name; ``changes``: model -> (the latest recorded deployment change, its
    fingerprint)."""
    changes, since, until, all_turns = changes or {}, since or {}, until or {}, all_turns or set()
    loaded, duplicates = dict(loaded), {}
    for key, rows in loaded.items():  # a final infrastructure try is in records/ too: once, as attempt_stats has it
        counted = {r["run_id"] for r in rows if r["kind"] == "records" and r["run_id"]}
        loaded[key] = [r for r in rows if not (r["kind"] == "infrastructure" and r["run_id"] in counted)]
        duplicates[key] = len(rows) - len(loaded[key])
    stalls = stall_bins([row for rows in loaded.values() for row in rows])
    start_of = {m: max(t for t in (since.get(m), changes.get(m, (None,))[0]) if t is not None)
                for m in set(since) | set(changes)}
    by_model: dict[str, dict] = {}
    result: dict[str, dict] = {}
    for source, (out_key, phase) in sources.items():
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in loaded[out_key]:
            if row["phase"] == phase:
                groups[names.get(row["model"], row["model"])].append(row)
        models = {}
        for model, rows in sorted(groups.items()):
            lo, hi = start_of.get(model), until.get(model)
            before = sum(1 for r in rows if lo is not None and r["started"] < lo)
            after = sum(1 for r in rows if hi is not None and r["started"] >= hi)
            kept = [r for r in rows if (lo is None or r["started"] >= lo) and (hi is None or r["started"] < hi)]
            fps = sorted({fp for r in kept for fp in r["fingerprints"]})
            if len(fps) > 1:
                raise Refused(f"{model} in {source}: its records after the deployment cut carry {len(fps)} system "
                              f"fingerprints ({', '.join(fps)}): two deployments mixed; give --since {model}=ISO at "
                              f"the latest change (or --until before it)")
            if model in changes and fps and fps != [changes[model][1]]:
                raise Refused(f"{model} in {source}: fingerprint {fps[0]} after its recorded deployment change to "
                              f"{changes[model][1]}")
            by_model.setdefault(model, {"fingerprints": set()})["fingerprints"].update(fps)
            calls = [(r_i, s, c) for r_i, r in enumerate(kept) for s, c in r["calls"]]
            sync = [c for _, s, c in calls if sync_minute(s)]
            stall = [c for _, s, c in calls if not sync_minute(s) and int(s // BIN_S) in stalls]
            calm = [(r_i, c) for r_i, s, c in calls if not sync_minute(s) and int(s // BIN_S) not in stalls]
            calm_span = [c for _, s, c in calls if not span_meets(s, s + c, stalls)]
            outcomes = [r for r in kept if r["kind"] == "records" and not r["final_infra"]]
            wall = [r for r in outcomes if r["cap"] == "wall"]
            stall_free = [r for r in outcomes
                          if not any(b in stalls for s, c in r["calls"]
                                     for b in range(int(s // BIN_S), int((s + c) // BIN_S) + 1))]
            calm_attempts = len({r_i for r_i, _ in calm})
            p50 = median([c for _, c in calm])
            rate = len(wall) / len(outcomes) if outcomes else None
            why = []
            if calm_attempts < min_attempts:
                why.append(f"calm turns from {calm_attempts} attempts (< {min_attempts})")
            if len(outcomes) < min_attempts:
                why.append(f"{len(outcomes)} agent outcomes (< {min_attempts})")
            latency = round(p50, 2) if p50 is not None and calm_attempts >= min_attempts else None
            basis = "calm"
            if (source, model) in all_turns:  # a ruling's exception: every turn's p50 (no calm minutes to read)
                p_all, basis = median([c for _, _, c in calls]), "all_turns"
                latency = round(p_all, 2) if p_all is not None and len(outcomes) >= min_attempts else None
                why = [w for w in why if not w.startswith("calm turns")]
            wallcap = round(max(rate, floor), 4) if rate is not None and len(outcomes) >= min_attempts else None
            if latency is not None and latency <= 0:  # run_pilot refuses it: --latency-baseline-s must be > 0
                why.append(f"the p50 {p50!r} s rounds to 0")
                latency = None
            if wallcap is not None and not 0 <= wallcap < 1:  # run_pilot refuses it: 0 <= --wallcap-baseline < 1
                why.append(f"the wall-cap rate {rate!r} rounds to {wallcap}")
                wallcap = None
            models[model] = {
                "records_before_deployment": before, "records_after_until": after,
                "turns": len(calls), "turns_sync": len(sync), "turns_stall": len(stall), "turns_calm": len(calm),
                "calm_attempts": calm_attempts, "p50_calm_s": p50, "p50_calm_span_s": median(calm_span),
                "turns_calm_span": len(calm_span), "p50_all_s": median([c for _, _, c in calls]),
                "attempts": len(outcomes), "wall_capped": len(wall), "wallcap_rate": rate,
                "attempts_stall_free": len(stall_free),
                "wallcap_rate_stall_free": (sum(r["cap"] == "wall" for r in stall_free) / len(stall_free)
                                            if stall_free else None),
                "latency_baseline_s": latency, "latency_basis": basis, "wallcap_baseline": wallcap,
                "fingerprints": fps, "why_none": "; ".join(why) or None}
        result[source] = {"out": out_key, "phase": phase, "models": models}
    # each value on its own: a model may have one (a launch measures a missing latency value on calm calls, and runs
    # without a wall-cap trigger where it has no wall-cap value)
    flags = {source: {m: {"latency_baseline_s": v["latency_baseline_s"], "wallcap_baseline": v["wallcap_baseline"]}
                      for m, v in body["models"].items()
                      if v["latency_baseline_s"] is not None or v["wallcap_baseline"] is not None}
             for source, body in result.items()}
    deployment = {m: {"since": iso(start_of[m]) if m in start_of else None,
                      "since_from": ("MODEL_CHANGED.json" if m in changes and start_of[m] == changes[m][0]
                                     else "--since" if m in start_of else None),
                      "until": iso(until[m]) if m in until else None,
                      "fingerprints": sorted(v["fingerprints"])}
                  for m, v in sorted(by_model.items())}
    return {"stall_bins": sorted(iso(b * BIN_S) for b in stalls), "deployment": deployment, "sources": result,
            "flags": flags, "infrastructure_duplicates": duplicates}


def load_out(out: Path, *, as_of: float) -> dict:
    """One OUT's records and infrastructure tries that ended by ``as_of`` (compact), its route -> name map
    (served-models.json) and its latest deployment change per model (MODEL_CHANGED.json)."""
    rows = []
    for kind in ("records", "infrastructure"):
        d = out / kind
        for p in sorted(d.glob("*.json")) if d.is_dir() else []:
            rec = json.loads(p.read_text(encoding="utf-8"))
            row = compact(rec, kind, f"{kind}/{p.name}")
            if row["ended"] <= as_of:
                rows.append(row)
    names, served_sha256 = {}, None
    served = out / "served-models.json"
    if served.exists():
        raw = served.read_bytes()
        served_sha256 = hashlib.sha256(raw).hexdigest()
        for name, entry in (json.loads(raw).get("models") or {}).items():
            if entry.get("route_model"):
                names[entry["route_model"]] = name
    changes: dict[str, tuple[float, str]] = {}
    used = []  # the changes logged by as_of: a later one must not move the cut of a re-run with the same as_of
    changed = out / "MODEL_CHANGED.json"
    if changed.exists():
        for c in json.loads(changed.read_text(encoding="utf-8")).get("changes") or []:
            at = parse_ts(c["at"])
            if at > as_of:
                continue
            used.append(c)
            if c["model"] not in changes or at > changes[c["model"]][0]:
                changes[c["model"]] = (at, c.get("system_fingerprint"))
    files = sorted(r["file"] for r in rows)
    return {"rows": rows, "names": names, "changes": changes,
            "records": sum(r["kind"] == "records" for r in rows),
            "infrastructure": sum(r["kind"] == "infrastructure" for r in rows),
            "files_sha256": hashlib.sha256("\n".join(files).encode("utf-8")).hexdigest(),
            "served_models_sha256": served_sha256,
            "changes_used_sha256": hashlib.sha256(json.dumps(used, sort_keys=True).encode("utf-8")).hexdigest()}


def table(result: dict) -> str:
    lines = [(f"{'source':18s} {'model':24s} {'p50 calm':>8s} {'span':>6s} {'all':>6s} {'turns':>6s} {'calm':>6s} "
              f"{'sync':>5s} {'stall':>5s} {'att':>5s} {'wall':>4s} {'rate':>6s} {'L flag':>7s} {'W flag':>7s}  why")]

    def f(x, nd=2):
        return "-" if x is None else f"{x:.{nd}f}"
    for source, body in result["sources"].items():
        for model, v in body["models"].items():
            lines.append(f"{source:18s} {model:24s} {f(v['p50_calm_s']):>8s} {f(v['p50_calm_span_s']):>6s} "
                         f"{f(v['p50_all_s']):>6s} {v['turns']:6d} {v['turns_calm']:6d} {v['turns_sync']:5d} "
                         f"{v['turns_stall']:5d} {v['attempts']:5d} {v['wall_capped']:4d} {f(v['wallcap_rate'], 4):>6s} "
                         f"{f(v['latency_baseline_s']):>7s} {f(v['wallcap_baseline'], 4):>7s}  {v['why_none'] or ''}")
    return "\n".join(lines)


def _per_model_ts(p: argparse.ArgumentParser, items: list[str], flag: str) -> dict[str, float]:
    out = {}
    for item in items:
        model, sep, raw = item.partition("=")
        try:
            out[model] = parse_ts(raw)
        except ValueError:
            sep = ""
        if not sep or not model:
            p.error(f"{flag} {item!r}: expected MODEL=ISO-TIME")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--source", action="append", required=True, metavar="NAME=OUT@PHASE",
                   help=f"one OUT's records of one phase ({', '.join(PHASES)}) (repeatable)")
    p.add_argument("--as-of", default=None, help="only records that ended by this time (default: now)")
    p.add_argument("--since", action="append", default=[], metavar="MODEL=ISO",
                   help="the model's deployment start, if later than its MODEL_CHANGED.json change (repeatable)")
    p.add_argument("--until", action="append", default=[], metavar="MODEL=ISO",
                   help="leave out the model's records started at or after this time (a fault) (repeatable)")
    p.add_argument("--all-turns", action="append", default=[], metavar="SOURCE=MODEL",
                   help="take this source's model's latency value from every turn, not the calm ones (a ruling's "
                        "exception, e.g. a dev run that lay inside a sync window) (repeatable)")
    p.add_argument("--min-attempts", type=int, default=MIN_ATTEMPTS)
    p.add_argument("--floor", type=float, default=WALLCAP_FLOOR)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    sources = {}
    for item in args.source:
        name, sep, rest = item.partition("=")
        out, at, phase = rest.rpartition("@")
        if not sep or not name or not at or phase not in PHASES:
            p.error(f"--source {item!r}: expected NAME=OUT@PHASE with PHASE one of {', '.join(PHASES)}")
        sources[name] = (str(Path(out).expanduser()), phase)
    as_of = parse_ts(args.as_of) if args.as_of else float(int(time.time()))
    if as_of > time.time():  # records still to end would change the answer on a re-run
        p.error(f"--as-of {args.as_of} is in the future")
    since, until = _per_model_ts(p, args.since, "--since"), _per_model_ts(p, args.until, "--until")
    loaded, names, changes, inputs = {}, {}, {}, {}
    for out in sorted({o for o, _ in sources.values()}):
        got = load_out(Path(out), as_of=as_of)
        loaded[out] = got["rows"]
        names.update(got["names"])
        for model, change in got["changes"].items():
            if model not in changes or change[0] > changes[model][0]:
                changes[model] = change
        inputs[out] = {k: got[k] for k in ("records", "infrastructure", "files_sha256", "served_models_sha256",
                                           "changes_used_sha256")}
    try:
        all_turns = set()
        for item in args.all_turns:
            source, sep, model = item.partition("=")
            if not sep or source not in sources or not model:
                raise Refused(f"--all-turns {item!r}: expected SOURCE=MODEL with SOURCE one of the --source names")
            all_turns.add((source, model))
        result = measure(sources, loaded, names=names, min_attempts=args.min_attempts, changes=changes,
                         since=since, until=until, floor=args.floor, all_turns=all_turns)
        absent = sorted(f"{s}={m}" for s, m in all_turns if m not in result["sources"][s]["models"])
        if absent:  # a mistyped name would leave the ruling's value unused
            raise Refused(f"--all-turns {absent}: no such model in that source")
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    doc = {"schema": SCHEMA, "as_of": iso(as_of), "definitions": DEFINITIONS,
           "parameters": {"min_attempts": args.min_attempts, "floor": args.floor,
                          "since": {m: iso(t) for m, t in since.items()},
                          "until": {m: iso(t) for m, t in until.items()},
                          "all_turns": sorted(f"{s}={m}" for s, m in all_turns)},
           "inputs": inputs, **result}
    args.out.write_text(json.dumps(doc, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    print(table(result))
    print(f"stall bins {len(result['stall_bins'])}; written {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
