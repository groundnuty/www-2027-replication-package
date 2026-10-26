#!/usr/bin/env python3
"""Set aside an attempt lost to a defect in our harness, so a resume re-runs it.

Design ruling (design log): a harness defect is not an agent error. The attempt's
record is moved, byte for byte unchanged and under its own name, to OUT/harness/
(it stays a valid, replayable run record), and the decision is written beside
it in a sidecar named after it, OUT/harness/<record name>.setaside.json for
<record name>.json (label ``failed:harness``, the original label, the defect and
the ruling): <claim digest>.<run_id>.setaside.json for a record the runner named
by its attempt's claim digest, <run_id>.setaside.json for an older one. Its line
leaves runs.jsonl and its claim is deleted (the attempt's marker stays, so both
files keep their split by name), so ``run_pilot.py`` with the same arguments re-runs exactly that
attempt with the fixed harness. Reports read the sidecars and count them per
arm. It refuses to run while a runner holds the run's lock.

Crash safety: the steps run in the order move the record, drop its runs.jsonl
line, delete its claim, write the sidecar. The sidecar comes last, so its
presence means the set-aside finished. A run that stopped part way leaves the
record in harness/ without a sidecar; running the same command again detects
that and finishes the remaining steps (each is idempotent). The claim is
released only if no record of the attempt is in records/: a runner may already
have re-run it, and then the claim belongs to the replacement. The intermediate
states are harmless meanwhile: the record is in no counted place, a report sees
it as a paid try, and until its claim goes no runner re-runs it.

    uv run python scripts/relabel_harness_failure.py OUT --question QID --model ROUTE_MODEL \\
        --repetition N --defect "why" --ruling URL
    uv run python scripts/relabel_harness_failure.py GRID_OUT --arm CELL --question QID ...

A development-grid OUT (scripts/run_pilot.py --grid-manifest; the design log) holds every
cell of a graph, so an attempt is (arm, question, model, repetition): ``--arm`` names the cell, and
is required there and refused elsewhere. In a grid this is also the supported path for an initial
attempt (natural or prompted) that failed infrastructure on every try (unscored, and recorded, so a
resume counts it as done): set aside, its phase runs it again, and the branch phase, which waits for
every natural and prompted attempt to have a counted record, can start. In a grid such a decision is labelled
``failed:infrastructure`` by default, not ``failed:harness``: an outage is no harness defect.
``--label`` says otherwise where the ruling does (an "infrastructure" outcome that was the
harness's misreading); outside a grid the default stays ``failed:harness``. A branch set aside
(its parent was replaced, which the branch phase refuses) is branched again from the current
parent on the next start.

A record whose online checker decision diverges from its offline recompute (the replay check holds
the batch) is set aside on the divergence alone (design ruling, design log, 20 Oct 07:26):

    uv run python scripts/relabel_harness_failure.py OUT --diverged RUN_ID --ruling URL [--defect "why"]

It works from the stored replay check's held record (its file, the sha256 of its bytes, its
attempt, the online and the offline decision) and never opens the record. The check must cover
every record now in the batch, by count and by hash (the only change it allows is its own
divergence set-asides), and the halt rule is applied again to its per-record inputs, never to the
verdict it stored (release review, item 6). The record is moved
unopened once its bytes are the ones the check held, and its score and outcome are neither read nor
printed. The sidecar records the divergence, the sha256 of the record's bytes and the
classification (label ``failed:harness``, failure_type harness_defect, classification
checker_divergence); the check counts it as its own only while it says what the check held
(release re-review, item 2). A set-aside record is left out
of the recompute and the analysis. Its claim is kept: condition 3 holds for every such set-aside,
so its attempt is re-run only once a fix is recorded, under that fix (scripts/run_pilot.py never
starts nor reclaims it before; release review, item 2). Refused once the check HALTS the record's
phase (more than 1% of its records diverge, or divergences occur in more than one model: the
checker code is suspect). Once the cause is fixed and committed and the batch verified again under
the fixed code (code whose tree digest differs from the code of every divergent decision of the
phase: another commit of the same code is no fix), one command sets aside every divergent record of
the phase, halted or not, and
stamps the fix (the fixed code's commit and tree digest, the defect, the ruling) on every
checker_divergence sidecar of it, releasing their claims:

    uv run python scripts/relabel_harness_failure.py OUT --checker-fix PHASE --defect "why" --ruling URL

A run's phase is ``run``; a grid's is natural, prompted or branch. The phase then re-runs its
set-aside attempts only under the fixed code (scripts/run_pilot.py refuses other code), so their
records name it. The attempt path (--question ...) refuses, before it finishes or opens anything,
an attempt whose record the stored check holds (in records/, or in harness/ where an interrupted
--diverged left it) and any attempt of a phase the check halts; it never opens a held record.

A confirmed infrastructure-fault window (change-log entries 129, 148 and 149; scripts/infra_fault_monitor.py prints
the command) sets aside, in one command, every attempt of one model with any model turn in the window (an assistant
turn whose interval, its start to its start plus its wall time, meets the window, or the attempt's start in it;
harness/agent/infra_fault.py active_in), in every arm and whatever its outcome, as ``failed:infrastructure``, each as the attempt path would (record kept byte for byte in harness/, its
sidecar tagged with the window, its runs.jsonl line dropped, its claim released so a start of its phase re-runs it):

    uv run python scripts/relabel_harness_failure.py OUT --window-model ROUTE_MODEL --window-start ISO \\
        --window-end ISO|open --defect "why" --ruling URL

The selection reads each record's route model, ``started_at`` and its turns' times (``tokens.per_turn``'s started_s and
wall_s, on the record's clock), never its outcome, arm or score: start inclusive, end exclusive, both ISO times with their zone (``open``: no end yet). The OUT is read once, so a window of
hundreds of attempts in an OUT of thousands costs one pass. Refused before anything moves: a record of the model the
stored replay check holds (its start time would need the held record opened: set it aside with --diverged first), a
selected attempt of a phase the check halts, and a selected record without exactly one runs.jsonl line. It takes the
OUT's lock like every path here, so it runs while no runner runs on the OUT. Running the same command again finishes
an interrupted batch and sets nothing aside twice (a record already set aside is in harness/, no longer in records/).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infra_fault_monitor import STORE_COPIES_FILE, STORE_PREFIX  # noqa: E402
from infra_fault_monitor import read_log as read_monitor_log  # noqa: E402
from infra_fault_monitor import windows as monitor_windows  # noqa: E402
from run_pilot import (  # noqa: E402
    _RUN_FILE_NAME,
    Refused,
    acquire_lock,
    attempt_key,
    claim_digest,
    claim_path,
    is_grid_out,
    record_arm,
)

from harness.agent.infra_fault import active_in, route_names  # noqa: E402
from harness.agent.record import utc_now  # noqa: E402
from harness.agent.replay_check import (  # noqa: E402
    CHECK_FILE,
    CHECKER_DIVERGENCE,
    PLAIN_PHASE,
    cell_phases,
    check_covers,
    check_groups,
    divergence_sidecar,
    halt_message,
)
from harness.agent.store_fault import attempts_with_query_in_window, store_record_events  # noqa: E402

SIDECAR_SCHEMA = "setaside/1"
LABELS = ("failed:harness", "failed:infrastructure")
SIDECAR_SUFFIX = ".setaside.json"


def _write_atomic(path: Path, text: str) -> None:
    # a sidecar's temporary file keeps the sidecar's name (<claim digest>.<run_id>.setaside.<random>.tmp), so one a
    # crash leaves in harness/ is placed by its name like every run file (confirmation review of 1b7b7ec)
    prefix = f"{path.name.removesuffix('.json')}." if path.name.endswith(".json") else f".{path.name}."
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=prefix, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _attempt(rec: dict, grid: bool = False) -> tuple:
    """The attempt a record is of: (question, model, repetition), in a grid with its arm first."""
    return (record_arm(rec), *_plain(rec)) if grid else _plain(rec)


def _plain(rec: dict) -> tuple:
    return (rec["question_id"], rec["model"], rec["repetition"])


def _sidecar(harness_dir: Path, record_name: str) -> Path:
    """The decision beside the record ``record_name`` (``<name>.json`` -> ``<name>.setaside.json``): named, like the
    record, by the attempt's claim digest when the runner named the record so, by the run id for an older record."""
    return harness_dir / f"{record_name.removesuffix('.json')}{SIDECAR_SUFFIX}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--question", default=None)
    ap.add_argument("--model", default=None, help="the route model id as recorded, e.g. Qwen/Qwen3.8-27B")
    ap.add_argument("--repetition", default=None, type=int)
    ap.add_argument("--arm", default=None, help="a grid run's cell, e.g. beta or fb:curated (a grid OUT only)")
    ap.add_argument("--label", choices=LABELS, default=None,
                    help="the decision's label (default failed:harness; in a grid, failed:infrastructure for a "
                         "record that failed infrastructure on every try)")
    ap.add_argument("--diverged", default=None, metavar="RUN_ID",
                    help="set aside the record of this run, which the stored replay check holds (its online checker "
                         "decision diverges from the offline recompute), without opening it")
    ap.add_argument("--checker-fix", default=None, metavar="PHASE",
                    help="the phase whose checker code is fixed (halted, or with divergences set aside one by one): "
                         "set aside all its divergent records and stamp the fix, from a replay check made under the "
                         "fixed code")
    ap.add_argument("--window-model", default=None, metavar="ROUTE_MODEL",
                    help="a confirmed infrastructure-fault window (change-log entries 129, 148 and 149): set aside "
                         "every attempt of this route model with a model turn in the window, in every arm, as "
                         "failed:infrastructure")
    ap.add_argument("--window-store", default=None, metavar="COPY",
                    help="a confirmed store time-out window of the OUT's store copy (change-log entry 149, item 2): "
                         "set aside every attempt, of any model, with a query on the copy overlapping the window")
    ap.add_argument("--monitor-state", type=Path, default=None, metavar="DIR",
                    help="the fault monitor's directory: a window is set aside only when its log holds that window, "
                         "confirmed (entry 148: a trigger alone authorises nothing)")
    ap.add_argument("--window-start", default=None, metavar="ISO", help="the window's start (inclusive, with zone)")
    ap.add_argument("--window-end", default=None, metavar="ISO|open", help="the window's end (exclusive), or open")
    ap.add_argument("--defect", default=None, help="what in the harness failed")
    ap.add_argument("--ruling", required=True, help="URL of the decision to treat it as a harness failure")
    args = ap.parse_args(argv)
    attempt = (args.question, args.model, args.repetition)
    if args.window_model is not None and args.window_store is not None:
        ap.error("--window-model and --window-store exclude each other: a window is a model's or a store copy's")
    window = (args.window_model or args.window_store, args.window_start, args.window_end)
    modes = [args.diverged is not None, args.checker_fix is not None, any(x is not None for x in attempt),
             any(x is not None for x in window)]
    if sum(modes) != 1:
        ap.error("give one of --diverged RUN_ID, --checker-fix PHASE, --question, --model and --repetition, or "
                 "--window-model, --window-start and --window-end")
    if modes[2] and None in attempt:
        ap.error("--question, --model and --repetition go together")
    if modes[3] and None in window:
        ap.error("--window-model, --window-start and --window-end go together")
    if modes[3] and args.monitor_state is None:
        ap.error("a window needs --monitor-state: the monitor's log must hold it as a confirmed window (entry 148)")
    if (modes[1] or modes[2] or modes[3]) and not args.defect:
        ap.error("--defect is required")
    if (modes[0] or modes[1]) and (args.arm is not None or args.label is not None):
        ap.error("--arm and --label are the attempt path's: a divergence's attempt and label are fixed")
    if modes[3] and (args.arm is not None or args.label is not None):
        ap.error("--arm and --label are the attempt path's: a window takes every arm, labelled failed:infrastructure")

    try:
        lock = acquire_lock(args.out)  # no runner may be writing to this run while it is edited
    except Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    try:
        if modes[0]:
            return _set_aside_diverged(args)
        if modes[1]:
            return _checker_fix(args)
        if modes[3]:
            return _set_aside_window(args)
        return _set_aside(args)
    finally:
        os.close(lock)


# -- a checker divergence (design ruling, design log, 20 Oct 07:26) --------------------------------------------------

DIVERGENCE_DEFECT = "its online checker decision diverges from the offline recompute"


def _read_check(out: Path) -> dict | None:
    """The stored replay check, when it covers the batch as it is now (release review, item 6: a classification
    stands on a count of every record now in the batch), with the halt rule applied again to its per-record inputs
    (``check["divergence"]["groups"]``: never the verdict it stored); else None, said why."""
    path = out / CHECK_FILE
    try:
        check = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"{path}: no readable replay check ({exc}); run scripts/verify_batch.py {out} first", file=sys.stderr)
        return None
    groups = check_groups(check) if "divergence" in check else None
    if groups is None:
        print(f"{path}: made before the halt rule's inputs were stored; run scripts/verify_batch.py {out} again",
              file=sys.stderr)
        return None
    stale = check_covers(out, check)
    if stale:
        print(f"refused: {stale}", file=sys.stderr)
        return None
    return {**check, "divergence": {**check["divergence"], "groups": groups}}


def _group_of(check: dict, phase: str | None) -> dict | None:
    """The halt rule's count of a phase in the check, as ``_read_check`` recomputed it."""
    return next((g for g in check["divergence"].get("groups") or [] if g.get("phase") == phase), None)


def _file_sha256(path: Path) -> str:
    """The sha256 of a file's bytes, read in binary and never decoded."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _set_aside_diverged(args: argparse.Namespace) -> int:
    check = _read_check(args.out)
    if check is None:
        return 2
    entry = next((h for h in check.get("held_records") or [] if h.get("run_id") == args.diverged), None)
    if entry is None:
        print(f"run {args.diverged}: the stored replay check holds no record of it (only a record whose online "
              f"checker decision diverges is set aside this way)", file=sys.stderr)
        return 2
    group = _group_of(check, entry.get("phase"))
    if group is None or group.get("halted"):
        halted = halt_message(args.out, check, entry.get("phase")) or f"phase {entry.get('phase')} is not counted"
        print(f"refused: records are not set aside one by one: {halted}", file=sys.stderr)
        return 2
    return _divergence_set_aside(args.out, check, entry, ruling=args.ruling, defect=args.defect or DIVERGENCE_DEFECT,
                                 fix=None)


def _index_run_id(line: str) -> object:
    """A runs.jsonl line's run id. The line also carries the attempt's outcome label, which a divergence set-aside
    never reads: the parser's hook keeps only run_id of each object as it is built, so no other field is kept."""
    return json.loads(line, object_pairs_hook=lambda pairs: {k: v for k, v in pairs if k == "run_id"}).get("run_id")


def _divergence_set_aside(out: Path, check: dict, entry: dict, *, ruling: str, defect: str, fix: dict | None) -> int:
    """Steps 1-4 of the set-aside for one held record, from its held entry alone (the record is never opened); each
    step idempotent, so an interrupted set-aside finishes when run again."""
    sub, _, name = str(entry.get("file")).partition("/")
    harness_dir = out / "harness"
    source, moved, sidecar = out / str(entry.get("file")), harness_dir / name, _sidecar(harness_dir, name)
    if sidecar.exists():
        print(f"{name}: already set aside", file=sys.stderr)
        return 2
    index = out / "runs.jsonl"
    lines = index.read_text(encoding="utf-8").splitlines(keepends=True) if index.exists() else []
    run_ids = [_index_run_id(ln) for ln in lines]
    found = run_ids.count(entry["run_id"])
    if sub == "records" and source.exists():
        if _file_sha256(source) != entry.get("sha256"):
            print(f"{entry['file']}: its bytes are not the ones the replay check held; verify the batch again",
                  file=sys.stderr)
            return 2
        if found != 1:
            print(f"runs.jsonl: expected one line for run {entry['run_id']}, found {found}", file=sys.stderr)
            return 2
        harness_dir.mkdir(exist_ok=True)
        os.replace(source, moved)  # step 1: the record itself, unopened and unchanged
    elif not (moved.exists() and _file_sha256(moved) == entry.get("sha256")):  # else: an interrupted set-aside
        print(f"{entry['file']}: not found as the replay check held it", file=sys.stderr)
        return 2
    if found > 1:
        print(f"runs.jsonl: more than one line for run {entry['run_id']}; resolve by hand", file=sys.stderr)
        return 2
    if found:  # step 2
        _write_atomic(index, "".join(ln for ln, run in zip(lines, run_ids, strict=True) if run != entry["run_id"]))
    grid = is_grid_out(out)
    if fix is not None:  # step 3, under a fix only: the attempt is re-run after the fix, under it (condition 3)
        _release_claim(out, attempt_key(entry["question_id"], entry["model"], entry["repetition"],
                                        entry["arm"] if grid else None))
    # what it takes from the check and the held record, which check_covers holds it to (release re-review, item 2)
    decision = {
        **divergence_sidecar(check, entry, grid=grid),
        "defect": defect,
        "ruling": ruling,
        "set_aside_at": utc_now(),
        **({"fix": fix} if fix is not None else {}),
    }
    _write_atomic(sidecar, json.dumps(decision, ensure_ascii=False, indent=1, sort_keys=True))  # step 4: last
    rerun = ("a start of its phase re-runs it under the fixed code" if fix is not None else
             "its claim is kept: it is re-run only once a fix is recorded (--checker-fix PHASE), under that code")
    print(f"set aside {name} as failed:harness (checker_divergence; decision in {sidecar.name}); {rerun}")
    return 0


def _release_claim(out: Path, key: tuple) -> None:
    """Delete the attempt's claim, so a start of its phase re-runs it, unless a replacement record of it is in
    records/ (named by the attempt's claim digest): the claim is then the replacement's."""
    digest = claim_digest(key)
    if not any((m := _RUN_FILE_NAME.fullmatch(p.name)) and m["digest"] == digest
               for p in (out / "records").glob("*.json")):
        claim_path(out / "claims", key).unlink(missing_ok=True)


def _checker_fix(args: argparse.Namespace) -> int:
    """Condition (ii) and (iii): a phase's divergent records set aside together once its cause is fixed, with the fix
    recorded on every checker_divergence sidecar of the phase (stamped ones stop counting, and their claims go, so
    a start re-runs their attempts under the fix). A halted phase is released only this way; one whose divergences
    were set aside one by one below the halt needs it too before they are re-run (release review, item 2)."""
    check = _read_check(args.out)  # current, its halt groups recomputed
    if check is None:
        return 2
    code = check.get("code") or {}
    if not code.get("git_commit") or code.get("git_dirty") is not False:
        print(f"{args.out}: the replay check was not made under committed, clean code; commit the fix and verify "
              f"again", file=sys.stderr)
        return 2
    phase = args.checker_fix
    group = _group_of(check, phase)
    if group is None or not group.get("diverged"):
        print(f"phase {phase}: the stored replay check counts no divergence of it (held, or set aside with no fix), so "
              f"there is nothing to fix", file=sys.stderr)
        return 2
    if code["git_commit"] in (group.get("codes") or []):
        print(f"refused: the replay check was made under code {code['git_commit'][:12]}, which made divergent "
              f"decisions of phase {phase}; fix the cause, commit it and verify the batch under the fixed code",
              file=sys.stderr)
        return 2
    # another commit id is no fix when its code is the same (an empty commit): the code digest must differ from the
    # digest of the code of every divergent decision (release review, item 3)
    trees = group.get("trees")
    if not code.get("tree_sha256") or trees is None or None in trees or code["tree_sha256"] in trees:
        known = "unknown" if trees is None or None in trees else ", ".join(t[:12] for t in trees)
        print(f"refused: the replay check was made under code tree {str(code.get('tree_sha256'))[:12]}; the divergent "
              f"decisions of phase {phase} were made under code tree(s) {known}: a fix must be other code (its tree "
              f"digest differs from theirs), not another commit of the same code", file=sys.stderr)
        return 2
    fix = {"code": code["git_commit"], "tree_sha256": code.get("tree_sha256"), "defect": args.defect,
           "ruling": args.ruling, "fixed_at": utc_now(),
           "halted": {k: group.get(k) for k in ("phase", "records", "diverged", "models", "codes", "reasons",
                                                 "halted")}}
    held = [h for h in check.get("held_records") or [] if h.get("phase") == phase]
    for entry in held:
        if _divergence_set_aside(args.out, check, entry, ruling=args.ruling, defect=args.defect, fix=fix) != 0:
            return 2
    stamped = 0
    grid = is_grid_out(args.out)
    for path in sorted((args.out / "harness").glob(f"*{SIDECAR_SUFFIX}")):
        decision = json.loads(path.read_text(encoding="utf-8"))
        if decision.get("classification") == CHECKER_DIVERGENCE and not decision.get("fix") and \
                decision.get("phase") == phase:
            _write_atomic(path, json.dumps({**decision, "fix": fix}, ensure_ascii=False, indent=1, sort_keys=True))
            # its claim, kept while no fix was recorded, goes now: a start re-runs the attempt under the fix
            _release_claim(args.out, attempt_key(decision["question_id"], decision["model"], decision["repetition"],
                                                 decision.get("arm") if grid else None))
            stamped += 1
    print(f"phase {args.checker_fix}: {len(held)} divergent records set aside and {stamped} earlier set-asides stamped "
          f"with the fix at code {code['git_commit'][:12]}; verify the batch again, then start the phase under that "
          f"code to re-run them")
    return 0


# -- an attempt named by its key --------------------------------------------------------------------------------


def _guard_held_and_halted(args: argparse.Namespace, want: tuple) -> set[str] | None:
    """The held/halted guard of the attempt path, before anything is opened or finished (release review, item 4):
    refused (None, said why) while the stored replay check halts the attempt's phase, or holds a record of the
    attempt (in records/, or in harness/ where an interrupted --diverged left it): such a record is set aside with
    --diverged, which the halt rule governs and which never opens it. Otherwise the names of the files the check
    holds, which the attempt path then never opens (no check: none)."""
    path = args.out / CHECK_FILE
    if not path.is_file():
        return set()
    try:
        check = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"{path}: not readable ({exc}), so whether the attempt is held or its phase halted cannot be told; run "
              f"scripts/verify_batch.py {args.out} again", file=sys.stderr)
        return None
    phase = PLAIN_PHASE if not args.grid else (cell_phases(args.out) or {}).get(args.arm, f"unpinned:{args.arm}")
    groups = check_groups(check)  # the rule applied again to the check's inputs, where it stored them
    if groups is not None:
        check = {**check, "divergence": {**(check.get("divergence") or {}), "groups": groups}}
    halted = halt_message(args.out, check, phase)
    if halted:
        print(f"refused: {halted}", file=sys.stderr)
        return None
    held = check.get("held_records") or []
    keys = ("arm", "question_id", "model", "repetition") if args.grid else ("question_id", "model", "repetition")
    if any(k not in h for h in held for k in keys):
        print(f"{path}: made before held records named their attempts; run scripts/verify_batch.py {args.out} again",
              file=sys.stderr)
        return None
    mine = [h for h in held if tuple(h[k] for k in keys) == want]
    if mine:  # it would bypass the halt rule, and reading it would read its score
        print(f"{mine[0].get('file')}: the replay check holds it (its online checker decision diverges); set it aside "
              f"with --diverged {mine[0].get('run_id')}, which the halt rule governs", file=sys.stderr)
        return None
    return {str(h.get("file")).partition("/")[2] for h in held}


def _set_aside(args: argparse.Namespace) -> int:
    # A grid OUT holds every cell of a graph: (question, model, repetition) names one attempt per arm, and its
    # claims are keyed with the arm (run_pilot.attempt_key).
    args.grid = is_grid_out(args.out)
    if args.grid and args.arm is None:
        print(f"{args.out} is a grid run, whose attempts are keyed by arm: name the cell with --arm", file=sys.stderr)
        return 2
    if not args.grid and args.arm is not None:
        print(f"--arm: {args.out} is not a grid run; its attempts are (question, model, repetition)", file=sys.stderr)
        return 2
    want = (args.arm, args.question, args.model, args.repetition) if args.grid else (
        args.question, args.model, args.repetition)
    held = _guard_held_and_halted(args, want)  # first: before any finishing, and before any record is opened
    if held is None:
        return 2
    harness_dir = args.out / "harness"
    # A run of this script that stopped part way left the record in harness/ without its sidecar.
    pending = []
    for path in sorted(harness_dir.glob("*.json")):
        if path.name.endswith(SIDECAR_SUFFIX) or path.name in held:  # a held one is never opened
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        if (_attempt(rec, args.grid) == want and "harness_failure" not in rec
                and not _sidecar(harness_dir, path.name).exists()):
            pending.append((path, rec))
    if len(pending) > 1:
        print(f"{len(pending)} unfinished set-asides of that attempt in harness/; resolve by hand", file=sys.stderr)
        return 2
    if pending:
        path, rec = pending[0]
        print(f"finishing the interrupted set-aside of {path.name}")
        return _finish(args, rec, path.name, resuming=True)

    matches = []
    for path in sorted((args.out / "records").glob("*.json")):
        if path.name in held:
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        if _attempt(rec, args.grid) == want:
            matches.append((path, rec))
    if len(matches) != 1:
        print(f"expected exactly one record for that attempt, found {len(matches)}", file=sys.stderr)
        return 2
    path, rec = matches[0]
    if _sidecar(harness_dir, path.name).exists() or (harness_dir / path.name).exists():
        print("already set aside", file=sys.stderr)
        return 2
    # Check the index before changing anything, so a refusal leaves the run as it was.
    index = args.out / "runs.jsonl"
    lines = index.read_text(encoding="utf-8").splitlines(keepends=True) if index.exists() else []
    found = sum(json.loads(ln).get("run_id") == rec["run_id"] for ln in lines)
    if found != 1:
        print(f"runs.jsonl: expected one line for run {rec['run_id']}, found {found}", file=sys.stderr)
        return 2
    harness_dir.mkdir(exist_ok=True)
    os.replace(path, harness_dir / path.name)  # step 1: the record itself, unchanged
    return _finish(args, rec, path.name, resuming=False)


# -- an infrastructure-fault window (change-log entry 129, item 2) --------------------------------------------------

WINDOW_LABEL = "failed:infrastructure"
WINDOW_SELECTION = "any model turn in the window (entry 149): route model, started_at and turn times only"
STORE_WINDOW_SELECTION = ("any query on the store copy overlapping the window (entry 149, item 2): query times "
                          "only, every model and arm")


def _window_time(text: str) -> datetime:
    t = datetime.fromisoformat(text)
    if t.tzinfo is None:
        raise ValueError(f"{text!r} has no time zone")
    return t.astimezone(UTC)


def _stamp(t: datetime | None) -> str | None:
    return None if t is None else t.isoformat(timespec="milliseconds")


def _set_aside_window(args: argparse.Namespace) -> int:
    """Every attempt of --window-model with a model turn in [--window-start, --window-end), in every arm and whatever
    its outcome, set aside as failed:infrastructure (module docstring). Everything that can refuse is checked before
    anything moves; the selection reads a record's route model and its times, nothing else."""
    args.grid = is_grid_out(args.out)
    try:
        start = _window_time(args.window_start)
        end = None if args.window_end == "open" else _window_time(args.window_end)
    except ValueError as exc:
        print(f"refused: the window's times: {exc}", file=sys.stderr)
        return 2
    if end is not None and end <= start:
        print(f"refused: the window's end {_stamp(end)} is not after its start {_stamp(start)}", file=sys.stderr)
        return 2
    model, copy = args.window_model, getattr(args, "window_store", None)
    subject = f"{STORE_PREFIX}{copy}" if copy is not None else route_names(args.out).get(model)
    confirmed = [w for w in monitor_windows(read_monitor_log(args.monitor_state), subject or "")
                 if _window_time(w["start"]) == start and (None if w["end"] is None else _window_time(w["end"])) == end]
    if copy is not None:  # the OUT must be on that copy, as the monitor watched it
        try:
            watched = json.loads((args.monitor_state / STORE_COPIES_FILE).read_text(encoding="utf-8"))["outs"]
        except (OSError, ValueError, KeyError) as exc:
            print(f"refused: the monitor's store-copy mapping cannot be read ({exc})", file=sys.stderr)
            return 2
        if watched.get(str(Path(args.out).resolve())) != copy:
            print(f"refused: the monitor watched {args.out} on copy {watched.get(str(Path(args.out).resolve()))!r}, "
                  f"not {copy}", file=sys.stderr)
            return 2
    if not confirmed:
        print(f"refused: {args.monitor_state} holds no confirmed window {_stamp(start)} .. {_stamp(end) or 'open'} of "
              f"{subject or model} (a hold that no probe, fingerprint change or provider note confirmed relabels "
              f"nothing; entry 148)", file=sys.stderr)
        return 2
    if copy is not None:  # a store copy's window: every model's attempts with a query on it in the window
        names = route_names(args.out)
        window = {"store_copy": copy, "start": _stamp(start), "end": _stamp(end), "selection": STORE_WINDOW_SELECTION}

        def in_window(rec: dict) -> bool:  # the selection: the queries' times on the OUT's copy, nothing else
            ev = store_record_events(rec, names, copy)
            return not isinstance(ev, str) and bool(attempts_with_query_in_window([ev], copy, start, end))
    else:
        window = {"model": model, "start": _stamp(start), "end": _stamp(end), "selection": WINDOW_SELECTION}

        def in_window(rec: dict) -> bool:  # the selection: the route model and the attempt's turn times, nothing else
            return rec.get("model") == model and active_in(rec, start, end)

    check, held_files = None, set()
    path = args.out / CHECK_FILE
    if path.is_file():
        try:
            check = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"{path}: not readable ({exc}), so whether a record of the window is held or its phase halted "
                  f"cannot be told; run scripts/verify_batch.py {args.out} again", file=sys.stderr)
            return 2
        groups = check_groups(check)
        if groups is not None:
            check = {**check, "divergence": {**(check.get("divergence") or {}), "groups": groups}}
        held = check.get("held_records") or []
        mine = [h for h in held if copy is not None or h.get("model") == model]
        if mine:  # its start would need the held record opened, which would read its score
            print(f"refused: the replay check holds {len(mine)} record(s) of {model} (their online checker decision "
                  f"diverges), e.g. {mine[0].get('file')}; set them aside with --diverged first, which the halt rule "
                  f"governs, then run the window", file=sys.stderr)
            return 2
        held_files = {str(h.get("file")).partition("/")[2] for h in held}

    harness_dir = args.out / "harness"
    pending = []  # an earlier run that stopped part way: in harness/ without its sidecar
    for p in sorted(harness_dir.glob("*.json")) if harness_dir.is_dir() else []:
        if p.name.endswith(SIDECAR_SUFFIX) or p.name in held_files or _sidecar(harness_dir, p.name).exists():
            continue
        rec = json.loads(p.read_text(encoding="utf-8"))
        if "harness_failure" not in rec and in_window(rec):
            pending.append((p, rec))
    selected, present = [], Counter()
    for p in sorted((args.out / "records").glob("*.json")):
        if p.name in held_files:  # another model's held record (the model's own refused above): never opened
            continue
        rec = json.loads(p.read_text(encoding="utf-8"))
        present[_attempt(rec, args.grid)] += 1
        if in_window(rec):
            selected.append((p, rec))
    problems = []
    if check is not None:
        phases = cell_phases(args.out) or {}
        for _, rec in pending + selected:
            phase = PLAIN_PHASE if not args.grid else phases.get(record_arm(rec), f"unpinned:{record_arm(rec)}")
            halted = halt_message(args.out, check, phase)
            if halted:
                problems.append(halted)
                break
    index = args.out / "runs.jsonl"
    lines = index.read_text(encoding="utf-8").splitlines(keepends=True) if index.exists() else []
    in_index = Counter(_index_run_id(ln) for ln in lines if ln.strip())
    for p, rec in selected:
        if in_index[rec["run_id"]] != 1:
            problems.append(f"runs.jsonl: expected one line for run {rec['run_id']}, found {in_index[rec['run_id']]}")
        if (harness_dir / p.name).exists() or _sidecar(harness_dir, p.name).exists():
            problems.append(f"{p.name} is in records/ and already in harness/; resolve by hand")
    for _, rec in pending:
        if in_index[rec["run_id"]] > 1:
            problems.append(f"runs.jsonl: more than one line for run {rec['run_id']}; resolve by hand")
    if problems:
        print("refused, nothing set aside:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 2

    # The attempt path's four steps, each over the whole batch and each idempotent, the sidecars last: a run that
    # stops part way leaves records in harness/ without a sidecar, which the same command finishes (pending above).
    args.label = WINDOW_LABEL
    batch = list(pending)
    if selected:
        harness_dir.mkdir(exist_ok=True)
    for p, rec in selected:  # step 1: each record itself, unchanged
        os.replace(p, harness_dir / p.name)
        present[_attempt(rec, args.grid)] -= 1
        batch.append((harness_dir / p.name, rec))
    ids = {rec["run_id"] for _, rec in batch}
    kept = [ln for ln in lines if not ln.strip() or _index_run_id(ln) not in ids]
    if len(kept) != len(lines):
        _write_atomic(index, "".join(kept))  # step 2: their runs.jsonl lines, in one rewrite
    for _, rec in batch:  # step 3: each claim, unless a replacement record of the attempt is in records/ and holds it
        if present[_attempt(rec, args.grid)] <= 0:
            claim_path(args.out / "claims", attempt_key(*_plain(rec), record_arm(rec) if args.grid else None)).unlink(
                missing_ok=True)
    for path, rec in batch:  # step 4, last: each decision
        decision = _decision(args, rec, path.name, extra={"window": window})
        _write_atomic(_sidecar(harness_dir, path.name), json.dumps(decision, ensure_ascii=False, indent=1,
                                                                   sort_keys=True))
    arms = Counter(record_arm(rec) for _, rec in batch)
    print(f"window {window['start']} .. {window['end'] or 'open'} of {model or 'store copy ' + copy}: "
          f"{len(selected)} attempt(s) set aside as "
          f"{WINDOW_LABEL}, {len(pending)} interrupted set-aside(s) finished, by arm {dict(sorted(arms.items()))} "
          f"(decisions in harness/*{SIDECAR_SUFFIX}); "
          f"{'a start of each phase re-runs them' if args.grid else 'a resume re-runs them'}")
    return 0


def _decision(args: argparse.Namespace, rec: dict, name: str, extra: dict | None = None) -> dict:
    """The sidecar of a set-aside: the attempt, the label (``args.label``, else in a grid failed:infrastructure for
    an outage that outlasted every re-run, else failed:harness), the original outcome, the defect and the ruling."""
    outcome = rec["outcome"]
    # in a grid, an outage that outlasted every re-run is no harness defect unless the ruling says so (--label)
    outage = args.grid and outcome.get("failure_type") == "infrastructure"
    label = args.label or ("failed:infrastructure" if outage else "failed:harness")
    return {
        "schema": SIDECAR_SCHEMA,
        "run_id": rec["run_id"],
        "record": name,
        "question_id": rec["question_id"],
        "model": rec["model"],
        "repetition": rec["repetition"],
        **({"arm": record_arm(rec)} if args.grid else {}),
        "label": label,
        "original_label": outcome.get("label"),
        "original_failure_type": outcome.get("failure_type"),
        "defect": args.defect,
        "ruling": args.ruling,
        "set_aside_at": utc_now(),
        **(extra or {}),
    }


def _finish(args: argparse.Namespace, rec: dict, name: str, *, resuming: bool) -> int:
    """Steps 2-4, each idempotent, so an interrupted set-aside can be finished by running again."""
    index = args.out / "runs.jsonl"
    lines = index.read_text(encoding="utf-8").splitlines(keepends=True) if index.exists() else []
    kept = [ln for ln in lines if json.loads(ln).get("run_id") != rec["run_id"]]
    if len(lines) - len(kept) > 1:
        print(f"runs.jsonl: more than one line for run {rec['run_id']}; resolve by hand", file=sys.stderr)
        return 2
    if len(kept) != len(lines):
        _write_atomic(index, "".join(kept))  # step 2
    # step 3: release the claim so a resume re-runs the attempt, unless a replacement already
    # ran (an interrupted set-aside can be finished after a runner re-ran the attempt): its
    # record is in records/ and the claim is now that record's.
    replaced = any(_attempt(json.loads(p.read_text(encoding="utf-8")), args.grid) == _attempt(rec, args.grid)
                   for p in (args.out / "records").glob("*.json"))
    if not replaced:
        key = attempt_key(*_plain(rec), record_arm(rec) if args.grid else None)
        claim_path(args.out / "claims", key).unlink(missing_ok=True)
    decision = _decision(args, rec, name)
    sidecar = _sidecar(args.out / "harness", name)
    _write_atomic(sidecar, json.dumps(decision, ensure_ascii=False, indent=1, sort_keys=True))  # step 4: last
    rerun = ("a start of its grid phase re-runs it" if args.grid else "a resume with the same arguments re-runs it")
    again = " after resuming" if resuming else ""
    print(f"set aside {name} as {decision['label']} (decision in {sidecar.name}){again}; {rerun}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
