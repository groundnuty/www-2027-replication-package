#!/usr/bin/env python3
"""Locate and check the audit-replay matrix of one plain-run OUT-phase that an e189 look hands the frozen exporter (science
ruling (B), 26 Oct 2026): the counterpart of driver/replay_matrices.py for the private audit replays.

driver/replay_matrices.py (reviewed, unchanged) locates the Tier 2 matrices through the Tier 2 tree layout and refuses an OUT
name outside its own pattern. This tool takes its matrix from one results folder per OUT and reuses the reviewed module for
everything that is not that location: the matrix reader, the staging copy, the run-id and claim-file patterns, the ledger
check. It loads that module BY PATH (``python3 -I`` leaves the script's directory off sys.path); nothing of it is copied.

Two subcommands, stdlib only, printing NAMES, COUNTS and sha256s only (never a run id, a row or a value), exit 0 (OK) or 2
(REFUSED, one line ``REFUSED <code> k=v ...`` on stderr):

  check-a15         marker, folder and matrix are bound by one sha256, the matrix is copied into a private stage, and every
                    record of the OUT has exactly its rows
  ledger-check-a15  after the export: the ledger's replay rows are exactly the matrix's runs that have an output, once per arm
                    (--stage repeated, one per partition of a union: across all the stages, a run id in two refuses)

    python3 -I a15_replay_matrices.py check-a15 --audit-root R --done-dir D --out-dir OUTDIR --out-name OUT --phase natural \\
        --stage STAGE
    python3 -I a15_replay_matrices.py ledger-check-a15 --stage STAGE --ledger LEDGER

THE INPUTS. ``--out-name`` and ``--phase`` must be one of the OUT_PHASES pairs below (an exact list, no pattern). The done
marker ``D/<OUT>-<phase>`` must be ONE newline-ended line ``<64 hex> ok <UTC>``, optionally followed by a disclosure after a
space; its first token is the matrix's sha256 (a disclosure is printed as ``disclosure=1`` and never interpreted). The folder
``R/audit-<OUT>`` must be a real directory (not a link) holding ``matrix.jsonl.gz``, a regular file whose sha256 is that first
token; ``run.json`` (schema ``a15-gated-replay/1``, ``final`` true, ``build.sha256`` naming the same matrix sha256,
``pins.sources`` naming exactly the SOURCES below), of which only those fields are read and no value is printed; and ``rc``
saying ``0 <token>``. ``--out-dir`` is the directory the exporter gets with ``--records`` (the OUT or its analysis view);
``--out-name`` is given, never taken from its basename.

THE COVERAGE CHECK, on the staged COPY. The OUT must be a plain run: ``order.json`` holds its order (a directory with any
``order-*.json`` is a grid and is refused), each ``records/<40 hex>.<run id>.json`` is named by the claim digest of its
attempt, the first 40 hex of the sha256 of the JSON of [question, route model, repetition] (route models from
served-models.json; a runner name with no route of its own, or two keys with one digest, is refused). Every record must be
named by the order (one that is not is refused). Then the matrix must hold EXACTLY the OUT's records: a record without rows
or a run id of no record refuses, naming the counts (``missing``, ``extra``), never a name. Each run id has exactly one row
per arm and variant (ARMS x VARIANTS below: a repeated row, another arm or a missing arm refuses), every row says whether the
run has an output (a bool, the same on all of the run's rows) and, with an output, its scorer label (``read_matrix`` of the
reviewed module). Whether a matrix may hold fewer records than the OUT is not decided here: any difference refuses.

A SELECTION OF THE OUT'S RECORDS (two matrix sources that together hold each record once). ``--exclude-names FILE:SHA256`` and
``--only-names FILE:SHA256`` (mutually exclusive) take a pinned file of record file names (FILE absolute, its sha256 checked BEFORE it
is parsed; one name a line, each a record file name, sorted, none twice, at least one; every name must be a record of the OUT). The
matrix must then hold EXACTLY the OUT's records without the listed ones (``--exclude-names``) or exactly the listed ones
(``--only-names``); everything else is as without them, and the COVERAGE line adds ``selection=`` and ``names_listed=`` (a count).
For one names file the two selections are disjoint and together are the OUT's records, so two checks of the same OUT, one with each
option and each into its own stage, cover every record exactly once; ``ledger-check-a15 --stage A --stage B`` then refuses a run id
in both. ``--audit-name NAME`` (the folder is R/audit-NAME) and ``--marker-name NAME`` (the marker is the file D/NAME) point the
check at a results folder and marker that are not named by the OUT; the folder must have the same structure and the checks on it
are the same. A general ``--select`` over several named sources is not built.

A SELECT, SEVERAL OUTS AND THE JOIN. ``--select-names FILE:SHA256`` (exclusive with the two above) keeps the whole-OUT coverage check on the
original matrix, whose sha256 is bound to the marker as always; the STAGED matrix is then a copy holding only the rows of the listed records
(each selected row's bytes as in the original, gzip with no time, named audit.selected.matrix.jsonl.gz, listed in matrices.list), and its run
ids must be exactly the listed names' (asserted by the same exact-match check). The manifest then has a seventh column, original_sha256, and
the first output line a selected_sha256. A matrix over SEVERAL OUTs is served in one form only: ``--out-dir`` repeated (at least two),
``--only-names`` repeated (the wanted run ids are the union of the lists; no name twice; every name a record of one OUT; no record file
name or run id in two OUTs), ``--out-name`` one of COMBINED_NAMES and ``--audit-name`` and ``--marker-name`` equal to it; its pair is no
entry of the pinned list. ``join-check-a15 --out-dir D... --stage S...`` reads the run ids from the staged matrices themselves and requires
them pairwise disjoint and together exactly the records of all the OUTs (``join-differs missing= extra= twice=``).

THE OUTPUT in STAGE (mode 700, must not exist; files mode 600): the copy of the matrix, ``matrices.list`` (its absolute path:
the exporter's ``--matrix``), ``replay-matrices-manifest.tsv`` (the reviewed check's columns) and ``replay-outputs.list`` (the
run ids with an output, for ledger-check-a15). NO replay-routing file is written: this tool reads no launch record, so the line
``ROUTING ... kind=a15 unrouted=1`` says so and the analysis counts these replay rows ``not_routed``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import zlib
from collections import Counter
from pathlib import Path


def _load_reviewed_module():
    """The reviewed replay_matrices.py beside this file, loaded by path (``-I`` puts no directory on sys.path). A module of that very
    file that is already loaded (a test's, or another loader's) is reused, so ``Refused`` is one class."""
    path = Path(__file__).resolve().parent / "replay_matrices.py"
    have = sys.modules.get("replay_matrices")
    if have is not None and getattr(have, "__file__", None) and Path(have.__file__).resolve() == path:
        return have
    name = "replay_matrices" if have is None else "_audit_replay_matrices_reviewed"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # the module's dataclasses look themselves up here
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


rm = _load_reviewed_module()
Refused = rm.Refused

# The arm set of the audit replay: the sources and unions its launcher gives the wrapper (the --source and --union arguments of
# a15_audit_run.sh and a15h_audit.sbatch; the wrapper's Arms are the sources then the unions, harness/checkloop/matrix.py
# Arms.names). The wrapper has no OBQC arms here. A test pins both tuples; any other arm in a matrix is refused.
SOURCES = ("curated", "aligned-curated", "census-t5", "generated-1", "generated-2", "generated-3")
UNIONS = ("union", "union-aligned")
ARMS = SOURCES + UNIONS
VARIANTS = tuple(rm.VARIANTS)  # a test pins it against the reviewed module's

# The OUT-phase pairs this tool serves (exact names; the marker is D/<OUT>-<phase>). Tests monkeypatch these two lines.
NATURAL_OUTS = ("a15-sonnet", "a15-glm", "a15-qwen38-e180", "a15-deepseek", "a15-coder", "a15-gemma-p0of3", "a15-gemma-p1of3", "a15-gemma-p2of3")
PROMPTED_OUTS = ("a15-sonnet-census", "a15-sonnet-curated", "a15-sonnet-generated", "a15-qwen38-e180-census", "a15-qwen38-e180-curated", "a15-qwen38-e180-generated-p0of2", "a15-qwen38-e180-generated-p1of2")
PHASES = ("natural", "prompted")
# The name of a matrix that covers the records of SEVERAL OUTs (a results folder audit-<name> and the marker <name>-<phase> named by it, no
# <OUT>-<phase> pair among the OUTs). It is served only by a check with several --out-dir, only --only-names, --audit-name equal to it and
# --marker-name equal to <name>-<phase> (the marker's file name, as for every other replay). A test patches this line.
COMBINED_NAMES = ("a15-gemma-copied",)

RUN_SCHEMA = "a15-gated-replay/1"
MATRIX_NAME = "matrix.jsonl.gz"
MARKER_LINE = re.compile(rf"([0-9a-f]{{64}}) ok ({rm.UTC})( [^\n]+)?")
RC_LINE = re.compile(r"0 [0-9a-f]{1,64}\n?")
AUDIT_NAME = re.compile(r"[a-z0-9][a-z0-9.-]*")  # the part of the folder name after "audit-"
MARKER_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
NAMES_PIN = re.compile(r"(/[^\s:]*):([0-9a-f]{64})")


def out_phases() -> frozenset:
    return frozenset([(o, "natural") for o in NATURAL_OUTS] + [(o, "prompted") for o in PROMPTED_OUTS])


# -- the marker, the folder and the matrix ------------------------------------------------------------------------------------


def read_audit_marker(done_dir: Path, out: str, phase: str, marker_name: str | None = None) -> tuple[str, str, bool]:
    """(matrix sha256 = the marker's first token, sha256 of the marker file, whether a disclosure follows the timestamp). The marker
    is D/<OUT>-<phase> unless ``marker_name`` names another file of D."""
    path = done_dir / (marker_name or f"{out}-{phase}")
    if not os.path.lexists(path):
        raise Refused("no-done-marker", out=out, phase=phase)
    data = rm.read_small(path, 4096)  # refuses a link and a non-regular file
    text = data.decode("utf-8", "replace")
    if not text.endswith("\n") or text.count("\n") != 1:
        raise Refused("marker-format", out=out, phase=phase)
    m = MARKER_LINE.fullmatch(text[:-1])
    if not m:
        raise Refused("marker-format", out=out, phase=phase)
    return m[1], hashlib.sha256(data).hexdigest(), m[3] is not None


def read_audit_folder(audit_root: Path, out: str, marker_sha: str, audit_name: str | None = None) -> Path:
    """The matrix path of the folder ``audit-<OUT>`` (``audit-<audit_name>`` when that is given), after the folder, the matrix, the run
    description and the rc are checked against the marker's sha256. Refuses a link anywhere on the way."""
    folder = audit_root / f"audit-{audit_name or out}"
    try:
        st = os.lstat(folder)
    except OSError:
        raise Refused("audit-folder-missing", out=out) from None
    if not stat.S_ISDIR(st.st_mode):  # lstat: a link to a directory is not one
        raise Refused("audit-folder-not-a-directory", out=out)
    matrix = folder / MATRIX_NAME
    try:
        if not stat.S_ISREG(os.lstat(matrix).st_mode):
            raise Refused("matrix-missing-or-linked", out=out)
    except OSError:
        raise Refused("matrix-missing-or-linked", out=out) from None
    run = rm.read_json_file(folder / "run.json", "run.json")
    build = run.get("build") if isinstance(run, dict) else None
    shas = build.get("sha256") if isinstance(build, dict) else None
    pins = run.get("pins") if isinstance(run, dict) else None
    sources = pins.get("sources") if isinstance(pins, dict) else None
    if not isinstance(run, dict) or run.get("schema") != RUN_SCHEMA:
        raise Refused("run-json-differs", out=out, what="schema")
    if run.get("final") is not True:
        raise Refused("run-json-differs", out=out, what="not-final")
    if not isinstance(shas, dict) or shas.get(MATRIX_NAME) != marker_sha:
        raise Refused("run-json-differs", out=out, what="matrix-sha256")
    if not isinstance(sources, dict) or set(sources) != set(SOURCES):
        raise Refused("run-json-differs", out=out, what="sources")
    rc = rm.read_small(folder / "rc", 4096).decode("ascii", "replace")
    if not RC_LINE.fullmatch(rc):
        raise Refused("rc-not-zero", out=out)
    if rm.sha256_file(matrix) != marker_sha:
        raise Refused("matrix-sha-differs", out=out)
    return matrix


# -- the OUT's records ------------------------------------------------------------------------------------------------------------


def plain_digest(question, route: str, repetition) -> str:
    """The claim digest of a plain run's attempt (scripts/run_pilot.py claim_digest of attempt_key(question, route, repetition)
    with no arm): the first 40 hex of the sha256 of the JSON of [question, route model, repetition]."""
    return hashlib.sha256(json.dumps([question, route, repetition], ensure_ascii=False).encode("utf-8")).hexdigest()[:40]


def classify_plain(out_dir: Path) -> tuple[set[str], Counter]:
    """({the OUT's record file names}, counts): every ``records/`` name is named by the plain order, or the OUT is refused."""
    if any(n.startswith("order-") and n.endswith(".json") for n in os.listdir(out_dir)):
        raise Refused("order-not-plain")  # a grid OUT keeps one order file per phase
    path = out_dir / "order.json"
    if not os.path.lexists(path):
        raise Refused("no-order-file")
    served = rm.read_json_file(out_dir / "served-models.json", "served-models").get("models") or {}
    routes = {name: str(v["route_model"]) for name, v in served.items() if isinstance(v, dict) and v.get("route_model")}
    route_of = {name: r for name, r in routes.items() if list(routes.values()).count(r) == 1}  # one name per route only
    order = rm.read_json_file(path, "order").get("order")
    if not isinstance(order, list) or any(not (isinstance(k, list) and len(k) == 4) for k in order):
        raise Refused("order-malformed")
    digests: list[str | None] = []
    for name, _arm, question, repetition in order:  # the arm is the OUT's one condition: not part of a plain claim key
        digests.append(plain_digest(question, route_of[name], repetition) if name in route_of else None)
    unrouted = sum(d is None for d in digests)
    shared = sum(n for n in Counter(d for d in digests if d is not None).values() if n > 1)
    if unrouted or shared:
        raise Refused("order-keys-ambiguous", unrouted=unrouted, shared=shared)
    known = set(digests)
    names: set[str] = set()
    seen_digest: Counter = Counter()
    seen_run: Counter = Counter()
    unclassified = other = 0
    for n in os.listdir(out_dir / "records"):
        if n.endswith(".setaside.json"):
            continue
        m = rm.RUN_FILE.fullmatch(n)
        if not m:
            if n.endswith(".json"):
                other += 1
            continue
        seen_digest[m[1]] += 1
        seen_run[m[2]] += 1
        if m[1] in known:
            names.add(n)
        else:
            unclassified += 1
    twice = sum(1 for n in seen_digest.values() if n > 1)
    run_twice = sum(1 for n in seen_run.values() if n > 1)
    if unclassified or other or twice or run_twice:  # flags, not counts, as in the reviewed check
        raise Refused("records-unclassified", unclassified=int(bool(unclassified)), odd_names=int(bool(other)),
                      digest_twice=int(bool(twice)), run_id_twice=int(bool(run_twice)))
    return names, Counter(order_keys=len(order))


def coverage_a15(want: set[str], matrix: Path) -> tuple[int, int, set[str]]:
    """(rows, run ids in the matrix, run ids with an output) of the staged matrix once it holds exactly the run ids ``want`` with a
    row per arm and variant each, or Refused."""
    rows, problems, ids, with_output = rm.read_matrix(matrix, ARMS, VARIANTS)
    missing, extra = len(want - ids), len(ids - want)
    expected = len(ARMS) * len(VARIANTS) * len(want)
    if missing or extra or any(problems.values()) or rows != expected:
        # an output-flag problem is printed as a flag, not a count (a number about which runs have an output)
        shown = {k: (int(bool(v)) if k in rm.OUTPUT_PROBLEMS else v) for k, v in sorted(problems.items()) if v}
        raise Refused("matrix-differs-from-records", out_records=len(want), matrix_records=len(ids), missing=missing,
                      extra=extra, rows=rows, rows_expected=expected, **shown)
    return rows, len(ids), with_output


# -- a selection of the OUT's records by a pinned file of record names ---------------------------------------------------------------


def read_names(spec: str) -> list[str]:
    """The record file names of a ``FILE:SHA256`` pin: FILE absolute, its sha256 checked BEFORE it is parsed, then one name a line,
    each a record file name, sorted, none twice, at least one."""
    m = NAMES_PIN.fullmatch(spec)
    if not m or ".." in m[1].split("/"):
        raise Refused("names-pin-malformed")
    data = rm.read_small(Path(m[1]), 1 << 24)  # refuses a link and a non-regular file
    if hashlib.sha256(data).hexdigest() != m[2]:
        raise Refused("names-sha-differs")
    text = data.decode("utf-8", "replace")
    names = text.split("\n")
    if names[-1] == "":
        names.pop()  # the last line's newline
    if not names or any(not rm.RUN_FILE.fullmatch(n) for n in names) or names != sorted(names) or len(set(names)) != len(names):
        raise Refused("names-malformed")
    return names


def listed_names(records: set[str], specs: list[str]) -> list[str]:
    """The record names of one or several pinned files, in order: every name must be a record of the OUT(s), and no name may be listed
    twice (in one file or in two)."""
    listed: list[str] = []
    for spec in specs:
        listed += read_names(spec)
    if len(set(listed)) != len(listed):
        raise Refused("names-twice", twice=len(listed) - len(set(listed)))
    unknown = len(set(listed) - records)
    if unknown:
        raise Refused("names-not-records", unknown=unknown)
    return listed


def classify_outs(out_dirs: list[Path]) -> tuple[set[str], Counter]:
    """classify_plain of one OUT, or of several whose records are disjoint (a record file name or run id in two OUTs refuses)."""
    if len(out_dirs) == 1:
        return classify_plain(out_dirs[0])
    names: set[str] = set()
    counts: Counter = Counter()
    for d in out_dirs:
        got, c = classify_plain(d)
        if names & got or rm.run_ids_of(names) & rm.run_ids_of(got):
            raise Refused("name-in-two-outs")
        names |= got
        counts += c
    return names, counts


def filter_matrix(src: Path, dest: Path, keep: set[str]) -> str:
    """Write the rows of ``src`` whose run id is in ``keep`` to ``dest`` (mode 600, gzip with no time), each row's bytes as they are in
    the original; the sha256 of ``dest``."""
    fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as out, gzip.open(src, "rb") as rows:
        for line in rows:
            if line.strip() and json.loads(line)["run_id"] in keep:
                out.write(line if line.endswith(b"\n") else line + b"\n")
    return rm.sha256_file(dest)


def stage_run_ids(stage: Path) -> set[str]:
    """The run ids of the one matrix a stage lists in matrices.list (a file inside the stage), read from the matrix itself."""
    lines = rm.read_small(stage / "matrices.list", 1 << 20).decode("utf-8").splitlines()
    if len(lines) != 1:
        raise Refused("stage-list")
    path = Path(lines[0])
    if not Path(os.path.realpath(path)).is_relative_to(Path(os.path.realpath(stage))):
        raise Refused("stage-list")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return {json.loads(line)["run_id"] for line in f if line.strip()}


# -- command line -----------------------------------------------------------------------------------------------------------------


def write_new(path: Path, text: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(text)


def run_check(args: argparse.Namespace) -> int:
    out, phase = args.out_name, args.phase
    out_dirs = [Path(d) for d in args.out_dir]
    mode = ("exclude" if args.exclude_names is not None else "only" if args.only_names is not None
            else "select" if args.select_names is not None else None)  # an empty spec is a spec
    combined = out in COMBINED_NAMES
    if not combined and (out, phase) not in out_phases():
        raise Refused("bad-name")  # the offending text is not echoed
    if combined or len(out_dirs) > 1:  # a matrix of several OUTs: only so, and only with the results folder and marker named by it
        if not (combined and len(out_dirs) >= 2 and mode == "only" and args.audit_name == out
                and args.marker_name == f"{out}-{phase}"):
            raise Refused("bad-name")
    if not all(d.is_dir() for d in out_dirs):
        raise Refused("out-dir-missing")
    stage_dir = Path(os.path.abspath(args.stage))  # absolute: matrices.list is read from another working directory
    if os.path.lexists(stage_dir):
        raise Refused("stage-exists")
    for given, pattern in ((args.audit_name, AUDIT_NAME), (args.marker_name, MARKER_NAME)):
        if given is not None and not pattern.fullmatch(given):
            raise Refused("bad-name")
    marker_sha, marker_file_sha, disclosure = read_audit_marker(Path(args.done_dir), out, phase, args.marker_name)
    matrix = read_audit_folder(Path(args.audit_root), out, marker_sha, args.audit_name)
    names, counts = classify_outs(out_dirs)  # before any copy: a refusal leaves no stage
    specs = args.only_names if mode == "only" else [args.exclude_names if mode == "exclude" else args.select_names] if mode else []
    listed = listed_names(names, specs) if mode else []
    selected = (names - set(listed) if mode == "exclude" else set(listed) if mode == "only" else names)
    want = rm.run_ids_of(selected)  # what the ORIGINAL matrix must hold ("select": every record of the OUT, the rows kept are the listed ones)
    if not want:
        raise Refused("no-records")
    tree = f"audit-{args.audit_name or out}"
    stage_dir.mkdir(parents=True, mode=0o700)
    staged = rm.stage([rm.Located(phase, tree, "audit", matrix, marker_sha, ())], stage_dir, out)[0]
    rows, ids, outputs = coverage_a15(want, staged.matrix)
    counts[f"order_keys_{phase}"] = counts.pop("order_keys")
    counts[f"records_{phase}"] = ids
    counts[f"shards_{phase}"] = 1
    counts[f"rows_{phase}"] = rows
    served, served_sha, shown_ids, extra_columns, extra_cells = staged.matrix, marker_sha, ids, "", ""
    if mode == "select":  # the exporter gets a copy of the listed records' rows only; its run ids must be exactly the listed names'
        keep = rm.run_ids_of(set(listed))
        served = staged.matrix.with_name("audit.selected.matrix.jsonl.gz")
        served_sha = filter_matrix(staged.matrix, served, keep)
        rows_kept, shown_ids, outputs = coverage_a15(keep, served)
        counts[f"selected_{phase}"] = shown_ids
        counts[f"selected_rows_{phase}"] = rows_kept
        extra_columns, extra_cells = "\toriginal_sha256", f"\t{marker_sha}"
    write_new(stage_dir / "matrices.list", f"{served}\n")
    write_new(stage_dir / "replay-outputs.list", "".join(rid + "\n" for rid in sorted(outputs)))
    manifest = (f"out\tphase\ttree\tshard\tsha256\trecords{extra_columns}\n"
                f"{out}\t{phase}\t{tree}\taudit\t{served_sha}\t{shown_ids}{extra_cells}\n")
    write_new(stage_dir / "replay-matrices-manifest.tsv", manifest)
    print(f"LOCATED-A15 out={out} phase={phase} marker_sha256={marker_file_sha} matrix_sha256={marker_sha} "
          f"disclosure={int(disclosure)}" + (f" selected_sha256={served_sha}" if mode == "select" else ""))
    print(f"ROUTING out={out} kind=a15 unrouted=1 classes=none note=no-launch-records")
    selection = f" selection={mode} names_listed={len(listed)}" if mode else ""  # nothing is added without a selection
    print("COVERAGE out=" + out + "".join(f" {k}={v}" for k, v in sorted(counts.items())) + selection +
          f" matrices=1 manifest_sha256={hashlib.sha256(manifest.encode()).hexdigest()} OK")
    return 0


def run_join_check(args: argparse.Namespace) -> int:
    """The stages' run ids are pairwise disjoint and together exactly the records of all the OUTs: nothing missing, extra or twice."""
    names, _ = classify_outs([Path(d) for d in args.out_dir])
    records = rm.run_ids_of(names)
    seen: Counter = Counter()
    for stage in args.stage:
        seen.update(stage_run_ids(Path(stage)))
    twice = sum(1 for n in seen.values() if n > 1)
    missing, extra = len(records - set(seen)), len(set(seen) - records)
    if missing or extra or twice:
        raise Refused("join-differs", missing=missing, extra=extra, twice=twice)
    print(f"JOIN-CHECK kind=a15 stages={len(args.stage)} outs={len(args.out_dir)} records={len(records)} OK")
    return 0


def ledger_check_stages(stages: list[Path], ledger: Path):
    """The reviewed ledger check against ONE stage, or against several (the partitions of a union, one ledger across them): the runs
    with an output are the stages' lists together, and a run id in two stages refuses. For several stages the reviewed check runs on
    the combined list, written to a private directory inside the first stage and removed afterwards."""
    if len(stages) == 1:
        return rm.ledger_check(stages[0], ledger, ARMS)
    want: set[str] = set()
    for stage in stages:
        ids = set(rm.read_small(stage / "replay-outputs.list", 1 << 28).decode("utf-8").splitlines())
        if want & ids:
            raise Refused("stages-overlap")
        want |= ids
    work = Path(tempfile.mkdtemp(prefix=".combined-", dir=stages[0]))  # mode 700
    try:
        write_new(work / "replay-outputs.list", "".join(rid + "\n" for rid in sorted(want)))
        return rm.ledger_check(work, ledger, ARMS)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def run_ledger_check(args: argparse.Namespace) -> int:
    counts = ledger_check_stages([Path(s) for s in args.stage], Path(args.ledger))
    print("LEDGER-CHECK kind=a15" + "".join(f" {k}={v}" for k, v in sorted(counts.items())) + " OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = rm.QuietParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check-a15")
    c.add_argument("--audit-root", required=True, help="the folder holding audit-<OUT>/")
    c.add_argument("--done-dir", required=True, help="the done markers (hub done/, byte-copied)")
    c.add_argument("--out-dir", required=True, action="append", help="the OUT directory (or its view) the exporter gets with "
                   "--records; repeated only for a matrix of several OUTs (COMBINED_NAMES)")
    c.add_argument("--out-name", required=True, help="the OUT's name, one of the pinned pairs")
    c.add_argument("--phase", required=True, choices=PHASES)
    c.add_argument("--stage", required=True, help="a new private directory for the checked copy")
    c.add_argument("--audit-name", default=None, help="the results folder is R/audit-NAME instead of audit-<OUT>")
    c.add_argument("--marker-name", default=None, help="the marker is the file D/NAME instead of <OUT>-<phase>")
    sel = c.add_mutually_exclusive_group()
    sel.add_argument("--exclude-names", metavar="FILE:SHA256", default=None,
                     help="the matrix holds the OUT's records except those the pinned file of record names lists")
    sel.add_argument("--only-names", metavar="FILE:SHA256", default=None, action="append",
                     help="the matrix holds exactly the records the pinned file of record names lists (repeated: the union of the lists)")
    sel.add_argument("--select-names", metavar="FILE:SHA256", default=None,
                     help="the matrix holds exactly the OUT's records; the staged copy keeps only the rows of the listed records")
    jc = sub.add_parser("join-check-a15", help="the stages' run ids are disjoint and together the records of all the OUTs")
    jc.add_argument("--out-dir", required=True, action="append", help="an OUT directory (or its view); repeated")
    jc.add_argument("--stage", required=True, action="append", help="a stage of a successful check-a15; repeated")
    lc = sub.add_parser("ledger-check-a15", help="the exporter's ledger against the staged matrix")
    lc.add_argument("--stage", required=True, action="append", help="the stage directory of a successful check-a15; repeated, one "
                    "per partition of a union (one ledger across them)")
    lc.add_argument("--ledger", required=True, help="the ledger the exporter wrote from the staged matrix")
    args = ap.parse_args(argv)
    previous = os.umask(0o077)
    try:
        try:
            return {"check-a15": run_check, "join-check-a15": run_join_check, "ledger-check-a15": run_ledger_check}[args.cmd](args)
        except Refused as exc:
            print(f"REFUSED {exc}", file=sys.stderr)
            return 2
        except (OSError, EOFError, ValueError, KeyError, TypeError, AttributeError, zlib.error) as exc:  # a truncated gzip, a
            # malformed order file: the exception's class name only, never its text (which could quote a value)
            print(f"REFUSED io-error class={type(exc).__name__}", file=sys.stderr)
            return 2
    finally:
        os.umask(previous)  # a test calls main() in-process


if __name__ == "__main__":
    sys.exit(main())
