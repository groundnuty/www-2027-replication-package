#!/usr/bin/env python3
"""Locate and check the Tier 2 replay matrices that an e189 look hands the frozen exporter (design ruling (B), 25 Oct 2026).

The registered Tier 2 is "natural with offline checker replay": for a natural-only cell the replay is the only checker
source, and the prompted cohorts' off-diagonal arms (a source other than the one the cohort was prompted with) exist only there. The frozen exporter
(scripts/export_checker_ledger.py, 847cc275) takes the replay matrices with ``--matrix`` and the OUT with ``--records``. It
never asks whether every record has its rows: a record without them is simply absent from the ledger, and the analysis's own
coverage refusal (``build_arms``) fires only for an arm that has SOME rows. This tool is that missing check.

Three subcommands, stdlib only, all printing NAMES, COUNTS and sha256s only (never a run id, a row or a value, and no number
of the branch phase, entry 199), exit 0 (OK) or 2 (REFUSED, one line ``REFUSED <code> k=v ...``):

  locate        the matrix files the exporter must get for one OUT's phases, by the replay tree's documented layout
  check         locate, copy the files into a private stage directory (checked bytes == exported bytes), then the coverage check
  ledger-check  after the export: the ledger's replay rows are exactly the matrices' runs that have an output, once per arm

    python3 -I replay_matrices.py locate --root R --done-dir D --cells-tsv C --out-name OUT --phases natural,prompted
    python3 -I replay_matrices.py check  --root R --done-dir D --cells-tsv C --out-dir OUTDIR --phases natural,prompted \\
        --stage STAGE

THE LAYOUT (replay-tier2-20261023/PLAN.md, PLAN-CLASS-CELLS.md; bin/t2_worker.sh, seal.sh, t2_done.py, t2_cells_done.sh).
A tree ``R/<name>/`` holds, per phase P, ``shards-P.tsv`` (``shard<TAB>tree<TAB>model<TAB>records`` per line), ``shards-P/<shard>/
records/*.json`` (symlinks: the record file NAMES ``<40-hex claim digest>.<run id>.json``), and the checker's ONLINE pass:
``out/<shard>/matrix.jsonl.gz`` with ``runrec/<shard>.json`` (rc, sealed_utc and the sha256 of every file it sealed). The OBQC
pass writes a second matrix (``obqc/SEALED-obqc/<shard>/``) over the SAME arms plus obqc and obqc-where; the exporter has no arm
filter and refuses a repeated (run id, source, use) row, so only the online matrices are given and the OBQC arms are not
exported. An OUT-phase is replayed either whole in ``R/<OUT>/`` (done marker ``OK shards N ...``) or, when the class rule
split it, in ``R/<OUT>-cell-<fast|medium|slow>/`` trees and NOT in ``R/<OUT>/`` (marker ``OK cells K shards S ...``, and the hub's
cells.tsv lists the cells). A class cell is a subset of the OUT-phase's records, replayed on a copy of its own speed class: its
rows carry no class field (cohort = the grid cell) and the record sets of the cells are disjoint, so the exporter takes them as
they are. The done marker (hub ``done/<OUT>-<phase>``, staged beside cells.tsv) is the gate: a phase with none is refused.

THE LOCATOR refuses when: the phase has no marker, a malformed one, or a marker whose shard count (or cell count) is not the
layout's; the phase is in the main tree AND in cells, or in neither; the cells are not exactly the cells.tsv lines (or their
record counts, which must add up to the phase total); a shard list is malformed, its shard dirs or record counts differ from it;
a runrec is missing, has rc != 0, names another shard, or was sealed after the marker; a matrix is missing, is a symlink, is
not ``<tree>/out/<shard>/matrix.jsonl.gz`` (a ``superseded`` component, a symlink or a path outside the tree), or its sha256
is not the one the runrec sealed. It never globs: the files it names are those the shard list names.

THE COVERAGE CHECK (the OUT is the very directory ``--records`` and ``--batch`` get): every ``records/*.json`` name (a
``.setaside.json`` aside) is put into exactly one of the phases natural, prompted, branch by the OUT's own order files and
served-models.json (the claim-digest rule of phase_status.py: sha256 of the JSON of [arm, question, route model, repetition],
first 40 hex), and is refused if it matches none, two, or shares its digest with another record; an order key of a requested
phase with no route, or two keys with one digest, is refused. Then, per phase: the shards' record names ARE the OUT's records of
that phase (none missing, none extra, none in two shards); each shard's matrix holds exactly its records' run ids, each with
exactly one row per arm and variant (the worker's seven sources and the union, four variants), no repeated row, no other arm,
and every row says whether its run has an output (a bool, the same on all of the run's rows) and, with an output, its scorer
label. The exporter SKIPS a row without an output, and the analysis's ``build_arms`` checks only the arms the ledger has, so
the check also writes ``replay-outputs.list`` (the run ids with an output, mode 600) and ``ledger-check`` requires the exported
ledger to hold, for each of the eight arms, exactly those runs once: an export that dropped a run is refused.

    python3 -I replay_matrices.py ...   # constants below are the worker's (t2_worker.sh SOURCES and UNION), not options

THE CLASS ROUTING: check writes replay-routing/5. The first-line GNU time launch record is primary, bound by the
class table and the producer's per-tree TIMEFILES-<phase>.sha256, itself pinned by ONE TIMEFILES-INDEX TSV: a single index or the
consolidated union of the incremental indexes (TIMEFILES-INDEX-ALL-<tag>.tsv, see read_timefiles_manifest). A MISSING
line allows a per-shard t2-attest/1 fallback; a present but unlisted time file, changed listed bytes, or a launch
rejected by this consumer without an ok not_recorded attestation row is unverified. The live file's mtime gates the
claim/seal window but is not pinned. The operator's STATE t2-attest-signed line is verified when the analysis lead
sets the attestation PATH:SHA256 pin; this tool does not read STATE. Line 2 after a nonzero exit cannot be positive
evidence here because the done-check provides no reviewed consumer-readable per-shard accepted-rc record.

Counts go to stdout; a refusal prints its code and counts only. ``check --stage DIR`` creates DIR (mode 700, must not exist),
copies each matrix there (mode 600) while hashing it, requires the copy's sha256 to be the runrec's, runs the coverage check
on the COPIES, and writes ``matrices.list`` (one absolute path per line: the exporter's ``--matrix`` arguments) and
``replay-matrices-manifest.tsv`` (OUT, phase, tree, shard, sha256, records, rows): the replay rows carry no class or copy, so
this manifest is the package's only record of which class cell each matrix came from.
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import json
import os
import re
import stat
import sys
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# The worker's constants (bin/t2_worker.sh: SOURCES=curated,census-t5,census-t2,census-t20,generated-1,generated-2,generated-3
# and UNION=union=curated,census-t5), and harness/checkloop/matrix.py's VARIANTS keys. A test pins both lists.
SOURCES = ("curated", "census-t5", "census-t2", "census-t20", "generated-1", "generated-2", "generated-3")
UNION_ARMS = ("union",)
ARMS = SOURCES + UNION_ARMS
VARIANTS = ("primary", "heuristic_off", "branch_local_off", "both_off")
MATRIX_VERSION = "checker-replay-matrix/1"
OUTPUT_PROBLEMS = ("output_not_bool", "output_disagrees", "output_without_label")
REPLAY_PHASES = ("natural", "prompted")
ORDER_PHASES = ("natural", "prompted", "branch")
SPLIT = "test"

# t2_done.py SIB_OUTS/H6_OUTS and OUT_RE; plain OUT names only (no cell suffix).
SIB_OUTS = ("sib-rhea-sonnet", "sib-sonnet-32k", "sib-rhea-qwen3.8-27b-81920", "sib-qwen3.8-27b-81920",
            "sib-sonnet-32k-cap128k", "sib-rhea-sonnet-cap128k") + tuple(
                f"sib-rhea-qwen3.8-27b-81920-clamp-b{i:02d}" for i in range(1, 13))
H6_OUTS = ("rhea-h6-claude-opus-5-5", "opaque-h6-claude-opus-5-5", "uniprot-h6-claude-opus-5-5")
OUT_RE = re.compile(r"^(?:(rhea|opaque|corporate)-(sonnet|haiku|gemma-4-31b|glm-5\.3-flash|qwen3-coder-30b-a3b|qwen3\.8-27b|"
                    r"deepseek-v4-flash-0731)(-(e180|e184|e190|dh2|dsh)-p[0-9]+of[0-9]+)?|"
                    + "|".join(map(re.escape, H6_OUTS + SIB_OUTS)) + r")$")
CLASSES = ("fast", "medium", "slow")
SHARD_RE = re.compile(r"^[a-z0-9.-]+$")
# '<claim digest>.<run id>.json' (phase_status.py's RUN_FILE, with the run id narrowed to the runner's: a UUID, so letters, digits, "-" and "_";
# no whitespace, which the run-id lists below split on)
RUN_FILE = re.compile(r"^([0-9a-f]{40})\.([A-Za-z0-9_-]+)\.json$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
UTC = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z"
MARKER_PLAIN = re.compile(rf"OK shards (\d+) online (\d+) obqc (\d+) errors (0|accepted) ({UTC})")
MARKER_CELLS = re.compile(rf"OK cells (\d+) shards (\d+) online (\d+) obqc (\d+) errors (0|accepted) ({UTC})")


class Refused(Exception):
    """A refusal: a code and counts (and names of trees or shards), never a run id, a row or a value."""

    def __init__(self, code: str, **counts: object) -> None:
        self.code = code
        self.counts = counts
        super().__init__(code + "".join(f" {k}={v}" for k, v in counts.items()))


@dataclass
class Located:
    phase: str
    tree: str  # the tree's directory name (the OUT, or OUT-cell-CLASS)
    shard: str
    matrix: Path  # the file to read: the tree's own, or, after staging, the stage's copy
    sha256: str  # the sha256 the runrec sealed (the matrix file must have it)
    record_names: tuple[str, ...]  # the shard's record file names (never printed)
    sealed: float = 0.0  # the runrec's sealed_utc as an epoch: when the shard's online pass ended
    started: int | None = None  # when it was claimed, in nanoseconds: the mtime of the `worker` file of its claim (None: no claim to read)


# -- small helpers ------------------------------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def utc_epoch(text: str) -> float:
    return datetime.datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc).timestamp()


def read_small(path: Path, limit: int = 1 << 20) -> bytes:
    """A regular, non-symlink file's bytes (opened with O_NOFOLLOW), at most ``limit`` of them."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0))
    except OSError:
        raise Refused("unreadable", file=path.name) from None
    with os.fdopen(fd, "rb") as f:
        if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
            raise Refused("not-a-regular-file", file=path.name)
        data = f.read(limit + 1)
    if len(data) > limit:
        raise Refused("too-large", file=path.name)
    return data


def read_json_file(path: Path, what: str):
    try:
        return json.loads(read_small(path).decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise Refused("malformed-json", what=what) from None


def is_symlink(path: Path) -> bool:
    return path.is_symlink()


# -- the done marker and the cells table -------------------------------------------------------------------------------------


@dataclass
class Marker:
    sha256: str
    utc: str
    kind: str  # "shards" | "cells"
    cells: int | None
    shards: int


def read_marker(done_dir: Path, out: str, phase: str) -> Marker:
    path = done_dir / f"{out}-{phase}"
    if not (os.path.lexists(path)):
        raise Refused("no-done-marker", out=out, phase=phase)
    data = read_small(path, 4096)
    text = data.decode("utf-8", "replace")
    if not text.endswith("\n") or text.count("\n") != 1:
        raise Refused("marker-format", out=out, phase=phase)
    line = text[:-1]
    m = MARKER_CELLS.fullmatch(line)
    if m:
        k, s, on, ob, err, utc = int(m[1]), int(m[2]), int(m[3]), int(m[4]), m[5], m[6]
        if not (s == on == ob):
            raise Refused("marker-counts", out=out, phase=phase)
        return Marker(hashlib.sha256(data).hexdigest(), utc, "cells", k, s)
    m = MARKER_PLAIN.fullmatch(line)
    if m:
        s, on, ob, utc = int(m[1]), int(m[2]), int(m[3]), m[5]
        if not (s == on == ob):
            raise Refused("marker-counts", out=out, phase=phase)
        return Marker(hashlib.sha256(data).hexdigest(), utc, "shards", None, s)
    raise Refused("marker-format", out=out, phase=phase)


def read_cells_tsv(path: Path) -> tuple[list[tuple], str]:
    """(lines, sha256) of the staged cells.tsv: ``real OUT, phase, class, cell name, records in the cell, records in the phase``."""
    data = read_small(path)
    rows, seen = [], set()
    for ln in data.decode("utf-8").splitlines():
        if not ln.strip() or ln.startswith("#"):
            continue
        f = ln.split("\t")
        if len(f) != 6 or f[1] not in REPLAY_PHASES or f[2] not in CLASSES or f[3] != f"{f[0]}-cell-{f[2]}" \
                or not (f[4].isdigit() and f[5].isdigit()) or not OUT_RE.fullmatch(f[0]) or (f[0], f[1], f[2]) in seen:
            raise Refused("cells-tsv-malformed")
        seen.add((f[0], f[1], f[2]))
        rows.append((f[0], f[1], f[2], f[3], int(f[4]), int(f[5])))
    return rows, hashlib.sha256(data).hexdigest()


# -- the locator --------------------------------------------------------------------------------------------------------------


def shard_list(tree: Path, phase: str) -> list[tuple[str, int]]:
    """[(shard, records)] of ``tree/shards-PHASE.tsv`` checked against the shard dirs on disk (as t2_done.py does)."""
    tsv = tree / f"shards-{phase}.tsv"
    rows = [ln.split("\t") for ln in read_small(tsv).decode("utf-8").splitlines() if ln.strip()]
    if not rows:
        raise Refused("empty-shard-list", tree=tree.name, phase=phase)
    shards = [r[0] for r in rows]
    if len(set(shards)) != len(shards) or any(len(r) < 4 or r[1] != tree.name or not SHARD_RE.fullmatch(r[0])
                                              or not r[3].isdigit() for r in rows):
        raise Refused("shard-list-malformed", tree=tree.name, phase=phase)
    shard_dir = tree / f"shards-{phase}"
    if is_symlink(shard_dir) or not shard_dir.is_dir():
        raise Refused("shard-dirs-differ", tree=tree.name, phase=phase)
    on_disk = sorted(p.name for p in shard_dir.iterdir() if p.is_dir())
    if on_disk != sorted(shards):
        raise Refused("shard-dirs-differ", tree=tree.name, phase=phase)
    return [(r[0], int(r[3])) for r in rows]


def record_names(tree: Path, phase: str, shard: str, expected: int) -> tuple[str, ...]:
    rec_dir = tree / f"shards-{phase}" / shard / "records"
    if is_symlink(rec_dir.parent) or is_symlink(rec_dir) or not rec_dir.is_dir():
        raise Refused("record-count", tree=tree.name, shard=shard)
    names = sorted(os.listdir(rec_dir))
    if any(not RUN_FILE.fullmatch(n) for n in names) or len(names) != expected or len(set(names)) != len(names):
        raise Refused("record-count", tree=tree.name, shard=shard)
    return tuple(names)


def matrix_of_shard(tree: Path, shard: str, marker: Marker) -> tuple[Path, str, float]:
    """The shard's online matrix, the sha256 its runrec sealed and the sealed time (epoch); refuses anything but
    ``tree/out/<shard>/matrix.jsonl.gz``."""
    rr = read_json_file(tree / "runrec" / f"{shard}.json", "runrec")
    if not isinstance(rr, dict) or rr.get("shard") != shard or rr.get("rc") != 0 or isinstance(rr.get("rc"), bool):
        raise Refused("runrec-differs", tree=tree.name, shard=shard)
    sha, sealed = rr.get("matrix.jsonl.gz"), rr.get("sealed_utc")
    if not (isinstance(sha, str) and SHA_RE.fullmatch(sha)) or not (isinstance(sealed, str) and re.fullmatch(UTC, sealed)):
        raise Refused("runrec-differs", tree=tree.name, shard=shard)
    if utc_epoch(sealed) > utc_epoch(marker.utc):
        raise Refused("sealed-after-marker", tree=tree.name, shard=shard)
    out_dir = tree / "out"
    shard_dir = out_dir / shard
    path = shard_dir / "matrix.jsonl.gz"
    if any(is_symlink(p) for p in (out_dir, shard_dir, path)) or not os.path.lexists(path):
        raise Refused("matrix-missing-or-linked", tree=tree.name, shard=shard)
    real_tree = Path(os.path.realpath(tree))
    if Path(os.path.realpath(path)) != real_tree / "out" / shard / "matrix.jsonl.gz" or "superseded" in path.relative_to(tree).parts:
        raise Refused("matrix-outside-out", tree=tree.name, shard=shard)
    if not stat.S_ISREG(os.lstat(path).st_mode):
        raise Refused("matrix-missing-or-linked", tree=tree.name, shard=shard)
    if sha256_file(path) != sha:
        raise Refused("matrix-sha-differs", tree=tree.name, shard=shard)
    return path, sha, utc_epoch(sealed)


def claim_started(root: Path, tree: str, shard: str, sealed: float) -> int | None:
    """When the worker claimed the shard, in integer nanoseconds: the mtime of the ``worker`` file in ``<root>/claims/<tree>.<shard>.online``,
    written once at the claim (t2_worker.sh next_job) and never again. The shard's time file cannot be older than that. None when the claim
    directory or that file is not plain, or the file is dated after the seal. Times are compared in integer nanoseconds, never as floats."""
    claim = root / "claims" / f"{tree}.{shard}.online"
    try:
        if not stat.S_ISDIR(os.lstat(claim).st_mode):
            return None
        st = os.lstat(claim / "worker")
    except OSError:
        return None
    return st.st_mtime_ns if stat.S_ISREG(st.st_mode) and st.st_mtime_ns <= round(sealed) * NS else None


def locate_phase(root: Path, done_dir: Path, cells: list[tuple], out: str, phase: str) -> tuple[Marker, str, list[Located]]:
    """(marker, layout, located shards) of one OUT-phase. Never globs a matrix: the shard lists name them."""
    if not OUT_RE.fullmatch(out) or phase not in REPLAY_PHASES:
        raise Refused("bad-name", out=out, phase=phase)
    marker = read_marker(done_dir, out, phase)
    lines = [c for c in cells if c[0] == out and c[1] == phase]
    cell_re = re.compile(rf"^{re.escape(out)}-cell-({'|'.join(CLASSES)})$")
    with_list = sorted(p.name for p in root.iterdir()
                       if cell_re.fullmatch(p.name) and os.path.lexists(p / f"shards-{phase}.tsv"))
    main_has = os.path.lexists(root / out / f"shards-{phase}.tsv")
    if main_has and (lines or with_list):
        raise Refused("layout-both", out=out, phase=phase)
    if not main_has and not (lines or with_list):
        raise Refused("layout-neither", out=out, phase=phase)
    if main_has:
        layout, trees, expect = "main", [out], None
        if marker.kind != "shards":
            raise Refused("marker-layout", out=out, phase=phase)
    else:
        layout, trees, expect = "cells", sorted(c[3] for c in lines), {c[3]: c[4] for c in lines}
        if not lines or with_list != trees or marker.kind != "cells" or marker.cells != len(trees):
            raise Refused("cells-differ", out=out, phase=phase, cells_tsv=len(lines), trees=len(with_list))
    located: list[Located] = []
    per_tree: dict[str, int] = {}
    for name in trees:
        tree = root / name
        if is_symlink(tree) or not tree.is_dir():
            raise Refused("tree-missing", tree=name)
        recs = 0
        for shard, n in shard_list(tree, phase):
            names = record_names(tree, phase, shard, n)
            matrix, sha, sealed_at = matrix_of_shard(tree, shard, marker)
            located.append(Located(phase, name, shard, matrix, sha, names, sealed_at, claim_started(root, name, shard, sealed_at)))
            recs += n
        per_tree[name] = recs
    if marker.shards != len(located):
        raise Refused("marker-shards", out=out, phase=phase, marker=marker.shards, found=len(located))
    if expect is not None:
        if per_tree != expect or sum(per_tree.values()) != lines[0][5] or {c[5] for c in lines} != {lines[0][5]}:
            raise Refused("cells-records", out=out, phase=phase, found=sum(per_tree.values()), total=lines[0][5])
    names_all = [n for loc in located for n in loc.record_names]
    if len(set(names_all)) != len(names_all):
        raise Refused("record-in-two-shards", out=out, phase=phase)
    return marker, layout, located


# -- the coverage check -------------------------------------------------------------------------------------------------------


def claim_digest(arm: str, question: str, route: str, repetition) -> str:
    """phase_status.py's digest_of for a grid phase: sha256 of the JSON of [arm, question, route model, repetition]."""
    return hashlib.sha256(json.dumps([arm, question, route, repetition], ensure_ascii=False).encode("utf-8")).hexdigest()[:40]


def classify_records(out_dir: Path, wanted: tuple[str, ...]) -> tuple[dict[str, set[str]], Counter]:
    """({phase: the OUT's record file names of the phase}, counts). Every records/ name goes to exactly one phase."""
    served = read_json_file(out_dir / "served-models.json", "served-models").get("models") or {}
    routes = {name: str(v["route_model"]) for name, v in served.items() if isinstance(v, dict) and v.get("route_model")}
    route_of = {name: r for name, r in routes.items() if list(routes.values()).count(r) == 1}  # one name per route only
    phase_of_digest: dict[str, str] = {}
    counts: Counter = Counter()
    for phase in ORDER_PHASES:
        path = out_dir / f"order-{phase}.json"
        if not os.path.lexists(path):
            if phase in wanted:
                raise Refused("no-order-file", phase=phase)
            continue
        order = read_json_file(path, f"order-{phase}").get("order")
        if not isinstance(order, list):
            raise Refused("order-malformed", phase=phase)
        digests: list[str | None] = []
        for k in order:
            name, arm, qid, rep = k
            digests.append(claim_digest(arm, qid, route_of[name], rep) if name in route_of else None)
        if phase in wanted:  # entry 199: no number of the branch phase is ever printed (its order is read to classify records only)
            counts[f"order_keys_{phase}"] = len(order)
            unrouted = sum(d is None for d in digests)
            shared = sum(n for n in Counter(d for d in digests if d is not None).values() if n > 1)
            if unrouted or shared:
                raise Refused("order-keys-ambiguous", phase=phase, unrouted=unrouted, shared=shared)
        for d in set(x for x in digests if x is not None):
            if d in phase_of_digest and phase_of_digest[d] != phase:
                raise Refused("digest-in-two-phases", phase=phase)
            phase_of_digest[d] = phase
    by_phase: dict[str, set[str]] = {p: set() for p in ORDER_PHASES}
    seen_digest: Counter = Counter()
    seen_run: Counter = Counter()
    unclassified = other = 0
    rec_dir = out_dir / "records"
    for n in os.listdir(rec_dir):
        if n.endswith(".setaside.json"):
            continue
        m = RUN_FILE.fullmatch(n)
        if not m:
            if n.endswith(".json"):
                other += 1
            continue
        seen_digest[m[1]] += 1
        seen_run[m[2]] += 1
        phase = phase_of_digest.get(m[1])
        if phase is None:
            unclassified += 1
        else:
            by_phase[phase].add(n)
    twice = sum(1 for n in seen_digest.values() if n > 1)
    run_twice = sum(1 for n in seen_run.values() if n > 1)  # two files, two digests, one run id: the matrix rows are keyed by run id
    if unclassified or other or twice or run_twice:  # flags, not counts: a record of no phase may be a branch record (entry 199)
        raise Refused("records-unclassified", unclassified=int(bool(unclassified)), odd_names=int(bool(other)),
                      digest_twice=int(bool(twice)), run_id_twice=int(bool(run_twice)))
    return by_phase, counts


def read_matrix(path: Path, arms: tuple[str, ...], variants: tuple[str, ...]) -> tuple[int, Counter, set[str], set[str]]:
    """(rows, problems, run ids, run ids with an output) of one matrix: the (arm, variant) set of each run id must be exactly
    arms x variants, every row says whether its run has an output (``output``, a bool, the same on all of a run's rows, which the
    exporter reads: it skips every row without one) and a run with an output has its scorer label (``correct``, a bool; the
    exporter refuses an output without one). Problems are counted, never named."""
    per_run: dict[str, set[tuple[str, str]]] = {}
    outs: dict[str, set[bool]] = {}
    problems: Counter = Counter()
    rows = 0
    want = {(a, v) for a in arms for v in variants}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            rows += 1
            if not isinstance(r, dict) or not all(isinstance(r.get(k), str) for k in ("run_id", "arm", "variant")):
                problems["malformed_rows"] += 1
                continue
            if r.get("version") != MATRIX_VERSION:
                problems["wrong_version_rows"] += 1
            if r.get("split") != SPLIT:
                problems["wrong_split_rows"] += 1
            key = (r["arm"], r["variant"])
            seen = per_run.setdefault(r["run_id"], set())
            if key in seen:
                problems["duplicate_rows"] += 1
            elif key not in want:
                problems["extra_rows"] += 1
            seen.add(key)
            if not isinstance(r.get("output"), bool):
                problems["output_not_bool"] += 1
            else:
                outs.setdefault(r["run_id"], set()).add(r["output"])
                if r["output"] and not isinstance(r.get("correct"), bool):
                    problems["output_without_label"] += 1
    for seen in per_run.values():
        problems["missing_rows"] += len(want - seen)
    problems["output_disagrees"] += sum(1 for v in outs.values() if len(v) > 1)
    return rows, problems, set(per_run), {rid for rid, v in outs.items() if v == {True}}


def run_ids_of(names) -> set[str]:
    return {RUN_FILE.fullmatch(n)[2] for n in names}


def coverage(out_dir: Path, located: dict[str, list[Located]], arms: tuple[str, ...] = ARMS,
             variants: tuple[str, ...] = VARIANTS) -> tuple[Counter, set[str]]:
    """(counts, the run ids that have an output) of the coverage check, or Refused."""
    phases = tuple(p for p in REPLAY_PHASES if p in located)
    by_phase, counts = classify_records(out_dir, phases)
    total = Counter(counts)
    outputs: set[str] = set()
    per_run = len(arms) * len(variants)
    for phase in phases:
        shards = located[phase]
        in_shards = [n for loc in shards for n in loc.record_names]
        want = by_phase[phase]
        missing, extra = len(want - set(in_shards)), len(set(in_shards) - want)
        if missing or extra or len(set(in_shards)) != len(in_shards):
            raise Refused("shards-differ-from-out", phase=phase, out_records=len(want), shard_records=len(in_shards),
                          missing=missing, extra=extra)
        phase_rows = 0
        for loc in shards:
            rows, problems, ids, with_output = read_matrix(loc.matrix, arms, variants)
            ids_want = run_ids_of(loc.record_names)
            gone, stray = len(ids_want - ids), len(ids - ids_want)
            n_names = len(loc.record_names)
            if gone or stray or any(problems.values()) or rows != per_run * n_names or len(ids_want) != n_names:
                # a problem with the output flags is printed as a flag, not a count: how many rows lack a label or disagree is a
                # number about which runs have an output, a property of their results
                shown = {k: (int(bool(v)) if k in OUTPUT_PROBLEMS else v) for k, v in sorted(problems.items()) if v}
                raise Refused("matrix-differs-from-records", tree=loc.tree, shard=loc.shard, records=n_names,
                              records_without_rows=gone, rows_of_no_record=stray, rows=rows, rows_expected=per_run * n_names,
                              **shown)
            phase_rows += rows
            outputs |= with_output
        total[f"records_{phase}"] = len(in_shards)
        total[f"shards_{phase}"] = len(shards)
        total[f"rows_{phase}"] = phase_rows
    return total, outputs


def ledger_check(stage_dir: Path, ledger: Path, arms: tuple[str, ...] = ARMS) -> Counter:
    """The exporter's ledger against the staged matrices: for every arm, the ledger's replay rows are exactly the runs the
    matrices say have an output (the exporter skips a row with no output and so would drop a run silently), once each, and no
    other replay source is there. Counts, or Refused."""
    want = set(read_small(stage_dir / "replay-outputs.list", 1 << 28).decode("utf-8").splitlines())  # one run id a line
    if not want:
        raise Refused("no-outputs-listed")
    by_source: dict[str, Counter] = {}
    malformed = 0
    with open(ledger, encoding="utf-8") as f:
        header = f.readline().rstrip("\n")
        if header != json.dumps({"schema": "checker-ledger/2", "split": SPLIT}):
            raise Refused("ledger-header")
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("use") == "replay":
                if isinstance(r.get("source"), str) and isinstance(r.get("run_id"), str):
                    by_source.setdefault(r["source"], Counter())[r["run_id"]] += 1
                else:
                    malformed += 1  # an id or source that is not a string never matches a run of the list
    unknown = malformed + sum(sum(c.values()) for src, c in by_source.items() if src not in arms)
    missing = extra = duplicate = 0
    for arm in arms:
        got = by_source.get(arm, Counter())
        missing += len(want - set(got))
        extra += len(set(got) - want)
        duplicate += sum(1 for n in got.values() if n > 1)
    if missing or extra or duplicate or unknown:
        # flags: with a valid header and no replay rows, "missing" would be eight times the number of runs that have an output
        raise Refused("ledger-differs-from-matrices", missing=int(bool(missing)), extra=int(bool(extra)),
                      duplicate=int(bool(duplicate)), other_sources=int(bool(unknown)))
    return Counter(arms=len(arms))  # no number of outputs is printed: whether a run has one is a property of its result


# -- the class routing of the replay rows (science, 25 Oct 11:04Z; recorded-field rule, analysis lead 11:24Z) -----------------------

# The copy a shard's online pass ran on is taken ONLY from what the worker recorded of that very launch, never inferred: the worker
# runs the frozen checker as ``/usr/bin/time -v -o <tree>/logs/<shard>.time python scripts/checker_replay.py ... --endpoint "$EP" ...
# --note "Tier 2 $phase replay, freeze 126a111, $out shard $shard, $COPYID, run-node" ...`` (t2_worker.sh run_online), and GNU time
# writes the command it ran, arguments as launched, on the first line of that file ("Command being timed: ..."), when the process
# ends. So the file names the endpoint the checker really used (an environment override included), the copy id the worker meant,
# and the tree and shard it was for. Nothing else the worker writes names a copy: runrec/<shard>.json holds host, job, rc, the seal time
# and hashes; the matrix rows and the store log name none; the endpoint is also in the sealed accounting.json's meta, which reviewed
# code does not open. Only the "Command being timed" line is read, and only its endpoint, copy id, tree, shard and phase are kept.
CELL_TREE_RE = re.compile(r"^(.+)-cell-(fast|medium|slow)$")
UNVERIFIABLE = "unverifiable"  # a copy was recorded but the class table does not bind it (endpoint, copy id) to a class
NOT_RECORDED = "not_recorded"  # no usable record of the copy: no time file, not this shard's, not plain, or not parseable
ROUTING_SCHEMA = "replay-routing/5"
TIME_FILE_LIMIT = 1 << 16  # a time file is under a few KB; the first line holds the command
NS = 10**9  # nanoseconds in a second: every date is compared as an integer number of nanoseconds
MANIFEST_READ_LIMIT = 1 << 26  # the most bytes hashed for the manifest comparison (a larger file differs from any listed one in practice)
TIME_BEFORE_SEAL_S = 300  # the worker seals the shard right after the checker process ends (wait, then seal.sh), and GNU time writes its file at that end
TIME_AFTER_SEAL_S = 60  # clock skew only: the file is never written after the seal
# The WHOLE command as the worker launches it (t2_worker.sh run_online), anchored at both ends, every argument a token without whitespace
# except the note, whose fixed text names the phase, the tree, the shard and the copy id (the id limited to [A-Za-z0-9.-]); so text inside
# one argument can never supply another argument's value.
LAUNCH_COMMAND = re.compile(
    r"\S+ scripts/checker_replay\.py --records \S+ --sources-dir \S+ --sources \S+ --union \S+ --hierarchy \S+ "
    r"--endpoint http://([0-9A-Za-z.-]+:[0-9]{1,5})/ --split test "
    r"--note Tier 2 (natural|prompted) replay, freeze 126a111, (\S+) shard (\S+), ([A-Za-z0-9.-]+), run-node --out \S+")
COMMAND_LINE = re.compile(r'^\tCommand being timed: "(.*)"$')
EXIT_LINE = re.compile(r"^\tExit status: 0$", re.M)


@dataclass
class ClassTable:
    rows: set  # {(host:port, copy id, class)}: bin/cell-classes.tsv, the table t2_cell_copy_ok binds a cell's copy to its class with


def read_cell_classes(path: Path) -> tuple[ClassTable, str]:
    """(table, sha256) of the staged bin/cell-classes.tsv (endpoint, copy id, class, note), both from ONE read of the file; a copy
    id of two classes refuses."""
    data = read_small(path)
    rows: set = set()
    classes: dict[str, str] = {}
    for ln in data.decode("utf-8").splitlines():
        if not ln.strip() or ln.startswith("#"):
            continue
        f = ln.split("\t")
        if len(f) < 3 or f[2] not in CLASSES or not f[1] or not re.fullmatch(r"[A-Za-z0-9.-]+:[0-9]{1,5}", f[0]):
            raise Refused("cell-classes-malformed")
        if classes.setdefault(f[1], f[2]) != f[2]:
            raise Refused("cell-classes-conflict")
        rows.add((f[0], f[1], f[2]))
    return ClassTable(rows), hashlib.sha256(data).hexdigest()


def load_time_file(root: Path, tree: str, shard: str):
    """(bytes, mtime in nanoseconds, sha256) of ``<root>/<tree>/logs/<shard>.time`` read ONCE: the logs directory is opened with O_DIRECTORY|O_NOFOLLOW
    and the file relative to that descriptor with O_NOFOLLOW, so neither can be a symlink or be swapped between a check and the
    open; the file must be a regular file whose size and mtime are the same before and after the read (descriptor metadata). The
    bytes are whatever is there, parsed or not, so the manifest comparison and the parsing are of the same bytes. None otherwise."""
    try:
        dfd = os.open(root / tree / "logs", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError:
        return None
    try:
        fd = os.open(f"{shard}.time", os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0), dir_fd=dfd)
    except OSError:
        return None
    finally:
        os.close(dfd)
    with os.fdopen(fd, "rb") as f:
        before = os.fstat(f.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > MANIFEST_READ_LIMIT:
            return None
        data = f.read(MANIFEST_READ_LIMIT + 1)
        after = os.fstat(f.fileno())
    if len(data) > MANIFEST_READ_LIMIT or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or len(data) != after.st_size:
        return None  # it changed under the read
    return data, after.st_mtime_ns, hashlib.sha256(data).hexdigest()


def parse_launch_record(data: bytes, mtime_ns: int, tree: str, shard: str, phase: str, sealed: int, started_ns: int | None):
    """(endpoint host:port, copy id) of the shard's online launch from the bytes of its time file, or None when they are no usable
    record: at most TIME_FILE_LIMIT bytes, dated (``mtime_ns``, integer nanoseconds) within [seal - TIME_BEFORE_SEAL_S, seal + TIME_AFTER_SEAL_S] (``sealed`` whole seconds) (GNU time
    writes it when the checker process ends and the worker seals right after) and no earlier than the shard's claim (when the claim is
    there), say ``Exit status: 0``, and its first line must be GNU time's "Command being timed" with a command that is EXACTLY the
    worker's launch shape (LAUNCH_COMMAND) for THIS tree, shard and phase. Only those lines are read; nothing of the file but the two
    values is kept. The runrec does not hash the file (the worker's seal.sh does not): the manifest, when the caller has one, does."""
    if len(data) > TIME_FILE_LIMIT or mtime_ns > (sealed + TIME_AFTER_SEAL_S) * NS or mtime_ns < (sealed - TIME_BEFORE_SEAL_S) * NS:
        return None
    if started_ns is not None and mtime_ns < started_ns:
        return None
    text = data.decode("utf-8", "replace")
    m = COMMAND_LINE.fullmatch(text.split("\n", 1)[0])
    if not m or not EXIT_LINE.search(text):
        return None
    c = LAUNCH_COMMAND.fullmatch(m[1])
    if not c or (c[2], c[3], c[4]) != (phase, tree, shard):
        return None
    return c[1], c[5]


def read_launch_record(root: Path, tree: str, shard: str, phase: str, sealed: float, started: int | None):
    """(endpoint, copy id, sha256 of the file) of the shard's online launch, or None (load_time_file + parse_launch_record)."""
    t = load_time_file(root, tree, shard)
    rec = None if t is None else parse_launch_record(t[0], t[1], tree, shard, phase, round(sealed), started)
    return None if rec is None else (rec[0], rec[1], t[2])


def shard_replay_class(record, table: ClassTable) -> str:
    """The class of the recorded copy: the table's class for (endpoint, copy id) when that very pair is a row of it (the binding
    t2_cell_copy_ok makes for a cell); UNVERIFIABLE when a copy is recorded and the table does not bind it; NOT_RECORDED when
    nothing usable was recorded (never inferred from the store map, a config, a default or the cell's name)."""
    if record is None:
        return NOT_RECORDED
    classes = {cls for (ep, copy, cls) in table.rows if ep == record[0] and copy == record[1]}
    return classes.pop() if len(classes) == 1 else UNVERIFIABLE


# The producer's t2-attest/1 is a per-shard fallback. Recorded rows cannot
# attest, but their bindings must be checked against accepted launch records.
ATTESTATION_SCHEMA = "t2-attest/1"
ATTESTATION_LABEL = "operator attestation (from glue records)"
ENDPOINT_RE = re.compile(r"[0-9A-Za-z.-]+:[0-9]{1,5}")
UTC_RE = re.compile(UTC)
NOT_RECORDED_REASONS = {"time_file_missing", "time_file_not_regular", "time_file_empty", "first_line_exit_status",
                        "first_line_not_command", "command_not_checker_replay", "endpoint_missing", "endpoint_repeated",
                        "endpoint_malformed", "recorded_evidence_unvalidated", "recorded_evidence_rejected",
                        "command_shape_unexpected", "time_file_changed_while_read", "line2_exit_status_not_accepted"}


def unique_keys(pairs: list) -> dict:
    """object_pairs_hook: a JSON object that repeats a key is ambiguous (json.loads would keep the last), so it refuses."""
    out: dict = {}
    for k, v in pairs:
        if k in out:
            raise ValueError("repeated key")
        out[k] = v
    return out


@dataclass(frozen=True)
class Attestation:
    out: str
    phase: str
    shards: dict  # (tree, shard) -> (endpoint, copy id, class), only ok not_recorded rows
    states: dict  # (tree, shard) -> producer's recorded/not_recorded state
    recorded_bindings: dict  # (tree, shard) -> informational (endpoint, copy id, class)
    file_sha256: str
    content_sha256: str
    agent: str


def read_attestation(spec: str, out: str, phases: tuple[str, ...], table: ClassTable) -> Attestation:
    """Read a pinned t2-attest/1 file; an unverified phase is refused."""
    path_text, sep, pin = spec.rpartition(":")
    if not sep or not path_text or not SHA_RE.fullmatch(pin):
        raise Refused("attestation-pin-malformed")
    data = read_small(Path(path_text))
    file_sha = hashlib.sha256(data).hexdigest()
    if file_sha != pin:
        raise Refused("attestation-sha-differs")
    try:
        doc = json.loads(data.decode("utf-8"), object_pairs_hook=unique_keys)
    except ValueError:
        raise Refused("attestation-malformed") from None
    if not isinstance(doc, dict) or doc.get("schema") != ATTESTATION_SCHEMA or doc.get("label") != ATTESTATION_LABEL:
        raise Refused("attestation-malformed")
    # The producer signs every other field, including its signing disclosure.
    body = {k: v for k, v in doc.items() if k != "content_sha256"}
    content_sha = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")).hexdigest()
    if doc.get("content_sha256") != content_sha:
        raise Refused("attestation-content-sha-differs")
    if doc.get("out") != out or doc.get("phase") not in phases:
        raise Refused("attestation-names-another-out-phase")
    required = {"tool", "supersedes", "rulings", "shard_counts", "cells", "checks", "state_refs", "signing"}
    if not (required <= doc.keys() and isinstance(doc.get("out"), str) and isinstance(doc.get("phase"), str)
            and isinstance(doc.get("agent"), str) and doc["agent"]
            and isinstance(doc.get("generated_utc"), str) and UTC_RE.fullmatch(doc["generated_utc"])
            and doc.get("status") in ("replay_class_attested", "replay_class_unverified")
            and isinstance(doc.get("shards"), list) and doc["shards"]
            and isinstance(doc.get("evidence"), list) and doc["evidence"]
            and all(isinstance(e, dict) and isinstance(e.get("path"), str) and e["path"]
                    and isinstance(e.get("kind"), str) and e["kind"]
                    and isinstance(e.get("sha256"), str) and SHA_RE.fullmatch(e["sha256"]) for e in doc["evidence"])):
        raise Refused("attestation-malformed")
    if doc["status"] != "replay_class_attested":
        raise Refused("attestation-status-unverified")
    shards: dict = {}
    states: dict = {}
    recorded_bindings: dict = {}
    counts = Counter()
    row_fields = {"shard", "tree", "cell", "state", "state_reason", "claim_utc", "seal_utc", "run_end_utc", "rc",
                  "map_version", "map_key", "copy", "ep", "ep_sha256", "table_ep_sha256", "time_endpoint_sha256",
                  "time_evidence", "time_copy", "class", "class_basis", "basis", "evidence_label", "waiver", "checks",
                  "check_reasons", "ok", "reasons", "notes"}
    check_keys = {f"c{i}" for i in range(1, 5)}
    # evidence_label: the producer's t2_attest.py writes this exact text as a constant (LINE2_LABEL), whichever time-file index covered the
    # shard, and science ruled it; it is validated here as written, not tied to a manifest time (the label the analysis prints is
    # exploratory.py RECORDED_LABEL), and this consumer gives no row that label (line 2 is no usable launch record, see the docstring)
    for row in doc["shards"]:
        if not isinstance(row, dict) or not row_fields <= row.keys() or not isinstance(row.get("tree"), str) or not isinstance(row.get("shard"), str):
            raise Refused("attestation-shards-malformed")
        tree, shard = row["tree"], row["shard"]
        cell = CELL_TREE_RE.fullmatch(tree)
        if (tree != out and not (cell and cell.group(1) == out)) or not re.fullmatch(r"[a-z0-9.-]+", shard) or (tree, shard) in shards:
            raise Refused("attestation-shards-malformed")
        ep, copy, cls = row["ep"], row["copy"], row["class"]
        checks = row["checks"]
        if (row["cell"] != (cell.group(2) if cell else None) or row["state"] not in ("recorded", "not_recorded")
                or any(not isinstance(row[k], str) or not UTC_RE.fullmatch(row[k]) for k in ("claim_utc", "seal_utc", "run_end_utc"))
                or type(row["rc"]) is not int or row["rc"] < 0
                or not isinstance(row["map_version"], str) or not row["map_version"]
                or not isinstance(row["map_key"], str) or not row["map_key"]
                or (ep is not None and (not isinstance(ep, str) or not ENDPOINT_RE.fullmatch(ep)))
                or (copy is not None and (not isinstance(copy, str) or not copy))
                or (cls is not None and cls not in CLASSES)
                or row["ep_sha256"] != (hashlib.sha256(ep.encode()).hexdigest() if ep is not None else None)
                or not isinstance(checks, dict) or set(checks) != check_keys or any(type(v) is not bool for v in checks.values())
                or not isinstance(row["ok"], bool) or row["ok"] != all(checks.values())
                or not isinstance(row["check_reasons"], dict) or set(row["check_reasons"]) != check_keys
                or any(not isinstance(v, list) or any(not isinstance(x, str) for x in v) for v in row["check_reasons"].values())
                or not isinstance(row["reasons"], list) or not isinstance(row["notes"], list)
                or any(not isinstance(x, str) for x in row["reasons"] + row["notes"])
                or row["basis"] not in ("positive_time_endpoint", "chain_only", "time_endpoint_differs", "time_file_rejected",
                                        "line2_not_accepted", "unreconciled")
                or row["waiver"] not in (None, "ctime_birth_copy")
                or row["evidence_label"] not in (None, "recorded (worker launch record, line 2 after an accepted non-zero exit; unsealed; sha manifest taken 12:32Z)")):
            raise Refused("attestation-shards-malformed")
        if row["state"] == "recorded":
            if row["state_reason"] is not None:
                raise Refused("attestation-shards-malformed")
            if row["waiver"] == "ctime_birth_copy" and not row["ok"]:
                raise Refused("attestation-status-inconsistent")
            if row["ok"] and (ep is None or copy is None or cls is None):
                raise Refused("attestation-shards-malformed")
            recorded_bindings[(tree, shard)] = (ep, copy, cls)
        else:
            if ep is None or copy is None or cls is None:
                raise Refused("attestation-shards-malformed")
            if not isinstance(row["state_reason"], str) or row["state_reason"] not in NOT_RECORDED_REASONS:
                raise Refused("attestation-shards-malformed")
            if not row["ok"]:
                raise Refused("attestation-status-inconsistent")
            if (row["basis"] != "positive_time_endpoint" or row["time_endpoint_sha256"] != row["ep_sha256"]
                    or row["time_evidence"] is None
                    or row["time_copy"] != copy or row["state_reason"] in
                    ("time_file_missing", "time_file_not_regular", "time_file_empty", "first_line_not_command",
                     "command_not_checker_replay", "endpoint_missing", "endpoint_repeated", "endpoint_malformed",
                     "time_file_changed_while_read") or row["reasons"] or any(row["check_reasons"].values())):
                raise Refused("attestation-shards-malformed")
            if row["state_reason"] == "first_line_exit_status" and (row["rc"] < 1 or row["evidence_label"] is None):
                raise Refused("attestation-shards-malformed")
        counts[row["state"]] += 1
        counts["recorded_not_ok"] += row["state"] == "recorded" and not row["ok"]
        counts["not_recorded_chain_only"] += row["state"] == "not_recorded" and row["basis"] == "chain_only"
        counts["waived"] += row["waiver"] == "ctime_birth_copy"
        shards[(tree, shard)] = None
        states[(tree, shard)] = row.get("state")
        if row.get("state") != "not_recorded" or row.get("ok") is not True:
            continue
        if (not isinstance(ep, str) or not isinstance(copy, str) or not isinstance(cls, str)
                or (ep, copy, cls) not in table.rows or (cell and cls != cell.group(2))
                or row.get("ep_sha256") != hashlib.sha256(ep.encode()).hexdigest()
                or row.get("table_ep_sha256") != hashlib.sha256(ep.encode()).hexdigest()
                or row.get("checks") != {f"c{i}": True for i in range(1, 5)}):
            continue
        shards[(tree, shard)] = (ep, copy, cls)
    expected_counts = {"total": len(shards), "recorded": counts["recorded"], "not_recorded": counts["not_recorded"],
                       "recorded_not_ok": counts["recorded_not_ok"], "not_recorded_chain_only": counts["not_recorded_chain_only"],
                       "waived": counts["waived"]}
    if doc.get("shard_counts") != expected_counts:
        raise Refused("attestation-counts-inconsistent")
    # In an attested phase every cell must have shards and pass its own gating rows.
    cell_counts = Counter(tree for tree, _ in shards if CELL_TREE_RE.fullmatch(tree))
    cells = doc["cells"]
    if cell_counts:
        if (not isinstance(cells, list) or len(cells) != len(cell_counts)
                or any(not isinstance(c, dict) or not isinstance(c.get("cell"), str) or c["cell"] not in cell_counts
                       or c.get("class") != CELL_TREE_RE.fullmatch(c["cell"]).group(2)
                       or c.get("shards") != cell_counts[c["cell"]] or c.get("status") != "pass" for c in cells)
                or len({c["cell"] for c in cells}) != len(cells)):
            raise Refused("attestation-status-inconsistent")
    elif cells is not None:
        raise Refused("attestation-status-inconsistent")
    check_names = ("c1_no_override", "c2_maps_complete", "c3_no_unbacked_repoint", "c4_class_table")
    if (not isinstance(doc["checks"], dict) or set(doc["checks"]) != set(check_names)
            or any(not isinstance(doc["checks"][name], dict) or doc["checks"][name].get("pass") is not True
                   for name in check_names)):
        raise Refused("attestation-status-inconsistent")
    return Attestation(out, doc["phase"], shards, states, recorded_bindings, file_sha, content_sha, doc["agent"])


# r34's pinned index is ten-column TSV; each row pins a tree-phase text manifest. The producer also writes ONE consolidated union of its
# incremental indexes (25 Oct 18:51Z, TIMEFILES-INDEX-ALL-<tag>.tsv): a cell whose tree-phases were taken in different indexes needs
# one pin that covers all of them. The consolidated file is recognised by its name AND its exact layout (and refused when they
# disagree): line 1 the CONSOLIDATED marker comment, line 2 the twelve-column header comment, then one row per tree-phase (the
# ten single-index columns after a source index tag and that index's taken-at time), and the last line the one ``# sources:`` line
# listing the source indexes with their sha256. A tree-phase appears once; every source tag the rows cite is on the sources
# line and every sources-line tag is cited; a source tag has one taken-at time, on the tag's own UTC date. Each row's taken-at
# time is the time of the index that covered its tree-phase: the sha manifest label time of that tree-phase's shards.
INDEX_NAME = re.compile(r"TIMEFILES-INDEX-(\d{8}T\d{6}Z)\.tsv")
ALL_INDEX_NAME = re.compile(r"TIMEFILES-INDEX-ALL-([0-9]{8}T[0-9]{6}Z)\.tsv")
TIME_LINE = re.compile(r"([0-9a-f]{64}|MISSING)  (online|obqc)/logs/([A-Za-z0-9][A-Za-z0-9._-]{0,200})\.time")
INDEX_TAG = r"[0-9]{8}T[0-9]{6}Z"
ALL_INDEX_MARK = "# CONSOLIDATED union of the Tier 2 time-file sha manifest indexes ("
ALL_INDEX_HEADER = ("# source_index_tag\ttaken_at_utc\ttree\tphase\tshards\tonline_hashed\tonline_missing\tobqc_hashed\tobqc_missing\t"
                    "obqc_na\tmanifest_sha256\tstatus")
_SOURCE_ITEM = rf"TIMEFILES-INDEX-({INDEX_TAG})\.tsv sha256 ([0-9a-f]{{64}})"
ALL_SOURCES_LINE = re.compile(rf"# sources: ({_SOURCE_ITEM}(?:; {_SOURCE_ITEM})*)")
SOURCE_ITEM = re.compile(_SOURCE_ITEM)
# the header comment of a producer's single index, the one place its taken-at time is written
SOURCE_HEADER = re.compile(rf"# Tier 2 time-file sha manifests index, taken ({UTC}) \(.*\)", re.ASCII)


@dataclass(frozen=True)
class TimefilesManifest:
    files: dict  # "<tree>/logs/<shard>.time" -> sha256 or MISSING
    taken_at: str  # a single index: its tag. The consolidated index: the file's OWN tag, the consolidation time, never a label time
    file_sha256: str
    manifests: list  # tree, phase, file sha256; a consolidated index adds source_index (tag) and taken_at (that row's time)
    kind: str = "single"  # "single" or "consolidated"
    sources: tuple = ()  # consolidated: ({"tag", "file_sha256", "verified"}, ...), verified = the source index file was beside it and agrees


@dataclass(frozen=True)
class ConsolidatedIndex:
    rows: list  # (source tag, taken_at_utc, the ten single-index columns as a tuple), file order
    sources: dict  # source tag -> sha256 on the sources line
    taken: dict  # source tag -> its taken-at time


def check_index_row(parts: list, seen: set) -> None:
    """The ten single-index columns of one row, strictly; a tree-phase seen before, a bad name, phase, count, status or sha refuses.
    ``seen`` gets the row's (tree, phase)."""
    if len(parts) != 10:
        raise Refused("timefiles-manifest-malformed")
    tree, phase, shards, online, missing, obqc, obqc_missing, obqc_na, msha, status = parts
    if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,200}", tree) or phase not in REPLAY_PHASES
            or (tree, phase) in seen or any(not n.isdecimal() for n in (shards, online, missing, obqc, obqc_missing, obqc_na))
            or not (status in ("written", "exists-match", "exists-differs", "dry") or status.startswith("refused:"))
            or not (SHA_RE.fullmatch(msha) or msha == "-")):
        raise Refused("timefiles-manifest-malformed")
    seen.add((tree, phase))


def parse_consolidated_index(data: bytes) -> ConsolidatedIndex:
    """The consolidated union index from its bytes alone (a pure function: no file is opened), strictly; see the layout above."""
    try:
        text = data.decode("utf-8")
    except UnicodeError:
        raise Refused("timefiles-manifest-malformed") from None
    lines = text.split("\n")  # only "\n" ends a line here (splitlines would also split on other separators)
    if len(lines) < 5 or lines[-1] != "":
        raise Refused("timefiles-manifest-malformed")
    lines.pop()
    marker, header, rows, sources_line = lines[0], lines[1], lines[2:-1], lines[-1]
    sources_match = ALL_SOURCES_LINE.fullmatch(sources_line)
    if not (marker.startswith(ALL_INDEX_MARK) and marker.endswith(")") and header == ALL_INDEX_HEADER and sources_match
            and not any(r.startswith("#") for r in rows)):
        raise Refused("timefiles-manifest-malformed")
    sources: dict = {}
    for item in sources_match[1].split("; "):
        tag, sha = SOURCE_ITEM.fullmatch(item).groups()
        if tag in sources:
            raise Refused("timefiles-manifest-malformed")
        sources[tag] = sha
    parsed, seen, taken = [], set(), {}
    for row in rows:
        parts = row.split("\t")
        if len(parts) != 12:
            raise Refused("timefiles-manifest-malformed")
        tag, at = parts[0], parts[1]
        try:
            if not (re.fullmatch(INDEX_TAG, tag) and re.fullmatch(UTC, at, re.ASCII)):
                raise ValueError
            datetime.datetime.strptime(tag, "%Y%m%dT%H%M%SZ")
            utc_epoch(at)
        except ValueError:
            raise Refused("timefiles-manifest-malformed") from None
        if tag[:8] != at[:10].replace("-", "") or taken.setdefault(tag, at) != at:  # one time per index, on the tag's own date
            raise Refused("timefiles-manifest-malformed")
        check_index_row(parts[2:], seen)
        parsed.append((tag, at, tuple(parts[2:])))
    if set(sources) != set(taken):  # exactly the source tags the rows cite
        raise Refused("timefiles-manifest-malformed")
    return ConsolidatedIndex(parsed, sources, taken)


def verify_consolidated_sources(cons: ConsolidatedIndex, directory: Path) -> dict:
    """tag -> True (the source index ``TIMEFILES-INDEX-<tag>.tsv`` was beside the consolidated file and agrees with it) or False (it
    is not there, so nothing was compared). A source that IS there must have the sha256 on the sources line, a header whose taken-at
    time is its rows' time, and exactly its rows (the ten columns, as a multiset) among the consolidated rows of its tag; anything
    else refuses (``timefiles-manifest-source-differs``)."""
    verified = {}
    for tag, sha in cons.sources.items():
        path = directory / f"TIMEFILES-INDEX-{tag}.tsv"
        if not os.path.lexists(path):
            verified[tag] = False
            continue
        data = read_small(path, 1 << 26)
        try:
            lines = data.decode("utf-8").splitlines()
        except UnicodeError:
            raise Refused("timefiles-manifest-source-differs") from None
        header = SOURCE_HEADER.fullmatch(lines[0]) if lines else None
        have = sorted(tuple(line.split("\t")) for line in lines if not line.startswith("#"))
        want = sorted(row for t, _, row in cons.rows if t == tag)
        if hashlib.sha256(data).hexdigest() != sha or header is None or header[1] != cons.taken[tag] or have != want:
            raise Refused("timefiles-manifest-source-differs")
        verified[tag] = True
    return verified


def read_timefiles_manifest(spec: str, root: Path) -> TimefilesManifest:
    """Read the producer's pinned index (the ten-column single index, or the consolidated union index) and its tree-phase sha256sum
    manifests."""
    path_text, sep, pin = spec.rpartition(":")
    if not sep or not path_text or not SHA_RE.fullmatch(pin):
        raise Refused("timefiles-manifest-pin-malformed")
    data = read_small(Path(path_text), 1 << 26)
    file_sha = hashlib.sha256(data).hexdigest()
    if file_sha != pin:
        raise Refused("timefiles-manifest-sha-differs")
    match = INDEX_NAME.fullmatch(Path(path_text).name)
    match_all = ALL_INDEX_NAME.fullmatch(Path(path_text).name)
    if match is None and match_all is None:
        raise Refused("timefiles-manifest-malformed")
    sources: tuple = ()
    if match_all is not None:
        cons = parse_consolidated_index(data)
        verified = verify_consolidated_sources(cons, Path(path_text).parent)
        sources = tuple({"tag": t, "file_sha256": cons.sources[t], "verified": verified[t]} for t in sorted(cons.sources))
        rows = iter(cons.rows)
    else:
        try:
            lines = data.decode("utf-8").splitlines()
        except UnicodeError:
            raise Refused("timefiles-manifest-malformed") from None

        def single_rows(seen):
            for line in lines:
                if not line.startswith("#"):
                    parts = line.split("\t")
                    check_index_row(parts, seen)  # one row at a time: a bad row after a refused manifest is never reached
                    yield None, None, tuple(parts)
        rows = single_rows(set())
    files, manifests = {}, []
    for tag, taken, parts in rows:
        tree, phase, msha, status = parts[0], parts[1], parts[8], parts[9]
        if status not in ("written", "exists-match"):
            continue
        if not SHA_RE.fullmatch(msha):
            raise Refused("timefiles-manifest-malformed")
        rel = root / tree / f"TIMEFILES-{phase}.sha256"
        mdata = read_small(rel, 1 << 26)
        if hashlib.sha256(mdata).hexdigest() != msha:
            raise Refused("timefiles-manifest-sha-differs")
        manifests.append({"tree": tree, "phase": phase, "file_sha256": msha,
                          **({} if tag is None else {"source_index": tag, "taken_at": taken})})
        try:
            mlines = mdata.decode("utf-8").splitlines()
        except UnicodeError:
            raise Refused("timefiles-manifest-malformed") from None
        for mline in mlines:
            if mline.startswith("#"):
                continue
            found = TIME_LINE.fullmatch(mline)
            if found is None:
                raise Refused("timefiles-manifest-malformed")
            digest, pass_name, shard = found.groups()
            if pass_name == "obqc":
                continue
            key = f"{tree}/logs/{shard}.time"
            if key in files:
                raise Refused("timefiles-manifest-malformed")
            files[key] = digest
    if match_all is not None:
        return TimefilesManifest(files, match_all[1], file_sha, manifests, "consolidated", sources)
    return TimefilesManifest(files, match[1], file_sha, manifests)


BASES = ("proven", "recorded", "attested")


def build_routing(out: str, located: dict[str, list[Located]], root: Path, table: ClassTable, classes_sha: str,
                  manifest: TimefilesManifest, attestations: tuple[Attestation, ...] = ()) -> dict:
    """Class each located shard from a pinned first-line launch, then an ok not_recorded attestation row.

    Listed changed bytes and an unlisted present time file are unverified without fallback.
    A MISSING line or a matching but unusable file can fall back only to its own shard row.
    A producer-recorded row whose launch this consumer rejects cannot supply a fallback.
    """
    shards: dict[str, dict] = {}
    time_files: dict[str, str] = {}
    by_phase = {a.phase: a for a in attestations}
    for phase, locs in located.items():
        att = by_phase.get(phase)
        for loc in locs:
            entry = att.shards.get((loc.tree, loc.shard)) if att else None
            recorded_binding = att.recorded_bindings.get((loc.tree, loc.shard)) if att else None
            cell = CELL_TREE_RE.fullmatch(loc.tree)
            key = f"{loc.tree}/{phase}/{loc.shard}"
            key_in_manifest = f"{loc.tree}/logs/{loc.shard}.time"
            want = manifest.files.get(key_in_manifest)
            cls, basis, copy, subkind = UNVERIFIABLE, None, None, None
            if want is None:
                # A present file omitted by the producer's manifest is not an absence.
                if os.path.lexists(root / loc.tree / "logs" / f"{loc.shard}.time"):
                    subkind = "unlisted_time_file"
                else:
                    subkind = "no_launch_record"
            elif want == "MISSING":
                subkind = "no_launch_record"
            else:
                t = load_time_file(root, loc.tree, loc.shard)  # the bytes, once: compared with the manifest and parsed
                if t is None or t[2] != want:
                    cls = UNVERIFIABLE  # gone, not a plain file, or different bytes since the manifest was taken: no fallback
                else:
                    rec = parse_launch_record(t[0], t[1], loc.tree, loc.shard, phase, round(loc.sealed), loc.started)
                    if rec is not None:
                        got = shard_replay_class(rec, table)
                        if (got not in CLASSES or (cell and got != cell.group(2))
                                or (entry is not None and (rec[0], rec[1], got) != entry)
                                or (recorded_binding is not None and any(
                                    known is not None and known != observed
                                    for known, observed in zip(recorded_binding, (rec[0], rec[1], got))))):
                            cls = UNVERIFIABLE  # unbound by the table, a cell's copy of another class, or contradicted by the attestation
                        else:
                            cls, basis, copy = got, "proven" if cell else "recorded", {"endpoint": rec[0], "copy": rec[1]}
                            time_files[key] = want
                    else:
                        subkind = ("record_rejected_by_consumer" if att and att.states.get((loc.tree, loc.shard)) == "recorded"
                                   else "no_launch_record")
            if subkind == "no_launch_record" and entry is not None:
                cls, basis, copy = entry[2], "attested", {"endpoint": entry[0], "copy": entry[1]}
                subkind = None
            elif subkind == "no_launch_record" and att is None:
                subkind = "pending_attestation"
            shards[key] = {"class": cls, "basis": basis, "copy": copy, "runs": sorted(run_ids_of(loc.record_names))}
            if subkind is not None:
                shards[key]["subkind"] = subkind
    return {"schema": ROUTING_SCHEMA, "out": out, "shards": dict(sorted(shards.items())),
            "inputs": {"cell_classes_sha256": classes_sha, "time_files_sha256": dict(sorted(time_files.items())),
                       "timefiles_manifest": {"file_sha256": manifest.file_sha256, "taken_at": manifest.taken_at,
                                              "manifests": manifest.manifests,
                                              **({"sources": list(manifest.sources)} if manifest.kind == "consolidated" else {})},
                       "attestations": [{"out": a.out, "phase": a.phase, "file_sha256": a.file_sha256,
                                         "content_sha256": a.content_sha256, "agent": a.agent,
                                         "shards": [{"tree": t, "shard": s, "class": v[2], "copy": v[1], "endpoint": v[0]}
                                                    for (t, s), v in sorted(a.shards.items()) if v is not None]}
                                        for a in attestations]}}


# -- staging ------------------------------------------------------------------------------------------------------------------


def stage(located: list[Located], stage_dir: Path, out: str) -> list[Located]:
    """Copy each matrix into ``stage_dir`` (mode 600) while hashing it, require the runrec's sha256 of the copy, and return the
    located list pointing at the copies."""
    staged = []
    for loc in located:
        dest_dir = stage_dir / out / loc.phase / loc.tree
        dest_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        dest = dest_dir / f"{loc.shard}.matrix.jsonl.gz"
        h = hashlib.sha256()
        fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as w, open(loc.matrix, "rb") as r:
            for block in iter(lambda: r.read(1 << 20), b""):
                h.update(block)
                w.write(block)
        if h.hexdigest() != loc.sha256 or sha256_file(dest) != loc.sha256:
            raise Refused("matrix-sha-differs", tree=loc.tree, shard=loc.shard)
        staged.append(Located(loc.phase, loc.tree, loc.shard, dest, loc.sha256, loc.record_names, loc.sealed, loc.started))
    return staged


# -- command line -------------------------------------------------------------------------------------------------------------


def parse_phases(text: str) -> tuple[str, ...]:
    phases = tuple(text.split(","))
    if not phases or any(p not in REPLAY_PHASES for p in phases) or len(set(phases)) != len(phases):
        raise Refused("bad-phases")
    return phases


def do_locate(args: argparse.Namespace, out: str, phases: tuple[str, ...]) -> dict[str, list[Located]]:
    root, done = Path(args.root), Path(args.done_dir)
    cells, cells_sha = read_cells_tsv(Path(args.cells_tsv))
    print(f"CELLS-TSV sha256={cells_sha} lines={len(cells)}")
    located: dict[str, list[Located]] = {}
    for phase in phases:
        marker, layout, locs = locate_phase(root, done, cells, out, phase)
        located[phase] = locs
        print(f"LOCATED out={out} phase={phase} layout={layout} marker_sha256={marker.sha256} "
              f"trees={len({loc.tree for loc in locs})} shards={len(locs)} records={sum(len(loc.record_names) for loc in locs)}")
        for loc in locs:
            print(f"MATRIX out={out} phase={phase} tree={loc.tree} shard={loc.shard} sha256={loc.sha256} "
                  f"records={len(loc.record_names)}")
    return located


class QuietParser(argparse.ArgumentParser):
    """An argument error prints a fixed refusal, never the offending text (the output is names, counts and sha256s only)."""

    def error(self, message: str):
        print("REFUSED bad-arguments", file=sys.stderr)
        raise SystemExit(2)


def main(argv: list[str] | None = None) -> int:
    ap = QuietParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    lc = sub.add_parser("ledger-check", help="the exporter's ledger against the staged matrices")
    lc.add_argument("--stage", required=True, help="the stage directory of a successful check")
    lc.add_argument("--ledger", required=True, help="the ledger the exporter wrote from the staged matrices")
    for name in ("locate", "check"):
        s = sub.add_parser(name)
        s.add_argument("--root", required=True, help="the replay tree root (replay-tier2-20261023, or a staged copy)")
        s.add_argument("--done-dir", required=True, help="the done markers (hub done/, byte-copied)")
        s.add_argument("--cells-tsv", required=True, help="the hub's cells.tsv (byte-copied)")
        s.add_argument("--phases", required=True, help="natural, prompted or natural,prompted")
        if name == "locate":
            s.add_argument("--out-name", required=True, help="the OUT's name (its replay tree's name)")
        else:
            s.add_argument("--out-dir", required=True, help="the OUT directory the exporter gets with --records")
            s.add_argument("--stage", required=True, help="a new private directory for the checked copies")
            s.add_argument("--cell-classes", required=True, help="the replay tree's bin/cell-classes.tsv (a byte copy)")
            s.add_argument("--timefiles-manifest", required=True, metavar="PATH:SHA256",
                           help="tier2-replay's TIMEFILES-INDEX-<tag>.tsv, or its consolidated TIMEFILES-INDEX-ALL-<tag>.tsv, pinned by its sha256")
            s.add_argument("--attestation", action="append", default=[], metavar="PATH:SHA256",
                           help="the operator's per-shard t2-attest/1 of one OUT-phase, pinned by its "
                                "sha256; repeatable, at most one per phase; a fallback for shards with no recorded launch")
    args = ap.parse_args(argv)
    previous = os.umask(0o077)
    try:
        return run(args)
    finally:
        os.umask(previous)  # a test calls main() in-process


def run(args: argparse.Namespace) -> int:
    try:
        if args.cmd == "ledger-check":
            counts = ledger_check(Path(args.stage), Path(args.ledger))
            print("LEDGER-CHECK" + "".join(f" {k}={v}" for k, v in sorted(counts.items())) + " OK")
            return 0
        phases = parse_phases(args.phases)
        if args.cmd == "locate":
            do_locate(args, args.out_name, phases)
            return 0
        out_dir = Path(args.out_dir)
        if not out_dir.is_dir():  # a symlink to the OUT is fine (the runs tree links OUTs on /data); its basename is the tree's name
            raise Refused("out-dir-missing")
        out = out_dir.name
        # the phases asked for are exactly the replay phases the OUT has (its order files): a phase left out would be analysed
        # with no replay rows, and one the OUT does not have is a mistaken cell line
        present = tuple(p for p in REPLAY_PHASES if os.path.lexists(out_dir / f"order-{p}.json"))
        if set(present) != set(phases):
            raise Refused("phases-differ", requested=",".join(phases), present=",".join(present) or "none")
        stage_dir = Path(os.path.abspath(args.stage))  # absolute: matrices.list is read from another working directory
        if os.path.lexists(stage_dir):
            raise Refused("stage-exists")
        table, table_sha = read_cell_classes(Path(args.cell_classes))  # the class table and the attestations first: a refusal leaves no stage
        manifest = read_timefiles_manifest(args.timefiles_manifest, Path(args.root))
        if manifest.kind == "consolidated":  # counts only; says how many source indexes were beside the file and compared with it
            print(f"TIMEFILES-INDEX kind=consolidated sha256={manifest.file_sha256} tree_phases={len(manifest.manifests)} "
                  f"sources={len(manifest.sources)} sources_verified={sum(x['verified'] for x in manifest.sources)} "
                  f"sources_not_beside={sum(not x['verified'] for x in manifest.sources)}")
        atts = tuple(read_attestation(spec, out, phases, table) for spec in args.attestation)
        if len({a.phase for a in atts}) != len(atts):
            raise Refused("attestation-twice-for-a-phase")
        located = do_locate(args, out, phases)
        have_manifests = {(m["tree"], m["phase"]) for m in manifest.manifests}
        if any((loc.tree, phase) not in have_manifests for phase, locs in located.items() for loc in locs):
            raise Refused("timefiles-manifest-phase-missing")
        stage_dir.mkdir(parents=True, mode=0o700)
        staged = {p: stage(locs, stage_dir, out) for p, locs in located.items()}
        counts, outputs = coverage(out_dir, staged)
        order = [loc for p in phases for loc in staged[p]]
        lines = [str(loc.matrix) for loc in order]
        fd = os.open(stage_dir / "matrices.list", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write("".join(x + "\n" for x in lines))
        routing = build_routing(out, staged, Path(args.root), table, table_sha, manifest, atts)
        fd = os.open(stage_dir / "replay-routing.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(routing, sort_keys=True, indent=1) + "\n")
        fd = os.open(stage_dir / "replay-outputs.list", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:  # the run ids with an output: the ledger check's expectation (never printed)
            f.write("".join(rid + "\n" for rid in sorted(outputs)))
        manifest = "out\tphase\ttree\tshard\tsha256\trecords\n" + "".join(
            f"{out}\t{loc.phase}\t{loc.tree}\t{loc.shard}\t{loc.sha256}\t{len(loc.record_names)}\n" for loc in order)
        fd = os.open(stage_dir / "replay-matrices-manifest.tsv", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(manifest)
        by_tree: dict[str, Counter] = {}
        for key, info in routing["shards"].items():
            c = by_tree.setdefault(key.split("/", 1)[0], Counter())
            c[info["class"]] += 1
            c["attested"] += info["basis"] == "attested"
            c["pending_attestation"] += info.get("subkind") == "pending_attestation"
        for tree, n in sorted(by_tree.items()):
            kind = "cell" if CELL_TREE_RE.fullmatch(tree) else "main"
            print(f"ROUTING out={out} tree={tree} kind={kind} " + " ".join(f"{c}={n.get(c, 0)}" for c in (*CLASSES, UNVERIFIABLE, NOT_RECORDED))
                  + f" attested={n.get('attested', 0)} pending_attestation={n.get('pending_attestation', 0)}")
        print("COVERAGE out=" + out + "".join(f" {k}={v}" for k, v in sorted(counts.items())) +
              f" matrices={len(order)} manifest_sha256={hashlib.sha256(manifest.encode()).hexdigest()} OK")
        return 0
    except Refused as exc:
        print(f"REFUSED {exc}", file=sys.stderr)
        return 2
    except (OSError, EOFError, ValueError, KeyError, TypeError, AttributeError, zlib.error) as exc:  # a truncated gzip, a malformed
        # order file: the exception's class name only, never its text (which could quote a value)
        print(f"REFUSED io-error class={type(exc).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
