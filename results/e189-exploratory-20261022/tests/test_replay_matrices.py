"""Tests of driver/replay_matrices.py (the locator and the coverage check of the replay matrices a look hands the exporter).

Synthetic fixtures only: a replay tree, done markers, a cells table and a runner OUT are built here by hand, with literal
expectations. The fixture writer shares no code with the tool, and every expected count below is a number written by hand.

Hand-derived facts (computed once with hashlib; the claim digest is the first 40 hex of sha256 of the JSON of
[arm, question, route model, repetition], as bin/phase_status.py computes it):
    ["natural", "q1", "route-1", 1] -> 960bbd62f8c6395ab50736db2589ea3538565c0e   (q2: cb0fc965..., q3: 2ef635f7...)
    ["curated", "q1", "route-1", 1] -> ee24ccea46c23839ec84c3b5571596d5e547da8b   (q2: 61f4bf02..., q3: a017b643...)
    ["branch-curated", "q1", "route-1", 1] -> bd2116548e0bf2ddcb90322b8666d43d4f25aea3
A record has 8 arms x 4 variants = 32 matrix rows.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
DRIVER = HERE.parent / "driver" / "replay_matrices.py"
_spec = importlib.util.spec_from_file_location("replay_matrices", DRIVER)
rm = importlib.util.module_from_spec(_spec)
sys.modules["replay_matrices"] = rm
_spec.loader.exec_module(rm)

DIGEST = {
    ("natural", "q1"): "960bbd62f8c6395ab50736db2589ea3538565c0e",
    ("natural", "q2"): "cb0fc96507f9f3b410b14e938e32aff1eea20b3b",
    ("natural", "q3"): "2ef635f74041fdee8a1a0a5ccc4b5d3d340dd564",
    ("curated", "q1"): "ee24ccea46c23839ec84c3b5571596d5e547da8b",
    ("curated", "q2"): "61f4bf020689bd17d19e86b7724b26eab752f9f1",
    ("curated", "q3"): "a017b643563eba62164d3fddf7ae3cc06cdeb8ef",
    ("branch-curated", "q1"): "bd2116548e0bf2ddcb90322b8666d43d4f25aea3",
    ("branch-curated", "q2"): "133ff13a411d9860f6186923848203337bd8a567",
}
ARMS = ["curated", "census-t5", "census-t2", "census-t20", "generated-1", "generated-2", "generated-3", "union"]
VARIANTS = ["primary", "heuristic_off", "branch_local_off", "both_off"]
MARKER_UTC = "2026-10-24T16:00:00Z"
SEALED_UTC = "2026-10-24T15:00:00Z"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rec_name(arm: str, q: str, rid: str) -> str:
    return f"{DIGEST[(arm, q)]}.{rid}.json"


def matrix_bytes(run_ids, *, drop=(), dup=(), arms=ARMS, extra_arm=None, version="checker-replay-matrix/1", split="test",
                 no_output=(), omit_output=False, flip_output=(), unlabelled=()) -> bytes:
    """A shard's matrix by hand: 8 arms x 4 variants per run id, each row saying whether the run has an output (all do, except
    ``no_output``) and its scorer label; the keyword options spoil one thing each."""
    rows = []
    for rid in run_ids:
        has = rid not in no_output
        for arm in arms:
            for v in VARIANTS:
                if (rid, arm, v) in drop:
                    continue
                row = {"version": version, "run_id": rid, "arm": arm, "variant": v, "split": split, "status": "flagged",
                       "output": has, "correct": True if has else None}
                if omit_output:
                    del row["output"]
                if (rid, arm, v) in flip_output:
                    row["output"] = not has
                if rid in unlabelled and has:
                    del row["correct"]
                rows.append(row)
                if (rid, arm, v) in dup:
                    rows.append(dict(row))
        if extra_arm:
            rows.append({"version": version, "run_id": rid, "arm": extra_arm, "variant": "primary", "split": split,
                         "output": has, "correct": True if has else None})
    return gzip.compress("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode(), mtime=0)


def launch_command(tree, shard, phase, endpoint, copy, *, note_copy=None) -> str:
    """The command t2_worker.sh run_online launches, as GNU time prints it (arguments joined by spaces): the frozen checker with
    the worker's constant sources and union, the endpoint and the note naming the phase, tree, shard and copy id."""
    return (f"/home/u/www-2027/repo-freeze-126a111/.venv/bin/python scripts/checker_replay.py --records /r/{tree}/shards-{phase}/{shard} "
            f"--sources-dir data/sources/x --sources curated,census-t5,census-t2,census-t20,generated-1,generated-2,generated-3 "
            f"--union union=curated,census-t5 --hierarchy /h/shared-hierarchy-v4.json.gz --endpoint http://{endpoint}/ --split test "
            f"--note Tier 2 {phase} replay, freeze 126a111, {tree} shard {shard}, {note_copy or copy}, run-node --out /r/{tree}/out/{shard}")


class World:
    """A runner OUT (names, order files, served models) and a replay root (trees, markers, cells table), by hand."""

    def __init__(self, tmp: Path, out: str = "opaque-sonnet") -> None:
        self.out_name = out
        self.out = tmp / "runs" / out
        self.root = tmp / "replay"
        self.done = tmp / "done"
        self.cells = tmp / "cells.tsv"
        self.stage = tmp / "stage"
        for d in (self.out / "records", self.root, self.done):
            d.mkdir(parents=True)
        self.cells.write_text("# REAL OUT\tphase\tclass\tCELL\trecords in the cell\trecords in the phase\n")
        self.cell_classes = tmp / "cell-classes.tsv"
        self.cell_classes.write_text(
            "# endpoint\tcopy id\tspeed class\tnote\n"
            "192.0.2.105:7039\texample-copy-38\tfast\tx\n192.0.2.105:7043\texample-copy-39\tfast\tx\n127.0.0.1:47125\texample-copy-40\tslow\tx\n"
            "192.0.2.101:7033\texample-copy-12\tfast\tx\n127.0.0.1:47278\texample-copy-33\tslow\tx\n"
            "192.0.2.101:7041\texample-copy-9\tfast\tx\n")
        (self.out / "served-models.json").write_text(json.dumps({"models": {"m1": {"route_model": "route-1"}}}))
        self.order: dict[str, list] = {}
        self.runrecs: dict[tuple, dict] = {}

    # the OUT
    def add_records(self, phase: str, items) -> None:
        """items: [(arm, question, run id)] -> the OUT's record files and the phase's order keys."""
        for arm, q, rid in items:
            (self.out / "records" / rec_name(arm, q, rid)).write_text("{}")
            self.order.setdefault(phase, []).append(["m1", arm, q, 1])
        (self.out / f"order-{phase}.json").write_text(json.dumps({"order": self.order[phase]}))

    # the replay trees
    def tree(self, name: str) -> Path:
        t = self.root / name
        t.mkdir(exist_ok=True)
        return t

    def add_shard(self, tree: str, phase: str, shard: str, items, *, rows: bytes | None = None, sealed: str = SEALED_UTC,
                  launch=None) -> None:
        """items: [(arm, question, run id)]: the shard's record names (symlinks into the OUT) and its matrix. ``launch`` is
        (host:port, copy id) of the copy its online pass ran on, written into its time file ("none": no time file)."""
        t = self.tree(tree)
        rdir = t / f"shards-{phase}" / shard / "records"
        rdir.mkdir(parents=True)
        for arm, q, rid in items:
            os.symlink(self.out / "records" / rec_name(arm, q, rid), rdir / rec_name(arm, q, rid))
        with open(t / f"shards-{phase}.tsv", "a") as f:
            f.write(f"{shard}\t{tree}\tm1\t{len(items)}\n")
        data = rows if rows is not None else matrix_bytes([rid for _, _, rid in items])
        self.set_started(tree, shard, rm.utc_epoch(sealed) - 1200)  # the worker's claim dir: dated at the claim
        launch = launch or self.default_launch(tree)
        if launch != "none":
            self.write_time(tree, shard, phase, *launch, mtime=rm.utc_epoch(sealed) - 5)  # written when the checker ended
        (t / "out" / shard).mkdir(parents=True)
        (t / "out" / shard / "matrix.jsonl.gz").write_bytes(data)
        self.seal(tree, shard, data, sealed)

    def default_launch(self, tree: str):
        if tree.endswith("-cell-slow"):
            return ("127.0.0.1:47125", "example-copy-40")
        return ("192.0.2.101:7033", "example-copy-12") if tree.startswith("opaque") else ("192.0.2.105:7043", "example-copy-39")

    def write_time(self, tree: str, shard: str, phase: str, endpoint: str, copy: str, mtime: float | None = None, *,
                   command: str | None = None, exit_status: int = 0) -> None:
        """The shard's `<tree>/logs/<shard>.time` as GNU time -v writes it (the first line is the command as launched by
        t2_worker.sh run_online: format checked against /usr/bin/time on the spoke), the rest resource lines."""
        logs = self.root / tree / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        cmd = command or launch_command(tree, shard, phase, endpoint, copy)
        path = logs / f"{shard}.time"
        path.write_text(f'\tCommand being timed: "{cmd}"\n\tUser time (seconds): 1.00\n\tElapsed (wall clock) time (h:mm:ss or m:ss): 0:02.00\n'
                        f"\tExit status: {exit_status}\n")
        if mtime is not None:
            os.utime(path, (mtime, mtime))

    def set_started(self, tree: str, shard: str, epoch: float) -> None:
        """The worker's claim directory: its `worker` file, written at the claim, is dated ``epoch``; the directory itself and
        its `pid` file (written at the launch) are dated by the caller if the test wants them later."""
        claim = self.root / "claims" / f"{tree}.{shard}.online"
        claim.mkdir(parents=True, exist_ok=True)
        (claim / "worker").write_text("1234 5678\n")
        os.utime(claim / "worker", (epoch, epoch))

    def seal(self, tree: str, shard: str, data: bytes, sealed: str = SEALED_UTC, rc: int = 0) -> None:
        t = self.root / tree
        (t / "runrec").mkdir(exist_ok=True)
        (t / "runrec" / f"{shard}.json").write_text(json.dumps(
            {"shard": shard, "host": "run-node", "rc": rc, "sealed_utc": sealed, "matrix.jsonl.gz": sha(data),
             "store-log.jsonl.gz": "0" * 64, "sealed": {"accounting.json": "1" * 64, "summary.md": "2" * 64}}))

    def rewrite_matrix(self, tree: str, shard: str, data: bytes, *, reseal: bool = True) -> None:
        (self.root / tree / "out" / shard / "matrix.jsonl.gz").write_bytes(data)
        if reseal:
            self.seal(tree, shard, data)

    def mark(self, phase: str, line: str) -> None:
        (self.done / f"{self.out_name}-{phase}").write_text(line + "\n")

    def make_manifest(self, name="TIMEFILES-INDEX-20261025T123234Z.tsv", *, drop=(), override=None, taken_at="2026-10-25T12:32:34Z",
                      **over) -> str:
        """Synthetic r34 text manifests and their real ten-column pinned index."""
        files = {}
        for logs in sorted(self.root.glob("*/logs")):
            for f in sorted(logs.glob("*.time")):
                if f.is_file() and not f.is_symlink():
                    files[f"{logs.parent.name}/logs/{f.name}"] = sha(f.read_bytes())
        files = {k: v for k, v in files.items() if k not in drop}
        for k, v in (override or {}).items():
            files[k] = v
        index = []
        for tree in sorted(p.name for p in self.root.iterdir()
                           if p.is_dir() and any((p / f"shards-{phase}.tsv").exists() for phase in ("natural", "prompted"))):
            for phase in ("natural", "prompted"):
                shard_list = self.root / tree / f"shards-{phase}.tsv"
                if not shard_list.exists():
                    continue
                shards = [line.split("\t")[0] for line in shard_list.read_text().splitlines()]
                entries = []
                for shard in shards:
                    key = f"{tree}/logs/{shard}.time"
                    if key in drop:
                        continue
                    entries.append(f"{files[key] if key in files else 'MISSING'}  online/logs/{shard}.time")
                body = f"# Tier 2 time-file sha256 manifest (HASH ONLY) of {tree} {phase}, taken {taken_at} AFTER THE FACT\n"
                body += "".join(line + "\n" for line in sorted(entries))
                path = self.root / tree / f"TIMEFILES-{phase}.sha256"
                path.write_text(body)
                have = sum(not line.startswith("MISSING") for line in entries)
                index.append(f"{tree}\t{phase}\t{len(shards)}\t{have}\t{len(entries)-have}\t0\t0\t1\t{sha(path.read_bytes())}\twritten")
        path = self.stage.parent / name
        path.write_text("# tree\tphase\tshards\tonline_hashed\tonline_missing\tobqc_hashed\tobqc_missing\tobqc_na\tmanifest_sha256\tstatus\n"
                        + "".join(row + "\n" for row in index))
        return f"{path}:{sha(path.read_bytes())}"

    def routing_args(self) -> list:
        return ["--cell-classes", str(self.cell_classes)]

    def run(self, phases="natural,prompted", capsys=None, extra=()):
        extra = tuple(extra)
        if "--timefiles-manifest" not in extra:
            extra += ("--timefiles-manifest", self.make_manifest())
        rc = rm.main(["check", "--root", str(self.root), "--done-dir", str(self.done), "--cells-tsv", str(self.cells),
                      "--out-dir", str(self.out), "--phases", phases, "--stage", str(self.stage), *self.routing_args(), *extra])
        out, err = capsys.readouterr()
        # every output is counts, names and sha256s: never a run id or a claim digest
        assert "rid-" not in out + err
        assert not re.search(r"[0-9a-f]{40}\.", out + err)
        return rc, out, err


def good_world(tmp: Path) -> World:
    """opaque-sonnet whole: natural n1 n2 | n3 in two shards, prompted p1 p2 in one, one branch record; markers for both."""
    w = World(tmp)
    nat = [("natural", "q1", "rid-n1"), ("natural", "q2", "rid-n2"), ("natural", "q3", "rid-n3")]
    pro = [("curated", "q1", "rid-p1"), ("curated", "q2", "rid-p2")]
    w.add_records("natural", nat)
    w.add_records("prompted", pro)
    w.add_records("branch", [("branch-curated", "q1", "rid-b1")])
    w.add_shard("opaque-sonnet", "natural", "m1-natural-01of02", nat[:2])
    w.add_shard("opaque-sonnet", "natural", "m1-natural-02of02", nat[2:], rows=matrix_bytes(["rid-n3"], no_output={"rid-n3"}))
    w.add_shard("opaque-sonnet", "prompted", "m1-prompted-01of01", pro)
    w.mark("natural", f"OK shards 2 online 2 obqc 2 errors 0 {MARKER_UTC}")
    w.mark("prompted", f"OK shards 1 online 1 obqc 1 errors accepted {MARKER_UTC}")
    return w


def split_world(tmp: Path) -> World:
    """rhea-gemma-4-31b: natural whole in the main tree (n1 n2), prompted split into cells fast (p1 p2) and slow (p3); both
    cells call their one shard m1-prompted-01of01 (the same name in two trees)."""
    w = World(tmp, "rhea-gemma-4-31b")
    nat = [("natural", "q1", "rid-n1"), ("natural", "q2", "rid-n2")]
    pro = [("curated", "q1", "rid-p1"), ("curated", "q2", "rid-p2"), ("curated", "q3", "rid-p3")]
    w.add_records("natural", nat)
    w.add_records("prompted", pro)
    w.add_shard("rhea-gemma-4-31b", "natural", "m1-natural-01of01", nat)
    w.add_shard("rhea-gemma-4-31b-cell-fast", "prompted", "m1-prompted-01of01", pro[:2])
    w.add_shard("rhea-gemma-4-31b-cell-slow", "prompted", "m1-prompted-01of01", pro[2:])
    with open(w.cells, "a") as f:
        f.write("rhea-gemma-4-31b\tprompted\tfast\trhea-gemma-4-31b-cell-fast\t2\t3\n")
        f.write("rhea-gemma-4-31b\tprompted\tslow\trhea-gemma-4-31b-cell-slow\t1\t3\n")
    w.mark("natural", f"OK shards 1 online 1 obqc 1 errors 0 {MARKER_UTC}")
    w.mark("prompted", f"OK cells 2 shards 2 online 2 obqc 2 errors 0 {MARKER_UTC}")
    return w


# -- the constants ------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("out", [
    "rhea-h6-claude-opus-5-5", "opaque-h6-claude-opus-5-5", "uniprot-h6-claude-opus-5-5",
    "sib-rhea-sonnet", "sib-sonnet-32k", "sib-rhea-qwen3.8-27b-81920", "sib-qwen3.8-27b-81920",
    "sib-sonnet-32k-cap128k", "sib-rhea-sonnet-cap128k",
    *(f"sib-rhea-qwen3.8-27b-81920-clamp-b{i:02d}" for i in range(1, 13)),
])
def test_done_writer_out_names_pass_both_name_checks(tmp_path, out):
    # Exact plain OUT names from t2_done.py SIB_OUTS/H6_OUTS; no done marker is staged here.
    assert rm.OUT_RE.fullmatch(out)
    with pytest.raises(rm.Refused, match="no-done-marker"):
        rm.locate_phase(tmp_path, tmp_path, [], out, "natural")
    cells = tmp_path / "cells.tsv"
    cells.write_text(f"{out}\tnatural\tfast\t{out}-cell-fast\t1\t1\n")
    assert rm.read_cells_tsv(cells)[0][0][0] == out


@pytest.mark.parametrize("out", [
    "rhea-h6-claude-opus-5-4", "opaque-h6-claude-opus-5-5-2", "corporate-h6-claude-opus-5-5",
    "sib-h6-claude-opus-5-5", "sib-rhea-haiku", "sib-qwen3.8-27b-81921",
    "sib-rhea-qwen3.8-27b-81920-clamp-b00", "sib-rhea-qwen3.8-27b-81920-clamp-b13",
    "sib-rhea-qwen3.8-27b-81920-clamp-b1", "sib-sonnet-32k-cap128k-2",
])
def test_near_miss_done_writer_names_are_refused(tmp_path, out):
    assert rm.OUT_RE.fullmatch(out) is None
    with pytest.raises(rm.Refused, match="bad-name"):
        rm.locate_phase(tmp_path, tmp_path, [], out, "natural")
    cells = tmp_path / "cells.tsv"
    cells.write_text(f"{out}\tnatural\tfast\t{out}-cell-fast\t1\t1\n")
    with pytest.raises(rm.Refused, match="cells-tsv-malformed"):
        rm.read_cells_tsv(cells)


def test_the_pinned_arms_and_variants_are_the_workers_and_the_matrixs():
    # t2_worker.sh: SOURCES=curated,census-t5,census-t2,census-t20,generated-1,generated-2,generated-3, UNION=union=curated,census-t5
    assert list(rm.ARMS) == ARMS
    assert list(rm.VARIANTS) == VARIANTS
    try:
        from harness.checkloop.matrix import VARIANTS as FROZEN  # the frozen matrix's own variant names
    except ImportError:  # pragma: no cover - the frozen code is on the path in the suite
        pytest.skip("harness not importable")
    assert list(FROZEN) == VARIANTS


def test_the_claim_digest_is_the_phase_status_rule():
    for (arm, q), digest in DIGEST.items():
        assert rm.claim_digest(arm, q, "route-1", 1) == digest
    assert rm.claim_digest("natural", "q1", "route-2", 1) == "c530310c4c7ec707e3814ed68b785635c0d07f89"


# -- the good cases -----------------------------------------------------------------------------------------------------------


def test_a_whole_out_with_both_phases_passes_and_stages_the_online_matrices_only(tmp_path, capsys):
    w = good_world(tmp_path)
    # the OBQC pass's matrix of the same shard (same arms again): present in the tree, never located
    obqc = w.root / "opaque-sonnet" / "obqc" / "SEALED-obqc" / "m1-natural-01of02"
    obqc.mkdir(parents=True)
    (obqc / "matrix.jsonl.gz").write_bytes(matrix_bytes(["rid-n1", "rid-n2"], extra_arm="obqc"))
    # a superseded first run of a shard with a matrix of other content
    sup = w.root / "opaque-sonnet" / "superseded" / "first-run" / "online" / "out" / "m1-natural-02of02"
    sup.mkdir(parents=True)
    (sup / "matrix.jsonl.gz").write_bytes(matrix_bytes(["rid-n3", "rid-gone"]))
    rc, out, err = w.run(capsys=capsys)
    assert rc == 0 and err == ""
    lines = out.splitlines()
    marker_nat = f"OK shards 2 online 2 obqc 2 errors 0 {MARKER_UTC}\n".encode()
    marker_pro = f"OK shards 1 online 1 obqc 1 errors accepted {MARKER_UTC}\n".encode()
    cells_bytes = w.cells.read_bytes()
    assert lines[0] == f"CELLS-TSV sha256={sha(cells_bytes)} lines=0"
    assert lines[1] == (f"LOCATED out=opaque-sonnet phase=natural layout=main marker_sha256={sha(marker_nat)} "
                        f"trees=1 shards=2 records=3")
    m1 = matrix_bytes(["rid-n1", "rid-n2"])
    m2 = matrix_bytes(["rid-n3"], no_output={"rid-n3"})  # rid-n3 is an attempt with no output
    m3 = matrix_bytes(["rid-p1", "rid-p2"])
    assert lines[2] == f"MATRIX out=opaque-sonnet phase=natural tree=opaque-sonnet shard=m1-natural-01of02 sha256={sha(m1)} records=2"
    assert lines[3] == f"MATRIX out=opaque-sonnet phase=natural tree=opaque-sonnet shard=m1-natural-02of02 sha256={sha(m2)} records=1"
    assert lines[4] == (f"LOCATED out=opaque-sonnet phase=prompted layout=main marker_sha256={sha(marker_pro)} "
                        f"trees=1 shards=1 records=2")
    assert lines[5] == f"MATRIX out=opaque-sonnet phase=prompted tree=opaque-sonnet shard=m1-prompted-01of01 sha256={sha(m3)} records=2"
    manifest = ("out\tphase\ttree\tshard\tsha256\trecords\n"
                f"opaque-sonnet\tnatural\topaque-sonnet\tm1-natural-01of02\t{sha(m1)}\t2\n"
                f"opaque-sonnet\tnatural\topaque-sonnet\tm1-natural-02of02\t{sha(m2)}\t1\n"
                f"opaque-sonnet\tprompted\topaque-sonnet\tm1-prompted-01of01\t{sha(m3)}\t2\n")
    assert lines[6] == "ROUTING out=opaque-sonnet tree=opaque-sonnet kind=main fast=3 medium=0 slow=0 unverifiable=0 not_recorded=0 attested=0 pending_attestation=0"
    assert lines[7] == ("COVERAGE out=opaque-sonnet order_keys_natural=3 order_keys_prompted=2 "
                        "records_natural=3 records_prompted=2 rows_natural=96 rows_prompted=64 shards_natural=2 shards_prompted=1 "
                        f"matrices=3 manifest_sha256={sha(manifest.encode())} OK")
    assert len(lines) == 8
    # the stage: copies (mode 600) of the three online matrices, listed by absolute path in that order
    st = w.stage
    listed = (st / "matrices.list").read_text().splitlines()
    assert listed == [str(st / "opaque-sonnet/natural/opaque-sonnet/m1-natural-01of02.matrix.jsonl.gz"),
                      str(st / "opaque-sonnet/natural/opaque-sonnet/m1-natural-02of02.matrix.jsonl.gz"),
                      str(st / "opaque-sonnet/prompted/opaque-sonnet/m1-prompted-01of01.matrix.jsonl.gz")]
    assert [Path(p).read_bytes() for p in listed] == [m1, m2, m3]
    assert (st / "replay-matrices-manifest.tsv").read_text() == manifest
    assert (st / "replay-outputs.list").read_text().split() == ["rid-n1", "rid-n2", "rid-p1", "rid-p2"]  # rid-n3 has no output
    assert all(oct(Path(p).stat().st_mode & 0o777) == "0o600" for p in listed)
    assert oct(st.stat().st_mode & 0o777) == "0o700"
    assert not any("obqc" in p or "superseded" in p for p in listed)


def test_a_split_phase_is_located_in_its_cells_and_not_in_the_main_tree(tmp_path, capsys):
    w = split_world(tmp_path)
    rc, out, err = w.run(capsys=capsys)
    assert rc == 0 and err == ""
    lines = out.splitlines()
    cells_bytes = w.cells.read_bytes()
    assert lines[0] == f"CELLS-TSV sha256={sha(cells_bytes)} lines=2"
    marker = f"OK cells 2 shards 2 online 2 obqc 2 errors 0 {MARKER_UTC}\n".encode()
    assert lines[3] == (f"LOCATED out=rhea-gemma-4-31b phase=prompted layout=cells marker_sha256={sha(marker)} "
                        f"trees=2 shards=2 records=3")
    fast, slow = matrix_bytes(["rid-p1", "rid-p2"]), matrix_bytes(["rid-p3"])
    assert lines[4] == (f"MATRIX out=rhea-gemma-4-31b phase=prompted tree=rhea-gemma-4-31b-cell-fast shard=m1-prompted-01of01 "
                        f"sha256={sha(fast)} records=2")
    assert lines[5] == (f"MATRIX out=rhea-gemma-4-31b phase=prompted tree=rhea-gemma-4-31b-cell-slow shard=m1-prompted-01of01 "
                        f"sha256={sha(slow)} records=1")
    listed = (w.stage / "matrices.list").read_text().splitlines()
    assert [Path(p).name for p in listed] == ["m1-natural-01of01.matrix.jsonl.gz"] + 2 * ["m1-prompted-01of01.matrix.jsonl.gz"]
    # two cells with the same shard name keep separate copies, one directory per tree
    assert listed[1] != listed[2] and "cell-fast" in listed[1] and "cell-slow" in listed[2]
    assert "records_prompted=3" in lines[-1] and "rows_prompted=96" in lines[-1] and "records_natural=2" in lines[-1]


def natural_only_world(tmp: Path) -> World:
    """An OUT that holds the natural phase alone (as opaque-glm-5.3-flash's does): no order-prompted.json, no prompted tree."""
    w = World(tmp)
    nat = [("natural", "q1", "rid-n1"), ("natural", "q2", "rid-n2")]
    w.add_records("natural", nat)
    w.add_shard("opaque-sonnet", "natural", "m1-natural-01of01", nat)
    w.mark("natural", f"OK shards 1 online 1 obqc 1 errors 0 {MARKER_UTC}")
    return w


def test_an_out_with_the_natural_phase_alone_is_checked_for_it_alone(tmp_path, capsys):
    w = natural_only_world(tmp_path)
    rc, out, err = w.run("natural", capsys)
    assert rc == 0 and err == "" and out.splitlines()[-1].startswith("COVERAGE out=opaque-sonnet order_keys_natural=2 "
                                                                     "records_natural=2 rows_natural=64 shards_natural=1 matrices=1 ")


def test_a_phase_the_out_has_cannot_be_left_out_of_the_phases(tmp_path, capsys):
    w = good_world(tmp_path)  # the OUT has order-natural.json and order-prompted.json
    rc, out, err = w.run("natural", capsys)
    refused(rc, err, "phases-differ", "requested=natural", "present=natural,prompted")
    assert not w.stage.exists()
    rc, out, err = w.run("prompted", capsys)
    refused(rc, err, "phases-differ", "requested=prompted", "present=natural,prompted")


def test_a_phase_the_out_does_not_have_cannot_be_asked_for(tmp_path, capsys):
    w = natural_only_world(tmp_path)
    w.mark("prompted", f"OK shards 1 online 1 obqc 1 errors 0 {MARKER_UTC}")  # even with a marker lying about it
    rc, out, err = w.run("natural,prompted", capsys)
    refused(rc, err, "phases-differ", "requested=natural,prompted", "present=natural")


def test_a_bad_phases_argument_is_refused_without_an_echo(tmp_path, capsys):
    w = good_world(tmp_path)
    for text in ("branch", "natural,natural", "", "natural;rm"):
        rc, out, err = w.run(text, capsys)
        assert rc == 2 and err == "REFUSED bad-phases\n", text


def test_locate_alone_prints_the_files_and_stages_nothing(tmp_path, capsys):
    w = good_world(tmp_path)
    rc = rm.main(["locate", "--root", str(w.root), "--done-dir", str(w.done), "--cells-tsv", str(w.cells),
                  "--out-name", "opaque-sonnet", "--phases", "natural"])
    out, err = capsys.readouterr()
    assert rc == 0 and out.count("MATRIX ") == 2 and not w.stage.exists()


@pytest.mark.parametrize("out", ["rhea-h6-claude-opus-5-5", "opaque-h6-claude-opus-5-5", "uniprot-h6-claude-opus-5-5"])
def test_h6_natural_out_is_located_with_its_done_marker(tmp_path, capsys, out):
    w = World(tmp_path, out)
    items = [("natural", "q1", "rid-n1")]
    w.add_records("natural", items)
    w.add_shard(out, "natural", "m1-natural-01of01", items)
    w.mark("natural", f"OK shards 1 online 1 obqc 1 errors 0 {MARKER_UTC}")
    rc = rm.main(["locate", "--root", str(w.root), "--done-dir", str(w.done), "--cells-tsv", str(w.cells),
                  "--out-name", out, "--phases", "natural"])
    output, err = capsys.readouterr()
    assert rc == 0 and err == ""
    assert f"LOCATED out={out} phase=natural layout=main" in output
    assert output.count("MATRIX ") == 1


# -- the refusals: locator ----------------------------------------------------------------------------------------------------


def refused(rc, err, code, *counts):
    assert rc == 2, err
    assert err.startswith(f"REFUSED {code}"), err
    for c in counts:
        assert c in err.split(), err
    assert err.count("\n") == 1  # one line


def test_a_phase_without_a_done_marker_is_refused(tmp_path, capsys):
    w = good_world(tmp_path)
    (w.done / "opaque-sonnet-prompted").unlink()
    rc, out, err = w.run(capsys=capsys)
    refused(rc, err, "no-done-marker", "out=opaque-sonnet", "phase=prompted")


@pytest.mark.parametrize("line", [
    "OK shards 1 online 1 obqc 0 errors 0 2026-10-24T16:00:00Z",      # a pass missing
    "OK shards 1 online 1 obqc 1 errors 3 2026-10-24T16:00:00Z",      # an errors state the rule does not know
    "OK shards 1 online 1 obqc 1 errors 0",                          # no time
    "NOT-READY online-incomplete",
])
def test_a_malformed_marker_is_refused(tmp_path, capsys, line):
    w = good_world(tmp_path)
    w.mark("prompted", line)
    rc, out, err = w.run(capsys=capsys)
    assert rc == 2 and err.split()[1] in ("marker-format", "marker-counts"), err


def test_a_marker_whose_shard_count_is_not_the_layouts_is_refused(tmp_path, capsys):
    w = good_world(tmp_path)
    w.mark("natural", f"OK shards 3 online 3 obqc 3 errors 0 {MARKER_UTC}")
    rc, out, err = w.run(capsys=capsys)
    refused(rc, err, "marker-shards", "marker=3", "found=2")


def test_a_phase_in_the_main_tree_and_in_cells_is_refused(tmp_path, capsys):
    w = split_world(tmp_path)
    # a leftover shard list of the split phase in the main tree
    (w.root / "rhea-gemma-4-31b" / "shards-prompted.tsv").write_text("x\trhea-gemma-4-31b\tm1\t1\n")
    rc, out, err = w.run(capsys=capsys)
    refused(rc, err, "layout-both", "out=rhea-gemma-4-31b", "phase=prompted")


def test_a_phase_in_neither_the_main_tree_nor_cells_is_refused(tmp_path, capsys):
    w = good_world(tmp_path)
    (w.root / "opaque-sonnet" / "shards-prompted.tsv").unlink()
    rc, out, err = w.run(capsys=capsys)
    refused(rc, err, "layout-neither")


def test_a_cell_tree_the_table_does_not_list_is_refused(tmp_path, capsys):
    w = split_world(tmp_path)
    lines = w.cells.read_text().splitlines(keepends=True)
    w.cells.write_text("".join(lines[:2]))  # the slow cell's line is gone, its tree is not
    rc, out, err = w.run(capsys=capsys)
    refused(rc, err, "cells-differ", "cells_tsv=1", "trees=2")


def test_a_cells_table_whose_counts_are_not_the_trees_is_refused(tmp_path, capsys):
    w = split_world(tmp_path)
    text = w.cells.read_text().replace("cell-slow\t1\t3", "cell-slow\t2\t3")
    w.cells.write_text(text)
    rc, out, err = w.run(capsys=capsys)
    refused(rc, err, "cells-records", "found=3", "total=3")


def test_a_marker_naming_the_wrong_number_of_cells_is_refused(tmp_path, capsys):
    w = split_world(tmp_path)
    w.mark("prompted", f"OK cells 3 shards 2 online 2 obqc 2 errors 0 {MARKER_UTC}")
    rc, out, err = w.run(capsys=capsys)
    refused(rc, err, "cells-differ")


def test_a_malformed_cells_table_is_refused(tmp_path, capsys):
    w = split_world(tmp_path)
    with open(w.cells, "a") as f:
        f.write("rhea-gemma-4-31b\tprompted\tfast\twrong-name\t2\t3\n")
    rc, out, err = w.run(capsys=capsys)
    refused(rc, err, "cells-tsv-malformed")


def test_a_matrix_sealed_after_the_marker_is_refused(tmp_path, capsys):
    w = good_world(tmp_path)
    w.seal("opaque-sonnet", "m1-natural-02of02", matrix_bytes(["rid-n3"]), sealed="2026-10-24T16:00:01Z")
    rc, out, err = w.run(capsys=capsys)
    refused(rc, err, "sealed-after-marker", "tree=opaque-sonnet", "shard=m1-natural-02of02")


def test_a_runrec_with_a_failed_rc_or_another_shard_is_refused(tmp_path, capsys):
    w = good_world(tmp_path)
    w.seal("opaque-sonnet", "m1-natural-02of02", matrix_bytes(["rid-n3"]), rc=1)
    rc, out, err = w.run(capsys=capsys)
    refused(rc, err, "runrec-differs", "shard=m1-natural-02of02")


def test_a_missing_runrec_is_refused(tmp_path, capsys):
    w = good_world(tmp_path)
    (w.root / "opaque-sonnet" / "runrec" / "m1-natural-02of02.json").unlink()
    rc, out, err = w.run(capsys=capsys)
    assert rc == 2 and err.startswith("REFUSED unreadable"), err


def test_a_matrix_changed_after_the_seal_is_refused(tmp_path, capsys):
    w = good_world(tmp_path)
    w.rewrite_matrix("opaque-sonnet", "m1-natural-01of02", matrix_bytes(["rid-n1", "rid-n2"], extra_arm="x"), reseal=False)
    rc, out, err = w.run(capsys=capsys)
    refused(rc, err, "matrix-sha-differs", "shard=m1-natural-01of02")


def test_a_matrix_swapped_for_a_symlink_into_superseded_is_refused(tmp_path, capsys):
    w = good_world(tmp_path)
    t = w.root / "opaque-sonnet"
    good = (t / "out" / "m1-natural-02of02" / "matrix.jsonl.gz").read_bytes()
    sup = t / "superseded" / "first-run" / "out" / "m1-natural-02of02"
    sup.mkdir(parents=True)
    (sup / "matrix.jsonl.gz").write_bytes(good)  # the very bytes the runrec sealed: only its location is wrong
    (t / "out" / "m1-natural-02of02" / "matrix.jsonl.gz").unlink()
    os.symlink(sup / "matrix.jsonl.gz", t / "out" / "m1-natural-02of02" / "matrix.jsonl.gz")
    rc, out, err = w.run(capsys=capsys)
    refused(rc, err, "matrix-missing-or-linked", "shard=m1-natural-02of02")


def test_a_shard_dir_that_is_a_symlink_into_superseded_is_refused(tmp_path, capsys):
    w = good_world(tmp_path)
    t = w.root / "opaque-sonnet"
    sup = t / "superseded" / "first-run" / "out"
    sup.mkdir(parents=True)
    os.rename(t / "out" / "m1-natural-02of02", sup / "m1-natural-02of02")
    os.symlink(sup / "m1-natural-02of02", t / "out" / "m1-natural-02of02")
    rc, out, err = w.run(capsys=capsys)
    refused(rc, err, "matrix-missing-or-linked")


def test_a_missing_online_matrix_is_not_replaced_by_a_superseded_or_obqc_one(tmp_path, capsys):
    w = good_world(tmp_path)
    t = w.root / "opaque-sonnet"
    for place in (t / "superseded" / "first-run" / "online" / "out" / "m1-natural-02of02",
                  t / "obqc" / "SEALED-obqc" / "m1-natural-02of02"):
        place.mkdir(parents=True)
        (place / "matrix.jsonl.gz").write_bytes(matrix_bytes(["rid-n3"]))
    (t / "out" / "m1-natural-02of02" / "matrix.jsonl.gz").unlink()
    rc, out, err = w.run(capsys=capsys)
    refused(rc, err, "matrix-missing-or-linked", "shard=m1-natural-02of02")


def test_shard_dirs_that_differ_from_the_list_are_refused(tmp_path, capsys):
    w = good_world(tmp_path)
    (w.root / "opaque-sonnet" / "shards-natural" / "m1-natural-03of03" / "records").mkdir(parents=True)
    rc, out, err = w.run(capsys=capsys)
    refused(rc, err, "shard-dirs-differ", "tree=opaque-sonnet", "phase=natural")


def test_a_record_count_that_is_not_the_lists_is_refused(tmp_path, capsys):
    w = good_world(tmp_path)
    tsv = w.root / "opaque-sonnet" / "shards-natural.tsv"
    tsv.write_text(tsv.read_text().replace("m1-natural-01of02\topaque-sonnet\tm1\t2", "m1-natural-01of02\topaque-sonnet\tm1\t3"))
    rc, out, err = w.run(capsys=capsys)
    refused(rc, err, "record-count", "shard=m1-natural-01of02")


def test_a_record_in_two_shards_is_refused(tmp_path, capsys):
    w = good_world(tmp_path)
    r = w.root / "opaque-sonnet" / "shards-natural" / "m1-natural-02of02" / "records"
    os.symlink(w.out / "records" / rec_name("natural", "q1", "rid-n1"), r / rec_name("natural", "q1", "rid-n1"))
    tsv = w.root / "opaque-sonnet" / "shards-natural.tsv"
    tsv.write_text(tsv.read_text().replace("m1-natural-02of02\topaque-sonnet\tm1\t1", "m1-natural-02of02\topaque-sonnet\tm1\t2"))
    rc, out, err = w.run(capsys=capsys)
    refused(rc, err, "record-in-two-shards", "out=opaque-sonnet", "phase=natural")


def test_a_stage_that_exists_is_refused_and_a_changed_copy_is_refused(tmp_path, capsys):
    w = good_world(tmp_path)
    w.stage.mkdir()
    rc, out, err = w.run(capsys=capsys)
    refused(rc, err, "stage-exists")
    loc = rm.Located("natural", "t", "s", tmp_path / "x.gz", "0" * 64, ())
    (tmp_path / "x.gz").write_bytes(b"not the sealed bytes")
    with pytest.raises(rm.Refused) as exc:
        rm.stage([loc], tmp_path / "other-stage", "opaque-sonnet")
    assert exc.value.code == "matrix-sha-differs"


# -- the refusals: coverage ---------------------------------------------------------------------------------------------------


def test_a_record_without_its_matrix_rows_is_refused(tmp_path, capsys):
    w = good_world(tmp_path)
    # rid-n2's rows are all gone from its shard's matrix; the matrix is re-sealed, so the sealed hash is right
    keep = matrix_bytes(["rid-n1"])
    w.rewrite_matrix("opaque-sonnet", "m1-natural-01of02", keep)
    rc, out, err = w.run(capsys=capsys)
    refused(rc, err, "matrix-differs-from-records", "tree=opaque-sonnet", "shard=m1-natural-01of02", "records=2",
            "records_without_rows=1", "rows=32", "rows_expected=64")


def test_a_record_with_some_of_its_rows_missing_is_refused(tmp_path, capsys):
    w = good_world(tmp_path)
    data = matrix_bytes(["rid-n3"], drop={("rid-n3", "census-t2", "both_off")})
    w.rewrite_matrix("opaque-sonnet", "m1-natural-02of02", data)
    rc, out, err = w.run(capsys=capsys)
    refused(rc, err, "matrix-differs-from-records", "shard=m1-natural-02of02", "rows=31", "rows_expected=32", "missing_rows=1")


def test_a_duplicate_row_is_refused(tmp_path, capsys):
    w = good_world(tmp_path)
    data = matrix_bytes(["rid-n3"], dup={("rid-n3", "union", "primary")})
    w.rewrite_matrix("opaque-sonnet", "m1-natural-02of02", data)
    rc, out, err = w.run(capsys=capsys)
    refused(rc, err, "matrix-differs-from-records", "rows=33", "rows_expected=32", "duplicate_rows=1")


def test_a_row_of_an_arm_the_worker_does_not_replay_is_refused(tmp_path, capsys):
    w = good_world(tmp_path)
    w.rewrite_matrix("opaque-sonnet", "m1-natural-02of02", matrix_bytes(["rid-n3"], extra_arm="obqc"))
    rc, out, err = w.run(capsys=capsys)
    refused(rc, err, "matrix-differs-from-records", "extra_rows=1", "rows=33")


def test_the_obqc_matrix_of_a_shard_cannot_stand_in_for_its_online_one(tmp_path, capsys):
    w = good_world(tmp_path)
    w.rewrite_matrix("opaque-sonnet", "m1-natural-02of02", matrix_bytes(["rid-n3"], arms=ARMS + ["obqc", "obqc-where"]))
    rc, out, err = w.run(capsys=capsys)
    refused(rc, err, "matrix-differs-from-records", "extra_rows=8")


def test_rows_of_a_run_id_that_is_no_record_of_the_shard_are_refused(tmp_path, capsys):
    w = good_world(tmp_path)
    w.rewrite_matrix("opaque-sonnet", "m1-natural-02of02", matrix_bytes(["rid-n3", "rid-other"]))
    rc, out, err = w.run(capsys=capsys)
    refused(rc, err, "matrix-differs-from-records", "rows_of_no_record=1", "records=1")


def test_a_row_of_another_split_or_version_is_refused(tmp_path, capsys):
    w = good_world(tmp_path)
    w.rewrite_matrix("opaque-sonnet", "m1-natural-02of02", matrix_bytes(["rid-n3"], split="dev"))
    rc, out, err = w.run(capsys=capsys)
    refused(rc, err, "matrix-differs-from-records", "wrong_split_rows=32")
    w2 = good_world(tmp_path / "second")
    w2.rewrite_matrix("opaque-sonnet", "m1-natural-02of02", matrix_bytes(["rid-n3"], version="other/1"))
    rc, out, err = w2.run(capsys=capsys)
    refused(rc, err, "matrix-differs-from-records", "wrong_version_rows=32")


def test_a_record_of_the_out_that_no_shard_holds_is_refused(tmp_path, capsys):
    w = good_world(tmp_path)
    # the shard that held rid-n3 is dropped from the list and the dirs, with its marker count and matrix: the OUT still has the record
    t = w.root / "opaque-sonnet"
    (t / "shards-natural.tsv").write_text("m1-natural-01of02\topaque-sonnet\tm1\t2\n")
    shutil.rmtree(t / "shards-natural" / "m1-natural-02of02")
    w.mark("natural", f"OK shards 1 online 1 obqc 1 errors 0 {MARKER_UTC}")
    rc, out, err = w.run(capsys=capsys)
    refused(rc, err, "shards-differ-from-out", "phase=natural", "out_records=3", "shard_records=2", "missing=1", "extra=0")


def test_a_shard_record_that_is_no_longer_the_outs_is_refused(tmp_path, capsys):
    w = good_world(tmp_path)
    (w.out / "records" / rec_name("natural", "q3", "rid-n3")).unlink()  # voided and moved out after the replay
    rc, out, err = w.run(capsys=capsys)
    refused(rc, err, "shards-differ-from-out", "phase=natural", "out_records=2", "shard_records=3", "missing=0", "extra=1")


def test_a_record_a_second_attempt_of_the_same_digest_is_refused(tmp_path, capsys):
    w = good_world(tmp_path)
    (w.out / "records" / rec_name("natural", "q3", "rid-n3-again")).write_text("{}")
    rc, out, err = w.run(capsys=capsys)
    refused(rc, err, "records-unclassified", "digest_twice=1", "unclassified=0", "odd_names=0")


def test_a_record_that_no_order_names_is_refused_not_dropped(tmp_path, capsys):
    w = good_world(tmp_path)
    (w.out / "records" / ("f" * 40 + ".rid-stray.json")).write_text("{}")
    rc, out, err = w.run(capsys=capsys)
    refused(rc, err, "records-unclassified", "unclassified=1", "odd_names=0", "digest_twice=0")  # a flag, whatever the number


def test_a_json_file_of_an_odd_name_in_records_is_refused_and_setasides_are_ignored(tmp_path, capsys):
    w = good_world(tmp_path)
    (w.out / "records" / (DIGEST[("natural", "q1")] + ".rid-n1.setaside.json")).write_text("{}")
    (w.out / "records" / "notes.txt").write_text("x")
    rc, out, err = w.run(capsys=capsys)
    assert rc == 0, err
    (w.out / "records" / "odd.json").write_text("{}")
    w.stage.rename(tmp_path / "stage-first")
    rc, out, err = w.run(capsys=capsys)
    refused(rc, err, "records-unclassified", "odd_names=1")


def test_an_order_key_with_no_route_is_refused_and_so_is_a_shared_digest(tmp_path, capsys):
    w = good_world(tmp_path)
    order = w.order["natural"] + [["m-unrouted", "natural", "q9", 1]]
    (w.out / "order-natural.json").write_text(json.dumps({"order": order}))
    rc, out, err = w.run(capsys=capsys)
    refused(rc, err, "order-keys-ambiguous", "phase=natural", "unrouted=1", "shared=0")
    w2 = good_world(tmp_path / "second")
    (w2.out / "order-natural.json").write_text(json.dumps({"order": w2.order["natural"] + [["m1", "natural", "q1", 1]]}))
    rc, out, err = w2.run(capsys=capsys)
    refused(rc, err, "order-keys-ambiguous", "unrouted=0", "shared=2")


def test_two_names_with_one_route_have_no_route_and_are_refused(tmp_path, capsys):
    w = good_world(tmp_path)
    (w.out / "served-models.json").write_text(json.dumps({"models": {"m1": {"route_model": "route-1"},
                                                                       "m2": {"route_model": "route-1"}}}))
    rc, out, err = w.run(capsys=capsys)
    refused(rc, err, "order-keys-ambiguous", "unrouted=3")


def test_a_missing_order_file_of_a_requested_phase_is_refused(tmp_path, capsys):
    w = good_world(tmp_path)
    (w.out / "order-prompted.json").unlink()
    rc, out, err = w.run(capsys=capsys)
    refused(rc, err, "phases-differ", "requested=natural,prompted", "present=natural")  # the command line asks first
    with pytest.raises(rm.Refused) as exc:  # and the classification itself refuses a requested phase with no order file
        rm.classify_records(w.out, ("natural", "prompted"))
    assert exc.value.code == "no-order-file" and exc.value.counts == {"phase": "prompted"}


def test_a_digest_in_two_phases_is_refused(tmp_path, capsys):
    w = good_world(tmp_path)
    (w.out / "order-branch.json").write_text(json.dumps({"order": [["m1", "natural", "q1", 1]]}))
    rc, out, err = w.run(capsys=capsys)
    refused(rc, err, "digest-in-two-phases")


def test_the_branch_phase_is_not_required_to_have_matrix_rows(tmp_path, capsys):
    # the branch record rid-b1 has no shard and no rows: a branch record checks its parent's output, which the parent's rows cover
    w = good_world(tmp_path)
    rc, out, err = w.run(capsys=capsys)
    assert rc == 0 and "branch" not in out and "records_in_out" not in out  # entry 199: no branch number is printed


def test_a_truncated_matrix_is_a_refusal_with_the_class_name_only(tmp_path, capsys):
    w = good_world(tmp_path)
    data = matrix_bytes(["rid-n3"])[:-9]
    w.rewrite_matrix("opaque-sonnet", "m1-natural-02of02", data)
    rc, out, err = w.run(capsys=capsys)
    assert rc == 2 and err.startswith("REFUSED io-error class="), err


def test_the_command_line_exits_2_on_a_refusal_and_0_on_success(tmp_path):
    w = good_world(tmp_path)
    cmd = [sys.executable, "-I", str(DRIVER), "check", "--root", str(w.root), "--done-dir", str(w.done), "--cells-tsv",
           str(w.cells), "--out-dir", str(w.out), "--phases", "natural,prompted", "--stage", str(w.stage), *w.routing_args(),
           "--timefiles-manifest", w.make_manifest()]
    ok = subprocess.run(cmd, capture_output=True, text=True)
    assert ok.returncode == 0 and ok.stdout.rstrip().endswith(" OK")
    (w.done / "opaque-sonnet-natural").unlink()
    cmd[cmd.index("--stage") + 1] = str(tmp_path / "stage-2")
    bad = subprocess.run(cmd, capture_output=True, text=True)
    assert bad.returncode == 2 and bad.stderr == "REFUSED no-done-marker out=opaque-sonnet phase=natural\n"
    assert bad.stdout == f"CELLS-TSV sha256={sha(w.cells.read_bytes())} lines=0\n"  # the only line printed before the refusal


def test_a_symlinked_out_dir_is_the_out(tmp_path, capsys):
    # the runs tree links its OUTs to /data/...: the cells file names the link, whose basename is the replay tree's name
    w = good_world(tmp_path / "w")
    (tmp_path / "runs-links").mkdir()
    link = tmp_path / "runs-links" / "opaque-sonnet"
    os.symlink(w.out, link)
    rc = rm.main(["check", "--root", str(w.root), "--done-dir", str(w.done), "--cells-tsv", str(w.cells),
                  "--out-dir", str(link), "--phases", "natural,prompted", "--stage", str(w.stage), *w.routing_args(),
                  "--timefiles-manifest", w.make_manifest()])
    out, err = capsys.readouterr()
    assert rc == 0 and err == "" and out.splitlines()[-1].startswith("COVERAGE out=opaque-sonnet ")


def test_the_replay_root_may_be_a_symlink_but_a_tree_inside_it_may_not(tmp_path, capsys):
    w = good_world(tmp_path / "w")
    link = tmp_path / "root-link"
    os.symlink(w.root, link)  # replay-tier2-20261023 is a link to the day's directory
    w.root = link
    rc, out, err = w.run("natural,prompted", capsys)
    assert rc == 0, err
    w2 = good_world(tmp_path / "w2")
    real = w2.root / "opaque-sonnet"
    moved = tmp_path / "w2" / "moved-tree"
    os.rename(real, moved)
    os.symlink(moved, real)
    rc, out, err = w2.run("natural,prompted", capsys)
    refused(rc, err, "tree-missing", "tree=opaque-sonnet")


def test_main_leaves_the_processs_umask_alone(tmp_path, capsys):
    before = os.umask(0o022)
    try:
        w = good_world(tmp_path)
        w.run("natural,prompted", capsys)
        assert os.umask(0o022) == 0o022
    finally:
        os.umask(before)


def test_a_relative_stage_path_gives_absolute_entries(tmp_path, capsys, monkeypatch):
    w = good_world(tmp_path / "w")
    monkeypatch.chdir(tmp_path)
    rc = rm.main(["check", "--root", str(w.root), "--done-dir", str(w.done), "--cells-tsv", str(w.cells),
                  "--out-dir", str(w.out), "--phases", "natural,prompted", "--stage", "relative-stage", *w.routing_args(),
                  "--timefiles-manifest", w.make_manifest()])
    capsys.readouterr()
    assert rc == 0
    listed = (tmp_path / "relative-stage" / "matrices.list").read_text().splitlines()
    assert len(listed) == 3 and all(os.path.isabs(x) and x.startswith(str(tmp_path / "relative-stage")) for x in listed)


def test_an_argument_error_prints_a_fixed_refusal_and_not_the_argument(capsys):
    for argv in (["check", "--no-such-option", "secret-looking-value"], ["locate"], ["nothing"], []):
        with pytest.raises(SystemExit) as exc:
            rm.main(argv)
        out, err = capsys.readouterr()
        assert exc.value.code == 2 and out == "" and err == "REFUSED bad-arguments\n", argv


def test_no_number_of_the_branch_phase_is_ever_printed(tmp_path, capsys):
    # entry 199: never report a branch phase's covered/total. A branch order of two keys and two branch records, and two stray
    # records that match no order: the success output names no branch number, and the refusal is a flag whatever the number of strays
    w = good_world(tmp_path)
    w.add_records("branch", [("branch-curated", "q2", "rid-b2")])  # the world has one branch key and record; now two of each
    assert len(w.order["branch"]) == 2 and len(list((w.out / "records").glob("*.json"))) == 7  # 3 natural + 2 prompted + 2 branch
    rc, out, err = w.run(capsys=capsys)
    assert rc == 0 and "branch" not in out and "records_in_out" not in out and err == ""
    for i in (1, 2):
        (w.out / "records" / ("a" * 39 + f"{i}.rid-stray-{i}.json")).write_text("{}")
    w.stage.rename(tmp_path / "stage-moved")
    rc, out, err = w.run(capsys=capsys)
    assert rc == 2 and err == "REFUSED records-unclassified unclassified=1 odd_names=0 digest_twice=0 run_id_twice=0\n"


# -- the output flag, and the exported ledger against the matrices --------------------------------------------------------------


def test_a_matrix_whose_rows_say_nothing_about_an_output_is_refused(tmp_path, capsys):
    # the exporter skips every row without an output: 32 rows of a run with no `output` key would pass a row-count check and
    # yield no ledger row (external reviewer r3). Here the whole shard's rows lack it.
    w = good_world(tmp_path)
    w.rewrite_matrix("opaque-sonnet", "m1-natural-02of02", matrix_bytes(["rid-n3"], omit_output=True))
    rc, out, err = w.run(capsys=capsys)
    refused(rc, err, "matrix-differs-from-records", "shard=m1-natural-02of02", "output_not_bool=1")


def test_a_run_whose_rows_disagree_about_its_output_is_refused(tmp_path, capsys):
    w = good_world(tmp_path)
    w.rewrite_matrix("opaque-sonnet", "m1-natural-02of02", matrix_bytes(["rid-n3"], flip_output={("rid-n3", "union", "primary")}))
    rc, out, err = w.run(capsys=capsys)
    refused(rc, err, "matrix-differs-from-records", "output_disagrees=1")


def test_a_run_with_an_output_and_no_scorer_label_is_refused(tmp_path, capsys):
    w = good_world(tmp_path)
    w.rewrite_matrix("opaque-sonnet", "m1-natural-02of02", matrix_bytes(["rid-n3"], unlabelled={"rid-n3"}))
    rc, out, err = w.run(capsys=capsys)
    refused(rc, err, "matrix-differs-from-records", "output_without_label=1")


LEDGER_HEADER = json.dumps({"schema": "checker-ledger/2", "split": "test"})


def ledger_file(path: Path, runs, *, sources=ARMS, drop=(), dup=(), extra=(), header=LEDGER_HEADER, online=()) -> Path:
    """A ledger by hand: a replay row per (run, source), the keys the check reads."""
    rows = []
    for rid in runs:
        for src in sources:
            if (rid, src) in drop:
                continue
            rows.append({"split": "test", "run_id": rid, "source": src, "use": "replay", "status": "P"})
            if (rid, src) in dup:
                rows.append({"split": "test", "run_id": rid, "source": src, "use": "replay", "status": "P"})
    rows += [{"split": "test", "run_id": rid, "source": src, "use": "replay", "status": "P"} for rid, src in extra]
    rows += [{"split": "test", "run_id": rid, "source": src, "use": use, "status": "F"} for rid, src, use in online]
    path.write_text("\n".join([header] + [json.dumps(r) for r in rows]) + "\n")
    return path


def staged_good_world(tmp_path, capsys) -> World:
    w = good_world(tmp_path)
    rc, out, err = w.run(capsys=capsys)
    assert rc == 0, err
    return w


def ledger_check(w: World, ledger: Path, capsys):
    rc = rm.main(["ledger-check", "--stage", str(w.stage), "--ledger", str(ledger)])
    out, err = capsys.readouterr()
    assert "rid-" not in out + err
    return rc, out, err


OUTPUT_RUNS = ["rid-n1", "rid-n2", "rid-p1", "rid-p2"]  # rid-n3 has no output


def test_an_exported_ledger_holding_the_matrices_outputs_once_per_arm_passes(tmp_path, capsys):
    w = staged_good_world(tmp_path, capsys)
    led = ledger_file(tmp_path / "ledger.jsonl", OUTPUT_RUNS, online=[("rid-n1", "curated", "checker"), ("rid-n1", "union", "both")])
    rc, out, err = ledger_check(w, led, capsys)
    assert rc == 0 and err == "" and out == "LEDGER-CHECK arms=8 OK\n"


def test_an_export_that_dropped_a_run_is_refused(tmp_path, capsys):
    w = staged_good_world(tmp_path, capsys)
    led = ledger_file(tmp_path / "ledger.jsonl", OUTPUT_RUNS, drop={("rid-p2", "census-t2")})
    rc, out, err = ledger_check(w, led, capsys)
    assert rc == 2 and err == "REFUSED ledger-differs-from-matrices missing=1 extra=0 duplicate=0 other_sources=0\n"
    led = ledger_file(tmp_path / "ledger2.jsonl", OUTPUT_RUNS[:2] + OUTPUT_RUNS[3:])  # a whole run gone: 8 rows
    rc, out, err = ledger_check(w, led, capsys)
    assert rc == 2 and "missing=1 " in err  # a flag, though eight rows are gone


def test_an_export_with_a_run_without_an_output_a_repeat_or_another_replay_source_is_refused(tmp_path, capsys):
    w = staged_good_world(tmp_path, capsys)
    led = ledger_file(tmp_path / "a.jsonl", OUTPUT_RUNS, extra=[("rid-n3", "curated")])  # rid-n3 has no output
    rc, out, err = ledger_check(w, led, capsys)
    assert rc == 2 and err == "REFUSED ledger-differs-from-matrices missing=0 extra=1 duplicate=0 other_sources=0\n"
    led = ledger_file(tmp_path / "b.jsonl", OUTPUT_RUNS, dup={("rid-n1", "union")})
    rc, out, err = ledger_check(w, led, capsys)
    assert rc == 2 and "duplicate=1" in err
    led = ledger_file(tmp_path / "c.jsonl", OUTPUT_RUNS, extra=[("rid-n1", "obqc"), ("rid-n1", "curated/heuristic_off")])
    rc, out, err = ledger_check(w, led, capsys)
    assert rc == 2 and "other_sources=1" in err  # a flag, though two rows are of other sources


def test_a_ledger_of_another_header_or_an_empty_output_list_is_refused(tmp_path, capsys):
    w = staged_good_world(tmp_path, capsys)
    for header in (json.dumps({"schema": "checker-ledger/2", "split": "dev"}),
                   json.dumps({"schema": "checker-ledger/2", "split": "test", "exploratory": True}), "{}"):
        led = ledger_file(tmp_path / "h.jsonl", OUTPUT_RUNS, header=header)
        rc, out, err = ledger_check(w, led, capsys)
        assert rc == 2 and err == "REFUSED ledger-header\n"
    (w.stage / "replay-outputs.list").write_text("")
    rc, out, err = ledger_check(w, ledger_file(tmp_path / "ok.jsonl", OUTPUT_RUNS), capsys)
    assert rc == 2 and err == "REFUSED no-outputs-listed\n"


def test_two_record_files_with_one_run_id_are_refused(tmp_path, capsys):
    # Gemini: two files with different claim digests and the same run id would collapse to one id in a shard's matrix check
    w = good_world(tmp_path)
    (w.out / "records" / rec_name("natural", "q2", "rid-n2")).unlink()  # rid-n2's file is renamed to carry rid-n1's run id
    (w.out / "records" / f"{DIGEST[('natural', 'q2')]}.rid-n1.json").write_text("{}")
    rc, out, err = w.run(capsys=capsys)
    refused(rc, err, "records-unclassified", "run_id_twice=1", "digest_twice=0", "unclassified=0")


def test_a_run_id_with_whitespace_or_another_odd_character_is_refused_everywhere(tmp_path, capsys):
    # external reviewer r4: the run-id lists split on whitespace, so a record name carrying a space in its run id must never be accepted
    w = good_world(tmp_path)
    (w.out / "records" / f"{DIGEST[('natural', 'q1')]}.rid a.json").write_text("{}")
    rc, out, err = w.run(capsys=capsys)
    refused(rc, err, "records-unclassified", "odd_names=1")
    w2 = good_world(tmp_path / "second")
    rdir = w2.root / "opaque-sonnet" / "shards-natural" / "m1-natural-01of02" / "records"
    os.symlink(w2.out / "records" / "x", rdir / f"{DIGEST[('natural', 'q3')]}.rid a.json")
    tsv = w2.root / "opaque-sonnet" / "shards-natural.tsv"
    tsv.write_text(tsv.read_text().replace("m1-natural-01of02\topaque-sonnet\tm1\t2", "m1-natural-01of02\topaque-sonnet\tm1\t3"))
    rc, out, err = w2.run(capsys=capsys)
    refused(rc, err, "record-count", "shard=m1-natural-01of02")
    for odd in ("rid\nx", "rid;x", "rid/x", "rid x", "r\u00e9"):
        assert rm.RUN_FILE.fullmatch(f"{DIGEST[('natural', 'q1')]}.{odd}.json") is None, repr(odd)
    assert rm.RUN_FILE.fullmatch(f"{DIGEST[('natural', 'q1')]}.4f1c2a9e-0b7d-4c53-9a7e-2d6f8b1a3c55.json")  # a UUID, as the runner writes


def test_a_ledger_row_whose_run_id_or_source_is_not_a_string_is_refused(tmp_path, capsys):
    # external reviewer r5: str() would let the number 1 match a run id "1"
    w = staged_good_world(tmp_path, capsys)
    (w.stage / "replay-outputs.list").write_text("1\n")
    rows = [{"split": "test", "run_id": 1, "source": "curated", "use": "replay"}]
    rows += [{"split": "test", "run_id": "1", "source": a, "use": "replay"} for a in ARMS]
    led = tmp_path / "numeric.jsonl"
    led.write_text("\n".join([LEDGER_HEADER] + [json.dumps(r) for r in rows]) + "\n")
    rc, out, err = ledger_check(w, led, capsys)
    assert rc == 2 and err == "REFUSED ledger-differs-from-matrices missing=0 extra=0 duplicate=0 other_sources=1\n"


def test_no_number_derived_from_the_output_flags_is_printed(tmp_path, capsys):
    # whether a run has an output is a property of its result: the success lines of check and ledger-check carry none, though the
    # fixture has an attempt without one (rid-n3) and the staged list holds the other four
    w = good_world(tmp_path)
    rc, out, err = w.run(capsys=capsys)
    assert rc == 0 and "output" not in out
    led = ledger_file(tmp_path / "ledger.jsonl", OUTPUT_RUNS)
    rc, out, err = ledger_check(w, led, capsys)
    assert rc == 0 and out == "LEDGER-CHECK arms=8 OK\n"


def test_no_refusal_prints_a_count_derived_from_the_output_flags(tmp_path, capsys):
    # external reviewer r6: a refusal about the output flags, or a ledger with no replay rows at all, must not reveal how many runs have an output
    w = good_world(tmp_path)
    w.rewrite_matrix("opaque-sonnet", "m1-natural-01of02", matrix_bytes(["rid-n1", "rid-n2"], unlabelled={"rid-n1", "rid-n2"}))
    rc, out, err = w.run(capsys=capsys)  # 64 rows lack their label, which is two runs of 32
    assert rc == 2 and err == ("REFUSED matrix-differs-from-records tree=opaque-sonnet shard=m1-natural-01of02 records=2 "
                               "records_without_rows=0 rows_of_no_record=0 rows=64 rows_expected=64 output_without_label=1\n")
    w2 = staged_good_world(tmp_path / "second", capsys)
    led = ledger_file(tmp_path / "empty.jsonl", [])  # a valid header and no replay row: 4 outputs x 8 arms are missing
    rc, out, err = ledger_check(w2, led, capsys)
    assert rc == 2 and err == "REFUSED ledger-differs-from-matrices missing=1 extra=0 duplicate=0 other_sources=0\n"


def test_a_matrix_with_invalid_deflate_data_is_a_sanitized_refusal(tmp_path, capsys):
    # an independent reviewer: zlib.error is neither OSError nor ValueError, and escaped the one-line refusal
    w = good_world(tmp_path)
    w.rewrite_matrix("opaque-sonnet", "m1-natural-02of02", b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff" + b"\xff" * 20)
    rc, out, err = w.run(capsys=capsys)
    assert rc == 2 and err == "REFUSED io-error class=error\n"


# -- the class routing of the replay rows: the copy is taken ONLY from the worker's recorded launch (analysis lead, 11:24Z) -----------


def routing_doc(w: World) -> dict:
    """The routing file as the tool wrote it (schema 5: shards -> {class, basis, runs}) with convenience views: ``records``
    (run -> class), ``bases`` (run -> basis), ``trees`` (tree -> kind, shard classes, attested shards) and ``attested_runs``."""
    doc = json.loads((w.stage / "replay-routing.json").read_text())
    doc["records"], doc["bases"], doc["trees"], doc["attested_runs"] = {}, {}, {}, {}
    for key, info in sorted(doc["shards"].items()):
        tree, rest = key.split("/", 1)
        t = doc["trees"].setdefault(tree, {"kind": "cell" if "-cell-" in tree else "main", "shards": {}, "attested": []})
        t["shards"][rest] = info["class"]
        for rid in info["runs"]:
            doc["records"][rid] = info["class"]
            if info["basis"]:
                doc["bases"][rid] = info["basis"]
        if info["basis"] == "attested":
            t["attested"].append(rest)
            doc["attested_runs"][key] = info["runs"]
    return doc


def test_the_time_file_format_is_gnu_times(tmp_path):
    # the command line GNU time -v wrote on the spoke for the same argument shapes (a run of /usr/bin/time -v -o f echo ... on run-node)
    first = ('\tCommand being timed: "echo --records /a/b --endpoint http://127.0.0.1:47125/ --split test --note Tier 2 natural replay, '
             'freeze 126a111, rhea-gemma-4-31b-cell-slow shard m1-natural-01of02, example-copy-40, run-node --out /o"')
    assert rm.COMMAND_LINE.fullmatch(first)
    cmd = launch_command("rhea-gemma-4-31b-cell-slow", "m1-natural-01of02", "natural", "127.0.0.1:47125", "example-copy-40")
    m = rm.LAUNCH_COMMAND.fullmatch(cmd)
    assert m and m.groups() == ("127.0.0.1:47125", "natural", "rhea-gemma-4-31b-cell-slow", "m1-natural-01of02", "example-copy-40")


def test_text_inside_an_argument_cannot_supply_another_arguments_value(tmp_path):
    # external reviewer r15: T2_COPYID_OPAQUE_ARM64 is unrestricted in the worker; a copy id carrying "--out /x --endpoint http://..." must not parse
    real = launch_command("opaque-sonnet", "s1", "natural", "127.0.0.1:47278", "example-copy-33")
    assert rm.LAUNCH_COMMAND.fullmatch(real)
    injected_id = "x, run-node --out /tmp/x --endpoint http://192.0.2.101:7033/ --split test --note Tier 2 natural replay, freeze 126a111, opaque-sonnet shard s1, example-copy-12"
    assert rm.LAUNCH_COMMAND.fullmatch(launch_command("opaque-sonnet", "s1", "natural", "127.0.0.1:47278", "x", note_copy=injected_id)) is None
    # an endpoint argument that carries a second endpoint inside one argv element (a valid URL never contains a space)
    spaced = real.replace("--endpoint http://127.0.0.1:47278/", "--endpoint http://127.0.0.1:47278/ --endpoint http://192.0.2.101:7033")
    assert rm.LAUNCH_COMMAND.fullmatch(spaced) is None
    # nothing before or after the worker's shape is tolerated
    assert rm.LAUNCH_COMMAND.fullmatch(real + " --endpoint http://192.0.2.101:7033/") is None


def test_the_routing_of_cells_and_a_whole_phase_tree_is_each_shards_recorded_copy(tmp_path, capsys):
    w = split_world(tmp_path)
    rc, out, err = w.run(capsys=capsys)
    assert rc == 0, err
    lines = out.splitlines()
    assert "ROUTING out=rhea-gemma-4-31b tree=rhea-gemma-4-31b kind=main fast=1 medium=0 slow=0 unverifiable=0 not_recorded=0 attested=0 pending_attestation=0" in lines
    assert "ROUTING out=rhea-gemma-4-31b tree=rhea-gemma-4-31b-cell-fast kind=cell fast=1 medium=0 slow=0 unverifiable=0 not_recorded=0 attested=0 pending_attestation=0" in lines
    assert "ROUTING out=rhea-gemma-4-31b tree=rhea-gemma-4-31b-cell-slow kind=cell fast=0 medium=0 slow=1 unverifiable=0 not_recorded=0 attested=0 pending_attestation=0" in lines
    doc = routing_doc(w)
    assert doc["schema"] == "replay-routing/5" and doc["out"] == "rhea-gemma-4-31b"
    assert doc["records"] == {"rid-n1": "fast", "rid-n2": "fast", "rid-p1": "fast", "rid-p2": "fast", "rid-p3": "slow"}
    assert doc["trees"]["rhea-gemma-4-31b-cell-slow"] == {"kind": "cell", "shards": {"prompted/m1-prompted-01of01": "slow"}, "attested": []}
    assert doc["inputs"]["cell_classes_sha256"] == sha(w.cell_classes.read_bytes())
    t = w.root / "rhea-gemma-4-31b-cell-slow" / "logs" / "m1-prompted-01of01.time"
    assert doc["inputs"]["time_files_sha256"]["rhea-gemma-4-31b-cell-slow/prompted/m1-prompted-01of01"] == sha(t.read_bytes())
    assert set(doc["inputs"]) == {"cell_classes_sha256", "time_files_sha256", "timefiles_manifest", "attestations"}
    assert doc["inputs"]["attestations"] == [] and doc["inputs"]["timefiles_manifest"]["taken_at"] == "20261025T123234Z"
    assert len(doc["inputs"]["timefiles_manifest"]["manifests"]) == 3
    assert doc["attested_runs"] == {} and all(t["attested"] == [] for t in doc["trees"].values())  # nothing inferred, nothing attested
    assert oct((w.stage / "replay-routing.json").stat().st_mode & 0o777) == "0o600"


def test_the_recorded_endpoint_not_the_cells_name_gives_the_class(tmp_path, capsys):
    # a shard of cell-fast whose time file records the slow copy is routed slow: the class is the record's, never the tree name's
    w = split_world(tmp_path)
    w.write_time("rhea-gemma-4-31b-cell-fast", "m1-prompted-01of01", "prompted", "127.0.0.1:47125", "example-copy-40",
                 mtime=rm.utc_epoch(SEALED_UTC) - 5)
    rc, out, err = w.run(capsys=capsys)
    assert rc == 0, err
    # a cell whose recorded copy is of ANOTHER class than the cell's: the worker's class gate failed, so it is unverifiable (an exception), not routed by the record
    assert routing_doc(w)["records"]["rid-p1"] == "unverifiable" and routing_doc(w)["records"]["rid-p2"] == "unverifiable"


def test_an_endpoint_override_shows_in_the_recorded_command(tmp_path, capsys):
    # the worker's copy id says fast (7033) while T2_EP_OPAQUE sent the checker to a slow copy's endpoint: the pair is not a row of the table
    w = good_world(tmp_path)
    w.write_time("opaque-sonnet", "m1-natural-02of02", "natural", "127.0.0.1:47278", "example-copy-12", mtime=rm.utc_epoch(SEALED_UTC) - 5)
    rc, out, err = w.run(capsys=capsys)
    assert rc == 0, err
    shards = routing_doc(w)["trees"]["opaque-sonnet"]["shards"]
    assert shards["natural/m1-natural-02of02"] == "unverifiable"  # recorded, but the table does not bind that pair
    assert shards["natural/m1-natural-01of02"] == "fast" and shards["prompted/m1-prompted-01of01"] == "fast"


def test_a_shard_with_no_time_file_is_not_recorded_and_never_inferred(tmp_path, capsys):
    w = good_world(tmp_path)
    (w.root / "opaque-sonnet" / "logs" / "m1-natural-02of02.time").unlink()
    rc, out, err = w.run(capsys=capsys)
    assert rc == 0, err
    assert routing_doc(w)["records"]["rid-n3"] == "unverifiable"
    assert routing_doc(w)["records"]["rid-n1"] == "fast"
    assert "ROUTING out=opaque-sonnet tree=opaque-sonnet kind=main fast=2 medium=0 slow=0 unverifiable=1 not_recorded=0 attested=0 pending_attestation=1" in out.splitlines()
    # and no map, config or default is consulted: a replay root with none of them behaves the same
    assert not (w.root / "store-map.tsv").exists() and not (w.root / "stores.d").exists()


@pytest.mark.parametrize("what", ["other-shard", "other-tree", "other-phase", "two-endpoints", "injected-copy-id", "no-endpoint", "no-note",
                                  "first-line", "huge", "symlink", "after-seal", "before-claim", "long-before-seal", "dir", "exit-status",
                                  "no-exit-line", "extra-argument", "other-script"])
def test_a_time_file_that_is_not_this_shards_launch_is_not_recorded(tmp_path, capsys, what):
    w = good_world(tmp_path)
    tree, shard, phase = "opaque-sonnet", "m1-natural-02of02", "natural"
    mt = rm.utc_epoch(SEALED_UTC) - 5
    ep, copy = "192.0.2.101:7033", "example-copy-12"
    base = launch_command(tree, shard, phase, ep, copy)
    path = w.root / tree / "logs" / f"{shard}.time"

    def put(cmd, **kw):
        w.write_time(tree, shard, phase, ep, copy, mt, command=cmd, **kw)
    if what == "other-shard":
        put(base.replace(f"shard {shard},", "shard m1-natural-01of02,"))
    elif what == "other-tree":
        put(base.replace(f"{tree} shard", "opaque-haiku shard"))
    elif what == "other-phase":
        put(base.replace("Tier 2 natural", "Tier 2 prompted"))
    elif what == "two-endpoints":
        put(base.replace(f"--endpoint http://{ep}/", f"--endpoint http://127.0.0.1:47278/ --endpoint http://{ep}/"))
    elif what == "injected-copy-id":
        put(launch_command(tree, shard, phase, "127.0.0.1:47278", "x", note_copy=f"x, run-node --out /t --endpoint http://{ep}/ --note Tier 2 {phase} replay, freeze 126a111, {tree} shard {shard}, {copy}"))
    elif what == "no-endpoint":
        put(base.replace(f"--endpoint http://{ep}/ ", ""))
    elif what == "no-note":
        put(base.replace("--note", "--notes"))
    elif what == "extra-argument":
        put(base + " --obqc")
    elif what == "other-script":
        put(base.replace("scripts/checker_replay.py", "scripts/bank_replay.py"))
    elif what == "exit-status":
        put(base, exit_status=1)
    elif what == "no-exit-line":
        path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    elif what == "first-line":
        path.write_text("Command being timed: no tab\n" + path.read_text())
    elif what == "huge":
        path.write_text(path.read_text() + "x" * 70000)
    elif what == "symlink":
        real = w.root / "real.time"
        real.write_text(path.read_text())
        path.unlink()
        os.symlink(real, path)
    elif what == "after-seal":
        os.utime(path, (rm.utc_epoch(SEALED_UTC) + 3600,) * 2)  # a later file: not the one written when this shard's checker ended
    elif what == "before-claim":
        w.set_started(tree, shard, rm.utc_epoch(SEALED_UTC) - 10)  # claimed 10 s before the seal
        os.utime(path, (rm.utc_epoch(SEALED_UTC) - 100,) * 2)  # inside the seal window but older than the claim: a previous run's file
    elif what == "long-before-seal":
        shutil.rmtree(w.root / "claims")  # no claim to lower-bound it: the seal window alone rejects a file 400 s before the seal
        os.utime(path, (rm.utc_epoch(SEALED_UTC) - 400,) * 2)
    else:
        path.unlink()
        path.mkdir()
    rc, out, err = w.run(capsys=capsys)
    assert rc == 0, err
    assert routing_doc(w)["records"]["rid-n3"] == "unverifiable", what
    assert routing_doc(w)["records"]["rid-n1"] == "fast"


def test_a_time_file_just_inside_the_seal_window_is_recorded_without_a_claim(tmp_path, capsys):
    w = good_world(tmp_path)
    shutil.rmtree(w.root / "claims")
    os.utime(w.root / "opaque-sonnet" / "logs" / "m1-natural-02of02.time", (rm.utc_epoch(SEALED_UTC) - 299,) * 2)
    rc, out, err = w.run(capsys=capsys)
    assert rc == 0 and routing_doc(w)["records"]["rid-n3"] == "fast"


def test_a_time_file_a_moment_after_the_seal_is_within_the_skew(tmp_path, capsys):
    w = good_world(tmp_path)
    os.utime(w.root / "opaque-sonnet" / "logs" / "m1-natural-02of02.time", (rm.utc_epoch(SEALED_UTC) + 30,) * 2)
    rc, out, err = w.run(capsys=capsys)
    assert rc == 0 and routing_doc(w)["records"]["rid-n3"] == "fast"


def test_a_shard_with_no_claim_is_still_routed_by_its_time_file(tmp_path, capsys):
    # the claim only lower-bounds the time file's date; its absence drops that one check, not the record
    w = good_world(tmp_path)
    shutil.rmtree(w.root / "claims")
    rc, out, err = w.run(capsys=capsys)
    assert rc == 0 and routing_doc(w)["records"]["rid-n3"] == "fast"


def test_a_recorded_copy_the_table_does_not_bind_is_unverifiable(tmp_path, capsys):
    w = good_world(tmp_path)
    mt = rm.utc_epoch(SEALED_UTC) - 5
    w.write_time("opaque-sonnet", "m1-natural-02of02", "natural", "192.0.2.101:7033", "opaque-rhea-unlisted", mt)  # copy not in the table
    w.write_time("opaque-sonnet", "m1-natural-01of02", "natural", "10.0.0.9:1234", "example-copy-12", mt)  # endpoint not in the table
    rc, out, err = w.run(capsys=capsys)
    assert rc == 0, err
    shards = routing_doc(w)["trees"]["opaque-sonnet"]["shards"]
    assert shards == {"natural/m1-natural-01of02": "unverifiable", "natural/m1-natural-02of02": "unverifiable",
                      "prompted/m1-prompted-01of01": "fast"}


def test_a_table_row_of_the_endpoint_for_another_copy_does_not_bind_this_one(tmp_path, capsys):
    # the pair (endpoint, copy id) must be ONE row: the endpoint of one copy with the id of another is not bound
    w = good_world(tmp_path)
    w.write_time("opaque-sonnet", "m1-natural-02of02", "natural", "127.0.0.1:47278", "example-copy-12", rm.utc_epoch(SEALED_UTC) - 5)
    rc, out, err = w.run(capsys=capsys)
    assert rc == 0 and routing_doc(w)["records"]["rid-n3"] == "unverifiable"


def test_no_store_map_config_or_attestation_is_an_input_any_more(tmp_path, capsys):
    w = good_world(tmp_path)
    for flag in ("--store-map-dir", "--attest-main-tree-routing"):
        with pytest.raises(SystemExit):
            rm.main(["check", "--root", str(w.root), "--done-dir", str(w.done), "--cells-tsv", str(w.cells), "--out-dir", str(w.out),
                     "--phases", "natural,prompted", "--stage", str(w.stage), "--cell-classes", str(w.cell_classes), flag, "x"])
        assert capsys.readouterr().err == "REFUSED bad-arguments\n"
    with pytest.raises(SystemExit):  # --cell-classes is still required
        rm.main(["check", "--root", str(w.root), "--done-dir", str(w.done), "--cells-tsv", str(w.cells), "--out-dir", str(w.out),
                 "--phases", "natural,prompted", "--stage", str(w.stage)])
    assert capsys.readouterr().err == "REFUSED bad-arguments\n"


@pytest.mark.parametrize("what", ["cell-classes-conflict", "cell-classes-malformed"])
def test_malformed_class_tables_are_refused(tmp_path, capsys, what):
    w = good_world(tmp_path)
    extra = "1.2.3.4:5\texample-copy-38\tslow\tx\n" if what == "cell-classes-conflict" else "1.2.3.4:5\texample-copy-38\tfastest\tx\n"
    w.cell_classes.write_text(w.cell_classes.read_text() + extra)
    rc, out, err = w.run(capsys=capsys)
    assert rc == 2 and err.startswith(f"REFUSED {what}"), err


# -- the operator attestation, per OUT-phase (design ruling, 25 Oct): a fallback and a conflict check, never an override ------------


def make_attestation(w: World, name="att.json", *, phase="natural", out=None, trees=None, **over):
    """A synthetic t2-attest/1 file with the producer's per-shard shape and canonical hash."""
    out = out or w.out_name
    tree_rows = trees if trees is not None else [{"tree": out, "class": "fast", "copy": "example-copy-12", "endpoint": "192.0.2.101:7033"}]
    rows = []
    for t in tree_rows:
        sl = w.root / t["tree"] / f"shards-{phase}.tsv"
        for line in sl.read_text().splitlines() if sl.exists() else []:
            shard = line.split("\t")[0]
            time_path = w.root / t["tree"] / "logs" / f"{shard}.time"
            first_line = time_path.read_text().splitlines()[0] if time_path.exists() else ""
            line2 = first_line == "Command exited with non-zero status 1"
            missing = not time_path.exists()
            state = "not_recorded" if missing or line2 else "recorded"
            endpoint_hash = sha(t["endpoint"].encode())
            cell_class = t["tree"].rsplit("-cell-", 1)[1] if "-cell-" in t["tree"] else None
            rows.append({"shard": shard, "tree": t["tree"], "cell": cell_class,
                         "state": state, "state_reason": "time_file_missing" if missing else "first_line_exit_status" if line2 else None,
                         "claim_utc": "2026-10-24T14:40:00Z", "seal_utc": SEALED_UTC, "run_end_utc": SEALED_UTC, "rc": 1 if line2 else 0,
                         "map_version": "store-map.tsv", "map_key": f"{t['tree'].split('-', 1)[0]}@{t['copy']}",
                         "copy": t["copy"], "ep": t["endpoint"],
                         "ep_sha256": endpoint_hash, "table_ep_sha256": endpoint_hash,
                         "time_endpoint_sha256": endpoint_hash if line2 or not missing else None,
                         "time_evidence": "line_2_after_accepted_exit" if line2 else "first_line" if not missing else None,
                         "time_copy": t["copy"] if line2 or not missing else None,
                         "class": t["class"], "class_basis": None,
                         "basis": "chain_only" if missing else "positive_time_endpoint",
                         "evidence_label": ("recorded (worker launch record, line 2 after an accepted non-zero exit; unsealed; sha manifest taken 12:32Z)"
                                            if line2 else None), "waiver": None,
                         "checks": {f"c{i}": not missing for i in range(1, 5)},
                         "check_reasons": {f"c{i}": ["chain_only_not_accepted"] if missing else [] for i in range(1, 5)},
                         "ok": not missing, "reasons": ["chain_only_not_accepted"] if missing else [], "notes": []})
    not_recorded_ok = all(r["ok"] for r in rows if r["state"] == "not_recorded")
    doc = {"schema": "t2-attest/1", "label": "operator attestation (from glue records)", "out": out, "phase": phase,
           "status": "replay_class_attested" if rows and not_recorded_ok else "replay_class_unverified",
           "generated_utc": "2026-10-25T13:00:00Z", "agent": "tier2-replay-5",
           "tool": {"name": "t2_attest.py", "sha256": "d" * 64}, "supersedes": None,
           "rulings": [],
           "shard_counts": {"total": len(rows), "recorded": sum(r["state"] == "recorded" for r in rows),
                            "not_recorded": sum(r["state"] == "not_recorded" for r in rows),
                            "recorded_not_ok": sum(r["state"] == "recorded" and not r["ok"] for r in rows),
                            "not_recorded_chain_only": sum(r["basis"] == "chain_only" for r in rows),
                            "waived": 0},
           "cells": ([{"cell": t["tree"], "class": t["tree"].rsplit("-cell-", 1)[1],
                       "shards": sum(r["tree"] == t["tree"] for r in rows),
                       "status": "pass" if all(r["ok"] for r in rows if r["tree"] == t["tree"] and r["state"] == "not_recorded") else "fail"}
                      for t in tree_rows] if any("-cell-" in t["tree"] for t in tree_rows) else None),
           "checks": {k: {"pass": not_recorded_ok, "reasons": []} for k in
                                    ("c1_no_override", "c2_maps_complete", "c3_no_unbacked_repoint", "c4_class_table")},
           "evidence": [{"path": "/glue/store-map.tsv", "sha256": "e" * 64, "kind": "store-map-live"}],
           "state_refs": {}, "shards": rows, "signing": {"method": "state", "assumption": "test",
                                                         "state_file": "STATE", "state_event": "t2-attest-signed"}}
    doc.update(over)
    if "content_sha256" not in over:
        doc["content_sha256"] = sha(json.dumps({k: v for k, v in doc.items() if k != "content_sha256"},
                                               sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode())
    path = w.stage.parent / name
    path.write_text(json.dumps(doc))
    return f"{path}:{sha(path.read_bytes())}", path, doc


def attested_world(tmp_path) -> World:
    """A valid producer fallback: GNU time line 1 is a nonzero exit, and line 2 is the accepted launch."""
    w = good_world(tmp_path)
    path = w.root / "opaque-sonnet" / "logs" / "m1-natural-02of02.time"
    path.write_text("Command exited with non-zero status 1\n" + path.read_text().replace("Exit status: 0", "Exit status: 1"))
    os.utime(path, (rm.utc_epoch(SEALED_UTC) - 5,) * 2)
    return w


def test_an_attestation_covers_a_shard_with_no_recorded_launch_under_its_own_label(tmp_path, capsys):
    w = attested_world(tmp_path)
    pin, path, doc = make_attestation(w)
    rc, out, err = w.run("natural,prompted", capsys, extra=("--attestation", pin))
    assert rc == 0, err
    assert "ROUTING out=opaque-sonnet tree=opaque-sonnet kind=main fast=3 medium=0 slow=0 unverifiable=0 not_recorded=0 attested=1 pending_attestation=0" in out.splitlines()
    r = routing_doc(w)
    assert r["records"] == {"rid-n1": "fast", "rid-n2": "fast", "rid-n3": "fast", "rid-p1": "fast", "rid-p2": "fast"}
    assert r["attested_runs"] == {"opaque-sonnet/natural/m1-natural-02of02": ["rid-n3"]}  # only the shard with no recorded launch
    assert r["trees"]["opaque-sonnet"]["attested"] == ["natural/m1-natural-02of02"]
    assert r["inputs"]["attestations"] == [{"out": "opaque-sonnet", "phase": "natural",
                                           "shards": [{"tree": "opaque-sonnet", "shard": "m1-natural-02of02", "class": "fast", "copy": "example-copy-12", "endpoint": "192.0.2.101:7033"}],
                                           "file_sha256": sha(path.read_bytes()), "content_sha256": doc["content_sha256"],
                                           "agent": "tier2-replay-5"}]
    assert r["shards"]["opaque-sonnet/natural/m1-natural-02of02"]["copy"] == {"endpoint": "192.0.2.101:7033", "copy": "example-copy-12"}
    assert sorted(r["inputs"]["time_files_sha256"]) == ["opaque-sonnet/natural/m1-natural-01of02", "opaque-sonnet/prompted/m1-prompted-01of01"]


def test_real_format_positive_t2_attestation_is_canonical_and_consistent(tmp_path, capsys):
    w = attested_world(tmp_path)
    pin, path, doc = make_attestation(w)
    required = {"shard", "tree", "cell", "state", "state_reason", "claim_utc", "seal_utc", "run_end_utc", "rc",
                "map_version", "map_key", "copy", "ep", "ep_sha256", "table_ep_sha256", "time_endpoint_sha256",
                "time_evidence", "time_copy", "class", "class_basis", "basis", "evidence_label", "waiver", "checks",
                "check_reasons", "ok", "reasons", "notes"}
    assert all(set(row) == required for row in doc["shards"])
    assert doc["status"] == "replay_class_attested"
    assert doc["shard_counts"] == {"total": 2, "recorded": 1, "not_recorded": 1, "recorded_not_ok": 0,
                                    "not_recorded_chain_only": 0, "waived": 0}
    fallback = doc["shards"][1]
    assert (fallback["state"], fallback["state_reason"], fallback["basis"], fallback["rc"], fallback["ok"]) == (
        "not_recorded", "first_line_exit_status", "positive_time_endpoint", 1, True)
    assert fallback["time_endpoint_sha256"] == fallback["ep_sha256"] == fallback["table_ep_sha256"]
    assert fallback["time_copy"] == fallback["copy"] and all(fallback["checks"].values())
    # SCHEMA.md: canonicalise the entire object except content_sha256; signing is included.
    canonical = json.dumps({k: v for k, v in doc.items() if k != "content_sha256"},
                           sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    assert doc["content_sha256"] == sha(canonical) and pin == f"{path}:{sha(path.read_bytes())}"
    rc, out, err = w.run("natural,prompted", capsys, extra=("--attestation", pin))
    assert rc == 0, err
    assert routing_doc(w)["shards"]["opaque-sonnet/natural/m1-natural-02of02"]["basis"] == "attested"


def test_attestation_hash_binds_signing_block(tmp_path, capsys):
    w = attested_world(tmp_path)
    _, _, doc = make_attestation(w)
    doc["signing"]["assumption"] = "changed after signing"
    pin, _, _ = make_attestation(w, name="changed-signing.json", **doc)
    rc, _, err = w.run("natural,prompted", capsys, extra=("--attestation", pin))
    assert rc == 2 and err == "REFUSED attestation-content-sha-differs\n"


@pytest.mark.parametrize("mutation,code", [
    ("missing_reason", "attestation-shards-malformed"),
    ("missing_endpoint", "attestation-shards-malformed"),
    ("missing_time_evidence", "attestation-shards-malformed"),
    ("chain_only", "attestation-shards-malformed"),
    ("failed_row_in_attested_phase", "attestation-status-inconsistent"),
    ("wrong_counts", "attestation-counts-inconsistent"),
])
def test_internally_inconsistent_attestation_is_refused(tmp_path, capsys, mutation, code):
    w = attested_world(tmp_path)
    _, _, doc = make_attestation(w)
    row = doc["shards"][1]
    if mutation == "missing_reason":
        row["state_reason"] = "time_file_missing"
    elif mutation == "missing_endpoint":
        row["time_endpoint_sha256"] = None
    elif mutation == "missing_time_evidence":
        row["time_evidence"] = None
    elif mutation == "chain_only":
        row["basis"] = "chain_only"
    elif mutation == "failed_row_in_attested_phase":
        row["checks"]["c1"] = False
        row["ok"] = False
        row["check_reasons"]["c1"] = ["time_endpoint_differs_from_expected"]
        row["reasons"] = ["time_endpoint_differs_from_expected"]
    else:
        doc["shard_counts"]["not_recorded"] = 0
    doc.pop("content_sha256")
    pin, _, _ = make_attestation(w, name="inconsistent.json", **doc)
    rc, out, err = w.run("natural,prompted", capsys, extra=("--attestation", pin))
    assert rc == 2 and err == f"REFUSED {code}\n"


def test_without_an_attestation_the_same_shard_is_not_recorded(tmp_path, capsys):
    w = attested_world(tmp_path)
    rc, out, err = w.run("natural,prompted", capsys)
    assert rc == 0 and routing_doc(w)["records"]["rid-n3"] == "unverifiable" and routing_doc(w)["attested_runs"] == {}
    assert routing_doc(w)["shards"]["opaque-sonnet/natural/m1-natural-02of02"]["subkind"] == "pending_attestation"
    assert "pending_attestation=1" in out


def test_all_recorded_phase_needs_no_attestation_and_has_no_pending_label(tmp_path, capsys):
    w = good_world(tmp_path)
    rc, out, err = w.run("natural,prompted", capsys)
    assert rc == 0, err
    assert all("subkind" not in row for row in routing_doc(w)["shards"].values())
    assert all("pending_attestation=0" in line for line in out.splitlines() if line.startswith("ROUTING "))


def test_an_attestation_never_overrides_a_recorded_launch(tmp_path, capsys):
    # every shard here has its launch recorded: the attestation (of the same copy) adds nothing and labels nothing
    w = good_world(tmp_path)
    pin, _, _ = make_attestation(w)
    rc, out, err = w.run("natural,prompted", capsys, extra=("--attestation", pin))
    assert rc == 0, err
    assert routing_doc(w)["attested_runs"] == {} and set(routing_doc(w)["records"].values()) == {"fast"}


def test_a_recorded_launch_conflicting_with_an_informational_recorded_attestation_row_is_unverifiable(tmp_path, capsys):
    w = good_world(tmp_path)
    # the attestation says copy 7033 (fast); shard 02of02's recorded launch is on siteb-8 (slow, a row of the table): they disagree
    w.write_time("opaque-sonnet", "m1-natural-02of02", "natural", "127.0.0.1:47278", "example-copy-33", rm.utc_epoch(SEALED_UTC) - 5)
    _, _, doc = make_attestation(w)
    row = doc["shards"][1]
    row["checks"]["c1"] = False
    row["ok"] = False
    row["check_reasons"]["c1"] = ["time_endpoint_differs_from_expected"]
    row["reasons"] = ["time_endpoint_differs_from_expected"]
    doc["shard_counts"]["recorded_not_ok"] = 1
    doc.pop("content_sha256")
    pin, _, _ = make_attestation(w, name="conflict.json", **doc)
    rc, out, err = w.run("natural,prompted", capsys, extra=("--attestation", pin))
    assert rc == 0, err
    shards = routing_doc(w)["trees"]["opaque-sonnet"]["shards"]
    assert shards["natural/m1-natural-02of02"] == "unverifiable" and shards["natural/m1-natural-01of02"] == "fast"
    assert routing_doc(w)["attested_runs"] == {}


@pytest.mark.parametrize("field,value", [("copy", "example-copy-33"), ("class", "slow")])
def test_recorded_attestation_copy_or_class_conflict_is_unverifiable(tmp_path, capsys, field, value):
    w = attested_world(tmp_path)
    _, _, doc = make_attestation(w)
    row = doc["shards"][0]
    row[field] = value
    row["checks"]["c4"] = False
    row["ok"] = False
    row["check_reasons"]["c4"] = ["class_row_names_other_copy"]
    row["reasons"] = ["class_row_names_other_copy"]
    doc["shard_counts"]["recorded_not_ok"] = 1
    doc.pop("content_sha256")
    pin, _, _ = make_attestation(w, name="recorded-conflict.json", **doc)
    rc, out, err = w.run("natural,prompted", capsys, extra=("--attestation", pin))
    assert rc == 0, err
    shards = routing_doc(w)["shards"]
    assert shards["opaque-sonnet/natural/m1-natural-01of02"]["class"] == "unverifiable"
    assert shards["opaque-sonnet/natural/m1-natural-02of02"]["basis"] == "attested"


def test_failed_informational_recorded_row_does_not_fail_attested_phase(tmp_path, capsys):
    w = attested_world(tmp_path)
    _, _, doc = make_attestation(w)
    row = doc["shards"][0]
    row["class"] = None  # the producer's class lookup failed on a non-gating recorded row
    row["table_ep_sha256"] = None
    row["checks"]["c4"] = False
    row["check_reasons"]["c4"] = ["copy_not_in_class_table"]
    row["ok"] = False
    row["reasons"] = ["copy_not_in_class_table"]
    doc["shard_counts"]["recorded_not_ok"] = 1
    doc.pop("content_sha256")
    pin, _, doc = make_attestation(w, name="recorded-not-ok.json", **doc)
    assert doc["status"] == "replay_class_attested"
    rc, _, err = w.run("natural,prompted", capsys, extra=("--attestation", pin))
    assert rc == 0, err
    shards = routing_doc(w)["shards"]
    assert shards["opaque-sonnet/natural/m1-natural-01of02"]["class"] == "fast"
    assert shards["opaque-sonnet/natural/m1-natural-02of02"]["basis"] == "attested"


def test_failed_waived_recorded_row_gates_phase(tmp_path, capsys):
    w = attested_world(tmp_path)
    _, _, doc = make_attestation(w)
    row = doc["shards"][0]
    row["waiver"] = "ctime_birth_copy"
    row["checks"]["c4"] = False
    row["check_reasons"]["c4"] = ["copy_not_in_class_table"]
    row["ok"] = False
    row["reasons"] = ["copy_not_in_class_table"]
    doc["shard_counts"].update(recorded_not_ok=1, waived=1)
    doc.pop("content_sha256")
    pin, _, _ = make_attestation(w, name="failed-waiver.json", **doc)
    rc, _, err = w.run("natural,prompted", capsys, extra=("--attestation", pin))
    assert rc == 2 and err == "REFUSED attestation-status-inconsistent\n"


def test_an_attestation_of_a_cell_tree_covers_that_cells_unrecorded_shards_only(tmp_path, capsys):
    w = split_world(tmp_path)
    path = w.root / "rhea-gemma-4-31b-cell-slow" / "logs" / "m1-prompted-01of01.time"
    path.write_text("Command exited with non-zero status 1\n" + path.read_text().replace("Exit status: 0", "Exit status: 1"))
    os.utime(path, (rm.utc_epoch(SEALED_UTC) - 5,) * 2)
    trees = [{"tree": "rhea-gemma-4-31b-cell-fast", "class": "fast", "copy": "example-copy-39", "endpoint": "192.0.2.105:7043"},
             {"tree": "rhea-gemma-4-31b-cell-slow", "class": "slow", "copy": "example-copy-40", "endpoint": "127.0.0.1:47125"}]
    pin, _, _ = make_attestation(w, phase="prompted", trees=trees)
    rc, out, err = w.run("natural,prompted", capsys, extra=("--attestation", pin))
    assert rc == 0, err
    assert routing_doc(w)["records"] == {"rid-n1": "fast", "rid-n2": "fast", "rid-p1": "fast", "rid-p2": "fast", "rid-p3": "slow"}
    assert routing_doc(w)["attested_runs"] == {"rhea-gemma-4-31b-cell-slow/prompted/m1-prompted-01of01": ["rid-p3"]}  # only the slow cell is named


def test_t2_attestation_status_hash_and_per_shard_checks(tmp_path, capsys):
    w = attested_world(tmp_path)
    base, _, doc = make_attestation(w)
    assert base.endswith(sha(Path(base.rsplit(":", 1)[0]).read_bytes()))
    for change, expected in [
        (lambda d: d["shards"][1].update(ok=False), "attestation-shards-malformed"),
        (lambda d: d["shards"][1].update(state="recorded"), "attestation-shards-malformed"),
    ]:
        d = json.loads(json.dumps(doc))
        change(d)
        d.pop("content_sha256")
        pin, _, _ = make_attestation(w, name="changed.json", **d)
        rc, out, err = w.run("natural,prompted", capsys, extra=("--attestation", pin))
        assert rc == 2 and err == f"REFUSED {expected}\n"
    bad, _, _ = make_attestation(w, name="bad-hash.json", content_sha256="0" * 64)
    rc, out, err = w.run("natural,prompted", capsys, extra=("--attestation", bad))
    assert rc == 2 and err == "REFUSED attestation-content-sha-differs\n"
    bad, _, _ = make_attestation(w, name="bad-status.json", status="replay_class_unverified")
    rc, out, err = w.run("natural,prompted", capsys, extra=("--attestation", bad))
    assert rc == 2 and err == "REFUSED attestation-status-unverified\n"


def test_attested_cell_class_must_equal_the_cell(tmp_path, capsys):
    w = split_world(tmp_path)
    tree, shard = "rhea-gemma-4-31b-cell-slow", "m1-prompted-01of01"
    path = w.root / tree / "logs" / f"{shard}.time"
    path.write_text("Command exited with non-zero status 1\n" + path.read_text().replace("Exit status: 0", "Exit status: 1"))
    os.utime(path, (rm.utc_epoch(SEALED_UTC) - 5,) * 2)
    pin, _, doc = make_attestation(w, phase="prompted",
                                   trees=[{"tree": tree, "class": "fast", "copy": "example-copy-39", "endpoint": "192.0.2.105:7043"}])
    rc, out, err = w.run("natural,prompted", capsys, extra=("--attestation", pin))
    assert rc == 0, err
    assert doc["shards"][0]["ok"] is True
    assert routing_doc(w)["shards"][f"{tree}/prompted/{shard}"]["class"] == "unverifiable"


def test_two_recorded_files_cannot_be_overridden_by_one_attested_copy(tmp_path, capsys):
    w = good_world(tmp_path)
    for shard in ("m1-natural-01of02", "m1-natural-02of02"):
        w.write_time("opaque-sonnet", shard, "natural", "127.0.0.1:47278", "example-copy-33",
                     rm.utc_epoch(SEALED_UTC) - 5)
    pin = w.make_manifest()
    att, _, doc = make_attestation(w)
    rc, out, err = w.run("natural,prompted", capsys, extra=("--timefiles-manifest", pin, "--attestation", att))
    assert rc == 0, err
    assert [routing_doc(w)["shards"][f"opaque-sonnet/natural/{shard}"]["class"]
            for shard in ("m1-natural-01of02", "m1-natural-02of02")] == ["unverifiable", "unverifiable"]
    shutil.rmtree(w.stage)
    pin = w.make_manifest(drop=("opaque-sonnet/logs/m1-natural-02of02.time",))
    rc, out, err = w.run("natural,prompted", capsys, extra=("--timefiles-manifest", pin, "--attestation", att))
    assert rc == 0, err
    assert routing_doc(w)["shards"]["opaque-sonnet/natural/m1-natural-02of02"]["subkind"] == "unlisted_time_file"


def test_present_unlisted_file_cannot_use_an_eligible_attestation_row(tmp_path, capsys):
    w = attested_world(tmp_path)
    att, _, doc = make_attestation(w)
    assert doc["shards"][1]["state"] == "not_recorded" and doc["shards"][1]["ok"] is True
    pin = w.make_manifest(drop=("opaque-sonnet/logs/m1-natural-02of02.time",))
    rc, _, err = w.run("natural,prompted", capsys, extra=("--timefiles-manifest", pin, "--attestation", att))
    assert rc == 0, err
    row = routing_doc(w)["shards"]["opaque-sonnet/natural/m1-natural-02of02"]
    assert row["class"] == "unverifiable" and row["basis"] is None and row["subkind"] == "unlisted_time_file"


def test_missing_line_allows_fallback_and_line_two_requires_an_unavailable_rc_record(tmp_path, capsys):
    w = attested_world(tmp_path)
    att, _, _ = make_attestation(w)
    shard = "m1-natural-02of02"
    path = w.root / "opaque-sonnet" / "logs" / f"{shard}.time"
    path.unlink()  # manifest sees MISSING after the signed positive-evidence attestation was made
    rc, out, err = w.run("natural,prompted", capsys, extra=("--attestation", att))
    assert rc == 0, err
    assert "MISSING  online/logs/m1-natural-02of02.time" in (
        w.root / "opaque-sonnet" / "TIMEFILES-natural.sha256").read_text()
    assert routing_doc(w)["shards"]["opaque-sonnet/natural/m1-natural-02of02"]["basis"] == "attested"
    shutil.rmtree(w.stage)
    w.write_time("opaque-sonnet", shard, "natural", "192.0.2.101:7033", "example-copy-12",
                 rm.utc_epoch(SEALED_UTC) - 5, exit_status=1)
    path.write_text("Command exited with non-zero status 1\n" + path.read_text())
    os.utime(path, (rm.utc_epoch(SEALED_UTC) - 5,) * 2)
    pin = w.make_manifest()
    rc, out, err = w.run("natural,prompted", capsys, extra=("--timefiles-manifest", pin))
    assert rc == 0, err
    assert routing_doc(w)["shards"][f"opaque-sonnet/natural/{shard}"]["class"] == "unverifiable"
    shutil.rmtree(w.stage)
    att, _, _ = make_attestation(w, name="line2.json")
    rc, out, err = w.run("natural,prompted", capsys, extra=("--timefiles-manifest", pin, "--attestation", att))
    assert rc == 0, err
    assert routing_doc(w)["shards"][f"opaque-sonnet/natural/{shard}"]["basis"] == "attested"


def test_a_producer_recorded_row_rejected_by_the_consumer_is_unverified(tmp_path, capsys):
    w = good_world(tmp_path)
    shard = "m1-natural-02of02"
    path = w.root / "opaque-sonnet" / "logs" / f"{shard}.time"
    path.write_text('Command being timed: "python scripts/checker_replay.py --endpoint http://192.0.2.101:7033/"\n')
    os.utime(path, (rm.utc_epoch(SEALED_UTC) - 5,) * 2)
    att, _, doc = make_attestation(w)
    assert doc["shards"][1]["state"] == "recorded"
    rc, out, err = w.run("natural,prompted", capsys, extra=("--attestation", att))
    assert rc == 0, err
    row = routing_doc(w)["shards"][f"opaque-sonnet/natural/{shard}"]
    assert row["class"] == "unverifiable" and row["subkind"] == "record_rejected_by_consumer"


def test_the_pin_must_match_the_file_and_be_well_formed(tmp_path, capsys):
    w = attested_world(tmp_path)
    pin, path, _ = make_attestation(w)
    for bad, code in ((f"{path}:{'0' * 64}", "attestation-sha-differs"), (str(path), "attestation-pin-malformed"),
                      (f"{path}:short", "attestation-pin-malformed"), (f":{pin.rsplit(':', 1)[1]}", "attestation-pin-malformed"),
                      (f"{path}.absent:{'0' * 64}", "unreadable")):
        rc, out, err = w.run("natural,prompted", capsys, extra=("--attestation", bad))
        assert rc == 2 and err.startswith(f"REFUSED {code}"), (bad, err)
    # not JSON, with its own sha256 pinned
    path.write_text("not json")
    rc, out, err = w.run("natural,prompted", capsys, extra=("--attestation", f"{path}:{sha(path.read_bytes())}"))
    assert rc == 2 and err == "REFUSED attestation-malformed\n"


def test_an_attestation_for_a_phase_this_call_does_not_check_is_refused(tmp_path, capsys):
    w = natural_only_world(tmp_path)
    pin, _, _ = make_attestation(w, phase="prompted")
    rc, out, err = w.run("natural", capsys, extra=("--attestation", pin))
    assert rc == 2 and err == "REFUSED attestation-names-another-out-phase\n"


def test_two_attestations_for_one_phase_are_refused(tmp_path, capsys):
    w = attested_world(tmp_path)
    a, _, _ = make_attestation(w, "a1.json")
    b, _, _ = make_attestation(w, "a2.json", agent="someone-else")
    rc, out, err = w.run("natural,prompted", capsys, extra=("--attestation", a, "--attestation", b))
    assert rc == 2 and err == "REFUSED attestation-twice-for-a-phase\n"
    c, _, _ = make_attestation(w, "a3.json", phase="prompted")
    rc, out, err = w.run("natural,prompted", capsys, extra=("--attestation", a, "--attestation", c))  # one per phase: fine
    assert rc == 0, err


def test_the_global_attestation_flag_is_gone(tmp_path, capsys):
    w = good_world(tmp_path)
    with pytest.raises(SystemExit):
        w.run("natural,prompted", capsys, extra=("--attest-main-tree-routing",))
    assert capsys.readouterr().err == "REFUSED bad-arguments\n"


def test_a_json_object_that_repeats_a_key_is_refused_in_an_attestation(tmp_path, capsys):
    # external reviewer r17: json.loads keeps the last of two equal keys, so two conflicting `out` values (or conditions) passed
    w = attested_world(tmp_path)
    pin, path, doc = make_attestation(w)
    text = path.read_text()
    for name, bad in (("top", text.replace('"out": "opaque-sonnet"', '"out": "opaque-haiku", "out": "opaque-sonnet"', 1)),
                      ("nested", text.replace('"ok": true', '"ok": false, "ok": true', 1)),
                      ("tree", text.replace('"class": "fast"', '"class": "slow", "class": "fast"', 1))):
        assert bad != text, name
        path.write_text(bad)
        rc, out, err = w.run("natural,prompted", capsys, extra=("--attestation", f"{path}:{sha(path.read_bytes())}"))
        assert rc == 2 and err == "REFUSED attestation-malformed\n", (name, err)
        assert not w.stage.exists()
    path.write_text(text)  # the original, unrepeated file is still accepted
    rc, out, err = w.run("natural,prompted", capsys, extra=("--attestation", pin))
    assert rc == 0, err


# -- the sha manifest of the time files and the basis of a class (analysis lead, 12:04Z) -----------------------------------------


def test_the_basis_of_each_classed_shard(tmp_path, capsys):
    w = split_world(tmp_path)
    rc, out, err = w.run(capsys=capsys)
    assert rc == 0, err
    shards = routing_doc(w)["shards"]
    assert {k: v["basis"] for k, v in shards.items()} == {
        "rhea-gemma-4-31b/natural/m1-natural-01of01": "recorded",  # a whole-phase tree: the worker's launch record
        "rhea-gemma-4-31b-cell-fast/prompted/m1-prompted-01of01": "proven",  # a class cell whose recorded copy has the cell's class
        "rhea-gemma-4-31b-cell-slow/prompted/m1-prompted-01of01": "proven"}
    assert routing_doc(w)["bases"] == {"rid-n1": "recorded", "rid-n2": "recorded", "rid-p1": "proven", "rid-p2": "proven", "rid-p3": "proven"}
    m = routing_doc(w)["inputs"]["timefiles_manifest"]
    assert set(m) == {"file_sha256", "taken_at", "manifests"}


def test_a_time_file_the_manifest_does_not_list_is_as_if_absent_and_the_fallback_may_apply(tmp_path, capsys):
    w = attested_world(tmp_path)  # 02of02 has no time file
    pin = w.make_manifest(drop=("opaque-sonnet/logs/m1-natural-01of02.time",))  # 01of02's file exists but is not listed
    rc, out, err = w.run("natural,prompted", capsys, extra=("--timefiles-manifest", pin))
    assert rc == 0, err
    shards = routing_doc(w)["shards"]
    assert shards["opaque-sonnet/natural/m1-natural-01of02"]["class"] == "unverifiable"
    assert shards["opaque-sonnet/natural/m1-natural-02of02"]["class"] == "unverifiable"
    att, _, _ = make_attestation(w)
    shutil.rmtree(w.stage)
    rc, out, err = w.run("natural,prompted", capsys, extra=("--timefiles-manifest", pin, "--attestation", att))
    assert rc == 0, err
    shards = routing_doc(w)["shards"]
    assert shards["opaque-sonnet/natural/m1-natural-01of02"]["class"] == "unverifiable" and shards["opaque-sonnet/natural/m1-natural-02of02"]["basis"] == "attested"
    assert shards["opaque-sonnet/prompted/m1-prompted-01of01"]["basis"] == "recorded"  # listed and unchanged: the fallback does not touch it


def test_a_time_file_that_differs_from_the_manifest_is_unverifiable_and_never_attested(tmp_path, capsys):
    w = good_world(tmp_path)
    pin = w.make_manifest(override={"opaque-sonnet/logs/m1-natural-02of02.time": "0" * 64})
    att, _, _ = make_attestation(w)
    rc, out, err = w.run("natural,prompted", capsys, extra=("--timefiles-manifest", pin, "--attestation", att))
    assert rc == 0, err
    s = routing_doc(w)["shards"]["opaque-sonnet/natural/m1-natural-02of02"]
    assert s["class"] == "unverifiable" and s["basis"] is None  # changed since the manifest was taken: no fallback


def test_a_cells_recorded_copy_of_another_class_is_unverifiable(tmp_path, capsys):
    w = split_world(tmp_path)
    w.write_time("rhea-gemma-4-31b-cell-fast", "m1-prompted-01of01", "prompted", "127.0.0.1:47125", "example-copy-40", rm.utc_epoch(SEALED_UTC) - 5)
    rc, out, err = w.run(capsys=capsys)
    assert rc == 0, err
    s = routing_doc(w)["shards"]["rhea-gemma-4-31b-cell-fast/prompted/m1-prompted-01of01"]
    assert s["class"] == "unverifiable" and s["basis"] is None  # the worker's gate by class failed: an exception, not routed by the record


def test_a_symlinked_logs_dir_is_not_the_workers(tmp_path, capsys):
    w = good_world(tmp_path)
    logs = w.root / "opaque-sonnet" / "logs"
    real = w.root / "opaque-sonnet" / "real-logs"
    os.rename(logs, real)
    os.symlink(real, logs)
    rc, out, err = w.run(capsys=capsys)  # the manifest (built through the link) lists the files; they are not plain files of a real logs/
    assert rc == 0, err
    assert {i["class"] for i in routing_doc(w)["shards"].values()} == {"unverifiable"}
    shutil.rmtree(w.stage)
    pin = w.make_manifest(drop=tuple(f"opaque-sonnet/logs/{f.name}" for f in real.glob("*.time")),
                         override={"opaque-sonnet/logs/other-shard.time": "a" * 64})  # unlisted: as if absent (a manifest lists something)
    rc, out, err = w.run(capsys=capsys, extra=("--timefiles-manifest", pin))
    assert rc == 0, err
    assert {i["class"] for i in routing_doc(w)["shards"].values()} == {"unverifiable"}


def test_real_format_index_and_manifest_pins(tmp_path, capsys):
    w = good_world(tmp_path)
    pin = w.make_manifest()
    index = Path(pin.rsplit(":", 1)[0])
    assert "\twritten\n" in index.read_text()
    manifest = w.root / "opaque-sonnet" / "TIMEFILES-natural.sha256"
    assert "  online/logs/m1-natural-01of02.time" in manifest.read_text()
    rc, out, err = w.run("natural,prompted", capsys, extra=("--timefiles-manifest", pin))
    assert rc == 0, err
    shutil.rmtree(w.stage)
    manifest.write_text(manifest.read_text() + "# changed\n")
    rc, out, err = w.run("natural,prompted", capsys, extra=("--timefiles-manifest", pin))
    assert rc == 2 and err == "REFUSED timefiles-manifest-sha-differs\n"


def test_the_manifest_is_required(tmp_path, capsys):
    w = good_world(tmp_path)
    with pytest.raises(SystemExit):
        rm.main(["check", "--root", str(w.root), "--done-dir", str(w.done), "--cells-tsv", str(w.cells), "--out-dir", str(w.out),
                 "--phases", "natural,prompted", "--stage", str(w.stage), "--cell-classes", str(w.cell_classes)])
    assert capsys.readouterr().err == "REFUSED bad-arguments\n"


def test_a_listed_time_file_that_changed_into_something_unparseable_is_unverifiable_and_never_attested(tmp_path, capsys):
    # external reviewer r18: the manifest comparison is by bytes, before any parsing: a changed file that no longer parses must not fall back
    w = good_world(tmp_path)
    pin = w.make_manifest()  # the manifest as taken: lists 02of02's real file
    (w.root / "opaque-sonnet" / "logs" / "m1-natural-02of02.time").write_text("garbage\n")
    att, _, _ = make_attestation(w)
    rc, out, err = w.run("natural,prompted", capsys, extra=("--timefiles-manifest", pin, "--attestation", att))
    assert rc == 0, err
    s = routing_doc(w)["shards"]["opaque-sonnet/natural/m1-natural-02of02"]
    assert s["class"] == "unverifiable" and s["basis"] is None and s["copy"] is None
    # a listed file that is gone altogether: the same
    shutil.rmtree(w.stage)
    (w.root / "opaque-sonnet" / "logs" / "m1-natural-02of02.time").unlink()
    rc, out, err = w.run("natural,prompted", capsys, extra=("--timefiles-manifest", pin, "--attestation", att))
    assert rc == 0, err
    assert routing_doc(w)["shards"]["opaque-sonnet/natural/m1-natural-02of02"]["class"] == "unverifiable"


def test_an_unlisted_present_time_file_is_unverified_even_with_an_attestation(tmp_path, capsys):
    w = good_world(tmp_path)
    pin = w.make_manifest(drop=("opaque-sonnet/logs/m1-natural-02of02.time",))
    att, _, _ = make_attestation(w)
    rc, out, err = w.run("natural,prompted", capsys, extra=("--timefiles-manifest", pin, "--attestation", att))
    assert rc == 0, err
    r = routing_doc(w)
    assert r["shards"]["opaque-sonnet/natural/m1-natural-02of02"]["class"] == "unverifiable"
    assert r["shards"]["opaque-sonnet/natural/m1-natural-02of02"]["subkind"] == "unlisted_time_file"
    assert "opaque-sonnet/natural/m1-natural-02of02" not in r["inputs"]["time_files_sha256"]


def test_every_classed_shard_names_the_copy_it_rests_on(tmp_path, capsys):
    w = split_world(tmp_path)
    rc, out, err = w.run(capsys=capsys)
    assert rc == 0, err
    c = {k: v["copy"] for k, v in routing_doc(w)["shards"].items()}
    assert c == {"rhea-gemma-4-31b/natural/m1-natural-01of01": {"endpoint": "192.0.2.105:7043", "copy": "example-copy-39"},
                 "rhea-gemma-4-31b-cell-fast/prompted/m1-prompted-01of01": {"endpoint": "192.0.2.105:7043", "copy": "example-copy-39"},
                 "rhea-gemma-4-31b-cell-slow/prompted/m1-prompted-01of01": {"endpoint": "127.0.0.1:47125", "copy": "example-copy-40"}}


def test_a_listed_time_file_is_read_once_so_the_manifest_comparison_and_the_parse_see_the_same_bytes(tmp_path, capsys, monkeypatch):
    # external reviewer r19: two reads left a window in which the file could match the manifest and then parse as unusable (and fall back)
    w = good_world(tmp_path)
    calls = []
    real = rm.load_time_file

    def counting(root, tree, shard):
        calls.append((tree, shard))
        return real(root, tree, shard)
    monkeypatch.setattr(rm, "load_time_file", counting)
    rc, out, err = w.run("natural,prompted", capsys)
    assert rc == 0, err
    assert sorted(calls) == [("opaque-sonnet", "m1-natural-01of02"), ("opaque-sonnet", "m1-natural-02of02"), ("opaque-sonnet", "m1-prompted-01of01")]
    # a file whose bytes equal the manifest's but which does not parse as this shard's launch is not usable (the fallback may apply)...
    w2 = good_world(tmp_path / "second")
    path = w2.root / "opaque-sonnet" / "logs" / "m1-natural-02of02.time"
    path.write_text("garbage that the manifest was taken of\n")
    rc, out, err = w2.run("natural,prompted", capsys)  # the manifest is taken of the file as it now is
    assert rc == 0, err
    assert routing_doc(w2)["shards"]["opaque-sonnet/natural/m1-natural-02of02"]["class"] == "unverifiable"
    # ...but if the bytes the loader returns differ from the listed ones (a change between a stat and the read), it is unverifiable
    shutil.rmtree(w2.stage)
    pin = w2.make_manifest()
    monkeypatch.setattr(rm, "load_time_file", lambda r, t, sh: (b"changed\n", 0.0, sha(b"changed\n")) if sh == "m1-natural-02of02" else real(r, t, sh))
    att, _, _ = make_attestation(w2)
    rc, out, err = w2.run("natural,prompted", capsys, extra=("--timefiles-manifest", pin, "--attestation", att))
    assert rc == 0, err
    assert routing_doc(w2)["shards"]["opaque-sonnet/natural/m1-natural-02of02"]["class"] == "unverifiable"


def test_live_date_still_gates_a_manifest_matching_launch(tmp_path, capsys):
    w = good_world(tmp_path)
    pin = w.make_manifest()
    path = w.root / "opaque-sonnet" / "logs" / "m1-natural-02of02.time"
    os.utime(path, (rm.utc_epoch(SEALED_UTC) - 400,) * 2)
    rc, out, err = w.run("natural,prompted", capsys, extra=("--timefiles-manifest", pin))
    assert rc == 0, err
    assert routing_doc(w)["records"]["rid-n3"] == "unverifiable"


def test_the_date_windows_are_compared_in_integer_nanoseconds():
    # external reviewer r22: a date one nanosecond before the claim was rounded onto it by a float conversion and accepted
    seal = rm.utc_epoch(SEALED_UTC)
    sealed = round(seal)
    claim_ns = (sealed - 100) * 10**9 + 123_456_789  # a 2026 epoch is ~1.8e18 ns: far beyond a float's 2**53 of exact integers
    data = (f'\tCommand being timed: "{launch_command("t", "s", "natural", "192.0.2.101:7033", "example-copy-12")}"\n\tExit status: 0\n').encode()
    assert claim_ns > 2**53
    ok = rm.parse_launch_record(data, claim_ns, "t", "s", "natural", sealed, claim_ns)
    assert ok == ("192.0.2.101:7033", "example-copy-12")  # exactly at the claim
    assert rm.parse_launch_record(data, claim_ns - 1, "t", "s", "natural", sealed, claim_ns) is None  # one nanosecond before it
    assert rm.parse_launch_record(data, claim_ns + 1, "t", "s", "natural", sealed, claim_ns) is not None
    # the seal window: [seal - 300 s, seal + 60 s], exact at both ends
    lo, hi = (sealed - rm.TIME_BEFORE_SEAL_S) * 10**9, (sealed + rm.TIME_AFTER_SEAL_S) * 10**9
    assert rm.parse_launch_record(data, lo, "t", "s", "natural", sealed, None) is not None
    assert rm.parse_launch_record(data, lo - 1, "t", "s", "natural", sealed, None) is None
    assert rm.parse_launch_record(data, hi, "t", "s", "natural", sealed, None) is not None
    assert rm.parse_launch_record(data, hi + 1, "t", "s", "natural", sealed, None) is None


def test_the_claim_is_read_in_integer_nanoseconds(tmp_path):
    w = good_world(tmp_path)
    claim = w.root / "claims" / "opaque-sonnet.m1-natural-02of02.online" / "worker"
    ns = (round(rm.utc_epoch(SEALED_UTC)) - 100) * 10**9 + 7
    os.utime(claim, ns=(ns, ns))
    got = rm.claim_started(w.root, "opaque-sonnet", "m1-natural-02of02", rm.utc_epoch(SEALED_UTC))
    assert got == ns and isinstance(got, int)
    os.utime(claim, ns=((round(rm.utc_epoch(SEALED_UTC)) + 1) * 10**9,) * 2)  # after the seal
    assert rm.claim_started(w.root, "opaque-sonnet", "m1-natural-02of02", rm.utc_epoch(SEALED_UTC)) is None


def test_a_file_that_changes_under_the_read_is_not_loaded(tmp_path, monkeypatch):
    w = good_world(tmp_path)
    logs = w.root / "opaque-sonnet" / "logs"
    assert rm.load_time_file(w.root, "opaque-sonnet", "m1-natural-02of02") is not None
    real_fstat = os.fstat
    calls = {"n": 0}

    def shifting(fd):
        st = real_fstat(fd)
        calls["n"] += 1
        if calls["n"] % 2 == 0:  # the second fstat (after the read) sees a newer mtime
            class S:
                st_mode, st_size, st_mtime_ns = st.st_mode, st.st_size, st.st_mtime_ns + 1_000_000_000
            return S
        return st
    monkeypatch.setattr(os, "fstat", shifting)
    assert rm.load_time_file(w.root, "opaque-sonnet", "m1-natural-02of02") is None
    monkeypatch.undo()
    # a symlinked file and a symlinked logs dir are not loaded
    real = logs / "m1-natural-02of02.time"
    os.rename(real, logs / "elsewhere.time")
    os.symlink(logs / "elsewhere.time", real)
    assert rm.load_time_file(w.root, "opaque-sonnet", "m1-natural-02of02") is None


# -- the consolidated union of the time-file sha manifest indexes (analysis lead 5, 25 Oct 18:57Z) ----------------------------------
# The producer took its indexes incrementally (12:32Z, 15:31Z, ...), each listing only its own tree-phases, and consolidated them
# into ONE file so that a cell whose tree-phases were taken at different times has one pin. The layout below is written out by
# hand from the producer's real file (marker comment, twelve-column header comment, one row per tree-phase, the one sources line).

TAG_A, TAKEN_A = "20261025T123234Z", "2026-10-25T12:32:38Z"
TAG_B, TAKEN_B = "20261025T153123Z", "2026-10-25T15:31:25Z"
TAG_C = "20261025T162644Z"
ALL_NAME = "TIMEFILES-INDEX-ALL-20261025T185104Z.tsv"
ALL_MARK = ("# CONSOLIDATED union of the Tier 2 time-file sha manifest indexes (derived mechanically by tier2-replay-5 from the index "
            "files below; NOT a new measurement; each row keeps its own index tag and taken-at time; a tree-phase appears once)")
ALL_HEADER = ("# source_index_tag\ttaken_at_utc\ttree\tphase\tshards\tonline_hashed\tonline_missing\tobqc_hashed\tobqc_missing\t"
              "obqc_na\tmanifest_sha256\tstatus")
SRC_COLUMNS = "# tree\tphase\tshards\tonline_hashed\tonline_missing\tobqc_hashed\tobqc_missing\tobqc_na\tmanifest_sha256\tstatus"
OTHER_ROW = "other-tree\tnatural\t1\t1\t0\t0\t0\t1\t" + "a" * 64 + "\twritten"  # a well-formed row of a tree the world does not have


class Consolidated:
    """good_world's two tree-phases, opaque-sonnet natural and opaque-sonnet prompted, in two incremental indexes as the producer
    writes them (A: natural, taken 12:32:38Z; B: prompted, taken 15:31:25Z), and the union file of both written by hand."""

    def __init__(self, w: World) -> None:
        self.w = w
        single = Path(w.make_manifest().rsplit(":", 1)[0])  # writes the per-tree-phase manifests; only its two rows are used
        rows = [line for line in single.read_text().splitlines() if line and not line.startswith("#")]
        single.unlink()
        assert [r.split("\t")[:2] for r in rows] == [["opaque-sonnet", "natural"], ["opaque-sonnet", "prompted"]]
        self.nat, self.pro = rows
        self.dir = w.stage.parent / "indexes"
        self.dir.mkdir()
        self.sha: dict[str, str] = {}
        self.put_source(TAG_A, self.source_text(TAKEN_A, self.nat))
        self.put_source(TAG_B, self.source_text(TAKEN_B, self.pro))

    @staticmethod
    def source_text(taken: str, *rows: str) -> str:
        return (f"# Tier 2 time-file sha manifests index, taken {taken} (after the fact; guards only against changes from now on)\n"
                + SRC_COLUMNS + "\n" + "".join(r + "\n" for r in rows))

    def put_source(self, tag: str, text: str) -> str:
        path = self.dir / f"TIMEFILES-INDEX-{tag}.tsv"
        path.write_text(text)
        self.sha[tag] = sha(text.encode())
        return self.sha[tag]

    def rows(self) -> list[str]:
        return [f"{TAG_A}\t{TAKEN_A}\t{self.nat}", f"{TAG_B}\t{TAKEN_B}\t{self.pro}"]

    def sources_line(self, tags=(TAG_A, TAG_B)) -> str:
        return "# sources: " + "; ".join(f"TIMEFILES-INDEX-{t}.tsv sha256 {self.sha.get(t, 'c' * 64)}" for t in tags)

    def text(self, rows=None, sources=None, mark=ALL_MARK, header=ALL_HEADER) -> str:
        rows = self.rows() if rows is None else rows
        return "".join(line + "\n" for line in [mark, header, *rows, self.sources_line() if sources is None else sources])

    def pin(self, text: str | None = None, name: str = ALL_NAME) -> str:
        data = (self.text() if text is None else text).encode()
        (self.dir / name).write_bytes(data)
        return f"{self.dir / name}:{sha(data)}"

    def run(self, pin: str, capsys):
        return self.w.run("natural,prompted", capsys, extra=("--timefiles-manifest", pin))


def test_the_consolidated_index_covers_a_cell_whose_tree_phases_come_from_two_indexes(tmp_path, capsys):
    w = good_world(tmp_path)
    c = Consolidated(w)
    for tag in (TAG_A, TAG_B):  # each incremental index alone lists one of the cell's two tree-phases: the other has no manifest
        src = c.dir / f"TIMEFILES-INDEX-{tag}.tsv"
        rc, out, err = c.run(f"{src}:{sha(src.read_bytes())}", capsys)
        assert rc == 2 and err == "REFUSED timefiles-manifest-phase-missing\n", err
    pin = c.pin()
    rc, out, err = c.run(pin, capsys)
    assert rc == 0, err
    all_sha = pin.rsplit(":", 1)[1]
    assert (f"TIMEFILES-INDEX kind=consolidated sha256={all_sha} tree_phases=2 sources=2 sources_verified=2 sources_not_beside=0\n"
            in out)
    nat = sha((w.root / "opaque-sonnet" / "TIMEFILES-natural.sha256").read_bytes())
    pro = sha((w.root / "opaque-sonnet" / "TIMEFILES-prompted.sha256").read_bytes())
    assert routing_doc(w)["inputs"]["timefiles_manifest"] == {
        "file_sha256": all_sha, "taken_at": "20261025T185104Z",  # the union file's own tag: a consolidation time, not a row's
        "manifests": [{"tree": "opaque-sonnet", "phase": "natural", "file_sha256": nat, "source_index": TAG_A, "taken_at": TAKEN_A},
                      {"tree": "opaque-sonnet", "phase": "prompted", "file_sha256": pro, "source_index": TAG_B, "taken_at": TAKEN_B}],
        "sources": [{"tag": TAG_A, "file_sha256": c.sha[TAG_A], "verified": True},
                    {"tag": TAG_B, "file_sha256": c.sha[TAG_B], "verified": True}]}
    # the classing is the single index's: the time files are checked against the manifests the union row pinned
    assert {i["basis"] for i in routing_doc(w)["shards"].values()} == {"recorded"}
    assert {i["class"] for i in routing_doc(w)["shards"].values()} == {"fast"}


def test_a_single_index_keeps_its_routing_shape_and_prints_no_index_line(tmp_path, capsys):
    w = good_world(tmp_path)
    rc, out, err = w.run("natural,prompted", capsys)
    assert rc == 0, err
    m = routing_doc(w)["inputs"]["timefiles_manifest"]
    assert set(m) == {"file_sha256", "taken_at", "manifests"} and m["taken_at"] == "20261025T123234Z"
    assert [set(e) for e in m["manifests"]] == [{"tree", "phase", "file_sha256"}] * 2
    assert "TIMEFILES-INDEX" not in out


def test_the_consolidated_pin_still_pins_every_tree_phase_manifest_and_its_own_bytes(tmp_path, capsys):
    w = good_world(tmp_path)
    c = Consolidated(w)
    pin = c.pin()
    path = Path(pin.rsplit(":", 1)[0])
    path.write_text(path.read_text().replace("\twritten\n", "\twritten \n", 1))  # the file changed after it was pinned
    rc, out, err = c.run(pin, capsys)
    assert rc == 2 and err == "REFUSED timefiles-manifest-sha-differs\n"
    pin = c.pin()
    manifest = w.root / "opaque-sonnet" / "TIMEFILES-prompted.sha256"
    manifest.write_text(manifest.read_text() + "# changed\n")  # a tree-phase manifest the union row pins
    rc, out, err = c.run(pin, capsys)
    assert rc == 2 and err == "REFUSED timefiles-manifest-sha-differs\n"


def _malformed_variants(c: Consolidated) -> dict:
    a, b = c.rows()
    swap = lambda row, old, new: row.replace(old, new, 1)  # noqa: E731
    sources = c.sources_line()
    full = c.text()
    return {
        "tree-phase-listed-in-two-indexes": c.text(rows=[a, b, f"{TAG_B}\t{TAKEN_B}\t{c.nat}"]),
        "tree-phase-listed-twice-in-one-index": c.text(rows=[a, a, b]),
        "row-with-eleven-columns": c.text(rows=[a.rsplit("\t", 1)[0], b]),
        "row-with-thirteen-columns": c.text(rows=[a + "\textra", b]),
        "taken-at-without-seconds": c.text(rows=[swap(a, TAKEN_A, "2026-10-25T12:32Z"), b]),
        "taken-at-month-13": c.text(rows=[swap(a, TAKEN_A, "2026-13-25T12:32:38Z"), b]),
        "taken-at-on-another-date-than-its-tag": c.text(rows=[swap(a, TAKEN_A, "2026-10-26T12:32:38Z"), b]),
        "one-index-with-two-taken-at-times": c.text(rows=[a, b, f"{TAG_B}\t2026-10-25T15:31:26Z\t{OTHER_ROW}"]),
        "unknown-status": c.text(rows=[swap(a, "\twritten", "\tweird"), b]),
        "non-numeric-count": c.text(rows=[swap(a, f"\t{c.nat.split(chr(9))[2]}\t", "\tx\t"), b]),
        "short-manifest-sha": c.text(rows=[a.replace(c.nat.split("\t")[8], "abc"), b]),
        "tag-not-a-tag": c.text(rows=[swap(a, TAG_A, "20261025"), b]),
        "blank-line-among-rows": c.text(rows=[a, "", b]),
        "comment-line-among-rows": c.text(rows=[a, "# another note", b]),
        "no-marker-comment": c.text(mark="# some other file"),
        "marker-comment-unclosed": c.text(mark=ALL_MARK[:-1]),
        "columns-header-reordered": c.text(header=ALL_HEADER.replace("tree\tphase", "phase\ttree")),
        "no-final-newline": full[:-1],
        "no-sources-line": "".join(line + "\n" for line in [ALL_MARK, ALL_HEADER, a, b]),
        "sources-line-lists-an-uncited-tag": c.text(sources=c.sources_line((TAG_A, TAG_B, TAG_C))),
        "sources-line-omits-a-cited-tag": c.text(sources=c.sources_line((TAG_A,))),
        "sources-line-repeats-a-tag": c.text(sources=c.sources_line((TAG_A, TAG_B, TAG_B))),
        "sources-line-with-a-short-sha": c.text(sources=sources.replace(c.sha[TAG_B], c.sha[TAG_B][:60])),
        "sources-line-not-last": "".join(line + "\n" for line in [ALL_MARK, ALL_HEADER, a, sources, b]),
        "no-rows": c.text(rows=[]),
        "crlf-line-ends": full.replace("\n", "\r\n"),
        "row-of-the-third-tag-uncited": c.text(rows=[a, b, f"{TAG_C}\t2026-10-25T16:26:45Z\t{OTHER_ROW}"]),
        "second-sources-line": c.text(rows=[a, b, sources]),
    }


MALFORMED_CASES = (
    "tree-phase-listed-in-two-indexes", "tree-phase-listed-twice-in-one-index", "row-with-eleven-columns",
    "row-with-thirteen-columns", "taken-at-without-seconds", "taken-at-month-13", "taken-at-on-another-date-than-its-tag",
    "one-index-with-two-taken-at-times", "unknown-status", "non-numeric-count", "short-manifest-sha", "tag-not-a-tag",
    "blank-line-among-rows", "comment-line-among-rows", "no-marker-comment", "marker-comment-unclosed",
    "columns-header-reordered", "no-final-newline", "no-sources-line", "sources-line-lists-an-uncited-tag",
    "sources-line-omits-a-cited-tag", "sources-line-repeats-a-tag", "sources-line-with-a-short-sha", "sources-line-not-last",
    "no-rows", "crlf-line-ends", "row-of-the-third-tag-uncited", "second-sources-line")


@pytest.mark.parametrize("case", MALFORMED_CASES)
def test_a_consolidated_index_that_is_not_exactly_the_producers_layout_is_refused(tmp_path, capsys, case):
    w = good_world(tmp_path)
    c = Consolidated(w)
    text = _malformed_variants(c)[case]
    rc, out, err = c.run(c.pin(text), capsys)
    assert rc == 2 and err == "REFUSED timefiles-manifest-malformed\n", (case, err)
    assert not w.stage.exists()  # refused before the stage is made


def test_the_consolidated_name_and_its_layout_must_agree(tmp_path, capsys):
    w = good_world(tmp_path)
    c = Consolidated(w)
    single = c.source_text(TAKEN_A, c.nat, c.pro)  # the ten-column layout
    rc, out, err = c.run(c.pin(single), capsys)  # under the consolidated name
    assert rc == 2 and err == "REFUSED timefiles-manifest-malformed\n", err
    rc, out, err = c.run(c.pin(c.text(), name="TIMEFILES-INDEX-20261025T185104Z.tsv"), capsys)  # the consolidated layout under a single name
    assert rc == 2 and err == "REFUSED timefiles-manifest-malformed\n", err
    for name in ("TIMEFILES-INDEX-ALL.tsv", "TIMEFILES-INDEX-ALL-20261025T1851Z.tsv", "TIMEFILES-INDEX-ALL-20261025T185104Z.txt",
                 "TIMEFILES-INDEX-ALL-20261025T185104Z.tsv.bak", "xTIMEFILES-INDEX-ALL-20261025T185104Z.tsv"):
        rc, out, err = c.run(c.pin(name=name), capsys)
        assert rc == 2 and err == "REFUSED timefiles-manifest-malformed\n", (name, err)
    assert not w.stage.exists()


def test_a_source_index_beside_the_union_must_agree_with_it(tmp_path, capsys):
    def case(tag, mutate, expect):
        w = good_world(tmp_path / tag)
        c = Consolidated(w)
        mutate(c)
        rc, out, err = c.run(c.pin(), capsys)
        assert rc == 2 and err == f"REFUSED {expect}\n", (tag, err)
        assert not w.stage.exists()

    differs = "timefiles-manifest-source-differs"
    # the source file changed after the union was made: its sha256 is not the one on the sources line
    case("source-bytes-changed", lambda c: (c.dir / f"TIMEFILES-INDEX-{TAG_A}.tsv").write_text(
        (c.dir / f"TIMEFILES-INDEX-{TAG_A}.tsv").read_text() + "# note\n"), differs)

    def other_count(c):  # the source has another count in its row; the sources line carries the new sha, so only the rows differ
        row = c.nat.split("\t")
        row[2] = str(int(row[2]) + 1)
        c.put_source(TAG_A, c.source_text(TAKEN_A, "\t".join(row)))
    case("source-row-differs", other_count, differs)
    case("source-has-a-row-the-union-dropped", lambda c: c.put_source(TAG_A, c.source_text(TAKEN_A, c.nat, OTHER_ROW)), differs)
    case("source-lacks-a-row-the-union-has", lambda c: c.put_source(TAG_A, c.source_text(TAKEN_A)), differs)
    case("source-taken-at-is-not-the-rows", lambda c: c.put_source(TAG_A, c.source_text("2026-10-25T12:33:38Z", c.nat)), differs)
    case("source-without-the-taken-at-header", lambda c: c.put_source(
        TAG_A, "# tree-phase manifests\n" + SRC_COLUMNS + "\n" + c.nat + "\n"), differs)

    def to_symlink(c):  # a source that is there but is no regular file is no absent source
        path = c.dir / f"TIMEFILES-INDEX-{TAG_A}.tsv"
        real = c.dir / "elsewhere.tsv"
        path.rename(real)
        path.symlink_to(real)
    w = good_world(tmp_path / "source-is-a-symlink")
    c = Consolidated(w)
    to_symlink(c)
    rc, out, err = c.run(c.pin(), capsys)
    assert rc == 2 and err.startswith("REFUSED unreadable"), err


def test_a_source_index_that_is_not_beside_the_union_is_reported_as_not_compared(tmp_path, capsys):
    w = good_world(tmp_path)
    c = Consolidated(w)
    pin = c.pin()
    (c.dir / f"TIMEFILES-INDEX-{TAG_B}.tsv").unlink()
    rc, out, err = c.run(pin, capsys)
    assert rc == 0, err
    all_sha = pin.rsplit(":", 1)[1]
    assert (f"TIMEFILES-INDEX kind=consolidated sha256={all_sha} tree_phases=2 sources=2 sources_verified=1 sources_not_beside=1\n"
            in out)
    assert routing_doc(w)["inputs"]["timefiles_manifest"]["sources"] == [
        {"tag": TAG_A, "file_sha256": c.sha[TAG_A], "verified": True}, {"tag": TAG_B, "file_sha256": c.sha[TAG_B], "verified": False}]
    # the row's own time still comes from the union's row
    assert [(e["source_index"], e["taken_at"]) for e in routing_doc(w)["inputs"]["timefiles_manifest"]["manifests"]] == [
        (TAG_A, TAKEN_A), (TAG_B, TAKEN_B)]


def test_a_union_row_the_producer_marked_refused_pins_no_manifest_as_in_a_single_index(tmp_path, capsys):
    w = good_world(tmp_path)
    c = Consolidated(w)
    a, b = c.rows()
    refused = b.replace("\twritten", "\trefused:exists-differs")  # a row that pins no manifest, as in a single index
    c.put_source(TAG_B, c.source_text(TAKEN_B, c.pro.replace("\twritten", "\trefused:exists-differs")))
    rc, out, err = c.run(c.pin(c.text(rows=[a, refused])), capsys)
    assert rc == 2 and err == "REFUSED timefiles-manifest-phase-missing\n", err
