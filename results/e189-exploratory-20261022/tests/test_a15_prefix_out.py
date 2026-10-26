"""Tests of driver/a15_prefix_out.py, the names-only subset OUT builder. Synthetic OUTs only; expected values are written by hand."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import stat
import sys
from pathlib import Path

import pytest

DRIVER = Path(__file__).resolve().parents[1] / "driver" / "a15_prefix_out.py"
spec = importlib.util.spec_from_file_location("a15_prefix_out_under_test", DRIVER)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

UUID = "01234567-89ab-cdef-0123-456789abcdef"
NAME = "synthetic-p0of3"
AT = "2026-10-26T08:31:00Z"
EARLY, LATE = 1_792_000_000, 1_794_000_000  # before and after AT (2026-10-26T08:31:00Z = 1_793_003_460)


def digest(n: int) -> str:
    return f"{n:040x}"


def rec(n: int) -> str:
    return f"{digest(n)}.{UUID}.json"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class Out:
    """An OUT with records 1..5 (1-3 final early, 4 final late, 5 an open attempt: claim files only), one try file for record 2 and one for 5."""

    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.src = tmp / "runs" / NAME
        for d in ("records", "claims", "infrastructure", "holds"):
            (self.src / d).mkdir(parents=True)
        for n in range(1, 6):
            (self.src / "claims" / f"{digest(n)}.claim").write_text(f"claim {n}")
            (self.src / "claims" / f"{digest(n)}.attempt").write_text(f"attempt {n}")
        for n in range(1, 5):
            f = self.src / "records" / rec(n)
            f.write_text(f"record {n}")
            os.utime(f, (EARLY if n < 4 else LATE,) * 2)
        (self.src / "infrastructure" / f"{digest(2)}.{UUID.replace('0', '1', 1)}.json").write_text("try")
        (self.src / "infrastructure" / f"{digest(5)}.{UUID}.json").write_text("open try")
        self.top = {"manifest.json": b'{"m": 1}', "served-models.json": b'{"s": 1}', "order.json": b'{"o": 1}',
                    "concurrency-events.jsonl": b"e1\n", "admitted-fingerprints.json": b"{}", "model-fronts.json": b"{}"}
        for name, data in self.top.items():
            (self.src / name).write_bytes(data)
        for name in ("runs.jsonl", "runner.log", "gate-queries.jsonl", "partition-ledger.json", ".lock"):
            (self.src / name).write_text("left out")
        self.starts = tmp / "runs" / f"{NAME}.starts"
        self.starts.write_bytes(b"start line\n")
        self.dst = tmp / "stage" / NAME
        self.dst.parent.mkdir()

    def keep(self, names, tag="keep") -> str:
        p = self.tmp / f"{tag}.names"
        data = "".join(f"{n}\n" for n in names).encode()
        p.write_bytes(data)
        return f"{p}:{sha(data)}"

    def argv(self, *keeps, at=AT, pins=None, starts_sha=None, dst=None):
        pins = pins if pins is not None else {k: sha(self.top[k]) for k in ("manifest.json", "served-models.json")}
        a = ["--src", str(self.src), "--dst", str(dst or self.dst), "--at", at]
        for k in keeps:
            a += ["--keep", k]
        for k, v in pins.items():
            a += ["--pin", f"{k}={v}"]
        return a + ["--starts", str(self.starts), "--starts-sha256", starts_sha or sha(self.starts.read_bytes())]


def run(capsys, argv):
    rc = mod.main(argv)
    out, err = capsys.readouterr()
    return rc, out, err


def tree(root: Path):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def test_it_builds_the_subset_from_byte_copies_of_records_claims_tries_and_the_files_that_can_change(tmp_path, capsys):
    o = Out(tmp_path)
    before = tree(o.src)
    rc, out, err = run(capsys, o.argv(o.keep([rec(1), rec(3)], "a"), o.keep([rec(2)], "b")))
    assert (rc, err) == (0, "") and out == "PREFIX-OUT records=3 claims=6 infrastructure=1 copied=6 OK\n"
    d = o.dst
    assert sorted(os.listdir(d / "records")) == sorted([rec(1), rec(2), rec(3)])
    assert sorted(os.listdir(d / "claims")) == sorted(f"{digest(n)}{s}" for n in (1, 2, 3) for s in (".claim", ".attempt"))
    assert os.listdir(d / "infrastructure") == [f"{digest(2)}.{UUID.replace('0', '1', 1)}.json"]  # record 5's try (open attempt) is not carried
    for n in (1, 2, 3):  # a byte copy: the same bytes and modification time, another file, mode 600, no link to the OUT's file
        mine, theirs = d / "records" / rec(n), o.src / "records" / rec(n)
        assert mine.read_bytes() == theirs.read_bytes() and not os.path.samefile(mine, theirs)
        assert os.stat(mine).st_mtime_ns == os.stat(theirs).st_mtime_ns and stat.S_IMODE(os.stat(mine).st_mode) == 0o600
        assert os.stat(mine).st_nlink == 1 and os.stat(theirs).st_nlink == 1
    for name, data in o.top.items():
        assert (d / name).read_bytes() == data and not os.path.samefile(d / name, o.src / name)
    assert (d.parent / f"{NAME}.starts").read_bytes() == o.starts.read_bytes()
    assert sorted(os.listdir(d)) == sorted(["records", "claims", "infrastructure", *o.top])  # nothing left out is carried
    assert stat.S_IMODE(os.stat(d).st_mode) == 0o700 and stat.S_IMODE(os.stat(d / "manifest.json").st_mode) == 0o600
    assert tree(o.src) == before  # nothing under the OUT is written


def test_a_copied_file_does_not_follow_the_out_later(tmp_path, capsys):
    o = Out(tmp_path)
    assert run(capsys, o.argv(o.keep([rec(1)])))[0] == 0
    (o.src / "concurrency-events.jsonl").write_bytes(b"e1\ne2\n")  # the running OUT appends
    assert (o.dst / "concurrency-events.jsonl").read_bytes() == b"e1\n"


@pytest.mark.parametrize("what", ["dst-exists", "name", "time", "empty-keep", "keep-twice-in-one", "keep-twice-in-two", "keep-sha", "keep-not-a-record-name",
                                  "keep-not-a-record", "record-after-the-time", "claim-missing", "attempt-missing", "unknown-infrastructure-name",
                                  "harness-not-empty", "aborted-not-empty", "manifest-pin", "served-pin", "pin-twice", "pin-missing", "unknown-pin",
                                  "starts-sha", "manifest-missing", "record-is-a-link", "claim-is-a-link", "keep-format"])
def test_it_refuses_before_anything_is_written(tmp_path, capsys, what):
    o = Out(tmp_path)
    keeps = [o.keep([rec(1), rec(2)])]
    kw: dict = {}
    if what == "dst-exists":
        o.dst.mkdir()
    elif what == "name":
        kw["dst"] = o.dst.parent / "another-name"
    elif what == "time":
        kw["at"] = "2026-10-26 08:31:00"
    elif what == "empty-keep":
        keeps = [o.keep([])]
    elif what == "keep-twice-in-one":
        keeps = [o.keep([rec(1), rec(1)])]
    elif what == "keep-twice-in-two":
        keeps = [o.keep([rec(1)], "a"), o.keep([rec(1), rec(2)], "b")]
    elif what == "keep-sha":
        keeps = [keeps[0][:-1] + ("0" if keeps[0][-1] != "0" else "1")]
    elif what == "keep-not-a-record-name":
        keeps = [o.keep(["not-a-record.json"])]
    elif what == "keep-not-a-record":
        keeps = [o.keep([rec(1), rec(9)])]
    elif what == "keep-format":
        p = o.tmp / "nonl.names"
        p.write_bytes(rec(1).encode())
        keeps = [f"{p}:{sha(p.read_bytes())}"]
    elif what == "record-after-the-time":
        keeps = [o.keep([rec(1), rec(4)])]
    elif what == "claim-missing":
        (o.src / "claims" / f"{digest(2)}.claim").unlink()
    elif what == "attempt-missing":
        (o.src / "claims" / f"{digest(1)}.attempt").unlink()
    elif what == "unknown-infrastructure-name":
        (o.src / "infrastructure" / "stray.txt").write_text("x")
    elif what == "harness-not-empty":
        (o.src / "harness").mkdir()
        (o.src / "harness" / "x").write_text("x")
    elif what == "aborted-not-empty":
        (o.src / "aborted").mkdir()
        (o.src / "aborted" / "x").write_text("x")
    elif what == "manifest-pin":
        kw["pins"] = {"manifest.json": "0" * 64, "served-models.json": sha(o.top["served-models.json"])}
    elif what == "served-pin":
        kw["pins"] = {"manifest.json": sha(o.top["manifest.json"]), "served-models.json": "1" * 64}
    elif what == "pin-missing":
        kw["pins"] = {"manifest.json": sha(o.top["manifest.json"])}
    elif what == "unknown-pin":
        kw["pins"] = {"manifest.json": sha(o.top["manifest.json"]), "served-models.json": sha(o.top["served-models.json"]), "order.json": sha(o.top["order.json"])}
    elif what == "starts-sha":
        kw["starts_sha"] = "2" * 64
    elif what == "manifest-missing":
        (o.src / "manifest.json").unlink()
    elif what == "record-is-a-link":
        (o.src / "records" / rec(2)).unlink()
        (o.src / "records" / rec(2)).symlink_to(o.src / "records" / rec(1))
    elif what == "claim-is-a-link":
        (o.src / "claims" / f"{digest(1)}.claim").unlink()
        (o.src / "claims" / f"{digest(1)}.claim").symlink_to(o.src / "claims" / f"{digest(2)}.claim")
    argv = o.argv(*keeps, **kw)
    if what == "pin-twice":
        argv += ["--pin", f"manifest.json={sha(o.top['manifest.json'])}"]
    stage_before = tree(o.dst.parent)
    rc, out, err = run(capsys, argv)
    assert rc == 2 and out == "" and err.startswith("REFUSED ") and "\n" == err[-1] and err.count("\n") == 1, (what, err)
    assert tree(o.dst.parent) == stage_before  # nothing was created: no directory, no start record copy


def test_a_failed_write_removes_only_the_directory_it_made(tmp_path, capsys, monkeypatch):
    o = Out(tmp_path)
    real = mod.copy_file
    calls = []

    def flaky(a, b):
        calls.append(b)
        if len(calls) == 3:
            raise OSError("disk full")
        return real(a, b)

    monkeypatch.setattr(mod, "copy_file", flaky)
    keep = o.keep([rec(1), rec(2), rec(3)])
    rc, out, err = run(capsys, o.argv(keep))
    assert (rc, out, err) == (2, "", "REFUSED write-failed\n")
    assert not os.path.lexists(o.dst) and not os.path.lexists(o.dst.parent / f"{NAME}.starts"), "the build's own leftovers are removed"
    assert sorted(os.listdir(o.src / "records")) == sorted(rec(n) for n in range(1, 5))  # the OUT keeps every file
    assert (o.src / "records" / rec(1)).read_text() == "record 1"


def test_no_file_of_the_oUT_gets_another_link(tmp_path, capsys):
    o = Out(tmp_path)
    assert run(capsys, o.argv(o.keep([rec(1), rec(2), rec(3)])))[0] == 0
    for d in ("records", "claims", "infrastructure"):
        assert all(os.stat(o.src / d / f).st_nlink == 1 for f in os.listdir(o.src / d))  # the runner's completeness checks refuse an extra link
        assert all(os.stat(o.dst / d / f).st_nlink == 1 for f in os.listdir(o.dst / d))


def test_a_copy_that_differs_from_what_was_read_is_refused_and_leaves_nothing(tmp_path, capsys, monkeypatch):
    o = Out(tmp_path)

    def bad_write(src, dest, data=None):  # writes other bytes than it reports
        data = data if data is not None else src.read_bytes()
        dest.write_bytes(b"x" + data)
        return sha(data)

    monkeypatch.setattr(mod, "copy_new", bad_write)
    before = tree(o.dst.parent)
    rc, out, err = run(capsys, o.argv(o.keep([rec(1)])))
    assert (rc, out, err) == (2, "", "REFUSED copy-differs\n") and tree(o.dst.parent) == before
