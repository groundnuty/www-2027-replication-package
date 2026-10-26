#!/usr/bin/env python3
"""A subset OUT of a plain test OUT, from a pinned list of record file names (entry 208 exploratory look, private-table graphs).

    python3 -I a15_prefix_out.py --src OUT --dst STAGE_OUT --at 2026-10-26T08:31:00Z
        --keep FILE:SHA256 [--keep FILE:SHA256 ...]
        --pin manifest.json=SHA256 --pin served-models.json=SHA256 --starts FILE --starts-sha256 SHA256

The analysis reads every record in the OUT it is given and has no record-name filter, and the identity view tool links every file of
records/ into its view. A look over the final records of an OUT that is still running (a seeded prefix, its records beyond it, records carried over from
another run) therefore needs an OUT that holds those records only. This builds it, from NAMES: no record is opened.

* STAGE_OUT is new, mode 700, and has the basename of OUT (the identity view tool wants the OUT's name and ``<name>.starts`` beside it).
* records/: a BYTE COPY (a new mode-600 file, its sha256 checked after the write, its modification time kept) for each name the pinned lists
  give (each must be a record file name, listed once, with a modification time at or before ``--at``, so a record that could still change is
  refused). Never a hard link: a link raises the live file's link count, which the runner's completeness checks refuse (26 Oct, p1 of the
  a15-gemma partitions). claims/: the two files (``.claim``, ``.attempt``) of each kept record's claim digest, copied the same way.
  infrastructure/: the try files of kept records only (a try of an open or a left-out attempt is not carried).
  harness/ and aborted/ must be empty or absent (refused otherwise: nothing here knows what they hold).
* the files that can change while the OUT runs are COPIED, byte for byte, mode 600: manifest.json and served-models.json (their sha256 must be
  the pinned ones), order.json, admitted-fingerprints.json, model-fronts.json, replay-check.json and concurrency-events.jsonl where the OUT
  has them. The start record ``--starts`` is copied beside STAGE_OUT as ``<name>.starts`` (its sha256 pinned). Anything else (the runner's
  logs, the query gate's files, runs.jsonl, the operator control directories) is left out.
* nothing under OUT is written. A link anywhere in the OUT's path to a kept file refuses.

Output: ``PREFIX-OUT records=N claims=N infrastructure=N copied=N OK`` on stdout and exit 0, or ``REFUSED <reason>`` on stderr and exit 2.
Never a name or a count of anything else."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import os
import re
import shutil
import stat
import sys
from pathlib import Path

RECORD = re.compile(r"([0-9a-f]{40})\.([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\.json")
SHA = re.compile(r"[0-9a-f]{64}")
PIN = re.compile(r"(/[^\s:]*):([0-9a-f]{64})")
UTC = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
PINNED_COPIES = ("manifest.json", "served-models.json")
OTHER_COPIES = ("order.json", "admitted-fingerprints.json", "model-fronts.json", "replay-check.json", "concurrency-events.jsonl")
EMPTY_DIRS = ("harness", "aborted")


class Refused(Exception):
    pass


def regular(path: Path) -> os.stat_result:
    """lstat of a regular file that is not a link."""
    try:
        st = os.lstat(path)
    except OSError:
        raise Refused("missing-file") from None
    if not stat.S_ISREG(st.st_mode):
        raise Refused("not-a-regular-file")
    return st


def read_bytes(path: Path) -> bytes:
    regular(path)
    return path.read_bytes()


def read_names(spec: str) -> list[str]:
    m = PIN.fullmatch(spec)
    if not m:
        raise Refused("keep-pin-malformed")
    data = read_bytes(Path(m[1]))
    if hashlib.sha256(data).hexdigest() != m[2]:
        raise Refused("keep-sha-differs")
    if data and not data.endswith(b"\n"):
        raise Refused("keep-format")
    names = data.decode("utf-8").split("\n")[:-1] if data else []
    if any(not RECORD.fullmatch(n) for n in names):
        raise Refused("keep-not-record-names")
    return names


def copy_new(src: Path, dest: Path, data: bytes | None = None) -> str:
    """A byte copy of the regular file ``src`` at the new ``dest`` (mode 600); its sha256."""
    data = read_bytes(src) if data is None else data
    fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return hashlib.sha256(data).hexdigest()


def copy_file(src: Path, dest: Path) -> None:
    """A byte copy of the regular file ``src`` at the new ``dest``: mode 600, no link and no reflink (the bytes are read and written), the sha256
    of what was written checked, the modification time of ``src`` kept."""
    st = regular(src)
    data = src.read_bytes()
    sha = copy_new(src, dest, data)
    if sha != hashlib.sha256(dest.read_bytes()).hexdigest():
        raise Refused("copy-differs")
    os.utime(dest, ns=(st.st_atime_ns, st.st_mtime_ns))


def build(a: argparse.Namespace) -> str:
    src, dst = Path(a.src), Path(a.dst)
    if not src.is_dir() or src.is_symlink() or not dst.parent.is_dir():
        raise Refused("bad-directory")
    if dst.name != src.name or os.path.lexists(dst):
        raise Refused("bad-name")
    if not UTC.fullmatch(a.at):
        raise Refused("bad-time")
    at = datetime.datetime.strptime(a.at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc).timestamp()
    keep: list[str] = []
    for spec in a.keep:
        keep += read_names(spec)
    if not keep or len(set(keep)) != len(keep):
        raise Refused("keep-empty-or-twice")
    pins = {}
    for item in a.pin:
        name, sep, sha = item.partition("=")
        if not sep or name not in PINNED_COPIES or not SHA.fullmatch(sha) or name in pins:
            raise Refused("pin-malformed")
        pins[name] = sha
    if set(pins) != set(PINNED_COPIES) or not SHA.fullmatch(a.starts_sha256):
        raise Refused("pin-malformed")
    digests = {RECORD.fullmatch(n)[1] for n in keep}
    for d in EMPTY_DIRS:
        p = src / d
        if os.path.lexists(p) and (not p.is_dir() or p.is_symlink() or any(p.iterdir())):
            raise Refused("unsupported-directory")
    for d in ("records", "claims", "infrastructure"):
        if not (src / d).is_dir() or (src / d).is_symlink():
            raise Refused("bad-directory")
    # every check before the first write
    for n in keep:
        if regular(src / "records" / n).st_mtime > at:
            raise Refused("record-after-the-time")
    for d in digests:
        for suffix in (".claim", ".attempt"):
            regular(src / "claims" / f"{d}{suffix}")
    tries = []
    for f in sorted(os.listdir(src / "infrastructure")):
        m = RECORD.fullmatch(f)
        if not m:
            raise Refused("unknown-infrastructure-name")
        regular(src / "infrastructure" / f)
        if m[1] in digests:
            tries.append(f)
    copies = {}
    for name in PINNED_COPIES + OTHER_COPIES:
        if os.path.lexists(src / name):
            data = read_bytes(src / name)
            if name in pins and hashlib.sha256(data).hexdigest() != pins[name]:
                raise Refused("pinned-file-differs")
            copies[name] = data
    if set(PINNED_COPIES) - set(copies):
        raise Refused("pinned-file-missing")
    starts = read_bytes(Path(a.starts))
    if hashlib.sha256(starts).hexdigest() != a.starts_sha256:
        raise Refused("starts-sha-differs")
    os.mkdir(dst, 0o700)
    try:
        for d, names in (("records", keep), ("claims", [f"{x}{s}" for x in sorted(digests) for s in (".claim", ".attempt")]), ("infrastructure", tries)):
            os.mkdir(dst / d, 0o700)
            for n in names:
                copy_file(src / d / n, dst / d / n)
        for name, data in copies.items():
            copy_new(src / name, dst / name, data)
        copy_new(Path(a.starts), dst.parent / f"{dst.name}.starts", starts)
    except (OSError, Refused) as e:  # this build's own new directory only (the OUT is never written)
        shutil.rmtree(dst, ignore_errors=True)
        (dst.parent / f"{dst.name}.starts").unlink(missing_ok=True)
        raise Refused(str(e) if isinstance(e, Refused) else "write-failed") from None
    return f"PREFIX-OUT records={len(keep)} claims={2 * len(digests)} infrastructure={len(tries)} copied={len(copies)} OK"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--at", required=True, help="every kept record's modification time must be at or before this UTC time")
    p.add_argument("--keep", action="append", required=True, metavar="FILE:SHA256", help="a pinned file of record file names; repeated")
    p.add_argument("--pin", action="append", default=[], metavar="NAME=SHA256", help="manifest.json and served-models.json, once each")
    p.add_argument("--starts", required=True, help="the start record, copied beside DST as <name>.starts")
    p.add_argument("--starts-sha256", required=True)
    a = p.parse_args(argv)
    try:
        print(build(a))
        return 0
    except Refused as e:
        print(f"REFUSED {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
