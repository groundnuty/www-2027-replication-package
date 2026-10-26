#!/usr/bin/env python3
"""Join OUT records' per-turn response ids to the e167 front's provenance log: every turn a front-routed record keeps
must join to exactly one backend (science, 22 Oct: the log is part of the record).

From each record (OUT/records, OUT/aborted and the set-aside OUT/infrastructure and OUT/harness, *.json or *.json.gz;
harness/'s *.setaside.json decisions are skipped unread) only ``route.front_url`` and each transcript turn's ``role``
and ``response_id`` are read: never an outcome, score, answer or checker field (blinding). Set-aside records count in
every figure and also as set_aside_records/set_aside_turns; a record file that is not readable JSON counts as
unreadable_records, and the OUT cannot pass. A record whose route has no ``front_url`` was written before the
switch-over and does not count. A turn of a front-routed record:
- joined: its id has a response line (the line the front fsyncs before the client sees a byte carrying the id)
  under exactly one backend; an id only in an end line does not join;
- ambiguous: under several backends;
- inherited: not in the log, but its id is also in a record without front_url (a branch copies its parent's
  initial phase byte for byte; give the parents' OUTs with --parent-out when they are elsewhere);
- without_id: the turn has no response id;
- unjoined: none of these. It must be 0, with ambiguous and without_id; the exit code is 1 otherwise.
A reply in the log that no record keeps is an orphan: a stream broken mid-way, one a client stream guard rejected,
a client gone; the client retried it, so it is no turn (counted by outcome).

Modes (lead ruling, 22 Oct): --totals-only (the default) reports per OUT the counts above summed over every
phase and arm, and no per-backend figure; --balance adds the per-backend table and refuses unless --analysis-started
is given (branch-phase turn counts per backend reflect flag rates).

    join.py --log provenance.jsonl --out OUT [--out OUT2 ...] [--parent-out DIR ...]
    join.py --log provenance.jsonl --out OUT --balance --analysis-started"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import sys
import zlib
from pathlib import Path
from typing import Any

DAMAGED = (OSError, ValueError, EOFError, zlib.error)  # unreadable, truncated or corrupt (gzip included) files


class LogUnreadable(Exception):
    pass


RECORD_DIRS = ("records", "aborted", "infrastructure", "harness")  # the runner's record and set-aside directories
SET_ASIDE_DIRS = ("infrastructure", "harness")  # run_pilot.py:4216-4217 (a set-aside attempt's turns still join)
SIDECAR_SUFFIX = ".setaside.json"  # harness/'s set-aside decisions: not records, and never read (they hold labels)
EXAMPLES = 20


def id_key(rid: Any) -> str | None:
    """A response id as JSON (so true and 1, or "1" and 1, stay distinct), or None for no id: a turn keeps whatever
    non-null id the assembler or the SDK kept, and the front logs the same value."""
    return None if rid is None else json.dumps(rid, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def record_turn_ids(record: Any) -> tuple[bool, list]:
    """(whether the record went through a front, each assistant turn's response id in order). Reads only
    ``route.front_url`` and the turns' ``role`` and ``response_id``."""
    if not isinstance(record, dict):
        return False, []
    route = record.get("route")
    front = isinstance(route, dict) and bool(route.get("front_url"))
    ids = []
    transcript = record.get("transcript")
    for turn in transcript if isinstance(transcript, list) else []:
        if isinstance(turn, dict) and turn.get("role") == "assistant":
            ids.append(turn.get("response_id"))
    return front, ids


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    return (gzip.decompress(raw) if path.name.endswith(".gz") else raw).decode("utf-8")


def record_files(out: Path) -> list[Path]:
    return [p for sub in RECORD_DIRS for p in sorted((out / sub).glob("*"))
            if p.is_file() and (p.name.endswith(".json") or p.name.endswith(".json.gz"))
            and not p.name.removesuffix(".gz").endswith(SIDECAR_SUFFIX)]


def read_records(out: Path) -> list[tuple[str, bool, list] | tuple[str, None, None]]:
    """(name, front-routed, turn ids) per record file; (name, None, None) for a file that is not readable JSON."""
    rows: list = []
    for path in record_files(out):
        name = f"{path.parent.name}/{path.name}"
        try:
            data = json.loads(read_text(path))
        except DAMAGED:
            rows.append((name, None, None))
            continue
        rows.append((name, *record_turn_ids(data)))
    return rows


def read_log(paths: list[Path]) -> dict:
    """id -> backends (response lines only), the pool's replies (run, seq, id, backend) and each reply's end
    outcome."""
    backends: dict[str, set] = collections.defaultdict(set)
    replies, outcomes, unreadable = [], {}, 0
    for path in paths:
        try:
            text = read_text(path)
        except DAMAGED as exc:  # the joins cannot be known: no report at all
            raise LogUnreadable(f"{path}: {type(exc).__name__}: {exc}") from None
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                x = json.loads(line)
            except ValueError:
                unreadable += 1  # a line a crash cut short
                continue
            if not isinstance(x, dict):
                unreadable += 1
                continue
            key = (x.get("front_run"), x.get("seq"))
            if x.get("kind") == "response":
                rid = id_key(x.get("response_id"))
                if rid is not None:
                    backends[rid].add(x.get("backend"))
                    if x.get("listener") == "pool":
                        replies.append((key, rid, x.get("backend")))
            elif x.get("kind") == "end":  # its outcome only: an id joins through a response line, the one the
                outcomes[key] = x.get("outcome")  # front puts on disk before the client sees a byte
    return {"backends": backends, "replies": replies, "outcomes": outcomes, "unreadable": unreadable}


def join(logs: list[Path], outs: list[Path], parents: list[Path], *, balance: bool) -> tuple[dict, bool]:
    log = read_log(logs)
    by_out = {out: read_records(out) for out in outs}
    pre_front_ids, kept = set(), set()
    for out in outs + parents:
        for _, front, ids in (by_out[out] if out in by_out else read_records(out)):
            if ids is None:
                continue
            kept.update(k for k in map(id_key, ids) if k is not None)
            if not front:
                pre_front_ids.update(k for k in map(id_key, ids) if k is not None)
    reports, clean = [], True
    for out, rows in by_out.items():
        c = collections.Counter()
        per_backend: dict[str, dict] = collections.defaultdict(lambda: {"turns": 0, "responses": set()})
        examples = []
        for name, front, ids in rows:
            c["records"] += 1
            if ids is None:  # its turns cannot be checked: the OUT cannot pass
                c["unreadable_records"] += 1
                if len(examples) < EXAMPLES:
                    examples.append({"record": name, "unreadable": True})
                continue
            if name.split("/", 1)[0] in SET_ASIDE_DIRS:
                c["set_aside_records"] += 1
                c["set_aside_turns"] += len(ids)
            if not front:
                c["pre_front_records"] += 1
                c["pre_front_turns"] += len(ids)
                continue
            c["front_records"] += 1
            for index, raw in enumerate(ids):
                c["turns"] += 1
                rid = id_key(raw)
                if rid is None:
                    c["without_id"] += 1
                    continue
                found = log["backends"].get(rid, set())
                if len(found) == 1:
                    c["joined"] += 1
                    (backend,) = found
                    per_backend[backend]["turns"] += 1
                    per_backend[backend]["responses"].add(rid)
                elif found:
                    c["ambiguous"] += 1
                elif rid in pre_front_ids:
                    c["inherited"] += 1
                else:
                    c["unjoined"] += 1
                    if len(examples) < EXAMPLES:
                        examples.append({"record": name, "assistant_turn": index})
        report = {"out": str(out), **{k: c[k] for k in ("records", "front_records", "turns", "joined", "unjoined",
                                                        "inherited", "ambiguous", "without_id", "pre_front_records",
                                                        "pre_front_turns", "set_aside_records", "set_aside_turns",
                                                        "unreadable_records")}}
        clean = clean and not (c["unjoined"] or c["ambiguous"] or c["without_id"] or c["unreadable_records"])
        if balance:
            report["by_backend"] = {b: {"turns": v["turns"], "responses": len(v["responses"])}
                                    for b, v in sorted(per_backend.items())}
            report["unjoined_examples"] = examples
        reports.append(report)
    orphans = [(key, rid, backend) for key, rid, backend in log["replies"] if rid not in kept]
    summary: dict[str, Any] = {
        "responses": len(log["replies"]), "orphans": len(orphans),
        "orphans_by_outcome": dict(sorted(collections.Counter(
            log["outcomes"].get(key) or "no_end_line" for key, _, _ in orphans).items())),
        "unreadable_lines": log["unreadable"]}
    if balance:
        table: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        for key, _, backend in orphans:
            table[backend][log["outcomes"].get(key) or "no_end_line"] += 1
        summary["orphans_by_backend"] = {b: dict(sorted(v.items())) for b, v in sorted(table.items())}
    return {"mode": "balance" if balance else "totals-only", "outs": reports, "log": summary,
            "note": "orphans count log replies no given OUT keeps: give every OUT the front served for a full count"}, \
        clean


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--log", action="append", required=True, type=Path, help="provenance log (.jsonl[.gz])")
    parser.add_argument("--out", action="append", required=True, type=Path, help="an OUT directory")
    parser.add_argument("--parent-out", action="append", default=[], type=Path,
                        help="an OUT whose records a branch copied (for inherited turns); not reported")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--totals-only", action="store_true", help="the default: per-OUT totals, no backend split")
    mode.add_argument("--balance", action="store_true", help="per-backend table; needs --analysis-started")
    parser.add_argument("--analysis-started", action="store_true",
                        help="confirms the analysis has started, so the per-backend table may be produced")
    args = parser.parse_args(argv)
    if args.balance and not args.analysis_started:
        print("join.py: --balance gives per-backend turn counts, which reflect flag rates in branch phases; it runs "
              "only at analysis time, with --analysis-started (lead ruling, 22 Oct)", file=sys.stderr)
        return 2
    try:
        report, clean = join(args.log, args.out, args.parent_out, balance=args.balance)
    except LogUnreadable as exc:
        print(f"join.py: the provenance log cannot be read, so no turn can be joined: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=1))
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
