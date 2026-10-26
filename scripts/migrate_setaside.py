#!/usr/bin/env python3
"""One-off: convert an attempt set aside by the earlier relabel script to the current form.

The earlier script edited the record in OUT/harness/ (outcome relabelled, a
``harness_failure`` field added), so it no longer loads or replays. This puts
the original, byte-unchanged record back in its place and writes the decision
to the sidecar OUT/harness/<run_id>.setaside.json, as relabel_harness_failure.py
now does. The original must load with RunRecord.from_json, replay to the same
record, and be the same attempt as the edited copy, or nothing is changed. The
edited copy is kept under OUT/harness-migration-backup/ (outside harness/, so
no report counts it). It refuses to run while a runner holds the run's lock.

Crash safety: the steps run in the order back up the edited copy, restore the
original, write the sidecar; each write is atomic and the sidecar comes last,
so its presence means the migration finished. Running the same command again
after a stop finishes the job: a backup left by an earlier run is reused (and
must equal the edited copy), and a restored record without a sidecar gets its
sidecar from the backup. Meanwhile a restored record without a sidecar is only
not yet counted as set aside.

    uv run python scripts/migrate_setaside.py OUT --run-id RUN_ID --original PATH \\
        --defect "why" --ruling URL
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from relabel_harness_failure import SIDECAR_SCHEMA  # noqa: E402
from run_pilot import Refused, acquire_lock  # noqa: E402

from harness.agent.record import RunRecord, utc_now  # noqa: E402
from harness.agent.replay import ReplayDivergence, replay_attempt  # noqa: E402


def _write_atomic(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--original", required=True, type=Path, help="the byte-unchanged record, e.g. from a backup")
    ap.add_argument("--defect", required=True)
    ap.add_argument("--ruling", required=True)
    args = ap.parse_args(argv)
    try:
        lock = acquire_lock(args.out)
    except Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    try:
        return _migrate(args)
    finally:
        os.close(lock)


def _migrate(args: argparse.Namespace) -> int:
    harness = args.out / "harness"
    target = harness / f"{args.run_id}.json"
    sidecar = harness / f"{args.run_id}.setaside.json"
    backup = args.out / "harness-migration-backup"
    backup_path = backup / f"{args.run_id}.edited.json"
    if sidecar.exists():
        print(f"{sidecar} exists: already migrated", file=sys.stderr)
        return 2
    target_bytes = target.read_bytes()
    original_bytes = args.original.read_bytes()
    if "harness_failure" in json.loads(target_bytes):
        edited_bytes, restored = target_bytes, False
        if backup_path.exists() and backup_path.read_bytes() != edited_bytes:
            print(f"{backup_path} exists and differs from the edited record; resolve by hand", file=sys.stderr)
            return 2
    elif target_bytes == original_bytes and backup_path.exists():
        edited_bytes, restored = backup_path.read_bytes(), True  # an earlier run restored it, then stopped
        print(f"finishing an interrupted migration of {target.name}")
    else:
        print(f"{target} is not a record edited by the earlier relabel script", file=sys.stderr)
        return 2
    edited = json.loads(edited_bytes)
    if "harness_failure" not in edited:
        print(f"{backup_path} is not a record edited by the earlier relabel script", file=sys.stderr)
        return 2
    original = json.loads(original_bytes)
    try:
        RunRecord.from_json(original)
        replay_attempt(original)  # whole-record check: the original reproduces itself
    except (ValueError, ReplayDivergence) as exc:
        print(f"{args.original} does not load or replay: {exc}", file=sys.stderr)
        return 2
    same = ("run_id", "question_id", "model", "repetition", "final_rows", "transcript", "tool_calls")
    differs = [k for k in same if original.get(k) != edited.get(k)]
    edited_label = edited["outcome"].get("original_label")
    if differs or edited_label != original["outcome"].get("label"):
        print(f"the original and the edited copy are not the same attempt (differ in {differs or ['outcome label']})",
              file=sys.stderr)
        return 2

    earlier = edited["harness_failure"]
    decision = {
        "schema": SIDECAR_SCHEMA,
        "run_id": original["run_id"],
        "record": target.name,
        "question_id": original["question_id"],
        "model": original["model"],
        "repetition": original["repetition"],
        "label": "failed:harness",
        "original_label": original["outcome"].get("label"),
        "original_failure_type": original["outcome"].get("failure_type"),
        "defect": args.defect,
        "ruling": args.ruling,
        "set_aside_at": earlier.get("set_aside_at"),
        "migrated_at": utc_now(),
        "migrated_from": "an edited record (earlier relabel script); the original restored from a backup",
    }
    if not restored:
        if not backup_path.exists():
            backup.mkdir(exist_ok=True)
            _write_atomic(backup_path, edited_bytes)  # step 1: keep the edited copy
        _write_atomic(target, original_bytes)  # step 2: the original, byte for byte
    _write_atomic(sidecar, json.dumps(decision, ensure_ascii=False, indent=1, sort_keys=True).encode("utf-8"))  # last
    print(f"restored the original {target.name} and wrote {sidecar.name} (edited copy kept in {backup.name}/)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
