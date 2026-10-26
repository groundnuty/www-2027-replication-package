#!/usr/bin/env python3
"""Replay check of one batch of run records, the gate before any analysis (design log).

    uv run python scripts/verify_batch.py OUT [--sources-dir DIR --hierarchy FILE [--source-file NAME=PATH ...]]

Every record in OUT/records, OUT/infrastructure, OUT/harness and OUT/aborted must load,
write back to its stored bytes and replay offline; the set-aside decisions and
OUT/concurrency-events.jsonl are hashed with them, not replayed. Every online checker decision
is decided again offline from the record's recorded store responses (dress rehearsal F3), with
the compilers of the sources its arm names, read from --sources-dir (NAME.json.gz or NAME.json;
--source-file for another file) over the shared hierarchy --hierarchy and built on first use: a
decision that diverges holds the batch (held, held_records), and a checker record with no
sources given is not verified. A phase whose divergences exceed 1% of its records, or occur in
more than one model, is HALTED (halted; exit 1); the set-aside records and their re-runs are
counted per arm (set_aside) and printed. A real source's compiler over a real hierarchy takes gigabytes:
run that on the run host, never on the agents VM. Writes OUT/replay-check.json
(harness/agent/replay_check.py describes it) and exits 0 when every record passed, 1
otherwise. pilot_summary.py refuses a batch unless its check passed, it is not held, its digest
matches the current inputs, and it was made under the code in use, committed.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

# Allow running as a plain script from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.agent.replay_check import (  # noqa: E402
    CHECK_FILE,
    halt_message,
    held_message,
    recompute_decisions,
    setaside_lines,
    verify_batch,
    write_check,
)


def recompute_of(sources_dir: Path | None, source_files: dict[str, Path], hierarchy_path: Path | None):
    """The online checker decisions' recompute over the compilers of the sources given, each built on first use;
    None when no hierarchy was given (then a checker record is not verified)."""
    if hierarchy_path is None:
        return None
    from harness.checkloop.recompute import ArmUnavailable, DecisionRecompute
    from harness.checks.compiler import CheckCompiler
    from harness.checks.hierarchy import SharedHierarchy
    from harness.sources.model import OntologySource

    hierarchy: list[SharedHierarchy] = []  # read once, on the first source that needs it

    def load(name: str) -> CheckCompiler:
        path = source_files.get(name)
        if path is None and sources_dir is not None:
            path = next((sources_dir / f"{name}{suffix}" for suffix in (".json.gz", ".json")
                         if (sources_dir / f"{name}{suffix}").is_file()), None)
        if path is None:
            raise ArmUnavailable(f"no source {name!r} in {sources_dir} or the --source-file list")
        data = path.read_bytes()
        source = OntologySource.from_json(json.loads(gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data))
        if not hierarchy:
            hierarchy.append(SharedHierarchy.load(hierarchy_path))
        return CheckCompiler(source, hierarchy[0])

    return DecisionRecompute(load=load)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("out", type=Path, help="the batch directory (a pilot runner OUT)")
    p.add_argument("--sources-dir", type=Path, default=None,
                   help="the graph's sources, to recompute every online checker decision (NAME.json.gz or NAME.json)")
    p.add_argument("--source-file", action="append", default=[], metavar="NAME=PATH",
                   help="read source NAME from PATH instead of the sources directory")
    p.add_argument("--hierarchy", type=Path, default=None, help="the store's saved shared hierarchy (rule 29)")
    p.add_argument("--decisions-json", type=Path, default=None,
                   help="only recompute the online checker decisions and write them to this file, as the runner asks "
                        "after a phase (no replay check is written)")
    p.add_argument("--progress-jsonl", type=Path, default=None,
                   help="with --decisions-json: also append each decision to this file as it is made (one JSON line, "
                        "flushed), so a recompute stopped part way leaves the decisions it completed")
    args = p.parse_args(argv)
    if not args.out.is_dir():
        p.error(f"{args.out} is not a directory")
    if (args.sources_dir is not None or bool(args.source_file)) != (args.hierarchy is not None):
        p.error("--sources-dir (or --source-file) and --hierarchy go together")
    if args.progress_jsonl is not None and args.decisions_json is None:
        p.error("--progress-jsonl goes with --decisions-json")
    files = {k: Path(v) for k, _, v in (x.partition("=") for x in args.source_file)}
    recompute = recompute_of(args.sources_dir, files, args.hierarchy)
    if args.decisions_json is not None:
        found = recompute_decisions(args.out, recompute, progress=args.progress_jsonl)
        args.decisions_json.write_text(json.dumps({"decisions": found}, ensure_ascii=False), encoding="utf-8")
        print(f"{args.out}: {sum(v is True for v in found.values())} online checker decisions recomputed to "
              f"themselves, {sum(isinstance(v, list) and v[1] for v in found.values())} diverge, "
              f"{sum(isinstance(v, list) and not v[1] for v in found.values())} could not be recomputed")
        return 0
    check = verify_batch(args.out, recompute=recompute)
    write_check(args.out, check)
    print(f"{args.out}: {check['n_ok']} of {check['n_records']} records load, write back byte for byte and replay, "
          f"{check['decisions_recomputed']} online checker decisions recomputed to themselves; records_sha256 "
          f"{check['records_sha256']}; written to {args.out / CHECK_FILE}")
    for line in setaside_lines(check):
        print(line)
    for failure in check["failures"]:
        print(f"  FAILED {failure['file']} ({failure['stage']}): {failure['error']}", file=sys.stderr)
    held = held_message(args.out, check)
    if held:
        print(held, file=sys.stderr)
    halted = halt_message(args.out, check)
    if halted:
        print(halted, file=sys.stderr)
    return 0 if check["n_ok"] == check["n_records"] and not halted else 1


if __name__ == "__main__":
    sys.exit(main())
