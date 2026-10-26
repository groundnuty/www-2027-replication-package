#!/usr/bin/env python3
"""The dev baselines of the infrastructure-fault rule (change-log entry 129, item 2; science, design log, 21 Oct 07:51):
per model and per arm, the call-less-turn and no-submission counts of the dev records, by the definitions of
harness/agent/infra_fault.py, written with each source's cut, what it read and left out, the sha256 of the run ids it
read, and the code that counted them.

    uv run python scripts/infra_dev_baselines.py --runs-root ~/www-2027/runs \\
        --source uniprot-sp-2026_03=dev-natural-20261020/uniprot@2026-10-21T05:45:00+00:00 \\
        --source rhea-142=dev-natural-20261020/rhea --source corporate=dev-natural-corporate-20261020/corporate \\
        --source opaque-rhea-142=dev-natural-opaque-20261020/opaque-rhea-142 \\
        --out data/runs/infra-fault/dev-baselines.json

A source is NAME=OUT[@CUT], OUT under --runs-root; a record is counted only if its attempt started before the cut
(UniProt's prompted phase ran into GLM's server fault from 05:45Z, so its records from then on are no dev rate; the
cut is by start time, for every model, never by outcome). It reads the OUTs and writes nothing to them (one record at
a time; run it niced where the records are, writing nothing but the output file there: run-node hosts a15).

The branch section (design ruling, design log, 21 Oct 09:31): the post-fix branch rates, from each --branch-source's
branch records that retried (their live retry only), per model, with the pooled rate for a model with fewer than 20
retries. With --branch-commit (repeatable) only records whose harness commit is one of those count (the fix code).
Without a branch source it is "pending". Once the post-fix branch runs finish, fill it in place, leaving the rest of
the file as it was:

    uv run python scripts/infra_dev_baselines.py --runs-root ~/www-2027/runs --branch-only \
        --branch-source lean-retry=dev-feedback-lean-retry-rerun-20261021/uniprot \
        --branch-source rehearsal2=<rehearsal 2's branch rerun OUT> --branch-commit <fix commit> \
        --out data/runs/infra-fault/dev-baselines.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from harness.agent import infra_fault as fault  # noqa: E402


def parse_source(text: str) -> tuple[str, str, str | None]:
    name, sep, rest = text.partition("=")
    if not sep or not name or not rest:
        raise argparse.ArgumentTypeError(f"{text!r}: a source is NAME=OUT[@CUT]")
    rel, _, cut = rest.partition("@")
    if cut:
        try:
            cut = fault.parse_time(cut).isoformat()
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{text!r}: the cut {cut!r} is not an ISO time with its zone ({exc})")
    return name, rel, cut or None


def parse_branch_source(text: str) -> tuple[str, str]:
    name, sep, rel = text.partition("=")
    if not sep or not name or not rel:
        raise argparse.ArgumentTypeError(f"{text!r}: a branch source is NAME=OUT")
    return name, rel


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs-root", type=Path, required=True, help="the directory the sources' OUT paths are under")
    ap.add_argument("--source", type=parse_source, action="append", default=[], metavar="NAME=OUT[@CUT]")
    ap.add_argument("--branch-source", type=parse_branch_source, action="append", default=[], metavar="NAME=OUT",
                    help="a post-fix branch run's OUT (repeatable): the branch section")
    ap.add_argument("--branch-commit", action="append", default=None, metavar="COMMIT",
                    help="count only branch records whose harness commit is this (repeatable; default: every one)")
    ap.add_argument("--branch-only", action="store_true",
                    help="rewrite only the branch section of the existing --out file")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    for flag, given in (("--source", args.source), ("--branch-source", args.branch_source)):
        names = [s[0] for s in given]
        if len(set(names)) != len(names):
            ap.error(f"{flag}: a source is named twice: {names}")
    if args.branch_only and (args.source or not args.branch_source or not args.out.is_file()):
        ap.error("--branch-only takes --branch-source and an existing --out, and no --source")
    if not args.branch_only and not args.source:
        ap.error("--source is required (or --branch-only)")
    # the code that counted them, named as every record names its own
    from run_pilot import harness_commit

    code, at = harness_commit(), datetime.now(UTC).isoformat(timespec="seconds")
    branch = fault.branch_section(args.branch_source, args.runs_root, args.branch_commit)
    if args.branch_source:
        branch.update(code=code, computed_at=at)
    if args.branch_only:
        doc = json.loads(args.out.read_text(encoding="utf-8"))
        doc["branch"] = branch
    else:
        doc = fault.baselines_doc(args.source, args.runs_root)
        doc.update(branch=branch, code=code, computed_at=at)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # atomically: a running monitor reads this file again on every pass
    tmp = args.out.with_name(f".{args.out.name}.tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, args.out)
    for model, entry in (doc["branch"].get("models") or {}).items():
        c = entry["all"]
        print(f"branch {model:24} turns {c['call_less']:>6}/{c['turns']:<7} call-less, retries {c['no_submission']:>4}/"
              f"{c['attempts']:<5} without a submission" + ("" if c["attempts"] >= fault.BRANCH_MIN_RETRIES else
                                                            " (fewer than the minimum: the pooled rate applies)"))
    if args.branch_only:
        return 0
    for model, entry in doc["models"].items():
        for arm, c in entry["arms"].items():
            print(f"{model:24} {arm:12} turns {c['call_less']:>6}/{c['turns']:<7} call-less "
                  f"({c['call_less_rate']}), attempts {c['no_submission']:>4}/{c['attempts']:<5} without a "
                  f"submission ({c['no_submission_rate']})")
    for s in doc["sources"]:
        print(f"{s['name']}: {s['records']} records before the cut (of {s['as_read']['records']} read), used "
              f"{s['records_used']}, left out {s['left_out']}, untimed turns {s['untimed_turns']}, their run ids' "
              f"sha256 {s['run_ids_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
