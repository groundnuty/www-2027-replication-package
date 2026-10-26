#!/usr/bin/env python3
"""Export the checker loop to the analysis pipeline's checker-ledger/2 (design log).

Reads replay matrices (scripts/checker_replay.py's matrix.jsonl.gz of agent outputs, and
scripts/bank_replay.py's of the fault bank's items) and checker-cell run records, and writes one
ledger row per initial output, source and use (harness/checkloop/ledger.py):

    uv run python scripts/export_checker_ledger.py --split dev --matrix OUT/matrix.jsonl.gz [BANK_OUT/matrix.jsonl.gz] \\
        [--records RUN_DIR_OR_ARCHIVE ...] [--variants] \\
        [--bank-manifest BANK/manifest.json --frozen-config data/analysis/prereg-config.json] \\
        --out ledger.jsonl

--matrix takes one or more matrices (repeat the flag or list them); a row twice across them (the
same run id, source and use) refuses the export.

--split (dev or test) is required and is named in the file's header line: a /2 file holds one
split. Every matrix row, run record (plain ones too) and bank item given must be of that split;
one of another split, or of none, refuses the export (exit 2, naming its id) and nothing is
written.

Every agent row (replay or online) names the question set its batch ran on, from the batch's own
manifest (a plain run's fingerprint.questions_sha256, a grid run's fingerprint.shared.questions_sha256;
for a records archive, the manifest.json beside it, once the replay-check.json beside both shows the
archive holds the records it checked: harness/checkloop/ledger.py archive_bound): the analysis refuses
a row of its graph without the frozen one. An online row names its own record's batch's set; a replay
row the set of the batch holding its run's record, so it must be a record of a batch given with
--records, and a run id that is a record of two batches given refuses the export (either set could
be the row's). A batch with no manifest, one naming no question set, an archive with nothing beside it
binding it to its manifest, or an agent row of no given batch (missing provenance) refuses the export,
unless --exploratory: its rows then name none, counted in the summary line. A record naming another
set, a run id of two batches, an archive that is not the records its replay check covers, or a
manifest the analysis cannot read refuse it even then.

--exploratory marks the file as an exploratory export in its header ({"schema": "checker-ledger/2",
"split": ..., "exploratory": true}), whatever it used of the allowance: the analysis reads it without
a registered config and refuses it with one (a test run or a confirmatory run), from the header alone.

Each --records path is one batch. A dev grid's OUT directory (its manifest.json names the grid's
cells) exports each branch cell's record keyed by the attempt it checked: its parent
(checker.branched_from), which must be a record of the same OUT in the cell's parent cell, and the
parent's cell as the cohort, as the analysis counts attempts; a plain run's records keep their own
run id and condition. A grid branch record given without its OUT's manifest (in an archive, or
records copied out of the OUT) is refused, as is one whose parent is missing or another attempt.

A confirmatory export passes --bank-manifest and --frozen-config: the manifest's sha256
must equal the frozen config's bank_manifest_sha256, its items (items.jsonl, or items.jsonl.gz as
the frozen UniProt bank is committed) the bytes it froze, and every bank row must be one of the
manifest's items with its metadata; each bank row then names the question set the manifest
records the bank was built on (questions_sha256), which the analysis checks against the frozen
one. Bank rows without --bank-manifest are refused unless --exploratory is given.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from harness.checkloop.ledger import (
    BANK,
    QUESTION_SET,
    SPLITS,
    LedgerError,
    MissingProvenance,
    bank_question_set,
    check_bank_manifest,
    from_matrix_row,
    from_online_record,
    frozen_bank,
    frozen_bank_sha256,
    grid_attempts,
    records_question_set,
    require_split,
    require_split_name,
    write_ledger,
)
from harness.checkloop.matrix import load_records, read_jsonl_gz


def export(matrix: Path | Sequence[Path] | None, records: list[Path], *, split: str, variants: bool = False,
           bank_manifest: Path | None = None, frozen_config: Path | None = None,
           exploratory: bool = False) -> tuple[list[dict], dict]:
    require_split_name(split)
    matrices = [] if matrix is None else [matrix] if isinstance(matrix, (str, Path)) else list(matrix)
    replayed: list[dict] = []  # the matrices' rows
    online: list[dict] = []  # the checker cells' rows, each named its question set from the batch its record came from
    skipped = {"no_output": 0, "not_scored": 0}
    # run id -> (its batch, the batch's question set, None where an exploratory export cannot read it): every record of
    # every batch, so that a run id of two batches refuses the export rather than taking either batch's set
    batch_of: dict[str, tuple[str, str | None]] = {}
    unnamed: list[str] = []  # batches whose question set is unknown (an exploratory export only)
    unstamped = 0
    for path in matrices:
        for r in read_jsonl_gz(Path(path)):
            require_split(r["run_id"], r.get("split"), split)  # every input, those the export skips included
            row = from_matrix_row(r, variants=variants)
            if row is not None:
                replayed.append(row)
            elif not r.get("output"):
                skipped["no_output"] += 1
    for path in records:  # one batch per path: a grid branch record's parent is a record of its own OUT
        batch = list(load_records([path]))
        attempts = grid_attempts(path, batch)  # a grid record given without its OUT's manifest is refused first
        try:
            sha: str | None = records_question_set(path, batch)
        except MissingProvenance:  # only a question set the batch cannot name; any other refusal stands
            if not exploratory:
                raise
            sha = None
            unnamed.append(str(path))
        for rec in batch:
            seen = batch_of.get(rec["run_id"])
            if seen is not None:
                raise LedgerError(f"run id {rec['run_id']} is a record of two batches given ({seen[0]} and {path}): "
                                  f"which question set its rows name is ambiguous")
            batch_of[rec["run_id"]] = (str(path), sha)
        for rec in batch:
            require_split(rec["run_id"], rec.get("split"), split)
            if rec.get("checker") is None:
                continue  # a plain record's initial output reaches the ledger through the replay matrix
            row = from_online_record(rec, attempt=attempts.get(rec["run_id"]))
            if row is None:
                skipped["not_scored"] += 1
                continue
            row[QUESTION_SET] = sha  # the batch the record came from (a branch row is keyed by its parent, of this OUT)
            unstamped += sha is None
            online.append(row)
    # a replay row names the question set of the batch whose record it is (the analysis keys each row on (id,
    # questions_sha256)); an online row named its own above
    for r in replayed:
        if r["cohort"] == BANK:
            continue
        found = batch_of.get(r["run_id"])
        if found is None:
            if not exploratory:
                raise MissingProvenance(f"agent row {r['run_id']} ({r['source']}) is no record of a batch given with "
                                        f"--records whose manifest names its question set: give its batch (the OUT, "
                                        f"or the records archive beside its manifest.json and replay-check.json)")
            unstamped += 1
        elif found[1] is None:
            unstamped += 1
        r[QUESTION_SET] = None if found is None else found[1]
    rows = replayed + online
    bank = [r for r in rows if r["cohort"] == BANK]
    info: dict = {"rows": len(rows), "bank_rows": len(bank), "skipped": skipped}
    if unstamped or unnamed:  # exploratory only: rows the analysis refuses under any registered config
        info.update(unstamped_agent_rows=unstamped, batches_without_question_set=unnamed)
    if bank_manifest is not None:
        if frozen_config is None:
            raise LedgerError("--bank-manifest needs --frozen-config: its sha256 is checked against the frozen one")
        frozen = frozen_bank_sha256(frozen_config)
        info["bank"] = check_bank_manifest(bank, bank_manifest, frozen, split=split)
        question_set = bank_question_set(frozen_bank(bank_manifest, frozen)[0])
        for r in bank:  # the analysis keys each row on (id, questions_sha256)
            r[QUESTION_SET] = question_set
        info["bank_questions_sha256"] = question_set
    elif bank and not exploratory:
        raise LedgerError("the export holds fault-bank rows: pass --bank-manifest and --frozen-config (or "
                          "--exploratory for a non-confirmatory export)")
    return rows, info


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--split", required=True, choices=SPLITS, help="the one split of the file (named in its header)")
    p.add_argument("--matrix", type=Path, nargs="+", action="extend", default=[],
                   help="replay matrices (checker_replay.py's, bank_replay.py's); repeat or list them")
    p.add_argument("--records", type=Path, nargs="*", default=[])
    p.add_argument("--variants", action="store_true", help="also export the switch-off variants as <arm>/<variant>")
    p.add_argument("--bank-manifest", type=Path)
    p.add_argument("--frozen-config", type=Path)
    p.add_argument("--exploratory", action="store_true",
                   help="a non-confirmatory export, marked so in its header (a registered analysis refuses it): bank "
                        "rows without the frozen manifest, and agent rows whose question set is unknown, naming none")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    try:
        rows, info = export(args.matrix, args.records, split=args.split, variants=args.variants,
                            bank_manifest=args.bank_manifest, frozen_config=args.frozen_config,
                            exploratory=args.exploratory)
        write_ledger(args.out, rows, split=args.split, exploratory=args.exploratory)
    except LedgerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {info['rows']} {args.split} ledger rows ({info['bank_rows']} fault-bank) to {args.out}; "
          f"skipped {info['skipped']}"
          + (f"; bank {info['bank']}" if "bank" in info else "")
          + ("; marked exploratory (a registered analysis refuses it)" if args.exploratory else "")
          + (f"; exploratory: {info['unstamped_agent_rows']} agent rows name no question set"
             if info.get("unstamped_agent_rows") else "")
          + (f"; batches whose question set is unknown: {info['batches_without_question_set']}"
             if info.get("batches_without_question_set") else ""), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
