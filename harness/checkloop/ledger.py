"""The checker loop's export to the analysis pipeline: ``checker-ledger/2`` (design log).

The pipeline (harness/analysis/inputs.py on the analysis branch) reads one row per
initial output, source and use, and validates every row; this module writes exactly
those fields:

* replay rows (use ``replay``, detection only) from the offline replay matrix: one per
  output and arm, the primary variant under the arm's name and, when asked, a switch-off
  variant as ``<arm>/<variant>``;
* online rows (use ``checker``, or ``both`` when the cohort was prompted with the arm's
  own source) from checker-cell run records (agent-run/12): the initial answer's cell and
  its retry. A flagged output's retry that was lost (its phase failed on infrastructure,
  its answer was never scored, or the record holds no retry at all) is exported as
  ``retry_status`` "missing", never as an unretried flagged row (prereg section 4). A plain
  run's record is keyed by its own run id and condition; a dev grid's branch record (a branch
  cell's, checker.branched_from naming its parent) by the attempt it checked, its parent's run
  id and cell, which the export takes from the grid OUT's manifest and records (``grid_attempts``);
* fault-bank rows (cohort ``bank``, use ``replay``): the bank's items.jsonl id is the run
  id, with the item's question, base question, kind and fault class.

A /2 file holds one split, named in its header (exactly ``{"schema": "checker-ledger/2",
"split": "dev"}``) and leading every row (``{"split": "dev", ...`` then the fields in sorted
order: the loader reads that prefix first, so it must not be ``sort_keys``): a row holds its
split only through its question, so the writer, not the reader, guarantees the file. Every
matrix row, run record and bank item handed to the export must carry the file's split; one that
carries another (or none) refuses the export, naming its id, and nothing is written.
``result_rows`` (/2 only) is how many rows the checked query returned: for an output, the
store's count for rule 22's checked query; for a bank item, the count recorded when the bank was
built; None when not recorded (an ASK, no checked query, a refused query).
``graph_outside_census`` (/2 only) is the patterns confined to a graph neither the census nor the
shared hierarchy describes (entry 72); the compiler of this branch does not report it, so it is
None (not recorded) unless a matrix row carries it. ``questions_sha256`` (/2 only) is the question set of the row's
question, which the analysis checks against the frozen one: the export writes it on every agent row from the manifest
of the batch its record came from (``batch_question_set``; an archive's only once the replay check beside it binds it
to that manifest, ``archive_bound``), and on bank rows from the frozen bank manifest's record of the set the bank was
built on. An exploratory export's header says so (``"exploratory": true``), and a registered analysis refuses it. /1
files (no split in the header or the rows, neither count) are still read, by ``read_ledger``.

The status is the source-attributed output status (rule 8): flagged is F, a substantive or
vacuous pass is P, unknown (no_query included) is U; ``substantive`` is a flag or a
substantive pass, as the checker decided it (contract v2), never recomputed.

For a confirmatory export the bank manifest must be the frozen one: its sha256 equals the
frozen config's ``bank_manifest_sha256``, its items.jsonl (or items.jsonl.gz, decompressed) is the file it froze, and
every bank row is one of its items with the same metadata (``check_bank_manifest``). A bank row's model must be
``matrix.BANK_MODEL`` ("bank"), the analysis's bank_model.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import tarfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.checkloop.checked import checked_result_rows, rules_of
from harness.checkloop.grid import GridError, batch_cohorts, read_grid_cells
from harness.checkloop.matrix import BANK_ITEMS, BANK_MODEL, bank_kind
from harness.checkloop.online import CHECKER_VERSIONS, GENERIC_ON_FULL_ARM
from harness.checks.outcomes import CHECKS

LEDGER_SCHEMA = "checker-ledger/2"
LEDGER_SCHEMA_V1 = "checker-ledger/1"  # old files: no split in the header, no result_rows
SPLITS = ("dev", "test")
# The pipeline's CheckedOutput fields, in its order, with their defaults (None: required). /1 has neither count.
FIELDS_V1: dict[str, Any] = {
    "run_id": None, "graph": None, "model": None, "cohort": None, "source": None, "use": None,
    "question_id": None, "base_question_id": None, "initial_correct": None, "status": None, "substantive": None,
    "any_unknown": False, "flagged_check_types": (), "branch_local_only": False, "data_fault": False,
    "checked_query": "unrecorded", "number_traceability": None, "retried": False, "retry_status": None,
    "retry_correct": None, "check_s": None, "evidence_triples": None, "retry_usd": None, "fault_class": None,
    "bank_item": None,
}
COUNTS = ("result_rows", "graph_outside_census")  # /2 only: a count or None (not recorded)
# /2 only: the question set the row's question belongs to (the analysis keys every row on (id, questions_sha256) and
# refuses a row of its graph that names another set or none, pipeline.check_questions_hash); written for bank rows from
# the frozen manifest's own record of the question set it was built on, None where the export does not know it
QUESTION_SET = "questions_sha256"
FIELDS: dict[str, Any] = {**FIELDS_V1, **dict.fromkeys(COUNTS), QUESTION_SET: None}
CARRIED = ("split",)  # on a row in memory so that the writer can check it; never written (the header names it)
REQUIRED = ("run_id", "graph", "model", "cohort", "source", "use", "question_id", "base_question_id",
            "initial_correct", "status", "substantive")
LETTER = {"flagged": "F", "substantive_pass": "P", "vacuous_pass": "P", "unknown": "U"}
ALIGNMENTS = ("aligned", "fallback", "none")
BANK = "bank"
_SHA256 = re.compile(r"[0-9a-f]{64}")
PLACEHOLDER = "[FILL]"
GENERIC_PREFIX = "generic_retry/"  # the matched generic control's rows, per source
EXPLORATORY = "exploratory"  # a /2 header's mark of an exploratory export (write_ledger), refused when registered


class LedgerError(ValueError):
    """An export that the analysis pipeline would refuse, or that is not the frozen bank."""


class MissingProvenance(LedgerError):
    """An agent row whose question set the export cannot read: its batch has no manifest, or one naming no question set,
    or is an archive with no replay check beside it to bind it to that manifest (``batch_question_set``), or the row is
    of no batch given. The only refusal an exploratory export turns into a row naming no question set; a record naming
    another set, a run id of two batches, an archive that is not the records bound beside it still refuse."""


def require_split(run_id: str, found: Any, split: str) -> None:
    """Refuse an input of another split, or of none: a /2 file holds one split. The message names the
    row's id and nothing else of it."""
    if found != split:
        raise LedgerError(f"row {run_id} is not of split {split!r}: a {LEDGER_SCHEMA} file holds one split")


def require_split_name(split: str) -> None:
    if split not in SPLITS:
        raise LedgerError(f"split must be one of {SPLITS}, not {split!r}")


def _row(**values: Any) -> dict[str, Any]:
    unknown = set(values) - set(FIELDS) - set(CARRIED)
    if unknown:
        raise LedgerError(f"not a {LEDGER_SCHEMA} field: {sorted(unknown)}")
    row = {k: values.get(k, default) for k, default in FIELDS.items()}
    row.update({k: values[k] for k in CARRIED if k in values})
    missing = [k for k in REQUIRED if row[k] is None]
    if missing:
        raise LedgerError(f"{values.get('run_id')}: required fields missing {missing}")
    for name in COUNTS:
        n = row[name]
        if n is not None and (not isinstance(n, int) or isinstance(n, bool) or n < 0):
            raise LedgerError(f"{values.get('run_id')}: {name} must be a count or None, not {n!r}")
    qs = row[QUESTION_SET]
    if qs is not None and not (isinstance(qs, str) and _SHA256.fullmatch(qs)):
        raise LedgerError(f"{values.get('run_id')}: {QUESTION_SET} must be a sha256 (64 lowercase hex) or None")
    row["flagged_check_types"] = list(row["flagged_check_types"])
    return row


def _check_types(fired_checks: Iterable[dict]) -> list[str]:
    """The check types that flagged (rule 9 keeps counts per check type), sorted."""
    return sorted({CHECKS[c["check_id"]][0] for c in fired_checks})


def _verdict_fields(v: dict) -> dict[str, Any]:
    status = v.get("status")
    if status not in LETTER:
        raise LedgerError(f"unknown output status {status!r}")
    fired = v.get("fired_checks") or []
    flagged = status == "flagged"
    return {
        "status": LETTER[status],
        "substantive": status in ("flagged", "substantive_pass"),
        "any_unknown": bool(v.get("any_unknown")),
        "flagged_check_types": _check_types(fired) if flagged else [],
        "branch_local_only": flagged and bool(fired) and all(c.get("branch_local") for c in fired),
        "data_fault": flagged and any(c.get("data_fault") for c in fired),
        "number_traceability": v.get("traceability"),
    }


def _alignment(status: str | None) -> str:
    return status if status in ALIGNMENTS else "unrecorded"


def from_matrix_row(row: dict, *, variants: bool = False) -> dict[str, Any] | None:
    """The replay row of one matrix row, or None: an attempt with no output is not a checked
    output, and a switch-off variant is exported only when ``variants`` is set. The row carries
    the matrix row's split (None when unrecorded) for ``write_ledger`` to check."""
    if not row.get("output") or (row["variant"] != "primary" and not variants):
        return None
    source = row["arm"] if row["variant"] == "primary" else f"{row['arm']}/{row['variant']}"
    bank = row.get("cohort") == BANK
    if bank and row.get("model") != BANK_MODEL:  # the analysis keys bank arms on the config's bank_model
        raise LedgerError(f"bank row {row['run_id']} ({source}) is of model {row.get('model')!r}, not {BANK_MODEL!r}: "
                          f"the analysis would find no registered bank arm for it (a matrix written before the bank "
                          f"rows' model was {BANK_MODEL!r}; replay the bank again)")
    if row.get("correct") is None:
        raise LedgerError(f"{row['run_id']}: an output with no scorer label cannot be exported")
    return _row(
        run_id=row["run_id"], graph=row["graph"], model=row["model"], cohort=row["cohort"], source=source,
        use="replay", question_id=row["question_id"],
        base_question_id=row.get("base_question_id") or row["question_id"], initial_correct=bool(row["correct"]),
        **_verdict_fields(row), checked_query="unrecorded" if bank else _alignment(row.get("checked_query_status")),
        evidence_triples=row.get("budget_used"), result_rows=row.get("result_rows"),
        graph_outside_census=row.get("graph_outside_census"), split=row.get("split"),
        fault_class=row.get("fault_class") if bank else None, bank_item=row.get("bank_item") if bank else None)


def _use(rec: dict, settings: dict) -> str:
    """``both`` when the cohort was prompted with the arm's own source (a single source whose
    kind names the prompt condition: curated, census, generated), else ``checker``."""
    sources = settings["arm"]["sources"]
    return "both" if len(sources) == 1 and rec.get("condition") == sources[0]["kind"] else "checker"


def from_online_record(rec: dict, *, attempt: tuple[str, str] | None = None) -> dict[str, Any] | None:
    """The online row of one checker-cell record, or None when its initial answer is not a
    scored output (that attempt is the attempt table's to count, failed or missing).

    The row is keyed by the record's own run id and condition (a plain run's checker cell is an
    attempt of its condition), unless ``attempt`` names the attempt whose output the record
    checked, as (run id, cohort): a grid branch record's parent and the parent's cell
    (``grid_attempts``), since the analysis counts no branch record as an attempt."""
    chk = rec.get("checker")
    if not chk:
        raise LedgerError(f"record {rec.get('run_id')} is not a checker cell's record")
    if attempt is not None and not (len(attempt) == 2 and all(isinstance(x, str) and x for x in attempt)):
        raise LedgerError(f"record {rec['run_id']}: the attempt it checked must be named by a run id and a cohort")
    initial, retry, settings = chk["initial"], chk["retry"], chk["settings"]
    if settings.get("mode") not in ("feedback", "generic"):  # a baseline's branch cell (design log) has no arm
        raise LedgerError(f"record {rec.get('run_id')}: mode {settings.get('mode')!r} is not a checker arm's; "
                          "the online rows cover the feedback and generic arms only")
    version = settings.get("version")
    if version not in CHECKER_VERSIONS:
        raise LedgerError(f"record {rec['run_id']} is of the unknown checker version {version!r}")
    if settings["mode"] == "generic" and version in GENERIC_ON_FULL_ARM:
        # its stored verdict and retry are the whole arm's, so exporting it would write a row flagged and retried on
        # another source's flag as the matched control of the curated-flagged set (C5); never exported
        raise LedgerError(f"record {rec['run_id']} is a generic cell of {version}, which retried on any source's flag, not "
                          f"the curated flags: the {LEDGER_SCHEMA} export refuses it")
    score = initial.get("score") or {}
    if initial["outcome"]["status"] != "submitted" or score.get("outcome") != "scored":
        return None
    v = initial.get("verdict") or {}
    fields = _verdict_fields(v)
    arm = settings["arm"]["name"]
    out = {"retried": False, "retry_status": None, "retry_correct": None}
    if v.get("retry"):
        out["retried"] = True
        if retry is None or rec["outcome"].get("failure_type") == "infrastructure":
            out.update(retry_status="missing", retry_correct=False)  # a lost retry (prereg section 4)
        else:
            end = rec["outcome"]["status"]
            final = rec.get("score") or {}
            if end == "submitted" and final.get("outcome") != "scored":
                out.update(retry_status="missing", retry_correct=False)  # never scored: missing
            else:
                out.update(retry_status=end, retry_correct=end == "submitted" and bool(final.get("exact")))
    elif retry is not None:
        raise LedgerError(f"{rec['run_id']}: a retry after an output that no check flagged (rule 9)")
    budget = sum((r.get("evidence") or {}).get("budget_used") or 0 for r in initial.get("reports") or [])
    result_rows = checked_result_rows(rec["tool_calls"], initial.get("final_rows"), rules_of(score),
                                      upto=initial.get("submission_index"))
    run_id, cohort = attempt if attempt is not None else (rec["run_id"], rec["condition"])
    return _row(
        run_id=run_id, graph=rec["store"]["name"], model=rec["model"], cohort=cohort,
        source=(GENERIC_PREFIX + arm) if settings["mode"] == "generic" else arm, use=_use(rec, settings),
        question_id=rec["question_id"], base_question_id=rec.get("base_question_id") or rec["question_id"],
        initial_correct=bool(score.get("exact")), **fields,
        checked_query=_alignment(initial.get("checked_query_status")), evidence_triples=budget,
        result_rows=result_rows, split=rec.get("split"), **out)


# -- a dev grid's OUT -------------------------------------------------------------------------------

def grid_attempts(out: Path, records: list[dict]) -> dict[str, tuple[str, str]]:
    """The attempt (run id, cohort) that each grid branch record of one OUT checked, by the branch
    record's run id: its parent (checker.branched_from) and the parent's cell. Every record of the
    OUT is read as the analysis reads it (harness/checkloop/grid.py: the cells its manifest pins,
    each record's prompt its cell's, a branch record's parent a record of the same OUT in the cell's
    parent cell with its question, route model, repetition and store); empty for a plain run's OUT
    or an archive, where a record in a cell other than its condition (a grid record given without
    its manifest) is refused rather than keyed as an attempt. Refusals raise LedgerError."""
    try:
        _, parents = batch_cohorts(records, read_grid_cells(out) if Path(out).is_dir() else None)
    except GridError as exc:
        raise LedgerError(str(exc)) from None
    return {run: (p["run_id"], p["load"]["arm"]) for run, p in parents.items()}


# -- the question set of an agent row ------------------------------------------------------------


def batch_question_set(path: Path) -> str:
    """The question set a batch ran on, from its own manifest as the analysis reads it (harness/analysis/gate.py
    read_manifest: a plain run's ``fingerprint.questions_sha256``, a grid run's ``fingerprint.shared.questions_sha256``):
    an OUT directory's manifest.json, or for a records archive the manifest.json beside it (results/<run>/<model>/
    holds both), once the archive is shown to hold that run's records (``archive_bound``). ``records_question_set``
    with no records."""
    return records_question_set(path, ())


def _member_name(archive: Path, name: str) -> str:
    """An archive member's path relative to the OUT it was packed from; one that could name another place, refused."""
    rel = name
    while rel.startswith("./"):
        rel = rel[2:]
    if not rel or rel.startswith("/") or ".." in Path(rel).parts:
        raise LedgerError(f"{archive}: its member {name!r} is no path inside a run's OUT")
    return rel


def archive_bound(archive: Path) -> None:
    """A records archive is bound to the manifest.json beside it only through the run's replay check beside both
    (results/<run>/<model>/: manifest.json, replay-check.json, records.tar.gz, as the runner's OUT left them): the
    check's digest (``records_sha256``, harness/analysis/gate.py's rule for its schema, /1 over the replayed records, /2
    over every analysis input) is recomputed from the archive's members, and must be the check's, over the records the
    check counts, every record the export reads from the archive (matrix.load_raw_records: a ``.json`` member under a
    ``records/`` directory) among them; the check must have passed (gate.require_replay_check, here on the archive's
    members instead of an OUT's files). The manifest records no run id of its own, so this is the analysis's binding of
    an OUT's records to its manifest (the directory both are in) carried to the archive: an archive moved together with
    its replay check beside another run's manifest is not caught here. Raises MissingProvenance when no replay check is
    beside the archive, LedgerError when the archive is not the records it covers."""
    from harness.analysis.gate import (
        CHECK_FILE,
        CHECK_SCHEMA,
        CHECK_SCHEMA_2,
        input_kind,
    )

    archive = Path(archive)
    if not (archive.is_file() and archive.name.endswith((".tar.gz", ".tgz"))):
        raise LedgerError(f"{archive}: neither a run's OUT directory nor a records archive (records.tar.gz)")
    check_path = archive.parent / CHECK_FILE
    if not check_path.is_file():
        raise MissingProvenance(f"{archive}: no {CHECK_FILE} beside it, so nothing binds its records to the "
                                f"manifest.json beside it")
    try:
        check = json.loads(check_path.read_text(encoding="utf-8"))
    except ValueError:
        raise LedgerError(f"{check_path}: not readable JSON, so nothing binds {archive.name} to its manifest") from None
    schema = check.get("schema") if isinstance(check, dict) else None
    if schema not in (CHECK_SCHEMA, CHECK_SCHEMA_2):
        raise LedgerError(f"{check_path}: unknown schema {schema!r} (neither {CHECK_SCHEMA} nor {CHECK_SCHEMA_2})")
    if check.get("n_ok") != check.get("n_records") or check.get("failures"):
        raise LedgerError(f"{archive}: its replay check failed ({check.get('n_ok')} of {check.get('n_records')} "
                          f"records passed)")
    files: dict[str, str] = {}
    read: list[str] = []  # the members the export reads as records (matrix.load_raw_records)
    with tarfile.open(archive) as tar:
        for m in tar.getmembers():
            if not m.isfile():
                continue
            rel = _member_name(archive, m.name)
            if rel in files:
                raise LedgerError(f"{archive}: its member {rel} is in it twice")
            files[rel] = hashlib.sha256(tar.extractfile(m).read()).hexdigest()
            if (m.name.endswith(".json") and not m.name.endswith(".setaside.json")
                    and Path(m.name).parent.name == "records"):
                read.append(rel)
    kinds = ("replay",) if schema == CHECK_SCHEMA else ("replay", "hashed")  # /1: the replayed records alone
    inputs = sorted(rel for rel in files if input_kind(rel) in kinds)
    replayed = [rel for rel in inputs if input_kind(rel) == "replay"]
    hashed = [rel for rel in inputs if input_kind(rel) == "hashed"]
    digest = hashlib.sha256(json.dumps(sorted([rel, files[rel]] for rel in inputs),
                                       separators=(",", ":")).encode("utf-8")).hexdigest()
    uncovered = sorted(set(read) - set(replayed))
    differs = [what for what, bad in (
        (f"records it reads that the check does not cover {uncovered[:3]}", uncovered),
        (f"{len(replayed)} records, the check {check.get('n_records')}", check.get("n_records") != len(replayed)),
        (f"inputs hashed only {hashed}, the check's {check.get('hashed_only')}",
         schema == CHECK_SCHEMA_2 and check.get("hashed_only") != hashed),
        ("its records_sha256 differs", check.get("records_sha256") != digest)) if bad]
    if differs:
        raise LedgerError(f"{archive}: not the records its replay check covers ({CHECK_FILE} beside it: "
                          f"{'; '.join(differs)}), so it is not bound to the manifest.json beside it")


def records_question_set(path: Path, records: Iterable[dict]) -> str:
    """The question set of one batch's records (``batch_question_set``). Every conflict is looked for before a missing
    provenance is reported, so that an exploratory export, which excuses the latter only, never masks one: a manifest
    the analysis refuses, an archive that is not the records its replay check beside it covers (``archive_bound``), a
    record carrying a question set of its own (``questions_sha256``) other than its manifest's, or records of the batch
    carrying two, raise LedgerError; then a batch with no manifest, a manifest naming no question set, or an archive
    with no replay check beside it raise MissingProvenance."""
    from harness.analysis.gate import BatchRefused, read_manifest

    path = Path(path)
    batch = path if path.is_dir() else path.parent
    missing: list[str] = []
    sha = None
    if not (batch / "manifest.json").is_file():
        missing.append(f"{batch}: no manifest.json, so the question set its records ran on is unknown")
    else:
        try:
            _, settings = read_manifest(batch, "the question set its records ran on is")
        except BatchRefused as exc:  # a manifest there that the analysis refuses: not missing, so never exploratory
            raise LedgerError(str(exc)) from None
        sha = settings.get("questions_sha256")
        if not (isinstance(sha, str) and _SHA256.fullmatch(sha)):
            missing.append(f"{batch}: its manifest names no question set (questions_sha256), so its rows cannot name "
                           f"theirs")
            sha = None
    if not path.is_dir():
        try:
            archive_bound(path)
        except MissingProvenance as exc:
            missing.append(str(exc))
    carried: dict[str, str] = {}  # a question set a record carries -> the first record carrying it
    for rec in records:
        named = rec.get(QUESTION_SET)
        if named is None:
            continue
        if sha is not None and named != sha:
            raise LedgerError(f"record {rec.get('run_id')} names the question set {named}, its batch's manifest "
                              f"{sha} ({path})")
        carried.setdefault(named, rec.get("run_id"))
    if len(carried) > 1:
        raise LedgerError(f"records of one batch name two question sets ({path}: {sorted(carried.items())[:2]})")
    if missing:
        raise MissingProvenance("; ".join(missing))
    return sha


# -- the frozen fault bank ------------------------------------------------------------------


def frozen_bank_sha256(config_path: Path) -> str:
    """The frozen config's ``bank_manifest_sha256``; an unfilled one refuses the export."""
    sha = json.loads(Path(config_path).read_text(encoding="utf-8")).get("bank_manifest_sha256")
    if not sha or sha == PLACEHOLDER:
        raise LedgerError(f"{config_path}: bank_manifest_sha256 is not filled in (prereg section 5 'Fault bank')")
    return sha


def bank_items_bytes(manifest: Path) -> tuple[Path, bytes]:
    """The bank's items file beside its manifest as the manifest froze it (``items.jsonl``): that file, or, where the
    repository keeps it gzip-compressed (``items.jsonl.gz``, as the frozen UniProt bank is committed), its decompressed
    bytes; empty when neither is there. The caller checks the bytes against the sha256 the manifest records (as the
    analysis reads it, harness/analysis/inputs.py bank_items_bytes)."""
    plain = Path(manifest).parent / "items.jsonl"
    if plain.is_file():
        return plain, plain.read_bytes()
    packed = plain.with_name("items.jsonl.gz")
    if packed.is_file():
        return packed, gzip.decompress(packed.read_bytes())
    return plain, b""


def frozen_bank(manifest: Path, frozen_sha256: str) -> tuple[dict, list[dict]]:
    """(manifest, items) of the frozen bank: the manifest's sha256 is ``frozen_sha256`` and its items are the bytes it
    froze (``bank_items_bytes``, checked by the sha256 the manifest records); LedgerError otherwise."""
    manifest = Path(manifest)
    actual = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if actual != frozen_sha256:
        raise LedgerError(f"{manifest}: sha256 {actual} is not the frozen bank manifest's {frozen_sha256}")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    items_path, raw = bank_items_bytes(manifest)
    if hashlib.sha256(raw).hexdigest() != (data.get("files") or {}).get("items.jsonl"):
        raise LedgerError(f"{items_path}: not the items.jsonl the manifest froze")
    return data, [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]


def bank_question_set(data: dict) -> str:
    """The question set the bank was built on, as its manifest records it (``inputs.questions.sha256``)."""
    sha = ((data.get("inputs") or {}).get("questions") or {}).get("sha256")
    if not (isinstance(sha, str) and _SHA256.fullmatch(sha)):
        raise LedgerError("the bank manifest records no question set (inputs.questions.sha256)")
    return sha


def check_bank_manifest(rows: Iterable[dict], manifest: Path, frozen_sha256: str, *, split: str) -> dict[str, int]:
    """Every bank row is an item of the frozen manifest with the item's metadata: the manifest's
    sha256 is the frozen one, its items.jsonl the file it froze (``frozen_bank``), each row's run id, question,
    base question, kind and fault class are the item's, and the item is of ``split`` (the file's).
    Returns counts per kind."""
    _, all_items = frozen_bank(manifest, frozen_sha256)
    items = {i["id"]: i for i in all_items if i.get("bank") in BANK_ITEMS}
    counts: dict[str, int] = {}
    for r in rows:
        if r["cohort"] != BANK:
            continue
        item = items.get(r["run_id"])
        if item is None:
            raise LedgerError(f"bank row {r['run_id']} ({r['source']}) is not a checked item of the frozen manifest")
        require_split(r["run_id"], item.get("split"), split)
        kind = bank_kind(item)
        want = (item["question_id"], item.get("base_question_id") or item["question_id"], kind,
                item.get("fault_class") if kind == "answer_changing" else None)
        got = (r["question_id"], r["base_question_id"], r["bank_item"], r["fault_class"])
        if got != want:
            raise LedgerError(f"bank row {r['run_id']} ({r['source']}) is {got}, the frozen item is {want}")
        counts[kind] = counts.get(kind, 0) + 1
    return counts


# -- the file -----------------------------------------------------------------------------------


def _unique(rows: list[dict]) -> None:
    dup = sorted(k for k, n in Counter((r["run_id"], r["source"], r["use"]) for r in rows).items() if n > 1)
    if dup:
        raise LedgerError(f"duplicate ledger rows {dup[:3]}")


def write_ledger(path: Path, rows: Iterable[dict], *, split: str, exploratory: bool = False) -> Path:
    """JSON lines whose first line names the schema and the split, one row per line: the split
    first, then the fields in sorted order (the loader's prefix check). Every row must carry
    ``split`` (as the ``from_*`` functions leave it): one of another split, or of none, is
    refused naming its id, and nothing is written. A duplicate (run id, source, use) is refused.
    An exploratory export's header also says so (``"exploratory": true``, after the split), which a
    registered analysis refuses from the header alone (harness/analysis/inputs.py); any other
    export's header is exactly the schema and the split."""
    require_split_name(split)
    rows = list(rows)
    for r in rows:
        require_split(r["run_id"], r.get("split"), split)
        fields = set(r) - set(CARRIED)
        if fields != set(FIELDS):
            raise LedgerError(f"{r['run_id']}: not the {LEDGER_SCHEMA} fields: {sorted(fields ^ set(FIELDS))}")
    _unique(rows)
    header = json.dumps({"schema": LEDGER_SCHEMA, "split": split, **({EXPLORATORY: True} if exploratory else {})})
    # the split first, then the fields sorted (json.dumps without sort_keys, which would put split in the middle)
    lines = [header] + [json.dumps({"split": split, **{k: r[k] for k in sorted(FIELDS)}}) for r in rows]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return Path(path)


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    if len({k for k, _ in pairs}) != len(pairs):
        raise LedgerError("a row repeats a key")
    return dict(pairs)


@dataclass(frozen=True)
class Ledger:
    schema: str
    split: str | None  # a /2 header names it; a /1 header names none
    rows: list[dict[str, Any]]  # every field of the schema's FIELDS, no split (a /1 row's counts read as None)
    exploratory: bool = False  # a /2 header marking an exploratory export


def read_ledger(path: Path) -> Ledger:
    """A ledger file of either schema: /1 (old files: a header naming no split, rows without
    the counts) or /2 (a header naming dev or test, and ``"exploratory": true`` for an exploratory
    export, each row beginning with that split). Anything
    else is refused; a message names the file, the row's position and the problem, never a row's
    content."""
    lines = [ln for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    try:
        header = json.loads(lines[0]) if lines else None
    except ValueError:
        header = None
    if not isinstance(header, dict):
        raise LedgerError(f"{path}: the first line is not a ledger header")
    schema = header.get("schema")
    exploratory = header.get(EXPLORATORY) is True and set(header) == {"schema", "split", EXPLORATORY}
    if schema == LEDGER_SCHEMA_V1 and set(header) == {"schema"}:
        split, fields = None, FIELDS_V1
    elif (schema == LEDGER_SCHEMA and (set(header) == {"schema", "split"} or exploratory)
          and header["split"] in SPLITS):
        split, fields = header["split"], FIELDS
    else:
        raise LedgerError(f"{path}: the header is not {LEDGER_SCHEMA_V1} (no split) or {LEDGER_SCHEMA} with a "
                          f"split of {SPLITS} (and, for an exploratory export, \"{EXPLORATORY}\": true)")
    rows = []
    for n, line in enumerate(lines[1:], 1):
        data = json.loads(line, object_pairs_hook=_unique_keys)
        if split is not None:  # /2: the row begins with its split, which is the header's
            if next(iter(data), None) != "split":
                raise LedgerError(f"{path} row {n}: a {schema} row begins with its split field")
            if data.pop("split") != split:
                raise LedgerError(f"{path} row {n}: its split is not the header's (a {schema} file holds one split)")
        unknown = sorted(set(data) - set(fields))
        if unknown:
            raise LedgerError(f"{path} row {n}: not a {schema} field: {unknown}")
        rows.append(_row(**data))
    _unique(rows)
    return Ledger(schema, split, rows, exploratory)
