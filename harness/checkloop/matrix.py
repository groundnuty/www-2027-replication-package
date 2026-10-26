"""The offline replay matrix (design log, AC6, AC7, AC9): every source's checks,
the union's and the detection-switch variants, on every record's INITIAL
output, with no model call.

For each record of a batch (any cohort), the initial output's checked query is
picked by rule 22; each source's compiled checks run once on it, through one
store client shared by every source (a recording client, or a replay of the
recorded store log, so the matrix itself replays with no store); the union arm
reads the reports of its sources; and each arm is read under every switch
variant (the no-common-subclass rule and branch-local flags, on and off; rule
23). Each (record, arm, variant) gives one row: status, fired checks,
traceability, checked-query status and strata; the six-cell accounting is
computed per (model, cohort, arm, variant) from those rows.
"""

from __future__ import annotations

import gzip
import io
import json
import tarfile
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.checkloop.accounting import NOT_ESTIMABLE, OutputRow, account
from harness.checkloop.arms import verdict
from harness.checkloop.checked import checked_output, checked_result_rows, rules_of
from harness.checkloop.grid import batch_cohorts, read_grid_cells
from harness.checkloop.store_log import RecordingClient, ReplayStoreClient
from harness.checks.compiler import CheckCompiler, Output
from harness.checks.query import analyse
from harness.terms import row_from_json

VARIANTS = {  # rule 23: the detection switches replayed on and off; "primary" is the frozen online setting
    "primary": (True, True),
    "heuristic_off": (False, True),
    "branch_local_off": (True, False),
    "both_off": (False, False),
}
STRATA = ("checked_query_status", "template", "negation", "any_unknown", "unsupported")
MATRIX_VERSION = "checker-replay-matrix/1"


def load_raw_records(paths: Sequence[Path]) -> Iterator[str]:
    """Run records, as stored, from record directories (``<dir>/records/*.json`` or
    ``<dir>/*.json``) and record archives (``records.tar.gz``: members under
    ``records/`` only, so set-aside attempts in ``harness/`` stay out), in a fixed order."""
    for path in paths:
        if path.is_file() and path.name.endswith((".tar.gz", ".tgz")):
            with tarfile.open(path) as tar:
                members = sorted((m for m in tar.getmembers() if m.isfile() and m.name.endswith(".json")
                                  and not m.name.endswith(".setaside.json")
                                  and Path(m.name).parent.name == "records"), key=lambda m: m.name)
                for m in members:
                    yield tar.extractfile(m).read().decode("utf-8")
        else:
            folder = path / "records" if (path / "records").is_dir() else path
            for f in sorted(folder.glob("*.json")):
                if not f.name.endswith(".setaside.json"):
                    yield f.read_text(encoding="utf-8")


def load_records(paths: Sequence[Path]) -> Iterator[dict]:
    return (json.loads(raw) for raw in load_raw_records(paths))


def load_attempts(paths: Sequence[Path]) -> tuple[list[dict], dict[str, str]]:
    """The records whose initial outputs the matrix checks, and the cohort of each (by run id), as
    the analysis counts attempts (harness/checkloop/grid.py): each path is one OUT. A plain run's
    records, each of its condition. A dev grid's OUT (its manifest.json pins its cells): its initial
    cells' records, each of its cell (generated-1, not its condition "generated"); its branch cells'
    records are left out, since each checks its parent's output, which the parent's rows cover, but
    each is checked first (its parent must be in the OUT). Anything the analysis would refuse raises
    GridError."""
    records: list[dict] = []
    cohorts: dict[str, str] = {}
    for path in paths:
        batch = list(load_records([path]))
        found, _ = batch_cohorts(batch, read_grid_cells(path) if path.is_dir() else None)
        records += [r for r in batch if r["run_id"] in found]
        cohorts.update(found)
    return records, cohorts


def initial_of(rec: dict) -> tuple[dict, list | None, dict | None]:
    """(outcome, final rows, score) of the record's initial output: a checker cell's
    ``checker.initial``, else the record itself."""
    chk = rec.get("checker")
    if chk:
        i = chk["initial"]
        return i["outcome"], i["final_rows"], i["score"]
    return rec["outcome"], rec.get("final_rows"), rec.get("score")


@dataclass
class Arms:
    """The sources' compilers (built once) and the arms read from them: each
    source alone, and every union named in ``unions`` (arm -> its sources)."""

    compilers: dict[str, CheckCompiler]
    unions: dict[str, tuple[str, ...]]

    @property
    def names(self) -> list[str]:
        return list(self.compilers) + list(self.unions)

    def sources_of(self, arm: str) -> tuple[str, ...]:
        return self.unions.get(arm, (arm,))


def check_record(rec: dict, arms: Arms, client: Any, cohort: str | None = None) -> list[dict]:
    """The matrix rows of one record: one per (arm, variant), of ``cohort`` (``load_attempts``; by
    default the record's condition, a plain run's one arm)."""
    outcome, final_rows, score = initial_of(rec)
    base = {
        "run_id": rec["run_id"], "graph": (rec.get("store") or {}).get("name"), "question_id": rec["question_id"],
        "base_question_id": rec.get("base_question_id"),
        "template": rec.get("template"), "split": rec.get("split"), "model": rec["model"],
        "cohort": cohort or rec.get("condition"), "repetition": rec.get("repetition"), "label": outcome.get("label"),
        "output": outcome["status"] == "submitted",
        "correct": (score or {}).get("exact") if outcome["status"] == "submitted" else None,
    }
    rows: list[dict] = []
    if not base["output"]:
        for arm in arms.names:
            for variant in VARIANTS:
                rows.append({**base, "arm": arm, "variant": variant, "status": None, "checked_query_status": None})
        return rows
    upto = (rec.get("checker") or {}).get("initial", {}).get("submission_index")
    output, cq = checked_output(rec["tool_calls"], final_rows, rules_of(score), upto=upto,
                                row_cap=rec["store"].get("row_cap"))
    answer = [r if isinstance(r, tuple) else row_from_json(r) for r in (final_rows or [])]
    # the ledger's result_rows (the empty-result stratum): what the store returned for the checked query
    base = {**base, "result_rows": checked_result_rows(rec["tool_calls"], final_rows, rules_of(score), upto=upto)}
    return _arm_rows(base, output, cq, arms, client, answer)


def _arm_rows(base: dict, output: Output | None, cq: str, arms: Arms, client: Any, answer: list) -> list[dict]:
    """One row per (arm, variant) of one checked output: each source's checks run once, and
    an arm reads its sources' reports (rule 39)."""
    reports: dict[str, list] = {}
    if output is not None:
        qa = analyse(output.query)
        for name, compiler in arms.compilers.items():
            reports[name] = [compiler.check(output, client, qa=qa)]
    rows: list[dict] = []
    for arm in arms.names:
        arm_reports = [r for s in arms.sources_of(arm) for r in reports.get(s, [])]
        for variant, (heuristic, branch_local) in VARIANTS.items():
            v = verdict(arm_reports, heuristic=heuristic, branch_local=branch_local, no_query=output is None,
                        answer=answer)
            first = arm_reports[0] if arm_reports else None
            rows.append({**base, "arm": arm, "variant": variant, "status": v.status, "any_unknown": v.any_unknown,
                         "retry": v.retry, "traceability": v.traceability, "per_source": v.per_source,
                         "fired_checks": v.fired_checks, "fired": v.fired, "reason": v.reason,
                         "checked_query_status": cq, "checked_query": output.query if output else None,
                         "negation": bool(first and first.negation),
                         "unsupported": sorted({c for r in arm_reports for c in r.unsupported_constructs}),
                         "tooling_limits": sorted({t for r in arm_reports for t in r.tooling_limits}),
                         # rule 39: an arm's retrieval cost is the sum of its sources' own budgets used
                         "budget_used": sum(r.evidence.get("budget_used") or 0 for r in arm_reports),
                         # entry 72: patterns of the query's graph the census and the shared hierarchy do not
                         # describe, summed over the arm's sources (0 for a non-census source)
                         "graph_outside_census": sum(r.graph_outside_census for r in arm_reports)})
    return rows


# -- the fault bank (prereg section 3 "Bank metrics"; E1.3) ------------------------------

BANK = "bank"  # the cohort of fault-bank rows (the analysis pipeline's checker-ledger/1)
# the model of fault-bank rows: the analysis keys a bank arm as (graph, bank_model, "bank", "replay", source), and the
# frozen config's bank_model is "bank" (harness/analysis/prereg.py PreregConfig.bank); rows under another model name
# would load and then be missing from every registered bank arm
BANK_MODEL = "bank"
# The checked items' ``bank`` values -> their kind; a control of kind "rewrite" (an answer-preserving
# rewrite: bank "control" in bank v2) is a "rewrite", as the analysis reads it (bank_kind).
BANK_ITEMS = {"control": "untouched", "answer_changing": "answer_changing", "answer_neutral": "answer_neutral"}


def bank_kind(item: dict) -> str:
    """A checked bank item's kind as the analysis pipeline reads it (feat/analysis inputs.py,
    check_bank_manifest): "rewrite" for an item of kind "rewrite", else by its ``bank`` value."""
    return "rewrite" if item.get("kind") == "rewrite" else BANK_ITEMS[item["bank"]]


def bank_items(items: Iterable[dict], split: str | None = None) -> list[dict]:
    """The checked items of a frozen bank's items.jsonl: untouched gold controls and the
    mutations that were not rejected, of ``split`` when given, in the file's order."""
    return [i for i in items if i.get("bank") in BANK_ITEMS and (split is None or i.get("split") == split)]


def check_bank_item(item: dict, arms: Arms, client: Any, graph: str) -> list[dict]:
    """The matrix rows of one bank item: every source's checks on the item's query, read as
    a gold query (role "gold": a flag is a data fault only when witnessed, rule 10). The row
    is the item: its id is the run id, with its question, base question, kind and fault class,
    so the ledger export can be checked against the frozen manifest item by item. An
    answer-changing mutation is wrong (W), every control right (C)."""
    kind = bank_kind(item)
    base = {
        "run_id": item["id"], "graph": graph, "question_id": item["question_id"],
        "base_question_id": item.get("base_question_id") or item["question_id"], "template": item.get("template"),
        "split": item.get("split"), "model": BANK_MODEL, "cohort": BANK, "repetition": None, "label": item["bank"],
        "output": True, "correct": kind != "answer_changing", "bank_item": kind,
        "fault_class": item.get("fault_class") if kind == "answer_changing" else None,
        "operator": item.get("operator"),
        "result_rows": _row_count(item),  # the registered empty-result stratum (bank_rates)
    }
    output = Output(query=item["query"], answer=[], role="gold", trajectory_rows=[])
    return _arm_rows(base, output, "given", arms, client, [])


def _row_count(item: dict) -> int | None:
    """How many rows the item's query returned when the bank was built (None: not recorded)."""
    n = (item.get("answer_effect") or {}).get("n_rows")
    return n if isinstance(n, int) and not isinstance(n, bool) else None


def run_bank(items: Iterable[dict], arms: Arms, graph: str, *, live_client: Any = None,
             store_logs: dict[str, dict] | None = None) -> tuple[list[dict], dict[str, dict]]:
    """(rows, store logs by item id) for the checked items, as ``run_matrix`` for records."""
    rows: list[dict] = []
    logs: dict[str, dict] = {}
    for item in items:
        client = (ReplayStoreClient(store_logs.get(item["id"], {"version": "checker-store-log/1", "entries": []}))
                  if store_logs is not None else RecordingClient(live_client))
        rows += check_bank_item(item, arms, client, graph)
        if store_logs is not None:
            problem = client.divergence()
            if problem:
                raise RuntimeError(f"bank item {item['id']}: the store log does not reproduce the checks: {problem}")
            logs[item["id"]] = store_logs.get(item["id"], {"version": "checker-store-log/1", "entries": []})
        else:
            logs[item["id"]] = client.log()
    return rows, logs


# -- the empty-result stratum (design ruling 4, change-log entry 59: registered) ------------

RESULT_STRATA = ("returns_rows", "empty", "no_row_count")
EMPTY_RETRY_ARM = "empty-retry"
EMPTY_RESULT_CHECK = "empty_result"  # retry on empty's one trigger, as a check id the ledger can name


def result_stratum(n_rows: int | None) -> str:
    """Whether a bank item's query returns rows: "empty", "returns_rows", or "no_row_count"."""
    if n_rows is None:
        return "no_row_count"
    return "empty" if n_rows == 0 else "returns_rows"


def empty_retry_bank_rows(items: Iterable[dict], graph: str) -> list[dict]:
    """Retry on empty read on the bank, the stratum's reference: it flags exactly the checked
    items whose query returned no rows when the bank was built (an item with no recorded
    count is unknown). The same rows as ``check_bank_item``'s, under the arm "empty-retry"."""
    rows = []
    for item in bank_items(items):
        kind = bank_kind(item)
        n = _row_count(item)
        rows.append({
            "run_id": item["id"], "graph": graph, "question_id": item["question_id"],
            "base_question_id": item.get("base_question_id") or item["question_id"], "template": item.get("template"),
            "split": item.get("split"), "model": BANK_MODEL, "cohort": BANK, "repetition": None, "label": item["bank"],
            "output": True, "correct": kind != "answer_changing", "bank_item": kind,
            "fault_class": item.get("fault_class") if kind == "answer_changing" else None,
            "operator": item.get("operator"), "result_rows": n, "arm": EMPTY_RETRY_ARM, "variant": "primary",
            "status": "unknown" if n is None else ("flagged" if n == 0 else "substantive_pass"),
            "fired_checks": [{"check_id": EMPTY_RESULT_CHECK, "reason": "empty_result", "key": "-", "source":
                              EMPTY_RETRY_ARM, "branch_local": False, "data_fault": False}] if n == 0 else [],
            "checked_query_status": "given"})
    return rows


def _bank_cell(rows: Sequence[dict]) -> dict[str, Any]:
    def ratio(num: int, den: int):
        return num / den if den else NOT_ESTIMABLE

    wrong = [r for r in rows if not r["correct"]]
    right = [r for r in rows if r["correct"]]
    w_f = sum(1 for r in wrong if r["status"] == "flagged")
    c_f = sum(1 for r in right if r["status"] == "flagged")
    by_kind: dict[str, dict[str, int]] = {}
    for r in rows:
        k = by_kind.setdefault(r.get("fault_class") or r.get("bank_item") or "unrecorded", {"n": 0, "flagged": 0})
        k["n"] += 1
        k["flagged"] += r["status"] == "flagged"
    return {"N": len(rows), "W": len(wrong), "C": len(right), "W_F": w_f, "C_F": c_f,
            "unknown": sum(1 for r in rows if r["status"] == "unknown"),
            "detection_recall": ratio(w_f, len(wrong)), "false_flag_rate": ratio(c_f, len(right)),
            "flag_precision": ratio(w_f, w_f + c_f), "by_kind": by_kind}


def bank_rates(rows: Iterable[dict]) -> dict[str, dict[str, dict[str, Any]]]:
    """Every bank rate per arm and variant, over all checked items and split by whether the
    item's query returns rows (``result_rows``: the registered empty-result stratum). Add
    ``empty_retry_bank_rows`` to the rows for the reference. A row's status is the arm's
    output status (a judge's reject is "flagged", accept a pass, unparsed "unknown"); an
    answer-changing item is wrong, a control or an answer-neutral mutation right; an unknown
    is not a flag. Rates with a zero denominator are "not estimable"."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        if r.get("cohort") == BANK and r.get("output") and r.get("status") is not None:
            groups.setdefault((r["arm"], r["variant"]), []).append(r)
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for (arm, variant), rs in sorted(groups.items()):
        cell = {"all": _bank_cell(rs)}
        for stratum in RESULT_STRATA:
            part = [r for r in rs if result_stratum(r.get("result_rows")) == stratum]
            if part:
                cell[stratum] = _bank_cell(part)
        out.setdefault(arm, {})[variant] = cell
    return out


def initial_identity(rec: dict) -> str:
    """The run whose initial output a record holds: a branched retry arm's is the record it
    was branched from; every other record's is its own."""
    return (rec.get("checker") or {}).get("branched_from") or rec["run_id"]


def one_per_initial_output(records: Iterable[dict]) -> list[dict]:
    """Each initial output once, in the given order: a plain record and the retry arms branched
    from it share one initial output, which the matrix counts once (the plain record is kept
    when it is given, else the first record that holds it)."""
    chosen: dict[str, dict] = {}
    for rec in records:
        key = initial_identity(rec)
        if key not in chosen or (chosen[key].get("checker") and rec["run_id"] == key):
            chosen[key] = rec
    return list(chosen.values())


def run_matrix(records: Iterable[dict], arms: Arms, *, live_client: Any = None,
               store_logs: dict[str, dict] | None = None, progress=None,
               cohorts: dict[str, str] | None = None) -> tuple[list[dict], dict[str, dict]]:
    """(rows, store logs by run id). Each initial output is checked once
    (``one_per_initial_output``). With ``store_logs`` the store is never read:
    each record's checks are served from its log, and a log that does not cover
    them is an error. ``cohorts`` (``load_attempts``) gives each record's cohort;
    without it, or for a record it does not name, the cohort is the condition."""
    rows: list[dict] = []
    logs: dict[str, dict] = {}
    for i, rec in enumerate(one_per_initial_output(records)):
        if store_logs is not None:
            client = ReplayStoreClient(store_logs.get(rec["run_id"], {"version": "checker-store-log/1",
                                                                     "entries": []}))
        else:
            client = RecordingClient(live_client)
        rows += check_record(rec, arms, client, (cohorts or {}).get(rec["run_id"]))
        if store_logs is not None:
            problem = client.divergence()
            if problem:
                raise RuntimeError(f"record {rec['run_id']}: the store log does not reproduce the checks: {problem}")
            logs[rec["run_id"]] = store_logs.get(rec["run_id"], {"version": "checker-store-log/1", "entries": []})
        else:
            logs[rec["run_id"]] = client.log()
        if progress:
            progress(i + 1, rec)
    return rows, logs


def to_output_row(row: dict) -> OutputRow:
    return OutputRow(run_id=row["run_id"], correct=row["correct"], status=row["status"], output=row["output"],
                     checked_query_status=row.get("checked_query_status") or "none", template=row.get("template"),
                     negation=bool(row.get("negation")), any_unknown=bool(row.get("any_unknown")),
                     unsupported=tuple(row.get("unsupported") or ()), no_query=row.get("reason") == "no_query")


def accounting(rows: Sequence[dict]) -> dict[str, Any]:
    """Six-cell accounting per model, cohort, arm and variant (strata under every one)."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["model"], r["cohort"], r["arm"], r["variant"]), []).append(r)
    out: dict[str, Any] = {}
    for (model, cohort, arm, variant), rs in sorted(groups.items()):
        a = account([to_output_row(r) for r in rs], strata=STRATA)
        if sum(a["cells"].values()) != a["N_out"]:
            raise AssertionError(f"{model}/{arm}/{variant}: the six cells do not sum to N_out")
        a["budget_used"] = sum(r.get("budget_used") or 0 for r in rs)  # rule 39: a union's is its sources' sum
        out.setdefault(model, {}).setdefault(cohort, {}).setdefault(arm, {})[variant] = a
    return out


def write_jsonl_gz(path: Path, items: Iterable[Any]) -> None:
    """Deterministic gzip (no timestamp, no file name), so a replay can be compared byte for byte."""
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0) as gz:
        for item in items:
            gz.write((json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
    path.write_bytes(buf.getvalue())


def read_jsonl_gz(path: Path) -> list[Any]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
