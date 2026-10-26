"""The analysis inputs: one row per attempt (from run records) and one row per checked output.

``Attempt`` is built from a run record (harness/agent/record.py) joined with its question;
an attempt-table file holds the same rows, so simulated data enters through the same loader.
``CheckedOutput`` is the checker loop's per-output row (E1.2, design log AC7), defined here as
the schema this pipeline reads (``checker-ledger/1``, and ``checker-ledger/2``: one split per file,
named in its header and leading every row; an exploratory export's header says so, and a registered
analysis refuses it): one row per initial output, source and use. Both files
are JSON lines whose first line is ``{"schema": ...}``; every loader reads that line alone, unbuffered,
and decides from it before reading any row.

Conventions (prereg = paper/preregistration.md draft 3; proposal §7.6 "Accounting"; contract v2):
- an *output* is a submission the scorer scored. Failed, capped and abstained attempts are
  scored failures: kept, and counted wrong (prereg §4 "Missing data");
- an attempt still unscored after the re-runs is **missing** (prereg §4): an attempt that failed
  infrastructure on every try, a submission whose scoring failed, or a registered (question,
  repetition) with no record at all (``complete_design``). Missing attempts count wrong in the
  primary analysis and are dropped in the sensitivity analysis (pipeline and prereg modules);
- earlier infrastructure tries and ``failed:harness`` set-asides are counted per arm;
- the ledger's ``status`` is the source-attributed output status: number traceability is
  its own layer (rule 8) and never makes an output flagged;
- every flagged output of an online arm (use ``checker`` or ``both``) has its retry (rule 9;
  prereg §5): a flagged row without one is refused, and a retry that was lost is recorded as
  ``retry_status`` "missing", which counts toward the missing share (prereg §4);
- ``substantive`` comes from the checker loop as computed there (four-way status, v2), never
  recomputed here;
- fault-bank rows (cohort ``bank``, prereg §3 "Bank metrics", §4.1) carry the item's kind: an
  answer-changing mutation (W, with its fault class), an answer-neutral mutation, an
  untouched gold control or an answer-preserving rewrite (the last three C). The false-flag rate
  of §4.1 uses the gold and the rewrites; answer-neutral mutations are their own stratum.
"""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.agent.cost import estimate_record_cost
from harness.agent.engine_flags import engine_flags
from harness.analysis.gate import PROMPT_PINS, GridCells

ATTEMPT_SCHEMA = "attempt-table/1"
LEDGER_SCHEMA = "checker-ledger/1"
SINGLE_SPLIT_LEDGER = "checker-ledger/2"  # one split per file: named in its header, leading every row (_ledger_rows)
EXPLORATORY = "exploratory"  # a /2 header's mark of an exploratory export, which a registered analysis refuses
LEDGER_SCHEMAS = (LEDGER_SCHEMA, SINGLE_SPLIT_LEDGER)
# "refused" (agent-run/17; design log, change-log entry 91): the provider declined or the final message declines the task
# (the runner's refusal-v1), scored wrong (prereg §2 "Refusals", §4)
STATUSES = ("submitted", "abstained", "failed", "capped", "refused", "missing")
MISSING_REASONS = ("infrastructure", "unscored", "absent")
BANK = "bank"
BANK_ITEMS = ("answer_changing", "answer_neutral", "untouched", "rewrite")
ALIGNMENTS = ("aligned", "fallback", "none", "unrecorded")
OUTPUT_STATUSES = ("F", "P", "U")
LEDGER_USES = ("checker", "both", "replay")
NT_OUTCOMES = ("pass", "flag", "unknown", "not_applicable")
UNRECORDED = "unrecorded"

# The runner's refusal setting (a record's agent_config.refusal_rule; feat/e0.1-skeleton 3675da5, harness/agent/loop.py)
# per registered refusal rule: "refused" runs REFUSAL_RULE, refusal-v1 (its REFUSAL_PHRASES); "failed" runs none (a refusal
# was failed:other, a textual decline failed:no_submission), and is what a record before agent-run/17 ran under
# (replay.config_of's default for a record without the setting)
REFUSAL_SETTINGS = {"refusal-v1": "refused"}
NO_REFUSAL_RULE = "failed"
REFUSAL_SIGNALS = ("provider:", "phrase:")  # outcome.refusal.signal: "provider:<stop reason>" or "phrase:<name>"

# (call, "aligned" | "fallback" | "none"): contract v2 rule 22, scripts/pilot_summary.checked_call
CheckedCallFn = Callable[[dict], tuple[dict | None, str]]


class InputError(ValueError):
    """An input row or file is malformed or inconsistent with the others."""


@dataclass(frozen=True)
class Attempt:
    run_id: str
    graph: str
    model: str
    cohort: str  # the prompt condition the output was produced under
    question_id: str
    base_question_id: str
    template: str
    split: str
    repetition: int
    status: str
    output: bool  # a scored submission: counted in N_out
    correct: bool  # exact match by the scorer; False whenever there is no output
    f1: float
    rules_version: str | None = None
    gold_empty: str = UNRECORDED  # "yes" | "no" | "unrecorded"
    tier: str = UNRECORDED
    engine_sensitive: str = UNRECORDED  # "yes" | "no" | "unrecorded"
    nested_graph: str = UNRECORDED  # the checked query's engine flag: "yes" | "no" | "unrecorded"
    alignment: str = UNRECORDED
    sib_stratum: str = UNRECORDED
    cost_usd: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    wall_s: float | None = None
    tool_calls: int = 0
    missing_reason: str | None = None  # set exactly when status is "missing"
    questions_sha256: str | None = None  # the question set the run used: ids are keyed on (id, questions_sha256)
    refusal_signal: str | None = None  # set exactly when status is "refused": "provider:<stop reason>" | "phrase:<name>"

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise InputError(f"{self.run_id}: unknown attempt status {self.status!r}")
        if (self.status == "refused") != (self.refusal_signal is not None):
            raise InputError(f"{self.run_id}: a refused attempt, and only it, records its refusal signal")
        if self.refusal_signal is not None and not self.refusal_signal.startswith(REFUSAL_SIGNALS):
            raise InputError(f"{self.run_id}: refusal signal {self.refusal_signal!r} is neither provider:<stop reason> "
                             f"nor phrase:<name>")
        if (self.status == "missing") != (self.missing_reason is not None):
            raise InputError(f"{self.run_id}: a missing attempt, and only it, records why it is missing")
        if self.missing_reason is not None and self.missing_reason not in MISSING_REASONS:
            raise InputError(f"{self.run_id}: unknown missing reason {self.missing_reason!r}")
        if self.output and self.status != "submitted":
            raise InputError(f"{self.run_id}: an output must be a submission")
        if self.correct and not self.output:
            raise InputError(f"{self.run_id}: correct without an output")
        if self.alignment not in ALIGNMENTS:
            raise InputError(f"{self.run_id}: unknown alignment {self.alignment!r}")


@dataclass(frozen=True)
class CheckedOutput:
    run_id: str  # the initial output's run record
    graph: str
    model: str
    cohort: str
    source: str  # open-ended: curated, census, generated, union, curated+doc, generic_retry/<source>, ...
    use: str  # checker | both (online, with retries) | replay (detection only)
    question_id: str
    base_question_id: str
    initial_correct: bool  # C/W by the scorer
    status: str  # F | P | U
    substantive: bool  # at least one substantive ontology-derived check decided (v2)
    any_unknown: bool = False
    flagged_check_types: tuple[str, ...] = ()
    branch_local_only: bool = False  # every flag is branch-local (rules 7, 17)
    data_fault: bool = False  # a flag annotated as a witnessed data fault (rule 10)
    checked_query: str = UNRECORDED  # aligned | fallback (rule 22)
    number_traceability: str | None = None  # its own layer (rule 8)
    retried: bool = False
    retry_status: str | None = None
    retry_correct: bool | None = None  # by the scorer; False when the retry failed or abstained
    check_s: float | None = None
    evidence_triples: int | None = None
    retry_usd: float | None = None
    fault_class: str | None = None  # bank rows: the class of an answer-changing mutation
    bank_item: str | None = None  # bank rows: answer_changing | answer_neutral | untouched | rewrite
    questions_sha256: str | None = None  # the question set of the output's question (or of the bank's item)
    # the rows the checked query returned: to the agent for an output (rule 22's checked query), when the bank was built
    # for a bank item; None when not recorded (the empty-result stratum, retry on empty's trigger; prereg §7)
    result_rows: int | None = None
    # the patterns confined to a graph neither the census nor the shared hierarchy describes (OutputReport, entry 72);
    # None when not recorded (checker-ledger/1)
    graph_outside_census: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "flagged_check_types", tuple(self.flagged_check_types))
        where = f"{self.run_id} ({self.source}, {self.use})"
        for name in ("result_rows", "graph_outside_census"):
            n = getattr(self, name)
            if n is not None and (not isinstance(n, int) or isinstance(n, bool) or n < 0):
                raise InputError(f"{where}: {name} must be a count or None, not {n!r}")
        if self.status not in OUTPUT_STATUSES:
            raise InputError(f"{where}: status must be one of F, P, U, not {self.status!r}")
        if self.use not in LEDGER_USES:
            raise InputError(f"{where}: use must be one of {LEDGER_USES}, not {self.use!r}")
        if (self.status == "F") != bool(self.flagged_check_types):
            raise InputError(f"{where}: a flagged output names the check types that flagged, and only it")
        if self.branch_local_only and self.status != "F":
            raise InputError(f"{where}: branch-local only applies to a flagged output")
        if self.checked_query not in ALIGNMENTS:
            raise InputError(f"{where}: unknown checked-query status {self.checked_query!r}")
        if self.number_traceability is not None and self.number_traceability not in NT_OUTCOMES:
            raise InputError(f"{where}: unknown number-traceability outcome {self.number_traceability!r}")
        if self.retried:
            if self.use == "replay":
                raise InputError(f"{where}: replay is detection only and never retries")
            if self.status != "F":
                raise InputError(f"{where}: only a flagged output is retried (rule 9)")
            if self.retry_status not in STATUSES or self.retry_correct is None:
                raise InputError(f"{where}: a retry records its status and whether it is correct")
            if self.retry_correct and self.retry_status != "submitted":
                raise InputError(f"{where}: a retry is correct only as a submission")
        elif self.retry_status is not None or self.retry_correct is not None:
            raise InputError(f"{where}: retry fields set on an output that was not retried")
        elif self.status == "F" and self.use != "replay":
            # rule 9 and the frozen online settings (prereg §5): every flag of an online arm is retried.
            # A flagged row without its retry would keep the initial answer and bias the net change, so it
            # is refused; a retry that was lost is recorded as retry_status "missing" (§4 "Missing data")
            raise InputError(f"{where}: a flagged output of an online arm was not retried; record a lost retry "
                             f"as retry_status 'missing'")
        if self.cohort == BANK:
            if self.use != "replay":
                raise InputError(f"{where}: a fault-bank row is detection only (use replay)")
            if self.bank_item not in BANK_ITEMS:
                raise InputError(f"{where}: a fault-bank row needs bank_item in {BANK_ITEMS}")
            if self.initial_correct != (self.bank_item != "answer_changing"):
                raise InputError(f"{where}: an answer-changing mutation is W and every control is C")
            if (self.bank_item == "answer_changing") != (self.fault_class is not None):
                raise InputError(f"{where}: an answer-changing mutation, and only it, has a fault class")
        elif self.bank_item is not None:
            raise InputError(f"{where}: bank_item set outside the fault bank")

    @property
    def retry_missing(self) -> bool:
        return self.retried and self.retry_status == "missing"


# --- attempts from run records ---------------------------------------------------------------


def _yes_no(value: Any) -> str:
    return "yes" if value else "no"


def _token_totals(record: dict) -> tuple[int, int]:
    totals = (record.get("tokens") or {}).get("totals") or {}
    n_in = sum(int(totals.get(k) or 0) for k in ("input_tokens", "cache_read_tokens", "cache_write_tokens"))
    return n_in, int(totals.get("output_tokens") or 0)


def store_name(record: dict) -> str:
    """The store a record names (``store.name``), its arm's graph. The runner keeps it from the run's store description
    (harness/agent/metadata.py STORE_KEYS), and every real description names its store; a record that names none is
    refused, naming the record and the field, since its arm's graph would be unknown."""
    store = record.get("store")
    if not isinstance(store, dict):
        raise InputError(f"{record.get('run_id')}: it has no store description (store), so its arm's graph is unknown")
    name = store.get("name")
    if not isinstance(name, str) or not name:
        raise InputError(f"{record.get('run_id')}: its store names no store (store.name), so its arm's graph is "
                         f"unknown; the store description the run was given must name its store")
    return name


def attempt_from_record(record: dict, question: dict | None, *, checked_call: CheckedCallFn | None = None,
                        tier_if_missing: str | None = None, cohort: str | None = None) -> Attempt:
    """One attempt row. ``question`` is the record's question from the questions file the batch
    ran on (strata come from it); ``checked_call`` gives the checked query (rule 22), whose
    engine flags are read from the record or, for records older than agent-run/5, recomputed
    from the query text; ``tier_if_missing`` labels questions from files written before the
    two tiers (design change log entry 6) and is recorded by the caller. ``cohort`` is the
    record's prompt arm as its batch decides it (load_batch: a grid record's cell), its prompt
    condition when not given (a plain run's one arm)."""
    outcome = record["outcome"]
    score = record.get("score") or {}
    output = outcome["status"] == "submitted" and score.get("outcome") == "scored"
    status, missing_reason = outcome["status"], None
    if is_infrastructure(record):
        status, missing_reason = "missing", "infrastructure"
    elif outcome["status"] == "submitted" and not output:
        status, missing_reason = "missing", "unscored"
    question = question or {}
    alignment, nested = UNRECORDED, UNRECORDED
    if checked_call is not None:
        call, alignment = checked_call(record) if outcome["status"] == "submitted" else (None, "none")
        flags = (call.get("engine_flags") or engine_flags(call["arguments"].get("query", ""))) if call else []
        nested = _yes_no("nested_graph" in flags)
    if output and "gold_empty" in score:
        gold_empty = _yes_no(score["gold_empty"])
    elif "gold_row_count" in question:
        gold_empty = _yes_no(question["gold_row_count"] == 0)
    else:
        gold_empty = UNRECORDED
    n_in, n_out = _token_totals(record)
    return Attempt(
        run_id=record["run_id"],
        graph=store_name(record),
        model=record["model"],
        cohort=record["condition"] if cohort is None else cohort,
        question_id=record["question_id"],
        base_question_id=record.get("base_question_id") or record["question_id"],
        template=record.get("template") or question.get("template") or UNRECORDED,
        split=record.get("split") or question.get("split") or UNRECORDED,
        repetition=int(record["repetition"]),
        status=status,
        output=output,
        correct=output and bool(score.get("exact")),
        f1=float(score.get("f1") or 0.0) if output else 0.0,
        rules_version=(score.get("rules") or {}).get("version"),
        gold_empty=gold_empty,
        tier=question.get("tier") or tier_if_missing or UNRECORDED,
        engine_sensitive=_yes_no(question["engine_sensitive"]) if "engine_sensitive" in question else UNRECORDED,
        nested_graph=nested,
        alignment=alignment,
        sib_stratum=question.get("sib_stratum") or UNRECORDED,
        cost_usd=estimate_record_cost(record).get("usd"),
        input_tokens=n_in,
        output_tokens=n_out,
        wall_s=record.get("wall_s"),
        tool_calls=len(record.get("tool_calls") or []),
        missing_reason=missing_reason,
        refusal_signal=(outcome.get("refusal") or {}).get("signal") if status == "refused" else None,
    )


def refusal_setting(record: dict) -> str:
    """The runner's refusal setting a record ran under (agent_config.refusal_rule), "failed" when it records none (a
    record before agent-run/17, as replay.config_of reads it)."""
    return (record.get("agent_config") or {}).get("refusal_rule", NO_REFUSAL_RULE)


def final_message(record: dict) -> dict[str, str | None]:
    """A record's final message: its last assistant turn's text, and the outcome's error message (a provider's refusal
    can leave the text empty, its stop reason in the message)."""
    turns = [t for t in record.get("transcript") or [] if t.get("role") == "assistant"]
    return {"text": turns[-1].get("text") if turns else None, "error_message": record["outcome"].get("error_message")}


def is_infrastructure(record: dict) -> bool:
    return record["outcome"].get("failure_type") == "infrastructure"


# Change-log entry 161 (b): a tool call the output cap cut (the reply stopped at the output limit inside a tool call)
# is an output-cap hit, not a malformed call. The runner records it as it always has, failed:malformed_tool_call with
# the call's status "truncated" and the call never run (harness.agent.loop), so every record replays byte for byte;
# the analysis labels it from that mark. It is scored wrong either way: only its label changes.
OUTPUT_CAP_LABEL = "capped:output"
OUTPUT_CAP_RULE = ("output-cap-v1: failed:malformed_tool_call whose cut call the runner marked truncated (stop reason at "
                   "the output limit inside a tool call) is labelled capped:output (change-log entry 161)")


def output_cap_hit(record: dict) -> bool:
    """Whether the record's attempt ended on a tool call the output cap cut (OUTPUT_CAP_RULE)."""
    outcome = record.get("outcome") or {}
    return (outcome.get("status") == "failed" and outcome.get("failure_type") == "malformed_tool_call"
            and any(c.get("status") == "truncated" for c in record.get("tool_calls") or []))


def outcome_class(record: dict) -> str:
    """The attempt's outcome label, as the runner's (submitted | abstained | refused | failed:<type> | capped:<cap>),
    with an output-cap hit labelled capped:output (OUTPUT_CAP_RULE)."""
    if output_cap_hit(record):
        return OUTPUT_CAP_LABEL
    outcome = record["outcome"]
    if outcome["status"] == "capped":
        return f"capped:{outcome['cap']}"
    if outcome["status"] == "failed":
        return f"failed:{outcome['failure_type']}"
    return outcome["status"]


def output_cap_report(records: Iterable[dict], arm_of: Callable[[dict], str]) -> dict[str, Any]:
    """The output-cap hits among ``records`` (OUTPUT_CAP_RULE): their count per arm (``arm_of``) and their run ids."""
    hits = [r for r in records if output_cap_hit(r)]
    return {"label": OUTPUT_CAP_LABEL, "rule": OUTPUT_CAP_RULE,
            "arms": dict(sorted(Counter(arm_of(r) for r in hits).items())),
            "run_ids": sorted(r["run_id"] for r in hits)}


@dataclass
class Batch:
    path: Path
    records: list[dict]  # records/ of the prompt arms, infrastructure finals included
    side_counts: dict[tuple[str, str, str], Counter]  # (graph, model, cohort) -> infra and harness counts
    cohorts: dict[str, str] = dataclasses.field(default_factory=dict)  # run id -> its prompt arm's cohort
    branch: list[dict] = dataclasses.field(default_factory=list)  # a grid's branch-cell records: checker arms'
    branch_side: dict[str, Counter] = dataclasses.field(default_factory=dict)  # branch cell -> its side counts


def _read_json_dir(path: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(path.glob("*.json"))
            if not p.name.endswith(".setaside.json")]


def _cell(record: dict) -> Any:
    return (record.get("load") or {}).get("arm")


def _prompt_differs(record: dict, pins: dict) -> list[str]:
    """The prompt pins (gate.PROMPT_PINS) a record's prompt does not have; a prompt naming another condition than its
    record is refused."""
    prompt = record.get("prompt") or {}
    if prompt.get("condition") != record.get("condition"):
        raise InputError(f"{record.get('run_id')}: its prompt names the condition {prompt.get('condition')!r}, not its "
                         f"own")
    found = {"condition": record.get("condition"), "prompt_id": prompt.get("id"),
             "template_sha256": prompt.get("template_sha256"), "nudge_sha256": prompt.get("nudge_sha256"),
             "ontology_block_sha256": (record.get("agent_config") or {}).get("ontology_block_sha256")}
    return [pin for pin in PROMPT_PINS if found[pin] != pins[pin]]


def record_cohort(record: dict, cells: GridCells | None) -> str | None:
    """A record's prompt arm, from its batch's layout, or None for a grid's branch-cell record. A plain run has one arm,
    its condition (scripts/run_pilot.py: its records' load.arm, when they record one, is it). A grid record's cell is its
    load.arm (scripts/run_pilot.py record_arm), which its manifest must pin (gate.grid_cells): an initial cell is the
    record's cohort, and its prompt must be the one pinned for that cell, field by field (gate.PROMPT_PINS; the
    condition alone would let a generated-1 record pass as generated-2, both prompted under "generated"); a branch
    cell's record checks its parent's output (its checker, branched_from) and is no attempt of a prompt arm. Anything
    else, a record whose two disagree, is refused, naming the record and the fields, never their values."""
    run, cell = record.get("run_id"), _cell(record)
    if cells is None:
        if cell is not None and cell != record.get("condition"):
            raise InputError(f"{run}: its cell {cell!r} is not its condition {record.get('condition')!r}, a plain "
                             f"run's one arm")
        return record["condition"]
    if not isinstance(cell, str) or not cell:
        raise InputError(f"{run}: a grid run's record names no cell (load.arm), so its arm is unknown")
    if cell in cells.branch:
        return None
    if cell not in cells.initial:
        raise InputError(f"{run}: its cell {cell!r} is not one its batch's manifest pins")
    differs = _prompt_differs(record, cells.initial[cell])
    if differs:
        raise InputError(f"{run}: its prompt ({differs}) is not the one its batch's manifest pins for its cell {cell!r}")
    if record.get("checker") is not None:
        raise InputError(f"{run}: an initial cell's record carries a checker (its cell {cell!r} runs no check)")
    return cell


def _check_branch(record: dict, cells: GridCells, by_run: dict[str, dict]) -> None:
    """A branch cell's record: it runs under its parent cell's prompt (the checker ledger takes its cohort from it), and
    its checker names its parent (branched_from), a record of the cell's parent cell in the batch with its question,
    route model and repetition."""
    run, cell = record.get("run_id"), _cell(record)
    differs = _prompt_differs(record, cells.initial[cells.branch[cell]])
    if differs:
        raise InputError(f"{run}: its prompt ({differs}) is not the one its batch's manifest pins for its cell's parent "
                         f"{cells.branch[cell]!r}")
    checker = record.get("checker")
    if not isinstance(checker, dict) or not checker.get("branched_from"):
        raise InputError(f"{run}: a branch cell's record carries no checker naming its parent (checker.branched_from); "
                         f"its cell is {cell!r}")
    parent = by_run.get(checker["branched_from"])
    if parent is None or _cell(parent) != cells.branch[cell]:
        raise InputError(f"{run}: its parent {checker['branched_from']} is not a record of its cell's parent "
                         f"{cells.branch[cell]!r} in the batch")
    if any(parent.get(k) != record.get(k) for k in ("question_id", "model", "repetition")):
        raise InputError(f"{run}: its parent {checker['branched_from']} is another attempt's (question, route model, "
                         f"repetition)")


def load_batch(batch: Path, cells: GridCells | None = None) -> Batch:
    """The records of one runner OUT directory, and per arm: attempts that failed
    infrastructure on every try (``infra_final``, left out of N_att), earlier infrastructure
    tries (``infra_tries``) and ``failed:harness`` set-asides (``failed_harness``). ``cells`` are
    a grid batch's (gate.grid_cells; None for a plain run's): each record's arm is decided by
    record_cohort, and a branch cell's records are kept apart (``branch``, ``branch_side``)."""
    batch = Path(batch)
    loaded = _read_json_dir(batch / "records")
    by_run = {r["run_id"]: r for r in loaded}
    records, branch, cohorts = [], [], {}
    for r in loaded:
        cohort = record_cohort(r, cells)
        if cohort is None:
            _check_branch(r, cells, by_run)
            branch.append(r)
        else:
            records.append(r)
            cohorts[r["run_id"]] = cohort
    counts: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    branch_side: dict[str, Counter] = defaultdict(Counter)

    def count(r: dict, what: str) -> None:
        cohort = record_cohort(r, cells)
        if cohort is None:
            branch_side[_cell(r)][what] += 1
        else:
            counts[(store_name(r), r["model"], cohort)][what] += 1

    for r in records:
        if is_infrastructure(r):
            count(r, "infra_final")
    for r in branch:
        if is_infrastructure(r):
            count(r, "infra_final")
    finals = set(by_run)
    for r in _read_json_dir(batch / "infrastructure"):
        if r["run_id"] not in finals:  # the final try of an all-infrastructure attempt is in both
            count(r, "infra_tries")
    for r in _read_json_dir(batch / "harness"):
        count(r, "failed_harness")
    return Batch(batch, records, dict(counts), cohorts, branch, dict(branch_side))


def check_one_stack(batches: Iterable[tuple[Path, list[dict], dict | None]], *, require_route: bool = False) -> None:
    """One stack per (model, graph) (science, design log, 21 Oct 16:36; change-log entry 144): a model's arms in one graph
    are never pooled across stacks. ``batches``: each OUT's path, its records (prompt arms and branch cells) and its
    deployment-identity.json (None without one). Refused, naming every case at once:

    * a (graph, model) whose records carry more than one route base URL (route.base_url, the endpoint the runner
      recorded; a record naming none counts as one more), in one OUT or across OUTs; the URLs are named by sha256 only
      (the replication package carries no hosts);
    * a (graph, model) whose OUTs' deployment identities name more than one stack (``stacks_by_route_model``);
    * an OUT whose deployment identity names no stack for a model it has records of (the mapping must hold);
    * with ``require_route`` (a confirmatory analysis), a record that names no route base URL (none or empty), an
      OUT with no deployment-identity.json (freeze-day e writes one in every test OUT before its first start), and a
      (graph, model) whose records name more than one store endpoint (a model's arms never straddle store replicas;
      science, design log, 21 Oct 17:28)."""
    urls: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    stacks: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    stores: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    problems: list[str] = []
    for path, records, identity in batches:
        named = None if identity is None else (identity.get("stacks_by_route_model") or {})
        if identity is None and require_route:
            problems.append(f"{path}: no deployment-identity.json (each test OUT's stack per model)")
        unnamed, routeless = set(), set()
        for r in records:
            key = (store_name(r), r["model"])
            url = (r.get("route") or {}).get("base_url") or None  # empty: no endpoint either
            if url is None and require_route:
                routeless.add(r["model"])
            urls[key]["none" if url is None else "sha256:" + hashlib.sha256(str(url).encode()).hexdigest()[:12]].add(
                str(path))
            endpoint = (r.get("store") or {}).get("endpoint")
            stores[key]["none" if not endpoint else "sha256:" + hashlib.sha256(str(endpoint).encode()).hexdigest()[:12]
                        ].add(str(path))
            if named is not None:
                if r["model"] in named:
                    stacks[key][str(named[r["model"]])].add(str(path))
                else:
                    unnamed.add(r["model"])
        problems += [f"{path}: its deployment-identity.json names no stack for {m}" for m in sorted(unnamed)]
        problems += [f"{path}: records of {m} name no route base URL" for m in sorted(routeless)]
    problems += [f"{model} in {graph}: {len(seen)} route base URLs ("
                 + "; ".join(f"{url} in {', '.join(sorted(paths))}" for url, paths in sorted(seen.items())) + ")"
                 for (graph, model), seen in sorted(urls.items()) if len(seen) > 1]
    problems += [f"{model} in {graph}: {len(seen)} stacks ("
                 + "; ".join(f"{stack} in {', '.join(sorted(paths))}" for stack, paths in sorted(seen.items())) + ")"
                 for (graph, model), seen in sorted(stacks.items()) if len(seen) > 1]
    if require_route:
        problems += [f"{model} in {graph}: {len(seen)} store endpoints ("
                     + "; ".join(f"{e} in {', '.join(sorted(paths))}" for e, paths in sorted(seen.items())) + ")"
                     for (graph, model), seen in sorted(stores.items()) if len(seen) > 1]
    if problems:
        raise InputError("one stack per (model, graph) (change-log entry 144): a model's arms in a graph are not "
                         "pooled across stacks: " + " | ".join(problems))


def load_questions(path: Path) -> dict[str, dict]:
    questions = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            q = json.loads(line)
            questions[q["id"]] = q
    return questions


def attempts_from_batch(batch: Batch, questions: dict[str, dict] | None, **kw) -> list[Attempt]:
    """Every record of the batch as an attempt; one that failed infrastructure on every try is
    a missing attempt (prereg §4)."""
    rows = []
    for r in batch.records:
        question = (questions or {}).get(r["question_id"])
        if questions is not None and question is None:
            raise InputError(f"{r['run_id']}: question {r['question_id']} is not in the questions file")
        rows.append(attempt_from_record(r, question, cohort=batch.cohorts.get(r["run_id"]), **kw))
    return rows


def complete_design(attempts: Iterable[Attempt], questions: dict[str, dict], n_reps: int) -> list[Attempt]:
    """The registered design, every (question, repetition) of every cohort present: a
    duplicate attempt, or one outside the design, is refused; a registered attempt with no
    record becomes a missing attempt (prereg §4 "Missing data"; "Fixed stopping": exactly
    ``n_reps`` repetitions of every registered question)."""
    rows = list(attempts)
    seen: dict[tuple, str] = {}
    cohorts = sorted({(a.graph, a.model, a.cohort) for a in rows})
    for a in rows:
        key = (a.graph, a.model, a.cohort, a.question_id, a.repetition)
        if key in seen:
            raise InputError(f"attempts {seen[key]} and {a.run_id} are the same registered attempt {key}")
        seen[key] = a.run_id
        if a.question_id not in questions or not 0 <= a.repetition < n_reps:
            raise InputError(f"{a.run_id}: question {a.question_id} repetition {a.repetition} is outside the design")
    for graph, model, cohort in cohorts:
        for qid, q in sorted(questions.items()):
            for rep in range(n_reps):
                if (graph, model, cohort, qid, rep) in seen:
                    continue
                rows.append(Attempt(
                    run_id=f"missing:{graph}:{model}:{cohort}:{qid}:{rep}", graph=graph, model=model, cohort=cohort,
                    question_id=qid, base_question_id=q.get("base_question_id") or qid,
                    template=q.get("template") or UNRECORDED, split=q.get("split") or UNRECORDED, repetition=rep,
                    status="missing", output=False, correct=False, f1=0.0, tier=q.get("tier") or UNRECORDED,
                    missing_reason="absent"))
    return rows


# --- the fault bank's frozen manifest (prereg §3 "Bank metrics", §5 "Fault bank") ----------------

BANK_SCHEMA = "fault-bank/1"
# the manifest's ``bank`` value -> the ledger's bank_item; a rewrite is an item of kind "rewrite" (bank v2: bank "control")
_BANK_KINDS = {"control": "untouched", "answer_changing": "answer_changing", "answer_neutral": "answer_neutral",
               "rewrite": "rewrite"}


def _bank_meta(question_id: str, base_question_id: str, bank_item: str, fault_class: str | None) -> tuple:
    return question_id, base_question_id, bank_item, fault_class if bank_item == "answer_changing" else None


def check_bank_pairing(rows: Iterable[CheckedOutput]) -> None:
    """Every bank source of a graph holds the same items with the same metadata (prereg §3: one
    denominator common to all sources). A source missing an unflagged mutation would silently raise
    its recall while another source keeps the mutation in its denominator, so any difference is
    refused."""
    by_source: dict[tuple[str, str], dict[str, tuple]] = defaultdict(dict)
    for r in rows:
        if r.cohort == BANK:
            by_source[(r.graph, r.source)][r.run_id] = _bank_meta(r.question_id, r.base_question_id, r.bank_item,
                                                                  r.fault_class)
    for graph in sorted({g for g, _ in by_source}):
        sources = sorted(s for g, s in by_source if g == graph)
        first = by_source[(graph, sources[0])]
        for other in sources[1:]:
            items = by_source[(graph, other)]
            if set(items) != set(first):
                diff = sorted(set(items) ^ set(first))
                raise InputError(f"{graph}: bank sources {sources[0]} and {other} do not hold the same items "
                                 f"({len(diff)} differ, e.g. {diff[:3]})")
            for item_id, meta in first.items():
                if items[item_id] != meta:
                    raise InputError(f"{graph}: bank item {item_id} differs between bank sources {sources[0]} "
                                     f"{meta} and {other} {items[item_id]}")


# The mutations the builder's summary counts in its fault_mix (harness/faults/manifest.py summarise): a rejected draw and
# a descriptive one (the type-consistent stratum, summarised apart in descriptive_mix) are not in it.
_MIX_BANKS = ("answer_changing", "answer_neutral")


def _manifest_counts(items: list[dict], split: str) -> dict[str, Any]:
    """The manifest summary's figures recomputed from its items: questions of the split, and per
    fault class the answer-changing and answer-neutral mutations (``_MIX_BANKS``), as the builder counts them."""
    mix: dict[str, Counter] = defaultdict(Counter)
    for i in items:
        if i["split"] == split and i["kind"] == "mutation" and i["bank"] in _MIX_BANKS:
            mix[i["fault_class"]][i["bank"]] += 1
    questions = {i["question_id"] for i in items if i["split"] == split}
    return {"questions": len(questions), "fault_mix": {fc: dict(c) for fc, c in sorted(mix.items())}}


def bank_items_bytes(manifest: Path) -> tuple[Path, bytes]:
    """The bank's items file beside its manifest, as the manifest froze it (``items.jsonl``): that file, or, where the
    repository keeps it gzip-compressed (``items.jsonl.gz``, as the frozen UniProt bank is committed), its decompressed
    bytes; empty when neither is there. The caller checks the bytes against the sha256 the manifest records."""
    plain = Path(manifest).parent / "items.jsonl"
    if plain.is_file():
        return plain, plain.read_bytes()
    packed = plain.with_name("items.jsonl.gz")
    if packed.is_file():
        return packed, gzip.decompress(packed.read_bytes())
    return plain, b""


def read_question_ids(path: Path) -> list[str]:
    """A declared question subset: a JSON array of question ids, or one id per line. Empty, repeated or non-string ids
    are refused."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        ids = json.loads(text) if text.lstrip().startswith("[") else [ln.strip() for ln in text.splitlines() if ln.strip()]
    except ValueError as exc:
        raise InputError(f"{path}: not a JSON array of question ids ({exc})") from None
    if not ids or not all(isinstance(q, str) and q for q in ids):
        raise InputError(f"{path}: a question subset is a non-empty list of question ids")
    if len(set(ids)) != len(ids):
        raise InputError(f"{path}: a question id is listed twice")
    return ids


def question_subset_sha256(ids: Iterable[str]) -> str:
    """The sha256 of a question subset: of the compact JSON array of its ids, sorted."""
    return hashlib.sha256(json.dumps(sorted(ids), separators=(",", ":")).encode("utf-8")).hexdigest()


def check_bank_manifest(rows: Iterable[CheckedOutput], manifest: Path, sha256: str, split: str,
                        questions: Iterable[str] | None = None) -> dict[str, Any]:
    """The bank rows of the confirmatory analysis equal the frozen fault bank (prereg §3, §5): the
    manifest's sha256 is the frozen one; its ``items.jsonl`` is the file it froze (by the sha256 the
    manifest records); the manifest's summary counts for the split agree with those items (stratum
    keys such as ``type_wrong@near`` are sub-counts and are skipped); and every bank source's rows are
    exactly the split's items (untouched controls, rewrites and mutations that were not rejected), each with the
    item's id as ``run_id`` and its question, base question, kind and fault class. Anything else is
    refused. Returns the item counts.

    ``questions`` declares a subset of the split's questions (a test run only: the pipeline refuses it in a
    confirmatory run): the summary is still checked on the whole split, and the rows must then be exactly the items of
    those questions; a declared question that is not a question of the split's items is refused."""
    manifest = Path(manifest)
    actual = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if actual != sha256:
        raise InputError(f"{manifest}: sha256 {actual} is not the frozen bank manifest's {sha256}")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("schema") != BANK_SCHEMA:
        raise InputError(f"{manifest}: not a {BANK_SCHEMA} manifest")
    items_path, raw = bank_items_bytes(manifest)
    frozen_items = (data.get("files") or {}).get("items.jsonl")
    if hashlib.sha256(raw).hexdigest() != frozen_items:
        raise InputError(f"{items_path}: not the items.jsonl the manifest froze (sha256 {frozen_items})")
    items = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    recomputed = _manifest_counts(items, split)
    summary = data.get("summary") or {}
    stated_mix = {fc: dict(c) for fc, c in sorted((summary.get("fault_mix") or {}).get(split, {}).items())
                  if "@" not in fc}
    stated = {"questions": (summary.get("questions") or {}).get(split), "fault_mix": stated_mix}
    if stated != recomputed:
        raise InputError(f"{manifest}: the summary's {split} counts {stated} disagree with its items {recomputed}")
    expected = {i["id"]: _bank_meta(i["question_id"], i.get("base_question_id") or i["question_id"],
                                    "rewrite" if i.get("kind") == "rewrite" else _BANK_KINDS[i["bank"]], i.get("fault_class"))
                for i in items if i["split"] == split and i["bank"] in _BANK_KINDS}
    population = f"frozen manifest's {split} split"
    subset = None
    if questions is not None:
        questions = list(questions)
        if len(set(questions)) != len(questions):
            raise InputError("the declared question subset names a question twice")
        subset = sorted(set(questions))
        unknown = sorted(set(subset) - {meta[0] for meta in expected.values()})
        if not subset or unknown:
            raise InputError(f"the declared question subset names {len(unknown)} question(s) with no checked item in the "
                             f"{population}, e.g. {unknown[:3]}" if unknown else "the declared question subset is empty")
        expected = {k: meta for k, meta in expected.items() if meta[0] in set(subset)}
        population = f"declared subset ({len(subset)} questions) of the {population}"
    by_source: dict[str, dict[str, tuple]] = defaultdict(dict)
    for r in rows:
        if r.cohort == BANK:
            by_source[r.source][r.run_id] = _bank_meta(r.question_id, r.base_question_id, r.bank_item, r.fault_class)
    if not by_source:
        raise InputError("the ledger has no fault-bank rows")
    for source, got in sorted(by_source.items()):
        absent, extra = sorted(set(expected) - set(got)), sorted(set(got) - set(expected))
        if absent:
            raise InputError(f"{source}: {len(absent)} item(s) of the {population} are not in the ledger, "
                             f"e.g. {absent[:3]}")
        if extra:
            raise InputError(f"{source}: {len(extra)} ledger row(s) are not in the {population}, e.g. {extra[:3]}")
        for item_id, meta in expected.items():
            if got[item_id] != meta:
                raise InputError(f"{source}: bank item {item_id} is {got[item_id]} in the ledger but {meta} in the "
                                 f"frozen manifest")
    kinds = Counter(meta[2] for meta in expected.values())
    out = {"items": len(expected), "sources": sorted(by_source), **{k: kinds[k] for k in sorted(kinds)}}
    if subset is not None:
        out["question_subset"] = {"questions": len(subset), "sha256": question_subset_sha256(subset)}
    return out


# --- files ----------------------------------------------------------------------------------

HEADER_LIMIT = 4096  # bytes: a header line is a few dozen
ROW_LIMIT = 1 << 20  # bytes: a ledger row is a few hundred
SPLIT_NAME_LIMIT = 64
# A checker-ledger/2 row's rest is read this many bytes at a time, so a read runs at most _CHUNK - 1 = 9 bytes into
# the next row: never past the 10 bytes '{"split":"' with which every row must begin (more when spaced), so no byte
# of a row's split value, nor anything after it, is read before that row's split is checked.
_CHUNK = 10


class _Reader:
    """An unbuffered reader that never reads ahead: ``take_byte`` and ``take_line`` read exactly what they return;
    ``take_rest`` reads to the end of the current line in _CHUNK-byte reads and keeps the at most _CHUNK - 1 bytes read
    past its newline for the next ``take_byte``. Only ``read`` is called on the file. (Its method names are its own:
    inference.py's closure rule binds every in-repo method named as a registered function reads an attribute.)"""

    def __init__(self, fh: Any) -> None:
        self.fh, self.carry = fh, bytearray()

    def take_byte(self) -> bytes:
        if self.carry:
            b = bytes(self.carry[:1])
            del self.carry[:1]
            return b
        return self.fh.read(1)

    def take_line(self, limit: int) -> bytes | None:
        """The line without its newline, byte by byte (the whole rest at the end of the file); None past ``limit``."""
        out = bytearray()
        while len(out) < limit:
            b = self.take_byte()
            if not b or b == b"\n":
                return bytes(out)
            out += b
        return None

    def take_rest(self, limit: int) -> bytes | None:
        """The rest of the current line without its newline; None past ``limit``."""
        out = bytearray()
        while len(out) <= limit:
            chunk = bytes(self.carry) if self.carry else self.fh.read(_CHUNK)
            self.carry.clear()
            if not chunk:
                return bytes(out)
            end = chunk.find(b"\n")
            if end >= 0:
                self.carry += chunk[end + 1:]
                return bytes(out + chunk[:end])
            out += chunk
        return None

    def take_all(self) -> bytes:
        return bytes(self.carry) + self.fh.read()


def first_line(path: Path, limit: int = HEADER_LIMIT) -> bytes | None:
    """The file's first line without its newline, read unbuffered one byte at a time so that no byte after the newline
    is read (a buffered readline reads ahead into the rows); the whole file when it has no newline; None when the line
    is longer than ``limit`` bytes."""
    with open(path, "rb", buffering=0) as fh:
        return _Reader(fh).take_line(limit)


def _header(r: _Reader, path: Path) -> dict[str, Any]:
    raw = r.take_line(HEADER_LIMIT)
    try:
        header = json.loads(raw) if raw is not None else None
    except ValueError:
        header = None
    if not isinstance(header, dict):
        raise InputError(f"{path}: its first line is not a header (a JSON object of at most {HEADER_LIMIT} bytes); "
                         f"nothing after it read")
    return header


def _split_name(value: Any) -> bool:
    return (isinstance(value, str) and 0 < len(value) <= SPLIT_NAME_LIMIT
            and all(c.isascii() and (c.isalnum() or c in "_-") for c in value))


def _write_jsonl(path: Path, schema: str, rows: Iterable[Any]) -> Path:
    lines = [json.dumps({"schema": schema})]
    lines.extend(json.dumps(dataclasses.asdict(r), sort_keys=True) for r in rows)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return Path(path)


def _rows(r: _Reader) -> list[dict]:
    return [json.loads(ln) for ln in r.take_all().decode("utf-8").splitlines() if ln.strip()]


def _read_jsonl(path: Path, schema: str) -> list[dict]:
    """The rows of a file whose first line, read alone, is ``{"schema": schema}``; any other is refused with nothing
    past that line read."""
    with open(path, "rb", buffering=0) as fh:
        r = _Reader(fh)
        try:
            header = _header(r, path)
        except InputError:
            header = {}
        if header.get("schema") != schema:
            raise InputError(f"{path}: not a {schema} file (its first line must be {{\"schema\": \"{schema}\"}}); "
                             f"nothing after it read")
        return _rows(r)


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    if len({k for k, _ in pairs}) != len(pairs):
        raise ValueError("duplicate keys")
    return dict(pairs)


def _single_split_row(r: _Reader, path: Path, n: int, first: bytes, split: str) -> dict:
    """Row ``n`` of a checker-ledger/2 file, whose first byte ``first`` was read (_single_split_rows)."""
    raw = bytearray(first)

    def take() -> bytes:
        b = r.take_byte()
        raw.extend(b)
        return b

    def refuse(why: str) -> InputError:
        return InputError(f"{path} row {n}: {why}; refused at its byte {len(raw)}, nothing after it read")

    if first == b"\n":
        raise refuse("an empty line, where a checker-ledger/2 row begins with its split field")
    form = 'a checker-ledger/2 row begins with its split field, \'{"split": "<split>"\''
    if first != b"{":
        raise refuse(form)
    b = take()
    for want in b'"split"':
        if b != bytes([want]):
            raise refuse(form)
        b = take()
    while b == b" ":
        b = take()
    if b != b":":
        raise refuse(form)
    b = take()
    while b == b" ":
        b = take()
    if b != b'"':
        raise refuse(form)
    value = bytearray()
    while (b := take()) != b'"':
        if not b or len(value) >= SPLIT_NAME_LIMIT or not (b.isalnum() or b in (b"_", b"-")):
            raise refuse("its split is not a split name")
        value += b
    if value.decode("ascii") != split:
        raise refuse(f"its split is not the header's {split!r} (a checker-ledger/2 file holds one split)")
    rest = r.take_rest(ROW_LIMIT)
    if rest is None:
        raise refuse(f"longer than {ROW_LIMIT} bytes")
    raw += rest
    try:
        data = json.loads(bytes(raw), object_pairs_hook=_unique_keys)
    except ValueError:
        raise refuse("not a JSON object with unique keys") from None
    if not isinstance(data, dict) or data.get("split") != split:
        raise refuse("not a JSON object with unique keys")
    del data["split"]
    return data


def _single_split_rows(r: _Reader, path: Path, split: str) -> Iterable[dict]:
    """The rows of a checker-ledger/2 file, one at a time. Each must begin with its split field, '{"split": "<name>"'
    (spaces allowed after the field name and after the colon, none before the brace), read byte by byte and compared
    with ``split`` (the header's, already equal to the run's) before any other byte of the row is read: the first byte
    that departs from that form, or a split other than ``split``, refuses the file there, naming the row's position
    only (never a byte of its content). Only a row of the split is then read to its end and parsed (JSON, unique keys;
    its split field dropped)."""
    n = 0
    while first := r.take_byte():
        n += 1
        yield _single_split_row(r, path, n, first, split)


def _ledger_rows(path: Path, split: str | None, single_split: bool, registered: bool = False) -> list[dict]:
    """The rows of a checker ledger, decided from its header line alone before anything else is read: a checker-ledger/1
    header (``{"schema": "checker-ledger/1"}``) or a checker-ledger/2 header, exactly ``{"schema": "checker-ledger/2",
    "split": <split>}`` or, for an exploratory export (the checker loop's export with --exploratory), exactly that and
    ``"exploratory": true``, naming ``split`` (the run's) when given; anything else is refused with nothing past that
    line read. A /1 file's rows hold their split only through their questions, so where a single split is required
    (``single_split``: a test run) a /1 file is refused from its header too; an exploratory export is refused from its
    header where the analysis is a registered one (``registered``: a config given, a test run or a confirmatory run),
    since its rows may name no question set or the frozen bank. A /2 file is read row by row (_single_split_rows).

    The trust boundary: that a /2 file holds one split is its writer's guarantee (feat/checker-loop's checker-ledger/2
    export: --split required, an input row of another split refused, the header naming the split, every row's split
    field written first); the per-row check here is the backstop, which stops at the first row that breaks it with no
    byte of that row past its split field read (at most _CHUNK - 1 bytes of a row are read ahead, all within the bytes
    that must spell its split field's name)."""
    with open(path, "rb", buffering=0) as fh:
        r = _Reader(fh)
        header = _header(r, path)
        schema = header.get("schema")
        if schema not in LEDGER_SCHEMAS:
            raise InputError(f"{path}: not a checker ledger (its first line must be the header {{\"schema\": "
                             f"\"{LEDGER_SCHEMA}\"}} or a {SINGLE_SPLIT_LEDGER} header naming its split); nothing after "
                             f"it read")
        if schema == LEDGER_SCHEMA:
            if single_split:
                raise InputError(f"{path}: a {LEDGER_SCHEMA} file's rows hold their split only through their questions, "
                                 f"so where a single split is required (a test run) it is refused from its header; "
                                 f"nothing after it read")
            return _rows(r)
        exploratory = set(header) == {"schema", "split", EXPLORATORY} and header[EXPLORATORY] is True
        if set(header) != {"schema", "split"} and not exploratory:
            raise InputError(f"{path}: a {SINGLE_SPLIT_LEDGER} header must be exactly {{\"schema\", \"split\"}} (or "
                             f"those and \"{EXPLORATORY}\": true), has {sorted(header)}; nothing after it read")
        if exploratory and registered:
            raise InputError(f"{path}: an exploratory export (its header says so), which a registered analysis does "
                             f"not read: export the ledger without --exploratory; nothing after its header read")
        named = header["split"]
        if not _split_name(named):
            raise InputError(f"{path}: its header's split is not a split name; nothing after it read")
        if split is not None and named != split:
            raise InputError(f"{path}: its header names the split {named!r}, not the run's {split!r}; nothing after it "
                             f"read")
        return list(_single_split_rows(r, path, named))


def _construct(cls: type, data: dict, path: Path, n: int):
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise InputError(f"{path} row {n}: unknown fields {sorted(unknown)}")
    try:
        return cls(**data)
    except TypeError as exc:
        raise InputError(f"{path} row {n}: {exc}") from None


def write_attempt_table(path: Path, attempts: Iterable[Attempt]) -> Path:
    return _write_jsonl(path, ATTEMPT_SCHEMA, attempts)


def load_attempt_table(path: Path) -> list[Attempt]:
    rows = [_construct(Attempt, d, path, i) for i, d in enumerate(_read_jsonl(path, ATTEMPT_SCHEMA), 1)]
    _unique(rows, lambda a: a.run_id, path)
    return rows


def write_checker_ledger(path: Path, rows: Iterable[CheckedOutput]) -> Path:
    return _write_jsonl(path, LEDGER_SCHEMA, rows)


def load_checker_ledger(path: Path, split: str | None = None, *, single_split: bool = False,
                        registered: bool = False) -> list[CheckedOutput]:
    """A checker ledger's rows (_ledger_rows: the header decides alone, an exploratory export refused where
    ``registered``; a checker-ledger/2 file is read row by row, each row's split checked first against the header's and
    the run's ``split``)."""
    rows = [_construct(CheckedOutput, d, path, i)
            for i, d in enumerate(_ledger_rows(path, split, single_split, registered), 1)]
    _unique(rows, lambda r: (r.run_id, r.source, r.use), path)
    return rows


def _unique(rows: list, key: Callable, path: Path) -> None:
    dup = [k for k, n in Counter(key(r) for r in rows).items() if n > 1]
    if dup:
        raise InputError(f"{path}: duplicate rows {dup[:3]}")
