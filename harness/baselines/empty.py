"""Retry on empty (design log, AC1; proposal section 7.6, "execution-only"): the cheapest
rival to every ontology check.

A submitted output whose checked query (rule 22, the query every checker arm reads)
is an executed SELECT that returned no rows gets one retry, with one fixed statement
that the result was empty. It runs as a branch cell like the generic control
(``gen:any``): ``branch_attempt(parent, EmptyRetryChecker(budget), route, sparql)``
replays the parent's initial phase and calls the model only when the trigger fires.

Nothing else triggers it: an ASK result, an output with no executed query, and a
checked query the store refused (an error or a time-out has no rows but is not an
empty result). The trigger status and the checked call's execution status are
recorded, so the proposal's wider execution-only reading (errors and time-outs too)
can be read offline from the same records.

An empty checked query is always the last query executed before the submission:
rule 22 aligns an answer only with a result whose rows hold it, which an empty
result never does, so the message can say "final" truthfully.

The message follows rule 38's wording rules and change-log entry 30: a statement of
what was observed, with no reason and no instruction tail, so the arm differs from
a plain resubmission only by the information that the result was empty.

Design ruling 2 of change-log entry 59 keeps the trigger narrow and has the wider
reading computed offline as a sensitivity analysis: ``trigger_category`` reads each
initial output's category from its record (empty, rows, ASK false or true, refused,
timed out, no query), and ``empty_sensitivity`` gives the detection rates of the
registered reading and of each wider one over a batch of records. Only detection can be
read this way: a wider-only output was never retried.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

from harness.agent.record import RunRecord
from harness.agent.replay import ReplayDivergence
from harness.checkloop.accounting import NOT_ESTIMABLE
from harness.checkloop.checked import (
    checked_call,
    executed_queries,
    received_rows,
    rules_of,
    submissions,
)
from harness.checkloop.matrix import initial_of, one_per_initial_output
from harness.checkloop.feedback import retry_turn_text
from harness.checkloop.online import (
    BASELINE_MODES,
    CheckInput,
    Decision,
    RetryBudget,
    _matching,
    recorded_set_aside,
    set_aside_on,
    set_aside_setting,
)
from harness.score import ScoringRules
from harness.terms import row_from_json

EMPTY_VERSION = "empty-retry/1"
EMPTY_RETRY_ID = "empty-retry-v1"
EMPTY_RETRY_TEXT = "Your final query returned no rows."
MODE = "empty"
TRIGGER = "the rule-22 checked query is an executed SELECT whose result has no rows"


def message_sha256() -> str:
    return hashlib.sha256(EMPTY_RETRY_TEXT.encode("utf-8")).hexdigest()


def trigger(tool_calls: list[dict], final_rows: list | None, rules: ScoringRules, upto: int) -> tuple[
        dict | None, str, dict[str, Any]]:
    """(checked call, rule-22 status, verdict) for the submission at log index ``upto``.

    The verdict's ``status``: "empty" (the only one that retries), "rows", "ask" (a
    boolean result), "error" (the store refused the checked query; ``execution`` holds
    the error kind) or "no_query"."""
    calls = executed_queries(tool_calls, upto)
    answer = [r if isinstance(r, tuple) else row_from_json(r) for r in (final_rows or [])]
    call, status = checked_call(calls, answer, rules)
    if call is None:
        return None, status, {"status": "no_query", "retry": False, "rows": None, "execution": None}
    if call.get("error"):
        return call, status, {"status": "error", "retry": False, "rows": None,
                              "execution": (call["error"] or {}).get("kind") or "error"}
    rows = received_rows(call)
    if rows is None:
        return call, status, {"status": "ask", "retry": False, "rows": None, "execution": "executed"}
    total = (call.get("result_meta") or {}).get("total_rows")
    n = total if isinstance(total, int) else len(rows)
    return call, status, {"status": "empty" if n == 0 else "rows", "retry": n == 0, "rows": n,
                          "execution": "executed"}


class EmptyRetryChecker:
    """The retry-on-empty cell's checker, in the interface the agent loop and
    ``branch_attempt`` call (``budget``, ``settings()``, ``check()``)."""

    mode = MODE
    label = MODE  # its name among an output's branch arms (it has no ontology arm)

    def __init__(self, *, budget: RetryBudget, set_aside: bool | None = None):
        self.budget = budget
        self.set_aside = set_aside_on(set_aside)  # entry 126: the retry turn's closing sentence, as in every arm

    def settings(self) -> dict[str, Any]:
        return {
            "version": EMPTY_VERSION,
            "mode": MODE,
            "trigger": TRIGGER,
            "message": {"id": EMPTY_RETRY_ID, "sha256": message_sha256()},
            "retry_budget": asdict(self.budget),
            **set_aside_setting(self.set_aside),
        }

    def check(self, inp: CheckInput) -> Decision:
        call, status, verdict = trigger(list(inp.tool_calls), inp.final_rows, inp.rules, inp.submission_index)
        r = inp.rules
        initial = {
            "matching": {"multiset": r.multiset, "rel_tol": r.rel_tol, "abs_tol": r.abs_tol,
                         "column_order": r.column_order},
            "checked_query": call["arguments"]["query"] if call is not None else None,
            "checked_query_status": status,
            "verdict": verdict,
            "reports": [],  # no checks run: the keys every checker decision carries (RecordedChecker)
            "store_log": None,
        }
        retry = verdict["retry"]
        return Decision(retry, BASELINE_MODES[MODE] if retry else None, EMPTY_RETRY_TEXT if retry else None,
                        json.loads(json.dumps(initial, ensure_ascii=False)),
                        retry_turn_text(EMPTY_RETRY_TEXT, self.set_aside) if retry else None)


def replay_empty_decision(record: RunRecord | dict) -> Decision:
    """Recompute a retry-on-empty decision from the record's own tool-call log, with no
    store and no model. Raises ReplayDivergence unless the checked query, its status,
    the verdict, the message and the retry decision are the recorded ones."""
    rec = record if isinstance(record, RunRecord) else RunRecord.from_json(record)
    chk = rec.checker
    if chk is None or chk["settings"].get("mode") != MODE:
        raise ValueError("the record is not a retry-on-empty cell's record")
    initial = chk["initial"]
    if initial["submission_index"] is None:
        raise ValueError("the record has no checked initial submission")
    settings = chk["settings"]
    if settings["message"] != {"id": EMPTY_RETRY_ID, "sha256": message_sha256()}:
        raise ReplayDivergence("the retry-on-empty message differs from the recorded one")
    checker = EmptyRetryChecker(budget=RetryBudget(**settings["retry_budget"]), set_aside=recorded_set_aside(settings))
    decision = checker.check(CheckInput(rec.tool_calls, initial["final_rows"], _matching(rec),
                                        rec.store.get("row_cap"), initial["submission_index"]))
    for key in ("checked_query", "checked_query_status", "verdict"):
        if decision.initial[key] != initial[key]:
            raise ReplayDivergence(f"the recomputed retry-on-empty decision differs at {key}")
    retry = chk["retry"]
    if decision.retry != (retry is not None) or (retry and decision.message != retry["message"]):
        raise ReplayDivergence("the recomputed retry decision or message differs from the recorded one")
    return decision


# -- the wider reading, offline (design ruling 2, change-log entry 59) ----------------------

CATEGORIES = ("empty", "rows", "ask_false", "ask_true", "refused", "timeout", "no_query")
# The registered trigger, and the widest execution-only reading: any submitted output whose checked
# query gave the agent no usable rows. Each wider category is also read on its own beside the narrow one.
READINGS: dict[str, frozenset[str]] = {
    "narrow": frozenset({"empty"}),
    "wide": frozenset({"empty", "refused", "timeout", "ask_false", "no_query"}),
    **{f"narrow+{c}": frozenset({"empty", c}) for c in ("refused", "timeout", "ask_false", "no_query")},
}


def trigger_category(record: dict) -> str | None:
    """The trigger category of a record's initial output (CATEGORIES), recomputed from its
    tool-call log at the initial submission; None when the attempt submitted nothing. A
    store error is "timeout" when the store timed out, else "refused"."""
    outcome, final_rows, score = initial_of(record)
    if outcome["status"] != "submitted":
        return None
    chk = record.get("checker")
    if chk:
        upto, rules = chk["initial"]["submission_index"], ScoringRules(**chk["initial"]["matching"])
    else:
        upto, rules = submissions(record["tool_calls"])[-1], rules_of(score)
    call, _status, v = trigger(record["tool_calls"], final_rows, rules, upto)
    if v["status"] == "error":
        return "timeout" if v["execution"] == "timeout" else "refused"
    if v["status"] == "ask":
        return "ask_true" if (call.get("result_meta") or {}).get("boolean") else "ask_false"
    return v["status"]


def _ratio(num: int, den: int) -> float | str:
    return num / den if den else NOT_ESTIMABLE


def empty_sensitivity(records: list[dict], readings: dict[str, frozenset[str]] = READINGS) -> dict[str, Any]:
    """Detection under each reading over the initial outputs of ``records`` (each counted once,
    a branch with its parent): per category, how many were right and wrong by the scorer;
    per reading, the flagged right (C_F) and wrong (W_F) outputs, flag precision, false-flag
    rate and detection recall. An output with no scorer label is left out and counted."""
    cats = {c: {"right": 0, "wrong": 0} for c in CATEGORIES}
    labelled: list[tuple[str, bool]] = []
    unscored = 0
    for rec in one_per_initial_output(records):
        category = trigger_category(rec)
        if category is None:
            continue
        exact = (initial_of(rec)[2] or {}).get("exact")
        if exact is None:
            unscored += 1
            continue
        cats[category]["right" if exact else "wrong"] += 1
        labelled.append((category, bool(exact)))
    out_readings = {}
    for name, flagged in readings.items():
        c_f = sum(1 for c, ok in labelled if ok and c in flagged)
        w_f = sum(1 for c, ok in labelled if not ok and c in flagged)
        n_c = sum(1 for _, ok in labelled if ok)
        n_w = len(labelled) - n_c
        out_readings[name] = {"categories": sorted(flagged), "C_F": c_f, "W_F": w_f, "C": n_c, "W": n_w,
                              "flag_precision": _ratio(w_f, w_f + c_f), "false_flag_rate": _ratio(c_f, n_c),
                              "detection_recall": _ratio(w_f, n_w)}
    return {"outputs": len(labelled), "unscored": unscored, "categories": cats, "readings": out_readings}
