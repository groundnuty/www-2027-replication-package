"""Which query an output is checked on (S0.1 contract v2, rule 22), read from a
run record's tool-call log.

The checked query is the last executed query whose result rows contain all the
submitted rows under the scorer's matching ("aligned"); if there is none, it is
the last executed query ("fallback"). The matching is the scorer's own
(``score_rows``, called directly; change-log entry 30), with the result as the
reference. A stored result cut short is read on the rows it kept. An output
with no executed query has no checked query ("none"; reason ``no_query``). This
is the one implementation; the pilot summary and the checker loop both use it.

An output is one submission: the initial one, or the retried one after checker
feedback. The retried output's trajectory is everything the agent executed
before its own submission, the initial phase included.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from typing import Any

from harness.checks.compiler import Output
from harness.score import ScoringRules
from harness.score.scorer import score_rows  # rule 22: the scorer's own matching
from harness.terms import Row, row_from_json, rows_from_sparql_json


def answer_in_result(submitted, result_rows, rules: ScoringRules | None = None) -> bool:
    """Whether one query result contains every submitted row, under the scorer's matching.

    Rule 22 (change-log entry 30): the scorer's own row matching, called directly. The
    result may have more columns than the answer, so each choice of as many result columns
    as the answer has (in their order) is scored as the reference with ``score_rows``; the
    scorer then searches the column orders it permits, reads numeric strings as numbers
    where the reference column is numeric, and applies set or multiset semantics. The
    result contains the answer when every (canonical) submitted row is matched.

    The checker keeps the scorer-rules-v4 matching whatever rules it is given (``ScoringRules.as_v4``): change-log
    entry 120 changed the scoring of agent outputs only, and the checker's semantics stay exactly as calibrated (C2
    and C5 rest on them; the lead's decision in the design log Oct).
    """
    rules = (rules or ScoringRules()).as_v4()
    rows = [tuple(r) for r in submitted]
    results = [tuple(r) for r in result_rows]
    if not rows or not results:
        return False
    arity, width = len(rows[0]), len(results[0])
    if any(len(r) != arity for r in rows) or width < arity:
        return False
    for columns in combinations(range(width), arity):
        reference = [tuple(r[c] for c in columns) for r in results]
        score = score_rows(rows, reference, rules)
        if score.n_answer_rows and score.n_matched == score.n_answer_rows:
            return True
    return False


# -- reading the tool-call log ---------------------------------------------------------


def submissions(tool_calls: Sequence[dict]) -> list[int]:
    """Indices (in the log) of the terminal submit_report calls, in order: the initial
    output, then the retried one when there was a retry."""
    return [i for i, c in enumerate(tool_calls) if c.get("name") == "submit_report" and c.get("terminal") == "submitted"]


def executed_queries(tool_calls: Sequence[dict], upto: int | None = None) -> list[dict]:
    """Executed sparql_query calls before log index ``upto`` (default: the first terminal
    submission), in order."""
    if upto is None:
        subs = submissions(tool_calls)
        upto = subs[0] if subs else len(tool_calls)
    return [c for c in tool_calls[:upto] if c.get("name") == "sparql_query" and c.get("status") == "executed"]


def received_rows(call: dict, row_cap: int | None = None) -> list[Row] | None:
    """The rows the agent received from one executed query, or None when the call holds
    no rows (an ASK result). The client cuts the stored bindings to the row cap already."""
    raw = call.get("result_raw")
    if not isinstance(raw, dict) or "results" not in raw:
        return None
    _, rows = rows_from_sparql_json(raw)
    meta = call.get("result_meta") or {}
    if meta.get("truncated") and row_cap is not None:
        rows = rows[:row_cap]
    return rows


def result_row_count(call: dict) -> int | None:
    """How many rows the store returned for one executed query: the store's total when the call
    records it (the stored rows are a prefix, cut to the row caps), else the stored rows; None when
    the call holds no rows (an ASK, a refused or failed query), never 0."""
    raw = call.get("result_raw")
    if not isinstance(raw, dict) or not isinstance(raw.get("results"), dict):
        return None
    total = (call.get("result_meta") or {}).get("total_rows")
    if isinstance(total, int) and not isinstance(total, bool) and total >= 0:
        return total
    bindings = raw["results"].get("bindings")
    return len(bindings) if isinstance(bindings, list) else None


def checked_call(calls: Sequence[dict], submitted: Sequence[Row], rules: ScoringRules) -> tuple[dict | None, str]:
    """Rule 22 over executed calls: (call, "aligned" | "fallback"), or (None, "none")."""
    if not calls:
        return None, "none"
    for call in reversed(calls):
        raw = call.get("result_raw")
        if not raw or "results" not in raw:
            continue  # no rows to align with (ASK, or a refused query)
        # a stored result cut short is a candidate when its retained rows hold the answer
        _, rows = rows_from_sparql_json(raw)
        if answer_in_result(submitted, rows, rules):
            return call, "aligned"
    return calls[-1], "fallback"


def rules_of(score: dict | None, fallback: Any = None) -> ScoringRules:
    """The scoring rules a record was scored under, else ``fallback``, else the defaults.
    An older rules version is read for its matching settings only."""
    for obj in ((score or {}).get("rules"), fallback):
        if isinstance(obj, ScoringRules):
            return obj
        if isinstance(obj, dict):
            fields = {k: obj[k] for k in ("multiset", "rel_tol", "abs_tol", "column_order") if k in obj}
            try:
                return ScoringRules(**fields)
            except (TypeError, ValueError):
                continue
    return ScoringRules()


def checked_output(tool_calls: Sequence[dict], final_rows: Sequence[Any] | None, rules: ScoringRules, *,
                   upto: int | None = None, row_cap: int | None = None) -> tuple[Output | None, str]:
    """The checker's input for the submission at log index ``upto`` (default: the initial
    one): (Output, status) with status "aligned" or "fallback", or (None, "none") when the
    agent executed no query before submitting."""
    calls = executed_queries(tool_calls, upto)
    answer = [r if isinstance(r, tuple) else row_from_json(r) for r in (final_rows or [])]
    call, status = checked_call(calls, answer, rules)
    if call is None:
        return None, status
    trajectory = [rows for c in calls if (rows := received_rows(c, row_cap)) is not None]
    output = Output(query=call["arguments"]["query"], answer=answer, result_rows=received_rows(call, row_cap),
                    role="agent", trajectory_rows=trajectory, checked_status=status,
                    store_accepted=not call.get("error"))  # a fallback may be a query the store refused
    return output, status


def checked_result_rows(tool_calls: Sequence[dict], final_rows: Sequence[Any] | None, rules: ScoringRules, *,
                        upto: int | None = None) -> int | None:
    """The number of rows rule 22's checked query returned for the submission at log index ``upto``
    (default: the initial one); None when it has no checked query or its result holds no rows."""
    answer = [r if isinstance(r, tuple) else row_from_json(r) for r in (final_rows or [])]
    call, _ = checked_call(executed_queries(tool_calls, upto), answer, rules)
    return None if call is None else result_row_count(call)
