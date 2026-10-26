"""The LLM judge (design log, AC2; proposal section 7.6; change-log entry 51): the agent's
own model reads the question, the agent's query, the result the graph returned for it
and the answer, and accepts or rejects the answer with a reason.

What the judge sees is fixed by one template (``TEMPLATE``, hashed by
``template_sha256``) with two variants:

* ``agent_output``: the question text (the agent's first user turn), the rule-22
  checked query (the query every checker arm reads), its result as the agent's tool
  returned it (the record's ``result_shown``, verbatim: cut to the run's row cap with
  ``total_rows`` and ``truncated``, exactly what the agent saw), and the submitted
  rows, cut to the same row cap in the same form;
* ``bank_item``: the question text, the bank item's query and its result, rendered by
  the agent's own tool code on a fresh execution (``shown_result``); the result is the
  answer.

Items are built only through ``record_judge_item`` and ``bank_judge_item``, which read
the fields above and nothing else: never an id, the template, the split, the score,
the gold query or rows, a mutation's label, operator or fault class, the agent's
system prompt, or any ontology source. ``render`` takes the built item only.

The judge gives its verdict by calling one tool, ``submit_verdict``, with ``verdict``
("accept" or "reject") and ``reason`` (``parse_verdict``): exactly one call, with exactly
those arguments, is a verdict; the reply's text is recorded and never read for one. No
call, several calls, other arguments, or a reply cut by the output limit is unparsed:
counted on its own, never as an accept. (Template v1 asked for a JSON object alone; the
dev pilot found most Sonnet replies reason in prose first, and any text protocol that
lets the judge reason first can read an example of the format as a verdict, which a
tool call cannot be.)

A call runs on the agent's route with the agent's settings (model, thinking, effort,
output cap per turn), except prompt caching: each judge input is unique, so the
conversation-level cache would write every input at the cache-write price and never
read it back; ``judge_route`` turns caching off and the route's settings record it.
Calls are interactive: the harness has no Batch API path yet.

The reject-then-retry cell (``JudgeRetryChecker``; design ruling 1 of change-log entry 59
and its follow-up) is a branch cell like the checker arms: the judge reads the initial
output, a reject gives one retry, and an accept or an unparsed reply gives none. The retry's
message is the judge's reason, verbatim, in a neutral wrapper (``judge-retry-reason-v1``, the
default: design log's AC2); rule 38 governs the wrapper only (no instruction tail, change-log
entry 30). The wrapper alone (``judge-retry-v1``) is the sensitivity variant. Unparsed is its
own status, read like a checker's unknown (``judge_row``). The judge's call (its verdict, reason, usage and cost) is recorded in the
cell's ``checker.initial.judge``; ``replay_judge_decision`` recomputes the decision from the
record with no model.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from string import Template
from typing import Any

from harness.agent.cost import (
    LONG_PROMPT_THRESHOLD,
    estimate_cost,
    long_prompt_price_for,
    price_for,
)
from harness.agent.loop import same_model
from harness.agent.providers import OUTPUT_LIMIT_STOPS, REFUSAL_STOPS
from harness.agent.record import RunRecord
from harness.agent.replay import ReplayDivergence
from harness.agent.tools import (
    MAX_RESULT_CHARS,
    RAW_ROW_CAP,
    ToolSpec,
    compact_json,
    run_sparql,
)
from harness.checkloop.accounting import OutputRow, online_row
from harness.checkloop.checked import (
    checked_call,
    executed_queries,
    rules_of,
    submissions,
)
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
from harness.terms import row_from_json

JUDGE_VERSION = "llm-judge/1"
TEMPLATE_ID = "llm-judge-v3"  # v1: a JSON object alone; v2: a final JSON line; v3: a tool call
KINDS = ("agent_output", "bank_item")

_FORMAT = ("A query result is JSON, as the agent's query tool returned it: the result columns, the rows (each cell "
           "an RDF term object, or null for an unbound value), total_rows, and truncated (true when the rows were "
           "cut to the first $row_cap). An ASK query's result is {\"boolean\": ...}; a query the graph refused "
           "shows its error.")
_VERDICT = ("Give your verdict by calling the submit_verdict tool once, with verdict \"accept\" or \"reject\" and "
            "a reason in one or two sentences.")
_TOOL = {
    "name": "submit_verdict",
    "description": "Submit your verdict on the answer. Call it exactly once.",
    "input_schema": {
        "type": "object",
        "properties": {"verdict": {"type": "string", "enum": ["accept", "reject"]},
                       "reason": {"type": "string", "description": "Why, in one or two sentences."}},
        "required": ["verdict", "reason"],
    },
}
# The query shown is the rule-22 checked query (the one every checker arm reads): the last query whose
# result holds the answer, else the last query run. It is not always the agent's last query, so the
# text says "the agent's query", never "final".
TEMPLATE: dict[str, Any] = {
    "system": {
        "agent_output": (
            "You check answers to questions about an RDF knowledge graph. An agent answered the question by "
            "querying the graph with SPARQL and submitting answer rows. You are shown the question, the agent's "
            "query, the result the graph returned for that query, and the submitted answer. Decide whether the "
            "submitted answer is a correct and complete answer to the question.\n\n" + _FORMAT
            + " The submitted answer is JSON in the same form: its rows (at most the first $row_cap), total_rows "
            "and truncated.\n\n" + _VERDICT),
        "bank_item": (
            "You check answers to questions about an RDF knowledge graph. An agent answered the question by "
            "querying the graph with SPARQL; its answer is the result of its query. You are shown the question, "
            "the agent's query and the result the graph returned for that query. Decide whether that result is a "
            "correct and complete answer to the question.\n\n" + _FORMAT + "\n\n" + _VERDICT),
    },
    "user": {
        "agent_output": ("Question:\n$question\n\nQuery:\n$query\n\nResult of the query:\n$result\n\n"
                         "Submitted answer:\n$answer"),
        "bank_item": "Question:\n$question\n\nQuery:\n$query\n\nResult of the query:\n$result",
    },
    "no_query": "(no query was run)",
    "no_result": "(no result)",
    "tool": _TOOL,
    # How a reply is read (parse_verdict): exactly one call of the tool, its arguments exactly these keys,
    # a verdict spelled exactly as listed and a non-empty reason. Anything else is unparsed, with one of
    # these statuses.
    "reply": {"verdicts": ["accept", "reject"], "keys": ["reason", "verdict"],
              "unparsed": ["length", "refusal", "no_call", "multiple_calls", "bad_call"]},
}


SUBMIT_VERDICT = ToolSpec(**TEMPLATE["tool"])


def template_sha256() -> str:
    """The hash of everything that fixes what the judge is shown and how its reply is read
    (canonical JSON of TEMPLATE: sorted keys, no spaces, UTF-8); recorded with every call."""
    return _sha256(TEMPLATE)


def _sha256(obj: Any) -> str:
    text = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class JudgeItem:
    """Everything the judge is shown, already rendered as the agent saw it."""

    kind: str  # agent_output | bank_item
    question: str
    query: str | None
    result: str | None  # the query's result as the agent's tool returns it
    row_cap: int
    answer: str | None = None  # agent outputs: the submitted rows

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown judge item kind {self.kind!r}; known: {KINDS}")
        if (self.answer is None) != (self.kind == "bank_item"):
            raise ValueError("an agent output is shown with its answer; a bank item's answer is its result")


@dataclass(frozen=True)
class Rendered:
    variant: str
    system: str
    user: str
    input_sha256: str  # canonical JSON of {"system", "user"}: what was sent, nothing else


def render(item: JudgeItem) -> Rendered:
    fill = {"row_cap": str(item.row_cap), "question": item.question,
            "query": item.query if item.query is not None else TEMPLATE["no_query"],
            "result": item.result if item.result is not None else TEMPLATE["no_result"],
            "answer": item.answer or ""}
    system = Template(TEMPLATE["system"][item.kind]).substitute(fill)
    user = Template(TEMPLATE["user"][item.kind]).substitute(fill)
    return Rendered(item.kind, system, user, _sha256({"system": system, "user": user}))


def _capped_rows(rows: list, cap: int) -> str:
    """Submitted rows in the tool result's form, cut to the row cap like a query result."""
    return compact_json({"rows": rows[:cap], "total_rows": len(rows), "truncated": len(rows) > cap})


def question_text(record: dict) -> str:
    """The question as the agent saw it: the record's first user turn."""
    transcript = record["transcript"]
    first = transcript[0] if transcript else {}
    if first.get("role") != "user" or first.get("kind") != "question":
        raise ValueError("the record does not open with the question turn")
    return first["text"]


def output_judge_item(question: str, tool_calls: Sequence[dict], final_rows: Sequence | None, rules: Any,
                      row_cap: int, upto: int) -> tuple[JudgeItem, str]:
    """(the judge's view of the output submitted at log index ``upto``, rule 22's checked-query
    status): the rule-22 checked query among the calls before it, that call's shown result,
    and the submitted rows."""
    answer = [r if isinstance(r, tuple) else row_from_json(r) for r in (final_rows or [])]
    call, status = checked_call(executed_queries(list(tool_calls), upto), answer, rules)
    if call is not None and not isinstance(call.get("result_shown"), str):
        raise ValueError("the checked call has no shown result")
    item = JudgeItem(kind="agent_output", question=question,
                     query=call["arguments"]["query"] if call is not None else None,
                     result=call["result_shown"] if call is not None else None, row_cap=row_cap,
                     answer=_capped_rows(list(final_rows or []), row_cap))
    return item, status


def record_judge_item(record: dict) -> JudgeItem:
    """The judge's view of a run record's initial output (a checker cell's: its initial
    answer). Raises ValueError when there is no submitted output to judge."""
    question = question_text(record)
    subs = submissions(record["tool_calls"])
    if not subs:
        raise ValueError(f"record {record.get('run_id')} has no submitted output to judge")
    chk = record.get("checker")
    final_rows = chk["initial"]["final_rows"] if chk else record["final_rows"]
    try:
        item, _ = output_judge_item(question, record["tool_calls"], final_rows, rules_of(record.get("score")),
                                    record["store"]["row_cap"], subs[0])
    except ValueError as exc:
        raise ValueError(f"record {record.get('run_id')}: {exc}") from None
    return item


def shown_result(client: Any, query: str, *, max_result_chars: int | None = MAX_RESULT_CHARS) -> str:
    """A query's result exactly as the agent's sparql_query tool shows it (row cap, total,
    truncation, the character cap, or the error), from one execution on ``client``. The judge
    sees what the agent would see: the cap is the agent's (the frozen 80,000 by default; pass a
    model's own cap when it has one)."""
    return run_sparql({"query": query}, client, RAW_ROW_CAP, max_result_chars).shown


def bank_judge_item(bank_item: dict, question_row: dict, result: str, *, row_cap: int) -> JudgeItem:
    """The judge's view of a fault-bank item: its question's text, its own query and that
    query's result (``shown_result``). Reads the item's ``query`` and the question row's
    ``question`` only."""
    return JudgeItem(kind="bank_item", question=question_row["question"], query=bank_item["query"], result=result,
                     row_cap=row_cap)


def parse_verdict(tool_calls: list[dict], stop_reason: str | None,
                  refusal: dict | None = None) -> tuple[str | None, str | None, str]:
    """(verdict, reason, status) from the reply's tool calls, read strictly (TEMPLATE["reply"]).
    Status "ok", or why the reply is unparsed: "length" (cut by the output limit), "refusal"
    (a refusal stop, or a refusal the provider reported beside any call), "no_call",
    "multiple_calls" or "bad_call" (another tool, arguments that did not parse, other keys,
    a verdict not spelled as listed, or no reason)."""
    if stop_reason in OUTPUT_LIMIT_STOPS:
        return None, None, "length"
    if stop_reason in REFUSAL_STOPS or refusal:
        return None, None, "refusal"
    if not tool_calls:
        return None, None, "no_call"
    if len(tool_calls) > 1:
        return None, None, "multiple_calls"
    c = tool_calls[0]
    args = c.get("arguments")
    if c.get("name") != TEMPLATE["tool"]["name"] or c.get("error") or not isinstance(args, dict):
        return None, None, "bad_call"
    verdict, reason = args.get("verdict"), args.get("reason")
    if (sorted(args) != TEMPLATE["reply"]["keys"] or verdict not in TEMPLATE["reply"]["verdicts"]
            or not isinstance(reason, str) or not reason.strip()):
        return None, None, "bad_call"
    return verdict, reason, "ok"


def judge_cost(route_info: dict, usage: dict) -> dict[str, Any]:
    """``estimate_cost``, with the estimate withheld (USD None) on a route that charges for
    tokens when a count it charges for was not reported, so unknown spend never reads as $0.
    Cache counts are exempt only on a route with prompt caching off: nothing is cached then.
    A call is one request, so on a model priced by prompt length (Haiku 5.5) it is priced at
    the tier of its own prompt (``request_cost``), not at the long-prompt upper bound."""
    caching = bool((route_info.get("settings") or {}).get("prompt_cache"))
    cost = request_cost(route_info, usage) if not caching else estimate_cost(route_info, usage)
    prices, _ = price_for(route_info)
    exempt = () if caching else ("cache_read_tokens", "cache_write_tokens")
    if prices and any(prices.values()) and any(k not in exempt for k in cost["unreported"]):
        cost = {**cost, "usd": None}
    return cost


def request_cost(route_info: dict, usage: dict) -> dict[str, Any]:
    """The cost of one request with prompt caching off: input and output tokens at the list
    price of the request's own prompt-length tier (cost.py), cache counts read as zero."""
    prices, basis = price_for(route_info)
    long_prices = long_prompt_price_for(route_info)
    if prices is None or long_prices is None:  # one price at any length
        return estimate_cost(route_info, usage)
    unreported = [k for k in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")
                  if usage.get(k) is None]
    prompt = sum(usage.get(k) or 0 for k in ("input_tokens", "cache_read_tokens", "cache_write_tokens"))
    tier = long_prices if prompt > LONG_PROMPT_THRESHOLD else prices
    usd = ((usage.get("input_tokens") or 0) * tier["input"] + (usage.get("output_tokens") or 0) * tier["output"]) / 1e6
    basis = f"{basis}; one request priced at its own prompt length ({prompt} tokens, threshold {LONG_PROMPT_THRESHOLD})"
    return {"usd": round(usd, 6), "is_estimate": True, "basis": basis, "unreported": unreported}


def judge_route(route: Any) -> Any:
    """The agent's route for judge calls: the same model and settings, with prompt caching
    off (see the module docstring); a route without caching is used as it is."""
    if getattr(route, "prompt_cache", False):
        route = copy.copy(route)
        route.prompt_cache = False
    return route


def call_judge(route: Any, item: JudgeItem, *, max_tokens: int | None = None) -> dict[str, Any]:
    """One judge call; returns its record (judge-call/1). A provider failure propagates
    (``ProviderError``): the caller records it."""
    r = render(item)
    step = route.step(r.system, [{"role": "user", "kind": "judge_input", "text": r.user}], (SUBMIT_VERDICT,),
                      max_tokens=max_tokens)
    verdict, reason, status = parse_verdict(step.tool_calls, step.stop_reason, step.refusal)
    info = route.route_info()
    reasoning = step.native.get("reasoning_content") or step.native.get("reasoning")
    thinking = [b for b in (step.native.get("content") or []) if isinstance(b, dict) and b.get("type") == "thinking"]
    return {
        "version": JUDGE_VERSION,
        "template": {"id": TEMPLATE_ID, "variant": r.variant, "sha256": template_sha256()},
        "input_sha256": r.input_sha256,
        "row_cap": item.row_cap,
        "model": route.model,
        "served_model": step.model,
        "same_model": same_model(route.model, step.model),
        "route": info,
        "verdict": verdict,
        "reason": reason,
        "parse_status": status,
        "text": step.text,  # recorded, never read for a verdict
        "tool_calls": step.tool_calls,
        "leaked_tool_call": step.leaked_tool_call,
        "reasoning_present": bool(reasoning) or bool(thinking),
        "stop_reason": step.stop_reason,
        "refusal": step.refusal,
        "usage": step.usage,
        "cost": judge_cost(info, step.usage),
        "max_tokens": step.max_tokens,
        "latency_s": step.latency_s,
        "wall_s": step.wall_s,
        "retries": step.retries,
        "response_id": step.response_id,
    }


# -- the reject-then-retry cell (design ruling 1, change-log entry 59) --------------------

JUDGE_CELL_VERSION = "judge-retry/1"
JUDGE_MODE = "judge"
# message id -> template. The default carries the judge's reason verbatim (design log, AC2; the science
# ruling on the follow-up round); the wrapper alone is the sensitivity variant.
JUDGE_MESSAGES = {"judge-retry-reason-v1": "Your answer was checked and rejected. Reason: {reason}",
                  "judge-retry-v1": "Your answer was checked and rejected."}
JUDGE_RETRY_ID = "judge-retry-reason-v1"
JUDGE_SENSITIVITY_ID = "judge-retry-v1"
JUDGE_RETRY_TEMPLATE = JUDGE_MESSAGES[JUDGE_RETRY_ID]
JUDGE_TRIGGER = "the agent's own model, judging the initial output with llm-judge-v3, rejects it"
# How accounting reads a verdict (judge_row): a reject is a flag, an accept a pass, and an unparsed
# reply an unknown (reported as its own count, never retried).
JUDGE_STATUS = {"reject": "flagged", "accept": "substantive_pass", "unparsed": "unknown"}


def judge_message_sha256(message_id: str = JUDGE_RETRY_ID) -> str:
    return hashlib.sha256(JUDGE_MESSAGES[message_id].encode("utf-8")).hexdigest()


def judge_message(message_id: str, reason: str | None) -> str:
    """The retry message: the template, with the judge's reason where it has a place for one."""
    return JUDGE_MESSAGES[message_id].replace("{reason}", reason or "")


def _verdict(judged: dict) -> dict[str, Any]:
    status = judged["verdict"] or "unparsed"
    return {"status": status, "retry": status == "reject", "reason": judged["reason"],
            "parse_status": judged["parse_status"]}


class JudgeRetryChecker:
    """The judge cell's checker, in the interface the agent loop and ``branch_attempt`` call
    (``budget``, ``settings()``, ``check()``). ``question`` is the question text as the agent
    saw it; ``route`` is the agent's own route (``judge_route`` turns its caching off). A
    provider failure of the judge's call propagates: the attempt is re-run, never counted."""

    mode = JUDGE_MODE
    label = JUDGE_MODE  # its name among an output's branch arms

    def __init__(self, route: Any, question: str, *, budget: RetryBudget, max_tokens: int | None = None,
                 message_id: str = JUDGE_RETRY_ID, set_aside: bool | None = None):
        if message_id not in JUDGE_MESSAGES:
            raise ValueError(f"unknown judge retry message {message_id!r}; known: {sorted(JUDGE_MESSAGES)}")
        self.route = judge_route(route)
        self.question = question
        self.budget = budget
        self.max_tokens = max_tokens
        self.message_id = message_id
        self.set_aside = set_aside_on(set_aside)  # entry 126: the retry turn's closing sentence, as in every arm

    @classmethod
    def for_record(cls, record: RunRecord | dict, route: Any, *, budget: RetryBudget,
                   max_tokens: int | None = None, message_id: str = JUDGE_RETRY_ID) -> JudgeRetryChecker:
        """The cell for a recorded output (to branch from it): the record's question, judged by
        the record's own model."""
        rec = record.to_json() if isinstance(record, RunRecord) else record
        if getattr(route, "model", None) != rec["model"]:
            raise ValueError(f"the judge is the agent's own model: {getattr(route, 'model', None)!r} cannot judge an "
                             f"output of {rec['model']!r}")
        return cls(route, question_text(rec), budget=budget, max_tokens=max_tokens, message_id=message_id)

    def settings(self) -> dict[str, Any]:
        return {
            "version": JUDGE_CELL_VERSION,
            "mode": JUDGE_MODE,
            "trigger": JUDGE_TRIGGER,
            "judge": {"version": JUDGE_VERSION, "template": {"id": TEMPLATE_ID, "sha256": template_sha256()},
                      "model": self.route.model, "max_tokens": self.max_tokens},
            "message": {"id": self.message_id, "sha256": judge_message_sha256(self.message_id)},
            "retry_budget": asdict(self.budget),
            **set_aside_setting(self.set_aside),
        }

    def check(self, inp: CheckInput) -> Decision:
        item, status = output_judge_item(self.question, inp.tool_calls, inp.final_rows, inp.rules, inp.row_cap,
                                         inp.submission_index)
        judged = call_judge(self.route, item, max_tokens=self.max_tokens)
        verdict = _verdict(judged)
        r = inp.rules
        initial = {
            "matching": {"multiset": r.multiset, "rel_tol": r.rel_tol, "abs_tol": r.abs_tol,
                         "column_order": r.column_order},
            "checked_query": item.query,
            "checked_query_status": status,
            "verdict": verdict,
            "reports": [],  # no checks run: the keys every checker decision carries
            "store_log": None,
            "judge": judged,
        }
        retry = verdict["retry"]
        message = judge_message(self.message_id, judged["reason"]) if retry else None
        return Decision(retry, BASELINE_MODES[JUDGE_MODE] if retry else None, message,
                        json.loads(json.dumps(initial, ensure_ascii=False)),
                        retry_turn_text(message, self.set_aside) if retry else None)


def replay_judge_decision(record: RunRecord | dict) -> Decision:
    """Recompute a judge cell's decision from its record, with no model and no store: the
    judge's input is rebuilt from the tool-call log and must hash to the recorded one, the
    recorded reply is read again, and the verdict, the retry decision and the message must
    be the recorded ones (ReplayDivergence otherwise)."""
    rec = record if isinstance(record, RunRecord) else RunRecord.from_json(record)
    chk = rec.checker
    if chk is None or chk["settings"].get("mode") != JUDGE_MODE:
        raise ValueError("the record is not a judge cell's record")
    initial = chk["initial"]
    if initial["submission_index"] is None:
        raise ValueError("the record has no checked initial submission")
    settings = chk["settings"]
    message_id = settings["message"]["id"]
    if message_id not in JUDGE_MESSAGES or settings["message"]["sha256"] != judge_message_sha256(message_id):
        raise ReplayDivergence("the judge cell's message template differs from the recorded one")
    if settings["judge"]["template"] != {"id": TEMPLATE_ID, "sha256": template_sha256()}:
        raise ReplayDivergence("the judge template differs from the recorded one")
    item, status = output_judge_item(question_text(rec.to_json()), rec.tool_calls, initial["final_rows"],
                                     _matching(rec), rec.store.get("row_cap"), initial["submission_index"])
    judged = initial["judge"]
    if render(item).input_sha256 != judged["input_sha256"]:
        raise ReplayDivergence("the judge's input rebuilt from the record differs from the recorded input")
    if (item.query, status) != (initial["checked_query"], initial["checked_query_status"]):
        raise ReplayDivergence("the recomputed checked query differs from the recorded one")
    parsed = parse_verdict(judged["tool_calls"], judged["stop_reason"], judged["refusal"])
    if parsed != (judged["verdict"], judged["reason"], judged["parse_status"]) or initial["verdict"] != _verdict(judged):
        raise ReplayDivergence("the judge's reply, read again, gives another verdict than the recorded one")
    retry = initial["verdict"]["retry"]
    message = judge_message(message_id, judged["reason"]) if retry else None
    rec_retry = chk["retry"]
    if retry != (rec_retry is not None) or (retry and rec_retry["message"] != message):
        raise ReplayDivergence("the recomputed retry decision or message differs from the recorded one")
    return Decision(retry, BASELINE_MODES[JUDGE_MODE] if retry else None, message,
                    {k: initial[k] for k in ("checked_query", "checked_query_status", "verdict", "judge")},
                    retry_turn_text(message, recorded_set_aside(settings)) if retry else None)


def judge_row(rec: dict) -> OutputRow:
    """The accounting row of one judge cell record: its verdict read as a checker status
    (JUDGE_STATUS), the retry as any checker arm's."""
    chk = rec.get("checker")
    if not chk or chk["settings"].get("mode") != JUDGE_MODE:
        raise ValueError(f"record {rec.get('run_id')} is not a judge cell's record")
    v = chk["initial"].get("verdict") or {}
    as_checker = {"status": JUDGE_STATUS.get(v.get("status")), "any_unknown": v.get("status") == "unparsed"}
    return online_row({**rec, "checker": {**chk, "initial": {**chk["initial"], "verdict": as_checker}}})
