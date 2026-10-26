"""One closed-book probe try: a question, a route, one provider call -> a probe record.

A try has two halves, kept apart so that the store is used one query at a time
and a store outage never costs a second model call:

1. ``call_sync`` (or the Batch API path, batch.py) makes the one provider call and
   returns a ``Reply``: the request as sent, the provider's whole response as JSON
   (or the failure that ended the call), and how the call went. The runner
   persists it before going on.
2. ``build_record`` turns a reply into the record. It parses the response with the
   route's own parser (providers.py), so a reply read back from disk, from a batch
   result, or from a record (replay.py) goes through exactly the same code; then:

   * the provider failed: ``failed:<failure type>`` as in the agent loop (an outage
     or overload past the route's retries is ``failed:infrastructure`` from the
     provider: never scored, re-run); a context overflow is ``capped:context``, kept
     although one short call cannot overflow; a refusal (stop reason ``refusal`` or
     ``content_filter``) is ``failed:other`` with the agent loop's message;
   * probe (a): the answer is parsed (parse.py, under the config's answer format): ``answered`` and scored, or
     ``unparseable`` (scored wrong, counted separately); the answer entry's ``declined``
     names the reporting stratum of rule ``declined-v1`` (parse.py), which changes neither
     the label nor the score;
   * probe (b): the query is extracted (parse.py) and executed through the agent's
     SPARQL tool code (tools.run_sparql: SERVICE refused before sending, the raw
     result capped as the agent's is), on the graph's own endpoint with the agent
     runs' row cap and time-out, wrapped in the store-outage retries and probing of
     infra.py. Its rows (cut to the row cap, as the agent would see them) are scored:
     ``answered``. A query that did not run cleanly is ``query_error:<class>``, scored
     wrong, with the class one of ``no_query`` (nothing extracted), ``not_a_query``
     (the client refuses it as not one read-only SELECT or ASK query, before sending;
     prose around a query lands here), ``parse_error`` (the store could not parse it),
     ``refused`` (SERVICE), ``store_error`` (any other error the store returned for
     the query) or ``timeout``. A store outage (infra.py) is ``failed:infrastructure``
     from the store: never scored; the runner waits for the store and executes the
     same reply's query again (the reply does not depend on the store).

Outcome labels: ``answered``, ``unparseable``, ``query_error:<class>``,
``failed:<failure type>``, ``capped:context``. Every scored outcome other than
``answered`` is scored as "failed" (wrong) with its label in the score's notes.
``failed:infrastructure`` and ``failed:harness`` (a fault in our own code while executing
or reading probe (b)'s query) are never scored.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Callable

from harness.agent.errors import HarnessAbort, InternalToolError
from harness.agent.infra import DEFAULT_LIMIT_S as DEFAULT_INFRA_LIMIT_S
from harness.agent.infra import StoreUnavailable
from harness.agent.loop import (
    _assistant_turn,
    _term_rows,
    _turn_tokens,
    require_frozen_scorer,
    scoring_rules,
    token_totals,
)
from harness.agent.providers import (
    CONTEXT_STOPS,
    OUTPUT_LIMIT_STOPS,
    StepFailure,
    StepResult,
    _to_dict,
    classify_stream_exception,
    parse_anthropic_response,
    parse_openai_response,
)
from harness.agent.record import harness_commit, utc_now
from harness.agent.tools import RAW_ROW_CAP, run_sparql
from harness.probe.parse import (
    ANSWER_FORMAT,
    DECLINE_RULE,
    QUERY_EXTRACTION,
    declined,
    extract_query,
    parse_answer,
)
from harness.probe.prompts import ProbeSpec
from harness.probe.record import ProbeRecord
from harness.terms import XSD, row_to_json, rows_from_sparql_json

ERROR_MESSAGE_LIMIT = 2000
QUERY_ERROR_CLASSES = ("no_query", "not_a_query", "parse_error", "refused", "store_error", "timeout")
# What the reported "parse" rate of probe (b) counts: the query could not be read as one query.
PARSE_CLASSES = ("no_query", "not_a_query", "parse_error")
_ERROR_KIND_CLASS = {"federation_refused": "refused", "not_read_only": "not_a_query", "parse": "parse_error",
                     "timeout": "timeout"}


@dataclass(frozen=True)
class ProbeConfig:
    max_tokens: int = 16000  # the agent runs' output cap per call
    iri_rule: str = "registered_schemes_v3"  # submit_report's rule, for probe (a)'s cells
    answer_format: str = ANSWER_FORMAT  # probe (a)'s reading (parse.py); a record replays under its own
    query_extraction: str = QUERY_EXTRACTION
    decline_rule: str = DECLINE_RULE  # probe (a)'s "declined" reporting stratum (parse.py)
    raw_row_cap: int = RAW_ROW_CAP  # SPARQL JSON rows kept in the record, as for the agent's queries
    infra_retry_s: float = DEFAULT_INFRA_LIMIT_S  # per query, store-outage retries (infra.py)


def probe_question(q: dict) -> dict:
    """The fields of a question-file row a probe record keeps."""
    return {"question_id": q["id"], "base_question_id": q.get("base_question_id"), "template": q.get("template"),
            "tier": q.get("tier"), "split": q.get("split"), "graph": q.get("graph"), "text": q["question"]}


# --- the request ---------------------------------------------------------------------------


def build_request(route: Any, spec: ProbeSpec, question_text: str, max_tokens: int) -> dict:
    """The keyword arguments of the probe's one call: the route's own request with no tools."""
    return route.request(spec.system, [{"role": "user", "kind": "question", "text": question_text}], (), max_tokens)


def _system_text(request: dict) -> str | None:
    system = request.get("system")
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        return "".join(b.get("text", "") for b in system if isinstance(b, dict))
    messages = request.get("messages") or []
    if messages and messages[0].get("role") == "system":
        return messages[0].get("content")
    return None


def request_problem(request: dict, spec: ProbeSpec, question_text: str, model: str, max_tokens: int) -> str | None:
    """Why the request is not the probe's (None when it is): model, output cap, no tools, the
    probe's system prompt and the question as the only user turn."""
    if request.get("model") != model:
        return f"the request names model {request.get('model')!r}, not {model!r}"
    if request.get("max_tokens") != max_tokens:
        return f"the request's max_tokens is {request.get('max_tokens')!r}, not {max_tokens}"
    if "tools" in request or "tool_choice" in request:
        return "the request carries tools"
    if _system_text(request) != spec.system:
        return f"the request's system prompt is not {spec.id}'s"
    users = [m for m in request.get("messages") or [] if m.get("role") != "system"]
    if users != [{"role": "user", "content": question_text}]:
        return "the request's messages are not the question alone"
    return None


# --- the call --------------------------------------------------------------------------------


@dataclass
class Reply:
    """What one provider call left, before any parsing: persisted by the runner between the
    call and the record, and rebuilt from a record by replay."""

    request: dict
    call: dict
    response: dict | None = None
    provider_failure: dict | None = None

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "Reply":
        return cls(**data)


def failure_json(failure: StepFailure) -> dict:
    return {"failure_type": failure.failure_type, "message": failure.message, "details": failure.details or None}


def call_sync(route: Any, request: dict, *, clock: Callable[[], float] = time.monotonic) -> Reply:
    """One synchronous call through the route's client and retries (providers.py).

    A route that streams (an OpenAI-compatible route with ``streaming``: the Gateway models, design ruling on
    design log) sends the request as built (``stream: true`` is in it) through the route's own transport,
    which reads the stream to its end inside the same retries and returns the reply a non-streamed request
    returns; the stored response, its parse and the record are then the same as without streaming. Every
    other call, and the Batch API path (batch.py, Anthropic only), is made as before."""
    started_at, t0 = utc_now(), clock()
    call: dict[str, Any] = {"mode": "sync", "batch_id": None, "custom_id": None, "started_at": started_at}
    try:
        client = route._ensure_client()
        if route.provider != "anthropic" and getattr(route, "streaming", False):
            # the stream reader returns the assembled reply and its time to the first byte, which a probe record
            # does not keep (closed-book-probe/1 has no first_byte_s)
            (resp, _first_byte), latency, retries, errors, wall = route._call(
                lambda timeout: route._read_stream(client, request, timeout), None, classify=classify_stream_exception)
        else:
            create = client.messages.create if route.provider == "anthropic" else client.chat.completions.create
            resp, latency, retries, errors, wall = route._call(lambda timeout: create(**request, timeout=timeout),
                                                               None)
    except StepFailure as failure:  # ProviderError included
        call.update({"latency_s": 0.0, "wall_s": clock() - t0, "retries": failure.details.get("retries") or 0,
                     "retry_errors": [], "ended_at": utc_now()})
        return Reply(request, call, None, failure_json(failure))
    except HarnessAbort:
        raise
    except Exception as exc:  # an unexpected client fault: recorded, not raised (as the agent loop does)
        call.update({"latency_s": 0.0, "wall_s": clock() - t0, "retries": 0, "retry_errors": [], "ended_at": utc_now()})
        return Reply(request, call, None, {"failure_type": "other", "message": f"{type(exc).__name__}: {exc}",
                                           "details": None})
    call.update({"latency_s": latency, "wall_s": wall, "retries": retries, "retry_errors": errors,
                 "ended_at": utc_now()})
    return Reply(request, call, _to_dict(resp), None)


def step_from_reply(reply: Reply, provider: str) -> StepResult | None:
    """The route's parse of the stored response (None when the call failed)."""
    if reply.response is None:
        return None
    call = reply.call
    kw = {"latency_s": call.get("latency_s") or 0.0, "wall_s": call.get("wall_s") or 0.0,
          "retries": call.get("retries") or 0, "retry_errors": list(call.get("retry_errors") or [])}
    if provider == "anthropic":
        from anthropic.types import Message

        step = parse_anthropic_response(Message.model_validate(reply.response), **kw)
    else:
        from openai.types.chat import ChatCompletion

        step = parse_openai_response(ChatCompletion.model_validate(reply.response), **kw)
    step.max_tokens = reply.request.get("max_tokens")
    return step


# --- the record --------------------------------------------------------------------------------


def _outcome(status: str, *, failure_type: str | None = None, cap: str | None = None,
             query_error: str | None = None, error_message: str | None = None,
             provider_error: dict | None = None, stage: str | None = None) -> dict:
    if status == "failed":
        label = f"failed:{failure_type}"
    elif status == "capped":
        label = f"capped:{cap}"
    elif status == "query_error":
        label = f"query_error:{query_error}"
    else:
        label = status
    return {"status": status, "failure_type": failure_type, "cap": cap, "query_error": query_error,
            "error_message": error_message[:ERROR_MESSAGE_LIMIT] if error_message else error_message,
            "provider_error": provider_error, "stage": stage, "label": label,
            "infra_source": stage if failure_type == "infrastructure" else None}


def _provider_outcome(failure: dict) -> dict:
    kind, message, details = failure["failure_type"], failure.get("message") or "", failure.get("details")
    if kind == "context_overflow":
        return _outcome("capped", cap="context", error_message=message, provider_error=details, stage="provider")
    return _outcome("failed", failure_type=kind, error_message=message, provider_error=details, stage="provider")


def rows_of(meta: dict, raw: dict, row_cap: int | None) -> list[tuple]:
    """A query's rows as the agent's tool would hold them: from the stored raw result, cut to the
    row cap; an ASK result is one row holding the boolean the store answered (the raw result's)."""
    from harness.terms import Literal

    if meta["form"] == "ASK":
        value = (raw or {}).get("boolean")
        if not isinstance(value, bool):
            raise ValueError("the stored ASK result holds no boolean")
        return [(Literal("true" if value else "false", XSD + "boolean"),)]
    _, rows = rows_from_sparql_json(raw or {"head": {"vars": []}, "results": {"bindings": []}})
    return rows[:row_cap] if meta["truncated"] and row_cap is not None else rows


def _execute(text: str, source: str, sparql: Any, raw_row_cap: int) -> tuple[dict, list | None, dict | None]:
    """Probe (b)'s query -> (query entry, final rows or None, outcome or None when answered).

    The raw result is kept up to the larger of ``raw_row_cap`` and the client's row cap, as the agent
    loop keeps it, so the scored rows are exactly the rows the agent's tool would expose. An unexpected
    exception in our code while executing the query or reading its result (anything but a SparqlError,
    which is the query's, or a store outage) is a harness failure: ``failed:harness``, never scored,
    set aside and run again once fixed (as for agent runs); the record keeps its message, so it replays
    as the same failure whatever the code does later."""
    from harness.agent.engine_flags import engine_flags

    entry: dict[str, Any] = {"text": text, "source": source, "engine_flags": engine_flags(text) if text else [],
                             "status": "not_sent", "error_class": None, "error": None, "client_called": False,
                             "result_raw": None, "result_raw_truncated": False, "result_meta": None,
                             "elapsed_s": 0.0, "infra_retries": [], "infra_s": 0.0}
    if not text:
        entry.update({"error_class": "no_query", "error": {"kind": "no_query", "message": "no query in the reply"}})
        return entry, None, _outcome("query_error", query_error="no_query", error_message="no query in the reply",
                                     stage="parse")
    row_cap = sparql.settings().get("row_cap")
    raw_cap = max(raw_row_cap, row_cap or 0)
    res = rows = None
    try:
        res = run_sparql({"query": text}, sparql, raw_cap)
        if res.error is None:
            rows = [row_to_json(r) for r in rows_of(res.result_meta, res.result_raw, row_cap)]
    except HarnessAbort:
        raise
    except StoreUnavailable as down:
        entry.update({"status": "infrastructure", "client_called": True, "infra_retries": down.retries,
                      "infra_s": down.infra_s, "error": {"kind": "infrastructure", "message": down.message,
                                                         "suspected_query": down.suspected_query}})
        return entry, None, _outcome("failed", failure_type="infrastructure", error_message=down.message,
                                     stage="store")
    except Exception as exc:  # a fault in our harness or client, neither the query's nor the model's
        message = exc.message if isinstance(exc, InternalToolError) else f"{type(exc).__name__}: {exc}"
        called = res is not None or bool(getattr(exc, "client_called", False))
        entry.update({"status": "crashed", "client_called": called, "error": {"kind": "internal", "message": message}})
        return entry, None, _outcome("failed", failure_type="harness", error_message=message, stage="harness")
    entry.update({"client_called": res.client_called, "result_raw": res.result_raw,
                  "result_raw_truncated": res.result_raw_truncated, "result_meta": res.result_meta,
                  "elapsed_s": res.elapsed_s, "infra_retries": res.infra_retries, "infra_s": res.infra_s})
    if res.error is not None:
        cls = _ERROR_KIND_CLASS.get(res.error["kind"], "store_error")
        entry.update({"status": "refused" if res.refused else "error", "error_class": cls, "error": res.error})
        return entry, None, _outcome("query_error", query_error=cls, error_message=res.error["message"],
                                     stage="store")
    entry["status"] = "executed"
    return entry, rows, None


UNSCORED_FAILURES = ("infrastructure", "harness")  # never scored: re-run (infrastructure) or set aside (harness)


def _score(outcome: dict, final_rows: list | None, gold: dict) -> dict | None:
    from harness.score import score_attempt

    if outcome["failure_type"] in UNSCORED_FAILURES:
        return None
    require_frozen_scorer()  # every scoring path, the probes' live, replay and rescoring included
    gold_rows = _term_rows(gold["rows"])
    rules = scoring_rules(gold["rules"])
    if outcome["status"] == "answered":
        return score_attempt("submitted", _term_rows(final_rows), gold_rows, rules).to_json()
    score = score_attempt("failed", None, gold_rows, rules).to_json()
    if outcome["status"] != "failed":  # capped, unparseable, query_error: kept distinguishable
        score["notes"] = [*score["notes"], outcome["label"]]
    return score


def derive(spec: ProbeSpec, question_text: str, reply: Reply, provider: str, model: str, gold: dict,
           config: ProbeConfig, sparql: Any) -> dict:
    """Everything in a record that follows from the reply (and, for probe (b), the store): the
    transcript, token counts, outcome, answer or query entry, final rows and score. The live run
    and replay both call this."""
    step = step_from_reply(reply, provider)
    transcript = [{"role": "user", "kind": "question", "text": question_text}]
    per_turn: list[dict] = []
    answer = query = final_rows = None
    if step is not None:
        transcript.append(_assistant_turn(step, 1))
        turn = _turn_tokens(step, 1, 0.0, model)
        del turn["first_byte_s"]  # agent-run/16's per-turn field; a closed-book-probe/1 record has none
        del turn["system_fingerprint"]  # agent-run/21's, likewise
        per_turn.append(turn)
    if reply.provider_failure is not None:
        outcome = _provider_outcome(reply.provider_failure)
    elif step.refusal is not None:
        outcome = _outcome("failed", failure_type="other", stage="model", error_message=(
            f"the provider declined to answer (stop_reason={step.stop_reason}, "
            f"category={step.refusal.get('category')}, explanation={step.refusal.get('explanation')})"))
    elif step.stop_reason in CONTEXT_STOPS:
        message = f"the prompt no longer fits the context window (stop_reason={step.stop_reason})"
        outcome = _outcome("capped", cap="context", stage="model", error_message=message)
    elif spec.probe == "answer":
        rows, form, problems = parse_answer(step.text, config.iri_rule, config.answer_format)
        answer = {"form": form, "n_rows": None if rows is None else len(rows), "problems": problems,
                  "declined": declined(step.text, rows, step.stop_reason in OUTPUT_LIMIT_STOPS)}
        if rows is None:
            outcome = _outcome("unparseable", stage="parse", error_message="; ".join(problems))
        else:
            final_rows, outcome = rows, _outcome("answered")
    else:
        text, source = extract_query(step.text)
        query, final_rows, failed = _execute(text, source, sparql, config.raw_row_cap)
        outcome = failed or _outcome("answered")
    outcome["truncated"] = bool(step is not None and step.stop_reason in OUTPUT_LIMIT_STOPS)
    outcome["served_models"] = sorted({t["served_model"] for t in per_turn if t["served_model"]})
    outcome["model_mismatch_turns"] = sum(t["model_mismatch"] for t in per_turn)
    return {"transcript": transcript, "tokens": {"totals": token_totals(per_turn), "per_turn": per_turn},
            "outcome": outcome, "answer": answer, "query": query, "final_rows": final_rows,
            "score": _score(outcome, final_rows, gold)}


def build_record(*, spec: ProbeSpec, question: dict, reply: Reply, route_info: dict, gold: dict,
                 config: ProbeConfig, repetition: int, store: dict | None = None, sparql: Any = None,
                 replay_of: str | None = None, runner: dict | None = None,
                 clock: Callable[[], float] = time.monotonic) -> ProbeRecord:
    """The record of one try from its reply. ``store`` is the sanitised store description (probe (b);
    the row cap and time-out are added from ``sparql.settings()``)."""
    if spec.probe == "query" and sparql is None:
        raise ValueError("probe (b) needs a SPARQL client")
    t0 = clock()
    model = route_info["model"]
    store_info = None
    if spec.probe == "query":
        settings = sparql.settings()
        store_info = {**(store or {}), "sparql_timeout_s": settings.get("timeout_s"),
                      "row_cap": settings.get("row_cap")}
    parts = derive(spec, question["text"], reply, route_info.get("provider"), model, gold, config, sparql)
    ended_at = utc_now()
    started_at = reply.call.get("started_at") or ended_at
    return ProbeRecord(
        run_id=str(uuid.uuid4()), probe=spec.probe, question_id=question["question_id"],
        base_question_id=question.get("base_question_id"), template=question.get("template"),
        tier=question.get("tier"), split=question.get("split"), graph=question.get("graph"), repetition=repetition,
        model=model, route=route_info, call=reply.call, prompt=spec.record(), config=asdict(config),
        store=store_info, harness=harness_commit(), started_at=started_at, ended_at=ended_at,
        wall_s=float(reply.call.get("wall_s") or 0.0) + (clock() - t0), question_text=question["text"],
        request=reply.request, response=reply.response, provider_failure=reply.provider_failure,
        transcript=parts["transcript"], tokens=parts["tokens"], outcome=parts["outcome"], answer=parts["answer"],
        query=parts["query"], final_rows=parts["final_rows"], gold=gold, score=parts["score"],
        replay_of=replay_of, runner=runner)


def gold_of(q: dict) -> dict:
    """The gold rows and scoring rules a record carries (from a question-file row)."""
    return {"rows": q["gold_rows"], "rules": q["scoring_rules"]}

