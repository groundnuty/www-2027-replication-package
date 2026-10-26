#!/usr/bin/env python3
"""Pilot summary over one or more pilot output directories (design log, AC7).

Per model: accuracy (exact, F1) with base-question bootstrap intervals,
N_att/N_out, failure types, cap and truncation rates, infrastructure failures,
runs per hour, tokens and estimated cost per run; spend over every paid try
(records/, infrastructure/, aborted/ and harness set-asides, each try once), which stays
out of the accuracy denominator; every model that has any record, even one
whose attempts all failed infrastructure; and, from the check compiler's
query analysis, two figures the study lead asked for:
  - the checked query of each output under contract v2 rule 22 (the last
    query whose rows contain the submitted rows, "aligned", else the last one,
    "fallback"), with shares per model;
  - constructs outside the v2 fragment in that query, per construct (rule 19;
    rdfs:subClassOf+ and * are in the fragment).
Load and latency (the design log), per model and per arm (the record's
``load.arm``, else its prompt condition): median and p90 per-turn model latency
(the request that succeeded) and call time (the request with its retries and
backoff, which the runner's guard rails read), retries, the model calls over
300 s and the provider retries (a call that failed in the end included, with its
time and retries; design log: a streamed call may take up to 600 s), the agent outcomes the
guard rails do not act on (turn caps, wall caps, context caps, truncated
replies, provider refusals, and submissions refused for an IRI that is not
absolute; design log), a flag on each pair of arms whose capped:context or
provider-refusal rates differ by more than one percentage point,
infrastructure failures by source (store or provider; records and every failed
try), the distribution of attempts in flight at each attempt's start, the
runner's concurrency events, and a flag when the arms' latency or call-time
medians differ by more than 25%. Store time-outs (design log), per model and per
arm: SPARQL calls that came back as a time-out, by how the store reported it
(its own status, a 200 cut by the cap, or no answer before the HTTP backstop,
twice), the requests that joined a failed identical computation twice (QLever),
and the store's queries in flight at those attempts where its client could read
them (not on QLever).

A directory is summarised only when its replay check passed, still matches its
inputs (records, set-aside decisions, concurrency events) and was made under the
committed code in use (harness/agent/replay_check.py); otherwise the summary is
refused: run scripts/verify_batch.py on it first, with the code committed.

    uv run python scripts/pilot_summary.py OUT_DIR [OUT_DIR ...] --out pilot.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from harness.agent.cost import estimate_record_cost
from harness.agent.engine_flags import engine_flags
from harness.agent.replay_check import BatchNotVerified, require_current_check
from harness.agent.tools import NOT_ABSOLUTE, NOT_ABSOLUTE_DATATYPE, UNKNOWN_SCHEME, UNKNOWN_SCHEME_DATATYPE
from harness.checkloop.checked import (
    answer_in_result,  # rule 22's matcher, shared with the checker loop
)
from harness.checks.query import analyse
from harness.pilot_report import render_markdown, summarize, summarize_by_model
from harness.score import ScoringRules
from harness.terms import row_from_json, rows_from_sparql_json

SETASIDE_SUFFIX = ".setaside.json"
EVENTS_FILE = "concurrency-events.jsonl"
ARM_SPREAD_LIMIT = 0.25  # latency medians across arms may differ by this share of the lowest
# A streamed agent call may take up to 600 s (the route's request time-out; design ruling in the design log): the
# summary counts the calls above half of that.
LONG_CALL_S = 300.0
ARM_RATE_GAP = 0.01  # outcome rates of two arms may differ by this much (one percentage point)
# Outcome strata compared between arms (design log): name -> the outcomes key of its count.
ARM_RATE_STRATA = {"capped_context": "capped_context", "provider_refusal": "provider_refusals"}


def load(dirs: list[Path]) -> list[dict]:
    records = []
    for d in dirs:
        for p in sorted((d / "records").glob("*.json")):
            records.append(json.loads(p.read_text()))
    return records


def load_paid_tries(dirs: list[Path]) -> list[dict]:
    """Every recorded try that was paid for, once each: records/, infrastructure/ tries
    (the final try of an attempt that failed on every try is in both), aborted/ attempts
    (a configuration error can come after tokens were spent) and set-aside records."""
    seen, tries = set(), []
    for d in dirs:
        for sub in ("records", "infrastructure", "aborted", "harness"):
            for p in sorted((d / sub).glob("*.json")):
                if p.name.endswith(SETASIDE_SUFFIX):
                    continue
                rec = json.loads(p.read_text())
                if rec.get("run_id") in seen:
                    continue
                seen.add(rec.get("run_id"))
                tries.append(rec)
    return tries


def load_set_asides(dirs: list[Path]) -> list[dict]:
    """Set-aside decisions: the sidecars, and records set aside by the earlier relabel script
    (which edited the record and added ``harness_failure``)."""
    decisions = []
    for d in dirs:
        harness = d / "harness"
        for p in sorted(harness.glob(f"*{SETASIDE_SUFFIX}")):
            decisions.append(json.loads(p.read_text()))
        sidecar_runs = {x.get("run_id") for x in decisions}
        for p in sorted(harness.glob("*.json")):
            if p.name.endswith(SETASIDE_SUFFIX):
                continue
            rec = json.loads(p.read_text())
            if "harness_failure" in rec and rec.get("run_id") not in sidecar_runs:
                decisions.append({"run_id": rec["run_id"], "model": rec["model"], "label": "failed:harness",
                                  **rec["harness_failure"]})
    return decisions


def executed_queries(record: dict) -> list[dict]:
    """Executed sparql_query calls before the terminal submit_report, in order."""
    calls = []
    for call in record.get("tool_calls", []):
        if call.get("name") == "submit_report" and call.get("terminal"):
            break
        if call.get("name") == "sparql_query" and call.get("status") == "executed":
            calls.append(call)
    return calls


def checked_call(record: dict) -> tuple[dict | None, str]:
    """Contract v2 rule 22: the last query whose result rows contain every
    submitted row ("aligned"), else the last query executed ("fallback")."""
    calls = executed_queries(record)
    if not calls:
        return None, "none"
    submitted = [row_from_json(x) for x in record.get("final_rows") or []]
    rules = record_rules(record)
    for call in reversed(calls):
        raw = call.get("result_raw")
        if not raw or "results" not in raw or call.get("result_raw_truncated"):
            continue  # no rows to align with (ASK, or truncated)
        _, rows = rows_from_sparql_json(raw)
        if answer_in_result(submitted, rows, rules):
            return call, "aligned"
    return calls[-1], "fallback"


def record_rules(record: dict) -> ScoringRules:
    """The rules the record was scored under, or the current defaults."""
    try:
        return ScoringRules.from_json((record.get("score") or {}).get("rules") or {})
    except (ValueError, KeyError, TypeError):
        return ScoringRules()


SUBCLASS = "http://www.w3.org/2000/01/rdf-schema#subClassOf"


def v2_unsupported(query: str) -> list[str]:
    """Constructs outside the v2 fragment. Rule 19 v2 admits rdfs:subClassOf+
    and * (read as membership, F28a), so an unbounded path counts only when it
    is over another predicate."""
    qa = analyse(query)
    if qa.error:
        return ["parser_mismatch"]  # QLever ran it, rdflib cannot parse it: a tooling limit, its own cause
    out = [c for c in qa.unsupported if c != "unbounded_path"]
    if "unbounded_path" in qa.unsupported and _other_unbounded_paths(query):
        out.append("unbounded_path")
    return sorted(out)


def _other_unbounded_paths(query: str) -> bool:
    from rdflib.paths import MulPath

    from harness.sparql_grammar import PARSER_LOCK, parse_query, translate_query

    found = []

    def walk(node):
        if isinstance(node, MulPath):
            if not (str(node.path) == SUBCLASS and node.mod in ("+", "*")):
                found.append(node)
            walk(node.path)
        elif isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, (list, tuple, set)):
            for v in node:
                walk(v)
        elif hasattr(node, "args"):  # rdflib Path containers (sequence, alternative)
            for v in node.args:
                walk(v)

    with PARSER_LOCK:  # rdflib's grammar is not thread-safe (harness/sparql_grammar.py)
        walk(translate_query(parse_query(query)).algebra)
    return bool(found)


def query_figures(records: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r["outcome"]["status"] == "submitted":
            by_model[r["model"]].append(r)
    for model, recs in sorted(by_model.items()):
        constructs, alignment, engine = Counter(), Counter(), defaultdict(list)
        for r in recs:
            call, how = checked_call(r)
            alignment[how] += 1
            if call is not None:
                query = call["arguments"].get("query", "")
                constructs.update(set(v2_unsupported(query)))
                # recomputed from the query text, so records older than agent-run/5 are covered
                for flag in call.get("engine_flags") or engine_flags(query):
                    engine[flag].append(bool((r.get("score") or {}).get("exact")))
        n = len(recs)
        out[model] = {
            "outputs": n,
            "checked_query": {k: {"n": v, "share": v / n} for k, v in sorted(alignment.items())},
            "unsupported_by_construct": {c: {"n": k, "share": k / n} for c, k in constructs.most_common()},
            # engine-affected strata (design log): outputs whose checked query has a shape the engine gets wrong
            "engine_affected": {f: {"n": len(v), "share": len(v) / n, "exact": sum(v) / len(v)}
                                for f, v in sorted(engine.items())},
        }
    return out


def _percentile(ordered: list, q: float):
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]  # nearest rank, as pilot_report


def _turns(records: list[dict]) -> list[dict]:
    return [t for r in records for t in (r.get("tokens") or {}).get("per_turn", [])]


def _median_p90(values: list[float]) -> tuple[float | None, float | None]:
    ordered = sorted(values)
    return (statistics.median(ordered), _percentile(ordered, 0.9)) if ordered else (None, None)


def _latency(records: list[dict]) -> dict:
    turns = _turns(records)
    latency = [t["latency_s"] for t in turns if isinstance(t.get("latency_s"), (int, float))]
    # the call time as the guard rails read it: wall_s, or the request time where the call was not timed
    calls = [t.get("wall_s") or t["latency_s"] for t in turns if isinstance(t.get("wall_s") or t.get("latency_s"),
                                                                           (int, float))]
    (median, p90), (call_median, call_p90) = _median_p90(latency), _median_p90(calls)
    retries = sum(t.get("retries") or 0 for t in turns)
    # a model call that failed in the end is in the outcome, not in a turn: its retries, and its time (from the
    # call's start, the outcome's final_check_s, to the attempt's end, which the failure ended at once)
    failed = [r for r in records if r["outcome"].get("stage") == "provider"]
    failed_call_retries = sum((r["outcome"].get("provider_error") or {}).get("retries") or 0 for r in failed)
    failed_call_s = [r["wall_s"] - r["outcome"]["final_check_s"] for r in failed
                     if isinstance(r.get("wall_s"), (int, float))
                     and isinstance(r["outcome"].get("final_check_s"), (int, float))]
    return {"attempts": len(records), "turns": len(latency), "median_s": median, "p90_s": p90,
            "call_median_s": call_median, "call_p90_s": call_p90,
            "retries": retries,
            "turns_with_retries": sum(1 for t in turns if t.get("retries")),
            "calls_over_300_s": sum(1 for c in [*calls, *failed_call_s] if c > LONG_CALL_S),
            "provider_retries": retries + failed_call_retries}


def _iri_refusals(record: dict) -> int:
    """submit_report calls refused for an IRI (a cell's, or a literal's datatype) that is not absolute."""
    messages = [(c.get("error") or {}).get("message") or "" for c in record.get("tool_calls", [])
                if c.get("name") == "submit_report"]
    marks = (NOT_ABSOLUTE, NOT_ABSOLUTE_DATATYPE, UNKNOWN_SCHEME, UNKNOWN_SCHEME_DATATYPE)  # every rule's wording
    return sum(1 for m in messages if any(mark in m for mark in marks))


def _provider_refusal(record: dict) -> bool:
    """The provider declined to answer: Anthropic stop_reason "refusal", OpenAI-compatible finish_reason
    "content_filter" (the turn's ``refusal``, providers.py)."""
    return any(t.get("refusal") is not None for t in record.get("transcript", []) if t.get("role") == "assistant")


def _outcomes(records: list[dict]) -> dict:
    """The agent outcomes the guard rails count but never act on, over N_att (no infrastructure failures)."""
    recs = [r for r in records if r["outcome"].get("failure_type") != "infrastructure"]
    refusals = [_iri_refusals(r) for r in recs]
    n = len(recs)
    wall = [r for r in recs if r["outcome"].get("cap") == "wall"]
    counts = {"attempts": n, "capped_turns": sum(r["outcome"].get("cap") == "turns" for r in recs),
              "capped_context": sum(r["outcome"].get("cap") == "context" for r in recs),
              "provider_refusals": sum(1 for r in recs if _provider_refusal(r)),
              "capped_wall": len(wall),
              "capped_wall_provider": sum(r["outcome"].get("stage") == "provider" for r in wall),
              "truncated": sum(1 for r in recs if any(t.get("truncated") for t in _turns([r]))),
              "iri_refusals": sum(refusals), "attempts_with_iri_refusal": sum(1 for n in refusals if n)}
    rates = {"capped_turns": "capped_turns_rate", "capped_wall": "capped_wall_rate", "truncated": "truncated_rate",
             "capped_context": "capped_context_rate", "provider_refusals": "provider_refusal_rate"}
    return {**counts, **{rate: counts[k] / n if n else None for k, rate in rates.items()}}


def _arm_rate_flags(by_arm: dict[str, dict]) -> dict[str, list[dict]]:
    """Per stratum, each pair of arms whose rates differ by more than one percentage point."""
    flags: dict[str, list[dict]] = {}
    arms = sorted(a for a, o in by_arm.items() if o["attempts"])
    for stratum, key in ARM_RATE_STRATA.items():
        rate_key = "provider_refusal_rate" if stratum == "provider_refusal" else f"{key}_rate"
        flags[stratum] = [{"arms": [a, b], "rates": [by_arm[a][rate_key], by_arm[b][rate_key]],
                           "difference": abs(by_arm[a][rate_key] - by_arm[b][rate_key])}
                          for i, a in enumerate(arms) for b in arms[i + 1:]
                          if abs(by_arm[a][rate_key] - by_arm[b][rate_key]) > ARM_RATE_GAP + 1e-12]
    return flags


def _infra_source(record: dict) -> str:
    # records before agent-run/7 hold no infra_source; the stage says where the failure was
    return record["outcome"].get("infra_source") or record["outcome"].get("stage") or "store"


def _spread(medians: list[float]) -> float | None:
    medians = [m for m in medians if m]
    return max(medians) / min(medians) - 1 if len(medians) > 1 else None


def _distribution(values: list[int]) -> dict:
    if not values:
        return {"n": 0, "median": None, "p90": None, "max": None, "histogram": {}}
    ordered = sorted(values)
    return {"n": len(ordered), "median": statistics.median(ordered), "p90": _percentile(ordered, 0.9),
            "max": ordered[-1], "histogram": {str(k): v for k, v in sorted(Counter(ordered).items())}}


def _arm(record: dict) -> str:
    return (record.get("load") or {}).get("arm") or record.get("condition") or "none"


def load_events(dirs: list[Path]) -> list[dict]:
    events = []
    for d in dirs:
        path = d / EVENTS_FILE
        if path.is_file():
            events.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return events


# How a time-out reached the harness (harness/sparql.py, harness/agent/infra.py), from its message.
_TIMEOUT_SOURCES = (("partial_200", "partial result"),  # HTTP 200 cut by the cap (QLever's export, Virtuoso)
                    ("no_answer", "did not finish in time"),  # the HTTP backstop, again on the re-send
                    ("no_answer", "query timed out after"),
                    ("http_status", "http "))  # the store's own time-out status (QLever 429, a 504)
# QLever's "Waited for a result from another thread which then failed", again on the re-send: an http
# error of unknown cause (infra.py), counted beside the time-outs.
_JOINED_FAILED = "waited for a result from another thread which then failed"
QLEVER_IN_FLIGHT = ("not readable: QLever lists its active queries only to its admin (?cmd=dump-active-queries "
                    "answers 403 without the access token, checked on 9bebc60)")
NO_IN_FLIGHT = "not readable: the store client does not report the store's in-flight queries"


def _timeout_source(message: str) -> str:
    lowered = message.lower()
    return next((source for source, mark in _TIMEOUT_SOURCES if mark in lowered), "other")


def _timeouts(records: list[dict]) -> dict:
    """Store time-outs over N_att: SPARQL calls that reached the store client and came back as a
    time-out, by how the store reported it."""
    recs = [r for r in records if r["outcome"].get("failure_type") != "infrastructure"]
    calls = [c for r in recs for c in r.get("tool_calls", [])
             if c.get("name") == "sparql_query" and c.get("client_called")]
    sources = Counter(_timeout_source((c.get("error") or {}).get("message") or "") for c in calls
                      if (c.get("error") or {}).get("kind") == "timeout")
    n = sum(sources.values())
    return {"attempts": len(recs), "sparql_calls": len(calls), "timeouts": n,
            "attempts_with_timeout": sum(1 for r in recs if any((c.get("error") or {}).get("kind") == "timeout"
                                                                for c in r.get("tool_calls", []))),
            "timeout_rate": n / len(calls) if calls else None, "by_source": dict(sorted(sources.items())),
            "joined_failed": sum(1 for c in calls
                                 if _JOINED_FAILED in ((c.get("error") or {}).get("message") or "").lower())}


def _server_in_flight(records: list[dict]) -> dict:
    """The store's in-flight count at the start and end of each attempt that had a time-out, where the
    store client read it (``load.store_connections_*``, agent-run/13); otherwise why it is missing."""
    hit = [r for r in records if any((c.get("error") or {}).get("kind") == "timeout" for c in r.get("tool_calls", []))]
    loads = [r["load"] for r in hit if isinstance(r.get("load"), dict)
             and isinstance(r["load"].get("store_connections_start"), int)]
    if loads:
        return {"readable": True, "start": _distribution([x["store_connections_start"] for x in loads]),
                "end": _distribution([x["store_connections_end"] for x in loads
                                      if isinstance(x.get("store_connections_end"), int)])}
    qlever = bool(hit) and all((r.get("store") or {}).get("engine") == "QLever" for r in hit)
    return {"readable": False, "note": QLEVER_IN_FLIGHT if qlever else NO_IN_FLIGHT}


def load_infra_tries(dirs: list[Path]) -> list[dict]:
    return [json.loads(p.read_text()) for d in dirs for p in sorted((d / "infrastructure").glob("*.json"))]


def load_figures(records: list[dict], events: list[dict], infra_tries: list[dict] | None = None) -> dict[str, dict]:
    """Per model: per-turn latency and call time overall and per arm, the agent outcomes, the
    infrastructure failures by source, attempts in flight at each attempt's start (records from
    agent-run/6 on), and the runner's concurrency events."""
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_model[r["model"]].append(r)
    tries_by_model: dict[str, list[dict]] = defaultdict(list)
    for t in infra_tries or []:
        tries_by_model[t["model"]].append(t)
    events_by_model: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        events_by_model[e.get("route_model") or e.get("model")].append(e)
    out: dict[str, dict] = {}
    for model in sorted(set(by_model) | set(events_by_model)):
        recs = by_model.get(model, [])
        arms: dict[str, list[dict]] = defaultdict(list)
        for r in recs:
            arms[_arm(r)].append(r)
        latency = {**_latency(recs), "by_arm": {arm: _latency(rs) for arm, rs in sorted(arms.items())}}
        spread = _spread([a["median_s"] for a in latency["by_arm"].values()])
        call_spread = _spread([a["call_median_s"] for a in latency["by_arm"].values()])
        loads = [r["load"] for r in recs if isinstance(r.get("load"), dict)]
        infra_records = Counter(_infra_source(r) for r in recs if r["outcome"].get("failure_type") == "infrastructure")
        infra_tries = Counter(_infra_source(t) for t in tries_by_model.get(model, []))
        out[model] = {
            "latency": latency,
            "arm_latency_spread": spread,
            "arm_call_spread": call_spread,
            "arm_latency_flag": any(x is not None and x > ARM_SPREAD_LIMIT for x in (spread, call_spread)),
            "outcomes": {**_outcomes(recs), "by_arm": {arm: _outcomes(rs) for arm, rs in sorted(arms.items())}},
            "arm_rate_flags": _arm_rate_flags({arm: _outcomes(rs) for arm, rs in arms.items()}),
            "infrastructure": {"records": dict(sorted(infra_records.items())),
                               "tries": dict(sorted(infra_tries.items()))},
            "in_flight_model_start": _distribution([x["in_flight_model_start"] for x in loads
                                                    if isinstance(x.get("in_flight_model_start"), int)]),
            "in_flight_total_start": _distribution([x["in_flight_total_start"] for x in loads
                                                    if isinstance(x.get("in_flight_total_start"), int)]),
            "concurrency_events": events_by_model.get(model, []),
            "timeouts": {**_timeouts(recs), "by_arm": {arm: _timeouts(rs) for arm, rs in sorted(arms.items())},
                         "server_in_flight": _server_in_flight(recs)},
        }
    return out


def _num(v) -> str:
    return "n/a" if v is None else f"{v:.3g}"


def _pct(v) -> str:
    return "n/a" if v is None else f"{100 * v:.1f}%"


def load_markdown(figures: dict[str, dict]) -> list[str]:
    md = ["## Load and latency (the design log)", ""]
    for model, f in figures.items():
        lat = f["latency"]
        arms = "; ".join(f"**{arm}** median {_num(a['median_s'])} s, p90 {_num(a['p90_s'])} s, call time median "
                         f"{_num(a['call_median_s'])} s ({a['attempts']} attempts, {a['turns']} turns, "
                         f"{a['retries']} retries, {a['calls_over_300_s']} calls over 300 s, "
                         f"{a['provider_retries']} provider retries)" for arm, a in lat["by_arm"].items())
        spreads = [f"{what} {100 * v:.0f}%" for what, v in (("latency", f["arm_latency_spread"]),
                                                             ("call time", f["arm_call_spread"])) if v is not None]
        flag = (f"; **flag**: per-turn medians across arms differ by more than 25% ({', '.join(spreads)})"
                if f["arm_latency_flag"] else "")
        o = f["outcomes"]
        outcomes = "; ".join(f"**{arm}** turn caps {a['capped_turns']}, wall caps {a['capped_wall']} "
                             f"({a['capped_wall_provider']} in a provider retry), capped:context "
                             f"{a['capped_context']} ({_pct(a['capped_context_rate'])}), provider refusals "
                             f"{a['provider_refusals']} ({_pct(a['provider_refusal_rate'])}), truncated {a['truncated']}, "
                             f"IRIs refused as not absolute {a['iri_refusals']} (in {a['attempts_with_iri_refusal']} "
                             f"attempts), of {a['attempts']}" for arm, a in o["by_arm"].items())
        infra = f["infrastructure"]
        names = {"capped_context": "capped:context", "provider_refusal": "provider-refusal"}
        rate_flags = "".join(f"; **flag**: {names[stratum]} rates of arms {x['arms'][0]} and {x['arms'][1]} differ by "
                             f"{100 * x['difference']:.1f} points ({_pct(x['rates'][0])} vs {_pct(x['rates'][1])})"
                             for stratum, found in f["arm_rate_flags"].items() for x in found)
        in_flight = f["in_flight_model_start"]
        load = ("not recorded (records before agent-run/6)" if not in_flight["n"] else
                f"median {_num(in_flight['median'])}, p90 {in_flight['p90']}, max {in_flight['max']}, "
                f"histogram {in_flight['histogram']}; in total across models: max "
                f"{f['in_flight_total_start']['max']}")
        events = "; ".join(f"{e['kind']} {e.get('from')} -> {e.get('to')} at {e.get('at')}"
                           + (f" [{', '.join(e['triggers'])}]" if e.get("triggers") else "")
                           + (f" ({e['reason']})" if e.get("reason") else "")
                           for e in f["concurrency_events"] if e.get("kind") == "step_down") or "no step-downs"
        t = f["timeouts"]
        by_source = ", ".join(f"{k} {v}" for k, v in t["by_source"].items()) or "none"
        t_arms = "; ".join(f"**{arm}** {a['timeouts']} of {a['sparql_calls']} ({_pct(a['timeout_rate'])})"
                           for arm, a in t["by_arm"].items())
        inflight = t["server_in_flight"]
        inflight_text = (f"start median {_num(inflight['start']['median'])}, max {inflight['start']['max']}; end max "
                         f"{inflight['end']['max']}" if inflight["readable"] else inflight["note"])
        timeouts = (f"store time-outs {t['timeouts']} of {t['sparql_calls']} SPARQL calls ({_pct(t['timeout_rate'])}; "
                    f"{by_source}) in {t['attempts_with_timeout']} of {t['attempts']} attempts, by arm: "
                    f"{t_arms or 'none'}; {t['joined_failed']} joined failed computations shown to the model; "
                    f"store's queries in flight at time-outs: {inflight_text}")
        md.append(f"- **{model}**: per-turn latency median {_num(lat['median_s'])} s, p90 {_num(lat['p90_s'])} s; "
                  f"call time (retries included) median {_num(lat['call_median_s'])} s, p90 "
                  f"{_num(lat['call_p90_s'])} s; {lat['retries']} retries over {lat['turns']} turns of "
                  f"{lat['attempts']} attempts; {lat['calls_over_300_s']} model calls over 300 s, "
                  f"{lat['provider_retries']} provider retries (those of failed calls included); by arm: "
                  f"{arms}{flag}; agent outcomes (never a step-down "
                  f"trigger), by arm: {outcomes or 'none'}{rate_flags}; infrastructure failures by source: records "
                  f"{infra['records'] or 'none'}, failed tries {infra['tries'] or 'none'}; attempts in flight for "
                  f"the model at each start: {load}; concurrency: {events}; {timeouts}.")
    return md


def report(dirs: list[Path], *, require_replay_check: bool = True) -> dict:
    """The summary over the pilot directories: ``summary`` per model, ``checked_queries``,
    ``load`` (load and latency per model) and the ``markdown`` of all three.

    Refused (BatchNotVerified) for a directory without a passing, current replay check;
    ``require_replay_check=False`` is for tests on hand-made records that cannot replay."""
    if require_replay_check:
        for d in dirs:
            require_current_check(d)
    records = load(dirs)
    summary, md, figures = _build(dirs, records)
    loads = load_figures(records, load_events(dirs), load_infra_tries(dirs))
    return {"summary": summary, "checked_queries": figures, "load": loads,
            "markdown": md + "\n" + "\n".join(load_markdown(loads)) + "\n"}


def build(dirs: list[Path], *, require_replay_check: bool = True) -> tuple[dict, str, dict]:
    """(summary per model, markdown, checked-query figures); see ``report``."""
    rep = report(dirs, require_replay_check=require_replay_check)
    return rep["summary"], rep["markdown"], rep["checked_queries"]


def _build(dirs: list[Path], records: list[dict]) -> tuple[dict, str, dict]:
    tries = load_paid_tries(dirs)
    set_asides = load_set_asides(dirs)
    infra = [r for r in records if r["outcome"].get("failure_type") == "infrastructure"]
    scored = [r for r in records if r["outcome"].get("failure_type") != "infrastructure"]
    cost_fn = lambda route, tokens: None  # noqa: E731  (cost per record below)
    by_model = summarize_by_model(scored, cost_fn)
    models = sorted({r["model"] for r in records} | {t["model"] for t in tries} | {a["model"] for a in set_asides})
    summary = {model: by_model.get(model) or summarize([], cost_fn) for model in models}

    def usd(recs: list[dict]) -> list[float]:
        return [c for c in (estimate_record_cost(r).get("usd") for r in recs) if c is not None]

    for model, s in summary.items():
        recs = [r for r in scored if r["model"] == model]
        paid = [t for t in tries if t["model"] == model]
        record_costs, paid_costs = usd(recs), usd(paid)
        s["cost_usd_per_run_mean"] = sum(record_costs) / len(record_costs) if record_costs else None
        s["cost_usd_records"] = sum(record_costs) if record_costs else None
        # spend: every paid try once (outages and set-asides included), outside the accuracy denominator
        s["cost_usd_total"] = sum(paid_costs) if paid_costs else None
        s["paid_tries"] = len(paid)
        n_infra = sum(1 for r in infra if r["model"] == model)
        s["infra_failures"] = n_infra
        s["infra_failure_rate"] = n_infra / (len(recs) + n_infra) if recs or n_infra else None
        mine = [a for a in set_asides if a["model"] == model]
        # a set-aside is a harness defect, or an outage past every re-run set aside to re-run it (a grid's path)
        s["failed_harness"] = sum(1 for a in mine if a.get("label", "failed:harness") != "failed:infrastructure")
        outages = sum(1 for a in mine if a.get("label") == "failed:infrastructure")
        if outages:  # only when there is one, so summaries made before the grid's set-aside path reproduce
            s["infra_set_aside"] = outages
    figures = query_figures(scored)
    md = ["# Pilot summary", "", render_markdown(summary), "",
          "## Per model: cost, infrastructure, checked queries", ""]
    for model, s in summary.items():
        f = figures.get(model, {})
        md.append(
            f"- **{model}**: cost/run (est.) {s['cost_usd_per_run_mean']}, total spend over {s['paid_tries']} paid "
            f"tries {s['cost_usd_total']} (scored records {s['cost_usd_records']}); "
            f"infrastructure failures {s['infra_failures']} (rate {s['infra_failure_rate']}); "
            f"failed:harness set aside and re-run {s['failed_harness']}; "
            + (f"failed:infrastructure set aside and re-run {s['infra_set_aside']}; " if "infra_set_aside" in s else "")
            + f"checked query (v2 rule 22) {f.get('checked_query')}; "
            f"constructs outside the v2 fragment, per construct (rule 19) {f.get('unsupported_by_construct')}; "
            f"engine-affected strata {f.get('engine_affected') or 'none'}."
        )
    return summary, "\n".join(md) + "\n", figures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    try:
        rep = report(args.dirs)
    except BatchNotVerified as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
    md = rep["markdown"]
    args.out.write_text(md)
    args.out.with_suffix(".json").write_text(json.dumps({"summary": rep["summary"],
                                                         "checked_queries": rep["checked_queries"],
                                                         "load": rep["load"]}, indent=2, default=str))
    print(md, end="")


if __name__ == "__main__":
    main()
