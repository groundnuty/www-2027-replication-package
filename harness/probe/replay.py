"""Offline replay of a probe record: no model, no store.

``replay_probe`` rebuilds the record from what it stores: it checks the request is
still the probe's (the pinned system prompt of the current code, the question as
the only user turn, no tools, the model and output cap), re-parses the stored
response with the route's parser, re-reads the answer or re-extracts the query,
serves probe (b)'s recorded store result (``RecordedProbeSparql``: the same query
text, the recorded error or the stored raw result, the recorded store retries),
and re-scores against the gold rows the record carries. With ``check`` it raises
``ReplayDivergence`` unless the rebuilt transcript, token counts, outcome, answer,
query entry (its ``elapsed_s`` wall time aside), final rows and score equal the
stored ones, and the recorded store call was used exactly when one was made.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from harness.agent.errors import InternalToolError
from harness.agent.infra import StoreUnavailable
from harness.agent.replay import ReplayDivergence, first_difference
from harness.probe.prompts import get_probe
from harness.probe.record import ProbeRecord
from harness.probe.run import ProbeConfig, Reply, build_record, request_problem
from harness.score.rules import RULES_VERSION_V4, RULES_VERSION_V5
from harness.sparql import QueryResult, SparqlError
from harness.terms import rows_from_sparql_json

DERIVED = ("transcript", "tokens", "outcome", "answer", "query", "final_rows", "score")


def _as_record(record: ProbeRecord | dict) -> ProbeRecord:
    return record if isinstance(record, ProbeRecord) else ProbeRecord.from_json(record)


class RecordedProbeSparql:
    """Serves probe (b)'s one recorded store call."""

    replays_infra_retries = True

    def __init__(self, record: ProbeRecord):
        store = record.store or {}
        self._settings = {"endpoint": store.get("endpoint_label"), "timeout_s": store.get("sparql_timeout_s"),
                          "row_cap": store.get("row_cap")}
        self._entry = record.query or {}
        self._expected = bool(self._entry.get("client_called"))
        self.calls_served = 0
        self.last_retries: list[dict] = []
        self.last_infra_s = 0.0
        self.last_raw_truncated = False

    def settings(self) -> dict[str, Any]:
        return dict(self._settings)

    @property
    def exhausted(self) -> bool:
        return self.calls_served == (1 if self._expected else 0)

    def query(self, query: str) -> QueryResult:
        entry = self._entry
        if not self._expected or self.calls_served:
            raise ReplayDivergence("the replay sent a query the record does not hold")
        if query != entry.get("text"):
            raise ReplayDivergence("the query differs from the recorded one")
        self.calls_served += 1
        self.last_retries = list(entry.get("infra_retries") or [])
        self.last_infra_s = float(entry.get("infra_s") or 0.0)
        self.last_raw_truncated = bool(entry.get("result_raw_truncated"))
        error = entry.get("error")
        if error:
            if error["kind"] == "infrastructure":
                raise StoreUnavailable(error["message"], self.last_retries, self.last_infra_s,
                                       suspected_query=bool(error.get("suspected_query")))
            if error["kind"] == "internal":
                raise InternalToolError(error["message"], client_called=True)
            raise SparqlError(error["message"], error["kind"], error.get("status"))
        meta, raw = entry["result_meta"], entry.get("result_raw") or {}
        if meta["form"] == "ASK":  # the store's answer is the raw boolean; the meta must say the same
            value = raw.get("boolean")
            if not isinstance(value, bool) or meta.get("boolean") is not value:
                raise ReplayDivergence("the stored ASK result's raw boolean and its meta disagree")
            return QueryResult("ASK", [], [], value, 0, False, meta["elapsed_s"], raw)
        problem = _select_meta_problem(meta, raw, self._settings["row_cap"])
        if problem:
            raise ReplayDivergence(f"the stored SELECT result's meta disagrees with its raw result: {problem}")
        _, rows = rows_from_sparql_json(raw)
        if meta["truncated"]:
            rows = rows[: self._settings["row_cap"]]
        return QueryResult(meta["form"], list(meta["columns"]), rows, None, meta["total_rows"], meta["truncated"],
                           meta["elapsed_s"], raw)


def _select_meta_problem(meta: dict, raw: dict, row_cap: int | None) -> str | None:
    """Why a stored SELECT result's meta does not follow from its raw result (None when it does)."""
    head, results = raw.get("head") or {}, raw.get("results") or {}
    bindings = results.get("bindings") if isinstance(results, dict) else None
    if not isinstance(bindings, list) or list(meta.get("columns") or []) != list(head.get("vars") or []):
        return "the columns differ from the raw result's variables"
    truncated = row_cap is not None and meta.get("total_rows", 0) > row_cap
    if bool(meta.get("truncated")) != truncated:
        return f"truncated is {meta.get('truncated')} for {meta.get('total_rows')} rows and row cap {row_cap}"
    if not truncated and meta.get("total_rows") != len(bindings):
        return f"total_rows is {meta.get('total_rows')} for {len(bindings)} stored rows"
    if truncated and len(bindings) < row_cap:  # the client keeps a truncated result's first row_cap rows
        return f"truncated, but the {len(bindings)} stored rows do not fill the row cap {row_cap}"
    return None


def _json(obj: Any) -> Any:
    return json.loads(json.dumps(obj, ensure_ascii=False))


def _without_elapsed(entry: dict | None) -> dict | None:
    return None if entry is None else {k: v for k, v in entry.items() if k != "elapsed_s"}


def replay_probe(record: ProbeRecord | dict, *, check: bool = True) -> ProbeRecord:
    """Rebuild the record offline; the new record has a new ``run_id`` and ``replay_of`` set."""
    rec = _as_record(record)
    spec = get_probe(rec.prompt.get("id") or rec.probe)  # a retired prompt id replays under its own text
    if spec.probe != rec.probe or rec.prompt.get("sha256") != spec.system_sha256:
        raise ReplayDivergence(f"the record's prompt is not {spec.id} of the current code (sha256 differs)")
    config = ProbeConfig(**rec.config)
    problem = request_problem(rec.request, spec, rec.question_text, rec.model, config.max_tokens)
    if problem:
        raise ReplayDivergence(f"the recorded request is not the probe's: {problem}")
    sparql = RecordedProbeSparql(rec) if spec.probe == "query" else None
    question = {"question_id": rec.question_id, "base_question_id": rec.base_question_id, "template": rec.template,
                "tier": rec.tier, "split": rec.split, "graph": rec.graph, "text": rec.question_text}
    store = None
    if rec.store is not None:
        store = {k: v for k, v in rec.store.items() if k not in ("sparql_timeout_s", "row_cap")}
    # a record scored under an earlier rules version (scorer-rules-v4 or v5) replays under it: its gold carries the
    # question's unversioned rules, which now read as the current version (change-log entries 120 and 127), so the
    # replay scores with the rules the stored score names
    stored_rules = (rec.score or {}).get("rules")
    scoring_gold = rec.gold
    if isinstance(stored_rules, dict) and stored_rules.get("version") in (RULES_VERSION_V4, RULES_VERSION_V5) and \
            isinstance(rec.gold, dict):
        gold_rules = rec.gold.get("rules") or {}
        changed = [k for k in ("multiset", "rel_tol", "abs_tol", "column_order") if gold_rules.get(k) != stored_rules.get(k)]
        if changed:  # the stored score must have been made under the record's own gold rules (review of 4b27c1f)
            raise ReplayDivergence(f"the stored score's rules differ from the record's gold rules at {', '.join(changed)}")
        scoring_gold = {**rec.gold, "rules": stored_rules}
    new = build_record(spec=spec, question=question, reply=Reply(rec.request, rec.call, rec.response,
                                                                  rec.provider_failure),
                       route_info=rec.route, gold=scoring_gold, config=config, repetition=rec.repetition, store=store,
                       sparql=sparql, replay_of=rec.run_id)
    new = replace(new, gold=rec.gold)
    if check:
        if (new.store or None) != (rec.store or None):
            raise ReplayDivergence("the replayed store description differs from the stored one")
        for name in DERIVED:
            replayed, recorded = getattr(new, name), getattr(rec, name)
            if name == "query":
                replayed, recorded = _without_elapsed(replayed), _without_elapsed(recorded)
            where = first_difference(_json(recorded), _json(replayed), name)
            if where is not None:
                raise ReplayDivergence(f"the replayed probe record differs from the stored one at {where}")
        if sparql is not None and not sparql.exhausted:
            raise ReplayDivergence("the replay did not use the recorded store call")
    return new


def reparse_answer(record: ProbeRecord | dict, answer_format: str) -> ProbeRecord:
    """Probe (a)'s stored reply read again under another answer format (parse.py) and re-scored against
    the gold rows the record carries: no model, no store. The copy has a new ``run_id``, ``replay_of``
    set and the format in its config; it replays under that format."""
    rec = _as_record(record)
    if rec.probe != "answer":
        raise ValueError(f"only probe (a) has an answer format; this record is probe {rec.probe!r}")
    spec = get_probe(rec.prompt.get("id") or rec.probe)
    config = replace(ProbeConfig(**rec.config), answer_format=answer_format)
    question = {"question_id": rec.question_id, "base_question_id": rec.base_question_id, "template": rec.template,
                "tier": rec.tier, "split": rec.split, "graph": rec.graph, "text": rec.question_text}
    return build_record(spec=spec, question=question, reply=Reply(rec.request, rec.call, rec.response,
                                                                   rec.provider_failure),
                        route_info=rec.route, gold=rec.gold, config=config, repetition=rec.repetition,
                        replay_of=rec.run_id, runner=rec.runner)


def rescore_probe(record: ProbeRecord | dict, gold_rows: Any, rules: Any) -> ProbeRecord:
    """A copy of the record re-scored against other gold rows or rules (not a replay)."""
    from harness.probe.run import _score

    rec = _as_record(record)
    gold = {"rows": gold_rows, "rules": rules}
    return replace(rec, gold=gold, score=_score(rec.outcome, rec.final_rows, gold))
