"""How often each per-model harness setting fired, per arm (change-log entry 129, item 4): descriptive, beside the
analysis, never confirmatory.

The runner's table gives some models their own settings (design log; design rulings, change-log entries 110, 112 and
113): Gemma's report-key rule json_string_cells_v1 (how submit_report reads a cell object's strings), Qwen3.6's
reasoning-call rule (a call written into the reasoning channel runs) and each model's caps. A setting is fixed per model
and the same across that model's arms, so its firing rate should not differ by arm; this table shows it does not, with
every numerator beside its denominator and integer counts only.

The report-key rule's firings are decided again from each record: every submit_report call the tool read under the rule
is read twice through the runner's own tools.run_submit (the parser and the number texts exactly as the run read them),
with the record's iri rule, under the strict reading (report_key_rule None) and under the record's rule. The call's JSON
text (``arguments_raw``) is in its turn's ``tool_calls`` (the tool-call log keeps it only for a malformed call); a call
the reasoning-call rule took has none, as in the run. A call is unchanged (both accept the same rows), rescued (only the
rule accepts), reread (both accept, the rows differ), lost (only the strict reading accepts), refused (neither) or the
rule's fault (a bare number with no JSON text: the run crashed there too). A rule reading that does not give the
recorded verdict, or the record's final rows for its final report, is counted (``calls_not_reproduced``), never
silently taken. A record the collector cannot read is counted and listed (``errors``) and adds nothing else; ``records``
counts every record added, so the rule histograms sum to records minus errors.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import Any

COUNTS = ("records", "errors", "submit_calls", "calls_unchanged", "calls_rescued", "calls_reread", "calls_lost",
          "calls_refused", "calls_rule_fault", "calls_not_reproduced", "records_final_report",
          "records_final_report_rescued", "assistant_turns", "turns_recovered_taken", "turns_recovered_not_taken",
          "records_with_recovered_call")
HISTOGRAMS = ("report_key_rule", "reasoning_call_rule", "caps")
ROW = ("records", "errors", "report_key_rule", "submit_calls", "calls_unchanged", "calls_rescued", "calls_reread",
       "calls_lost", "calls_refused", "calls_rule_fault", "calls_not_reproduced", "records_final_report",
       "records_final_report_rescued", "reasoning_call_rule", "assistant_turns", "turns_recovered_taken",
       "turns_recovered_not_taken", "records_with_recovered_call", "caps")
STATUS = ("descriptive (change-log entry 129, item 4): how often each per-model harness setting fired in each arm, to "
          "show the settings do not differ by arm; it is never confirmatory and changes no arm's numbers")
NONE = "none"  # no rule: the strict reading, or a call in the reasoning channel failing leaked_tool_call
IRI_RULE_BEFORE_7 = "any_scheme"  # replay.config_of: the iri rule of a record without one (before agent-run/7)
RULE_READS = ("executed", "crashed")  # log statuses of a submit_report call the tool read (a crash: raised inside it)
RESCUED = ("calls_rescued", "calls_reread")
MAX_ERRORS = 50


class ModelSettings:
    """The counts, added one record at a time (``add``) and read with ``table``."""

    def __init__(self) -> None:
        self.counts: dict[str, Counter] = defaultdict(Counter)
        self.histograms: dict[str, dict[str, Counter]] = defaultdict(_histograms)
        self.errors: list[str] = []

    def add(self, record: Mapping[str, Any], arm: str) -> None:
        counts = self.counts[arm]
        counts["records"] += 1
        try:
            own, histograms = _read(record)
        except Exception as exc:  # noqa: BLE001 - any record the collector cannot read: counted and shown
            counts["errors"] += 1
            self.errors.append(f"{_run_id(record)}: {type(exc).__name__}: {str(exc)[:200]}")
            return
        counts.update(own)
        for name in HISTOGRAMS:
            self.histograms[arm][name].update(histograms[name])

    def table(self) -> dict[str, Any]:
        arms = {arm: _row(self.counts[arm], self.histograms[arm]) for arm in sorted(self.counts)}
        totals_counts: Counter = Counter()
        totals_histograms = _histograms()
        for arm in arms:
            totals_counts.update(self.counts[arm])
            for name in HISTOGRAMS:
                totals_histograms[name].update(self.histograms[arm][name])
        return {"descriptive": True, "status": STATUS, "arms": arms,
                "totals": _row(totals_counts, totals_histograms), "errors": self.errors[:MAX_ERRORS],
                "errors_omitted": max(0, len(self.errors) - MAX_ERRORS)}  # listed: the first MAX_ERRORS


def _histograms() -> dict[str, Counter]:
    from harness.agent.loop import CAPS

    return {"report_key_rule": Counter(), "reasoning_call_rule": Counter(), "caps": Counter(dict.fromkeys(CAPS, 0))}


def _row(counts: Counter, histograms: Mapping[str, Counter]) -> dict[str, Any]:
    return {f: dict(sorted(histograms[f].items())) if f in HISTOGRAMS else counts[f] for f in ROW}


def _run_id(record: Any) -> Any:
    try:
        return record.get("run_id")
    except (AttributeError, TypeError):  # not a mapping
        return None


def _as(value: Any, kind: type, what: str) -> Any:
    if not isinstance(value, kind):
        raise TypeError(f"{what} must be a JSON {'object' if kind is Mapping else 'array'}, got {type(value).__name__}")
    return value


def _read(record: Any) -> tuple[Counter, dict[str, Counter]]:
    """One record's counts; raises on anything it cannot read, before any count is kept."""
    record = _as(record, Mapping, "a run record")
    config = _as(record["agent_config"], Mapping, "agent_config")
    transcript = _as(record["transcript"], list, "transcript")
    counts: Counter = Counter()
    histograms = {name: Counter() for name in HISTOGRAMS}
    key_rule, call_rule = config.get("report_key_rule"), config.get("reasoning_call_rule")
    histograms["report_key_rule"][NONE if key_rule is None else str(key_rule)] += 1
    histograms["reasoning_call_rule"][NONE if call_rule is None else str(call_rule)] += 1
    outcome = _as(record["outcome"], Mapping, "outcome")
    if key_rule is not None:
        _decode_firings(record, config, key_rule, transcript, outcome.get("status") == "submitted", counts)
    _recovered_calls(transcript, call_rule, counts)
    cap = _cap(outcome)
    if cap is not None:
        histograms["caps"][cap] += 1
    return counts, histograms


def _decode_firings(record: Mapping, config: Mapping, key_rule: Any, transcript: list, submitted: bool,
                    counts: Counter) -> None:
    """The rule's firings over the record's submit_report calls. A checker cell's record can hold two accepted calls
    (loop._check_and_retry): the final report is the last one, of a record that ended submitted, and only its rows are
    the record's ``final_rows``."""
    from harness.agent.tools import REPORT_KEY_RULES, SUBMIT_REPORT, run_submit

    if key_rule not in REPORT_KEY_RULES:  # checked first: a ValueError of the rule's reading is then its number fault
        raise ValueError(f"unknown report-key rule {key_rule!r}; known: {', '.join(REPORT_KEY_RULES)}")
    iri_rule = config.get("iri_rule", IRI_RULE_BEFORE_7)
    last = None  # the last accepted call: its class, its rule reading and whether its verdict was reproduced
    for entry in _as(record["tool_calls"], list, "tool_calls"):
        entry = _as(entry, Mapping, "a tool_calls entry")
        if entry.get("name") != SUBMIT_REPORT.name or entry.get("status") not in RULE_READS:
            continue  # never read by the tool: skipped, malformed, truncated
        arguments, raw = entry["arguments"], _arguments_raw(transcript, entry)
        strict = run_submit(arguments, iri_rule, None, raw)
        try:
            ruled = run_submit(arguments, iri_rule, key_rule, raw)
        except ValueError:  # the rule's harness fault (tools._json_number_text): the run raised there as well
            ruled = None
        accepted = entry.get("terminal") == "submitted"
        if ruled is None:
            kind, reproduced = "calls_rule_fault", entry["status"] == "crashed"
        else:
            kind = _kind(strict.is_error, ruled.is_error, strict.final_rows == ruled.final_rows)
            reproduced = entry["status"] == "executed" and accepted == (not ruled.is_error)
        counts["submit_calls"] += 1
        counts[kind] += 1
        counts["calls_not_reproduced"] += not reproduced
        if accepted:
            last = (kind, ruled, reproduced)
    if submitted and last is not None:
        kind, ruled, reproduced = last
        counts["records_final_report"] += 1
        counts["records_final_report_rescued"] += kind in RESCUED
        # a verdict reproduced but not the record's final rows
        counts["calls_not_reproduced"] += (reproduced and ruled is not None and not ruled.is_error
                                           and ruled.final_rows != record.get("final_rows"))


def _kind(strict_refused: bool, rule_refused: bool, same_rows: bool) -> str:
    if not strict_refused and not rule_refused:
        return "calls_unchanged" if same_rows else "calls_reread"
    if strict_refused and rule_refused:
        return "calls_refused"
    return "calls_rescued" if strict_refused else "calls_lost"


def _arguments_raw(transcript: list, entry: Mapping) -> str | None:
    """The call's JSON text as the run passed it to the tool: from its turn's ``tool_calls`` (loop.py, agent-run/21),
    and None for a call the reasoning-call rule took (it has none)."""
    if entry.get("source") == "reasoning":
        return None
    for turn in transcript:
        if isinstance(turn, Mapping) and turn.get("role") == "assistant" and turn.get("turn") == entry.get("turn"):
            for call in turn.get("tool_calls") or []:
                if isinstance(call, Mapping) and call.get("id") == entry.get("id"):
                    return call.get("arguments_raw")
    raise LookupError(f"the call {entry.get('id')!r} of turn {entry.get('turn')!r} is in no assistant turn's "
                      f"tool_calls")


def _recovered_calls(transcript: list, call_rule: Any, counts: Counter) -> None:
    taken = 0
    for turn in transcript:
        turn = _as(turn, Mapping, "a transcript turn")
        if turn.get("role") != "assistant":
            continue
        found = turn.get("recovered_tool_call")
        if call_rule is None:
            if found is not None:
                raise ValueError(f"turn {turn.get('turn')!r} has a recovered_tool_call under no reasoning-call rule")
            continue
        counts["assistant_turns"] += 1
        if found is None:
            continue
        flag = found.get("taken") if isinstance(found, Mapping) else None
        if flag is True:
            taken += 1
        elif flag is False:
            counts["turns_recovered_not_taken"] += 1
        else:
            raise ValueError(f"turn {turn.get('turn')!r}: recovered_tool_call has no boolean taken: {flag!r}")
    counts["turns_recovered_taken"] += taken
    counts["records_with_recovered_call"] += taken > 0


def _cap(outcome: Mapping) -> str | None:
    """The cap an outcome ended at (loop.CAPS, or any other the record names), None when it is not capped."""
    status = outcome.get("status")
    if status == "capped":
        return str(outcome.get("cap"))
    if isinstance(status, str) and status.startswith("capped:"):
        return status[len("capped:"):]
    return None
