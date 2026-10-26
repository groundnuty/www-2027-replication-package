"""How many queries a store read through a query gate may have running at once, and when that number steps down.

A store's description (harness/gated_store.py) may set its gate's ``query_gate``:

    {"slots": 3, "step_down": [2, 1], "timeout_baseline": {"timeouts": 4, "queries": 200}}

``slots`` is how many queries may run at once (harness/query_gate.py; without the setting, one). ``step_down`` lists
the lower sizes in the order they are stepped to. The limit is either relative, ``timeout_baseline``: the share of the
store's queries that timed out with one slot, as counts; or absolute, ``timeout_limit`` instead, e.g.
``{"timeouts": 40, "queries": 100}`` (design log, design rulings of 21 Oct). A size of one slot takes neither.

The rule: over the last ``TIMEOUT_WINDOW`` queries sent at the current size, when the share that timed out is above
the limit (absolute: the given share; relative: ``max(TIMEOUT_FACTOR x baseline, baseline + TIMEOUT_MARGIN)``), the
gate steps down to the next size of ``step_down`` below the current one. It never steps up within a version of the rule:
the runner starts each start of a run at the lowest size an earlier step-down of the same version reached, whatever
else that start's setting said. A setting may name its ``rule_version`` and a ``rule_note`` (the ruling and its
evidence): a new version starts at its own configured size, the history of the earlier ones stays as it was, and the
runner records the change (scripts/run_pilot.py). Only a full window decides; after a step the window starts again, and a query sent before the step
(an earlier epoch of the gate) is no evidence about the new size. A trigger at the last size is logged and changes
nothing (the window starts again). The shares are compared as exact fractions.

What is counted is the one function ``call_outcomes``, for the run as it goes (VirtuosoClient.observe_call, per tool
call) and for a baseline measured from records alike (``timeout_counts``): each request of a tool call that the
agent's query sent, its one re-send included, but not a probe (harness/agent/infra.py: a call's ``infra_retries`` name
every failed request and every answered probe, in the order sent, so a call's ``query_timings`` entry ``i`` is a probe
when its ``infra_retries`` entry ``i`` is one) and not a request the gate kept back (outcome ``error:gate``: nothing was
sent). A request timed out when its outcome is a time-out of any kind (``timeout:``: the client's own deadline, the
answer past the wall-time share, a flagged answer, an HTTP 206), each of which the agent is shown as one time-out.

Every query the rule takes is logged (``TimeoutStepDown``'s ``on_query``: its sequence number, time, epoch and
whether it timed out), and ``recompute`` gives the events again from those lines alone.
"""

from __future__ import annotations

import dataclasses
import threading
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from typing import Any

TIMEOUT_WINDOW = 100
TIMEOUT_FACTOR = 2
TIMEOUT_MARGIN = Fraction(1, 100)  # one percentage point
LIMIT_KEYS = ("timeout_baseline", "timeout_limit")  # exactly one of them, for more than one slot
RULE_KEYS = ("rule_version", "rule_note")  # optional: the rule's version, and the ruling and evidence behind it


def _whole(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class GateSetting:
    slots: int
    step_down: tuple[int, ...] = ()
    baseline_timeouts: int = 0
    baseline_queries: int = 0
    limit_timeouts: int = 0  # an absolute limit (``timeout_limit``), in place of the baseline
    limit_queries: int = 0
    rule_version: str | None = None
    rule_note: str | None = None

    @property
    def baseline(self) -> Fraction | None:
        return Fraction(self.baseline_timeouts, self.baseline_queries) if self.baseline_queries else None

    @property
    def absolute_limit(self) -> Fraction | None:
        return Fraction(self.limit_timeouts, self.limit_queries) if self.limit_queries else None

    @property
    def limit(self) -> Fraction | None:
        """The share of a window's queries above which the gate steps down."""
        if self.absolute_limit is not None:
            return self.absolute_limit
        return timeout_limit(self.baseline) if self.baseline is not None else None

    def to_json(self) -> dict:
        if self.limit_queries:
            out = {"slots": self.slots, "step_down": list(self.step_down),
                   "timeout_limit": {"timeouts": self.limit_timeouts, "queries": self.limit_queries}}
        else:
            out = {"slots": self.slots, "step_down": list(self.step_down),
                   "timeout_baseline": {"timeouts": self.baseline_timeouts, "queries": self.baseline_queries}
                   if self.baseline_queries else None}
        if self.rule_version is not None:  # only when set, so an older setting reads and records as before
            out["rule_version"] = self.rule_version
        if self.rule_note is not None:
            out["rule_note"] = self.rule_note
        return out


def parse_setting(obj: Any) -> GateSetting:
    """A description's ``query_gate``, checked (ValueError otherwise)."""
    if not isinstance(obj, dict) or not any(set(obj) - set(RULE_KEYS) == {"slots", "step_down", k} for k in LIMIT_KEYS):
        raise ValueError("query_gate must name exactly slots, step_down and one of " + " or ".join(LIMIT_KEYS)
                         + ", and may name " + " and ".join(RULE_KEYS))
    for key in RULE_KEYS:
        if key in obj and not (isinstance(obj[key], str) and obj[key].strip()):
            raise ValueError(f"query_gate.{key} must be a non-empty string")
    if "rule_note" in obj and "rule_version" not in obj:
        raise ValueError("query_gate.rule_note goes with a rule_version")
    if obj.get("slots") == 1 and any(key in obj for key in RULE_KEYS):  # one slot has no step-down rule to version
        raise ValueError("a gate of one slot has no step-down rule, so no rule_version or rule_note")
    return dataclasses.replace(_parse_sizes_and_limit(obj), rule_version=obj.get("rule_version"),
                               rule_note=obj.get("rule_note"))


def _parse_sizes_and_limit(obj: dict) -> GateSetting:
    limit_key = "timeout_limit" if "timeout_limit" in obj else "timeout_baseline"
    slots, steps, base = obj["slots"], obj["step_down"], obj[limit_key]
    if not (_whole(slots) and slots >= 1):
        raise ValueError("query_gate.slots must be a whole number of at least 1")
    if not (isinstance(steps, list) and all(_whole(n) for n in steps)):
        raise ValueError("query_gate.step_down must be a list of whole numbers")
    if any(not 1 <= n < prev for prev, n in zip([slots, *steps], steps)):
        raise ValueError("query_gate.step_down must go down from slots, each size at least 1")
    if slots == 1:
        if steps or base is not None:
            raise ValueError(f"a gate of one slot has no step_down and no {limit_key}")
        return GateSetting(1)
    if not (isinstance(base, dict) and set(base) == {"timeouts", "queries"} and _whole(base["timeouts"])
            and _whole(base["queries"]) and 0 <= base["timeouts"] <= base["queries"] and base["queries"] >= 1):
        raise ValueError(f"query_gate.{limit_key} must be {{\"timeouts\": k, \"queries\": n}}, 0 <= k <= n, n >= 1")
    if limit_key == "timeout_limit":
        return GateSetting(slots, tuple(steps), limit_timeouts=base["timeouts"], limit_queries=base["queries"])
    return GateSetting(slots, tuple(steps), base["timeouts"], base["queries"])


def timeout_limit(baseline: Fraction) -> Fraction:
    """The share of a window's queries above which the gate steps down."""
    return max(TIMEOUT_FACTOR * baseline, baseline + TIMEOUT_MARGIN)


def call_outcomes(query_timings: list | None, infra_retries: list | None) -> list[tuple[int, bool]]:
    """(index into ``query_timings``, timed out) for each request of one tool call that counts (module docstring)."""
    retries = infra_retries or []
    out = []
    for i, entry in enumerate(query_timings or []):
        if not isinstance(entry, dict):
            continue
        retry = retries[i] if i < len(retries) and isinstance(retries[i], dict) else {}
        if retry.get("query") == "probe":
            continue
        outcome = str(entry.get("outcome") or "")
        if outcome == "error:gate":
            continue
        out.append((i, outcome.startswith("timeout:")))
    return out


def timeout_counts(records: Iterable[dict]) -> tuple[int, int]:
    """(time-outs, queries) over the tool calls of ``records`` (records as JSON), counted by ``call_outcomes``."""
    timeouts = queries = 0
    for rec in records:
        for call in rec.get("tool_calls") or []:
            for _, timed_out in call_outcomes(call.get("query_timings"), call.get("infra_retries")):
                queries += 1
                timeouts += timed_out
    return timeouts, queries


class StepDownRule:
    """The rule's state: the size allowed, the epoch and the window of the current epoch. ``observe`` takes one
    query (its epoch and whether it timed out) and returns the event it causes, or None."""

    def __init__(self, setting: GateSetting, allowed: int):
        if setting.slots < 2 or setting.limit is None:
            raise ValueError("the step-down rule is for a gate of several slots with a time-out limit or baseline")
        if not (_whole(allowed) and 1 <= allowed <= setting.slots):
            raise ValueError(f"{allowed!r} slot(s) is not between 1 and this gate's {setting.slots}")
        self.setting = setting
        self.allowed = allowed  # any size up to the slots: a run keeps the lowest size an earlier start reached
        self.epoch = 0
        self.limit = setting.limit
        self.window: deque[bool] = deque(maxlen=TIMEOUT_WINDOW)

    def observe(self, epoch: int, timed_out: bool) -> dict | None:
        if epoch != self.epoch:  # sent before the latest step: no evidence about this size
            return None
        self.window.append(bool(timed_out))
        if len(self.window) < TIMEOUT_WINDOW:
            return None
        timeouts = sum(self.window)
        if Fraction(timeouts, TIMEOUT_WINDOW) <= self.limit:
            return None
        lower = [n for n in self.setting.step_down if n < self.allowed]
        old = self.allowed
        new = lower[0] if lower else old
        st = self.setting
        above = (f"{st.limit_timeouts}/{st.limit_queries}" if st.absolute_limit is not None else
                 f"max({TIMEOUT_FACTOR} x {st.baseline_timeouts}/{st.baseline_queries}, "
                 f"{st.baseline_timeouts}/{st.baseline_queries} + {TIMEOUT_MARGIN.numerator}/{TIMEOUT_MARGIN.denominator})")
        event = {"kind": "step_down" if lower else "trigger_at_lowest", "from": old, "to": new, "epoch": self.epoch,
                 "window": {"queries": TIMEOUT_WINDOW, "timeouts": timeouts},
                 "limit": {"numerator": self.limit.numerator, "denominator": self.limit.denominator},
                 "reason": f"{timeouts} of the last {TIMEOUT_WINDOW} queries timed out, above {above}"}
        self.allowed = new
        self.epoch += 1
        self.window.clear()
        return event


def rule_json(setting: GateSetting | None = None) -> dict:
    """The rule's constants, for the events that record a start (with ``setting``: the limit it gives, exactly)."""
    out = {"window": TIMEOUT_WINDOW, "factor": TIMEOUT_FACTOR,
           "margin": {"numerator": TIMEOUT_MARGIN.numerator, "denominator": TIMEOUT_MARGIN.denominator},
           "counted": "requests of the agent's tool calls that went out, probes excluded; timed out: outcome timeout:*"}
    if setting is not None and setting.limit is not None:
        out.update(kind="absolute" if setting.absolute_limit is not None else "relative",
                   limit={"numerator": setting.limit.numerator, "denominator": setting.limit.denominator})
    return out


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class TimeoutStepDown:
    """The rule run against a live gate: ``observe`` takes each query as its tool call ends (from any thread), logs it
    (``on_query``), and on a trigger changes the gate (``QueryGate.step_down``, or a new epoch at the last size) and
    reports the event (``on_event``), all under one lock, so the log's order is the order the rule saw.

    A log that cannot be written never fails the tool call that told the rule (the attempt's result stands) and never
    keeps the gate from stepping down: the first such failure is kept in ``error`` and passed to ``on_error`` (the
    runner then starts no new attempt). So that no step-down can be lost to a resume, the logs come first: a query
    counts only once its line is written (the rule's state is always what ``recompute`` gives from the lines), and a
    step-down's event is written before the gate changes; a resume recomputes each earlier start's events from its
    lines as well (scripts/run_pilot.py, gate_start_level)."""

    def __init__(self, gate: Any, setting: GateSetting, *, now: Callable[[], str] = _utc_now,
                 on_query: Callable[[dict], None] | None = None, on_event: Callable[[dict], None] | None = None,
                 on_error: Callable[[BaseException], None] | None = None):
        allowed, epoch = gate.level()
        if epoch != 0:
            raise ValueError("the step-down rule starts with the gate's first epoch")
        self.gate = gate
        self.rule = StepDownRule(setting, allowed)
        self.now = now
        self.on_query = on_query
        self.on_event = on_event
        self.on_error = on_error
        self.error: BaseException | None = None
        self.seq = 0
        self._lock = threading.Lock()

    def _log(self, callback: Callable[[dict], None] | None, item: dict) -> bool:
        """Write ``item`` with ``callback``; False when it could not be written (kept in ``error``)."""
        if callback is None:
            return True
        try:
            callback(item)
            return True
        except Exception as exc:  # noqa: BLE001 - kept and reported, never raised into the tool call
            if self.error is None:
                self.error = exc
                if self.on_error is not None:
                    try:
                        self.on_error(exc)
                    except Exception:  # noqa: BLE001, S110 - the error is kept in ``error`` all the same
                        pass
            return False

    def observe(self, epoch: int, timed_out: bool) -> dict | None:
        with self._lock:
            self.seq += 1
            line = {"seq": self.seq, "at": self.now(), "epoch": epoch, "timed_out": bool(timed_out)}
            if not self._log(self.on_query, line):  # not on record: not counted (the rule stays what its lines give)
                return None
            event = self.rule.observe(epoch, timed_out)
            if event is None:
                return None
            event.update(at=line["at"], seq=line["seq"])
            self._log(self.on_event, event)  # first on record; the gate steps down whether or not it could be
            if event["to"] < event["from"]:
                self.gate.step_down(event["to"])
            else:
                self.gate.new_epoch()
            if self.gate.level() != (self.rule.allowed, self.rule.epoch):
                raise RuntimeError("the gate's level and the step-down rule's disagree")
            return event


def recompute(lines: Iterable[dict], setting: GateSetting, allowed: int) -> list[dict]:
    """The events of one start again, from its logged queries (``on_query``'s lines, in order) and the size it
    started at."""
    rule = StepDownRule(setting, allowed)
    events = []
    for line in lines:
        event = rule.observe(line["epoch"], line["timed_out"])
        if event is not None:
            event.update(at=line["at"], seq=line["seq"])
            events.append(event)
    return events
