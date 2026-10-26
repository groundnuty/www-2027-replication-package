"""The infrastructure-fault rule for the test runs (change-log entries 129, 148 and 149; science, design log, 21 Oct 07:51,
18:04 and 18:10; pre-registration section 5), after GLM's server fault of 21 Oct (entry 124): one set of definitions,
used by the dev baselines (scripts/infra_dev_baselines.py) and the monitor (scripts/infra_fault_monitor.py) alike, so
the 10x comparison compares like with like.

* Trigger (it only HOLDS the model; entry 148): in a clock-aligned 15-minute bin of UTC (:00, :15, :30, :45), a model's
  share of call-less assistant turns, or of attempts ending without a submission, passes its trigger rate (strictly)
  with at least MIN_EVENTS such events in the bin. A cell's trigger rate is FACTOR times its dev rate (the frozen
  per-model rates of the fault-rule file, data/runs/infra-fault/fault-rule.json, bound by the frozen config); a
  zero-rate cell keeps "MIN_EVENTS events in a bin". Entry 129's "+40 points" variant for high-baseline cells is
  dropped (entry 148: no frozen numbers; no P model's cell comes near it). The share pools the model's arms; the
  threshold weights each arm's trigger rate by the bin's own arm mix (sum of n_arm * trigger_arm over sum of n_arm), as
  the expected share does its dev rate: per arm where the dev records hold the arm, else the model's rate over all its
  dev arms. A model or arm with no rate is never evaluated: it is reported as unmonitored. Nothing is relabelled at
  trigger time: whether the attempts are an infrastructure failure is decided by the confirmation
  (harness/agent/fault_probe.py, the frozen fault probe; a served-fingerprint change; a provider-confirmed incident).
* The branch phase (ruling of 09:31): a branch record that retried counts its live retry alone (its turns after the
  parent's replayed ones), each turn at its real start, ``ended_at - (wall_s - started_s)`` (the record's clock
  continues the parent's), and its retry's ending; it is judged against the post-fix branch rates (the baselines'
  branch section, under the arm key "branch:<cell>"): the model's own where it has at least BRANCH_MIN_RETRIES
  post-fix retries, else the pooled post-fix rate. Until that section is computed the branch phase has no baseline
  and its attempts are reported as unmonitored. A branch record that did not flag made no request: left out.
* A call-less assistant turn: an assistant turn of the transcript whose ``tool_calls`` is empty, as recorded (prose, a
  call written as text, an empty reply). It is binned by its own start: the record's ``started_at`` plus its
  ``tokens.per_turn`` entry's ``started_s``, matched by turn number; a turn with no such entry is left out of both the
  bins and the baselines, and counted (``untimed_turns``).
* An attempt ending without a submission: outcome label failed:no_submission or refused (the agent loop's endings of a
  model that stops calling tools; a textual decline is ``refused`` from agent-run/17, failed:no_submission before), as
  the GLM diagnosis counted them. A capped attempt kept calling, an abstention is a submission (its tool) and a leaked
  or malformed call is a call-format failure: none is counted. It is binned by the attempt's start.
* Read: OUT/records/*.json only (aborted, infrastructure and set-aside attempts live elsewhere; a write in progress is
  a .tmp). Left out, and counted by reason: an attempt that failed infrastructure on every try (the provider's outcome,
  not the model's); a branch record that did not flag ("no_retry"); a record whose route model is not in the OUT's
  manifest. The initial phases' dev baselines leave out every branch record ("branch").
* Window (only once CONFIRMED): from the first triggering bin's start until the fault probe passes again. Every
  attempt of the model with ANY model turn in the window (entry 149; ``active_in``: an assistant turn whose interval,
  its start to its start plus its wall time, meets the window, or the attempt's own start in it), in every arm, is an
  infrastructure failure and re-run (scripts/relabel_harness_failure.py --window-model). A hold that is NOT confirmed
  relabels nothing: its attempts stand, scored as usual, and the model resumes; the hold is recorded.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from pathlib import Path
from typing import Any

BIN_S = 15 * 60  # bins start at :00, :15, :30 and :45 of UTC (bin_start)
FACTOR = 10
MIN_EVENTS = 5
BRANCH_MIN_RETRIES = 20  # a model's own post-fix branch rate needs this many retries, else the pooled one
BRANCH_PREFIX = "branch:"  # a branch cell's arm key, so its rate is the branch section's
NO_SUBMISSION_LABELS = frozenset({"failed:no_submission", "refused"})
# metric -> (the events it counts, what they are a share of)
METRICS = {"call_less_turns": ("call_less", "turns"), "no_submission": ("no_submission", "attempts")}
COUNTS = ("turns", "call_less", "attempts", "no_submission")
BASELINES_SCHEMA = "infra-fault-baselines/1"
RULE_SCHEMA = "infra-fault-rule/1"  # data/runs/infra-fault/fault-rule.json: the frozen rates, probe sets and rule
HOLDS_DIR = "holds"  # OUT/holds/<runner model name>.json: the monitor's hold, read by the runner before each claim
SERVED_VERSION_HOLD = "served-version"  # OUT/holds/<model>.served-version.json: a setting changed after the freeze
RULE = {"rule": ("change-log entries 129 (item 2), 148 and 149 (science, design log, 21 Oct 07:51, 18:04 and 18:10; "
                 "pre-registration section 5): the trigger only holds; a window needs confirmation"),
        "bin_s": BIN_S, "bins": "clock-aligned 15-minute bins of UTC (:00, :15, :30, :45)", "factor": FACTOR,
        "min_events": MIN_EVENTS, "compare": "events >= min_events and events > sum over arms of n x factor x dev rate",
        "branch_min_retries": BRANCH_MIN_RETRIES,
        "call_less_turn": "an assistant turn with no tool call as recorded, binned by its own start",
        "no_submission": sorted(NO_SUBMISSION_LABELS),
        "left_out": ["failed:infrastructure records",
                     ("branch records (from the initial phases' baselines; a branch section of its own), and branch "
                      "records that did not flag"),
                     "routes not in the OUT's manifest", "turns with no recorded start"],
        "window": ("confirmed only: from the first triggering bin's start until the fault probe passes again; every "
                   "attempt of the model with any model turn in it, every arm (entry 149)")}


def trigger_rate(rate: Fraction) -> Fraction:
    """A cell's trigger rate: FACTOR times its dev rate (entry 148: the "+40 points" variant is dropped)."""
    return FACTOR * rate


def parse_time(text: str) -> datetime:
    """An ISO time with its zone, in UTC; a time without its zone is refused (ValueError)."""
    t = datetime.fromisoformat(text)
    if t.tzinfo is None:
        raise ValueError(f"{text!r} has no time zone")
    return t.astimezone(UTC)


def bin_start(t: datetime) -> datetime:
    """The start of the aligned 15-minute bin of UTC that holds ``t``."""
    epoch = int(t.timestamp())
    return datetime.fromtimestamp(epoch - epoch % BIN_S, tz=UTC)


@dataclass(frozen=True)
class AttemptEvents:
    """What the rule reads of one record."""

    run_id: str
    model: str  # the runner's name of the model (the OUT's manifest routes), never the route model
    arm: str
    started_at: datetime
    turns: tuple[tuple[datetime, bool], ...]  # each timed assistant turn: (its start, call-less)
    no_submission: bool
    untimed_turns: int = 0
    branch: bool = False  # a branch record's live retry (its arm is "branch:<cell>")


def record_events(rec: dict, name_of: Mapping[str, str]) -> AttemptEvents | str:
    """The rule's events of one record, or why it is left out: "infrastructure", "no_retry" (a branch that did not
    flag), "unmapped", or "untimed" (a branch record without ended_at and wall_s: its live turns cannot be placed in
    time, and counted with none it would pass for an attempt observed). A branch record's events are its live
    retry's (module docstring)."""
    outcome = rec.get("outcome") or {}
    if outcome.get("failure_type") == "infrastructure":
        return "infrastructure"
    checker = rec.get("checker")
    if checker is not None and not checker.get("retry"):
        return "no_retry"
    name = name_of.get(rec.get("model"))
    if name is None:
        return "unmapped"
    start = parse_time(rec["started_at"])
    offsets = {t.get("turn"): t.get("started_s") for t in (rec.get("tokens") or {}).get("per_turn") or []}
    replayed = 0
    if checker is not None:  # the parent's turns replay first; the live ones continue the parent's clock
        replayed = ((checker.get("initial") or {}).get("outcome") or {}).get("turns") or 0
        ended, wall = rec.get("ended_at"), rec.get("wall_s")
        if not (isinstance(ended, str) and isinstance(wall, (int, float))):
            return "untimed"
        end = parse_time(ended)
        at_clock = lambda s: end - timedelta(seconds=wall - s)  # noqa: E731
    else:
        at_clock = lambda s: start + timedelta(seconds=s)  # noqa: E731
    turns, untimed = [], 0
    for t in rec.get("transcript") or []:
        if t.get("role") != "assistant" or (t.get("turn") or 0) <= replayed:
            continue
        s = offsets.get(t.get("turn"))
        if not isinstance(s, (int, float)):
            untimed += 1
            continue
        turns.append((at_clock(s), not t.get("tool_calls")))
    arm = (rec.get("load") or {}).get("arm") or rec.get("condition") or "none"
    return AttemptEvents(run_id=rec["run_id"], model=name, arm=BRANCH_PREFIX + arm if checker is not None else arm,
                         started_at=start, turns=tuple(turns), branch=checker is not None,
                         no_submission=outcome.get("label") in NO_SUBMISSION_LABELS, untimed_turns=untimed)


def route_names(out: Path) -> dict[str, str]:
    """Route model -> the runner's name, from OUT/manifest.json's routes; a route model two names share is refused."""
    manifest = json.loads((Path(out) / "manifest.json").read_text(encoding="utf-8"))
    routes = (manifest.get("fingerprint") or {}).get("routes") or {}
    names: dict[str, str] = {}
    for name, route in routes.items():
        model = route.get("model")
        if model in names:
            raise ValueError(f"{out}: route model {model!r} is both {names[model]} and {name}")
        names[model] = name
    return names


def record_files(out: Path) -> list[Path]:
    return sorted((Path(out) / "records").glob("*.json"))


def out_events(out: Path, cache: dict[Path, AttemptEvents | str] | None = None
               ) -> tuple[list[AttemptEvents], Counter]:
    """Every record of OUT/records/ as AttemptEvents, and how many were left out by reason. A record never changes
    once written, so ``cache`` (path -> events) spares a monitor reading it again; a path gone is dropped from it."""
    names = route_names(out)
    files = record_files(out)
    if cache is not None:
        for gone in {p for p in cache if p.parent == Path(out) / "records"} - set(files):
            del cache[gone]
    evs, left_out = [], Counter()
    for path in files:
        e = cache.get(path) if cache is not None else None
        if e is None:
            e = record_events(json.loads(path.read_text(encoding="utf-8")), names)
            if cache is not None:
                cache[path] = e
        if isinstance(e, str):
            left_out[e] += 1
        else:
            evs.append(e)
    return evs, left_out


# --- baselines ----------------------------------------------------------------------------------


@dataclass
class Tally:
    turns: int = 0
    call_less: int = 0
    attempts: int = 0
    no_submission: int = 0

    def add(self, e: AttemptEvents) -> None:
        self.attempts += 1
        self.no_submission += e.no_submission
        self.turns += len(e.turns)
        self.call_less += sum(cl for _, cl in e.turns)

    def counts(self) -> dict[str, int]:
        return {k: getattr(self, k) for k in COUNTS}


def _with_rates(counts: dict[str, int]) -> dict[str, Any]:
    """The counts and, for reading, their rates (the monitor reads the counts; a rate is None with no denominator)."""
    out: dict[str, Any] = dict(counts)
    for metric, (num, den) in METRICS.items():
        key = "call_less_rate" if metric == "call_less_turns" else "no_submission_rate"
        out[key] = round(counts[num] / counts[den], 6) if counts[den] else None
    return out


def baselines_doc(sources: list[tuple[str, str, str | None]], root: Path) -> dict:
    """The dev baselines from ``sources``, each (name, OUT path under ``root``, cut or None): per model and arm, and per
    model over its arms, the counts of every record that started before the source's cut (all without one), with a
    per-graph (per-source) breakdown. Per source: the records before the cut (the inputs: their count, the sha256 of
    their sorted run ids, how many were used and how many left out by reason), which stay the same while a live OUT
    grows after its cut, and, as read at the time, every record and those after the cut (``as_read``)."""
    arms: dict[str, dict[str, Tally]] = {}
    graphs: dict[tuple[str, str, str], Tally] = {}
    described = []
    for name, rel, cut in sources:
        out = Path(root) / rel
        names = route_names(out)
        cut_at = parse_time(cut) if cut else None
        run_ids, read, after_cut, left_out, used, untimed = [], 0, 0, Counter(), 0, 0
        for path in record_files(out):
            rec = json.loads(path.read_text(encoding="utf-8"))
            read += 1
            if cut_at is not None and parse_time(rec["started_at"]) >= cut_at:
                after_cut += 1
                continue
            run_ids.append(rec["run_id"])
            if rec.get("checker") is not None:  # the branch phase has a section of its own
                left_out["branch"] += 1
                continue
            e = record_events(rec, names)
            if isinstance(e, str):
                left_out[e] += 1
                continue
            used += 1
            untimed += e.untimed_turns
            arms.setdefault(e.model, {}).setdefault(e.arm, Tally()).add(e)
            graphs.setdefault((e.model, e.arm, name), Tally()).add(e)
        described.append({"name": name, "out": rel, "cut": cut_at.isoformat() if cut_at else None,
                          "records": len(run_ids), "records_used": used, "left_out": dict(sorted(left_out.items())),
                          "untimed_turns": untimed,
                          "run_ids_sha256": hashlib.sha256("\n".join(sorted(run_ids)).encode("utf-8")).hexdigest(),
                          "as_read": {"records": read, "after_cut": after_cut}})
    models = {}
    for model in sorted(arms):
        total = {k: sum(getattr(t, k) for t in arms[model].values()) for k in COUNTS}
        models[model] = {"all": _with_rates(total), "arms": {
            arm: {**_with_rates(t.counts()), "by_graph": {g: graphs[(m, a, g)].counts()
                                                          for (m, a, g) in sorted(graphs) if (m, a) == (model, arm)}}
            for arm, t in sorted(arms[model].items())}}
    return {"schema": BASELINES_SCHEMA, "rule": RULE, "sources": described, "models": models,
            "branch": branch_section([], root)}


BRANCH_RULE = ("post-fix branch rates (design ruling, design log, 21 Oct 09:31): per model, the live retries of the "
               "branch records of the post-fix branch runs (Claude's from the lean retry-cells rerun on the fix code, "
               "every P model's from rehearsal 2's branch rerun and the targeted retry test); a model with fewer than "
               f"{BRANCH_MIN_RETRIES} retries takes the pooled post-fix rate; pending until those runs finish")


def branch_section(sources: list[tuple[str, str]], root: Path, accepted_code: list[str] | None = None) -> dict:
    """The baselines' branch section from ``sources`` (name, OUT path under ``root``): per model (and per cell, for
    reading) the counts of every branch record that retried, its live retry only, and the pooled counts. With
    ``accepted_code`` a record whose harness commit is not one of them is left out ("other_code"): the post-fix rates
    count retries on the fix code only. No source: "pending", and the branch phase has no baseline."""
    if not sources:
        return {"status": "pending", "rule": BRANCH_RULE, "min_retries": BRANCH_MIN_RETRIES, "sources": [],
                "models": {}, "pooled": None}
    models: dict[str, dict[str, Tally]] = {}
    described = []
    for name, rel in sources:
        out = Path(root) / rel
        names = route_names(out)
        read, used, left_out, code = 0, 0, Counter(), Counter()
        for path in record_files(out):
            rec = json.loads(path.read_text(encoding="utf-8"))
            read += 1
            if rec.get("checker") is None:
                left_out["initial"] += 1
                continue
            commit = (rec.get("harness") or {}).get("git_commit")
            if accepted_code is not None and commit not in accepted_code:
                left_out["other_code"] += 1
                continue
            e = record_events(rec, names)
            if isinstance(e, str):
                left_out[e] += 1
                continue
            used += 1
            code[str(commit)] += 1
            cells = models.setdefault(e.model, {})
            cells.setdefault(e.arm.removeprefix(BRANCH_PREFIX), Tally()).add(e)
        described.append({"name": name, "out": rel, "records": read, "records_used": used,
                          "left_out": dict(sorted(left_out.items())), "code": dict(sorted(code.items()))})
    out_models = {}
    for model in sorted(models):
        total = {k: sum(getattr(t, k) for t in models[model].values()) for k in COUNTS}
        out_models[model] = {"all": _with_rates(total),
                             "arms": {a: _with_rates(t.counts()) for a, t in sorted(models[model].items())}}
    pooled = {k: sum(m["all"][k] for m in out_models.values()) for k in COUNTS}
    # "empty": the sources gave no retry (none flagged, or none on the accepted code): still no branch baseline
    return {"status": "computed" if pooled["attempts"] else "empty", "rule": BRANCH_RULE,
            "min_retries": BRANCH_MIN_RETRIES, "sources": described, "accepted_code": accepted_code,
            "models": out_models, "pooled": _with_rates(pooled)}


@dataclass(frozen=True)
class Baselines:
    """The dev counts the monitor compares against (a baselines file's ``models``, and its ``branch`` section)."""

    models: dict[str, dict[str, Any]]
    branch: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_doc(cls, doc: dict) -> Baselines:
        if doc.get("schema") != BASELINES_SCHEMA:
            raise ValueError(f"baselines schema {doc.get('schema')!r}, expected {BASELINES_SCHEMA!r}")
        return cls(models=doc["models"], branch=doc.get("branch") or {})

    @classmethod
    def from_rule(cls, doc: dict) -> Baselines:
        """The frozen per-model dev rates of a fault-rule file (load_rule): its ``dev_rates.models``, counts per model
        and arm; the branch section as frozen there (pending: the branch phase is unmonitored)."""
        rates = doc.get("dev_rates") or {}
        return cls(models=rates["models"], branch=rates.get("branch") or {})

    @property
    def branch_status(self) -> str:
        return self.branch.get("status") or "pending"

    @classmethod
    def load(cls, path: Path) -> Baselines:
        return cls.from_doc(json.loads(Path(path).read_text(encoding="utf-8")))

    def rate(self, model: str, arm: str, metric: str) -> tuple[Fraction, str] | None:
        """The dev rate of ``metric`` for the model's arm ("arm"), else over all its dev arms ("model"); None for a
        model with no dev records (unmonitored). A branch cell ("branch:<cell>") takes the post-fix branch rate: the
        model's ("branch") with at least BRANCH_MIN_RETRIES retries, else the pooled one ("branch_pooled"); None while
        the branch section is pending."""
        num, den = METRICS[metric]
        if arm.startswith(BRANCH_PREFIX):
            if self.branch_status != "computed":
                return None
            own = (self.branch.get("models") or {}).get(model)
            if own is not None and own["all"]["attempts"] >= BRANCH_MIN_RETRIES and own["all"][den]:
                return Fraction(own["all"][num], own["all"][den]), "branch"
            pooled = self.branch.get("pooled")
            if pooled and pooled[den]:
                return Fraction(pooled[num], pooled[den]), "branch_pooled"
            return None
        entry = self.models.get(model)
        if entry is None:
            return None
        own = entry["arms"].get(arm)
        if own is not None and own[den]:
            return Fraction(own[num], own[den]), "arm"
        if entry["all"][den]:
            return Fraction(entry["all"][num], entry["all"][den]), "model"
        return None


# --- the trigger --------------------------------------------------------------------------------


@dataclass(frozen=True)
class BinResult:
    model: str
    start: datetime
    metric: str
    events: int
    denominator: int
    share: float
    expected: float  # the dev rates weighted by the bin's arm mix
    threshold: float  # the trigger rates weighted likewise: the share must pass it
    triggered: bool
    by_arm: dict[str, tuple[int, int, float, str]] = field(default_factory=dict)  # arm -> (events, n, rate, source)

    def describe(self) -> dict[str, Any]:
        return {"bin": self.start.isoformat(timespec="milliseconds"), "metric": self.metric, "events": self.events,
                "denominator": self.denominator, "share": round(self.share, 6), "expected": round(self.expected, 6),
                "threshold": round(self.threshold, 6),
                "by_arm": {a: {"events": v[0], "n": v[1], "dev_rate": round(v[2], 6), "dev_rate_of": v[3]}
                           for a, v in sorted(self.by_arm.items())}}


def _judge(model: str, metric: str, cells: Mapping[str, list[int]], baselines: Baselines
           ) -> tuple[int, int, Fraction, Fraction, dict] | None:
    """Pooled events, denominator, expected events and threshold events (each arm's trigger rate times its n) of
    ``cells`` (arm -> [events, n]); None if nothing is monitored."""
    events = n = 0
    expected_events = threshold_events = Fraction(0)
    by_arm = {}
    for arm, (num, den) in sorted(cells.items()):
        rate = baselines.rate(model, arm, metric)
        if rate is None or not den:
            continue
        events, n = events + num, n + den
        expected_events += den * rate[0]
        threshold_events += den * trigger_rate(rate[0])
        by_arm[arm] = (num, den, float(rate[0]), rate[1])
    if not n:
        return None
    return events, n, expected_events, threshold_events, by_arm


def triggers(events: int, threshold_events: Fraction) -> bool:
    """The rule: at least MIN_EVENTS events, and strictly more than the trigger rates allow (exact)."""
    return events >= MIN_EVENTS and events > threshold_events


def _cells(evs: Iterable[AttemptEvents]) -> dict[tuple[str, datetime, str], dict[str, list[int]]]:
    cells: dict[tuple[str, datetime, str], dict[str, list[int]]] = {}
    for e in evs:
        c = cells.setdefault((e.model, bin_start(e.started_at), "no_submission"), {}).setdefault(e.arm, [0, 0])
        c[0] += e.no_submission
        c[1] += 1
        for t, call_less in e.turns:
            c = cells.setdefault((e.model, bin_start(t), "call_less_turns"), {}).setdefault(e.arm, [0, 0])
            c[0] += call_less
            c[1] += 1
    return cells


def _since(evs: Iterable[AttemptEvents], since: Mapping[str, datetime] | None) -> list[AttemptEvents]:
    """The attempts each model started at or after its ``since`` (a resume), with their turns."""
    since = since or {}
    return [e for e in evs if e.model not in since or e.started_at >= since[e.model]]


def evaluate(evs: Iterable[AttemptEvents], baselines: Baselines,
             since: Mapping[str, datetime] | None = None) -> list[BinResult]:
    """Every (model, bin, metric) with a monitored denominator, judged by the rule; per model only its attempts started
    at or after ``since[model]`` count (after a resume the old fault bins are the window's, not a new trigger)."""
    results = []
    for (model, start, metric), cells in sorted(_cells(_since(evs, since)).items()):
        judged = _judge(model, metric, cells, baselines)
        if judged is None:
            continue
        events, n, expected_events, threshold_events, by_arm = judged
        results.append(BinResult(model=model, start=start, metric=metric, events=events, denominator=n,
                                 share=events / n, expected=float(expected_events / n),
                                 threshold=float(threshold_events / n), triggered=triggers(events, threshold_events),
                                 by_arm=by_arm))
    return results


def first_triggers(results: Iterable[BinResult]) -> dict[str, BinResult]:
    """Per model, its earliest triggering bin (both metrics of a bin: the call-less one first)."""
    first: dict[str, BinResult] = {}
    for r in sorted(results, key=lambda r: (r.start, r.metric)):
        if r.triggered and r.model not in first:
            first[r.model] = r
    return first


def monitored(baselines: Baselines, e: AttemptEvents) -> bool:
    return baselines.rate(e.model, e.arm, "no_submission") is not None


def unmonitored(evs: Iterable[AttemptEvents], baselines: Baselines,
                since: Mapping[str, datetime] | None = None) -> dict[str, dict[str, dict[str, int]]]:
    """Per model and arm with records but no rate (no dev records of the model; a branch cell while the branch section
    is pending), their attempts and timed turns: the rule cannot see them."""
    out: dict[str, dict[str, dict[str, int]]] = {}
    for e in _since(evs, since):
        if not monitored(baselines, e):
            c = out.setdefault(e.model, {}).setdefault(e.arm, {"attempts": 0, "turns": 0})
            c["attempts"] += 1
            c["turns"] += len(e.turns)
    return out


# --- the window (entry 149) -----------------------------------------------------------------------


def turn_intervals(rec: dict) -> list[tuple[datetime, datetime]]:
    """Each timed assistant turn of a record as (its start, its start plus its wall time), on the record's clock (a
    branch record's continues its parent's, record_events); a turn with no recorded start is left out, one with no wall
    time is an instant. Read: started_at, ended_at, wall_s, the transcript's turn numbers and tokens.per_turn's
    started_s and wall_s; never an outcome."""
    start = parse_time(rec["started_at"])
    per_turn = {t.get("turn"): t for t in (rec.get("tokens") or {}).get("per_turn") or []}
    checker = rec.get("checker")
    replayed = 0
    if checker is not None:
        replayed = ((checker.get("initial") or {}).get("outcome") or {}).get("turns") or 0
        ended, wall = rec.get("ended_at"), rec.get("wall_s")
        if not (isinstance(ended, str) and isinstance(wall, (int, float))):
            return []
        end = parse_time(ended)
        at_clock = lambda s: end - timedelta(seconds=wall - s)  # noqa: E731
    else:
        at_clock = lambda s: start + timedelta(seconds=s)  # noqa: E731
    out = []
    for t in rec.get("transcript") or []:
        if t.get("role") != "assistant" or (t.get("turn") or 0) <= replayed:
            continue
        entry = per_turn.get(t.get("turn")) or {}
        s = entry.get("started_s")
        if not isinstance(s, (int, float)):
            continue
        w = entry.get("wall_s")
        t0 = at_clock(s)
        out.append((t0, t0 + timedelta(seconds=w if isinstance(w, (int, float)) and w > 0 else 0)))
    return out


def active_in(rec: dict, start: datetime, end: datetime | None) -> bool:
    """Whether the attempt had a model turn in the window [start, end) (end None: open): one of its turn intervals
    [start, end) meets it, or the attempt started in it (an attempt that failed before its first reply). Entry 149: every such attempt of
    the model, in every arm, is set aside once the window is confirmed."""
    def inside(t: datetime) -> bool:
        return start <= t and (end is None or t < end)

    if inside(parse_time(rec["started_at"])):
        return True
    # a turn is [its start, its end): it meets the window when it ends after the start and starts before the end
    return any(inside(t0) or (t1 > start and t0 < start) for t0, t1 in turn_intervals(rec))


# --- the frozen fault-rule file ---------------------------------------------------------------------


def rule_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rule(path: Path, pin: Path | None = None) -> dict:
    """The fault-rule file (RULE_SCHEMA), refused unless its trigger parameters are this module's and, with ``pin`` (the
    frozen config's pin, data/freeze/frozen-config.json), unless its sha256 is the pin's ``fault_rule.sha256`` for this
    path: the monitor and the probe run only on the frozen rule."""
    raw = Path(path).read_bytes()
    doc = json.loads(raw)
    if doc.get("schema") != RULE_SCHEMA:
        raise ValueError(f"{path}: schema {doc.get('schema')!r}, expected {RULE_SCHEMA!r}")
    trig = doc.get("trigger") or {}
    want = {"bin_s": BIN_S, "factor": FACTOR, "min_events": MIN_EVENTS}
    got = {k: trig.get(k) for k in want}
    if got != want:
        raise ValueError(f"{path}: trigger {got} is not this code's {want}")
    if pin is not None:
        bound = (json.loads(Path(pin).read_text(encoding="utf-8")).get("fault_rule") or {})
        value = hashlib.sha256(raw).hexdigest()
        if bound.get("sha256") != value:
            raise ValueError(f"{path}: sha256 {value} is not the frozen pin's {bound.get('sha256')!r} ({pin})")
        if Path(bound.get("path") or "").name != Path(path).name:
            raise ValueError(f"{pin} pins {bound.get('path')!r}, not {path}")
    return doc


# --- the hold the runner reads --------------------------------------------------------------------


def hold_path(out: Path, name: str) -> Path:
    """The infrastructure-fault monitor's hold (it writes and releases only this one)."""
    return Path(out) / HOLDS_DIR / f"{name}.json"


def served_version_hold_path(out: Path, name: str) -> Path:
    """The hold of a model whose per-model setting changed after the freeze (scripts/run_pilot.py, change-log entry
    129, item 3): a served-version change, so it has no release in this OUT."""
    return Path(out) / HOLDS_DIR / f"{name}.{SERVED_VERSION_HOLD}.json"


def store_hold_path(out: Path, name: str, copy: str) -> Path:
    """The store time-out monitor's hold of model ``name`` in OUT while its store copy is held (entry 149, item 2)."""
    return Path(out) / HOLDS_DIR / f"{name}.store-{copy}.json"


def read_hold(out: Path, name: str) -> dict | None:
    """A hold of model ``name`` in OUT (the monitor's, else a served-version change's, else a held store copy's), or
    None. A hold file that cannot be read still holds the model: a half-written or damaged hold must never let the
    model run."""
    stores = sorted((Path(out) / HOLDS_DIR).glob(f"{name}.store-*.json"))
    path = next((p for p in (hold_path(out, name), served_version_hold_path(out, name), *stores) if p.exists()), None)
    if path is None:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"model": name, "unreadable": str(exc)}
    return data if isinstance(data, dict) else {"model": name, "unreadable": "not a JSON object"}
