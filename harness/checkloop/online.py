"""The checker inside the agent loop: the "checker" and "both" cells (design log,
AC1-AC3, AC5, AC8).

After the initial submission, ``OnlineChecker`` picks the checked query (rule
22), runs the cell's arm (one source, or the union) through a recording store
client, and decides: a retry only when an ontology-derived check flags (rule
9). The message is the fixed-format feedback ("feedback" mode) or the generic
"Please re-check your answer." ("generic" mode, the matched control). The generic
retry runs exactly on the curated-flagged set (C5 compares the feedback retry with
it on the outputs the curated checker flags): a generic cell decides on the arm's
curated source or sources alone, whatever else its arm holds. The loop
(harness/agent/loop.py) appends it as a user turn and allows exactly one more
submission under ``RetryBudget`` (the attempt's own caps, pre-registration section 3).

Everything the decision used is recorded: the arm's identity (source and
hierarchy hashes), the switches, the budget, the feedback format's hash, the
checked query, the verdict, the check reports and the store responses the
checks read. ``replay_decision`` recomputes the decision from the recorded
inputs with no store and no model; ``RecordedChecker`` serves a recorded
decision back when a replay checks the loop alone.

``branch_attempt`` runs a retry arm from a recorded attempt's initial phase:
the initial phase is replayed from the record (no model call, no store), then
the live route takes over for the retry. That is how the generic control, and
every source's retry, run on the same flagged initial outputs (AC3: "on the
same flagged set"), and how a checker cell shares its initial outputs with the
prompt-only cell of the same cohort.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from harness.agent.infra import DEFAULT_LIMIT_S, RetryingSparql
from harness.agent.loop import AgentConfig, _skipped_result, run_attempt
from harness.agent.record import RunRecord
from harness.agent.replay import (
    RecordedRoute,
    RecordedSparql,
    ReplayClock,
    ReplayDivergence,
    _canonical,
    _question,
    config_of,
)
from harness.checkloop.arms import Arm, verdict
from harness.checkloop.checked import checked_output
from harness.checkloop.feedback import (
    GENERIC_RETRY_TEXT,
    generic_sha256,
    render_lines,
    retry_turn_text,
    set_aside_sha256,
    set_aside_text,
)
from harness.checkloop.store_log import RecordingClient, ReplayStoreClient
from harness.score import ScoringRules
from harness.terms import row_from_json

# The online decision's version, recorded in every record's checker settings; a record replays (and exports) under the
# rule its version names. /1: every cell decides on its whole arm, a generic cell too (so it retried on a census flag);
# /2: a generic cell decides on the arm's curated sources alone (the generic retry runs on the curated-flagged set, B10).
CHECKER_VERSION = "checker-loop/2"
CHECKER_VERSIONS = ("checker-loop/1", CHECKER_VERSION)
GENERIC_ON_FULL_ARM = frozenset({"checker-loop/1"})  # versions whose generic cell decided on its whole arm
# Entry 126: a new checker closes every retry turn with the set-aside sentence (feedback.SET_ASIDE_TEXT, entry 135's
# final wording) and names it in its settings ("set_aside": its sha256); a record made before has no such setting, and
# its retry turn replays without; a record made under entry 126's sentence replays with that sentence.
SET_ASIDE_DEFAULT = True


def set_aside_on(set_aside: bool | str | None = None) -> bool | str:
    """Whether a checker closes its retry turns with a set-aside sentence: ``set_aside`` (True: this code's; a sentence:
    a record's, replayed; False: none), else the default."""
    return SET_ASIDE_DEFAULT if set_aside is None else set_aside


def set_aside_setting(on: bool | str) -> dict[str, Any]:
    """The checker settings' entry naming the set-aside sentence by its sha256: this code's when ``on`` is True, the
    sentence ``on`` is otherwise (a record's, replayed); none when off (a record made before entry 126)."""
    if not on:
        return {}
    return {"set_aside": {"sha256": set_aside_sha256(None if on is True else on)}}


def recorded_set_aside(settings: dict[str, Any]) -> bool | str:
    """The set-aside sentence a record's retry turn closed with: False when its settings have no set-aside entry (a
    record made before entry 126), else the known sentence they name (feedback.SET_ASIDE_TEXTS: entry 126's or entry
    135's), with which the record replays; ReplayDivergence when they name any other, or none (an explicit null is not
    an absent entry: no checker writes one)."""
    if "set_aside" not in settings:
        return False
    recorded = settings["set_aside"]
    text = set_aside_text(recorded.get("sha256")) if isinstance(recorded, dict) and set(recorded) == {"sha256"} else None
    if text is None:
        raise ReplayDivergence("the record's retry turn closed with a set-aside sentence this code does not know")
    return text
MODES = {"feedback": "checker_feedback", "generic": "generic_retry"}  # mode -> the user turn's kind
# The baselines' branch cells (harness/baselines, design log): mode -> the retry message's user-turn kind.
# They run through branch_attempt like a checker arm but are not OnlineChecker modes.
BASELINE_MODES = {"empty": "empty_retry", "judge": "judge_retry"}
# Every user-turn kind that opens a retry phase: where a branch cuts a record's initial phase.
RETRY_KINDS = frozenset(MODES.values()) | frozenset(BASELINE_MODES.values())
GENERIC_TRIGGER_KIND = "curated"  # the source kind whose flags trigger the generic retry (and decide its verdict)


@dataclass(frozen=True)
class RetryBudget:
    """The retry's own budget, the same for the feedback and the generic arm:
    exactly one more submission, within these caps counted from the retry's start.
    Pre-registration section 3: a retry gets the same turn and wall caps (and the
    same tools) as the initial attempt, so the caps have no default here; build
    them with ``RetryBudget.matching(config)`` from the attempt's AgentConfig. The
    output cap per turn needs no field: the retry runs under the attempt's config."""

    max_turns: int
    max_wall_s: float
    max_nudges: int
    submissions: int = 1

    def __post_init__(self) -> None:
        if self.submissions != 1:
            raise ValueError("a flag triggers at most one retry: exactly one more submission")

    @classmethod
    def matching(cls, config: AgentConfig) -> RetryBudget:
        return cls(max_turns=config.max_turns, max_wall_s=config.max_wall_s, max_nudges=config.max_nudges)


@dataclass(frozen=True)
class Switches:
    """Rule 23: the online retry settings, frozen at pre-registration: every
    ontology-derived check on, both detection switches included."""

    heuristic: bool = True
    branch_local: bool = True


FROZEN_SWITCHES = Switches()


@dataclass
class CheckInput:
    tool_calls: Sequence[dict]
    final_rows: Sequence[Any] | None
    rules: ScoringRules
    row_cap: int | None
    submission_index: int


@dataclass
class Decision:
    retry: bool
    message_kind: str | None
    message: str | None  # the retry's content: the feedback lines, or the generic text (recorded as checker.retry.message)
    initial: dict[str, Any]  # merged into the record's checker.initial
    turn: str | None = None  # the retry turn's whole text, when it is more than the content (entry 126's sentence)


class OnlineChecker:
    """``store_client``: the store the arm's footprint checks read (live), or a
    ``ReplayStoreClient`` over a recorded log (replay)."""

    def __init__(self, arm: Arm, store_client: Any, mode: str = "feedback", *, budget: RetryBudget,
                 switches: Switches = FROZEN_SWITCHES, version: str = CHECKER_VERSION, set_aside: bool | None = None):
        if mode not in MODES:
            raise ValueError(f"unknown checker mode {mode!r}; known: {sorted(MODES)}")
        if version not in CHECKER_VERSIONS:
            raise ValueError(f"unknown checker version {version!r}; known: {list(CHECKER_VERSIONS)}")
        self.version = version
        self.arm = arm
        # the checks that decide: the whole arm, except that the generic retry runs exactly on the curated-flagged set
        # (spec trace B10), so a generic cell's verdict, and with it the ledger row's status, is the curated sources'.
        # A replay of an older record (checker-loop/1) decides as that record did: on its whole arm.
        self.deciding = arm
        if mode == "generic" and version not in GENERIC_ON_FULL_ARM:
            curated = tuple(c for c in arm.compilers if c.source.kind == GENERIC_TRIGGER_KIND)
            if not curated:
                raise ValueError(f"a generic cell is triggered by the {GENERIC_TRIGGER_KIND} source's flags: arm "
                                 f"{arm.name!r} has none")
            self.deciding = Arm(arm.name, curated)
        self.store_client = store_client
        self.mode = mode
        self.budget = budget
        self.switches = switches
        self.set_aside = set_aside_on(set_aside)  # entry 126; False: a record made before
        self.last_client: RecordingClient | None = None

    def settings(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "mode": self.mode,
            "arm": self.arm.identity(),
            "switches": asdict(self.switches),
            "retry_budget": asdict(self.budget),
            "feedback_format": self.arm.feedback_format(),
            "generic_sha256": generic_sha256(),
            **set_aside_setting(self.set_aside),
        }

    def check(self, inp: CheckInput) -> Decision:
        output, status = checked_output(inp.tool_calls, inp.final_rows, inp.rules, upto=inp.submission_index,
                                        row_cap=inp.row_cap)
        client = RecordingClient(self.store_client)
        self.last_client = client
        if output is None:
            reports, v = [], verdict([], no_query=True, answer=[r if isinstance(r, tuple) else row_from_json(r)
                                                                for r in (inp.final_rows or [])])
        else:
            reports = self.deciding.check(output, client)
            v = verdict(reports, heuristic=self.switches.heuristic, branch_local=self.switches.branch_local)
        message = kind = None
        if v.retry:
            kind = MODES[self.mode]
            message = render_lines(v.fired) if self.mode == "feedback" else GENERIC_RETRY_TEXT
        r = inp.rules
        initial = {
            "matching": {"multiset": r.multiset, "rel_tol": r.rel_tol, "abs_tol": r.abs_tol,
                         "column_order": r.column_order},  # rule 22's matching settings
            "checked_query": output.query if output is not None else None,
            "checked_query_status": status,
            "verdict": v.to_json(),
            "reports": [r.to_json() for r in reports],
            "store_log": client.log(),
        }
        turn = retry_turn_text(message, self.set_aside) if v.retry else None
        return Decision(v.retry, kind, message, json.loads(json.dumps(initial, ensure_ascii=False)), turn)


# The fields of a record's checker.initial that the loop itself writes (harness/agent/loop.py).
_LOOP_FIELDS = frozenset({"outcome", "final_rows", "score", "submission_index"})


class RecordedChecker:
    """Serves a record's checker decision back, so a replay can check the loop
    (transcript, retry phase, tool calls) without the sources."""

    def __init__(self, record: RunRecord | dict):
        rec = record if isinstance(record, RunRecord) else RunRecord.from_json(record)
        if rec.checker is None:
            raise ValueError("the record has no checker decision")
        self._checker = rec.checker
        budget = rec.checker["settings"]["retry_budget"]
        self.budget = RetryBudget(**budget)

    def settings(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._checker["settings"]))

    def check(self, inp: CheckInput) -> Decision:
        initial = self._checker["initial"]
        if inp.submission_index != initial["submission_index"]:
            raise ReplayDivergence("the replayed initial submission is not the recorded one")
        retry = self._checker["retry"]
        # Everything the decision put in the initial record (a baseline's decision adds its own keys,
        # such as the judge's call), not the loop's own fields.
        keep = [k for k in initial if k not in _LOOP_FIELDS]
        turn = retry_turn_text(retry["message"], recorded_set_aside(self._checker["settings"])) if retry else None
        return Decision(retry is not None, retry["message_kind"] if retry else None,
                        retry["message"] if retry else None, {k: json.loads(json.dumps(initial[k])) for k in keep}, turn)


def _without_scores(checker: dict) -> dict:
    out = json.loads(json.dumps(checker))
    out["initial"]["score"] = None
    return out


def replay_decision(record: RunRecord | dict, arm: Arm) -> Decision:
    """Recompute a recorded checker decision offline: the same arm (checked by its
    recorded identity) over the recorded store responses. Raises ReplayDivergence
    unless the verdict, the reports, the message and the retry decision are the
    recorded ones and every recorded store response was used. A decision recorded
    before rule 10's kind existed (change-log entry 108) is compared without it."""
    rec = record if isinstance(record, RunRecord) else RunRecord.from_json(record)
    chk = rec.checker
    if chk is None or chk["initial"]["submission_index"] is None:
        raise ValueError("the record has no checked initial submission")
    settings = chk["settings"]
    if settings.get("version") not in CHECKER_VERSIONS:
        raise ReplayDivergence(f"the recorded checker version {settings.get('version')!r} is not one of "
                               f"{list(CHECKER_VERSIONS)}")
    if arm.identity() != settings["arm"]:
        raise ReplayDivergence("the arm's sources or shared hierarchy differ from the recorded ones")
    if settings["feedback_format"] != arm.feedback_format():
        raise ReplayDivergence("the feedback format differs from the recorded one")
    replay = ReplayStoreClient(chk["initial"]["store_log"])
    checker = OnlineChecker(arm, replay, settings["mode"], budget=RetryBudget(**settings["retry_budget"]),
                            switches=Switches(**settings["switches"]), version=settings["version"],
                            set_aside=recorded_set_aside(settings))
    initial = chk["initial"]
    decision = checker.check(CheckInput(rec.tool_calls, initial["final_rows"], _matching(rec),
                                        rec.store.get("row_cap"), initial["submission_index"]))

    def diverges(message: str) -> ReplayDivergence:
        # the decision recomputed offline goes with the divergence: a set-aside records it beside the online one
        exc = ReplayDivergence(message)
        exc.recomputed = decision_summary(decision.initial, decision.retry)
        return exc

    if replay.missing:  # a query the log lacks: the compiler read it as a failed query
        raise diverges(f"checker store replay: {replay.divergence()}")
    got = _without_fault_kind(decision.initial) if _fault_kind_unrecorded(initial) else decision.initial
    for key in ("checked_query", "checked_query_status", "verdict", "reports"):
        if got[key] != initial[key]:
            raise diverges(f"the recomputed checker decision differs at {key}")
    if replay.unused:
        raise diverges(f"checker store replay: {replay.divergence()}")
    retry = chk["retry"]
    if decision.retry != (retry is not None) or (retry and decision.message != retry["message"]):
        raise diverges("the recomputed retry decision or message differs from the recorded one")
    return decision


FAULT_KIND = "data_fault_kind"  # rule 10's kind (change-log entry 108), added after checker-loop/2 records were written


def _fault_kind_unrecorded(initial: dict) -> bool:
    """A decision recorded before rule 10's kind existed: a data fault among its fired checks, or in its reports, that
    names no kind (the kind is written on data faults only). Its replay compares everything else; the kind, a reporting
    label no verdict reads, is the one field such a record cannot hold."""
    if any(c.get("data_fault") and FAULT_KIND not in c for c in (initial.get("verdict") or {}).get("fired_checks") or []):
        return True
    return any("data_fault" in r.get("annotations", ()) and FAULT_KIND not in r.get("evidence", {})
               for rep in initial.get("reports") or [] for r in rep.get("results") or [])


def _without_fault_kind(initial: dict) -> dict:
    """A recomputed decision as it would have been recorded before the kind existed."""
    out = json.loads(json.dumps(initial))
    for c in (out.get("verdict") or {}).get("fired_checks") or []:
        c.pop(FAULT_KIND, None)
    for rep in out.get("reports") or []:
        for r in rep.get("results") or []:
            r.get("evidence", {}).pop(FAULT_KIND, None)
    return out


def decision_summary(initial: dict, retry: bool) -> dict[str, Any]:
    """A checker decision as a divergence is recorded (the replay check's held records, a set-aside sidecar): its
    verdict's status, its checked query's status and whether it retried; never the initial answer's score."""
    return {"verdict": (initial.get("verdict") or {}).get("status"),
            "checked_query_status": initial.get("checked_query_status"), "retry": bool(retry)}


def _matching(rec: RunRecord) -> ScoringRules:
    """The settings rule 22's matching used online, as recorded."""
    return ScoringRules(**rec.checker["initial"]["matching"])


# -- branching a retry arm from a recorded initial phase ---------------------------------


def initial_transcript(record: RunRecord) -> list[dict]:
    """The transcript of a record's initial phase: all of it, or for a checker cell's record
    everything before its retry message."""
    for i, t in enumerate(record.transcript):
        if t["role"] == "user" and t.get("kind") in RETRY_KINDS:
            return record.transcript[:i]
    return record.transcript


def _same_initial_phase(branch: list[dict], recorded: list[dict]) -> bool:
    """The branch's transcript starts with the recorded initial phase, byte for byte. A record
    from before skipped calls were answered (before agent-run/14) may lack their results, which the
    branch adds to the last tool turn before the retry; nothing else may differ."""
    n = len(recorded)
    if len(branch) < n or _canonical(branch[: n - 1]) != _canonical(recorded[: n - 1]):
        return False
    got, want = branch[n - 1], recorded[n - 1]
    if _canonical(got) == _canonical(want):
        return True
    if got.get("role") != "tool" or want.get("role") != "tool":
        return False
    k = len(want["results"])
    return (_canonical(got["results"][:k]) == _canonical(want["results"])
            and all(_canonical(r) == _canonical(_skipped_result({"id": r["tool_call_id"], "name": r["name"]}))
                    for r in got["results"][k:]))


class _PrefixRoute:
    """The recorded assistant turns of the initial phase, then the live route. Before the
    live route's first step, the whole initial phase the retry sees (the terminal tool
    response included) must be the recorded one; ``on_start`` is then called once, just
    before that first live request (the replayed steps are no request)."""

    def __init__(self, record: RunRecord, live: Any, turns: int, on_start: Callable[[], None] | None = None):
        self.recorded = RecordedRoute(record)
        self.initial = initial_transcript(record)
        self.live = live
        self.turns = turns
        self.model = record.model
        self.validated = False
        self.on_start = on_start
        if getattr(live, "model", record.model) != record.model:
            raise ValueError(f"the live route serves {live.model!r}, the record was made with {record.model!r}")

    def route_info(self) -> dict:
        return self.live.route_info()

    def step(self, system, transcript, tools, **kw):
        if self.recorded.steps_served < self.turns:
            return self.recorded.step(system, transcript, tools, **kw)
        if not self.validated:
            if not _same_initial_phase(transcript, self.initial):
                raise ReplayDivergence("the branch's initial phase differs from the recorded one")
            self.validated = True
            if self.on_start is not None:
                self.on_start()
        return self.live.step(system, transcript, tools, **kw)


class _PrefixSparql:
    """The recorded SPARQL calls of the initial phase, then the live store (with
    the loop's usual outage retries)."""

    replays_infra_retries = True

    def __init__(self, record: RunRecord, live: Any, calls: int, infra_retry_s: float, sleep: Callable):
        self.recorded = RecordedSparql(record)
        self.live = RetryingSparql(live, limit_s=infra_retry_s, sleep=sleep)
        self.calls = calls
        self.current: Any = self.recorded
        mine, theirs = self.recorded.settings(), live.settings()
        if (mine["timeout_s"], mine["row_cap"]) != (theirs.get("timeout_s"), theirs.get("row_cap")):
            raise ValueError("the live store's time-out or row cap differs from the record's: the prompt would differ")

    def settings(self) -> dict:
        return self.recorded.settings()

    def query(self, query: str):
        self.current = self.recorded if self.recorded.calls_served < self.calls else self.live
        return self.current.query(query)

    @property
    def last_retries(self):
        return self.current.last_retries

    @property
    def last_infra_s(self):
        return self.current.last_infra_s

    @property
    def last_raw_truncated(self):
        return getattr(self.current, "last_raw_truncated", False)


class _PrefixClock:
    """The recorded clock reads of the initial phase, then live time continuing
    from the recorded end of that phase."""

    def __init__(self, record: RunRecord, turns: int, live: Callable[[], float], initial_end_s: float):
        self.recorded = ReplayClock(record)
        self.reads = 1 + turns  # t0, then one read per model step
        self.live = live
        self.initial_end_s = initial_end_s
        self.n = 0
        self.switched_at: float | None = None

    def __call__(self) -> float:
        self.n += 1
        if self.n <= self.reads:
            return self.recorded()
        if self.switched_at is None:
            self.switched_at = self.live()
        return self.initial_end_s + (self.live() - self.switched_at)


def sampling_seed(order_seed: int, parent_run_id: str, arm: str) -> int:
    """The sampling seed of one retry arm on one output: the first 31 bits of
    sha256("<order seed>|<parent run id>|<arm>"), so every arm's retry randomness is fixed
    where the route takes a seed (the consolidated analysis plan, design log)."""
    digest = hashlib.sha256(f"{order_seed}|{parent_run_id}|{arm}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def arm_label(checker: Any) -> str:
    """An arm's name in the branch record: "<mode>/<arm name>" (feedback or generic), or a
    baseline's own label (a baseline has no arm)."""
    label = getattr(checker, "label", None)
    return label if label is not None else f"{checker.mode}/{checker.arm.name}"


def arm_order(order_seed: int, parent_run_id: str, labels: Sequence[str]) -> list[str]:
    """The order in which the arms of one output run: a seeded shuffle of the sorted labels."""
    order = sorted(labels)
    random.Random(sampling_seed(order_seed, parent_run_id, "order")).shuffle(order)
    return order


def _parent(rec: RunRecord) -> str:
    return (rec.checker or {}).get("branched_from") or rec.run_id


def branch_attempt(record: RunRecord | dict, checker: Any, route: Any, sparql: Any, *, gold_rows: Any = None,
                   rules: Any = None, out_dir: str | None = None, clock: Callable[[], float] = time.monotonic,
                   sleep: Callable[[float], None] = time.sleep, order_seed: int | None = None,
                   label: str | None = None, order: Sequence[str] | None = None, position: int = 0,
                   on_start: Callable[[], None] | None = None,
                   served_check: Callable[[str | None, str | None], Any] | None = None) -> RunRecord:
    """A new attempt whose initial phase is ``record``'s, replayed, followed by the
    checker's decision and, on a flag, a live retry. ``record`` must end in a
    submission; a record of a checker cell is branched at its initial submission.

    With ``order_seed`` the retry's sampling is seeded, on a route that takes a seed
    (``seedable``: the OpenAI-compatible routes), with ``sampling_seed(order_seed, parent,
    label)``; the record's ``checker.branch`` holds the seed (None on a route that takes
    none, such as Anthropic's), the arm's label and its place in the output's arm order.

    ``on_start`` is called once, immediately before the retry's first live model request, and
    never when the arm does not flag: what run_attempt's ``on_start`` is to an initial attempt
    (the grid runner's start gate). ``served_check`` is run_attempt's: every turn, the replayed ones included
    (which passed it live), is checked against the model's first served id."""
    rec = record if isinstance(record, RunRecord) else RunRecord.from_json(record)
    branch = None
    if order_seed is not None:
        label = label or arm_label(checker)
        seedable = bool(getattr(route, "seedable", False))
        seed = sampling_seed(order_seed, _parent(rec), label) if seedable else None
        if seed is not None:
            route = route.with_seed(seed)
        branch = {"order_seed": order_seed, "arm": label, "sampling_seed": seed, "seedable": seedable,
                  "order": list(order) if order is not None else [label], "position": position}
    if rec.checker is not None:
        initial_outcome = rec.checker["initial"]["outcome"]
        end_s = rec.checker["retry"]["started_s"] if rec.checker["retry"] else rec.wall_s
    else:
        initial_outcome, end_s = rec.outcome, rec.wall_s
    if initial_outcome["status"] != "submitted":
        raise ValueError(f"only a submitted initial output can be checked (the record's is {initial_outcome['label']})")
    turns = initial_outcome["turns"]
    calls = sum(1 for c in rec.tool_calls if c.get("client_called") and c["turn"] <= turns)
    config = config_of(rec)  # an older record's initial phase replays under the rules it ran under
    if checker.budget != RetryBudget.matching(config):
        raise ValueError(f"pre-registration section 3: the retry gets the attempt's own caps "
                         f"({RetryBudget.matching(config)}), not {checker.budget}")
    store = {k: v for k, v in rec.store.items() if k not in ("sparql_timeout_s", "row_cap")}
    return run_attempt(_question(rec), _PrefixRoute(rec, route, turns, on_start),
                       _PrefixSparql(rec, sparql, calls, config.infra_retry_s or DEFAULT_LIMIT_S, sleep),
                       config, gold_rows, rules, repetition=rec.repetition, store=store, out_dir=out_dir,
                       clock=_PrefixClock(rec, turns, clock, end_s), sleep=sleep, checker=checker,
                       branched_from=rec.run_id, branch=branch, served_check=served_check)


def branch_arms(record: RunRecord | dict, checkers: dict[str, Any], route_for: Callable[[str], Any],
                sparql_for: Callable[[str], Any], *, order_seed: int, **kw: Any) -> list[RunRecord]:
    """Every arm of one flagged output (label -> checker, e.g. "feedback/curated" and
    "generic/curated"), branched from the same recorded initial phase, run one after the
    other in a seeded random order (``arm_order``) that each record keeps with its own place
    in it; each arm's retry gets its own sampling seed where the route takes one."""
    rec = record if isinstance(record, RunRecord) else RunRecord.from_json(record)
    order = arm_order(order_seed, _parent(rec), list(checkers))
    return [branch_attempt(rec, checkers[label], route_for(label), sparql_for(label), order_seed=order_seed,
                           label=label, order=order, position=i, **kw)
            for i, label in enumerate(order)]
