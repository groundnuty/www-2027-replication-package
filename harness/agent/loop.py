"""One attempt: a question, a model route and a SPARQL endpoint -> a run record.

The loop alternates model steps and tool calls until the model submits a
report, abstains, or a budget or failure ends the attempt.

* A reply with tool calls: the calls run in order. The first terminal event (a
  valid ``submit_report``, ``abstain``, or a tool call whose arguments could not
  be read) ends the attempt; later calls in the same reply are recorded as
  skipped. Otherwise every result goes back to the model in one tool turn.
* A reply without tool calls is nudged ("Call submit_report or abstain.") up to
  ``max_nudges`` times, then the attempt fails ``no_submission``.
* A tool call written into the text instead of the tool-call field is detected
  and never executed; the attempt fails ``leaked_tool_call``. Under a model's
  reasoning-call rule (``AgentConfig.reasoning_call_rule``, agent-run/22: Qwen3.6's,
  design ruling in the design log Oct 00:09), a reply with no parsed call whose
  reasoning channel holds exactly one well-formed call to an offered tool with
  valid arguments (providers.reasoning_tool_call) runs that call as if parsed: the
  turn keeps what the server sent (no ``tool_calls``, its ``leaked_tool_call``) and
  records ``recovered_tool_call`` (the rule, the span and text taken, from its
  ``offset``, and the call with the id ``reasoning-call-t<turn>``), the call's log
  entry has ``source`` "reasoning", and the next request carries the call as the
  turn's (with an ``echo_reasoning`` route, the reasoning before the call). Any other
  such reply, or one cut at the output limit, fails ``leaked_tool_call`` as before, its
  ``recovered_tool_call`` saying why the rule did not take it.
* A tool call that cannot be read (arguments that are not a JSON object, or a
  reply cut off by the output limit inside a tool call) is not executed; the
  attempt fails ``malformed_tool_call``.
* The turn and wall caps are checked before each model step; a submission
  returned by a step that started inside the caps is accepted. An attempt
  stopped by a cap (including a model call the attempt's own deadline cut off)
  gets status ``capped`` with ``cap`` "turns" or "wall", rendered
  ``capped:turns`` / ``capped:wall``; it is scored as a failure but stays
  distinguishable, and carries a ``looping`` verdict (see looping.py).
* A context overflow (the provider's error, by the table in providers.py, or a
  context-window stop reason) is the cap "context", ``capped:context``: an agent
  outcome like a turn cap (design log), with the provider's message in
  ``error_message``. Under ``AgentConfig.context_overflow="failed"``, the rule
  of records before agent-run/9 (replay.py), it is ``failed:context_overflow``.
* Every turn records ``truncated``: the reply stopped at the output limit.
* A store outage (infra.py) is not shown to the model: the SPARQL tool waits
  and retries, outside the wall cap. If the store stays down past
  ``infra_retry_s`` the attempt fails ``infrastructure`` (``failed:infrastructure``),
  is not scored, and should be re-run rather than counted. A provider that is
  overloaded or unreachable past its retries (providers.py) ends the attempt the
  same way. The outcome's ``infra_source`` says which: ``store`` or ``provider``
  (None for every other outcome). The route decides what is infrastructure, so a
  replay, which raises the recorded failure type, reproduces either.
* A store shared through a query gate (agent-run/20): the waits for the gate
  and for the server time-out after a request are infrastructure time as well
  (infra.py), outside the wall cap. So that no attempt waits without end,
  ``AgentConfig.total_cap_factor`` (set by the runner for such a store only)
  ends a phase whose whole time, waits included, has reached that many times
  its wall cap as ``failed:infrastructure`` (stage "store"; design ruling,
  design log, 20 Oct: 3), never scored and re-run. It decides only while the
  agent's own time (its wall-cap time, waits excluded) is under its cap
  (design ruling, design log, 23:51): once the agent's own time has reached the
  cap, the attempt ends as it would with no wait at all, ``capped:wall`` (or
  the turn cap, checked first, as before), so long waits never earn an
  attempt a re-run that one without them would not get; a step the wall
  budget cut off ran to the agent's own cap, so it is ``capped:wall``. Under
  the cap, the ceiling is checked before each model step, after the turn and
  wall caps; after each step, on the turn's start plus the step's recorded
  ``wall_s`` (retries included), before its reply is read for a refusal, a
  context stop, a missing call or its calls (a served-model change keeps its
  precedence: it stops the model); and before each tool call, adding the
  whole wall time of each call run so far, so a reply's later calls do not
  run, nor its submission count, once the ceiling has passed. All read
  recorded values only (a replay serves each call's recorded wall time).
* Every turn records the model the provider says answered (``served_model``)
  and whether it differs from the requested one (``model_mismatch``; a dated
  snapshot of the requested id is the same model); the outcome counts such
  turns in ``model_mismatch_turns`` so a silent fallback is visible.
* A refusal (rule ``refusal-v1``, REFUSAL_RULE below: the provider declined,
  or the final message after the last nudge declines the task in the first
  person) ends the attempt ``refused``: scored wrong, its signal in
  ``outcome.refusal``. A crash inside our own code on valid input (not a store
  error, not the model; malformed agent input is a validation error returned to
  the agent, tools.py) ends it ``failed:harness_defect`` with the exception's
  class and stack hash: never scored; the pilot runner sets it aside for a
  re-run after the fix (design log, change-log entry 91). Before agent-run/17 both
  were ``failed:other`` (and a textual decline ``failed:no_submission``):
  ``AgentConfig.refusal_rule`` and ``tool_crash_rule`` "failed", under which
  older records replay.
* ``served_check`` (the pilot runner's, change-log entry 91) is called with each
  turn's served model id and system fingerprint as the turn arrives; when it
  reports a change from the model's first served id in the run, or from its
  first fingerprint (design ruling in the design log Oct 23:24; agent-run/21), the
  attempt ends at that turn ``failed:model_changed`` (agent-run/17), never
  scored, the turn's calls not run. Every turn records its
  ``system_fingerprint``, null when the server sent none.
* Scoring uses the frozen scorer only: its digest is checked before an attempt
  that will be scored starts and again in ``score_outcome`` (rescoring
  included), and any other scorer raises ScorerNotFrozen.
* Every turn records ``first_byte_s`` (providers.py). The outcome counts the
  replies the route's stream guards rejected (``stream_guard_firings``: the
  rejected streams, and per guard the rejections it was part of), over every
  turn's ``retry_errors`` and the failed step's error, so a rate per model and
  arm can be read from the records.

The tool turn that follows a terminal call is recorded in the transcript but,
the attempt being over, never sent to the model.

In a checker cell (``checker``, harness/checkloop/online.py) a submitted answer
is checked; on a flag from an ontology-derived check the attempt continues: the
skipped calls of the terminal reply get results, the checker's message (the
fixed-format feedback, or the generic "Please re-check your answer.") is appended as a
user turn, and a second phase runs under the checker's retry budget (its own
turn, wall and nudge caps) until exactly one more terminal event.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from harness.agent.engine_flags import engine_flags
from harness.engine_suspect import ENGINE_SUSPECT, suspect
from harness.agent.errors import HarnessAbort, InternalToolError
from harness.agent.infra import DEFAULT_LIMIT_S as DEFAULT_INFRA_LIMIT_S
from harness.agent.infra import RetryingSparql, StoreUnavailable
from harness.agent.looping import is_looping, loop_evidence
from harness.agent.metadata import sanitize_store
from harness.agent.prompts import get_condition, load_block, user_message
from harness.agent.providers import (CONTEXT_STOPS, NATIVE_OPENAI, OUTPUT_LIMIT_STOPS, REASONING_CALL_RULES,
                                     STREAM_GUARDS, Provider, StepFailure, StepResult, reasoning_text,
                                     reasoning_tool_call)
from harness.agent.record import RunRecord, harness_commit, utc_now
from harness.agent.tools import (MAX_RESULT_CHARS, MIN_RESULT_CHARS, RAW_ROW_CAP, REPORT_KEY_RULES, SparqlLike,
                                 ToolSpec, agent_tools, execute_tool, render_error)

STATUSES = ("submitted", "abstained", "failed", "capped", "refused")
CAPS = ("turns", "wall", "context")
CONTEXT_OVERFLOW_RULES = ("capped", "failed")
REFUSAL_RULES = ("refused", "failed")  # refused (agent-run/17), or "failed": failed:other / no_submission as before
TOOL_CRASH_RULES = ("harness_defect", "failed")  # failed:harness_defect (agent-run/17), or "failed": failed:other
FAILURE_TYPES = frozenset({
    "no_submission", "provider_error", "malformed_tool_call", "leaked_tool_call", "context_overflow", "other",
    "infrastructure",  # the SPARQL store stayed down past the retry limit: never scored, re-run instead
    "harness_defect",  # a crash inside our own code on valid input: never scored, set aside and re-run after the fix
    "model_changed",  # a turn served by another model id than the model's first in the run: never scored, aborted
})
UNSCORED_FAILURES = frozenset({"infrastructure", "harness_defect", "model_changed"})

# Refusal, rule "refusal-v1" (design ruling in the design log, change-log entry 91; its text rule as the confirmation review of
# 57e148c required it: first-person declining of the task, never a bare word). An attempt is ``refused`` when it ends
# without a submission and either
# * the provider declined (the refusal screen's primary label, feat/prompt-conditions scripts/refusal_screen.py):
#   Anthropic stop_reason "refusal", OpenAI-compatible finish_reason "content_filter" or the message's ``refusal``
#   field (the turn's ``refusal``, providers.py): signal "provider:<stop reason>"; or
# * it would end ``no_submission`` (no tool call after the last nudge) and its final message's text (the content
#   channel, never the reasoning), not cut by the output cap and holding no submission written as text, declines the
#   task in the first person: it matches one of REFUSAL_PHRASES below, tried in order: signal "phrase:<name>".
# A submission written as text is a parseable JSON object holding, at any depth, a "rows" key or a term cell (an
# object whose "type" is iri, uri, literal, typed-literal or bnode and that has a "value"): an answer attempt, so not
# a refusal whatever else the reply says. Any other brace ("I refuse {this}.") changes nothing.
# The text is read in lower case, with typographic apostrophes as straight ones and each run of white space as one
# space. Every phrase but against_guidelines starts at the speaker, the word "i" (not inside a word, not after a
# hyphen or an apostrophe), which is not the speaker after a word naming a numbered thing (NUMBERED_THINGS: "complex
# I", "photosystem I", "type I"). The phrases, exactly:
# * cannot_do: i + one of " can't", " cannot", " can not", " won't", " will not", " am unable to", "'m unable to",
#   " am not able to", "'m not able to"; then optionally " be able to"; then optionally one word ending in "ly"; then
#   a verb of doing the task: answer, help (not "help but" and not "help" + a word ending in "ing", idioms), assist,
#   provide, comply, fulfil, fulfill, share, respond, engage, participate; or "do" + that | this | so | it; or
#   "complete" + this | that | the | your + request | task; unless the negation is an inability (not " won't" or
#   " will not", which decline by will) and the rest of its clause (up to the next . ; ! or ?) holds yet, until,
#   before, without, first, "more" + queries | data | information | results, or "need(s) to" + query | run | check |
#   look | search | explore: the agent is stuck, not declining ("I cannot answer yet; I need to query the graph
#   first.", "I cannot answer because I need to query the graph."; but "I will not provide more information.");
# * cannot_sentence: i + one of " can't", " cannot", " can not", " won't", " will not", then "." or "!" (optionally
#   after one space): the sentence ends there ("I'm sorry, but I can't.");
# * decline: i + optionally one of " must", " have to", " need to", " will", "'ll", " would", "'d", " shall",
#   " should", " am going to", "'m going to"; then optionally one word ending in "ly"; then " decline" (a whole word);
#   or i + " am" | "'m", optionally a word ending in "ly", then " declining";
# * refuse: the same with " refuse" and " refusing" ("I am refusing to answer this question.");
# * against_guidelines: "against my guidelines" (whole words; "my" is the speaker).
# So "The population will decline by 10%.", "Cells refuse to divide" and "The store was unable to answer" (third
# person) are no refusal, nor "I can't help but notice", "I will not need more queries" or "I'm unable to find the
# class" (first person, but no declined task). The screen's keywords map onto these phrases, but for the ones that are
# no refusal in a loop with a query tool: bare "i'm unable to", "i am unable to", "i won't" and "i will not" now need
# a verb of doing the task, "not able to comply" needs its speaker, and every keyword starts at a whole word (it
# matched inside one: "Delhi cannot provide"). declined-v1's cannot_answer verbs that say the agent is stuck or
# unsure (determine, know, tell, say, recall, identify, list) are left out, like its dont_know, not_sure, no_access
# and needs_query patterns and its empty reply. A refused attempt is scored wrong ("refused" in the score's notes)
# and stays separable for the refusal-excluded secondary analysis (``outcome.refusal``).
REFUSAL_RULE = "refusal-v1"
# Words a roman numeral I follows in the domain's prose. Words that can also stand just before a speaking "I" ("at this
# stage I cannot", "in this step I won't") are left out, so such a refusal is still read.
NUMBERED_THINGS = frozenset({
    "complex", "photosystem", "class", "type", "subtype", "phase", "group", "grade", "subunit", "isoform", "domain",
    "family", "subfamily", "category", "factor", "receptor", "collagen", "kingdom", "division", "tier",
})
_SPEAKER = r"(?<![\w'-])i"
_NEGATED = r"(?: can't| cannot| can not| won't| will not| am unable to|'m unable to| am not able to|'m not able to)"
_TASK_VERB = (r"(?:answer|help(?! but\b)(?! \w+ing\b)|assist|provide|comply|fulfil|fulfill|share|respond|engage"
              r"|participate|do (?:that|this|so|it)|complete (?:this|that|the|your) (?:request|task))\b")
_MODAL = r"(?: must| have to| need to| will|'ll| would|'d| shall| should| am going to|'m going to)?(?: \w+ly)?"
REFUSAL_PHRASES: tuple[tuple[str, re.Pattern, bool], ...] = (  # (name, pattern, starts at the speaker)
    ("cannot_do", re.compile(rf"{_SPEAKER}{_NEGATED}(?: be able to)?(?: \w+ly)? {_TASK_VERB}"), True),
    ("cannot_sentence", re.compile(rf"{_SPEAKER}(?: can't| cannot| can not| won't| will not) ?[.!]"), True),
    ("decline", re.compile(rf"{_SPEAKER}(?:{_MODAL} decline\b|(?: am|'m)(?: \w+ly)? declining\b)"), True),
    ("refuse", re.compile(rf"{_SPEAKER}(?:{_MODAL} refuse\b|(?: am|'m)(?: \w+ly)? refusing\b)"), True),
    ("against_guidelines", re.compile(r"\bagainst my guidelines\b"), False),
)
_WORD_BEFORE = re.compile(r"([a-z0-9]+) $")
# A cannot_do match whose clause (up to the next . ; ! or ?) says the agent is not done yet is the agent being stuck,
# not declining: "I cannot answer yet; I need to query the graph first."
_STUCK_CLAUSE = re.compile(r"\b(?:yet|until|before|without|first|more (?:queries|data|information|results)"
                           r"|needs? to (?:query|run|check|look|search|explore))\b")
# Only an inability ("can't", "cannot", "unable") can be the agent being stuck; "won't" and "will not" decline by will.
_VOLITIONAL = re.compile(r" won't| will not")
_CLAUSE_END = re.compile(r"[.;!?]")
_CELL_TYPES = frozenset({"iri", "uri", "literal", "typed-literal", "bnode"})


def _submission_like(obj: Any) -> bool:
    """Whether a decoded JSON value holds, at any depth, a "rows" key or a term cell (iteratively: no depth limit)."""
    stack = [obj]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            kind = item.get("type")
            if "rows" in item or (isinstance(kind, str) and kind in _CELL_TYPES and "value" in item):
                return True
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return False


def holds_submission(text: str) -> bool:
    """Whether ``text`` holds a submission written as text: a parseable JSON object, starting at any "{", that is
    submission-like (``_submission_like``)."""
    decoder = json.JSONDecoder()
    at = text.find("{")
    while at != -1:
        try:
            obj, end = decoder.raw_decode(text, at)
        except (ValueError, RecursionError):
            at = text.find("{", at + 1)
            continue
        if _submission_like(obj):
            return True
        at = text.find("{", end)
    return False


def _normalised_text(text: str) -> str:
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u02bc", "'")
    return re.sub(r"\s+", " ", text).lower()


def declined_text(text: str | None, truncated: bool) -> str | None:
    """refusal-v1's signal for a final message that declines the task, or None (module comment above)."""
    reply = (text or "").strip()
    if truncated or not reply or holds_submission(reply):
        return None
    lowered = _normalised_text(reply)
    for name, pattern, at_speaker in REFUSAL_PHRASES:
        for match in pattern.finditer(lowered):
            before = _WORD_BEFORE.search(lowered, max(0, match.start() - 40), match.start()) if at_speaker else None
            if before is not None and before.group(1) in NUMBERED_THINGS:
                continue
            if name == "cannot_do" and not _VOLITIONAL.search(match.group(0)):
                end = _CLAUSE_END.search(lowered, match.end())
                if _STUCK_CLAUSE.search(lowered, match.start(), end.start() if end else len(lowered)):
                    continue
            return f"phrase:{name}"
    return None


def harness_defect_of(exc: BaseException) -> dict:
    """A crash inside our tool code: the exception's class (the client's own, when it was wrapped as an
    InternalToolError) and the sha256 of its stack, the (path, function) of each frame from where it was caught to
    where it was raised, without line numbers (so an unrelated edit does not split one defect into two hashes)."""
    import traceback
    from pathlib import Path

    root = exc.__cause__ if isinstance(exc, InternalToolError) and exc.__cause__ is not None else exc
    base = Path(__file__).resolve().parent.parent.parent
    frames = []
    for frame in traceback.extract_tb(root.__traceback__):
        path = Path(frame.filename).resolve()
        try:
            where = path.relative_to(base).as_posix()
        except ValueError:
            where = path.name
        frames.append([where, frame.name])
    name = type(root).__name__
    stack = hashlib.sha256(json.dumps([name, frames], separators=(",", ":")).encode("utf-8")).hexdigest()
    return {"exception": name, "stack_sha256": stack}
USAGE_FIELDS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")
ERROR_MESSAGE_LIMIT = 2000


# The frozen scorer (preregistration §3): the sha256 of its closed module set, harness/score/digest.py
# (feat/scorer-digest 05492e6). Nothing is scored by any other scorer, at run time or at analysis time (design log,
# change-log entry 91; the confirmation review of 57e148c). Since change-log entry 127 it is scorer-rules-v6
# (feat/scorer-entry127), which replaced 3d6b1e6afe70fd2a81b764c6d3f97fbec66088b3bd5710d0daa893c16b30606e (v5,
# change-log entry 120), which replaced 4fca135fa146d9308b50a251af7da883234aa781011b098c8a27fa93924ef8ac (v4).
FROZEN_SCORER_DIGEST = "f359683f65f8e2a2e3846ca7f2de86feeff69851af4bf0b0f599ea057ada8c10"


class ScorerNotFrozen(Exception):
    """The scorer on disk is not the frozen one, so nothing may be scored with it."""


def require_frozen_scorer() -> str:
    """The scorer's digest (harness/score/digest.py, read when called, from ``digest.ROOT``), or ScorerNotFrozen
    unless it is the frozen one: a missing module of the frozen set, or any other digest."""
    from pathlib import Path

    from harness.score import digest

    try:
        value = digest.scorer_digest(digest.ROOT)
    except FileNotFoundError as exc:
        missing = Path(exc.filename or "?")
        try:
            missing = missing.relative_to(digest.ROOT)
        except ValueError:
            pass
        raise ScorerNotFrozen(f"the scorer is not the frozen one: {missing.as_posix()} is missing (this tree's scorer "
                              f"predates the frozen scorer {FROZEN_SCORER_DIGEST}, feat/scorer-digest 05492e6)"
                              ) from None
    if value != FROZEN_SCORER_DIGEST:
        raise ScorerNotFrozen(f"the scorer's digest is {value}, not the frozen {FROZEN_SCORER_DIGEST} "
                              f"(preregistration §3, feat/scorer-digest 05492e6)")
    return value


class ScoringError(Exception):
    """Scoring failed after the attempt ran. Carries the unscored record (already
    written when ``out_dir`` was given) so the attempt can be rescored later."""

    def __init__(self, record: RunRecord):
        super().__init__(f"scoring failed for run {record.run_id}; the record is kept unscored")
        self.record = record


# A reasoning-call rule on a route that echoes reasoning is refused (External review of a2ea65c, item 2; lead, 21
# Oct): the echo would send back a turn's reasoning only up to the call the rule ran (providers._echoed_reasoning), a
# combination no model has and science has not ruled on.
ECHO_WITH_REASONING_CALL = ("a reasoning-call rule and a route that echoes reasoning (echo_reasoning) are not combined "
                            "until science rules on it: the echo would send a turn's reasoning only up to the call the "
                            "rule ran")
# The agent config's per-model rules, each recorded only when set (AgentConfig.to_json)
RULE_FIELDS = ("report_key_rule", "reasoning_call_rule")
REASONING_CALL_ID = "reasoning-call-t{turn}"  # the id a call taken from turn <turn>'s reasoning is given and sent with


@dataclass(frozen=True)
class AgentConfig:
    max_turns: int = 30
    max_wall_s: float = 900.0
    max_tokens: int | None = 16000  # per model call; None leaves the route's own default
    max_nudges: int = 2
    condition: str = "none"
    raw_row_cap: int = RAW_ROW_CAP  # SPARQL JSON rows kept per query in the record
    infra_retry_s: float = DEFAULT_INFRA_LIMIT_S  # store-outage retries per query (infra.py), outside the wall cap
    iri_rule: str = "registered_schemes_v3"  # which submitted IRIs are absolute (tools.py; design log)
    context_overflow: str = "capped"  # capped:context (design log), or "failed": failed:context_overflow as before
    # The calls of a reply after a submission or abstention are skipped; from agent-run/14 the record
    # answers each in the reply's tool turn, so a retry branched from it (harness/checkloop) starts
    # from the same bytes. False is the rule of earlier records (replay.py).
    answer_skipped_calls: bool = True
    # Suspect classes of the store (from its private vocabulary, harness/engine_suspect.py): a query of the shape
    # the store answers wrongly gets the engine flag "engine_suspect" (design log). Empty: never flagged.
    engine_suspect_classes: tuple[str, ...] = ()
    # Characters of one sparql_query result shown to the model, for every model, arm and graph (design log, change-log
    # entry 66; tools.py). None: no cap, as before agent-run/15, with the earlier tool description. The pilot runner
    # sets it, the turn and wall caps and max_tokens per model (scripts/run_pilot.py MODELS; design log).
    max_result_chars: int | None = MAX_RESULT_CHARS
    # How the route calls the model, as the pilot runner's model table set it (agent-run/16; design log): "streamed" or
    # "non-streamed", and the stream guards on (providers.py STREAM_GUARDS). Recorded only, so the analysis reads
    # every per-model setting in one place; the route itself acts on them (route.settings says the same). None and
    # () outside the runner, and for a record written before agent-run/16.
    transport: str | None = None
    stream_guards: tuple[str, ...] = ()
    # design log, change-log entry 91 (agent-run/17): a refusal ends ``refused`` (REFUSAL_RULE), and a crash inside our
    # code on valid input ``failed:harness_defect``; "failed" keeps the rule before (failed:other,
    # failed:no_submission), under which a record written before agent-run/17 replays (replay.config_of).
    refusal_rule: str = "refused"
    tool_crash_rule: str = "harness_defect"
    # The graph the run is on (the store's name; agent-run/17), recorded with the tool-result cap resolved for it: a
    # model's cap may differ per graph (scripts/run_pilot.py MODELS; design log). None outside the runner.
    graph: str | None = None
    # An ontology condition's block, by its sha256 in the block store (prompts.BLOCK_DIR); None for "none"
    # (agent-run/18). Last, so a "none" config's other fields keep their order.
    ontology_block_sha256: str | None = None
    # agent-run/20: a phase whose whole time, waits for a shared query gate included, reaches this many times its wall
    # cap while the agent's own time is under its cap ends failed:infrastructure (the runner sets 3 for a gated store;
    # design rulings, design log, 20 Oct). None: no such limit, as before (left out of the record).
    total_cap_factor: float | None = None
    # How submit_report reads a cell object's strings (tools.REPORT_KEY_RULES; agent-run/21): a per-model setting of
    # the pilot runner's table (Gemma's "json_string_cells_v1", change-log entries 110 and 112, completed by the
    # follow-up of 21 Oct 00:20: a type "number" is read as a literal). None, as before: left
    # out of the record, so every other model's records and fingerprints are what they were; an older record replays
    # without it.
    report_key_rule: str | None = None
    # Whether a call written into the reasoning channel runs (providers.REASONING_CALL_RULES; Qwen3.6's
    # "reasoning_xml_call_v1"; design rulings in the design log Oct 00:09 and 01:31; agent-run/22): a per-model setting
    # of the pilot runner's table, None (left out of the record, so every other model's records and fingerprints are
    # what they were) unless set. An older record replays without it.
    reasoning_call_rule: str | None = None

    def to_json(self) -> dict:
        """The config as a record stores it. A field added later is left out at its default, so a "none"
        record's agent_config (and a pilot's fingerprint) is what it was before the field existed."""
        out = dataclasses.asdict(self)
        for key in ("ontology_block_sha256", "total_cap_factor", *RULE_FIELDS):
            if out[key] is None:
                del out[key]
        return out


TRANSPORTS = ("streamed", "non-streamed")


def tools_sha256(tools: tuple[ToolSpec, ...]) -> str:
    spec = [dataclasses.asdict(t) for t in tools]
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _outcome(status: str, *, failure_type: str | None = None, cap: str | None = None,
             error_message: str | None = None, abstain_reason: str | None = None,
             provider_error: dict | None = None, stage: str | None = None, refusal: dict | None = None,
             harness_defect: dict | None = None, model_changed: dict | None = None) -> dict:
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}")
    if failure_type is not None and failure_type not in FAILURE_TYPES:
        raise ValueError(f"unknown failure type {failure_type!r}")
    if (status == "capped") != (cap in CAPS):
        raise ValueError(f"status {status!r} with cap {cap!r}")
    return {
        "status": status,
        "failure_type": failure_type,
        "cap": cap,
        "abstain_reason": abstain_reason,
        "error_message": error_message[:ERROR_MESSAGE_LIMIT] if error_message else error_message,
        "provider_error": provider_error,
        "stage": stage,
        "refusal": refusal,  # refused: {"rule", "signal"} (agent-run/17)
        "harness_defect": harness_defect,  # failed:harness_defect: {"exception", "stack_sha256"} (agent-run/17)
        "model_changed": model_changed,  # failed:model_changed: {"first_served_model", "served_model"} (agent-run/17)
    }


def _failed(failure_type: str, message: str, **kw) -> dict:
    return _outcome("failed", failure_type=failure_type, error_message=message, **kw)


def _capped(cap: str, message: str, **kw) -> dict:
    return _outcome("capped", cap=cap, error_message=message, **kw)


def outcome_label(outcome: dict) -> str:
    """submitted | abstained | refused | failed:<failure_type> | capped:turns | capped:wall | capped:context"""
    status = outcome["status"]
    if status == "capped":
        return f"capped:{outcome['cap']}"
    if status == "failed":
        return f"failed:{outcome['failure_type']}"
    return status


def _assistant_turn(step: StepResult, turn: int) -> dict:
    return {
        "role": "assistant",
        "turn": turn,
        "text": step.text,
        "tool_calls": [dict(c) for c in step.tool_calls],
        "stop_reason": step.stop_reason,
        "response_id": step.response_id,
        "model": step.model,
        "leaked_tool_call": step.leaked_tool_call,
        "refusal": step.refusal,
        "native_format": step.native_format,
        "native": step.native,
    }


_SNAPSHOT_SUFFIX = re.compile(r"-\d{8}")


def same_model(requested: str, served: str | None) -> bool | None:
    """Whether the served model is the requested one (None when the provider did not say).

    A dated snapshot of the requested id (``<requested>-YYYYMMDD``) counts as the same model.
    """
    if not served:
        return None
    if served == requested:
        return True
    return served.startswith(requested) and _SNAPSHOT_SUFFIX.fullmatch(served[len(requested):]) is not None


def _turn_tokens(step: StepResult, turn: int, started_s: float, requested_model: str) -> dict:
    out = {
        "turn": turn,
        "started_s": started_s,
        **{k: step.usage.get(k) for k in USAGE_FIELDS},
        "max_tokens": step.max_tokens,
        "latency_s": step.latency_s,
        "first_byte_s": step.first_byte_s,
        "wall_s": step.wall_s,
        "retries": step.retries,
        "retry_errors": step.retry_errors,
        "stop_reason": step.stop_reason,
        "truncated": step.stop_reason in OUTPUT_LIMIT_STOPS,
        "served_model": step.model,
        "model_mismatch": same_model(requested_model, step.model) is False,
        "usage_raw": step.usage_raw,
        "system_fingerprint": step.system_fingerprint,  # agent-run/21: null when the server sent none
    }
    if step.sent_messages_sha256 is not None:  # agent-run/22: an echoing route's request, checked by every replay
        out["sent_messages_sha256"] = step.sent_messages_sha256
    return out


def token_totals(per_turn: list[dict]) -> dict:
    totals: dict[str, Any] = {}
    for key in USAGE_FIELDS:
        values = [t[key] for t in per_turn if t.get(key) is not None]
        totals[key] = sum(values) if values else None
    inputs = [totals[k] for k in ("input_tokens", "cache_read_tokens", "cache_write_tokens") if totals[k] is not None]
    totals["total_input_tokens"] = sum(inputs) if inputs else None
    totals["turns"] = len(per_turn)
    totals["retries"] = sum(t.get("retries") or 0 for t in per_turn)
    totals["truncated_turns"] = sum(bool(t.get("truncated")) for t in per_turn)
    # Cache writes by lifetime, where the provider breaks them down (Anthropic's usage.cache_creation).
    for ttl in ("5m", "1h"):
        values = [t["usage_raw"]["cache_creation"].get(f"ephemeral_{ttl}_input_tokens")
                  for t in per_turn if isinstance((t.get("usage_raw") or {}).get("cache_creation"), dict)]
        values = [v for v in values if isinstance(v, int)]
        totals[f"cache_write_{ttl}_tokens"] = sum(values) if values else None
    return totals


def stream_guard_firings(per_turn: list[dict], provider_error: dict | None) -> dict:
    """The replies the stream guards rejected in an attempt: in every turn's retry_errors and in the failed
    step's error (whose rejections reach no turn), as ``rejected_streams`` and, per guard, the rejections it
    was part of (both guards can reject one reply)."""
    found = [e["stream_guard"] for t in per_turn for e in t.get("retry_errors") or []
             if isinstance(e, dict) and isinstance(e.get("stream_guard"), dict)]
    found += [g for g in (provider_error or {}).get("stream_guard") or [] if isinstance(g, dict)]
    return {"rejected_streams": len(found), **{name: sum(name in g for g in found) for name in STREAM_GUARDS}}


def _term_rows(rows: Any) -> list[tuple]:
    """Rows of term JSON (or of ``harness.terms`` terms) -> rows of terms."""
    from harness.terms import BNode, IRI, Literal, term_from_json

    return [tuple(c if isinstance(c, (IRI, Literal, BNode)) else term_from_json(c) for c in row) for row in rows]


def scoring_rules(rules: Any):
    """A ``ScoringRules``, from itself, its ``to_json`` form, or a plain dict of its fields."""
    from harness.score import ScoringRules

    if isinstance(rules, ScoringRules):
        return rules
    if isinstance(rules, dict):
        return ScoringRules.from_json(rules) if "version" in rules else ScoringRules(**rules)
    raise TypeError(f"scoring rules must be ScoringRules or a dict, got {type(rules).__name__}")


def score_outcome(outcome: dict, final_rows: list | None, gold_rows: Any, rules: Any) -> dict:
    """The one place the scorer is called (imported lazily so scoring stays optional).

    A capped attempt is scored as "failed" (wrong), and its ``capped:*`` label is
    appended to the score's notes so it stays distinguishable. An attempt that is
    never scored (``UNSCORED_FAILURES``) is refused, and so is any scorer but the
    frozen one (``require_frozen_scorer``), before the scorer is called.
    """
    if outcome.get("failure_type") in UNSCORED_FAILURES:
        raise ValueError(f"a failed:{outcome['failure_type']} attempt is never scored")
    require_frozen_scorer()
    from harness.score import gold_from_json, score_attempt

    status = outcome["status"]
    answer = _term_rows(final_rows) if final_rows is not None else None
    # A gold rule (tie-aware, either-value; harness.score.question_gold) comes as an object, gold rows as a list.
    gold = gold_from_json(gold_rows) if isinstance(gold_rows, dict) else _term_rows(gold_rows)
    score = score_attempt("failed" if status in ("capped", "refused") else status, answer, gold,
                          scoring_rules(rules)).to_json()
    if status in ("capped", "refused"):
        score["notes"] = [*score["notes"], outcome_label(outcome)]
    return score


@dataclass
class _Attempt:
    """What an attempt has built so far, carried across its phases: the initial
    answer, then, in a checker cell, one retry after the checker's message."""

    transcript: list[dict]
    calls_log: list[dict] = dataclasses.field(default_factory=list)
    per_turn: list[dict] = dataclasses.field(default_factory=list)
    turn: int = 0
    nudges: int = 0
    infra_s: float = 0.0  # time spent waiting out store outages; not charged to the wall cap
    on_start: Callable[[], None] | None = None  # called once, before the attempt's first provider request


SKIPPED_RESULT = "not executed: the attempt ended at an earlier call in this turn"


def _skipped_result(call: dict) -> dict:
    """The tool result of a call skipped because the attempt ended at an earlier call of its reply."""
    return {"tool_call_id": call.get("id"), "name": call.get("name"), "content": render_error("skipped", SKIPPED_RESULT),
            "is_error": True}


def _run_phase(att: _Attempt, *, route: Provider, sparql: SparqlLike, cond, system: str, tools, config: AgentConfig,
               raw_cap: int, clock: Callable[[], float], t0: float, phase_start: float, max_turns: int,
               max_wall_s: float, max_nudges: int,
               served_check: Callable[[str | None, str | None], Any] | None = None) -> tuple[dict, list | None]:
    """Model steps and tool calls until a terminal event. ``phase_start`` is the
    phase's start as elapsed time since ``t0``; the turn, wall and nudge caps count
    from it, so a retry gets its own budget. Times are recorded relative to ``t0``,
    and the caps are computed from recorded values only, so a replay decides them
    identically."""
    outcome: dict | None = None
    final_rows: list | None = None
    phase_turns = 0
    phase_nudges = 0
    infra_at_start = att.infra_s
    while outcome is None:
        elapsed = clock() - t0
        charged = (elapsed - phase_start) - (att.infra_s - infra_at_start)
        factor = config.total_cap_factor
        if phase_turns >= max_turns:
            outcome = _capped("turns", f"no submission within {max_turns} turns", stage="loop")
            outcome["final_check_s"] = elapsed
            break
        if charged >= max_wall_s:
            outcome = _capped("wall", f"no submission within {max_wall_s:g} s of wall time", stage="loop")
            outcome["final_check_s"] = elapsed
            break
        if factor is not None and elapsed - phase_start >= factor * max_wall_s:  # the agent's own time is under its cap
            outcome = _failed("infrastructure", f"the attempt took {elapsed - phase_start:.0f} s with its waits for "
                                                f"the store, {factor:g} times its {max_wall_s:g} s wall cap",
                              stage="store")
            outcome["final_check_s"] = elapsed
            break
        if att.on_start is not None and att.turn == 0 and att.nudges == 0:
            att.on_start()
            att.on_start = None
        try:
            step = route.step(system, att.transcript, tools, max_tokens=config.max_tokens,
                              time_left_s=max_wall_s - charged)
        except StepFailure as failure:
            # the attempt's own deadline cut the call off: the step ran to the agent's own cap, so this is capped:wall
            # whatever the waits before it (design ruling, design log, 23:51)
            if failure.failure_type == "budget_wall":
                outcome = _capped("wall", failure.message, provider_error=failure.details or None,
                                  stage="provider")
            elif failure.failure_type == "context_overflow" and config.context_overflow == "capped":
                outcome = _capped("context", failure.message, provider_error=failure.details or None,
                                  stage="provider")
            else:
                outcome = _failed(failure.failure_type, failure.message, provider_error=failure.details or None,
                                  stage="provider")
            outcome["final_check_s"] = elapsed
            break
        except HarnessAbort:
            raise
        except Exception as exc:  # an unexpected reply shape: recorded, not raised
            outcome = _failed("other", f"{type(exc).__name__}: {exc}", stage="provider")
            outcome["final_check_s"] = elapsed
            break
        att.turn += 1
        phase_turns += 1
        att.transcript.append(_assistant_turn(step, att.turn))
        att.per_turn.append(_turn_tokens(step, att.turn, elapsed, route.model))

        change = served_check(step.model, step.system_fingerprint) if served_check is not None else None
        if change is not None:  # this turn is another model's: nothing in it is the model's, nothing of it runs
            changed = dict(change) if isinstance(change, dict) else {"first_served_model": change[0],
                                                                     "served_model": change[1]}
            for call in step.tool_calls:
                att.calls_log.append(_call_entry(len(att.calls_log), att.turn, call, status="skipped", error={
                    "kind": "skipped", "message": "not executed: the served model changed at this turn"}))
            if "system_fingerprint" in changed:
                message = (f"turn {att.turn} was served by {changed['served_model']!r} with system fingerprint "
                           f"{changed['system_fingerprint']!r}, not {changed['first_system_fingerprint']!r}, the "
                           f"model's first in this run: the attempt is aborted unscored")
            else:
                message = (f"turn {att.turn} was served by {changed['served_model']!r}, not by "
                           f"{changed['first_served_model']!r}, the model's first served id in this run: the attempt "
                           f"is aborted unscored")
            outcome = _failed("model_changed", message, stage="model", model_changed=changed)
            break
        if (factor is not None and charged + (step.wall_s or 0.0) < max_wall_s
                and elapsed - phase_start + (step.wall_s or 0.0) >= factor * max_wall_s):
            # the step (its recorded wall time) took the phase past the ceiling while the agent's own time is under its
            # cap: whatever the reply holds (a refusal, no call, calls), it is not taken and none of its calls runs
            for call in step.tool_calls:
                att.calls_log.append(_call_entry(len(att.calls_log), att.turn, call, status="skipped", error={
                    "kind": "skipped", "message": "not executed: the attempt passed its ceiling at this turn"}))
            outcome = _failed("infrastructure", f"the attempt took at least {elapsed - phase_start + step.wall_s:.0f} s "
                                                f"with its waits for the store, {factor:g} times its {max_wall_s:g} s "
                                                f"wall cap", stage="store")
            break
        if step.refusal is not None:
            message = (f"the provider declined to answer (stop_reason={step.stop_reason}, "
                       f"category={step.refusal.get('category')}, explanation={step.refusal.get('explanation')})")
            if config.refusal_rule == "refused":
                outcome = _outcome("refused", error_message=message, stage="model", refusal={
                    "rule": REFUSAL_RULE, "signal": f"provider:{step.stop_reason or 'refusal'}"})
            else:  # the rule before agent-run/17
                outcome = _failed("other", message, stage="model")
            break
        if step.stop_reason in CONTEXT_STOPS:
            message = f"the prompt no longer fits the context window (stop_reason={step.stop_reason})"
            outcome = (_capped("context", message, stage="model") if config.context_overflow == "capped"
                       else _failed("context_overflow", message, stage="model"))
            break
        if step.tool_calls and step.stop_reason in OUTPUT_LIMIT_STOPS:
            for call in step.tool_calls:
                att.calls_log.append(_call_entry(len(att.calls_log), att.turn, call, status="truncated", error={
                    "kind": "truncated", "message": f"the reply hit the output limit (stop_reason="
                                                    f"{step.stop_reason}); the call was not executed"}))
            outcome = _failed("malformed_tool_call", f"the reply hit the output limit (stop_reason="
                                                     f"{step.stop_reason}) inside a tool call; it was not executed",
                              stage="model")
            break
        calls = step.tool_calls
        if not calls and step.leaked_tool_call is not None and config.reasoning_call_rule is not None \
                and step.leaked_tool_call["channel"] == "reasoning" and step.native_format == NATIVE_OPENAI:
            calls = _reasoning_call(att, step, config.reasoning_call_rule, tools)
        if not calls:
            if step.leaked_tool_call is not None:
                leak = step.leaked_tool_call
                outcome = _failed("leaked_tool_call", f"a tool call ({leak['shape']}) was written into the "
                                                      f"{leak['channel']} channel instead of the tool-call field; "
                                                      f"it was not executed", stage="model")
                break
            if phase_nudges >= max_nudges:
                signal = (declined_text(step.text, step.stop_reason in OUTPUT_LIMIT_STOPS)
                          if config.refusal_rule == "refused" else None)
                if signal is not None:
                    outcome = _outcome("refused", error_message=f"the model declined the task after {phase_nudges} "
                                                                f"nudge(s) ({signal})", stage="model",
                                       refusal={"rule": REFUSAL_RULE, "signal": signal})
                else:
                    outcome = _failed("no_submission", f"the model answered without calling a tool after "
                                                       f"{phase_nudges} nudge(s)", stage="model")
                break
            att.nudges += 1
            phase_nudges += 1
            att.transcript.append({"role": "user", "kind": "nudge", "text": cond.nudge})
            continue

        results: list[dict] = []
        ended: dict | None = None
        # the turn so far, from recorded values only (so a replay decides the same): its model step (with retries)
        # and the whole wall time of each call run (its waits, its answer or error; a replay serves the recorded one)
        turn_spent = turn_own = step.wall_s or 0.0  # the whole turn so far, and the agent's own part of it
        for call in calls:
            index = len(att.calls_log)
            if ended is None and factor is not None and charged + turn_own < max_wall_s:
                # past the ceiling while the agent's own time is under its cap, no further call runs and no submission
                # of this reply is taken
                reached = elapsed - phase_start + turn_spent
                if reached >= factor * max_wall_s:
                    ended = _failed("infrastructure", f"the attempt took at least {reached:.0f} s with its waits for "
                                                      f"the store, {factor:g} times its {max_wall_s:g} s wall cap",
                                    stage="store")
            if ended is not None:
                att.calls_log.append(_call_entry(index, att.turn, call, status="skipped", error={
                    "kind": "skipped", "message": SKIPPED_RESULT}))
                if config.answer_skipped_calls and ended["status"] in ("submitted", "abstained"):
                    results.append(_skipped_result(call))
                continue
            if call.get("error") is not None or call.get("arguments") is None:
                problem = call.get("error") or "the tool call has no arguments"
                att.calls_log.append(_call_entry(index, att.turn, call, status="malformed",
                                             error={"kind": "malformed_arguments", "message": problem}))
                ended = _failed("malformed_tool_call", f"tool call {call.get('name')!r} could not be read: {problem}",
                                stage="model")
                continue
            try:
                res = execute_tool(call["name"], call["arguments"], sparql, raw_row_cap=raw_cap,
                                   iri_rule=config.iri_rule, max_result_chars=config.max_result_chars,
                                   report_key_rule=config.report_key_rule, arguments_raw=call.get("arguments_raw"))
            except HarnessAbort:
                raise
            except StoreUnavailable as down:  # not shown to the model, not the agent's failure
                entry = _call_entry(index, att.turn, call, status="infrastructure",
                                    error={"kind": "infrastructure", "message": down.message,
                                           "suspected_query": down.suspected_query})
                entry.update({"client_called": True, "infra_retries": down.retries, "infra_s": down.infra_s,
                              "query_timings": down.query_timings})
                att.calls_log.append(entry)
                att.infra_s += down.infra_s
                ended = _failed("infrastructure", down.message, stage="store")
                continue
            except Exception as exc:  # a harness or client fault, not a tool error for the model
                internal = exc if isinstance(exc, InternalToolError) else None
                message = internal.message if internal else f"{type(exc).__name__}: {exc}"
                entry = _call_entry(index, att.turn, call, status="crashed",
                                    error={"kind": "internal", "message": message})
                entry["client_called"] = bool(internal and internal.client_called)
                att.calls_log.append(entry)
                if config.tool_crash_rule == "harness_defect":
                    # a replay raises the recorded defect (replay.RecordedSparql), whose stack it cannot rebuild
                    defect = getattr(exc, "recorded_defect", None) or harness_defect_of(exc)
                    entry["error"].update(defect)
                    ended = _failed("harness_defect", f"tool {call['name']!r} failed inside the harness "
                                                      f"({defect['exception']}, stack {defect['stack_sha256'][:12]}): "
                                                      f"{message}", stage="tool", harness_defect=dict(defect))
                else:  # the rule before agent-run/17
                    ended = _failed("other", f"tool {call['name']!r} failed inside the harness: {message}",
                                    stage="tool")
                continue
            entry = _call_entry(index, att.turn, call, status="refused" if res.refused else "executed", error=res.error)
            entry.update({
                "result_shown": res.shown,
                "is_error": res.is_error,
                "result_raw": res.result_raw,
                "result_raw_truncated": res.result_raw_truncated,
                "result_meta": res.result_meta,
                "elapsed_s": res.elapsed_s,
                "client_called": res.client_called,
                "terminal": res.terminal,
                "infra_retries": res.infra_retries,
                "infra_s": res.infra_s,
                "query_timings": res.query_timings,
            })
            att.infra_s += res.infra_s
            turn_spent += max(res.elapsed_s, res.infra_s)  # a call takes at least its waits
            turn_own += max(0.0, res.elapsed_s - res.infra_s)
            att.calls_log.append(entry)
            results.append({"tool_call_id": call["id"], "name": call["name"], "content": res.shown,
                            "is_error": res.is_error})
            if res.terminal == "submitted":
                final_rows = res.final_rows
                ended = _outcome("submitted")
            elif res.terminal == "abstained":
                ended = _outcome("abstained", abstain_reason=res.abstain_reason)
        if results:
            att.transcript.append({"role": "tool", "results": results})
        outcome = ended

    return outcome, final_rows


def _reasoning_call(att: _Attempt, step: StepResult, rule: str, tools) -> list[dict]:
    """Under a reasoning-call rule, the call a reply with none parsed wrote into its reasoning channel, as the one call
    of the turn (or none: the reply then fails leaked_tool_call as before); either way the turn records what the rule
    found (``recovered_tool_call``). The call carries ``source`` "reasoning" into its log entry; it is sent back as the
    turn's call (providers.turn_calls)."""
    turn = att.transcript[-1]
    if step.stop_reason in OUTPUT_LIMIT_STOPS:  # a reply cut off is not read for a call, as a parsed one is not run
        turn["recovered_tool_call"] = {"rule": rule, "taken": False, "reason": "output_limit",
                                       "detail": f"the reply hit the output limit (stop_reason={step.stop_reason})"}
        return []
    text = reasoning_text(step.native)
    found = reasoning_tool_call(text, tools)
    if not found["taken"]:
        turn["recovered_tool_call"] = {"rule": rule, **found}
        return []
    call = {"id": REASONING_CALL_ID.format(turn=att.turn), "name": found["name"], "arguments": found["arguments"]}
    turn["recovered_tool_call"] = {"rule": rule, "taken": True, "channel": "reasoning", "offset": found["start"],
                                   "end": found["end"], "text": text[found["start"]:found["end"]], "call": call}
    return [{**call, "source": "reasoning"}]


def _finish_outcome(outcome: dict, att: _Attempt) -> dict:
    """The attempt-level fields of an outcome, over everything the attempt did so far."""
    outcome["turns"] = att.turn
    outcome["nudges"] = att.nudges
    outcome["label"] = outcome_label(outcome)
    outcome["infra_s"] = round(att.infra_s, 3)
    outcome["infra_retries"] = sum(len(c["infra_retries"]) for c in att.calls_log)
    outcome["infra_source"] = outcome["stage"] if outcome["failure_type"] == "infrastructure" else None
    outcome["model_mismatch_turns"] = sum(t["model_mismatch"] for t in att.per_turn)
    outcome["served_models"] = sorted({t["served_model"] for t in att.per_turn if t["served_model"]})
    outcome["stream_guard_firings"] = stream_guard_firings(att.per_turn, outcome.get("provider_error"))
    outcome["looping"] = is_looping(att.calls_log) if outcome["status"] == "capped" else None
    outcome["loop_evidence"] = loop_evidence(att.calls_log)
    # design log (change-log entry 57): the engine-suspect stratum is the union of its parts, each recorded where it is
    # seen; the runner sees the agent's executed queries (the checker records the checked query and the footprint)
    outcome["engine_suspect"] = {
        "any_agent_query": any(ENGINE_SUSPECT in c["engine_flags"] and c["status"] == "executed" for c in att.calls_log)}
    return outcome


def _flag_engine_suspect(att: _Attempt, config: AgentConfig) -> None:
    """Add the engine flag "engine_suspect" to each call of the store's suspect shape not flagged yet (a checker
    cell flags its initial phase before the retry runs, and the whole attempt at its end)."""
    if config.engine_suspect_classes:
        for c in att.calls_log:
            if ENGINE_SUSPECT not in c["engine_flags"] and _engine_suspect(c, config.engine_suspect_classes):
                c["engine_flags"] = [*c["engine_flags"], ENGINE_SUSPECT]


def _answer_skipped_calls(att: _Attempt) -> None:
    """Before a retry, every tool call of the terminal reply gets a result (the
    provider APIs refuse a tool call left without one); calls after the
    submission were skipped and are answered as such."""
    last = att.transcript[-1]
    reply = next(t for t in reversed(att.transcript) if t["role"] == "assistant")
    answered = {r["tool_call_id"] for r in last.get("results", [])} if last["role"] == "tool" else set()
    missing = [c for c in reply["tool_calls"] if c.get("id") not in answered]
    if not missing:
        return
    if last["role"] != "tool":
        last = {"role": "tool", "results": []}
        att.transcript.append(last)
    for c in missing:
        last["results"].append(_skipped_result(c))


def run_attempt(
    question: dict,
    route: Provider,
    sparql: SparqlLike,
    config: AgentConfig = AgentConfig(),
    gold_rows: Any = None,
    rules: Any = None,
    *,
    repetition: int = 0,
    store: dict | None = None,
    out_dir: str | None = None,
    clock: Callable[[], float] = time.monotonic,
    replay_of: str | None = None,
    sleep: Callable[[float], None] = time.sleep,
    on_start: Callable[[], None] | None = None,
    checker: Any = None,
    branched_from: str | None = None,
    branch: dict | None = None,
    served_check: Callable[[str | None, str | None], Any] | None = None,
) -> RunRecord:
    """Run one attempt and return its record (written to ``out_dir`` when given).

    ``store`` describes the SPARQL store (engine, version, image digest, endpoint
    label); the SPARQL time-out and row cap are added from ``sparql.settings()``.
    Scoring runs only when both ``gold_rows`` (gold rows, or a gold rule from
    ``harness.score.question_gold``) and ``rules`` are given, and never
    for an attempt that failed ``infrastructure``, ``harness_defect`` or
    ``model_changed``. ``sleep`` is used only for the
    waits between store-outage retries. ``on_start`` is called once, immediately
    before the first provider request (never, when the attempt makes none): the
    moment the attempt actually starts, which the pilot runner numbers.
    ``served_check`` is called with each turn's served model id and system
    fingerprint as the turn arrives, and returns None, or the outcome's
    ``model_changed`` (a dict; or the pair (the model's first served id, this
    one)) to end the attempt there ``failed:model_changed``. When the attempt will be scored,
    the scorer must be the frozen one: checked before the first provider request
    (ScorerNotFrozen, raised, never recorded).

    ``checker`` (the "checker" and "both" cells, harness/checkloop/online.py):
    after the initial submission the checker runs the cell's checks; when it
    asks for a retry, its message is appended as a user turn and the model gets
    exactly one more submission under the checker's retry budget. The record's
    ``outcome``, ``final_rows`` and ``score`` are then the end-to-end answer; the
    initial answer, its checks and the message are in ``checker``.
    ``branched_from`` names the record whose initial phase this attempt replays
    (a retry arm branched from a shared initial output); ``branch`` records how the
    arm was run (its order among the output's arms, its sampling seed).
    """
    question_id = question.get("question_id")
    if not isinstance(question_id, str) or not question_id:
        raise ValueError("question has no 'question_id'")
    if config.context_overflow not in CONTEXT_OVERFLOW_RULES:
        raise ValueError(f"unknown context_overflow rule {config.context_overflow!r}; known: "
                         f"{', '.join(CONTEXT_OVERFLOW_RULES)}")
    cap = config.max_result_chars
    if cap is not None and (not isinstance(cap, int) or isinstance(cap, bool) or cap < MIN_RESULT_CHARS):
        raise ValueError(f"max_result_chars must be None or an integer of at least {MIN_RESULT_CHARS}, got {cap!r}")
    if checker is not None and (checker.budget.max_turns, checker.budget.max_wall_s, checker.budget.max_nudges) != (
            config.max_turns, config.max_wall_s, config.max_nudges):
        raise ValueError(f"pre-registration section 3: the retry gets the attempt's own caps ({config.max_turns} "
                         f"turns, {config.max_wall_s:g} s, {config.max_nudges} nudges), not {checker.budget}")
    if config.refusal_rule not in REFUSAL_RULES or config.tool_crash_rule not in TOOL_CRASH_RULES:
        raise ValueError(f"unknown refusal or tool-crash rule ({config.refusal_rule!r}, {config.tool_crash_rule!r})")
    if config.total_cap_factor is not None and not config.total_cap_factor >= 1:
        raise ValueError(f"total_cap_factor must be None or at least 1, got {config.total_cap_factor!r}")
    if config.transport is not None and config.transport not in TRANSPORTS:
        raise ValueError(f"unknown transport {config.transport!r}; known: {', '.join(TRANSPORTS)}")
    if not set(config.stream_guards) <= set(STREAM_GUARDS):
        raise ValueError(f"unknown stream guards {sorted(set(config.stream_guards) - set(STREAM_GUARDS))}")
    if config.report_key_rule is not None and config.report_key_rule not in REPORT_KEY_RULES:
        raise ValueError(f"unknown report-key rule {config.report_key_rule!r}; known: {', '.join(REPORT_KEY_RULES)}")
    if config.reasoning_call_rule is not None and config.reasoning_call_rule not in REASONING_CALL_RULES:
        raise ValueError(f"unknown reasoning-call rule {config.reasoning_call_rule!r}; known: "
                         f"{', '.join(REASONING_CALL_RULES)}")
    if config.reasoning_call_rule is not None and ((route.route_info() or {}).get("settings") or {}).get(
            "echo_reasoning") is not None:
        raise ValueError(ECHO_WITH_REASONING_CALL)
    if gold_rows is not None and rules is not None:
        require_frozen_scorer()  # before any request: an attempt that cannot be scored is not run
    cond = get_condition(config.condition)
    if not getattr(sparql, "replays_infra_retries", False):  # a replay serves the recorded retries instead
        sparql = RetryingSparql(sparql, limit_s=config.infra_retry_s, sleep=sleep)
    settings = sparql.settings()
    row_cap = settings.get("row_cap")
    timeout_s = settings.get("timeout_s")
    block = load_block(config.ontology_block_sha256) if config.ontology_block_sha256 is not None else None
    system = cond.render(row_cap=row_cap, timeout_s=timeout_s, ontology_block=block)
    tools = agent_tools(config.max_result_chars)
    raw_cap = max(config.raw_row_cap, row_cap)
    store_info = {**sanitize_store(store), "sparql_timeout_s": timeout_s, "row_cap": row_cap}
    prompt_info = {**cond.record(system, block), "tools_sha256": tools_sha256(tools)}

    att = _Attempt([{"role": "user", "kind": "question", "text": user_message(question)}], on_start=on_start)

    started_at = utc_now()
    t0 = clock()
    phase = {"route": route, "sparql": sparql, "cond": cond, "system": system, "tools": tools, "config": config,
             "raw_cap": raw_cap, "clock": clock, "t0": t0, "served_check": served_check}
    outcome, final_rows = _run_phase(att, **phase, phase_start=0.0, max_turns=config.max_turns,
                                     max_wall_s=config.max_wall_s, max_nudges=config.max_nudges)
    checked = None
    if checker is not None:
        checked = _check_and_retry(att, checker, outcome, final_rows, phase, rules, row_cap, branched_from, branch)
        outcome, final_rows = checked.outcome, checked.final_rows
    wall_s = clock() - t0
    ended_at = utc_now()
    _flag_engine_suspect(att, config)
    _finish_outcome(outcome, att)

    record = RunRecord(
        run_id=str(uuid.uuid4()),
        question_id=question_id,
        base_question_id=question.get("base_question_id"),
        template=question.get("template"),
        split=question.get("split"),
        repetition=repetition,
        condition=config.condition,
        model=route.model,
        route=route.route_info(),
        store=store_info,
        prompt=prompt_info,
        agent_config=_agent_config(config),
        harness=harness_commit(),
        started_at=started_at,
        ended_at=ended_at,
        wall_s=wall_s,
        transcript=att.transcript,
        tool_calls=att.calls_log,
        tokens={"totals": token_totals(att.per_turn), "per_turn": att.per_turn},
        outcome=outcome,
        final_rows=final_rows,
        score=None,
        replay_of=replay_of,
        checker=checked.record if checked is not None else None,
    )
    if gold_rows is not None and rules is not None:
        try:
            if outcome["failure_type"] not in UNSCORED_FAILURES:
                record.score = score_outcome(outcome, final_rows, gold_rows, rules)
            if checked is not None:  # the initial answer is scored even when the retry lost its store
                checked.score_initial(gold_rows, rules)
        except ScorerNotFrozen:
            raise  # the scorer changed during the attempt: nothing is scored or kept by another scorer
        except Exception as exc:
            if out_dir is not None:
                record.write(out_dir)
            raise ScoringError(record) from exc
    if out_dir is not None:
        record.write(out_dir)
    return record


@dataclass
class _Checked:
    record: dict  # the record's ``checker`` field
    outcome: dict  # the end-to-end outcome (the retry's, when there was one)
    final_rows: list | None

    def score_initial(self, gold_rows: Any, rules: Any) -> None:
        initial = self.record["initial"]
        if initial["outcome"]["failure_type"] not in UNSCORED_FAILURES:
            initial["score"] = score_outcome(initial["outcome"], initial["final_rows"], gold_rows, rules)


def _check_and_retry(att: _Attempt, checker: Any, outcome: dict, final_rows: list | None, phase: dict,
                     rules: Any, row_cap: int | None, branched_from: str | None, branch: dict | None) -> _Checked:
    """Check the initial submission; on the checker's request, one retry."""
    from harness.checkloop.checked import submissions
    from harness.checkloop.online import CheckInput
    from harness.score import ScoringRules

    _flag_engine_suspect(att, phase["config"])
    initial = {"outcome": _finish_outcome(json.loads(json.dumps(outcome)), att),
               "final_rows": json.loads(json.dumps(final_rows)), "score": None, "submission_index": None,
               "matching": None, "checked_query": None, "checked_query_status": None, "verdict": None, "reports": [],
               "store_log": None}
    retry = None
    if outcome["status"] == "submitted":
        index = submissions(att.calls_log)[-1]
        matching = scoring_rules(rules) if rules is not None else ScoringRules()
        decision = checker.check(CheckInput(att.calls_log, final_rows, matching, row_cap, index))
        initial["submission_index"] = index
        initial.update(decision.initial)
        if decision.retry:
            _answer_skipped_calls(att)
            # the content, closed (entry 126) by the set-aside sentence where the checker adds it; a baseline's decision
            # has no turn of its own
            text = getattr(decision, "turn", None) or decision.message
            att.transcript.append({"role": "user", "kind": decision.message_kind, "text": text})
            started = phase["clock"]() - phase["t0"]
            budget = checker.budget
            outcome, final_rows = _run_phase(att, **phase, phase_start=started, max_turns=budget.max_turns,
                                             max_wall_s=budget.max_wall_s, max_nudges=budget.max_nudges)
            subs = submissions(att.calls_log)
            retry = {"message_kind": decision.message_kind, "message": decision.message, "started_s": started,
                     "turns": att.turn - initial["outcome"]["turns"],
                     "nudges": att.nudges - initial["outcome"]["nudges"],  # nudges are counted per phase
                     "submission_index": subs[-1] if outcome["status"] == "submitted" else None}
    record = {"settings": checker.settings(), "branched_from": branched_from, "initial": initial, "retry": retry}
    if branch is not None:  # a branched arm: its order among the output's arms and its sampling seed
        record["branch"] = branch
    return _Checked(record, outcome, final_rows)


def _engine_flags(call: dict) -> list[str]:
    args = call.get("arguments")
    if call.get("name") != "sparql_query" or not isinstance(args, dict) or not isinstance(args.get("query"), str):
        return []
    return engine_flags(args["query"])


def _agent_config(config: AgentConfig) -> dict:
    """The config as the record holds it; engine_suspect_classes only when set (as a list), stream_guards as a list,
    and the ontology block's hash only under an ontology condition (AgentConfig.to_json), so records of stores without
    suspect classes and of the "none" condition are unchanged."""
    out = config.to_json()
    classes = out.pop("engine_suspect_classes")
    out["stream_guards"] = list(out["stream_guards"])
    if classes:
        out["engine_suspect_classes"] = list(classes)
    return out


def _engine_suspect(call: dict, classes) -> bool:
    args = call.get("arguments")
    if call.get("name") != "sparql_query" or not isinstance(args, dict) or not isinstance(args.get("query"), str):
        return False
    return suspect(args["query"], classes)


def _call_entry(index: int, turn: int, call: dict, *, status: str, error: dict | None) -> dict:
    entry = {
        "index": index,
        "turn": turn,
        "id": call.get("id"),
        "name": call.get("name"),
        "arguments": call.get("arguments"),
        "engine_flags": _engine_flags(call),  # query shapes the pinned engine gets wrong (engine_flags.py)
        "status": status,  # executed | refused | skipped | malformed | truncated | crashed | infrastructure
        "result_shown": None,
        "is_error": error is not None,
        "result_raw": None,
        "result_raw_truncated": False,
        "result_meta": None,
        "error": error,
        "elapsed_s": 0.0,
        "client_called": False,  # whether the SPARQL client's query() was called
        "terminal": None,
        "infra_retries": [],  # store-outage retries behind this call (infra.py)
        "infra_s": 0.0,
        "query_timings": None,  # the store client's own timing of each request behind this call, if it keeps one
    }
    if call.get("arguments_raw") is not None and (status == "malformed" or call.get("arguments") is None):
        entry["arguments_raw"] = call["arguments_raw"]
    if call.get("source") is not None:  # a call taken from the reasoning channel (agent-run/22): only then
        entry["source"] = call["source"]
    return entry

