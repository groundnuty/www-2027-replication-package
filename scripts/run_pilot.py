#!/usr/bin/env python3
"""Pilot runner: every (question, model, repetition) of one split, resumable.

    uv run python scripts/run_pilot.py --questions questions.jsonl --split dev \\
        --models qwen3.8-27b claude-sonnet-5-5 --reps 3 --order-seed 20261018 \\
        --concurrency 4 --concurrency-for qwen3.8-27b=18 --latency-baseline-s qwen3.8-27b=6.5 \\
        --endpoint http://HOST:PORT/ --store-json store.json --out runs/pilot
        [--condition curated --prompts data/prompts/rhea-142]

Writes one run record per attempt to ``OUT/records/<claim digest>.<run_id>.json``
and one index line per finished attempt to ``OUT/runs.jsonl``. Every file written
for an attempt (records/, infrastructure/, aborted/) is named by its attempt's
claim digest, and the attempt's marker ``OUT/claims/<claim digest>.attempt``
(its key and split, written before its first try and never deleted) names its
split, so each file can be placed in its split without being opened (design log).
Files of earlier runs are named ``<run_id>.json``; every reader takes both.

Resuming. The command can be re-run into the same OUT; an attempt that already
has a record is skipped. Three guards keep a resumed run one experiment:

* ``OUT/manifest.json`` fingerprints the run on first start (question file
  content, store description, endpoint, row cap, SPARQL time-out, agent config,
  prompt condition and each model's route settings). A later start must match
  it exactly or is refused with the differences listed. A model not yet in the
  manifest may be added; a model already in it must keep its route settings.
* ``OUT/.lock`` is held (flock, non-blocking) for the whole run; a second
  runner on the same OUT is refused. OUT must be on a local file system and
  used from one host at a time: flock is not reliable across hosts on network
  file systems, and the claims below are only a secondary guard there. The
  runner warns when OUT is on NFS, shared FS, CIFS/SMB or Lustre.
* Each attempt is claimed with ``OUT/claims/<key>.claim`` (created exclusively)
  before it starts, and the claim is kept after its record is written, so no
  runner can take the attempt again even where flock does not reach (a network
  file system that locks per node). A claim is a lease (pid, host, heartbeat):
  a running attempt refreshes its heartbeat every ``--claim-heartbeat-s``. A
  claim without a record may stand for an attempt that was paid for, so it is
  skipped until a restart with ``--reclaim``, which deletes only claims that
  have no record and whose heartbeat is older than ``--claim-timeout-s`` or
  whose process is gone (on this host); a live attempt's lease is never taken.
  Claims of attempts moved to ``OUT/aborted/`` are deleted, so those can be
  retried; their markers stay. A claim also records the attempt's split.

Order (the design log). Every attempt of the run, (model, arm, question,
repetition), is ordered by the sha256 of the seed and the attempt (``order_key``),
a seeded random order in which server load never correlates with an arm, a
question or a repetition. Because each attempt's key depends on nothing else,
adding repetitions or models never moves the attempts already planned.

Each model has its own stream and its own workers: comparisons are within a
model, and within each model's stream the arms, questions and repetitions are
interleaved in that order. A model's attempts start strictly in the order (with
warm-up, the first attempt of each arm starts first): a worker waits until its
attempt is the model's next, and the next is let through only when this one is
at its first provider request (``StartGate``, run_attempt's ``on_start``), so a
worker paused anywhere before its request holds the next back, however many
workers there are. The models' streams run side by side, so the global order is
not kept across models. What actually happened is recorded per attempt in
``load``: ``start_index`` (among every attempt this runner started, taken at the
first provider request), ``model_start_index`` (among its model's) and
``runner_started_at`` (which start of the runner); an attempt that never made a
provider request has no start index.

``--order-seed`` is required when the run has more than one model (or arm); for
one model it defaults to the seed recorded in the manifest, else to one derived
from the question file's sha256. The manifest records the seed and rule and,
per start whose order differs, the sha256 of the order and of the set of
attempts (outside the fingerprint, since a resume may add repetitions or
models); ``OUT/order.json`` holds the order itself. A resume rebuilds the order,
skips what is recorded, and is refused with another seed or rule, or when the
attempts it shares with the recorded order come out in another relative order.
The arm is the prompt condition, one per run without the grid; a grid run has
one arm per cell (below).

Warm-up. With ``--warmup`` (the default for Anthropic routes, off for
OpenAI-compatible ones) a model's first pending attempt of each arm is its
warm-up: it starts at its own place in the order and runs to completion before
the model's next attempt starts, so the arm's prompt prefix is written to the
cache once and read by every later attempt of the arm. Attempts that started
before it go on (their prefix is another arm's). No attempt is moved, so the
recorded starts keep the seeded order (review of a2ab839: pulling the warm-ups
to the front let a later arm start first).

Load. Every record the runner writes carries ``load`` (record.py, agent-run/6
and /10): the arm, the position in this start's order and the attempt's order
key, its actual start indices (above), the model's configured and allowed workers,
the attempts in flight (this one included) for the model and in total at the
try's start and end, and the try's UTC start and end. Replay does not check it.

Concurrency and guard rails. ``--concurrency`` is the configured number of
attempts in flight per model; ``--concurrency-for MODEL=N`` overrides it for one
model (the Anthropic route keeps its own limit this way). The runner steps a
model down by a third (at least 1) when any of three triggers fires over the
tries or attempts that started at its current size (``GuardRails``; science
ruling in the design log):

* latency: the median per-turn call time (the request with its retries and
  backoff, the record's per-turn ``wall_s``) over every turn of every try of the
  last 50 attempts is over twice the baseline (``--latency-baseline-s
  MODEL=SECONDS``; with ``--latency-baseline-calm MODEL`` the median of the
  model's first 50 calm calls of this start, outside the Gateway sync minutes and
  the all-model stall bins, and inactive until then (``CalmBaseline``; science
  ruling, design log, 21 Oct 15:43); else the median of the first 50 attempts);
* wall-cap rate: of the last 200 attempts, the share capped by the wall clock
  is more than one percentage point over ``--wallcap-baseline MODEL=RATE`` (the
  model's rate on the tier at low concurrency); inactive, and logged so,
  without it;
* infrastructure: more than 2% of the last 50 attempts had a try that failed
  infrastructure (store or provider), even when a re-run then succeeded.

Turn caps, context caps (``capped:context``) and truncated replies are the
agent's behaviour and never trigger a step-down; nor are they re-run. The runner never steps up. Running attempts finish; new ones wait for
a free slot. Every start, baseline and step-down (with the triggers that fired
and the window's figures) is logged and appended to
``OUT/concurrency-events.jsonl``; a resume starts at the last stepped-down size
unless the configured size changed or ``--reset-concurrency`` is given.

Replay check. At the end of every run the whole OUT is checked
(harness/agent/replay_check.py): every record must load, write back byte for
byte and replay. The result is written to ``OUT/replay-check.json``, which the
pilot summary requires.

Each finished attempt also reports its cache hit rate and an estimated cost in
the index, and a warning when any turn was answered by a model other than the
one requested (a silent fallback; the record marks the turns).

An attempt that failed because the route is misconfigured (the provider
answered 401/403/404) is written to ``OUT/aborted/`` instead, does not count as
done, and stops that model's remaining attempts.

Store outages and provider overload. A try that failed ``infrastructure`` is
written to ``OUT/infrastructure/``; its outcome's ``infra_source`` says whether
the SPARQL store stayed down past ``--infra-retry-s`` or the model provider
failed past its own retries (429, 5xx, 529, time-outs, connection errors;
providers.py). After a store outage the runner probes until the store answers
(again up to ``--infra-retry-s``); after a provider failure it waits
``PROVIDER_RERUN_WAIT_S`` (60 s) without probing the store. Either way it
re-runs the attempt, up to ``--infra-reruns`` times, holding the attempt's slot.
Non-transient provider errors (400, 401/403, 404, context overflow) are not
re-run: they stay ``provider_error`` (or abort the model, below).

If every try fails, the last one is recorded in ``OUT/records/`` as
``failed:infrastructure`` (unscored, outside N_att, reported per model with its
rate, and by source in the index's ``infra_source``). If the store does
not come back, the claim is released and the model stops; so does
``--max-infra-failures`` consecutive attempts failing on every try. The
record's ``runner.infra_tries`` and the index's ``infra_tries`` count the tries
of an attempt that failed infrastructure.

A store read through a query gate (harness/gated_store.py; design log, design ruling of 20 Oct). A store
description that names a Virtuoso engine has ``private_files``: the paths of the private credential, graph, row-limit
and gate files. Such a store is never read with the plain shared client: every try gets a client of its own
(``VirtuosoClient.from_files``), which sends the data graph as default and named graph and Virtuoso's millisecond
time-out of twice ``--sparql-timeout`` with every request, and whose requests wait their turn at the one gate file all
clients of the store share (``--gate-wait-s`` at most; past that the attempt fails infrastructure, never a tool error
the model sees). One query at a time therefore holds across the runner's threads and every other process using the
gate, and after a request without a whole answer the client waits, still holding the gate, until the server time-out
has passed. The waits are infrastructure time, recorded per query (agent-run/20) and outside the wall cap; an attempt
whose whole time reaches three times its wall cap while its own time is under the cap fails infrastructure
(``total_cap_factor``) and is re-run. The
client's settings are pinned in the manifest (``store_client``), the store's connection count is read before and after
each try (``load``), and the run ends with the share of its wall time the gate was busy for this runner (held or
queued for; ``OUT/gate-busy.jsonl``). The description, every file it names and OUT must be outside the repository and
owner-only; files are created mode 600, and the runner's messages, which can carry the store's content, go to
``OUT/runner.log`` (mode 600) while its own output gets counts and timings only. ``--preflight-log LOG`` sends only
the store's triple count through the same client and gate, appends it to LOG (mode 600) and exits.

A description's ``query_gate`` (harness/gate_step_down.py) gives its gate several slots: that many queries at once,
stepping down to the next lower size of its ``step_down`` when, over the last 100 queries at the current size, the
share that timed out is above its limit (absolute, ``timeout_limit``; or max(2 x its time-out baseline, the baseline
+ 1 point)); never up: a start begins at the lowest size any earlier start of the OUT reached, whatever its setting.
The setting is not fingerprinted (``gated.public_description``): each start records it, the size it starts at and the
rule in the manifest's ``starts`` entry and in ``OUT/gate-events.jsonl``; each step-down, with its time and window, goes to that
file, to the manifest's ``query_gate_events`` and to the private log; every query the rule took is logged in
``OUT/gate-queries.jsonl``, from which the events can be recomputed. One runner at a time runs on such a gate
(``<gate>.runner.lock``). A description's ``query_timeout`` sets the clients' time-outs (gated_store.py); the run's
``--sparql-timeout`` must equal its client time-out.

Transport. The LLM gateway routes that ``MODELS`` marks ``streaming=True`` send
each model call as a stream: the same request plus ``stream: true`` (providers.py,
module docstring) and ``stream_options: {"include_usage": true}``, recorded per
record in ``route.settings`` (``streaming``, ``stream_options``), and those it
marks with a stream guard reject a stream that lost events (providers.py;
``stream_guards`` in the settings). The transport and the guards are part of
the manifest's route settings, so an OUT never resumes across them.

Caps. Each model's entry in ``MODELS`` states its caps: the characters of one
query result the model is shown and its output tokens per call (design ruling
in the design log). No option sets them, so every arm and cohort of a model runs with
the same; they are in every record's ``agent_config`` and in the manifest's
entry for the model (``routes.<model>.agent_config``, with the tools hash they
give). The calibration rounds 1 and 2 (results/cal-rhea-r1-20261018,
cal-uniprot-r1-20261018, cal-uniprot-round2-20261019 and the Rhea round 2: Qwen
and Sonnet) ran non-streamed: against a later streamed run of the same model,
that is a transport difference with identical requests (the same body but for
``stream`` and ``stream_options``), not a change of what the model was sent.

Keys are read from the environment by the routes and never printed:
ANTHROPIC_API_KEY for Claude, LLM_GATEWAY_API_KEY (or its alias
CENTREB_API_KEY) for the HPCNet LLM Gateway models. Every log line passes
through the record writer's secret redaction; a variable counts as a secret when
its name contains KEY, TOKEN, SECRET or PASSWORD and its value is at least 16
characters and not only digits. The run logs those variables' names once, never
their values.

The development grid (E1.5a, the design log). With ``--grid-manifest M
--graph G --phase natural|prompted|branch`` the runner runs one graph's phase of
the grid that scripts/build_dev_grid.py planned, every model and cell of the
phase in one seeded order, into one OUT per graph: three registered phases in one
output directory (natural: the none cohort; prompted: the prompted cohorts,
interleaved among themselves; branch: every checker arm, its records beside their
parents); the launch design is that script's docstring, items 1-7:

* Gates first. Any split but dev needs the frozen pre-registration
  (build_dev_grid.split_gate: the exact line "Status: frozen" at a commit on the
  science repository's origin/main, fetched now), checked before anything else
  is read. The spec file must be the one the manifest was built from (its
  sha256), the manifest is rebuilt from it with every input of the phase checked
  against its hash (``--for-launch --phase``: natural needs only the question
  file, the store description and the settings, so it launches while prompts and
  sources are FILL), and the attempts file must match both.
* Everything else comes from the manifest: models (each route must serve the
  spec's route model), questions (development only in the dev grid), repetitions,
  the phase's seed and order, store description and the endpoint it names (one
  store per graph for every phase), row cap and time-out, each model's workers
  (``runner_settings.concurrency``) and, in the branch phase, the retry's nudges and
  one submission, the switches, feedback format and generic text (refused unless
  the frozen ones and the running checker's). Giving
  any of those flags (``--endpoint`` and ``--concurrency`` included) is refused.
  The Gateway models' workers of the phase must sum within
  ``runner_settings.gateway_budget`` (Anthropic routes outside it); each start's
  workers and budget, and the code it runs on, are appended to the manifest's
  ``starts``. Each model runs its own cells and repetitions per phase (the
  manifest's ``model_cells`` and ``model_repetitions``). A model the spec's
  ``runner_settings.request_timeout_s`` or ``runner_settings.max_turns`` names
  (change-log entry 180) runs on its route rebuilt with that request time-out,
  or with that turn cap, as the manifest recorded them.
* The code (dress rehearsal F1): every run, grid or not, starts only on code it
  can name (``replay_check.require_identified_code``): a git checkout, or a deploy
  whose COMMIT and TREE files hold and whose tree digest recomputed from disk is
  its TREE. Every record and the manifest carry ``harness`` = {git_commit,
  git_dirty, source, tree_sha256}. A grid start also needs that code clean and at
  the spec's ``code.harness_commit`` and ``code.compiler_commit`` (each compared
  once filled; the branch phase needs both: ``check_running_code``).
* Per model, never grid-wide: its turn, wall, output-token and tool-result caps
  (the last per graph) and its transport and stream guards come from this runner's
  model table (``MODELS``, ``model_caps``; design log, entries 90, 111 and 119: 120
  turns and 3,600 s, GLM's 240 and 7,200 s), as the grid manifest recorded them per graph
  (``model_settings``); a route built otherwise is refused, each record carries
  its own, and the run pins each model's caps, tools hash and, in the branch
  phase, its retry budget (its own turn and wall caps: the retry gets the
  attempt's own, pre-registration section 3).
* The order: the phase's attempts from ``attempts.jsonl.gz``; the runner
  recomputes every phase's order (per role, its models crossed with its cells,
  merged by ``order_key``) and refuses unless it is the manifest's, by the phase's
  ``order_sha256``. The start gate keeps each model's starts in that order, so
  the arms are interleaved within the phase.
* The arm is in the attempt key: (arm, question, model, repetition), in
  ``existing_keys`` (``load.arm``), the claims, the index line (``arm``) and the
  summary (per model and arm). A run without the grid keeps the key (question,
  model, repetition), so its claims do not move.
* Initial cells (the natural and prompted cohorts) run with an AgentConfig each:
  the model's caps, the cohort's condition and, for an ontology cohort, its
  block by hash (an AgentConfig without ``ontology_block_sha256``, as before issue
  16, refuses such a cohort rather than run it unprompted). Each has its warm-up
  (``--warmup``, above: at its place in the order).
* Branch cells run ``branch_attempt`` on the parent record of (parent cohort,
  question, model, repetition) in the same OUT (none for the fb and gen cells,
  census for the deployment row), with ``OnlineChecker(Arm(cell arm, the
  compilers of the cell's sources for the model's role), store, mode,
  RetryBudget(the parent's turn and wall caps, 2 nudges))``; a generic cell's sources follow
  the role (the manifest's cells). A parent that did not submit has nothing to
  check: no record, counted ``inherited``. A branch attempt starts at its retry's
  first model request; one that does not flag makes none. No warm-up
  (``--warmup`` is refused): the retry continues the parent's transcript. The
  phase starts only when phases natural and prompted are complete and replayed:
  every planned attempt of both, of every model, has a record that did not fail
  infrastructure (a crashed attempt, its claim kept, is not; an attempt that failed
  infrastructure on every try is set aside with relabel_harness_failure.py
  ``--arm`` and run again in its phase), and OUT's replay check passed on the
  records as they are now. A recorded branch must have been branched from the
  record now in OUT for its parent: one whose parent was set aside and re-run
  since is refused (set it aside too, and this phase branches it again).
* ``OUT/manifest.json`` pins the graph's grid per phase (``check_grid_manifest``):
  the shared settings (question file, store and endpoint, row cap, time-out, the
  base agent config, split, repetitions, freeze) and each model's route, checked
  at every start; per phase started so far, its seed, cells, inputs (prompted: its
  prompts manifest, blocks and sources; branch: the hierarchy, its sources, the
  code hashes), arms (an initial cell's prompt hashes; a branch cell's
  ``OnlineChecker.settings()`` per role) and each model's role and own order.
  Every start fingerprints every pinned phase again, so a later phase cannot change
  an earlier one's arms; once a phase has a record, its seed, cells, inputs and
  arms are fixed (an arm may join only a phase with none). A model may join while
  it has no record in the run (the DeepSeek panel model): every other model's own
  order is unchanged, since a key depends only on the seed and the attempt. The
  spec's sha256 is not pinned: the spec gains later phases' inputs, the workers
  and models after a run starts. Each phase records its order in the manifest's
  ``grid_order`` and ``OUT/order-<phase>.json``.

Code faults (design ruling in the design log, change-log entry 91). A refusal is an
agent outcome, ``refused`` (loop.py), recorded and scored wrong like any other.
A crash inside our own tool code ends the attempt ``failed:harness_defect``:
the runner sets it aside at once as relabel_harness_failure.py would (the
record and a setaside/1 sidecar, label ``failed:harness``, in OUT/harness/,
no index line, the claim released), so a resume re-runs it once the defect is
fixed. Each model's first served model id in an OUT is the id of its first
served turn, kept in OUT/served-models.json (written under a lock before any
other turn of the model is checked, kept across resumes). Every turn is checked
as it arrives: a turn served by another id (any difference, a dated snapshot
included) ends its attempt at once ``failed:model_changed`` (loop.py), which
is written unscored to OUT/aborted/ with ``runner.aborted`` (model_changed),
its claim released; no new attempt of the model starts from then on (attempts
already running go on, each checked turn by turn), and the change is logged
and appended to OUT/MODEL_CHANGED.json. A redeploy or move of a model before the
freeze is a recorded deployment change (design ruling in the design log Oct
23:24): a start with ``--deployment-change MODEL --ruling REF`` takes the one
system fingerprint recorded there since the model's baseline, its served id
unchanged, as the new baseline, records it in served-models.json and the
manifest's ``deployment_changes`` and holds its turns to it; the attempts
aborted at the change re-run, as every aborted attempt does (its claim is
released). After the freeze (an OUT whose pins hold it: a test grid) the model
does not continue in place: the change is a served-version change of it
(change-log entry 129, item 3), recorded in the manifest's
served_version_changes and held through OUT/holds/<model>.served-version.json,
with an alert, as a per-model setting changed after the freeze is; its
served-models.json baseline stays, and the other models go on. The scorer must be the frozen one: its
digest (harness/score/digest.py, loop.FROZEN_SCORER_DIGEST) is checked before
anything is written and recorded in the manifest.

Exit status: 0 all scheduled attempts recorded (attempts skipped for a stale
claim are reported in the log; none awaits a checker fix); 1 some attempts
crashed in the harness (logged, claim kept, no record), or were set aside as
harness defects; 2 refused before starting (unknown model, missing key, no
questions, not the frozen scorer, no tool-result cap for the graph, manifest
mismatch, duplicate records, OUT locked) or stopped by an unexpected error; 3 a
model was stopped (misconfigured, its served model changed, the store did not
come back, too many consecutive attempts failed infrastructure on every try, or
a hold holds it: OUT/holds/<model>.json from the infrastructure-fault monitor or
<model>.served-version.json for a setting changed after the freeze, change-log
entry 129, read before each claim of the model); 4 some attempts were recorded as failed:infrastructure after all re-runs;
5 the replay check found records that do not load, write back or replay, or a
decision that diverges, or it HALTS a phase, or a periodic check halted the
branch phase while it ran (after 3, before 1, 6 and 4); 6 every scheduled
attempt was run except those awaiting a checker fix (set aside as checker
divergences with no fix recorded: none is run before its fix), so the run is
not complete (after 1, before 4: a launcher must not take 6 for done).

A HALTED phase (design ruling, design log, 20 Oct 07:26: more than 1% of its
records' online checker decisions diverge from the offline recompute, or
divergences occur in more than one model; harness/agent/replay_check.py) starts
no attempt until the cause is fixed: scripts/relabel_harness_failure.py
--checker-fix then sets its divergent records aside with the fix recorded, and a
later start of the phase re-runs them only under the fixed code (a grid phase's
pinned code commits move to it once, recorded in the manifest's code_changes).
A record set aside one by one on its divergence below the halt waits the same
way: its claim is kept and no start runs or reclaims its attempt until a fix is
recorded for its phase (--checker-fix), then it is re-run under that fix.

While a grid's branch phase runs, its checker decisions completed so far are
recomputed in a fresh interpreter every --periodic-check-s (2 hours by
default; niced, and skipped while the previous check runs or the CPU is
short). Divergences above the halt threshold stop new launches at once: the
attempts in flight finish and are recorded, an attempt not yet at its first
provider request is not launched (no record; its claim is released), the halt
is recorded in the manifest's periodic_checks (and stands in every replay check
until a checker fix of the phase), and the run exits 5. Each check's time,
record count and result is recorded there; it prints counts only, never a score.
"""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import gzip
import hashlib
import importlib.util
import json
import math
import os
import re
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections import Counter, deque
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable
from urllib.parse import urlsplit

# Allow running as a plain script from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.agent.cost import cache_hit_rate, estimate_record_cost  # noqa: E402
from harness.agent.errors import HarnessAbort  # noqa: E402
from harness.agent.infra import wait_for_store  # noqa: E402
from harness.agent.infra_fault import hold_path, read_hold, served_version_hold_path  # noqa: E402
from harness.agent.loop import (  # noqa: E402
    ECHO_WITH_REASONING_CALL,
    FROZEN_SCORER_DIGEST,  # noqa: F401  (the runner's frozen digest, read as run_pilot.FROZEN_SCORER_DIGEST)
    RULE_FIELDS,
    UNSCORED_FAILURES,
    AgentConfig,
    ScorerNotFrozen,
    ScoringError,
    require_frozen_scorer,
    run_attempt,
    token_totals,
    tools_sha256,
)
from harness.agent.metadata import sanitize_store  # noqa: E402
from harness.agent.prompts import CONDITIONS, add_block_store, get_condition, load_block, sha256_text  # noqa: E402
from harness.agent.providers import (  # noqa: E402
    MIXED_FINGERPRINTS,
    REASONING_CALL_RULES,
    STREAM_GUARDS,
    AnthropicRoute,
    OpenAICompatRoute,
    Provider,
    _public_url,
)
from harness.agent.record import (  # noqa: E402
    RunRecord,
    harness_commit,
    redact_secrets,
    redact_structure,
    secret_names,
    utc_now,
)
from harness.agent.replay_check import (  # noqa: E402
    CHECK_FILE,
    CHECKER_DIVERGENCE,
    PLAIN_PHASE,
    CodeNotIdentified,
    current_records_sha256,
    divergence_so_far,
    halt_message,
    held_message,
    read_progress,
    read_sidecars,
    require_identified_code,
    setaside_lines,
    verify_batch,
    write_check,
)
from harness.engine_suspect import suspect_classes_from_vocab  # noqa: E402
from harness.agent.tools import (  # noqa: E402
    MAX_RESULT_CHARS,
    MIN_RESULT_CHARS,
    REPORT_KEY_RULES,
    agent_tools,
)
from harness import gated_store as gated  # noqa: E402
from harness.checkloop.store_log import CHECKER_ROW_CAP, CHECKER_TIMEOUT_S  # noqa: E402
from harness.gate_step_down import TimeoutStepDown, parse_setting, recompute, rule_json  # noqa: E402
from harness.query_gate import QueryGate  # noqa: E402
from harness.score import question_gold  # noqa: E402

GATEWAY_KEY_ENVS = ("LLM_GATEWAY_API_KEY", "CENTREB_API_KEY")
ANTHROPIC_KEY_ENVS = ("ANTHROPIC_API_KEY",)
CONFIG_ERROR_CLASSES = frozenset({"auth", "not_found"})
QUESTION_FIELDS = ("id", "split", "question")
MANIFEST_SCHEMA = "pilot-manifest/1"
ORDER_RULE = ("sorted by sha256 of '<seed>|' followed by the compact JSON of [model, arm, question id, "
              "repetition]")
EVENTS_FILE = "concurrency-events.jsonl"
GATE_BUSY_FILE = "gate-busy.jsonl"  # a store read through a query gate: each start's busy share of the gate
# a gate of several slots (its description's query_gate): each start and step-down, and every query the rule took
GATE_EVENTS_FILE = "gate-events.jsonl"
GATE_QUERIES_FILE = "gate-queries.jsonl"
# every read-modify-write of OUT/manifest.json made while attempts run (the periodic check, the gate's events)
MANIFEST_LOCK = threading.Lock()
# The guard rails (the design log, design ruling), fixed in advance: not run options.
LATENCY_WINDOW = 50  # attempts per model for the median per-turn call time
LATENCY_FACTOR = 2.0  # ... against the baseline
WALLCAP_WINDOW = 200  # attempts per model for the wall-cap rate
WALLCAP_MARGIN = 0.01  # ... against the baseline rate: more than one percentage point over it
INFRA_WINDOW = 50  # attempts per model for the infrastructure failure rate
INFRA_RATE_LIMIT = 0.02
PROVIDER_RERUN_WAIT_S = 60.0  # pause before re-running an attempt the provider failed (its retries already backed off)
DEFAULT_ARM = AgentConfig().condition  # the one arm of a run without the grid
REPO_ROOT = Path(__file__).resolve().parent.parent
DEV_SPLIT = "dev"  # the one split a run may make calls on without the frozen pre-registration (plain_split_gate)
DEFAULT_GRID_SPEC = REPO_ROOT / "data" / "runs" / "dev-grid" / "grid.json"
GRID_PHASES = ("natural", "prompted", "branch")  # three registered phases, one OUT per graph (design log)
INITIAL_PHASES = ("natural", "prompted")  # the cohorts' phases: full agent runs; branch is the checker arms'
GRID_OUT_SCHEMA = "pilot-grid-manifest/1"  # OUT/manifest.json of a grid run: per-phase pins
GATEWAY_PROVIDER = "openai-compatible"  # the HPCNet LLM Gateway routes, which share runner_settings.gateway_budget
# The development grid's frozen settings (pre-registration section 3; design log): the spec must hold exactly these.
# Nothing of the agent's caps is grid-wide: each model's turn, wall, output and tool-result caps are its own
# (``model_caps``, the runner's table; design log, entry 90), and so are its retry's turn and wall caps (the retry gets the
# attempt's own); the spec fixes only the retry's nudges and its one submission.
FROZEN_AGENT_SETTINGS: dict = {}
FROZEN_RETRY_BUDGET = {"max_nudges": 2, "submissions": 1}
# The frozen scorer's digest is loop.FROZEN_SCORER_DIGEST (preregistration §3): the runner refuses any other scorer
# before anything is written (design log, change-log entry 91), and every scoring path checks it again.


def gateway_key_env(environ: dict[str, str] | None = None) -> str:
    """The first of LLM_GATEWAY_API_KEY, CENTREB_API_KEY that is set (the name, not the value)."""
    env = os.environ if environ is None else environ
    for name in GATEWAY_KEY_ENVS:
        if (env.get(name) or "").strip():
            return name
    return GATEWAY_KEY_ENVS[0]


# The agent config's per-model fields (design ruling in the design log, entry 90): the caps a ModelSpec states and the
# transport and stream guards of its route. They are in each record's agent_config and the manifest's entry for the
# model, never in the run's shared agent config.
MODEL_FIELDS = ("max_turns", "max_wall_s", "max_tokens", "max_result_chars", "transport", "stream_guards")
# Each rule a model may set (RULE_FIELDS), as its messages name it
RULE_NAMES = {"report_key_rule": "report-key", "reasoning_call_rule": "reasoning-call"}
assert tuple(RULE_NAMES) == RULE_FIELDS


def rule_choices(field: str) -> tuple[str, ...]:
    """The known values of a rule field, read when called (tools.REPORT_KEY_RULES, providers.REASONING_CALL_RULES)."""
    return {"report_key_rule": REPORT_KEY_RULES, "reasoning_call_rule": REASONING_CALL_RULES}[field]


def route_transport(settings: dict) -> tuple[str, tuple[str, ...]]:
    """A route's transport and the stream guards it has on, from its recorded settings (providers.py)."""
    guards = settings.get("stream_guards") or {}
    return ("streamed" if settings.get("streaming") else "non-streamed",
            tuple(name for name in STREAM_GUARDS if guards.get(name)))


@dataclass(frozen=True)
class ModelSpec:
    """One model of the table: its route and the model's caps (design ruling in the design log, entry 90): its turns
    (``max_turns``) and wall seconds (``max_wall_s``) per attempt, its output tokens per model call (``max_tokens``)
    and the characters of one sparql_query result it is shown (``max_result_chars``, tools.py). The caps, and its
    route's transport and stream guards, go into the agent config of every attempt of the model (``agent_config``),
    so into each record's agent_config and the manifest's entry for the model; no run option sets any of them, so
    every arm, cohort and graph of a model has the same. The defaults are the agent config's own (for test tables);
    every entry of ``MODELS`` states its caps. The rules (RULE_FIELDS, each None unless the model's entry sets it):
    ``report_key_rule``, how submit_report reads a cell object's strings (tools.REPORT_KEY_RULES; change-log entries
    110 and 112: Gemma's); ``reasoning_call_rule``, whether a call written into the reasoning channel runs (providers.
    REASONING_CALL_RULES: Qwen3.6's; design rulings in the design log Oct 00:09 and 01:31). Each is a setting like
    the caps, recorded only when set (the agent config's, the manifest's entry for the model and the grid manifest's
    model settings), so every other model's records and pins are what they were. A run whose earlier phases pinned a
    model otherwise (another rule, other caps) takes the table's values once, as the recorded move ``PIN_MOVES``
    declares (check_grid_manifest)."""

    factory: Callable[[], Provider]
    key_envs: tuple[str, ...] = ()  # at least one must be set; empty means no key is needed
    # one cap for every graph, or {graph: cap} with no default: a run on a graph it lacks is refused (design log)
    max_result_chars: int | Mapping[str, int] = MAX_RESULT_CHARS
    max_tokens: int = 16_000
    max_turns: int = AgentConfig.max_turns
    max_wall_s: float = AgentConfig.max_wall_s
    report_key_rule: str | None = None
    reasoning_call_rule: str | None = None

    def __post_init__(self) -> None:
        def integer(name: str, value: Any, low: int) -> None:
            if not isinstance(value, int) or isinstance(value, bool) or value < low:
                raise ValueError(f"{name} must be an integer of at least {low}, got {value!r}")

        caps = self.max_result_chars
        if isinstance(caps, Mapping):
            if not caps or not all(isinstance(g, str) and g for g in caps):
                raise ValueError(f"max_result_chars per graph needs at least one graph name, got {caps!r}")
            for graph, cap in caps.items():
                integer(f"max_result_chars[{graph!r}]", cap, MIN_RESULT_CHARS)
            object.__setattr__(self, "max_result_chars", MappingProxyType(dict(caps)))  # read-only from here
        else:
            integer("max_result_chars", caps, MIN_RESULT_CHARS)
        integer("max_tokens", self.max_tokens, 1)
        integer("max_turns", self.max_turns, 0)
        if not isinstance(self.max_wall_s, (int, float)) or isinstance(self.max_wall_s, bool) or self.max_wall_s <= 0:
            raise ValueError(f"max_wall_s must be a positive number of seconds, got {self.max_wall_s!r}")
        for field, what in RULE_NAMES.items():
            if getattr(self, field) is not None and getattr(self, field) not in rule_choices(field):
                raise ValueError(f"unknown {what} rule {getattr(self, field)!r}; known: "
                                 f"{', '.join(rule_choices(field))}")

    def rules(self) -> dict[str, str]:
        """The rules this model sets (RULE_FIELDS), by field; none for most models."""
        return {field: getattr(self, field) for field in RULE_FIELDS if getattr(self, field) is not None}

    def result_chars_for(self, graph: str | None) -> int:
        """The tool-result cap on ``graph``: the one cap, or the graph's own (KeyError when the mapping lacks it)."""
        caps = self.max_result_chars
        if not isinstance(caps, Mapping):
            return caps
        if graph not in caps:
            raise KeyError(f"no tool-result cap for graph {graph!r} (caps for {', '.join(sorted(caps))})")
        return caps[graph]

    def agent_config(self, base: AgentConfig, route_settings: dict) -> AgentConfig:
        """``base`` (the run's shared settings, its graph included) with this model's caps on that graph and its
        route's transport and guards."""
        transport, guards = route_transport(route_settings)
        if self.reasoning_call_rule is not None and route_settings.get("echo_reasoning") is not None:
            raise ValueError(ECHO_WITH_REASONING_CALL)
        return replace(base, max_turns=self.max_turns, max_wall_s=float(self.max_wall_s), max_tokens=self.max_tokens,
                       max_result_chars=self.result_chars_for(base.graph), transport=transport, stream_guards=guards,
                       **{field: getattr(self, field) for field in RULE_FIELDS})


def model_caps(spec: ModelSpec, graph: str | None) -> dict:
    """A model's four caps on ``graph`` (the runner's table; design log, entry 90): turns, wall seconds, output tokens per
    call and the tool-result characters resolved for the graph (KeyError when the model has no cap for it). A grid
    run takes every one of them from here, per model, never from a grid-wide setting; its retry gets the same turn
    and wall caps (pre-registration section 3)."""
    return {"max_turns": spec.max_turns, "max_wall_s": float(spec.max_wall_s), "max_tokens": spec.max_tokens,
            "max_result_chars": spec.result_chars_for(graph)}


def model_settings(name: str, spec: ModelSpec, graphs: list[str]) -> dict:
    """What the grid manifest records of a model, from the runner's table: its route model, provider and route
    settings (the transport: streaming, its stream options and guards, parallel tool calls ...) and its caps on each
    of ``graphs`` (``model_caps``; KeyError when it has no tool-result cap for one)."""
    route = spec.factory()
    info = route.route_info()
    out = {"route_model": route.model, "provider": info.get("provider"), "route_settings": info.get("settings"),
           "caps": {graph: model_caps(spec, graph) for graph in graphs}}
    out.update(spec.rules())  # only those set: every other model's settings are what they were
    return out


# --- a stack's own request time-out and turn cap per model (change-log entry 180) ---------------------------------
# Science, the design log: 18:54Z, option (b), a grid spec may give a model its own request time-out,
# runner_settings.request_timeout_s {model: seconds}, a recorded runner setting of the stack the spec runs on; 19:05Z,
# by the same mechanism its own turn cap, runner_settings.max_turns {model: turns}. The runner (_run_store) and the
# grid build (build_dev_grid's model_settings) both take the model's entry through grid_models, so the manifest's route
# settings and caps, the OUT's pins and every record carry the same values (settings.request_timeout_s, 600 s by
# default, and the caps' max_turns are recorded there today). A plain run (the only path of a store read through a query
# gate) takes the same object from --runner-settings FILE (plain_runner_settings, through grid_models too): its
# manifest pins each model's route and caps, so a resume under other settings is refused. A model the settings do not
# name, and every model of a run without them, keeps its table entry exactly as before.
REQUEST_TIMEOUT_KEY = "request_timeout_s"
TURN_CAP_KEY = "max_turns"
RUNNER_MODEL_KEYS = (REQUEST_TIMEOUT_KEY, TURN_CAP_KEY)


def _positive_seconds(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        seconds = float(value)  # a JSON integer beyond a float's range is no number of seconds
    except OverflowError:
        return False
    return math.isfinite(seconds) and seconds > 0


def _positive_turns(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def runner_model_problems(spec: Mapping[str, Any], models: Mapping[str, ModelSpec], *,
                          scope: str = "the grid") -> list[str]:
    """What is wrong with ``spec``'s per-model runner settings (nothing when neither key is present): each must map at
    least one model of the grid (``spec["models"]``) of the runner's table to its value, a positive, finite number of
    seconds (``request_timeout_s``) or a positive whole number of turns (``max_turns``); a model given a request
    time-out must have a route rebuilt with another one (an OpenAI-compatible route, by ``with_options``)."""
    runner = spec.get("runner_settings")
    if not isinstance(runner, Mapping):
        return []
    grid = {m.get("name") for m in spec.get("models") or [] if isinstance(m, Mapping)}
    problems = []
    for key, valid, what in ((REQUEST_TIMEOUT_KEY, _positive_seconds, "a positive, finite number of seconds"),
                             (TURN_CAP_KEY, _positive_turns, "a positive whole number of turns")):
        if key not in runner:
            continue
        where = f"runner_settings.{key}"
        values = runner[key]
        if not isinstance(values, Mapping) or not values:
            problems.append(f"{where}: {values!r} is not a mapping of at least one model to its value (leave the key "
                            f"out for every model's own, from the runner's table)")
            continue
        for name, value in values.items():
            if not valid(value):
                problems.append(f"{where}.{name}: {value!r} is not {what}")
            if name not in grid:
                problems.append(f"{where}.{name}: not a model of {scope}")
            elif name not in models:
                problems.append(f"{where}.{name}: not a model of the runner's table")
            elif key == REQUEST_TIMEOUT_KEY and not takes_front(models[name].factory()):
                # takes_front's test is this one: an OpenAI-compatible route that with_options rebuilds
                problems.append(f"{where}.{name}: its route cannot be rebuilt with another request time-out (only an "
                                f"OpenAI-compatible route, by with_options, can)")
    return problems


def _timeout_factory(factory: Callable[[], Provider], seconds: float) -> Callable[[], Provider]:
    def build() -> Provider:
        route = factory()
        rebuilt = route.with_options(request_timeout_s=seconds)
        own, there = route.route_info(), rebuilt.route_info()
        want = {**(own.get("settings") or {}), REQUEST_TIMEOUT_KEY: seconds}
        if ({k: own.get(k) for k in ("provider", "base_url", "model")}
                != {k: there.get(k) for k in ("provider", "base_url", "model")}
                or there.get("settings") != want or getattr(rebuilt, REQUEST_TIMEOUT_KEY, None) != seconds):
            raise ValueError(f"the route rebuilt with request_timeout_s {seconds} differs from the table's route in more "
                             f"than its request time-out")
        return rebuilt
    return build


def grid_models(models: Mapping[str, ModelSpec], spec: Mapping[str, Any], *,
                scope: str = "the grid") -> Mapping[str, ModelSpec]:
    """``models`` with each model ``spec``'s per-model runner settings name given them: its route built with its
    request time-out (a float, like the 600.0 default) and its turn cap; ``models`` itself, the same object, when the
    spec sets neither. ValueError with ``runner_model_problems``'s findings."""
    runner = spec.get("runner_settings")
    if not isinstance(runner, Mapping) or not any(key in runner for key in RUNNER_MODEL_KEYS):
        return models
    problems = runner_model_problems(spec, models, scope=scope)
    if problems:
        raise ValueError("; ".join(problems))
    out = dict(models)
    for name, seconds in (runner.get(REQUEST_TIMEOUT_KEY) or {}).items():
        out[name] = replace(out[name], factory=_timeout_factory(out[name].factory, float(seconds)))
    for name, turns in (runner.get(TURN_CAP_KEY) or {}).items():
        out[name] = replace(out[name], max_turns=turns)
    return out


def plain_runner_settings(path: Path, names: list[str], models: Mapping[str, ModelSpec]
                          ) -> tuple[Mapping[str, ModelSpec], dict]:
    """A plain run's per-model settings (``--runner-settings``): the JSON object in ``path``, holding
    ``request_timeout_s`` and/or ``max_turns`` exactly as a grid spec's runner_settings does, for models of the run
    (``names``, its --models); -> (the table through ``grid_models``, the object). Refused when it cannot be read, is
    not such an object (no other key), or ``runner_model_problems`` finds anything."""
    def unique(pairs: list[tuple[str, Any]]) -> dict:
        keys = [k for k, _ in pairs]
        twice = sorted({k for k in keys if keys.count(k) > 1})
        if twice:  # json.loads would keep the last silently, and the run would record only that one
            raise ValueError(f"a key given twice in one object: {', '.join(twice)}")
        return dict(pairs)

    try:
        settings = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise Refused(f"--runner-settings {path}: {exc}") from None
    if not isinstance(settings, dict) or not settings or set(settings) - set(RUNNER_MODEL_KEYS):
        raise Refused(f"--runner-settings {path}: a JSON object with {' and/or '.join(RUNNER_MODEL_KEYS)} and nothing "
                      f"else, got {sorted(settings) if isinstance(settings, dict) else type(settings).__name__}")
    spec = {"models": [{"name": n} for n in names], "runner_settings": settings}
    try:
        return grid_models(models, spec, scope="the run (--models)"), settings
    except ValueError as exc:
        raise Refused(f"--runner-settings {path}: {exc}") from None


def say_runner_settings(runner: Any, say: Callable[[str], None]) -> None:
    own = {key: runner[key] for key in RUNNER_MODEL_KEYS if isinstance(runner, Mapping) and key in runner}
    if own:
        say("per-model runner settings (change-log entry 180): " + "; ".join(
            f"{key} " + ", ".join(f"{m} {v}" for m, v in sorted(values.items())) for key, values in own.items()))


def _gateway(model: str, *, max_result_chars: int | Mapping[str, int], max_tokens: int, max_turns: int,
           max_wall_s: float, report_key_rule: str | None = None, reasoning_call_rule: str | None = None,
           **route_options: Any) -> ModelSpec:
    """A HPCNet LLM Gateway (CentreB) route, with the model's caps (``ModelSpec``; stated in every entry). A Llama
    3.3 route needs ``parallel_tool_calls=False``: its chat template takes one tool call per turn
    (tests/test_run_pilot.py checks every route for it).

    ``streaming=True`` sends the request as a stream (the gateway cuts a reply that has not started within
    180 s; www2027-reviews/gateway): the same request plus ``stream: true`` and ``stream_options``
    ``{"include_usage": true}``. The second field is what the protocol needs for a stream to carry its usage;
    the Gateway sends usage on every chunk without it, but the route asks as any conforming endpoint requires
    (streaming review, design log). ``guard_call_identity`` and ``guard_token_gap`` turn on the stream guards G1
    (narrow) and G2 of a streamed route (providers.py; design log)."""
    return ModelSpec(lambda: OpenAICompatRoute(model=model, api_key_env=gateway_key_env(), **route_options),
                     GATEWAY_KEY_ENVS, max_result_chars=max_result_chars, max_tokens=max_tokens, max_turns=max_turns,
                     max_wall_s=max_wall_s, report_key_rule=report_key_rule, reasoning_call_rule=reasoning_call_rule)


# The panel (preregistration §6, owner's approval of 18 Oct; ids from the 18 Oct CentreB catalogue,
# results/centreb-catalogue-20261018): GLM-5.3-Flash, Gemma-4-31B, Llama-3.3-70B (one tool call per turn)
# and Claude Haiku 5.5. One setting per family (design ruling in the design log): every Claude route sends
# CLAUDE_EFFORT explicitly, in the agent runs and the closed-book probes alike (Haiku 5.5's own default
# would be "medium"), with adaptive thinking and the 1-hour prompt cache. Claude Opus 5.5 runs as an agent in
# the natural cohort only (E4.3, the exploratory self-check crossing; not in C5's model set), with the same
# settings as Sonnet.
#
# Transport and stream guards (design rulings in the design log; results/stream-integrity-20261019): one setting
# per model, uniformly across arms, cohorts and graphs, from the start of its runs. Qwen3.8 (until the cap audit
# decides its transport), GLM, Gemma, Qwen3-Coder and Qwen3.6 stream, with both stream guards (G1 narrow: a tool
# call without an id or a function name; G2: the token gap; providers.py). Both DeepSeek models and Llama do not
# stream: DeepSeek's agent turns are short, and Llama's broken streamed calls are its own unreadable JSON, which no
# guard should retry. Bielik does not stream either (below). A model that does not stream must finish each turn
# within the gateway's 180 s to the first byte at its output cap, so every entry states its caps on one line
# (``ModelSpec``; design log, entry 90): turns, wall seconds, output tokens per call (16,000, but 32,768 for Qwen3.8, GLM
# and Qwen3.6 from entry 161) and the
# characters of one tool result. The final values (design rulings in the design log, entries 90 and later; the Qwen3.8 dev
# replay and the cap audit, www2027-reviews/cap-audit/): 120 turns and 3,600 s for every model but GLM, whose caps
# double to 240 and 7,200 s from UniProt dev prompted on (entry 111, its causation replay; its natural records keep 120
# and 3,600 s, a recorded move, PIN_MOVES). Gemma's rise from 60 turns and 1,800 s to the others' 120 and 3,600 s from
# UniProt dev prompted on, with its report-key rule (entry 119; its natural records keep 60 and 1,800 s, the same
# recorded move). From the freeze on (entry 160), 7,200 s for every model on every graph, and 240 turns for Qwen3.8 in
# all its arms (its capped opaque-Rhea dev attempts were progressing and would reach 120 turns before 7,200 s); every
# other turn cap stays. From the freeze on (entry 161), Qwen3.8, GLM and Qwen3.6 take their measured output limit,
# 32,768 tokens per call (GLM's SiteA deploy serves a 1,048,576-token context); every other model keeps 16,000. Test OUTs
# pin these at launch; the dev records keep the caps they ran under. Tool-result caps per
# graph for
# Qwen3.8 (UniProt 224,000, Rhea 80,000, the ruled floor), Gemma (188,000 and 62,000) and Qwen3-Coder (179,000 and
# 65,000), the Rhea values at the lower end of the Qwen3.8 proxy;
# Qwen3.6 (exploratory) as Qwen3.8; the 400,000 guard for the Claude models, GLM and both DeepSeek models (Opus, not
# in the rulings, takes the other Claude models' guard); Llama and Bielik (exploratory) 80,000. On the corporate graph
# (text2sparql-ck26) every ruled model is shown 400,000 characters (change-log entry 94: corporate caps 400k for all
# models), so the per-graph models name it, and Llama's cap is per graph too (80,000 on UniProt and Rhea); Bielik, in
# no grid, keeps its one 80,000. A run on a graph a
# per-graph mapping lacks is refused, and the record holds the graph with the cap resolved for it. A run
# never resumes across transports, guards or caps: they are in the manifest's entry for the model, and every record's
# agent_config holds them all. This is the one model table: every route the runner builds comes from it
# (tests/test_agent_streaming.py).
CLAUDE_EFFORT = "high"
MODELS: dict[str, ModelSpec] = {
    "qwen3.8-27b": _gateway("Qwen/Qwen3.8-27B",
                          max_turns=240, max_wall_s=7_200, max_tokens=32_768,
                          max_result_chars={"uniprot-sp-2026_03": 224_000, "rhea-142": 80_000, "opaque-rhea-142": 80_000, "text2sparql-ck26": 400_000, "a15": 80_000},
                          streaming=True, guard_call_identity=True, guard_token_gap=True),
    "deepseek-v4-flash-0731": _gateway("deepseek-ai/DeepSeek-V4-Flash-0731",
                                     max_turns=120, max_wall_s=7_200, max_tokens=16_000, max_result_chars=400_000),
    "deepseek-v4.1-flash": _gateway("deepseek-ai/DeepSeek-V4.1-Flash",
                                  max_turns=120, max_wall_s=7_200, max_tokens=16_000, max_result_chars=400_000),
    "claude-sonnet-5-5": ModelSpec(lambda: AnthropicRoute(model="claude-sonnet-5-5", effort=CLAUDE_EFFORT),
                                   ANTHROPIC_KEY_ENVS,
                                   max_turns=120, max_wall_s=7_200, max_tokens=16_000, max_result_chars=400_000),
    "glm-5.3-flash": _gateway("zai-org/GLM-5.3-Flash",
                            max_turns=240, max_wall_s=7_200, max_tokens=32_768, max_result_chars=400_000,
                            streaming=True, guard_call_identity=True, guard_token_gap=True),
    "gemma-4-31b": _gateway("google/gemma-4-31B",
                          max_turns=120, max_wall_s=7_200, max_tokens=16_000,
                          max_result_chars={"uniprot-sp-2026_03": 188_000, "rhea-142": 62_000, "opaque-rhea-142": 62_000, "text2sparql-ck26": 400_000, "a15": 53_000},
                          report_key_rule="json_string_cells_v1",
                          streaming=True, guard_call_identity=True, guard_token_gap=True),
    "llama-3.3-70b": _gateway("meta-llama/Llama-3.3-70B-Instruct",
                            max_turns=120, max_wall_s=7_200, max_tokens=16_000,
                            max_result_chars={"uniprot-sp-2026_03": 80_000, "rhea-142": 80_000, "opaque-rhea-142": 80_000, "text2sparql-ck26": 400_000},
                            parallel_tool_calls=False),
    "claude-haiku-5-5": ModelSpec(lambda: AnthropicRoute(model="claude-haiku-5-5", effort=CLAUDE_EFFORT),
                                  ANTHROPIC_KEY_ENVS,
                                  max_turns=120, max_wall_s=7_200, max_tokens=16_000, max_result_chars=400_000),
    "claude-opus-5-5": ModelSpec(lambda: AnthropicRoute(model="claude-opus-5-5", effort=CLAUDE_EFFORT),
                                 ANTHROPIC_KEY_ENVS,
                                 max_turns=120, max_wall_s=7_200, max_tokens=16_000, max_result_chars=400_000),
    # Exploratory candidates for widening the panel (design log, change-log entry 80; measured by the reliability gate
    # only, nothing joins the panel): results/panel-widening-20261019. Each passed the assembler check on captured
    # pairs; like Llama's, their non-streamed message always has tool_calls. Each accepted a replayed multi-call turn.
    # Bielik does not stream, but not because the stream cuts it: its broken tool calls are its own invalid JSON,
    # delivered at its normal characters per token (token accounting, panel-widening.md section 0). Streamed they end
    # an attempt as malformed_tool_call, non-streamed as leaked_tool_call, at about the same rate (7 and 6 of 20).
    "qwen3-coder-30b-a3b": _gateway("Qwen/Qwen3-Coder-30B-A3B-Instruct",
                                  max_turns=120, max_wall_s=7_200, max_tokens=16_000,
                                  max_result_chars={"uniprot-sp-2026_03": 179_000, "rhea-142": 65_000, "opaque-rhea-142": 65_000, "text2sparql-ck26": 400_000, "a15": 54_000},
                                  streaming=True, stream_empty_tool_calls=True,
                                  guard_call_identity=True, guard_token_gap=True),
    "qwen3.6-35b-a3b": _gateway("Qwen/Qwen3.6-35B-A3B",
                              max_turns=120, max_wall_s=7_200, max_tokens=32_768,
                              max_result_chars={"uniprot-sp-2026_03": 224_000, "rhea-142": 80_000, "opaque-rhea-142": 80_000, "text2sparql-ck26": 400_000, "a15": 80_000},
                              reasoning_call_rule="reasoning_xml_call_v1",
                              streaming=True, stream_empty_tool_calls=True,
                              guard_call_identity=True, guard_token_gap=True),
    "bielik-11b-v3.0": _gateway("speakleash/Bielik-11B-v3.0-Instruct",
                              max_turns=120, max_wall_s=7_200, max_tokens=16_000, max_result_chars=80_000),
}


# The recorded moves of a model's pins a grid run accepts (check_grid_manifest): per model, the agent-config fields that
# move, from the values its earlier phases pinned ("from"; None: absent) to the ones its MODELS entry now gives ("to"),
# and the ruling. A run whose earlier phases pinned the model at "from" (they have its records, which keep their pins
# and replay under them) takes "to" once, at the first start of a phase with no record, written into the manifest's
# ``pin_moves``; any other difference of its pins is refused as before.
PIN_MOVES: dict[str, dict[str, Any]] = {
    # Gemma's rule and its caps move together: a model's pins move once (entry 119 beside entries 110 and 112)
    "gemma-4-31b": {"from": {"report_key_rule": None, "max_turns": 60, "max_wall_s": 1800.0},
                    "to": {"report_key_rule": "json_string_cells_v1", "max_turns": 120, "max_wall_s": 3600.0},
                    "ruling": "study lead and owner, design log, 20 Oct 20:42, 20:43 and 22:53, and the follow-up of "
                              "21 Oct 00:20, which completes the rule (a type number is read as a literal); change-log "
                              "entries 110 and 112, and the 00:20 follow-up; and its caps, design ruling in the design log, "
                              "21 Oct 02:19, change-log entry 119; pre-registration section 5 (agent settings)"},
    "glm-5.3-flash": {"from": {"max_turns": 120, "max_wall_s": 3600.0}, "to": {"max_turns": 240, "max_wall_s": 7200.0},
                      "ruling": "study lead, design log, 20 Oct 20:40 (the 1% rule), on GLM's causation replay "
                                "(the design log:55); change-log entry 111"},
    "qwen3.6-35b-a3b": {"from": {"reasoning_call_rule": None}, "to": {"reasoning_call_rule": "reasoning_xml_call_v1"},
                        "ruling": "design rulings in the design log Oct 00:09 and 01:31 (Qwen3.6's reasoning-channel "
                                  "call rule, judged on its leaked_tool_call rate; adopted, change-log entry 113)"},
}
# Where each field a move may name sits in a model's pins: its route entry (the manifest's routes) and its entry in a
# phase (GridPlan.model_orders; a retry budget only in the branch phase).
_MOVE_PATHS: dict[str, dict[str, tuple[tuple[str, ...], ...]]] = {
    "route": {**{rule: (("agent_config", rule),) for rule in RULE_FIELDS}, "max_turns": (("agent_config", "max_turns"),),
              "max_wall_s": (("agent_config", "max_wall_s"),)},
    "phase": {**{rule: ((rule,),) for rule in RULE_FIELDS},
              "max_turns": (("caps", "max_turns"), ("retry_budget", "max_turns")),
              "max_wall_s": (("caps", "max_wall_s"), ("retry_budget", "max_wall_s"))},
}
# A per-model setting changed after the freeze counts as a served-version change for that model (change-log entry
# 129, item 3; science, design log, 21 Oct 07:51; pre-registration section 5: "if a served version changes, that model's
# runs stop and are all re-run"). An OUT is after the freeze when its pins hold the freeze (fingerprint.shared.grid.
# freeze: the frozen pre-registration's commit, which a test grid must name before it starts, build_dev_grid.
# split_gate; a plain run of any split but dev pins it as fingerprint.freeze, plain_split_gate, change-log entry 128).
# There no declared move (PIN_MOVES) is taken: the model is held, ONLY it (design ruling, design log, 21 Oct
# 09:31), through its own hold file (OUT/holds/<model>.served-version.json, which the runner reads as it reads the
# fault monitor's and which nothing in this OUT releases), its pins in this OUT stay the ones it ran under, the change
# is recorded in the manifest's served_version_changes and alerted, and every other model goes on. No attempt of the
# model runs under the new setting beside records made under the old one; all of its runs are re-run in a new OUT.
SERVED_VERSION_AFTER_FREEZE = (
    "its pins differ from the ones this OUT recorded after the freeze (the pre-registration frozen at {commit}). A "
    "per-model setting changed after the freeze counts as a served-version change of {model} (change-log entry 129, "
    "item 3; pre-registration section 5): its runs stop and are all re-run. No pin move is taken after the freeze "
    "(PIN_MOVES included): {model} is held in this OUT for good ({hold}), its pins here stay the ones it ran under, "
    "and the other models go on; run every attempt of {model} again under the new setting in a new --out")


def frozen_out(fingerprint: dict) -> dict | None:
    """The freeze an OUT's pins hold, or None: a development OUT. A grid OUT's is its manifest's
    fingerprint.shared.grid.freeze; a plain OUT's, of any split but dev, its fingerprint.freeze (plain_split_gate,
    change-log entry 128), so a plain run after the freeze takes the same served-version rule as a test grid."""
    freeze = ((fingerprint.get("shared") or {}).get("grid") or {}).get("freeze")
    if freeze is None:
        freeze = fingerprint.get("freeze")
    if freeze is None:
        return None
    return freeze if isinstance(freeze, dict) else {"commit": str(freeze)}  # anything else held there: frozen too


class Refused(Exception):
    """The run cannot start; the message says why (exit status 2)."""


class HaltedBeforeStart(HarnessAbort):
    """The phase halted (a periodic check) before the attempt's first provider request: it is not launched, its claim
    is released and nothing of it is recorded. A HarnessAbort, so the agent loop raises it rather than recording it."""


class HeldBeforeStart(HaltedBeforeStart):
    """The infrastructure-fault monitor held the model (OUT/holds/<model>.json; change-log entry 129) before the
    attempt's first provider request: like a halt, it is not launched, its claim is released and nothing is recorded."""


def parse_args(argv: list[str] | None, models: dict[str, ModelSpec]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    # Flags marked "grid: from the manifest" default to None, so a grid run can tell that one was given.
    p.add_argument("--questions", type=Path, default=None,
                   help="question file (JSON lines); required without --grid-manifest (grid: from the manifest)")
    p.add_argument("--split", default=None, help="default dev (grid: from the manifest)")
    # A plain run of any split but dev passes the grid's freeze check (plain_split_gate): the same inputs as a grid
    # spec's freeze block (freeze.prereg_repo, freeze.freeze_commit). A grid takes them from its spec.
    p.add_argument("--prereg-repo", default=None,
                   help="a plain run of a split other than dev: the science repository whose paper/preregistration.md "
                        "is frozen (a relative path is under the code root); grid: from the spec's freeze")
    p.add_argument("--freeze-commit", default=None,
                   help="a plain run of a split other than dev: the commit of --prereg-repo with the line 'Status: "
                        "frozen', on its origin's main; grid: from the spec's freeze")
    p.add_argument("--models", nargs="+", default=None,
                   help=f"one or more of: {', '.join(models)}; required without --grid-manifest (grid: from the "
                        f"manifest)")
    p.add_argument("--runner-settings", type=Path, default=None,
                   help='plain runs (change-log entry 180): a JSON file {"request_timeout_s": {model: seconds}, '
                        '"max_turns": {model: turns}} (either or both), applied to those models of --models as a grid '
                        "spec's runner_settings are; pinned with each model's route and caps (grid: from its spec)")
    p.add_argument("--reps", type=int, default=None, help="default 3 (grid: from the manifest)")
    p.add_argument("--concurrency", type=int, default=None,
                   help="attempts in flight per model (configured workers; default 4) (grid: from the manifest)")
    p.add_argument("--concurrency-for", action="append", default=[], metavar="MODEL=N",
                   help="configured workers for one model, instead of --concurrency (repeatable) (grid: from the "
                        "manifest)")
    p.add_argument("--order-seed", type=int, default=None,
                   help="seed of the interleaved order of attempts; required with more than one model (one model: "
                        "the seed in the manifest, else one derived from the question file's sha256) (grid: the "
                        "phase's seed)")
    p.add_argument("--latency-baseline-s", action="append", default=[], metavar="MODEL=SECONDS",
                   help="the model's median per-turn call time (the request with its retries and backoff) at low "
                        f"concurrency: concurrency steps down when the median over the last {LATENCY_WINDOW} "
                        f"attempts is over {LATENCY_FACTOR:g}x this (default: the median of the first "
                        f"{LATENCY_WINDOW} attempts) (repeatable)")
    p.add_argument("--wallcap-baseline", action="append", default=[], metavar="MODEL=RATE",
                   help="the model's wall-cap rate on this tier at low concurrency (0 to 1): concurrency steps down "
                        f"when the rate over the last {WALLCAP_WINDOW} attempts is more than "
                        f"{100 * WALLCAP_MARGIN:g} percentage point over it (without it this trigger is inactive) "
                        "(repeatable)")
    p.add_argument("--latency-baseline-calm", action="append", default=[], metavar="MODEL",
                   help=f"measure the model's latency baseline on its first {CALM_CALLS} calm calls of this start (not "
                        "in a sync minute, not in an all-model stall bin of this runner's models); its latency trigger "
                        "is inactive until then (design ruling, design log, 21 Oct 15:43; repeatable)")
    p.add_argument("--reset-concurrency", action="store_true",
                   help="start every model at its configured workers even if an earlier start of this OUT "
                        "stepped it down")
    p.add_argument("--endpoint", default=None,
                   help="SPARQL endpoint URL; required without --grid-manifest (grid: the store description's)")
    p.add_argument("--store-json", type=Path, default=None,
                   help="JSON file describing the store; required without --grid-manifest (grid: from the manifest)")
    p.add_argument("--row-cap", type=int, default=None, help="default 500 (grid: from the manifest)")
    p.add_argument("--sparql-timeout", type=float, default=None, help="default 60 (grid: from the manifest)")
    p.add_argument("--out", type=Path, default=None, help="the run's directory (required, except with --preflight-log)")
    p.add_argument("--condition", default="none", choices=sorted(CONDITIONS),
                   help="the prompt condition (not in a grid: each cell's comes from the manifest)")
    p.add_argument("--prompts", type=Path, help="an ontology condition's prompt directory (data/prompts/<graph>, "
                                                "built by scripts/build_prompts.py): its manifest names the block")
    # No --max-turns, --max-wall-s, --max-tokens or result-cap option: each model's caps are in its MODELS entry
    # (design log, entry 90), so no two arms or cohorts of one model can run with different ones.
    p.add_argument("--limit", type=int, default=None, help="only the first N questions of the split (not in a grid)")
    p.add_argument("--allow-code-change", action="store_true",
                   help="resume a run (not a grid run) under other code than its last start's; the manifest records "
                        "the start as allowed (F1 review, item 11)")
    p.add_argument("--allow-store-change", metavar="NOTE", default=None,
                   help="resume a run on another endpoint or version of the same store: only the store's location and "
                        "identity may differ (its endpoint, its description's descriptive keys, the graph digests), "
                        "every safeguard and other setting must match; the change, NOTE and the attempts made before "
                        "it are recorded in the manifest. (A gate's slots and step-down rule are no part of the "
                        "fingerprint: each start records them, and a new rule version is recorded as rule_changed.)")
    p.add_argument("--reclaim", action="store_true",
                   help="delete stale claims (no record, heartbeat older than --claim-timeout-s, or a dead process "
                        "on this host) so those attempts run again")
    p.add_argument("--claim-timeout-s", type=float, default=1800.0,
                   help="a claim whose heartbeat is older than this counts as stale for --reclaim")
    p.add_argument("--claim-heartbeat-s", type=float, default=60.0,
                   help="how often a running attempt refreshes its claim's heartbeat")
    p.add_argument("--engine-suspect-vocab", type=Path, default=None,
                   help="private vocabulary file whose suspect classes mark the engine-suspect query shape "
                        "(harness/engine_suspect.py); default none (the environment is never read for this)")
    p.add_argument("--infra-retry-s", type=float, default=AgentConfig.infra_retry_s,
                   help="per query, how long to wait out a SPARQL store outage before the attempt fails "
                        "infrastructure (not charged to the wall cap)")
    p.add_argument("--infra-reruns", type=int, default=3,
                   help="re-run an attempt that failed infrastructure up to this many times, once the store "
                        "answers again")
    p.add_argument("--max-infra-failures", type=int, default=3,
                   help="stop a model after this many consecutive attempts failed infrastructure on every try")
    p.add_argument("--gate-wait-s", type=float, default=gated.GATE_WAIT_S,
                   help="a store read through a query gate (its description's private_files): the longest one query "
                        "waits for the gate before its attempt fails infrastructure")
    p.add_argument("--preflight-log", type=Path, default=None,
                   help="a store read through a query gate: send only its triple count through the run's client and "
                        "gate, append it to this private log (mode 600, outside the repository; never printed) and "
                        "exit; needs --store-json and --endpoint only")
    p.add_argument("--deployment-change", action="append", default=[], metavar="MODEL",
                   help="a recorded deployment change of MODEL (design ruling in the design log Oct 23:24; after the "
                        "freeze a served-version change that holds MODEL, change-log entry 129): at this "
                        "start, after its identity checks, the one system fingerprint the runner recorded for MODEL "
                        "in MODEL_CHANGED.json since its baseline, on turns of its own served id, becomes its new "
                        "baseline, recorded in served-models.json and the manifest's deployment_changes; needs "
                        "--ruling")
    p.add_argument("--ruling", default=None, help="the ruling a --deployment-change rests on, recorded with it")
    p.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=None,
                   help="run each model's first attempt of each arm alone before the pool starts, so it writes the "
                        "prompt cache the others read (default: on for Anthropic routes, off for OpenAI-compatible; "
                        "never in a grid's branch phase)")
    grid = p.add_argument_group("development grid (the design log)")
    grid.add_argument("--grid-manifest", type=Path, default=None,
                      help="manifest.json from scripts/build_dev_grid.py: run one graph's phase of the grid")
    grid.add_argument("--grid-spec", type=Path, default=DEFAULT_GRID_SPEC,
                      help="the grid spec the manifest was built from (its sha256 must be the manifest's)")
    grid.add_argument("--grid-root", type=Path, default=REPO_ROOT,
                      help="the directory the spec's paths are relative to")
    grid.add_argument("--graph", default=None, help="the graph to run (one OUT per graph)")
    grid.add_argument("--phase", choices=GRID_PHASES, default=None,
                      help="natural (the none cohort), prompted (the prompted cohorts) and branch (every checker "
                           "arm, once both are complete)")
    grid.add_argument("--periodic-check-s", type=float, default=PERIODIC_CHECK_S,
                      help="in the branch phase, recompute the checker decisions of the records completed so far in a "
                           "fresh interpreter this often, in seconds (0: never), and stop new launches when their "
                           "divergences halt the phase (default every 2 hours)")
    grid.add_argument("--periodic-check-max-load", type=float, default=PERIODIC_MAX_LOAD,
                      help="skip a periodic check while the one-minute load average per CPU is at least this")
    args = p.parse_args(argv)
    if args.deployment_change and not (args.ruling or "").strip():
        p.error("--deployment-change needs --ruling: the ruling the recorded deployment change rests on")
    if args.ruling is not None and not args.deployment_change:
        p.error("--ruling goes with --deployment-change")
    twice = sorted({m for m in args.deployment_change if args.deployment_change.count(m) > 1})
    if twice:
        p.error(f"--deployment-change names {', '.join(twice)} twice")
    if not (math.isfinite(args.gate_wait_s) and args.gate_wait_s > 0):
        p.error("--gate-wait-s must be a finite positive number of seconds")
    if args.preflight_log is not None:
        if args.store_json is None or args.endpoint is None or args.grid_manifest is not None:
            p.error("--preflight-log needs --store-json and --endpoint, and no grid")
        if args.runner_settings is not None:
            p.error("--runner-settings: a preflight runs no model")
        if args.sparql_timeout is None:
            args.sparql_timeout = 60.0
        if args.row_cap is None:
            args.row_cap = 500
        return args
    if args.out is None:
        p.error("the following arguments are required: --out")
    derived = {"--questions": args.questions, "--split": args.split, "--models": args.models, "--reps": args.reps,
               "--runner-settings": args.runner_settings,
               "--order-seed": args.order_seed, "--store-json": args.store_json, "--row-cap": args.row_cap,
               "--sparql-timeout": args.sparql_timeout, "--limit": args.limit, "--endpoint": args.endpoint,
               "--concurrency": args.concurrency, "--concurrency-for": args.concurrency_for or None,
               "--prereg-repo": args.prereg_repo, "--freeze-commit": args.freeze_commit}
    if args.grid_manifest is not None:
        given = [flag for flag, value in derived.items() if value is not None]
        if given:
            p.error(f"{', '.join(given)}: a grid run takes {'these' if len(given) > 1 else 'this'} from --grid-manifest; "
                    f"nothing of the manifest is overridden")
        if args.graph is None or args.phase is None:
            p.error("--grid-manifest needs --graph and --phase")
        if args.phase == "branch" and args.warmup:
            p.error("--warmup: a branch cell's retry continues its parent's transcript, so the branch phase has no "
                    "warm-up")
    else:
        if args.graph is not None or args.phase is not None:
            p.error("--graph and --phase need --grid-manifest")
        missing = [flag for flag, value in (("--questions", args.questions), ("--models", args.models),
                                            ("--store-json", args.store_json), ("--endpoint", args.endpoint))
                   if value is None]
        if missing:
            p.error(f"the following arguments are required: {', '.join(missing)}")
        for name, default in (("split", "dev"), ("reps", 3), ("row_cap", 500), ("sparql_timeout", 60.0),
                              ("concurrency", 4)):
            if getattr(args, name) is None:
                setattr(args, name, default)
        if args.reps < 1 or args.row_cap < 1:
            p.error("--reps and --row-cap must be positive")
        if args.split == DEV_SPLIT and (args.prereg_repo is not None or args.freeze_commit is not None):
            p.error("--prereg-repo and --freeze-commit open a plain run of a split other than dev (the frozen "
                    "pre-registration); a dev run takes neither")
    if (args.concurrency is not None and args.concurrency < 1) or args.max_infra_failures < 1:
        p.error("--concurrency and --max-infra-failures must be positive")
    if not 0 < args.claim_heartbeat_s < args.claim_timeout_s:
        p.error("--claim-heartbeat-s must be positive and less than --claim-timeout-s")
    if args.infra_retry_s < 0 or args.infra_reruns < 0:
        p.error("--infra-retry-s and --infra-reruns must not be negative")
    if args.periodic_check_s < 0 or args.periodic_check_max_load <= 0:
        p.error("--periodic-check-s must not be negative and --periodic-check-max-load must be positive")
    names = args.models  # a grid's models come from its manifest: their names are checked there (check_per_model)
    args.concurrency_for = _per_model(p, args.concurrency_for, "--concurrency-for", names, int, lambda n: n >= 1)
    args.latency_baseline = _per_model(p, args.latency_baseline_s, "--latency-baseline-s", names, float,
                                       lambda v: v > 0)
    args.wallcap_baseline = _per_model(p, args.wallcap_baseline, "--wallcap-baseline", names, float,
                                       lambda v: 0 <= v < 1)
    unknown = [m for m in args.latency_baseline_calm if names is not None and m not in names]
    both = sorted(set(args.latency_baseline_calm) & set(args.latency_baseline))
    if unknown or both:
        p.error(f"--latency-baseline-calm {unknown or both}: " + ("not one of --models" if unknown else
                "also given a fixed --latency-baseline-s"))
    args.latency_baseline_calm = set(args.latency_baseline_calm)
    if args.grid_manifest is None and len(set(args.models)) * len(run_arms()) > 1 and args.order_seed is None:
        p.error("--order-seed is required when a run has more than one model or arm (the interleaved order is "
                "pre-registered by its seed)")
    return args


def _per_model(p: argparse.ArgumentParser, items: list[str], flag: str, names: list[str] | None, cast: Callable,
               valid: Callable) -> dict:
    """``names`` None (a grid run): any model name is taken here and checked against the manifest's models."""
    out = {}
    for item in items:
        name, sep, raw = item.partition("=")
        try:
            value = cast(raw)
        except ValueError:
            value = None
        if not sep or (names is not None and name not in names) or value is None or not valid(value):
            p.error(f"{flag} {item!r}: expected MODEL=VALUE with MODEL one of --models and a valid value")
        out[name] = value
    return out


def run_arms() -> list[str]:
    """The arms of a run without the grid, interleaved with its models in the order: its prompt
    condition, one per run. A grid run's arms are its cells, and its attempt keys and claims hold the
    arm (``attempt_key``)."""
    return [AgentConfig().condition]


# --- questions and records ------------------------------------------------------------


def load_questions(path: Path, split: str) -> list[dict]:
    questions, seen = [], set()
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            q = json.loads(line)
            missing = [f for f in QUESTION_FIELDS if f not in q]
            if missing:
                raise ValueError(f"{path}:{n}: question is missing {missing}")
            if q["split"] != split:
                continue
            if q["id"] in seen:
                raise ValueError(f"{path}:{n}: duplicate question id {q['id']!r}")
            question_gold(q)  # a malformed gold rule fails here, before any attempt
            seen.add(q["id"])
            questions.append(q)
    return questions


def attempt_question(q: dict) -> dict:
    return {"question_id": q["id"], "base_question_id": q.get("base_question_id"), "template": q.get("template"),
            "split": q.get("split"), "text": q["question"]}


def attempt_key(question_id: str, route_model: str, repetition: int, arm: str | None = None) -> tuple:
    """(question, model, repetition) in a run without the grid, whose one arm needs no place in the key (and
    whose claims are hashed from it); (arm, question, model, repetition) in a grid, whose OUT holds every cell."""
    return (question_id, route_model, repetition) if arm is None else (arm, question_id, route_model, repetition)


def record_arm(data: dict) -> str:
    """The arm a record ran in: its ``load.arm`` (a grid record's cell), else its prompt condition."""
    return (data.get("load") or {}).get("arm") or data.get("condition") or DEFAULT_ARM


def record_paths(records_dir: Path, *, grid: bool = False) -> dict[tuple, Path]:
    """Each attempt key that has a record, with the record's path (in a grid the key holds the arm);
    refuses when two records share a key."""
    seen: dict[tuple, Path] = {}
    duplicates: list[str] = []
    if records_dir.is_dir():
        for path in sorted(records_dir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            key = attempt_key(data["question_id"], data["model"], data["repetition"], record_arm(data) if grid else None)
            if key in seen:
                duplicates.append(f"{key}: {seen[key].name} and {path.name}")
            else:
                seen[key] = path
    if duplicates:
        raise Refused(f"records share an attempt key ({'arm, ' if grid else ''}question, model, repetition):\n  "
                      + "\n  ".join(duplicates))
    return seen


def existing_keys(records_dir: Path, *, grid: bool = False) -> set[tuple]:
    """The attempt keys that have a record; refuses when two records share a key."""
    return set(record_paths(records_dir, grid=grid))


def live_turns(rec: Any) -> list[dict]:
    """The turns this try asked the model for: every turn, but in a branch record (a checker arm branched
    from a parent record) only those after the initial phase, which is the parent's, replayed."""
    per_turn = rec.tokens.get("per_turn") or []
    checker = rec.checker if hasattr(rec, "checker") else None
    if checker and checker.get("branched_from"):
        initial_turns = checker["initial"]["outcome"]["turns"]
        return [t for t in per_turn if t["turn"] > initial_turns]
    return per_turn


def index_line(rec: RunRecord, model_name: str) -> dict[str, Any]:
    """One finished attempt. Tokens, cache hit rate and cost are over the turns the attempt asked the model for
    (``paid_turns``): a branch record's replayed initial phase was paid for by its parent's line."""
    score = rec.score or {}
    branch = bool(rec.checker and rec.checker.get("branched_from"))
    paid = live_turns(rec)
    totals = token_totals(paid) if branch else rec.tokens["totals"]
    return {
        "run_id": rec.run_id,
        "question_id": rec.question_id,
        "model": model_name,
        "route_model": rec.model,
        "repetition": rec.repetition,
        "arm": record_arm({"load": rec.load, "condition": rec.condition}),
        "status": rec.outcome["label"],
        "exact": score.get("exact"),
        "f1": score.get("f1"),
        "wall_s": round(rec.wall_s, 3),
        "turns": rec.outcome["turns"],
        "looping": rec.outcome["looping"],
        "truncated_turns": totals["truncated_turns"],
        "model_mismatch_turns": rec.outcome["model_mismatch_turns"],
        # tries of this attempt that failed infrastructure (the last one too when every try failed)
        "infra_tries": (rec.runner or {}).get("infra_tries", 0),
        "infra_source": rec.outcome.get("infra_source"),  # store or provider, for failed:infrastructure
        "tokens": {k: totals[k] for k in ("input_tokens", "output_tokens", "cache_read_tokens",
                                          "cache_write_tokens")},
        "paid_turns": len(paid),
        "cache_hit_rate": cache_hit_rate(totals),
        "cost_usd_estimate": estimate_record_cost({"route": rec.route, "tokens": {"per_turn": paid, "totals": totals}}
                                                  if branch else rec)["usd"],
    }


def is_config_failure(rec: RunRecord) -> bool:
    error = rec.outcome.get("provider_error") or {}
    return (rec.outcome["status"] == "failed" and rec.outcome["failure_type"] == "provider_error"
            and error.get("error_class") in CONFIG_ERROR_CLASSES)


# --- the manifest ---------------------------------------------------------------------


def route_fingerprint(route: Provider) -> dict:
    """What identifies a route's behaviour: provider, base URL, model and settings (not the SDK version)."""
    info = route.route_info()
    return {k: info.get(k) for k in ("provider", "base_url", "model", "settings")}


# --- a model's endpoints behind the loopback front (change-log entry 167) ---------------------------------------
# Science, design log, 08:40-08:58Z: endpoints are joined to a model's running OUTs through one loopback-only front on the
# runner's host. OUT/model-fronts.json maps a model to the front; its route then contacts the front while recording the
# base URL the OUT pinned (a declared deviation: route.base_url is the route's identity, not the URL contacted; the
# records add front_url, and the front's log joins each response id to the backend that served it).

MODEL_FRONTS_FILE = "model-fronts.json"
MODEL_FRONTS_SCHEMA = "model-fronts/1"
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


def front_url_problem(url: Any) -> str | None:
    """Why ``url`` cannot be the front's (or one of its per-backend listeners'), or None: an http(s) URL on a loopback
    host, a port optional, with no user, password, query or fragment (the URL is recorded)."""
    problem = "must be an http(s) URL on a loopback host (127.0.0.1, localhost or ::1) with no user, query or fragment"
    if not isinstance(url, str) or not url:
        return problem
    try:
        parts = urlsplit(url)
        parts.port  # noqa: B018 - a port that is not a number raises here
    except ValueError:
        return problem
    if (parts.scheme not in ("http", "https") or parts.hostname not in LOOPBACK_HOSTS or parts.username is not None
            or parts.password is not None or parts.query or parts.fragment):
        return problem
    return None


class FrontRoute:
    """A model's route through the loopback front: every request goes to ``front_url`` through the route rebuilt there
    (``with_options(base_url=front_url)``: the same model, settings and key, every check of its constructor run again),
    while ``route_info`` reports the route's own base URL, the OUT's pinned route identity, and adds ``front_url``. So
    the manifest's pins, the served-version rule (a changed route after the freeze) and the analysis's one base URL per
    (graph, model) see the route they recorded. ``with_seed`` seeds both and keeps the front (a branch arm's retry,
    harness/checkloop/online.py); every other attribute is the rebuilt route's."""

    def __init__(self, route: Any, front_url: str, *, _transport: Any = None):
        if _transport is None:
            if not takes_front(route):
                raise TypeError(f"a {type(route).__name__} cannot take a front: only an OpenAI-compatible route, "
                                f"rebuilt on another base URL by with_options, can")
            if getattr(route, "_client", None) is not None:  # with_options keeps it, and the URL it was made for
                raise ValueError("a route that already holds a client cannot take a front")
            _transport = route.with_options(base_url=front_url)
            own, there = route.route_info(), _transport.route_info()
            if any(own.get(k) != there.get(k) for k in ("provider", "model", "settings")):
                raise ValueError("the route rebuilt on the front differs from the route in more than its base URL")
        if getattr(_transport, "base_url", None) != front_url:  # the URL its requests go to (its client is made there)
            raise ValueError("the route rebuilt on the front does not send to the front's URL")
        self.__dict__.update(_route=route, _transport=_transport, front_url=front_url)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):  # copy and pickle look these up on a bare instance
            raise AttributeError(name)
        try:
            transport = self.__dict__["_transport"]
        except KeyError:
            raise AttributeError(name) from None
        return getattr(transport, name)

    def __repr__(self) -> str:
        return f"FrontRoute({self._route!r}, front {self.front_url!r})"

    def step(self, *args: Any, **kwargs: Any):
        return self._transport.step(*args, **kwargs)

    def route_info(self) -> dict:
        return {**self._transport.route_info(), "base_url": self._route.route_info().get("base_url"),
                "front_url": _public_url(self.front_url)}

    def with_seed(self, seed: int) -> FrontRoute:
        return FrontRoute(self._route.with_seed(seed), self.front_url, _transport=self._transport.with_seed(seed))

    def with_options(self, **overrides: Any) -> FrontRoute:
        """The route with other options, still through the front (its base URL stays the pinned one)."""
        if "base_url" in overrides:
            raise ValueError("a route through the front keeps its pinned base URL; map another front instead")
        return FrontRoute(self._route.with_options(**overrides), self.front_url)


def takes_front(route: Any) -> bool:
    """Whether ``route`` can go through the front: an OpenAI-compatible route (the front speaks chat completions),
    rebuilt on another base URL by ``with_options``."""
    return (callable(getattr(route, "with_options", None))
            and (route.route_info() or {}).get("provider") == OpenAICompatRoute.provider)


def _front_factory(factory: Callable[[], Provider], front_url: str) -> Callable[[], Provider]:
    return lambda: FrontRoute(factory(), front_url)


def _url_name(url: Any) -> str:
    """A URL named in a message by its sha256 only (the replication package carries no hosts)."""
    return "none" if url is None else "sha256:" + hashlib.sha256(str(url).encode("utf-8")).hexdigest()[:12]


def model_fronts(out: Path, models: Mapping[str, ModelSpec], names: Any,
                 say: Callable[[str], None] = lambda msg: None) -> tuple[dict[str, str], str | None]:
    """OUT/model-fronts.json, read once per start before any route is built: {"models": {name: {"route_base_url",
    "front_url"}}} (an optional "schema", model-fronts/1; any other key is the front's own). Returns each model of
    ``names`` it maps -> its front URL, and the file's sha256 (({}, None) without a file). Refused, before anything is
    written: a file that is not that object; and, for a model this start runs, an entry without both URLs, a front URL
    that is not loopback (``front_url_problem``), a route that cannot take a front (``takes_front``) or a route_base_url
    other than the route's own base URL and the one the OUT's manifest pinned for the model; and a name that is no
    model of the runner's table (a misspelt name would leave its model on the pinned URL unseen). The entry of a model
    of the table this start does not run is not read (``say`` names them)."""
    path = out / MODEL_FRONTS_FILE
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return {}, None
    except OSError as exc:
        raise Refused(f"{path} cannot be read ({exc})") from None
    try:
        doc = json.loads(raw)
    except ValueError as exc:
        raise Refused(f"{path} cannot be read as JSON ({exc})") from None
    if not isinstance(doc, dict) or not isinstance(doc.get("models"), dict):
        raise Refused(f"{path}: not an object with a models object")
    if doc.get("schema", MODEL_FRONTS_SCHEMA) != MODEL_FRONTS_SCHEMA:
        raise Refused(f"{path}: schema {doc.get('schema')!r}, not {MODEL_FRONTS_SCHEMA}")
    pinned = _pinned_base_urls(out)
    run = set(names)
    fronts: dict[str, str] = {}
    problems: list[str] = []
    not_run = []
    for name, entry in doc["models"].items():
        if name not in models:
            problems.append(f"{name}: not a model of the runner's table")
            continue
        if name not in run:
            not_run.append(name)
            continue
        if not isinstance(entry, dict) or not all(isinstance(entry.get(k), str) and entry.get(k)
                                                  for k in ("route_base_url", "front_url")):
            problems.append(f"{name}: needs route_base_url and front_url, each a URL")
            continue
        why = front_url_problem(entry["front_url"])
        if why is not None:
            problems.append(f"{name}: front_url {why}")
            continue
        route = models[name].factory()
        if not takes_front(route):
            problems.append(f"{name}: its route cannot take a front (only an OpenAI-compatible route, rebuilt on "
                            f"another base URL)")
            continue
        want, own = entry["route_base_url"], route.route_info().get("base_url")
        if want != own:
            problems.append(f"{name}: route_base_url ({_url_name(want)}) is not the route's own base URL "
                            f"({_url_name(own)})")
        if name in pinned and pinned[name] != want:
            problems.append(f"{name}: route_base_url ({_url_name(want)}) is not the base URL the manifest pinned for "
                            f"it ({_url_name(pinned[name])})")
        fronts[name] = entry["front_url"]
    if problems:
        raise Refused(f"{path}: " + "; ".join(problems))
    if not_run:
        say(f"model fronts: {', '.join(not_run)} mapped in {MODEL_FRONTS_FILE} but not run by this start")
    return fronts, hashlib.sha256(raw).hexdigest()


def _pinned_base_urls(out: Path) -> dict[str, Any]:
    """Each model's base URL in the OUT's manifest (fingerprint.routes, in a plain run's and a grid's alike)."""
    path = out / "manifest.json"
    if not path.exists():
        return {}
    try:
        routes = (json.loads(path.read_text(encoding="utf-8")).get("fingerprint") or {}).get("routes") or {}
    except (OSError, ValueError, AttributeError) as exc:
        raise Refused(f"{path} cannot be read ({exc})") from None
    return {name: route.get("base_url") for name, route in routes.items() if isinstance(route, dict)}


def _file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


def _store_graphs(store: dict) -> set[str]:
    graphs = store.get("graphs")
    return set(graphs) if isinstance(graphs, (dict, list)) else set()


def ontology_block(condition: str, prompts_dir: Path | None, store: dict | None = None) -> str | None:
    """The sha256 of the block an ontology condition runs with: the one its graph's prompt manifest names under
    the condition's own key, refused unless the manifest marks it selectable (never a provisional render, never a
    block of an opaque graph whose input leak check failed or is missing) and its layout is the store's: every
    graph of the layout among the store description's graphs, and the same index identity where both name one
    (External review of b552d62, items 2 and 3)."""
    if not get_condition(condition).ontology:
        if prompts_dir is not None:
            raise Refused("--prompts is only for an ontology condition")
        return None
    if prompts_dir is None:
        raise Refused(f"condition {condition!r} needs --prompts data/prompts/<graph>")
    path = prompts_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    entry = manifest["conditions"].get(condition)
    if entry is None or entry.get("provisional"):
        why = manifest.get("skipped", {}).get(condition)
        raise Refused(f"{path} has no {condition!r} prompt" + (f": {why}" if why else ""))
    if manifest.get("opaque"):
        lc = manifest.get("leak_check")
        if not isinstance(lc, dict) or "leak" not in lc:
            raise Refused(f"{path}: an opaque graph's prompts with no input leak check; rebuild them")
        if lc["leak"] is not False or (lc.get("hits") or {}).get(condition):
            raise Refused(f"{path}: the input leak check failed; no block of these prompts may be used")
    if entry.get("selectable") is not True:
        raise Refused(f"{path}: the {condition!r} prompt is not selectable")
    store = store or {}
    served, wanted = _store_graphs(store), set(manifest.get("layout_graphs") or [])
    if not served:
        raise Refused("the store description names no graphs, so the prompts' graph cannot be checked")
    if not wanted or not wanted <= served:
        raise Refused(f"{path}: the prompts are for layout {manifest.get('layout')!r} (graphs {sorted(wanted)}), "
                      f"not the store described (graphs {sorted(served)})")
    if manifest.get("store_identity") and store.get("identity") and manifest["store_identity"] != store["identity"]:
        raise Refused(f"{path}: the prompts were built from index identity {manifest['store_identity']!r}, the store "
                      f"describes {store['identity']!r}")
    sha = entry["block"]["sha256"]
    if manifest.get("private"):  # a private graph's blocks stay in the private store its manifest names (a15)
        store_rel = entry["block"].get("store")
        block_dir = (prompts_dir / store_rel).resolve().parent if store_rel else None
        if block_dir is None or block_dir.is_relative_to(REPO_ROOT.resolve()) or (block_dir.stat().st_mode & 0o077):
            raise Refused(f"{path}: a private graph's block store must be a private directory outside the repository")
        add_block_store(block_dir)
    load_block(sha)  # present in the block store and intact
    return sha


def model_fingerprint(route: Provider, config: AgentConfig) -> dict:
    """A model's entry in the manifest: its route's fingerprint, its agent config's per-model fields (its caps,
    transport and stream guards) and the hash of the tools they give. A model already in the manifest must keep
    all three."""
    per_model = {k: getattr(config, k) for k in MODEL_FIELDS}
    per_model["stream_guards"] = list(per_model["stream_guards"])
    for field in RULE_FIELDS:  # only those set: every other model's entry is what it was
        if getattr(config, field) is not None:
            per_model[field] = getattr(config, field)
    return {**route_fingerprint(route), "agent_config": per_model,
            "tools_sha256": tools_sha256(agent_tools(config.max_result_chars))}


def run_fingerprint(args: argparse.Namespace, store: dict, config: AgentConfig, routes: dict[str, dict],
                    scorer_digest: str | None = None, store_client: dict | None = None) -> dict:
    """The run's settings: ``config`` is the agent config every model shares (its per-model fields are in each
    model's entry in ``routes``, ``model_fingerprint``). ``store_client``: the settings of a store read through a
    query gate's clients (graph digest, server time-out, row limit set; not the gate's wait bound, which may change
    between starts like ``infra_retry_s``), only for such a store."""
    cond = get_condition(config.condition)
    prompt = {"condition": cond.condition, "id": cond.id, "template_sha256": cond.template_sha256,
              "nudge_sha256": sha256_text(cond.nudge)}  # the tools hash is per model, in ``routes``
    if config.ontology_block_sha256 is not None:
        prompt["ontology_block_sha256"] = config.ontology_block_sha256
    fp = {
        "questions_sha256": hashlib.sha256(args.questions.read_bytes()).hexdigest(),
        "store": sanitize_store(store),  # allowed keys only, no credentials in URLs (metadata.py)
        "endpoint": _public_url(args.endpoint),
        "row_cap": args.row_cap,
        "sparql_timeout_s": args.sparql_timeout,
        # infra_retry_s only decides how long a store outage is waited out; attempts that hit one are re-run,
        # never counted, so it may change between starts of one run.
        "agent_config": {k: v for k, v in config.to_json().items() if k != "infra_retry_s" and k not in MODEL_FIELDS},
        "prompt": prompt,
        "routes": routes,
        "scorer_digest": scorer_digest,  # the frozen scorer's (frozen_scorer_digest): a resume needs the same
    }
    if store_client is not None:
        fp["store_client"] = store_client
    freeze = getattr(args, "freeze", None)  # a plain run the frozen pre-registration opened (plain_split_gate) only
    if freeze is not None:
        fp["freeze"] = {k: freeze[k] for k in ("commit", "preregistration_sha256")}
    return fp


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(obj, dict) and obj:
        out: dict[str, Any] = {}
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
        return out
    return {prefix: obj}


def fingerprint_diff(recorded: dict, now: dict) -> list[str]:
    a, b = _flatten(recorded), _flatten(now)
    diff = []
    for path in sorted(set(a) | set(b)):
        if path not in a:
            diff.append(f"{path}: absent when the run started, now {b[path]!r}")
        elif path not in b:
            diff.append(f"{path}: recorded {a[path]!r}, now absent")
        elif a[path] != b[path]:
            diff.append(f"{path}: recorded {a[path]!r}, now {b[path]!r}")
    return diff


def _redacted_json(obj: Any) -> Any:
    """``obj`` in JSON form with every environment secret value replaced, as the record writer
    does. The parsed values are redacted, not the serialised text, so the result is always JSON."""
    return redact_structure(json.loads(json.dumps(obj, ensure_ascii=False)))[0]


def earlier_layout(fingerprint: dict) -> bool:
    """Whether a stored fingerprint has the layout before agent-run/16: a per-model field (a cap) or the tools hash
    in the shared part, or a model's entry without its own agent config. Such a run is refused, not migrated
    (``check_manifest``): equal settings would not make it one experiment, since the rules its attempts ran under
    changed with the layout. A fingerprint of another kind (the closed-book probes', scripts/run_probes.py, which
    names its ``kind`` and has no agent config) is never of the earlier layout."""
    if "kind" in fingerprint:
        return False
    shared = fingerprint.get("agent_config") or {}
    routes = fingerprint.get("routes") or {}
    return (any(k in shared for k in MODEL_FIELDS) or "tools_sha256" in (fingerprint.get("prompt") or {})
            or any(isinstance(r, dict) and "agent_config" not in r for r in routes.values()))


def normalise_fingerprint(recorded: dict) -> dict:
    """A stored fingerprint in today's form: store metadata sanitised (manifests written
    before the sanitiser hold it raw) and secret values redacted. Idempotent."""
    out = dict(recorded)
    if isinstance(out.get("store"), dict):
        out["store"] = sanitize_store(out["store"])
    return _redacted_json(out)


def _write_json_atomic(path: Path, data: dict) -> None:
    text = json.dumps(_redacted_json(data), ensure_ascii=False, indent=1, sort_keys=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _same_code(before: dict, now: dict) -> bool:
    """Whether a start runs the code an earlier start ran: the same commit and, when the earlier start recorded one
    (from F1 on), the same tree digest."""
    return (before.get("git_commit") == now.get("git_commit")
            and before.get("tree_sha256") in (None, now.get("tree_sha256")))


# What a recorded store change (--allow-store-change) may change in a run's fingerprint: where the store is and which
# version it is, never the graph's name, the client's time-outs and limits, or anything about the agents.
STORE_FINGERPRINT_KEYS = ("store", "endpoint", "store_client")
STORE_CHANGE_PATHS = ("endpoint", "store.endpoint", "store.version", "store.engine", "store.image", "store.image_digest",
                      "store.identity", "store.settings", "store.description", "store.url", "store.endpoint_label",
                      "store.server_query_timeout", "store.default_graph_distinct_triples", "store.graphs",
                      "store_client.endpoint", "store_client.default_graph_sha256", "store_client.named_graphs_sha256")


def _store_change_allowed(path: str) -> bool:
    return any(path == p or path.startswith(p + ".") for p in STORE_CHANGE_PATHS)


def _changed_paths(recorded: dict, now: dict) -> list[str]:
    a, b = _flatten(recorded), _flatten(now)
    missing = object()
    return sorted(p for p in set(a) | set(b) if a.get(p, missing) != b.get(p, missing))


def check_manifest(out: Path, fingerprint: dict, *, has_records: bool,
                   say: Callable[[str], None] = lambda msg: None, allow_code_change: bool = False,
                   start_fields: dict | None = None, allow_store_change: str | None = None) -> list[str]:
    """Write the manifest on first start; on resume, raise Refused unless it matches.

    On resume the stored fingerprint is normalised the same way as the new one
    (``normalise_fingerprint``), so a manifest written before the store sanitiser still
    matches; ``say`` reports when that changed anything. Every start appends the code it runs on
    to the manifest's ``starts`` (F1 review, item 11), and a resume under other code than the last
    start's (another commit or tree digest) is refused unless ``allow_code_change``, which the start
    records (``code_change_allowed``). ``start_fields`` are added to this start's entry (a gate of several slots: its
    setting and the size it starts at). With ``allow_store_change`` (its note), a resume may differ in the store's
    location and identity only (STORE_CHANGE_PATHS): the change, the note and the run ids of the records made before
    it are appended to the manifest's ``store_changes``, the fingerprint takes the new store, and this start's entry
    says so (``store_changed``). Returns the names of models this start added to the manifest.
    """
    # Compare in the form that is stored: JSON, with environment secret values redacted.
    fingerprint = _redacted_json(fingerprint)
    path = out / "manifest.json"
    code = harness_commit()
    if not path.exists():
        if has_records:
            raise Refused(f"{out} holds records but no manifest.json, so it cannot be checked; use a new --out")
        _write_json_atomic(path, {"schema": MANIFEST_SCHEMA, "created_at": utc_now(), "harness": code,
                                  "starts": [{"at": utc_now(), "harness": code, **(start_fields or {})}],
                                  "fingerprint": fingerprint})
        return list(fingerprint["routes"])
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise Refused(f"{path}: unknown manifest schema {manifest.get('schema')!r}")
    recorded = manifest["fingerprint"]
    stored = recorded
    if earlier_layout(stored):
        commit = (manifest.get("harness") or {}).get("git_commit")
        raise Refused(f"{out} was started under the earlier manifest layout (one agent config shared by every model, "
                      f"from before each model's caps, transport and stream guards became its own, agent-run/16); "
                      f"its attempts ran under other rules (no stream guards, no bare_json leak rule), so it is not "
                      f"resumed under this harness: start a fresh --out, or finish it with the harness it started "
                      f"under ({str(commit)[:12] if commit else 'see its manifest'})")
    recorded = normalise_fingerprint(stored)
    if recorded != stored:
        say(f"manifest: the stored fingerprint was normalised (store metadata sanitised, secret values redacted) "
            f"before comparing; {path.name} is unchanged")
    rest_now = {k: v for k, v in fingerprint.items() if k != "routes"}
    rest_recorded = {k: v for k, v in recorded.items() if k != "routes"}
    shared = {name: fingerprint["routes"][name] for name in fingerprint["routes"] if name in recorded["routes"]}
    diff = fingerprint_diff({**rest_recorded, "routes": {n: recorded["routes"][n] for n in shared}},
                            {**rest_now, "routes": shared})
    store_change = None
    others: list[str] = []
    if diff and allow_store_change is not None:
        changed = _changed_paths({**rest_recorded, "routes": {n: recorded["routes"][n] for n in shared}},
                                 {**rest_now, "routes": shared})
        others = [p for p in changed if not _store_change_allowed(p)]
        if not others:
            records_dir = out / "records"
            before = sorted(p.name.split(".")[1] for p in records_dir.glob("*.json")) if records_dir.is_dir() else []
            store_change = {"at": utc_now(), "note": allow_store_change, "changed": changed,
                            "from": {k: recorded.get(k) for k in STORE_FINGERPRINT_KEYS},
                            "to": {k: fingerprint.get(k) for k in STORE_FINGERPRINT_KEYS},
                            "pre_change_records": len(before), "pre_change_run_ids": before}
    if diff and store_change is None:  # refused, with or without --allow-store-change
        frozen = frozen_out(recorded)
        if frozen is not None:  # after the freeze a model's own changed pins are a served-version change of the model
            # (change-log entry 129, item 3): the plain path holds no model, so the start is refused as before, nothing
            # written, and each such change is alerted at once, naming the model and its changed pins
            for name in sorted(shared):
                own = fingerprint_diff(recorded["routes"][name], shared[name])
                if own:
                    pins = ", ".join(f"routes.{name}.{d.split(':', 1)[0]}" for d in own)
                    say(f"ALERT served-version change after the freeze: {name}: its pins changed ({pins}) in a plain "
                        f"run after the freeze (the pre-registration frozen at {str(frozen.get('commit'))[:12]}); the "
                        f"plain path holds no model, so this start is refused, nothing written: run every attempt of "
                        f"{name} again under the new setting in a new --out (change-log entry 129, item 3)")
        if others:
            raise Refused(f"{out}: a store change may change the store's location and identity only, not "
                          + ", ".join(others) + "; resume needs an exact match otherwise")
        raise Refused(f"{out} was started with different settings; resume needs an exact match "
                      f"(use a new --out for a new experiment):\n  " + "\n  ".join(diff))
    elif not diff and allow_store_change is not None:
        say("manifest: --allow-store-change given, but the store is the one recorded: no store change to record")
    last = (manifest.get("starts") or [{"harness": manifest.get("harness") or {}}])[-1]["harness"] or {}
    changed = not _same_code(last, code)
    if changed and not allow_code_change:
        raise Refused(f"{out} was last started under code {str(last.get('git_commit'))[:12]} (tree "
                      f"{str(last.get('tree_sha256'))[:12]}); this start runs {str(code.get('git_commit'))[:12]} (tree "
                      f"{str(code.get('tree_sha256'))[:12]}), so its records would mix code: resume under the same "
                      f"code, use a new --out, or pass --allow-code-change, which the manifest records")
    if store_change is not None:  # every check passed: the change goes on record with this start
        manifest.setdefault("store_changes", []).append(store_change)
        for key in STORE_FINGERPRINT_KEYS:  # later resumes compare with the new store
            if key in fingerprint:
                stored[key] = fingerprint[key]
            else:
                stored.pop(key, None)
        say(f"manifest: store change recorded ({len(store_change['changed'])} setting(s) of the store's location and "
            f"identity; {store_change['pre_change_records']} attempt(s) made before it): {allow_store_change}")
    added = [name for name in fingerprint["routes"] if name not in recorded["routes"]]
    if added:
        stored["routes"].update({name: fingerprint["routes"][name] for name in added})  # the file's own form
        manifest.setdefault("models_added", []).append({"at": utc_now(), "models": added})
    manifest.setdefault("starts", []).append({"at": utc_now(), "harness": code,
                                              **({"code_change_allowed": True} if changed else {}),
                                              **(start_fields or {}),
                                              **({"store_changed": {k: store_change[k] for k in (
                                                  "at", "note", "pre_change_records")}} if store_change else {})})
    if changed:
        say(f"manifest: resumed under other code ({str(last.get('git_commit'))[:12]} -> "
            f"{str(code.get('git_commit'))[:12]}), allowed by --allow-code-change and recorded")
    _write_json_atomic(path, manifest)
    return added


# --- the latency baseline measured on calm calls ------------------------------------------


CALM_CALLS = 50  # calls (turns) per model
CALM_SYNC_MINUTES = ((6, 14), (36, 44))  # the Gateway front end's 30-minute sync, every UTC hour
CALM_EVEN_HOUR_SYNC_MINUTES = (52, 58)  # and its pair at even UTC hours
STALL_BIN_S = 120  # the stall tools' all-model stall bin (lead, 20 Oct)
STALL_FIRST_BYTE_S = 15.0
STALL_MIN_MODELS = 3
STALL_MIN_TURNS = 3
CALM_START_MARGIN_S = 1.0  # an attempt's in-flight mark is set this much before its first request


def calm_sync_minute(ts: float) -> bool:
    t = time.gmtime(ts)
    if any(lo <= t.tm_min <= hi for lo, hi in CALM_SYNC_MINUTES):
        return True
    lo, hi = CALM_EVEN_HOUR_SYNC_MINUTES
    return t.tm_hour % 2 == 0 and lo <= t.tm_min <= hi


def calm_turn_offset(rec: Any, live: list[dict]) -> float:
    """The wall-clock time a try's turn offsets (started_s) count from. A branch record's offsets run on from its
    replayed parent's, while its started_at is the retry's own start: its live turns are anchored at the retry's first
    request as stamped (load.started_at), else at started_at less the parent's replayed time (wall_s less the retry's
    own ended_at - started_at)."""
    from datetime import datetime

    t0 = datetime.fromisoformat(rec.started_at).timestamp()
    checker = getattr(rec, "checker", None)
    if not (checker and checker.get("branched_from")):
        return t0
    stamped = (getattr(rec, "load", None) or {}).get("started_at")
    if stamped:
        return datetime.fromisoformat(stamped).timestamp() - float(live[0]["started_s"])
    replayed = float(rec.wall_s) - (datetime.fromisoformat(rec.ended_at).timestamp() - t0)
    return t0 - replayed


def _utc_iso(ts: float) -> str:
    from datetime import UTC, datetime

    return datetime.fromtimestamp(ts, tz=UTC).isoformat(timespec="milliseconds")


class CalmBaseline:
    """The latency baseline of the models started with --latency-baseline-calm (design ruling, the design log,
    21 Oct 15:43: where a phase has no dev value, the baseline is measured live on the model's first 50 calm-period
    calls, never on the raw first window, and its latency step-down is inactive until then). A call is a live turn of
    a try of this start, placed at its request's start and timed as the guard times it (wall_s, else latency_s); it is
    calm when it starts outside the sync minutes and outside every all-model stall bin: a 2-minute UTC bin in which 3
    or more of this runner's models have a first-byte p90 (xs sorted, xs[min(n - 1, int(0.9 n))]) over 15 s with at
    least 3 turns each (the stall tools' definition). With fewer than 3 models in the runner no such bin can be told,
    and only the sync minutes are left out (said in the baseline's event). A bin is judged only once it is over and no
    attempt that started before its end is still running (each attempt is marked in flight from just before its first
    request until it is done), so no later turn can change it; the baseline is the median of the first 50 calm calls
    in time order, measured once per start."""

    def __init__(self, models: list[str], calm_models, calls: int = CALM_CALLS):
        self._lock = threading.Lock()
        self.models = sorted(set(models))
        self.calm_models = sorted(set(calm_models))
        self.stall_detection = len(self.models) >= STALL_MIN_MODELS
        self.calls_needed = calls
        self.in_flight: dict[Any, float] = {}
        self.first_bytes: dict[int, dict[str, list[float]]] = {}
        self.calls: dict[str, list[tuple[float, float]]] = {m: [] for m in self.calm_models}
        self.measured: set[str] = set()

    def started(self, key: Any, at: float) -> None:
        with self._lock:
            self.in_flight[key] = at

    def finished(self, key: Any) -> None:
        with self._lock:
            self.in_flight.pop(key, None)

    def add(self, name: str, tries: list) -> None:
        """Finished tries of a model, each given once (every model: the stall bins read them all)."""
        with self._lock:
            if len(self.measured) == len(self.calm_models):  # nothing left to measure: no more data is kept
                return
            for rec in tries:
                live = live_turns(rec)
                if not live:
                    continue
                offset = calm_turn_offset(rec, live)
                for t in live:
                    start = offset + float(t["started_s"])
                    fb = t.get("first_byte_s")
                    if isinstance(fb, (int, float)):
                        self.first_bytes.setdefault(int(start // STALL_BIN_S), {}).setdefault(name, []).append(
                            float(fb))
                    call = t.get("wall_s") or t.get("latency_s")
                    if name in self.calls and isinstance(call, (int, float)):
                        self.calls[name].append((start, float(call)))

    def _stall_bin(self, b: int) -> bool:
        over = 0
        for xs in self.first_bytes.get(b, {}).values():
            if len(xs) >= STALL_MIN_TURNS and sorted(xs)[min(len(xs) - 1, int(0.9 * len(xs)))] > STALL_FIRST_BYTE_S:
                over += 1
        return over >= STALL_MIN_MODELS

    def ready(self, now: float) -> dict[str, dict]:
        """The models whose first 50 calm calls are now settled: model -> the baseline and how it was measured."""
        out = {}
        with self._lock:
            horizon = min([now, *self.in_flight.values()])
            for m in self.calm_models:
                if m in self.measured:
                    continue
                taken, sync, stall = [], 0, 0
                for start, call in sorted(self.calls[m]):
                    b = int(start // STALL_BIN_S)
                    if (b + 1) * STALL_BIN_S > horizon:  # its bin may still change: so may every later one
                        break
                    if calm_sync_minute(start):
                        sync += 1
                    elif self.stall_detection and self._stall_bin(b):
                        stall += 1
                    else:
                        taken.append((start, call))
                        if len(taken) == self.calls_needed:
                            break
                if len(taken) == self.calls_needed:
                    self.measured.add(m)
                    out[m] = {"baseline_s": statistics.median(c for _, c in taken), "calls": self.calls_needed,
                              "first_call_at": _utc_iso(taken[0][0]), "last_call_at": _utc_iso(taken[-1][0]),
                              "skipped_sync": sync, "skipped_stall": stall, "stall_detection": self.stall_detection,
                              "models": len(self.models)}
        return out


# --- the order of attempts ----------------------------------------------------------------


def order_key(seed: int, attempt: tuple | list) -> str:
    """An attempt's place in the order: the sha256 of the seed and the attempt. It depends on nothing
    else, so attempts added to the plan (repetitions, models) never move the ones already in it."""
    return hashlib.sha256(f"{seed}|{json.dumps(list(attempt), separators=(',', ':'))}".encode("utf-8")).hexdigest()


def interleaved_order(models: list[str], arms: list[str], question_ids: list[str], reps: int,
                      seed: int) -> list[tuple[str, str, str, int]]:
    """Every attempt of the run as (model, arm, question id, repetition), in a seeded random order:
    sorted by ``order_key``. The order depends only on the seed and the attempts (not on the order of
    --models), a resume rebuilds it exactly, and a larger plan keeps the relative order of the
    attempts it shares with a smaller one (runner review)."""
    plan = {(m, a, q, rep) for m in models for a in arms for q in question_ids for rep in range(reps)}
    return sorted(plan, key=lambda attempt: order_key(seed, attempt))


def order_sha256(order: list) -> str:
    return hashlib.sha256(json.dumps([list(x) for x in order], separators=(",", ":")).encode("utf-8")).hexdigest()


def default_order_seed(questions: Path) -> int:
    return int(hashlib.sha256(questions.read_bytes()).hexdigest()[:16], 16)


def recorded_order_seed(out: Path) -> int | None:
    path = out / "manifest.json"
    if not path.exists():
        return None
    order = json.loads(path.read_text(encoding="utf-8")).get("order")
    return order.get("seed") if isinstance(order, dict) else None


def record_order(out: Path, *, seed: int, seed_source: str, order: list, resumed: bool,
                 say: Callable[[str], None], phase: str | None = None) -> None:
    """Record this start's order: the seed and rule in the manifest (outside the fingerprint, since a
    resume may grow the plan) with a history entry per distinct order, and the order itself in
    ``OUT/order.json``. Refuses a resume under another rule, and one whose order puts the attempts it
    shares with the recorded order in another relative order: the remaining attempts would silently
    run in a different sequence. A grid's ``phase`` has its own seed and order: the manifest's
    ``grid_order.<phase>`` and ``OUT/order-<phase>.json``."""
    path = out / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    entry = {"at": utc_now(), "attempts": len(order), "plan_sha256": order_sha256(sorted(order)),
             "order_sha256": order_sha256(order)}
    block = manifest.get("order") if phase is None else (manifest.get("grid_order") or {}).get(phase)
    order_file = out / ("order.json" if phase is None else f"order-{phase}.json")
    if block is None:
        if resumed:
            say(f"manifest: no order recorded (the run started before seeded interleaving); attempts already "
                f"recorded ran in the earlier order, the rest follow seed {seed}")
        block = {"seed": seed, "seed_source": seed_source, "rule": ORDER_RULE, "history": []}
    elif block.get("rule") != ORDER_RULE:
        raise Refused(f"{out} was ordered by another rule ({block.get('rule') or block.get('shuffle')!r}), so its "
                      f"order cannot be kept; use a new --out")
    elif block.get("seed") != seed:
        raise Refused(f"{out} was started with order seed {block.get('seed')}; resume needs the same --order-seed")
    if order_file.exists():
        recorded = [tuple(x) for x in json.loads(order_file.read_text(encoding="utf-8"))["order"]]
        shared = set(recorded) & set(order)
        if [x for x in recorded if x in shared] != [x for x in order if x in shared]:
            raise Refused(f"the {len(shared)} attempts this start shares with {order_file.name} come out in another "
                          f"relative order: the order does not reproduce here; use a new --out")
    if block["history"] and block["history"][-1]["order_sha256"] == entry["order_sha256"] and order_file.exists():
        return
    block["history"].append(entry)
    if phase is None:
        manifest["order"] = block
    else:
        manifest.setdefault("grid_order", {})[phase] = block
    _write_json_atomic(order_file, {"seed": seed, "rule": ORDER_RULE, "order": [list(x) for x in order]})
    _write_json_atomic(path, manifest)


# --- concurrency and guard rails -------------------------------------------------------------


class Throttle:
    """Attempts in flight per model, each under an allowed size that starts at the model's
    configured workers and only ever goes down: a semaphore per model that can shrink (a
    threading.Semaphore cannot). A step-down lets running attempts finish."""

    def __init__(self, configured: dict[str, int], allowed: dict[str, int] | None = None):
        self._cond = threading.Condition()
        self.configured = dict(configured)
        self.allowed = {name: min(n, (allowed or {}).get(name, n)) for name, n in configured.items()}
        self.in_flight = dict.fromkeys(configured, 0)

    def acquire(self, name: str) -> None:
        with self._cond:
            while self.in_flight[name] >= self.allowed[name]:
                self._cond.wait()
            self.in_flight[name] += 1

    def release(self, name: str) -> None:
        with self._cond:
            self.in_flight[name] -= 1
            self._cond.notify_all()

    def snapshot(self, name: str) -> dict[str, int]:
        with self._cond:
            return {"workers_configured": self.configured[name], "workers_allowed": self.allowed[name],
                    "in_flight_model": self.in_flight[name], "in_flight_total": sum(self.in_flight.values())}

    def step_down(self, name: str) -> tuple[int, int]:
        """Allow a third fewer attempts in flight (at least 1); returns (from, to)."""
        with self._cond:
            old = self.allowed[name]
            self.allowed[name] = max(1, old * 2 // 3)
            return old, self.allowed[name]


class StartGate:
    """Starts each model's attempts strictly in their order, and numbers the starts that happen.

    The dispatcher hands a model's attempts to its workers with consecutive tickets. A worker waits
    (``wait``) until its ticket is the model's next; the ticket passes on only when its attempt is
    at its first provider request (``start``, called by run_attempt's ``on_start`` just before it),
    or ends without one (``skip``: a model stopped, the attempt claimed elsewhere, an exception, an
    attempt with no request), so a worker paused anywhere before its request holds the next one
    back, and no ticket can hold it up for good. ``start`` assigns the attempt's index among every
    attempt this runner started and among its model's, runs ``at_start`` (the throttle snapshot and
    the timestamp) while still holding the gate, and releases the gate as its last step. What is
    left between the release and the HTTP send (the loop's own code and the client's request
    building) is the accepted residual: nothing in it waits on another worker."""

    def __init__(self, names: list[str]):
        self._cond = threading.Condition()
        self.next_ticket = dict.fromkeys(names, 0)
        self.started = dict.fromkeys(names, 0)
        self.started_total = 0

    def wait(self, name: str, ticket: int) -> None:
        with self._cond:
            while self.next_ticket[name] != ticket:
                self._cond.wait()

    def _pass(self, name: str, ticket: int) -> None:
        assert self.next_ticket[name] == ticket, "a ticket passes only on its turn"
        self.next_ticket[name] += 1
        self._cond.notify_all()

    def start(self, name: str, ticket: int, at_start: Callable[[], None] | None = None) -> tuple[int, int]:
        with self._cond:
            index = (self.started_total, self.started[name])
            if at_start is not None:
                at_start()
            self.started_total += 1
            self.started[name] += 1
            self._pass(name, ticket)  # the last step: the next attempt may start now
            return index

    def skip(self, name: str, ticket: int) -> None:
        with self._cond:
            while self.next_ticket[name] != ticket:
                self._cond.wait()
            self._pass(name, ticket)


def _call_times(rec: Any) -> list[float]:
    # The call time: the request with its retries and backoff (wall_s). A route that does not time the
    # whole call leaves wall_s at 0; its request time is then the best measure there is.
    return [float(t.get("wall_s") or t["latency_s"]) for t in live_turns(rec)
            if isinstance(t.get("wall_s") or t.get("latency_s"), (int, float))]


def _infra_source(rec: Any) -> str | None:
    if rec.outcome.get("failure_type") != "infrastructure":
        return None
    return rec.outcome.get("infra_source") or "store"


def attempt_stats(final: Any, failed_tries: list, in_flight: int) -> dict:
    """What the guard rails read from one finished attempt: its final record, and the tries before it
    that failed infrastructure (the final try of an attempt that failed on every try is among them)."""
    tries = [t for t in failed_tries if t is not final]
    per_turn = live_turns(final)  # a branch record's replayed initial phase is its parent's
    return {"cap": final.outcome.get("cap"),  # turns or wall
            "cap_stage": final.outcome.get("stage") if final.outcome.get("cap") else None,  # provider: in a retry
            "truncated": any(t.get("truncated") for t in per_turn),  # a reply stopped at the output limit
            "infrastructure": sorted({src for t in [*tries, final] if (src := _infra_source(t))}),
            "final_infra": _infra_source(final) is not None,  # not an agent outcome
            "call_s": [x for t in [*tries, final] for x in _call_times(t)],  # every turn of every try
            "in_flight": in_flight}


class GuardRails:
    """The step-down rule for one model (the design log, design ruling). Three triggers, each
    over the attempts that started at the current size:

    * latency: the median per-turn call time (retries and backoff included) over every turn of
      every try of the last 50 attempts is over twice the baseline (without one, the first full
      window's median becomes it);
    * wall-cap rate: of the last 200 attempts that ended as agent outcomes (not infrastructure
      failures), the share capped by the wall clock is more than one percentage point over the
      model's rate on the same tier at low concurrency (inactive without that baseline);
    * infrastructure: of the last 50 attempts, more than 2% had a try that failed infrastructure
      (the store or the provider), even when a re-run then succeeded.

    Turn caps, context caps and truncated replies are the agent's behaviour: counted, never acted on.
    Every window starts again after a step-down, so what caused one cannot cause the next, and an
    attempt that started before it (``epoch``) is no evidence about the new size."""

    def __init__(self, baseline_s: float | None = None, wallcap_baseline: float | None = None,
                 first_window: bool = True):
        self.recent: deque[dict] = deque(maxlen=max(LATENCY_WINDOW, INFRA_WINDOW))
        # False: the baseline is measured on calm calls (CalmBaseline) and set from outside; until then the latency
        # trigger is inactive (design ruling, design log, 21 Oct 15:43)
        self.first_window = first_window
        self.outcomes: deque[dict] = deque(maxlen=WALLCAP_WINDOW)
        self.baseline_s = baseline_s
        self.wallcap_baseline = wallcap_baseline
        self.epoch = 0

    def stats(self) -> dict:
        latency = list(self.recent)[-LATENCY_WINDOW:]
        infra = list(self.recent)[-INFRA_WINDOW:]
        outcomes = list(self.outcomes)
        calls = [x for a in latency for x in a["call_s"]]
        median = statistics.median(calls) if calls else None
        sources = Counter(src for a in infra for src in a["infrastructure"])
        failed = sum(1 for a in infra if a["infrastructure"])
        wall = [a for a in outcomes if a["cap"] == "wall"]
        return {"latency_attempts": len(latency), "turns": len(calls), "call_median_s": median,
                "baseline_s": self.baseline_s,
                "latency_ratio": median / self.baseline_s if median is not None and self.baseline_s else None,
                "infrastructure_attempts": len(infra), "infrastructure_failed": failed,
                "infrastructure": dict(sorted(sources.items())),
                "infrastructure_rate": failed / len(infra) if infra else 0.0,
                "wallcap_attempts": len(outcomes), "wallcap_baseline": self.wallcap_baseline,
                "wallcap_rate": len(wall) / len(outcomes) if outcomes else 0.0,
                # the agent outcomes over the same attempts, for the record only
                "capped_wall": len(wall), "capped_wall_provider": sum(a["cap_stage"] == "provider" for a in wall),
                "capped_turns": sum(a["cap"] == "turns" for a in outcomes),
                "capped_context": sum(a["cap"] == "context" for a in outcomes),
                "truncated": sum(bool(a["truncated"]) for a in outcomes),
                "in_flight_median": statistics.median(a["in_flight"] for a in latency) if latency else None}

    def observe(self, stats: dict, epoch: int) -> list[dict]:
        """Take one finished attempt; returns the events it causes ("baseline", "step_down")."""
        if epoch != self.epoch:
            return []
        self.recent.append(stats)
        if not stats["final_infra"]:
            self.outcomes.append(stats)
        events, triggers, reasons = [], [], []
        window = self.stats()
        if (self.baseline_s is None and self.first_window and window["latency_attempts"] >= LATENCY_WINDOW
                and window["call_median_s"]):
            self.baseline_s = window["call_median_s"]
            window = self.stats()
            events.append({"kind": "baseline", "reason": f"median per-turn call time of the first "
                                                         f"{LATENCY_WINDOW} attempts", "window": window})
        ratio = window["latency_ratio"]
        if window["latency_attempts"] >= LATENCY_WINDOW and ratio is not None and ratio > LATENCY_FACTOR:
            triggers.append("latency")
            reasons.append(f"median per-turn call time {window['call_median_s']:.3g} s is {ratio:.2f}x the "
                           f"baseline {self.baseline_s:.3g} s (> {LATENCY_FACTOR:g}x)")
        if (self.wallcap_baseline is not None and window["wallcap_attempts"] >= WALLCAP_WINDOW
                and window["wallcap_rate"] > self.wallcap_baseline + WALLCAP_MARGIN + 1e-9):
            triggers.append("wall_cap_rate")
            reasons.append(f"wall caps {window['capped_wall']} of {window['wallcap_attempts']} attempts "
                           f"({window['wallcap_rate']:.1%}, {window['capped_wall_provider']} in a provider retry) "
                           f"> baseline {self.wallcap_baseline:.1%} + {100 * WALLCAP_MARGIN:g} point")
        if window["infrastructure_attempts"] >= INFRA_WINDOW and window["infrastructure_rate"] > INFRA_RATE_LIMIT:
            triggers.append("infrastructure")
            reasons.append(f"attempts with an infrastructure failure {window['infrastructure_failed']} of "
                           f"{window['infrastructure_attempts']} ({window['infrastructure_rate']:.1%}: "
                           f"{window['infrastructure']}) > {INFRA_RATE_LIMIT:.0%}")
        if triggers:
            events.append({"kind": "step_down", "triggers": triggers, "reason": "; ".join(reasons),
                           "window": window})
            self.epoch += 1
            self.recent.clear()
            self.outcomes.clear()
        return events


def read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --- the lock and the claims --------------------------------------------------------------

NETWORK_FS_TYPES = frozenset({"nfs", "nfs4", "ceph", "cifs", "smb3", "smbfs", "lustre"})


def filesystem_type(path: Path, mounts: Path = Path("/proc/mounts")) -> str | None:
    """The type of the file system holding ``path`` (the longest mount point above it), or None."""
    try:
        lines = mounts.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    target = os.path.realpath(path)
    best, best_type = "", None
    for line in lines:
        fields = line.split()
        if len(fields) < 3:
            continue
        point = fields[1].replace("\\040", " ")
        inside = target == point or target.startswith(point.rstrip("/") + "/")
        if inside and len(point) >= len(best):
            best, best_type = point, fields[2]
    return best_type


def is_network_fs(fs_type: str | None) -> bool:
    return bool(fs_type) and (fs_type in NETWORK_FS_TYPES or fs_type.startswith("fuse.ceph"))


def acquire_lock(out: Path) -> int:
    fd = os.open(out / ".lock", os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise Refused(f"another runner holds {out / '.lock'}; refusing to run two runners on one --out") from None
    except OSError as exc:
        os.close(fd)
        raise Refused(f"cannot lock {out / '.lock'}: the file system does not support flock ({exc})") from None
    return fd


def release_lock(fd: int) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


# --- harness defects and a change of the served model (design log, change-log entry 91) -------------------------------

SETASIDE_SCHEMA = "setaside/1"  # the set-aside decision, as scripts/relabel_harness_failure.py writes it
HARNESS_DEFECT_RULING = ("automatic: a crash inside the harness's own tool code is a harness defect, never the "
                         "agent's failure (design ruling in the design log, change-log entry 91)")
SERVED_MODELS_FILE = "served-models.json"
MODEL_CHANGED_FILE = "MODEL_CHANGED.json"
# Change-log entry 167 (science, design log, 08:40-08:58Z): the fingerprints of the endpoints admitted to serve a model in
# this OUT, appended by the admission tool (whole file renamed into place) and never shrunk
ADMITTED_FILE = "admitted-fingerprints.json"
ADMITTED_SCHEMA = "admitted-fingerprints/1"


def read_admitted(path: Path) -> dict[str, list[str]]:
    """Each model's admitted system fingerprints in ``path`` (OUT/admitted-fingerprints.json), sorted, by the runner's
    model name; {} when there is no file. ValueError when it is malformed anywhere: no admitted-fingerprints/1 object,
    a model without an ``admitted`` list, or an entry without one non-empty ``system_fingerprint`` (several joined
    included). Copied in scripts/fault_probe.py and scripts/infra_fault_monitor.py (read_admitted, which must stay the
    same; tests/test_e167_admitted_fingerprints.py holds the three to one result)."""
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return {}
    doc = json.loads(raw)
    if not isinstance(doc, dict) or doc.get("schema") != ADMITTED_SCHEMA or not isinstance(doc.get("models"), dict):
        raise ValueError(f"not an {ADMITTED_SCHEMA} file with a models object")
    out = {}
    for model, entry in doc["models"].items():
        admitted = entry.get("admitted") if isinstance(entry, dict) else None
        if not isinstance(admitted, list):
            raise ValueError(f"{model}: no admitted list")  # noqa: TRY004 - the file's content
        fingerprints = set()
        for item in admitted:
            fp = item.get("system_fingerprint") if isinstance(item, dict) else None
            if not isinstance(fp, str) or not fp or MIXED_FINGERPRINTS in fp:
                raise ValueError(f"{model}: an admitted entry without one system fingerprint")
            fingerprints.add(fp)
        out[model] = sorted(fingerprints)
    return out


def write_harness_defect_setaside(harness_dir: Path, record_name: str, rec: RunRecord) -> Path:
    """The set-aside decision beside a failed:harness_defect record in harness/ (relabel's sidecar, setaside/1),
    named after the record; its temporary file is named so too."""
    defect = rec.outcome["harness_defect"]
    decision = {"schema": SETASIDE_SCHEMA, "run_id": rec.run_id, "record": record_name,
                "question_id": rec.question_id, "model": rec.model, "repetition": rec.repetition,
                "label": "failed:harness", "original_label": rec.outcome["label"],
                "original_failure_type": rec.outcome["failure_type"],
                "defect": f"{defect['exception']} (stack {defect['stack_sha256']}): {rec.outcome['error_message']}",
                "ruling": HARNESS_DEFECT_RULING, "set_aside_at": utc_now()}
    stem = record_name.removesuffix(".json")
    path = harness_dir / f"{stem}.setaside.json"
    fd, tmp = tempfile.mkstemp(dir=harness_dir, prefix=f"{stem}.setaside.", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(_redacted_json(decision), ensure_ascii=False, indent=1, sort_keys=True))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


class ServedModels:
    """Each model's first served model id in an OUT (OUT/served-models.json, kept across resumes), and every change
    from it (OUT/MODEL_CHANGED.json). The first id is the one of the model's first served turn, whichever attempt it
    belongs to: ``check`` is called for every turn as it arrives (run_attempt's ``served_check``) and, under one lock,
    sets it and writes it to disk before any other turn of the model is checked. A change is any turn whose served id
    is not exactly the first one (a dated snapshot of the same id included, which the model-mismatch check takes for
    the same model); a turn whose provider names no model is not one.

    The server's system fingerprint likewise (design ruling in the design log Oct 23:24): the model's first fingerprinted
    turn sets its ``system_fingerprint`` here, and any later turn with another one is a change, compared as the whole
    string (the 20 Oct probes, www2027-reviews/gemma-probe-20261020, saw one fingerprint per model across its nodes,
    "vllm-<version>-tp<degree>-<hash>"; the node is the reply's separate ``_consumer_node_name``). A turn without a
    fingerprint (some servers send none) is no change and sets nothing; a turn whose provider names no model is still
    held to the model's fingerprint.

    Endpoints joined to a running OUT (change-log entry 167; science, design log, 08:40-08:58Z): a turn whose fingerprint
    differs from the model's first passes when the first and it are both in the model's admitted set (ADMITTED_FILE,
    read again whenever the file is another: inode, modification time or size, so an admission applies from the next
    turn). A reply carrying several fingerprints is a change whatever is admitted, and the first fingerprint is set as
    before, admitted or not. A malformed file is ignored whole (a warning through ``say``, once per version of the
    file), which is the check as it was; so is a missing file or a model the file does not name. A fingerprint change of
    a model with an admitted set adds ``admitted_system_fingerprints`` (the set it was checked against, sorted) to the
    change, from which replay rebuilds the check (harness/agent/replay.py). Replay needs the set to only grow while
    attempts run: an entry removed, or the file made malformed (which empties the set), would leave an earlier turn that
    passed on an admitted fingerprint failing on replay."""

    _UNREAD = object()

    def __init__(self, out: Path, say: Callable[[str], None] | None = None):
        self.out = out
        self.path = out / SERVED_MODELS_FILE
        self.lock = threading.Lock()
        data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        self.models: dict[str, dict] = dict(data.get("models") or {})
        self.say = say or (lambda msg: None)
        self.admitted_path = out / ADMITTED_FILE
        self._admitted_key: Any = self._UNREAD
        self._admitted: dict[str, list[str]] = {}

    def admitted(self, name: str) -> list[str]:
        """The model's admitted fingerprints now, sorted ([] for none); called under the lock (the copy read is
        shared by the workers)."""
        try:
            st = self.admitted_path.stat()
        except FileNotFoundError:
            key = None
        except OSError as exc:  # read below, and refused there with its reason
            key = ("unreadable", type(exc).__name__)
        else:
            key = (st.st_ino, st.st_mtime_ns, st.st_size)
        if key != self._admitted_key:
            self._admitted_key = key
            try:
                self._admitted = read_admitted(self.admitted_path) if key is not None else {}
            except (OSError, ValueError) as exc:
                self._admitted = {}
                self.say(f"warning: {self.admitted_path} is ignored ({type(exc).__name__}: {exc}): no fingerprint is "
                         f"admitted beside each model's first until it is repaired (change-log entry 167)")
        return self._admitted.get(name, [])

    def check(self, name: str, route_model: str, served: str | None, fingerprint: str | None = None, *,
              question_id: str | None = None, repetition: int | None = None) -> dict | None:
        """None, or the outcome's ``model_changed``: {"first_served_model", "served_model"} when this turn's id differs
        from the model's first, and with "first_system_fingerprint" and "system_fingerprint" when its fingerprint
        differs from the model's first or is several joined (one reply from servers that differ: a change by itself,
        never a baseline), unless both fingerprints are admitted (the class docstring; the change then adds
        "admitted_system_fingerprints" when the model has an admitted set). The model's first turn naming a model sets
        its first id, and its first turn with one fingerprint sets that (each written before the next turn is checked,
        whichever comes first), and passes."""
        mixed = bool(fingerprint) and MIXED_FINGERPRINTS in fingerprint
        with self.lock:
            entry = self.models.get(name)
            if entry is None:
                if not served and (not fingerprint or mixed):  # nothing to set a baseline from
                    return None if not mixed else {"first_served_model": None, "served_model": served,
                                                   "first_system_fingerprint": None, "system_fingerprint": fingerprint}
                entry = self.models[name] = {"served_model": served or None, "route_model": route_model,
                                             "question_id": question_id, "repetition": repetition, "at": utc_now()}
                if fingerprint and not mixed:
                    entry["system_fingerprint"] = fingerprint
                _write_json_atomic(self.path, {"schema": "served-models/1", "models": self.models})
                if not mixed:
                    return None
            first = entry["served_model"]
            if served and first is None:  # the fingerprint came first: the first turn naming a model sets its id
                entry.update(served_model=served, served_question_id=question_id, served_repetition=repetition,
                             served_at=utc_now())
                _write_json_atomic(self.path, {"schema": "served-models/1", "models": self.models})
                first = served
            if served and served != first:
                return {"first_served_model": first, "served_model": served}
            if not fingerprint:
                return None
            known = entry.get("system_fingerprint")
            if known is None and not mixed:
                entry.update(system_fingerprint=fingerprint, system_fingerprint_question_id=question_id,
                             system_fingerprint_repetition=repetition, system_fingerprint_at=utc_now())
                _write_json_atomic(self.path, {"schema": "served-models/1", "models": self.models})
                return None
            if fingerprint == known and not mixed:
                return None
            admitted = self.admitted(name)
        if not mixed and known in admitted and fingerprint in admitted:  # entry 167: both admitted endpoints
            return None
        change = {"first_served_model": first, "served_model": served, "first_system_fingerprint": known,
                  "system_fingerprint": fingerprint}
        if admitted:
            change["admitted_system_fingerprints"] = list(admitted)
        return change

    def mark_change(self, name: str, route_model: str, rec: RunRecord, changed: dict) -> None:
        with self.lock:
            path = self.out / MODEL_CHANGED_FILE
            data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"schema": "model-changed/1",
                                                                                       "changes": []}
            data["changes"].append({"model": name, "route_model": route_model, **changed, "run_id": rec.run_id,
                                    "question_id": rec.question_id, "repetition": rec.repetition, "at": utc_now()})
            _write_json_atomic(path, data)


def claim_digest(key: tuple[str, str, int]) -> str:
    """The attempt's name in OUT: it names its claim, its marker and every file written for it."""
    return hashlib.sha256(json.dumps(list(key), ensure_ascii=False).encode("utf-8")).hexdigest()[:40]


def claim_path(claims_dir: Path, key: tuple[str, str, int]) -> Path:
    return claims_dir / f"{claim_digest(key)}.claim"


DEPLOYMENT_CHANGE_RULE = ("design ruling in the design log Oct 23:24: a redeploy or move of a model before the freeze is "
                          "a recorded deployment change, identity check first, its time logged")


def deployment_changes(out: Path, names: list[str], ruling: str | None, running: list[str],
                       route_models: dict[str, str]) -> list[dict]:
    """The recorded deployment changes a start's --deployment-change asks for (DEPLOYMENT_CHANGE_RULE), checked before
    the start writes anything (before its identity checks, which may write the manifest; it is recorded after them);
    Refused unless each is one. A model's new baseline is the one system fingerprint the runner itself recorded for
    it in MODEL_CHANGED.json since its current baseline: the entries after the last one an earlier change of the
    model consumed, whose changes are from that baseline fingerprint on turns of the model's own served id (or
    naming none), and no served-id change since the baseline was set. A reply carrying several fingerprints joined
    passes only when they are the old and the new one. Refused too: a model this start does not run, a model with no
    baseline fingerprint, no recorded change (nothing to re-baseline: the no-op is refused, not recorded), and a
    model whose change under this ruling is already recorded (a second change needs its own ruling). A change the
    manifest records but served-models.json does not yet hold (the start stopped between the two writes) is
    completed by the same option and ruling: ``{"complete": True, ...}``."""
    if not names:
        return []

    def read(name: str) -> dict:
        path = out / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    served, changes = read(SERVED_MODELS_FILE).get("models") or {}, read(MODEL_CHANGED_FILE).get("changes") or []
    recorded = read("manifest.json").get("deployment_changes") or []
    plans = []
    for name in names:
        what = f"--deployment-change {name}"
        if name not in running:
            raise Refused(f"{what}: this start does not run {name}")
        entry = served.get(name) or {}
        old, own = entry.get("system_fingerprint"), entry.get("served_model")
        if not old:
            raise Refused(f"{what}: {name} has no system fingerprint baseline in {SERVED_MODELS_FILE}, so there is "
                          f"nothing to re-baseline")
        mine = [d for d in recorded if d.get("model") == name]
        applied = entry.get("deployment_changes") or []
        again = [d for d in mine if d.get("ruling") == ruling]
        if again and not any(a.get("ruling") == ruling for a in applied) \
                and again[-1].get("from_system_fingerprint") == old:  # recorded, its baseline never written
            plans.append({**again[-1], "complete": True})
            continue
        if again or any(a.get("ruling") == ruling for a in applied):
            raise Refused(f"{what}: a deployment change of {name} is already recorded under this ruling; a second "
                          f"change needs its own ruling")
        since = entry.get("system_fingerprint_at") or entry.get("at") or ""
        after = max((i for d in mine + applied for i in d.get("model_changed_entries") or []), default=-1)
        seen = [(i, c) for i, c in enumerate(changes) if i > after and c.get("model") == name and (
            c.get("first_system_fingerprint") == old if "system_fingerprint" in c else (c.get("at") or "") >= since)]
        moved = [c for _, c in seen if "system_fingerprint" not in c or c.get("served_model") not in (None, own)]
        if moved:
            raise Refused(f"{what}: the served id changed from {own!r} to {moved[0].get('served_model')!r} (run "
                          f"{moved[0].get('run_id')}); a deployment change keeps the served id, only the fingerprint "
                          f"may change")
        single = sorted({c["system_fingerprint"] for _, c in seen if MIXED_FINGERPRINTS not in c["system_fingerprint"]})
        if not single:
            raise Refused(f"{what}: no change of {name}'s system fingerprint from {old!r} is recorded in "
                          f"{MODEL_CHANGED_FILE} (the runner records it at the first turn that carries another), so "
                          f"there is nothing to re-baseline")
        if len(single) > 1:
            raise Refused(f"{what}: turns carried several new fingerprints ({', '.join(single)}); a new baseline is "
                          f"one fingerprint")
        new = single[0]
        joined = [c for _, c in seen if not set(c["system_fingerprint"].split(MIXED_FINGERPRINTS)) <= {old, new}]
        if joined:
            raise Refused(f"{what}: a reply carried several fingerprints ({joined[0]['system_fingerprint']!r}, run "
                          f"{joined[0].get('run_id')}), not only {old!r} and {new!r}")
        evidence = [{k: c.get(k) for k in ("run_id", "question_id", "repetition", "at", "served_model",
                                           "system_fingerprint")} for _, c in seen]
        plans.append({"model": name, "route_model": route_models[name], "served_model": own,
                      "from_system_fingerprint": old, "to_system_fingerprint": new, "ruling": ruling,
                      "rule": DEPLOYMENT_CHANGE_RULE, "evidence": evidence,
                      "model_changed_entries": [i for i, _ in seen]})
    return plans


def deployments_after_freeze(out: Path, plans: list[dict], say: Callable[[str], None]) -> list[dict]:
    """The checked deployment changes of a start (deployment_changes), in an OUT after the freeze: none continues in
    place (change-log entry 129, item 3). Each is a served-version change of its model: recorded in the manifest's
    served_version_changes first (once per model and ruling, a model already held included), then the model is held
    through its served-version hold file, with an alert (served_version_holds), and served-models.json is left as it
    is. Returns the plans to record as deployment changes: all of them in a development OUT, none after the freeze."""
    if not plans:
        return plans
    path = out / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    frozen = frozen_out(manifest.get("fingerprint") or {})
    if frozen is None:
        return plans
    commit = str(frozen.get("commit"))
    recorded = manifest.get("served_version_changes") or []
    new = []
    for plan in plans:
        m = plan["model"]
        # once per model and ruling: a model already held (a setting changed after the freeze, an earlier deployment
        # change) still gets this change's evidence recorded, and alerted
        if any(c.get("model") == m and (c.get("deployment_change") or {}).get("ruling") == plan.get("ruling")
               for c in recorded + new):
            continue
        what = (f"deployment change: system fingerprint {plan.get('from_system_fingerprint')!r} -> "
                f"{plan.get('to_system_fingerprint')!r} (--deployment-change, {plan.get('ruling')})")
        new.append({"model": m, "route_model": plan.get("route_model"), "at": utc_now(), "differences": [what],
                    "freeze_commit": commit,
                    "rule": SERVED_VERSION_AFTER_FREEZE.format(model=m, commit=commit[:12],
                                                               hold="its served-version hold"),
                    "harness": harness_commit(),
                    "deployment_change": {k: plan.get(k) for k in ("from_system_fingerprint", "to_system_fingerprint",
                                                                   "served_model", "ruling", "evidence",
                                                                   "model_changed_entries")}})
    if new:  # the record first: a stop before the hold leaves it recorded, and the next start holds the model
        manifest.setdefault("served_version_changes", []).extend(new)
        _write_json_atomic(path, manifest)
    served_version_holds(out, frozen, {e["model"]: e["differences"] for e in new}, {p["model"] for p in plans}, new,
                         say)
    return []


def record_deployment_changes(out: Path, plans: list[dict], say: Callable[[str], None]) -> None:
    """Record checked deployment changes (deployment_changes) as a start's last step before any attempt: each in the
    manifest's ``deployment_changes`` (with its time and the harness commit), then the model's new baseline in
    served-models.json (its fingerprint, set at this time by the first turn seen on it, and the change in the entry's
    own ``deployment_changes``, which ServedModels keeps)."""
    if not plans:
        return
    at, code = utc_now(), harness_commit()
    done = [plan if plan.get("complete") else {**plan, "at": at, "harness": code} for plan in plans]
    path = out / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.setdefault("deployment_changes", []).extend(change for change in done if not change.get("complete"))
    _write_json_atomic(path, manifest)  # the decision first: a stop before the next write is completed by a re-run
    served_path = out / SERVED_MODELS_FILE
    served = json.loads(served_path.read_text(encoding="utf-8"))
    for change in done:
        entry, first = served["models"][change["model"]], change["evidence"][0]
        entry.update(system_fingerprint=change["to_system_fingerprint"], system_fingerprint_at=change["at"],
                     system_fingerprint_question_id=first["question_id"],
                     system_fingerprint_repetition=first["repetition"])
        entry.setdefault("deployment_changes", []).append(
            {k: change[k] for k in ("at", "from_system_fingerprint", "to_system_fingerprint", "ruling",
                                    "model_changed_entries")})
    _write_json_atomic(served_path, served)
    for change in done:
        say(f"deployment change: {change['model']}'s system fingerprint baseline moved from "
            f"{change['from_system_fingerprint']!r} to {change['to_system_fingerprint']!r} at {change['at']} "
            f"({change['ruling']}{'; completing the change recorded then' if change.get('complete') else ''}; "
            f"{len(change['evidence'])} recorded turn(s) on it); recorded in manifest.json and {SERVED_MODELS_FILE}, "
            f"and its turns are held to the new fingerprint from now")


# Every file the runner writes for an attempt (records/, infrastructure/, aborted/, and harness/ once set aside, its
# sidecar too) is named "<claim digest>.<run id>.json" (the sidecar ".setaside.json"), and the attempt's marker
# OUT/claims/<claim digest>.attempt (ATTEMPT_MARKER_SCHEMA: its key and split), written before its first try and never
# deleted, outlives its claim (an aborted attempt's claim is deleted so it can be retried, and so is a set-aside one's).
# So a reader places every file in its attempt and split by its name and the marker, without opening it (the
# analysis's test-run guard, design log). A temporary file a crash leaves behind is named the same way, with
# ".<random>.tmp" in place of ".json" (RunRecord.write), so it is placed by its name too; no reader takes it for a run
# file. Files of runs before this have "<run id>.json" and no marker; every reader still takes them.
ATTEMPT_MARKER_SUFFIX = ".attempt"
ATTEMPT_MARKER_SCHEMA = "attempt-marker/1"
_RUN_FILE_NAME = re.compile(
    r"(?P<digest>[0-9a-f]{40})\.(?P<run_id>[^./]+)(?:\.setaside)?(?:\.json|\.[A-Za-z0-9_]+\.tmp)")


def attempt_marker_path(claims_dir: Path, key: tuple[str, str, int]) -> Path:
    return claims_dir / f"{claim_digest(key)}{ATTEMPT_MARKER_SUFFIX}"


def write_attempt_marker(claims_dir: Path, key: tuple[str, str, int], split: str) -> Path:
    """The attempt's permanent marker (its key and split); an existing one must say the same."""
    path = attempt_marker_path(claims_dir, key)
    marker = {"schema": ATTEMPT_MARKER_SCHEMA, "key": list(key), "split": split}
    if path.exists():
        found = json.loads(path.read_text(encoding="utf-8"))
        if found != marker:
            raise Refused(f"{path.name} names {found.get('key')} in split {found.get('split')!r}, not {list(key)} in "
                          f"split {split!r}")
        return path
    _write_json_atomic(path, marker)
    return path


def run_file_attempt(out: Path, name: str) -> dict | None:
    """The marker of the attempt a run file under OUT belongs to, from the file's name alone (the file is not
    opened); None for a name without a claim digest (a run before attempt markers) or a digest without a marker."""
    m = _RUN_FILE_NAME.fullmatch(name)
    if m is None:
        return None
    path = out / "claims" / f"{m['digest']}{ATTEMPT_MARKER_SUFFIX}"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _read_claim(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) and isinstance(data.get("key"), list) else None
    except (OSError, ValueError):  # a claim cut off mid-write: it still blocks its path
        return None


def _claim_key(path: Path) -> tuple | None:
    data = _read_claim(path)
    return tuple(data["key"]) if data else None


def claimed_keys(claims_dir: Path) -> set[tuple[str, str, int]]:
    return {key for path in claims_dir.glob("*.claim") if (key := _claim_key(path)) is not None}


def _pid_alive(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:  # it exists but belongs to someone else
        return True
    return True


def lease_is_stale(path: Path, *, timeout_s: float, now: float | None = None, host: str | None = None) -> bool:
    """A claim is a lease: stale when its heartbeat is older than ``timeout_s``, or when it was
    taken on this host by a process that no longer exists. An unreadable claim ages by mtime."""
    now = time.time() if now is None else now
    host = socket.gethostname() if host is None else host
    data = _read_claim(path)
    if data is None:
        try:
            return now - path.stat().st_mtime > timeout_s
        except OSError:
            return False
    beat = data.get("heartbeat_at", data.get("claimed_at"))
    if not isinstance(beat, (int, float)) or now - beat > timeout_s:
        return True
    return data.get("host") == host and not _pid_alive(data.get("pid"))


def reclaim(claims_dir: Path, done: set[tuple[str, str, int]], *, timeout_s: float,
            now: float | None = None, host: str | None = None) -> int:
    """Delete stale leases of attempts that have no record; live leases and recorded attempts keep theirs."""
    removed = 0
    for path in claims_dir.glob("*.claim"):
        if _claim_key(path) in done:
            continue
        if lease_is_stale(path, timeout_s=timeout_s, now=now, host=host):
            path.unlink(missing_ok=True)
            removed += 1
    return removed


def _lease(key: tuple[str, str, int], claimed_at: float, heartbeat_at: float, split: str | None = None) -> dict:
    lease = {"key": list(key), "pid": os.getpid(), "host": socket.gethostname(), "claimed_at": claimed_at,
             "heartbeat_at": heartbeat_at}
    if split is not None:  # the attempt's split (its marker holds it for good)
        lease["split"] = split
    return lease


def try_claim(claims_dir: Path, key: tuple[str, str, int], split: str | None = None) -> Path | None:
    path = claim_path(claims_dir, key)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return None
    now = time.time()
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(_lease(key, now, now, split), fh)
    return path


def refresh_lease(path: Path, claimed_at: float) -> bool:
    """Rewrite our claim with a fresh heartbeat (atomically, so it is never seen half-written).

    Returns False, writing nothing, when the claim is no longer ours (another runner reclaimed
    it after our heartbeat stalled): our refresh must never overwrite their lease.
    """
    data = _read_claim(path)
    if data is None or (data.get("pid"), data.get("host"), data.get("claimed_at")) != (
            os.getpid(), socket.gethostname(), claimed_at):
        return False
    lease = _lease(tuple(data["key"]), claimed_at, time.time(), data.get("split"))
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(lease, fh)
    os.replace(tmp, path)
    return True


# --- summary ----------------------------------------------------------------------------


def summarise(index_path: Path, model_names: list[str], *, by_arm: bool = False,
              keep: Callable[[dict], bool] | None = None) -> dict[str, dict]:
    """Per model (with ``by_arm``, per model and arm: a grid's cells are separate experiments), from the
    index, over its lines ``keep`` keeps (every line by default). ``accuracy_exact`` and ``mean_f1`` are over all
    attempts (N_att: an unscored, failed, abstained or capped attempt adds 0); the ``*_out`` figures are over the
    attempts that submitted a scored answer (N_out).

    Attempts that failed infrastructure on every try are outside N_att: they are
    counted in ``infra_failures``, with ``infra_failure_rate`` over all recorded
    attempts (N_att + infra_failures). ``infra_tries`` counts every try that
    failed infrastructure and ``infra_reruns`` the re-runs those tries caused."""
    groups: dict[str, Any] = {}
    if not index_path.exists():
        return groups
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["model"] not in model_names or (keep is not None and not keep(row)):
            continue
        if by_arm:
            groups.setdefault(row["model"], {}).setdefault(row.get("arm") or DEFAULT_ARM, []).append(row)
        else:
            groups.setdefault(row["model"], []).append(row)
    if by_arm:
        return {model: {arm: _summary(rows) for arm, rows in arms.items()} for model, arms in groups.items()}
    return {model: _summary(rows) for model, rows in groups.items()}


def _summary(rows: list[dict]) -> dict:
    s = {"N_att": 0, "N_out": 0, "n_exact": 0, "n_unscored": 0, "f1_sum": 0.0, "exact_out": 0, "f1_out_sum": 0.0,
         "by_status": {}, "capped": 0, "looping": 0, "turns": 0, "truncated_turns": 0, "infra_failures": 0,
         "infra_tries": 0, "infra_reruns": 0, "infra_failures_by_source": {}}
    for row in rows:
        s["by_status"][row["status"]] = s["by_status"].get(row["status"], 0) + 1
        infra_tries = row.get("infra_tries") or 0
        final_infra = row["status"] == "failed:infrastructure"
        s["infra_tries"] += infra_tries
        s["infra_reruns"] += infra_tries - 1 if final_infra else infra_tries
        if final_infra:
            s["infra_failures"] += 1
            source = row.get("infra_source") or "store"  # index lines before agent-run/7: only the store
            s["infra_failures_by_source"][source] = s["infra_failures_by_source"].get(source, 0) + 1
            continue
        s["N_att"] += 1
        if row["exact"] is None:
            s["n_unscored"] += 1
        else:
            s["n_exact"] += bool(row["exact"])
            s["f1_sum"] += row["f1"] or 0.0
            if row["status"] == "submitted":
                s["N_out"] += 1
                s["exact_out"] += bool(row["exact"])
                s["f1_out_sum"] += row["f1"] or 0.0
        if row["status"].startswith("capped:"):
            s["capped"] += 1
            s["looping"] += bool(row["looping"])
        s["turns"] += row["turns"] or 0
        s["truncated_turns"] += row["truncated_turns"] or 0
    n_att, n_out = s["N_att"], s["N_out"]
    s["accuracy_exact"] = s["n_exact"] / n_att if n_att else None
    s["mean_f1"] = s.pop("f1_sum") / n_att if n_att else None
    s["accuracy_exact_out"] = s.pop("exact_out") / n_out if n_out else None
    s["mean_f1_out"] = s.pop("f1_out_sum") / n_out if n_out else None
    s["truncated_turn_share"] = s["truncated_turns"] / s["turns"] if s["turns"] else None
    recorded = n_att + s["infra_failures"]
    s["infra_failure_rate"] = s["infra_failures"] / recorded if recorded else None
    return s


# --- the development grid (the design log) ----------------------------------------------


def _grid_builder():
    """scripts/build_dev_grid.py, whose gates, spec checks and expansion the launcher reuses, never copies. It
    takes this module as its runner, so a run started as ``python scripts/run_pilot.py`` loads no second copy."""
    module = sys.modules.get("build_dev_grid")
    if module is None:
        sys.modules.setdefault("run_pilot", sys.modules[__name__])
        spec = importlib.util.spec_from_file_location("build_dev_grid",
                                                      Path(__file__).resolve().parent / "build_dev_grid.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["build_dev_grid"] = module
        spec.loader.exec_module(module)
    return module


@dataclass
class GridPlan:
    """One graph's phase of the development grid: what the manifest plans, re-derived by the runner."""

    graph: str
    phase: str
    seed: int
    spec: dict
    spec_sha256: str
    attempts_sha256: str
    root: Path
    freeze: dict | None
    questions: list[dict]
    models: list[str]  # the phase's models, in spec order
    route_models: dict[str, str]  # model name -> the route model the spec names
    roles: dict[str, str]
    cells: dict[str, dict[str, dict]]  # role -> cell id -> the manifest's cell (a generic cell's sources per role)
    order: list[tuple[str, str, str, int]]  # (model, cell, question id, repetition), the phase's order
    initial: list[tuple[str, str, str, int]]  # every attempt of the graph's phases natural and prompted
    orders: dict[str, list[tuple[str, str, str, int]]]  # every phase's order (the graph's), by phase
    seeds: dict[str, int]  # every phase's seed
    model_settings: dict[str, dict]  # per model: route model, provider, route settings and caps (the manifest's)
    model_cells: dict[str, list[str]]  # per model: the cells it runs (its own cohort list; the manifest's)
    model_repetitions: dict[str, dict[str, int]]  # per model: its repetitions per phase (the manifest's)
    gateway: dict = dataclasses.field(default_factory=dict)  # this start's Gateway budget and workers (apply_grid_plan)

    @property
    def graph_spec(self) -> dict:
        return self.spec["graphs"][self.graph]

    def cell(self, model: str, cell_id: str) -> dict:
        return self.cells[self.roles[model]][cell_id]

    def is_branch(self, cell_id: str) -> bool:
        return any(cs.get(cell_id, {}).get("phase") == "branch" for cs in self.cells.values())

    def phase_cells(self, phase: str) -> list[str]:
        """Every cell of ``phase`` some model runs: cohorts in spec order, then branch cells in spec order."""
        used = {c for m, ids in self.model_cells.items() for c in ids
                if self.cells[self.roles[m]][c]["phase"] == phase}
        ids = [*self.spec["cohorts"], *(b["id"] for b in self.spec.get("branch_cells") or [])]
        return [c for c in ids if c in used]

    def cohorts(self, phase: str | None = None) -> list[str]:
        """The initial cells of ``phase`` (of both initial phases, when None), in spec order."""
        return [c for c in self.spec["cohorts"] if c in {x for ph in ((phase,) if phase else INITIAL_PHASES)
                                                          for x in self.phase_cells(ph)}]

    def model_caps(self, model: str) -> dict:
        """The model's caps on this plan's graph, as the manifest recorded them from the runner's table."""
        caps = self.model_settings[model]["caps"]
        if self.graph not in caps:
            raise Refused(f"{model}: the manifest records no caps for graph {self.graph!r}")
        return caps[self.graph]

    def model_orders(self, phase: str) -> dict[str, dict]:
        """Per model of ``phase``: its role, the sha256 of its own attempts in the phase's order, its caps on the
        graph and the tools hash they give, and in the branch phase its retry budget (its own turn and wall caps,
        the frozen nudges). A model's stream runs in this order (the start gate), and it depends on no other model:
        one may join later."""
        out = {}
        for m in self.spec["models"]:
            mine = [x for x in self.orders[phase] if x[0] == m["name"]]
            if mine:
                caps = self.model_caps(m["name"])
                out[m["name"]] = {"role": m["role"], "route_model": m["route_model"], "attempts": len(mine),
                                  "order_sha256": order_sha256(mine), "caps": caps,
                                  "tools_sha256": tools_sha256(agent_tools(caps["max_result_chars"]))}
                # The rules an initial attempt of the phase runs under, only those set. A branch attempt continues
                # its parent under the parent's recorded config (online.branch_attempt: its replayed prefix and its
                # retry alike, as with its caps), so the branch phase pins none of its own.
                if phase in INITIAL_PHASES:
                    out[m["name"]].update({field: self.model_settings[m["name"]][field] for field in RULE_FIELDS
                                           if self.model_settings[m["name"]].get(field) is not None})
                if phase == "branch":
                    out[m["name"]]["retry_budget"] = {"max_turns": caps["max_turns"], "max_wall_s": caps["max_wall_s"],
                                                      **FROZEN_RETRY_BUDGET}
        return out

    def sibling_order(self, model: str, cell_id: str, question_id: str, rep: int) -> tuple[list[str], int]:
        """The branch cells of one output (its model's role's cells on the same parent) in their order in the
        phase, and this cell's place among them (the record's ``checker.branch``)."""
        parent = self.cell(model, cell_id)["parent"]
        siblings = [c for c, cell in self.cells[self.roles[model]].items()
                    if cell["phase"] == "branch" and cell["parent"] == parent and c in self.model_cells[model]]
        ordered = sorted(siblings, key=lambda c: order_key(self.seed, (model, c, question_id, rep)))
        return ordered, ordered.index(cell_id)


def load_grid_plan(manifest_path: Path, spec_path: Path, root: Path, graph: str, phase: str) -> GridPlan:
    """The phase's plan, after every gate: the freeze for any split but dev (first, before anything else is
    read), the spec file the manifest was built from, the manifest rebuilt from that spec with every input of
    the phase checked against its hash (``build(..., for_launch=True, phase=phase)``), the attempts file, the
    runner's own order against the manifest's, the frozen settings the phase uses, and the code it runs on
    (``check_running_code``)."""
    build = _grid_builder()
    manifest_path, spec_path, root = Path(manifest_path), Path(spec_path), Path(root)
    try:
        spec_bytes = spec_path.read_bytes()
        spec = json.loads(spec_bytes)
        freeze = build.split_gate(spec, root)  # nothing of a test grid is looked at before the freeze is shown
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != build.MANIFEST_SCHEMA:
            raise Refused(f"{manifest_path}: schema {manifest.get('schema')!r}, expected {build.MANIFEST_SCHEMA!r}")
        spec_sha = build.sha256_bytes(spec_bytes)
        if manifest.get("spec_sha256") != spec_sha or manifest.get("spec") != spec:
            raise Refused(f"{manifest_path} was built from another spec than {spec_path} (sha256 {spec_sha}, the "
                          f"manifest's {manifest.get('spec_sha256')}); rebuild it with scripts/build_dev_grid.py")
        # the gate of this phase on this graph only: another graph's FILL does not hold it back (M6)
        rebuilt, _ = build.build(spec, spec_sha, root, for_launch=True, phase=phase, graph=graph)
        gs = spec["graphs"].get(graph)
        if gs is None:
            raise Refused(f"--graph {graph!r} is not a graph of the grid ({', '.join(spec['graphs'])})")
        questions = build.select_questions(root / gs["questions"]["path"], gs.get("question_ids"), spec["split"])
    except build.Refused as exc:
        raise Refused(f"the grid cannot launch: {exc}") from None
    raw = gzip.decompress((manifest_path.parent / manifest["attempts_file"]).read_bytes())
    attempts_sha = hashlib.sha256(raw).hexdigest()
    if attempts_sha != manifest.get("attempts_sha256"):
        raise Refused(f"{manifest['attempts_file']} does not match the manifest's attempts_sha256: the attempts were "
                      f"edited; rebuild the manifest")
    stale = [k for k in ("attempts_sha256", "order", "cells", "roles", "model_settings", "model_cells",
                         "model_repetitions") if rebuilt.get(k) != manifest.get(k)]
    if stale:
        raise Refused(f"{manifest_path} no longer matches its spec and inputs ({', '.join(stale)} differ from a "
                      f"rebuild); rebuild it")
    for q in questions:
        missing = [f for f in QUESTION_FIELDS if f not in q]
        if missing:
            raise Refused(f"question {q.get('id')!r} is missing {missing}")
    planned = [json.loads(x) for x in raw.decode("utf-8").splitlines()]
    roles = manifest["roles"]
    cells = {role: {c["id"]: c for c in cs} for role, cs in manifest["cells"].items()}
    qids = [q["id"] for q in questions]
    orders, seeds = {}, {}
    for ph in GRID_PHASES:
        mine = sorted((a for a in planned if a["graph"] == graph and a["phase"] == ph), key=lambda a: a["position"])
        order = [(a["model"], a["cell"], a["question_id"], a["repetition"]) for a in mine]
        seed = build.phase_seed(spec, ph)
        # The runner's own order: each model crossed with its own cells and repetitions of the phase, merged by
        # order_key.
        merged: list[tuple[str, str, str, int]] = []
        for m, ids in manifest["model_cells"].items():
            mine_ids = [c for c in ids if cells[roles[m]][c]["phase"] == ph]
            if mine_ids:
                merged += interleaved_order([m], mine_ids, qids, manifest["model_repetitions"][m][ph], seed)
        runner_order = sorted(merged, key=lambda attempt: order_key(seed, attempt))
        recorded = next((p["graphs"].get(graph, {}).get("order_sha256") for p in manifest["order"]["phases"]
                         if p["phase"] == ph), None)
        if (runner_order != order or order_sha256(runner_order) != recorded
                or [a["position"] for a in mine] != list(range(len(mine)))
                or any(a["seed"] != seed or a["order_key"] != order_key(seed, x) or a.get("role") != roles[x[0]]
                       for a, x in zip(mine, order))):
            raise Refused(f"the runner's order of graph {graph}, phase {ph} (seed {seed}, order sha256 "
                          f"{order_sha256(runner_order)}) is not the manifest's ({recorded}); rebuild the manifest")
        orders[ph], seeds[ph] = order, seed
    if not orders[phase]:
        raise Refused(f"graph {graph} has no attempts in phase {phase}")
    _check_frozen(spec, phase)
    check_running_code(spec, phase, harness_commit())
    order = orders[phase]
    models = [m["name"] for m in spec["models"] if any(o[0] == m["name"] for o in order)]
    return GridPlan(graph=graph, phase=phase, seed=seeds[phase], spec=spec, spec_sha256=spec_sha,
                    attempts_sha256=attempts_sha, root=root, freeze=freeze, questions=questions, models=models,
                    route_models={m["name"]: m["route_model"] for m in spec["models"]}, roles=roles, cells=cells,
                    order=order, initial=[x for ph in INITIAL_PHASES for x in orders[ph]], orders=orders, seeds=seeds,
                    model_settings=manifest["model_settings"], model_cells=manifest["model_cells"],
                    model_repetitions=manifest["model_repetitions"])


CODE_COMMITS = ("harness_commit", "compiler_commit")  # the spec's code fields that name a commit of this code


def check_running_code(spec: dict, phase: str, code: dict) -> None:
    """The grid runs only on code it can name, clean, at the commit its spec names (dress rehearsal F1): ``code``
    (``harness_commit()``: git, or a deploy's COMMIT and TREE with the tree recomputed from disk) must have a commit
    and no change from it, and the spec's ``code.harness_commit`` and ``code.compiler_commit`` must both be that
    commit (the compiler has no identity of its own: it is a file of the same tree). A commit still FILL is not
    compared; the phase gate refuses one where the phase needs it (branch), so branch always compares both."""
    placeholder = _grid_builder().PLACEHOLDER
    problems = []
    commit = code.get("git_commit")
    if not commit:
        problems.append(f"the running code names no commit (source {code.get('source')}): a record would name none")
    elif code.get("git_dirty") is not False:
        problems.append(f"the running code (commit {commit} from {code.get('source')}) differs from that commit, or "
                        f"that cannot be told: its records would name code that did not run")
    for key in CODE_COMMITS:
        want = (spec.get("code") or {}).get(key)
        if want is None and phase == "branch":
            problems.append(f"code.{key}: the spec names none; the branch phase runs only on the commit it names")
        elif commit and want not in (None, placeholder) and want != commit:
            problems.append(f"code.{key} {want} is not the running code's commit {commit}")
    if problems:
        raise Refused("the grid cannot launch on this code (deploy the commit the spec names, with COMMIT and TREE "
                      "from scripts/write_tree_digest.py, or fill the spec's code commits with the running one):\n  "
                      + "\n  ".join(problems))


def _check_frozen(spec: dict, phase: str) -> None:
    """The grid runs only under the frozen settings: in every phase the agent settings (none grid-wide: every cap
    is the model's own) and the runner settings; in the branch phase also the retry budget (its nudges and its one
    submission; its turn and wall caps are the model's), the switches and the checker's own feedback format and
    generic text (whose hashes the spec pins)."""
    from harness.checkloop.feedback import format_sha256, generic_sha256
    from harness.checkloop.online import FROZEN_SWITCHES

    problems = []
    if spec.get("agent_settings") != FROZEN_AGENT_SETTINGS:
        problems.append(f"agent_settings {spec.get('agent_settings')} are not the frozen ones {FROZEN_AGENT_SETTINGS} "
                        f"(each model's turn, wall, output and tool-result caps are its own, from the runner's table)")
    runner = spec.get("runner_settings") or {}
    if not (isinstance(runner.get("row_cap"), int) and runner["row_cap"] > 0
            and isinstance(runner.get("sparql_timeout_s"), (int, float)) and runner["sparql_timeout_s"] > 0):
        problems.append(f"runner_settings {runner}: needs a positive row_cap and sparql_timeout_s")
    if phase == "branch":
        if spec.get("retry_budget") != FROZEN_RETRY_BUDGET:
            problems.append(f"retry_budget {spec.get('retry_budget')} is not the frozen one {FROZEN_RETRY_BUDGET} "
                            f"(pre-registration section 3: the retry gets the attempt's own caps, the model's)")
        if spec.get("switches") != asdict(FROZEN_SWITCHES):
            problems.append(f"switches {spec.get('switches')} are not the frozen ones {asdict(FROZEN_SWITCHES)} "
                            f"(rule 23)")
        code = spec.get("code") or {}
        if code.get("feedback_format_sha256") != format_sha256():
            problems.append(f"code.feedback_format_sha256 {code.get('feedback_format_sha256')} is not the running "
                            f"checker's feedback format ({format_sha256()})")
        if code.get("generic_text_sha256") != generic_sha256():
            problems.append(f"code.generic_text_sha256 {code.get('generic_text_sha256')} is not the running "
                            f"checker's generic text ({generic_sha256()})")
    if problems:
        raise Refused("the grid's settings are not the ones it may run under:\n  " + "\n  ".join(problems))


def apply_grid_plan(args: argparse.Namespace, plan: GridPlan) -> None:
    """Every setting the manifest holds, into the run's arguments (none was given: parse_args refuses that): the
    question file, the store description and the endpoint it names (one store per graph for every phase), the
    runner and agent settings, and each model's workers."""
    gs = plan.graph_spec
    args.questions = plan.root / gs["questions"]["path"]
    args.split = plan.spec["split"]
    args.models = list(plan.models)
    args.reps = plan.spec["repetitions"]
    args.order_seed = plan.seed
    args.store_json = plan.root / gs["store_json"]["path"]
    endpoint = json.loads(args.store_json.read_text(encoding="utf-8")).get("endpoint")
    if not isinstance(endpoint, str) or not endpoint.startswith(("http://", "https://")):
        raise Refused(f"graphs.{plan.graph}.store_json ({gs['store_json']['path']}) names no http(s) endpoint: the "
                      f"grid's store is the one its description names, for every phase")
    args.endpoint = endpoint
    args.row_cap = plan.spec["runner_settings"]["row_cap"]
    args.sparql_timeout = float(plan.spec["runner_settings"]["sparql_timeout_s"])
    # each model's caps are its own (the runner's table, as the manifest's model_settings recorded them)
    workers = plan.spec["runner_settings"]["concurrency"]
    args.concurrency_for = {m: workers[m] for m in plan.models}  # build_dev_grid refused a FILL for these
    args.concurrency = max(args.concurrency_for.values())
    # the Gateway models' workers together within the budget (build_dev_grid refused the launch otherwise)
    gateway = {m: workers[m] for m in plan.models if plan.model_settings[m]["provider"] == GATEWAY_PROVIDER}
    budget = plan.spec["runner_settings"]["gateway_budget"]
    if sum(gateway.values()) > budget:
        raise Refused(f"Gateway models of phase {plan.phase}: {sum(gateway.values())} workers {gateway} > gateway_budget {budget}")
    plan.gateway = {"gateway_budget": budget, "gateway_workers": dict(sorted(gateway.items()))}
    if plan.phase == "branch":
        args.warmup = False  # a branch's retry continues its parent's transcript: no prefix of its own to cache
    named = {n for flag in (args.latency_baseline, args.wallcap_baseline, args.latency_baseline_calm) for n in flag}
    if named - set(plan.models):
        raise Refused(f"per-model flags name {sorted(named - set(plan.models))}, which are not models of this phase "
                      f"({', '.join(plan.models)})")


def grid_cell_config(base: AgentConfig, cohort_id: str, cohort: dict, block_sha256: str | None) -> AgentConfig:
    """An initial cell's AgentConfig: ``base`` (the shared settings) with the cohort's prompt condition and, for
    a cohort prompted with an ontology block, the block by its sha256 (design log's
    ``AgentConfig.ontology_block_sha256``). A harness whose AgentConfig cannot name a block refuses such a
    cohort: run under the bare condition, its records would claim a prompt the model never saw."""
    try:
        get_condition(cohort["condition"])
    except ValueError as exc:
        raise Refused(f"cohort {cohort_id!r}: {exc}") from None
    if cohort.get("source") is None:
        return dataclasses.replace(base, condition=cohort["condition"])
    if "ontology_block_sha256" not in {f.name for f in dataclasses.fields(base)}:
        raise Refused(f"cohort {cohort_id!r} is prompted with an ontology block (source {cohort['source']!r}), but this "
                      f"harness's AgentConfig has no ontology_block_sha256: the ontology prompt conditions (design log, "
                      f"feat/prompt-conditions) are not in it")
    if not block_sha256:
        raise Refused(f"cohort {cohort_id!r}: the spec names no prompt block for it")
    return dataclasses.replace(base, condition=cohort["condition"], ontology_block_sha256=block_sha256)


def _read_json(path: Path) -> Any:
    data = Path(path).read_bytes()
    return json.loads(gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data)


def grid_arms(plan: GridPlan) -> dict[tuple[str, str], Any]:
    """Each branch cell's checker arm per role: ``Arm(cell arm, the compilers of the cell's sources for that
    role)``, every compiler over the graph's one shared hierarchy (rule 16). A source must carry its spec name:
    its checks report under it (``per_source``), and the coverage check compares those with the cell's."""
    from harness.checkloop.arms import Arm
    from harness.checks.compiler import CheckCompiler
    from harness.checks.hierarchy import SharedHierarchy
    from harness.sources.model import OntologySource

    wanted = [(cid, role, cell) for role, cs in plan.cells.items() for cid, cell in cs.items()
              if cell["phase"] == "branch"]
    if not wanted:
        return {}
    gs = plan.graph_spec
    hierarchy = SharedHierarchy.from_json(_read_json(plan.root / gs["shared_hierarchy"]["path"]))
    compilers = {}
    for name in sorted({s for _, _, cell in wanted for s in cell["sources"]}):
        source = OntologySource.from_json(_read_json(plan.root / gs["sources"][name]["path"]))
        if source.name != name:
            raise Refused(f"graphs.{plan.graph}.sources.{name}: the file's source is named {source.name!r}, so its "
                          f"checks would report under that name, not {name!r}")
        compilers[name] = CheckCompiler(source, hierarchy)
    from harness.checkloop.online import GENERIC_TRIGGER_KIND

    for cid, role, cell in wanted:  # the manifest's deciding sources are the ones the checker will decide on (M7)
        if cell["mode"] == "generic" and "verdict_sources" in cell:
            curated = [s for s in cell["sources"] if compilers[s].source.kind == GENERIC_TRIGGER_KIND]
            if curated != cell["verdict_sources"]:
                raise Refused(f"{cid} ({role}): the manifest's verdict sources {cell['verdict_sources']} are not the "
                              f"arm's sources of kind {GENERIC_TRIGGER_KIND!r} {curated}, which the generic cell "
                              f"decides on (checker-loop/2)")
    return {(cid, role): Arm(cell["arm"], tuple(compilers[s] for s in cell["sources"])) for cid, role, cell in wanted}


def phase_inputs(plan: GridPlan, phase: str) -> dict:
    """The hashes of what ``phase`` reads beyond the shared settings (pinned with the phase): prompted, the
    prompts manifest and its cohorts' blocks and sources; branch, the shared hierarchy, its cells' sources and
    the code it was built at. Natural reads only the shared settings."""
    gs = plan.graph_spec
    if phase == "natural":
        return {}
    if phase == "prompted":
        cohorts = plan.cohorts("prompted")
        sources = sorted({plan.spec["cohorts"][c].get("source") for c in cohorts} - {None})
        return {"prompts_manifest": gs["prompts_manifest"]["sha256"],
                "prompt_blocks": {c: (gs.get("prompt_blocks") or {}).get(c, {}).get("sha256") for c in cohorts
                                  if plan.spec["cohorts"][c].get("source") is not None},
                "sources": {s: gs["sources"][s]["sha256"] for s in sources}}
    sources = sorted({s for cs in plan.cells.values() for cell in cs.values() if cell["phase"] == "branch"
                      for s in cell["sources"]})
    return {"shared_hierarchy": gs["shared_hierarchy"]["sha256"],
            "sources": {s: gs["sources"][s]["sha256"] for s in sources},
            "code": dict(plan.spec.get("code") or {})}


def phase_arm_fingerprints(plan: GridPlan, phase: str, configs: dict[str, AgentConfig],
                           arms: dict[tuple[str, str], Any], switches: Any) -> dict[str, dict]:
    """Per arm of ``phase``: an initial cell's prompt (condition, id, template, block and nudge hashes); a branch
    cell's ``OnlineChecker.settings()`` per role (arm identity with source and hierarchy hashes, mode, switches,
    feedback format and generic text hashes) without its retry budget, which is each model's own (its turn and wall
    caps, one role mixing models of different caps) and pinned per model (``GridPlan.model_orders``)."""
    from harness.checkloop.online import OnlineChecker, RetryBudget

    no_budget = RetryBudget(max_turns=0, max_wall_s=0.0, max_nudges=0)  # dropped from the settings below

    out: dict[str, dict] = {}
    if phase != "branch":
        for cid in plan.cohorts(phase):
            config = configs[cid]
            cond = get_condition(config.condition)
            # the tools hash depends on the model's tool-result cap: it is pinned per model (model_orders)
            out[cid] = {"condition": cond.condition, "prompt_id": cond.id, "template_sha256": cond.template_sha256,
                        "ontology_block_sha256": getattr(config, "ontology_block_sha256", None),
                        "nudge_sha256": sha256_text(cond.nudge)}
        return out
    for (cid, role), arm in sorted(arms.items()):
        cell = plan.cells[role][cid]
        entry = out.setdefault(cid, {"parent": cell["parent"], "by_role": {}})
        settings = OnlineChecker(arm, None, cell["mode"], budget=no_budget, switches=switches).settings()
        del settings["retry_budget"]
        entry["by_role"][role] = settings
    return out


def grid_fingerprint(args: argparse.Namespace, store: dict, base: AgentConfig, routes: dict[str, dict],
                     plan: GridPlan, phases: dict[str, dict], scorer_digest: str | None = None,
                     store_client: dict | None = None) -> dict:
    """What one graph's grid run is pinned to, in OUT/manifest.json (``check_grid_manifest``): ``shared``, the
    settings every phase runs under (question file, store and its endpoint, row cap, SPARQL time-out, the base
    agent config, split, repetitions, freeze, the frozen scorer's digest); ``routes``, each model's route settings;
    ``phases``, per phase started so far, its seed, cells, inputs (``phase_inputs``), arms
    (``phase_arm_fingerprints``) and each model's role, own order and caps (``GridPlan.model_orders``). The spec's
    sha256 is not pinned: the spec gains its later phases' inputs, and models, after a run starts."""
    fp = run_fingerprint(args, store, base, routes, scorer_digest, store_client)
    del fp["prompt"]  # per initial cell, in each phase's arms
    fp["agent_config"].pop("condition", None)  # per initial cell; each model's caps are in model_orders
    routes = fp.pop("routes")
    freeze = plan.freeze
    fp["grid"] = {"graph": plan.graph, "split": plan.spec["split"], "repetitions": plan.spec["repetitions"],
                  "order_rule": ORDER_RULE,
                  "freeze": None if freeze is None else {k: freeze[k] for k in ("commit", "preregistration_sha256")}}
    return {"shared": fp, "routes": routes, "phases": phases}


def _records_of(briefs: dict[tuple, Any], *, route_model: str | None = None, cells: Any = None) -> bool:
    return any((route_model is None or key[2] == route_model) and (cells is None or key[0] in cells)
               for key in briefs)


def check_grid_manifest(out: Path, fingerprint: dict, briefs: dict[tuple, Any], plan: GridPlan,
                        say: Callable[[str], None] = lambda msg: None) -> None:
    """Write the pins on the run's first start; on every later start check them and add what may be added.
    Refused: any change to the shared settings; a changed route, role or own order of a model with a record;
    any change to a recorded phase's seed, cells, inputs or arms once that phase has a record (an arm is added
    only to a phase with none); a model added to a phase while it has a record in the run. Added: a phase
    started for the first time, a model with no record yet (the DeepSeek panel model), and its route.

    A model's pins move once, as a recorded move (PIN_MOVES: Gemma's report-key rule with its turn and wall caps,
    change-log entries 110, 112 and 119; GLM's turn and wall caps, entry 111; Qwen3.6's reasoning-call rule, entry 113):
    when the only differences of the pins are the declared fields moved from their declared earlier values to the new
    ones, in the model's route and in its entries in the phases that have its records, the start is the first of a
    phase with no record, it runs the model, and every earlier phase holds a record of each of the model's attempts
    that is ever scored, the route takes the new values, the earlier phases keep their pins (their records replay under
    them), and ``pin_moves`` records the move with its ruling. A model the start does not run (an exploratory model of
    phase natural alone) has its move recorded for its phases with records, its route pins kept (``route_kept``;
    pin_moves). From then on those phases' entries are checked with the move undone, and a start of one of them that
    would run an attempt of the model is refused: it would run under the moved pins.

    After the freeze (the OUT's pins hold the freeze: a test grid) no move is taken: a per-model setting changed then
    is a served-version change of the model (change-log entry 129, item 3; SERVED_VERSION_AFTER_FREEZE), and the
    start is refused, naming each such model, before anything is written."""
    fingerprint = _redacted_json(fingerprint)
    path = out / "manifest.json"
    if not path.exists():
        if briefs:
            raise Refused(f"{out} holds records but no manifest.json, so it cannot be checked; use a new --out")
        _write_json_atomic(path, {"schema": GRID_OUT_SCHEMA, "created_at": utc_now(), "harness": harness_commit(),
                                  "fingerprint": fingerprint})
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != GRID_OUT_SCHEMA:
        raise Refused(f"{path}: schema {manifest.get('schema')!r}, not a grid run's ({GRID_OUT_SCHEMA}); use a new "
                      f"--out")
    stored = manifest["fingerprint"]
    recorded = _redacted_json({**stored, "shared": normalise_fingerprint(stored["shared"])})
    # after the freeze tag a per-model setting change is a served-version change of the model (change-log entry 129,
    # item 3): no declared move is taken, and the start is refused below with what the served-version rule asks
    frozen = frozen_out(stored)
    held_sv = {c["model"] for c in manifest.get("served_version_changes") or []}  # held here for good
    sv: dict[str, list[str]] = {}  # frozen: a model whose own pins differ now -> how (held, not refused)
    moved = recorded_pin_moves(manifest)
    for m, mv in moved.items():  # an earlier phase restarted after the move runs no attempt of the model
        left = attempts_to_run(plan, briefs, m, plan.phase) if plan.phase in mv["phases_before"] and m in plan.models \
            else 0
        if left:
            raise Refused(f"{out}: {m}'s pins moved from {mv['from']} to {mv['to']} at phase {mv['phase']} "
                          f"({mv['at']}; "
                          f"{mv['ruling']}); phase {plan.phase} pinned it before the move, and {left} of its attempts "
                          f"there would run under the moved pins. They cannot run in this OUT: this code would run "
                          f"them under the moved pins, and the earlier code is refused since the route moved. Ask the "
                          f"study lead for a ruling on them, or run them in a new --out")
    moving: dict[str, list[str]] = {}  # a model whose pins take its declared move now -> its phases with records
    routes_moving: set[str] = set()  # those whose route entry is the declared move
    diff = [f"shared.{d}" for d in fingerprint_diff(recorded["shared"], fingerprint["shared"])]
    for name, route in fingerprint["routes"].items():
        if name in recorded["routes"]:
            route_diff = fingerprint_diff(recorded["routes"][name], route)
            if route_diff and not frozen and name not in moved and name in PIN_MOVES and _takes_move(
                    recorded["routes"][name], route, PIN_MOVES[name], "route"):
                moving.setdefault(name, [])
                routes_moving.add(name)
            elif route_diff and frozen:  # a served-version change: the model is held, its route pins kept
                sv.setdefault(name, []).extend(f"routes.{name}.{d}" for d in route_diff)
            else:
                diff += [f"routes.{name}.{d}" for d in route_diff]
    added: list[str] = []
    code_changes: list[dict] = []
    phases = dict(recorded["phases"])
    for phase, now in fingerprint["phases"].items():
        before = recorded["phases"].get(phase)
        if before is None:
            if frozen:  # a model held as a served-version change is never pinned under its new settings
                now = {**now, "models": {m: e for m, e in now["models"].items() if m not in held_sv | set(sv)}}
            phases[phase] = now
            added.append(f"phase {phase}")
            continue
        if not _records_of(briefs, cells=set(before["cells"]) | set(now["cells"])):
            if frozen:
                # after the freeze a model's pins in a phase hold once written, whether or not anything ran under
                # them: changed or gone (gone and back would pin it again) is a served-version change of the model
                for m in sorted(set(before["models"]) | set(now["models"])):
                    if m not in now["models"]:
                        sv.setdefault(m, []).append(f"phases.{phase}.models.{m}: pinned in the phase, gone after the "
                                                    f"freeze")
                    elif m not in before["models"]:
                        sv.setdefault(m, []).append(f"phases.{phase}.models.{m}: joins the pinned phase after the "
                                                    f"freeze")
                    else:
                        sv.setdefault(m, []).extend(f"phases.{phase}.models.{m}.{d}" for d in fingerprint_diff(
                            before["models"][m], now["models"][m]))
                sv = {m: d for m, d in sv.items() if d}
                keep = held_sv | set(sv)
                now = {**now, "models": {**{m: e for m, e in now["models"].items() if m not in keep},
                                         **{m: e for m, e in before["models"].items() if m in keep}}}
            if before != now:  # nothing ran under the earlier pins: the phase is pinned as it is now
                phases[phase] = now
                added.append(f"phase {phase} (pinned again: it had no record)")
            continue
        for key in ("seed", "cells"):
            diff += [f"phases.{phase}.{d}" for d in fingerprint_diff({key: before[key]}, {key: now[key]})]
        inputs_diff = fingerprint_diff({"inputs": before["inputs"]}, {"inputs": now["inputs"]})
        code_diff = [d for d in inputs_diff if d.startswith(tuple(f"inputs.code.{k}:" for k in CODE_COMMITS))]
        change = checker_fix_code_change(out, phase, plan.phase, before["inputs"], now["inputs"],
                                         manifest.get("code_changes") or []) if code_diff else None
        if change is not None:  # the phase's checker-fix re-runs, on the fixed code their fix recorded
            inputs_diff = [d for d in inputs_diff if d not in code_diff]
            code_changes.append(change)
            before = {**before, "inputs": {**before["inputs"], "code": {**before["inputs"]["code"],
                                                                        **{k: change["to"] for k in CODE_COMMITS}}}}
        diff += [f"phases.{phase}.{d}" for d in inputs_diff]
        for cid in sorted(set(before["arms"]) | set(now["arms"])):
            if cid not in now["arms"]:
                diff.append(f"phases.{phase}.arms.{cid}: an arm the phase has records of is gone")
            elif cid not in before["arms"]:
                diff.append(f"phases.{phase}.arms.{cid}: a phase's arms are fixed once it has a record in {out}")
            else:
                diff += [f"phases.{phase}.arms.{cid}.{d}" for d in fingerprint_diff(before["arms"][cid],
                                                                                      now["arms"][cid])]
        models = {}
        for m, entry in before["models"].items():
            if m in now["models"]:
                entry_diff = fingerprint_diff(entry, now["models"][m])
                if not entry_diff:
                    pass
                elif m in moved and phase in moved[m]["phases_before"] and _takes_move(
                        entry, now["models"][m], moved[m], "phase"):
                    pass  # pinned before the model's recorded move: its records keep the earlier pins
                elif not frozen and m not in moved and m in PIN_MOVES and _takes_move(entry, now["models"][m],
                                                                                         PIN_MOVES[m], "phase"):
                    moving.setdefault(m, []).append(phase)
                elif frozen:  # a served-version change: the model is held, its entry kept
                    sv.setdefault(m, []).extend(f"phases.{phase}.models.{m}.{d}" for d in entry_diff)
                else:
                    diff += [f"phases.{phase}.models.{m}.{d}" for d in entry_diff]
                models[m] = entry
            elif frozen:  # gone after the freeze, with records in the phase or not: held, its entry kept
                sv.setdefault(m, []).append(f"phases.{phase}.models.{m}: pinned in the phase, gone after the freeze")
                models[m] = entry
            elif _records_of(briefs, route_model=entry["route_model"], cells=set(before["cells"])):
                diff.append(f"phases.{phase}.models.{m}: a model the phase has records of is gone")
        for m, entry in now["models"].items():
            if m not in before["models"]:
                if frozen:  # joins a pinned phase after the freeze: held, never pinned there under its new settings
                    sv.setdefault(m, []).append(f"phases.{phase}.models.{m}: joins the pinned phase after the freeze")
                elif _records_of(briefs, route_model=entry["route_model"], cells=set(now["cells"])):
                    diff.append(f"phases.{phase}.models.{m}: joins a phase it already has records in")
                else:
                    models[m] = entry
                    added.append(f"model {m} in phase {phase}")
        phases[phase] = {**before, "models": models}
    if diff and not frozen:
        hinted = [m for m in PIN_MOVES if m not in moved and (m in moving or any(
            f".{m}." in d and any(f".{field}:" in d for field in PIN_MOVES[m]["to"]) for d in diff))]
        diff += [f"{m}: its pins would also move from {PIN_MOVES[m]['from']} to {PIN_MOVES[m]['to']} (PIN_MOVES), a "
                 f"move taken only when it is the one difference" for m in hinted]
    if diff and frozen and sv:
        # a model's own change that also changes a phase's cells or arms (a cohort only it uses) is no per-model pin:
        # the OUT is refused as before, and the served-version change alerted, naming the model
        commit = str(frozen.get("commit"))[:12]
        diff += [f"{m}: {SERVED_VERSION_AFTER_FREEZE.format(model=m, commit=commit, hold='its served-version hold')}"
                 f" (not held: the start is refused for the differences above that are not its own pins)"
                 for m in sorted(sv)]
        say(f"ALERT served-version change after the freeze: {', '.join(sorted(sv))}: its own pins changed together "
            f"with pins that are not one model's (a phase's cells or arms), so this start is refused, nothing written; "
            f"see the refusal")
    if diff:
        diff += [f"(recorded: {m}'s pins moved from {mv['from']} to {mv['to']} at phase {mv['phase']}, {mv['at']}; "
                 f"they move once)" for m, mv in moved.items()]
        raise Refused(f"{out} was started with other settings; a start must keep every pin it has (use a new --out "
                      f"for a new experiment):\n  " + "\n  ".join(diff))
    with_records = {m: [ph for ph, pinned in recorded["phases"].items() if m in pinned["models"]
                        and _records_of(briefs, cells=set(pinned["cells"]))] for m in moving}
    new_moves = pin_moves(out, moving, routes_moving, with_records, fingerprint, briefs, plan)
    new_routes = {n: r for n, r in fingerprint["routes"].items() if n not in recorded["routes"]}
    new_sv = served_version_records(frozen, sv, held_sv, recorded, fingerprint) if frozen else []
    if new_sv:
        manifest.setdefault("served_version_changes", []).extend(new_sv)
    if added or new_routes or code_changes or new_moves or new_sv:
        stored["routes"].update(new_routes)
        for move in new_moves:
            if move["model"] in fingerprint["routes"]:  # a model this start does not run keeps its route pins
                stored["routes"][move["model"]] = fingerprint["routes"][move["model"]]
            manifest.setdefault("pin_moves", []).append(move)
            say(f"manifest: {move['model']}'s pins moved from {move['from']} to {move['to']} at phase {move['phase']} "
                f"({move['ruling']}); phases {', '.join(move['phases_before']) or 'none'} keep the earlier pins, and "
                f"recorded")
        stored["phases"] = phases
        if added or new_routes:
            manifest.setdefault("additions", []).append({"at": utc_now(), "added": added,
                                                         "routes": sorted(new_routes)})
            say(f"manifest: added {', '.join(added + [f'route {n}' for n in new_routes])}")
        for change in code_changes:
            manifest.setdefault("code_changes", []).append(change)
            say(f"manifest: phase {change['phase']} re-pinned from code {str(change['from'])[:12]} to the checker "
                f"fix's code {change['to'][:12]} for its {change['fix_reruns']} set-aside attempts, and recorded")
        _write_json_atomic(path, manifest)
    if frozen:  # after the manifest records each change: a stop in between leaves it recorded, held at the next start
        served_version_holds(out, frozen, sv, held_sv | {c["model"] for c in new_sv}, new_sv, say)


def served_version_records(frozen: dict, sv: dict[str, list[str]], held_sv: set[str], recorded: dict,
                           fingerprint: dict) -> list[dict]:
    """After the freeze: the manifest's new served_version_changes entries, one per model whose own pins differ now
    (``sv``) and that no earlier start recorded (``held_sv``): the durable record the holds are written from."""
    commit = str(frozen.get("commit"))
    routes = {**fingerprint["routes"], **recorded["routes"]}  # the route the model ran under, else the new one
    out = []
    for m in sorted(set(sv) - held_sv):
        rule = SERVED_VERSION_AFTER_FREEZE.format(model=m, commit=commit[:12], hold="its served-version hold")
        out.append({"model": m, "route_model": (routes.get(m) or {}).get("model"), "at": utc_now(),
                    "differences": sv[m], "freeze_commit": commit, "rule": rule, "harness": harness_commit()})
    return out


def served_version_holds(out: Path, frozen: dict, sv: dict[str, list[str]], held: set[str], new: list[dict],
                         say: Callable[[str], None]) -> None:
    """Hold every model the manifest records as a served-version change (``held``) through its served-version hold
    file, written when missing (removed by hand, or a stop after the manifest write), and alert each new one."""
    commit = str(frozen.get("commit"))
    for m in sorted(held):
        path = served_version_hold_path(out, m)
        reason = SERVED_VERSION_AFTER_FREEZE.format(model=m, commit=commit[:12], hold=path)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_json_atomic(path, {"kind": "served_version_change", "model": m, "freeze_commit": commit,
                                      "at": utc_now(), "differences": sv.get(m), "reason": reason,
                                      "release": "none in this OUT: the model's runs are all re-run in a new OUT"})
    for entry in new:
        m = entry["model"]
        say(f"ALERT served-version change: {m}: "
            f"{SERVED_VERSION_AFTER_FREEZE.format(model=m, commit=commit[:12], hold=served_version_hold_path(out, m))}"
            f". Differences: {'; '.join(entry['differences'])}")


def _held(entry: Any, keys: tuple[str, ...]) -> dict | None:
    for key in keys:
        entry = entry.get(key) if isinstance(entry, dict) else None
    return entry if isinstance(entry, dict) else None


def _takes_move(before: dict, now: dict, move: dict, kind: str) -> bool:
    """Whether the pins ``now`` (a model's route entry, ``kind`` "route", or its entry in a phase, "phase") are
    ``before`` with each field ``move`` names moved from its "from" value (None: absent) to its "to" value, and nothing
    else changed."""
    undone = json.loads(json.dumps(now))
    moved_any = False
    for field, to in move["to"].items():
        earlier = move["from"][field]
        for path in _MOVE_PATHS[kind][field]:
            held_before, held_now = _held(before, path[:-1]), _held(undone, path[:-1])
            if held_before is None and held_now is None:
                continue  # not in this entry (a retry budget outside the branch phase)
            if held_before is None or held_now is None:
                return False
            if held_before.get(path[-1]) != earlier or held_now.get(path[-1]) != to:
                return False
            if earlier is None:
                held_now.pop(path[-1], None)
            else:
                held_now[path[-1]] = earlier
            moved_any = True
    return moved_any and not fingerprint_diff(before, undone)


def attempts_to_run(plan: GridPlan, briefs: dict[tuple, Any], model: str, phase: str) -> int:
    """How many of ``model``'s planned attempts in ``phase`` a start of it would run: those with no record, and
    those whose record is never scored (UNSCORED_FAILURES), which are set aside and run again."""
    route = plan.route_models[model]
    left = 0
    for m, cell, qid, rep in plan.orders[phase]:
        if m == model:
            brief = briefs.get(attempt_key(qid, route, rep, cell))
            left += brief is None or brief.failure_type in UNSCORED_FAILURES
    return left


def recorded_pin_moves(manifest: dict) -> dict[str, dict]:
    """A grid OUT manifest's recorded pin moves, per model (``report_key_moves``: the same records, as a development
    commit of this change named them)."""
    return {mv["model"]: mv for mv in (manifest.get("report_key_moves") or []) + (manifest.get("pin_moves") or [])}


def pin_moves(out: Path, moving: dict[str, list[str]], routes_moving: set[str], with_records: dict[str, list[str]],
              fingerprint: dict, briefs: dict[tuple, Any], plan: GridPlan) -> list[dict]:
    """The recorded moves of the models in ``moving`` (each with the phases whose pins of it are the earlier ones),
    for the manifest's ``pin_moves``; Refused unless each is the move check_grid_manifest describes: its route and
    its entry in every phase with records (``with_records``) move together, at the first start of a phase with no
    record that runs the model, with every earlier attempt of it recorded.

    A model this start does not run (an exploratory model of phase natural alone, at the start of prompted) has no
    route in it, and none of its attempts runs: the move records that its entries in every phase with records (which
    must all differ by the declared move alone) keep the earlier pins, so its records are held to them (the coverage
    check), and its route pins are left as they are (``route_kept``); an earlier phase restarted after the move is
    refused as for any moved model (check_grid_manifest). Refusing it would refuse the start of every other model."""
    problems = []
    started_cells = set(fingerprint["phases"][plan.phase]["cells"])
    for m, earlier in moving.items():
        move = PIN_MOVES[m]
        if m not in fingerprint["routes"]:
            if sorted(earlier) != sorted(with_records[m]):
                problems.append(f"{m}: its entries in every phase with records ({', '.join(with_records[m])}) move "
                                f"from {move['from']} to {move['to']} together, and here only phases "
                                f"{', '.join(earlier) or 'none'} do")
            continue
        if m not in routes_moving or sorted(earlier) != sorted(with_records[m]):
            problems.append(f"{m}: its route and its entries in every phase with records "
                            f"({', '.join(with_records[m])}) "
                            f"move from {move['from']} to {move['to']} together, and here only "
                            f"{'its route and ' if m in routes_moving else ''}phases {', '.join(earlier) or 'none'} do")
        if _records_of(briefs, cells=started_cells):
            problems.append(f"{m}: its pins move from {move['from']} to {move['to']} at the first start of a phase "
                            f"with no record, and phase {plan.phase} has records, which ran under the earlier pins")
            continue
        for phase in earlier:
            left = attempts_to_run(plan, briefs, m, phase)
            if left:
                problems.append(f"{m}: {left} of its attempts in phase {phase} have no record that is ever scored; "
                                f"they run under the earlier pins ({move['from']}), so finish them first: start phase "
                                f"{phase} on the code it ran on, then start phase {plan.phase} on this code; or use a "
                                f"new --out")
    if problems:
        raise Refused(f"{out}: a model's pins move once, as a recorded move, and this start is not that move:\n  "
                      + "\n  ".join(problems))
    return [{"at": utc_now(), "model": m, "route_model": plan.route_models.get(m), "phase": plan.phase,
             "from": dict(PIN_MOVES[m]["from"]), "to": dict(PIN_MOVES[m]["to"]), "phases_before": earlier,
             "branch_retries": "a branch attempt runs under its parent's recorded config (pre-registration section "
                               "3): one whose parent is in phases_before keeps the from values; its phase's pinned "
                               "caps and retry budget are the table's",
             "ruling": PIN_MOVES[m]["ruling"], "harness": harness_commit(),
             **({} if m in fingerprint["routes"] else {"route_kept": f"phase {plan.phase} does not run {m}: its route "
                                                                     f"pins are not compared, and stay as recorded"})}
            for m, earlier in moving.items()]


def checker_fix_code_change(out: Path, phase: str, started: str, before: dict, now: dict,
                            earlier: list[dict] = ()) -> dict | None:
    """Design ruling (design log, 20 Oct 07:26), condition (iii): a phase whose divergent records were set aside under a
    checker fix (relabel_harness_failure.py --checker-fix) re-runs them, in a later start of the same phase, under the
    fixed code its fix recorded. So the phase being started (``started``) may change the code commits it pinned
    (``before`` and ``now``: its pinned inputs) to that code, once, and only while attempts of it set aside under that
    fix wait to be re-run; any other change of its pins is refused as before. The change, for the manifest's
    ``code_changes``; None when it is not that one. Refused when the manifest's ``code_changes`` (``earlier``) already
    record a move of the phase (release review, item 5): the ruling allows one, and a second fix of the same phase
    means the first did not fix the cause, which needs a fresh decision, recorded explicitly, before its pins move
    again."""
    to = {(now.get("code") or {}).get(k) for k in CODE_COMMITS}
    pending = pending_fixed_reruns(out, phase) if phase == started else []
    if not pending or len(to) != 1 or {s["fix"].get("code") for s in pending} != to:
        return None
    moved = [c for c in earlier if c.get("phase") == phase]
    if moved:
        last = moved[-1]
        raise Refused(f"{out}: phase {phase}'s pinned code moved once already, to a checker fix ({str(last.get('from'))[:12]}"
                      f" -> {str(last.get('to'))[:12]} at {last.get('at')}, for {last.get('fix_reruns')} set-aside "
                      f"attempts); its {len(pending)} attempts set aside under the fix at {next(iter(to))[:12]} need "
                      f"a fresh decision, recorded explicitly, before its pins move a second time: the first fix did "
                      f"not fix the cause (design ruling, design log, 20 Oct 07:26, condition 3)")
    return {"at": utc_now(), "phase": phase, "from": (before.get("code") or {}).get("harness_commit"),
            "to": next(iter(to)), "fix_reruns": len(pending),
            "rulings": sorted({str(s["fix"].get("ruling")) for s in pending})}


def _grid_pins(out: Path) -> dict | None:
    """A grid run's pins in OUT/manifest.json, or None (no manifest yet, or not a grid run's)."""
    path = Path(out) / "manifest.json"
    if not path.exists():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return manifest.get("fingerprint") if manifest.get("schema") == GRID_OUT_SCHEMA else None


def is_grid_out(out: Path) -> bool:
    """Whether OUT holds a grid run (its manifest is the grid's)."""
    path = Path(out) / "manifest.json"
    return path.exists() and json.loads(path.read_text(encoding="utf-8")).get("schema") == GRID_OUT_SCHEMA


VERIFY_SCRIPT = Path(__file__).resolve().parent / "verify_batch.py"


@dataclasses.dataclass
class StoppedRecompute:
    """A fresh-process recompute stopped part way (the periodic check's at the phase's end): the decisions it had
    completed, by file (``replay_check.read_progress``)."""

    decisions: dict


def decisions_in_fresh_process(out: Path, plan: GridPlan, say: Callable[[str], None], *, nice: int = 0,
                               stop: threading.Event | None = None) -> dict | StoppedRecompute | None:
    """Every online checker decision in OUT recomputed by scripts/verify_batch.py --decisions-json in a fresh
    interpreter of this code, from the records' recorded store responses, with the compilers of the graph's branch
    sources over its shared hierarchy (the spec's files, whose hashes the launch gate checked): a process whose state
    broke while the phase ran (the rehearsal's rdflib grammar, F2) would recompute its own broken decisions and agree
    with them (F3). None when that process did not finish: then no checker decision counts as recomputed. ``nice``
    lowers the process's priority by that much (the periodic check's, so the attempts keep the CPU first); ``stop``
    set ends it early: a StoppedRecompute with the decisions it had completed (its --progress-jsonl lines; entry 125),
    which count as recomputed nowhere but in the stopped check's own record."""
    gs = plan.graph_spec
    sources = sorted({s for cs in plan.cells.values() for cell in cs.values() if cell["phase"] == "branch"
                      for s in cell["sources"]})
    fd, target = tempfile.mkstemp(dir=out, prefix=".decisions.", suffix=".json")
    os.close(fd)
    fd, progress = tempfile.mkstemp(dir=out, prefix=".decisions.", suffix=".progress.jsonl")
    os.close(fd)
    cmd = [sys.executable, str(VERIFY_SCRIPT), str(out), "--decisions-json", target, "--progress-jsonl", progress,
           "--hierarchy", str(plan.root / gs["shared_hierarchy"]["path"])]
    for name in sources:
        cmd += ["--source-file", f"{name}={plan.root / gs['sources'][name]['path']}"]
    # its output goes to a file, not a pipe: the process is polled (``stop``), and a full pipe would block it
    with tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace") as output:
        try:
            proc = subprocess.Popen(cmd, stdout=output, stderr=subprocess.STDOUT, text=True)
            if nice:
                try:
                    os.setpriority(os.PRIO_PROCESS, proc.pid, min(19, os.getpriority(os.PRIO_PROCESS, 0) + nice))
                except OSError:  # it ran first, or the priority cannot be set here: it runs at the runner's
                    pass
            while True:
                try:
                    proc.wait(timeout=None if stop is None else 0.2)
                    break
                except subprocess.TimeoutExpired:
                    if stop.is_set():
                        proc.terminate()
                        proc.wait()
                        return StoppedRecompute(read_progress(Path(progress)))
            text = Path(target).read_text(encoding="utf-8")
        finally:
            Path(target).unlink(missing_ok=True)
            Path(progress).unlink(missing_ok=True)
        output.seek(0)
        printed = output.read()
    if proc.returncode != 0 or not text:
        tail = " | ".join(printed.strip().splitlines()[-5:])
        say(f"error: the checker decisions' recompute in a fresh process exited {proc.returncode}: {tail}")
        return None
    return json.loads(text)["decisions"]


PERIODIC_CHECK_S = 7200.0  # the branch phase's periodic recompute: every 2 hours unless --periodic-check-s says
PERIODIC_MAX_LOAD = 1.0  # a periodic check is skipped while the 1-minute load average per CPU is at least this
PERIODIC_NICE = 10  # and runs this much nicer than the runner


class PeriodicCheck:
    """Science, design log, 20 Oct 10:39 (item 7 of the release work): while a branch phase runs, the online checker
    decisions of the records completed so far are recomputed every ``interval_s`` in a fresh interpreter (never in
    this process: F3), niced; a cycle is skipped while the previous one still runs, or while the CPU is short (the
    one-minute load average per CPU at least ``max_load``). When the divergences so far exceed the halt threshold
    (more than 1% of the phase's completed records, or divergences in more than one model: replay_check's rule),
    ``halt`` is set: the runner launches and claims nothing new, the attempts in flight finish and are recorded, and
    it exits 5; the halt stands until a checker fix of the phase (replay_check.standing_halts). A divergence below the
    threshold is left to the end-of-phase check, the gate. Each check (its time, record count and result), each
    skipped cycle and a check the phase's end stopped (with the decisions it had completed; entry 125) is recorded in the
    OUT manifest's ``periodic_checks``. Counts and the verdict only: no score is
    read or printed. ``recompute(stop)`` is the fresh process (decisions, or None when it did not finish)."""

    def __init__(self, out: Path, phase: str, recompute: Callable[[threading.Event], dict | None],
                 say: Callable[[str], None], *, interval_s: float, max_load: float,
                 load: Callable[[], tuple] = os.getloadavg, cpus: int | None = None,
                 clock: Callable[[], float] = time.monotonic, wait: Callable[[float], bool] | None = None,
                 manifest: Path | None = None):
        self.out, self.phase, self.recompute, self.say = Path(out), phase, recompute, say
        self.interval_s, self.max_load, self.load, self.clock = interval_s, max_load, load, clock
        self.cpus = cpus or os.cpu_count() or 1
        self.halt, self.stop = threading.Event(), threading.Event()
        self.wait = wait or self.stop.wait
        self.manifest = manifest or self.out / "manifest.json"
        self._lock = threading.Lock()
        self._decide = threading.Lock()  # the stop and a halt decided together (finish, cycle)

    def record(self, entry: dict) -> None:
        with self._lock, MANIFEST_LOCK:
            data = json.loads(self.manifest.read_text(encoding="utf-8"))
            data.setdefault("periodic_checks", []).append({"at": utc_now(), "phase": self.phase, **entry})
            _write_json_atomic(self.manifest, data)

    def run(self) -> None:
        due = self.clock() + self.interval_s
        while not self.wait(max(0.0, due - self.clock())):
            load = self.load()[0] / self.cpus
            if load >= self.max_load:
                self.record({"skipped": "cpu short", "load_per_cpu": round(load, 2)})
                self.say(f"periodic check of phase {self.phase} skipped: load {load:.2f} per CPU")
                due += self.interval_s
                continue
            result = self.cycle()
            if result is None:  # stopped while it ran: the phase's attempts ended
                return
            missed = int((self.clock() - due) // self.interval_s)  # the ticks that fell due while it ran
            if missed > 0:
                self.record({"skipped": "previous check still running", "ticks": missed})
            due += self.interval_s * (missed + 1)
            if result.get("halted"):  # cycle set the halt before it wrote or said anything
                return

    def finish(self) -> None:
        """The phase's attempts ended: stop the checks. Under the lock that ``cycle`` decides under, so a check either
        halted the phase before this, or is recorded as stopped (never halting after it)."""
        with self._decide:
            self.stop.set()

    def cycle(self) -> dict | None:
        at, started = utc_now(), time.monotonic()
        decisions = self.recompute(self.stop)
        entry = None
        if decisions is not None and not isinstance(decisions, StoppedRecompute):
            entry = {"at": at, **divergence_so_far(self.out, decisions, self.phase),
                     "seconds": round(time.monotonic() - started, 1)}
        with self._decide:  # the stop and the halt decided together (finish): never a halt after the phase's end
            stopped = self.stop.is_set()
            if not stopped and entry is not None and entry["halted"]:
                # first: the manifest write and the log line can block (a full stdout pipe), and no launch may wait
                # on them (release re-review, item 3)
                self.halt.set()
        if stopped:  # the phase's attempts ended while it ran: recorded as stopped, never dropped (entry 125)
            self.record_stopped(at, started, decisions)
            return None
        if entry is None:
            entry = {"at": at, "error": "the fresh-process recompute did not finish", "halted": False}
            self.record(entry)
            self.say(f"warning: periodic check of phase {self.phase}: its fresh-process recompute did not finish")
            return entry
        self.record(entry)
        counts = (f"periodic check: {entry['records']} records of phase {self.phase} so far, {entry['diverged']} "
                  f"diverge ({entry['decisions_recomputed']} decisions recomputed to themselves, "
                  f"{entry['not_recomputed']} not recomputed)")
        if entry["halted"]:
            self.say(f"error: {counts}: phase {self.phase} HALTED ({' and '.join(entry['reasons'])}): no new attempt "
                     f"starts; the attempts in flight finish and are recorded, then the run exits 5")
        else:
            self.say(f"{counts}; not halted")
        return entry

    def record_stopped(self, at: str, started: float, result: Any) -> None:
        """A check the phase's end stopped while it ran (science, design log, change-log entry 125): recorded with the
        decisions it had completed (``StoppedRecompute``; none when its process left none) and the halt rule's counts
        over them, never acted on: the end-of-phase recompute, the gate, decides every record next."""
        partial = (result.decisions if isinstance(result, StoppedRecompute)
                   else result if isinstance(result, dict) else {})  # completed just as the phase ended: all of it
        counts = divergence_so_far(self.out, partial, self.phase)
        entry = {"at": at, "stopped": "the phase's attempts ended while it ran; the end-of-phase recompute follows",
                 **{k: counts[k] for k in ("records", "decisions_recomputed", "not_recomputed", "held", "diverged",
                                           "models")},
                 "halt_rule_met": counts["halted"], "halted": False, "seconds": round(time.monotonic() - started, 1)}
        self.record(entry)
        self.say(f"periodic check of phase {self.phase} stopped at the phase's end: {entry['records']} records decided "
                 f"so far, {entry['diverged']} diverge ({entry['decisions_recomputed']} recomputed to themselves); the "
                 f"end-of-phase recompute follows")


def verified_run_ids(decisions: dict | None) -> set[str]:
    """The run ids of the records in records/ whose online checker decision the fresh-process recompute
    (``decisions_in_fresh_process``) decided again to itself, read from the records' names (none is opened)."""
    found = set()
    for rel, decided in (decisions or {}).items():
        sub, _, name = rel.partition("/")
        if decided is True and sub == "records":
            m = _RUN_FILE_NAME.fullmatch(name)
            found.add(m["run_id"] if m else name.removesuffix(".json"))
    return found


def pending_fixed_reruns(out: Path, phase: str) -> list[dict]:
    """The sidecars of phase ``phase``'s attempts set aside as checker divergences with a fix (relabel_harness_failure.py
    --checker-fix) whose attempt has no record in records/ yet, one per attempt: its latest set-aside (a re-run under
    a fix that was itself set aside again has two sidecars, and only the later one waits). A re-run is written by
    this runner, so it is named by its attempt's claim digest: the sidecar's own name gives the digest, or, for a
    record named before attempt markers, its attempt key does."""
    grid = is_grid_out(out)
    digests = {m["digest"] for p in (out / "records").glob("*.json") if (m := _RUN_FILE_NAME.fullmatch(p.name))}
    latest: dict[str, dict] = {}
    for name, sidecar in read_sidecars(out).items():
        if sidecar.get("classification") != CHECKER_DIVERGENCE or sidecar.get("phase") != phase:
            continue
        m = _RUN_FILE_NAME.fullmatch(name)
        digest = m["digest"] if m else claim_digest(attempt_key(
            sidecar["question_id"], sidecar["model"], sidecar["repetition"], sidecar.get("arm") if grid else None))
        if digest not in latest or str(sidecar.get("set_aside_at")) > str(latest[digest].get("set_aside_at")):
            latest[digest] = sidecar
    return [s for digest, s in latest.items() if s.get("fix") and digest not in digests]


def awaiting_checker_fix(out: Path) -> dict[tuple, dict]:
    """The attempts set aside as checker divergences with no fix recorded yet (relabel_harness_failure.py --diverged,
    below the halt), by attempt key, with their sidecars. Condition 3 of the design ruling (design log, 20 Oct 07:26)
    holds for every such set-aside (release review, item 2): its attempt is re-run only after a fix is recorded
    (--checker-fix), under it; until then its claim is kept, and no start runs or reclaims it."""
    grid = is_grid_out(out)
    return {attempt_key(s["question_id"], s["model"], s["repetition"], s.get("arm") if grid else None): s
            for s in read_sidecars(out).values()
            if s.get("classification") == CHECKER_DIVERGENCE and not s.get("fix")}


def _fixed_code(fix: dict, code: dict) -> bool:
    """Whether the running ``code`` (clean) is the code a checker fix recorded: its commit, or its tree digest."""
    if code.get("git_dirty") is not False:
        return False
    return fix.get("code") == code.get("git_commit") or (
        fix.get("tree_sha256") is not None and fix.get("tree_sha256") == code.get("tree_sha256"))


def require_not_halted(out: Path, phase: str, code: dict) -> None:
    """Design ruling (design log, 20 Oct 07:26), condition (ii) and (iii): no attempt of a HALTED phase starts (its
    divergences exceed 1% of its records, or occur in more than one model: the checker code is suspect) until the
    cause is fixed, and attempts set aside under that fix are re-run only under the fixed code it recorded, so that
    their re-run records name it. The halt is the stored replay check's (decided where the hold is, after each
    phase): it takes effect at the phase's next start."""
    path = out / CHECK_FILE
    if path.is_file():
        try:
            check = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise Refused(f"{path}: not readable ({exc}), so whether phase {phase} is halted cannot be told; run "
                          f"scripts/verify_batch.py {out} first") from None
        halted = halt_message(out, check, phase)
        if halted:
            raise Refused(halted)
    wrong = [s for s in pending_fixed_reruns(out, phase) if not _fixed_code(s["fix"], code)]
    if wrong:
        fixes = sorted({str(s["fix"].get("code"))[:12] for s in wrong})
        raise Refused(f"{out}: {len(wrong)} attempt(s) of phase {phase} set aside as checker divergences wait to be "
                      f"re-run under the fixed code their fix recorded ({', '.join(fixes)}); the running code is "
                      f"{str(code.get('git_commit'))[:12]}{'' if code.get('git_dirty') is False else ' (not clean)'}: "
                      f"run the fixed code")


def require_replayed_records(out: Path) -> None:
    """The branch phase starts once phases natural and prompted are recorded and pass their replay check
    (design ruling in the design log): OUT's replay check must have passed, on the records as they are now."""
    rerun = f"run scripts/verify_batch.py {out}, then start the branch phase again"
    if not (out / "manifest.json").exists():
        raise Refused(f"{out}: phases natural and prompted have not run here (no manifest.json); the branch phase "
                      f"branches from their records")
    path = out / CHECK_FILE
    if not path.is_file():
        raise Refused(f"{out}: no {CHECK_FILE}, so the initial phases' records never passed the replay check; {rerun}")
    try:
        check = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise Refused(f"{path}: not readable ({exc}); {rerun}") from None
    held = held_message(out, check)
    if held:
        raise Refused(f"{held}; re-run or set aside those records, then {rerun}")
    if check.get("failures") or check.get("n_ok") != check.get("n_records"):
        raise Refused(f"{out}: the replay check failed for {len(check.get('failures') or [])} of "
                      f"{check.get('n_records')} records ({CHECK_FILE}); fix them, then {rerun}")
    if check.get("records_sha256") != current_records_sha256(out):
        raise Refused(f"{out}: records changed since the replay check; {rerun}")


@dataclass(frozen=True)
class RecordBrief:
    """What the grid's gates read of a record in OUT/records/."""

    path: Path
    run_id: str
    status: str | None
    failure_type: str | None
    branched_from: str | None  # a branch record's parent run id

    @property
    def outcome(self) -> dict:
        return {"status": self.status, "failure_type": self.failure_type}


def record_briefs(records_dir: Path) -> dict[tuple, RecordBrief]:
    """Each grid attempt key (arm, question, model, repetition) that has a record, with what the gates read of
    it, from one read of every record; refuses when two records share a key."""
    out: dict[tuple, RecordBrief] = {}
    for key, path in record_paths(records_dir, grid=True).items():
        data = json.loads(path.read_text(encoding="utf-8"))
        outcome = data.get("outcome") or {}
        out[key] = RecordBrief(path, data["run_id"], outcome.get("status"), outcome.get("failure_type"),
                               (data.get("checker") or {}).get("branched_from"))
    return out


def relabel_hint(arm: str, model_route: str, qid: str, rep: int, out: Path) -> str:
    return (f"scripts/relabel_harness_failure.py {out} --arm {arm} --question {qid} --model {model_route} "
            f"--repetition {rep} --defect ... --ruling ...")


def require_initial_complete(plan: GridPlan, briefs: dict[tuple, RecordBrief], claims_dir: Path, out: Path) -> None:
    """The branch phase starts once phases natural and prompted are complete (design ruling in the design log): every
    planned attempt of both, of every model the branch phase runs, has a record that did not fail infrastructure. A
    crashed or stopped attempt (its claim kept, no record), one never started, and one that failed
    infrastructure on every try (unscored: the coverage check counts it missing) each block the phase, listed
    with what clears it."""
    problems = []
    phase_of = {c: cell["phase"] for cs in plan.cells.values() for c, cell in cs.items()}
    # An exploratory model (natural only: no branch cell) has no record that is a parent, so its attempts gate
    # nothing here: only the initial attempts of the models with branch cells are counted.
    branch_models = {o[0] for o in plan.orders["branch"]}
    gating = [x for x in plan.initial if x[0] in branch_models]
    for model, cell, qid, rep in gating:
        route = plan.route_models[model]
        key = attempt_key(qid, route, rep, cell)
        brief = briefs.get(key)
        phase = phase_of[cell]
        if brief is None:
            if claim_path(claims_dir, key).exists():
                why = (f"claimed, no record (a crashed or stopped attempt): run phase {phase} with --reclaim once "
                       f"its lease is stale")
            else:
                why = f"no record: run phase {phase} (again)"
        elif brief.failure_type == "infrastructure":
            why = (f"failed:infrastructure on every try ({brief.path.name}): set it aside with "
                   f"{relabel_hint(cell, route, qid, rep, out)}, then run phase {phase} again")
        else:
            continue
        problems.append(f"{model} {cell} {qid} rep {rep}: {why}")
    if problems:
        raise Refused(f"phases natural and prompted are not complete: {len(problems)} of {len(gating)} planned "
                      f"attempts of graph {plan.graph} have no counted record, so no branch may start:\n  "
                      + "\n  ".join(problems[:50]) + ("\n  ..." if len(problems) > 50 else ""))


def require_current_parents(plan: GridPlan, briefs: dict[tuple, RecordBrief], out: Path) -> None:
    """Every recorded branch of the phase was branched from the record now in OUT for its parent. A parent set
    aside and re-run since leaves its branches checking an answer that is no longer the cohort's: refused,
    never kept; setting the branch aside lets this phase branch it again from the current parent."""
    problems = []
    for model, cell, qid, rep in plan.order:
        route = plan.route_models[model]
        brief = briefs.get(attempt_key(qid, route, rep, cell))
        if brief is None:
            continue
        parent_cell = plan.cell(model, cell)["parent"]
        parent = briefs.get(attempt_key(qid, route, rep, parent_cell))
        if parent is None or brief.branched_from != parent.run_id:
            now = "none" if parent is None else parent.run_id
            problems.append(f"{model} {cell} {qid} rep {rep} ({brief.path.name}): branched from {brief.branched_from}, "
                            f"the current {parent_cell} record is {now}; set it aside with "
                            f"{relabel_hint(cell, route, qid, rep, out)}")
    if problems:
        raise Refused(f"{len(problems)} recorded branch attempts were branched from a parent record that is no "
                      f"longer the one in {out}:\n  " + "\n  ".join(problems[:50])
                      + ("\n  ..." if len(problems) > 50 else ""))


def branch_parent_state(parent: dict | None) -> str:
    """What a branch attempt can do with its parent record (its outcome is read): "ready" (it submitted: its
    answer is checked), "inherited" (it did not submit: nothing to check, the initial answer stands, and no
    branch record is written) or "held_back" (missing, or failed infrastructure: unscored and to be re-run;
    the branch phase does not start while any initial attempt is so, ``require_initial_complete``)."""
    if parent is None:
        return "held_back"
    outcome = parent.get("outcome") or {}
    if outcome.get("failure_type") == "infrastructure":
        return "held_back"
    return "ready" if outcome.get("status") == "submitted" else "inherited"


# --- the run ----------------------------------------------------------------------------


def run(argv: list[str] | None = None, *, models: dict[str, ModelSpec] = MODELS,
        sparql_factory: Callable[[argparse.Namespace], Any] | None = None,
        log: Callable[[str], None] | None = None, sleep: Callable[[float], None] = time.sleep) -> int:
    args = parse_args(argv, models)
    sink = log or (lambda msg: print(msg, file=sys.stderr, flush=True))

    def say(msg: str) -> None:
        sink(redact_secrets(str(msg))[0])

    try:
        if args.preflight_log is not None:
            return _preflight(args, say)
        return _run(args, models, sparql_factory, say, sleep)
    except Refused as exc:
        private_log = getattr(args, "private_log", None)
        if private_log is not None:  # a store read through a gate: a reason can quote its description or sources
            _log_private(private_log, f"error: {exc}")
            say(f"error: refused (the reason is in {private_log})")
        else:
            say(f"error: {exc}")
        return 2
    except Exception as exc:  # printed through the redaction, never as a raw traceback
        private_log = getattr(args, "private_log", None)
        if private_log is not None:  # a store read through a gate: the message can carry its content
            _log_private(private_log, f"error: {type(exc).__name__}: {exc}")
            say(f"error: {type(exc).__name__} (its message is in {private_log})")
        else:
            say(f"error: {type(exc).__name__}: {exc}")
        return 2


def _log_private(path: Path, msg: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as fh:
        fh.write(f"{utc_now()} {redact_secrets(str(msg))[0]}\n")


def gated_store_of(store: dict, store_json: Path) -> gated.GatedStore | None:
    """The private part of a store read through a query gate, its files and its description checked (outside the
    repository, owner-only); None for any other store."""
    try:
        gs = gated.gated_store(store)
        if gs is not None:
            gated.check_private_path(store_json, REPO_ROOT, "the store description")
            gated.check_private(gs, REPO_ROOT)
    except (gated.DescriptionError, OSError) as exc:
        raise Refused(f"store description: {exc}") from None
    return gs


def _check_gated(gs: gated.GatedStore, args: argparse.Namespace) -> None:
    """The run's endpoint is the description's, and its client time-out the gated store's (60 s, so the server's
    120 s; or the description's own query_timeout)."""
    if gs.endpoint is not None and gs.endpoint != args.endpoint:
        raise Refused("--endpoint is not the endpoint the store description names")
    if gs.timeout is not None:
        if args.sparql_timeout != gs.timeout.client_s:
            raise Refused(f"the store description sets a {gs.timeout.client_s:g} s client time-out (refusing answers "
                          f"from {gs.timeout.refuse_from_s:g} s, the server's {gs.timeout.server_s:g} s), not "
                          f"{args.sparql_timeout:g} s")
    elif args.sparql_timeout != gated.CLIENT_TIMEOUT_S:
        raise Refused(f"a store read through a query gate runs with a {gated.CLIENT_TIMEOUT_S:g} s client time-out (the "
                      f"server's {2 * gated.CLIENT_TIMEOUT_S:g} s), not {args.sparql_timeout:g} s")


def gate_start_level(events_path: Path, setting: dict, queries_path: Path | None = None) -> tuple[int, str | None]:
    """The size a gate of several slots starts at: never above the lowest size an earlier step-down of this OUT
    reached, whatever setting that start had (a step-down is permanent for the run: a resume, a new setting or more
    slots never step it back up); else all its slots. A configured size is no step-down: a run started with fewer
    slots may be given more by its setting. The step-downs are those logged and, with ``queries_path``, those
    recomputed from each earlier start's logged queries (its setting and size from its start event), so a step-down
    whose event could not be written, or a crash before it, is not lost. Only step-downs of the setting's own
    ``rule_version`` count: a new version of the rule starts at its configured size, and the history of the others
    stays as it was."""
    version = setting.get("rule_version")

    def same_version(e: dict) -> bool:  # an event of the same version of the rule (an older setting: version None)
        return (e.get("setting") or {}).get("rule_version") == version

    events = read_events(events_path)
    reached = [e["to"] for e in events if e.get("kind") == "step_down" and isinstance(e.get("to"), int) and same_version(e)]
    if queries_path is not None and queries_path.exists():
        lines = read_query_lines(queries_path)
        for start in (e for e in events if e.get("kind") == "start" and same_version(e)):
            mine = [x for x in lines if x.get("start") == start.get("start")]
            if mine:
                reached += [e["to"] for e in recompute(mine, parse_setting(start["setting"]), start["to"])
                            if e["kind"] == "step_down"]
    if reached and min(reached) < setting["slots"]:
        return min(reached), "stepped down in an earlier start"
    return setting["slots"], None


def read_query_lines(path: Path) -> list[dict]:
    """The step-down rule's query log, strictly: every line whole (newline-terminated JSON). A line the writer cut
    short was not counted by the rule (append_query_line), so a log ending without a newline, or with a line that is
    not JSON, is refused rather than read differently from how the rule counted it."""
    text = path.read_text(encoding="utf-8")
    if text and not text.endswith("\n"):
        raise Refused(f"{path.name} ends in a line cut short: the step-down history cannot be read as the rule counted "
                      f"it (check the file before resuming)")
    try:
        return [json.loads(line) for line in text.split("\n")[:-1]]
    except ValueError:
        raise Refused(f"{path.name} holds a line that is not whole: the step-down history cannot be read as the rule "
                      f"counted it (check the file before resuming)") from None


def append_query_line(path: Path, line: dict) -> None:
    """Append one line of the step-down rule's query log (created owner-only). It raises only when the line is not
    whole on disk (nothing written, or cut short), never after a whole line was written, so the rule counts exactly
    the queries a recompute from the log will count (gate_step_down.TimeoutStepDown)."""
    data = (json.dumps(line, sort_keys=True) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        written = os.write(fd, data)
    finally:
        try:
            os.close(fd)
        except OSError:  # the line is on disk or not by now; closing cannot change which
            pass
    if written != len(data):
        raise OSError(f"the query log line was cut short ({written} of {len(data)} bytes)")


def gate_rule_change(events_path: Path, setting: dict) -> dict | None:
    """The change of the rule's version from the last earlier start of this OUT to ``setting``, or None (no earlier
    start, or the same version)."""
    starts = [e for e in read_events(events_path) if e.get("kind") == "start"]
    if not starts:
        return None
    before = (starts[-1].get("setting") or {}).get("rule_version")
    if before == setting.get("rule_version"):
        return None
    return {"from_version": before, "to_version": setting.get("rule_version"), "note": setting.get("rule_note")}


def start_gate_rule(rule: TimeoutStepDown, start: dict, out: Path, started_at: str, say: Callable[[str], None],
                    public: Callable[[str], None] | None, halt: threading.Event) -> None:
    """Log the start of a gate of several slots and connect its rule's logs: every query it takes to
    OUT/gate-queries.jsonl, each event to OUT/gate-events.jsonl, the manifest's ``query_gate_events`` and the private
    log, and a count-only line out. A log that cannot be written sets ``halt``: no new attempt starts (the running ones
    finish, their results standing), since a resume reads these logs for the size to start at."""
    events_path, queries_path, manifest_path = out / GATE_EVENTS_FILE, out / GATE_QUERIES_FILE, out / "manifest.json"

    def tell(msg: str) -> None:
        say(msg)
        if public is not None:
            public(msg)

    def on_query(line: dict) -> None:
        append_query_line(queries_path, {**line, "start": started_at})

    def on_event(event: dict) -> None:
        full = {**event, "start": started_at, "setting": start["setting"]}
        with MANIFEST_LOCK:
            gated.append_private(events_path, full)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.setdefault("query_gate_events", []).append(full)
            _write_json_atomic(manifest_path, manifest)
        what = (f"step down {event['from']} -> {event['to']} slots" if event["kind"] == "step_down"
                else f"trigger at the lowest size, {event['to']} slots, kept")
        tell(f"gate: {what} at {event['at']} ({event['window']['timeouts']} of the last {event['window']['queries']} "
             f"queries timed out; {GATE_EVENTS_FILE})")

    store_changed = start.get("store_changed")
    if store_changed is not None:  # another endpoint or version of the store (check_manifest): on record first
        gated.append_private(events_path, {"at": started_at, "kind": "store_changed", "start": started_at, "from": None,
                                           "to": start["allowed"], **store_changed, "setting": start["setting"]})
    change = start.get("rule_changed")
    if change is not None:  # a new version of the rule: on record before the start it begins at
        gated.append_private(events_path, {"at": started_at, "kind": "rule_changed", "start": started_at, "from": None,
                                           "to": start["allowed"], **change, "setting": start["setting"]})
        say(f"gate: rule changed (version {change['from_version']} -> {change['to_version']}): {change['note']}")
        if public is not None:  # the note can carry the evidence: the private log only
            public(f"gate: rule changed (version {change['from_version']} -> {change['to_version']}); its note is in "
                   f"the private log")
    gated.append_private(events_path, {"at": started_at, "kind": "start", "start": started_at, "from": None,
                                       "to": start["allowed"], "reason": start["reason"], "setting": start["setting"],
                                       "rule": start["rule"]})
    def on_error(exc: BaseException) -> None:
        halt.set()
        say(f"error: a log of the gate's step-down rule could not be written ({type(exc).__name__}: {exc}); no new "
            f"attempt starts, and the gate's events in OUT may be incomplete")
        if public is not None:
            public(f"error: a log of the gate's step-down rule could not be written ({type(exc).__name__}); no new "
                   f"attempt starts")

    rule.on_query, rule.on_event, rule.on_error = on_query, on_event, on_error
    tell(f"gate: {start['allowed']} of {start['setting']['slots']} slots allowed"
         + (f" ({start['reason']})" if start["reason"] else "")
         + f"; steps {start['setting']['step_down']}, over {start['rule']['window']} queries")


def _private_out(out: Path) -> None:
    """OUT of a run on a store read through a gate: outside the repository, and everything already in it owner-only."""
    if out.resolve().is_relative_to(REPO_ROOT.resolve()):
        raise Refused(f"{out} is inside the repository; a run on this store writes its OUT outside it")
    if out.is_symlink():
        raise Refused(f"{out} is a symbolic link; a run on this store writes into a directory of its own")
    if not out.exists():
        return
    linked = [p for p in out.rglob("*") if p.is_symlink()]  # a link could carry what is written out of OUT
    if linked:
        raise Refused(f"{out}: {len(linked)} symbolic link(s) in it; OUT holds no links")
    shared = [p for p in (out, *out.rglob("*")) if not gated.owner_only(p)]
    if shared:
        raise Refused(f"{out}: {len(shared)} path(s) in it can be read by others (OUT and everything in it must be "
                      f"owner-only: chmod -R go-rwx)")


def _preflight(args, say) -> int:
    """--preflight-log: the store's triple count through the run's client and gate, into the private log only."""
    store = json.loads(args.store_json.read_text(encoding="utf-8"))
    gs = gated_store_of(store, args.store_json)
    if gs is None:
        raise Refused("--preflight-log is for a store read through a query gate (a description with private_files)")
    _check_gated(gs, args)
    log = Path(args.preflight_log).expanduser()
    if log.resolve().is_relative_to(REPO_ROOT.resolve()):
        raise Refused("--preflight-log is inside the repository; the log is private")
    if log.exists() and not gated.owner_only(log):
        raise Refused("--preflight-log must be readable by its owner only (mode 600)")
    make, _ = gated.client_factory(gs, endpoint=args.endpoint, timeout_s=args.sparql_timeout, row_cap=args.row_cap,
                                   gate_wait_s=args.gate_wait_s)
    ok = gated.preflight_count(make, log, now=utc_now)
    say(f"preflight: {'the count was read' if ok else 'the count could not be read'}; written to {log} (mode 600)")
    return 0 if ok else 2


def refuse_echo_with_reasoning_call(names: list[str] | None, models: dict[str, ModelSpec]) -> None:
    """Refused: a model of ``names`` whose reasoning-call rule would run on a route that echoes reasoning (loop.py,
    ECHO_WITH_REASONING_CALL). Checked from the table alone, before anything is written."""
    for name in names or ():
        spec = models.get(name)
        if spec is not None and spec.reasoning_call_rule is not None and (
                spec.factory().route_info().get("settings") or {}).get("echo_reasoning") is not None:
            raise Refused(f"{name}: {ECHO_WITH_REASONING_CALL}")


def _run(args, models, sparql_factory, say, sleep) -> int:
    """A run, after the gated store's privacy: a store read through a query gate (its description's private_files) has
    its OUT checked and made, and its messages sent to the private log, before anything of the run is read."""
    refuse_echo_with_reasoning_call(args.models, models)  # a grid names its models later (_run_store)
    gs = None
    if args.grid_manifest is None:
        gs = gated_store_of(json.loads(args.store_json.read_text(encoding="utf-8")), args.store_json)
    if gs is None:
        return _run_store(args, models, sparql_factory, say, sleep)
    if sparql_factory is not None:
        raise Refused("a store read through a query gate takes its clients from its description")
    _check_gated(gs, args)
    _private_out(args.out)
    umask = os.umask(0o077)
    try:
        args.out.mkdir(mode=0o700, parents=True, exist_ok=True)
        args.private_log = args.out / "runner.log"
        say(f"private store: the runner's messages are in {args.private_log} (mode 600); here only counts and timings")

        def private(msg: str) -> None:  # every message of the run goes to the private log only
            _log_private(args.private_log, msg)

        runner_lock = None
        if gs.slots > 1:  # one runner at a time steps a gate of several slots down
            try:
                runner_lock = gated.lock_gate_runner(gs)
            except (gated.DescriptionError, OSError) as exc:
                raise Refused(f"the store's gate: {exc}") from None
        try:
            return _run_store(args, models, None, private, sleep, gs=gs, public=say)
        finally:
            if runner_lock is not None:
                os.close(runner_lock)
    finally:
        os.umask(umask)


def _run_store(args, models, sparql_factory, say, sleep, *, gs: gated.GatedStore | None = None,
               public: Callable[[str], None] | None = None) -> int:
    say(f"redaction: environment variables treated as secrets: {', '.join(secret_names()) or 'none'}")
    # first: every record and the manifest name the code they ran from (F1), so code that cannot be named never runs
    try:
        code = require_identified_code(harness_commit())
    except CodeNotIdentified as exc:
        raise Refused(str(exc)) from None
    say(f"code: commit {code['git_commit']} from {code['source']}, tree {code['tree_sha256']}"
        + (" (uncommitted changes)" if code["git_dirty"] else ""))
    plan = None
    if args.grid_manifest is not None:  # the gates come first: nothing is written before they pass
        plan = load_grid_plan(args.grid_manifest, args.grid_spec, args.grid_root, args.graph, args.phase)
        apply_grid_plan(args, plan)
        if gated.gated_store(json.loads(args.store_json.read_text(encoding="utf-8"))) is not None:
            raise Refused("a store read through a query gate does not run in a grid (no grid names one yet)")
        # change-log entry 180: a model the spec gives its own request time-out or turn cap runs with them, as the
        # manifest's model_settings recorded them (the grid build takes the model through the same grid_models)
        try:
            models = grid_models(models, plan.spec)
        except ValueError as exc:
            raise Refused(f"the grid spec's per-model runner settings: {exc}") from None
        say_runner_settings(plan.spec.get("runner_settings"), say)
        say(f"grid: graph {plan.graph}, phase {plan.phase}, seed {plan.seed}: {len(plan.order)} attempts over "
            f"{len(plan.models)} model(s) and {len({o[1] for o in plan.order})} cell(s), split {plan.spec['split']}"
            + (f", pre-registration frozen at {plan.freeze['commit'][:12]}" if plan.freeze else ""))
    problems = []
    for name in args.models:
        spec = models.get(name)
        if spec is None:
            problems.append(f"unknown model {name!r}; known: {', '.join(models)}")
        elif spec.key_envs and not any((os.environ.get(e) or "").strip() for e in spec.key_envs):
            problems.append(f"{name}: no API key; set {' or '.join(spec.key_envs)}")
    if plan is not None:
        questions = plan.questions
    else:
        questions = load_questions(args.questions, args.split)
        if args.limit is not None:
            questions = questions[: args.limit]
        if args.runner_settings is not None:  # change-log entry 180: as a grid spec's, through the same grid_models
            models, args.runner_settings_given = plain_runner_settings(args.runner_settings, args.models, models)
            say_runner_settings(args.runner_settings_given, say)
        # before anything asks a store or a model: a run of any split but dev is opened by the frozen pre-registration
        args.freeze = plain_split_gate(args, questions)
        if args.freeze is not None:
            say(f"split {args.split}: pre-registration frozen at {args.freeze['commit'][:12]} "
                f"(preregistration sha256 {args.freeze['preregistration_sha256'][:12]}; origin's main "
                f"{str(args.freeze.get('origin_main'))[:12]})")
    if not questions:
        problems.append(f"no questions with split {args.split!r} in {args.questions}")
    if problems:
        for msg in problems:
            say(f"error: {msg}")
        if public is not None:
            public(f"error: {len(problems)} problem(s) before the start (in {args.private_log})")
        return 2
    scorer = frozen_scorer_digest()  # before anything is written: a run with another scorer is refused
    say(f"scorer: digest {scorer} (the frozen scorer)")
    graph = run_graph(json.loads(args.store_json.read_text(encoding="utf-8")))
    for name in args.models:  # likewise a model without a tool-result cap for this graph
        try:
            models[name].result_chars_for(graph)
        except KeyError as exc:
            raise Refused(f"{name}: {exc.args[0]}; a run on this graph needs one in its MODELS entry") from None
    refuse_echo_with_reasoning_call(args.models, models)  # and a grid's, now that its plan has named them
    return _run_out(args, questions, models, sparql_factory, say, sleep, plan, scorer, gs=gs, public=public)


def _run_out(args, questions, models, sparql_factory, say, sleep, plan, scorer, *, gs=None, public=None) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    fs_type = filesystem_type(args.out)
    if is_network_fs(fs_type):
        say(f"warning: {args.out} is on a network file system ({fs_type}); an OUT directory should be on a local "
            f"file system and used from one host at a time (the lock holds per host; claims are only a "
            f"secondary guard)")
    lock_fd = acquire_lock(args.out)
    try:
        return _run_locked(args, questions, models, sparql_factory, say, sleep, plan, scorer, gs=gs, public=public)
    finally:
        release_lock(lock_fd)


def checker_args(args: argparse.Namespace) -> argparse.Namespace:
    """The run's arguments as the online checker's store client takes them (science, design log, change-log entry 125):
    the checker's own row cap (none: rule 39's retrieval budget bounds its reads) and time-out, the offline replays';
    every other argument the run's. A factory a caller passes (``run(sparql_factory=...)``) is asked with these for the
    checker's client, and with the run's own for the agents'."""
    return argparse.Namespace(**{**vars(args), "row_cap": CHECKER_ROW_CAP, "sparql_timeout": CHECKER_TIMEOUT_S})


def store_client_factories(args: argparse.Namespace, sparql_factory: Callable[[argparse.Namespace], Any] | None,
                           checker_factory: Callable[[argparse.Namespace], Any] | None = None,
                           ) -> tuple[Callable[[argparse.Namespace], Any], Callable[[argparse.Namespace], Any]]:
    """The agents' store clients and the online checker's (entry 125): the agents read with the run's row cap and
    time-out (pinned in the run's fingerprint), the checker through its own client without that cap, built as the
    offline replays build theirs, so an online decision is the offline one on the same output. With no factory given,
    one shared client each; a gated store's ``checker_factory`` (its own clients on the store's one gate) is kept; a
    factory a caller passed makes both, the checker's asked with ``checker_args``."""
    if sparql_factory is None:
        from harness.sparql import SparqlClient

        shared = SparqlClient(args.endpoint, timeout_s=args.sparql_timeout, row_cap=args.row_cap)
        made: list[Any] = []  # the checker's, made on its first use only: a run with no checker makes none
        lock = threading.Lock()

        def checker(_args: argparse.Namespace) -> Any:
            with lock:
                if not made:
                    made.append(SparqlClient(args.endpoint, timeout_s=CHECKER_TIMEOUT_S, row_cap=CHECKER_ROW_CAP))
                return made[0]

        return (lambda _args: shared), checker
    if checker_factory is None:
        given = sparql_factory
        checker_factory = lambda a: given(checker_args(a))  # noqa: E731
    return sparql_factory, checker_factory


def plain_split_gate(args: argparse.Namespace, questions: list[dict]) -> dict | None:
    """A plain run (no grid manifest) that would make calls on questions of any split but dev passes the grid's own
    freeze check (scripts/build_dev_grid.py split_gate, the same function, with a grid spec's freeze block made of
    --prereg-repo and --freeze-commit, a relative repository resolved under the code root), before any store query or
    model request: every harness gates test-split calls on the line 'Status: frozen' at the commit given to it
    (pre-registration, 'Freezing'; change-log entry 55). The plain path is the only one a store read through a query
    gate can take (a grid refuses one), so a15's test calls are gated here. The split decided is the run's --split and
    that of every question it loaded, so a question file whose loaded records hold a test item is gated too. None for a
    development run, which is unchanged: no gate, nothing recorded. The freeze it returns is pinned in the run's
    fingerprint (``run_fingerprint``), so every start is checked again and a resume keeps the same freeze."""
    splits = sorted({args.split, *(q.get("split") for q in questions)} - {DEV_SPLIT})
    if not splits:
        return None
    split = args.split if args.split != DEV_SPLIT else splits[0]
    spec = {"split": split, "freeze": {"prereg_repo": args.prereg_repo, "freeze_commit": args.freeze_commit}}
    build = _grid_builder()
    try:
        freeze = build.split_gate(spec, REPO_ROOT)
    except build.Refused as exc:
        raise Refused(f"a plain run of split {split!r} makes test-split calls only under the frozen pre-registration: "
                      f"give --prereg-repo and --freeze-commit (a commit whose paper/preregistration.md holds the line "
                      f"'Status: frozen', on its origin's main). {exc}") from None
    if not isinstance(freeze, dict) or not freeze.get("commit") or not freeze.get("preregistration_sha256"):
        raise Refused(f"a plain run of split {split!r}: the freeze check returned no frozen commit")
    return freeze


def run_graph(store: dict) -> str | None:
    """The graph a run is on: its store description's ``name`` (stores/<graph>/store.json), else its ``store``."""
    graph = store.get("name") or store.get("store")
    return graph if isinstance(graph, str) and graph else None


def frozen_scorer_digest() -> str:
    """The scorer's digest, refused unless it is the frozen one (loop.require_frozen_scorer; preregistration §3,
    design log, change-log entry 91): a run scores with the frozen scorer or not at all."""
    try:
        return require_frozen_scorer()
    except ScorerNotFrozen as exc:
        raise Refused(f"{exc}; run from a tree with the frozen scorer") from None


def store_connections(client: Any) -> int | None:
    """The store's current client connections, when its client can read them (VirtuosoClient.active_connections,
    readable without admin rights; design log); None otherwise or on any error."""
    probe = getattr(client, "active_connections", None)
    if not callable(probe):
        inner = getattr(client, "inner", None)
        probe = getattr(inner, "active_connections", None)
    if not callable(probe):
        return None
    try:
        n = probe()
    except Exception:
        return None
    return n if isinstance(n, int) else None


def engine_suspect_classes(path: Path | None) -> tuple[str, ...]:
    """Suspect classes for the engine-suspect flag from the private vocabulary file given with --engine-suspect-vocab
    (harness/engine_suspect.py), or none. Only the explicit option counts, never the environment: the classes are
    written into every record's agent config, so a run on another store must not pick them up by accident."""
    if path is None:
        return ()
    return tuple(suspect_classes_from_vocab(json.loads(Path(path).expanduser().read_text(encoding="utf-8"))))


def _run_locked(args, questions, models, sparql_factory, say, sleep, plan: GridPlan | None = None,
                scorer: str | None = None, *, gs: gated.GatedStore | None = None,
                public: Callable[[str], None] | None = None) -> int:
    # change-log entry 167: a model OUT/model-fronts.json maps contacts the loopback front through a route that keeps
    # its pinned base URL (FrontRoute), in every route built from here (the pins' and each try's); a bad mapping is
    # refused before anything is written
    fronts, fronts_sha = model_fronts(args.out, models, args.models, say)
    if fronts:
        models = {**models, **{n: replace(models[n], factory=_front_factory(models[n].factory, url))
                               for n, url in fronts.items()}}
        say("model fronts: " + ", ".join(f"{n} through {url}" for n, url in sorted(fronts.items()))
            + f" ({MODEL_FRONTS_FILE} sha256 {fronts_sha[:12]})")
    # each start records the two files' sha256 as it read them (only those present)
    start_files = {k: v for k, v in (("model_fronts_sha256", fronts_sha),
                                     ("admitted_fingerprints_sha256", _file_sha256(args.out / ADMITTED_FILE)))
                   if v is not None}
    store = json.loads(args.store_json.read_text(encoding="utf-8"))
    if gs is not None:  # the gate's setting is recorded per start, not fingerprinted or kept in records
        store = gated.public_description(store)
    # The settings every model shares; each model's attempts run with its own caps, transport and guards
    # (ModelSpec.agent_config), keyed by the model alone, so the same for every arm, cohort and phase of the model,
    # in a grid as outside one.
    config = AgentConfig(infra_retry_s=args.infra_retry_s, graph=run_graph(store),
                         engine_suspect_classes=engine_suspect_classes(args.engine_suspect_vocab),
                         condition=args.condition,
                         ontology_block_sha256=ontology_block(args.condition, args.prompts, store))
    query_gate = store_client = checker_factory = gate_rule = gate_start = None
    gate_since = time.time()
    if gs is not None:  # one client per try, all on the store's one gate; a phase ends at 3x its wall cap
        config = replace(config, total_cap_factor=gated.TOTAL_CAP_FACTOR)
        if gs.slots > 1:  # several queries at once, stepped down by the time-out rule (gate_step_down.py)
            allowed, reason = gate_start_level(args.out / GATE_EVENTS_FILE, gs.gate.to_json(),
                                               args.out / GATE_QUERIES_FILE)
            change = gate_rule_change(args.out / GATE_EVENTS_FILE, gs.gate.to_json())
            query_gate = QueryGate(gs.files["gate"], slots=gs.slots, allowed=allowed)
            gate_rule = TimeoutStepDown(query_gate, gs.gate)
            gate_start = {"setting": gs.gate.to_json(), "allowed": allowed, "reason": reason, "rule": rule_json(gs.gate),
                          **({"rule_changed": change} if change is not None else {})}
        sparql_factory, query_gate = gated.client_factory(gs, endpoint=args.endpoint, timeout_s=args.sparql_timeout,
                                                          row_cap=args.row_cap, gate_wait_s=args.gate_wait_s,
                                                          gate=query_gate, monitor=gate_rule)
        # the checker's clients: on the same gate (its slots and step-down level) and with the gate's client time-out,
        # without the agents' row cap; no monitor, since the step-down rule counts the agents' tool calls only
        checker_factory, _ = gated.client_factory(gs, endpoint=args.endpoint, timeout_s=args.sparql_timeout,
                                                  row_cap=CHECKER_ROW_CAP, gate_wait_s=args.gate_wait_s,
                                                  gate=query_gate)
        sample = sparql_factory(args)  # its settings only: nothing is sent
        store_client = {k: v for k, v in sample.settings().items() if k != "gate_wait_s"}
        gated.close_client(sample)

    def tell(msg: str) -> None:  # the run's closing counts: to the private log and, for a private store, out
        say(msg)
        if public is not None:
            public(msg)
    model_configs: dict[str, AgentConfig] = {}
    route_settings: dict[str, dict] = {}
    route_models = {}
    routes = {}
    for name in args.models:
        route = models[name].factory()
        route_models[name] = route.model
        route_settings[name] = route.route_info()["settings"]
        model_configs[name] = models[name].agent_config(config, route_settings[name])
        routes[name] = model_fingerprint(route, model_configs[name])
    records_dir = args.out / "records"
    aborted_dir = args.out / "aborted"
    claims_dir = args.out / "claims"
    index_path = args.out / "runs.jsonl"
    # a HALTED phase starts no attempt until its checker code is fixed, and its fixed set-asides re-run on that code
    require_not_halted(args.out, plan.phase if plan is not None else PLAIN_PHASE, harness_commit())
    # The arms: a run's one prompt condition; in a grid the cells of the phase, each initial cell with its own
    # AgentConfig and each branch cell with its checker arm per role. The phases already pinned in OUT are
    # fingerprinted again, so a later phase cannot change an earlier one's arms.
    configs: dict[str, AgentConfig] = {config.condition: config}
    checker_arms: dict[tuple[str, str], Any] = {}
    switches = None
    if plan is not None:
        from harness.checkloop.online import Switches

        wrong = {n: m for n, m in route_models.items() if m != plan.route_models[n]}
        if wrong:
            raise Refused("routes serve other models than the grid spec names: " + "; ".join(
                f"{n} serves {m!r}, the spec names {plan.route_models[n]!r}" for n, m in wrong.items()))
        # each route is run exactly as the manifest recorded it from the runner's table: its transport and caps
        drift = []
        for name in args.models:
            ms = plan.model_settings[name]
            if (routes[name]["provider"], routes[name]["settings"]) != (ms["provider"], ms["route_settings"]):
                drift.append(f"{name}: route settings {routes[name]['settings']} (provider {routes[name]['provider']}) "
                             f"are not the manifest's {ms['route_settings']} (provider {ms['provider']})")
            if model_caps(models[name], plan.graph) != plan.model_caps(name):
                drift.append(f"{name}: caps {model_caps(models[name], plan.graph)} on {plan.graph} are not the "
                             f"manifest's {plan.model_caps(name)}")
            for field, what in RULE_NAMES.items():
                if getattr(models[name], field) != ms.get(field):
                    drift.append(f"{name}: {what} rule {getattr(models[name], field)!r} is not the manifest's "
                                 f"{ms.get(field)!r}")
        if drift:
            raise Refused("the routes are not the ones the manifest recorded from the runner's model table "
                          "(rebuild the manifest with this table):\n  " + "\n  ".join(drift))
        briefs = record_briefs(records_dir)
        if plan.phase == "branch":  # natural and prompted complete and replayed, every recorded branch on its parent
            require_replayed_records(args.out)
            require_initial_complete(plan, briefs, claims_dir, args.out)
            require_current_parents(plan, briefs, args.out)
        pinned = set((_grid_pins(args.out) or {}).get("phases") or {})
        pin = [ph for ph in GRID_PHASES if ph == plan.phase or ph in pinned]
        blocks = plan.graph_spec.get("prompt_blocks") or {}
        configs = {c: grid_cell_config(config, c, plan.spec["cohorts"][c], (blocks.get(c) or {}).get("sha256"))
                   for ph in pin if ph != "branch" for c in plan.cohorts(ph)}
        if "branch" in pin:  # the sources and the hierarchy are the branch phase's inputs
            checker_arms = grid_arms(plan)
            # rule 23: every switch on; the retry budget is each try's, from its parent's caps (run_try)
            switches = Switches(**plan.spec["switches"])
        phases = {ph: {"seed": plan.seeds[ph], "cells": plan.phase_cells(ph), "inputs": phase_inputs(plan, ph),
                       "arms": phase_arm_fingerprints(plan, ph, configs, checker_arms, switches),
                       "models": plan.model_orders(ph)} for ph in pin}
    done = set(briefs) if plan is not None else existing_keys(records_dir)
    # checked before the identity checks below, which may write the manifest, so a refused one leaves nothing; it is
    # recorded after them, as the start's last step before any attempt (record_deployment_changes)
    deployments = deployment_changes(args.out, args.deployment_change, args.ruling,
                                     list(dict.fromkeys(args.models)), route_models)
    if plan is not None:
        seed, seed_source = plan.seed, f"grid manifest, phase {plan.phase}"
        check_grid_manifest(args.out, grid_fingerprint(args, store, config, routes, plan, phases, scorer, store_client),
                            briefs, plan, say=say)
    else:
        recorded_seed = recorded_order_seed(args.out)
        if args.order_seed is not None and recorded_seed is not None and args.order_seed != recorded_seed:
            raise Refused(f"{args.out} was started with order seed {recorded_seed}, not {args.order_seed}; a resume "
                          f"keeps the order (use a new --out for a new order)")
        if args.order_seed is not None:
            seed, seed_source = args.order_seed, "flag"
        elif recorded_seed is not None:
            seed, seed_source = recorded_seed, "manifest"
        else:
            seed, seed_source = default_order_seed(args.questions), "questions_sha256"
        added = check_manifest(args.out, run_fingerprint(args, store, config, routes, scorer, store_client),
                               has_records=bool(done), say=say, allow_code_change=args.allow_code_change,
                               start_fields={**({"query_gate": gate_start} if gate_start is not None else {}),
                                             **start_files,
                                             **({"runner_settings": args.runner_settings_given}
                                                if getattr(args, "runner_settings_given", None) else {})} or None,
                               allow_store_change=args.allow_store_change)
        if args.allow_store_change is not None:
            store_changed = json.loads((args.out / "manifest.json").read_text(encoding="utf-8"))["starts"][-1].get(
                "store_changed")
            if store_changed is not None:
                if public is not None:
                    public(f"manifest: store change recorded ({store_changed['pre_change_records']} attempt(s) made "
                           f"before it)")
                if gate_start is not None:
                    gate_start["store_changed"] = store_changed
        if added:
            say(f"manifest: recorded settings for {', '.join(added)}")
    model_names = list(dict.fromkeys(args.models))
    if plan is not None:
        order = list(plan.order)
        arms = list(dict.fromkeys(o[1] for o in order))
        manifest_path = args.out / "manifest.json"
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if plan.freeze is not None:  # the latest start's freeze check, with its fetch of origin's main
            data["grid_freeze"] = plan.freeze
        # every start: its phase, each model's workers and the Gateway budget they were checked against (the workers
        # are no pin: they may change between starts), and the code it runs on (F1: the phases of one OUT may run on
        # different commits; each record names its own too)
        data.setdefault("starts", []).append({"at": utc_now(), "phase": plan.phase, "concurrency": args.concurrency_for,
                                              **plan.gateway, "harness": harness_commit(),
                                              **({"query_gate": gate_start} if gate_start is not None else {}),
                                              **start_files})
        _write_json_atomic(manifest_path, data)
    else:
        arms = [config.condition]
        order = interleaved_order(model_names, arms, [q["id"] for q in questions], args.reps, seed)
    order_sha = order_sha256(order)
    phase_done = {k for k in done if k[0] in arms} if plan is not None else done
    record_order(args.out, seed=seed, seed_source=seed_source, order=order, resumed=bool(phase_done), say=say,
                 phase=plan.phase if plan is not None else None)
    runner_started_at = utc_now()  # with start_index, orders the actual starts across resumes
    say(f"order: {len(order)} attempts over {len(model_names)} model(s) and {len(arms)} arm(s), shuffled with seed "
        f"{seed} ({seed_source}); order sha256 {order_sha}")
    records_dir.mkdir(parents=True, exist_ok=True)
    claims_dir.mkdir(parents=True, exist_ok=True)
    awaiting = set(awaiting_checker_fix(args.out)) - done  # set aside on a divergence, no fix recorded: never re-run
    if args.reclaim:  # nor reclaimed: their claims stay until the fix
        say(f"reclaim: deleted {reclaim(claims_dir, done | awaiting, timeout_s=args.claim_timeout_s)} stale "
            f"claim(s); live leases are kept")
    claimed = claimed_keys(claims_dir) - done  # claims without a record: stale (or another runner's)

    sparql_factory, checker_factory = store_client_factories(args, sparql_factory, checker_factory)
    index_lock = threading.Lock()
    stops = {name: threading.Event() for name in args.models}
    # the branch phase's periodic fresh-process recompute (science, design log, 20 Oct 10:39): its halt stops new launches
    periodic = None
    if plan is not None and plan.phase == "branch" and args.periodic_check_s > 0:
        periodic = PeriodicCheck(args.out, plan.phase, lambda stop: decisions_in_fresh_process(
            args.out, plan, say, nice=PERIODIC_NICE, stop=stop), say, interval_s=args.periodic_check_s,
            max_load=args.periodic_check_max_load)
    halt = periodic.halt if periodic is not None else threading.Event()
    if gate_rule is not None:  # a log of the gate's rule that cannot be written halts new launches
        start_gate_rule(gate_rule, gate_start, args.out, runner_started_at, say, public, halt)
    counts = {"run": 0, "skipped": 0, "stale_claims": 0, "awaiting_fix": 0, "claimed_elsewhere": 0, "aborted": 0,
              "errors": 0, "infrastructure": 0, "infra_reruns": 0, "infra_final": 0, "models_stopped_infrastructure": 0,
              "inherited": 0, "harness_defects": 0, "model_changed": 0, "models_held": 0}
    infra_dir = args.out / "infrastructure"
    harness_dir = args.out / "harness"
    # after the freeze (a test grid, or a plain run the freeze opened) none continues in place: a served-version change,
    # the model held
    deployments = deployments_after_freeze(args.out, deployments, say)
    record_deployment_changes(args.out, deployments, say)  # this start's checked deployment changes, if any
    # each model's first served id in this OUT (design log, change-log entry 91), and its admitted fingerprints (entry 167)
    served = ServedModels(args.out, say=say)
    changed_models: set[str] = set()  # models whose served id changed in this run (counted once each)
    infra_streak = {name: 0 for name in args.models}  # consecutive attempts that failed infrastructure on every try
    counts_lock = threading.Lock()
    held_models: set[str] = set()  # held by the infrastructure-fault monitor in this start (counted once each)

    def bump(key: str) -> None:
        with counts_lock:
            counts[key] += 1

    def held_now(name: str) -> bool:
        """Whether the infrastructure-fault monitor holds ``name`` (OUT/holds/<name>.json; change-log entry 129): read
        before each claim of the model, so a hold written while the phase runs takes effect at its next attempt. A held
        model starts no attempt from then on, as one whose served version changed (attempts in flight go on); the
        others run."""
        hold = read_hold(args.out, name)
        if hold is None:
            return False
        stops[name].set()
        with counts_lock:
            first = name not in held_models
            held_models.add(name)
            if first:
                counts["models_held"] += 1
        if first and hold.get("kind") == "served_version_change":
            say(f"warning: {name}: held as a served-version change after the freeze "
                f"({served_version_hold_path(args.out, name)}): no attempt of {name} starts in this OUT, the others "
                f"go on")
        elif first:
            say(f"warning: {name}: held by the infrastructure-fault monitor ({hold_path(args.out, name)}, window from "
                f"{hold.get('window_start')}): no attempt of {name} starts from now on, the others go on; the monitor "
                f"releases it when its probe passes, and a restart of this OUT resumes it")
        return True

    for name in model_names:  # a model held before this start runs nothing
        held_now(name)

    # Concurrency: configured workers per model, the allowed size (which a resume keeps where an earlier
    # start stepped it down), the latency baseline, and the events log of every change.
    events_path = args.out / EVENTS_FILE
    earlier_events = read_events(events_path)
    configured = {name: args.concurrency_for.get(name, args.concurrency) for name in model_names}
    allowed, guards, baseline_source, wallcap_source, start_reason = {}, {}, {}, {}, {}
    for name in model_names:
        mine = [e for e in earlier_events if e.get("model") == name]
        levels = [e for e in mine if e.get("kind") in ("start", "step_down")]
        allowed[name], start_reason[name] = configured[name], None
        if levels and levels[-1].get("configured") == configured[name] and levels[-1]["to"] < configured[name]:
            if args.reset_concurrency:
                start_reason[name] = "--reset-concurrency"
            else:
                allowed[name] = levels[-1]["to"]
                start_reason[name] = "stepped down in an earlier start"
                say(f"{name}: {allowed[name]} attempts in flight of {configured[name]} configured: stepped down in "
                    f"an earlier start of this OUT and never stepped up automatically (--reset-concurrency "
                    f"starts at {configured[name]})")
        recorded = [e["baseline_s"] for e in mine if e.get("baseline_s")]  # from the flag or a first window
        if name in args.latency_baseline:
            baseline, baseline_source[name] = args.latency_baseline[name], "flag"
        elif name in args.latency_baseline_calm:  # measured in this start, never the value of an earlier one
            baseline, baseline_source[name] = None, "calm_first_50"
            say(f"{name}: the latency baseline is the median of its first {CALM_CALLS} calm calls in this start; its "
                f"latency trigger is inactive until then"
                + ("" if len(set(model_names)) >= STALL_MIN_MODELS else
                   f" (sync minutes only: {len(set(model_names))} model(s) in this runner, too few for an all-model "
                   f"stall bin)"))
        elif recorded:
            baseline, baseline_source[name] = recorded[-1], "recorded"
        else:
            baseline, baseline_source[name] = None, None
            say(f"{name}: no --latency-baseline-s; the median per-turn call time of the first {LATENCY_WINDOW} "
                f"tries becomes the baseline (it reflects this run's own load, not the low-concurrency one)")
        recorded_wallcap = [e["wallcap_baseline"] for e in mine if e.get("wallcap_baseline") is not None]
        if name in args.wallcap_baseline:
            wallcap, wallcap_source[name] = args.wallcap_baseline[name], "flag"
        elif recorded_wallcap:  # a resume keeps the trigger the run started with
            wallcap, wallcap_source[name] = recorded_wallcap[-1], "recorded"
        else:
            wallcap, wallcap_source[name] = None, None
            say(f"{name}: wall-cap trigger inactive (no --wallcap-baseline); the latency and infrastructure "
                f"triggers still apply")
        guards[name] = GuardRails(baseline, wallcap, first_window=name not in args.latency_baseline_calm)
    throttle = Throttle(configured, allowed)
    events_lock = threading.Lock()  # also serialises the guard rails' updates
    calm = (CalmBaseline(model_names, args.latency_baseline_calm & set(model_names))
            if args.latency_baseline_calm & set(model_names) else None)

    def log_event(name: str, kind: str, **fields: Any) -> None:
        event = {"at": utc_now(), "kind": kind, "model": name, "route_model": route_models[name],
                 "configured": configured[name], "from": None, "to": None, "reason": None,
                 "baseline_s": guards[name].baseline_s, "baseline_source": baseline_source[name],
                 "wallcap_baseline": guards[name].wallcap_baseline, "wallcap_source": wallcap_source[name],
                 "window": None,
                 **fields}
        with events_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
            fh.flush()

    with events_lock:
        for name in model_names:
            log_event(name, "start", to=allowed[name], reason=start_reason[name])

    def calm_check(now: float) -> None:
        """The calm baselines now settled (held under events_lock): set and logged, each once."""
        for model, measured in calm.ready(now).items():
            guards[model].baseline_s = measured["baseline_s"]
            reason = (f"median call time of the first {measured['calls']} calm calls "
                      f"({measured['first_call_at']} to {measured['last_call_at']}; skipped "
                      f"{measured['skipped_sync']} in sync minutes"
                      + (f", {measured['skipped_stall']} in all-model stall bins)" if measured["stall_detection"] else
                         f"; no all-model stall bins with {measured['models']} model(s) in this runner)"))
            say(f"{model}: latency baseline {measured['baseline_s']:.3g} s ({reason})")
            log_event(model, "baseline", reason=reason, window=measured)

    def observe(name: str, stats: dict, epoch: int) -> None:
        with events_lock:
            if calm is not None:  # before the guard reads this attempt: a baseline settled meanwhile applies to it
                calm_check(time.time())
            for event in guards[name].observe(stats, epoch):
                if event["kind"] == "baseline":
                    baseline_source[name] = "first_window"
                    say(f"{name}: latency baseline {guards[name].baseline_s:.3g} s ({event['reason']})")
                    log_event(name, "baseline", reason=event["reason"], window=event["window"])
                    continue
                old, new = throttle.step_down(name)
                if new == old:
                    say(f"warning: [{name}] guard rails: {event['reason']}; already at 1 attempt in flight")
                    continue
                say(f"warning: [{name}] step down: {old} -> {new} attempts in flight ({event['reason']})")
                log_event(name, "step_down", **{"from": old, "to": new}, triggers=event["triggers"],
                          reason=event["reason"], window=event["window"])

    # Leases: a running attempt's claim is refreshed by a heartbeat thread. Releasing a claim and
    # refreshing one hold the same lock, so a refresh can never re-create a claim just released.
    leases_lock = threading.Lock()
    active_leases: dict[Path, float] = {}  # our claims being refreshed -> their claimed_at
    heartbeat_stop = threading.Event()

    def heartbeat() -> None:
        while not heartbeat_stop.wait(args.claim_heartbeat_s):
            with leases_lock:
                for path, claimed_at in list(active_leases.items()):
                    try:
                        ours = refresh_lease(path, claimed_at)
                    except OSError:
                        continue
                    if not ours:
                        del active_leases[path]
                        say(f"warning: lost the lease {path.name}: another runner reclaimed it")

    def release(path: Path, *, delete: bool) -> None:
        with leases_lock:
            mine = active_leases.pop(path, None) is not None
            if delete and mine:  # never delete a lease another runner holds now
                path.unlink(missing_ok=True)

    def key_of(name: str, arm: str, qid: str, rep: int) -> tuple:
        return attempt_key(qid, route_models[name], rep, arm if plan is not None else None)

    def attempt(name: str, ticket: int, position: int, arm: str, q: dict, rep: int, parent: Path | None) -> None:
        started: dict[str, Any] = {}  # set at the actual start: the first provider request
        try:
            if stops[name].is_set() or halt.is_set() or held_now(name):
                return
            key = key_of(name, arm, q["id"], rep)
            claim = try_claim(claims_dir, key, q["split"])
            if claim is None:  # claimed since this run looked: by another runner, never run twice
                bump("claimed_elsewhere")
                return
            gate.wait(name, ticket)  # the model's next attempt in order; the next waits until this one starts
            if halt.is_set():  # the phase halted while it waited its turn: it launches nothing, its claim goes
                claim.unlink(missing_ok=True)
                return
            if held_now(name):  # held by the infrastructure-fault monitor while it waited: likewise
                claim.unlink(missing_ok=True)
                return
            with events_lock:
                epoch = guards[name].epoch  # the size this attempt runs at

            def on_start(at_start: Callable[[], None]) -> None:
                # the launch boundary, just before the first provider request: a halt set since the checks above (the
                # worker may have blocked making a store client, or in the checker) stops it here, and so does a hold
                # the infrastructure-fault monitor wrote meanwhile
                if halt.is_set():
                    raise HaltedBeforeStart(f"phase halted before {q['id']} rep {rep} of {name} started")
                if held_now(name):
                    raise HeldBeforeStart(f"{name} held before {q['id']} rep {rep} started")

                def stamp_then_check() -> None:
                    at_start()  # the start's stamp reads the store's connections, which can block
                    if halt.is_set():  # before the gate counts the start: the finally below skips this ticket
                        raise HaltedBeforeStart(f"phase halted before {q['id']} rep {rep} of {name} started")
                    if held_now(name):
                        raise HeldBeforeStart(f"{name} held before {q['id']} rep {rep} started")

                if calm is not None:  # before the first request: no turn of it can start earlier
                    calm.started(key, time.time() - CALM_START_MARGIN_S)
                    started["calm"] = key
                started["index"] = gate.start(name, ticket, stamp_then_check)

            run_attempt_claimed(name, position, arm, q, rep, claim, epoch, on_start, started, parent)
        finally:
            if calm is not None and "calm" in started:
                calm.finished(started["calm"])
                with events_lock:  # its bins may be final now
                    calm_check(time.time())
            if "index" not in started:  # no provider request (stopped, claimed elsewhere, an error, no turn)
                gate.skip(name, ticket)

    def cell_config(name: str, arm: str) -> AgentConfig:
        """An initial cell's config for one model: the arm's with the model's caps on the graph, transport and guards
        (ModelSpec.agent_config), in a grid as outside one."""
        return models[name].agent_config(configs[arm], route_settings[name])

    def run_try(name: str, arm: str, q: dict, rep: int, route: Provider, client: Any, on_start: Callable[[], None],
                parent: RunRecord | None,
                served_check: Callable[[str | None, str | None], Any] | None = None) -> RunRecord:
        """One try: an initial cell's full agent run under its AgentConfig with its model's own caps, or a branch
        cell's check of its parent's output with, on a flag, one live retry (which starts at the retry's first model
        request, under the parent's config and caps). Every turn is checked against the model's first served id
        (``served_check``)."""
        if parent is None:
            return run_attempt(attempt_question(q), route, client, cell_config(name, arm), question_gold(q),
                               q.get("scoring_rules"), repetition=rep, store=store, sleep=sleep, on_start=on_start,
                               served_check=served_check)
        from harness.agent.replay import config_of
        from harness.checkloop.online import OnlineChecker, RetryBudget, branch_attempt

        cell = plan.cell(name, arm)
        # Pre-registration section 3: the retry gets the attempt's own caps, its parent's (its model's); the nudges
        # are the frozen ones (branch_attempt refuses any other budget)
        parent_config = config_of(parent)
        budget = RetryBudget(max_turns=parent_config.max_turns, max_wall_s=parent_config.max_wall_s,
                             max_nudges=FROZEN_RETRY_BUDGET["max_nudges"])
        checker_client = checker_factory(args)  # its own client: no agents' row cap (entry 125)
        checker = OnlineChecker(checker_arms[(arm, plan.roles[name])], checker_client, cell["mode"],
                                budget=budget, switches=switches)  # one per try: it keeps the try's store log
        siblings, place = plan.sibling_order(name, arm, q["id"], rep)
        try:
            return branch_attempt(parent, checker, route, client, gold_rows=question_gold(q),
                                  rules=q.get("scoring_rules"), sleep=sleep, order_seed=seed, label=arm, order=siblings,
                                  position=place, on_start=on_start, served_check=served_check)
        finally:
            gated.close_client(checker_client)

    def run_attempt_claimed(name: str, position: int, arm: str, q: dict, rep: int, claim: Path, epoch: int,
                            first_start: Callable[[Callable[[], None]], None], started: dict,
                            parent_path: Path | None = None) -> None:
        with leases_lock:
            active_leases[claim] = (_read_claim(claim) or {}).get("claimed_at")
        clients: list = []  # every store client made for the attempt's tries, closed on every way out
        try:
            parent = (RunRecord.from_json(json.loads(parent_path.read_text(encoding="utf-8")))
                      if parent_path is not None else None)
            # every file of the attempt is named by its claim digest, and its marker (key, split) is written before
            # its first try and kept for good, so each file can be placed in its split by name
            key = key_of(name, arm, q["id"], rep)
            digest = claim_digest(key)
            write_attempt_marker(claims_dir, key, q["split"])
            failed_tries: list[str] = []  # run ids of this attempt's tries that failed infrastructure
            failed_recs: list[RunRecord] = []  # and their records, for the guard rails
            first_in_flight = None
            while True:
                route = models[name].factory()
                at_try: dict[str, Any] = {}

                def on_start(first: bool = not failed_tries, at_try: dict = at_try) -> None:
                    """This try's first provider request: snapshot and time it (inside the gate on the
                    attempt's first try, which then lets the model's next attempt through)."""
                    def stamp() -> None:
                        at_try["snapshot"], at_try["at"] = throttle.snapshot(name), utc_now()

                    if first:
                        first_start(stamp)
                    else:
                        stamp()

                client = sparql_factory(args)
                clients.append(client)
                at_try["client"] = client
                # the store's connections as the try starts: read before the attempt's clock starts, since the probe
                # is a query and on a store read through a gate it waits its turn
                at_try["store_connections"] = store_connections(client)

                def served_check(model: str | None, fingerprint: str | None, name: str = name, q_id: str = q["id"],
                                 rep: int = rep):
                    """Each turn's served id and system fingerprint, as it arrives: a change ends the attempt
                    (loop.py) and, from now, no new attempt of the model starts."""
                    change = served.check(name, route_models[name], model, fingerprint, question_id=q_id,
                                          repetition=rep)
                    if change is not None:
                        stops[name].set()
                    return change

                try:
                    if not failed_tries and halt.is_set():  # the halt came while the store client was made
                        raise HaltedBeforeStart(f"phase halted before {q['id']} rep {rep} of {name} started")
                    if not failed_tries and held_now(name):  # so did a hold
                        raise HeldBeforeStart(f"{name} held before {q['id']} rep {rep} started")
                    rec = run_try(name, arm, q, rep, route, client, on_start, parent, served_check)
                except ScoringError as exc:
                    rec = exc.record
                    say(f"warning: [{name}] {q['id']} rep {rep}: scoring failed: {exc.__cause__!r}")
                except HaltedBeforeStart as exc:  # not launched: no record, and its claim goes so a later start runs it
                    release(claim, delete=True)
                    why = ("the infrastructure-fault monitor held the model" if isinstance(exc, HeldBeforeStart)
                           else "the phase halted")
                    say(f"[{name}] {q['id']} rep {rep}: not launched, {why} before its first provider request; its "
                        f"claim is released")
                    return
                end, ended_at = throttle.snapshot(name), utc_now()
                # a try that made no provider request has no start; it is timed from now
                start, started_at = at_try.get("snapshot", end), at_try.get("at", ended_at)
                if first_in_flight is None:
                    first_in_flight = start["in_flight_model"]
                start_index, model_start_index = started.get("index", (None, None))
                rec.load = {"arm": arm, "position": position, "order_size": len(order), "order_seed": seed,
                            "order_sha256": order_sha, "order_key": order_key(seed, (name, arm, q["id"], rep)),
                            "start_index": start_index, "model_start_index": model_start_index,
                            "runner_started_at": runner_started_at,
                            "workers_configured": start["workers_configured"],
                            "workers_allowed": start["workers_allowed"],
                            "in_flight_model_start": start["in_flight_model"],
                            "in_flight_total_start": start["in_flight_total"],
                            "in_flight_model_end": end["in_flight_model"], "in_flight_total_end": end["in_flight_total"],
                            "started_at": started_at, "ended_at": ended_at}
                if calm is not None:  # each try once, as it ends: an attempt that stops early keeps its turns too
                    calm.add(name, [rec])
                end_connections = store_connections(client)
                gated.close_client(client)
                if at_try.get("store_connections") is not None or end_connections is not None:
                    # agent-run/13: the store's own count of current client connections (our probe included)
                    rec.load["store_connections_start"] = at_try.get("store_connections")
                    rec.load["store_connections_end"] = end_connections
                if rec.outcome["failure_type"] == "model_changed":  # a turn served by another id: unscored, aborted
                    changed = dict(rec.outcome["model_changed"])
                    first, now = changed["first_served_model"], changed["served_model"]
                    rec.score = None
                    rec.runner = {"aborted": {"reason": "model_changed", **changed}}
                    rec.write(aborted_dir, prefix=digest)
                    release(claim, delete=True)  # not done: a later run (with the model's new version) takes it
                    served.mark_change(name, route_models[name], rec, changed)
                    stops[name].set()  # set already by the check; kept so the stop never depends on the callback
                    with counts_lock:
                        if name not in changed_models:
                            changed_models.add(name)
                            counts["model_changed"] += 1
                    what = (f"the served model changed from {first!r} to {now!r}" if "system_fingerprint" not in changed
                            else f"the server's system fingerprint changed from "
                                 f"{changed['first_system_fingerprint']!r} to {changed['system_fingerprint']!r}")
                    say(f"error: [{name}] {what} at turn {rec.outcome['turns']} of {q['id']} rep {rep}: the attempt "
                        f"is aborted unscored (aborted/) and no new attempt of {name} starts; see {MODEL_CHANGED_FILE}")
                    return
                if is_config_failure(rec):
                    rec.write(aborted_dir, prefix=digest)
                    release(claim, delete=True)
                    stops[name].set()
                    bump("aborted")
                    say(f"error: [{name}] stopped: {rec.outcome['error_message'][:300]}")
                    return
                if rec.outcome["failure_type"] == "harness_defect":  # our tool code crashed: set aside, re-run later
                    rec.runner = {"infra_tries": len(failed_tries), "infra_try_run_ids": failed_tries}
                    path = rec.write(harness_dir, prefix=digest)
                    write_harness_defect_setaside(harness_dir, path.name, rec)
                    release(claim, delete=True)  # a resume re-runs it, after the fix
                    bump("harness_defects")
                    defect = rec.outcome["harness_defect"]
                    say(f"error: [{name}] {q['id']} rep {rep}: harness defect ({defect['exception']}, stack "
                        f"{defect['stack_sha256'][:12]}): {rec.outcome['error_message'][:200]}; set aside in "
                        f"harness/ as failed:harness, unscored, and re-run by a resume once it is fixed")
                    return
                if rec.outcome["failure_type"] != "infrastructure":
                    break
                # The store or the provider failed: every try keeps its own record (and counts for the guard
                # rails); re-run once the store answers again, or after a pause for the provider.
                failed_tries.append(rec.run_id)
                failed_recs.append(rec)
                rec.runner = {"infra_try": len(failed_tries)}
                rec.write(infra_dir, prefix=digest)
                bump("infrastructure")
                source = rec.outcome.get("infra_source") or "store"
                if len(failed_tries) > args.infra_reruns:
                    say(f"warning: [{name}] {q['id']} rep {rep}: "
                        + (f"store unavailable on all {len(failed_tries)} tries" if source == "store" else
                           f"infrastructure ({source}) on all {len(failed_tries)} tries")
                        + "; recorded as failed:infrastructure")
                    break
                if source == "provider":  # the store is not in question: no probe, a pause (the slot is kept)
                    say(f"warning: [{name}] {q['id']} rep {rep}: provider unavailable "
                        f"({rec.outcome['error_message'][:200]}); waiting {PROVIDER_RERUN_WAIT_S:g} s before re-run "
                        f"{len(failed_tries)} of {args.infra_reruns}")
                    sleep(PROVIDER_RERUN_WAIT_S)
                    bump("infra_reruns")
                    continue
                say(f"warning: [{name}] {q['id']} rep {rep}: store unavailable "
                    f"({rec.outcome['error_message'][:200]}); waiting for it before re-run "
                    f"{len(failed_tries)} of {args.infra_reruns}")
                probe_client = sparql_factory(args)
                try:
                    answered, _ = wait_for_store(probe_client, limit_s=args.infra_retry_s, sleep=sleep)
                finally:
                    gated.close_client(probe_client)
                if not answered:
                    release(claim, delete=True)  # not done: a later run retries it
                    with counts_lock:
                        if not stops[name].is_set():
                            stops[name].set()
                            counts["models_stopped_infrastructure"] += 1
                    say(f"error: [{name}] stopped: the store did not come back within {args.infra_retry_s:g} s")
                    return
                bump("infra_reruns")
            rec.runner = {"infra_tries": len(failed_tries), "infra_try_run_ids": failed_tries}
            final_infra = rec.outcome["failure_type"] == "infrastructure"
            with counts_lock:
                infra_streak[name] = infra_streak[name] + 1 if final_infra else 0
                stop = final_infra and infra_streak[name] >= args.max_infra_failures and not stops[name].is_set()
                if stop:
                    stops[name].set()
                    counts["models_stopped_infrastructure"] += 1
                if final_infra:
                    counts["infra_final"] += 1
            if stop:
                say(f"error: [{name}] stopped after {args.max_infra_failures} consecutive attempts failed "
                    f"infrastructure on every try")
            rec.write(records_dir, prefix=digest)
            if parent is None or "index" in started or failed_recs:  # a branch that asked the model nothing is
                observe(name, attempt_stats(rec, failed_recs, first_in_flight), epoch)  # no load: one entry per attempt
            if rec.outcome["model_mismatch_turns"]:
                say(f"warning: [{name}] {q['id']} rep {rep}: {rec.outcome['model_mismatch_turns']} turn(s) answered "
                    f"by {', '.join(rec.outcome['served_models'])}, not the requested {rec.model} "
                    f"(possible silent fallback; marked in the record)")
            line = index_line(rec, name)
            with index_lock, index_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(line, ensure_ascii=False) + "\n")
                fh.flush()
            bump("run")  # the claim stays: it marks the attempt as taken for any runner, on any node
            if parent is None:  # an initial cell: no checker in the loop
                f1 = "-" if line["f1"] is None else f"{line['f1']:.2f}"
                say(f"[{name}] {q['id']} rep {rep}: {line['status']} exact={line['exact']} f1={f1} "
                    f"turns={line['turns']} {line['wall_s']:.0f}s")
            else:  # a branch: its checker decision is verified only after the phase, and a divergent one is set
                # aside before its score is seen (design ruling, design log, 20 Oct 07:26), so no score here
                say(f"[{name}] {q['id']} rep {rep}: {line['status']} turns={line['turns']} {line['wall_s']:.0f}s")
        except Exception as exc:  # one attempt's crash must not stop the others; its claim is kept
            bump("errors")
            say(f"error: [{name}] {q['id']} rep {rep}: {type(exc).__name__}: {exc} "
                f"(claim kept; it goes stale after --claim-timeout-s, then --reclaim retries it)")
            say("".join(traceback.format_exception_only(type(exc), exc)).strip())
        finally:
            release(claim, delete=False)  # no longer refreshed; a recorded attempt keeps its claim for good
            for made in clients:
                gated.close_client(made)

    # One pending list in the seeded order; each model's attempts start in that order, each when its
    # model has a free slot, so one model waiting (warm-up, a step-down) never holds back another.
    by_id = {q["id"]: q for q in questions}
    pending: dict[str, list[tuple[int, str, dict, int, Path | None]]] = {name: [] for name in model_names}
    for position, (name, arm, qid, rep) in enumerate(order):
        key = key_of(name, arm, qid, rep)
        if key in done:
            bump("skipped")
            continue
        if key in awaiting:
            bump("awaiting_fix")
            continue
        if key in claimed:
            bump("stale_claims")
            continue
        parent_path = None
        if plan is not None and plan.is_branch(arm):  # its parent record of the same output, in this OUT
            parent = briefs.get(key_of(name, plan.cell(name, arm)["parent"], qid, rep))
            state = branch_parent_state({"outcome": parent.outcome} if parent is not None else None)
            if state == "inherited":  # nothing to check: the initial answer stands, no record
                bump("inherited")
                continue
            if state != "ready":  # require_initial_complete refused every such parent before anything started
                raise Refused(f"{name} {arm} {qid} rep {rep}: its parent record is {state}")
            parent_path = parent.path
        pending[name].append((position, arm, by_id[qid], rep, parent_path))
    pools = {name: ThreadPoolExecutor(max_workers=configured[name], thread_name_prefix=name) for name in model_names}

    gate = StartGate(model_names)

    def run_slot(name: str, ticket: int, item: tuple[int, str, dict, int, Path | None]) -> None:
        try:
            attempt(name, ticket, *item)
        finally:
            throttle.release(name)

    def schedule(name: str) -> None:
        if models[name].key_envs == GATEWAY_KEY_ENVS:
            say(f"{name}: route model {route_models[name]}, key from {gateway_key_env()}")
        say(f"{name}: {len(pending[name])} attempts pending, up to {allowed[name]} in flight "
            f"({configured[name]} configured)")
        items = list(pending[name])
        tickets = iter(range(len(items)))  # the model's start sequence: the order
        warm = args.warmup if args.warmup is not None else routes[name]["provider"] == "anthropic"
        # Warm-up: each initial arm's first pending attempt (each arm has its own prompt prefix), at its own place
        # in the order, run to completion before the model's next attempt starts, so every later attempt of the
        # arm reads the prefix it wrote. Attempts started before it go on (theirs is another arm's prefix); no
        # attempt moves, so the starts keep the seeded order. Never a branch cell (item[4]: its parent record).
        firsts = {}
        if warm:
            for item in items:
                if item[4] is None:
                    firsts.setdefault(item[1], item[0])
        warmups = set(firsts.values())  # positions in the order
        for item in items:
            throttle.acquire(name)  # waits for a slot under the model's current allowed size
            ticket = next(tickets)
            if stops[name].is_set() or halt.is_set() or held_now(name):
                throttle.release(name)
                gate.skip(name, ticket)  # the rest are never handed out; nothing waits on them
                break
            if item[0] in warmups:
                say(f"{name}: warm-up: {item[2]['id']} rep {item[3]} ({item[1]}) runs to completion before the "
                    f"model's next attempt starts, so its prompt cache is written once")
                run_slot(name, ticket, item)
            else:
                pools[name].submit(run_slot, name, ticket, item)

    schedulers = [threading.Thread(target=schedule, args=(name,), name=f"schedule-{name}") for name in model_names]
    beat = threading.Thread(target=heartbeat, name="claim-heartbeat", daemon=True)
    beat.start()
    checking = None
    if periodic is not None:
        say(f"periodic check: phase {plan.phase}'s checker decisions recomputed in a fresh interpreter every "
            f"{args.periodic_check_s:g} s while it runs")
        checking = threading.Thread(target=periodic.run, name="periodic-check", daemon=True)
        checking.start()
    try:
        for thread in schedulers:
            thread.start()
        for thread in schedulers:
            thread.join()
    finally:
        for pool in pools.values():
            pool.shutdown(wait=True)
        if calm is not None:  # no attempt of this runner is left to add a turn: every bin is final
            with events_lock:
                calm_check(float("inf"))
        heartbeat_stop.set()
        beat.join()
        if checking is not None:  # the attempts ended: a check still running is stopped (the final check follows)
            periodic.finish()
            checking.join()
    if halt.is_set() and gate_rule is not None and gate_rule.error is not None:
        say("error: no attempt was launched after a log of the gate's step-down rule could not be written; the "
            "attempts in flight finished and are recorded")
    elif halt.is_set():
        say(f"error: phase {plan.phase} HALTED by a periodic check while it ran: no attempt was launched after it, "
            f"the attempts in flight finished and are recorded (manifest periodic_checks)")

    # the standing gate: every record loads, writes back and replays, and every online checker decision is decided
    # again offline from its recorded store responses (F3; a decision that diverges holds the batch, one that cannot
    # be recomputed fails). Once the OUT has branch records, in a fresh interpreter over the grid's sources: this
    # process cannot vouch for its own decisions (in the rehearsal its rdflib grammar broke while the phase ran, F2).
    decisions = None
    if plan is not None and checker_arms:
        decisions = decisions_in_fresh_process(args.out, plan, say)
        check = verify_batch(args.out, decisions=decisions or {})  # none: no decision counts as recomputed
    else:
        from harness.checkloop.recompute import DecisionRecompute

        check = verify_batch(args.out, recompute=DecisionRecompute.from_arms(checker_arms))
    write_check(args.out, check)
    held = held_message(args.out, check)
    if held:
        say(f"error: {held}; the batch cannot enter analysis, and the branch phase does not start on it")
    halted = halt_message(args.out, check)
    if halted:
        say(f"error: {halted}")
    if check["failures"]:
        say(f"error: replay check failed for {len(check['failures'])} of {check['n_records']} records ({CHECK_FILE}); "
            f"the batch cannot enter analysis: " + "; ".join(f"{f['file']} ({f['stage']}): {f['error'][:200]}"
                                                              for f in check["failures"][:5]))
    else:
        tell(f"replay check: {check['n_ok']} of {check['n_records']} records load, write back byte for byte and "
             f"replay, {check['decisions_recomputed']} online checker decisions recomputed to themselves ({CHECK_FILE})")

    branch_counts = (f", {counts['inherited']} inherited (parent did not submit)"
                     if plan is not None and plan.phase == "branch" else "")
    awaiting_fix = (f"{counts['awaiting_fix']} awaiting a checker fix (set aside as checker divergences: re-run once "
                    f"a fix is recorded, under it), " if counts["awaiting_fix"] else "")
    tell(f"done: {counts['run']} run, {counts['skipped']} skipped (already recorded){branch_counts}, "
        f"{counts['aborted']} aborted, {counts['errors']} errors, {counts['stale_claims']} skipped with a stale "
        f"claim{' (rerun with --reclaim to retry them)' if counts['stale_claims'] else ''}, {awaiting_fix}"
        f"{counts['claimed_elsewhere']} claimed by another runner; store outages: {counts['infrastructure']} "
        f"failed tries, {counts['infra_reruns']} re-runs, {counts['infra_final']} attempts failed on every try")
    if plan is not None:  # per model and cell: the cells are separate experiments
        # a branch cell's scores only over the records whose decision the fresh-process recompute verified, and none
        # of a held or halted batch (item 1 of the release review): a divergent decision is set aside unseen
        verified = verified_run_ids(decisions)
        withheld = bool(held or halted or decisions is None)
        by_model = summarise(index_path, args.models, by_arm=True,
                             keep=lambda row: not plan.is_branch(row.get("arm")) or row["run_id"] in verified)
        for name in model_names:
            for arm in arms:
                if plan.is_branch(arm) and withheld:
                    say(f"{name} {arm}: score summary withheld: the branch decisions are not all verified (the batch "
                        f"is held or halted, or their recompute did not finish)")
                elif arm in by_model.get(name, {}):
                    say(f"{name} {arm}: {json.dumps(by_model[name][arm], sort_keys=True)}")
    else:
        for name, s in summarise(index_path, args.models).items():
            say(f"{name}: {json.dumps(s, sort_keys=True)}")
    for line in setaside_lines(check):  # per arm: the set-aside records and how many of their attempts were re-run
        say(line)
    for name in sorted(held_models):
        say(f"{name}: held by the infrastructure-fault monitor: its attempts not yet started wait for its release "
            f"(its probe passing, scripts/infra_fault_monitor.py), then a start of the phase runs them")
    if query_gate is not None:  # the share of this start's wall time the gate was held or queued for by this runner
        busy = {"at": utc_now(), "runner_started_at": runner_started_at, "gate_wait_bound_s": args.gate_wait_s,
                **query_gate.stats.busy_shares(gate_since, time.time())}
        if gate_rule is not None:
            busy["gate_slots"] = {"slots": query_gate.slots, "allowed_start": gate_start["allowed"],
                                  "allowed_end": query_gate.level()[0]}
        gated.append_private(args.out / "gate-busy.jsonl", busy)
        share = {k: f"{busy[k]:.1%}" if busy[k] is not None else "n/a" for k in ("busy_share", "held_share",
                                                                                 "queued_share")}
        tell(f"gate: busy {share['busy_share']} of this runner's {busy['wall_s']:.0f} s (held {share['held_share']} "
             f"over {busy['queries_held']} queries, queued for {share['queued_share']}); other processes' holds are "
             f"seen only as this runner's queue ({GATE_BUSY_FILE})")
    if counts["aborted"] or counts["models_stopped_infrastructure"] or counts["model_changed"] or counts["models_held"]:
        return 3
    if check["failures"] or halted or halt.is_set():
        return 5
    if counts["errors"] or counts["harness_defects"]:
        return 1
    if counts["awaiting_fix"]:  # not complete: those attempts run only once their fix is recorded
        return 6
    return 4 if counts["infra_final"] else 0


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
