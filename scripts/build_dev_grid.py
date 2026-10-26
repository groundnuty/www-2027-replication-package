#!/usr/bin/env python3
"""The development-split full grid (E1.5a, the design log): expand the grid spec into the
run manifest, every attempt (graph, model, cell, question, repetition) in its seeded order.

    uv run python scripts/build_dev_grid.py --spec data/runs/dev-grid/grid.json --out OUT
    uv run python scripts/build_dev_grid.py --spec data/runs/dev-grid/grid.json --out OUT --for-launch [--phase P]

Writes ``OUT/manifest.json`` (the spec, its sha256, the inputs still unfilled, the counts per cell,
each phase with its seed, cells and order sha256 per graph, the offline replay jobs) and
``OUT/attempts.jsonl.gz`` (one line per attempt, in order). ``scripts/check_dev_grid_coverage.py``
reads both.

Models and roles. Every model has a role, and a role names the cells its models run: the must
models (claude-sonnet-5-5, qwen3.8-27b) every cell; the model panel (glm-5.3-flash, gemma-4-31b,
claude-haiku-5-5, qwen3-coder-30b-a3b and deepseek-v4-flash-0731, the seventh of P once #21 decided it;
design log, change-log entry 82) the cohorts none, curated and census, and on none fb:curated and gen:any; the exploratory
models (llama-3.3-70b, qwen3.6-35b-a3b) the none cohort only, never a prompted or branch cell and
never in P (``exploratory`` may run natural cohorts only, with no branch cell). Every attempt line,
replay job and coverage row names its model's role, so the analysis never pools the roles. Same
frozen settings, phases, seeds and split refusal for every role. ``runner_settings.concurrency``
gives each model's workers (set from the load test, design log; not part of a run's fingerprint).
No agent setting is grid-wide: each model's turn, wall, output and tool-result caps (design log,
entry 90; the tool-result cap per graph) and its transport (route settings: streaming and its
options, stream guards, parallel tool calls) come from the runner's model table, and the manifest
records them per model and graph (``model_settings``), never as a grid-wide setting; every record
carries its own, and a retry gets its attempt's turn and wall caps (the spec fixes only its nudges).
Per model, too (design log, change-log entry 87): a model may list its own cohorts (``models[].cohorts``,
a subset of its role's that keeps every parent of its role's branch cells; qwen3.8-27b's dev prompted
phase runs curated and census only) and its repetitions per phase (``models[].repetitions``, a
branch never more than its parent's phase); the manifest records both (``model_cells``,
``model_repetitions``). A test grid runs 3 repetitions in every phase for every model of P, fixed.
``runner_settings.gateway_budget`` caps the workers of a phase's Gateway models together (by the route's
provider in the runner's table; Anthropic routes are outside it): ``--for-launch`` refuses a phase over
it, and the manifest records each phase's planned Gateway load (``gateway_load``).
``runner_settings.request_timeout_s`` and ``runner_settings.max_turns`` (change-log entry 180; optional) give a model
of the grid its own request time-out in seconds (an OpenAI-compatible route only) and its own turn cap, recorded
settings of the stack the spec runs on: the model is built with them here and by the runner alike
(``run_pilot.grid_models``), so its ``model_settings`` (route settings and caps) record them. Without them every model
keeps its table's (a 600 s request time-out; its own turn cap).
A model may be added to the spec later (as the DeepSeek panel model was): every other model's attempts keep
their relative order in every phase, since a key depends only on the seed and the attempt, and their
cells; the runner admits it while it has no record in the run.

Cells. A graph has two kinds of cell (pre-registration section 3: every checker arm of a cohort
starts from the same initial outputs):

* initial cells, the prompt cohorts (none, curated, census, generated draw 1): each attempt is a
  full agent run; each cohort names its phase (none: ``natural``; the others: ``prompted``);
* branch cells, the checker arms on the natural cohort, the generic control and the deployment
  row: each attempt branches from its parent cohort's record of the same (model, question,
  repetition) (``harness.checkloop.online.branch_attempt``) and calls the model only when its arm
  flags (one retry). Each branch attempt names its parent attempt, its mode and the sources whose
  checks run in it, in the manifest.

The generic control is one cell, ``gen:any`` (design ruling in the design log). Its arm's sources are not
listed in the spec but follow from the role (trigger ``role_feedback_arms``): every source of the role's
own feedback arms on the same cohort, so curated, census-t5 and generated draws 1-3 for the must models,
and curated alone for the panel, whose only checker arm is curated. Under checker-loop/2 (spec trace
B10, design log entry 91; harness/checkloop/online.py) it decides on the curated source alone, whatever
else its arm holds: it retries, with the generic message, every natural-cohort output the curated
source flags, once, and its record keeps the curated source's verdict
(``checker.initial.verdict.per_source``). Each branch cell names the sources whose verdicts decide it
(``verdict_sources``): a feedback cell its whole arm, a generic cell its role's curated sources (a source
keyed ``curated``, or ``curated+...``; the launcher refuses a manifest whose list is not the arm's
sources of kind curated), and the coverage check expects exactly those verdicts (dress rehearsal M7). The deployment row checks the census
cohort with curated and with each generated draw.

The offline replay (every source of the graph, including the replay-only census thresholds 2 and
20, the union and every detection-switch variant, on every cohort's initial outputs) is a separate
job list with no model calls (``replay_jobs``), each job with the rows its matrix must hold;
``check_dev_grid_coverage.py --replay`` checks them.

Development split only. The question set is loaded whole and every requested question must be
in it with split ``dev``; a test id in the spec's ``question_ids`` is refused, as is a development
question whose base question is also a test question. Without ``question_ids`` every development
question of the file runs. Any other split (the test grid to come) is gated by the freeze:
``split_gate`` requires ``freeze.prereg_repo`` and ``freeze.freeze_commit``, a commit whose
paper/preregistration.md holds the exact whole line ``Status: frozen`` and that is on the science
repository's main line as it is now: origin's main is fetched first (a failed fetch refuses, since a
local ref may be stale) and the commit must be it or one of its ancestors (``frozen_preregistration``,
kept in step with scripts/run_probes.py on feat/closed-book), checked before any question is loaded
and by ``assert_launchable``; the manifest records the commit, the document's sha256, and the fetched
origin/main commit and its fetch time.

Inputs. Every input is named by an exact hash (``FILL`` until it is final): each graph's question
set, store description, prompts manifest, ontology blocks, shared hierarchy and source files, and
the code (compiler and harness commits, check contract, feedback format, generic text). A filled
hash that does not match its file, or a malformed one, is always refused. A preview build (no
``--for-launch``) tolerates placeholders and files not built yet, and writes ``launchable`` per
phase with them listed (``placeholders_by_phase``, ``missing_by_phase``). ``--for-launch --phase P``
refuses while an input of phase P is left (exit 2); without ``--phase``, while any is. The inputs of
a phase (``phase_placeholders``): natural, the question files, the store descriptions (each holds its
store's endpoint and identity: one store per graph for every phase), the agent and runner settings
and its models' concurrency; prompted, those, the prompts manifest, its cohorts' blocks and sources,
and its models' concurrency; branch, everything (the code hashes, the shared hierarchy, every source,
the replay-only ones included). So natural can launch while the prompts are FILL. The gate is per graph
too (dress rehearsal M6): a phase on a graph (``--for-launch --phase P --graph G``, as the launcher runs it)
needs that graph's inputs of the phase and the inputs of no graph (the code, the settings), never another
graph's, so a FILL in Rhea's prompts does not hold UniProt's prompted phase back; the manifest records
``launchable_by_graph``, ``placeholders_by_graph`` and ``missing_by_graph`` (``launchable`` and its
companions are every graph's together). The launcher runs that check first (``assert_launchable``) and
compares the manifest's ``spec_sha256`` with the spec.

Order: three registered phases in one output directory per graph (design ruling in the design log; the
test runs use the same structure). ``natural`` is one seeded order over the none cohort and every
model; ``prompted`` one seeded order over the prompted cohorts (curated, census, generated-1),
interleaved among themselves; ``branch``, once both are complete and pass their replay check, one
seeded order over every branch cell (fb and gen cells branch from none, the deployment row from
census). A branch attempt cannot share an order with its own parent, which may come after it. Each
phase has its own seed in the spec (``order.phases.<phase>.seed``), recorded with the phase in the
manifest. Within a graph a phase's order is the runner's scheme (``run_pilot.interleaved_order``:
sorted by ``order_key``, the sha256 of the seed and [model, cell, question id, repetition]); with
one seed for every phase, as in grid.json, the orders are the full plan's order cut by phase, since a
key depends only on the seed and the attempt. The graphs are separate stores and separate runs.

Launch design (scripts/run_pilot.py ``--grid-manifest``). One run_pilot invocation per cell is ruled
out: the cells would run one after another, so server load would correlate with the arm
(the design log). The launcher runs run_pilot once per graph and phase, every model and cell of
that phase in one seeded order, all into one OUT per graph (the branch records sit beside their
parents). The runner change that needs:

1. Attempts from the manifest: an option (``--grid-manifest M --graph G --phase P``) that takes the
   phase's attempts from ``attempts.jsonl.gz`` (or recomputes them: per role, its models crossed
   with its cells, merged by ``order_key``) instead of crossing every model with every arm, which
   would give the panel the must cells; the phase's seed is the order seed, and the runner's order
   sha256 must equal the manifest's (the phase's ``graphs.<graph>.order_sha256``). Each record's
   ``load`` carries the cell id as ``arm`` and the phase's seed as ``order_seed``; the coverage
   check requires both.
2. The arm in the attempt key: ``attempt_key``, ``existing_keys`` (reading ``load.arm``),
   ``claim_path`` and the claims, the index line and the summary take (arm, question, model,
   repetition); today two arms in one OUT collide on (question, model, repetition).
3. An ``AgentConfig`` per initial cell (condition and ``ontology_block_sha256`` from the manifest),
   with the frozen settings (60 turns, 1,800 s, 16k output tokens per turn).
4. A fingerprint per arm in ``manifest.json``: the prompt block (condition, id, template, block,
   nudge and tools hashes) per initial cell, and ``OnlineChecker.settings()`` per branch cell (arm
   identity with source and hierarchy hashes, mode, switches, retry budget, feedback format and
   generic text hashes), per phase, with the phase's inputs and each model's order. A start refuses a
   change to any phase already recorded; a phase's arms are fixed once it has a record, and a model
   joins only while it has none.
5. Branch cells: ``run_attempt_claimed`` calls ``branch_attempt(parent, OnlineChecker(arm, store,
   mode, budget), route, sparql, gold_rows, rules)`` with the parent record of (parent cell,
   question, model, repetition) from the same OUT, instead of ``run_attempt``; the arm is
   ``Arm(cell arm, the compilers of the cell's sources for the model's role)`` (``gen:any``: the five
   checker sources for a must model, curated for a panel model; mode generic). A
   parent that did not submit has nothing to check: no branch record is written and the coverage
   check reports it as inherited. The branch phase does not start until every natural and prompted
   attempt has a counted record and every recorded branch's parent is the record now in the run.
6. The retry budget passed explicitly (``RetryBudget(max_turns=60, max_wall_s=1800)``): its default
   is 30 turns and 900 s, and pre-registration section 3 gives the retry the initial caps.
7. Warm-up per initial cell (each cohort has its own prompt prefix), as today per arm; none for
   branch cells, whose retry continues the parent's transcript.

Exit status: 0 manifest written; 2 refused (malformed spec, a hash that does not match its
file, a test-split or unknown question, a split other than dev without a frozen pre-registration,
or, with ``--for-launch``, any placeholder or missing input of the phase given, or of any phase).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _load_run_pilot():
    """The runner, whose order scheme and models are reused, never copied: the module already
    loaded under that name (the tests load it so), else scripts/run_pilot.py."""
    if "run_pilot" in sys.modules:
        return sys.modules["run_pilot"]
    spec = importlib.util.spec_from_file_location("run_pilot", Path(__file__).resolve().parent / "run_pilot.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_pilot"] = module
    spec.loader.exec_module(module)
    return module


run_pilot = _load_run_pilot()
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPEC = ROOT / "data" / "runs" / "dev-grid" / "grid.json"
SPEC_SCHEMA = "dev-grid-spec/2"
MANIFEST_SCHEMA = "dev-grid-manifest/2"
ATTEMPTS_FILE = "attempts.jsonl.gz"
PLACEHOLDER = "FILL"
SPLIT = "dev"
SPLITS = {"dev": "development", "test": "test"}
MODES = ("feedback", "generic")
ROLES = ("must", "panel", "exploratory")
EXPLORATORY = "exploratory"  # natural cohorts only, no branch cell, never in P
GENERIC_TRIGGER = "role_feedback_arms"  # a generic cell's arm: the sources of its role's feedback arms
GENERIC_DECIDES = "curated"  # ... of which it decides on the curated ones alone (checker-loop/2, spec trace B10)
# The freeze gate, kept in step with scripts/run_probes.py on feat/closed-book (frozen_preregistration, with the
# origin/main checks of the reviews of 81f7af2 and c61ca83) until the two share a module.
PREREG_PATH = "paper/preregistration.md"
# The exact whole line, case-sensitive (design ruling in the design log: "Status: draft (not frozen)" must not open it).
_FROZEN_STATUS = re.compile(r"^Status: frozen$", re.MULTILINE)
_COMMITISH = re.compile(r"[0-9a-fA-F]{7,40}")
ORIGIN_MAIN = "refs/remotes/origin/main"
FETCH_TIMEOUT_S = 120.0
PHASES = ("natural", "prompted", "branch")
INITIAL_PHASES = ("natural", "prompted")  # the cohorts' phases; branch is the checker arms'
FROZEN_AGENT_KEYS: tuple[str, ...] = ()  # nothing grid-wide: every cap is the model's own (design log, entry 90)
MODEL_CAPS = ("max_turns", "max_wall_s", "max_tokens", "max_result_chars")  # the runner's table (run_pilot.model_caps)
TEST_REPETITIONS = 3  # a test grid's every phase and every model of P (the calibration assumes 210 x 3)
P_ROLES = ("must", "panel")
GATEWAY_PROVIDER = "openai-compatible"  # the HPCNet LLM Gateway routes, which share gateway_budget
_NATURAL_GRAPH_INPUTS = ("questions", "store_json", "question_ids")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


class Refused(Exception):
    """The spec cannot be built (or launched); the message lists every reason."""


# --- the spec -------------------------------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def placeholders(obj: Any, prefix: str = "") -> list[str]:
    """The dotted paths of every value still ``FILL``."""
    if isinstance(obj, dict):
        return [p for k, v in obj.items() for p in placeholders(v, f"{prefix}.{k}" if prefix else str(k))]
    if isinstance(obj, list):
        return [p for i, v in enumerate(obj) for p in placeholders(v, f"{prefix}[{i}]")]
    return [prefix] if obj == PLACEHOLDER else []


def _model_cohorts(spec: dict, m: dict) -> list[str]:
    """The cohorts a model runs: its own list (design log, entry 87: Qwen3.8's dev prompted phase drops
    generated-1), else its role's."""
    role_cohorts = ((spec.get("roles") or {}).get(m.get("role")) or {}).get("cohorts") or []
    return list(m["cohorts"]) if isinstance(m.get("cohorts"), list) else list(role_cohorts)


def model_cell_ids(spec: dict, m: dict) -> list[str]:
    """The cells a model runs: its cohorts, then its role's branch cells whose parent it runs."""
    mine = _model_cohorts(spec, m)
    rs = (spec.get("roles") or {}).get(m.get("role")) or {}
    branch = [b["id"] for b in spec.get("branch_cells") or []
              if b.get("id") in (rs.get("branch_cells") or []) and b.get("parent") in mine]
    return [c for c in spec.get("cohorts") or {} if c in mine] + branch


def model_repetitions(spec: dict, m: dict) -> dict[str, int]:
    """A model's repetitions per phase: its own (``repetitions`` per phase; a ruling may cut Qwen3.8's dev
    prompted and branch phases to 2), else the spec's."""
    own = m.get("repetitions") if isinstance(m.get("repetitions"), dict) else {}
    return {ph: own.get(ph, spec.get("repetitions")) for ph in PHASES}


def _model_cell_problems(spec: dict) -> list[str]:
    problems = []
    cohorts = spec.get("cohorts") or {}
    branch = {b.get("id"): b for b in spec.get("branch_cells") or []}
    for m in spec.get("models") or []:
        name, rs = m.get("name"), (spec.get("roles") or {}).get(m.get("role")) or {}
        if "cohorts" in m:
            if not isinstance(m["cohorts"], list) or not m["cohorts"]:
                problems.append(f"models.{name}.cohorts: a non-empty list of its role's cohorts")
                continue
            problems += [f"models.{name}.cohorts: {c!r} is not a cohort of its role" for c in m["cohorts"]
                         if c not in (rs.get("cohorts") or [])]
            for cid in rs.get("branch_cells") or []:
                parent = (branch.get(cid) or {}).get("parent")
                if parent is not None and parent not in m["cohorts"]:
                    problems.append(f"models.{name}.cohorts: drops {parent!r}, the parent of its role's branch cell "
                                    f"{cid!r}")
        if "repetitions" in m:
            own = m["repetitions"]
            if not isinstance(own, dict) or set(own) - set(PHASES) or not all(
                    isinstance(v, int) and not isinstance(v, bool) and v >= 1 for v in own.values()):
                problems.append(f"models.{name}.repetitions: positive integers per phase ({', '.join(PHASES)})")
                continue
        reps = model_repetitions(spec, m)
        if not all(isinstance(v, int) for v in reps.values()):
            continue
        cells = model_cell_ids(spec, m)
        for cid in cells:
            parent = (branch.get(cid) or {}).get("parent")
            if parent is None or parent not in cohorts:
                continue
            ph = cohorts[parent].get("phase")
            if ph in reps and reps["branch"] > reps[ph]:
                problems.append(f"models.{name}.repetitions: branch {reps['branch']} > {ph} {reps[ph]}, the phase of "
                                f"its branch cells' parent {parent!r}")
                break
        if spec.get("split") == "test" and m.get("role") in P_ROLES and set(reps.values()) != {TEST_REPETITIONS}:
            problems.append(f"models.{name}: a test grid runs {TEST_REPETITIONS} repetitions in every phase for every "
                            f"model of P (the calibration assumes 210 x {TEST_REPETITIONS}), not {reps}")
    return problems


def phase_models(spec: dict, phase: str) -> list[str]:
    """The models that run a cell of ``phase``, in spec order."""
    cohorts = spec.get("cohorts") or {}

    def runs(m: dict) -> bool:
        ids = model_cell_ids(spec, m)
        if phase == "branch":
            return any(c not in cohorts for c in ids)
        return any((cohorts.get(c) or {}).get("phase") == phase for c in ids)
    return [m["name"] for m in spec.get("models") or [] if runs(m)]


def gateway_workers(spec: dict, phase: str) -> dict[str, Any]:
    """The workers of the phase's Gateway models (by their route's provider in the runner's table): together they
    must stay within ``runner_settings.gateway_budget`` (Anthropic routes are outside it)."""
    concurrency = (spec.get("runner_settings") or {}).get("concurrency") or {}
    out = {}
    for name in phase_models(spec, phase):
        spec_entry = run_pilot.MODELS.get(name)
        if spec_entry is not None and spec_entry.factory().route_info().get("provider") == GATEWAY_PROVIDER:
            out[name] = concurrency.get(name)
    return dict(sorted(out.items()))


def _gateway_total(spec: dict, phase: str) -> int | None:
    workers = gateway_workers(spec, phase)
    return sum(workers.values()) if all(isinstance(n, int) for n in workers.values()) else None


def gateway_problems(spec: dict, phase: str) -> list[str]:
    budget = (spec.get("runner_settings") or {}).get("gateway_budget")
    workers = gateway_workers(spec, phase)
    if not isinstance(budget, int) or not all(isinstance(n, int) for n in workers.values()):
        return []  # a FILL is refused as a placeholder
    total = sum(workers.values())
    if total <= budget:
        return []
    return [f"Gateway models of phase {phase}: {total} workers ("
            + ", ".join(f"{m} {n}" for m, n in workers.items()) + f") > gateway_budget {budget}"]


def _of_graph(path: str, graph: str | None) -> bool:
    """Whether the input at dotted ``path`` gates a launch on ``graph`` (every graph, when None): an input of no graph
    (the code, the settings, the models) gates every graph; a graph's own input only that graph (M6)."""
    parts = path.split(".")
    return graph is None or parts[0] != "graphs" or len(parts) < 2 or parts[1] == graph


def _in_phase(spec: dict, path: str, phase: str) -> bool:
    """Whether the input at dotted ``path`` is one a launch of ``phase`` needs: natural, the question files, the
    store descriptions, the agent and runner settings; prompted, those and its prompts and cohorts' sources;
    branch, everything. A model's concurrency counts only for the phases its role runs."""
    parts = path.split(".")
    if parts[:2] == ["runner_settings", "concurrency"]:
        return ".".join(parts[2:]) in phase_models(spec, phase)
    if phase == "branch":
        return True
    if parts[0] in ("runner_settings", "agent_settings", "models", "repetitions"):
        return True
    if parts[0] != "graphs" or len(parts) < 3:
        return False
    if parts[2] in _NATURAL_GRAPH_INPUTS:
        return True
    if phase != "prompted":
        return False
    prompted = [c for c, cs in (spec.get("cohorts") or {}).items() if cs.get("phase") == "prompted"]
    if parts[2] == "prompts_manifest":
        return True
    if parts[2] == "prompt_blocks":
        return len(parts) > 3 and parts[3] in prompted
    if parts[2] == "sources":
        sources = {(spec["cohorts"][c] or {}).get("source") for c in prompted}
        return len(parts) > 3 and parts[3] in sources
    return False


def phase_placeholders(spec: dict, phase: str, graph: str | None = None) -> list[str]:
    """The placeholders a launch of ``phase`` on ``graph`` (on every graph, when None) refuses."""
    return [p for p in placeholders(spec) if _in_phase(spec, p, phase) and _of_graph(p, graph)]


def _hash_fields(spec: dict) -> list[tuple[str, Any, re.Pattern]]:
    """Every hash or commit field: (dotted path, value, the form a filled value must have)."""
    out = []
    for key, value in (spec.get("code") or {}).items():
        out.append((f"code.{key}", value, _COMMIT if key.endswith("_commit") else _SHA256))
    for g, gs in (spec.get("graphs") or {}).items():
        for ref_path, ref in _graph_refs(g, gs):
            out.append((f"{ref_path}.sha256", ref.get("sha256"), _SHA256))
    return out


def _graph_refs(graph: str, gs: dict) -> list[tuple[str, dict]]:
    """A graph's inputs as (dotted path, {path?, sha256}); an ontology block has no path (its file
    is named by its hash in the block store)."""
    base = f"graphs.{graph}"
    refs = [(f"{base}.{k}", gs[k]) for k in ("questions", "store_json", "prompts_manifest", "shared_hierarchy")
            if isinstance(gs.get(k), dict)]
    refs += [(f"{base}.sources.{name}", ref) for name, ref in (gs.get("sources") or {}).items()]
    refs += [(f"{base}.prompt_blocks.{name}", ref) for name, ref in (gs.get("prompt_blocks") or {}).items()]
    return refs


def validate_spec(spec: dict) -> list[str]:
    """Structural problems; a spec with any is never built."""
    problems = []
    if spec.get("schema") != SPEC_SCHEMA:
        problems.append(f"schema {spec.get('schema')!r}, expected {SPEC_SCHEMA!r}")
    split = spec.get("split")
    if split not in SPLITS:
        problems.append(f"split {split!r}: one of {sorted(SPLITS)}")
    elif split != SPLIT:
        freeze = spec.get("freeze") or {}
        if not isinstance(freeze.get("prereg_repo"), str) or not isinstance(freeze.get("freeze_commit"), str):
            problems.append(f"split {split!r} needs freeze.prereg_repo and freeze.freeze_commit: a frozen "
                            f"pre-registration gates every split but the development split")
    reps = spec.get("repetitions")
    if not isinstance(reps, int) or isinstance(reps, bool) or reps < 1:
        problems.append(f"repetitions {reps!r}: a positive integer")
    order = spec.get("order") or {}
    for phase in PHASES:
        seed = ((order.get("phases") or {}).get(phase) or {}).get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            problems.append(f"order.phases.{phase}.seed: an integer")
    if order.get("rule") != run_pilot.ORDER_RULE:
        problems.append(f"order.rule {order.get('rule')!r} is not the runner's ({run_pilot.ORDER_RULE!r})")
    for block in ("agent_settings", "retry_budget", "switches"):
        if not isinstance(spec.get(block), dict):
            problems.append(f"{block}: missing")
    missing = [k for k in FROZEN_AGENT_KEYS if k not in (spec.get("agent_settings") or {})]
    if missing:
        problems.append(f"agent_settings: missing {missing}")
    problems += [f"agent_settings.{k}: a model's own cap (the runner's model table), not a grid-wide setting"
                 for k in MODEL_CAPS if k in (spec.get("agent_settings") or {})]
    names = [m.get("name") for m in spec.get("models") or []]
    if not names:
        problems.append("models: none")
    if len(set(names)) != len(names):
        problems.append(f"models: duplicate names {names}")
    concurrency = (spec.get("runner_settings") or {}).get("concurrency")
    if not isinstance(concurrency, dict):
        problems.append("runner_settings.concurrency: a worker count (or FILL) per model")
        concurrency = {}
    for name in names:
        if name not in concurrency:
            problems.append(f"runner_settings.concurrency: no entry for {name}")
    for name, n in concurrency.items():
        if name not in names:
            problems.append(f"runner_settings.concurrency.{name}: not a model of the grid")
        elif n != PLACEHOLDER and (not isinstance(n, int) or isinstance(n, bool) or n < 1):
            problems.append(f"runner_settings.concurrency.{name}: {n!r} is neither FILL nor a positive integer")
    budget = (spec.get("runner_settings") or {}).get("gateway_budget")
    if budget != PLACEHOLDER and (not isinstance(budget, int) or isinstance(budget, bool) or budget < 1):
        problems.append(f"runner_settings.gateway_budget: {budget!r} is neither FILL nor a positive integer (the Gateway "
                        f"models' workers of a phase together)")
    problems += _model_cell_problems(spec)
    problems += run_pilot.runner_model_problems(spec, run_pilot.MODELS)  # change-log entry 180
    for m in spec.get("models") or []:
        if m.get("name") not in run_pilot.MODELS:
            problems.append(f"model {m.get('name')!r} is not a runner model ({', '.join(run_pilot.MODELS)})")
        else:  # a model runs on every graph of the grid: it needs a tool-result cap for each
            for graph in spec.get("graphs") or {}:
                try:
                    run_pilot.model_caps(run_pilot.MODELS[m["name"]], graph)
                except KeyError as exc:
                    problems.append(f"model {m['name']!r}: {exc.args[0]} in the runner's table")
        if not m.get("route_model"):
            problems.append(f"model {m.get('name')!r}: no route_model")
        if m.get("role") not in (spec.get("roles") or {}):
            problems.append(f"model {m.get('name')!r}: role {m.get('role')!r} is not in roles")
    cohorts = spec.get("cohorts") or {}
    for c, cs in cohorts.items():
        if cs.get("phase") not in INITIAL_PHASES:
            problems.append(f"cohort {c!r}: phase {cs.get('phase')!r}, one of {INITIAL_PHASES}")
    branch = {b.get("id"): b for b in spec.get("branch_cells") or []}
    for role, rs in (spec.get("roles") or {}).items():
        if role == EXPLORATORY and (rs.get("branch_cells") or any(
                (cohorts.get(c) or {}).get("phase") != "natural" for c in rs.get("cohorts") or [])):
            problems.append(f"roles.{role}: an exploratory role runs the natural cohorts only, with no branch cell")
        if role not in ROLES:
            problems.append(f"roles.{role}: one of {ROLES}")
        mine = rs.get("cohorts") or []
        if not mine:
            problems.append(f"roles.{role}: no cohorts")
        problems += [f"roles.{role}: cohort {c!r} is not a cohort" for c in mine if c not in cohorts]
        for cid in rs.get("branch_cells") or []:
            b = branch.get(cid)
            if b is None:
                problems.append(f"roles.{role}: branch cell {cid!r} is not a branch cell")
            elif b.get("parent") not in mine:
                problems.append(f"roles.{role}: branch cell {cid!r} needs its parent cohort {b.get('parent')!r}")
            elif b.get("mode") == "generic" and not _role_feedback_sources(spec, role, b["parent"]):
                problems.append(f"roles.{role}: generic cell {cid!r} has no feedback arm on {b['parent']!r} to "
                                f"trigger it")
            elif b.get("mode") == "generic" and not any(map(is_curated, _role_feedback_sources(spec, role,
                                                                                               b["parent"]))):
                problems.append(f"roles.{role}: generic cell {cid!r} has no curated source among its feedback arms on "
                                f"{b['parent']!r} to decide it: under checker-loop/2 a generic cell decides on the "
                                f"curated source alone")
    if not cohorts:
        problems.append("cohorts: none")
    graphs = spec.get("graphs") or {}
    if not graphs:
        problems.append("graphs: none")
    ids = list(cohorts)
    for b in spec.get("branch_cells") or []:
        ids.append(b.get("id"))
        if b.get("parent") not in cohorts:
            problems.append(f"branch cell {b.get('id')!r}: parent {b.get('parent')!r} is not a cohort")
        if b.get("mode") not in MODES:
            problems.append(f"branch cell {b.get('id')!r}: mode {b.get('mode')!r}, one of {MODES}")
        if b.get("mode") == "generic":
            if b.get("trigger") != GENERIC_TRIGGER or "sources" in b or not b.get("arm"):
                problems.append(f"branch cell {b.get('id')!r}: a generic cell names its arm and "
                                f"trigger {GENERIC_TRIGGER!r}; its sources follow from each role's feedback arms")
        elif not b.get("arm") or not b.get("sources"):
            problems.append(f"branch cell {b.get('id')!r}: needs an arm name and its sources")
    if len(set(ids)) != len(ids) or not all(isinstance(i, str) and i for i in ids):
        problems.append(f"cell ids must be unique non-empty strings: {ids}")
    for g, gs in graphs.items():
        sources = gs.get("sources") or {}
        blocks = gs.get("prompt_blocks") or {}
        for c, cs in cohorts.items():
            if cs.get("source") is not None:
                if cs["source"] not in sources:
                    problems.append(f"graphs.{g}: cohort {c!r} needs source {cs['source']!r}")
                if c not in blocks:
                    problems.append(f"graphs.{g}: cohort {c!r} needs its prompt block")
        for b in spec.get("branch_cells") or []:
            for s in b.get("sources") or []:
                if s not in sources:
                    problems.append(f"graphs.{g}: branch cell {b.get('id')!r} needs source {s!r}")
        for name, members in ((spec.get("replay") or {}).get("unions") or {}).items():
            for s in members:
                if s not in sources:
                    problems.append(f"graphs.{g}: replay union {name!r} needs source {s!r}")
        if not isinstance(gs.get("questions"), dict) or not gs["questions"].get("path"):
            problems.append(f"graphs.{g}.questions: needs a path")
        qids = gs.get("question_ids")
        if qids is not None and (not isinstance(qids, list) or not qids):
            problems.append(f"graphs.{g}.question_ids: null (every development question) or a non-empty list")
        for ref_path, ref in _graph_refs(g, gs):
            if "prompt_blocks" not in ref_path and not ref.get("path"):
                problems.append(f"{ref_path}: needs a path")
    for path, value, form in _hash_fields(spec):
        if value != PLACEHOLDER and not (isinstance(value, str) and form.match(value)):
            kind = "a 40-hex commit" if form is _COMMIT else "a lowercase sha256"
            problems.append(f"{path}: {value!r} is neither {PLACEHOLDER} nor {kind}")
    return problems


def input_problems(spec: dict, root: Path, phase: str | None = None,
                   graph: str | None = None) -> dict[str, list[str]]:
    """Each input checked against its file: ``missing`` (no file at its path yet, its hash filled or still FILL:
    a FILL must not hide a wrong path, dress rehearsal M5; with ``phase``, only that phase's inputs; with ``graph``,
    only that graph's and the inputs of no graph) and ``mismatched`` (a filled hash that is not the file's sha256;
    always every input). A prompt block still FILL has no path to check (its file is named by its hash)."""
    missing, mismatched = [], []
    store = root / spec.get("block_store", "data/prompts/blocks")
    for g, gs in (spec.get("graphs") or {}).items():
        for ref_path, ref in _graph_refs(g, gs):
            declared = ref.get("sha256")
            if declared == PLACEHOLDER and ref.get("path"):
                path = root / ref["path"]
                if not path.is_file() and (phase is None or _in_phase(spec, f"{ref_path}.sha256", phase)) and \
                        _of_graph(ref_path, graph):
                    missing.append(f"{ref_path}: {path} does not exist (its hash is still {PLACEHOLDER})")
                continue
            if declared == PLACEHOLDER or not isinstance(declared, str) or not _SHA256.match(declared):
                continue
            path = root / ref["path"] if ref.get("path") else store / f"{declared}.txt"
            if not path.is_file():
                if (phase is None or _in_phase(spec, f"{ref_path}.sha256", phase)) and _of_graph(ref_path, graph):
                    missing.append(f"{ref_path}: {path} does not exist")
            elif (actual := sha256_file(path)) != declared:
                mismatched.append(f"{ref_path}: {path} has sha256 {actual}, the spec says {declared}")
    return {"missing": missing, "mismatched": mismatched}


def assert_launchable(spec: dict, root: Path, phase: str | None = None, graph: str | None = None) -> None:
    """The launcher's gate: refuses while an input of ``phase`` (of any phase, when None) on ``graph`` (on any graph,
    when None: an input of no graph gates every graph, a graph's own input only that graph) is a placeholder or
    missing, while any filled input is not the declared file, and, for any split but dev, unless the
    pre-registration is frozen (``split_gate``)."""
    if spec.get("split") in SPLITS and spec.get("split") != SPLIT:
        split_gate(spec, root)  # first: nothing of a test grid is looked at before the freeze is shown
    if phase is not None and phase not in PHASES:
        raise Refused(f"phase {phase!r}: one of {PHASES}")
    if graph is not None and graph not in (spec.get("graphs") or {}):
        raise Refused(f"graph {graph!r} is not a graph of the grid ({', '.join(spec.get('graphs') or {})})")
    found = input_problems(spec, root, phase, graph)
    unfilled = [p for p in placeholders(spec) if _of_graph(p, graph)] if phase is None else phase_placeholders(
        spec, phase, graph)
    problems = (validate_spec(spec) + [f"{p}: placeholder" for p in unfilled]
                + found["missing"] + found["mismatched"])
    if not problems:
        for ph in (PHASES if phase is None else (phase,)):
            problems += gateway_problems(spec, ph)
    if problems:
        raise Refused("the grid cannot launch:\n  " + "\n  ".join(problems))


def frozen_preregistration(repo: Path, commit: str) -> dict:
    """Check the freeze: ``commit`` is a commit of the git repository ``repo``, its
    paper/preregistration.md holds the exact whole line ``Status: frozen``, and it is on the
    science repository's main line: an ancestor of origin's main as fetched now (a failed fetch
    refuses; the local ref may be stale). Returns what the manifest records: the repository, the
    full commit, the document's sha256 at that commit, and the fetched origin/main commit and the
    time it was fetched (``origin_main``, ``origin_main_fetched_at``). (Kept in step with
    scripts/run_probes.py, feat/closed-book.)"""
    def git(*args: str, text: bool = True, timeout: float = 30, env: dict | None = None):
        try:
            return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=text, timeout=timeout,
                                  check=False, env=env)
        except subprocess.TimeoutExpired:
            raise Refused(f"freeze.prereg_repo {repo}: git {args[0]} did not finish within {timeout:g} s") from None
        except (OSError, subprocess.SubprocessError) as exc:
            raise Refused(f"freeze.prereg_repo {repo}: git cannot be run ({exc})") from None

    if not _COMMITISH.fullmatch(commit or "") or git("cat-file", "-e", f"{commit}^{{commit}}").returncode != 0:
        raise Refused(f"freeze.freeze_commit {commit} is not a commit of {repo}")
    full = git("rev-parse", f"{commit}^{{commit}}").stdout.strip()
    shown = git("show", f"{full}:{PREREG_PATH}", text=False)
    if shown.returncode != 0:
        raise Refused(f"freeze.freeze_commit {full[:12]}: {PREREG_PATH} is not in that commit of {repo}")
    document = shown.stdout
    if not _FROZEN_STATUS.search(document.decode("utf-8", errors="replace").replace("\r\n", "\n")):
        raise Refused(f"freeze.freeze_commit {full[:12]}: {PREREG_PATH} at that commit has no line 'Status: frozen'; "
                      f"the pre-registration is not frozen there")
    # Review of 81f7af2: a branch commit saying "Status: frozen" while main holds the draft must not open the gate.
    # Review of c61ca83: judged on origin's main as it is now, fetched (forced: main may have been rewritten), never
    # on a local ref that may be stale; no prompt for credentials, and a failed fetch refuses.
    fetched_at = datetime.now(UTC).isoformat(timespec="milliseconds")
    fetch = git("fetch", "--quiet", "--no-tags", "origin", f"+refs/heads/main:{ORIGIN_MAIN}", timeout=FETCH_TIMEOUT_S,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    if fetch.returncode != 0:
        why = " ".join(fetch.stderr.split())[:300]
        raise Refused(f"freeze.prereg_repo {repo}: could not fetch origin's main ({why}); the freeze is checked "
                      f"against the origin's main line as it is now, never a local copy: fix the credentials or "
                      f"network (git -C {repo} fetch origin main must work), then run again")
    main = git("rev-parse", "--verify", "--quiet", f"{ORIGIN_MAIN}^{{commit}}")
    if main.returncode != 0:
        raise Refused(f"freeze.prereg_repo {repo}: {ORIGIN_MAIN} is missing after fetching it; run again")
    origin_main = main.stdout.strip()
    on_main = git("merge-base", "--is-ancestor", full, ORIGIN_MAIN)
    if on_main.returncode == 1:
        raise Refused(f"freeze.freeze_commit {full[:12]} is not on origin/main ({origin_main[:12]}) of {repo}: the "
                      f"freeze must be a commit of the science repository's main line (fetch first if it was merged "
                      f"since)")
    if on_main.returncode != 0:
        raise Refused(f"freeze.freeze_commit {full[:12]}: git merge-base --is-ancestor failed in {repo} "
                      f"({on_main.stderr.strip()[:200]})")
    return {"repo": str(repo), "commit": full, "path": PREREG_PATH,
            "preregistration_sha256": hashlib.sha256(document).hexdigest(), "origin_main": origin_main,
            "origin_main_fetched_at": fetched_at}


def split_gate(spec: dict, root: Path) -> dict | None:
    """None for the development split; for any other split (the test grid to come) the frozen
    pre-registration that opens it (``frozen_preregistration``), checked before any question is
    loaded, or Refused."""
    split = spec.get("split")
    if split == SPLIT:
        return None
    freeze = spec.get("freeze") or {}
    if not freeze.get("prereg_repo") or not freeze.get("freeze_commit"):
        raise Refused(f"split {split!r}: needs freeze.prereg_repo and freeze.freeze_commit (a frozen pre-registration)")
    repo = Path(freeze["prereg_repo"])
    return frozen_preregistration(repo if repo.is_absolute() else root / repo, freeze["freeze_commit"])


def load_spec(path: Path) -> tuple[dict, str]:
    data = path.read_bytes()
    return json.loads(data), sha256_bytes(data)


# --- questions ------------------------------------------------------------------------------


def select_questions(path: Path, ids: list[str] | None = None, split: str = SPLIT) -> list[dict]:
    """The questions of ``split`` to run, in file order. ``ids`` (or every question of the split
    when None) must all be in the file with that split: in the development grid a test id is refused,
    and so is a development question whose base question is a test question."""
    rows, problems = [], []
    by_id: dict[str, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            q = json.loads(line)
            if not q.get("id") or not q.get("split"):
                raise Refused(f"{path}:{n}: a question needs an id and a split")
            if q["id"] in by_id:
                raise Refused(f"{path}:{n}: duplicate question id {q['id']!r}")
            by_id[q["id"]] = q
            rows.append(q)
    if ids is None:
        chosen = [q for q in rows if q["split"] == split]
    else:
        if len(set(ids)) != len(ids):
            problems.append("question_ids lists an id twice")
        for qid in ids:
            q = by_id.get(qid)
            if q is None:
                problems.append(f"{qid}: not in {path}")
            elif q["split"] != split:
                problems.append(f"{qid}: split {q['split']!r}; only {SPLITS.get(split, split)} questions run")
        wanted = set(ids)
        chosen = [q for q in rows if q["id"] in wanted]
    other_bases = {q.get("base_question_id") or q["id"]: q["split"] for q in rows if q["split"] != split}
    for q in chosen:
        if q["split"] != split:  # unreachable after the checks above; the run must never see one
            problems.append(f"{q['id']}: split {q['split']!r}")
        elif (base := q.get("base_question_id") or q["id"]) in other_bases:
            problems.append(f"{q['id']}: its base question {base!r} is also in the {other_bases[base]} split")
    if not chosen and not problems:
        problems.append(f"no {SPLITS.get(split, split)} questions in {path}")
    if problems:
        raise Refused(f"{path}: refused:\n  " + "\n  ".join(problems))
    return chosen


# --- expansion ------------------------------------------------------------------------------


def _role_feedback_sources(spec: dict, role: str, parent: str) -> list[str]:
    """Every source of the role's feedback arms on ``parent``, in spec order: what a generic cell of
    that role retries on (design ruling in the design log: the generic control is triggered when any source
    flags; for the panel, whose only checker arm is curated, that is curated)."""
    mine = set(spec["roles"][role].get("branch_cells") or [])
    out: list[str] = []
    for b in spec.get("branch_cells") or []:
        if b.get("id") in mine and b.get("mode") == "feedback" and b.get("parent") == parent:
            out += [s for s in b.get("sources") or [] if s not in out]
    return out


def is_curated(source: str) -> bool:
    """Whether the spec's source key names a curated source (``curated``, ``curated+doc``): the sources a generic
    cell decides on (checker-loop/2)."""
    return source == GENERIC_DECIDES or source.startswith(GENERIC_DECIDES + "+")


def cells(spec: dict, role: str) -> list[dict]:
    """The role's cells, in spec order; a generic cell's sources are its role's feedback-arm sources, and its
    ``verdict_sources`` the curated ones among them (a feedback cell's are its sources)."""
    rs = spec["roles"][role]
    out = [{"id": c, "phase": cs["phase"], "parent": None, "condition": cs["condition"], "source": cs.get("source")}
           for c, cs in spec["cohorts"].items() if c in rs["cohorts"]]
    for b in spec.get("branch_cells") or []:
        if b["id"] in (rs.get("branch_cells") or []):
            sources = (_role_feedback_sources(spec, role, b["parent"]) if b["mode"] == "generic"
                       else list(b["sources"]))
            verdict = [s for s in sources if is_curated(s)] if b["mode"] == "generic" else list(sources)
            out.append({"id": b["id"], "phase": "branch", "parent": b["parent"], "mode": b["mode"], "arm": b["arm"],
                        "sources": sources, "verdict_sources": verdict})
    return out


def role_of(spec: dict) -> dict[str, str]:
    return {m["name"]: m["role"] for m in spec["models"]}


def phase_cells(spec: dict, phase: str) -> list[str]:
    """Every cell of the phase, of any role, in spec order."""
    ids = [*spec["cohorts"], *(b["id"] for b in spec.get("branch_cells") or [])]
    used = {c["id"] for role in spec["roles"] for c in cells(spec, role) if c["phase"] == phase}
    return [c for c in ids if c in used]


def phase_seed(spec: dict, phase: str) -> int:
    return spec["order"]["phases"][phase]["seed"]


def expand(spec: dict, questions: dict[str, list[dict]]) -> list[dict]:
    """Every attempt, per graph and phase in the runner's seeded order (each phase with its own seed).
    A branch attempt names its parent attempt, its mode and the sources whose checks run in it (for
    the generic control, every source whose flag triggers its retry)."""
    roles = role_of(spec)
    by_role = {role: {c["id"]: c for c in cells(spec, role)} for role in spec["roles"]}
    attempts = []
    for graph in spec["graphs"]:
        qids = [q["id"] for q in questions[graph]]
        for phase in PHASES:
            seed = phase_seed(spec, phase)
            # one seeded order over every model's attempts: each model is a product of its own cells and
            # repetitions of the phase, and the runner's rule (sorted by order_key) merges them
            plan = []
            for m in spec["models"]:
                ids = [c for c in model_cell_ids(spec, m) if by_role[m["role"]][c]["phase"] == phase]
                if ids:
                    plan += run_pilot.interleaved_order([m["name"]], ids, qids, model_repetitions(spec, m)[phase], seed)
            order = sorted(plan, key=lambda x: run_pilot.order_key(seed, x))
            for position, (m, cell, qid, rep) in enumerate(order):
                c = by_role[roles[m]][cell]
                a = {"graph": graph, "phase": phase, "seed": seed, "position": position, "model": m, "role": roles[m],
                     "cell": cell, "question_id": qid, "repetition": rep,
                     "order_key": run_pilot.order_key(seed, (m, cell, qid, rep)),
                     "parent": [m, c["parent"], qid, rep] if c["parent"] else None}
                if c["phase"] == "branch":
                    a.update(mode=c["mode"], arm=c["arm"], sources=c["sources"], verdict_sources=c["verdict_sources"])
                attempts.append(a)
    split = spec["split"]
    allowed = {graph: {q["id"] for q in qs if q["split"] == split} for graph, qs in questions.items()}
    stray = [a for a in attempts if a["question_id"] not in allowed[a["graph"]]]
    if stray:
        raise Refused(f"{len(stray)} attempts name a question outside the {SPLITS[split]} split")
    return attempts


def phases(spec: dict, attempts: list[dict]) -> list[dict]:
    """Each phase: its seed, its cells, when it starts, and per graph the order's sha256 as the runner
    records it (``order_sha256`` of the [model, arm, question, repetition] list, and of the plan)."""
    out = []
    for phase in PHASES:
        graphs = {}
        for graph in spec["graphs"]:
            items = sorted((a for a in attempts if a["graph"] == graph and a["phase"] == phase),
                           key=lambda a: a["position"])
            order = [(a["model"], a["cell"], a["question_id"], a["repetition"]) for a in items]
            graphs[graph] = {"attempts": len(order), "order_sha256": run_pilot.order_sha256(order),
                             "plan_sha256": run_pilot.order_sha256(sorted(order))}
        out.append({"phase": phase, "seed": phase_seed(spec, phase), "cells": phase_cells(spec, phase),
                    "cells_by_role": {role: [c["id"] for c in cells(spec, role) if c["phase"] == phase]
                                      for role in spec["roles"]},
                    "starts": spec["order"]["phases"][phase].get("starts"), "graphs": graphs})
    return out


def counts(spec: dict, attempts: list[dict]) -> dict[str, dict[str, dict[str, int]]]:
    """Attempts per graph, model (in spec order) and cell."""
    out: dict[str, dict[str, dict[str, int]]] = {g: {m["name"]: {} for m in spec["models"]} for g in spec["graphs"]}
    for a in attempts:
        per = out.setdefault(a["graph"], {}).setdefault(a["model"], {})
        per[a["cell"]] = per.get(a["cell"], 0) + 1
    return out


def replay_jobs(spec: dict, questions: dict[str, list[dict]]) -> list[dict]:
    """The offline replay: per graph, model and cohort, every source of the graph (replay-only sources
    such as census-t2 and census-t20 included), the unions and every detection-switch variant on the
    cohort's initial outputs. No model calls. A job's matrix has one row per initial record, arm and
    variant (``harness.checkloop.matrix``: an output that was not submitted still has its rows), so it
    expects ``expected_outputs x len(arms) x len(variants)`` rows."""
    replay = spec.get("replay") or {}
    unions = replay.get("unions") or {}
    variants = list(replay.get("variants") or [])
    jobs = []
    for graph, gs in spec["graphs"].items():
        arms = [*gs["sources"], *unions]
        for m in spec["models"]:
            reps = model_repetitions(spec, m)
            for cohort in spec["cohorts"]:
                if cohort not in _model_cohorts(spec, m):
                    continue
                outputs = len(questions[graph]) * reps[spec["cohorts"][cohort]["phase"]]
                jobs.append({"graph": graph, "model": m["name"], "role": m["role"], "cohort": cohort,
                             "sources": list(gs["sources"]), "unions": unions, "arms": arms, "variants": variants,
                             "expected_outputs": outputs, "expected_rows": outputs * len(arms) * len(variants)})
    return jobs


def attempts_text(attempts: list[dict]) -> bytes:
    return "".join(json.dumps(a, separators=(",", ":"), ensure_ascii=False) + "\n" for a in attempts).encode("utf-8")


def build(spec: dict, spec_sha256: str, root: Path, *, for_launch: bool,
          phase: str | None = None, graph: str | None = None) -> tuple[dict, list[dict]]:
    """The manifest and the attempts. ``for_launch``: refuse unless ``phase`` (every phase, when None) can
    launch on ``graph`` (every graph, when None; ``assert_launchable``)."""
    problems = validate_spec(spec)
    if problems:
        raise Refused("the grid spec is malformed:\n  " + "\n  ".join(problems))
    freeze = split_gate(spec, root)  # before any input is read
    found = input_problems(spec, root)
    if found["mismatched"]:
        raise Refused("inputs do not match their declared sha256:\n  " + "\n  ".join(found["mismatched"]))
    if for_launch:
        assert_launchable(spec, root, phase, graph)
    questions, question_info = {}, {}
    for g, gs in spec["graphs"].items():
        path = root / gs["questions"]["path"]
        if not path.is_file():
            raise Refused(f"graphs.{g}.questions: {path} does not exist")
        questions[g] = select_questions(path, gs.get("question_ids"), spec["split"])
        ids = [q["id"] for q in questions[g]]
        question_info[g] = {"path": gs["questions"]["path"], "sha256": sha256_file(path),
                            f"{spec['split']}_questions": len(ids),
                            "ids_sha256": sha256_bytes(json.dumps(ids, separators=(",", ":")).encode("utf-8"))}
    attempts = expand(spec, questions)
    per_cell = counts(spec, attempts)
    table = run_pilot.grid_models(run_pilot.MODELS, spec)  # entry 180; validate_spec refused what it refuses
    roles = role_of(spec)
    phase_of = {c["id"]: c["phase"] for role in spec["roles"] for c in cells(spec, role)}
    per_graph_model = {graph: {m: {"role": roles[m], "attempts": sum(n.values()),
                                   **{ph: sum(v for c, v in n.items() if phase_of[c] == ph) for ph in PHASES}}
                               for m, n in models.items()} for graph, models in per_cell.items()}
    initial = sum(1 for a in attempts if a["phase"] != "branch")
    per_role = {role: {"models": [m for m, r in roles.items() if r == role],
                       "attempts": sum(1 for a in attempts if roles[a["model"]] == role),
                       **{ph: sum(1 for a in attempts if roles[a["model"]] == role and a["phase"] == ph)
                          for ph in PHASES}}
                for role in spec["roles"]}
    unfilled = placeholders(spec)
    by_phase = {ph: phase_placeholders(spec, ph) for ph in PHASES}
    missing_by_phase = {ph: input_problems(spec, root, ph)["missing"] for ph in PHASES}
    by_graph = {g: {ph: phase_placeholders(spec, ph, g) for ph in PHASES} for g in spec["graphs"]}
    missing_by_graph = {g: {ph: input_problems(spec, root, ph, g)["missing"] for ph in PHASES} for g in spec["graphs"]}
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "spec_sha256": spec_sha256,
        "spec": spec,
        "launchable": {ph: not by_phase[ph] and not missing_by_phase[ph] for ph in PHASES},  # every graph's
        "placeholders": unfilled,
        "placeholders_by_phase": by_phase,
        "missing_inputs": found["missing"],
        "missing_by_phase": missing_by_phase,
        # per graph (M6): a phase on a graph launches when that graph's inputs of the phase, and those of no graph,
        # are filled and on disk
        "launchable_by_graph": {g: {ph: not by_graph[g][ph] and not missing_by_graph[g][ph] for ph in PHASES}
                                for g in spec["graphs"]},
        "placeholders_by_graph": by_graph,
        "missing_by_graph": missing_by_graph,
        "freeze": freeze,
        "questions": question_info,
        "roles": roles,
        "cells": {role: cells(spec, role) for role in spec["roles"]},
        # each model's own cells (its cohort list) and repetitions per phase (design log, entry 87)
        "model_cells": {m["name"]: model_cell_ids(spec, m) for m in spec["models"]},
        "model_repetitions": {m["name"]: model_repetitions(spec, m) for m in spec["models"]},
        # the Gateway models' workers per phase against the budget, as planned
        "gateway_load": {"budget": (spec.get("runner_settings") or {}).get("gateway_budget"),
                       **{ph: _gateway_total(spec, ph) for ph in PHASES}},
        "order": {"rule": run_pilot.ORDER_RULE, "phases": phases(spec, attempts)},
        "counts": per_cell,
        "totals": {"attempts": len(attempts), "initial_attempts": initial, "branch_attempts": len(attempts) - initial,
                   "per_role": per_role, "per_graph_model": per_graph_model},
        "model_calls": {
            "initial_attempts": initial,
            "branch_retries_at_most": len(attempts) - initial,
            "note": "an initial attempt (natural or prompted) is one agent run; a branch attempt replays its "
                    "parent's initial phase and "
                    "calls the model only when its arm flags (one retry), so its retries are its parent cohort's "
                    "flag rate times its attempts",
        },
        # each model's route model, provider, route settings (its transport) and caps, from the runner's table
        # (change-log entry 180: a model the spec gives its own request time-out or turn cap, as the runner does)
        "model_settings": {m["name"]: run_pilot.model_settings(m["name"], table[m["name"]], list(spec["graphs"]))
                           for m in spec["models"]},
        "replay_jobs": replay_jobs(spec, questions),
        "attempts_file": ATTEMPTS_FILE,
        "attempts_sha256": sha256_bytes(attempts_text(attempts)),
    }
    return manifest, attempts


def write(out: Path, manifest: dict, attempts: list[dict]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0, filename="") as gz:  # reproducible bytes
        gz.write(attempts_text(attempts))
    (out / ATTEMPTS_FILE).write_bytes(buf.getvalue())
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    p.add_argument("--root", type=Path, default=ROOT, help="the directory the spec's paths are relative to")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--for-launch", action="store_true",
                   help="refuse while an input (of --phase, else of any phase) is a placeholder, missing or not the "
                        "declared file")
    p.add_argument("--phase", choices=PHASES, default=None, help="with --for-launch: the phase to be launched")
    p.add_argument("--graph", default=None, help="with --for-launch: the graph it is launched on (default: every graph)")
    args = p.parse_args(argv)
    if args.phase is not None and not args.for_launch:
        p.error("--phase needs --for-launch")
    if args.graph is not None and not args.for_launch:
        p.error("--graph needs --for-launch")
    try:
        spec, spec_sha = load_spec(args.spec)
        manifest, attempts = build(spec, spec_sha, args.root, for_launch=args.for_launch, phase=args.phase,
                                   graph=args.graph)
    except Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    write(args.out, manifest, attempts)
    t = manifest["totals"]
    print(f"{t['attempts']} attempts ({t['initial_attempts']} initial, {t['branch_attempts']} branch); "
          f"{len(manifest['replay_jobs'])} offline replay jobs")
    def per_phase(n: dict) -> str:
        return ", ".join(f"{n[ph]} {ph}" for ph in PHASES)

    for role, n in t["per_role"].items():
        print(f"  {role} ({', '.join(n['models'])}; {len(manifest['cells'][role])} cells): {n['attempts']} "
              f"({per_phase(n)})")
    for graph, models in t["per_graph_model"].items():
        for m, n in models.items():
            print(f"  {graph} {m} ({n['role']}): {n['attempts']} ({per_phase(n)})")
    for ph in PHASES:
        print(f"phase {ph}: launchable {manifest['launchable'][ph]} ({len(manifest['placeholders_by_phase'][ph])} "
              f"placeholders, {len(manifest['missing_by_phase'][ph])} missing inputs)")
        for g in spec["graphs"]:
            print(f"  on {g}: launchable {manifest['launchable_by_graph'][g][ph]} "
                  f"({len(manifest['placeholders_by_graph'][g][ph])} placeholders, "
                  f"{len(manifest['missing_by_graph'][g][ph])} missing inputs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
