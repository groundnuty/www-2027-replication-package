#!/usr/bin/env python3
"""Closed-book probe runner (design log, E1.7): every (model, probe, question, repetition) of one graph.

    uv run python scripts/run_probes.py --graph rhea-142 --models must panel --order-seed 20261018 \\
        --endpoint http://localhost:7039/ --store-json store.json --out runs/probes
    uv run python scripts/run_probes.py --dry-run --graph rhea-142 uniprot-sp-2026_03 --models all

Probes (harness/probe): (a) ``closed-book-answer-v1``, answer from memory; (b)
``closed-book-query-v1``, write one query, which is executed on the graph's store.
One provider call per try, no tools, the agent runs' output cap (16k), each route's
own sampling and effort settings, 3 repetitions. Records go to ``OUT/<graph>/records``
(``closed-book-probe/1``); every record must replay (the replay check runs at the end
and writes ``OUT/<graph>/replay-check.json``, as for agent batches).

Graphs. ``--graphs-config`` (default data/probes/graphs.json) lists the graphs and their
question files; a graph whose file is absent or pending cannot run, and ``--dry-run`` says
so. A run covers one graph; ``--dry-run`` takes several. Graph binding is the agent's:
queries go unchanged to the graph's own endpoint (``--endpoint``), with the agent runs'
row cap and SPARQL time-out (``--row-cap`` 500, ``--sparql-timeout`` 60 s; at most 290 s).

Splits. Only the dev split runs by default. The test split, and every split of a graph
marked ``"freeze_gated": true`` (the SIB question files: the held-out test), is refused
unless ``--allow-test`` comes with ``--prereg-repo PATH --freeze-commit SHA``: SHA must be a
commit of the git repository PATH (``git cat-file -e SHA^{commit}``) whose
paper/preregistration.md holds the exact whole line ``Status: frozen`` (the study's single
freeze signal, design log), and it must be on the science repository's main line as it is now: the
gate first fetches origin's main (``git fetch origin +refs/heads/main:refs/remotes/origin/main``,
no credential prompt; a failed fetch refuses, since a local ref may be stale), then requires SHA
to be that commit or one of its ancestors (``git merge-base --is-ancestor``). The full commit and
the document's sha256 go into the manifest's fingerprint; the fetched origin/main commit and the
fetch time of each start go into its ``freeze_checks`` (outside the fingerprint, so a resume after
main moved on is the same run). The
refusal comes before any question is loaded, also under ``--dry-run``.

heldout/. The runner refuses any path under a ``heldout/`` directory (the graph list, a
question file, the store description, the run directory, the pre-registration repository) before
reading it: the opaque map lives there and the harness never reads it.

Order (run_pilot's scheme). Every attempt (model, probe id, question id, repetition) is
ordered by the sha256 of the seed and the attempt (``order_key``); ``--order-seed`` is
required with more than one model or probe. Synchronous models start their attempts
strictly in that order (``StartGate``), ``--concurrency`` at a time; Claude models go
through the Message Batches API (``--anthropic-mode batch``, the default), submitted in
that order (the API's processing order cannot be set). The manifest (``manifest.json``)
fingerprints the run and a resume must match it, as for the pilot runner.

Two phases. First every pending attempt's call is made and its reply (request, response
or failure) is kept in ``OUT/<graph>/replies/`` until its record exists; then the records
are built in order, probe (b)'s queries executed one at a time (one heavy query at a time
on a shared store). A resumed run builds the records of kept replies without calling
again, and polls a submitted batch (``OUT/<graph>/batches/``) instead of submitting it again.

Failures. As in the pilot runner: a provider that stays overloaded or unreachable past its
retries is infrastructure (the try is kept in ``infrastructure/``; the call is made again
after ``PROVIDER_RERUN_WAIT_S``, or resubmitted in the next batch, up to ``--infra-reruns``
times); a misconfigured route (401/403/404, or an authentication, permission, billing or
not-found batch error) goes to ``aborted/`` and stops that model. A store outage while
executing probe (b)'s query is infrastructure too: the try is kept in ``infrastructure/``,
the runner waits for the store (probing it, up to ``--infra-retry-s``) and executes the same
reply's query again: the model's reply does not depend on the store, so no call is repeated.
If every try fails the last one is recorded in ``records/`` as ``failed:infrastructure``
(never scored). Keys are read from the environment by the routes and never printed.

Private graphs (``"private": true``, a15): the run directory is created mode 0700, the log
goes to ``OUT/<graph>/run.log`` (mode 0600) instead of the terminal, which gets counts only;
only scripts/probe_summary.py's rates-only summary is meant to leave the directory.

Harness faults. An unexpected exception in our own code while executing or reading probe (b)'s
query is ``failed:harness`` (harness/probe/run.py): never scored, written to ``harness/`` with
its set-aside decision (``<run_id>.setaside.json``, the format of
scripts/relabel_harness_failure.py), its reply kept, so a run after the fix builds the record
without a new call.

Batches. A batch create is never retried: if it fails (or its response is lost), the marker
``batches/submitting-*.json`` holding its requests stays, the model stops, and a later start
refuses until ``--reconcile-batches`` settles it (harness/probe/batch.py: the batch whose
custom ids are exactly the marker's is adopted, waited for if still running). A listing that
finds no such batch proves nothing while the create is recent (a new batch may not be listed
yet). The marker is stamped on disk ``create_started_at`` immediately before the create and
``create_returned_at`` (with ``create_error``) as soon as the create returns or raises. A marker
never started (the run stopped before the create) holds requests no batch can hold: they are
submitted at once. A create that returned an error counts the grace period from its return (a
pause before or during the create cannot start it early): under ``RECONCILE_GRACE_S`` (30
minutes) the marker stays, the model stops and the operator is told when to reconcile again;
past it the batches are listed a second time, after ``RECONCILE_RELIST_S``, and only when that
too finds none is the marker dropped and its requests submitted again, once. A create that
started but never said it returned (the process stopped around it, perhaps after a long pause)
is never submitted again automatically, however old: the operator checks the Anthropic console
and, if no batch holds the requests, runs ``--reconcile-batches --confirm-no-batch MARKER``,
which submits them once and keeps the marker as ``confirmed-no-batch-MARKER`` with who confirmed
and when. A listed batch is adopted whatever the stamps say. Custom ids carry the
run's nonce (64 random bits made once per run directory, kept in the manifest's fingerprint), so
two runs of the same attempts in one account never adopt each other's batches; a run directory
from before nonces is given one on resume if it never used the Batches API, else refused.
Collection claims one custom id at a time and skips any that already has a reply, a record or a
try; the tries of each attempt are counted from its ``infrastructure/`` records, so a
resubmission's custom id (run, attempt and try number) is always new and a restart never resets
the rerun budget.

Exit status: 0 all attempts recorded; 1 an attempt crashed in the runner; 2 refused before
starting; 3 a model stopped (misconfigured route, a batch submission that failed, a reconcile
within the grace period that found no batch: run it again later, or a create that never said it
returned: check the console, then --confirm-no-batch) or the store did not come back; 4 some
attempts recorded as failed:infrastructure after all re-runs; 5 the replay check found records
that do not load, write back or replay; 6 some attempts set aside as failed:harness.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import math
import os
import re
import secrets
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

# Allow running as a plain script from the repository root (and importing the pilot runner beside it).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness.agent.cost import estimate_record_cost  # noqa: E402
from harness.agent.infra import RetryingSparql, wait_for_store  # noqa: E402
from harness.agent.metadata import sanitize_store  # noqa: E402
from harness.agent.providers import AnthropicRoute, _public_url  # noqa: E402
from harness.agent.record import harness_commit, redact_secrets, secret_names, utc_now  # noqa: E402
from harness.agent.replay_check import (  # noqa: E402
    CHECK_FILE,
    CodeNotIdentified,
    require_identified_code,
    verify_batch,
    write_check,
)
from harness.agent.tools import RAW_ROW_CAP  # noqa: E402
from harness.probe.batch import AnthropicBatches, custom_id, reply_from_result  # noqa: E402
from harness.probe.prompts import get_probe  # noqa: E402
from harness.probe.record import ProbeRecord, write_text_atomic  # noqa: E402
from harness.probe.run import (  # noqa: E402
    ProbeConfig,
    Reply,
    build_record,
    build_request,
    call_sync,
    gold_of,
    probe_question,
)
from run_pilot import (  # noqa: E402
    CONFIG_ERROR_CLASSES,
    MODELS,
    PROVIDER_RERUN_WAIT_S,
    ModelSpec,
    Refused,
    StartGate,
    _write_json_atomic,
    acquire_lock,
    check_manifest,
    default_order_seed,
    interleaved_order,
    load_questions,
    order_key,
    order_sha256,
    record_order,
    release_lock,
    route_fingerprint,
)

REPO = Path(__file__).resolve().parent.parent
DEFAULT_GRAPHS = REPO / "data" / "probes" / "graphs.json"
GRAPHS_SCHEMA = "probe-graphs/1"
MUST_MODELS = ("claude-sonnet-5-5", "qwen3.8-27b", "deepseek-v4-flash-0731", "deepseek-v4.1-flash")
PANEL_MODELS = ("glm-5.3-flash", "gemma-4-31b", "llama-3.3-70b", "claude-haiku-5-5")
MODEL_GROUPS = {"must": MUST_MODELS, "panel": PANEL_MODELS, "all": MUST_MODELS + PANEL_MODELS}
MAX_SPARQL_TIMEOUT_S = 290.0
# Cost estimate of --dry-run: the round-1 calibration's Sonnet 5.5 records (UniProt and Rhea, 378 attempts)
# give the first turn's output, one call with adaptive thinking that writes a query: median 128 and 131
# tokens, mean 133 and 131; and 3.38 characters of question text per input token (a fit of the first
# turn's input tokens on question length). Both are proxies for a probe call; the cap bound is printed too.
EST_OUTPUT_TOKENS = 132
EST_CHARS_PER_TOKEN = 3.38
# Reconciling a batch submission (review of 81f7af2): a batch created moments ago may not be in the listing yet,
# so finding none settles a marker only once it is this old, and after a second listing this long after the first.
RECONCILE_GRACE_S = 30 * 60
RECONCILE_RELIST_S = 60.0


# --- configuration -------------------------------------------------------------------------------


def load_graphs(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != GRAPHS_SCHEMA or not isinstance(data.get("graphs"), dict):
        raise Refused(f"{path}: not a {GRAPHS_SCHEMA} graph list")
    return data["graphs"]


def questions_path(graph: dict) -> Path | None:
    if not graph.get("questions"):
        return None
    path = Path(graph["questions"])
    return path if path.is_absolute() else REPO / path


def expand_models(names: list[str]) -> list[str]:
    out: list[str] = []
    for name in names:
        out.extend(MODEL_GROUPS.get(name, (name,)))
    return list(dict.fromkeys(out))


PREREG_PATH = "paper/preregistration.md"
# The exact whole line, case-sensitive (design ruling in the design log: "Status: draft (not frozen)" must not open it).
_FROZEN_STATUS = re.compile(r"^Status: frozen$", re.MULTILINE)
_COMMITISH = re.compile(r"[0-9a-fA-F]{7,40}")
ORIGIN_MAIN = "refs/remotes/origin/main"
FETCH_TIMEOUT_S = 120.0
FREEZE_OBSERVED = ("origin_main", "origin_main_fetched_at")  # what each start saw: recorded, not compared


def frozen_preregistration(repo: Path, commit: str) -> dict:
    """Check the freeze: ``commit`` is a commit of the git repository ``repo``, its
    paper/preregistration.md holds the exact whole line ``Status: frozen``, and it is on the
    science repository's main line: an ancestor of origin's main as fetched now (a failed fetch
    refuses; the local ref may be stale). Returns what the manifest records: the repository, the
    full commit, the document's sha256 at that commit, and the fetched origin/main commit and the
    time it was fetched (``origin_main``, ``origin_main_fetched_at``)."""
    def git(*args: str, text: bool = True, timeout: float = 30, env: dict | None = None):
        try:
            return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=text, timeout=timeout,
                                  check=False, env=env)
        except subprocess.TimeoutExpired:
            raise Refused(f"--prereg-repo {repo}: git {args[0]} did not finish within {timeout:g} s") from None
        except (OSError, subprocess.SubprocessError) as exc:
            raise Refused(f"--prereg-repo {repo}: git cannot be run ({exc})") from None

    if not _COMMITISH.fullmatch(commit or "") or git("cat-file", "-e", f"{commit}^{{commit}}").returncode != 0:
        raise Refused(f"--freeze-commit {commit} is not a commit of {repo}")
    full = git("rev-parse", f"{commit}^{{commit}}").stdout.strip()
    shown = git("show", f"{full}:{PREREG_PATH}", text=False)
    if shown.returncode != 0:
        raise Refused(f"--freeze-commit {full[:12]}: {PREREG_PATH} is not in that commit of {repo}")
    document = shown.stdout
    if not _FROZEN_STATUS.search(document.decode("utf-8", errors="replace").replace("\r\n", "\n")):
        raise Refused(f"--freeze-commit {full[:12]}: {PREREG_PATH} at that commit has no line 'Status: frozen'; "
                      f"the pre-registration is not frozen there")
    # Review of 81f7af2: a branch commit saying "Status: frozen" while main holds the draft must not open the gate.
    # Review of c61ca83: judged on origin's main as it is now, fetched (forced: main may have been rewritten), never
    # on a local ref that may be stale; no prompt for credentials, and a failed fetch refuses.
    fetched_at = utc_now()
    fetch = git("fetch", "--quiet", "--no-tags", "origin", f"+refs/heads/main:{ORIGIN_MAIN}", timeout=FETCH_TIMEOUT_S,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    if fetch.returncode != 0:
        why = " ".join(fetch.stderr.split())[:300]
        raise Refused(f"--prereg-repo {repo}: could not fetch origin's main ({why}); the freeze is checked against "
                      f"the origin's main line as it is now, never a local copy: fix the credentials or network "
                      f"(git -C {repo} fetch origin main must work), then run again")
    main = git("rev-parse", "--verify", "--quiet", f"{ORIGIN_MAIN}^{{commit}}")
    if main.returncode != 0:
        raise Refused(f"--prereg-repo {repo}: {ORIGIN_MAIN} is missing after fetching it; run again")
    origin_main = main.stdout.strip()
    on_main = git("merge-base", "--is-ancestor", full, ORIGIN_MAIN)
    if on_main.returncode == 1:
        raise Refused(f"--freeze-commit {full[:12]} is not on origin/main ({origin_main[:12]}) of {repo}: the freeze "
                      f"must be a commit of the science repository's main line (fetch first if it was merged since)")
    if on_main.returncode != 0:
        raise Refused(f"--freeze-commit {full[:12]}: git merge-base --is-ancestor failed in {repo} "
                      f"({on_main.stderr.strip()[:200]})")
    return {"repo": str(repo), "commit": full, "path": PREREG_PATH,
            "preregistration_sha256": hashlib.sha256(document).hexdigest(), "origin_main": origin_main,
            "origin_main_fetched_at": fetched_at}


def check_split(args: argparse.Namespace, gated: list[str] | None = None) -> dict | None:
    """Refuse the test split, and any split of a freeze-gated graph (the SIB questions, the held-out test),
    unless --allow-test comes with a frozen pre-registration (--prereg-repo, --freeze-commit); checked
    before any question is loaded."""
    freeze_args = args.prereg_repo is not None or args.freeze_commit is not None
    if args.split == "dev" and not gated:
        if args.allow_test or freeze_args:
            raise Refused("--allow-test, --prereg-repo and --freeze-commit are for the test split and freeze-gated "
                          "graphs only")
        return None
    if not args.allow_test or args.prereg_repo is None or args.freeze_commit is None:
        what = f"the {args.split} split" if args.split != "dev" else f"the freeze-gated graph(s) {', '.join(gated)}"
        raise Refused(f"{what} is refused: test-split and SIB probes run only after the pre-registration freeze; "
                      f"pass --allow-test with --prereg-repo PATH --freeze-commit SHA (the frozen commit)")
    return frozen_preregistration(args.prereg_repo, args.freeze_commit)


def refuse_heldout(path: Path | None, what: str) -> None:
    """The harness never reads anything under a heldout/ directory (the opaque map lives there)."""
    if path is None:
        return
    for candidate in (Path(path), Path(path).resolve()):
        if "heldout" in candidate.parts:
            raise Refused(f"{what} {path} is under heldout/: the opaque map there is never read by the harness")


def selected_graphs(args: argparse.Namespace) -> dict[str, dict]:
    """The graphs named by --graph, after refusing any path under heldout/ (the graph list, the store
    description, the run directory, the pre-registration repository and each question file)."""
    for path, what in ((args.graphs_config, "--graphs-config"), (args.store_json, "--store-json"),
                       (args.out, "--out"), (args.prereg_repo, "--prereg-repo")):
        refuse_heldout(path, what)
    graphs = load_graphs(args.graphs_config)
    chosen = {}
    for name in args.graph:
        if name not in graphs:
            raise Refused(f"unknown graph {name!r}; known: {', '.join(graphs)}")
        refuse_heldout(questions_path(graphs[name]), f"graph {name}'s question file")
        chosen[name] = graphs[name]
    return chosen


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--graphs-config", type=Path, default=DEFAULT_GRAPHS)
    p.add_argument("--graph", nargs="+", required=True, help="graph name(s) from --graphs-config (one for a run)")
    p.add_argument("--models", nargs="+", required=True,
                   help=f"model names, or the groups must ({', '.join(MUST_MODELS)}), panel "
                        f"({', '.join(PANEL_MODELS)}) and all")
    p.add_argument("--probes", nargs="+", choices=("answer", "query"), default=["answer", "query"])
    p.add_argument("--split", choices=("dev", "test"), default="dev")
    p.add_argument("--allow-test", action="store_true")
    p.add_argument("--prereg-repo", type=Path, default=None, help="the science repository (git) holding "
                   f"{PREREG_PATH}; with --freeze-commit, the freeze that unlocks the test split")
    p.add_argument("--freeze-commit", default=None, help="the commit of the frozen pre-registration")
    p.add_argument("--reconcile-batches", action="store_true",
                   help="before submitting, settle every batch submission whose id was never written (a create "
                        "that failed or lost its response): adopt the batch that holds its requests; else submit "
                        "once if the create never started, wait out the grace period from its return if it "
                        "returned an error, and stop for --confirm-no-batch if it started but never returned")
    p.add_argument("--confirm-no-batch", action="append", default=[], metavar="MARKER",
                   help="with --reconcile-batches: the operator has checked the Anthropic console and no batch holds "
                        "the requests of this marker (batches/MARKER), whose create never said it returned; its "
                        "requests are submitted once, and who confirmed and when is kept")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--order-seed", type=int, default=None)
    p.add_argument("--allow-code-change", action="store_true",
                   help="resume a run under other code than its last start's; the manifest records the start as "
                        "allowed (F1 review, item 11)")
    p.add_argument("--concurrency", type=int, default=4, help="synchronous calls in flight per model")
    p.add_argument("--concurrency-for", action="append", default=[], metavar="MODEL=N")
    p.add_argument("--endpoint", help="the graph's SPARQL endpoint URL (probe (b))")
    p.add_argument("--store-json", type=Path, help="JSON file describing the store")
    p.add_argument("--row-cap", type=int, default=500)
    p.add_argument("--sparql-timeout", type=float, default=60.0)
    p.add_argument("--max-tokens", type=int, default=ProbeConfig.max_tokens)
    p.add_argument("--infra-retry-s", type=float, default=ProbeConfig.infra_retry_s)
    p.add_argument("--infra-reruns", type=int, default=3)
    p.add_argument("--anthropic-mode", choices=("batch", "sync"), default="batch")
    p.add_argument("--batch-poll-s", type=float, default=60.0)
    p.add_argument("--out", type=Path, help="run directory; each graph writes OUT/<graph>/")
    p.add_argument("--limit", type=int, default=None, help="only the first N questions of the split")
    p.add_argument("--dry-run", action="store_true", help="count the calls and estimate the Anthropic cost; no calls")
    p.add_argument("--est-output-tokens", type=int, default=EST_OUTPUT_TOKENS)
    p.add_argument("--est-chars-per-token", type=float, default=EST_CHARS_PER_TOKEN)
    args = p.parse_args(argv)
    if args.reps < 1 or args.concurrency < 1 or args.row_cap < 1 or args.infra_reruns < 0:
        p.error("--reps, --concurrency and --row-cap must be positive, --infra-reruns not negative")
    if args.confirm_no_batch and not args.reconcile_batches:
        p.error("--confirm-no-batch needs --reconcile-batches")
    if args.row_cap > RAW_ROW_CAP:  # the scored rows are cut from the stored raw result (tools.RAW_ROW_CAP)
        p.error(f"--row-cap must be at most {RAW_ROW_CAP}, the rows of a raw result a record keeps")
    if not 0 < args.sparql_timeout <= MAX_SPARQL_TIMEOUT_S:
        p.error(f"--sparql-timeout must be positive and at most {MAX_SPARQL_TIMEOUT_S:g} s")
    args.models = expand_models(args.models)
    concurrency_for = {}
    for item in args.concurrency_for:
        name, _, raw = item.partition("=")
        if name not in args.models or not raw.isdigit() or int(raw) < 1:
            p.error(f"--concurrency-for {item!r}: expected MODEL=N with MODEL one of --models")
        concurrency_for[name] = int(raw)
    args.concurrency_for = concurrency_for
    if not args.dry_run:
        if len(args.graph) != 1:
            p.error("a run covers one graph; --dry-run takes several")
        missing = [f for f in ("endpoint", "store_json", "out") if getattr(args, f) is None]
        if missing and "query" in args.probes:
            p.error(f"a run needs {', '.join('--' + m.replace('_', '-') for m in missing)}")
        if args.out is None:
            p.error("a run needs --out")
        if len(args.models) * len(args.probes) > 1 and args.order_seed is None:
            p.error("--order-seed is required when a run has more than one model or probe")
    return args


def probe_route(spec: ModelSpec) -> Any:
    """The model's agent route, used for one call with no tools; a Claude route caches only the static
    system prefix (the question is the item)."""
    route = spec.factory()
    if isinstance(route, AnthropicRoute):
        route.auto_cache = False
    return route


def call_mode(route: Any, args: argparse.Namespace) -> str:
    return "batch" if route.provider == "anthropic" and args.anthropic_mode == "batch" else "sync"


# --- dry run -------------------------------------------------------------------------------------


def dry_run(args: argparse.Namespace, models: dict[str, ModelSpec], say: Callable[[str], None],
            graphs: dict[str, dict]) -> int:
    unknown = [m for m in args.models if m not in models]
    if unknown:
        raise Refused(f"unknown model(s) {unknown}; known: {', '.join(models)}")
    routes = {name: probe_route(models[name]) for name in args.models}
    say(f"dry run: split {args.split}, {args.reps} repetition(s), probes {', '.join(args.probes)}; no calls")
    say("| graph | model | probe | mode | questions | calls | est. input tok | est. USD (central) | USD bound (cap) |")
    say("|---|---|---|---|---:|---:|---:|---:|---:|")
    total = {"calls": 0, "anthropic_calls": 0, "usd": 0.0, "usd_cap": 0.0}
    for graph_name in args.graph:
        graph = graphs[graph_name]
        path = questions_path(graph)
        if path is None or not path.is_file():
            why = graph.get("pending") or f"{graph.get('questions')} is not in this checkout"
            say(f"| {graph_name} | — | — | — | absent | 0 | | | | ({why})")
            continue
        questions = load_questions(path, args.split)[: args.limit]
        for name in args.models:
            route = routes[name]
            mode = call_mode(route, args)
            for probe in args.probes:
                spec = get_probe(probe)
                calls = len(questions) * args.reps
                total["calls"] += calls
                cells = ["", "", ""]
                if route.provider == "anthropic":
                    inputs = [math.ceil((len(spec.system) + len(q["question"])) / args.est_chars_per_token)
                              for q in questions] * args.reps
                    tokens_in = sum(inputs)
                    usd = projected_usd(route.route_info(), mode, inputs, args.est_output_tokens)
                    usd_cap = projected_usd(route.route_info(), mode, inputs, args.max_tokens)
                    total["anthropic_calls"] += calls
                    total["usd"] += usd or 0.0
                    total["usd_cap"] += usd_cap or 0.0
                    cells = [str(tokens_in), f"{usd:.4f}", f"{usd_cap:.2f}"]
                say(f"| {graph_name} | {name} | {spec.id} | {mode} | {len(questions)} | {calls} | "
                    f"{' | '.join(cells)} |")
    say(f"total: {total['calls']} calls, {total['anthropic_calls']} of them to Claude; estimated Claude cost "
        f"${total['usd']:.4f} at {args.est_output_tokens} output tokens per call (cap bound ${total['usd_cap']:.2f} "
        f"at {args.max_tokens}); no cache reads assumed (the probe prompts are under the 512-token cacheable "
        f"minimum); HPCNet routes cost nothing per token")
    return 0


def _totals(tokens_in: int, tokens_out: int) -> dict:
    return {"input_tokens": tokens_in, "output_tokens": tokens_out, "cache_read_tokens": 0, "cache_write_tokens": 0,
            "cache_write_5m_tokens": 0, "cache_write_1h_tokens": 0}


def projected_usd(route_info: dict, mode: str, inputs: list[int], output_per_call: int) -> float | None:
    """The projected cost of calls with these input token counts, priced as their records will be
    (cost.estimate_record_cost): each call is one request, so a model priced by prompt length (Haiku 5.5) is
    priced call by call at the rate for its own prompt; summed totals would take the long-prompt rate."""
    per_call = [_totals(n, output_per_call) for n in inputs]
    record = {"route": route_info, "call": {"mode": mode},
              "tokens": {"totals": _totals(sum(inputs), output_per_call * len(inputs)), "per_turn": per_call}}
    return estimate_record_cost(record)["usd"]


# --- the run -------------------------------------------------------------------------------------


def attempt_digest(attempt: tuple) -> str:
    return hashlib.sha256(json.dumps(list(attempt), separators=(",", ":")).encode()).hexdigest()[:40]


def run_nonce(out: Path) -> str:
    """The run's nonce: 64 random bits (16 hex digits) made once per run directory and kept in its manifest's
    fingerprint, which a resume must match; batch custom ids carry it (harness/probe/batch.py). A run started
    before nonces is given one on resume if it never used the Batches API (nothing to confuse), else refused."""
    path = out / "manifest.json"
    if not path.exists():
        return f"{secrets.randbits(64):016x}"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    nonce = manifest.get("fingerprint", {}).get("run_nonce")
    if isinstance(nonce, str) and re.fullmatch(r"[0-9a-f]{16}", nonce):
        return nonce
    if nonce is not None or used_batches(out):
        raise Refused(f"{path} has no run nonce (the run was started before batch custom ids carried one) and the "
                      f"run used the Message Batches API, so its batches cannot be told from another run's: use a "
                      f"new --out")
    nonce = f"{secrets.randbits(64):016x}"
    manifest["fingerprint"]["run_nonce"] = nonce
    manifest["run_nonce_assigned"] = {"at": utc_now(), "nonce": nonce, "why": "resumed a run started before run "
                                      "nonces that never used the Message Batches API"}
    _write_json_atomic(path, manifest)
    return nonce


def _operator() -> str:
    """Who is running the runner (for a recorded confirmation)."""
    try:
        return getpass.getuser()
    except (KeyError, OSError):
        return os.environ.get("USER") or "unknown"


def used_batches(out: Path) -> bool:
    """Whether a run directory holds any trace of the Message Batches API: a batch file or marker, or a kept
    reply or record of a batch call."""
    if any((out / "batches").glob("*.json")):
        return True
    for sub in ("replies", "records", "infrastructure", "aborted", "harness"):
        for p in (out / sub).glob("*.json"):
            data = json.loads(p.read_text(encoding="utf-8"))
            call = data.get("call") or (data.get("reply") or {}).get("call") or {}
            if call.get("mode") == "batch" or call.get("custom_id") or call.get("batch_id"):
                return True
    return False


def record_key(data: dict) -> tuple:
    return (data["question_id"], data["model"], data["prompt"]["id"], data["repetition"])


def existing_keys(records_dir: Path) -> set[tuple]:
    seen: dict[tuple, str] = {}
    dupes = []
    for path in sorted(records_dir.glob("*.json")) if records_dir.is_dir() else []:
        key = record_key(json.loads(path.read_text(encoding="utf-8")))
        if key in seen:
            dupes.append(f"{key}: {seen[key]} and {path.name}")
        seen[key] = path.name
    if dupes:
        raise Refused("records share an attempt key (question, model, probe, repetition):\n  " + "\n  ".join(dupes))
    return set(seen)


def is_config_failure(reply: Reply) -> bool:
    failure = reply.provider_failure or {}
    return (failure.get("failure_type") == "provider_error"
            and (failure.get("details") or {}).get("error_class") in CONFIG_ERROR_CLASSES)


def is_provider_infra(reply: Reply) -> bool:
    return (reply.provider_failure or {}).get("failure_type") == "infrastructure"


def run(argv: list[str] | None = None, *, models: dict[str, ModelSpec] = MODELS,
        sparql_factory: Callable[[argparse.Namespace], Any] | None = None,
        batches_factory: Callable[[Any, argparse.Namespace], Any] | None = None,
        log: Callable[[str], None] | None = None, sleep: Callable[[float], None] = time.sleep) -> int:
    args = parse_args(argv)
    sink = log or (lambda msg: print(msg, file=sys.stderr, flush=True))

    def say(msg: str) -> None:
        sink(redact_secrets(str(msg))[0])

    try:
        graphs = selected_graphs(args)
        freeze = check_split(args, [name for name, g in graphs.items() if g.get("freeze_gated")])
        if args.dry_run:
            return dry_run(args, models, say, graphs)
        return _run(args, models, sparql_factory, batches_factory, say, sleep, graphs, freeze)
    except Refused as exc:
        say(f"error: {exc}")
        return 2
    except Exception as exc:  # printed through the redaction, never as a raw traceback
        say(f"error: {type(exc).__name__}: {exc}")
        return 2


def _run(args, models, sparql_factory, batches_factory, say, sleep, graphs, freeze) -> int:
    try:  # every probe record names the code it ran from (dress rehearsal F1): code that cannot be named never runs
        code = require_identified_code(harness_commit())
    except CodeNotIdentified as exc:
        raise Refused(str(exc)) from None
    say(f"code: commit {code['git_commit']} from {code['source']}, tree {code['tree_sha256']}"
        + (" (uncommitted changes)" if code["git_dirty"] else ""))
    graph_name = args.graph[0]
    graph = graphs[graph_name]
    path = questions_path(graph)
    if path is None or not path.is_file():
        raise Refused(f"graph {graph_name}: no question file ({graph.get('pending') or graph.get('questions')})")
    problems = []
    for name in args.models:
        spec = models.get(name)
        if spec is None:
            problems.append(f"unknown model {name!r}; known: {', '.join(models)}")
        elif spec.key_envs and not any((os.environ.get(e) or "").strip() for e in spec.key_envs):
            problems.append(f"{name}: no API key; set {' or '.join(spec.key_envs)}")
    questions = load_questions(path, args.split)[: args.limit]
    if not questions:
        problems.append(f"no questions with split {args.split!r} in {path}")
    if problems:
        raise Refused("; ".join(problems))
    out = args.out / graph_name
    private = bool(graph.get("private"))
    # A private graph's files (records, replies, batches, manifest, order, check, log) are created mode 600
    # and its directories 700 (AGENTS.md: a15 content stays in mode-600 files); the mask is restored after.
    old_mask = os.umask(0o077) if private else None
    try:
        return _run_graph(args, models, sparql_factory, batches_factory, say, sleep, out, graph_name, graph,
                          path, questions, freeze, private)
    finally:
        if old_mask is not None:
            os.umask(old_mask)


def _run_graph(args, models, sparql_factory, batches_factory, say, sleep, out, graph_name, graph, path, questions,
               freeze, private) -> int:
    out.mkdir(parents=True, exist_ok=True)
    if private:
        os.chmod(out, 0o700)
        log_path = out / "run.log"
        fd = os.open(log_path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        os.close(fd)
        terminal = say

        def say(msg: str) -> None:  # noqa: F811 - a private graph's log stays in its directory
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(redact_secrets(str(msg))[0] + "\n")

        terminal(f"{graph_name} is private: the log is {log_path}; this terminal gets counts only")
    else:
        terminal = say
    lock = acquire_lock(out)
    try:
        return _run_locked(args, models, sparql_factory, batches_factory, say, terminal, sleep, out, graph_name,
                           path, questions, freeze)
    finally:
        release_lock(lock)


def _run_locked(args, models, sparql_factory, batches_factory, say, terminal, sleep, out, graph_name, path,
                questions, freeze) -> int:
    say(f"redaction: environment variables treated as secrets: {', '.join(secret_names()) or 'none'}")
    config = ProbeConfig(max_tokens=args.max_tokens, infra_retry_s=args.infra_retry_s)
    store = sanitize_store(json.loads(args.store_json.read_text(encoding="utf-8"))) if args.store_json else None
    specs = [get_probe(p) for p in args.probes]
    model_names = list(dict.fromkeys(args.models))
    routes = {name: probe_route(models[name]) for name in model_names}
    modes = {name: call_mode(routes[name], args) for name in model_names}
    nonce = run_nonce(out)
    fingerprint = {
        "kind": "closed-book-probes",
        "graph": graph_name,
        "run_nonce": nonce,
        "questions_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "split": args.split,
        "freeze": None if freeze is None else {k: v for k, v in freeze.items() if k not in FREEZE_OBSERVED},
        "store": store,
        "endpoint": _public_url(args.endpoint) if args.endpoint else None,
        "row_cap": args.row_cap,
        "sparql_timeout_s": args.sparql_timeout,
        "config": {k: v for k, v in asdict(config).items() if k != "infra_retry_s"},
        "probes": {s.id: {"system_sha256": s.system_sha256, "instruction_sha256": s.instruction_sha256}
                   for s in specs},
        "routes": {name: {**route_fingerprint(routes[name]), "call_mode": modes[name]} for name in model_names},
    }
    records_dir, infra_dir, aborted_dir = out / "records", out / "infrastructure", out / "aborted"
    harness_dir = out / "harness"
    replies_dir, batches_dir = out / "replies", out / "batches"
    done = existing_keys(records_dir)
    added = check_manifest(out, fingerprint, has_records=bool(done), say=say, allow_code_change=args.allow_code_change)
    if added:
        say(f"manifest: recorded settings for {', '.join(added)}")
    if freeze is not None:  # origin/main as this start saw it: recorded, not compared (it moves on with each fetch)
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        manifest.setdefault("freeze_checks", []).append({"at": utc_now(), **{k: freeze[k] for k in FREEZE_OBSERVED}})
        _write_json_atomic(out / "manifest.json", manifest)
    seed = args.order_seed if args.order_seed is not None else default_order_seed(path)
    order = interleaved_order(model_names, [s.id for s in specs], [q["id"] for q in questions], args.reps, seed)
    record_order(out, seed=seed, seed_source="flag" if args.order_seed is not None else "questions_sha256",
                 order=order, resumed=bool(done), say=say)
    order_sha = order_sha256(order)
    say(f"order: {len(order)} attempts over {len(model_names)} model(s) and {len(specs)} probe(s), seed {seed}; "
        f"order sha256 {order_sha}")
    by_id = {q["id"]: q for q in questions}
    route_models = {name: routes[name].model for name in model_names}
    position = {attempt: i for i, attempt in enumerate(order)}
    for d in (records_dir, replies_dir, batches_dir):
        d.mkdir(parents=True, exist_ok=True)
    counts = {"recorded": 0, "skipped": 0, "aborted": 0, "infra_tries": 0, "infra_final": 0, "errors": 0,
              "stopped": 0, "store_lost": 0, "harness": 0}
    lock = threading.Lock()
    stops = {name: threading.Event() for name in model_names}

    def bump(key: str, n: int = 1) -> None:
        with lock:
            counts[key] += n

    def key_of(attempt: tuple) -> tuple:
        name, probe_id, qid, rep = attempt
        return (qid, route_models[name], probe_id, rep)

    def runner_info(attempt: tuple) -> dict:
        return {"attempt": list(attempt), "position": position[attempt], "order_key": order_key(seed, attempt),
                "order_seed": seed, "order_sha256": order_sha}

    def reply_path(attempt: tuple) -> Path:
        return replies_dir / f"{attempt_digest(attempt)}.json"

    def keep_reply(attempt: tuple, reply: Reply, route_info: dict, runner: dict) -> None:
        write_text_atomic(replies_dir, reply_path(attempt).name, json.dumps(
            {"attempt": list(attempt), "route": route_info, "reply": reply.to_json(), "runner": runner},
            ensure_ascii=False, indent=1))

    def record_of(attempt: tuple, reply: Reply, route_info: dict, runner: dict, sparql: Any = None) -> ProbeRecord:
        name, probe_id, qid, rep = attempt
        q = by_id[qid]
        return build_record(spec=get_probe(probe_id), question={**probe_question(q), "graph": graph_name},
                            reply=reply, route_info=route_info, gold=gold_of(q), config=config, repetition=rep,
                            store=store, sparql=sparql, runner=runner)

    def abort(name: str, attempt: tuple, reply: Reply, route_info: dict) -> None:
        rec = record_of(attempt, reply, route_info, runner_info(attempt), sparql=_null_sparql(args))
        rec.write(aborted_dir)
        bump("aborted")
        if not stops[name].is_set():
            stops[name].set()
            bump("stopped")
        say(f"error: [{name}] stopped: misconfigured route: {(reply.provider_failure or {}).get('message', '')[:300]}")

    # What earlier starts left on disk, so that a crash at any point neither repeats nor forgets a try: the
    # provider failures of each attempt (its infrastructure/ tries, which count against --infra-reruns and
    # number its next try) and every batch custom id that already has a reply, a record or a try.
    provider_tries: dict[tuple, list] = {}
    store_tries: dict[tuple, list] = {}  # the store outages met while executing each attempt's query
    claimed: set[str] = set()
    for sub in ("infrastructure", "aborted", "records", "harness"):
        for path in sorted((out / sub).glob("*.json")) if (out / sub).is_dir() else []:
            data = json.loads(path.read_text(encoding="utf-8"))
            if "schema_version" not in data:
                continue  # a set-aside decision, not a record
            if (data.get("call") or {}).get("custom_id"):
                claimed.add(data["call"]["custom_id"])
            runner = data.get("runner") or {}
            if sub == "infrastructure" and runner.get("attempt"):
                kind = store_tries if runner.get("infra_source") == "store" else provider_tries
                kind.setdefault(tuple(runner["attempt"]), []).append(
                    (runner.get("infra_try") or 0, data["started_at"], data["run_id"]))
    provider_tries = {a: [run_id for *_, run_id in sorted(t)] for a, t in provider_tries.items()}
    store_tries = {a: [run_id for *_, run_id in sorted(t)] for a, t in store_tries.items()}
    for path in replies_dir.glob("*.json"):
        cid = (json.loads(path.read_text(encoding="utf-8"))["reply"].get("call") or {}).get("custom_id")
        if cid:
            claimed.add(cid)

    def tries_of(attempt: tuple) -> list[str]:
        with lock:
            return provider_tries.setdefault(attempt, [])

    def provider_infra_try(attempt: tuple, reply: Reply, route_info: dict) -> None:
        tries = tries_of(attempt)
        runner = {**runner_info(attempt), "infra_try": len(tries) + 1, "infra_source": "provider"}
        rec = record_of(attempt, reply, route_info, runner, sparql=_null_sparql(args))
        rec.write(infra_dir)  # on disk first: a crash after this still counts the try
        with lock:
            tries.append(rec.run_id)
        bump("infra_tries")

    markers = sorted(batches_dir.glob("submitting-*.json"))
    if markers and not args.reconcile_batches:
        raise Refused(f"{markers[0]} is left from a batch submission whose batch id was never written (the create "
                      f"failed or its response was lost, so the batch may exist and be paid for): run again with "
                      f"--reconcile-batches, which adopts the batch that holds its requests or drops the marker")

    # Phase 1: the calls. Pending attempts per model, in order; a kept reply is not called again.
    pending: dict[str, list[tuple]] = {name: [] for name in model_names}
    for attempt in order:
        if key_of(attempt) in done:
            bump("skipped")
        elif not reply_path(attempt).exists():
            pending[attempt[0]].append(attempt)
    gate = StartGate(model_names)

    def sync_model(name: str) -> None:
        items = pending[name]
        workers = args.concurrency_for.get(name, args.concurrency)
        say(f"{name}: {len(items)} calls pending, synchronous, {workers} in flight, route model {route_models[name]}")

        def one(ticket: int, attempt: tuple) -> None:
            started = False
            try:
                if stops[name].is_set():
                    return
                _, probe_id, qid, _ = attempt
                tries = tries_of(attempt)  # an earlier start's tries count too
                start_index = None
                while True:
                    route = probe_route(models[name])
                    request = build_request(route, get_probe(probe_id), by_id[qid]["question"], config.max_tokens)
                    if not started:  # the model's attempts start strictly in order (run_pilot's StartGate)
                        gate.wait(name, ticket)
                        start_index = list(gate.start(name, ticket))
                        started = True
                    reply = call_sync(route, request)
                    if is_config_failure(reply):
                        abort(name, attempt, reply, route.route_info())
                        return
                    if is_provider_infra(reply) and len(tries) < args.infra_reruns:
                        provider_infra_try(attempt, reply, route.route_info())
                        say(f"warning: [{name}] provider unavailable; re-call {len(tries)} of {args.infra_reruns} "
                            f"after {PROVIDER_RERUN_WAIT_S:g} s")
                        sleep(PROVIDER_RERUN_WAIT_S)
                        continue
                    runner = {**runner_info(attempt), "start_index": start_index,
                              "provider_infra_run_ids": list(tries)}
                    keep_reply(attempt, reply, route.route_info(), runner)
                    return
            except Exception as exc:  # one attempt's crash must not stop the others; nothing is recorded
                bump("errors")
                say(f"error: [{name}] {attempt[1]} rep {attempt[3]}: {type(exc).__name__}: {exc}")
                say("".join(traceback.format_exception_only(type(exc), exc)).strip())
            finally:
                if not started:
                    gate.skip(name, ticket)

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=name) as pool:
            for ticket, attempt in enumerate(items):
                pool.submit(one, ticket, attempt)

    def batch_model(name: str) -> None:
        route = routes[name]
        batches = batches_factory(route, args) if batches_factory else AnthropicBatches(
            route, poll_s=args.batch_poll_s, sleep=sleep)
        for marker_path in sorted(batches_dir.glob(f"submitting-{name}-*.json")):  # --reconcile-batches
            if not reconcile(name, batches, marker_path):
                if not stops[name].is_set():
                    stops[name].set()
                    bump("stopped")
                return
        # A batch submitted by an earlier start (or adopted just now) is collected, never submitted again.
        for state_path in sorted(batches_dir.glob("msgbatch*.json")):
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state["model"] == name and state["status"] != "collected":
                collect(name, batches, state_path, state)
        round_no = 0
        while not stops[name].is_set():
            items = [a for a in pending[name] if not reply_path(a).exists() and key_of(a) not in done]
            if not items:
                break
            requests = {}
            for attempt in items:  # the try number comes from disk, so a resubmission always has a new id
                _, probe_id, qid, _ = attempt
                cid = custom_id(attempt, len(tries_of(attempt)), nonce)
                requests[cid] = {"attempt": list(attempt), "request": build_request(
                    route, get_probe(probe_id), by_id[qid]["question"], config.max_tokens)}
            state = {"model": name, "status": "submitting", "round": round_no, "created_at": utc_now(),
                     "create_started_at": None, "requests": requests}
            marker = write_text_atomic(batches_dir, f"submitting-{name}-{uuid.uuid4().hex[:12]}.json",
                                       json.dumps(state))
            say(f"{name}: submitting {len(requests)} requests to the Message Batches API (round {round_no})")
            # Stamped on disk immediately before the create and again as soon as it returns or raises (reviews of
            # c61ca83 and e352b53): unstamped, no create was made; started but never returned, the process stopped
            # at an unknown point around the create, and only the operator can settle the marker.
            state["create_started_at"] = utc_now()
            write_text_atomic(batches_dir, marker.name, json.dumps(state))
            try:
                batch_id = batches.submit([(cid, r["request"]) for cid, r in requests.items()])
            except Exception as exc:  # never retried: the batch may exist; the marker stays for reconciliation
                state.update({"create_returned_at": utc_now(), "create_error": f"{type(exc).__name__}: {exc}"[:500]})
                write_text_atomic(batches_dir, marker.name, json.dumps(state))
                if not stops[name].is_set():
                    stops[name].set()
                    bump("stopped")
                say(f"error: [{name}] stopped: the batch submission failed ({type(exc).__name__}: {str(exc)[:300]}); "
                    f"the batch may have been created, so it is not submitted again: run again with "
                    f"--reconcile-batches ({marker.name} holds its requests)")
                return
            state.update({"batch_id": batch_id, "status": "submitted", "submitted_at": state["create_started_at"],
                          "create_returned_at": utc_now()})
            state_path = write_text_atomic(batches_dir, f"{batch_id}.json", json.dumps(state))
            marker.unlink()
            say(f"{name}: batch {batch_id} submitted with {len(requests)} requests (round {round_no})")
            collect(name, batches, state_path, state)
            round_no += 1

    def reconcile(name: str, batches: Any, marker_path: Path) -> bool:
        """Settle a submission marker: adopt the batch that holds exactly its custom ids, or drop it when no batch
        does and that is sure: the create never started; or it returned an error, the grace period from that
        return has passed, and two listings find nothing; or, for a create that never said it returned, the
        operator confirmed (--confirm-no-batch). False: not settled, stop."""
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        ids = list(marker["requests"])
        if "create_started_at" in marker and marker["create_started_at"] is None:
            say(f"{name}: reconcile: the create of {marker_path.name} never started (the run stopped before it), so "
                f"no batch holds its requests; they are submitted now")
            marker_path.unlink()
            return True
        submitted_at = marker.get("create_started_at") or marker["created_at"]
        found = batches.find_submitted(ids, marker["created_at"], say=say)
        returned_at = marker.get("create_returned_at")
        if not found and not returned_at:
            # Started (or a marker of c61ca83, written before the create with no stamps) but never said it returned:
            # the process stopped at an unknown point around the create, possibly after a long pause, so no age
            # makes "not listed" mean "not created". Never submitted again without the operator's word.
            if marker_path.name not in args.confirm_no_batch:
                say(f"error: [{name}] stopped: reconcile: {marker_path.name} says its create started at "
                    f"{marker.get('create_started_at') or 'an unknown time (a marker without create stamps)'} but "
                    f"never that it returned, and no batch listed holds its {len(ids)} requests; a batch may still "
                    f"exist, so they are never submitted again automatically. Check the Anthropic console (or the "
                    f"batch list) for a batch of {len(ids)} requests created since {marker['created_at']}; if there "
                    f"is none, run again with --reconcile-batches --confirm-no-batch {marker_path.name}")
                return False
            confirmation = {"confirmed_by": _operator(), "host": socket.gethostname(), "confirmed_at": utc_now(),
                            "flag": f"--confirm-no-batch {marker_path.name}"}
            write_text_atomic(batches_dir, f"confirmed-no-batch-{marker_path.name}",
                              json.dumps({**marker, "confirmation": confirmation}))
            marker_path.unlink()
            say(f"{name}: reconcile: {confirmation['confirmed_by']} confirmed at {confirmation['confirmed_at']} that "
                f"no batch holds the requests of {marker_path.name} (kept as confirmed-no-batch-{marker_path.name}); "
                f"they are submitted again, once")
            return True
        if not found:
            returned = datetime.fromisoformat(returned_at)
            age_s = (datetime.now(UTC) - returned).total_seconds()
            if age_s < RECONCILE_GRACE_S:
                retry_at = (returned + timedelta(seconds=RECONCILE_GRACE_S)).isoformat(timespec="seconds")
                say(f"error: [{name}] stopped: reconcile: no batch listed holds the requests of {marker_path.name} "
                    f"yet, but its create returned ({marker.get('create_error', 'an error')[:200]}) only "
                    f"{age_s / 60:.0f} min ago and a new batch may not be listed so soon; nothing is submitted: run "
                    f"--reconcile-batches again after {retry_at}")
                return False
            say(f"{name}: reconcile: no batch listed holds the requests of {marker_path.name}; listing again in "
                f"{RECONCILE_RELIST_S:g} s before submitting them")
            sleep(RECONCILE_RELIST_S)
            found = batches.find_submitted(ids, marker["created_at"], say=say)
        if not found:
            say(f"{name}: reconcile: no batch holds the requests of {marker_path.name} in two listings "
                f"{age_s / 60:.0f} min after its create returned an error; none was created, so they are submitted "
                f"again, once")
        if len(found) > 1:
            say(f"warning: [{name}] reconcile: {len(found)} batches hold the requests of {marker_path.name} "
                f"({', '.join(found)}); each custom id is claimed once")
        for batch_id in found:
            say(f"{name}: reconcile: batch {batch_id} holds the requests of {marker_path.name}; adopted")
            write_text_atomic(batches_dir, f"{batch_id}.json", json.dumps({
                **marker, "batch_id": batch_id, "status": "submitted", "submitted_at": submitted_at,
                "reconciled_from": marker_path.name}))
        marker_path.unlink()
        return True

    def collect(name: str, batches: Any, state_path: Path, state: dict) -> None:
        """Claim a batch's results, one custom id at a time: a result whose custom id already has a reply, a
        record or a try is skipped, so collecting again after a crash repeats nothing."""
        ended = batches.wait(state["batch_id"], say)
        results = batches.results(state["batch_id"])
        route_info = routes[name].route_info()
        wall = _seconds_between(state.get("submitted_at"), ended.get("ended_at"))
        for cid, entry in state["requests"].items():
            attempt = tuple(entry["attempt"])
            if attempt not in position:  # submitted by a start with another plan: its reply is not kept
                say(f"warning: [{name}] batch {state['batch_id']} holds an attempt outside this run's plan; skipped")
                continue
            if cid in claimed or key_of(attempt) in done or reply_path(attempt).exists():
                continue
            call = {"mode": "batch", "batch_id": state["batch_id"], "custom_id": cid,
                    "started_at": state.get("submitted_at"), "batch_ended_at": ended.get("ended_at"), "wall_s": wall}
            result = results.get(cid, {"type": "expired"})  # a request the results do not hold was not processed
            reply = reply_from_result(result, entry["request"], call)
            reply.call["ended_at"] = ended.get("ended_at")
            if is_config_failure(reply):
                if not stops[name].is_set():  # one aborted record says the route is misconfigured
                    abort(name, attempt, reply, route_info)
                    claimed.add(cid)
                continue
            if is_provider_infra(reply) and len(tries_of(attempt)) < args.infra_reruns:
                provider_infra_try(attempt, reply, route_info)
            else:
                keep_reply(attempt, reply, route_info, {**runner_info(attempt), "start_index": None,
                                                        "provider_infra_run_ids": list(tries_of(attempt))})
            claimed.add(cid)
        state["status"] = "collected"
        write_text_atomic(state_path.parent, state_path.name, json.dumps(state))

    threads = []
    for name in model_names:
        target = batch_model if modes[name] == "batch" else sync_model
        threads.append(threading.Thread(target=_guarded(target, name, say, bump), name=f"calls-{name}"))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Phase 2: the records, in order; probe (b)'s queries one at a time.
    if sparql_factory is None:
        from harness.sparql import SparqlClient

        client = SparqlClient(args.endpoint, timeout_s=args.sparql_timeout, row_cap=args.row_cap) \
            if args.endpoint else None
        sparql_factory = lambda _args: client  # noqa: E731
    kept = []
    for p in replies_dir.glob("*.json"):
        data = json.loads(p.read_text(encoding="utf-8"))
        attempt = tuple(data["attempt"])
        if attempt in position:
            kept.append((position[attempt], p, attempt, data))
    for _, p, attempt, data in sorted(kept, key=lambda item: item[0]):
        name, probe_id, qid, rep = attempt
        if key_of(attempt) in done:
            p.unlink()
            continue
        reply, route_info, runner = Reply.from_json(data["reply"]), data["route"], data["runner"]
        outages = store_tries.setdefault(attempt, [])  # an earlier start's outages count too
        while True:
            sparql = None
            if probe_id == get_probe("query").id:
                sparql = RetryingSparql(sparql_factory(args), limit_s=args.infra_retry_s, sleep=sleep)
            rec = record_of(attempt, reply, route_info, runner, sparql)
            store_down = rec.outcome["failure_type"] == "infrastructure" and rec.outcome["infra_source"] == "store"
            if not store_down or len(outages) >= args.infra_reruns:
                break
            rec.runner = {**runner, "infra_try": len(outages) + 1, "infra_source": "store"}
            rec.write(infra_dir)
            outages.append(rec.run_id)
            bump("infra_tries")
            say(f"warning: [{name}] store unavailable while executing a probe query; waiting for it before "
                f"executing the same query again ({len(outages)} of {args.infra_reruns})")
            answered, _ = wait_for_store(sparql_factory(args), limit_s=args.infra_retry_s, sleep=sleep)
            if not answered:
                bump("store_lost")
                say(f"error: the store did not come back within {args.infra_retry_s:g} s; the kept replies wait "
                    f"for a resumed run")
                return finish(args, out, counts, say, terminal, stopped=True)
        if rec.outcome["failure_type"] == "harness":  # our defect: set aside, the reply kept for a run after the fix
            set_aside(rec, harness_dir)
            bump("harness")
            say(f"error: [{name}] {probe_id} rep {rep}: failed:harness ({rec.outcome['error_message'][:200]}); set "
                f"aside in harness/, its reply kept: fix the harness and run again")
            continue
        infra_ids = list(runner.get("provider_infra_run_ids") or []) + outages
        rec.runner = {**runner, "infra_tries": len(infra_ids), "infra_try_run_ids": infra_ids}
        if rec.outcome["failure_type"] == "infrastructure":
            bump("infra_final")
        rec.write(records_dir)
        done.add(key_of(attempt))
        p.unlink()
        bump("recorded")
        exact = (rec.score or {}).get("exact")
        say(f"[{name}] {probe_id} {qid} rep {rep}: {rec.outcome['label']} exact={exact}")
    return finish(args, out, counts, say, terminal, stopped=bool(counts["stopped"]))


SETASIDE_SCHEMA = "setaside/1"  # scripts/relabel_harness_failure.py's decision format


def set_aside(rec: ProbeRecord, harness_dir: Path) -> None:
    """Write a failed:harness record to harness/ and its decision beside it (last, as the agent's set-aside
    does): it is replayed by the gate, counted by no rate, and the attempt runs again after the fix."""
    path = rec.write(harness_dir)
    decision = {"schema": SETASIDE_SCHEMA, "run_id": rec.run_id, "record": path.name, "question_id": rec.question_id,
                "model": rec.model, "repetition": rec.repetition, "probe": rec.prompt["id"], "label": "failed:harness",
                "original_label": rec.outcome["label"], "original_failure_type": rec.outcome["failure_type"],
                "defect": rec.outcome["error_message"], "ruling": "automatic: an unexpected exception in the harness "
                "while executing or reading a probe query (closed-book review of 05649ea)", "decided_at": utc_now()}
    write_text_atomic(harness_dir, f"{rec.run_id}.setaside.json",
                      json.dumps(decision, ensure_ascii=False, indent=1, sort_keys=True))


def _seconds_between(start: str | None, end: str | None) -> float:
    try:
        return max(0.0, (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds())
    except (TypeError, ValueError):
        return 0.0


def _null_sparql(args: argparse.Namespace) -> Any:
    """For a record of a failed call to probe (b), which runs no query: a client that is never called."""
    class _Unused:
        def settings(self) -> dict:
            return {"endpoint": _public_url(args.endpoint) if args.endpoint else None,
                    "timeout_s": args.sparql_timeout, "row_cap": args.row_cap}

        def query(self, query: str):
            raise AssertionError("no query runs for a failed call")

    return _Unused()


def _guarded(target: Callable[[str], None], name: str, say: Callable[[str], None],
             bump: Callable[..., None]) -> Callable[[], None]:
    def run_target() -> None:
        try:
            target(name)
        except Refused as exc:
            bump("errors")
            say(f"error: [{name}] {exc}")
        except Exception as exc:  # a model's thread must report, not vanish
            bump("errors")
            say(f"error: [{name}] {type(exc).__name__}: {exc}")

    return run_target


def finish(args, out: Path, counts: dict, say, terminal, *, stopped: bool) -> int:
    check = verify_batch(out)
    write_check(out, check)
    if check["failures"]:
        say(f"error: replay check failed for {len(check['failures'])} of {check['n_records']} records ({CHECK_FILE})")
    else:
        say(f"replay check: {check['n_ok']} of {check['n_records']} records load, write back and replay ({CHECK_FILE})")
    line = (f"done: {counts['recorded']} recorded, {counts['skipped']} already recorded, {counts['aborted']} aborted, "
            f"{counts['errors']} errors, {counts['harness']} set aside as failed:harness, "
            f"{counts['infra_tries']} infrastructure tries, {counts['infra_final']} "
            f"attempts failed infrastructure on every try; replay check {check['n_ok']}/{check['n_records']}")
    say(line)
    if terminal is not say:
        terminal(line)
    if stopped or counts["store_lost"]:
        return 3
    if check["failures"]:
        return 5
    if counts["errors"]:
        return 1
    if counts["harness"]:
        return 6
    return 4 if counts["infra_final"] else 0


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
