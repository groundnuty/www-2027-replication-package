"""H6 "within a run" (the design log; docs/h6-within-spec.md is the contract; the step refuses a spec file whose
sha256 is not the one it was started with).

    python -I scripts/h6_within_run.py --spec-sha256 H --config data/analysis/prereg-config.json --config-sha256 H \\
        --results-root DIR --out DIR/h6-within.json \\
        --agent sonnet=claude-sonnet-5-5 --agent haiku=claude-haiku-5-5 ... \\
        --cohorts sonnet=curated,census,generated-1 --cohorts haiku=curated,census ... \\
        --graph uniprot=TREE --matrices uniprot=LIST-FILE [--runrec uniprot=TREE/runrec]

``LIST-FILE`` names one matrix file per line, relative to the tree (``out/<shard>/matrix.jsonl.gz``). The step reads
those files read-only, hashes each before it is parsed and again after the whole analysis (a file that changed is
refused), checks each against the sealed ``runrec/<shard>.json`` when ``--runrec`` is given, and writes ONE results file
(mode 600 in a mode 700 directory under ``--results-root``).

Blinding (the design log): standard output carries only counts of rows, file names and sha256s; a refusal names
counts, file names and the command line's own labels, never a value read from a row; an unexpected error names only its
class. Every number goes to the results file.

The helpers of the registered H6 step (``h6_analysis.py``, the design log, unchanged) and the frozen metric code
(``harness.analysis.metrics``) are imported, not copied. Every computation below is a pure function over arrays
(``collect``, ``recall_quantity``, ``silent_quantities``, ``analyse_graph``); ``main`` is a thin wrapper.
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import io
import json
import re
import sys
import zlib
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import h6_analysis as h  # noqa: E402  (the registered H6 step: its helpers are used unchanged)
from harness.analysis.metrics import METRICS, PROPORTION, metric_arrays, metric_of  # noqa: E402
from harness.analysis.prereg import Clusters, PreregConfig  # noqa: E402
from harness.checkloop.accounting import CELLS  # noqa: E402
from harness.checkloop.matrix import VARIANTS  # noqa: E402

Refused = h.Refused
Quantity = h.Quantity

SCHEMA = "h6-within-run/1"
VARIANT = "primary"  # the frozen online setting; the design log names no other variant
NATURAL = "none"  # the cohort with no ontology in the prompt (the reference)
PROMPT_COHORTS = ("curated", "census", "generated-1")  # the prompt cohorts of the Tier 1 grid, in this order
# the checker whose source is the prompt cohort's own ontology: the "both" cell is the replay row of this arm. The prompt
# the prompt index of each graph (data/prompts/<graph>/index.json) names these source files: curated.json.gz, census-t5.json, generated-1.json
OWN_SOURCE = {"curated": "curated", "census": "census-t5", "generated-1": "generated-1"}
SOURCES = tuple(OWN_SOURCE.values())  # the three checkers of the design log, in this order
PAIRS = tuple((a, b) for i, a in enumerate(SOURCES) for b in SOURCES[i + 1:])
# UniProt Tier 1: the models that ran the generated-1 prompt cohort (the 3 x 3 of the grid; science, 24 Oct). The panel itself
# is the `panel` of the preregistration config (seven models), whose hash the step pins.
UNIPROT_GENERATED_MODELS = frozenset({"claude-sonnet-5-5", "Qwen/Qwen3.8-27B"})
COMMON_COHORTS = ("curated", "census")  # the cohorts every model ran: the headline Delta_s is over these (science, 24 Oct)
COMMON_SOURCES = tuple(OWN_SOURCE[p] for p in COMMON_COHORTS)
SPLIT = h.SPLIT
MIN_WRONG = h.MIN_WRONG
N_REPS = PreregConfig().n_reps  # repetitions of a question in every cell (prereg 4.4): 210 test questions x 3 = 630

COL, C_COLS, W_COLS, ROW_COLUMN, STATUSES = h.COL, h.C_COLS, h.W_COLS, h.ROW_COLUMN, h.STATUSES
NAME = h.NAME
# a matrix file of a tree: out/<shard>/matrix.jsonl.gz, optionally under one directory of the tree (an OUT's own replay tree)
MATRIX_PATH = re.compile(r"(?:[A-Za-z0-9][A-Za-z0-9_.-]*/)?out/[A-Za-z0-9][A-Za-z0-9_.-]*/matrix\.jsonl\.gz")

CODE_FILES = ("scripts/h6_within_run.py", "scripts/h6_analysis.py", "harness/analysis/stats.py",
              "harness/analysis/prereg.py", "harness/analysis/metrics.py", "harness/checkloop/matrix.py")


def pair_label(pair: tuple[str, str]) -> str:
    return f"{pair[0]}|{pair[1]}"


# --- the command line's parts: agents, cohort sets, matrix lists ---------------------------------------------------------


def parse_cohorts(items: Sequence[str], agents: Mapping[str, str]) -> dict[str, tuple[str, ...]]:
    """``NAME=curated,census[,generated-1]`` per agent: the prompt cohorts the agent ran, in the fixed order. Every
    agent needs an entry with at least ``curated`` and ``census``; anything else is refused."""
    out: dict[str, tuple[str, ...]] = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep or name not in agents or name in out:
            raise Refused("--cohorts must be NAME=COHORT,COHORT[,COHORT] once for each listed agent")
        got = [c for c in value.split(",") if c]
        if len(set(got)) != len(got) or any(c not in PROMPT_COHORTS for c in got):
            raise Refused(f"--cohorts {name}: the prompt cohorts are {', '.join(PROMPT_COHORTS)}, each once")
        if "curated" not in got or "census" not in got:
            raise Refused(f"--cohorts {name}: every agent ran at least curated and census")
        out[name] = tuple(c for c in PROMPT_COHORTS if c in got)
    missing = [a for a in agents if a not in out]
    if missing:
        raise Refused(f"--cohorts is missing for the agents {missing}")
    return out


def read_matrix_list(path: Path, tree: Path, label: str) -> tuple[list[tuple[str, Path]], str]:
    """(relative name, absolute file name) of every matrix file the list names, and the list file's sha256. Each line is
    ``out/<shard>/matrix.jsonl.gz`` (optionally under one directory of the tree), without ``..``, named once; the file must
    be a regular file that, with every symbolic link resolved, lies under the tree."""
    try:
        data = Path(path).read_bytes()
        lines = [x for x in data.decode("utf-8").split("\n") if x.strip()]
        root = Path(tree).resolve(strict=True)
    except (OSError, UnicodeDecodeError):
        raise Refused(f"{label}: the matrix list or the tree cannot be read") from None
    if not lines or len(set(lines)) != len(lines):
        raise Refused(f"{label}: the matrix list must name at least one file, once each")
    files = []
    for line in lines:
        if not MATRIX_PATH.fullmatch(line):
            raise Refused(f"{label}: a line of the matrix list is not out/<shard>/matrix.jsonl.gz under the tree")
        full = Path(tree) / line
        try:
            real = full.resolve(strict=True)
        except OSError:
            raise Refused(f"{label}: {line} cannot be found") from None
        if root not in real.parents or not real.is_file():
            raise Refused(f"{label}: {line} is not a regular file under the tree")
        files.append((line, full))
    return files, hashlib.sha256(data).hexdigest()


# --- reading: the bytes are hashed, then parsed -------------------------------------------------------------------------


def read_bytes(path: Path, label: str) -> bytes:
    try:
        return Path(path).read_bytes()
    except OSError:
        raise Refused(f"{label}: cannot be read") from None


def rows_of(data: bytes, label: str) -> Iterator[dict]:
    """The rows of one gzip JSON-lines file (already read into memory). Nothing of a row is put in a message."""
    try:
        with gzip.open(io.BytesIO(data), "rt", encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    raise Refused(f"{label}: line {n} is not JSON") from None
                if not isinstance(row, dict):
                    raise Refused(f"{label}: line {n} is not a JSON object")
                yield row
    except (OSError, EOFError, ValueError, zlib.error) as e:
        raise Refused(f"{label}: unreadable ({type(e).__name__})") from None


# --- collecting the cells of one tree -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Tables:
    """What the analysis needs of a validated tree: per (agent, cohort, source) the six cell counts per base question and
    per (agent, cohort) the attempts per base question (every attempt, including those with no output)."""

    index: object
    cells: Mapping[tuple[str, str, str], np.ndarray]  # (agent, cohort, source) -> (n_questions, 6), CELLS order
    att: Mapping[tuple[str, str], np.ndarray]  # (agent, cohort) -> (n_questions,) attempts per question
    info: Mapping[tuple[str, str], dict]  # (agent, cohort) -> n_attempts, n_no_output, n_wrong, n_correct
    rows: int
    hashes: Mapping[str, str]  # matrix file -> sha256 of the bytes that were parsed


def collect(label: str, datas: Sequence[tuple[str, bytes]], index, agents: Mapping[str, str],
            cohorts: Mapping[str, Sequence[str]], expected_graph: str | None = None,
            reps: int | None = N_REPS) -> Tables:
    """Read every matrix of a tree (as bytes), refuse what the specification's section 3 refuses, and count the six cells
    per base question of the three checkers at the primary variant, for every (agent, cohort)."""
    agent_of = {model: name for name, model in agents.items()}
    expected = {a: {NATURAL, *cohorts[a]} for a in agents}
    allowed_cohorts = set().union(*expected.values())
    nq = len(index)
    viol = h.Violations()
    runs: dict[str, tuple] = {}  # run id -> (agent, cohort, base question, output, correct)
    seen: set[tuple[str, str]] = set()  # (run id, source)
    covered: dict[tuple[str, str, str], set[str]] = defaultdict(set)  # (agent, cohort, source) -> run ids
    cells: dict[tuple[str, str, str], np.ndarray] = {}
    att: dict[tuple[str, str], np.ndarray] = {}
    n_rows = 0
    hashes: dict[str, str] = {}
    for name, data in datas:
        hashes[name] = hashlib.sha256(data).hexdigest()
        for row in rows_of(data, f"{label} {name}"):
            n_rows += 1
            ok = True
            cohort, model, arm, variant, rid = (row.get(k) for k in ("cohort", "model", "arm", "variant", "run_id"))
            agent = agent_of.get(model) if isinstance(model, str) else None
            if not (isinstance(cohort, str) and cohort in allowed_cohorts):
                viol.add("cohort outside the expected cohorts", name)
                ok = False
            if agent is None:
                viol.add("model that is not a listed agent", name)
                ok = False
            elif isinstance(cohort, str) and cohort in allowed_cohorts and cohort not in expected[agent]:
                viol.add("cohort the agent did not run", name)
                ok = False
            if row.get("split") != SPLIT:
                viol.add(f"split other than {SPLIT}", name)
                ok = False
            if expected_graph is not None and row.get("graph") != expected_graph:
                viol.add("graph other than the tree's", name)
                ok = False
            if not (isinstance(variant, str) and variant in VARIANTS):
                viol.add("variant outside the four", name)
                ok = False
            if not (isinstance(arm, str) and arm):
                viol.add("arm that is not a non-empty string", name)
                ok = False
            if not (isinstance(rid, str) and rid):
                viol.add("run id that is not a non-empty string", name)
                ok = False
            if not ok or arm not in SOURCES or variant != VARIANT:
                continue  # the other arms and variants of the matrix are not read
            output, status, correct = row.get("output"), row.get("status"), row.get("correct")
            if not isinstance(output, bool):
                viol.add("output that is not a boolean", name)
                ok = False
            elif output:
                if not (isinstance(status, str) and status in STATUSES):
                    viol.add("output row with a status outside the four", name)
                    ok = False
                if not isinstance(correct, bool):
                    viol.add("output row with correct that is not a boolean", name)
                    ok = False
            else:
                if status is not None:
                    viol.add("non-output row with a status", name)
                    ok = False
                if not (correct is None or correct is False):
                    viol.add("non-output row with a correct that is neither null nor false", name)
                    ok = False
            base = row.get("base_question_id")
            if not (isinstance(base, str) and base in index):
                viol.add("base question missing from the index", name)
                ok = False
            if not ok:
                continue
            info = (agent, cohort, base, output, correct if output else None)
            first = rid not in runs
            if runs.setdefault(rid, info) != info:
                viol.add("run id whose agent, cohort, base question, output or correct differs between rows", name)
                continue
            if (rid, arm) in seen:
                viol.add("run id and source seen twice", name)
                continue
            seen.add((rid, arm))
            covered[(agent, cohort, arm)].add(rid)
            if first:
                a = att.get((agent, cohort))
                if a is None:
                    a = att[(agent, cohort)] = np.zeros(nq, dtype=np.int64)
                a[index.position(base)] += 1
            cell = cells.get((agent, cohort, arm))
            if cell is None:
                cell = cells[(agent, cohort, arm)] = np.zeros((nq, len(CELLS)), dtype=np.int64)
            if output:
                cell[index.position(base), ROW_COLUMN[(correct, status)]] += 1
    viol.raise_if_any(label)

    by_cell: dict[tuple[str, str], set[str]] = defaultdict(set)
    for rid, (agent, cohort, *_rest) in runs.items():
        by_cell[(agent, cohort)].add(rid)
    for a in agents:
        for p in (NATURAL, *cohorts[a]):
            if (a, p) not in by_cell:
                raise Refused(f"{label}: agent {a} has no rows in its expected cohort {p}")
    off = [(a, p, s) for (a, p), ids in sorted(by_cell.items()) for s in SOURCES if covered.get((a, p, s)) != ids]
    if off:
        raise Refused(f"{label}: {len(off)} (agent, cohort, source) groups do not hold exactly the cohort's attempts")
    if reps is not None:
        for (a, p), ids in sorted(by_cell.items()):
            if len(ids) != reps * nq or not np.all(att[(a, p)] == reps):
                raise Refused(f"{label}: an (agent, cohort) cell does not hold {reps} attempts of each of the "
                              f"{nq} test questions ({len(ids)} attempts found)")
    summary: dict[tuple[str, str], dict] = {}
    for (a, p), ids in by_cell.items():
        mine = [runs[r] for r in ids]
        summary[(a, p)] = {"n_attempts": len(mine), "n_no_output": sum(1 for r in mine if not r[3]),
                           "n_wrong": sum(1 for r in mine if r[3] and not r[4]),
                           "n_correct": sum(1 for r in mine if r[3] and r[4])}
    for (a, p), s in summary.items():
        for src in SOURCES:
            cells.setdefault((a, p, src), np.zeros((nq, len(CELLS)), dtype=np.int64))
        ref = cells[(a, p, SOURCES[0])]
        for src in SOURCES:  # every checker sees the same wrong and correct outputs, question by question (checked)
            cell = cells[(a, p, src)]
            if (int(cell[:, W_COLS].sum()), int(cell[:, C_COLS].sum())) != (s["n_wrong"], s["n_correct"]):
                raise Refused(f"{label}: the cells of a checker do not add up to the cohort's wrong and correct outputs")
            if not (np.array_equal(cell[:, W_COLS].sum(axis=1), ref[:, W_COLS].sum(axis=1))
                    and np.array_equal(cell[:, C_COLS].sum(axis=1), ref[:, C_COLS].sum(axis=1))):
                raise Refused(f"{label}: two checkers disagree on a question's wrong or correct outputs")
    return Tables(index, cells, att, summary, n_rows, hashes)


# --- quantities: pure functions over arrays ------------------------------------------------------------------------------


def recall_quantity(clusters: Clusters, cell: np.ndarray) -> Quantity:
    """R = W_F / W, the ratio of pooled counts (per question numerator and denominator summed over its attempts)."""
    c = np.asarray(cell, dtype=np.float64)
    return h.ratio_quantity(clusters, c[:, COL["W_F"]], c[:, W_COLS].sum(axis=1))


def silent_quantities(clusters: Clusters, cell: np.ndarray, att: np.ndarray) -> dict[str, Quantity]:
    """The unconditional silent-error rate of one (cohort, checker) cell, from the frozen code's quantities (the design log): the
    frozen ``silent_failure`` (wrong outputs per attempt, ``harness.analysis.metrics.METRICS``, equal weight per question)
    minus the checker's catches per attempt (flagged wrong outputs per attempt, the same weighting). Wrong and not flagged
    is a substantive pass, a vacuous pass or unknown. Both terms are returned with the difference."""
    c = np.asarray(cell, dtype=np.float64)
    n = np.asarray(att, dtype=np.float64)
    vectors = {"n": n, "att": n, "W": c[:, W_COLS].sum(axis=1), "W_F": c[:, COL["W_F"]]}
    sf = h.ratio_quantity(clusters, *metric_of(vectors, "silent_failure"))
    catches = h.ratio_quantity(clusters, *metric_arrays(vectors, "W_F", "att", PROPORTION))
    return {"silent_failure": sf, "catches_per_attempt": catches, "silent_error": sf - catches}


def flagged(n_wrong: Mapping[str, int], keys: Sequence[str]) -> dict:
    """The < 20 rule: the entries (agents or cohorts) whose cell holds fewer than 20 wrong outputs."""
    low = [k for k in keys if n_wrong[k] < MIN_WRONG]
    return {"uninformative": bool(low), "uninformative_cells": low}


def with_flag(q: Quantity, flag: Mapping) -> dict:
    return {**q.summary(), **flag}


# --- one graph's results -------------------------------------------------------------------------------------------------


def own_cohort(source: str) -> str:
    return next(p for p, s in OWN_SOURCE.items() if s == source)


def delta_parts(R: Mapping[tuple[str, str], Quantity], low: Mapping[str, bool], source: str,
                cohort_set: Sequence[str]) -> dict | None:
    """Delta_s for one agent: R(s | p = s) minus the unweighted mean of R(s | p) over the OTHER prompt cohorts p of
    ``cohort_set``. None when the cohort of s is not in the set or no other cohort is. ``R[(p, s)]`` is the agent's
    recall of checker s on cohort p; ``low[p]`` is the agent's < 20 flag of the cohort."""
    own = own_cohort(source)
    others = [p for p in cohort_set if p != own]
    if own not in cohort_set or not others:
        return None
    own_q = R[(own, source)]
    others_q = h.mean_quantity([R[(p, source)] for p in others])
    used = [own, *others]
    return {"own": own_q, "others": others_q, "delta": own_q - others_q, "cohorts": used,
            "uninformative_cells": [p for p in used if low[p]]}


def pool_entry(items: Sequence[tuple[str, Quantity, Sequence, int]]) -> dict:
    """Pooled over the panel with equal weight per model (science, 24 Oct): the unweighted mean of the models' estimates on
    the graph's one K. ``items`` are (model, estimate, below-20 cells it uses, wrong outputs in the cells it uses). A model
    whose estimate is undefined (no wrong output in a cell it uses) is excluded from the mean and listed; the models that
    enter are listed. A resample in which an entering model's estimate is undefined is undefined for the pool (counted in
    ``undefined_share``). The pool is flagged uninformative when ANY model of the cell (entering or excluded) has a cell
    below 20 wrong outputs, with those models named; the wrong outputs summed over all the models are given beside it."""
    used = [i for i in items if i[1].point is not None]
    if used:
        q = h.mean_quantity([i[1] for i in used])
    else:
        q = Quantity(None, np.full(len(items[0][1].draws), np.nan))
    low = [i[0] for i in items if i[2]]
    return {**q.summary(), "models": [i[0] for i in used], "n_models": len(used),
            "excluded_models": [i[0] for i in items if i[1].point is None], "uninformative": bool(low),
            "uninformative_models": low, "n_wrong_total": sum(i[3] for i in items)}


def delta_block(parts: dict) -> dict:
    flag = {"uninformative": bool(parts["uninformative_cells"]), "uninformative_cells": parts["uninformative_cells"]}
    return {"cohorts": parts["cohorts"], "own": parts["own"].summary(), "others": parts["others"].summary(),
            "delta": with_flag(parts["delta"], flag)}


def analyse_graph(tables: Tables, clusters: Clusters, agents: Mapping[str, str],
                  cohorts: Mapping[str, Sequence[str]]) -> dict:
    """The ``agents``, ``cells``, ``delta_headline``, ``delta_three_cohort``, ``checker_differences`` and ``pooled`` blocks
    of one graph (spec section 2). The HEADLINE Delta_s is over the cohorts curated and census, for every model and
    pooled over all of them; the three-cohort Delta_s (with generated-1) is reported for the models that ran that cohort,
    per model and pooled over those models (science, 24 Oct)."""
    names = list(agents)
    n_wrong = {(a, p): tables.info[(a, p)]["n_wrong"] for (a, p) in tables.info}
    all_cohorts = {a: (NATURAL, *cohorts[a]) for a in names}
    R: dict[tuple[str, str, str], Quantity] = {}
    SF: dict[tuple[str, str], Quantity] = {}
    CATCH: dict[tuple[str, str, str], Quantity] = {}
    SIL: dict[tuple[str, str, str], Quantity] = {}
    out: dict = {"agents": {}, "cells": {}, "delta_headline": {}, "delta_three_cohort": {}, "checker_differences": {}}
    for a in names:
        out["agents"][a] = {"model": agents[a], "cohorts": {}}
        out["cells"][a] = {}
        for p in all_cohorts[a]:
            info = tables.info[(a, p)]
            first = tables.cells[(a, p, SOURCES[0])]
            wrong_clusters = int(np.count_nonzero(clusters.sum(first[:, W_COLS].sum(axis=1))))
            flag = flagged({p: n_wrong[(a, p)]}, [p])
            out["agents"][a]["cohorts"][p] = {**info, "n_clusters_with_wrong": wrong_clusters,
                                              "uninformative": flag["uninformative"]}
            block = {}
            for s in SOURCES:
                cell = tables.cells[(a, p, s)]
                R[(a, p, s)] = recall_quantity(clusters, cell)
                parts = silent_quantities(clusters, cell, tables.att[(a, p)])
                if (a, p) in SF and not (parts["silent_failure"].point == SF[(a, p)].point and np.array_equal(
                        parts["silent_failure"].draws, SF[(a, p)].draws, equal_nan=True)):
                    raise Refused("the frozen silent_failure differs between the checkers of one cohort")
                SF[(a, p)], CATCH[(a, p, s)], SIL[(a, p, s)] = (parts["silent_failure"], parts["catches_per_attempt"],
                                                                parts["silent_error"])
                block[s] = {"counts": {c: int(x) for c, x in zip(CELLS, cell.sum(axis=0))},
                            "recall": with_flag(R[(a, p, s)], flag),
                            "catches_per_attempt": with_flag(CATCH[(a, p, s)], flag),
                            "silent_error": with_flag(SIL[(a, p, s)], flag)}
            out["cells"][a][p] = {"silent_failure": with_flag(SF[(a, p)], flag), "sources": block}
    # Delta_s per agent (natural excluded: it holds "some ontology in the prompt" constant)
    deltas: dict[tuple[str, str, str], dict] = {}  # (reading, agent, source) -> parts
    for a in names:
        low = {p: n_wrong[(a, p)] < MIN_WRONG for p in cohorts[a]}
        out["delta_headline"][a], out["delta_three_cohort"][a] = {}, {}
        for reading, cohort_set, srcs, key in (("headline", COMMON_COHORTS, COMMON_SOURCES, "delta_headline"),
                                               ("three", PROMPT_COHORTS, SOURCES, "delta_three_cohort")):
            if not set(cohort_set) <= set(cohorts[a]):
                out[key].pop(a)
                continue
            for s in srcs:
                deltas[(reading, a, s)] = parts = delta_parts({(p, s): R[(a, p, s)] for p in cohort_set}, low, s, cohort_set)
                out[key][a][s] = delta_block(parts)
    # the difference between two checkers on the same outputs, for each cohort (natural as a reference)
    for a in names:
        out["checker_differences"][a] = {}
        for p in all_cohorts[a]:
            flag = flagged({p: n_wrong[(a, p)]}, [p])
            out["checker_differences"][a][p] = {
                pair_label(pr): with_flag(R[(a, p, pr[0])] - R[(a, p, pr[1])], flag) for pr in PAIRS}
    # pooled over the panel, equal weight per model, over the models that ran the cell
    lowcell = lambda a, p: [p] if n_wrong[(a, p)] < MIN_WRONG else []  # noqa: E731
    pooled: dict = {"models": list(names), "cells": {}, "checker_differences": {}, "delta_headline": {},
                    "delta_three_cohort": {}}
    for p in (NATURAL, *PROMPT_COHORTS):
        who = [a for a in names if p in all_cohorts[a]]
        if not who:
            continue
        pooled["cells"][p] = {
            "silent_failure": pool_entry([(a, SF[(a, p)], lowcell(a, p), n_wrong[(a, p)]) for a in who]),
            "sources": {s: {"recall": pool_entry([(a, R[(a, p, s)], lowcell(a, p), n_wrong[(a, p)]) for a in who]),
                            "catches_per_attempt": pool_entry([(a, CATCH[(a, p, s)], lowcell(a, p), n_wrong[(a, p)]) for a in who]),
                            "silent_error": pool_entry([(a, SIL[(a, p, s)], lowcell(a, p), n_wrong[(a, p)]) for a in who])}
                        for s in SOURCES}}
        pooled["checker_differences"][p] = {
            pair_label(pr): pool_entry([(a, R[(a, p, pr[0])] - R[(a, p, pr[1])], lowcell(a, p), n_wrong[(a, p)]) for a in who])
            for pr in PAIRS}
    for reading, key, srcs in (("headline", "delta_headline", COMMON_SOURCES), ("three", "delta_three_cohort", SOURCES)):
        for s in srcs:
            who = [a for a in names if (reading, a, s) in deltas]
            if not who:
                continue
            pooled[key][s] = pool_entry([
                (a, deltas[(reading, a, s)]["delta"], deltas[(reading, a, s)]["uninformative_cells"],
                 sum(n_wrong[(a, p)] for p in deltas[(reading, a, s)]["cohorts"])) for a in who])
    out["pooled"] = pooled
    return out

# --- the run -------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphInput:
    name: str
    tree: Path
    matrices: tuple[tuple[str, Path], ...]  # (name relative to the tree, absolute file name)
    list_sha256: str
    pins: h.GraphPins
    runrec: Path | None = None  # the sealed runrec/<shard>.json directory of the tree, when the matrices were sealed there


def check_runrec(label: str, runrec: Path, name: str, sha: str) -> None:
    """The matrix's sha256 must be the one the replay sealed (``runrec/<shard>.json``, key ``matrix.jsonl.gz``) and the
    shard's rc 0. Only those two fields are read."""
    shard = Path(name).parent.name
    try:
        rec = json.loads((Path(runrec) / f"{shard}.json").read_bytes())
        sealed, rc = rec["matrix.jsonl.gz"], rec["rc"]
    except (OSError, ValueError, KeyError, TypeError):
        raise Refused(f"{label}: the sealed record of {name} cannot be read") from None
    if sealed != sha or rc != 0:
        raise Refused(f"{label}: {name} is not the sealed matrix (sha256 or rc differs from its runrec)")


def check_uniprot_panel(config: Path, agents: Mapping[str, str], cohorts: Mapping[str, Sequence[str]]) -> None:
    """The silent-omission guard of the UniProt pools: the listed agents must be exactly the seven models of the
    preregistration config's panel, and the agents with the generated-1 cohort exactly Sonnet and Qwen3.8 (the models of
    the grid's three prompt cohorts). A matrix list and an agent list that both leave a model out would otherwise give a
    plausible pool of fewer models."""
    try:
        panel = json.loads(Path(config).read_bytes())["panel"]
        panel = set(panel) if isinstance(panel, list) and all(isinstance(x, str) for x in panel) else None
    except (OSError, ValueError, KeyError, TypeError):
        panel = None
    if not panel:
        raise Refused("config: its panel cannot be read")
    if set(agents.values()) != panel:
        raise Refused(f"uniprot: the {len(agents)} listed agents are not the config's panel of {len(panel)} models")
    generated = {agents[a] for a in agents if "generated-1" in cohorts[a]}
    if generated != UNIPROT_GENERATED_MODELS:
        raise Refused(f"uniprot: the {len(generated)} agents with the generated-1 cohort are not "
                      f"Sonnet and Qwen3.8")


def analyse(graphs: Sequence[GraphInput], agents: Mapping[str, str], cohorts: Mapping[str, Sequence[str]], *,
            config: Path, config_sha256: str, spec: Path, spec_sha256: str, created_utc: str | None = None,
            reps: int | None = N_REPS, real_graph: bool = False) -> dict:
    """The whole analysis; returns the results dictionary (``h.write_results`` puts it in a file)."""
    if not agents or len(set(agents.values())) != len(agents):
        raise Refused("the panel must list at least one agent, each with its own model")
    names = [g.name for g in graphs]
    if not graphs or len(set(names)) != len(names) or any(not NAME.fullmatch(n) or n == "meta" for n in names):
        raise Refused("graph names must be distinct labels and not 'meta'")
    hashes: dict = {"spec": h.check_pin("spec", spec, spec_sha256)}
    cfg, hashes["config"] = h.load_config(config, config_sha256)
    hashes["code"] = {rel: h.sha256_file(ROOT / rel, rel) for rel in CODE_FILES}
    results: dict = {}
    rows_read: dict[str, int] = {}
    not_listed: dict[str, int] = {}
    cluster_info: dict[str, dict] = {}
    runrec_checked: dict[str, int] = {}
    hashes["graphs"] = {}
    for gi in graphs:
        if real_graph and gi.name == "uniprot":
            check_uniprot_panel(config, agents, cohorts)
        index = h.build_index(gi.pins, gi.name)
        cmap = h.load_cluster_map(gi.pins.cluster_map, gi.pins.cluster_map_sha256, gi.name)
        datas = [(name, read_bytes(path, f"{gi.name} {name}")) for name, path in gi.matrices]
        tables = collect(gi.name, datas, index, agents, cohorts, h.REAL_GRAPH_DIRS.get(gi.name) if real_graph else None, reps)
        if gi.runrec is not None:
            for name, sha in tables.hashes.items():
                check_runrec(gi.name, gi.runrec, name, sha)
            runrec_checked[gi.name] = len(tables.hashes)
        listed = {Path(name).parent.name for name, _ in gi.matrices}
        out_dir = gi.tree / "out"
        not_listed[gi.name] = (sum(1 for p in out_dir.iterdir() if p.is_dir() and p.name not in listed)
                               if out_dir.is_dir() else 0)
        clusters = h.resample(cmap, index, cfg, f"{gi.name} cluster map")  # ONE K per graph
        results[gi.name] = analyse_graph(tables, clusters, agents, cohorts)
        results[gi.name]["design"] = {"expected_attempts_per_cell": None if reps is None else reps * len(index)}
        rows_read[gi.name] = tables.rows
        cluster_info[gi.name] = clusters.summary()
        hashes["graphs"][gi.name] = {"questions": gi.pins.questions_sha256, "cluster_map": gi.pins.cluster_map_sha256,
                                     "tier_questions": gi.pins.tier_questions_sha256, "matrix_list": gi.list_sha256,
                                     "matrices": dict(tables.hashes), "matrices_after": {}}
        for name, path in gi.matrices:  # the files must be exactly the bytes that were parsed (the design log "Process")
            after = h.sha256_file(path, f"{gi.name} {name}")
            hashes["graphs"][gi.name]["matrices_after"][name] = after
            if after != tables.hashes[name]:
                raise Refused(f"{gi.name}: {name} changed while it was read")
    results["meta"] = {
        "schema": SCHEMA, "created_utc": created_utc or h.utc_now(), "seed": cfg.seed, "n_boot": cfg.n_boot,
        "scheme": "unstratified", "level": h.LEVEL, "min_wrong": MIN_WRONG, "variant": VARIANT,
        "agents": dict(agents), "cohorts": {a: list(c) for a, c in cohorts.items()},
        "own_source": dict(OWN_SOURCE), "sources": list(SOURCES), "pairs": [pair_label(p) for p in PAIRS],
        "silent_error_rule": ("wrong and not flagged per attempt = the frozen silent_failure (W / att, equal weight per "
                              "question) minus the checker's catches per attempt (W_F / att, the same weighting)"),
        "hashes": hashes, "rows_read": rows_read, "matrix_dirs_not_listed": not_listed, "clusters": cluster_info,
        "runrec_checked": runrec_checked}
    return results


def flatten_hashes(meta: Mapping) -> list[tuple[str, str]]:
    """(file label, sha256) pairs of everything the run hashed, for the standard output."""
    hh = meta["hashes"]
    out = [("spec", hh["spec"]), ("config", hh["config"])]
    out += [(f"code/{rel}", sha) for rel, sha in hh["code"].items()]
    for graph, gh in hh["graphs"].items():
        out += [(f"{graph}/questions", gh["questions"]), (f"{graph}/cluster-map", gh["cluster_map"]),
                (f"{graph}/matrix-list", gh["matrix_list"])]
        if gh.get("tier_questions"):
            out.append((f"{graph}/tier-questions", gh["tier_questions"]))
        out += [(f"{graph}/{rel}", sha) for rel, sha in gh["matrices"].items()]
    return out


def print_summary(meta: Mapping, out: Path, out_sha256: str, file=None) -> None:
    file = sys.stdout if file is None else file
    for graph, n in sorted(meta["rows_read"].items()):
        print(f"rows {graph} {n}", file=file)
    for graph, n in sorted(meta["runrec_checked"].items()):
        print(f"runrec-checked {graph} {n}", file=file)
    for label, sha in flatten_hashes(meta):
        print(f"file {label} {sha}", file=file)
    print(f"output {out} {out_sha256}", file=file)


# --- the command line -----------------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="h6_within_run.py", description=__doc__.split("\n\n")[0])
    p.add_argument("--spec", type=Path, default=ROOT / "docs/h6-within-spec.md")
    p.add_argument("--spec-sha256", required=True)
    p.add_argument("--config", type=Path, default=ROOT / "data/analysis/prereg-config.json")
    p.add_argument("--config-sha256", required=True)
    p.add_argument("--results-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--agent", action="append", default=[], metavar="NAME=MODEL-ID")
    p.add_argument("--cohorts", action="append", default=[], metavar="NAME=COHORT,COHORT[,COHORT]")
    p.add_argument("--graph", action="append", default=[], metavar="GRAPH=TREE-DIR")
    p.add_argument("--matrices", action="append", default=[], metavar="GRAPH=LIST-FILE")
    p.add_argument("--runrec", action="append", default=[], metavar="GRAPH=RUNREC-DIR")
    p.add_argument("--questions", action="append", default=[], metavar="GRAPH=FILE")
    p.add_argument("--cluster-map", action="append", default=[], metavar="GRAPH=FILE")
    p.add_argument("--tier-questions", action="append", default=[], metavar="GRAPH=FILE")
    return p


def graphs_from_args(args: argparse.Namespace, pins: Mapping[str, h.GraphPins]) -> list[GraphInput]:
    trees = h._by_graph(args.graph, "--graph")
    lists = h._by_graph(args.matrices, "--matrices")
    runrecs = h._by_graph(args.runrec, "--runrec")
    where = {field: h._by_graph(getattr(args, field), "--" + field.replace("_", "-"))
             for field in ("questions", "cluster_map", "tier_questions")}
    if any(len(v) != 1 for v in (*trees.values(), *lists.values(), *runrecs.values(),
                                 *(v for w in where.values() for v in w.values()))):
        raise Refused("--graph, --matrices, --runrec and the file locations are given once per graph")
    unknown = sorted((set(lists) | set(runrecs) | {g for w in where.values() for g in w}) - set(trees))
    if unknown or not trees:
        raise Refused(f"every graph named needs a --graph entry; unlisted: {unknown}")
    out = []
    for name, (tree,) in trees.items():
        if name not in pins:
            raise Refused(f"graph {name} has no pinned inputs")
        if name not in lists:
            raise Refused(f"graph {name} has no --matrices list")
        pin = pins[name]
        for field, paths in where.items():
            if name in paths:
                if field == "tier_questions" and pin.tier_questions_sha256 is None:
                    raise Refused(f"graph {name} has no pinned tier file")
                pin = dataclasses.replace(pin, **{field: Path(paths[name][0])})
        files, list_sha = read_matrix_list(Path(lists[name][0]), Path(tree), name)
        out.append(GraphInput(name, Path(tree), tuple(files), list_sha, pin,
                              Path(runrecs[name][0]) if name in runrecs else None))
    return out


def main(argv: Sequence[str] | None = None, *, pins: Mapping[str, h.GraphPins] | None = None,
         reps: int | None = N_REPS) -> int:
    """0: written; 2: refused (the message names counts, file names and labels); 3: an unexpected error (class only)."""
    try:
        args = build_parser().parse_args(argv)
        agents = dict(h.parse_agent(a) for a in args.agent)
        if len(agents) != len(args.agent):
            raise Refused("an agent is listed twice")
        cohorts = parse_cohorts(args.cohorts, agents)
        h.check_output_location(args.out, args.results_root)  # before any computation
        real = pins is None
        graphs = graphs_from_args(args, h.real_pins(ROOT, args.config) if real else pins)
        results = analyse(graphs, agents, cohorts, config=args.config, config_sha256=args.config_sha256,
                          spec=args.spec, spec_sha256=args.spec_sha256, reps=reps, real_graph=real)
        sha = h.write_results(results, args.out, args.results_root)
        print_summary(results["meta"], args.out, sha)
        return 0
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001 - nothing of an unexpected error is printed but its class
        print(f"error: {type(e).__name__}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
