"""H6 self-check crossing: the analysis step (docs/h6-spec.md is the contract; the step refuses a spec file whose
sha256 is not the one it was started with).

    python -I scripts/h6_analysis.py --spec-sha256 H --config data/analysis/prereg-config.json --config-sha256 H \\
        --results-root DIR --out DIR/h6.json --draft-pins data/sources-h6/PINS-ds.json \\
        --agent opus=claude-opus-5-5 --agent deepseek=deepseek-ai/DeepSeek-V4-Flash-0731 \\
        --generator opus=opus:generated-1,generated-2,generated-3 \\
        --generator deepseek=deepseek:generated-ds-1,generated-ds-2,generated-ds-3 \\
        --generator opaque/deepseek=deepseek:generated-ds-2,generated-ds-3 \\
        --graph uniprot=TREE_U --shards uniprot=s1,s2 [--bank uniprot=BANK/matrix.jsonl.gz] \\
        --graph rhea=TREE_R --shards rhea=s1,s2 --drop uniprot=generated-ds-1

It reads the replay matrices of one H6 tree per graph (``<tree>/out/<shard>/matrix.jsonl.gz``) and writes ONE results
file (mode 600 in a mode 700 directory under ``--results-root``). Blinding (the design log): standard output
carries only counts of rows, file names and sha256s; a refusal names counts, file names and the command line's own
labels, never a value read from a row; an unexpected error names only its class. Every number goes to the results
file.

Every computation below is a pure function over arrays (``metric_vectors``, ``ratio_quantity``, ``mean_over_drafts``,
``own_other_delta``, ``interaction``, ``analyse_graph``, ``analyse_bank``), and the command line (``main``) is a thin
wrapper over ``analyse`` and ``write_results``.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import functools
import gzip
import hashlib
import json
import operator
import os
import re
import stat
import sys
import zlib
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.analysis import stats  # noqa: E402
from harness.analysis.inputs import InputError  # noqa: E402
from harness.analysis.prereg import (  # noqa: E402
    ClusterMap,
    Clusters,
    PreregConfig,
    PreregError,
    index_from_questions,
)
from harness.checkloop.accounting import _LETTER, CELLS  # noqa: E402
from harness.checkloop.matrix import BANK, BANK_ITEMS, BANK_MODEL, VARIANTS  # noqa: E402

SCHEMA = "h6-analysis/1"
SEED = 20261026
N_BOOT = 10000
LEVEL = 0.95
MIN_WRONG = 20  # prereg section 6: a recall on the errors of fewer wrong outputs is uninformative
N_DRAWS = 3  # a generator has up to three drafts
COHORT, SPLIT = "none", "test"
STATIC_ARMS = ("curated", "census-t5", "union", "union-qw")  # union-qw: the union of the three Qwen drafts (the 3 x 3)
STATUSES = ("flagged", "substantive_pass", "vacuous_pass", "unknown")
BANK_KINDS = tuple(sorted({*BANK_ITEMS.values(), "rewrite"}))  # untouched, answer_changing, answer_neutral, rewrite
NOT_AVAILABLE = "not available"
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
SHARD = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")

COL = {c: i for i, c in enumerate(CELLS)}
C_COLS = [COL[c] for c in CELLS if c.startswith("C_")]
W_COLS = [COL[c] for c in CELLS if c.startswith("W_")]
ROW_COLUMN = {(correct, status): COL[f"{'C' if correct else 'W'}_{_LETTER[status]}"]
              for correct in (True, False) for status in STATUSES}

CODE_FILES = ("scripts/h6_analysis.py", "harness/analysis/stats.py", "harness/analysis/prereg.py",
              "harness/checkloop/matrix.py")


class Refused(Exception):
    """A condition of the specification's section 3 or of the output location. The message names counts, file names
    and the command line's own labels only, never a value read from a row (the design log)."""


# --- the command line's parts: agents, generators, pins -----------------------------------------------------------


@dataclass(frozen=True)
class Generator:
    name: str
    agent: str  # the agent whose model made the drafts (own(g))
    drafts: tuple[str, ...]  # arm names of the drafts that exist


@dataclass(frozen=True)
class GraphPins:
    """The pinned input files of one graph (file names absolute or relative to the working directory)."""

    questions: Path
    questions_sha256: str
    cluster_map: Path
    cluster_map_sha256: str
    tier_questions: Path | None = None  # opaque Rhea: the natural file whose tiers order the index
    tier_questions_sha256: str | None = None


@dataclass(frozen=True)
class GraphInput:
    name: str
    tree: Path  # holds out/<shard>/matrix.jsonl.gz
    shards: tuple[str, ...]
    pins: GraphPins
    bank: tuple[Path, ...] = ()  # bank matrix files (spec section 4); empty: "bank: not available"
    generators: Mapping[str, Generator] | None = None  # this graph's generators when they differ from the run's
    # draft arm -> its source file, for the structural sensitivity rule (spec section 1); with the sha256 each must have
    draft_sources: Mapping[str, Path] | None = None
    draft_source_sha256: Mapping[str, str] | None = None


STRUCTURAL_RULE = ("every draft, of either generator, whose source declares no property gets a sensitivity without it "
                   "(spec section 1, ruling H as amended 24 Oct)")


def property_counts(label: str, sources: Mapping[str, Path], expected: Mapping[str, str] | None) -> dict[str, int]:
    """The number of properties each draft's source declares (the length of its ``properties`` map), from the source
    files themselves: a structural fact of the input, not an outcome. Each file is read ONCE; the bytes that are hashed
    against ``expected`` (when given) are the bytes that are parsed."""
    counts: dict[str, int] = {}
    for arm, path in sorted(sources.items()):
        try:
            data = Path(path).read_bytes()
        except OSError:
            raise Refused(f"{label}: the source of draft {arm} cannot be read") from None
        if expected is not None:
            if arm not in expected:
                raise Refused(f"{label}: no pinned sha256 for the source of draft {arm}")
            actual = hashlib.sha256(data).hexdigest()
            if actual != expected[arm]:
                raise Refused(f"{label} source of {arm}: sha256 {actual} is not the pinned {expected[arm]}")
        try:
            props = json.loads(data)["properties"]
            if not isinstance(props, dict):
                raise TypeError
            counts[arm] = len(props)
        except (ValueError, KeyError, TypeError):
            raise Refused(f"{label}: the source of draft {arm} cannot be read as a source") from None
    return counts


def structural_drops(counts: Mapping[str, int], generators: Mapping[str, Generator]) -> tuple[list[str], list[str]]:
    """(drafts to drop one at a time, generators skipped because none of their drafts declares a property)."""
    drops, skipped = [], []
    for g, gen in sorted(generators.items()):
        zero = [d for d in gen.drafts if counts[d] == 0]
        if zero and len(zero) == len(gen.drafts):
            skipped.append(g)
        else:
            drops += zero
    return drops, skipped


def parse_agent(text: str) -> tuple[str, str]:
    name, sep, model = text.partition("=")
    if not sep or not NAME.fullmatch(name) or not model.strip():
        raise Refused("--agent must be NAME=MODEL-ID")
    return name, model


def parse_generator(text: str) -> tuple[str | None, Generator]:
    """``[GRAPH/]NAME=AGENT:ARM,ARM,...``; a GRAPH prefix gives that graph's own draft list."""
    left, sep, right = text.partition("=")
    graph, slash, name = left.rpartition("/")
    agent, sep2, arms = right.partition(":")
    drafts = tuple(a for a in arms.split(",") if a)
    if (not sep or not sep2 or not NAME.fullmatch(name) or not NAME.fullmatch(agent) or not drafts
            or (slash and not NAME.fullmatch(graph)) or any(not NAME.fullmatch(d) for d in drafts)):
        raise Refused("--generator must be [GRAPH/]NAME=AGENT:ARM,ARM,...")
    return (graph if slash else None), Generator(name, agent, drafts)


def check_generators(agents: Mapping[str, str], generators: Mapping[str, Generator], n_draws: int = N_DRAWS) -> None:
    if not generators:
        raise Refused("no generator is listed")
    arms: list[str] = []
    for name, g in generators.items():
        if g.agent not in agents:
            raise Refused(f"generator {name}: its agent {g.agent} is not a listed agent")
        if len(g.drafts) > n_draws:
            raise Refused(f"generator {name}: {len(g.drafts)} drafts listed, at most {n_draws} exist")
        arms += g.drafts
    if len(set(arms)) != len(arms):
        raise Refused("a draft arm is listed twice")
    clash = sorted(set(arms) & set(STATIC_ARMS))
    if clash:
        raise Refused(f"draft arms named like a static arm: {clash}")
    # the interaction contrast is a 2 x 2 contrast: two agents with one generator each, or three agents with one
    # generator each, reported as the three PAIRWISE 2 x 2 interactions (spec section 1). A pooled interaction over
    # three agents is not defined: pooling the other agents' errors by count would stop agent difficulty cancelling.
    if len(agents) not in (2, 3) or sorted(g.agent for g in generators.values()) != sorted(agents):
        raise Refused("the interaction is defined for two (or three, pairwise) agents with one generator each")


# --- files, hashes, rows -------------------------------------------------------------------------------------------


def sha256_file(path: Path, label: str) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        raise Refused(f"{label}: cannot be read") from None
    return h.hexdigest()


def check_pin(label: str, path: Path, expected: str) -> str:
    actual = sha256_file(path, label)
    if actual != expected:
        raise Refused(f"{label}: sha256 {actual} is not the pinned {expected}")
    return actual


def read_matrix(path: Path, label: str) -> Iterator[dict]:
    """The rows of one gzip JSON-lines file, with the gzip and json modules only. Nothing of a row is ever put in a
    message."""
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
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
    except (OSError, EOFError, ValueError, zlib.error) as e:  # gzip.BadGzipFile is an OSError
        raise Refused(f"{label}: unreadable ({type(e).__name__})") from None


class Violations:
    """Counts of the rows that broke a condition, by condition, with the files that held them."""

    def __init__(self) -> None:
        self.rows: Counter[str] = Counter()
        self.files: dict[str, set[str]] = defaultdict(set)

    def add(self, kind: str, file: str) -> None:
        self.rows[kind] += 1
        self.files[kind].add(file)

    def raise_if_any(self, where: str) -> None:
        if self.rows:
            parts = [f"{kind} in {n} rows (files: {', '.join(sorted(self.files[kind]))})"
                     for kind, n in sorted(self.rows.items())]
            raise Refused(f"{where}: " + "; ".join(parts))


# --- the question index and the clusters -----------------------------------------------------------------------------


def read_questions(path: Path, label: str) -> dict[str, dict]:
    """Only ``id``, ``base_question_id``, ``template``, ``tier`` and ``split`` of each line are read."""
    questions: dict[str, dict] = {}
    try:
        with open(path, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    raise Refused(f"{label}: line {n} is not JSON") from None
                if not isinstance(rec, dict) or not isinstance(rec.get("id"), str):
                    raise Refused(f"{label}: line {n} is not a question")
                if rec["id"] in questions:
                    raise Refused(f"{label}: line {n} repeats an earlier question id")
                questions[rec["id"]] = {k: rec[k] for k in ("id", "base_question_id", "template", "tier", "split")
                                        if rec.get(k) is not None}
    except (OSError, UnicodeDecodeError):
        raise Refused(f"{label}: cannot be read") from None
    return questions


def build_index(pins: GraphPins, label: str) -> stats.QuestionIndex:
    """The test-split questions of the graph's question file as a QuestionIndex (strata ``template|tier``, which only
    order it). The opaque file has no tier: the tier of the same base question in the natural Rhea file is used."""
    check_pin(f"{label} question file", pins.questions, pins.questions_sha256)
    questions = read_questions(pins.questions, f"{label} question file")
    if pins.tier_questions is not None:
        if pins.tier_questions_sha256 is None:
            raise Refused(f"{label}: the tier file has no pinned sha256")
        check_pin(f"{label} tier question file", pins.tier_questions, pins.tier_questions_sha256)
        tiers: dict[str, str] = {}
        for q in read_questions(pins.tier_questions, f"{label} tier question file").values():
            if q.get("tier") is not None and tiers.setdefault(q.get("base_question_id") or q["id"], q["tier"]) != q["tier"]:
                raise Refused(f"{label} tier question file: a base question has two tiers")
        missing = 0
        for qid, q in list(questions.items()):
            if q.get("tier") is None:
                tier = tiers.get(q.get("base_question_id") or q["id"])
                if tier is not None:
                    questions[qid] = {**q, "tier": tier}
                elif q.get("split") == SPLIT:
                    missing += 1
        if missing:
            raise Refused(f"{label}: {missing} test questions have no tier in the natural question file")
    try:
        index = index_from_questions(questions, SPLIT)
    except InputError as e:
        raise Refused(f"{label} question file: {withheld(e)}") from None
    if not len(index):
        raise Refused(f"{label} question file: no test-split question")
    return index


# The study's loaders quote ids from the input files in their messages, so no loader text is ever forwarded: a message
# is mapped to a fixed reason (and the leading count, if it starts with one); anything else is "message withheld".
_REASONS = (
    ("are not in the cluster map", "base questions are not in the cluster map"),
    ("has two strata", "a base question has two strata"),
    ("span more than one template", "a cluster spans more than one template"),
    ("are not their question file's", "templates of the cluster map differ from the question file's"),
    ("stated clusters are not the connected components", "stated clusters are not the connected components of their anchors"),
    ("malformed questions", "malformed questions in the cluster map"),
    ("is not a cluster-map", "not a cluster-map file"),
)


def withheld(e: Exception) -> str:
    """A fixed description of an upstream loader error: never a piece of its text (it may quote an id)."""
    text = str(e)
    count = re.match(r"(\d+) ", text)
    for needle, reason in _REASONS:
        if needle in text:
            return f"{count.group(1)} {reason}" if count else reason
    return f"{type(e).__name__}, message withheld"


def load_cluster_map(path: Path, sha256: str, label: str) -> ClusterMap:
    check_pin(f"{label} cluster map", path, sha256)
    try:
        return ClusterMap.load(path, sha256)
    except PreregError as e:
        raise Refused(f"{label} cluster map: {withheld(e)}") from None


def resample(cmap: ClusterMap, index: stats.QuestionIndex, cfg: PreregConfig, label: str) -> Clusters:
    """The index's clusters and the ONE resample matrix K of the graph (``ClusterMap.resampling``)."""
    try:
        return cmap.resampling(index, cfg)
    except PreregError as e:
        raise Refused(f"{label}: {withheld(e)}") from None


def load_config(path: Path, sha256: str) -> tuple[PreregConfig, str]:
    """``seed`` and ``n_boot`` from the parsed config; anything but the registered two is refused."""
    actual = check_pin("config", path, sha256)
    try:
        data = json.loads(Path(path).read_bytes())
    except ValueError:
        raise Refused("config: not JSON") from None
    seed, n_boot = (data.get("seed"), data.get("n_boot")) if isinstance(data, dict) else (None, None)
    if type(seed) is not int or seed != SEED:
        raise Refused(f"config: seed is not the registered {SEED}")
    if type(n_boot) is not int or n_boot != N_BOOT:
        raise Refused(f"config: n_boot is not the registered {N_BOOT}")
    return PreregConfig(seed=seed, n_boot=n_boot), actual


# --- the attempts of one tree: reading and validating -----------------------------------------------------------------


@dataclass(frozen=True)
class GraphTables:
    """What the analysis needs of a validated tree: per (variant, agent, arm) the six cell counts per base question."""

    index: stats.QuestionIndex
    arms: tuple[str, ...]
    agents: Mapping[str, dict]  # name -> model, n_attempts, n_no_output, n_wrong, n_correct
    cells: Mapping[tuple[str, str, str], np.ndarray]  # (variant, agent, arm) -> (n_questions, 6) counts, CELLS order
    rows: int


def collect_tree(label: str, files: Sequence[tuple[str, Path]], index: stats.QuestionIndex,
                 agents: Mapping[str, str], drafts: Sequence[str], expected_graph: str | None = None) -> GraphTables:
    """Read every matrix file of a tree, refuse what section 3 refuses, and count the six cells per base question."""
    agent_of = {model: name for name, model in agents.items()}
    allowed = {*STATIC_ARMS, *drafts}
    nq = len(index)
    viol = Violations()
    runs: dict[str, tuple] = {}  # run id -> (agent, base question, output, correct)
    seen: set[tuple[str, str, str]] = set()
    groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    cells: dict[tuple[str, str, str], np.ndarray] = {}
    n_rows = 0
    for name, path in files:
        for row in read_matrix(path, f"{label} {name}"):
            n_rows += 1
            ok = True
            if row.get("cohort") != COHORT:
                viol.add(f"cohort other than {COHORT}", name)
                ok = False
            if row.get("split") != SPLIT:
                viol.add(f"split other than {SPLIT}", name)
                ok = False
            if expected_graph is not None and row.get("graph") != expected_graph:
                viol.add("graph other than the tree's", name)
                ok = False
            model, arm, variant, rid = (row.get(k) for k in ("model", "arm", "variant", "run_id"))
            agent = agent_of.get(model) if isinstance(model, str) else None
            if agent is None:
                viol.add("model that is not a listed agent", name)
                ok = False
            if not (isinstance(arm, str) and arm in allowed):
                viol.add("arm that is neither static nor a listed draft", name)
                ok = False
            if not (isinstance(variant, str) and variant in VARIANTS):
                viol.add("variant outside the four", name)
                ok = False
            if not (isinstance(rid, str) and rid):
                viol.add("run id that is not a non-empty string", name)
                ok = False
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
            info = (agent, base, output, correct if output else None)
            # one attempt may be replayed in more than one shard group when the groups hold DISJOINT arms (the 3 x 3: an
            # output replayed against the base arms in one shard and against the Qwen drafts in another); the same
            # (run id, arm, variant) twice is refused below, and an attempt must read the same everywhere
            if runs.setdefault(rid, info) != info:
                viol.add("run id whose agent, base question, output or correct differs between rows", name)
                continue
            key = (rid, arm, variant)
            if key in seen:
                viol.add("run id, arm and variant seen twice", name)
                continue
            seen.add(key)
            groups[(arm, variant)].add(rid)
            cell = cells.get((variant, agent, arm))
            if cell is None:
                cell = cells[(variant, agent, arm)] = np.zeros((nq, len(CELLS)), dtype=np.int64)
            if output:
                cell[index.position(base), ROW_COLUMN[(correct, status)]] += 1
    viol.raise_if_any(label)

    all_runs = set(runs)
    present = sorted({arm for arm, _ in groups})
    off = sum(1 for arm in present for v in VARIANTS if groups.get((arm, v)) != all_runs)
    if off:
        raise Refused(f"{label}: {off} of {len(present) * len(VARIANTS)} (arm, variant) groups do not hold exactly "
                      f"the tree's {len(all_runs)} attempts")
    without = [a for a in agents if not any(r[0] == a for r in runs.values())]
    if without:
        raise Refused(f"{label}: listed agents with no rows: {without}")
    no_draft = [d for d in drafts if d not in present]
    if no_draft:
        raise Refused(f"{label}: listed drafts with no rows: {no_draft}")

    summary: dict[str, dict] = {}
    for a, model in agents.items():
        mine = [r for r in runs.values() if r[0] == a]
        summary[a] = {"model": model, "n_attempts": len(mine), "n_no_output": sum(1 for r in mine if not r[2]),
                      "n_wrong": sum(1 for r in mine if r[2] and not r[3]),
                      "n_correct": sum(1 for r in mine if r[2] and r[3])}
    for v in VARIANTS:
        for a in agents:
            for arm in present:
                cells.setdefault((v, a, arm), np.zeros((nq, len(CELLS)), dtype=np.int64))
    for (v, a, arm), cell in cells.items():  # n_wrong and n_correct do not depend on the arm or the variant (checked)
        if (int(cell[:, W_COLS].sum()), int(cell[:, C_COLS].sum())) != (summary[a]["n_wrong"], summary[a]["n_correct"]):
            raise Refused(f"{label}: the cells of an arm do not add up to the attempts' wrong and correct outputs")
    return GraphTables(index, tuple(present), summary, cells, n_rows)


# --- quantities: pure functions over arrays ------------------------------------------------------------------------


@dataclass(frozen=True)
class Quantity:
    """A point estimate (None when undefined) and its draws (NaN where undefined) over the graph's ONE K."""

    point: float | None
    draws: np.ndarray

    def __sub__(self, other: Quantity) -> Quantity:
        point = None if self.point is None or other.point is None else self.point - other.point
        return Quantity(point, self.draws - other.draws)

    def __add__(self, other: Quantity) -> Quantity:
        point = None if self.point is None or other.point is None else self.point + other.point
        return Quantity(point, self.draws + other.draws)

    def summary(self, level: float = LEVEL) -> dict:
        iv = stats.interval(self.draws, level)
        return {"point": self.point, "lo": iv["lo"], "hi": iv["hi"], "level": iv["level"],
                "undefined_share": iv["undefined_share"]}


def mean_quantity(qs: Sequence[Quantity]) -> Quantity:
    """The unweighted mean of quantities on the same K: undefined (None, NaN) wherever any one is."""
    point = None if any(q.point is None for q in qs) else sum(q.point for q in qs) / len(qs)
    return Quantity(point, functools.reduce(operator.add, (q.draws for q in qs)) / len(qs))


def ratio_quantity(clusters: Clusters, num: np.ndarray, den: np.ndarray) -> Quantity:
    """The ratio of sums of per-question vectors: summed per cluster, drawn with the graph's K (``stats.ratio_draws``).
    The point estimate is sum(num) / sum(den), None for an empty denominator; never 0."""
    cn, cd = clusters.sum(num), clusters.sum(den)
    total = float(cd.sum())
    return Quantity(float(cn.sum()) / total if total > 0 else None, stats.ratio_draws(clusters.matrix, cn, cd))


def metric_vectors(cell: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-question (numerator, denominator) of R = W_F / W, FF = C_F / C and U = (C_U + W_U) / (W + C) from the
    (n_questions, 6) counts of a cell (the sums over a question's attempts: the ratio-of-sums rule)."""
    c = np.asarray(cell, dtype=np.float64)
    w, r = c[:, W_COLS].sum(axis=1), c[:, C_COLS].sum(axis=1)
    return {"R": (c[:, COL["W_F"]], w), "FF": (c[:, COL["C_F"]], r),
            "U": (c[:, COL["C_U"]] + c[:, COL["W_U"]], w + r), "WU": (c[:, COL["W_U"]], w)}


def mean_over_drafts(nums: Sequence[np.ndarray], dens: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """A generator's mean over its drafts: the numerator is the mean of the draft numerators per question, the
    denominator the common one (every draft is applied to the same outputs; refused otherwise)."""
    if not nums or len(nums) != len(dens):
        raise Refused("a mean over drafts needs at least one draft")
    for d in dens[1:]:
        if not np.array_equal(d, dens[0]):
            raise Refused("the drafts of a generator do not share one denominator")
    return np.sum(np.stack([np.asarray(n, dtype=np.float64) for n in nums]), axis=0) / len(nums), dens[0]


def own_other_delta(clusters: Clusters, own: str, nums: Mapping[str, np.ndarray],
                    dens: Mapping[str, np.ndarray]) -> dict[str, Quantity]:
    """delta_g = R(g | own(g)) - the mean of R(g | a) over the other agents a (the unweighted mean of their recalls: with
    one other agent, that agent's recall). ``nums[a]`` is the generator's (mean) numerator on agent a, ``dens[a]`` the
    agent's denominator per question. The contrast is the elementwise difference of the component draws, so it is
    paired by construction."""
    others = [a for a in nums if a != own]
    if own not in nums or not others:
        raise Refused("own-minus-other needs the generator's own agent and at least one other agent")
    own_q = ratio_quantity(clusters, nums[own], dens[own])
    other_q = mean_quantity([ratio_quantity(clusters, nums[a], dens[a]) for a in others])
    return {"own": own_q, "others": other_q, "delta": own_q - other_q}


def pairwise_block(clusters: Clusters, names: Sequence[str], gen_of_agent: Mapping[str, str],
                   inputs: Mapping[str, tuple[Mapping[str, np.ndarray], Mapping[str, np.ndarray]]],
                   n_wrong: Mapping[str, int], flagged: bool = True) -> dict:
    """Every pairwise 2 x 2 interaction among the agents, in listing order ((0,1), (0,2), (1,2) for three agents), each
    computed exactly as the two-agent I from the two agents' errors and their two generators: delta_a = R(g_a | a) -
    R(g_a | b), delta_b = R(g_b | b) - R(g_b | a), I = delta_a + delta_b. ``inputs[g]`` is (nums, dens) of generator g
    per agent. The flag is the pair's own: either agent under 20 wrong outputs (``flagged`` False: no flag)."""
    def recall(g: str, a: str) -> Quantity:
        nums, dens = inputs[g]
        return ratio_quantity(clusters, nums[a], dens[a])

    out: dict = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ga, gb = gen_of_agent[a], gen_of_agent[b]
            d_a, d_b = recall(ga, a) - recall(ga, b), recall(gb, b) - recall(gb, a)
            flag = uninformative(n_wrong, [a, b]) if flagged else {}
            out[f"{a}|{b}"] = {"agents": [a, b], "generators": [ga, gb],
                               "delta": {ga: {**d_a.summary(), **flag}, gb: {**d_b.summary(), **flag}},
                               "I": {**(d_a + d_b).summary(), **flag}}
    return out


def interaction(deltas: Mapping[str, Quantity]) -> Quantity:
    """I = the sum of delta_g over the generators (in name order); undefined where any delta is."""
    return functools.reduce(operator.add, (deltas[g] for g in sorted(deltas)))


def uninformative(n_wrong: Mapping[str, int], agents: Sequence[str]) -> dict:
    """The < 20 rule: the agents whose recall is uninformative. A contrast uses every agent's errors."""
    low = [a for a in agents if n_wrong[a] < MIN_WRONG]
    return {"uninformative": bool(low), "uninformative_agents": low}


def draws_label(k: int, n_draws: int = N_DRAWS) -> str:
    return f"{k} of {n_draws} draws"


# --- one graph's results -----------------------------------------------------------------------------------------------


def draft_inputs(tables: GraphTables, variant: str, agents: Sequence[str], drafts: Sequence[str],
                 metric: str = "R") -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Per agent, the mean over ``drafts`` of the metric's numerator, and its common denominator (for R: the flagged-wrong
    numerator and the wrong-output denominator)."""
    nums, dens = {}, {}
    for a in agents:
        per = [metric_vectors(tables.cells[(variant, a, d)])[metric] for d in drafts]
        nums[a], dens[a] = mean_over_drafts([p[0] for p in per], [p[1] for p in per])
    return nums, dens


def delta_entry(clusters: Clusters, own: str, nums: Mapping[str, np.ndarray], dens: Mapping[str, np.ndarray],
                flag: Mapping, n_wrong: Mapping[str, int]) -> tuple[dict, Quantity]:
    """The delta block of one generator (or draft set): the two recalls it is made of carry their own < 20 flag (the
    own agent's, and any of the other agents'), the delta and its flag the contrast's."""
    parts = own_other_delta(clusters, own, nums, dens)
    low_others = [a for a in nums if a != own and n_wrong[a] < MIN_WRONG]
    entry = {"own_recall": {**parts["own"].summary(), "uninformative": n_wrong[own] < MIN_WRONG},
             "others_recall": {**parts["others"].summary(), "uninformative": bool(low_others),
                               "uninformative_agents": low_others},
             "delta": parts["delta"].summary(), **flag}
    return entry, parts["delta"]


def secondary_contrasts(clusters: Clusters, tables: GraphTables, variant: str, agents: Mapping[str, str],
                        generators: Mapping[str, Generator], n_wrong: Mapping[str, int]) -> dict:
    """The same own-minus-mean-of-others and pairwise 2 x 2 contrasts on the false-flag rate (C_F / C) and on the share
    of an agent's wrong outputs that the checks leave unknown (W_U / W), over the generator's mean of drafts: a lower
    recall on one's own errors can come from leniency (the false-flag contrast), from a check that cannot decide own
    outputs (the unknown share) or from blindness to own errors (neither). The unknown share is a ratio over wrong
    outputs, so each component carries the < 20 flag of the agents whose errors it uses; the false-flag rate none."""
    names = list(agents)
    gen_of_agent = {gen.agent: g for g, gen in generators.items()}
    out: dict = {}
    for metric, label, flagged in (("FF", "false_flag_rate", False), ("WU", "unknown_share_of_wrong", True)):
        inputs = {g: draft_inputs(tables, variant, names, generators[g].drafts, metric) for g in sorted(generators)}
        by_gen = {}
        for g in sorted(generators):
            own = generators[g].agent
            parts = own_other_delta(clusters, own, *inputs[g])
            low_others = [a for a in names if a != own and n_wrong[a] < MIN_WRONG]
            flags = {"own": {"uninformative": n_wrong[own] < MIN_WRONG},
                     "others": {"uninformative": bool(low_others), "uninformative_agents": low_others},
                     "delta": uninformative(n_wrong, names)}
            by_gen[g] = {k: {**parts[k].summary(), **(flags[k] if flagged else {})} for k in ("own", "others", "delta")}
        pairs = pairwise_block(clusters, names, gen_of_agent, inputs, n_wrong, flagged)
        first = pairs[f"{names[0]}|{names[1]}"]["I"]
        out[label] = {"by_generator": by_gen, "I": first, "pairwise": pairs}
    return out


def analyse_graph(tables: GraphTables, clusters: Clusters, agents: Mapping[str, str],
                  generators: Mapping[str, Generator], drops: Sequence[str] = (), n_draws: int = N_DRAWS) -> dict:
    """The ``agents``, ``cells``, ``generators`` and ``sensitivity`` blocks of one graph. With two agents ``I`` is the
    interaction; with three, ``I`` is the first pair's (the registered one) and ``pairwise`` holds all three 2 x 2
    interactions; a pooled interaction over three agents is not defined."""
    names = list(agents)
    n_wrong = {a: tables.agents[a]["n_wrong"] for a in names}
    flag = uninformative(n_wrong, names)
    gen_of_agent = {gen.agent: g for g, gen in generators.items()}
    first_pair = f"{names[0]}|{names[1]}"
    owner = {d: g for g, gen in generators.items() for d in gen.drafts}
    unknown = [d for d in drops if d not in owner]
    if unknown:
        raise Refused(f"sensitivity: {unknown} is not a draft of a listed generator")
    for d in drops:
        if len(generators[owner[d]].drafts) < 2:
            raise Refused(f"sensitivity: {d} is the only draft of its generator")
    first_arm = tables.arms[0]
    wrong_clusters = {a: int(np.count_nonzero(clusters.sum(tables.cells[("primary", a, first_arm)][:, W_COLS].sum(axis=1))))
                      for a in names}
    out: dict = {"agents": {a: {**tables.agents[a], "n_clusters_with_wrong": wrong_clusters[a],
                                "uninformative": n_wrong[a] < MIN_WRONG} for a in names},
                 "pair_order": [f"{a}|{b}" for i, a in enumerate(names) for b in names[i + 1:]],
                 "cells": {}, "generators": {}, "sensitivity": {}}
    for v in VARIANTS:
        out["cells"][v] = {}
        for a in names:
            out["cells"][v][a] = {}
            for arm in tables.arms:
                cell = tables.cells[(v, a, arm)]
                entry: dict = {"counts": {c: int(x) for c, x in zip(CELLS, cell.sum(axis=0))}}
                for metric, (num, den) in metric_vectors(cell).items():
                    entry[metric] = ratio_quantity(clusters, num, den).summary()
                entry["R"]["uninformative"] = n_wrong[a] < MIN_WRONG
                out["cells"][v][a][arm] = entry
        by_gen, inputs = {}, {}
        for g in sorted(generators):
            gen = generators[g]
            nums, dens = inputs[g] = draft_inputs(tables, v, names, gen.drafts)
            mean_entry, _ = delta_entry(clusters, gen.agent, nums, dens, flag, n_wrong)
            per_draft = {d: delta_entry(clusters, gen.agent, *draft_inputs(tables, v, names, (d,)), flag, n_wrong)[0]
                         for d in gen.drafts}
            by_gen[g] = {"agent": gen.agent, "drafts": list(gen.drafts), "draws": draws_label(len(gen.drafts), n_draws),
                         "mean_recall": {a: {**ratio_quantity(clusters, nums[a], dens[a]).summary(),
                                             "uninformative": n_wrong[a] < MIN_WRONG} for a in names},
                         "mean_delta": mean_entry, "per_draft_delta": per_draft}
        pairs = pairwise_block(clusters, names, gen_of_agent, inputs, n_wrong)
        out["generators"][v] = {"by_generator": by_gen, "I": {**pairs[first_pair]["I"],
                                                              "draws": {g: by_gen[g]["draws"] for g in sorted(by_gen)}},
                                "pairwise": pairs,
                                "pooled_interaction": "defined for two agents only" if len(names) == 3 else "the I above",
                                "secondary": secondary_contrasts(clusters, tables, v, agents, generators, n_wrong)}
        out["sensitivity"][v] = {}
        for d in drops:
            g = owner[d]
            rest = tuple(x for x in generators[g].drafts if x != d)
            sens_inputs = {**inputs, g: draft_inputs(tables, v, names, rest)}
            entry, _ = delta_entry(clusters, generators[g].agent, *sens_inputs[g], flag, n_wrong)
            sens_pairs = pairwise_block(clusters, names, gen_of_agent, sens_inputs, n_wrong)
            out["sensitivity"][v][d] = {"generator": g, "without": d, "drafts": list(rest),
                                        "draws": draws_label(len(rest), n_draws), "delta": entry,
                                        "I": sens_pairs[first_pair]["I"], "pairwise": sens_pairs}
    return out


# --- the fault-bank reference (spec section 4) -------------------------------------------------------------------


@dataclass(frozen=True)
class BankTables:
    index: stats.QuestionIndex
    # (variant, arm) -> {kind: (flagged per base question, items per base question)}, kind in answer_changing/untouched
    vectors: Mapping[tuple[str, str], Mapping[str, tuple[np.ndarray, np.ndarray]]]
    rows: int


def collect_bank(label: str, files: Sequence[tuple[str, Path]], generators: Mapping[str, Generator]) -> BankTables:
    """Read the bank matrices: rows of cohort ``bank`` and model ``bank``; the correct flag must follow the item's kind;
    every listed draft must cover the same items."""
    drafts = [d for g in generators.values() for d in g.drafts]
    allowed = {*STATIC_ARMS, *drafts}
    viol = Violations()
    items: dict[str, tuple] = {}  # item (run id) -> (kind, base question, template)
    seen: set[tuple[str, str, str]] = set()
    covered: dict[tuple[str, str], dict[str, bool]] = defaultdict(dict)  # (variant, arm) -> item -> flagged
    n_rows = 0
    for name, path in files:
        for row in read_matrix(path, f"{label} {name}"):
            n_rows += 1
            ok = True
            if row.get("cohort") != BANK:
                viol.add(f"cohort other than {BANK}", name)
                ok = False
            if row.get("model") != BANK_MODEL:
                viol.add(f"model other than {BANK_MODEL}", name)
                ok = False
            if row.get("split") != SPLIT:
                viol.add(f"bank row with a split other than {SPLIT}", name)
                ok = False
            arm, variant, rid, kind = (row.get(k) for k in ("arm", "variant", "run_id", "bank_item"))
            if not (isinstance(arm, str) and arm in allowed):
                viol.add("arm that is neither static nor a listed draft", name)
                ok = False
            if not (isinstance(variant, str) and variant in VARIANTS):
                viol.add("variant outside the four", name)
                ok = False
            if not (isinstance(rid, str) and rid):
                viol.add("item id that is not a non-empty string", name)
                ok = False
            if not (isinstance(kind, str) and kind in BANK_KINDS):
                viol.add("bank_item outside the four kinds", name)
                ok = False
            elif row.get("correct") is not (kind != "answer_changing"):
                viol.add("correct that does not follow bank_item", name)
                ok = False
            if row.get("output") is not True or row.get("status") not in STATUSES:
                viol.add("bank row without an output or with a status outside the four", name)
                ok = False
            base, template = row.get("base_question_id"), row.get("template")
            if not (isinstance(base, str) and base and isinstance(template, str) and template):
                viol.add("bank row without a base question or a template", name)
                ok = False
            if not ok:
                continue
            if items.setdefault(rid, (kind, base, template)) != (kind, base, template):
                viol.add("item whose kind, base question or template differs between rows", name)
                continue
            if (rid, arm, variant) in seen:
                viol.add("item, arm and variant seen twice", name)
                continue
            seen.add((rid, arm, variant))
            covered[(variant, arm)][rid] = row["status"] == "flagged"
    viol.raise_if_any(label)

    sets = [set(covered.get((v, d), {})) for v in VARIANTS for d in drafts]  # every draft in every variant
    if not sets or not sets[0] or any(s != sets[0] for s in sets):
        raise Refused(f"{label}: the listed drafts' bank rows do not cover the same items in every variant")
    templates: dict[str, set[str]] = defaultdict(set)
    for rid in sets[0]:  # the index holds the drafts' items only: an item only a static arm covers adds no cluster
        _, base, template = items[rid]
        templates[base].add(template)
    if any(len(t) > 1 for t in templates.values()):
        raise Refused(f"{label}: a base question has more than one template")
    index = stats.QuestionIndex.from_strata({b: f"{next(iter(t))}|-" for b, t in templates.items()})
    vectors: dict[tuple[str, str], dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for (v, arm), flagged in covered.items():
        per: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for kind in ("answer_changing", "untouched"):
            num, den = np.zeros(len(index)), np.zeros(len(index))
            for rid, was_flagged in flagged.items():
                k, base, _ = items[rid]
                if k == kind and base in index:  # a static arm may cover items outside the drafts' index: not counted
                    den[index.position(base)] += 1
                    num[index.position(base)] += was_flagged
            per[kind] = (num, den)
        vectors[(v, arm)] = per
    return BankTables(index, vectors, n_rows)


def analyse_bank(tables: BankTables, clusters: Clusters, generators: Mapping[str, Generator],
                 n_draws: int = N_DRAWS) -> dict:
    """Per variant and generator: bank recall (flagged / all over the answer-changing items) and bank false-flag rate
    (flagged / all over the untouched items), per draft and for the mean over drafts. Not combined with the agents'."""
    out: dict = {}
    for v in VARIANTS:
        out[v] = {}
        for g in sorted(generators):
            drafts = generators[g].drafts
            entry: dict = {"drafts": list(drafts), "draws": draws_label(len(drafts), n_draws)}
            for metric, kind in (("recall", "answer_changing"), ("false_flag", "untouched")):
                per = {d: tables.vectors[(v, d)][kind] for d in drafts}
                mean_num, den = mean_over_drafts([p[0] for p in per.values()], [p[1] for p in per.values()])
                entry[metric] = {"mean": ratio_quantity(clusters, mean_num, den).summary(),
                                 "per_draft": {d: ratio_quantity(clusters, *per[d]).summary() for d in drafts}}
                entry[f"n_{kind}"] = int(den.sum())
            out[v][g] = entry
    return out


def analyse_bank_sensitivity(tables: BankTables, clusters: Clusters, generators: Mapping[str, Generator],
                             drops: Sequence[str], n_draws: int = N_DRAWS) -> dict:
    """The bank reference of the generator's drafts without a named draft (spec section 4: the sensitivity)."""
    owner = {d: g for g, gen in generators.items() for d in gen.drafts}
    out: dict = {}
    for v in VARIANTS:
        out[v] = {}
        for d in drops:
            rest = tuple(x for x in generators[owner[d]].drafts if x != d)
            entry: dict = {"generator": owner[d], "without": d, "drafts": list(rest), "draws": draws_label(len(rest), n_draws)}
            for metric, kind in (("recall", "answer_changing"), ("false_flag", "untouched")):
                per = [tables.vectors[(v, x)][kind] for x in rest]
                mean_num, den = mean_over_drafts([p[0] for p in per], [p[1] for p in per])
                entry[metric] = {"mean": ratio_quantity(clusters, mean_num, den).summary()}
            out[v][d] = entry
    return out


# --- the run ---------------------------------------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def analyse(graphs: Sequence[GraphInput], agents: Mapping[str, str], generators: Mapping[str, Generator], *,
            config: Path, config_sha256: str, spec: Path, spec_sha256: str,
            drops: Sequence[tuple[str, str]] = (), draft_pins: Path | None = None, extra_draft_pins: Sequence[Path] = (),
            created_utc: str | None = None, n_draws: int = N_DRAWS, real_graph: bool = False) -> dict:
    """The whole analysis; returns the results dictionary (``write_results`` puts it in a file)."""
    if len(agents) not in (2, 3) or len(set(agents.values())) != len(agents):
        raise Refused("the interaction is defined for two (or three, pairwise) agents with distinct models")
    check_generators(agents, generators, n_draws)
    names = [g.name for g in graphs]
    if not graphs or len(set(names)) != len(names) or any(not NAME.fullmatch(n) or n == "meta" for n in names):
        raise Refused("graph names must be distinct labels and not 'meta'")
    if any(g not in names for g, _ in drops):
        raise Refused("--drop names a graph that is not listed")
    hashes: dict = {"spec": check_pin("spec", spec, spec_sha256)}
    cfg, hashes["config"] = load_config(config, config_sha256)
    hashes["draft_pins"] = sha256_file(draft_pins, "draft pins") if draft_pins is not None else None
    hashes["extra_draft_pins"] = {Path(p).name: sha256_file(Path(p), "extra draft pins") for p in extra_draft_pins}
    hashes["code"] = {rel: sha256_file(ROOT / rel, rel) for rel in CODE_FILES}
    results: dict = {}
    rows_read: dict[str, int] = {}
    unlisted_shards: dict[str, int] = {}
    structural_info: dict[str, dict] = {}
    cluster_info: dict[str, dict] = {}
    hashes["graphs"] = {}
    for gi in graphs:
        gens = {**generators, **(gi.generators or {})}
        check_generators(agents, gens, n_draws)
        index = build_index(gi.pins, gi.name)
        cmap = load_cluster_map(gi.pins.cluster_map, gi.pins.cluster_map_sha256, gi.name)
        files = []
        for shard in gi.shards:
            if not SHARD.fullmatch(shard):
                raise Refused(f"{gi.name}: shard name {shard} is not a plain name")
            files.append((f"out/{shard}/matrix.jsonl.gz", gi.tree / "out" / shard / "matrix.jsonl.gz"))
        if not files or len({f for f, _ in files}) != len(files):
            raise Refused(f"{gi.name}: the shards must be listed, once each")
        out_dir = gi.tree / "out"
        unlisted = sum(1 for p in out_dir.iterdir() if p.is_dir() and p.name not in set(gi.shards)) if out_dir.is_dir() else 0
        h: dict = {"questions": gi.pins.questions_sha256, "cluster_map": gi.pins.cluster_map_sha256,
                   "tier_questions": gi.pins.tier_questions_sha256,
                   "matrices": {f: sha256_file(p, f"{gi.name} {f}") for f, p in files}}
        drafts = [d for g in gens.values() for d in g.drafts]
        graph_drops = [d for g, d in drops if g == gi.name]
        if gi.draft_sources is not None:
            missing = [d for d in drafts if d not in gi.draft_sources]
            if missing:
                raise Refused(f"{gi.name}: the structural rule needs the source of every draft; missing: {missing}")
            counts = property_counts(gi.name, {d: gi.draft_sources[d] for d in drafts}, gi.draft_source_sha256)
            structural, skipped = structural_drops(counts, gens)
            structural_info[gi.name] = {"properties_per_draft": counts, "dropped": structural,
                                        "generators_without_any_property_skipped": skipped}
            graph_drops = sorted({*graph_drops, *structural})
        tables = collect_tree(gi.name, files, index, agents, drafts, REAL_GRAPH_DIRS.get(gi.name) if real_graph else None)
        rows_read[gi.name] = tables.rows
        unlisted_shards[gi.name] = unlisted
        clusters = resample(cmap, index, cfg, f"{gi.name} cluster map")  # ONE K per graph
        block = analyse_graph(tables, clusters, agents, gens, graph_drops, n_draws)
        block["bank"] = NOT_AVAILABLE
        cluster_info[gi.name] = {"agent": clusters.summary(), "bank": None}
        if gi.bank:
            bank_files = [(f"bank/{p.name}", p) for p in gi.bank]
            if len({f for f, _ in bank_files}) != len(bank_files):
                raise Refused(f"{gi.name}: two bank files share a name")
            h["bank"] = {f: sha256_file(p, f"{gi.name} {f}") for f, p in bank_files}
            bank = collect_bank(f"{gi.name} bank", bank_files, gens)
            bank_clusters = resample(cmap, bank.index, cfg, f"{gi.name} bank base questions")
            block["bank"] = analyse_bank(bank, bank_clusters, gens, n_draws)
            block["bank_sensitivity"] = analyse_bank_sensitivity(bank, bank_clusters, gens,
                                                                 graph_drops, n_draws)
            rows_read[f"{gi.name}:bank"] = bank.rows
            cluster_info[gi.name]["bank"] = bank_clusters.summary()
        results[gi.name] = block
        hashes["graphs"][gi.name] = h
    results["meta"] = {
        "schema": SCHEMA, "created_utc": created_utc or utc_now(), "seed": cfg.seed, "n_boot": cfg.n_boot,
        "scheme": "unstratified", "level": LEVEL, "min_wrong": MIN_WRONG, "n_draws": n_draws,
        "agents": dict(agents),
        "generators": {g: {"agent": gen.agent, "drafts": list(gen.drafts)} for g, gen in sorted(generators.items())},
        "graph_generators": {gi.name: {g: list(gen.drafts) for g, gen in sorted(gi.generators.items())}
                             for gi in graphs if gi.generators},
        "drops": [{"graph": g, "draft": d} for g, d in drops],
        "structural_rule": {"rule": STRUCTURAL_RULE, "applied": structural_info or "not applied"},
        "hashes": hashes, "rows_read": rows_read, "shard_dirs_not_listed": unlisted_shards, "clusters": cluster_info}
    return results


def flatten_hashes(meta: Mapping) -> list[tuple[str, str]]:
    """(file label, sha256) pairs of everything the run hashed, for the standard output."""
    h = meta["hashes"]
    out = [("spec", h["spec"]), ("config", h["config"])]
    if h.get("draft_pins"):
        out.append(("draft-pins", h["draft_pins"]))
    for name, sha in sorted((h.get("extra_draft_pins") or {}).items()):
        out.append((f"extra-draft-pins {name}", sha))
    out += [(f"code/{rel}", sha) for rel, sha in h["code"].items()]
    for graph, gh in h["graphs"].items():
        out += [(f"{graph}/questions", gh["questions"]), (f"{graph}/cluster-map", gh["cluster_map"])]
        if gh.get("tier_questions"):
            out.append((f"{graph}/tier-questions", gh["tier_questions"]))
        out += [(f"{graph}/{rel}", sha) for rel, sha in gh["matrices"].items()]
        out += [(f"{graph}/{rel}", sha) for rel, sha in gh.get("bank", {}).items()]
    return out


# --- the results file ---------------------------------------------------------------------------------------------------


def check_output_location(out: Path, results_root: Path) -> Path:
    """The output file must lie under ``results_root``, in a directory of mode 700, and not exist."""
    try:
        root = Path(results_root).resolve(strict=True)
        parent = Path(os.path.abspath(out)).parent.resolve(strict=True)
    except OSError:
        raise Refused("the results root or the output directory does not exist") from None
    final = parent / Path(out).name
    if Path(out).name in ("", ".", "..") or not (parent == root or root in parent.parents):
        raise Refused("the output file is not under the results root")
    if os.path.lexists(final):
        raise Refused("the output file already exists")
    if stat.S_IMODE(parent.stat().st_mode) != 0o700:
        raise Refused("the output directory is not mode 700")
    return final


def _create_exclusive(path: Path) -> int:
    return os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)


def write_results(results: Mapping, out: Path, results_root: Path) -> str:
    """Write the results (sorted keys, no NaN) as a new mode-600 file; returns its sha256. The data goes to a hidden
    partial file in the same directory and is linked to its final name only when complete (``os.link`` refuses an
    existing name), so the final file name never holds a partial result, even if the process is killed."""
    final = check_output_location(out, results_root)
    data = (json.dumps(results, sort_keys=True, indent=1, allow_nan=False) + "\n").encode("utf-8")
    partial = final.with_name(f".{final.name}.partial.{os.getpid()}")
    fd = _create_exclusive(partial)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if stat.S_IMODE(partial.stat().st_mode) != 0o600:
            raise Refused("the results file did not end up mode 600")
        try:
            os.link(partial, final)
        except FileExistsError:
            raise Refused("the results file exists") from None
    finally:
        partial.unlink(missing_ok=True)
    return hashlib.sha256(data).hexdigest()


# --- standard output: counts of rows, file names and sha256s only -------------------------------------------------------


def print_summary(meta: Mapping, out: Path, out_sha256: str, file: TextIO | None = None) -> None:
    file = sys.stdout if file is None else file
    for graph, n in sorted(meta["rows_read"].items()):
        print(f"rows {graph} {n}", file=file)
    for label, sha in flatten_hashes(meta):
        print(f"file {label} {sha}", file=file)
    print(f"output {out} {out_sha256}", file=file)


# --- the command line ----------------------------------------------------------------------------------------------------

SPEC_UNIPROT_QUESTIONS_SHA256 = "17eef284d7c89c64e5da42245b0b09f21215d2ebd01e086670057baae36c9184"
SPEC_UNIPROT_CLUSTER_MAP = "data/questions/uniprot-sp-2026_03/extension/clusters.json"
SPEC_RHEA_QUESTIONS_SHA256 = "b204846e7c20e82971ec65590298d25d54dd480efcea4f178e5044847dfa96b7"
SPEC_OPAQUE_QUESTIONS_SHA256 = "4153ad29e7d4d659a12425a685e8b460e95e0966930221adea8119190b263507"
SPEC_RHEA_CLUSTER_MAP_SHA256 = "1f7346fadeffb1d81074255a48c7f9f0eed9d48e8ac65dc8fcd73322aafd881d"
SPEC_OPAQUE_CLUSTER_MAP_SHA256 = "86fbd61c0765a29abc614c88fa71dccd4f0fafc6966245384019e7a6e2484441"
UNIPROT_GRAPH = "uniprot-sp-2026_03"
SPEC_DRAFT_PINS_SHA256 = "a42c6c0789bf06d96e1fdd385678737a803df7dc5587cbe670fdf87edc555e1c"  # data/sources-h6/PINS-ds.json


def real_pins(root: Path, config: Path) -> dict[str, GraphPins]:
    """The pinned inputs of the three graphs (spec sections 2 and 5). UniProt's cluster map and its sha256 come from
    the config (the frozen map of the preregistration config), whose own hash is checked by ``analyse``."""
    try:
        cfg = json.loads(Path(config).read_bytes())
        uni_map_sha = cfg["cluster_map_sha256"]
        if (cfg["cluster_map"] != SPEC_UNIPROT_CLUSTER_MAP
                or cfg["questions_sha256"][UNIPROT_GRAPH] != SPEC_UNIPROT_QUESTIONS_SHA256):
            raise Refused("config: its UniProt question-file or cluster-map entry is not the spec's")
    except (OSError, ValueError, KeyError, TypeError):
        raise Refused("config: cannot be read for the UniProt pins") from None
    rhea_questions = root / "data/questions-frozen/rhea-142/questions.jsonl"
    return {
        "uniprot": GraphPins(root / "data/questions/uniprot-sp-2026_03/questions.jsonl",
                             SPEC_UNIPROT_QUESTIONS_SHA256, root / SPEC_UNIPROT_CLUSTER_MAP, uni_map_sha),
        "rhea": GraphPins(rhea_questions, SPEC_RHEA_QUESTIONS_SHA256,
                          root / "data/analysis-h6/clusters/rhea-142.json", SPEC_RHEA_CLUSTER_MAP_SHA256),
        "opaque": GraphPins(root / "data/questions/opaque-rhea-142/questions.jsonl", SPEC_OPAQUE_QUESTIONS_SHA256,
                            root / "data/analysis-h6/clusters/opaque-rhea-142.json", SPEC_OPAQUE_CLUSTER_MAP_SHA256,
                            rhea_questions, SPEC_RHEA_QUESTIONS_SHA256),
    }


def _by_graph(items: Sequence[str], what: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for item in items:
        graph, sep, value = item.partition("=")
        if not sep or not NAME.fullmatch(graph) or not value:
            raise Refused(f"{what} must be GRAPH=VALUE")
        out[graph].append(value)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="h6_analysis.py", description=__doc__.split("\n\n")[0])
    p.add_argument("--spec", type=Path, default=ROOT / "docs/h6-spec.md")
    p.add_argument("--spec-sha256", required=True)
    p.add_argument("--config", type=Path, default=ROOT / "data/analysis/prereg-config.json")
    p.add_argument("--config-sha256", required=True)
    p.add_argument("--draft-pins", type=Path, default=ROOT / "data/sources-h6/PINS-ds.json")
    p.add_argument("--results-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--agent", action="append", default=[], metavar="NAME=MODEL-ID")
    p.add_argument("--generator", action="append", default=[], metavar="[GRAPH/]NAME=AGENT:ARM,ARM,...")
    p.add_argument("--graph", action="append", default=[], metavar="GRAPH=TREE-DIR")
    p.add_argument("--shards", action="append", default=[], metavar="GRAPH=SHARD,SHARD,...")
    p.add_argument("--bank", action="append", default=[], metavar="GRAPH=BANK-MATRIX-FILE")
    p.add_argument("--questions", action="append", default=[], metavar="GRAPH=FILE",
                   help="where the graph's pinned question file is (its sha256 stays the pinned one)")
    p.add_argument("--cluster-map", action="append", default=[], metavar="GRAPH=FILE",
                   help="where the graph's pinned cluster map is (its sha256 stays the pinned one)")
    p.add_argument("--tier-questions", action="append", default=[], metavar="GRAPH=FILE",
                   help="where the natural Rhea question file is, for the graph whose index takes its tiers from it")
    p.add_argument("--drop", action="append", default=[], metavar="GRAPH=DRAFT-ARM")
    p.add_argument("--extra-draft-pins", action="append", default=[], metavar="FILE=SHA256",
                   help="a further draft pin file (the Qwen drafts' PINS-qw.json for the 3 x 3), hashed against the "
                        "given sha256; the arms it pins join the pinned drafts of the structural rule")
    p.add_argument("--draft-source", action="append", default=[], metavar="GRAPH/ARM=SOURCE-FILE",
                   help="the source file of every draft of the graph (hashed against the draft pins): the structural "
                        "sensitivity rule reads how many properties each declares")
    return p


def draft_source_args(items: Sequence[str]) -> dict[str, dict[str, str]]:
    """``GRAPH/ARM=FILE`` items as graph -> arm -> file."""
    out: dict[str, dict[str, str]] = defaultdict(dict)
    for item in items:
        left, sep, path = item.partition("=")
        graph, slash, arm = left.partition("/")
        if not (sep and slash and NAME.fullmatch(graph) and NAME.fullmatch(arm) and path) or arm in out[graph]:
            raise Refused("--draft-source must be GRAPH/ARM=FILE, once per draft")
        out[graph][arm] = path
    return dict(out)


def pinned_draft_shas(pins_file: Path, labels: Mapping[str, str],
                      extra: Sequence[Path] = ()) -> dict[str, dict[str, str]]:
    """graph label -> draft arm -> the sha256 the draft pins hold (the DeepSeek drafts' ``source_sha256`` and the frozen
    Opus sources'), for the three real graphs; ``labels`` maps the command line's graph label to the pin record's
    graph directory name. ``extra`` are further pin files of the same form (the Qwen drafts' PINS-qw.json, UniProt
    only): their draws are added under the arm name ``generated-<tag>-<i>`` of the file's generator tag, and their
    frozen Opus sources must agree with the first file's."""
    try:
        pins = json.loads(Path(pins_file).read_bytes())
        out: dict[str, dict[str, str]] = {}
        for label, gdir in labels.items():
            shas = {f"generated-ds-{i}": d["source_sha256"] for i, d in pins["graphs"][gdir]["draws"].items()}
            shas.update({f"generated-{i}": sha for i, sha in pins["opus_sources"][gdir].items()})
            out[label] = shas
        for path in extra:
            more = json.loads(Path(path).read_bytes())
            tag = more["generator"]["tag"]
            for gdir, entry in more["graphs"].items():
                label = next((lab for lab, g in labels.items() if g == gdir), None)
                if label is None or not re.fullmatch(r"[a-z]{2,8}", tag):
                    raise Refused("an extra draft pin file names a graph or a generator tag that is not known")
                if more["opus_sources"][gdir] != pins["opus_sources"][gdir]:
                    raise Refused("an extra draft pin file holds other frozen Opus sources than the first")
                for i, d in entry["draws"].items():
                    arm = f"generated-{tag}-{i}"
                    if arm in out[label]:
                        raise Refused("an extra draft pin file repeats a draft arm")
                    out[label][arm] = d["source_sha256"]
        return out
    except (OSError, ValueError, KeyError, TypeError):
        raise Refused("the draft pins cannot be read for the structural rule") from None


REAL_GRAPH_DIRS = {"uniprot": "uniprot-sp-2026_03", "rhea": "rhea-142", "opaque": "opaque-rhea-142"}


def graphs_from_args(args: argparse.Namespace, pins: Mapping[str, GraphPins],
                     generators: Mapping[str | None, Mapping[str, Generator]],
                     expected_sha: Mapping[str, Mapping[str, str]] | None = None,
                     require_sources: bool = True) -> list[GraphInput]:
    draft_sources = draft_source_args(args.draft_source)
    expected_sha = expected_sha or {}
    trees = _by_graph(args.graph, "--graph")
    shards = _by_graph(args.shards, "--shards")
    banks = _by_graph(args.bank, "--bank")
    where = {field: _by_graph(getattr(args, field), "--" + field.replace("_", "-"))
             for field in ("questions", "cluster_map", "tier_questions")}
    if any(len(v) != 1 for v in (*trees.values(), *shards.values(), *(v for w in where.values() for v in w.values()))):
        raise Refused("--graph, --shards and the file locations are given once per graph")
    unknown = sorted((set(trees) | set(shards) | set(banks) | {g for g in generators if g}
                      | {g for w in where.values() for g in w}) - set(trees))
    if unknown or not trees:
        raise Refused(f"every graph named needs a --graph entry; unlisted: {unknown}")
    out = []
    for name, (tree,) in trees.items():
        if name not in pins:
            raise Refused(f"graph {name} has no pinned inputs")
        if name not in shards:
            raise Refused(f"graph {name} has no --shards")
        pin = pins[name]
        for field, paths in where.items():
            if name in paths:
                if field == "tier_questions" and pin.tier_questions_sha256 is None:
                    raise Refused(f"graph {name} has no pinned tier file")
                pin = dataclasses.replace(pin, **{field: Path(paths[name][0])})
        sources = draft_sources.get(name)
        if require_sources and not sources:
            raise Refused(f"graph {name}: the structural sensitivity rule needs --draft-source for every draft")
        out.append(GraphInput(name, Path(tree), tuple(s for s in shards[name][0].split(",") if s), pin,
                              tuple(Path(b) for b in banks.get(name, ())), generators.get(name) or None,
                              {a: Path(f) for a, f in sources.items()} if sources else None,
                              expected_sha.get(name) if sources else None))
    return out


def main(argv: Sequence[str] | None = None, *, pins: Mapping[str, GraphPins] | None = None) -> int:
    """0: written; 2: refused (the message names counts, file names and labels); 3: an unexpected error (class only)."""
    try:
        args = build_parser().parse_args(argv)
        agents = dict(parse_agent(a) for a in args.agent)
        if len(agents) != len(args.agent):
            raise Refused("an agent is listed twice")
        gens: dict[str | None, dict[str, Generator]] = defaultdict(dict)
        for text in args.generator:
            graph, gen = parse_generator(text)
            if gen.name in gens[graph]:
                raise Refused("a generator is listed twice")
            gens[graph][gen.name] = gen
        drops = []
        for graph, arms in _by_graph(args.drop, "--drop").items():
            drops += [(graph, arm) for arm in arms]
        check_output_location(args.out, args.results_root)  # before any computation
        real = pins is None
        if real:
            check_pin("draft pins", args.draft_pins, SPEC_DRAFT_PINS_SHA256)  # before the pins are read for their hashes
        extra = []
        for item in args.extra_draft_pins:
            path, sep, want = item.rpartition("=")
            if not (sep and path and re.fullmatch(r"[0-9a-f]{64}", want)):
                raise Refused("--extra-draft-pins must be FILE=SHA256")
            check_pin("extra draft pins", Path(path), want)
            extra.append(Path(path))
        expected = pinned_draft_shas(args.draft_pins, REAL_GRAPH_DIRS, extra) if real else None
        graphs = graphs_from_args(args, real_pins(ROOT, args.config) if real else pins, gens, expected,
                                  require_sources=real)
        results = analyse(graphs, agents, gens.get(None, {}), config=args.config, config_sha256=args.config_sha256,
                          spec=args.spec, spec_sha256=args.spec_sha256, drops=drops, draft_pins=args.draft_pins,
                          extra_draft_pins=extra,
                          real_graph=real)
        sha = write_results(results, args.out, args.results_root)
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
