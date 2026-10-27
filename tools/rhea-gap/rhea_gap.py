#!/usr/bin/env python3
"""Rhea natural vs opaque Rhea: the accuracy gap per model and prompt arm, with a PAIRED cluster bootstrap, beside each
arm's configuration shares on both graphs (26 Oct 2026).

For every model and every prompt arm (cohort, use "prompt") that BOTH graphs have, the difference

    accuracy(opaque Rhea) - accuracy(Rhea natural)

with a 95% percentile interval over R = 10,000 draws of the SHARED clusters (seed 20261026), each draw taken once and
applied to both graphs, so the two accuracies of a draw rest on the same resampled clusters (paired). Opaque Rhea 142's
base_question_id is the natural question's id; the two cluster maps assign the same cluster to every base id (checked);
the draws come from the NATURAL graph's cluster map (the design log's, ClusterMap.resampling, as the analysis takes them).

Where the numbers come from (nothing is re-scored, nothing is implemented again)
- results.json of a look holds per-arm estimates and intervals, never the per-question values. This tool therefore runs the
  looks' own loading procedure again on the same inputs (the cells files that drove the looks, pinned by sha256): the pinned
  exploratory.py's prepare() (its voided, partition-union, configuration and data-source checks included), then the frozen
  metrics.build_arms / question_vectors / metric_of("accuracy"): per question the mean over its attempts of the recorded
  correctness (a missing attempt counts wrong), then the mean over questions (prereg §3). The scorer is the one that
  wrote each record's correctness; nothing is scored here.
- The looks' results.json is read by this tool at run time as a CHECK, never printed: the recomputed accuracy estimate and
  interval of every prompt arm must equal the look's exactly, the arm sets must be equal, the OUT records' sha256 must be
  the look's, and the look's configuration shares must cover exactly the recomputed attempts. A cell
  that fails any check is not paired (its pairs are refused and listed).
- Configuration shares are the look's shares section of each prompt arm.

Stages (one process per cell, so the load check is passed between cells; the cell files stay private, mode 600)
    rhea_gap.py cell    --plan PLAN --name NAME --analysis-dir D --code-root C --owner-root R --out CELLDIR [--status F]
    rhea_gap.py combine --plan PLAN --analysis-dir D --code-root C --cells-dir CELLDIR --out OUT [--status F]
Neither prints a result: stdout carries counts and booleans only.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

CELL_SCHEMA = "rhea-gap-cell/1"
TABLE_SCHEMA = "rhea-gap-table/1"
STATUS_SCHEMA = "rhea-gap-status/1"
# The pinned analysis script every B look ran (the same bytes in both look directories); any other
# is refused
EXPLORATORY_SHA256 = "d97cf45489fa33a4f810cf9d527928ed3073936bac9b1104f14d3fd701777508"
NAT, OPQ = "rhea-142", "opaque-rhea-142"
SEED, N_BOOT, LEVEL = 20261026, 10_000, 0.95
SPLIT = "test"
PROMPT = "prompt"
# The unions of the looks' driver (its union allowlist): prefix -> (K, graph, pinned record sha256,
# binding). A union cell is the whole union ("A"), never a subset
UNIONS: dict[str, tuple[int, str, str | None, bool]] = {
    "opaque-qwen3.8-27b-run_group": (6, OPQ, None, False),
    "opaque-deepseek-v4-flash-0731-deepseek_b": (
        6, OPQ, "98158f127c0c7545d5c7255cfa7760dc6b9ff6a9d3b80b7929354cb003dc1cb7", True),
    "opaque-qwen3-coder-30b-a3b-deepseek": (
        2, OPQ, "639bb65af66828821bb774bf9e65b6194510fe343587c145a1615fb66b856714", False),
    "rhea-qwen3.8-27b-run_group": (5, NAT, None, False),
    "opaque-gemma-4-31b-run_group": (3, OPQ, None, False),
}
# The models both graphs hold, as the records name them (the looks' cells; the request is per model): combine refuses a
# set of cells that is not exactly these on BOTH graphs, so a model named differently on the two graphs cannot land silently
# among the unpaired
EXPECTED_MODELS = ("Qwen/Qwen3-Coder-30B-A3B-Instruct", "Qwen/Qwen3.8-27B", "claude-haiku-5-5", "claude-opus-5-5",
                   "claude-sonnet-5-5", "deepseek-ai/DeepSeek-V4-Flash-0731", "google/gemma-4-31B", "zai-org/GLM-5.3-Flash")
BINDING_SHA256 = "015aa5e74a391b266db346f480978f46d7f706aad2d022aa48160f173f14326b"
PLAN_SHA256 = ("0d0cf40446fcc60976a66bca893a3730589814c313afcb92c43d553b45cc20cc",
               "7efb050424205cb849b1c0eabe495a2f00ac65b79d6b1bcad8953ea44b85267d",
               "18f9bd250b7d3df31ef5cf88f4d4424ec7ec36d0fad5ad0aa3e302c353373bca",
               "6b5cff90edc2e4000cf90b3439c29db557aef04fdf9ca4ddf736ba7efb1c9c88",
               "d4a045074cdec610e350cf0b439f9dc0087eb0427fc18d19b3b48c4ceb40fcd7",
               "671bc3743ede5f89b78bb294abff53cc0ca71302b7f6a9027ba0b1fce37e5ac7")
STAMP_RE = re.compile(r"-(\d{8}T\d{6}Z)\.txt$")
PATH_RE = re.compile(r"^/[A-Za-z0-9._/+@=-]+$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class GapRefused(Exception):
    """The run may not go on; the message names the structural reason (never a result)."""


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --- the cells files of the looks (the inputs) ----------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class UnionSpec:
    prefix: str
    k: int
    record: str
    record_sha256: str
    binding: str | None


@dataclasses.dataclass(frozen=True)
class CellSpec:
    name: str
    graph: str
    outs: tuple[str, ...]
    questions: tuple[str, ...]
    store_records: tuple[str, ...]
    fronts: tuple[str, ...]
    need_gb: int
    cap_gb: int
    union: UnionSpec | None


def _path_list(field: str, what: str, *, required: bool) -> tuple[str, ...]:
    if field == "":
        if required:
            raise GapRefused(f"cells line: {what} is empty")
        return ()
    items = field.split(",")
    for p in items:
        if not PATH_RE.match(p) or "//" in p or "/../" in p or p.endswith("/"):
            raise GapRefused(f"cells line: {what} has an entry that is not a clean absolute file name")
    if len(set(items)) != len(items):
        raise GapRefused(f"cells line: {what} repeats an entry")
    return tuple(items)


def parse_cell_line(line: str) -> CellSpec:
    """One line of a cells file of the looks' driver (13 fields) or of its union variant (14 fields), strictly."""
    f = line.rstrip("\n").split("|")
    if len(f) not in (13, 14):
        raise GapRefused(f"cells line has {len(f)} fields, not 13 (a single OUT) or 14 (a union)")
    name, graph, outlist, q, ledger, a15, need, cap, store_records, fronts, phases = f[:11]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        raise GapRefused("cells line: the cell name is not a plain name")
    if graph not in (NAT, OPQ):
        raise GapRefused(f"cells line {name}: graph {graph!r} is neither {NAT} nor {OPQ}")
    if ledger != "1" or a15 != "0":
        raise GapRefused(f"cells line {name}: not a ledger=1, a15=0 cell")
    if phases not in ("natural", "prompted", "natural,prompted"):
        raise GapRefused(f"cells line {name}: phases {phases!r}")
    if not (need.isdigit() and cap.isdigit()):
        raise GapRefused(f"cells line {name}: need and cap are not integers")
    outs = _path_list(outlist, "the OUT list", required=True)
    qs = _path_list(q, "the questions", required=True)
    pf = _path_list(store_records, "the check records", required=False)
    fr = _path_list(fronts, "the request logs", required=False)
    union = None
    if len(f) == 13:
        if len(outs) != 1:
            raise GapRefused(f"cells line {name}: a single-OUT line lists {len(outs)} OUTs")
    else:
        parts = f[13].split(":")
        if len(parts) not in (4, 5):
            raise GapRefused(f"cells line {name}: the union spec has {len(parts)} parts")
        prefix, k, record, rsha = parts[:4]
        binding = parts[4] if len(parts) == 5 else None
        if prefix not in UNIONS:
            raise GapRefused(f"cells line {name}: union prefix {prefix!r} is not an allowlisted union")
        want_k, want_graph, want_rsha, want_binding = UNIONS[prefix]
        if not k.isdigit() or int(k) != want_k or graph != want_graph or len(outs) != want_k:
            raise GapRefused(f"cells line {name}: union K, graph or OUT count differs from the allowlist")
        for p, o in enumerate(outs):
            if o.rsplit("/", 1)[-1] != f"{prefix}-p{p}of{want_k}":
                raise GapRefused(f"cells line {name}: OUT {p} is not {prefix}-p{p}of{want_k}")
        if not HEX64.match(rsha) or (want_rsha is not None and rsha != want_rsha):
            raise GapRefused(f"cells line {name}: the partition record pin differs from the union's")
        if not PATH_RE.match(record) or record.rsplit("/", 1)[-1] in ("binding.json",) \
                or record.rsplit("/", 1)[-1].startswith("plan-p"):
            raise GapRefused(f"cells line {name}: the partition record file")
        if want_binding != (binding is not None) or (binding is not None and not PATH_RE.match(binding)):
            raise GapRefused(f"cells line {name}: the binding directory differs from the union's")
        union = UnionSpec(prefix, want_k, record, rsha, binding)
    return CellSpec(name, graph, outs, qs, pf, fr, int(need), int(cap), union)


@dataclasses.dataclass(frozen=True)
class PlanRow:
    cells_file: str
    cells_sha256: str
    name: str


def read_plan(path: Path) -> list[PlanRow]:
    rows = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) != 3 or not PATH_RE.match(f[0]) or not HEX64.match(f[1]) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", f[2]):
            raise GapRefused(f"plan line {n}: not <cells file>\\t<sha256>\\t<name>")
        rows.append(PlanRow(*f))
    names = [r.name for r in rows]
    if not rows or len(set(names)) != len(names):
        raise GapRefused("the plan is empty or repeats a cell name")
    return rows


def load_spec(row: PlanRow) -> tuple[CellSpec, str]:
    """The plan row's cells line (by name), the cells file checked by its pinned sha256; also the look's stamp."""
    path = Path(row.cells_file)
    raw = path.read_bytes()
    if sha256_bytes(raw) != row.cells_sha256:
        raise GapRefused(f"{path.name}: sha256 differs from the plan's")
    m = STAMP_RE.search(path.name)
    if not m:
        raise GapRefused(f"{path.name}: no look stamp in the file name")
    lines = [ln for ln in raw.decode("utf-8").splitlines() if ln.split("|", 1)[0] == row.name]
    if len(lines) != 1:
        raise GapRefused(f"{path.name}: {len(lines)} lines for cell {row.name}")
    return parse_cell_line(lines[0]), m.group(1)


def results_path(owner_root: Path, stamp: str, spec: CellSpec) -> Path:
    base = Path(owner_root) / f"look-{stamp}" / spec.name
    return base / "A" / "results.json" if spec.union else base / "results.json"


# --- loading the pinned analysis -------------------------------------------------------------------------------------


def load_exploratory(analysis_dir: Path) -> Any:
    analysis_dir = Path(analysis_dir).resolve()
    path = analysis_dir / "exploratory.py"
    if sha256_file(path) != EXPLORATORY_SHA256:
        raise GapRefused(f"{path}: not the exploratory.py the B looks ran (sha256 {sha256_file(path)[:12]})")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(analysis_dir))
    import exploratory  # noqa: PLC0415  (the pinned script, in place: it reads its config files from beside itself)
    if Path(exploratory.__file__).resolve() != path:
        raise GapRefused("exploratory was imported from another location")
    if (exploratory.SEED, exploratory.N_BOOT, exploratory.LEVEL) != (SEED, N_BOOT, LEVEL):
        raise GapRefused("the pinned exploratory.py has other seed, R or level than 20261026, 10,000, 0.95")
    return exploratory


# --- one cell ----------------------------------------------------------------------------------------------------------


def build_call(X: Any, spec: CellSpec, voided: Path) -> Any:
    """The looks' call (the driver and its union variant) without the ledgers and replay selection, which the prompt
    arms do not depend on."""
    kw: dict[str, Any] = dict(batches=tuple(Path(o) for o in spec.outs), questions=tuple(Path(q) for q in spec.questions),
                              graph=spec.graph, split=SPLIT, voided=(Path(voided),),
                              front_logs=tuple(Path(p) for p in spec.fronts), store_proofs=tuple(Path(p) for p in spec.store_records))
    u = spec.union
    if u is not None:
        record = Path(u.record)
        if not record.is_file() or record.is_symlink() or sha256_file(record) != u.record_sha256:
            raise GapRefused(f"{spec.name}: the partition record is missing or its sha256 differs")
        kw["partition_record"] = record
        if u.binding is not None:
            b = Path(u.binding)
            files = [b / "binding.json", *(b / f"plan-p{p}.json" for p in range(u.k))]
            pins = (BINDING_SHA256, *PLAN_SHA256[: u.k])
            for p, pin in zip(files, pins, strict=True):
                if not p.is_file() or p.is_symlink() or sha256_file(p) != pin:
                    raise GapRefused(f"{spec.name}: a partition binding or plan file is missing or differs from its pin")
            kw["partition_binding"] = files[0]
            kw["partition_plans"] = tuple(files[1:])
    return X.Call(**kw)


def share_tied(shares: Any, n_units: int) -> bool:
    """The look's configuration shares of an arm cover exactly the recomputed attempts: attempts with a record plus those
    without are the arm's units, and the rows' counts add up to the attempts with a record."""
    try:
        a, w, rows = shares["attempts"], shares["without record"], shares["rows"]
        counts = [r["n"] for r in rows.values()]
        return (a + w == n_units and sum(counts) == a and all(r["of"] == a for r in rows.values()))
    except (KeyError, TypeError, AttributeError):
        return False


def arm_checks(estimate: float | None, ci: list | None, n_units: int, look_arm: Any, look_shares: Any) -> dict[str, bool]:
    """The recomputed arm against the look's results.json (booleans only)."""
    try:
        acc = look_arm["metrics"]["accuracy"]
        return {"estimate_equals_look": acc["estimate"] == estimate, "ci_equals_look": acc["ci"] == ci,
                "n_att_equals_look": look_arm["counts"]["N_att"] == n_units,
                "shares_tied": share_tied(look_shares, n_units)}
    except (KeyError, TypeError):
        return {"estimate_equals_look": False, "ci_equals_look": False, "n_att_equals_look": False,
                "shares_tied": False}


def arm_accuracy(F: Any, units: Any, index: Any) -> tuple[Any, np.ndarray, np.ndarray, float | None]:
    """An arm's per-question accuracy by the frozen code: question_vectors, then metric_of("accuracy") (prereg §3: per
    question the mean over its attempts of the recorded correctness, a missing attempt counting wrong; the denominator 1 for
    a question with units) and the frozen point estimate."""
    vec = F.metrics.question_vectors(units, index)
    num, den = F.metrics.metric_of(vec, "accuracy")
    return vec, num, den, F.metrics.point_estimate(num, den)


def compact_shares(shares: dict) -> dict:
    rows = {d: {"n": r["n"], "share": r["share"]} for d, r in sorted(shares["rows"].items())}
    return {"attempts": shares["attempts"], "without_record": shares["without record"], "rows": rows}


def cell_main(args: argparse.Namespace) -> int:
    t0 = time.time()
    row = next((r for r in read_plan(args.plan) if r.name == args.name), None)
    if row is None:
        raise GapRefused(f"{args.name}: not in the plan")
    spec, stamp = load_spec(row)
    X = load_exploratory(args.analysis_dir)
    F = X.load_frozen(args.code_root)
    cfg, _ = X.frozen_config(F)
    owner = Path(args.owner_root)
    voided = owner / f"look-{stamp}" / "inputs" / "voided.jsonl"
    look_path = results_path(owner, stamp, spec)
    look_raw = look_path.read_bytes()
    look = json.loads(look_raw)
    loaded, _union, context = X.prepare(F, build_call(X, spec, voided), cfg=cfg)
    K, _clusters = X.cluster_counts(F, context["cluster_map"], loaded.index, context["cfg"])
    index = loaded.index
    arms = {k: u for k, u in X.attempt_arms(F, loaded).items() if k.use == F.metrics.PROMPT}
    models = {k.model for k in arms}
    if len(models) != 1 or not arms:
        raise GapRefused(f"{spec.name}: {len(arms)} prompt arms over {len(models)} models (one model per cell)")
    (model,) = models
    look_arms = look["arms"]
    look_labels = sorted(lab for lab, a in look_arms.items()
                         if a["key"]["use"] == PROMPT and a["key"]["cohort"] != F.inputs.BANK)
    out_arms: dict[str, Any] = {}
    for key, units in sorted(arms.items()):
        vec, num, den, est = arm_accuracy(F, units, index)
        ci = marginal_ci(F.stats, K, num, den)
        checks = arm_checks(est, ci, len(units), look_arms.get(key.label), look["deployments"]["shares"].get(key.label))
        out_arms[key.cohort] = {
            "label": key.label, "n_units": len(units), "n_questions_with_units": int((vec["n"] > 0).sum()),
            "estimate": est, "ci": ci, "num": [float(x) for x in num], "den": [float(x) for x in den],
            "shares": compact_shares(look["deployments"]["shares"][key.label]) if checks["shares_tied"] else None,
            "checks": checks}
    mine = sorted(a["label"] for a in out_arms.values())
    got = sorted((i["batch"], i["records_sha256"]) for i in look["settings"]["inputs"])
    have = sorted((i["batch"], i["records_sha256"]) for i in loaded.settings["inputs"])
    cell_checks = {"arm_set_equals_look": mine == look_labels, "records_sha256_equal_look": got == have,
                   "n_questions_equal_look": look["settings"]["n_questions"] == len(index)}
    ok = all(cell_checks.values()) and all(all(a["checks"].values()) for a in out_arms.values())
    doc = {
        "schema": CELL_SCHEMA, "name": spec.name, "graph": spec.graph, "model": model, "stamp": stamp,
        "union": None if spec.union is None else {"prefix": spec.union.prefix, "k": spec.union.k},
        "plan_row": dataclasses.asdict(row), "look_results_sha256": sha256_bytes(look_raw),
        "voided_sha256": sha256_file(voided), "exploratory_sha256": EXPLORATORY_SHA256,
        "tool_sha256": sha256_file(Path(__file__)), "code": F.identity,
        "inputs": [{"batch": i["batch"], "records_sha256": i["records_sha256"]} for i in loaded.settings["inputs"]],
        "index": {"questions": list(index.questions), "strata": list(index.strata),
                  "cluster": [context["cluster_map"].clusters[q] for q in index.questions]},
        "arms": out_arms, "checks": cell_checks, "ok": ok}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    write_private(out / f"{spec.name}.json", dump(doc))
    n_checks = len(cell_checks) + sum(len(a["checks"]) for a in out_arms.values())
    print(f"cell {spec.name}: {len(out_arms)} prompt arms, {len(index)} questions, {n_checks} checks, ok={ok}")
    write_status(args, {"stage": "cell", "name": spec.name, "rc": 0 if ok else 3, "ok": ok, "arms": len(out_arms),
                        "checks": cell_checks, "arm_checks_all_true": all(all(a["checks"].values()) for a in out_arms.values()),
                        "wall_s": round(time.time() - t0), "maxrss_mb": maxrss_mb()})
    return 0 if ok else 3


# --- combine: the paired bootstrap --------------------------------------------------------------------------------------


def point(x: np.ndarray, d: np.ndarray) -> float:
    return float(x.sum()) / float(d.sum())


def shared_K(X: Any, F: Any, cfg: Any, nat_map: Any, questions: list[str], strata: list[str]) -> np.ndarray:
    """The question-level resample weights of the natural graph's cluster map over the natural index: ClusterMap.resampling's
    B unstratified draws of the clusters (seed and R from the frozen config), each question taking its cluster's column."""
    index = F.stats.QuestionIndex(tuple(questions), tuple(strata))
    clusters = nat_map.resampling(index, cfg)
    return clusters.matrix[:, clusters.group]


def paired_gap(stats: Any, K: np.ndarray, x_nat: np.ndarray, d_nat: np.ndarray, x_opq: np.ndarray, d_opq: np.ndarray,
               level: float = LEVEL) -> dict[str, Any]:
    """opaque minus natural accuracy: the point estimate, and the percentile interval of the per-draw differences, each
    draw's two ratios taken on the SAME row of K (paired). A draw where either ratio is undefined is dropped from the
    interval (and counted in undefined_share), as the frozen interval does."""
    nat = stats.ratio_draws(K, x_nat, d_nat)
    opq = stats.ratio_draws(K, x_opq, d_opq)
    iv = stats.interval(opq - nat, level, "two")
    ci = [iv["lo"], iv["hi"]] if iv["lo"] is not None else None
    return {"estimate": point(x_opq, d_opq) - point(x_nat, d_nat), "ci": ci, "undefined_share": iv["undefined_share"]}


def marginal_ci(stats: Any, K: np.ndarray, x: np.ndarray, d: np.ndarray, level: float = LEVEL) -> list | None:
    iv = stats.interval(stats.ratio_draws(K, x, d), level, "two")
    return [iv["lo"], iv["hi"]] if iv["lo"] is not None else None


def align(nat_q: list[str], opq_q: list[str], opq_cluster: list[str], nat_cluster: list[str]) -> np.ndarray:
    """The positions of the natural index's questions in the opaque index. Refuses unless the two graphs have the same base
    questions, each once, in the same clusters."""
    if len(set(nat_q)) != len(nat_q) or len(set(opq_q)) != len(opq_q):
        raise GapRefused("a graph's index repeats a base question")
    if set(nat_q) != set(opq_q):
        raise GapRefused(f"the graphs' base questions differ ({len(set(nat_q) - set(opq_q))} only natural, "
                         f"{len(set(opq_q) - set(nat_q))} only opaque)")
    where = {q: i for i, q in enumerate(opq_q)}
    perm = np.array([where[q] for q in nat_q], dtype=np.intp)
    if [opq_cluster[i] for i in perm] != list(nat_cluster):
        raise GapRefused("the two cluster maps assign different clusters to a shared base question")
    return perm


def pair_arms(stats: Any, K: np.ndarray, nat_cell: dict, opq_cell: dict, cohort: str,
              not_configurations: Any = ()) -> dict[str, Any]:
    """One model x cohort pair from two cell documents."""
    ni, oi = nat_cell["index"], opq_cell["index"]
    perm = align(ni["questions"], oi["questions"], oi["cluster"], ni["cluster"])
    if K.shape[1] != len(ni["questions"]):
        raise GapRefused("the resample matrix does not match the natural index")
    na, oa = nat_cell["arms"][cohort], opq_cell["arms"][cohort]
    x_nat, d_nat = np.array(na["num"]), np.array(na["den"])
    x_opq, d_opq = np.array(oa["num"])[perm], np.array(oa["den"])[perm]
    if not ((d_nat > 0).all() and (d_opq > 0).all()):
        raise GapRefused(f"{nat_cell['model']} / {cohort}: a shared question has no units in an arm")
    gap = paired_gap(stats, K, x_nat, d_nat, x_opq, d_opq)
    same_order = ni["questions"] == oi["questions"]
    return {
        "model": nat_cell["model"], "arm": cohort, "n_questions": len(ni["questions"]),
        "natural": {"cell": nat_cell["name"], "estimate": na["estimate"], "ci": marginal_ci(stats, K, x_nat, d_nat),
                    "n_units": na["n_units"], "partitions": (nat_cell["union"] or {}).get("k"), "shares": na["shares"]},
        "opaque": {"cell": opq_cell["name"], "estimate": oa["estimate"], "ci": marginal_ci(stats, K, x_opq, d_opq),
                   "n_units": oa["n_units"], "partitions": (opq_cell["union"] or {}).get("k"), "shares": oa["shares"]},
        "difference_opaque_minus_natural": {**gap, "ci_excludes_zero": bool(gap["ci"] and (gap["ci"][0] > 0 or gap["ci"][1] < 0))},
        "configurations_on_both_graphs": sorted((set(na["shares"]["rows"]) & set(oa["shares"]["rows"])) - set(not_configurations)),
        "_same_order": same_order,
    }


def group_cells(docs: Any) -> dict[str, dict[str, dict]]:
    """Cell documents by graph, then model; two cells of one model on one graph would count its attempts twice."""
    by_graph: dict[str, dict[str, dict]] = {NAT: {}, OPQ: {}}
    for doc in docs:
        if doc["graph"] not in by_graph:
            raise GapRefused(f"{doc['name']}: graph {doc['graph']!r}")
        if doc["model"] in by_graph[doc["graph"]]:
            raise GapRefused(f"two cells for {doc['model']} on {doc['graph']}")
        by_graph[doc["graph"]][doc["model"]] = doc
    return by_graph


def check_models(by_graph: dict[str, dict[str, dict]], expected: Any = EXPECTED_MODELS) -> None:
    """Both graphs hold exactly the expected models: a missing, an extra or a differently named model refuses."""
    problems = []
    for graph, models in by_graph.items():
        missing, extra = sorted(set(expected) - set(models)), sorted(set(models) - set(expected))
        if missing:
            problems.append(f"{graph} lacks {missing}")
        if extra:
            problems.append(f"{graph} has unexpected {extra}")
    if problems:
        raise GapRefused("the models of the cells are not the expected set: " + "; ".join(problems))


def build_pairs(stats: Any, K: np.ndarray, by_graph: dict[str, dict[str, dict]],
                not_configurations: Any = ()) -> tuple[list, list, list, list, dict]:
    """Every model x prompt arm both graphs have, as a pair; what only one graph has, listed; a pair that cannot be made
    safely (a cell whose look cross-check failed, different base questions or clusters, a shared question without units)
    listed as refused with its reason, never computed."""
    pairs, unpaired_models, unpaired_arms, refused, repro = [], [], [], [], {}
    for model in sorted(set(by_graph[NAT]) | set(by_graph[OPQ])):
        n, o = by_graph[NAT].get(model), by_graph[OPQ].get(model)
        for graph, d, other in ((NAT, n, o), (OPQ, o, n)):
            if d is not None and other is None:
                unpaired_models.append({"model": model, "graph": graph, "cell": d["name"]})
        if n is None or o is None:
            continue
        bad = [d["name"] for d in (n, o) if not d["ok"]]
        if bad:
            refused.append({"model": model, "reason": f"the look cross-check failed for cell(s) {', '.join(bad)}"})
            continue
        for cohort in sorted(set(n["arms"]) ^ set(o["arms"])):
            unpaired_arms.append({"model": model, "arm": cohort, "graph": NAT if cohort in n["arms"] else OPQ})
        for cohort in sorted(set(n["arms"]) & set(o["arms"])):
            try:
                pair = pair_arms(stats, K, n, o, cohort, not_configurations)
            except GapRefused as exc:
                refused.append({"model": model, "reason": f"{cohort}: {exc}"})
                continue
            # the shared K reproduces the interval each look computed on its own index (natural always; opaque where
            # its index order is the natural one)
            same = pair.pop("_same_order")
            repro[f"{model}/{cohort}"] = {
                "natural_ci_reproduced": pair["natural"]["ci"] == n["arms"][cohort]["ci"],
                "opaque_ci_reproduced": (pair["opaque"]["ci"] == o["arms"][cohort]["ci"]) if same else None}
            pairs.append(pair)
    return pairs, unpaired_models, unpaired_arms, refused, repro


def fmt_pct(v: float | None) -> str:
    return "n/a" if v is None else f"{100 * v:.1f}"


def render_text(table: dict) -> str:
    lines = [f"Rhea natural vs opaque Rhea: accuracy gap (opaque - natural), paired cluster bootstrap",
             f"R = {table['bootstrap']['n_boot']}, seed {table['bootstrap']['seed']}, {table['bootstrap']['level'] * 100:g}% percentile "
             f"interval; draws of the shared clusters ({table['bootstrap']['n_clusters']} clusters over "
             f"{table['bootstrap']['n_questions']} base questions), one draw applied to both graphs.",
             "Accuracies and gaps in percent (gap in percentage points); each interval in brackets.", ""]
    head = f"{'model':30s} {'arm':12s} {'natural':>22s} {'opaque':>22s} {'gap (opaque - natural)':>28s}"
    lines += [head, "-" * len(head)]
    for p in table["pairs"]:
        n, o, g = p["natural"], p["opaque"], p["difference_opaque_minus_natural"]
        ci = lambda c: "n/a" if c is None else f"[{fmt_pct(c[0])}, {fmt_pct(c[1])}]"  # noqa: E731
        lines.append(f"{p['model']:30s} {p['arm']:12s} {fmt_pct(n['estimate']):>6s} {ci(n['ci']):>15s} "
                     f"{fmt_pct(o['estimate']):>6s} {ci(o['ci']):>15s} {100 * g['estimate']:>+8.1f} {ci(g['ci']):>18s}"
                     + ("  *" if g["ci_excludes_zero"] else ""))
        for side, d in (("natural", n), ("opaque", o)):
            sh = d["shares"]
            rows = ", ".join(f"{dep} {fmt_pct(r['share'])}% ({r['n']})" for dep, r in sh["rows"].items()) or "none"
            extra = f"; {sh['without_record']} without record" if sh["without_record"] else ""
            lines.append(f"    configurations, {side:8s} ({sh['attempts']} attempts{extra}): {rows}")
        lines.append("    on both graphs: " + (", ".join(p["configurations_on_both_graphs"]) or "none"))
    lines += ["", "* the paired interval excludes zero."]
    if table["unpaired_models"] or table["unpaired_arms"]:
        lines += ["", "Not paired (a graph lacks the model or the arm):"]
        lines += [f"  model {u['model']} only on {u['graph']} (cell {u['cell']})" for u in table["unpaired_models"]]
        lines += [f"  {u['model']} / {u['arm']} only on {u['graph']}" for u in table["unpaired_arms"]]
    if table["refused_pairs"]:
        lines += ["", "Refused (not computed):"] + [f"  {r['model']}: {r['reason']}" for r in table["refused_pairs"]]
    return "\n".join(lines) + "\n"


def combine_main(args: argparse.Namespace) -> int:
    t0 = time.time()
    plan = read_plan(args.plan)
    X = load_exploratory(args.analysis_dir)
    F = X.load_frozen(args.code_root)
    cfg, cfg_record = X.frozen_config(F)
    cells: dict[str, dict] = {}
    for row in plan:
        p = Path(args.cells_dir) / f"{row.name}.json"
        if not p.is_file():
            raise GapRefused(f"cell file for {row.name} is missing")
        doc = json.loads(p.read_bytes())
        if doc.get("schema") != CELL_SCHEMA or doc.get("name") != row.name:
            raise GapRefused(f"{row.name}: not a {CELL_SCHEMA} document of that name")
        if doc.get("tool_sha256") != sha256_file(Path(__file__)):
            raise GapRefused(f"{row.name}: written by another tool version")
        cells[row.name] = doc
    by_graph = group_cells(cells.values())
    check_models(by_graph)
    nat_docs = list(by_graph[NAT].values())
    if not nat_docs:
        raise GapRefused("no natural cell")
    first = nat_docs[0]["index"]
    if any(d["index"]["questions"] != first["questions"] or d["index"]["strata"] != first["strata"] for d in nat_docs):
        raise GapRefused("the natural cells do not share one question index")
    nat_map, nat_map_record = X.cluster_map_for(F, cfg, NAT)
    K = shared_K(X, F, cfg, nat_map, first["questions"], first["strata"])
    n_clusters = len(set(first["cluster"]))
    # the real configuration rows only, as compare_shares counts them (mixed, unattributed, no own turns and a retry without record are
    # not configurations)
    not_configurations = (*X.load_attribution().NOT_DEPLOYMENTS, X.RETRY_WITHOUT_RECORD)
    pairs, unpaired_models, unpaired_arms, refused, repro = build_pairs(F.stats, K, by_graph, not_configurations)
    reproduced = all(r["natural_ci_reproduced"] and r["opaque_ci_reproduced"] is not False for r in repro.values())
    table = {
        "schema": TABLE_SCHEMA,
        "what": "accuracy of opaque Rhea minus Rhea natural per model and prompt arm both graphs have, paired cluster "
                "bootstrap over the shared base questions; configuration shares of each arm beside it",
        "bootstrap": {"method": "paired: one draw of the shared clusters per resample, applied to both graphs; percentile "
                                "interval (frozen stats.interval, inverted-CDF quantiles) of the per-draw differences",
                      "n_boot": N_BOOT, "seed": SEED, "level": LEVEL, "n_clusters": n_clusters,
                      "n_questions": len(first["questions"]), "cluster_map": nat_map_record,
                      "resample_matrix_sha256": sha256_bytes(np.ascontiguousarray(K).tobytes()),
                      "config_sha256": cfg_record["config_sha256"]},
        "accuracy": "prereg §3 'accuracy': per question the mean over its attempts of the recorded correctness (a missing "
                    "attempt counts wrong), then the mean over questions; the frozen metrics.question_vectors/metric_of",
        "shares": "the shares section of the looks: per arm, the share of its "
                  "attempts per configuration",
        "pairs": sorted(pairs, key=lambda p: (p["model"], p["arm"])),
        "unpaired_models": unpaired_models, "unpaired_arms": unpaired_arms, "refused_pairs": refused,
        "inputs": {"plan_sha256": sha256_file(Path(args.plan)),
                   "cells": [{"name": c["name"], "graph": c["graph"], "model": c["model"], "look": f"look-{c['stamp']}",
                              "cells_file_sha256": c["plan_row"]["cells_sha256"], "look_results_sha256": c["look_results_sha256"],
                              "voided_sha256": c["voided_sha256"], "partitions": (c["union"] or {}).get("k"),
                              "outs": c["inputs"]} for c in cells.values()]},
        "code": {"tool_sha256": sha256_file(Path(__file__)), "exploratory_sha256": EXPLORATORY_SHA256,
                 "frozen": next(iter(cells.values()))["code"]},
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    write_private(out / "table.json", dump(table))
    write_private(out / "table.txt", render_text(table))
    ok = (not refused) and reproduced and all(d["ok"] for d in cells.values())
    status = {"stage": "combine", "rc": 0 if ok else 3, "ok": ok, "cells": len(cells), "natural_cells": len(by_graph[NAT]),
              "opaque_cells": len(by_graph[OPQ]), "pairs": len(pairs), "unpaired_models": len(unpaired_models),
              "unpaired_arms": len(unpaired_arms), "refused_pairs": [r["reason"] for r in refused],
              "shared_K_reproduces_looks_intervals": reproduced,
              "opaque_index_same_order_as_natural": sorted(k for k, v in repro.items() if v["opaque_ci_reproduced"] is not None),
              "cells_not_ok": sorted(d["name"] for d in cells.values() if not d["ok"]),
              "table_json_sha256": sha256_file(out / "table.json"), "table_txt_sha256": sha256_file(out / "table.txt"),
              "wall_s": round(time.time() - t0), "maxrss_mb": maxrss_mb()}
    write_status(args, status)
    print(f"combine: {len(cells)} cells, {len(pairs)} pairs, {len(refused)} refused, {len(unpaired_models)} unpaired models, "
          f"{len(unpaired_arms)} unpaired arms, K reproduces the looks' intervals={reproduced}, ok={ok}")
    return 0 if ok else 3


# --- small io ----------------------------------------------------------------------------------------------------------


def dump(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, indent=1, ensure_ascii=False, allow_nan=False) + "\n"


def write_private(path: Path, text: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)


def maxrss_mb() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def write_status(args: argparse.Namespace, doc: dict) -> None:
    if getattr(args, "status", None):
        line = json.dumps({"schema": STATUS_SCHEMA, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **doc},
                          sort_keys=True)
        fd = os.open(args.status, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("cell", "combine"):
        s = sub.add_parser(name)
        s.add_argument("--plan", type=Path, required=True)
        s.add_argument("--analysis-dir", type=Path, required=True)
        s.add_argument("--code-root", type=Path, required=True)
        s.add_argument("--out", type=Path, required=True)
        s.add_argument("--status", type=Path)
        if name == "cell":
            s.add_argument("--name", required=True)
            s.add_argument("--owner-root", type=Path, required=True)
        else:
            s.add_argument("--cells-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    try:
        return cell_main(args) if args.cmd == "cell" else combine_main(args)
    except GapRefused as exc:  # this tool's own refusals name structure only
        print(f"refused: {exc}", file=sys.stderr)
        write_status(args, {"stage": args.cmd, "rc": 2, "refused": str(exc)[:300]})
        return 2
    except BaseException as exc:  # noqa: BLE001  anything else: the class and the raising frame only, as look_wrap does
        last = traceback.extract_tb(exc.__traceback__)[-1]
        write_status(args, {"stage": args.cmd, "rc": 1, "error": f"{type(exc).__name__} {os.path.basename(last.filename)}:{last.lineno}"})
        return 1


if __name__ == "__main__":
    sys.exit(main())
