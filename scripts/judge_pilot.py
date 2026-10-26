#!/usr/bin/env python3
"""Dev pilot of the LLM judge (design log, first deliverable; change-log entry 51).

Judges two dev populations with each agent's own model and reports accept or reject
against the truth, with the cost of every call:

* natural-cohort outputs from run records (the calibration round-1 records): per
  records file, a seeded sample of submitted outputs with a checked query, half right
  and half wrong by the scorer, each judged by the model that produced it;
* fault-bank items that carry a question text: a seeded, stratified sample of the
  bank's DEV items (the test split is dropped before any label is read), each item's
  query executed once on the store and judged by every model given.

    uv run python scripts/judge_pilot.py \\
        --records cal-r1/sonnet/records.tar.gz cal-r1/qwen/records.tar.gz \\
        --bank-dir bank-uniprot-provisional-v2 --questions .../questions.jsonl \\
        --endpoint http://node-1.cluster-a.example:7020/ --bank-models claude-sonnet-5-5 qwen3.8-27b \\
        --anthropic-key-file ~/.config/www-2027/anthropic_api_key \\
        --gateway-key-file ~/.config/www-2027/gateway_api_key --out OUT

Writes, under OUT: ``selection.json`` (seed, rule, chosen ids), ``store.json`` (the store's
triples per graph against the bank manifest's), ``bank-results.jsonl`` (each bank query's
shown result), ``judge-inputs.jsonl`` (every rendered input with its hash),
``judge-calls.jsonl`` (one judge-call record per call), ``truth.jsonl`` (joined after the
calls), ``summary.json`` and ``summary.md``, and ``judge-input-examples.md`` (the exact
input of one wrong agent output and one answer-changing bank item).

With ``--bank-all`` (the design ruling in the design log's follow-up round) every judgeable dev item
is judged instead of the sample (``judgeable_bank``: a query, a question text, no retired
operator), with no records needed; ``--concurrency NAME=N`` sets the calls in flight per model,
``--context-limit NAME=TOKENS`` refuses the run before any call when an input could overflow a
model's window, and ``--resume`` continues a run in the same OUT: the bank results are read
back, every call with a reply is kept, and failed or missing calls are made.

Keys are read from their files into this process's environment, never printed. Store
queries run one at a time with a client time-out of at most 290 s.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import threading
import time
from collections.abc import Collection, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import anthropic  # noqa: F401  (imported here, in the main thread: the judge threads never import concurrently)
import openai  # noqa: F401

from harness.agent.providers import StepFailure
from harness.agent.replay_check import CODE_ROOT, tree_sha256
from harness.baselines.empty import trigger_category
from harness.baselines.judge import (
    JUDGE_STATUS,
    JUDGE_VERSION,
    TEMPLATE,
    TEMPLATE_ID,
    JudgeItem,
    bank_judge_item,
    call_judge,
    judge_route,
    record_judge_item,
    render,
    shown_result,
    template_sha256,
)
from harness.checkloop.matrix import (
    BANK,
    EMPTY_RETRY_ARM,
    bank_kind,
    bank_rates,
    load_raw_records,
)
from harness.sparql import SparqlClient

PILOT_VERSION = "judge-pilot/1"
MAX_TIMEOUT_S = 290.0  # QLever 9bebc60 caps time-outs at 300 s (stores/OPERATIONS.md)
RESERVED_PORTS = (":7022",)  # the UniProt calibration server: never used here
GRAPH_COUNTS = "SELECT ?g (COUNT(*) AS ?n) WHERE { GRAPH ?g { ?s ?p ?o } } GROUP BY ?g ORDER BY ?g"
SAMPLE_RULE = ("within each stratum, items sorted by sha256('<seed>|<id>') (hex) and the first k taken; bank strata "
               "in the order listed, skipping an item whose base question is already chosen")
# Bank strata (the truth is the bank's own label): answer-changing items are wrong, controls
# (the untouched gold, answer-preserving rewrites) and answer-neutral mutations are right.
BANK_STRATA: tuple[tuple[str, int], ...] = (
    ("wrong/type_wrong/d2", 2), ("wrong/type_wrong/d4", 2), ("wrong/vocabulary_absent", 2),
    ("wrong/stale_in_data", 2), ("wrong/meaning", 2),
    ("right/untouched_gold", 4), ("right/answer_preserving_rewrite", 4), ("right/answer_neutral_mutation", 2),
)
# Rewrite operators retired after this provisional bank was built (design log: the redundant join's
# queries are artificial, every sampled join dangling); their controls are never drawn.
RETIRED_OPERATORS = frozenset({"rewrite_redundant_join"})
KEY_FILES = {"anthropic": ("ANTHROPIC_API_KEY",), "gateway": ("LLM_GATEWAY_API_KEY",)}
# Fewer characters per token than any BPE tokenizer gives on this text: the window check's upper bound.
CHARS_PER_TOKEN_FLOOR = 2


def _load_run_pilot():
    spec = importlib.util.spec_from_file_location("run_pilot", Path(__file__).resolve().parent / "run_pilot.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("run_pilot", module)
    spec.loader.exec_module(module)
    return module


def order_key(seed: int, ident: str) -> str:
    return hashlib.sha256(f"{seed}|{ident}".encode()).hexdigest()


# -- agent outputs ----------------------------------------------------------------------


def output_truth(rec: dict) -> str | None:
    """"right" or "wrong" by the scorer's exact match on the initial output; None when the
    output was not scored."""
    score = (rec.get("checker") or {}).get("initial", {}).get("score") if rec.get("checker") else rec.get("score")
    if not score or score.get("outcome") != "scored":
        return None
    return "right" if score.get("exact") else "wrong"


def judgeable(rec: dict) -> bool:
    """A dev, natural-cohort, scored, submitted output with a checked query, from a plain run
    (a checker cell's or a branch's record repeats its parent's initial answer: never drawn)."""
    if rec.get("checker") is not None:
        return False
    if rec.get("split") != "dev" or rec.get("condition") != "none" or output_truth(rec) is None:
        return False
    try:
        return record_judge_item(rec).query is not None
    except ValueError:
        return False


def sample_outputs(records: Sequence[dict], seed: int, per_stratum: int) -> list[dict]:
    """``per_stratum`` right and ``per_stratum`` wrong judgeable outputs (see SAMPLE_RULE)."""
    chosen = []
    for truth in ("right", "wrong"):
        pool = sorted((r for r in records if judgeable(r) and output_truth(r) == truth),
                      key=lambda r: order_key(seed, r["run_id"]))
        if len(pool) < per_stratum:
            raise SystemExit(f"only {len(pool)} {truth} judgeable outputs for {per_stratum} wanted")
        chosen += pool[:per_stratum]
    return chosen


# -- bank items -------------------------------------------------------------------------


def dev_items(items: Iterable[dict]) -> list[dict]:
    """The bank's dev split. Test items are dropped here, on their split alone, before any
    other field is read: test bank labels stay sealed until the freeze."""
    return [it for it in items if it["split"] == "dev"]


def bank_stratum(item: dict) -> str | None:
    """A dev item's stratum (BANK_STRATA), or None (rejected, or outside every stratum)."""
    bank = item["bank"]
    if bank == "answer_changing":
        fc, d = item.get("fault_class"), item.get("distance")
        return f"wrong/{fc}/d{d}" if fc == "type_wrong" else f"wrong/{fc}"
    if bank == "control":
        return f"right/{item.get('control')}"
    if bank == "answer_neutral":
        return "right/answer_neutral_mutation"
    if bank == "descriptive":  # change-log entry 63: outside every estimand set, wrong when execution changed the answer
        label = (item.get("answer_effect") or {}).get("label")
        return f"wrong/{item.get('fault_class')}/{item.get('stratum')}" if label == "answer_changing" else None
    return None


def bank_truth(item: dict) -> str:
    stratum = bank_stratum(item)
    if stratum is None:
        raise ValueError(f"bank item {item['id']} has no truth for the pilot")
    return stratum.split("/", 1)[0]


def judgeable_bank(items: Iterable[dict], questions: dict[str, dict],
                   operators: Collection[str] | None = None) -> list[dict]:
    """Every dev item the judge can see, in the file's order: a query, a question text, and
    no retired operator (with ``operators``, only those operators' items). The test split is
    dropped first, on its split alone (dev_items)."""
    return [it for it in dev_items(items) if it.get("query") and it["base_question_id"] in questions
            and isinstance(questions[it["base_question_id"]].get("question"), str)
            and it.get("operator") not in RETIRED_OPERATORS
            and (operators is None or it.get("operator") in operators)]


def judged_kind(item: dict) -> str:
    """The analysis pipeline's kind (bank_kind), or "descriptive" for a descriptive stratum's
    item, which is in none of the checked bank's kinds (BANK_ITEMS)."""
    return "descriptive" if item["bank"] == "descriptive" else bank_kind(item)


def sample_bank(items: Iterable[dict], questions: dict[str, dict], seed: int,
                strata: Sequence[tuple[str, int]] = BANK_STRATA) -> list[dict]:
    """The stratified dev sample: items with a query and a question text, of no retired
    operator, from distinct base questions (see SAMPLE_RULE)."""
    dev = judgeable_bank(items, questions)
    used: set[str] = set()
    chosen = []
    for stratum, k in strata:
        pool = sorted((it for it in dev if bank_stratum(it) == stratum), key=lambda it: order_key(seed, it["id"]))
        picked = []
        for it in pool:
            if it["base_question_id"] not in used:
                picked.append(it)
                used.add(it["base_question_id"])
            if len(picked) == k:
                break
        if len(picked) < k:
            raise SystemExit(f"bank stratum {stratum}: {len(picked)} items for {k} wanted")
        chosen += picked
    return chosen


# -- results ----------------------------------------------------------------------------


def summarise(calls: Sequence[dict], truths: dict[str, str], result_rows: dict[str, int | None] | None = None,
              item_kinds: dict[str, tuple[str, str | None]] | None = None) -> dict[str, Any]:
    """Per (judge model, item kind): verdicts against the truth, and the cost per call. With
    ``result_rows`` (bank item key -> the rows its query returned when the bank was built), each
    bank cell's rates are also split by whether the query returns rows, beside retry on
    empty's on the same items (the registered empty-result stratum: design ruling 4 of
    change-log entry 59); a call that failed has no verdict and is left out of the split."""
    out: dict[str, Any] = {}
    for c in calls:
        cell = out.setdefault(c["judge_model"], {}).setdefault(c["item"]["kind"], {
            "n": 0, "table": {t: {"accept": 0, "reject": 0, "unparsed": 0, "error": 0} for t in ("right", "wrong")},
            "usd": 0.0, "priced": 0, "usd_unknown": 0, "input_tokens": 0, "output_tokens": 0})
        truth = truths[c["item"]["key"]]
        j = c.get("judge")
        verdict = "error" if j is None else (j["verdict"] or "unparsed")
        cell["n"] += 1
        cell["table"][truth][verdict] += 1
        if j is not None:
            usd = (j.get("cost") or {}).get("usd")
            if usd is None:
                cell["usd_unknown"] += 1
            else:
                cell["usd"] += usd
                cell["priced"] += 1
            cell["input_tokens"] += j["usage"].get("input_tokens") or 0
            cell["output_tokens"] += j["usage"].get("output_tokens") or 0
    for kinds in out.values():
        for cell in kinds.values():
            t = cell["table"]
            wrong_rejected, right_rejected = t["wrong"]["reject"], t["right"]["reject"]
            n_wrong, n_right = sum(t["wrong"].values()), sum(t["right"].values())
            cell["reject_recall"] = wrong_rejected / n_wrong if n_wrong else None
            cell["false_reject_rate"] = right_rejected / n_right if n_right else None
            rejected = wrong_rejected + right_rejected
            cell["reject_precision"] = wrong_rejected / rejected if rejected else None
            cell["usd"] = round(cell["usd"], 6)
            cell["usd_per_call"] = round(cell["usd"] / cell["priced"], 6) if cell["priced"] else None
    if result_rows is not None:
        for model, cells in out.items():
            if "bank_item" in cells:
                cells["bank_item"].update(_by_result(calls, truths, result_rows, model, item_kinds or {}))
    return out


def _by_result(calls: Sequence[dict], truths: dict[str, str], result_rows: dict[str, int | None],
               model: str, kinds: dict[str, tuple[str, str | None]]) -> dict[str, Any]:
    def row(key: str, arm: str, status: str) -> dict:
        bank_item, fault_class = kinds.get(key, (None, None))
        return {"run_id": key, "cohort": BANK, "output": True, "arm": arm, "variant": "primary", "status": status,
                "correct": truths[key] == "right", "result_rows": result_rows[key], "bank_item": bank_item,
                "fault_class": fault_class}

    judged = [c for c in calls if c["judge_model"] == model and c["item"]["kind"] == "bank_item" and c.get("judge")]
    rows = [row(c["item"]["key"], model, JUDGE_STATUS[c["judge"]["verdict"] or "unparsed"]) for c in judged]
    keys = sorted({c["item"]["key"] for c in judged})
    reference = [row(k, EMPTY_RETRY_ARM, "unknown" if result_rows[k] is None else
                     ("flagged" if result_rows[k] == 0 else "substantive_pass")) for k in keys]
    rates = bank_rates(rows + reference)
    return {"by_result": rates[model]["primary"], "empty_retry_reference": rates[EMPTY_RETRY_ARM]["primary"]}


def _fmt(x: Any) -> str:
    return f"{x:.2f}" if isinstance(x, float) else ("-" if x is None else str(x))


def summary_md(summary: dict, meta: dict) -> str:
    title = "# LLM judge on the full dev bank" if meta.get("mode") == "bank-all" else "# LLM judge dev pilot"
    lines = [title, "",
             (f"Template {meta['template']['id']} ({meta['template']['sha256']}); calls interactive (no Batch API "
              f"path in the harness); seed {meta['seed']}."), "",
             ("| Judge model | Items | N | right: accept / reject / unparsed / error | wrong: accept / reject / "
              "unparsed / error | reject recall | false-reject rate | reject precision | USD total | USD per call |"),
             "|---|---|---|---|---|---|---|---|---|---|"]
    for model, kinds in summary.items():
        for kind, c in kinds.items():
            r, w = c["table"]["right"], c["table"]["wrong"]
            per_call = "-" if c["usd_per_call"] is None else f"{c['usd_per_call']:.4f}"
            lines.append(f"| {model} | {kind} | {c['n']} | {r['accept']} / {r['reject']} / {r['unparsed']} / "
                         f"{r['error']} | {w['accept']} / {w['reject']} / {w['unparsed']} / {w['error']} | "
                         f"{_fmt(c['reject_recall'])} | {_fmt(c['false_reject_rate'])} | "
                         f"{_fmt(c['reject_precision'])} | {c['usd']:.4f} | {per_call} |")
    split = [(m, kinds["bank_item"]) for m, kinds in summary.items() if "by_result" in kinds.get("bank_item", {})]
    if split:
        lines += ["", ("Bank items by whether the query returns rows (the registered empty-result stratum): reject "
                       "recall on answer-changing items, false-reject rate on the others; retry on empty on the same "
                       "items is the reference."), "",
                  "| Judge model | Stratum | wrong | reject recall | right | false-reject rate | retry-on-empty recall |",
                  "|---|---|---|---|---|---|---|"]
        for model, c in split:
            for stratum in ("empty", "returns_rows", "no_row_count"):
                s, ref = c["by_result"].get(stratum), c["empty_retry_reference"].get(stratum)
                if s:
                    lines.append(f"| {model} | {stratum} | {s['W']} | {_fmt(s['detection_recall'])} | {s['C']} | "
                                 f"{_fmt(s['false_flag_rate'])} | {_fmt(ref['detection_recall'])} |")
    if meta.get("models"):
        lines += ["", "Calls made in this run, per judge (a resumed run counts only its own calls):", "",
                  "| Judge model | calls | in flight | failed | wall time, h | mean call, s |", "|---|---|---|---|---|---|"]
        for model, m in sorted(meta["models"].items()):
            done = m["calls"] - m["failed"]
            mean = f"{m['call_wall_s'] / done:.1f}" if done else "-"
            lines.append(f"| {model} | {m['calls']} | {m['concurrency']} | {m['failed']} | "
                         f"{m.get('wall_s', 0) / 3600:.2f} | {mean} |")
    total = sum(c["usd"] for kinds in summary.values() for c in kinds.values())
    unknown = sum(c["usd_unknown"] for kinds in summary.values() for c in kinds.values())
    failed = sum(c["table"][t]["error"] for kinds in summary.values() for c in kinds.values() for t in c["table"])
    lines += ["", (f"Total estimated cost: ${total:.4f} over the priced calls (list prices; Gateway models at no "
                   f"charge); calls with unknown cost: {unknown}; failed calls (no reply): {failed}."), ""]
    return "\n".join(lines)


def examples_md(examples: Sequence[tuple[str, dict]]) -> str:
    out = ["# Judge input examples", "",
           (f"The exact inputs the judge was sent ({TEMPLATE_ID}, template sha256 {template_sha256()}), rendered "
            "by the pilot's own path (scripts/judge_pilot.py). Each block is verbatim; the sha256 of each text is "
            "over its UTF-8 bytes, and the input sha256 is the one in the call record. Every call also carries "
            "the one tool below, through which the judge gives its verdict (part of the hashed template)."), "",
           "## The verdict tool (every call)", "", "````json", json.dumps(TEMPLATE["tool"], indent=1), "````", ""]
    for title, row in examples:
        out += [f"## {title}", "", f"- item: `{row['item']['key']}` (kind {row['item']['kind']})",
                f"- input sha256: `{row['input_sha256']}`",
                f"- system sha256: `{hashlib.sha256(row['system'].encode()).hexdigest()}`",
                f"- user sha256: `{hashlib.sha256(row['user'].encode()).hexdigest()}`", "", "### System", "",
                "````text", row["system"], "````", "", "### User", "", "````text", row["user"], "````", ""]
    return "\n".join(out)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_keys(anthropic_file: Path | None, gateway_file: Path | None) -> None:
    """Each key file into this process's environment (never printed or logged)."""
    for path, kind in ((anthropic_file, "anthropic"), (gateway_file, "gateway")):
        if path is None:
            continue
        value = Path(path).expanduser().read_text(encoding="utf-8").strip()
        if not value:
            raise SystemExit(f"the {kind} key file is empty")
        for env in KEY_FILES[kind]:
            os.environ[env] = value


def routes_by_model(models: dict) -> dict[str, tuple[str, Any]]:
    """Agent model id -> (MODELS name, route). Built after the keys are in the environment."""
    out = {}
    for name, spec in models.items():
        route = spec.factory()
        out[route.model] = (name, route)
    return out


def _pairs(specs: Sequence[str], what: str) -> dict[str, int]:
    """NAME=N arguments as {name: n}."""
    out = {}
    for spec in specs:
        name, sep, value = spec.partition("=")
        if not sep or not value.isdigit() or int(value) < 1:
            raise SystemExit(f"{what}: {spec!r} is not NAME=N with N >= 1")
        out[name] = int(value)
    return out


def estimated_tokens(rendered: dict) -> int:
    """An upper bound on an input's tokens: its characters (system, user and the verdict tool's
    schema) over CHARS_PER_TOKEN_FLOOR, fewer characters per token than any BPE tokenizer gives
    on English, SPARQL or result JSON."""
    chars = len(rendered["system"]) + len(rendered["user"]) + len(json.dumps(TEMPLATE["tool"]))
    return -(-chars // CHARS_PER_TOKEN_FLOOR)


def _latest(rows: Iterable[dict]) -> dict[tuple[str, str], dict]:
    """Per (item key, judge model): the last row with a reply, else the last failure."""
    out: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (row["item"]["key"], row["judge_model"])
        if row.get("judge") is not None or out.get(key, {}).get("judge") is None:
            out[key] = row
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--records", type=Path, nargs="*", default=[], help="records.tar.gz, one per agent model")
    p.add_argument("--bank-dir", type=Path, default=None,
                   help="the bank folder (items.jsonl, manifest.json); none: agent outputs only, no store")
    p.add_argument("--questions", type=Path, default=None, help="the bank manifest's questions file")
    p.add_argument("--endpoint", default=None, help="the store the bank was built on")
    p.add_argument("--timeout-s", type=float, default=MAX_TIMEOUT_S)
    p.add_argument("--row-cap", type=int, default=500, help="the agent runs' row cap (checked against the records)")
    p.add_argument("--bank-models", nargs="+", default=["claude-sonnet-5-5", "qwen3.8-27b"],
                   help="MODELS names that judge every bank item")
    p.add_argument("--bank-all", action="store_true",
                   help="judge every judgeable dev item (judgeable_bank), not the stratified sample")
    p.add_argument("--bank-operators", nargs="+", default=None, metavar="OPERATOR",
                   help="with --bank-all, only these operators' dev items (e.g. type_consistent_constant)")
    p.add_argument("--concurrency", nargs="+", default=[], metavar="NAME=N",
                   help="calls in flight per MODELS name (default 1)")
    p.add_argument("--context-limit", nargs="+", default=[], metavar="NAME=TOKENS",
                   help="refuse before any call if an input could overflow this model's context window")
    p.add_argument("--resume", action="store_true",
                   help="reuse OUT's bank results and every call with a reply; redo failed and missing calls")
    p.add_argument("--seed", type=int, default=20261019)
    p.add_argument("--per-stratum", type=int, default=5, help="right and wrong outputs per records file")
    p.add_argument("--outputs-all", action="store_true",
                   help="judge every judgeable output of each records file (judgeable), not the sample")
    p.add_argument("--output-models", nargs="+", default=None, metavar="NAME",
                   help="MODELS names that judge every output (default: each output by its own model)")
    p.add_argument("--anthropic-key-file", type=Path, default=None)
    p.add_argument("--gateway-key-file", type=Path, default=None)
    p.add_argument("--no-calls", action="store_true", help="select, execute and render only: no model call")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    if args.timeout_s > MAX_TIMEOUT_S:
        raise SystemExit(f"--timeout-s is capped at {MAX_TIMEOUT_S:g} s")
    if not args.records and args.bank_dir is None:
        raise SystemExit("nothing to judge: give --records, --bank-dir or both")
    if args.outputs_all and not args.records:
        raise SystemExit("--outputs-all needs --records")
    if args.bank_dir is not None and (args.questions is None or args.endpoint is None):
        raise SystemExit("--bank-dir needs --questions and --endpoint")
    if args.endpoint and any(port in args.endpoint for port in RESERVED_PORTS):
        raise SystemExit(f"{args.endpoint}: the calibration server is never used here")
    if args.bank_operators and not args.bank_all:
        raise SystemExit("--bank-operators applies to --bank-all only, not to the stratified sample")
    concurrency = _pairs(args.concurrency, "--concurrency")
    limits = _pairs(args.context_limit, "--context-limit")
    partial_path = args.out / "judge-calls.partial.jsonl"
    if partial_path.exists() and not args.resume:
        raise SystemExit(f"{partial_path} exists: pass --resume to continue that run, or use another --out")
    args.out.mkdir(parents=True, exist_ok=True)
    run_pilot = _load_run_pilot()
    read_keys(args.anthropic_key_file, args.gateway_key_file)
    routes = routes_by_model(run_pilot.MODELS)
    for name in args.output_models or []:
        if name not in run_pilot.MODELS:
            raise SystemExit(f"--output-models: {name!r} is not in run_pilot.MODELS")
    output_models = [run_pilot.MODELS[name].factory().model for name in args.output_models or []]

    # agent outputs: a seeded right/wrong sample (or, with --outputs-all, every judgeable output) per records
    # file, judged by its own model or by every --output-models model
    outputs: list[dict] = []
    for path in args.records:
        recs = [json.loads(raw) for raw in load_raw_records([path])]
        models = {r["model"] for r in recs}
        if len(models) != 1:
            raise SystemExit(f"{path}: records of {sorted(models)}; one model per file")
        model = models.pop()
        if model not in routes:
            raise SystemExit(f"{path}: no route in run_pilot.MODELS serves {model!r}")
        caps = {r["store"]["row_cap"] for r in recs}
        if caps != {args.row_cap}:
            raise SystemExit(f"{path}: row caps {sorted(caps)}, not {args.row_cap}")
        chosen = [r for r in recs if judgeable(r)] if args.outputs_all else \
            sample_outputs(recs, args.seed, args.per_stratum)
        for rec in chosen:
            outputs.append({"key": f"output:{rec['run_id']}", "record": rec, "model": model, "path": path})

    # bank items: the dev split only, then every judgeable item or a stratified sample; questions from the
    # manifest's file
    manifest, qsha, questions, bank = None, None, {}, []
    if args.bank_dir is not None:
        manifest = json.loads((args.bank_dir / "manifest.json").read_text(encoding="utf-8"))
        qsha = hashlib.sha256(args.questions.read_bytes()).hexdigest()
        if qsha != manifest["inputs"]["questions"]["sha256"]:
            raise SystemExit(f"{args.questions}: sha256 {qsha} is not the bank manifest's questions file")
        questions = {q["id"]: q for q in map(json.loads, args.questions.read_text(encoding="utf-8").splitlines())}
        with (args.bank_dir / "items.jsonl").open(encoding="utf-8") as f:
            items = dev_items(json.loads(line) for line in f)
        bank = (judgeable_bank(items, questions, args.bank_operators) if args.bank_all
                else sample_bank(items, questions, args.seed, BANK_STRATA))
    for name in args.bank_models:
        if name not in run_pilot.MODELS:
            raise SystemExit(f"--bank-models: {name!r} is not in run_pilot.MODELS")
    for name in list(concurrency) + list(limits):
        if name not in run_pilot.MODELS:
            raise SystemExit(f"{name!r} is not in run_pilot.MODELS")
    names = {routes[m][1].model: routes[m][0] for m in routes}  # served model id -> MODELS name
    bank_models = [run_pilot.MODELS[name].factory().model for name in args.bank_models] if bank else []
    selection = {"version": PILOT_VERSION, "mode": "bank-all" if args.bank_all else "pilot", "seed": args.seed,
                 "rule": ("every judgeable dev item" + (" of the operators listed" if args.bank_operators else "")
                          + ", in the file's order") if args.bank_all else SAMPLE_RULE,
                 "bank_strata": None if args.bank_all else BANK_STRATA,
                 "operators": sorted(args.bank_operators) if args.bank_operators else None,
                 "retired_operators": sorted(RETIRED_OPERATORS),
                 "outputs": [{"key": o["key"], "model": o["model"], "question_id": o["record"]["question_id"],
                              "records": str(o["path"]), "result": trigger_category(o["record"])}
                             for o in outputs],
                 "outputs_all": args.outputs_all, "output_models": output_models or None,
                 "bank": [{"key": f"bank:{it['id']}", "base_question_id": it["base_question_id"]} for it in bank],
                 "bank_models": bank_models}
    (args.out / "selection.json").write_text(json.dumps(selection, indent=1) + "\n")

    store: dict | None = None
    bank_rows: list[dict] = []
    if args.bank_dir is not None:
        # the store: its identity against the bank manifest's, then each bank query once, in turn (or, on a resumed
        # run, the results recorded for exactly these queries)
        client = SparqlClient(args.endpoint, timeout_s=args.timeout_s, row_cap=args.row_cap)
        t = time.monotonic()
        counts = {g.value: int(n.lexical) for g, n in client.query(GRAPH_COUNTS).rows}
        want = manifest["store"]["identity"]["graph_counts"]
        store = {"endpoint": args.endpoint, "graph_counts": counts, "matches_bank_manifest": counts == want,
                 "elapsed_s": time.monotonic() - t, "timeout_s": args.timeout_s, "row_cap": args.row_cap}
        (args.out / "store.json").write_text(json.dumps(store, indent=1, sort_keys=True) + "\n")
        if counts != want:
            raise SystemExit("the store's triples per graph differ from the bank manifest's: not the bank's store")
        results_path = args.out / "bank-results.jsonl"
        if args.resume and results_path.exists():
            bank_rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
            wanted = [(f"bank:{it['id']}", hashlib.sha256(it["query"].encode()).hexdigest()) for it in bank]
            if [(r["key"], r["query_sha256"]) for r in bank_rows] != wanted:
                raise SystemExit(f"{results_path} holds other items or queries than this run's: not resumable")
            print(f"bank results: {len(bank_rows)} read back from {results_path}", file=sys.stderr)
        else:
            bank_rows = []
            for it in bank:
                t = time.monotonic()
                shown = shown_result(client, it["query"])
                meta = json.loads(shown)
                bank_rows.append({"key": f"bank:{it['id']}",
                                  "query_sha256": hashlib.sha256(it["query"].encode()).hexdigest(),
                                  "shown": shown, "elapsed_s": round(time.monotonic() - t, 3),
                                  "total_rows": meta.get("total_rows"), "error": meta.get("error"),
                                  "bank_n_rows": (it.get("answer_effect") or {}).get("n_rows")})
                print(f"bank query {len(bank_rows)}/{len(bank)}: {meta.get('total_rows', meta.get('error'))} rows, "
                      f"{bank_rows[-1]['elapsed_s']}s", file=sys.stderr)
            write_jsonl(results_path, bank_rows)
        # every bank item must run as it did when the bank was built: a fresh error, a row count other than
        # the bank's own, or no recorded count to compare with, and its label could be about another result
        bad = [r["key"] for r in bank_rows if r["error"] is not None or r["bank_n_rows"] is None
               or r["total_rows"] != r["bank_n_rows"]]
        if bad:
            raise SystemExit(f"bank items that did not re-run as recorded (no model was called): {bad}")

    # render every input (the judge sees these items only)
    jobs: list[tuple[dict, JudgeItem, str]] = []  # (item ref, judge item, judging model)
    for o in outputs:
        item = record_judge_item(o["record"])
        for model in output_models or [o["model"]]:
            jobs.append(({"key": o["key"], "kind": "agent_output"}, item, model))
    shown_by_key = {r["key"]: r["shown"] for r in bank_rows}
    for it in bank:
        key = f"bank:{it['id']}"
        item = bank_judge_item(it, questions[it["base_question_id"]], shown_by_key[key], row_cap=args.row_cap)
        for model in bank_models:
            jobs.append(({"key": key, "kind": "bank_item"}, item, model))
    inputs = {}
    for ref, item, _ in jobs:
        r = render(item)
        inputs[ref["key"]] = {"item": ref, "input_sha256": r.input_sha256, "system": r.system, "user": r.user}
    write_jsonl(args.out / "judge-inputs.jsonl", inputs.values())

    # the context-window check: every input of a model, at an upper bound on its tokens, plus the route's
    # output cap, must fit the model's window, or no call is made
    context = {}
    for name, limit in limits.items():
        model = run_pilot.MODELS[name].factory()
        cap = model.route_info()["settings"]["default_max_tokens"]
        sizes = [estimated_tokens(inputs[ref["key"]]) for ref, _, m in jobs if m == model.model]
        worst = max(sizes, default=0)
        context[model.model] = {"limit": limit, "output_cap": cap, "max_input_tokens_upper_bound": worst,
                                "chars_per_token_floor": CHARS_PER_TOKEN_FLOOR, "inputs": len(sizes)}
        if worst + cap > limit:
            raise SystemExit(f"{name}: an input of up to {worst} tokens plus the {cap}-token output cap could "
                             f"overflow the model's context window of {limit} tokens (no model was called)")

    # the calls: per judging model a pool of --concurrency workers, models in parallel; a resumed run keeps
    # every call that has a reply and makes the rest
    previous = [json.loads(line) for line in partial_path.read_text(encoding="utf-8").splitlines()] \
        if partial_path.exists() else []
    done = {k for k, row in _latest(previous).items() if row.get("judge") is not None}
    todo = [j for j in jobs if (j[0]["key"], j[2]) not in done]
    calls: list[dict] = list(previous)
    lock = threading.Lock()
    stop = threading.Event()
    stops: list[str] = []
    timing: dict[str, dict] = {}
    partial = partial_path.open("a", encoding="utf-8")  # nothing spent is lost

    def run(model: str) -> None:
        name = names[model]
        local = threading.local()
        mine = [j for j in todo if j[2] == model]
        n = concurrency.get(name, 1)
        timing[model] = {"name": name, "concurrency": n, "calls": len(mine), "call_wall_s": 0.0, "failed": 0}

        def one(job: tuple[dict, JudgeItem, str]) -> None:
            ref, item, _ = job
            if stop.is_set():
                return
            if not hasattr(local, "route"):  # one route per worker thread
                local.route = judge_route(run_pilot.MODELS[name].factory())
            row = {"version": PILOT_VERSION, "item": ref, "judge_model": model, "models_name": name}
            try:
                row["judge"] = call_judge(local.route, item)
            except StepFailure as exc:
                row["judge"] = None
                row["failure"] = {"type": exc.failure_type, "message": exc.message[:500], "details": exc.details}
            except Exception as exc:  # noqa: BLE001 - recorded and counted; the other calls go on
                row["judge"] = None
                row["failure"] = {"type": "crash", "message": repr(exc)[:500], "details": None}
            j = row["judge"]
            with lock:
                if j is not None and j["input_sha256"] != inputs[ref["key"]]["input_sha256"]:
                    stops.append(f"{ref['key']}: the input sent is not the input rendered")
                    stop.set()
                if j is not None and j["same_model"] is False:
                    stops.append(f"{ref['key']}: {j['served_model']!r} answered for {model!r}")
                    stop.set()
                calls.append(row)
                partial.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                partial.flush()
                timing[model]["call_wall_s"] += j["wall_s"] if j else 0.0
                timing[model]["failed"] += j is None
                note = f", {j['wall_s']:.0f} s, {j['retries']} retries" if j else ""
                print(f"[{name}] {ref['key'][:60]}: {j['verdict'] if j else 'error'} "
                      f"({j['parse_status'] if j else row['failure']['type']}{note})", file=sys.stderr, flush=True)

        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=n) as pool:
            list(pool.map(one, mine))
        timing[model]["wall_s"] = round(time.monotonic() - t0, 3)
        timing[model]["call_wall_s"] = round(timing[model]["call_wall_s"], 3)

    if not args.no_calls:
        with ThreadPoolExecutor(max_workers=max(1, len({m for _, _, m in jobs}))) as pool:
            for f in [pool.submit(run, m) for m in sorted({m for _, _, m in jobs})]:
                f.result()
    partial.close()
    if stops:
        raise SystemExit(f"stopped: {stops[0]}")
    order = {(ref["key"], m): i for i, (ref, _, m) in enumerate(jobs)}
    latest = _latest(calls)
    calls = sorted((row for k, row in latest.items() if k in order), key=lambda c: order[(c["item"]["key"],
                                                                                          c["judge_model"])])
    write_jsonl(args.out / "judge-calls.jsonl", calls)

    # the truth, joined only now
    truths = {o["key"]: output_truth(o["record"]) for o in outputs}
    truths.update({f"bank:{it['id']}": bank_truth(it) for it in bank})
    strata = {f"bank:{it['id']}": bank_stratum(it) for it in bank}
    write_jsonl(args.out / "truth.jsonl", [{"key": k, "truth": v, "stratum": strata.get(k)} for k, v in truths.items()])
    meta = {"version": PILOT_VERSION, "judge_version": JUDGE_VERSION, "mode": selection["mode"], "seed": args.seed,
            "template": {"id": TEMPLATE_ID, "sha256": template_sha256()}, "code_tree_sha256": tree_sha256(CODE_ROOT),
            "records": [str(p) for p in args.records], "bank_dir": str(args.bank_dir) if args.bank_dir else None,
            "bank_manifest_sha256": hashlib.sha256((args.bank_dir / "manifest.json").read_bytes()).hexdigest()
            if args.bank_dir else None,
            "questions_sha256": qsha,
            "store": {k: store[k] for k in ("endpoint", "matches_bank_manifest")} if store else None,
            "calls": len(calls), "failed_calls": sum(1 for c in calls if c.get("judge") is None),
            "batch_api": False, "models": timing, "context": context, "resumed": bool(previous)}
    summary = summarise(calls, truths, {f"bank:{it['id']}": (it.get("answer_effect") or {}).get("n_rows")
                                        for it in bank},
                        {f"bank:{it['id']}": (judged_kind(it), it.get("fault_class")) for it in bank})
    (args.out / "summary.json").write_text(json.dumps({"meta": meta, "summary": summary}, indent=1) + "\n")
    (args.out / "summary.md").write_text(summary_md(summary, meta))

    # the exact input of one wrong agent output and one answer-changing bank item, where there are some
    examples = []
    wrong_output = next((o["key"] for o in outputs if truths[o["key"]] == "wrong"), None)
    if wrong_output:
        examples.append(("An agent output (wrong by the scorer)", inputs[wrong_output]))
    changing = next((f"bank:{it['id']}" for it in bank if bank_truth(it) == "wrong"), None)
    if changing:
        examples.append(("A bank item (answer-changing)", inputs[changing]))
    (args.out / "judge-input-examples.md").write_text(examples_md(examples))
    print(f"done: {len(calls)} judge calls, {len(bank_rows)} bank queries; summary in {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
