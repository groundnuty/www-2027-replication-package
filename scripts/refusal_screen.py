#!/usr/bin/env python3
"""Refusal screen (design log; change-log entry 34, amending 23): a diagnostic of provider refusals on dev questions.

Entry 34: the screen sends DEVELOPMENT questions only (probing test questions before the freeze would observe
model responses on test data) and EXCLUDES NOTHING (excluding by model- and prompt-dependent refusals would
redefine the population). It reports refusal counts by prompt condition and model. In the test runs every
question stays in the primary analysis (a refusal is a scored failure); the secondary analysis excludes a base
question refused in any compared arm of the test runs themselves, never on this screen's evidence.

    uv run python scripts/refusal_screen.py --dry-run
    uv run python scripts/refusal_screen.py --out results/refusal-screen-<date>/ [--graphs ...] [--models ...]

One probe per (graph, question, condition, model): the condition's system prompt (data/prompts/<graph>/, the
selectable entries only, each checked against its manifest and its block in the store) and the question text as
the only user message, with no tools and max output --max-tokens (1,024), sent through the harness's own route.
Every development question of the ruled question sets (QUESTION_SETS: the file's sha256 must start with the
ruled prefix), both tiers; test questions are dropped before ordering and counted in the manifest, and a task
list holding one is refused; every condition (none, curated, census, generated); every model of the panel. A
graph without prompts, or a condition without a selectable prompt, stops the screen before any call.

Primary label (``refused``): the provider's own signal only.
* Anthropic: stop_reason "refusal" (its stop_details category and explanation are recorded);
* OpenAI-compatible: finish_reason "content_filter", or the message's ``refusal`` field;
* an HTTP error whose message names a content or safety policy (POLICY_ERROR).
Secondary flag (``keyword_flag``), reported apart and never used for the primary label: the answer text holds
one of REFUSAL_KEYWORDS (case-insensitive substrings), the fixed list below.

Writes under --out: rows.jsonl (one row per probe, appended as it finishes: the signal, the finish reason, the
first 300 characters of text, the usage), summary.json (refusal counts per graph, condition and model; the
refused and keyword-flagged questions listed as a diagnostic, never as an exclusion; cache use per model) and
manifest.json (the screen's inputs). A rerun into the same --out skips probes already answered or refused,
sends errored ones again, and is refused if the inputs differ. The order is seeded (--seed) and the same for
every model; each model has its own pool (--concurrency, default 4, Anthropic 3; --model-concurrency
NAME=N). Keys are read from the environment at run time and never written. Anthropic routes cache the system
prompt for an hour (the route's default), so a condition's prompt is read from the cache after its first probe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_pilot import MODELS as RUN_MODELS  # noqa: E402

from harness.agent.prompts import get_condition, load_block, sha256_text, user_message  # noqa: E402
from harness.agent.providers import StepFailure  # noqa: E402
from harness.agent.record import redact_secrets, redact_structure  # noqa: E402
from harness.sources.tokens import count_tokens  # noqa: E402

SCHEMA = "refusal-screen/1"
GRAPHS = ("rhea-142", "uniprot-sp-2026_03")
CONDITIONS = ("none", "curated", "census", "generated")
# Asked for by name only (--conditions): the aligned curated arm, where a graph has one (a15).
KNOWN_CONDITIONS = CONDITIONS + ("aligned-curated",)
# The ruled question sets (design log): each question file's sha256 must start with its prefix. The frozen round-2 sets:
# Rhea feat/qgen b50e77d (replacing round 1's 00e84760), UniProt feat/qgen-uniprot 0b247dd (replacing round 1's
# b805acd9).
QUESTION_SETS = {"rhea-142": "b204846e", "uniprot-sp-2026_03": "17eef284"}
# Where a run reads each graph's questions without --questions: the ruled bytes, committed apart
# (data/questions-frozen/), because this branch's qgen tests still read the older data/questions/<graph>/ files.
QUESTION_FILES = {g: ROOT / "data" / "questions-frozen" / g / "questions.jsonl" for g in QUESTION_SETS}
MAX_TOKENS = 1024
SEED = 20261018
TEXT_HEAD = 300
POLICY_ERROR = re.compile(r"content[ _-]?(policy|filter|management|moderation)|safety|moderation|usage polic"
                          r"|flagged|harmful|prohibited content", re.IGNORECASE)
# The secondary flag's fixed list (lower case; matched as case-insensitive substrings of the answer text).
REFUSAL_KEYWORDS = (
    "i can't help", "i cannot help", "i can't assist", "i cannot assist", "i can't provide", "i cannot provide",
    "i'm not able to help", "i am not able to help", "i'm unable to", "i am unable to", "i won't", "i will not",
    "i must decline", "i'm sorry, but i can't", "i'm sorry, but i cannot", "against my guidelines",
    "not able to comply",
)


@dataclass(frozen=True)
class ModelSpec:
    factory: Callable[[int], Any]  # max output tokens -> route
    key_envs: tuple[str, ...] = ()
    concurrency: int = 4


def _from_table(name: str, concurrency: int) -> ModelSpec:
    """The runs' route from the one model table (scripts/run_pilot.py MODELS: route model, transport, effort and
    caching), with only the screen's output budget (design log: one table)."""
    spec = RUN_MODELS[name]

    def build(max_tokens: int) -> Any:
        route = spec.factory()
        route.max_tokens = max_tokens
        return route

    return ModelSpec(build, spec.key_envs, concurrency)


# The prompted panel P (change-log entry 82): Llama-3.3-70B is replaced by Qwen3-Coder-30B-A3B and runs exploratory in
# the natural cohort only, so it has no prompted cohort to screen; DeepSeek joins when design log decides. Gateway models
# screen 4 at a time, Claude models 3.
MODELS: dict[str, ModelSpec] = {
    "claude-sonnet-5-5": _from_table("claude-sonnet-5-5", 3),
    "qwen3.8-27b": _from_table("qwen3.8-27b", 4),
    "glm-5.3-flash": _from_table("glm-5.3-flash", 4),
    "gemma-4-31b": _from_table("gemma-4-31b", 4),
    "claude-haiku-5-5": _from_table("claude-haiku-5-5", 3),
    "qwen3-coder-30b-a3b": _from_table("qwen3-coder-30b-a3b", 4),
}


# --- inputs ------------------------------------------------------------------------------------------------


def default_questions(graph: str) -> Path:
    return QUESTION_FILES.get(graph, ROOT / "data" / "questions" / graph / "questions.jsonl")


def load_questions(path: Path, expect_prefix: str, graph: str) -> tuple[list[dict], str]:
    if not path.exists():
        raise SystemExit(f"{graph}: question file {path} does not exist")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if not sha.startswith(expect_prefix):
        raise SystemExit(f"{graph}: {path} has sha256 {sha[:16]}, not the ruled question set {expect_prefix}")
    questions = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [q["id"] for q in questions]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{graph}: {path} repeats question ids")
    return questions, sha


def load_prompts(prompts_dir: Path, conditions: tuple[str, ...], block_dir: Path | None = None) -> dict[str, dict]:
    """Each condition's system prompt, from its graph's prompt manifest: selectable entries only, the file
    matching the manifest's hash, and an ontology prompt equal to none-v1 + its stored block."""
    path = prompts_dir / "manifest.json"
    if not path.exists():
        raise SystemExit(f"{prompts_dir.name}: no prompts ({path} does not exist); build them first "
                         "(scripts/build_prompts.py)")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    none_entry = manifest["conditions"].get("none")
    none_text = (prompts_dir / none_entry["prompt"]["file"]).read_text(encoding="utf-8") if none_entry else None
    out = {}
    for c in conditions:
        entry = manifest["conditions"].get(c)
        if entry is None:
            why = manifest.get("skipped", {}).get(c)
            raise SystemExit(f"{path}: no {c!r} prompt" + (f" ({why})" if why else ""))
        if c != "none" and entry.get("selectable") is not True:
            raise SystemExit(f"{path}: the {c!r} prompt is not selectable")
        text = (prompts_dir / entry["prompt"]["file"]).read_text(encoding="utf-8")
        if sha256_text(text) != entry["prompt"]["sha256"]:
            raise SystemExit(f"{path}: the {c!r} prompt file does not match its manifest")
        if c != "none" and text != none_text + load_block(entry["block"]["sha256"], block_dir):
            raise SystemExit(f"{path}: the {c!r} prompt is not none-v1 + its stored block")
        get_condition(c)  # a known condition
        out[c] = {"text": text, "sha256": entry["prompt"]["sha256"]}
    return out


def assert_dev_only(questions: dict[str, dict[str, dict]]) -> None:
    """Change-log entry 34: no test-split question may be screened before the freeze."""
    bad = sorted(f"{g}:{qid}" for g, qs in questions.items() for qid, q in qs.items() if q.get("split") != "dev")
    if bad:
        raise SystemExit(f"refusing to screen test-split (or unsplit) questions before the freeze: {bad[:5]}"
                         + (f" and {len(bad) - 5} more" if len(bad) > 5 else ""))


def task_order(graphs: list[str], question_ids: dict[str, list[str]], conditions: tuple[str, ...],
               seed: int) -> list[tuple[str, str, str]]:
    tasks = [(g, q, c) for g in graphs for q in question_ids[g] for c in conditions]
    random.Random(seed).shuffle(tasks)
    return tasks


# --- one probe ---------------------------------------------------------------------------------------------


def keyword_hits(text: str) -> list[str]:
    low = (text or "").lower()
    return [k for k in REFUSAL_KEYWORDS if k in low]


def probe(route, system: str, question: str, max_tokens: int) -> dict:
    """One tool-free single-turn request, classified by the provider's own signal."""
    transcript = [{"role": "user", "kind": "question", "text": question}]
    try:
        step = route.step(system, transcript, (), max_tokens=max_tokens)
    except StepFailure as exc:
        message = redact_secrets(exc.message)[0][:500]
        details = exc.details or {}
        if POLICY_ERROR.search(exc.message):
            return {"status": "refused", "refused": True, "finish_reason": None, "text_head": "",
                    "signal": {"kind": "http_error", "value": details.get("status"), "category": None,
                               "explanation": message},
                    "keyword_flag": False, "keyword_hits": [], "usage": None, "served_model": None}
        return {"status": "error", "refused": False, "signal": None, "finish_reason": None, "text_head": "",
                "keyword_flag": False, "keyword_hits": [], "usage": None, "served_model": None,
                "error": {"failure_type": exc.failure_type, "error_class": details.get("error_class"),
                          "http_status": details.get("status"), "message": message}}
    except Exception as exc:  # noqa: BLE001 - an unexpected client error is recorded and sent again on a rerun
        return {"status": "error", "refused": False, "signal": None, "finish_reason": None, "text_head": "",
                "keyword_flag": False, "keyword_hits": [], "usage": None, "served_model": None,
                "error": {"failure_type": "other", "error_class": type(exc).__name__, "http_status": None,
                          "message": redact_secrets(str(exc))[0][:500]}}
    signal = None
    if step.stop_reason == "refusal":
        signal = {"kind": "stop_reason", "value": "refusal", "category": (step.refusal or {}).get("category"),
                  "explanation": (step.refusal or {}).get("explanation")}
    elif step.stop_reason == "content_filter":
        signal = {"kind": "finish_reason", "value": "content_filter", "category": None, "explanation": None}
    elif step.refusal is not None:  # an OpenAI-compatible message's own refusal field
        signal = {"kind": "message_refusal", "value": step.stop_reason, "category": None,
                  "explanation": step.refusal.get("explanation")}
    text = step.text or ""
    hits = keyword_hits(text)
    return {"status": "refused" if signal else "answered", "refused": signal is not None, "signal": signal,
            "finish_reason": step.stop_reason, "text_head": text[:TEXT_HEAD], "keyword_flag": bool(hits),
            "keyword_hits": hits, "usage": step.usage, "served_model": step.model}


# --- the summary --------------------------------------------------------------------------------------------


def key(row: dict) -> tuple:
    return row["graph"], row["question_id"], row["condition"], row["model"]


def read_rows(path: Path) -> dict[tuple, dict]:
    latest: dict[tuple, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                latest[key(row)] = row  # the latest row of a probe counts
    return latest


def summarize(latest: dict[tuple, dict]) -> dict:
    counts: dict[tuple, dict] = {}
    prone: dict[tuple, list] = defaultdict(list)
    worded: dict[tuple, list] = defaultdict(list)
    cache: dict[str, dict] = defaultdict(lambda: {"input_tokens": 0, "cache_read_tokens": 0,
                                                  "cache_write_tokens": 0, "output_tokens": 0})
    for row in latest.values():
        cell = counts.setdefault((row["graph"], row["condition"], row["model"]), {
            "graph": row["graph"], "condition": row["condition"], "model": row["model"], "probes": 0, "refused": 0,
            "errors": 0, "keyword_flagged": 0})
        cell["probes"] += 1
        cell["refused"] += row["refused"]
        cell["errors"] += row["status"] == "error"
        cell["keyword_flagged"] += row["keyword_flag"]
        where = {"condition": row["condition"], "model": row["model"]}
        if row["refused"]:
            prone[(row["graph"], row["question_id"])].append({**where, "signal": row["signal"]})
        if row["keyword_flag"]:
            worded[(row["graph"], row["question_id"])].append({**where, "keyword_hits": row["keyword_hits"]})
        for k in cache[row["model"]]:
            cache[row["model"]][k] += ((row.get("usage") or {}).get(k) or 0)
    order = sorted(counts)
    return {
        "probes": len(latest),
        "use": "diagnostic only (change-log entry 34): no question is excluded on this screen's evidence",
        "excludes_nothing": True,
        "refused_questions": [{"graph": g, "question_id": q, "refusals": sorted(v, key=lambda x: (
            x["condition"], x["model"]))} for (g, q), v in sorted(prone.items())],
        "keyword_flagged_questions": [{"graph": g, "question_id": q, "flags": sorted(v, key=lambda x: (
            x["condition"], x["model"]))} for (g, q), v in sorted(worded.items())],
        "counts": [counts[k] for k in order],
        "usage_by_model": dict(sorted(cache.items())),
        "keywords": list(REFUSAL_KEYWORDS),
    }


# --- the command --------------------------------------------------------------------------------------------


def _pairs(values: list[str], what: str) -> dict[str, str]:
    out = {}
    for v in values:
        name, sep, val = v.partition("=")
        if not sep:
            raise SystemExit(f"{what} takes NAME=VALUE, got {v!r}")
        out[name] = val
    return out


def main(argv: list[str] | None = None, *, models: dict[str, ModelSpec] = MODELS) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--graphs", nargs="+", default=list(GRAPHS))
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(KNOWN_CONDITIONS))
    ap.add_argument("--models", nargs="+", help=f"default: all of {', '.join(models)}")
    ap.add_argument("--prompts-root", type=Path, default=ROOT / "data" / "prompts")
    ap.add_argument("--block-dir", type=Path, default=None, help="the block store (default data/prompts/blocks)")
    ap.add_argument("--questions", nargs="+", action="extend", default=[],
                    help="GRAPH=PATH (default QUESTION_FILES, else data/questions/<graph>/questions.jsonl)")
    ap.add_argument("--question-sha", nargs="+", action="extend", default=[],
                    help="GRAPH=PREFIX, instead of the ruled prefix")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--concurrency", type=int, default=None, help="every model's pool (default 4, Anthropic 3)")
    ap.add_argument("--model-concurrency", nargs="+", action="extend", default=[], help="NAME=N")
    ap.add_argument("--limit", type=int, default=None, help="only the first N probes per model (a cache pilot)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true", help="count the calls and estimate the tokens; send nothing")
    args = ap.parse_args(argv)
    names = args.models or list(models)
    unknown = [n for n in names if n not in models]
    if unknown:
        raise SystemExit(f"unknown models {unknown}; known: {', '.join(models)}")
    conditions = tuple(args.conditions)
    paths = {g: Path(p) for g, p in _pairs(args.questions, "--questions").items()}
    prefixes = {**QUESTION_SETS, **_pairs(args.question_sha, "--question-sha")}

    questions, question_sha, prompts, not_sent = {}, {}, {}, {}
    for g in args.graphs:
        prompts[g] = load_prompts(args.prompts_root / g, conditions, args.block_dir)  # no prompts: stops here
        if g not in prefixes:
            raise SystemExit(f"{g}: no ruled question set (QUESTION_SETS) and no --question-sha")
        qs, question_sha[g] = load_questions(paths.get(g, default_questions(g)),
                                             prefixes[g], g)
        questions[g] = {q["id"]: q for q in qs if q.get("split") == "dev"}  # entry 34: dev only
        not_sent[g] = sum(1 for q in qs if q.get("split") != "dev")
    assert_dev_only(questions)
    tasks = task_order(args.graphs, {g: list(questions[g]) for g in args.graphs}, conditions, args.seed)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    routes = {n: models[n].factory(args.max_tokens) for n in names}  # no network until a probe is sent
    config = {
        "schema": SCHEMA, "graphs": args.graphs, "conditions": list(conditions), "models": names,
        "max_tokens": args.max_tokens, "seed": args.seed, "question_sha256": question_sha,
        "splits": ["dev"], "test_questions_not_sent": not_sent,
        "prompt_sha256": {g: {c: p["sha256"] for c, p in prompts[g].items()} for g in args.graphs},
        "routes": {n: r.route_info() if hasattr(r, "route_info") else {"model": n} for n, r in routes.items()},
        "keywords": list(REFUSAL_KEYWORDS), "policy_error": POLICY_ERROR.pattern,
    }
    out = args.out or ROOT / "results" / f"refusal-screen-{time.strftime('%Y%m%d')}"
    latest = read_rows(out / "rows.jsonl")
    done = {k for k, r in latest.items() if r["status"] in ("answered", "refused")}

    def todo(name: str) -> list[tuple[str, str, str]]:
        return [t for t in tasks if (t[0], t[1], t[2], name) not in done]

    if args.dry_run:
        per_graph_condition = {(g, c): count_tokens(prompts[g][c]["text"]) for g in args.graphs for c in conditions}
        estimate, calls = 0, 0
        for name in names:
            for g, q, c in todo(name):
                calls += 1
                estimate += per_graph_condition[(g, c)] + count_tokens(user_message(questions[g][q]))
        print(json.dumps({"calls": calls, "calls_per_model": len(tasks), "models": names,
                          "questions": {g: len(questions[g]) for g in args.graphs}, "conditions": list(conditions),
                          "system_prompt_o200k_tokens": {f"{g}/{c}": n for (g, c), n in per_graph_condition.items()},
                          "estimated_input_o200k_tokens": estimate,
                          "max_output_tokens_bound": calls * args.max_tokens}, indent=1))
        return 0

    problems = [f"{n}: no key; set {' or '.join(models[n].key_envs)}" for n in names
                if models[n].key_envs and not any((os.environ.get(e) or "").strip() for e in models[n].key_envs)]
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
        diff = sorted(k for k in set(recorded) | set(config) if k not in ("models", "routes")
                      and recorded.get(k) != redact_structure(config)[0].get(k))
        if diff:
            raise SystemExit(f"{out} was screened with other inputs ({', '.join(diff)}); use a new --out")
    manifest_path.write_text(json.dumps(redact_structure(config)[0], indent=1, sort_keys=True) + "\n",
                             encoding="utf-8")

    lock = threading.Lock()
    rows_path = out / "rows.jsonl"
    fixed = _pairs(args.model_concurrency, "--model-concurrency")

    def run_one(name: str, route, task: tuple[str, str, str]) -> None:
        g, qid, c = task
        q = questions[g][qid]
        result = probe(route, prompts[g][c]["text"], user_message(q), args.max_tokens)
        row = {"schema": SCHEMA, "graph": g, "question_id": qid, "split": q.get("split"), "tier": q.get("tier"),
               "template": q.get("template"), "condition": c, "prompt_sha256": prompts[g][c]["sha256"],
               "model": name, "route_model": getattr(route, "model", name), **result,
               "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        text = json.dumps(redact_structure(row)[0], ensure_ascii=False, sort_keys=True)
        with lock:
            with rows_path.open("a", encoding="utf-8") as fh:
                fh.write(text + "\n")

    pools = []
    for name in names:
        n = int(fixed.get(name) or args.concurrency or models[name].concurrency)
        pool = ThreadPoolExecutor(max_workers=n, thread_name_prefix=name)
        pools.append((name, pool, [pool.submit(run_one, name, routes[name], t) for t in todo(name)]))
    for name, pool, futures in pools:
        for f in futures:
            f.result()
        pool.shutdown()
    summary = summarize(read_rows(rows_path))
    (out / "summary.json").write_text(json.dumps(redact_structure(summary)[0], indent=1, ensure_ascii=False)
                                      + "\n", encoding="utf-8")
    print(f"{summary['probes']} probes (dev only); {len(summary['refused_questions'])} questions refused in some "
          f"probe; {len(summary['keyword_flagged_questions'])} keyword-flagged; nothing excluded", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
