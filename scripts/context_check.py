#!/usr/bin/env python3
"""Context-window check before the freeze (design log): does every model accept the largest request a run can make?

    uv run python scripts/context_check.py --dry-run           # sizes only, no network, no key
    uv run python scripts/context_check.py --out results/context-check-<date>.json [--models ...]

One request per model, sent once through the harness's own route (no retry):
* system: the largest prompt at the budget, across every graph and source in data/prompts/*/ (by o200k tokens
  of the prompt file; a provisional render counts, since the redraw replaces it at about the same size);
* conversation: the committed lookup-pilot record at the 95th percentile (nearest rank, ties by run id) of the
  final turn's prompt (input + cache read + cache write tokens, as each provider counted them), replayed as the
  input of that final turn: everything before its last assistant turn. Assistant turns are replayed from their
  neutral fields (text and tool calls), never the provider's native blocks: a thinking block is bound to the
  conversation that produced it and a new system prompt would make the replay an edited history;
* the agent's tools, and max output tokens 16,384.

A model flagged single-tool-call (Llama-3.3-70B: its chat template refuses an assistant turn with several tool
calls) gets a conversation whose assistant turns carry one call each, and its route sends
parallel_tool_calls=false. By default (--single-tool-call-transcript select) that conversation is the 95th
percentile of the records whose assistant turns each carry at most one call; with fewer than MIN_SELECT such
records, or with "split", it is the overall 95th-percentile record with each multi-call turn split into
consecutive single-call assistant/tool pairs, keeping every text, call and result. The report says which.

With --transcript synthetic the conversation is instead an N-turn run (--turns, default 60, the run cap) built
deterministically from real pilot turns: the question of the record with the longest final-turn prompt, then
its SPARQL call/result pairs (multi-call turns split), then the next longest record's, and so on, until N
single-call turns, renumbered with fresh call ids. It is single-call, so every model, Llama included, gets it.

Pass: the provider accepts the request and answers (any content). Fail: the provider's error, or a
context-window stop reason. A model whose key is not in the environment is "not run" and fails the check.
Exit status 0 when every model passed, 1 otherwise. The report (--out) holds sizes, statuses, errors and the
provider's usage, never a key. --dry-run prints the sizes (o200k tokens of each route's serialised request: an
estimate, not the provider's own count) and sends nothing.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import math
import os
import sys
import tarfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_pilot import MODELS as RUN_MODELS  # noqa: E402

from harness.agent.providers import (  # noqa: E402
    CONTEXT_STOPS,
    RetryPolicy,
    StepFailure,
)
from harness.agent.record import redact_secrets, redact_structure  # noqa: E402
from harness.agent.tools import TOOLS  # noqa: E402
from harness.sources.tokens import count_tokens  # noqa: E402

MAX_OUTPUT_TOKENS = 16_384
PERCENTILE = 0.95
ONCE = RetryPolicy(max_retries=0)
DEFAULT_RECORDS = "results/pilot-lookup-*/*/records.tar.gz"


MIN_SELECT = 20  # below this many single-call records a 95th percentile is the maximum; split instead


@dataclass(frozen=True)
class ModelSpec:
    factory: Callable[[], Any]
    key_envs: tuple[str, ...] = ()
    single_tool_call: bool = False  # the model's template takes one tool call per assistant turn


def _from_table(name: str) -> ModelSpec:
    """The runs' route from the one model table (scripts/run_pilot.py MODELS: route model, transport, effort and
    tool-call settings), with only the check's output budget and a single send (design log: one table)."""
    spec = RUN_MODELS[name]

    def build() -> Any:
        route = spec.factory()
        route.max_tokens = MAX_OUTPUT_TOKENS
        route.retry = ONCE
        return route

    return ModelSpec(build, spec.key_envs, getattr(spec.factory(), "parallel_tool_calls", None) is False)


# The routes as the runs use them: every model the runs had when the check was written, each from the one table.
MODELS: dict[str, ModelSpec] = {name: _from_table(name) for name in (
    "claude-sonnet-5-5", "claude-haiku-5-5", "qwen3.8-27b", "deepseek-v4-flash-0731", "deepseek-v4.1-flash",
    "glm-5.3-flash", "gemma-4-31b", "llama-3.3-70b")}


# --- the conversation -------------------------------------------------------------------------------------


def load_records(archives: list[Path]) -> list[dict]:
    out = []
    for archive in archives:
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith(".json") and not member.name.endswith(".setaside.json"):
                    out.append(json.loads(tar.extractfile(member).read().decode("utf-8")))
    return out


def final_turn_input(rec: dict) -> int | None:
    """The final turn's whole prompt as the provider counted it: input + cache read + cache write (a count the
    provider did not report is 0 here)."""
    turns = (rec.get("tokens") or {}).get("per_turn") or []
    if not turns:
        return None
    last = turns[-1]
    return sum(last.get(k) or 0 for k in ("input_tokens", "cache_read_tokens", "cache_write_tokens"))


def percentile_record(records: list[dict], q: float = PERCENTILE) -> dict:
    ranked = sorted(((n, r["run_id"], r) for r in records if (n := final_turn_input(r)) is not None),
                    key=lambda x: (x[0], x[1]))
    if not ranked:
        raise ValueError("no record has per-turn usage")
    return ranked[max(math.ceil(q * len(ranked)) - 1, 0)][2]


def replay_transcript(rec: dict) -> list[dict]:
    """The final turn's input: the transcript before its last assistant turn, assistant turns without native blocks."""
    transcript = rec["transcript"]
    last = max((i for i, t in enumerate(transcript) if t["role"] == "assistant"), default=len(transcript))
    out = []
    for turn in transcript[:last]:
        turn = copy.deepcopy(turn)
        if turn["role"] == "assistant":
            turn.pop("native", None)
            turn.pop("native_format", None)
        out.append(turn)
    return out


def single_call(transcript: list[dict]) -> bool:
    return all(len(t.get("tool_calls") or []) <= 1 for t in transcript if t["role"] == "assistant")


def split_tool_calls(transcript: list[dict]) -> list[dict]:
    """Each assistant turn with several calls becomes consecutive single-call assistant/tool pairs: the turn's text
    stays with its first call, each result follows its own call, a result matched to no call stays with the last."""
    out: list[dict] = []
    i = 0
    while i < len(transcript):
        turn = transcript[i]
        calls = turn.get("tool_calls") or [] if turn["role"] == "assistant" else []
        if len(calls) <= 1:
            out.append(copy.deepcopy(turn))
            i += 1
            continue
        results = transcript[i + 1]["results"] if i + 1 < len(transcript) and transcript[i + 1]["role"] == "tool" \
            else []
        by_id = {r["tool_call_id"]: r for r in results}
        for k, call in enumerate(calls):
            out.append({**copy.deepcopy(turn), "text": turn.get("text", "") if k == 0 else "",
                        "tool_calls": [copy.deepcopy(call)]})
            mine = [by_id[call["id"]]] if call["id"] in by_id else []
            if k == len(calls) - 1:
                mine += [r for r in results if r["tool_call_id"] not in {c["id"] for c in calls}]
            if mine:
                out.append({"role": "tool", "results": copy.deepcopy(mine)})
        i += 2 if results else 1
    return out


def choose_transcript(records: list[dict], q: float, *, single_tool_call: bool = False,
                      mode: str = "select") -> dict:
    """The conversation a model is sent, with how it was chosen."""
    rec = percentile_record(records, q)
    transcript, method = replay_transcript(rec), "as recorded"
    if single_tool_call and not single_call(transcript):
        singles = [r for r in records if single_call(replay_transcript(r)) and final_turn_input(r) is not None]
        if mode == "select" and len(singles) >= MIN_SELECT:
            rec = percentile_record(singles, q)
            transcript, method = replay_transcript(rec), "select"
        else:
            transcript = split_tool_calls(transcript)
            method = "split" if mode == "split" else (
                f"split (only {len(singles)} single-call record{'s' if len(singles) != 1 else ''}, "
                f"fewer than {MIN_SELECT})")
    elif single_tool_call:
        method = "as recorded (already one call per turn)"
    return {"run_id": rec["run_id"], "record": rec, "transcript": transcript, "method": method}


def by_weight(records: list[dict]) -> list[dict]:
    """Records by final-turn prompt, heaviest first, ties by run id."""
    ranked = sorted((r for r in records if final_turn_input(r) is not None),
                    key=lambda r: (-final_turn_input(r), r["run_id"]))
    if not ranked:
        raise ValueError("no record has per-turn usage")
    return ranked


def synthetic_transcript(records: list[dict], turns: int) -> list[dict]:
    """A ``turns``-turn conversation from real pilot turns, heaviest records first (by final-turn prompt, ties by
    run id): the first record's question, then SPARQL call/result pairs, one call per assistant turn."""
    ranked = by_weight(records)
    question = next(t for t in ranked[0]["transcript"] if t["role"] == "user")
    out = [copy.deepcopy(question)]
    n = 0
    for rec in ranked:
        conv = split_tool_calls(replay_transcript(rec))
        for i, turn in enumerate(conv[:-1]):
            nxt = conv[i + 1]
            if (turn["role"] != "assistant" or nxt["role"] != "tool"
                    or [c["name"] for c in turn.get("tool_calls") or []] != ["sparql_query"]):
                continue
            n += 1
            cid = f"call_{n:03d}"
            out.append({"role": "assistant", "turn": n, "text": turn.get("text", ""),
                        "tool_calls": [{**copy.deepcopy(turn["tool_calls"][0]), "id": cid}]})
            out.append({"role": "tool", "results": [{**copy.deepcopy(nxt["results"][0]), "tool_call_id": cid}]})
            if n == turns:
                return out
    raise ValueError(f"the records hold only {n} SPARQL call/result pairs, fewer than {turns} turns")


def transcript_info(chosen: dict, records: list[dict], q: float) -> dict:
    rec = chosen["record"]
    return {"run_id": rec["run_id"], "model": rec.get("model"), "method": chosen["method"],
            "records_ranked": len(records), "percentile": q,
            "recorded_final_turn_input_tokens": final_turn_input(rec), "turns": len(chosen["transcript"]),
            "o200k_tokens": count_tokens(json.dumps(chosen["transcript"], ensure_ascii=False))}


# --- the system prompt --------------------------------------------------------------------------------------


def largest_prompt(prompts_root: Path) -> dict:
    found = []
    for manifest_path in sorted(Path(prompts_root).glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, c in manifest["conditions"].items():
            if c["condition"] == "none":
                continue
            path = manifest_path.parent / c["prompt"]["file"]
            text = path.read_text(encoding="utf-8")
            found.append({"graph": manifest_path.parent.name, "key": key, "file": os.path.relpath(path, ROOT),
                          "tokens": count_tokens(text), "text": text})
    if not found:
        raise ValueError(f"no ontology prompt under {prompts_root}")
    return max(found, key=lambda p: (p["tokens"], p["graph"], p["key"]))


# --- one request ---------------------------------------------------------------------------------------------


def request_tokens(route, system: str, transcript: list[dict], max_tokens: int) -> int:
    """o200k tokens of the route's serialised request: an estimate of its size, the same measure for every route."""
    return count_tokens(json.dumps(route.request(system, transcript, TOOLS, max_tokens), ensure_ascii=False))


def check(route, system: str, transcript: list[dict], max_tokens: int) -> dict:
    try:
        step = route.step(system, transcript, TOOLS, max_tokens=max_tokens)
    except StepFailure as exc:
        return {"status": "fail", "failure_type": exc.failure_type, "error": redact_secrets(exc.message)[0][:2000],
                "error_class": (exc.details or {}).get("error_class"), "http_status": (exc.details or {}).get("status")}
    except Exception as exc:  # noqa: BLE001 - an unexpected client error is a failed check, reported
        return {"status": "fail", "failure_type": "other", "error_class": type(exc).__name__,
                "error": redact_secrets(str(exc))[0][:2000], "http_status": None}
    out = {"status": "pass", "stop_reason": step.stop_reason, "usage": step.usage, "served_model": step.model}
    if step.stop_reason in CONTEXT_STOPS:
        out.update(status="fail", failure_type="context_overflow",
                   error=f"the provider answered with stop reason {step.stop_reason}")
    return out


def main(argv: list[str] | None = None, *, models: dict[str, ModelSpec] = MODELS) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", nargs="+", help=f"default: all of {', '.join(models)}")
    ap.add_argument("--dry-run", action="store_true", help="print the request sizes; send nothing")
    ap.add_argument("--prompts-root", type=Path, default=ROOT / "data" / "prompts")
    ap.add_argument("--records", default=str(ROOT / DEFAULT_RECORDS), help="glob of record archives")
    ap.add_argument("--percentile", type=float, default=PERCENTILE)
    ap.add_argument("--max-tokens", type=int, default=MAX_OUTPUT_TOKENS)
    ap.add_argument("--transcript", choices=("p95", "synthetic"), default="p95",
                    help="the 95th-percentile pilot record (p95), or an N-turn run built from real pilot turns")
    ap.add_argument("--turns", type=int, default=60, help="turns of the synthetic conversation (the run cap)")
    ap.add_argument("--single-tool-call-transcript", choices=("select", "split"), default="select",
                    help="for a single-tool-call model: the 95th percentile of the single-call records (select), "
                         "or the overall 95th-percentile record with its multi-call turns split (split)")
    ap.add_argument("--out", type=Path, help="write the report here as well")
    args = ap.parse_args(argv)
    names = args.models or list(models)
    unknown = [n for n in names if n not in models]
    if unknown:
        ap.error(f"unknown models {unknown}; known: {', '.join(models)}")

    prompt = largest_prompt(args.prompts_root)
    archives = sorted(Path(p) for p in glob.glob(args.records))
    records = load_records(archives)
    if args.transcript == "synthetic":  # single-call by construction: every model gets it
        conv = synthetic_transcript(records, args.turns)
        top = by_weight(records)[0]  # whose question opens the conversation
        chosen = {"default": {"run_id": top["run_id"], "record": top, "transcript": conv,
                              "method": f"synthetic: {args.turns} single-call turns"}}
    else:
        chosen = {"default": choose_transcript(records, args.percentile)}
    if args.transcript == "p95" and any(models[n].single_tool_call for n in names):
        chosen["single_tool_call"] = choose_transcript(records, args.percentile, single_tool_call=True,
                                                       mode=args.single_tool_call_transcript)
    report: dict[str, Any] = {
        "prompt": {k: prompt[k] for k in ("graph", "key", "file", "tokens")},
        "archives": [os.path.relpath(a, ROOT) for a in archives],
        "transcripts": {k: transcript_info(c, records, args.percentile) for k, c in chosen.items()},
        "max_output_tokens": args.max_tokens,
        "requests": {},
    }
    report["transcript"] = report["transcripts"]["default"]  # the key earlier reports had
    routes, kinds = {}, {}
    for name in names:
        routes[name] = models[name].factory()
        kinds[name] = "single_tool_call" if models[name].single_tool_call and "single_tool_call" in chosen \
            else "default"
        size = request_tokens(routes[name], prompt["text"], chosen[kinds[name]]["transcript"], args.max_tokens)
        report["requests"][name] = {"model": routes[name].model, "transcript": kinds[name],
                                    "parallel_tool_calls": getattr(routes[name], "parallel_tool_calls", None),
                                    "request_o200k_tokens": size, "with_max_output_o200k": size + args.max_tokens}
    if args.dry_run:
        print(json.dumps(report, indent=1))
        return 0
    results = {}
    for name in names:
        envs = models[name].key_envs
        if envs and not any((os.environ.get(e) or "").strip() for e in envs):
            results[name] = {"status": "not run", "error": f"no key: set {' or '.join(envs)}"}
            continue
        results[name] = check(routes[name], prompt["text"], chosen[kinds[name]]["transcript"], args.max_tokens)
        print(f"{name}: {results[name]['status']}" + (f" ({results[name].get('error', '')[:200]})"
                                                     if results[name]["status"] != "pass" else ""), file=sys.stderr)
    report["results"] = results
    report["passed"] = all(r["status"] == "pass" for r in results.values())
    text = json.dumps(redact_structure(report)[0], indent=1, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
