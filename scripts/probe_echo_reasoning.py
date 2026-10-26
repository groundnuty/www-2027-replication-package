#!/usr/bin/env python3
"""The echo gate probe (results/gateway-mitigations-20261020/AB-PLAN.md, P-E): does a Gateway deployment render an echoed
reasoning channel, and under which wire name?

    uv run python scripts/probe_echo_reasoning.py --record <record.json> --turn K --model <table model> --out <out.json>

The request of assistant turn K of a recorded attempt is rebuilt exactly as the runner sent it (the condition's system
prompt and the agent's tools, both checked against the record's sha256 values, and the transcript before turn K), then
sent three times with max_tokens 1: without the echo, and with every earlier turn's reasoning echoed as
``reasoning_content`` and as ``reasoning`` (OpenAICompatRoute ``echo_reasoning``). The server's ``prompt_tokens`` tell
whether the template rendered the echo: it is "rendered" when the prompt grew by at least half the reasoning the echo
carried (``verdict``). Three requests of one output token each; no record is written, nothing is scored. Pick a turn
whose earlier turns carry at least MIN_EXPECTED_TOKENS of reasoning, or the probe refuses (it could not tell).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from harness.agent.loop import tools_sha256
from harness.agent.prompts import get_condition, sha256_text
from harness.agent.tools import agent_tools

CHARS_PER_TOKEN = 3.455  # reasoning characters per reasoning token, GLM and Qwen3.8 natural records (MATRIX.md)
NOISE_FLOOR_TOKENS = 16  # a template may add a few tokens around any assistant turn
MIN_EXPECTED_TOKENS = 200
ECHO_NAMES = ("reasoning_content", "reasoning")


def _load_run_pilot():
    if "run_pilot" in sys.modules:
        return sys.modules["run_pilot"]
    spec = importlib.util.spec_from_file_location("run_pilot", Path(__file__).resolve().parent / "run_pilot.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_pilot"] = module
    spec.loader.exec_module(module)
    return module


def prefix_before_turn(transcript: list[dict], turn: int) -> list[dict]:
    """The transcript entries before assistant turn ``turn``: what the runner sent for it."""
    for i, entry in enumerate(transcript):
        if entry.get("role") == "assistant" and entry.get("turn") == turn:
            return transcript[:i]
    raise ValueError(f"no assistant turn {turn} in the transcript")


def _reasoning(entry: dict) -> str:
    native = entry.get("native") or {}
    text = native.get("reasoning_content") or native.get("reasoning")
    return text if isinstance(text, str) else ""


def expected_echo_tokens(transcript: list[dict], per_turn: list[dict], turn: int,
                         chars_per_token: float = CHARS_PER_TOKEN) -> dict:
    """The reasoning tokens the echo adds to turn ``turn``'s request: each earlier turn's reported reasoning tokens,
    or its reasoning characters / ``chars_per_token`` (rounded up) where the server reported none."""
    reported = {t.get("turn"): ((t.get("usage_raw") or {}).get("completion_tokens_details") or {}).get(
        "reasoning_tokens") for t in per_turn}
    tokens = reported_turns = estimated_turns = echoed = 0
    for entry in prefix_before_turn(transcript, turn):
        if entry.get("role") != "assistant" or not _reasoning(entry):
            continue
        echoed += 1
        count = reported.get(entry.get("turn"))
        if isinstance(count, int) and count > 0:
            tokens += count
            reported_turns += 1
        else:
            tokens += math.ceil(len(_reasoning(entry)) / chars_per_token)
            estimated_turns += 1
    return {"tokens": tokens, "reported_turns": reported_turns, "estimated_turns": estimated_turns,
            "echoed_turns": echoed}


def verdict(base_prompt_tokens: int, echoed_prompt_tokens: int, expected_tokens: int) -> str:
    """rendered: the prompt grew by at least half the echoed reasoning; not_rendered: by no more than the noise floor
    (or 5% of it); partial: in between; nothing_to_echo: no earlier reasoning."""
    if expected_tokens <= 0:
        return "nothing_to_echo"
    delta = echoed_prompt_tokens - base_prompt_tokens
    if delta >= 0.5 * expected_tokens:
        return "rendered"
    if delta <= max(NOISE_FLOOR_TOKENS, 0.05 * expected_tokens):
        return "not_rendered"
    return "partial"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--record", type=Path, required=True)
    p.add_argument("--turn", type=int, required=True)
    p.add_argument("--model", required=True, help="the runner's table model the record ran (scripts/run_pilot.py)")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    rec = json.loads(args.record.read_text(encoding="utf-8"))
    run_pilot = _load_run_pilot()
    route = run_pilot.MODELS[args.model].factory()
    if route.model != rec["model"]:
        raise SystemExit(f"{args.model} serves {route.model}, the record ran {rec['model']}")
    system = get_condition(rec["condition"]).render(row_cap=rec["store"]["row_cap"],
                                                    timeout_s=rec["store"]["sparql_timeout_s"])
    if sha256_text(system) != rec["prompt"]["sha256"]:
        raise SystemExit("the rebuilt system prompt is not the record's (only a condition without a block is rebuilt)")
    tools = agent_tools(rec["agent_config"]["max_result_chars"])
    if tools_sha256(tools) != rec["prompt"]["tools_sha256"]:
        raise SystemExit("the rebuilt tools are not the record's")
    transcript = prefix_before_turn(rec["transcript"], args.turn)
    expected = expected_echo_tokens(rec["transcript"], rec["tokens"]["per_turn"], args.turn)
    if expected["tokens"] < MIN_EXPECTED_TOKENS:
        raise SystemExit(f"turn {args.turn} would echo only {expected['tokens']} reasoning tokens; pick a later turn")
    results: dict[str, Any] = {}
    for name in (None, *ECHO_NAMES):
        step = route.with_options(echo_reasoning=name).step(system, transcript, tools, max_tokens=1)
        results[name or "none"] = {"prompt_tokens": (step.usage_raw or {}).get("prompt_tokens"),
                                   "served_model": step.model}
    base = results["none"]["prompt_tokens"]
    out = {"record": rec["run_id"], "model": args.model, "turn": args.turn, "expected_echo": expected,
           "requests": results,
           "verdicts": {name: verdict(base, results[name]["prompt_tokens"], expected["tokens"])
                        for name in ECHO_NAMES}}
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(out["verdicts"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
