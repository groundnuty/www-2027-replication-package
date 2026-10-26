#!/usr/bin/env python3
"""Count the tokens of the agent's shared prompt prefix (tools + system prompt) on Claude.

    uv run python scripts/count_prompt_tokens.py --row-cap 500 --sparql-timeout 60 \\
        [--questions questions.jsonl | --question "..."]

Prompt caching on the 5.5 models caches a shared prefix only when it is at
least 512 tokens long. This calls the free token-counting endpoint
(``messages.count_tokens``) twice with the tools and system prompt the
Anthropic route sends: once with a one-character user turn (the prefix plus a
few tokens of message framing) and once with a sample question. It needs
ANTHROPIC_API_KEY and exits 2 without it; nothing is generated or billed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.agent.prompts import get_condition  # noqa: E402
from harness.agent.providers import AnthropicRoute  # noqa: E402
from harness.agent.record import redact_secrets  # noqa: E402
from harness.agent.tools import TOOLS, anthropic_tools  # noqa: E402

MIN_CACHEABLE_TOKENS = 512
SAMPLE_QUESTION = "Which reactions in the knowledge graph produce water?"


def count_prefix(client: Any, *, model: str, system: str, tools: list[dict], question: str) -> dict:
    def count(content: str) -> int:
        return client.messages.count_tokens(model=model, system=system, tools=tools,
                                            messages=[{"role": "user", "content": content}]).input_tokens

    prefix = count(".")
    full = count(question)
    return {
        "model": model,
        "prefix_tokens": prefix,  # tools + system + a one-character user turn
        "with_question_tokens": full,
        "question_tokens": full - prefix,
        "min_cacheable_tokens": MIN_CACHEABLE_TOKENS,
        "prefix_qualifies": prefix >= MIN_CACHEABLE_TOKENS,
    }


def _sample_question(args: argparse.Namespace) -> str:
    if args.question:
        return args.question
    if args.questions:
        with args.questions.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    return json.loads(line)["question"]
    return SAMPLE_QUESTION


def main(argv: list[str] | None = None, *, client: Any = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", default="claude-sonnet-5-5")
    p.add_argument("--row-cap", type=int, default=500)
    p.add_argument("--sparql-timeout", type=float, default=60.0)
    p.add_argument("--condition", default="none")
    p.add_argument("--questions", type=Path, help="question file; its first question is the sample")
    p.add_argument("--question", help="a sample question")
    args = p.parse_args(argv)
    system = get_condition(args.condition).render(row_cap=args.row_cap, timeout_s=args.sparql_timeout)
    tools = anthropic_tools(TOOLS)
    if client is None:
        key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
        if not key:
            print("ANTHROPIC_API_KEY is not set; nothing was counted", file=sys.stderr)
            return 2
        import anthropic

        client = anthropic.Anthropic(api_key=key, base_url=AnthropicRoute.DEFAULT_BASE_URL, max_retries=2)
    try:
        result = count_prefix(client, model=args.model, system=system, tools=tools, question=_sample_question(args))
    except Exception as exc:  # noqa: BLE001 - report, redacted, and fail
        print(redact_secrets(f"error: {type(exc).__name__}: {exc}")[0], file=sys.stderr)
        return 1
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
