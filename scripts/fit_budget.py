#!/usr/bin/env python3
"""The largest render budget at which every model in P fits change-log entry 60's rule (design log, entry 88).

Entry 60, in each model's own tokens: the prompt at budget B + the maximum loop context of its dev records + the
output reservation <= 0.95 x its window. Entry 88: one B for a set of graphs (Rhea and opaque Rhea), so the
prompt is the largest over every graph and every prompted arm (curated, census, generated draw 1), rendered at B
exactly as scripts/build_prompts.py renders it; B is the largest multiple of 1,000 o200k tokens at which every
model fits, the census cut by support within its layers, curated and the drafts whole.

    # 1. sizes: render each arm at each budget and count the whole system prompt in each model's tokenizer
    uv run python scripts/fit_budget.py measure --graphs rhea-142 opaque-rhea-142 --budgets 6000:80000:1000 \\
        --tokenizer qwen3.8-27b=PATH/Qwen_Qwen3.8-27B.tokenizer.json ... \\
        [--claude claude-sonnet-5-5 claude-haiku-5-5] --out sizes.json
    # 2. the budget, for each reservation (16,000 and 32,768): models.json = {model: {window, loop,
    #    loop_tool_cap_chars, turn1_overhead, source}}
    uv run python scripts/fit_budget.py compute --sizes sizes.json --models models.json --reservation 16000 \\
        --reading strict --tool-cap qwen3.8-27b=80000

Design log's ruling for Rhea: the strict reading (the default) adds turn 1's tools, question and template beyond the
system prompt (each model's turn1_overhead) to entry 60's sum; Qwen3.8 has per-graph tool caps, on Rhea not below
80,000 characters, and every model's loop maximum must have been measured under the tool cap its runs use. The
search covers every multiple of 1,000 in the measured range and returns the largest budget at which every model
fits, a smaller failing one notwithstanding (counts can be non-monotone); the range must end on a budget at which a
model fails, or at which every prompt renders whole, or it is refused (a fit at the end of a range that still fits
says nothing of the next budget). The output records its schema, reading,
reservation, tool caps, graphs and the sizes' sha256: a fit-bound ruling names this file, committed
(harness.sources.render_guard.authorised_budget).

`measure` reads only data/sources/<graph>/ (as build_prompts does; no store) and, for Claude, the free
count_tokens endpoint (the key read inside the process from ~/.config/www-2027/anthropic_api_key, never printed):
the system prompt's own tokens, count(system=prompt, user "x") - count(user "x"). Hugging Face tokenizers count
with no special tokens and no chat template. Every model is counted at every budget: the computation
(harness.sources.fit, shared with the render guard) refuses a missing count, as counts can be non-monotone. It also
records each prompt's sha256 and whether its render was cut, per graph, arm and budget, and the source each arm was
measured on (its file's sha256, its name and the sha256 of its canonical JSON): the render guard refuses a fit-bound
budget whose sizes were measured on other sources than the ones being rendered. The
compute command writes harness.sources.fit.fit_record's output, which records the sizes and models files it was
computed from (path from the repository root and sha256); a fit-bound ruling commits all three, in both branches.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import build_prompts as bp  # noqa: E402

from harness.agent.prompts import get_condition, ontology_block, sha256_text  # noqa: E402
from harness.sources import source_file  # noqa: E402
from harness.sources.fit import (  # noqa: E402  the computation itself, shared with the render guard
    READINGS,
    SCHEMA,
    STEP,
    FitError,
    compute,
    fit_record,
)
from harness.sources.imported import namespaces_sha256  # noqa: E402
from harness.sources.tokens import count_tokens  # noqa: E402

__all__ = ["READINGS", "SCHEMA", "FitError", "compute", "fit_record", "measure"]

ARMS = ("curated", "census", "generated")


def measure(graphs: dict[str, Path], budgets: list[int], counters: dict[str, Callable[[str], int]], *,
            row_cap: int = bp.DEFAULT_ROW_CAP, timeout_s: float = bp.DEFAULT_TIMEOUT_S, log=lambda m: None,
            owner_namespaces: dict[str, tuple[str, ...]] | None = None) -> dict:
    """Prompt tokens per graph, arm and budget, for every counter (and o200k), at every budget: compute needs a
    measured count at every candidate (External review of 50e8ec8, item 3). Also each prompt's sha256 and whether its
    render was cut (``prompts``), and the source each arm was measured on (``inputs``; External review of bebd077).
    The arms are every source scripts/build_prompts.py renders: curated, census and draw 1 on every graph, and the
    layout's extra arms (StoreLayout.extra_arms: the aligned curated source and drafts 2 and 3 on a15), the same
    entries build_prompts renders, so the render guard finds every rendered source among the measured ones.
    ``owner_namespaces``: per graph whose layout renders imported files by origin (rule 16 as amended), the owner's
    namespaces, required there and refused elsewhere; the sizes record only their sha256."""
    none_text = get_condition("none").render(row_cap=row_cap, timeout_s=timeout_s)
    out = {"schema": SCHEMA, "budgets": sorted(budgets), "graphs": {}, "row_cap": row_cap,
           "sparql_timeout_s": timeout_s, "inputs": {}, "prompts": {}}
    owner_namespaces = owner_namespaces or {}
    for graph, src_dir in graphs.items():
        layout = bp.LAYOUTS.get(graph)
        by_origin = bool(layout is not None and layout.imports_by_origin)
        if by_origin != (graph in owner_namespaces):
            raise SystemExit(f"{graph}: the owner's namespaces are " + ("required: its layout renders imported files "
                                                                         "by origin" if by_origin else
                                                                         "an input only where the layout renders "
                                                                         "imported files by origin"))
        ns = tuple(owner_namespaces.get(graph, ()))
        if ns:
            out.setdefault("imported", {})[graph] = {
                "owner_namespaces_sha256": namespaces_sha256(ns)}
        ctx = bp.load_context(src_dir, ns)
        entries, skipped = bp.select_sources(src_dir, None, layout.extra_arms if layout is not None else ())
        by_arm = {e.key: e for e in entries}
        missing = [a for a in ARMS if a not in by_arm]
        if missing:
            raise SystemExit(f"{graph}: no {missing} source ({skipped})")
        names = [e.key for e in entries]
        # Each arm's source by name and content (the render guard compares them with the sources being rendered).
        out["inputs"][graph] = {a: {"file": by_arm[a].path.name,
                                    "sha256": hashlib.sha256(by_arm[a].path.read_bytes()).hexdigest(),
                                    "source": source_file.name_of(by_arm[a].path),
                                    "content_sha256": source_file.content_sha256(by_arm[a].path)} for a in names}
        arms: dict[str, dict] = {}
        prompts: dict[str, dict] = {}
        for arm in names:
            src = bp.read_source(by_arm[arm].path)
            arms[arm], prompts[arm] = {}, {}
            for b in sorted(budgets):
                r = bp.render_source(src, ctx, b)
                prompt = get_condition(by_arm[arm].condition).render(row_cap=row_cap, timeout_s=timeout_s,
                                                                     ontology_block=ontology_block(r.text))
                assert prompt.startswith(none_text)
                row = {"o200k": count_tokens(prompt)}
                row.update({name: fn(prompt) for name, fn in counters.items()})
                arms[arm][str(b)] = row
                # Whether the render was cut: a range may end where every prompt renders whole (harness.sources.fit).
                prompts[arm][str(b)] = {"sha256": sha256_text(prompt), "truncated": r.manifest["truncated"]}
            log(f"{graph} {arm}: {len(budgets)} budgets")
        out["graphs"][graph] = arms
        out["prompts"][graph] = prompts
    return out


def hf_counter(path: Path) -> tuple[Callable[[str], int], dict]:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(path))
    info = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return (lambda text: len(tok.encode(text, add_special_tokens=False).ids)), info


def claude_counter(model: str, key_file: Path) -> Callable[[str], int]:
    """The system prompt's own tokens by the free count_tokens endpoint; the key never leaves this process."""
    import urllib.error
    import urllib.request

    key = key_file.read_text().strip()

    def raw(system: str | None) -> int:
        body = {"model": model, "messages": [{"role": "user", "content": "x"}]}
        if system is not None:
            body["system"] = system
        data = json.dumps(body).encode()
        for attempt in range(8):
            req = urllib.request.Request("https://api.anthropic.com/v1/messages/count_tokens", data=data, headers={
                "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    return json.loads(resp.read())["input_tokens"]
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 529):
                    time.sleep(min(60, 2 ** attempt * 2))
                    continue
                raise RuntimeError(f"count_tokens HTTP {e.code}: "
                                   f"{e.read().decode(errors='replace').replace(key, '[KEY]')[:300]}") from None
            except (urllib.error.URLError, TimeoutError):
                time.sleep(min(60, 2 ** attempt * 2))
        raise RuntimeError("count_tokens failed after retries")

    base = raw(None)
    return lambda text: raw(text) - base


def _budgets(spec: str) -> list[int]:
    """LO:HI:1000 -> every multiple of 1,000 from LO to HI (a coarser grid would skip candidates)."""
    lo, hi, step = (int(x) for x in spec.split(":"))
    if lo % STEP or hi % STEP or step != STEP or hi < lo:
        raise SystemExit(f"--budgets {spec}: every multiple of 1,000 o200k tokens between two multiples of 1,000 is "
                         "searched, so the step is 1,000")
    return list(range(lo, hi + 1, step))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--graphs", nargs="+", required=True)
    m.add_argument("--sources-root", type=Path, default=ROOT / "data" / "sources")
    m.add_argument("--budgets", required=True, help="LO:HI:STEP in o200k tokens, e.g. 6000:80000:1000")
    m.add_argument("--tokenizer", nargs="+", action="extend", default=[], help="MODEL=tokenizer.json")
    m.add_argument("--claude", nargs="*", default=[], help="Claude model ids counted with count_tokens (every budget)")
    m.add_argument("--key-file", type=Path, default=Path.home() / ".config" / "www-2027" / "anthropic_api_key")
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--owner-namespaces-file", nargs="+", action="extend", default=[],
                   help="GRAPH=FILE, the private file naming the owner's namespaces (layouts that render imports)")
    m.add_argument("--graph-dir", nargs="+", action="extend", default=[],
                   help="GRAPH=DIR, a graph's sources elsewhere than <sources-root>/<graph> (a private directory)")
    c = sub.add_parser("compute")
    c.add_argument("--sizes", type=Path, required=True)
    c.add_argument("--models", type=Path, required=True,
                   help="{model: {window, loop, loop_tool_cap_chars, turn1_overhead, source}}")
    c.add_argument("--reservation", type=int, required=True)
    c.add_argument("--reading", choices=sorted(READINGS), default="strict")
    c.add_argument("--tool-cap", nargs="+", action="extend", default=[], help="MODEL=CHARS, the tool cap its runs use")
    c.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    if args.cmd == "measure":
        budgets = _budgets(args.budgets)
        counters, tokenizers = {}, {}
        for spec in args.tokenizer:
            name, path = spec.split("=", 1)
            counters[name], tokenizers[name] = hf_counter(Path(path))
        counters.update({name: claude_counter(name, args.key_file) for name in args.claude})
        dirs = dict(spec.split("=", 1) for spec in args.graph_dir)
        owners = {g: bp.read_owner_namespaces(Path(f))
                  for g, f in (spec.split("=", 1) for spec in args.owner_namespaces_file)}
        sizes = measure({g: Path(dirs[g]) if g in dirs else args.sources_root / g for g in args.graphs}, budgets,
                        counters, log=lambda msg: print(msg, file=sys.stderr, flush=True), owner_namespaces=owners)
        sizes["tokenizers"] = tokenizers
        sizes["claude"] = {"models": args.claude, "method": "count_tokens(system) - count(no system)"}
        args.out.write_text(json.dumps(sizes, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        return 0
    caps = {}
    for spec in args.tool_cap:
        name, chars = spec.split("=", 1)
        caps[name] = int(chars)
    try:
        out = fit_record(args.sizes, args.models, args.reservation, args.reading, caps, ROOT)
    except FitError as e:
        raise SystemExit(str(e)) from None
    text = json.dumps(out, indent=1, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("reading", "reservation", "tool_caps", "budget", "binding",
                                          "failing_budgets", "failing_at_next") if k in out}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
