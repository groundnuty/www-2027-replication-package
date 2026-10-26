"""The fit computation of change-log entry 88: the largest render budget at which every model fits entry 60's rule.

Kept byte-identical on feat/sources and feat/prompt-conditions: scripts/fit_budget.py (prompt-conditions) measures
the prompts and writes a fit computation with ``fit_record``; harness.sources.render_guard recomputes it with
``compute`` before it accepts a fit-bound budget (External review of ee6fd40, item 1).

Entry 60, in each model's own tokens: the prompt at budget B (+ turn 1's tools, question and template by the strict
reading, design log's ruling for Rhea) + the maximum loop context of its dev records + the output reservation
<= 0.95 x its window. One B for all the measured graphs: the prompt is the largest over every graph and arm. Every
multiple of 1,000 o200k tokens between the ends of the range is a candidate and must carry a measured count for every
model, graph and arm (counts can be non-monotone, so a missing one is never bounded by a neighbour); B is the largest
candidate at which every model fits, a failing smaller one notwithstanding. The range must end on a budget at which
some model fails, or at which every measured prompt renders whole (``sizes["prompts"]``: no larger budget changes a
prompt then); a range that ends on a budget every model fits says nothing of the next one, so it is refused (external reviewer
review of bebd077, item 1).

The sizes (scripts/fit_budget.py measure) also record, per graph and arm, the source it was measured on
(``sizes["inputs"]``: its name and the sha256 of its canonical JSON, harness.sources.source_file.content_sha256), which
the render guard compares with the sources being rendered (item 2).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
from pathlib import Path

SCHEMA = "fit-budget/1"
LIMIT_SHARE = 0.95
STEP = 1000
RULE = ("change-log entry 60 (design log): prompt tokens at B + the maximum loop context of the model's dev records + "
        "the output reservation <= 0.95 x the model's window, each in the model's own tokens; entry 88: one B for "
        "all the given graphs, the largest multiple of 1,000 o200k tokens at which every model fits, the prompt "
        "being the largest over every graph and arm")
READINGS = {
    "literal": "entry 60 as written: the prompt + the loop maximum + the reservation",
    "strict": "design log's ruling for Rhea: entry 60 with turn 1's tools, question and template (beyond the system "
              "prompt) added, each model's turn1_overhead",
}
RHEA_GRAPHS = frozenset({"rhea-142", "opaque-rhea-142"})
# Design log's ruling: Qwen3.8 has per-graph tool caps, and on Rhea its cap is not below 80,000 characters.
MIN_TOOL_CAP_RHEA = {"qwen3.8-27b": 80_000}


class FitError(ValueError):
    """Inputs the fit computation refuses."""


def canonical_sha256(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _check_inputs(sizes: dict, models: dict, reading: str, tool_caps: dict[str, int]) -> list[int]:
    budgets = sorted(sizes["budgets"])
    if not budgets:
        raise FitError("no budgets were measured")
    off = [b for b in budgets if b % STEP]
    if off:
        raise FitError(f"the budgets are the multiples of 1,000 o200k tokens, not {off[:5]}")
    grid = list(range(budgets[0], budgets[-1] + STEP, STEP))
    if budgets != grid:
        missing = sorted(set(grid) - set(budgets))
        raise FitError(f"the sizes skip 1,000-token candidates ({missing[:5]}): every multiple of 1,000 in "
                       f"{budgets[0]}..{budgets[-1]} must be measured")
    for g, arms in sizes["graphs"].items():
        for arm_name, arm in arms.items():
            for m in models:
                lacking = [b for b in budgets if m not in arm.get(str(b), {})]
                if lacking:
                    raise FitError(f"{m} has no count at {lacking[:5]} on {g} {arm_name}: every candidate budget "
                                   "needs a measured count (no count is bounded by a neighbour)")
    if reading not in READINGS:
        raise FitError(f"reading {reading!r}: one of {sorted(READINGS)}")
    on_rhea = bool(RHEA_GRAPHS & set(sizes["graphs"]))
    for m, spec in models.items():
        if reading == "strict" and "turn1_overhead" not in spec:
            raise FitError(f"{m}: the strict reading needs its turn1_overhead (turn 1's tools, question and template)")
        if on_rhea and m in MIN_TOOL_CAP_RHEA:
            if m not in tool_caps:
                raise FitError(f"{m}: state its Rhea tool cap (not below {MIN_TOOL_CAP_RHEA[m]:,})")
            if tool_caps[m] < MIN_TOOL_CAP_RHEA[m]:
                raise FitError(f"{m}: its Rhea tool cap {tool_caps[m]:,} is below {MIN_TOOL_CAP_RHEA[m]:,} characters "
                               "(design log's ruling)")
        if m in tool_caps and spec.get("loop_tool_cap_chars") != tool_caps[m]:
            raise FitError(f"{m}: its loop maximum was measured under a {spec.get('loop_tool_cap_chars')} character "
                           f"cap, not the tool cap its runs use ({tool_caps[m]:,})")
    return budgets


def compute(sizes: dict, models: dict, reservation: int, reading: str = "strict",
            tool_caps: dict[str, int] | None = None) -> dict:
    """The largest budget over the whole measured range at which every model fits, with each model's row there."""
    tool_caps = dict(tool_caps or {})
    budgets = _check_inputs(sizes, models, reading, tool_caps)
    rows = {}
    for b in budgets:
        per = {}
        for m, spec in models.items():
            largest, where = None, None
            for g, arms in sizes["graphs"].items():
                for arm_name, arm in arms.items():
                    n = arm[str(b)][m]
                    if largest is None or n > largest:
                        largest, where = n, {"graph": g, "arm": arm_name}
            limit = LIMIT_SHARE * spec["window"]
            turn1 = spec.get("turn1_overhead", 0) if reading == "strict" else 0
            need = largest + turn1 + spec["loop"] + reservation
            per[m] = {"prompt_tokens": largest, "largest_at": where, "turn1_overhead": turn1, "loop": spec["loop"],
                      "reservation": reservation, "need": need, "limit": limit, "margin": limit - need,
                      "fits": need <= limit}
        rows[b] = per
    passing = [b for b in budgets if all(r["fits"] for r in rows[b].values())]
    budget = max(passing) if passing else None
    top = budgets[-1]
    whole = _whole_at(sizes, top)
    if budget == top and not whole:
        raise FitError(f"every model fits at {top}, the largest budget measured, and not every prompt renders whole "
                       "there: a larger budget may fit too; measure on to a budget at which a model fails (or at "
                       "which every prompt renders whole)")
    non_monotone = sorted({b for g in sizes["graphs"].values() for arm in g.values()
                           for b, nxt in itertools.pairwise(budgets) for m in models
                           if arm[str(b)][m] > arm[str(nxt)][m]})
    out = {"schema": SCHEMA, "rule": RULE, "reading": reading, "reading_rule": READINGS[reading],
           "reservation": reservation, "tool_caps": tool_caps, "graphs": sorted(sizes["graphs"]),
           "budgets": [budgets[0], budgets[-1]], "sizes_sha256": canonical_sha256(sizes), "budget": budget,
           "failing_budgets": [b for b in budgets if b not in passing], "non_monotone": non_monotone,
           "range_end": {"budget": top, "fits": top in passing, "whole": whole},
           "models": rows[budget] if budget is not None else {}, "inputs": models}
    if budget is not None:
        margins = {m: r["margin"] for m, r in rows[budget].items()}
        out["binding"] = sorted(m for m, v in margins.items() if v == min(margins.values()))
    nxt = (budget + STEP) if budget is not None else budgets[0]
    if nxt in rows:
        out["failing_at_next"] = sorted(m for m, r in rows[nxt].items() if not r["fits"])
        out["at_next_budget"] = rows[nxt]
    return out


def _whole_at(sizes: dict, budget: int) -> bool:
    """Whether every measured prompt (every graph and arm) is recorded as rendered whole at ``budget``: then every
    larger budget renders the same prompts."""
    prompts = sizes.get("prompts") or {}
    return all((prompts.get(g) or {}).get(arm, {}).get(str(budget), {}).get("truncated") is False
               for g, arms in sizes["graphs"].items() for arm in arms)


def _from_root(path: Path, root: Path) -> str:
    """The path as a fit computation records it: from the repository root when it is under it, else absolute."""
    path, root = Path(path).resolve(), Path(root).resolve()
    return path.relative_to(root).as_posix() if path.is_relative_to(root) else os.fspath(path)


def fit_record(sizes_path: Path, models_path: Path, reservation: int, reading: str, tool_caps: dict[str, int],
               root: Path) -> dict:
    """``compute`` on the sizes and models files, with both files recorded (path from ``root`` and sha256), so that the
    computation can be recomputed from what it names (harness.sources.render_guard)."""
    sizes = json.loads(Path(sizes_path).read_text(encoding="utf-8"))
    models = json.loads(Path(models_path).read_text(encoding="utf-8"))
    out = compute(sizes, models, reservation, reading, tool_caps)
    out["input_files"] = {key: {"path": _from_root(path, root), "sha256": hashlib.sha256(Path(path).read_bytes())
                                .hexdigest()} for key, path in (("sizes", sizes_path), ("models", models_path))}
    return out
