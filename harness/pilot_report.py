"""Pilot report over run records: throughput, tokens, cost, accuracy and failures.

The unit of analysis is the base question (proposal §7.3): each question's
attempts are averaged over its repetitions first, and the bootstrap resamples
base questions, keeping a question's repetitions together. Failed attempts and
abstentions count as wrong; N_att counts every attempt, N_out only scored
outputs.
"""

from __future__ import annotations

import json
import random
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

CostFn = Callable[[dict[str, Any], dict[str, int]], float | None]


def load_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        if path.is_dir():
            records.extend(json.loads(p.read_text()) for p in sorted(path.glob("*.json")))
        else:
            records.append(json.loads(path.read_text()))
    return records


def _tokens(record: dict[str, Any]) -> dict[str, int]:
    tokens = record.get("tokens") or {}
    totals = tokens.get("totals", tokens.get("total", tokens))
    return {k: int(totals.get(k) or 0) for k in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")}


def _scored(record: dict[str, Any]) -> dict[str, Any] | None:
    score = record.get("score") or {}
    return score if record["outcome"]["status"] == "submitted" and score.get("outcome") == "scored" else None


def _exact(record: dict[str, Any]) -> bool:
    return bool((_scored(record) or {}).get("exact"))


def _f1(record: dict[str, Any]) -> float:
    return float((_scored(record) or {}).get("f1") or 0.0)


def _dist(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p95": None}
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
    return {"mean": statistics.fmean(ordered), "median": statistics.median(ordered), "p95": p95}


def per_question_means(records: Sequence[dict[str, Any]], metric: Callable[[dict], float]) -> dict[str, float]:
    """Mean of a metric over each base question's attempts."""
    by_question: dict[str, list[float]] = defaultdict(list)
    for r in records:
        by_question[r["base_question_id"]].append(metric(r))
    return {q: statistics.fmean(v) for q, v in sorted(by_question.items())}


def bootstrap_interval(values: Sequence[float], n_boot: int = 10_000, seed: int = 0, level: float = 0.95) -> tuple[float, float] | None:
    """Percentile bootstrap of the mean, resampling the given units (base questions)."""
    if not values:
        return None
    rng = random.Random(seed)
    n = len(values)
    means = sorted(statistics.fmean(rng.choices(values, k=n)) for _ in range(n_boot))
    lo = means[int((1 - level) / 2 * n_boot)]
    hi = means[min(n_boot - 1, int((1 + level) / 2 * n_boot))]
    return lo, hi


def summarize(records: Sequence[dict[str, Any]], cost_fn: CostFn | None = None, n_boot: int = 10_000, seed: int = 0) -> dict[str, Any]:
    """Summary for one model (or any group of records)."""
    statuses = Counter(r["outcome"]["status"] for r in records)
    failures = Counter(r["outcome"].get("failure_type") or "unknown" for r in records if r["outcome"]["status"] == "failed")
    capped = [r for r in records if r["outcome"]["status"] == "capped"]
    caps = Counter(r["outcome"].get("cap") or "unknown" for r in capped)
    turn_totals = [(r.get("tokens") or {}).get("totals") or {} for r in records]
    n_turns = sum(int(t.get("turns") or 0) for t in turn_totals)
    n_truncated = sum(int(t.get("truncated_turns") or 0) for t in turn_totals)
    acc_by_q = per_question_means(records, lambda r: float(_exact(r)))
    f1_by_q = per_question_means(records, _f1)
    walls = [float(r["wall_s"]) for r in records if r.get("wall_s") is not None]
    tokens = [_tokens(r) for r in records]
    costs = [c for c in ((cost_fn(r.get("route") or {}, t) if cost_fn else None) for r, t in zip(records, tokens)) if c is not None]

    starts = [datetime.fromisoformat(r["started_at"]) for r in records if r.get("started_at")]
    ends = [datetime.fromisoformat(r["ended_at"]) for r in records if r.get("ended_at")]
    span_h = (max(ends) - min(starts)).total_seconds() / 3600 if starts and ends else None

    by_template: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_template[r.get("template") or "unknown"].append(r)

    return {
        "N_att": len(records),
        # A submission counts as an output only with an explicit scored result;
        # one whose scoring failed is still an attempt (and wrong), not an output.
        "N_out": sum(1 for r in records if r["outcome"]["status"] == "submitted" and (r.get("score") or {}).get("outcome") == "scored"),
        "n_unscored_submissions": sum(1 for r in records if r["outcome"]["status"] == "submitted" and (r.get("score") or {}).get("outcome") != "scored"),
        "n_abstained": statuses.get("abstained", 0),
        "n_failed": statuses.get("failed", 0),
        "failure_types": dict(sorted(failures.items())),
        "n_capped": len(capped),
        # capped:context (design log, agent-run/9) is listed once any attempt hit it, so earlier summaries keep their keys
        "cap_rate": {c: caps.get(c, 0) / len(records)
                     for c in ("turns", "wall", *(("context",) if caps.get("context") else ()))} if records else {},
        "n_capped_looping": sum(1 for r in capped if r["outcome"].get("looping")),
        "turns_total": n_turns,
        "truncated_turn_share": n_truncated / n_turns if n_turns else None,
        "n_base_questions": len(acc_by_q),
        "accuracy_exact": statistics.fmean(acc_by_q.values()) if acc_by_q else None,
        "accuracy_exact_ci95": bootstrap_interval(list(acc_by_q.values()), n_boot, seed),
        "mean_f1": statistics.fmean(f1_by_q.values()) if f1_by_q else None,
        "mean_f1_ci95": bootstrap_interval(list(f1_by_q.values()), n_boot, seed),
        "accuracy_by_template": {
            t: statistics.fmean(per_question_means(rs, lambda r: float(_exact(r))).values()) for t, rs in sorted(by_template.items())
        },
        "wall_s": _dist(walls),
        "runs_per_hour_sequential": 3600 / statistics.fmean(walls) if walls else None,
        "runs_per_hour_observed": len(records) / span_h if span_h else None,
        "input_tokens": _dist([t["input_tokens"] + t["cache_read_tokens"] + t["cache_write_tokens"] for t in tokens]),
        "output_tokens": _dist([t["output_tokens"] for t in tokens]),
        "cost_usd_per_run": _dist(costs),
        "cost_usd_total": sum(costs) if costs else None,
    }


def summarize_by_model(records: Sequence[dict[str, Any]], cost_fn: CostFn | None = None, **kw) -> dict[str, dict[str, Any]]:
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_model[r["model"]].append(r)
    return {m: summarize(rs, cost_fn, **kw) for m, rs in sorted(by_model.items())}


def _fmt(x: Any, nd: int = 1) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:,.{nd}f}"
    return f"{x:,}" if isinstance(x, int) else str(x)


def render_markdown(by_model: dict[str, dict[str, Any]]) -> str:
    lines = [
        "| Model | N_att | N_out | Abstained | Failed | Accuracy (exact) [95% CI] | Mean F1 [95% CI] | Runs/h (sequential) | Runs/h (observed) | Wall s (median) | Input tokens (median) | Output tokens (median) | Cost/run USD (mean, est.) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for model, s in by_model.items():
        acc, ci = s["accuracy_exact"], s["accuracy_exact_ci95"]
        f1, fci = s["mean_f1"], s["mean_f1_ci95"]
        lines.append(
            f"| {model} | {s['N_att']} | {s['N_out']} | {s['n_abstained']} | {s['n_failed']} | "
            f"{_fmt(acc, 3)} [{_fmt(ci[0], 3) if ci else 'n/a'}, {_fmt(ci[1], 3) if ci else 'n/a'}] | "
            f"{_fmt(f1, 3)} [{_fmt(fci[0], 3) if fci else 'n/a'}, {_fmt(fci[1], 3) if fci else 'n/a'}] | "
            f"{_fmt(s['runs_per_hour_sequential'])} | {_fmt(s['runs_per_hour_observed'])} | {_fmt(s['wall_s']['median'])} | "
            f"{_fmt(s['input_tokens']['median'], 0)} | {_fmt(s['output_tokens']['median'], 0)} | {_fmt(s['cost_usd_per_run']['mean'], 4)} |"
        )
    lines.append("")
    for model, s in by_model.items():
        fails = ", ".join(f"{k}: {v}" for k, v in s["failure_types"].items()) or "none"
        per_t = ", ".join(f"{t} {_fmt(a, 2)}" for t, a in s["accuracy_by_template"].items())
        caps = ", ".join(f"capped:{c} {_fmt(100 * v, 1)}%" for c, v in s["cap_rate"].items())
        trunc = s["truncated_turn_share"]
        flag = lambda v: " (over 1%)" if v is not None and v > 0.01 else ""
        lines.append(
            f"- **{model}**: failure types: {fails}. Caps: {caps} ({s['n_capped_looping']} of {s['n_capped']} capped runs looping)"
            f"{flag(max(s['cap_rate'].values(), default=0))}. Truncated turns: "
            f"{_fmt(100 * trunc, 2) if trunc is not None else 'n/a'}% of {s['turns_total']}{flag(trunc)}. Accuracy by template: {per_t}."
        )
    return "\n".join(lines) + "\n"
