#!/usr/bin/env python3
"""Closed-book probe report (design log, E1.7): per graph, model and probe.

    uv run python scripts/probe_summary.py OUT/rhea-142 OUT/uniprot-sp-2026_03 --out probes.md \\
        [--json probes.json] [--rates-only rates.json]

Each directory is one graph's probe run (scripts/run_probes.py); it is summarised only
when its replay check passed, still matches its records and was made under the committed
code in use (harness/agent/replay_check.py), as for agent batches.

Per graph, model and probe (records in ``records/``; a try kept in ``infrastructure/`` is
not an attempt):

* exact match over the attempts (an unparseable answer, a query that did not run cleanly,
  a provider failure or refusal and a context cap all count as wrong), with a 95%
  percentile-bootstrap interval clustered by question: each base question's attempts are
  averaged over its repetitions first and the bootstrap resamples base questions (the
  pilot report's method, harness/pilot_report.py); the same for mean F1;
* probe (a): three disjoint strata over the attempts that reached the model's reply:
  answered, declined (rule ``declined-v1``, harness/probe/parse.py: an empty-rows answer, an
  empty reply, or prose with no JSON object matching a fixed decline phrase) and unparseable
  output; a decline is scored wrong like unparseable output, and reported apart from it;
* probe (b): the parse rate (no query extracted, not one read query, or a store parse
  error), the refused rate (SERVICE), the store-error rate and the time-out rate, each over
  the attempts, and where the query came from (a ``sparql`` fence, a plain fence, the reply);
* attempts that failed infrastructure on every try and tries set aside as ``failed:harness``
  (our defects; both outside the attempts), the provider's own failures (stage "provider"
  only: a store or harness fault is never one), refusals, context caps and replies cut by
  the output cap.

A section for the natural-minus-opaque difference in probe (b) pairs each opaque graph
with its natural graph (``opaque_of`` in the graph list) when both are among the inputs;
the agent's opaque-minus-natural difference joins it once the main runs exist.

``--rates-only FILE`` writes the counts and rates alone (graph, model and probe names, no
question ids, texts, IRIs, queries or replies): the only output meant to leave a private
graph's directory (a15).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Allow running as a plain script from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.agent.replay_check import BatchNotVerified, require_current_check  # noqa: E402
from harness.pilot_report import bootstrap_interval, per_question_means  # noqa: E402
from harness.probe.prompts import get_probe  # noqa: E402
from harness.probe.run import PARSE_CLASSES, QUERY_ERROR_CLASSES  # noqa: E402

DEFAULT_GRAPHS = Path(__file__).resolve().parent.parent / "data" / "probes" / "graphs.json"
RATES_SCHEMA = "probe-rates/1"


def load_dir(path: Path) -> list[dict]:
    """The final records, and the records set aside as failed:harness (harness/, decisions excluded): counted
    apart, never as attempts."""
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((path / "records").glob("*.json"))]
    records += [json.loads(p.read_text(encoding="utf-8")) for p in sorted((path / "harness").glob("*.json"))
                if not p.name.endswith(".setaside.json")]
    return records


def _exact(r: dict) -> float:
    return float(bool((r.get("score") or {}).get("exact")))


def _f1(r: dict) -> float:
    return float((r.get("score") or {}).get("f1") or 0.0)


def _rate(n: int, d: int) -> float | None:
    return n / d if d else None


def _refused_by_provider(r: dict) -> bool:
    return any(t.get("refusal") is not None for t in r.get("transcript", []) if t.get("role") == "assistant")


def summarize_group(records: list[dict], probe: str, n_boot: int = 10_000, seed: int = 0) -> dict:
    infra = [r for r in records if r["outcome"]["failure_type"] == "infrastructure"]
    harness = [r for r in records if r["outcome"]["failure_type"] == "harness"]
    att = [r for r in records if r["outcome"]["failure_type"] not in ("infrastructure", "harness")]
    labels = Counter(r["outcome"]["label"] for r in att)
    exact_q = per_question_means(att, _exact)
    f1_q = per_question_means(att, _f1)
    n = len(att)
    out = {
        "n_attempts": n,
        "n_base_questions": len(exact_q),
        "exact": statistics.fmean(exact_q.values()) if exact_q else None,
        "exact_ci95": bootstrap_interval(list(exact_q.values()), n_boot, seed),
        "mean_f1": statistics.fmean(f1_q.values()) if f1_q else None,
        "mean_f1_ci95": bootstrap_interval(list(f1_q.values()), n_boot, seed),
        "answered_rate": _rate(labels.get("answered", 0), n),
        "infra_failures": len(infra),
        "harness_failures": len(harness),  # our defects, set aside: outside the attempts like infrastructure
        # the provider's own failures only (stage "provider"); a refusal is counted apart
        "provider_failure_rate": _rate(sum(1 for r in att if r["outcome"]["status"] == "failed"
                                           and r["outcome"].get("stage") == "provider"
                                           and not _refused_by_provider(r)), n),
        "provider_refusal_rate": _rate(sum(1 for r in att if _refused_by_provider(r)), n),
        "capped_context_rate": _rate(labels.get("capped:context", 0), n),
        "truncated_rate": _rate(sum(1 for r in att if r["outcome"].get("truncated")), n),
        "labels": dict(sorted(labels.items())),
    }
    if probe == "answer":
        declined = Counter((r.get("answer") or {}).get("declined") for r in att)
        declined.pop(None, None)
        n_declined = sum(declined.values())
        out["answered_rate"] = _rate(labels.get("answered", 0) - declined.get("empty_rows", 0), n)
        out["declined_rate"] = _rate(n_declined, n)
        out["declined_by_rule"] = dict(sorted(declined.items()))
        out["unparseable_rate"] = _rate(labels.get("unparseable", 0) - (n_declined - declined.get("empty_rows", 0)), n)
    else:
        classes = Counter(r["outcome"]["query_error"] for r in att if r["outcome"]["status"] == "query_error")
        out["parse_rate"] = _rate(sum(classes.get(c, 0) for c in PARSE_CLASSES), n)
        out["refused_rate"] = _rate(classes.get("refused", 0), n)
        out["store_error_rate"] = _rate(classes.get("store_error", 0), n)
        out["timeout_rate"] = _rate(classes.get("timeout", 0), n)
        out["query_error_rates"] = {c: _rate(classes.get(c, 0), n) for c in QUERY_ERROR_CLASSES}
        sources = Counter((r.get("query") or {}).get("source") for r in att if r.get("query"))
        out["query_source"] = {k: sources[k] for k in sorted(sources)}
    return out


def summarize(records: list[dict], **kw) -> dict[str, dict[str, dict[str, dict]]]:
    """graph -> model -> probe id -> figures."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:  # by prompt id: a retired prompt's records are never pooled with its successor's
        groups[(r.get("graph") or "?", r["model"], r["prompt"]["id"])].append(r)
    out: dict[str, dict[str, dict[str, dict]]] = {}
    for (graph, model, prompt_id), recs in sorted(groups.items()):
        out.setdefault(graph, {}).setdefault(model, {})[prompt_id] = summarize_group(
            recs, get_probe(prompt_id).probe, **kw)
    return out


def opaque_pairs(summary: dict, graphs: dict[str, dict]) -> list[dict]:
    """Probe (b)'s natural-minus-opaque exact-match difference, per model, where both graphs are summarised."""
    query_id = get_probe("query").id
    rows = []
    for opaque, spec in sorted(graphs.items()):
        natural = spec.get("opaque_of")
        if not natural:
            continue
        both = natural in summary and opaque in summary
        models = sorted(set(summary.get(natural, {})) | set(summary.get(opaque, {})))
        for model in models or [None]:
            a = (summary.get(natural, {}).get(model) or {}).get(query_id) or {}
            b = (summary.get(opaque, {}).get(model) or {}).get(query_id) or {}
            diff = a["exact"] - b["exact"] if both and a.get("exact") is not None and b.get("exact") is not None \
                else None
            rows.append({"natural": natural, "opaque": opaque, "model": model, "natural_exact": a.get("exact"),
                         "opaque_exact": b.get("exact"), "natural_minus_opaque": diff,
                         "agent_opaque_minus_natural": None})
    return rows


def rates_only(summary: dict) -> dict:
    """Counts and rates alone: names of graphs, models and probes, nothing from a question or a reply."""
    keep = ("n_attempts", "n_base_questions", "exact", "exact_ci95", "mean_f1", "mean_f1_ci95", "answered_rate",
            "infra_failures", "harness_failures", "provider_failure_rate", "provider_refusal_rate",
            "capped_context_rate",
            "truncated_rate", "declined_rate", "declined_by_rule", "unparseable_rate", "parse_rate", "refused_rate",
            "store_error_rate", "timeout_rate", "query_error_rates")
    return {"schema": RATES_SCHEMA, "graphs": {
        graph: {model: {probe: {k: v for k, v in fig.items() if k in keep} for probe, fig in probes.items()}
                for model, probes in models.items()} for graph, models in summary.items()}}


def _pct(v) -> str:
    return "—" if v is None else f"{100 * v:.1f}%"


def _ci(ci) -> str:
    return "—" if not ci else f"[{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]"


def render_markdown(summary: dict, pairs: list[dict]) -> str:
    lines = ["# Closed-book probes (E1.7)", "",
             "Exact match over attempts (wrong: unparseable, query errors, provider failures, refusals, context "
             "caps); 95% intervals by a bootstrap over base questions, repetitions averaged first.", ""]
    for graph, models in summary.items():
        lines += [f"## {graph}", "", "### Probe (a): answer from memory", "",
                  "| model | prompt | N | questions | exact | 95% CI | F1 | declined | unparseable | provider fail | "
                  "refusal | truncated | infra (excluded) |",
                  "|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
        for model, probes in models.items():
            for prompt_id, f in probes.items():
                if get_probe(prompt_id).probe != "answer":
                    continue
                lines.append(f"| {model} | {prompt_id} | {f['n_attempts']} | {f['n_base_questions']} | "
                             f"{_pct(f['exact'])} | {_ci(f['exact_ci95'])} | {_pct(f['mean_f1'])} | "
                             f"{_pct(f['declined_rate'])} | {_pct(f['unparseable_rate'])} | "
                             f"{_pct(f['provider_failure_rate'])} | {_pct(f['provider_refusal_rate'])} | "
                             f"{_pct(f['truncated_rate'])} | {f['infra_failures']} |")
        lines += ["", "### Probe (b): query from memory, executed", "",
                  "| model | N | questions | exact | 95% CI | F1 | parse | refused | store error | time-out | "
                  "provider fail | infra (excluded) | query from |",
                  "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|"]
        for model, probes in models.items():
            f = probes.get(get_probe("query").id)
            if f:
                source = ", ".join(f"{k}: {v}" for k, v in f["query_source"].items())
                lines.append(f"| {model} | {f['n_attempts']} | {f['n_base_questions']} | {_pct(f['exact'])} | "
                             f"{_ci(f['exact_ci95'])} | {_pct(f['mean_f1'])} | {_pct(f['parse_rate'])} | "
                             f"{_pct(f['refused_rate'])} | {_pct(f['store_error_rate'])} | {_pct(f['timeout_rate'])} "
                             f"| {_pct(f['provider_failure_rate'])} | {f['infra_failures']} | {source} |")
        lines.append("")
    lines += ["## Natural minus opaque, probe (b)", "",
              "Placeholder until both graphs of a pair are run: the difference of exact match (natural minus opaque) "
              "per model, beside the agent's opaque-minus-natural difference once the main runs exist.", "",
              "| natural | opaque | model | natural exact | opaque exact | natural − opaque | agent opaque − natural |",
              "|---|---|---|---:|---:|---:|---:|"]
    for row in pairs:
        diff = "pending" if row["natural_minus_opaque"] is None else f"{100 * row['natural_minus_opaque']:+.1f} pt"
        lines.append(f"| {row['natural']} | {row['opaque']} | {row['model'] or '—'} | {_pct(row['natural_exact'])} | "
                     f"{_pct(row['opaque_exact'])} | {diff} | pending (main runs) |")
    return "\n".join(lines) + "\n"


def build(dirs: list[Path], graphs_config: Path = DEFAULT_GRAPHS, *, require_replay_check: bool = True,
          n_boot: int = 10_000) -> tuple[dict, list[dict], str]:
    records: list[dict] = []
    for d in dirs:
        if require_replay_check:
            require_current_check(d)
        records += load_dir(d)
    summary = summarize(records, n_boot=n_boot)
    graphs = json.loads(graphs_config.read_text(encoding="utf-8"))["graphs"]
    pairs = opaque_pairs(summary, graphs)
    return summary, pairs, render_markdown(summary, pairs)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("dirs", nargs="+", type=Path, help="one probe run directory per graph (OUT/<graph>)")
    p.add_argument("--graphs-config", type=Path, default=DEFAULT_GRAPHS)
    p.add_argument("--out", type=Path, help="the report (Markdown); printed when omitted")
    p.add_argument("--json", type=Path, help="the figures as JSON")
    p.add_argument("--rates-only", type=Path, help="counts and rates alone (for a private graph)")
    args = p.parse_args(argv)
    try:
        summary, pairs, markdown = build(args.dirs, args.graphs_config)
    except BatchNotVerified as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.rates_only:
        args.rates_only.write_text(json.dumps(rates_only(summary), indent=1) + "\n", encoding="utf-8")
    if args.json:
        args.json.write_text(json.dumps({"summary": summary, "opaque_pairs": pairs}, indent=1) + "\n",
                             encoding="utf-8")
    if args.out:
        args.out.write_text(markdown, encoding="utf-8")
    elif not args.rates_only:
        print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
