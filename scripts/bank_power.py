#!/usr/bin/env python3
"""Power of the bank contrasts C2 (i), C2 (ii), C3 and C4 under pre-registration §4.1, for the
bank designs (a)-(d) (harness/analysis/bank_power.py).

The power uses only the bank's source-free structure: per-question item counts and their design
maxima. Detection rates are the grid's assumed truths. ``items.jsonl`` is read for the item id,
question, split, kind, bank and fault class only. --exclude drops listed item ids (Rhea's exposed
items, a Rhea-only stratum). --rewrites gives the achievable rewrites per question as JSON
{question_id: k}; without it, --k rewrites everywhere.

    uv run python scripts/bank_power.py --bank BANK_DIR --split test --label "UniProt provisional v1" \\
        --note "built on ..." [--cap 8] [--k 6 | --rewrites k.json] --out DIR [--workers 4]

Writes ``power.json`` and ``power.md`` to --out. No model calls, no store queries.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import hashlib
import json
import statistics
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.analysis.bank_bounds import BankDesign, load_bank_items
from harness.analysis.bank_power import (
    C4_H,
    CENSUS_FPR,
    CENTRAL_FPR,
    CURATED_FPR,
    LEVELS,
    POWER_METHODS,
    RECALL,
    SESOI,
    TYPE_WRONG_RECALL,
    class_counts,
    design_power,
    designs,
)
from harness.analysis.digest import code_digest

CLAIMS = ("C2 precision", "C2 recall", "C3", "C4")
LABELS = {"C2 precision": "C2 (i) precision", "C2 recall": "C2 (ii) type-wrong recall", "C3": "C3 precision",
          "C4": "C4 net benefit"}
METHOD_LABELS = {"eb": "EB (primary)", "kl": "KL (reference)", "kl+eb": "KL+EB split (not ruled)"}


def _sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _job(args: tuple) -> tuple[str, dict]:
    name, design, icc, sens, grid_reps, curve_reps, seed, i = args
    return name, design_power(design, icc, sens, grid_reps, curve_reps, np.random.default_rng([seed, 10 + i]))


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.1f}"


def _eff(x: float | None, unit: str) -> str:
    if x is None:
        return "not reached"
    return f"{100 * x:.0f} pts" if unit == "pts" else f"{x:g}"


def _kind(s: dict, kind: str, *fields: str) -> str:
    node = s.get(kind)
    return "–" if node is None else ", ".join(f"{node[f]:g}" if isinstance(node[f], float) else str(node[f])
                                              for f in fields)


def render(res: dict) -> str:
    p = res["power"]
    lo, hi = (f"{a:.6g}" for a in p["levels"])
    lines = [f"# Bank power by design under §4.1: {res['label']}", "",
             (f"- **Bank:** `{res['bank']}`, split {res['split']}, provisional: {res['provisional']}. items.jsonl "
              f"`{res['items_sha256']}`, manifest.json `{res['manifest_sha256']}`."),
             *([f"- {res['note']}"] if res.get("note") else []),
             (f"- **Excluded items:** {res['excluded']}. **Rewrites:** {res['rewrites']}. **Code digest:** "
              f"`{res['code_digest']}`."),
             (f"- **Assumed truths:** census recall on W {RECALL}, census type-wrong recall {TYPE_WRONG_RECALL}; "
              f"curated FPR grid {list(CURATED_FPR)}, census FPR grid {list(CENSUS_FPR)}; curves at curated "
              f"FPR {CENTRAL_FPR[0]} and recall {RECALL + 0.05} for both sources; C4 union-only f (vs curated, vs "
              f"census) central {tuple(round(x, 4) for x in C4_H['central'])}, low {C4_H['low']}."),
             (f"- **Heterogeneity:** per-question rates Beta(mean, ICC), so items are correlated within a question, "
              f"at ICC {p['icc']} (sensitivity {p['icc_sensitivity']}). Replicates: {p['curve_replicates']} per curve "
              f"point, {p['grid_replicates']} per grid cell."),
             (f"- **Precision** at the fixed prevalence {p['prevalence']}. **SESOIs:** C2 (i) and C3: 5 precision "
              "points; C2 (ii): 5 recall points; C4: Delta = 0.01."),
             ("- **Levels:** 0.05/6 (guaranteed by Holm; drives the expected-inconclusive call) and 0.05 (the most a "
              "claim is tested at). Power in %."), "",
             "## Structure per design", "",
             ("| Design | f | d (C2 i / C2 ii / C3 / C4) | Wrong: total, median, mean, M_W | Type-wrong: total, M_T | "
              "Rewrites: total, M_k |"), "|---|---|---|---|---|---|"]
    for name, e in p["designs"].items():
        s = e["summary"]
        d = s["d"]
        lines.append(f"| {name} | {s['f']} | {d['C2 precision']} / {d['C2 recall']} / {d['C3']} / {d['C4']} | "
                     f"{_kind(s, 'wrong', 'total', 'median', 'mean', 'max')} | {_kind(s, 'type_wrong', 'total', 'max')} | "
                     f"{_kind(s, 'rewrites', 'total', 'max')} |")
    for method in p["methods"]:
        lines += ["", f"## Side by side at the SESOI, and the smallest effect reaching 50% and 80%: {METHOD_LABELS[method]}",
                  "", "Effects: precision or recall points; C4 in Delta.", "",
                  ("| Claim | Design | Power at SESOI, 0.05/6 | at 0.05 | 50% at 0.05/6 | 80% at 0.05/6 | 50% at 0.05 | "
                   "80% at 0.05 |"), "|---|---|---|---|---|---|---|---|"]
        for claim in CLAIMS:
            unit = "delta" if claim == "C4" else "pts"
            for name, e in p["designs"].items():
                a, b = e["curves"]["summary"][claim][f"{method}@{lo}"], e["curves"]["summary"][claim][f"{method}@{hi}"]
                lines.append(f"| {LABELS[claim]} | {name} | {_pct(a['at_sesoi'])} | {_pct(b['at_sesoi'])} | "
                             f"{_eff(a['effect_50'], unit)} | {_eff(a['effect_80'], unit)} | {_eff(b['effect_50'], unit)} | "
                             f"{_eff(b['effect_80'], unit)} |")
    lines += ["", "## ICC sensitivity of the primary bound (EB): smallest effect reaching 50% / 80% at 0.05/6", "",
              "| Claim | Design | " + " | ".join(f"ICC {x}" for x in [p['icc'], *p['icc_sensitivity']]) + " |",
              "|---|---|" + "---|" * (1 + len(p["icc_sensitivity"]))]
    for claim in CLAIMS:
        unit = "delta" if claim == "C4" else "pts"
        for name, e in p["designs"].items():
            cells = [e["curves"]["summary"][claim][f"eb@{lo}"]]
            cells += [e["sensitivity"][f"{x:g}"][claim][f"eb@{lo}"] for x in p["icc_sensitivity"]]
            lines.append(f"| {LABELS[claim]} | {name} | " + " | ".join(
                f"{_eff(c['effect_50'], unit)} / {_eff(c['effect_80'], unit)}" for c in cells) + " |")

    def cell(pw: dict | None) -> str:
        if pw is None:
            return "infeasible"
        return f"{_pct(pw[f'eb@{lo}'])} / {_pct(pw[f'eb@{hi}'])} [{_pct(pw[f'kl@{lo}'])} / {_pct(pw[f'kl@{hi}'])}]"

    lines += ["", "## The ruling's grid (primary EB; KL reference in brackets): power at 0.05/6 / at 0.05", ""]
    for name, e in p["designs"].items():
        lines += [f"### {name}", "", ("C2 (i) and C3 cells: curated FPR, census (or generated) FPR, recall gap, and "
                                      "the precisions each implies at pi = 25%."), "",
                  "| Curated FPR | Other FPR | Recall gap | P curated | P other | Gap | C2 (i) | C3 |",
                  "|---|---|---|---|---|---|---|---|"]
        for c2, c3 in zip(e["grid"]["C2 precision"], e["grid"]["C3"]):
            gap = c2["curated_precision"] - c2["other_precision"]
            lines.append(f"| {c2['curated_fpr']} | {c2['other_fpr']} | {c2['recall_gap']} | "
                         f"{c2['curated_precision']:.3f} | {c2['other_precision']:.3f} | {100 * gap:.1f} | "
                         f"{cell(c2['power'])} | {cell(c3['power'])} |")
        lines += ["", "| C2 (ii) recall gap | Power |", "|---|---|"]
        lines += [f"| {r['gap']} | {cell(r['power'])} |" for r in e["grid"]["C2 recall"]]
        lines += ["", "| C4 union-only f | Delta | Power |", "|---|---|---|"]
        lines += [f"| {r['h']} ({r['h_vs_curated']:.4g}, {r['h_vs_census']:.4g}) | {r['delta']} | {cell(r['power'])} |"
                  for r in e["grid"]["C4"]]
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--exclude", type=Path, action="append", default=[])
    ap.add_argument("--label", required=True)
    ap.add_argument("--note")
    ap.add_argument("--cap", type=int, help="design (b)'s cap on wrong items per question (default: their median)")
    ap.add_argument("--k", type=int, default=6, help="rewrites per question for designs (c) and (d)")
    ap.add_argument("--rewrites", type=Path, help="JSON {question_id: achievable k}, instead of --k")
    ap.add_argument("--icc", type=float, default=0.2)
    ap.add_argument("--icc-sensitivity", type=float, nargs="*", default=[0.05, 0.5])
    ap.add_argument("--grid-replicates", type=int, default=1000)
    ap.add_argument("--curve-replicates", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20261028)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    digest = code_digest()  # the code at launch: the workers fork from this process's loaded modules
    items = load_bank_items(args.bank / "items.jsonl")
    exclude: set[str] = set()
    for path in args.exclude:
        exclude |= {ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
                    if ln.strip() and not ln.startswith("#")}
    base, info = BankDesign.from_items(items, args.split, exclude, gold_only=True)
    by_class = class_counts(items, args.split, exclude)
    order = sorted({i["question_id"] for i in items if i["split"] == args.split})
    if args.rewrites:
        kmap = json.loads(args.rewrites.read_text(encoding="utf-8"))
        rewrites = np.array([float(kmap.get(q, 0)) for q in order])
        rw_text = (f"achievable k from {args.rewrites} (sha256 {_sha(args.rewrites)}), median "
                   f"{statistics.median(rewrites.tolist()):g}, max {rewrites.max():g}, zero on {(rewrites == 0).sum()}")
    else:
        rewrites = np.full(len(order), float(args.k))
        rw_text = f"k = {args.k} per question (a placeholder until the achievable k)"
    cap = args.cap if args.cap is not None else int(statistics.median(base.wrong.tolist()))
    ds = designs(base, by_class, cap, rewrites, args.seed)
    jobs = [(name, d, args.icc, tuple(args.icc_sensitivity), args.grid_replicates, args.curve_replicates, args.seed, i)
            for i, (name, d) in enumerate(ds.items())]
    with Pool(min(args.workers, len(jobs))) as pool:
        results = dict(pool.map(_job, jobs))
    power = {"icc": args.icc, "icc_sensitivity": args.icc_sensitivity, "grid_replicates": args.grid_replicates,
             "curve_replicates": args.curve_replicates, "seed": args.seed, "levels": list(LEVELS),
             "methods": list(POWER_METHODS), "sesoi": SESOI, "prevalence": 0.25, "cap": cap,
             "designs": {name: results[name] for name in ds}}
    manifest = args.bank / "manifest.json"
    provisional = json.loads(manifest.read_text(encoding="utf-8")).get("provisional") if manifest.is_file() else None
    res = {"schema": "bank-power/2", "label": args.label, "note": args.note, "bank": str(args.bank),
           "split": args.split, "provisional": provisional, "items_sha256": _sha(args.bank / "items.jsonl"),
           "manifest_sha256": _sha(manifest), "excluded": info["excluded"], "questions": info["questions"],
           "rewrites": rw_text, "code_digest": digest, "power": power}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "power.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (args.out / "power.md").write_text(render(res), encoding="utf-8")
    print(f"{args.out}/power.md: {info['questions']} questions, cap {cap}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
