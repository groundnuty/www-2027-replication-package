#!/usr/bin/env python3
"""Size the bank-only extension (change-log entry 41): the smallest N_test at which C2, the IUT, reaches 80% power at
its size-calibrated alpha'(0.05/6), then the full power table at that N at the calibrated levels.

Structure only: the bank v2 test split's per-question counts (source-free fields), with k_q = min(available, 3)
rewrites excluding rewrite_redundant_join (entry 41). For N beyond 210 the per-question structure is resampled within
template x tier strata, keeping 30 test questions per template at n = 210 in proportion (10 lookup, 20 compositional);
the real extension's structure will replace it. Tests and estimands as harness/analysis/bank_tests.py.

    uv run python scripts/bank_size.py --bank BANK_V2 --out DIR [--workers 4]
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.analysis.bank_bounds import BankDesign, load_bank_items
from harness.analysis.bank_power import rewrite_counts
from harness.analysis.bank_tests import (
    DELTAS,
    EFFECTS,
    LEVELS,
    SESOI,
    TESTS,
    boot_matrix,
    lf_alpha,
    lf_null_p,
    sizing_power,
    type_one,
)
from harness.analysis.digest import code_digest

JOIN = "rewrite_redundant_join"
PER_TEMPLATE = (30, 45, 60, 75, 90, 105, 120, 150, 180, 240)  # test questions per template; 7 templates
CLAIMS = ("C2 ffr", "C2 recall", "C2", "C3", "C4")


def _base(bank: Path):
    items = load_bank_items(bank / "items.jsonl")
    base, _ = BankDesign.from_items(items, "test", gold_only=True)
    order = sorted({i["question_id"] for i in items if i["split"] == "test"})
    stratum = {}
    for i in items:
        if i["split"] == "test" and i.get("kind") == "control":
            stratum[i["question_id"]] = f"{i.get('template')}|{i.get('tier')}"
    k = rewrite_counts(items, "test", 3, {JOIN})
    design = replace(base, rewrites=k, gold_only=False)
    return design, [stratum[q] for q in order], k, hashlib.sha256((bank / "items.jsonl").read_bytes()).hexdigest()


def _at(design, strata, per_template, seed):
    factor_num = per_template  # enlarge by an integer factor of 15-question steps: resample each stratum to size
    rng = np.random.default_rng([seed, per_template])
    idx = []
    for s in sorted(set(strata)):
        members = [i for i, t in enumerate(strata) if t == s]
        size = len(members) * factor_num // 30
        idx.extend(members if size == len(members) else rng.choice(members, size=size, replace=True).tolist())
    idx = np.array(idx)
    return replace(design, wrong=design.wrong[idx], type_wrong=design.type_wrong[idx], gold=design.gold[idx],
                   rewrites=design.rewrites[idx], neutral=design.neutral[idx])


def _size_job(args):
    design, strata, per_template, seed, null_reps, reps, b = args
    d = _at(design, strata, per_template, seed)
    K = boot_matrix(d.n, b, seed + d.n)
    rng = np.random.default_rng([seed, d.n])
    alpha = lf_alpha(lf_null_p(d, "C2", null_reps, rng, K))
    power = sizing_power(d, "C2", SESOI["C2 ffr"], 0.2, reps, rng, K, alpha)
    return {"per_template": per_template, "n": d.n, "alpha": alpha, "power": power}


def _full_job(args):
    design, strata, per_template, seed, null_reps, reps, b = args
    d = _at(design, strata, per_template, seed)
    K = boot_matrix(d.n, b, seed + d.n + 1)
    rng = np.random.default_rng([seed, d.n, 1])
    out = {"per_template": per_template, "n": d.n, "claims": {}}
    for claim in CLAIMS:
        alpha = lf_alpha(lf_null_p(d, claim, null_reps, rng, K))
        grid = DELTAS if claim == "C4" else EFFECTS
        curve = [(e, sizing_power(d, claim, e, 0.2, reps, rng, K, alpha)) for e in grid]
        sesoi = SESOI["C2 ffr"] if claim == "C2" else SESOI[claim]
        keys = [f"{t}@{a:.6g}{c}" for t in TESTS for a in LEVELS for c in ("", "@cal")]

        def first(key, target, curve=curve):
            return next((e for e, p in curve if p is not None and p[key] >= target), None)

        at = dict(curve).get(sesoi)
        out["claims"][claim] = {"alpha": alpha, "at_sesoi": {k: at[k] for k in keys} if at else None,
                                "effect_50": {k: first(k, 0.5) for k in keys}, "effect_80": {k: first(k, 0.8) for k in keys}}
    out["type_one"] = type_one(d, 0.2, null_reps, rng, K)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--null-reps", type=int, default=4000)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261030)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--target", type=float, default=0.8)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    digest = code_digest()
    design, strata, k, sha = _base(args.bank)
    jobs = [(design, strata, t, args.seed, args.null_reps, args.reps, args.B) for t in PER_TEMPLATE]
    with Pool(args.workers) as pool:
        sizes = pool.map(_size_job, jobs)
    key = f"boot_t@{0.05 / 6:.6g}@cal"
    chosen = next((s for s in sizes if s["power"][key] >= args.target), None)
    full_at = sorted({30, chosen["per_template"] if chosen else PER_TEMPLATE[-1]})
    with Pool(min(args.workers, len(full_at))) as pool:
        full = pool.map(_full_job, [(design, strata, t, args.seed, args.null_reps, args.reps, args.B) for t in full_at])
    res = {"schema": "bank-size/1", "bank": str(args.bank), "items_sha256": sha, "code_digest": digest,
           "k_distribution": {str(int(v)): int((k == v).sum()) for v in sorted(set(k.tolist()))},
           "settings": {"null_reps": args.null_reps, "reps": args.reps, "B": args.B, "seed": args.seed, "icc": 0.2,
                        "lf_icc": [0.2, 0.5], "target": args.target},
           "sizes": sizes, "chosen": chosen, "full": full}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "size.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    lo = f"{0.05 / 6:.6g}"
    lines = ["# Sizing the bank-only extension (entry 41)", "",
             (f"- bank `{args.bank}` (items {sha[:12]}...), code digest `{digest}`; k_q = min(available, 3) without "
              f"{JOIN}: {res['k_distribution']}; structure resampled within template x tier strata beyond 210 (not a "
              "real bank)."), "",
             "| Per template | N | alpha'(0.05/6) boot-t / CR2 | C2 IUT power at alpha'(0.05/6), boot-t / CR2 | at nominal 0.05/6 |",
             "|---|---|---|---|---|"]
    for s in sizes:
        a, p = s["alpha"], s["power"]
        lines.append(f"| {s['per_template']} | {s['n']} | {a['boot_t'][lo]:.5f} / {a['cr2'][lo]:.5f} | "
                     f"{100 * p[f'boot_t@{lo}@cal']:.1f} / {100 * p[f'cr2@{lo}@cal']:.1f} | {100 * p[f'boot_t@{lo}']:.1f} |")
    lines += ["", f"**Chosen:** {chosen['n'] if chosen else 'not reached within the grid'}", ""]
    for f in full:
        lines += [f"## Full table at N = {f['n']} ({f['per_template']} per template), bootstrap-t at the calibrated alpha'", "",
                  "| Claim | alpha'(0.05/6) | SESOI power at alpha'(0.05/6) | 50% / 80% at alpha'(0.05/6) | alpha'(0.05) | SESOI power at alpha'(0.05) |",
                  "|---|---|---|---|---|---|"]
        for c, v in f["claims"].items():
            kk = f"boot_t@{lo}@cal"
            kh = "boot_t@0.05@cal"
            at = v["at_sesoi"] or {}
            lines.append(f"| {c} | {v['alpha']['boot_t'][lo]:.5f} | {100 * at.get(kk, float('nan')):.1f} | "
                         f"{v['effect_50'][kk]} / {v['effect_80'][kk]} | {v['alpha']['boot_t']['0.05']:.4f} | "
                         f"{100 * at.get(kh, float('nan')):.1f} |")
        lines.append("")
    (args.out / "size.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{args.out}/size.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
