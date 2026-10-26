#!/usr/bin/env python3
"""Power and type-I error of asymptotic tests for the redefined bank contrasts (the study lead's request of
18 October, design log; exploratory, not the registered analysis): the studentised paired cluster bootstrap and
the CR2 cluster-robust Wald test with Satterthwaite degrees of freedom (harness/analysis/bank_tests.py), on
designs (a) and (c), for n = 210, 2 x 210 and 3 x 210 questions (the larger banks resample the v1 test split's
question structure within strata).

Source-free: ``items.jsonl`` is read for the item id, question, split, kind, bank, fault class, template and
tier only.

    uv run python scripts/bank_tests_power.py --bank BANK_DIR --out DIR [--workers 4]
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

from dataclasses import replace

from harness.analysis.bank_bounds import BankDesign, load_bank_items
from harness.analysis.bank_power import class_counts, designs, rewrite_counts
from harness.analysis.bank_tests import (
    C4_H,
    CURATED_FPR,
    LEVELS,
    SESOI,
    TESTS,
    TW_RECALL,
    boot_matrix,
    calibrate,
    curves,
    enlarge,
    grid,
    null_p,
    type_one,
)
from harness.analysis.digest import code_digest

CLAIMS = ("C2 ffr", "C2 recall", "C2", "C3", "C4")
LABELS = {"C2 ffr": "C2 (i) FFR difference", "C2 recall": "C2 (ii) recall difference", "C2": "C2 IUT (both at the effect)",
          "C3": "C3 FFR difference (two-sided)", "C4": "C4 Delta (IUT)"}
TEST_LABELS = {"boot_t": "studentised bootstrap", "cr2": "CR2 + Satterthwaite (t, n - 1 df)"}


def _job(args):
    name, design, factor, icc, strata, seed, reps, null_reps, b, with_grid = args
    d = enlarge(design, strata, factor, seed)
    n = d.n
    K = boot_matrix(n, b, seed + n)
    rng = np.random.default_rng([seed, factor, int(icc * 1000), len(name)])
    nulls = null_p(d, icc, null_reps, rng, K)
    calibrated = {c: {t: {f"{a:.6g}": calibrate(nulls[c][t], a) for a in LEVELS} for t in TESTS} for c in nulls}
    out = {"design": name, "factor": factor, "n": n, "icc": icc, "calibrated_alpha": calibrated,
           "null_size": {c: {f"{t}@{a:.6g}": float(np.mean(nulls[c][t] <= a)) for t in TESTS for a in LEVELS}
                         for c in nulls},
           "curves": curves(d, icc, reps, rng, K, calibrated=calibrated), "type_one": type_one(d, icc, null_reps, rng, K)}
    if with_grid:
        out["grid"] = grid(d, icc, reps, rng, K)
    return out


def _pct(x):
    return "–" if x is None else f"{100 * x:.1f}"


def _eff(x, claim):
    if x is None:
        return "not reached"
    return f"{x:g}" if claim == "C4" else f"{100 * x:.0f} pts"


def render(res: dict) -> str:
    lo, hi = (f"{a:.6g}" for a in LEVELS)
    lines = [f"# Bank power of asymptotic tests (exploratory; not the registered analysis): {res['label']}", "",
             (f"- **Bank:** `{res['bank']}`, test split; items.jsonl `{res['items_sha256']}`. Built on the PRE-FREEZE round-2 "
              "question set (0f84b976...), with no source labels; source-free fields only."),
             ("- **Larger banks:** n = 420 and 630 resample the v1 test split's question structure with replacement within "
              "the 14 template x tier strata. A real larger bank from the frozen templates comes later."),
             (f"- **Tests:** {TEST_LABELS['boot_t']} (B = {res['B']}, unstratified, one resample matrix per bank size); "
              f"{TEST_LABELS['cr2']}. Both on the per-question contrasts u_q (see harness/analysis/bank_tests.py)."),
             (f"- **Designs:** (a) as built, f = f_gold; (c) plus k = 6 rewrites (placeholder), f = 0.5 f_gold + 0.5 f_rewrite. "
              f"Assumed truths: curated FFR {CURATED_FPR}, census type-wrong recall {TW_RECALL}; C4 union-only f (vs curated, "
              f"vs census) {C4_H['central']}."),
             *(["- **Design (c) variants** (k_q = min(cap, attainable rewrites); k_q = 0 takes the subset mean): "
                + "; ".join(f"{n}: cap {v['cap']}, excluding {v['excluded_operators'] or 'none'}, k distribution "
                            f"{v['k_distribution']}" for n, v in res["variants"].items())] if res.get("variants") else []),
             (f"- **Heterogeneity:** per-question Beta(mean, ICC) rates, ICC {res['icc']} (sensitivity {res['icc_sensitivity']}); "
              f"sources independent. Replicates: {res['reps']} per curve point at ICC {res['icc']} ({res['sens_reps']} at "
              f"the sensitivity ICCs), {res['null_reps']} for type-I."),
             (f"- **SESOIs** (the existing ones, read on these estimands): C2 (i) a 5-point FFR difference, C2 (ii) 5 recall "
              f"points, C3 a 5-point FFR difference (two-sided), C4 Delta = 0.01. Code digest `{res['code_digest']}`."), ""]
    for test in TESTS:
        lines += [f"## {TEST_LABELS[test]}: power at the SESOI and the smallest effect reaching 50% / 80%", "",
                  ("| Claim | Design | n | SESOI power 0.05/6 | SESOI power 0.05 | 50% at 0.05/6 | 80% at 0.05/6 | 50% at 0.05 | "
                   "80% at 0.05 |"),
                  "|---|---|---|---|---|---|---|---|---|"]
        for claim in CLAIMS:
            for r in res["runs"]:
                if r["icc"] != res["icc"]:
                    continue
                a, b = r["curves"]["summary"][claim][f"{test}@{lo}"], r["curves"]["summary"][claim][f"{test}@{hi}"]
                lines.append(f"| {LABELS[claim]} | {r['design']} | {r['n']} | {_pct(a['at_sesoi'])} | {_pct(b['at_sesoi'])} | "
                             f"{_eff(a['effect_50'], claim)} | {_eff(a['effect_80'], claim)} | {_eff(b['effect_50'], claim)} | "
                             f"{_eff(b['effect_80'], claim)} |")
        lines.append("")
    lines += [("## Power at the empirically size-calibrated level alpha' (from this run's null datasets; the nominal "
               "columns above are optimistic for any test whose size exceeds nominal)"), "",
              ("| Test | Claim | Design | n | alpha'(0.05/6) | SESOI power at alpha'(0.05/6) | 50% / 80% at alpha'(0.05/6) | "
               "alpha'(0.05) | SESOI power at alpha'(0.05) | 50% / 80% at alpha'(0.05) |"), "|---|---|---|---|---|---|---|---|---|---|"]
    for test in TESTS:
        for claim in CLAIMS:
            for r in res["runs"]:
                if r["icc"] != res["icc"]:
                    continue
                s = r["curves"]["summary"][claim]
                a, b = s[f"{test}@{lo}@cal"], s[f"{test}@{hi}@cal"]
                cal = r["calibrated_alpha"]
                if claim == "C2":
                    al = "/".join(f"{cal[c][test][lo]:.5f}" for c in ("C2 ffr", "C2 recall"))
                    ah = "/".join(f"{cal[c][test][hi]:.4f}" for c in ("C2 ffr", "C2 recall"))
                else:
                    al, ah = f"{cal[claim][test][lo]:.5f}", f"{cal[claim][test][hi]:.4f}"
                lines.append(f"| {TEST_LABELS[test]} | {LABELS[claim]} | {r['design']} | {r['n']} | {al} | {_pct(a['at_sesoi'])} | "
                             f"{_eff(a['effect_50'], claim)} / {_eff(a['effect_80'], claim)} | {ah} | {_pct(b['at_sesoi'])} | "
                             f"{_eff(b['effect_50'], claim)} / {_eff(b['effect_80'], claim)} |")
    lines.append("")
    lines += ["## Type-I error at each component's null boundary (rejection rate %, at 0.05/6 / at 0.05; nominal 0.83 / 5.0)", "",
              "| Test | Design | n | ICC | " + " | ".join(res["runs"][0]["type_one"]) + " |",
              "|---|---|---|---|" + "---|" * len(res["runs"][0]["type_one"])]
    for test in TESTS:
        for r in res["runs"]:
            cells = [f"{_pct(v[f'{test}@{lo}'])} / {_pct(v[f'{test}@{hi}'])}" for v in r["type_one"].values()]
            lines.append(f"| {TEST_LABELS[test]} | {r['design']} | {r['n']} | {r['icc']} | " + " | ".join(cells) + " |")
    lines += ["", "## ICC sensitivity: smallest effect reaching 50% at 0.05/6 (studentised bootstrap / CR2)", "",
              "| Claim | Design | n | " + " | ".join(f"ICC {x}" for x in [res['icc'], *res['icc_sensitivity']]) + " |",
              "|---|---|---|" + "---|" * (1 + len(res["icc_sensitivity"]))]
    for claim in CLAIMS:
        for design in sorted({r["design"] for r in res["runs"]}):
            for n in sorted({r["n"] for r in res["runs"]}):
                cells = []
                for icc in [res["icc"], *res["icc_sensitivity"]]:
                    r = next(x for x in res["runs"] if x["design"] == design and x["n"] == n and x["icc"] == icc)
                    s = r["curves"]["summary"][claim]
                    cells.append(f"{_eff(s[f'boot_t@{lo}']['effect_50'], claim)} / {_eff(s[f'cr2@{lo}']['effect_50'], claim)}")
                lines.append(f"| {LABELS[claim]} | {design} | {n} | " + " | ".join(cells) + " |")
    lines += ["", "## Grid (ICC central): C2 IUT and C3 power, studentised bootstrap / CR2, at 0.05/6", ""]
    for r in res["runs"]:
        if "grid" not in r:
            continue
        lines += [f"### {r['design']}, n = {r['n']}", "", "| Curated FFR | Census/draws FFR | Recall gap | C2 | C3 |", "|---|---|---|---|---|"]
        for g in r["grid"]:
            lines.append(f"| {g['curated_fpr']} | {g['other_fpr']} | {g['recall_gap']} | {_pct(g['C2'][f'boot_t@{lo}'])} / "
                         f"{_pct(g['C2'][f'cr2@{lo}'])} | {_pct(g['C3'][f'boot_t@{lo}'])} / {_pct(g['C3'][f'cr2@{lo}'])} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--label", default="UniProt provisional bank v1 (provisional)")
    ap.add_argument("--k", type=int, default=6, help="placeholder rewrites per question when no --variant is given")
    ap.add_argument("--variant", action="append", default=[],
                    help="NAME:CAP[:EXCLUDED_OPERATORS] design (c) at the bank's attainable rewrites, k_q = min(CAP, attainable)")
    ap.add_argument("--icc", type=float, default=0.2)
    ap.add_argument("--icc-sensitivity", type=float, nargs="*", default=[0.05, 0.5])
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--sens-reps", type=int, default=500)
    ap.add_argument("--null-reps", type=int, default=4000)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    digest = code_digest()  # the code at launch: the workers fork from this process's loaded modules
    items = load_bank_items(args.bank / "items.jsonl")
    base, _ = BankDesign.from_items(items, "test", gold_only=True)
    order = sorted({i["question_id"] for i in items if i["split"] == "test"})
    stratum = {}
    for i in items:
        if i["split"] == "test":
            stratum.setdefault(i["question_id"], f"{i.get('template')}|{i.get('tier')}")
    strata = [stratum[q] for q in order]
    cap = int(statistics.median(base.wrong.tolist()))
    variants: dict = {}
    ds = designs(base, class_counts(items, "test"), cap, np.full(len(order), float(args.k)), args.seed)
    chosen = {"a: as built": ds["a: as built"]}
    if not args.variant:
        chosen["c: plus rewrites"] = ds["c: plus rewrites"]
    for spec in args.variant:  # NAME:CAP[:EXCLUDED_OPERATOR,...]: the bank's attainable rewrites, capped
        name, cap_k, *excl = spec.split(":")
        k = rewrite_counts(items, "test", int(cap_k), set(",".join(excl).split(",")) - {""})
        chosen[f"c ({name})"] = replace(base, rewrites=k, gold_only=False)
        variants[f"c ({name})"] = {"cap": int(cap_k), "excluded_operators": sorted(set(",".join(excl).split(",")) - {""}),
                                   "k_distribution": {str(int(v)): int((k == v).sum()) for v in sorted(set(k.tolist()))}}
    jobs = []
    for name, d in chosen.items():
        for factor in (1, 2, 3):
            for icc in [args.icc, *args.icc_sensitivity]:
                central = icc == args.icc
                jobs.append((name, d, factor, icc, strata, args.seed, args.reps if central else args.sens_reps,
                             args.null_reps, args.B, central))
    with Pool(args.workers) as pool:
        runs = pool.map(_job, jobs)
    res = {"schema": "bank-tests-power/1", "label": args.label, "bank": str(args.bank),
           "items_sha256": hashlib.sha256((args.bank / "items.jsonl").read_bytes()).hexdigest(), "B": args.B,
           "icc": args.icc, "icc_sensitivity": args.icc_sensitivity, "reps": args.reps, "sens_reps": args.sens_reps,
           "null_reps": args.null_reps, "seed": args.seed, "sesoi": SESOI, "code_digest": digest, "runs": runs,
           "variants": variants}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "power.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (args.out / "power.md").write_text(render(res), encoding="utf-8")
    print(f"{args.out}/power.md: {len(runs)} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
