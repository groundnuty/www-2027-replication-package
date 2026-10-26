"""Power of the bank contrasts C2 (i), C2 (ii), C3 and C4 under pre-registration §4.1 (the
empirical-Bernstein projection), across the bank designs the study lead chooses between
(design log, design log).

**What the power may use** (the study lead's guard). Only source-free structure of the bank's
split: per-question item counts and their design maxima, and the report of median against
maximum. Detection rates are ASSUMED truths on the grid, never estimated from any source's labels.
``bank_bounds.load_bank_items`` reads only the item id, question, split, kind, bank and fault class.

**Designs:**
- **(a) as built.**
- **(b)** wrong items capped per question at a cap (the median) by seeded subsampling without
  replacement. A question with fewer keeps all of its items. The cap affects recall only.
- **(c)** as built, plus k_q answer-preserving rewrites per question (k = 6 everywhere until the
  achievable k arrives). Rewrites enter f through f_rewrite, with M = max k_q.
- **(d)** both.

Designs (a) and (b) have no rewrites, so their false-flag rate is f_gold alone
(``BankDesign.gold_only``). That is not §4.1's registered f = 0.5 f_gold + 0.5 f_rewrite, which
needs rewrites; the report says so.

**Data-generating process.** These are assumptions of the power grid, not of the analysis.
- Every coordinate of every source draws a per-question rate p_q from a Beta distribution with
  the coordinate's mean and intra-question correlation ``icc``, i.e. a = mu (1/icc - 1) and
  b = (1 - mu)(1/icc - 1). The count is Binomial(N_q, p_q).
- Rates are therefore heterogeneous across questions, and items within a question are correlated
  with correlation ``icc``. Every estimand equals its mu exactly.
- Sources are drawn independently. Each claim's bound is a union over the coordinates' marginal
  bounds, so the dependence between sources does not change power.
- **C4** is drawn on its own union-only coordinates. The union-only recall is g_s, solved from
  the target Delta_s = 0.25 g_s - 0.75 h_s, and h_s is the union-only false-flag rate.

**Power** is the share of replicates in which the claim holds:
- at a = 0.05/6, the level Holm guarantees; this drives the expected-inconclusive call;
- at a = 0.05, the most a claim can be tested at.

The power is computed for the primary bound (``eb``) and for the KL-Hoeffding reference
(``kl``).
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from harness.analysis.bank_bounds import (
    PREVALENCE,
    BankDesign,
    SourceCounts,
    UnionCounts,
    c2_precision_interval,
    c2_recall_interval,
    c3_interval,
    c4_interval,
)

CLASSES = ("type_wrong", "vocabulary_absent", "meaning", "stale_in_data")
LEVELS = (0.05 / 6, 0.05)
POWER_METHODS = ("eb", "kl", "kl+eb")  # eb the primary (§4.1); kl the withdrawn reference; kl+eb not ruled
SESOI = {"C2 precision": 0.05, "C2 recall": 0.05, "C3": 0.05, "C4": 0.01}  # §2 (C3 registered in entry 39)
RECALL = 0.55  # census recall on all wrong items (assumed); curated = RECALL + gap
TYPE_WRONG_RECALL = 0.5  # census type-wrong recall (assumed)
CURATED_FPR = (0.0, 0.02, 0.05)  # the ruling's grid
CENSUS_FPR = (0.05, 0.10, 0.20)
RECALL_GAPS = (0.0, 0.05, 0.15)
CENTRAL_FPR = (0.02, 0.10)  # (curated, census or generated) for the curves
EFFECT_GRID = tuple(round(0.01 * i, 2) for i in range(41))
DELTA_GRID = tuple(round(0.005 * i, 3) for i in range(41))


def c4_h(f_cur: float, f_cen: float) -> tuple[float, float]:
    """Union-only false-flag rates if the two sources flag controls independently: over curated it is
    f_cen (1 - f_cur), over census f_cur (1 - f_cen)."""
    return f_cen * (1 - f_cur), f_cur * (1 - f_cen)


C4_H = {"central": c4_h(*CENTRAL_FPR), "low": (0.02, 0.005)}  # union-only f over (curated, census)


# --- designs -----------------------------------------------------------------------------------


def class_counts(items: Iterable[Mapping[str, Any]], split: str, exclude: Collection[str] = ()) -> dict[str, np.ndarray]:
    """Wrong items per question by fault class, the split's questions sorted by id."""
    items = list(items)
    per: dict[str, Counter] = {k: Counter() for k in CLASSES}
    order = sorted({i["question_id"] for i in items if i["split"] == split})
    for i in items:
        if i["split"] == split and i["id"] not in exclude and i["bank"] == "answer_changing":
            if i["fault_class"] not in per:
                raise ValueError(f"item {i['id']}: unknown fault class {i['fault_class']!r}")
            per[i["fault_class"]][i["question_id"]] += 1
    return {k: np.array([float(per[k][q]) for q in order]) for k in CLASSES}


def rewrite_counts(items: Iterable[Mapping[str, Any]], split: str, cap: int,
                   exclude_operators: Collection[str] = ()) -> np.ndarray:
    """Rewrites per question (the split's questions sorted by id): min(cap, attainable), the attainable ones being
    the items of kind "rewrite" whose operator is not excluded (bank v2)."""
    items = list(items)
    order = sorted({i["question_id"] for i in items if i["split"] == split})
    have: Counter = Counter(i["question_id"] for i in items if i["split"] == split and i.get("kind") == "rewrite"
                            and i.get("operator") not in exclude_operators)
    return np.array([float(min(cap, have[q])) for q in order])


def cap_wrong(by_class: Mapping[str, np.ndarray], cap: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Design (b): per question keep min(W_q, cap) wrong items by uniform subsampling without replacement."""
    n = len(next(iter(by_class.values())))
    out = {k: np.zeros(n) for k in CLASSES}
    for q in range(n):
        labels = [k for k in CLASSES for _ in range(int(by_class[k][q]))]
        if len(labels) > cap:
            keep = sorted(rng.choice(len(labels), size=cap, replace=False))
            labels = [labels[i] for i in keep]
        for k in labels:
            out[k][q] += 1
    return out


def designs(base: BankDesign, by_class: Mapping[str, np.ndarray], cap: int, rewrites: np.ndarray,
            seed: int) -> dict[str, BankDesign]:
    """The four designs (a)-(d)."""
    capped = cap_wrong(by_class, cap, np.random.default_rng([seed, 2]))
    cw, ct = sum(capped.values()), capped["type_wrong"]
    none = np.zeros(base.n)
    rw = np.asarray(rewrites, dtype=np.float64)
    return {
        "a: as built": replace(base, rewrites=none, gold_only=True),
        f"b: wrong capped at {cap}": replace(base, wrong=cw, type_wrong=ct, rewrites=none, gold_only=True),
        "c: plus rewrites": replace(base, rewrites=rw, gold_only=False),
        f"d: capped at {cap}, plus rewrites": replace(base, wrong=cw, type_wrong=ct, rewrites=rw, gold_only=False),
    }


def design_summary(d: BankDesign) -> dict[str, Any]:
    out: dict[str, Any] = {"questions": d.n, "f": "f_gold only (not the registered f)" if d.gold_only else
                           "0.5 f_gold + 0.5 f_rewrite (registered)", "coordinates_per_source": d.coords_per_source,
                           "d": {c: d.d_coords(c) for c in ("C2 precision", "C2 recall", "C3", "C4")}}
    for kind in ("wrong", "type_wrong", "gold", "rewrites", "neutral"):
        v = getattr(d, kind)
        if len(v) == 0 or float(v.sum()) == 0:
            continue
        mx = float(v.max())
        out[kind] = {"total": int(v.sum()), "mean": float(v.mean()), "median": float(statistics.median(v.tolist())),
                     "max": int(mx), "share_at_max": float((v == mx).mean()), "max_over_mean": mx / float(v.mean())}
    return out


# --- simulation --------------------------------------------------------------------------------


def beta_binomial(n_q: np.ndarray, mu: float, icc: float, replicates: int, rng: np.random.Generator) -> np.ndarray:
    """(replicates x questions) counts: p_q ~ Beta(mu, icc) per replicate and question, then Binomial(N_q, p_q)."""
    shape = (replicates, len(n_q))
    if mu <= 0 or mu >= 1:
        p = np.full(shape, float(min(max(mu, 0.0), 1.0)))
    else:
        k = 1 / icc - 1
        p = rng.beta(mu * k, (1 - mu) * k, size=shape)
    return rng.binomial(np.broadcast_to(n_q, shape).astype(np.int64), p).astype(np.float64)


def simulate_source(d: BankDesign, recall: float, tw_recall: float, fpr: float, icc: float, replicates: int,
                    rng: np.random.Generator) -> SourceCounts:
    return SourceCounts(beta_binomial(d.wrong, recall, icc, replicates, rng),
                        beta_binomial(d.type_wrong, tw_recall, icc, replicates, rng),
                        beta_binomial(d.gold, fpr, icc, replicates, rng),
                        beta_binomial(d.rewrites, fpr, icc, replicates, rng))


def precision_at(recall: float, fpr: float, prevalence: float = PREVALENCE) -> float | None:
    den = prevalence * recall + (1 - prevalence) * fpr
    return prevalence * recall / den if den > 0 else None


def fpr_for_precision(recall: float, precision: float, prevalence: float = PREVALENCE) -> float | None:
    """The false-flag rate giving a precision at the prevalence; None when outside [0, 1]."""
    if precision <= 0:
        return None
    f = prevalence * recall * (1 - precision) / ((1 - prevalence) * precision)
    return f if 0 <= f <= 1 else None


def _keys(levels: Sequence[float], methods: Sequence[str]) -> list[str]:
    return [f"{m}@{a:.6g}" for m in methods for a in levels]


def c2_precision_power(d: BankDesign, r_cur: float, r_cen: float, f_cur: float, f_cen: float, icc: float,
                       replicates: int, rng: np.random.Generator, levels=LEVELS, methods=POWER_METHODS) -> dict:
    cur = simulate_source(d, r_cur, TYPE_WRONG_RECALL, f_cur, icc, replicates, rng)
    cen = simulate_source(d, r_cen, TYPE_WRONG_RECALL, f_cen, icc, replicates, rng)
    return {f"{m}@{a:.6g}": float(np.mean(c2_precision_interval(cur, cen, d, a, m)[0] > 0))
            for m in methods for a in levels}


def c2_recall_power(d: BankDesign, r_cur: float, r_cen: float, icc: float, replicates: int, rng: np.random.Generator,
                    levels=LEVELS, methods=POWER_METHODS) -> dict:
    k_cur = beta_binomial(d.type_wrong, r_cur, icc, replicates, rng)
    k_cen = beta_binomial(d.type_wrong, r_cen, icc, replicates, rng)
    return {f"{m}@{a:.6g}": float(np.mean(c2_recall_interval(k_cur, k_cen, d, a, m)[0] > 0))
            for m in methods for a in levels}


def c3_power(d: BankDesign, r_cur: float, r_gen: float, f_cur: float, f_gen: float, icc: float, replicates: int,
             rng: np.random.Generator, levels=LEVELS, methods=POWER_METHODS) -> dict:
    cur = simulate_source(d, r_cur, TYPE_WRONG_RECALL, f_cur, icc, replicates, rng)
    draws = [simulate_source(d, r_gen, TYPE_WRONG_RECALL, f_gen, icc, replicates, rng) for _ in range(3)]
    out = {}
    for m in methods:
        for a in levels:
            lo, hi = c3_interval(cur, draws, d, a, m)
            out[f"{m}@{a:.6g}"] = float(np.mean((lo > 0) | (hi < 0)))
    return out


def c4_power(d: BankDesign, delta: float, h: tuple[float, float], icc: float, replicates: int,
             rng: np.random.Generator, levels=LEVELS, methods=POWER_METHODS) -> dict | None:
    """C4's IUT against both sources at the same Delta: per comparison g_s = (Delta + 0.75 h_s) / 0.25."""
    g = [(delta + (1 - PREVALENCE) * hs) / PREVALENCE for hs in h]
    if max(g) > 1:
        return None
    unions = [UnionCounts(beta_binomial(d.wrong, gs, icc, replicates, rng), beta_binomial(d.gold, hs, icc, replicates, rng),
                          beta_binomial(d.rewrites, hs, icc, replicates, rng)) for gs, hs in zip(g, h)]
    out = {}
    for m in methods:
        for a in levels:
            ok = np.ones(replicates, dtype=bool)
            for u in unions:
                ok &= c4_interval(u, d, a, m)[0] > 0
            out[f"{m}@{a:.6g}"] = float(ok.mean())
    return out


# --- tables ------------------------------------------------------------------------------------


def _threshold(curve: Sequence[tuple[float, dict | None]], key: str, target: float) -> float | None:
    for effect, power in curve:
        if power is not None and power[key] >= target:
            return effect
    return None


def curves(d: BankDesign, icc: float, replicates: int, rng: np.random.Generator, levels=LEVELS,
           methods=POWER_METHODS) -> dict[str, Any]:
    """Power against effect size for each claim at central nuisance values, with the SESOI cell and the
    smallest effect on the grid reaching 50% and 80%:
    - C2 (i) and C3: the precision gap at pi = 25%, recall equal (RECALL + 0.05 for both), curated
      f = 0.02, the comparator's f solved from the gap;
    - C2 (ii): the type-wrong recall gap over census 0.5;
    - C4: Delta, with the central union-only f (C4_H)."""
    f_cur = CENTRAL_FPR[0]
    r = RECALL + 0.05
    p_cur = precision_at(r, f_cur)
    out: dict[str, Any] = {}
    for name, fn in (("C2 precision", c2_precision_power), ("C3", c3_power)):
        curve = []
        for gap in EFFECT_GRID:
            f_cmp = fpr_for_precision(r, p_cur - gap) if p_cur - gap > 0 else None
            curve.append((gap, None if f_cmp is None else fn(d, r, r, f_cur, f_cmp, icc, replicates, rng, levels,
                                                              methods)))
        out[name] = curve
    out["C2 recall"] = [(gap, c2_recall_power(d, TYPE_WRONG_RECALL + gap, TYPE_WRONG_RECALL, icc, replicates, rng,
                                              levels, methods) if TYPE_WRONG_RECALL + gap <= 1 else None)
                        for gap in EFFECT_GRID]
    out["C4"] = [(delta, c4_power(d, delta, C4_H["central"], icc, replicates, rng, levels, methods))
                 for delta in DELTA_GRID]
    summary = {}
    for name, curve in out.items():
        at = dict(curve).get(SESOI[name])
        summary[name] = {key: {"at_sesoi": None if at is None else at[key], "effect_50": _threshold(curve, key, 0.5),
                               "effect_80": _threshold(curve, key, 0.8)} for key in _keys(levels, methods)}
    return {"summary": summary, "curves": out, "curated_precision": p_cur, "recall": r, "curated_fpr": f_cur}


def grid(d: BankDesign, icc: float, replicates: int, rng: np.random.Generator, levels=LEVELS,
         methods=POWER_METHODS) -> dict[str, list]:
    """The ruling's grid: curated FPR x census (or generated) FPR x recall gap, with the precision each cell
    implies at pi = 25%; C2 (ii) by recall gap; C4 by Delta at both union-only f settings."""
    c2, c3 = [], []
    for f_cur in CURATED_FPR:
        for f_cen in CENSUS_FPR:
            for gap in RECALL_GAPS:
                r_cur, r_cen = RECALL + gap, RECALL
                cell = {"curated_fpr": f_cur, "other_fpr": f_cen, "recall_gap": gap, "curated_recall": r_cur,
                        "other_recall": r_cen, "curated_precision": precision_at(r_cur, f_cur),
                        "other_precision": precision_at(r_cen, f_cen)}
                c2.append({**cell, "power": c2_precision_power(d, r_cur, r_cen, f_cur, f_cen, icc, replicates, rng,
                                                               levels, methods)})
                c3.append({**cell, "power": c3_power(d, r_cur, r_cen, f_cur, f_cen, icc, replicates, rng, levels,
                                                     methods)})
    rec = [{"census": TYPE_WRONG_RECALL, "gap": gap,
            "power": c2_recall_power(d, TYPE_WRONG_RECALL + gap, TYPE_WRONG_RECALL, icc, replicates, rng, levels,
                                     methods)} for gap in RECALL_GAPS]
    c4 = [{"h": name, "h_vs_curated": h[0], "h_vs_census": h[1], "delta": delta,
           "power": c4_power(d, delta, h, icc, replicates, rng, levels, methods)}
          for name, h in C4_H.items() for delta in (0.0, 0.01, 0.02, 0.05)]
    return {"C2 precision": c2, "C2 recall": rec, "C3": c3, "C4": c4}


def power_by_design(base: BankDesign, by_class: Mapping[str, np.ndarray], cap: int, rewrites: np.ndarray, *,
                    icc: float = 0.2, icc_sensitivity: Sequence[float] = (0.5,), grid_replicates: int = 2000,
                    curve_replicates: int = 1000, seed: int = 20261028) -> dict[str, Any]:
    """Every design's structure, power curves with SESOI cells and thresholds, the grid, and the curves'
    summaries at the sensitivity ICCs."""
    out: dict[str, Any] = {"icc": icc, "icc_sensitivity": list(icc_sensitivity), "grid_replicates": grid_replicates,
                           "curve_replicates": curve_replicates, "seed": seed, "levels": list(LEVELS),
                           "methods": list(POWER_METHODS), "sesoi": SESOI, "prevalence": PREVALENCE,
                           "recall": RECALL, "type_wrong_recall": TYPE_WRONG_RECALL, "c4_h": C4_H, "designs": {}}
    for i, (name, d) in enumerate(designs(base, by_class, cap, rewrites, seed).items()):
        out["designs"][name] = design_power(d, icc, icc_sensitivity, grid_replicates, curve_replicates,
                                            np.random.default_rng([seed, 10 + i]))
    return out


def design_power(d: BankDesign, icc: float, icc_sensitivity: Sequence[float], grid_replicates: int,
                 curve_replicates: int, rng: np.random.Generator) -> dict[str, Any]:
    """One design's structure, power curves, grid and the primary bound's curves at the sensitivity ICCs."""
    entry = {"summary": design_summary(d), "curves": curves(d, icc, curve_replicates, rng),
             "grid": grid(d, icc, grid_replicates, rng)}
    entry["sensitivity"] = {f"{x:g}": curves(d, x, curve_replicates, rng, methods=("eb",))["summary"]
                            for x in icc_sensitivity}
    return entry
