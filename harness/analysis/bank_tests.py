"""Power of the clustered tests of the bank contrasts under pre-registration §4.1-§4.2 (change-log entry
40, the owner's decision of 18 October): per-question rates with equal weight per base question, the paired
unstratified studentised cluster bootstrap-t (primary) and the CR2-Satterthwaite Wald test (fallback).

**Estimands** (the request):
- **C2:** the IUT of (i) the false-flag-rate difference, census minus curated, > 0, and (ii) the
  type-wrong recall difference, curated minus census, > 0.
- **C3:** the false-flag-rate difference between curated and the mean of the three draws,
  two-sided.
- **C4:** §4.1's Delta_s for both s, as is.

f is §4.1's 0.5 f_gold + 0.5 f_rewrite in design (c), and f_gold alone in design (a), which has
no rewrites. **Per-question form** (§4.1). Every estimand is a mean over base questions, with
equal weight, of a per-question contrast u_q:
- C2(i): u_q = f_cen,q - f_cur,q with f_q = 0.5 g_q + 0.5 w_q / k_q when k_q >= 1, else g_q (entry 41),
  where g is the gold flag (0 or 1) and w the flagged rewrites of the k_q;
- C2(ii): u_q = (K_cur - K_cen)_q / T_q, over the questions with T_q > 0;
- C4: u_q = pi G_q / W_q - (1 - pi) f^u_q, f^u_q built from the union-only gold and rewrites as f_q.

**Tests**, both on the u_q of independent base questions:
- **(i) A paired cluster bootstrap, studentised (bootstrap-t).** Questions are resampled with
  replacement, unstratified, with one resample matrix shared by every dataset.
  - The statistic is T = u_bar / (s_u / sqrt(n)), and T* = (u_bar* - u_bar) / (s_u* / sqrt(n)).
  - One-sided p_up = (1 + #{T* >= T}) / (B + 1), p_down mirrored. Two-sided p = min(1, 2 min(p_up,
    p_down)) (§4.2).
  - An undefined T* (zero spread) does not support the alternative.
- **(ii) A CR2 cluster-robust Wald test with Satterthwaite degrees of freedom.** For an
  intercept-only model with one observation per cluster (the u_q), CR2's leverage adjustment is
  (1 - 1/n)^-1/2 for every cluster. That gives V = s_u^2 / n, and the Bell–McCaffrey degrees of
  freedom are exactly n - 1. The test is therefore the one-sample t test on u_q with n - 1 degrees
  of freedom (``stats.t_sf``).

``boot_t`` and ``t_test`` here are vectorised over simulated replicates. On one dataset they equal
``stats.studentise`` with ``stats.bootstrap_t_p`` and ``stats.cr2_p`` (the analysis's primitives), and
a test asserts it.

**Data-generating process** (assumed, as in bank_power).
- Every coordinate's per-question rate is Beta(mean, ICC), so items are correlated within a
  question, and the count is Binomial(N_q, rate).
- Sources are drawn independently. Positive dependence between sources would shrink var(u_q),
  so these powers are conservative in that respect.
- **Larger banks** (2 x 210 and 3 x 210 questions) are simulated by resampling the v1 test
  split's question structure (item counts by kind) with replacement within each template x tier
  stratum. A real larger bank from the frozen templates comes later.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from harness.analysis.bank_bounds import F_WEIGHTS, PREVALENCE, BankDesign
from harness.analysis.bank_power import beta_binomial
from harness.analysis.stats import t_sf

LEVELS = (0.05 / 6, 0.05)
TESTS = ("boot_t", "cr2")
SESOI = {"C2 ffr": 0.05, "C2 recall": 0.05, "C3": 0.05, "C4": 0.01}  # the existing 5-point and 0.01 SESOIs
CURATED_FPR = 0.02  # central curated false-flag rate for the curves
TW_RECALL = 0.5  # census type-wrong recall
RECALL = 0.6  # recall on all wrong items, for C4's union-only rates' context (not used by C2/C3)
C4_H = {"central": (0.098, 0.018), "low": (0.02, 0.005)}  # union-only f over (curated, census), as in bank_power
EFFECTS = tuple(round(0.01 * i, 2) for i in range(31))
DELTAS = tuple(round(0.005 * i, 3) for i in range(41))


# --- the tests -----------------------------------------------------------------------------------


def t_test(u: np.ndarray, sided: str) -> np.ndarray:
    """(ii) CR2 with one observation per cluster: the one-sample t test on u_q, n - 1 df (``u`` is
    (replicates, n)). One-sided for H1: mean > 0; two-sided otherwise. Zero spread with mean 0 gives 1."""
    n = u.shape[-1]
    mean, sd = u.mean(axis=-1), u.std(axis=-1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(sd > 0, mean / (sd / math.sqrt(n)), np.where(mean > 0, np.inf, np.where(mean < 0, -np.inf, 0.0)))
    return t_sf(t, n - 1) if sided == "greater" else np.minimum(1.0, 2 * t_sf(np.abs(t), n - 1))


def boot_matrix(n: int, b: int, seed: int) -> np.ndarray:
    """(B x n) resample counts: B unstratified multinomial draws of n questions."""
    return np.random.default_rng(seed).multinomial(n, np.full(n, 1.0 / n), size=b).astype(np.float64)


def boot_t(u: np.ndarray, K: np.ndarray, sided: str) -> np.ndarray:
    """(i) The studentised paired cluster bootstrap of u_bar (``u`` is (replicates, n), ``K`` (B, n)); two-sided
    is 2 min of the one-sided p's (§4.2)."""
    n, b = u.shape[-1], K.shape[0]
    mean, sd = u.mean(axis=-1), u.std(axis=-1, ddof=1)
    m_star = (K @ u.T) / n  # (B, replicates)
    sq_star = (K @ (u * u).T) / n
    sd_star = np.sqrt(np.maximum(sq_star - m_star**2, 0.0) * n / (n - 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_obs = np.where(sd > 0, mean / (sd / math.sqrt(n)), np.where(mean > 0, np.inf, 0.0))
        # a resample with no spread is undefined (NaN), never supporting (stats._t_star)
        t_star = np.where(sd_star > 0, (m_star - mean) / (sd_star / math.sqrt(n)), np.nan)
    p_up = (1 + np.sum(~(t_star < t_obs), axis=0)) / (b + 1)  # NaN counts against
    if sided == "greater":
        p = p_up
    else:
        p_down = (1 + np.sum(~(t_star > t_obs), axis=0)) / (b + 1)
        p = np.minimum(1.0, 2 * np.minimum(p_up, p_down))
    # no spread at all: every T* is undefined; a constant positive contrast is the strongest evidence there is
    supported = mean > 0 if sided == "greater" else mean != 0
    floor = 1 / (b + 1) if sided == "greater" else min(1.0, 2 / (b + 1))  # two-sided: 2 min, as stats.bootstrap_t_p
    return np.where(sd > 0, p, np.where(supported, floor, 1.0))


def test_p(u: np.ndarray, K: np.ndarray, sided: str) -> dict[str, np.ndarray]:
    return {"boot_t": boot_t(u, K, sided), "cr2": t_test(u, sided)}


# --- designs and larger banks -------------------------------------------------------------------


def enlarge(design: BankDesign, strata: Sequence[str], factor: int, seed: int) -> BankDesign:
    """A bank of factor x n questions: each stratum's question structures resampled with replacement to
    factor x its count (the v1 test split's structure; not a real larger bank)."""
    if factor == 1:
        return design
    rng = np.random.default_rng([seed, factor])
    idx = []
    for s in sorted(set(strata)):
        members = [i for i, t in enumerate(strata) if t == s]
        idx.extend(rng.choice(members, size=factor * len(members), replace=True).tolist())
    idx = np.array(idx)
    return replace(design, wrong=design.wrong[idx], type_wrong=design.type_wrong[idx], gold=design.gold[idx],
                   rewrites=design.rewrites[idx], neutral=design.neutral[idx] if len(design.neutral) else design.neutral)


def controls(d: BankDesign, f: float, icc: float, reps: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """One source's gold flag and flagged rewrites per question, sharing one per-question control rate ~ Beta(f, icc)
    (as simulate.generate does for the calibration)."""
    shape = (reps, d.n)
    if f <= 0 or f >= 1:
        p = np.full(shape, float(min(max(f, 0.0), 1.0)))
    else:
        k = 1 / icc - 1
        p = rng.beta(f * k, (1 - f) * k, size=shape)
    gold = rng.binomial(np.broadcast_to(d.gold, shape).astype(np.int64), p).astype(np.float64)
    rw = rng.binomial(np.broadcast_to(d.rewrites, shape).astype(np.int64), p).astype(np.float64)
    return gold, rw


def _ffr_u(d: BankDesign, f_a: float, f_b: float, icc: float, reps: int, rng) -> np.ndarray:
    """u_q of f_b - f_a (f = 0.5 f_gold + 0.5 f_rewrite, or f_gold when the design has no rewrites)."""
    gb, wb = controls(d, f_b, icc, reps, rng)
    ga, wa = controls(d, f_a, icc, reps, rng)
    if d.gold_only:
        return gb - ga
    return f_q(gb, wb, d.rewrites) - f_q(ga, wa, d.rewrites)


def f_q(gold: np.ndarray, rewrites: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Each question's false-flag rate (change-log entry 41): 0.5 f_gold + 0.5 f_rewrite when k_q >= 1, else f_gold."""
    share = np.where(k > 0, rewrites / np.where(k > 0, k, 1.0), 0.0)
    return np.where(k > 0, F_WEIGHTS[0] * gold + F_WEIGHTS[1] * share, gold)


def c2_ffr_u(d, f_cur, f_cen, icc, reps, rng):
    return _ffr_u(d, f_cur, f_cen, icc, reps, rng)


def c2_recall_u(d, r_cur, r_cen, icc, reps, rng):
    keep = d.type_wrong > 0  # questions with no type-wrong item are outside the mean (§4.1)
    t = d.type_wrong[keep]
    return (beta_binomial(t, r_cur, icc, reps, rng) - beta_binomial(t, r_cen, icc, reps, rng)) / t


def c3_u(d, f_cur, f_gen, icc, reps, rng):
    """u_q of f_cur - mean over the three draws of f_draw (two-sided)."""
    draws = [_ffr_u(d, 0.0, f_gen, icc, reps, rng) for _ in range(3)]  # each: f_draw - 0
    cur = _ffr_u(d, 0.0, f_cur, icc, reps, rng)
    return cur - np.mean(draws, axis=0)


def c4_u(d, g: float, h: float, icc, reps, rng):
    if (d.wrong <= 0).any():
        raise ValueError("C4's per-question form needs every question to hold a wrong item")
    gq = beta_binomial(d.wrong, g, icc, reps, rng) / d.wrong
    hg, hw = controls(d, h, icc, reps, rng)
    hf = hg if d.gold_only else f_q(hg, hw, d.rewrites)
    return PREVALENCE * gq - (1 - PREVALENCE) * hf


# --- power ---------------------------------------------------------------------------------------


def _rates(ps: Mapping[str, np.ndarray], levels: Sequence[float], calibrated: Mapping | None = None) -> dict[str, float]:
    """Rejection rates at each nominal level and, with ``calibrated`` ({test: {level: alpha'}}), at the
    empirically size-calibrated level (keys "<test>@<a>@cal")."""
    out = {f"{t}@{a:.6g}": float(np.mean(ps[t] <= a)) for t in TESTS for a in levels}
    if calibrated:
        out.update({f"{t}@{a:.6g}@cal": float(np.mean(ps[t] <= calibrated[t][f"{a:.6g}"])) for t in TESTS for a in levels})
    return out


def _iut(*parts: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {t: np.max([p[t] for p in parts], axis=0) for t in TESTS}


GRID_STEPS = 2000  # alpha' on the grid j * 0.0005 a, j = 1..2000 (capped at a, our reading)


def calibrate(null_p: np.ndarray, a: float) -> float:
    """The largest alpha' = j * 0.0005 a (j <= GRID_STEPS) whose rejection rate on the null p-values is <= a;
    0 when none (the empirical size calibration of §4.2's step 1, on this power run's null datasets)."""
    best = 0.0
    for j in range(1, GRID_STEPS + 1):
        x = j * 0.0005 * a
        if np.mean(null_p <= x) <= a:
            best = x
    return best


def claim_p(d: BankDesign, claim: str, effect: float, icc: float, reps: int, rng, K: np.ndarray,
            partner: float | None = None) -> dict[str, np.ndarray] | None:
    """Both tests' p-values for one claim at one effect. C2 is the IUT with its other component at
    ``partner`` (default: the same effect); "C2 ffr" and "C2 recall" are the components alone."""
    if claim == "C2 ffr":
        return test_p(c2_ffr_u(d, CURATED_FPR, CURATED_FPR + effect, icc, reps, rng), K, "greater")
    if claim == "C2 recall":
        if TW_RECALL + effect > 1:
            return None
        return test_p(c2_recall_u(d, TW_RECALL + effect, TW_RECALL, icc, reps, rng), K, "greater")
    if claim == "C2":
        other = effect if partner is None else partner
        a = test_p(c2_ffr_u(d, CURATED_FPR, CURATED_FPR + effect, icc, reps, rng), K, "greater")
        b = test_p(c2_recall_u(d, TW_RECALL + other, TW_RECALL, icc, reps, rng), K, "greater")
        return {"parts": (a, b), **_iut(a, b)}
    if claim == "C3":
        return test_p(c3_u(d, CURATED_FPR, CURATED_FPR + effect, icc, reps, rng), K, "two")
    if claim == "C4":
        h = C4_H["central"]
        g = [(effect + (1 - PREVALENCE) * hs) / PREVALENCE for hs in h]
        if max(g) > 1:
            return None
        return _iut(*[test_p(c4_u(d, gs, hs, icc, reps, rng), K, "greater") for gs, hs in zip(g, h)])
    raise ValueError(claim)


def claim_power(d: BankDesign, claim: str, effect: float, icc: float, reps: int, rng, K: np.ndarray,
                levels: Sequence[float] = LEVELS, calibrated: Mapping | None = None) -> dict[str, float] | None:
    ps = claim_p(d, claim, effect, icc, reps, rng, K)
    if ps is None:
        return None
    out = _rates(ps, levels)
    if calibrated:
        if claim == "C2":  # each component at its own calibrated level
            a, b = ps["parts"]
            out.update({f"{t}@{lv:.6g}@cal": float(np.mean((a[t] <= calibrated["C2 ffr"][t][f"{lv:.6g}"])
                                                           & (b[t] <= calibrated["C2 recall"][t][f"{lv:.6g}"])))
                        for t in TESTS for lv in levels})
        else:
            out.update({f"{t}@{lv:.6g}@cal": float(np.mean(ps[t] <= calibrated[claim][t][f"{lv:.6g}"]))
                        for t in TESTS for lv in levels})
    return out


def _threshold(curve, key: str, target: float) -> float | None:
    for e, p in curve:
        if p is not None and p[key] >= target:
            return e
    return None


def null_p(d: BankDesign, icc: float, datasets: int, rng, K: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """Null p-values at each claim's least-favourable boundary as the curves parameterise it: C2 (i) and C3 at
    equal false-flag rates CURATED_FPR, C2 (ii) at equal recall, C4 at Delta = 0 in the comparison whose test is
    larger (both comparisons simulated; the IUT's size is at most the larger)."""
    out = {"C2 ffr": test_p(c2_ffr_u(d, CURATED_FPR, CURATED_FPR, icc, datasets, rng), K, "greater"),
           "C2 recall": test_p(c2_recall_u(d, TW_RECALL, TW_RECALL, icc, datasets, rng), K, "greater"),
           "C3": test_p(c3_u(d, CURATED_FPR, CURATED_FPR, icc, datasets, rng), K, "two")}
    c4 = [test_p(c4_u(d, 3 * hs, hs, icc, datasets, rng), K, "greater") for hs in C4_H["central"]]
    out["C4"] = {t: np.min([c[t] for c in c4], axis=0) for t in TESTS}  # the more rejecting comparison, per dataset
    return out


def curves(d: BankDesign, icc: float, reps: int, rng, K: np.ndarray, levels: Sequence[float] = LEVELS,
           calibrated: Mapping | None = None) -> dict:
    """Power curves per claim; the SESOI cell and the smallest effect reaching 50% and 80% (at nominal levels
    and, with ``calibrated``, at the empirically size-calibrated levels)."""
    out, summary = {}, {}
    sesoi = {"C2": SESOI["C2 ffr"], **SESOI}
    keys = [f"{t}@{a:.6g}" for t in TESTS for a in levels]
    if calibrated:
        keys += [f"{k}@cal" for k in keys]
    for claim in ("C2 ffr", "C2 recall", "C2", "C3", "C4"):
        grid_ = DELTAS if claim == "C4" else EFFECTS
        curve = [(e, claim_power(d, claim, e, icc, reps, rng, K, levels, calibrated)) for e in grid_]
        out[claim] = curve
        at = dict(curve).get(sesoi[claim])
        summary[claim] = {key: {"at_sesoi": None if at is None else at[key], "effect_50": _threshold(curve, key, 0.5),
                                "effect_80": _threshold(curve, key, 0.8)} for key in keys}
    return {"summary": summary, "curves": out}


def type_one(d: BankDesign, icc: float, datasets: int, rng, K: np.ndarray, levels: Sequence[float] = LEVELS,
             fprs: Sequence[float] = (0.005, 0.01, 0.02, 0.10)) -> dict[str, Any]:
    """Each test's rejection rate at the null boundary of each component (the IUTs' least-favourable size):
    C2 (i) at equal false-flag rates, C2 (ii) at equal recall, C3 at equal rates (two-sided), and one C4
    comparison at Delta = 0 (g = 3h), at several false-flag rates down to 0.005."""
    out = {}
    for f in fprs:
        out[f"C2 ffr (f = {f})"] = _rates(test_p(c2_ffr_u(d, f, f, icc, datasets, rng), K, "greater"), levels)
        out[f"C3 (f = {f})"] = _rates(test_p(c3_u(d, f, f, icc, datasets, rng), K, "two"), levels)
        out[f"C4 one comparison (h = {f})"] = _rates(test_p(c4_u(d, 3 * f, f, icc, datasets, rng), K, "greater"),
                                                     levels)
    out[f"C2 recall (r = {TW_RECALL})"] = _rates(test_p(c2_recall_u(d, TW_RECALL, TW_RECALL, icc, datasets, rng), K,
                                                        "greater"), levels)
    return out


def grid(d: BankDesign, icc: float, reps: int, rng, K: np.ndarray, levels: Sequence[float] = LEVELS) -> list[dict]:
    """The ruling's grid for C2 (the IUT) and C3: curated FPR x census (or draws') FPR x type-wrong recall gap."""
    rows = []
    for f_cur in (0.0, 0.02, 0.05):
        for f_cen in (0.05, 0.10, 0.20):
            for gap in (0.0, 0.05, 0.15):
                a = test_p(c2_ffr_u(d, f_cur, f_cen, icc, reps, rng), K, "greater")
                b = test_p(c2_recall_u(d, TW_RECALL + gap, TW_RECALL, icc, reps, rng), K, "greater")
                c3 = test_p(c3_u(d, f_cur, f_cen, icc, reps, rng), K, "two")
                rows.append({"curated_fpr": f_cur, "other_fpr": f_cen, "recall_gap": gap,
                             "C2": _rates(_iut(a, b), levels), "C3": _rates(c3, levels)})
    return rows


# --- sizing the bank-only extension (change-log entry 41) ----------------------------------------
#
# The size-calibrated alpha'(a) of a claim is the largest grid value whose rejection rate is <= a in every one of the
# claim's least-favourable null configurations (prereg §4.2's step 1, here on this simulation's nulls), each config
# putting one component at its boundary and the other strongly true. Power is then the rejection rate at alpha'.

LF_ICC = (0.2, 0.5)


def lf_null_p(d: BankDesign, claim: str, reps: int, rng, K: np.ndarray) -> list[dict[str, np.ndarray]]:
    """The claim-level p-values (both tests) in each least-favourable null configuration of ``claim``."""
    out = []
    for icc in LF_ICC:
        if claim in ("C2", "C2 ffr"):
            for f in (0.005, 0.02, 0.10):  # (i) at its boundary, (ii) strongly true
                a = test_p(c2_ffr_u(d, f, f, icc, reps, rng), K, "greater")
                if claim == "C2 ffr":
                    out.append(a)
                    continue
                b = test_p(c2_recall_u(d, 0.8, 0.5, icc, reps, rng), K, "greater")
                out.append(_iut(a, b))
        if claim in ("C2", "C2 recall"):
            for r in (0.05, 0.5, 0.95):  # (ii) at its boundary, (i) strongly true
                b = test_p(c2_recall_u(d, r, r, icc, reps, rng), K, "greater")
                if claim == "C2 recall":
                    out.append(b)
                    continue
                a = test_p(c2_ffr_u(d, 0.02, 0.20, icc, reps, rng), K, "greater")
                out.append(_iut(a, b))
        if claim == "C3":
            for f in (0.005, 0.02, 0.10):
                out.append(test_p(c3_u(d, f, f, icc, reps, rng), K, "two"))
        if claim == "C4":
            for h in (0.005, 0.02, 0.05):  # one comparison at Delta = 0 (g = 3h), the other strongly positive
                a = test_p(c4_u(d, 3 * h, h, icc, reps, rng), K, "greater")
                b = test_p(c4_u(d, 0.6, 0.005, icc, reps, rng), K, "greater")
                out.append(_iut(a, b))
    return out


def lf_alpha(nulls: Sequence[Mapping[str, np.ndarray]], levels: Sequence[float] = LEVELS) -> dict[str, dict[str, float]]:
    """alpha'(a) per test: the smallest over configurations of the largest grid value with size <= a."""
    return {t: {f"{a:.6g}": min(calibrate(p[t], a) for p in nulls) for a in levels} for t in TESTS}


def sizing_power(d: BankDesign, claim: str, effect: float, icc: float, reps: int, rng, K: np.ndarray,
                 alpha: Mapping[str, Mapping[str, float]], levels: Sequence[float] = LEVELS) -> dict[str, float] | None:
    """Power of a claim (C2 the IUT) at one effect, at nominal levels and at the calibrated alpha'."""
    ps = claim_p(d, claim, effect, icc, reps, rng, K)
    if ps is None:
        return None
    out = _rates(ps, levels)
    out.update({f"{t}@{a:.6g}@cal": float(np.mean(ps[t] <= alpha[t][f"{a:.6g}"])) for t in TESTS for a in levels})
    return out
