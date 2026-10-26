"""The bank contrasts C2, C3 and C4 by an empirical-Bernstein confidence-region projection.

This implements pre-registration §4.1 (draft 4, after change-log entries 38 and 39). It is
analytic and valid in finite samples. Simulation only checks the implementation and computes
power (bank_power).

**Assumptions.** Base questions are independent; they need not share a mean, and any dependence
within a question is allowed. Inference is about the generator's question population,
conditional on the frozen mutation draws and on the three generated artefacts.

**Estimands** (§4.1, entry 39). They are fixed independently of the collection counts.
- **Recall r_s:** caught over all answer-changing items (C2 ii: caught type-wrong over all
  type-wrong items), i.e. sum_q E count_q / sum_q N_q.
- **False-flag rate:** f_s = 0.5 f_gold + 0.5 f_rewrite, with the weights fixed now.
  - f_gold is the flag rate on the untouched gold query, exactly one per question.
  - f_rewrite is the flag rate on the k answer-preserving rewrites per question.
  - Answer-neutral mutations are their own reported stratum, outside f.
- **Precision at a fixed prevalence pi = 25%:** P_s = pi r_s / (pi r_s + (1 - pi) f_s).
- **C4:** Delta_s = pi r_s^u - (1 - pi) f_s^u, where r_s^u is the union-only recall over s
  (union flags, s does not) and f_s^u the union-only false-flag rate, on the same strata and
  weights.

**Coordinates.** Each random per-question count is divided by the design maximum M of its own
kind in the frozen manifest, never by an observed flag count:
- M_W = max W_q for A (flagged wrong) and G (union-only wrong);
- M_T = max T_q for K (caught type-wrong);
- M = 1 for the gold flag and the union-only gold flag, since there is exactly one gold per
  question;
- M_k = max k_q for the flagged rewrites and the union-only rewrites.

A rate is then mu M / N_bar, clipped to [0, 1], where N_bar is the mean of its fixed per-question
denominators. ``BankDesign.gold_only`` drops the rewrite stratum (f = f_gold). The power tables
use it for designs without rewrites; it is not the registered estimand.

**Bound per coordinate** (``METHODS``):
- **``eb`` (primary):** Maurer and Pontil's empirical-Bernstein bound for the average
  expectation of independent, heterogeneous variables in [0, 1] (Theorem 11 of
  arXiv:0907.3740). With probability at least 1 - delta,
  (1/n) sum E X_i <= X_bar + sqrt(2 V_n ln(2/delta) / n) + 7 ln(2/delta) / (3 (n - 1)),
  where V_n is the sample variance (ddof 1). The lower side is the same bound on 1 - X.
- **Reference only** (entries 38 and 39 withdrew them as the primary):
  - ``kl``, the KL-Hoeffding bound;
  - ``betting``, Waudby-Smith and Ramdas' betting interval, which needs a common conditional
    mean;
  - ``kl+eb`` (not ruled): both KL-Hoeffding and empirical Bernstein at delta/2, keeping the
    tighter end. It is valid by a union bound, for heterogeneous questions.

**Allocation.** The tail allocations are frozen. One-sided claims (C2, C4) use one side per
coordinate at delta = a / d. C3, which is two-sided, uses a / (2d). Here d is the number of
random coordinates in the claim (``d_coords``): 2 for C2 (ii), and with rewrites 6 for C2 (i),
12 for C3 and 6 for C4. **Projection:** monotone bounds on r and f give the bounds on P and
Delta. Differences take the opposite bounds; C3's comparator bound is the mean of the three
draws' bounds.

**p-values** (§4.1). p = inf{a : the claim holds}, by bisection on log a. Nesting in a is checked
on a ladder of levels before inverting (``check_nesting``). An empty set gives 1.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

PREVALENCE = 0.25  # §4.1: precision and C4 at a fixed prevalence pi = 25%
PREVALENCES = (0.10, 0.25, 0.50)  # §4.1: also reported at 10% and 50% (and at the bank's own)
F_WEIGHTS = (0.5, 0.5)  # §4.1: f = 0.5 f_gold + 0.5 f_rewrite, weights fixed now
METHODS = ("eb", "kl", "kl+eb", "betting")  # eb primary (§4.1); the others reference only (entries 38, 39)
ORDER_SEED = (20261026, 2)  # the question order the reference betting bound bets in
TRUNCATION = 0.5  # betting reference: lambda_i(m) = min(lambda_i, c_trunc / m)
TUNING_ALPHA = 0.05 / 6  # betting reference: the claim-level a its bets are tuned at
ALPHA_FLOOR = 1e-300  # the smallest a the inversion searches; a claim holding there reports this p
BISECTION_STEPS = 64
P_STEPS = 64
NESTING_LADDER = (1e-12, 1e-8, 1e-6, 1e-4, 1e-3, 0.05 / 6, 0.01, 0.05, 0.1, 0.5, 1.0)
BANK_FIELDS = ("id", "question_id", "split", "kind", "bank", "fault_class", "template", "tier", "operator",
               "draw_rank", "internal_node")  # source-free only (internal_node: ruling (c), design log)
GOLD, REWRITE, NEUTRAL = "control", "rewrite", "answer_neutral"  # the manifest's bank values; rewrites by kind "rewrite"


class NestingError(ValueError):
    """A claim's bound is not nested in a, so inverting it would not give a valid p-value."""


# --- the per-coordinate bounds (§4.1) -----------------------------------------------------------


def eb_side_bounds(values: np.ndarray, delta) -> tuple[np.ndarray, np.ndarray]:
    """Maurer and Pontil, Theorem 11 (arXiv:0907.3740): bounds on the average expectation of
    independent variables in [0, 1], each side at delta, clipped to [0, 1]. ``values`` is (..., n),
    n >= 2."""
    values = np.asarray(values, dtype=np.float64)
    n = values.shape[-1]
    if n < 2:
        raise ValueError("the empirical-Bernstein bound needs at least two questions")
    xbar, var = values.mean(axis=-1), values.var(axis=-1, ddof=1)
    log_term = np.log(2 / np.asarray(delta, dtype=np.float64))
    width = np.sqrt(2 * var * log_term / n) + 7 * log_term / (3 * (n - 1))
    return np.clip(xbar - width, 0.0, 1.0), np.clip(xbar + width, 0.0, 1.0)


def kl(x: float | np.ndarray, m: float | np.ndarray) -> float | np.ndarray:
    """Bernoulli KL divergence with 0 log 0 = 0 and +inf where m excludes the mass of x (reference)."""
    x, m = np.asarray(x, dtype=np.float64), np.asarray(m, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        left = np.where(x > 0, x * np.log(np.where(x > 0, x, 1.0) / m), 0.0)
        right = np.where(x < 1, (1 - x) * np.log(np.where(x < 1, 1 - x, 1.0) / (1 - m)), 0.0)
    out = np.where(((x > 0) & (m <= 0)) | ((x < 1) & (m >= 1)), np.inf, left + right)
    return float(out) if out.ndim == 0 else out


def kl_side_bounds(xbar, n: int, delta) -> tuple[np.ndarray, np.ndarray]:
    """KL-Hoeffding (reference): the ends of {m : n kl(x_bar, m) <= log(1/delta)}, outer approximations."""
    x = np.asarray(xbar, dtype=np.float64)
    t = np.log(1 / np.asarray(delta, dtype=np.float64)) / n
    x, t = np.broadcast_arrays(x, t)
    lo, hi = np.zeros_like(x), x.copy()
    for _ in range(BISECTION_STEPS):
        mid = (lo + hi) / 2
        out = kl(x, mid) > t
        lo, hi = np.where(out, mid, lo), np.where(out, hi, mid)
    lower = np.where(kl(x, 0.0) <= t, 0.0, lo)
    lo, hi = x.copy(), np.ones_like(x)
    for _ in range(BISECTION_STEPS):
        mid = (lo + hi) / 2
        out = kl(x, mid) > t
        lo, hi = np.where(out, lo, mid), np.where(out, mid, hi)
    upper = np.where(kl(x, 1.0) <= t, 1.0, hi)
    return lower, upper


def registered_order(n: int) -> np.ndarray:
    return np.random.default_rng(list(ORDER_SEED)).permutation(n)


def plugin_bets(x: np.ndarray, tuning_delta: float) -> np.ndarray:
    """Betting reference: predictable plug-in bets for a fixed n (Waudby-Smith and Ramdas 2024)."""
    n = x.shape[-1]
    t = np.arange(1, n + 1)
    mu_hat = (0.5 + np.cumsum(x, axis=-1)) / (t + 1)
    sigma2 = (0.25 + np.cumsum((x - mu_hat) ** 2, axis=-1)) / (t + 1)
    prev = np.concatenate([np.full(x.shape[:-1] + (1,), 0.25), sigma2[..., :-1]], axis=-1)
    return np.sqrt(2 * math.log(1 / tuning_delta) / (n * prev))


def _max_log_capital(x: np.ndarray, bets: np.ndarray, m: np.ndarray) -> np.ndarray:
    m_ = m[..., None]
    with np.errstate(divide="ignore"):
        cap = np.where(m_ > 0, TRUNCATION / np.where(m_ > 0, m_, 1.0), np.inf)
    return np.cumsum(np.log1p(np.minimum(bets, cap) * (x - m_)), axis=-1).max(axis=-1)


def betting_lower(x: np.ndarray, delta, tuning_delta: float) -> np.ndarray:
    """Betting reference: inf{m : max_t K_t(m) < 1/delta} with ``x`` in betting order."""
    x = np.asarray(x, dtype=np.float64)
    bets = plugin_bets(x, tuning_delta)
    threshold = np.broadcast_to(np.log(1 / np.asarray(delta, dtype=np.float64)), x.shape[:-1])
    lo, hi = np.zeros(x.shape[:-1]), np.ones(x.shape[:-1])
    excluded_at_zero = _max_log_capital(x, bets, lo) >= threshold
    for _ in range(BISECTION_STEPS):
        mid = (lo + hi) / 2
        out = _max_log_capital(x, bets, mid) >= threshold
        lo, hi = np.where(out, mid, lo), np.where(out, hi, mid)
    return np.where(excluded_at_zero, lo, 0.0)


def side_bounds(values: np.ndarray, delta, method: str = "eb", tuning_delta: float | None = None):
    """(lower, upper) bounds on the average expectation of per-question values in [0, 1] (``values``
    is (..., n) in index order), each side at delta, by ``method``."""
    values = np.asarray(values, dtype=np.float64)
    if method == "eb":
        return eb_side_bounds(values, delta)
    if method == "kl":
        return kl_side_bounds(values.mean(axis=-1), values.shape[-1], delta)
    if method == "kl+eb":  # not ruled: both bounds at delta / 2, the tighter side kept (a union bound)
        half = np.asarray(delta, dtype=np.float64) / 2
        k_lo, k_hi = kl_side_bounds(values.mean(axis=-1), values.shape[-1], half)
        e_lo, e_hi = eb_side_bounds(values, half)
        return np.maximum(k_lo, e_lo), np.minimum(k_hi, e_hi)
    if method == "betting":
        x = values[..., registered_order(values.shape[-1])]
        tune = TUNING_ALPHA if tuning_delta is None else tuning_delta
        return betting_lower(x, delta, tune), 1.0 - betting_lower(1.0 - x, delta, tune)
    raise ValueError(f"unknown bound {method!r}")


def scaled(counts: np.ndarray, m: float) -> np.ndarray:
    """count_q / M. A count outside [0, M] means M is not the design maximum, which would void every
    bound, so it is refused."""
    counts = np.asarray(counts, dtype=np.float64)
    if m <= 0:
        raise ValueError(f"design maximum {m} must be positive")
    if (counts < 0).any() or (counts > m).any():
        raise ValueError(f"counts must lie in [0, {m}] (the design maximum), got [{counts.min()}, {counts.max()}]")
    return counts / m


# --- the design (M from the frozen manifest) ---------------------------------------------------


@dataclass(frozen=True)
class BankDesign:
    """Fixed per-question item counts of the frozen bank, over the analysis's questions in index order
    (§4.1): W_q (answer-changing), T_q (type-wrong), gold_q (untouched gold: exactly 1), k_q (rewrites),
    and the answer-neutral mutations (their own stratum, outside f). ``gold_only`` makes f = f_gold (a
    design without rewrites; not the registered estimand)."""

    wrong: np.ndarray
    type_wrong: np.ndarray
    gold: np.ndarray
    rewrites: np.ndarray
    neutral: np.ndarray = field(default_factory=lambda: np.zeros(0))
    gold_only: bool = False

    def __post_init__(self) -> None:
        n = len(self.wrong)
        for name in ("type_wrong", "gold", "rewrites"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} has {len(getattr(self, name))} questions, not {n}")
        if not self.gold_only and float(self.rewrites.sum()) <= 0:
            raise ValueError("the registered f needs the rewrite stratum (§4.1); none is in the bank")

    @property
    def n(self) -> int:
        return len(self.wrong)

    def m(self, kind: str) -> int:
        """The design maximum of a coordinate's kind: "wrong", "type_wrong", "gold" or "rewrites"."""
        return int(getattr(self, kind).max())

    def mean(self, kind: str) -> float:
        return float(getattr(self, kind).mean())

    @property
    def f_strata(self) -> tuple[tuple[str, float], ...]:
        """(kind, weight) of the false-flag strata (§4.1)."""
        return (("gold", 1.0),) if self.gold_only else (("gold", F_WEIGHTS[0]), ("rewrites", F_WEIGHTS[1]))

    @property
    def coords_per_source(self) -> int:
        """Random coordinates of one source's precision: its flagged wrong items and each f stratum."""
        return 1 + len(self.f_strata)

    def d_coords(self, claim: str) -> int:
        """d, the random coordinates of a claim (§4.1): C2 (ii) has one per source."""
        per = {"C2 precision": 2 * self.coords_per_source, "C2 recall": 2, "C3": 4 * self.coords_per_source,
               "C4": 2 * self.coords_per_source}
        return per[claim]

    @classmethod
    def from_items(cls, items: Iterable[Mapping[str, Any]], split: str, exclude: Collection[str] = (),
                   gold_only: bool | None = None) -> tuple[BankDesign, dict[str, Any]]:
        """From manifest items (``load_bank_items``): the split's questions sorted by id; rejected
        mutations and ``exclude``d item ids are left out. ``gold_only`` defaults to whether the bank
        has no rewrites."""
        counts = {k: Counter() for k in ("wrong", "type_wrong", "gold", "rewrites", "neutral")}
        questions, excluded = set(), 0
        for i in items:
            if i["split"] != split:
                continue
            questions.add(i["question_id"])
            if i["id"] in exclude:
                excluded += 1
                continue
            q = i["question_id"]
            if i.get("kind") == "rewrite" or i["bank"] == REWRITE:  # bank v2: kind "rewrite", bank "control"
                counts["rewrites"][q] += 1
            elif i["bank"] == "answer_changing":
                counts["wrong"][q] += 1
                counts["type_wrong"][q] += i["fault_class"] == "type_wrong"
            elif i["bank"] == GOLD:
                counts["gold"][q] += 1
            elif i["bank"] == NEUTRAL:
                counts["neutral"][q] += 1
        order = sorted(questions)
        arrays = {k: np.array([float(c[q]) for q in order]) for k, c in counts.items()}
        if gold_only is None:
            gold_only = float(arrays["rewrites"].sum()) <= 0
        return cls(gold_only=gold_only, **arrays), {"questions": len(order), "excluded": excluded}


def load_bank_items(path: Path) -> list[dict[str, Any]]:
    """The bank's items.jsonl, keeping only source-free fields (``BANK_FIELDS``): never any source's
    conformance label (the study lead's guard, design log). Template and tier give the strata."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out.append({k: row.get(k) for k in BANK_FIELDS})
    return out


@dataclass
class SourceCounts:
    """One source's random per-question counts (§4.1), each (n,) or (replicates, n): flagged wrong
    items, caught type-wrong items, the gold flag and flagged rewrites."""

    wrong: np.ndarray
    type_wrong: np.ndarray
    gold: np.ndarray
    rewrites: np.ndarray


@dataclass
class UnionCounts:
    """The union-only counts over one source s (§4.1, C4): wrong items, gold and rewrites the union
    flags and s does not."""

    wrong: np.ndarray
    gold: np.ndarray
    rewrites: np.ndarray


# --- projections (§4.1) -----------------------------------------------------------------------


def rate_bounds(counts: np.ndarray, design: BankDesign, kind: str, delta, method: str = "eb"):
    """Bounds on sum_q E count_q / sum_q N_q = mu M / N_bar, clipped to [0, 1]."""
    m = design.m(kind)
    lo, hi = side_bounds(scaled(counts, m), delta, method)
    s = m / design.mean(kind)
    return np.clip(np.asarray(lo) * s, 0, 1), np.clip(np.asarray(hi) * s, 0, 1)


def ffr_bounds(counts: Mapping[str, np.ndarray], design: BankDesign, delta, method: str = "eb"):
    """f = sum_j w_j f_j over the false-flag strata (§4.1): the weighted sum of the strata's bounds."""
    lo = hi = 0.0
    for kind, w in design.f_strata:
        l_, h_ = rate_bounds(counts[kind], design, kind, delta, method)
        lo, hi = lo + w * l_, hi + w * h_
    return np.asarray(lo), np.asarray(hi)


def project_precision(r_lo, r_hi, f_lo, f_hi, prevalence: float = PREVALENCE):
    """P = pi r / (pi r + (1 - pi) f) is increasing in r and decreasing in f, so L = P(L_r, U_f) and
    U = P(U_r, L_f); undefined (0/0) -> 0 below and 1 above."""
    r_lo, r_hi, f_lo, f_hi = (np.asarray(v, dtype=np.float64) for v in (r_lo, r_hi, f_lo, f_hi))
    lo_den = prevalence * r_lo + (1 - prevalence) * f_hi
    hi_den = prevalence * r_hi + (1 - prevalence) * f_lo
    with np.errstate(divide="ignore", invalid="ignore"):
        lo = np.where(lo_den > 0, prevalence * r_lo / np.where(lo_den > 0, lo_den, 1.0), 0.0)
        hi = np.where(hi_den > 0, prevalence * r_hi / np.where(hi_den > 0, hi_den, 1.0), 1.0)
    return lo, hi


def precision_bounds(src: SourceCounts, design: BankDesign, delta, method: str = "eb",
                     prevalence: float = PREVALENCE):
    r = rate_bounds(src.wrong, design, "wrong", delta, method)
    f = ffr_bounds({"gold": src.gold, "rewrites": src.rewrites}, design, delta, method)
    return project_precision(*r, *f, prevalence)


def benefit_bounds(u: UnionCounts, design: BankDesign, delta, method: str = "eb", prevalence: float = PREVALENCE):
    """Delta_s = pi r^u - (1 - pi) f^u: L takes L for r^u and U for f^u (§4.1)."""
    r_lo, r_hi = rate_bounds(u.wrong, design, "wrong", delta, method)
    f_lo, f_hi = ffr_bounds({"gold": u.gold, "rewrites": u.rewrites}, design, delta, method)
    return prevalence * r_lo - (1 - prevalence) * f_hi, prevalence * r_hi - (1 - prevalence) * f_lo


# --- the claims (§4.1) -------------------------------------------------------------------------


def c2_precision_interval(cur: SourceCounts, cen: SourceCounts, design: BankDesign, a, method: str = "eb"):
    """C2 (i): [L_cur - U_cen, U_cur - L_cen], every side at a / d (d = 2 x coordinates per source)."""
    delta = np.asarray(a) / design.d_coords("C2 precision")
    lo_cur, hi_cur = precision_bounds(cur, design, delta, method)
    lo_cen, hi_cen = precision_bounds(cen, design, delta, method)
    return lo_cur - hi_cen, hi_cur - lo_cen


def c2_recall_interval(cur_k, cen_k, design: BankDesign, a, method: str = "eb"):
    """C2 (ii): type-wrong recall, curated minus census, every side at a / 2."""
    delta = np.asarray(a) / design.d_coords("C2 recall")
    lo_cur, hi_cur = rate_bounds(cur_k, design, "type_wrong", delta, method)
    lo_cen, hi_cen = rate_bounds(cen_k, design, "type_wrong", delta, method)
    return lo_cur - hi_cen, hi_cur - lo_cen


def c3_interval(cur: SourceCounts, draws: Sequence[SourceCounts], design: BankDesign, a, method: str = "eb"):
    """C3: [L_cur - mean_g U_g, U_cur - mean_g L_g], every side at a / (2d) (d = 4 sources x coordinates);
    the generated bound is the mean of the three draws' bounds."""
    delta = np.asarray(a) / (2 * design.d_coords("C3"))
    lo_cur, hi_cur = precision_bounds(cur, design, delta, method)
    bounds = [precision_bounds(g, design, delta, method) for g in draws]
    lo_gen = np.mean([lo for lo, _ in bounds], axis=0)
    hi_gen = np.mean([hi for _, hi in bounds], axis=0)
    return lo_cur - hi_gen, hi_cur - lo_gen


def c4_interval(u: UnionCounts, design: BankDesign, a, method: str = "eb"):
    """C4 against one source: bounds on Delta_s, every side at a / d (d covers both comparisons)."""
    return benefit_bounds(u, design, np.asarray(a) / design.d_coords("C4"), method)


# --- inversion (§4.1: nesting checked, then p = inf{a : the claim holds}) -------------------------


def check_nesting(bound: Callable[[float], float], ladder: Sequence[float] = NESTING_LADDER,
                  increasing: bool = True) -> None:
    """Before inverting (§4.1): the claim's bound is monotone in a on a ladder of levels (a lower bound
    that rises with a, or with ``increasing`` False an upper bound that falls). NestingError otherwise."""
    values = [float(bound(a)) for a in ladder]
    pairs = list(pairwise(values))
    bad = [(a, b) for a, b in pairs if (b < a - 1e-12 if increasing else b > a + 1e-12)]
    if bad:
        raise NestingError(f"the bound is not nested in a on the ladder {list(ladder)}: {values}")


def invert(holds: Callable[[np.ndarray], np.ndarray], shape: tuple[int, ...] = ()) -> float | np.ndarray:
    """p = inf{a in (0, 1] : holds(a)} for a claim nested in a, by bisection on log a over
    [log ALPHA_FLOOR, 0]; 1 when it fails at a = 1; ALPHA_FLOOR when it holds there. The returned p
    is the end where the claim holds (never below the exact infimum)."""
    at_one = np.asarray(holds(np.ones(shape)), dtype=bool)
    lo, hi = np.full(shape, math.log(ALPHA_FLOOR)), np.zeros(shape)
    at_floor = np.asarray(holds(np.exp(lo)), dtype=bool)
    for _ in range(P_STEPS):
        mid = (lo + hi) / 2
        h = np.asarray(holds(np.exp(mid)), dtype=bool)
        lo, hi = np.where(h, lo, mid), np.where(h, mid, hi)
    p = np.where(at_one, np.exp(hi), 1.0)
    p = np.where(at_floor, ALPHA_FLOOR, p)
    return float(p) if p.ndim == 0 else p
