"""Statistics of the pre-registration (paper/preregistration.md, draft 4 after change-log entry 40; "prereg" below).

- **Resampling** (prereg §4.2): a paired, clustered bootstrap of complete question vectors. The registered
  analysis resamples UNSTRATIFIED (``scheme="unstratified"``, prereg.resamples); the stratified scheme (template x
  tier) remains for the exploratory per-arm numbers. ``QuestionIndex`` fixes the order: strata sorted by label,
  questions sorted by id within a stratum. ``resample_matrix`` draws from one ``numpy.random.default_rng(seed)``
  how often each question is drawn in each resample. Every arm, model, repetition and version of a question is
  carried by its column, so one matrix serves every quantity: contrasts are paired.
- **Ratios** (§3 "Weighting"): a resample's value is ``(K @ num) / (K @ den)``; a resample with
  an empty denominator is undefined (NaN), never zero.
- **One-sided p** (§4): for H: theta > theta0, ``(1 + #{theta*_b <= theta0}) / (B + 1)``; ties
  and undefined resamples count as not supporting (§3 "Non-estimability"). For H: theta <
  theta0 the mirror; a draw within ``TIE_TOLERANCE`` of the boundary is a tie. Two-sided: ``min(1, 2 min(p_up, p_down))`` (§2, C1a and C3).
- **Intervals** (§4 "Percentile intervals", "Intervals and multiplicity"): percentile
  intervals over the defined resamples, with the undefined share reported; quantiles use
  numpy's ``inverted_cdf`` method (an order statistic of the draws, no interpolation).
- **Holm** (§4) and the rank-based local levels used for Holm-adjusted intervals.
- **Exact bounds** (§3): Clopper-Pearson, for binary question-level events only.
- **The fallback** (§4 "Calibration gate"): a paired sign-flip permutation test on
  base-question-level differences.
- **Constrained ratios** (change-log entry 29's first step): the sources' ratios re-estimated under a
  linear constraint sum_s w_s R_s = delta (``constrained_ratios``). No longer used by the registered
  analysis: the score test built on them was withdrawn (entry 35) and its module removed.
- **The tests of draft 4 §4.2** (change-log entry 40): ``studentise`` (a linear combination of ratios of
  per-question sums, its linearised standard error and both in every resample), ``bootstrap_t_p`` and
  ``bootstrap_t_bound`` (the paired, studentised cluster bootstrap-t and its percentile-t bounds), and
  ``cr2_p`` (the CR2-Satterthwaite fallback; ``t_sf`` from the regularised incomplete beta function).
  The sign-flip test above is no longer registered for any claim.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

QUANTILE_METHOD = "inverted_cdf"
# A draw within this distance of the boundary is a tie (prereg §4: ties do not support the
# alternative). Differences of count ratios over a few thousand outputs are at least ~1e-6
# apart, so the tolerance absorbs only floating-point rounding (0.53 - 0.50 is 0.0300...03).
TIE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class QuestionIndex:
    """The base questions of an analysis, in the fixed order of §4's resampling."""

    questions: tuple[str, ...]
    strata: tuple[str, ...]

    @classmethod
    def from_strata(cls, strata: Mapping[str, str]) -> QuestionIndex:
        ordered = sorted(strata.items(), key=lambda kv: (kv[1], kv[0]))
        return cls(tuple(q for q, _ in ordered), tuple(s for _, s in ordered))

    def __len__(self) -> int:
        return len(self.questions)

    def position(self, question: str) -> int:
        return self._positions[question]

    def __contains__(self, question: str) -> bool:
        return question in self._positions

    @property
    def _positions(self) -> dict[str, int]:
        cache = self.__dict__.get("_cache")
        if cache is None:
            cache = {q: i for i, q in enumerate(self.questions)}
            object.__setattr__(self, "_cache", cache)
        return cache


SCHEMES = ("stratified", "unstratified", "rescaled")


def resample_matrix(index: QuestionIndex, n_boot: int, *, seed: int, scheme: str = "stratified") -> np.ndarray:
    """(n_boot x questions) draw weights.

    ``stratified`` (the per-arm exploratory numbers; prereg §4.2 registers ``unstratified``): per stratum, in sorted order, n_h draws
    with replacement from its n_h questions, from one ``default_rng(seed)``. The two others exist
    only to diagnose it (calibration): ``unstratified`` draws all questions as one stratum;
    ``rescaled`` draws n_h - 1 per stratum and weights each draw n_h / (n_h - 1) (Rao and Wu's
    rescaling bootstrap, which removes the (n_h - 1) / n_h shrinkage of the variance that
    within-stratum resampling causes); a stratum of one question keeps its single draw."""
    if scheme not in SCHEMES:
        raise ValueError(f"unknown resampling scheme {scheme!r}")
    rng = np.random.default_rng(seed)
    out = np.zeros((n_boot, len(index)), dtype=np.float64)
    strata = {"all": list(range(len(index)))} if scheme == "unstratified" else {
        s: [i for i, t in enumerate(index.strata) if t == s] for s in sorted(set(index.strata))}
    for cols in strata.values():
        n = len(cols)
        m = n - 1 if scheme == "rescaled" and n > 1 else n
        out[:, cols] = rng.multinomial(m, np.full(n, 1.0 / n), size=n_boot) * (n / m)
    return out


def ratio_draws(counts: np.ndarray, num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """The ratio of weighted sums in each resample (prereg §3 "Weighting"); NaN where the denominator
    is zero (§3 "Non-estimability")."""
    n, d = counts @ num, counts @ den
    out = np.full(len(d), np.nan)
    ok = d > 0
    out[ok] = n[ok] / d[ok]
    return out


def p_greater(draws: np.ndarray, theta0: float = 0.0) -> float:
    """One-sided p for H: theta > theta0 (prereg §4): ties and undefined resamples do not support it."""
    against = int(np.sum(np.isnan(draws) | (np.nan_to_num(draws, nan=0.0) <= theta0 + TIE_TOLERANCE)))
    return (1 + against) / (len(draws) + 1)


def p_less(draws: np.ndarray, theta0: float = 0.0) -> float:
    """One-sided p for H: theta < theta0 (prereg §4), the mirror of ``p_greater``."""
    against = int(np.sum(np.isnan(draws) | (np.nan_to_num(draws, nan=0.0) >= theta0 - TIE_TOLERANCE)))
    return (1 + against) / (len(draws) + 1)


def p_two_sided(p_up: float, p_down: float) -> float:
    """min(1, 2 min(p_up, p_down)) (prereg §2, C1a and C3)."""
    return min(1.0, 2 * min(p_up, p_down))


def interval(draws: np.ndarray, level: float, sided: str = "two") -> dict[str, float | None]:
    """Percentile interval over the defined resamples (prereg §4). ``sided`` is "two",
    "greater" (a lower bound at 1 - level; ``hi`` None) or "less" (an upper bound; ``lo`` None).
    Both bounds are None when no resample is defined."""
    ok = draws[~np.isnan(draws)]
    share = float(np.isnan(draws).mean()) if len(draws) else 0.0
    if ok.size == 0:
        return {"lo": None, "hi": None, "level": level, "sided": sided, "undefined_share": share}
    alpha = round(1 - level, 12)  # 1 - 0.95 is 0.0500...04 in floating point; quantiles are order statistics
    if sided == "two":
        lo, hi = np.quantile(ok, [alpha / 2, 1 - alpha / 2], method=QUANTILE_METHOD)
    elif sided == "greater":
        lo, hi = np.quantile(ok, alpha, method=QUANTILE_METHOD), None
    elif sided == "less":
        lo, hi = None, np.quantile(ok, 1 - alpha, method=QUANTILE_METHOD)
    else:
        raise ValueError(f"unknown side {sided!r}")
    return {"lo": None if lo is None else float(lo), "hi": None if hi is None else float(hi), "level": level,
            "sided": sided, "undefined_share": share}


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-adjusted p-values, in the input order (prereg §4)."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def holm_levels(pvalues: Sequence[float], alpha: float) -> list[float]:
    """Each hypothesis's local Holm level (prereg §4 "Intervals and multiplicity"), alpha / (m - j + 1)
    at its rank j (1 = smallest p;
    ties broken by input order): the level its Holm-adjusted interval is computed at."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    levels = [0.0] * m
    for rank, i in enumerate(order):
        levels[i] = alpha / (m - rank)
    return levels


def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p), summing the pmf by its log-space recurrence."""
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if k >= n else 0.0
    log_pmf = n * math.log1p(-p)
    step = math.log(p) - math.log1p(-p)
    total = math.exp(log_pmf)
    for i in range(k):
        log_pmf += math.log(n - i) - math.log(i + 1) + step
        total += math.exp(log_pmf)
    return min(1.0, total)


def _bisect(f, target: float, increasing: bool) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(60):  # 2**-60: far below any reported digit
        mid = (lo + hi) / 2
        if (f(mid) < target) == increasing:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float] | None:
    """Exact binomial interval for k of n binary question-level events (prereg §3); None for n = 0."""
    if n == 0:
        return None
    alpha = 1 - level
    lo = 0.0 if k == 0 else _bisect(lambda p: 1 - _binom_cdf(k - 1, n, p), alpha / 2, increasing=True)
    hi = 1.0 if k == n else _bisect(lambda p: _binom_cdf(k, n, p), alpha / 2, increasing=False)
    return lo, hi


def sign_matrix(n_questions: int, n_perm: int, *, seed: int) -> np.ndarray:
    """(n_perm x questions) random signs for the sign-flip test (prereg §4's fallback), from a
    stream of its own (``default_rng([seed, 1])``) so it never shares draws with the resamples."""
    rng = np.random.default_rng([seed, 1])
    return rng.choice((-1.0, 1.0), size=(n_perm, n_questions))


def signflip_p(d: np.ndarray, signs: np.ndarray, alternative: str) -> float:
    """Paired sign-flip permutation test of mean(d) = 0 on base-question-level values d (prereg
    §4). T = mean(d); T*_b = mean(s_b d). One-sided p = (1 + #{T* at least as extreme as T}) /
    (B + 1), a tie (within ``TIE_TOLERANCE``) counting as at least as extreme."""
    t, t_star = float(d.mean()), signs @ d / len(d)
    if alternative == "greater":
        extreme = int(np.sum(t_star >= t - TIE_TOLERANCE))
    elif alternative == "less":
        extreme = int(np.sum(t_star <= t + TIE_TOLERANCE))
    else:
        raise ValueError(f"unknown alternative {alternative!r}")
    return (1 + extreme) / (len(t_star) + 1)


def signflip_bound(d: np.ndarray, signs: np.ndarray, alpha: float, side: str) -> float:
    """A confidence bound by inverting the sign-flip test (prereg §4 fallback; the inversion is our
    reading, listed in the report): the lower bound is the smallest
    theta0 at which the one-sided test of mean(d) > theta0 (on d - theta0) no longer rejects at
    alpha; the upper bound mirrors it. The p-value is monotone in theta0, so bisection finds it."""
    alternative = "greater" if side == "lower" else "less"
    spread = float(np.abs(d).max()) + 1.0
    lo, hi = float(d.mean()) - spread, float(d.mean()) + spread
    rejects = (lambda th: signflip_p(d - th, signs, alternative) <= alpha)
    # lower: rejects for theta0 well below the mean, not above it; upper: the mirror
    for _ in range(60):
        mid = (lo + hi) / 2
        if rejects(mid) == (side == "lower"):
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# --- constrained ratios (change-log entry 29's estimating step) ----------------------------------

LAMBDA_ITERATIONS = 200  # bisection steps on the multiplier
CONSTRAINT_TOLERANCE = 1e-10  # |sum_s w_s R_s - delta| at convergence, else ConvergenceError
LAMBDA_LIMIT = 1e12  # times the largest denominator: a constraint needing more lies at the boundary


class ConvergenceError(ValueError):
    """The constrained ratios could not be found to tolerance (delta at or outside the feasible range)."""


def _ratio_at(c: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Each source's maximiser on [0, 1] of the concave A log R + (B - A) log(1 - R) - c R (``c``
    broadcasts against A and B), by its Karush-Kuhn-Tucker conditions:
    - A = 0: R = 0 if c >= -B, else the interior root 1 + B / c (the rational root is 0/0 at c = -B);
    - A = B: R = 1 if c <= B, else B / c;
    - 0 < A < B: the root in (0, 1) of c R^2 - (B + c) R + A = 0, evaluated without cancellation:
      2A / ((B + c) + sqrt(D)) when B + c >= 0, else ((B + c) - sqrt(D)) / (2c), D = (B + c)^2 - 4cA."""
    c, A, B = np.broadcast_arrays(np.asarray(c, dtype=np.float64), np.asarray(A, dtype=np.float64),
                                  np.asarray(B, dtype=np.float64))
    b = B + c
    root_d = np.sqrt(np.maximum(b * b - 4 * c * A, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        interior = np.where(b >= 0, 2 * A / (b + root_d), (b - root_d) / (2 * c))
        zero_edge = np.where(c >= -B, 0.0, 1 + B / c)
        one_edge = np.where(c <= B, 1.0, B / c)
    out = np.where(A <= 0, zero_edge, np.where(A >= B, one_edge, interior))
    return np.clip(out, 0.0, 1.0)


def constrained_ratios(A: np.ndarray, B: np.ndarray, w: np.ndarray, delta: float | np.ndarray) -> np.ndarray:
    """The sources' ratios under sum_s w_s R_s = delta: the maximiser of the binomial quasi-likelihood
    sum_s A_s log R_s + (B_s - A_s) log(1 - R_s) under the constraint, whose Lagrange conditions are
    A_s - B_s R_s = lambda w_s R_s (1 - R_s) (Farrington-Manning / Miettinen-Nurminen, generalised to a
    linear contrast). sum_s w_s R_s(lambda w_s) is continuous and non-increasing in lambda, so lambda is
    bracketed by doubling (up to ``LAMBDA_LIMIT`` times the largest denominator) and found by bisection;
    the constraint is then checked to ``CONSTRAINT_TOLERANCE``. ``A``, ``B`` are the pooled counts per
    source (every B_s > 0); ``delta`` a number or an array (one row of ratios per delta) strictly inside
    (sum min(0, w), sum max(0, w)). Raises ConvergenceError otherwise."""
    A, B, w = (np.asarray(x, dtype=np.float64) for x in (A, B, w))
    if (B <= 0).any() or (A < 0).any() or (A > B).any():
        raise ValueError("constrained ratios need 0 <= A_s <= B_s and B_s > 0 for every source")
    deltas = np.atleast_1d(np.asarray(delta, dtype=np.float64))
    low_end, high_end = float(np.minimum(w, 0).sum()), float(np.maximum(w, 0).sum())
    if ((deltas <= low_end) | (deltas >= high_end)).any():
        raise ConvergenceError(f"delta must lie strictly inside ({low_end}, {high_end})")

    def theta(lam: np.ndarray) -> np.ndarray:
        return (_ratio_at(lam[:, None] * w, A, B) * w).sum(axis=1)

    bound, limit = np.ones_like(deltas), LAMBDA_LIMIT * float(B.max())
    while True:
        open_ = (theta(-bound) < deltas) | (theta(bound) > deltas)
        if not open_.any():
            break
        if (bound[open_] >= limit).any():
            raise ConvergenceError("delta is too close to the feasible range's endpoint to be bracketed")
        bound = np.where(open_, 2 * bound, bound)
    lo, hi = -bound, bound
    for _ in range(LAMBDA_ITERATIONS):
        mid = (lo + hi) / 2
        above = theta(mid) > deltas
        lo, hi = np.where(above, mid, lo), np.where(above, hi, mid)
    lam = (lo + hi) / 2
    if (np.abs(theta(lam) - deltas) > CONSTRAINT_TOLERANCE).any():
        raise ConvergenceError("the constraint was not met to tolerance")
    R = _ratio_at(lam[:, None] * w, A, B)
    return R[0] if np.ndim(delta) == 0 else R


# --- the studentised cluster bootstrap and the CR2 fallback (prereg §4.2, change-log entry 40) ---------
#
# Every confirmatory statistic is a linear combination of ratios of per-question sums,
# theta = sum_j c_j (sum_q a_jq / sum_q b_jq), which covers each case:
# - a per-question mean over a subset S (a = 1_S x, b = 1_S);
# - a difference of such means;
# - C5's per-model ratios sum F (Y_fb - Y_gen) / sum F, averaged over the panel.
#
# Its linearised per-question values are psi_q = sum_j c_j (a_jq - theta_j b_jq) / b_bar_j, which sum to 0.
# The standard error is sqrt(sum_q psi_q^2 / (n (n - 1))), the usual sd / sqrt(n) for a plain mean.
# In a resample with counts K_q, every theta_j and b_bar_j is recomputed, and
# sum_q K_q psi*_q^2 expands into K @ (products of the a's and b's). The bootstrap standard error then
# needs only matrix products, and no question is ever dropped from a resample.


PAIRWISE_MAX = 4  # distinct columns up to which studentise uses column-pair products (cheaper than a B x n array)


@dataclass(frozen=True)
class Term:
    """One ratio of per-question sums in a statistic: c * sum(a) / sum(b)."""

    c: float
    a: np.ndarray
    b: np.ndarray


@dataclass
class Studentised:
    """A statistic, its standard error, and the same in every resample (NaN where a denominator is empty)."""

    estimate: float | None
    se: float | None
    psi: np.ndarray | None
    star: np.ndarray
    se_star: np.ndarray


def _theta(terms: Sequence[Term]) -> list[float] | None:
    out = []
    for t in terms:
        den = float(t.b.sum())
        if den <= 0:
            return None
        out.append(float(t.a.sum()) / den)
    return out


def studentise(K: np.ndarray, terms: Sequence[Term]) -> Studentised:
    """The statistic sum_j c_j sum(a_j)/sum(b_j), its linearised standard error, and both in every resample of
    the question-resample matrix K (B x n). In resample b, psi*_bq = sum_j (c_j / b_bar*_j)(a_jq - theta*_j b_jq)
    is a linear combination of the distinct a and b columns with resample-specific coefficients, so
    Psi* = coef (B x m) @ Z' (m x n) and se*_b^2 = sum_q K_bq psi*_bq^2 / (n (n - 1))."""
    n = K.shape[1]
    thetas = _theta(terms)
    cols: list[np.ndarray] = []
    keys: dict[bytes, int] = {}

    def col(vec: np.ndarray) -> int:
        key = np.ascontiguousarray(vec, dtype=np.float64).tobytes()
        if key not in keys:
            keys[key] = len(cols)
            cols.append(np.asarray(vec, dtype=np.float64))
        return keys[key]

    pos = [(col(t.a), col(t.b)) for t in terms]
    Z = np.column_stack(cols)  # (n x m), the distinct columns
    S = K @ Z  # (B x m)
    with np.errstate(divide="ignore", invalid="ignore"):
        th = [S[:, ia] / S[:, ib] for ia, ib in pos]
        bbar = [S[:, ib] / n for _, ib in pos]
        star = sum(t.c * th[j] for j, t in enumerate(terms))
        coef = np.zeros((K.shape[0], len(cols)))
        for j, (t, (ia, ib)) in enumerate(zip(terms, pos)):
            coef[:, ia] += t.c / bbar[j]
            coef[:, ib] -= t.c * th[j] / bbar[j]
        m = len(cols)
        if m <= PAIRWISE_MAX:  # coef_b' (Z' diag(K_b) Z) coef_b from the products of column pairs: no (B x n) array
            pairs = [(i, j) for i in range(m) for j in range(i, m)]
            M = K @ np.column_stack([Z[:, i] * Z[:, j] for i, j in pairs])
            total = np.zeros(K.shape[0])
            for col_, (i, j) in enumerate(pairs):
                total += (1.0 if i == j else 2.0) * coef[:, i] * coef[:, j] * M[:, col_]
        else:
            psi_star = coef @ Z.T  # (B x n)
            total = np.einsum("bq,bq->b", K, psi_star * psi_star)
        se_star = np.sqrt(np.maximum(total, 0.0) / (n * (n - 1)))
    star = np.where(np.isfinite(star), star, np.nan)
    se_star = np.where(np.isfinite(se_star), se_star, np.nan)
    if thetas is None:
        return Studentised(None, None, None, star, se_star)
    psi = sum(t.c * (t.a - thetas[j] * t.b) / (float(t.b.sum()) / n) for j, t in enumerate(terms))
    se = math.sqrt(float((psi**2).sum()) / (n * (n - 1)))
    return Studentised(sum(t.c * thetas[j] for j, t in enumerate(terms)), se, psi, star, se_star)


def _t_star(s: Studentised) -> np.ndarray:
    """T*_b = (theta*_b - theta_hat) / se*_b, undefined (NaN) wherever se*_b is zero or not finite: a resample whose
    drawn questions all have the same value says nothing about the spread and never supports a claim (§4.2)."""
    ok = np.isfinite(s.se_star) & (s.se_star > 0) & np.isfinite(s.star)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ok, (s.star - s.estimate) / np.where(ok, s.se_star, 1.0), np.nan)


def bootstrap_t_p(s: Studentised, theta0: float, alternative: str) -> float:
    """One-sided bootstrap-t p-value for H1: theta > theta0 ("greater") or theta < theta0 ("less") (prereg §4.2):
    T = (theta_hat - theta0) / se, T*_b = (theta*_b - theta_hat) / se*_b; p = (1 + #{T* at least as extreme as
    T}) / (B + 1); an undefined T* is non-supporting. A non-estimable statistic gives 1; zero spread gives
    1 / (B + 1) when the estimate is strictly on the alternative's side and 1 otherwise."""
    b = len(s.star)
    if s.estimate is None:
        return 1.0
    diff = s.estimate - theta0
    if s.se == 0:
        on_side = diff > 0 if alternative == "greater" else diff < 0
        return 1 / (b + 1) if on_side else 1.0
    t = diff / s.se
    t_star = _t_star(s)
    if alternative == "greater":
        against = int(np.sum(~(t_star < t)))
    elif alternative == "less":
        against = int(np.sum(~(t_star > t)))
    else:
        raise ValueError(f"unknown alternative {alternative!r}")
    return (1 + against) / (b + 1)


def bootstrap_t_bound(s: Studentised, alpha: float, side: str) -> float | None:
    """The percentile-t bound at level 1 - alpha: theta_hat - q_{1-alpha}(T*) se (lower) or theta_hat -
    q_alpha(T*) se (upper); an undefined T* counts as the most extreme value against the bound, and a bound that is
    then unbounded is None."""
    if s.estimate is None:
        return None
    t_star = _t_star(s)
    if side == "lower":
        q = np.quantile(np.where(np.isnan(t_star), np.inf, t_star), 1 - alpha, method=QUANTILE_METHOD)
    else:
        q = np.quantile(np.where(np.isnan(t_star), -np.inf, t_star), alpha, method=QUANTILE_METHOD)
    return float(s.estimate - q * s.se) if np.isfinite(q) else None  # None: unbounded on that side


def _betacf(a: float, b: float, x: np.ndarray, steps: int = 300) -> np.ndarray:
    """The continued fraction of the incomplete beta function (modified Lentz), vectorised over x."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = np.ones_like(x)
    d = 1.0 - qab * x / qap
    d = 1.0 / np.where(np.abs(d) < tiny, tiny, d)
    h = d.copy()
    for m in range(1, steps + 1):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)), -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1.0 + aa * d
            d = 1.0 / np.where(np.abs(d) < tiny, tiny, d)
            c = 1.0 + aa / c
            c = np.where(np.abs(c) < tiny, tiny, c)
            h = h * d * c
    return h


def _betacf_scalar(a: float, b: float, x: float) -> float:
    """The continued fraction for one x, to convergence (Numerical Recipes 6.4)."""
    tiny, eps = 1e-300, 1e-16
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 10_000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc(a: float, b: float, x) -> np.ndarray:
    """The regularised incomplete beta function I_x(a, b) (Numerical Recipes 6.4), vectorised over x."""
    if np.ndim(x) == 0:
        xs = float(x)
        if xs <= 0:
            return np.float64(0.0)
        if xs >= 1:
            return np.float64(1.0)
        bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(xs) + b * math.log1p(-xs))
        if xs < (a + 1) / (a + b + 2):
            return np.float64(bt * _betacf_scalar(a, b, xs) / a)
        return np.float64(1.0 - bt * _betacf_scalar(b, a, 1 - xs) / b)
    x = np.asarray(x, dtype=np.float64)
    out = np.where(x <= 0, 0.0, np.where(x >= 1, 1.0, np.nan))
    inner = (x > 0) & (x < 1)
    if inner.any():
        xi = x[inner]
        bt = np.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * np.log(xi) + b * np.log1p(-xi))
        direct = xi < (a + 1) / (a + b + 2)
        out[inner] = np.where(direct, bt * _betacf(a, b, np.where(direct, xi, 0.5)) / a,
                              1.0 - bt * _betacf(b, a, np.where(direct, 0.5, 1 - xi)) / b)
    return out


def t_sf(t, df: float) -> np.ndarray:
    """P(T > t) for Student's t with df degrees of freedom: 0.5 I_{df/(df+t^2)}(df/2, 1/2) for t >= 0."""
    t = np.asarray(t, dtype=np.float64)
    tail = 0.5 * betainc(df / 2, 0.5, df / (df + t * t))
    return np.where(t >= 0, tail, 1.0 - tail)


def cr2_p(s: Studentised, theta0: float, alternative: str) -> float:
    """The fallback (prereg §4.2): a cluster-robust Wald test with CR2 and Satterthwaite degrees of freedom on
    the per-question values. With one value per question (psi_q) in an intercept-only model, every cluster's
    CR2 adjustment is (1 - 1/n)^-1/2, so V = sum psi^2 / (n (n - 1)), the studentised statistic's se, and the
    Bell-McCaffrey degrees of freedom are n - 1: the one-sample t test (for C5's ratios, on its linearised
    values, our reading)."""
    if s.estimate is None:
        return 1.0
    n = len(s.psi)
    diff = s.estimate - theta0
    if s.se == 0:
        on_side = diff > 0 if alternative == "greater" else diff < 0
        return 0.0 if on_side else 1.0
    t = diff / s.se
    return float(t_sf(t, n - 1)) if alternative == "greater" else float(t_sf(-t, n - 1))
