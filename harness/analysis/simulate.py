"""Simulated datasets with known effects, for the certification of prereg §4.2 and the C5 power.

**Data-generating process.** These are assumptions of the simulation, not of the analysis.

Per model m of the panel P (prereg §2 C5) and of C1's models:
- **Accuracy.** Each base question draws its accuracy p_q from a Beta with mean ``accuracy`` and
  intra-question correlation ``icc``. This is a beta-binomial (ICC = 1 / (a + b + 1)), so the
  repetitions of a question are correlated. With probability ``cross_model`` a question's draw
  is shared by every model, which gives dependence across models; each model's marginal stays the
  same Beta.
- **Cohorts.** The census-prompt cohort and the natural cohort draw from p_q. The curated-prompt
  cohort draws from p_q shifted to the model's known accuracy difference ``c1``.
- **Flags.** On the natural outputs, curated checks flag a wrong output with P(F | W) and a correct
  one with P(F | C), set by the model's flag rate (``panel_flag_rates``) and the flags' precision.
  For the two families, census checks do the same, and the union flags what either flags.
- **Retries.** Each checker arm retries its flags. A wrong output is fixed with the arm's fix rate;
  a correct one is corrupted with ``corrupt``.
- **The generic control** retries the curated-flagged outputs. Its success probabilities are
  shifted from the curated retry's by the model's C5 effect, minus ``c5_spread`` times a
  question-level sign shared by every model and repetition. That makes the effect vary across
  questions while keeping its mean, so it adds dependence across models and repetitions.
- **Retry coupling.** With probability ``retry_coupling`` per output, every arm's retry uses one
  shared uniform: the arms become comonotone, with each marginal unchanged. Otherwise each arm draws
  its own.
- **Missing data.** Attempts are missing with probability ``missing`` and retries with
  ``retry_missing``.

**The fault bank** (§4.1) holds a fixed structure per question (``BankStructure``; in calibration the
real provisional bank's source-free counts):
- type-wrong and other answer-changing items;
- answer-neutral mutations;
- exactly one untouched gold;
- k rewrites.

Per question, each source draws a rate for each kind (type-wrong, other wrong, controls) from a Beta
with the source's mean and the bank's intra-question correlation. Its items are then flagged at that
rate, with sources independent: heterogeneous per-question rates and within-question correlation
(§4.2's configurations).

**Truth** is analytic (``truths``). Every registered estimand is a closed form of the parameters and
the structure, never computed by the pipeline. This includes C4's union-only shares,
E[p_other (1 - p_s)] = mu_other (1 - mu_s).

**Two paths, one generator.** ``dataset_vectors`` aggregates the draws into question vectors (the
array path, used by certification). ``dataset_objects`` turns them into attempts and ledger rows
that go through the files and loaders (the object path). The parity test asserts that both give
identical vectors and p-values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from harness.analysis.inputs import BANK, Attempt, CheckedOutput
from harness.analysis.prereg import PreregConfig, Vectors
from harness.analysis.stats import QuestionIndex

KINDS = ("type_wrong", "vocabulary_absent", "answer_neutral", "untouched", "rewrite")  # vocabulary_absent: the other wrong items
CHANGING = ("type_wrong", "vocabulary_absent")
CONTROLS = ("answer_neutral", "untouched", "rewrite")
LEDGER_ITEM = {"answer_neutral": "answer_neutral", "untouched": "untouched", "rewrite": "rewrite"}


@dataclass(frozen=True)
class BankStructure:
    """Fixed items per question of the simulated bank (prereg §4.1), aligned to the analysis's question index:
    type-wrong and other answer-changing items, answer-neutral mutations and rewrites; exactly one untouched gold."""

    type_wrong: tuple[int, ...]
    other_wrong: tuple[int, ...]
    neutral: tuple[int, ...]
    rewrites: tuple[int, ...]

    @classmethod
    def uniform(cls, n: int, type_wrong: int = 2, other_wrong: int = 2, neutral: int = 1, rewrites: int = 2) -> BankStructure:
        return cls((type_wrong,) * n, (other_wrong,) * n, (neutral,) * n, (rewrites,) * n)

    def counts(self, kind: str) -> np.ndarray:
        if kind == "untouched":
            return np.ones(len(self.type_wrong), dtype=np.int64)
        field_ = {"type_wrong": "type_wrong", "vocabulary_absent": "other_wrong", "answer_neutral": "neutral",
                  "rewrite": "rewrites"}[kind]
        return np.asarray(getattr(self, field_), dtype=np.int64)


def structure_from_bank(design_strata: list[str], wrong: np.ndarray, type_wrong: np.ndarray, neutral: np.ndarray,
                        index: QuestionIndex, rewrites: int | np.ndarray) -> BankStructure:
    """A real bank's per-question structure (source-free counts) assigned to the analysis's questions stratum by
    stratum, in sorted order, cycling within a stratum if the bank has fewer questions there (for certifying the
    procedure on the bank's real heterogeneity of item counts)."""
    by_stratum: dict[str, list[int]] = {}
    for i, s in enumerate(design_strata):
        by_stratum.setdefault(s, []).append(i)
    tw, ow, ne, rw = [], [], [], []
    k = np.full(len(design_strata), rewrites) if np.ndim(rewrites) == 0 else np.asarray(rewrites)
    seen: dict[str, int] = {}
    for s in index.strata:
        members = by_stratum.get(s) or [i for v in by_stratum.values() for i in v]
        j = members[seen.get(s, 0) % len(members)]
        seen[s] = seen.get(s, 0) + 1
        tw.append(int(type_wrong[j]))
        ow.append(int(wrong[j] - type_wrong[j]))
        ne.append(int(neutral[j]))
        rw.append(int(k[j]))
    return BankStructure(tuple(tw), tuple(ow), tuple(ne), tuple(rw))


@dataclass(frozen=True)
class SourceRates:
    """One bank source's rates (prereg §4.1): recall on type-wrong and on the other answer-changing items, and the
    false-flag rate on every control (gold, rewrites, answer-neutral)."""

    type_wrong: float = 0.5
    other_wrong: float = 0.6
    fpr: float = 0.05


@dataclass(frozen=True)
class Bank:
    """The simulated fault bank: its structure (None: 2 type-wrong, 2 other wrong, 1 neutral, 2 rewrites per
    question), each source's rates, and the intra-question correlation of every source's flags (per-question rates
    Beta(mean, icc), sources independent)."""

    structure: BankStructure | None = None
    curated: SourceRates = field(default_factory=SourceRates)
    census: SourceRates = field(default_factory=SourceRates)
    masked: tuple[SourceRates, SourceRates, SourceRates] = field(
        default_factory=lambda: (SourceRates(), SourceRates(), SourceRates()))
    icc: float = 0.2


@dataclass(frozen=True)
class Scenario:
    """One simulated configuration (prereg §4.2): accuracy, intra-question correlation, flag rates, retry
    rates, the true effect of every model-based contrast, panel heterogeneity and dependence, the bank,
    and missing data."""

    name: str
    accuracy: float = 0.65
    icc: float = 0.3
    flag_rate: tuple[float, float] = (0.10, 0.10)  # curated and census flags among natural outputs (two families)
    flag_precision: float = 0.6  # share of flags on wrong outputs
    curated_fix: float = 0.4
    corrupt: float = 0.1
    c1: tuple[float, ...] = (0.0, 0.0)  # per C1 model (cfg.c1a_models): curated-prompt - census-prompt accuracy
    c4: tuple[float, float] = (0.0, 0.0)  # the descriptive retry contrast: union - curated, union - census
    c5: tuple[float, ...] = (0.0, 0.0)  # per panel model (cfg.c5_models)
    panel_flag_rates: tuple[float, ...] = ()  # curated flag rate per panel model; empty = flag_rate[0] for all
    cross_model: float = 0.0
    retry_coupling: float = 0.0
    c5_spread: float = 0.0
    bank: Bank = field(default_factory=Bank)
    missing: float = 0.0
    retry_missing: float = 0.0
    n_reps: int = 3


# --- parameters and the analytic truth ------------------------------------------------------


def _flag_probs(rate: float, precision: float, mu: float) -> tuple[float, float]:
    """(P(F | W), P(F | C)) giving the flag rate and the flags' precision."""
    pw, pc = rate * precision / (1 - mu), rate * (1 - precision) / mu
    if not (0 <= pw <= 1 and 0 <= pc <= 1):
        raise ValueError(f"flag rate {rate} at precision {precision} is impossible at accuracy {mu}")
    return pw, pc


def _net(mu: float, pw: float, pc: float, fix: float, corrupt: float) -> float:
    return (1 - mu) * pw * fix - mu * pc * corrupt


def _fix_for(target: float, mu: float, pw: float, pc: float, corrupt: float) -> float:
    fix = (target + mu * pc * corrupt) / ((1 - mu) * pw)
    if not 0 <= fix <= 1:
        raise ValueError(f"net change {target} needs a fix rate of {fix:.3f}")
    return fix


@dataclass(frozen=True)
class Params:
    """Every probability of the generator, derived from a scenario and checked feasible."""

    beta: tuple[float, float]
    flags: dict[str, tuple[float, float]]  # panel model -> curated (P(F|W), P(F|C))
    family_flags: dict[str, tuple[float, float]]  # census and union (P(F|W), P(F|C)) for the families
    fix: dict[str, float]  # source -> fix rate (curated, census, union)
    generic: dict[str, tuple[float, float]]  # panel model -> (P(correct | W), P(correct | C)) after the generic retry
    c1: dict[str, float]
    c5: dict[str, float]
    bank_rates: dict[str, SourceRates]


def params(sc: Scenario, cfg: PreregConfig) -> Params:
    """Solve the generator's probabilities from a scenario's targets; infeasible targets are refused."""
    mu, k = sc.accuracy, 1 / sc.icc - 1
    panel = cfg.c5_models
    if len(sc.c5) != len(panel):
        raise ValueError(f"c5 gives {len(sc.c5)} effects for {len(panel)} panel models")
    if len(sc.c1) != len(cfg.c1a_models):
        raise ValueError(f"c1 gives {len(sc.c1)} differences for {len(cfg.c1a_models)} C1 models")
    rates = sc.panel_flag_rates or (sc.flag_rate[0],) * len(panel)
    if len(rates) != len(panel):
        raise ValueError(f"panel_flag_rates gives {len(rates)} rates for {len(panel)} panel models")
    flags = {m: _flag_probs(r, sc.flag_precision, mu) for m, r in zip(panel, rates)}
    pw_c, pc_c = _flag_probs(sc.flag_rate[0], sc.flag_precision, mu)
    pw_n, pc_n = _flag_probs(sc.flag_rate[1], sc.flag_precision, mu)
    pw_u, pc_u = 1 - (1 - pw_c) * (1 - pw_n), 1 - (1 - pc_c) * (1 - pc_n)
    net_c = _net(mu, pw_c, pc_c, sc.curated_fix, sc.corrupt)
    net_u = net_c + sc.c4[0]
    fix = {cfg.curated_source: sc.curated_fix, cfg.union_source: _fix_for(net_u, mu, pw_u, pc_u, sc.corrupt),
           cfg.census_source: _fix_for(net_u - sc.c4[1], mu, pw_n, pc_n, sc.corrupt)}
    generic = {}
    for m, theta in zip(panel, sc.c5):
        gw, gc = sc.curated_fix - theta, 1 - sc.corrupt - theta
        lo, hi = min(gw, gc) - sc.c5_spread, max(gw, gc) + sc.c5_spread
        if lo < 0 or hi > 1:
            raise ValueError(f"C5 effect {theta} with spread {sc.c5_spread} is impossible with fix {sc.curated_fix}")
        generic[m] = (gw, gc)
    for m, theta in zip(cfg.c1a_models, sc.c1):
        if not -mu <= theta <= 1 - mu:
            raise ValueError(f"C1 difference {theta} is impossible at accuracy {mu}")
    b = sc.bank
    rates_by_source = {cfg.curated_source: b.curated, cfg.census_source: b.census, **dict(zip(cfg.masked_sources, b.masked))}
    for name, r in rates_by_source.items():
        if not all(0 <= x <= 1 for x in (r.type_wrong, r.other_wrong, r.fpr)):
            raise ValueError(f"bank rates for {name} leave [0, 1]")
    if not 0 < b.icc < 1:
        raise ValueError("the bank's intra-question correlation must lie in (0, 1)")
    return Params(beta=(mu * k, (1 - mu) * k), flags=flags,
                  family_flags={cfg.census_source: (pw_n, pc_n), cfg.union_source: (pw_u, pc_u)}, fix=fix,
                  generic=generic, c1=dict(zip(cfg.c1a_models, sc.c1)), c5=dict(zip(panel, sc.c5)),
                  bank_rates=rates_by_source)


def bank_structure(sc: Scenario, n: int) -> BankStructure:
    return sc.bank.structure or BankStructure.uniform(n)


def truths(sc: Scenario, cfg: PreregConfig, n: int) -> dict[str, dict[str, float]]:
    """The true value of every registered component (prereg §2, §4.1), by contrast and component label, for a bank of
    n questions. Bank rates are per-question means: with Beta rates of mean mu every per-question share has mean mu,
    and independent sources flag an item that the union flags and s does not with mean mu_other (1 - mu_s)."""
    p = params(sc, cfg)
    st = bank_structure(sc, n)
    rates = p.bank_rates
    cur, cen = cfg.curated_source, cfg.census_source
    tw, ow = st.counts("type_wrong").astype(float), st.counts("vocabulary_absent").astype(float)
    w = tw + ow
    pi = cfg.prevalence

    def delta(over: str, other: str) -> float:
        a, b = rates[over], rates[other]
        share = (tw * b.type_wrong * (1 - a.type_wrong) + ow * b.other_wrong * (1 - a.other_wrong))[w > 0] / w[w > 0]
        return pi * float(share.mean()) - (1 - pi) * b.fpr * (1 - a.fpr)

    c1a = {"pooled over P": float(np.mean(sc.c1))} if cfg.c1a_pooled else dict(zip(cfg.models, sc.c1))
    return {
        "C1a": c1a, "C1b": {m: p.c1[m] for m in cfg.models},
        "C2": {"false-flag rate (census - curated)": rates[cen].fpr - rates[cur].fpr,
               "type-wrong recall (curated - census)": rates[cur].type_wrong - rates[cen].type_wrong},
        "C3": {"curated - mean(masked)": rates[cur].fpr - float(np.mean([rates[s].fpr for s in cfg.masked_sources]))},
        "C4": {f"union - {cur}": delta(cur, cen), f"union - {cen}": delta(cen, cur)},
        "C5": {"pooled over P": float(np.mean(sc.c5))},
        "C5 per model": dict(p.c5),
    }


def is_null(contrast: str, t: dict[str, float], cfg: PreregConfig, two_sided_c2: bool = False) -> bool:
    """Whether the contrast's claim is false under the truth (prereg §2's claims). C2 is two-sided since change-log
    entry 106 (``two_sided_c2``, as calibration step 3 reads it): null unless both components lie on one side; steps 1
    and 2 recorded the one-sided null, the default here so that their records can still be verified."""
    v = list(t.values())
    if contrast == "C1a" or (contrast == "C2" and two_sided_c2):
        return not (all(x > 1e-12 for x in v) or all(x < -1e-12 for x in v))
    if contrast == "C1b":
        return not all(abs(x) < cfg.c1_margin - 1e-12 for x in v)
    if contrast == "C3":
        return abs(v[0]) < 1e-12
    return not all(x > 1e-12 for x in v)  # C2, C4, C5: one-sided IUTs (C5 pooled: one component)


# --- one dataset ------------------------------------------------------------------------------


@dataclass
class Dataset:
    """Per-output draws (questions x repetitions) for agent arms, (items,) for the bank."""

    natural: dict[str, np.ndarray]  # model -> correct
    missing: dict[tuple[str, str], np.ndarray]  # (model, cohort) -> missing attempt
    prompt: dict[tuple[str, str], np.ndarray]  # (model, cohort) -> correct
    flags: dict[tuple[str, str], np.ndarray]  # (model, source) -> flagged (generic = curated)
    retry_ok: dict[tuple[str, str], np.ndarray]  # (model, source) -> retry correct (where flagged), before loss
    retry_missing: dict[tuple[str, str], np.ndarray]  # (model, source) -> flagged and the retry lost
    bank_q: np.ndarray
    bank_kind: np.ndarray  # index into KINDS
    bank_flags: dict[str, np.ndarray]  # source -> flagged
    bank_n: int = 0  # the bank's questions (the agent questions unless the structure holds more)


def _sources(cfg: PreregConfig, model: str) -> tuple[str, ...]:
    if model in cfg.models:
        return (cfg.curated_source, cfg.census_source, cfg.union_source)
    return (cfg.curated_source,)


def _prompt_models(cfg: PreregConfig) -> list[str]:
    return list(dict.fromkeys([*cfg.models, *(cfg.c1a_models if cfg.c1a_pooled else ())]))


def generate(sc: Scenario, index: QuestionIndex, cfg: PreregConfig, rng: np.random.Generator) -> Dataset:
    """One simulated dataset of every registered arm, by the process in the module docstring."""
    p = params(sc, cfg)
    q, r = len(index), sc.n_reps
    mu = sc.accuracy
    natural, missing, prompt, flags, retry_ok, retry_missing = {}, {}, {}, {}, {}, {}
    shared_pq = rng.beta(*p.beta, size=q)
    shared = rng.random(q) < sc.cross_model
    spread_sign = np.where(rng.random(q) < 0.5, -1.0, 1.0)[:, None]
    models = list(dict.fromkeys([*cfg.c5_models, *_prompt_models(cfg)]))
    for model in models:
        pq = np.where(shared, shared_pq, rng.beta(*p.beta, size=q))
        cohorts = [(cfg.natural_cohort, pq)]
        if model in _prompt_models(cfg):
            theta = p.c1.get(model, 0.0)
            p_cur = pq + theta / (1 - mu) * (1 - pq) if theta >= 0 else pq * (1 + theta / mu)
            cohorts += [(cfg.census_prompt_cohort, pq), (cfg.curated_prompt_cohort, p_cur)]
        for cohort, probs in cohorts:
            gone = rng.random((q, r)) < sc.missing
            correct = (rng.random((q, r)) < probs[:, None]) & ~gone
            missing[(model, cohort)] = gone
            if cohort == cfg.natural_cohort:
                natural[model] = correct
            else:
                prompt[(model, cohort)] = correct
        if model not in cfg.c5_models and model not in cfg.models:
            continue
        nat, out = natural[model], ~missing[(model, cfg.natural_cohort)]
        probs = {cfg.curated_source: p.flags.get(model) or _flag_probs(sc.flag_rate[0], sc.flag_precision, mu)}
        if model in cfg.models:
            probs.update(p.family_flags)
        u = rng.random((q, r))
        flags[(model, cfg.curated_source)] = out & np.where(nat, u < probs[cfg.curated_source][1],
                                                            u < probs[cfg.curated_source][0])
        if model in cfg.models:
            u = rng.random((q, r))
            pw_n, pc_n = probs[cfg.census_source]
            flags[(model, cfg.census_source)] = out & np.where(nat, u < pc_n, u < pw_n)
            flags[(model, cfg.union_source)] = flags[(model, cfg.curated_source)] | flags[(model, cfg.census_source)]
        flags[(model, cfg.generic_source)] = flags[(model, cfg.curated_source)]
        common = rng.random((q, r))
        couple = rng.random((q, r)) < sc.retry_coupling
        for s in _sources(cfg, model):
            uu = np.where(couple, common, rng.random((q, r)))
            lost = rng.random((q, r)) < sc.retry_missing
            retry_ok[(model, s)] = np.where(nat, uu >= sc.corrupt, uu < p.fix[s])
            retry_missing[(model, s)] = lost & flags[(model, s)]
        if model in cfg.c5_models:
            gw, gc = p.generic[model]
            gw_q, gc_q = gw - sc.c5_spread * spread_sign, gc - sc.c5_spread * spread_sign
            uu = np.where(couple, common, rng.random((q, r)))
            lost = rng.random((q, r)) < sc.retry_missing
            # comonotone with the checker arms where coupled: success is u < gw on a wrong output, u >= 1 - gc on a
            # correct one; with an own uniform the same events have probabilities gw and gc
            retry_ok[(model, cfg.generic_source)] = np.where(nat, uu >= 1 - gc_q, uu < gw_q)
            retry_missing[(model, cfg.generic_source)] = lost & flags[(model, cfg.generic_source)]
    st = bank_structure(sc, q)
    qb = len(st.type_wrong)  # the bank's questions: the agent questions, or more with a bank-only extension
    bank_q = np.concatenate([np.repeat(np.arange(qb), st.counts(k)) for k in KINDS])
    bank_kind = np.concatenate([np.full(int(st.counts(k).sum()), KINDS.index(k)) for k in KINDS])
    order = np.lexsort((bank_kind, bank_q))  # items grouped by question, kinds in KINDS order
    bank_q, bank_kind = bank_q[order], bank_kind[order]
    kk = 1 / sc.bank.icc - 1
    bank_flags = {}
    for s, r in p.bank_rates.items():
        prob = np.zeros(len(bank_q))
        for k, mu in (("type_wrong", r.type_wrong), ("vocabulary_absent", r.other_wrong), ("controls", r.fpr)):
            if mu <= 0 or mu >= 1:
                rate_q = np.full(qb, float(mu))
            else:
                rate_q = rng.beta(mu * kk, (1 - mu) * kk, size=qb)
            sel = np.isin(bank_kind, [KINDS.index(x) for x in CONTROLS]) if k == "controls" else bank_kind == KINDS.index(k)
            prob[sel] = rate_q[bank_q[sel]]
        bank_flags[s] = rng.random(len(bank_q)) < prob
    return Dataset(natural, missing, prompt, flags, retry_ok, retry_missing, bank_q, bank_kind, bank_flags, qb)


def _sum(values: np.ndarray, q: int) -> np.ndarray:
    return values.reshape(q, -1).sum(axis=1).astype(np.float64)


def dataset_vectors(ds: Dataset, index: QuestionIndex, cfg: PreregConfig) -> Vectors:
    """The array path: question vectors straight from the draws (metrics.question_vectors' sums, for the
    indicators the contrasts read)."""
    q = len(index)
    arms: dict = {}
    for model in _prompt_models(cfg):
        for cohort in (cfg.curated_prompt_cohort, cfg.census_prompt_cohort):
            gone = ds.missing[(model, cohort)]
            n = _sum(np.ones_like(gone), q)
            arms[cfg.prompt(model, cohort)] = {"n": n, "att": n, "correct": _sum(ds.prompt[(model, cohort)], q),
                                               "missing": _sum(gone, q)}
    for model in cfg.models:
        nat, out = ds.natural[model], ~ds.missing[(model, cfg.natural_cohort)]
        for s in (cfg.curated_source, cfg.census_source, cfg.union_source):
            f, ok = ds.flags[(model, s)], ds.retry_ok[(model, s)] & ~ds.retry_missing[(model, s)]
            fixed, corrupted = f & ~nat & out & ok, f & nat & ~ok
            n = _sum(np.ones_like(out), q)
            arms[cfg.checker(model, s)] = {"n": n, "att": n, "delta": _sum(fixed, q) - _sum(corrupted, q)}
    qb = ds.bank_n
    for s, flagged in ds.bank_flags.items():
        def bc(mask: np.ndarray) -> np.ndarray:
            return np.bincount(ds.bank_q, weights=mask.astype(np.float64), minlength=qb)

        kind = {k: ds.bank_kind == KINDS.index(k) for k in KINDS}
        changing = kind["type_wrong"] | kind["vocabulary_absent"]
        arms[cfg.bank(s)] = {"n": bc(np.ones_like(flagged)), "F": bc(flagged), "W": bc(changing),
                             "W_F": bc(flagged & changing), "tw": bc(kind["type_wrong"]),
                             "W_F_tw": bc(flagged & kind["type_wrong"]), "C_untouched": bc(kind["untouched"]),
                             "C_F_untouched": bc(flagged & kind["untouched"]), "C_rewrite": bc(kind["rewrite"]),
                             "C_F_rewrite": bc(flagged & kind["rewrite"]), "C_neutral": bc(kind["answer_neutral"]),
                             "C_F_neutral": bc(flagged & kind["answer_neutral"])}
    c5 = {}
    for model in cfg.c5_models:
        f = ds.flags[(model, cfg.curated_source)]
        yf = ds.retry_ok[(model, cfg.curated_source)] & ~ds.retry_missing[(model, cfg.curated_source)]
        g_lost = ds.retry_missing[(model, cfg.generic_source)]
        yg = ds.retry_ok[(model, cfg.generic_source)] & ~g_lost
        num = _sum(f & yf, q) - _sum(f & yg, q)
        c5[model] = {"num": num, "den": _sum(f, q), "n": _sum(np.ones_like(f), q), "num_adverse": num - _sum(f & g_lost, q)}
    union = {}
    cur, cen = ds.bank_flags[cfg.curated_source], ds.bank_flags[cfg.census_source]
    for over, a, b in ((cfg.curated_source, cur, cen), (cfg.census_source, cen, cur)):
        only = b & ~a
        union[over] = {name: np.bincount(ds.bank_q, weights=(only & mask).astype(np.float64), minlength=qb)
                       for name, mask in (("wrong", np.isin(ds.bank_kind, [KINDS.index(k) for k in CHANGING])),
                                          ("gold", ds.bank_kind == KINDS.index("untouched")),
                                          ("rewrites", ds.bank_kind == KINDS.index("rewrite")))}
    return Vectors(arms, c5, union)


def dataset_objects(ds: Dataset, index: QuestionIndex, questions: dict[str, dict],
                    cfg: PreregConfig) -> tuple[list[Attempt], list[CheckedOutput]]:
    """The object path: the same draws as attempts and checker-ledger rows, which the registered analysis
    reads through the loaders, so the simulation exercises the code that runs on the test records."""
    attempts, ledger = [], []

    def attempt(model: str, cohort: str, qi: int, rep: int, correct: bool, gone: bool) -> Attempt:
        qid = index.questions[qi]
        meta = questions[qid]
        return Attempt(run_id=f"{model}|{cohort}|{qid}|{rep}", graph=cfg.graph, model=model, cohort=cohort,
                       question_id=qid, base_question_id=qid, template=meta["template"], split=cfg.split,
                       repetition=rep, status="missing" if gone else "submitted", output=not gone,
                       correct=bool(correct) and not gone, f1=float(bool(correct) and not gone),
                       rules_version=cfg.scorer_version, tier=meta.get("tier", "unrecorded"),
                       missing_reason="absent" if gone else None)

    for model in _prompt_models(cfg):
        for cohort in (cfg.curated_prompt_cohort, cfg.census_prompt_cohort):
            for (qi, rep), correct in np.ndenumerate(ds.prompt[(model, cohort)]):
                attempts.append(attempt(model, cohort, qi, rep, correct, ds.missing[(model, cohort)][qi, rep]))
    for model in dict.fromkeys([*cfg.models, *cfg.c5_models]):
        nat = ds.natural[model]
        sources = (*_sources(cfg, model), cfg.generic_source)
        for (qi, rep), correct in np.ndenumerate(nat):
            gone = bool(ds.missing[(model, cfg.natural_cohort)][qi, rep])
            a = attempt(model, cfg.natural_cohort, qi, rep, correct, gone)
            attempts.append(a)
            if gone:
                continue
            for s in sources:
                flagged = bool(ds.flags[(model, s)][qi, rep])
                lost = bool(ds.retry_missing[(model, s)][qi, rep])
                ok = bool(ds.retry_ok[(model, s)][qi, rep]) and not lost
                ledger.append(CheckedOutput(
                    run_id=a.run_id, graph=cfg.graph, model=model, cohort=cfg.natural_cohort, source=s,
                    use=cfg.checker_use, question_id=a.question_id, base_question_id=a.base_question_id,
                    initial_correct=a.correct, status="F" if flagged else "P", substantive=False,
                    flagged_check_types=("class",) if flagged else (), retried=flagged,
                    retry_status=None if not flagged else ("missing" if lost else "submitted" if ok else "failed"),
                    retry_correct=None if not flagged else ok))
    for s, flagged in ds.bank_flags.items():
        for j, (qi, kind) in enumerate(zip(ds.bank_q, ds.bank_kind)):
            qid, name = (index.questions[qi] if ds.bank_n == len(index) else bank_question(qi)), KINDS[kind]
            changing = name in CHANGING
            ledger.append(CheckedOutput(
                run_id=f"bank|{qid}|{j}", graph=cfg.graph, model=cfg.bank_model, cohort=BANK, source=s,
                use=cfg.bank_use, question_id=qid, base_question_id=qid, initial_correct=not changing,
                status="F" if flagged[j] else "P", substantive=False,
                flagged_check_types=("class",) if flagged[j] else (),
                bank_item="answer_changing" if changing else LEDGER_ITEM[name], fault_class=name if changing else None))
    return attempts, ledger


def bank_question(i: int) -> str:
    """The id of a simulated bank question when the bank holds more questions than the agent runs."""
    return f"bank-q{i:05d}"


def bank_index(n: int) -> QuestionIndex:
    return QuestionIndex.from_strata({bank_question(i): "bank" for i in range(n)})


def icc_estimate(outcomes: np.ndarray) -> float:
    """The one-way ANOVA estimate of the intra-question correlation of binary repetitions (questions x
    repetitions), for placing development runs on the scenario grid."""
    q, r = outcomes.shape
    means = outcomes.mean(axis=1)
    msb = r * ((means - outcomes.mean()) ** 2).sum() / (q - 1)
    msw = ((outcomes - means[:, None]) ** 2).sum() / (q * (r - 1))
    return float((msb - msw) / (msb + (r - 1) * msw)) if msb + (r - 1) * msw > 0 else 0.0


def rate(hits: list[bool] | np.ndarray) -> dict[str, Any] | None:
    hits = np.asarray(hits, dtype=bool)
    if hits.size == 0:
        return None
    n, k = int(hits.size), int(hits.sum())
    return {"rate": k / n, "mc_se": math.sqrt((k / n) * (1 - k / n) / n), "k": k, "n": n}
