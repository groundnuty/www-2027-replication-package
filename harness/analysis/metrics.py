"""Ledger metrics per arm (prereg §3 "Outcome definitions"; proposal §7.6 "Accounting").

An *arm* is graph x model x cohort x use x source. Prompt arms (use ``prompt``) are a
cohort's attempts alone; checker arms (``checker``, ``both``, ``replay``) are the same
attempts joined with the ledger rows of one source and use; fault-bank arms (cohort ``bank``)
are the bank's items checked by one source. Every attempt of a cohort is a unit of its arms,
so N_att counts scored failures and missing attempts, which sit outside the checker matrix.

**Question vectors** (prereg §4 "Resampling"). Each arm is reduced to per-question sums of its
output-level indicators, aligned to the analysis's ``QuestionIndex``, plus the number of units
per question. Every quantity is then ``(K @ num) / (K @ den)`` for one resample matrix K.

**Weighting** (prereg §3 "Weighting"):
- per-output *proportions* over N_att (accuracy, net change, ...): averaged within a base
  question over its attempts, then over questions (equal weight per base question):
  ``num_q = mean_i num_i``, ``den_q = 1`` for each question with attempts in the arm;
- *ratios* (precision, recall, false-flag rate, and every other conditional rate): ratios of
  pooled counts, ``num_q = sum_i num_i``, ``den_q = sum_i den_i``, recomputed per resample.
A ratio whose denominator is empty is not estimable (None), never zero (§3 "Non-estimability").
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from harness.analysis.inputs import (
    BANK,
    UNRECORDED,
    Attempt,
    CheckedOutput,
    InputError,
    check_bank_pairing,
)
from harness.analysis.stats import QuestionIndex

PROMPT = "prompt"
NO_SOURCE = "-"
PROPORTION, RATIO = "proportion", "ratio"
TYPE_WRONG = "type_wrong"


@dataclass(frozen=True, order=True)
class ArmKey:
    """An arm: graph x model x cohort x use x source (proposal §7.1 design; prereg §1 "Run set")."""

    graph: str
    model: str
    cohort: str
    use: str
    source: str = NO_SOURCE

    @property
    def label(self) -> str:
        return f"{self.graph}|{self.model}|{self.cohort}|{self.use}|{self.source}"

    def to_json(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class Metric:
    """A metric as numerator and denominator indicators with its weighting (prereg §3)."""

    num: str
    den: str
    weighting: str  # PROPORTION (per question, then over questions) or RATIO (pooled counts), prereg §3
    kind: str  # "rate" (a share), "change" (a signed difference in rate points), "score" (mean F1)
    arms: str  # "all", "checker" (needs a ledger row) or "bank"
    definition: str


METRICS: dict[str, Metric] = {
    "accuracy": Metric("correct", "att", PROPORTION, "rate", "all",
                       "exact-match accuracy of the initial output over N_att; failures, abstentions and missing "
                       "attempts count wrong (prereg §3, §4)"),
    "mean_f1": Metric("f1", "att", PROPORTION, "score", "all", "row-level F1 of the initial output over N_att"),
    "output_rate": Metric("out", "att", PROPORTION, "rate", "all", "N_out / N_att"),
    "missing_rate": Metric("missing", "att", PROPORTION, "rate", "all", "missing attempts / N_att (prereg §4)"),
    "silent_failure": Metric("W", "att", PROPORTION, "rate", "all", "wrong scored submissions over N_att"),
    "accuracy_final": Metric("final_correct", "att", PROPORTION, "rate", "checker",
                             "accuracy after the loop over N_att; a failed, abstained or missing retry counts wrong"),
    "net_change": Metric("delta", "att", PROPORTION, "change", "checker",
                         "(fixes - corruptions) / N_att (prereg §3 'Accuracy and net change')"),
    "repair_yield": Metric("delta", "out", RATIO, "change", "checker", "(fixed - corrupted) / N_out"),
    "flag_rate": Metric("F", "out", RATIO, "rate", "checker", "F / N_out"),
    "flag_precision": Metric("W_F", "F", RATIO, "rate", "checker", "W_F / (W_F + C_F), pooled counts (prereg §3)"),
    "false_flag_rate": Metric("C_F", "C", RATIO, "rate", "checker", "C_F / (C_F + C_P + C_U), pooled counts"),
    "recall": Metric("W_F", "W", RATIO, "rate", "checker", "W_F / (W_F + W_P + W_U), pooled counts"),
    "error_certified": Metric("W_P", "P", RATIO, "rate", "checker", "W_P / (W_P + C_P)"),
    "error_not_intervened": Metric("W_PU", "PU", RATIO, "rate", "checker", "(W_P + W_U) / (W_P + C_P + W_U + C_U)"),
    "fix_rate": Metric("fixed", "W_F_retried", RATIO, "rate", "checker", "P(right after retry | wrong, flagged)"),
    "corruption_rate": Metric("corrupted", "C_F_retried", RATIO, "rate", "checker",
                              "P(wrong after retry | right, flagged)"),
    "unknown_rate": Metric("U", "out", RATIO, "rate", "checker", "U / N_out"),
    "substantive_coverage": Metric("sub", "out", RATIO, "rate", "checker",
                                   "outputs with a substantive ontology-derived check decided (v2) / N_out"),
    "substantive_pass_share": Metric("P_sub", "P", RATIO, "rate", "checker", "substantive passes / certified passes"),
    "number_traceability_flag_rate": Metric("nt_flag", "out", RATIO, "rate", "checker",
                                            "number-traceability flags / N_out (its own layer, rule 8)"),
    "recall_type_wrong": Metric("W_F_tw", "tw", RATIO, "rate", "bank",
                                "flagged type-wrong answer-changing mutations / all of them (prereg §3, C2 ii)"),
    "false_flag_rate_untouched": Metric("C_F_untouched", "C_untouched", RATIO, "rate", "bank",
                                        "false-flag rate on untouched gold controls (prereg §3)"),
    "false_flag_rate_neutral": Metric("C_F_neutral", "C_neutral", RATIO, "rate", "bank",
                                      "false-flag rate on answer-neutral mutations, their own stratum (prereg §4.1)"),
    "false_flag_rate_rewrite": Metric("C_F_rewrite", "C_rewrite", RATIO, "rate", "bank",
                                      "false-flag rate on answer-preserving rewrites (prereg §4.1 f_rewrite)"),
}
PROMPT_METRICS = tuple(m for m, spec in METRICS.items() if spec.arms == "all")
CHECKER_METRICS = tuple(m for m, spec in METRICS.items() if spec.arms in ("all", "checker"))
BANK_METRICS = ("flag_rate", "flag_precision", "false_flag_rate", "recall", "unknown_rate", "substantive_coverage",
                "recall_type_wrong", "false_flag_rate_untouched", "false_flag_rate_rewrite", "false_flag_rate_neutral")
# Exact bounds only for binary question-level events (prereg §3): "any false flag on q", "any corruption on q".
QUESTION_EVENTS = {"false_flag_rate": ("C_F", "C"), "corruption_rate": ("corrupted", "C_F_retried")}

INDICATORS = ("att", "out", "missing", "correct", "f1", "W", "C", "F", "P", "U", "C_F", "C_P", "C_U", "W_F", "W_P",
              "W_U", "PU", "W_PU", "sub", "P_sub", "P_vac", "retried", "retry_missing", "fixed", "corrupted",
              "W_F_retried", "C_F_retried", "final_correct", "delta", "nt_flag", "branch_local", "data_fault",
              "tw", "W_F_tw", "C_untouched", "C_F_untouched", "C_neutral", "C_F_neutral", "C_rewrite", "C_F_rewrite")


@dataclass(frozen=True)
class Unit:
    """One attempt of an arm with its ledger row and output-level indicators (proposal §7.6 "Accounting")."""

    cluster: str  # the base question
    attempt: Attempt
    checked: CheckedOutput | None
    ind: dict[str, float]


def indicators(a: Attempt, c: CheckedOutput | None) -> dict[str, float]:
    """Output-level indicators (proposal §7.6 "Accounting"; prereg §3): the six cells, retries
    (fix W->C, corruption C->W; a failed, abstained or missing retry is wrong), bank item kinds."""
    out, correct = a.output, a.correct
    wrong = out and not correct
    status = c.status if c is not None else None
    retried = bool(c and c.retried)
    retry_ok = bool(c and c.retry_correct)  # a failed, abstained or missing retry is wrong (prereg §3, §4)
    fixed = retried and wrong and retry_ok
    corrupted = retried and correct and not retry_ok
    item = c.bank_item if c is not None else None
    flagged = status == "F"
    tw = item == "answer_changing" and c is not None and c.fault_class == TYPE_WRONG
    return {
        "att": 1.0, "out": float(out), "missing": float(a.status == "missing"), "correct": float(correct),
        "f1": a.f1, "W": float(wrong),
        "C": float(correct and status is not None),  # C_F + C_P + C_U: correct outputs with a status
        "F": float(flagged), "P": float(status == "P"), "U": float(status == "U"),
        "C_F": float(correct and flagged), "C_P": float(correct and status == "P"), "C_U": float(correct and status == "U"),
        "W_F": float(wrong and flagged), "W_P": float(wrong and status == "P"), "W_U": float(wrong and status == "U"),
        "PU": float(status in ("P", "U")), "W_PU": float(wrong and status in ("P", "U")),
        "sub": float(bool(c and c.substantive)),
        "P_sub": float(status == "P" and bool(c and c.substantive)),
        "P_vac": float(status == "P" and not (c and c.substantive)),
        "retried": float(retried), "retry_missing": float(bool(c and c.retry_missing)),
        "fixed": float(fixed), "corrupted": float(corrupted),
        "W_F_retried": float(wrong and flagged and retried), "C_F_retried": float(correct and flagged and retried),
        "final_correct": float(retry_ok if retried else correct),
        "delta": float(fixed) - float(corrupted),
        "nt_flag": float(bool(c and c.number_traceability == "flag")),
        "branch_local": float(bool(c and c.branch_local_only)),
        "data_fault": float(bool(c and c.data_fault)),
        "tw": float(tw), "W_F_tw": float(tw and flagged),
        "C_untouched": float(item == "untouched"), "C_F_untouched": float(item == "untouched" and flagged),
        "C_neutral": float(item == "answer_neutral"), "C_F_neutral": float(item == "answer_neutral" and flagged),
        "C_rewrite": float(item == "rewrite"), "C_F_rewrite": float(item == "rewrite" and flagged),
    }


def _bank_attempt(c: CheckedOutput, questions: Mapping[str, dict]) -> Attempt:
    q = questions.get(c.question_id) or {}
    return Attempt(run_id=c.run_id, graph=c.graph, model=c.model, cohort=BANK, question_id=c.question_id,
                   base_question_id=c.base_question_id, template=q.get("template") or UNRECORDED,
                   split=q.get("split") or UNRECORDED, repetition=0, status="submitted", output=True,
                   correct=c.initial_correct, f1=float(c.initial_correct), tier=q.get("tier") or UNRECORDED)


def build_arms(attempts: Sequence[Attempt], ledger: Iterable[CheckedOutput],
               questions: Mapping[str, dict] | None = None) -> dict[ArmKey, list[Unit]]:
    """Proposal §7.6; prereg §3 "Shared initial outputs". Prompt arms for every cohort, one checker
    arm per (cohort, use, source) in the ledger,
    and one fault-bank arm per bank source.

    The ledger must cover each scored output of its cohort exactly once per arm, and agree with
    the scored record on the base question and on correctness; every bank source must hold the
    same items (inputs.check_bank_pairing). Anything else is refused, since a missing or stale row
    would bias every rate silently."""
    ledger = list(ledger)
    check_bank_pairing(ledger)
    by_cohort: dict[tuple[str, str, str], list[Attempt]] = defaultdict(list)
    by_run: dict[str, Attempt] = {}
    for a in attempts:
        by_cohort[(a.graph, a.model, a.cohort)].append(a)
        by_run[a.run_id] = a
    arms: dict[ArmKey, list[Unit]] = {}
    for (graph, model, cohort), rows in sorted(by_cohort.items()):
        arms[ArmKey(graph, model, cohort, PROMPT)] = [Unit(a.base_question_id, a, None, indicators(a, None))
                                                      for a in rows]
    grouped: dict[ArmKey, dict[str, CheckedOutput]] = defaultdict(dict)
    bank: dict[ArmKey, list[Unit]] = defaultdict(list)
    for c in ledger:
        key = ArmKey(c.graph, c.model, c.cohort, c.use, c.source)
        if c.cohort == BANK:
            a = _bank_attempt(c, questions or {})
            bank[key].append(Unit(a.base_question_id, a, c, indicators(a, c)))
            continue
        a = by_run.get(c.run_id)
        if a is None or (a.graph, a.model, a.cohort) != (c.graph, c.model, c.cohort):
            raise InputError(f"ledger row {c.run_id} ({c.source}, {c.use}) is not an attempt of its cohort "
                             f"{c.graph}/{c.model}/{c.cohort}")
        if not a.output:
            raise InputError(f"ledger row {c.run_id} ({c.source}, {c.use}): the attempt is not a scored output")
        if c.initial_correct != a.correct:
            raise InputError(f"ledger row {c.run_id} ({c.source}, {c.use}): initial_correct {c.initial_correct} "
                             f"disagrees with the scored record ({a.correct})")
        if c.base_question_id != a.base_question_id:
            raise InputError(f"ledger row {c.run_id}: base question {c.base_question_id} != {a.base_question_id}")
        grouped[key][c.run_id] = c
    for key, rows in sorted(grouped.items()):
        cohort = by_cohort[(key.graph, key.model, key.cohort)]
        missing = [a.run_id for a in cohort if a.output and a.run_id not in rows]
        if missing:
            raise InputError(f"arm {key.label}: no ledger row for {len(missing)} scored outputs, e.g. {missing[:3]}")
        arms[key] = [Unit(a.base_question_id, a, rows.get(a.run_id), indicators(a, rows.get(a.run_id)))
                     for a in cohort]
    arms.update(sorted(bank.items()))
    return arms


def question_vectors(units: Sequence[Unit], index: QuestionIndex,
                     names: Sequence[str] = INDICATORS) -> dict[str, np.ndarray]:
    """Per-question sums of each indicator over the units, aligned to the index, and ``n``,
    the units per question (prereg §4: the arm's part of every complete question vector)."""
    which = np.empty(len(units), dtype=np.int64)
    for i, u in enumerate(units):
        if u.cluster not in index:
            raise InputError(f"unit {u.attempt.run_id}: base question {u.cluster} is not in the analysis's question index")
        which[i] = index.position(u.cluster)
    q = len(index)
    out = {"n": np.bincount(which, minlength=q).astype(np.float64)}
    for name in names:
        values = np.fromiter((u.ind[name] for u in units), dtype=np.float64, count=len(units))
        out[name] = np.bincount(which, weights=values, minlength=q)
    return out


def metric_arrays(vectors: Mapping[str, np.ndarray], num: str, den: str, weighting: str) -> tuple[np.ndarray, np.ndarray]:
    """(num_q, den_q) of one metric under prereg §3's weighting."""
    if weighting == PROPORTION:
        n = vectors["n"]
        present = n > 0
        safe = np.where(present, n, 1.0)
        return np.where(present, vectors[num] / safe, 0.0), np.where(present, vectors[den] / safe, 0.0)
    if weighting == RATIO:
        return vectors[num], vectors[den]
    raise ValueError(f"unknown weighting {weighting!r}")


def metric_of(vectors: Mapping[str, np.ndarray], metric: str) -> tuple[np.ndarray, np.ndarray]:
    """(num_q, den_q) of a named metric under its prereg §3 weighting."""
    spec = METRICS[metric]
    return metric_arrays(vectors, spec.num, spec.den, spec.weighting)


def point_estimate(num: np.ndarray, den: np.ndarray) -> float | None:
    """sum num / sum den; None when the denominator is empty (prereg §3 "Non-estimability")."""
    d = float(den.sum())
    return float(num.sum()) / d if d > 0 else None


def arm_counts(units: Sequence[Unit]) -> dict[str, int]:
    """Raw counts (outputs, not per-question averages; proposal §7.6): the six cells sum to N_out."""
    total = Counter()
    for u in units:
        for k, v in u.ind.items():
            total[k] += v
    statuses = Counter(u.attempt.status for u in units)
    missing_reasons = Counter(u.attempt.missing_reason for u in units if u.attempt.missing_reason)
    return {
        "N_att": len(units), "N_out": int(total["out"]),
        **{k: int(total[k]) for k in ("C_F", "C_P", "C_U", "W_F", "W_P", "W_U")},
        "P_substantive": int(total["P_sub"]), "P_vacuous": int(total["P_vac"]),
        "retried": int(total["retried"]), "fixed": int(total["fixed"]), "corrupted": int(total["corrupted"]),
        "missing": int(total["missing"]), "retry_missing": int(total["retry_missing"]),
        **{f"missing_{r}": missing_reasons[r] for r in sorted(missing_reasons)},
        "branch_local_flags": int(total["branch_local"]), "data_fault_flags": int(total["data_fault"]),
        "number_traceability_flags": int(total["nt_flag"]),
        "abstained": statuses["abstained"], "failed": statuses["failed"], "capped": statuses["capped"],
        "refused": statuses["refused"],
        "n_questions": len({u.cluster for u in units}),
    }
