"""Power of the registered weighted Holm (prereg §4.4; design log, change-log entries 74, 76 and 91) on simulated
p-values, at the frozen alpha' table restricted to the dated step-2 addendum's usable levels, and the inputs it is
computed from.

- **The procedure is the registered one** (``family_power``): each claim's p* by multiplicity.calibrated_matrix over
  the usable levels (multiplicity.load_files, WeightedCalibration.final: a claim without a usable level, or every claim
  after a failed FWER check, at p* = 1), then multiplicity.step_down_rejections. A claim's power is its rejection rate.
- **Brackets.** Each cell studies some claims (the bank cells C2, C3 and C4; the C1 cells C1a or C1b; the C5 cell
  C5); every other claim of the family is a stand-in there, its inputs measured elsewhere or pending (C1 until the dev
  grid's prompted phase, C5 until the entry-77 replay). A stand-in is forced rejected (p* = its first weighted level
  alpha w / sum_family w, which is at most every threshold alpha w / sum_S w, so the step-down always reaches and
  rejects it) or forced not rejected (p* = 1). The weighted Holm is monotone in every p*, so the two bound what the
  other claims can do to a studied claim.
- **First-level power**: p* <= alpha w / sum_family w, the claim rejected at its first weighted level by weighted
  Bonferroni whatever the other claims do (prereg §4.4's claim-level interval level), reported beside the rejection
  power. ``choose_c1a`` compares the rejection powers by the registered weighted Holm (design log ruling 2: "calibrated
  power at the 3-point margin"; prereg §4.4: power before the freeze is under weighted Holm), which can reverse the
  first-level order when another claim's rejection raises C1a's level (the review of e479fc6).
- **C1's dev rates** (``c1_dev_rates``): per model of P and prompted arm (curated, census), the dev accuracy with equal
  weight per base question and the intra-question correlation of the repetitions (simulate.icc_estimate), from the
  analysis's own attempts of a grid OUT (inputs.load_batch with the grid's cells). The simulation takes one accuracy and
  one ICC for every model (simulate.Scenario), so ``c1_scenario`` uses the equal-weight mean over P of the census arm's
  accuracy and of the per-model ICCs, with every model's curated - census difference at the 3-point margin.
- **C5's inputs** (``read_c5_inputs``, ``c5_scenario``): the entry-77 offline replay's counts on the natural cohort's dev
  outputs per model of P (outputs, correct, flagged by the curated checks, flagged and wrong), ``c5-power-inputs/1``;
  any split but dev is refused. Every model's C5 effect is at the smallest effect of interest (10 points per flagged
  output); the retry rates are not measured before the branch phase (the calibration's defaults unless the file gives
  them).
- **The bank cells** (``bank_cells``): C2, C3 and C4 at their smallest effects of interest over the earlier power
  study's grid of baseline false-flag rates f and census type-wrong recalls r (results/power-holm-20261019:
  A, R and B, with ``c4_other_wrong``), on the final bank's source-free test-split structure (prereg §4.4: no per-source
  label of a test item). No dev-split bank replay under the final checker exists, so the grid stands in for the rates.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from harness.analysis import calibration as cal
from harness.analysis import multiplicity as mu
from harness.analysis import prereg
from harness.analysis.inputs import Attempt
from harness.analysis.simulate import (
    Bank,
    BankStructure,
    Scenario,
    SourceRates,
    icc_estimate,
    params,
    rate,
)

C5_SCHEMA = "c5-power-inputs/1"
DEV_RATES_SCHEMA = "c1-dev-rates/1"
AGENT_CLAIMS = ("C1a", "C1a pooled", "C1b", "C5")
SESOI = {"C2": 0.05, "C3": 0.05, "C4": 0.01, "C5": 0.10}
FS = (0.005, 0.02, 0.05, 0.10, 0.25)  # the earlier power study's grid (results/power-holm-20261019)
RECALLS = (0.3, 0.5)
MASKED_RECALL = 0.6
BANK_ICC = 0.2
OTHER_C1, OTHER_C5 = (0.02,) * 7, (0.25,) * 7  # the calibration's alternatives (E1, F1-F7) for claims not studied
PI = 0.25


class PowerInputError(ValueError):
    """An input the power run refuses."""


# --- the registered procedure on simulated p-values ---------------------------------------------------------------


def first_level(wc: mu.WeightedCalibration, members: Sequence[str], claim: str) -> float:
    """The claim's first weighted level alpha w_i / sum_family w (0.05 w_i / 16)."""
    return mu.level(wc.alpha, wc.weights[claim], sum(wc.weights[c] for c in members))


def calibrated(p: np.ndarray, wc: mu.WeightedCalibration, members: Sequence[str]) -> np.ndarray:
    """(datasets x members) p*: each claim's test's raw p (calibration.CLAIMS x prereg.TESTS in ``p``) through its usable
    levels (WeightedCalibration.final), 1 without."""
    tests, alphas = {}, {}
    for c in members:
        t, a = wc.final(c)
        tests[c], alphas[c] = t, a
    return mu.calibrated_matrix(p, members, tests, alphas, wc.weights, wc.alpha)


def rejections(p: np.ndarray, wc: mu.WeightedCalibration, members: Sequence[str],
               forced: Mapping[str, bool] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(datasets x members) p* before forcing, and the step-down's rejections with the claims in ``forced`` forced
    rejected (True: p* = the first weighted level) or not (False: p* = 1)."""
    forced = dict(forced or {})
    unknown = set(forced) - set(members)
    if unknown:
        raise PowerInputError(f"forced claims {sorted(unknown)} are not in the family {list(members)}")
    P = calibrated(p, wc, members)
    Q = P.copy()
    for c, rejected in forced.items():
        Q[:, list(members).index(c)] = first_level(wc, members, c) if rejected else 1.0
    return P, mu.step_down_rejections(Q, [wc.weights[c] for c in members], wc.alpha)


def family_power(p: np.ndarray, wc: mu.WeightedCalibration, members: Sequence[str],
                 forced: Mapping[str, bool] | None = None) -> dict[str, dict[str, Any]]:
    """Per claim of the family: its rejection rate by the weighted step-down (``power``), with the claims in ``forced``
    forced rejected (True) or not (False); and, for every claim not forced, its first-level power."""
    forced = dict(forced or {})
    P, R = rejections(p, wc, members, forced)
    out: dict[str, dict[str, Any]] = {}
    for j, c in enumerate(members):
        if c in forced:
            out[c] = {"forced": "rejected" if forced[c] else "not rejected"}
        else:
            first = first_level(wc, members, c)
            out[c] = {"first_level": rate(P[:, j] <= first), "level": first}
        out[c]["power"] = rate(R[:, j])
    return out


def _paired(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    d = a.astype(float) - b.astype(float)
    return {"difference": float(d.mean()), "mc_se": float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else 0.0}


def c1a_comparison(p: np.ndarray, ws: Mapping[str, mu.WeightedCalibration]) -> dict[str, Any]:
    """Ruling 2's inputs (change-log entry 104: rejection power under the full weighted Holm) on one cell's datasets,
    both forms on the same datasets (calibration.dataset_p computes both C1a p-values from one simulated dataset:
    common random numbers): each C1a form's rejection power by the registered weighted Holm as simulated
    (``choose_c1a``'s input) and its first-level power, each with its binomial Monte Carlo SE; the paired differences
    pooled - two-family with their SE (the sample SD of the per-dataset differences over sqrt(n)); the choice by each
    reading (pooled on a tie: "at least"); whether the two readings agree; and whether the weighted-Holm difference is
    within 2 SE (a close call)."""
    pooled_m, two_m = prereg.FAMILY_CLAIMS[prereg.POOLED], prereg.FAMILY_CLAIMS[prereg.TWO_FAMILY]
    Pp, Rp = rejections(p, ws[prereg.POOLED], pooled_m)
    Pt, Rt = rejections(p, ws[prereg.TWO_FAMILY], two_m)
    jp, jt = pooled_m.index("C1a pooled"), two_m.index("C1a")
    fp = Pp[:, jp] <= first_level(ws[prereg.POOLED], pooled_m, "C1a pooled")
    ft = Pt[:, jt] <= first_level(ws[prereg.TWO_FAMILY], two_m, "C1a")
    first = {prereg.POOLED: float(fp.mean()), prereg.TWO_FAMILY: float(ft.mean())}
    full = {prereg.POOLED: float(Rp[:, jp].mean()), prereg.TWO_FAMILY: float(Rt[:, jt].mean())}

    def se(x: float) -> float:  # a power's binomial Monte Carlo SE over the cell's datasets
        return math.sqrt(x * (1 - x) / len(p))

    first_diff, full_diff = _paired(fp, ft), _paired(Rp[:, jp], Rt[:, jt])
    choice_first = prereg.choose_c1a(first[prereg.POOLED], first[prereg.TWO_FAMILY])
    choice_full = prereg.choose_c1a(full[prereg.POOLED], full[prereg.TWO_FAMILY])
    return {"n": len(p), "paired": "both forms on the same simulated datasets (common random numbers)",
            "first_level": first, "full_family": full,
            "first_level_mc_se": {k: se(v) for k, v in first.items()},
            "full_family_mc_se": {k: se(v) for k, v in full.items()}, "first_level_difference": first_diff,
            "full_family_difference": full_diff, "choice_first_level": choice_first, "choice_full_family": choice_full,
            "definitions_agree": choice_first == choice_full,
            "close_call": abs(full_diff["difference"]) <= 2 * full_diff["mc_se"]}


ENTRY81_LIMIT = 0.02  # prereg §4.2 (entry 81): a dev capped:context gap curated - census above 2 points


def entry81(rows: Iterable[tuple[str, str, bool]], models: Sequence[str], arms: Sequence[str] = ("curated", "census"),
            limit: float = ENTRY81_LIMIT, unmeasurable_ok: Sequence[str] = ()) -> dict[str, Any]:
    """Entry 81's membership rule for the pooled C1a: per model of P and prompted arm, the share of its dev records
    capped by the context window (``rows``: model, arm, capped:context), and the models whose curated - census gap
    exceeds ``limit`` in absolute value (they leave the pooled C1a and are reported descriptively). A model or arm
    with no record is refused, but for a model in ``unmeasurable_ok`` (a fallback model, the review of 0fa4fa4): its gap
    is then unmeasurable, recorded under "unmeasurable", and "leave" is None, so the C1a choice waits for a ruling."""
    counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for m, arm, capped in rows:
        if m in models and arm in arms:
            counts[(m, arm)][0] += bool(capped)
            counts[(m, arm)][1] += 1
    out: dict[str, Any] = {"limit": limit, "models": {}, "leave": []}
    unmeasurable = []
    for m in models:
        row = {}
        for arm in arms:
            k, n = counts.get((m, arm), (0, 0))
            if n == 0 and m not in unmeasurable_ok:
                raise PowerInputError(f"{m}: no record in arm {arm} for entry 81's capped:context rate")
            row[arm] = {"capped_context": k, "records": n, "rate": k / n if n else None}
        if any(row[a]["records"] == 0 for a in arms):
            row["gap"] = None
            unmeasurable.append(m)
        else:
            row["gap"] = row[arms[0]]["rate"] - row[arms[-1]]["rate"]
        out["models"][m] = row
        if row["gap"] is not None and abs(row["gap"]) > limit:
            out["leave"].append(m)
    if unmeasurable:
        out["unmeasurable"], out["leave"] = unmeasurable, None
    return out


def bracketed(p: np.ndarray, wc: mu.WeightedCalibration, members: Sequence[str],
              pending: Iterable[str]) -> dict[str, dict[str, Any]]:
    """The family's power as simulated, and with every stand-in claim (``pending``) forced rejected and forced not
    rejected; when C5 is a stand-in, also with C5 alone of them rejected (the agent claims' likeliest state on the
    calibration's alternatives: C5 near certain, C1a and C1b rarely)."""
    pending = [c for c in members if c in set(pending)]
    out = {"as simulated": family_power(p, wc, members),
           "others rejected": family_power(p, wc, members, dict.fromkeys(pending, True)),
           "others not rejected": family_power(p, wc, members, dict.fromkeys(pending, False))}
    if "C5" in pending:
        out["only C5 of the others rejected"] = family_power(p, wc, members, {c: c == "C5" for c in pending})
    return out


# --- the bank cells ------------------------------------------------------------------------------------------------


def type_wrong_share(st: BankStructure) -> float:
    """mean_q T_q / W_q over the questions with wrong items: the type-wrong weight of C4's union-only recall."""
    tw, ow = np.asarray(st.type_wrong, float), np.asarray(st.other_wrong, float)
    w = tw + ow
    return float((tw[w > 0] / w[w > 0]).mean())


def c4_other_wrong(f_cur: float, f_cen: float, tw_cur: float, tw_cen: float, share: float,
                   delta: float = 0.01) -> tuple[float, float] | None:
    """Other-wrong recalls (curated a, census b) putting C4's IUT at its SESOI, min_s Delta_s = delta, with the other
    comparison as low as it can go (results/power-holm-20261019/power_holm.py, unchanged): x = b (1 - a) and
    y = a (1 - b) are the needed other-wrong union-only shares; None when no recalls reach it."""
    o = 1 - share
    x = ((delta + (1 - PI) * f_cen * (1 - f_cur)) / PI - share * tw_cen * (1 - tw_cur)) / o
    y = ((delta + (1 - PI) * f_cur * (1 - f_cen)) / PI - share * tw_cur * (1 - tw_cen)) / o
    if x <= 0 and y <= 0:
        return None
    if y <= 0:
        return (0.0, x) if x <= 1 else None
    if x <= 0:
        return (y, 0.0) if y <= 1 else None
    d = x - y
    disc = (1 - d) ** 2 - 4 * y
    if disc < 0:
        return None
    for a in (((1 - d) - math.sqrt(disc)) / 2, ((1 - d) + math.sqrt(disc)) / 2):
        b = a + d
        if 0 <= a <= 1 and 0 <= b <= 1:
            return a, b
    return None


BANK_STUDIED = ("C2", "C3", "C4")


@dataclass(frozen=True)
class Cell:
    """One simulated cell: its name, its scenario (None: infeasible or its inputs pending, never simulated), the claims
    it studies (their truths are the cell's point; every other claim of a family is a stand-in, bracketed at
    aggregation) and a note."""

    name: str
    scenario: Scenario | None
    studied: tuple[str, ...]
    note: str = ""

    def pending(self, members: Sequence[str]) -> list[str]:
        return [c for c in members if c not in self.studied]


def bank_cells(structure: BankStructure, c1: Sequence[float] = OTHER_C1, c5: Sequence[float] = OTHER_C5) -> list[Cell]:
    """The grid in a fixed order (the cell index is the dataset seed's second entry; an infeasible cell keeps its
    index). A: C2 and C3 at the SESOI (census f + 0.05, masked f + 0.05); R: C3 reversed (curated f + 0.05, masked f,
    census f + 0.10); B: as A with C4 at its SESOI (``c4_other_wrong``). Type-wrong recall: census r, curated r + 0.05;
    other-wrong recall equal to type-wrong in A and R; bank ICC 0.2; masked recall 0.6."""
    out, share = [], type_wrong_share(structure)
    for fam in ("A", "R", "B"):
        for r in RECALLS:
            for f in FS:
                name = f"{fam} f={f:g} r={r:g}"
                tc, tn = r + SESOI["C2"], r
                note = ""
                if fam == "A":
                    cur, cen, masked = SourceRates(tc, tc, f), SourceRates(tn, tn, f + 0.05), f + 0.05
                elif fam == "R":
                    cur, cen, masked = SourceRates(tc, tc, f + 0.05), SourceRates(tn, tn, f + 0.10), f
                else:
                    ab = c4_other_wrong(f, f + 0.05, tc, tn, share)
                    if ab is None:
                        out.append(Cell(name, None, BANK_STUDIED, "infeasible: no other-wrong recalls reach Delta = "
                                                                  "0.01 in both comparisons with C2 at its SESOI"))
                        continue
                    cur, cen, masked = SourceRates(tc, ab[0], f), SourceRates(tn, ab[1], f + 0.05), f + 0.05
                    note = f"other-wrong recall curated {ab[0]:.4f}, census {ab[1]:.4f}"
                bank = Bank(structure=structure, curated=cur, census=cen,
                            masked=tuple(SourceRates(MASKED_RECALL, MASKED_RECALL, masked) for _ in range(3)),
                            icc=BANK_ICC)
                out.append(Cell(name, Scenario(f"power {name}", c1=tuple(c1), c5=tuple(c5), bank=bank),
                                BANK_STUDIED, note))
    return out


DEV_BANK_CELLS = ("dev bank as observed", "dev bank baselines at the SESOI")


def dev_bank_cells(structure: BankStructure, c1: Sequence[float] = OTHER_C1, c5: Sequence[float] = OTHER_C5) -> list[Cell]:
    """The dev bank's rates (calibration.DEV_BANK: the dev bank replay's per-source means, type-wrong recall,
    other-wrong recall and f), the final-bank power's primary column (science, design log, 21 Oct 10:54): every source as
    observed on dev; and grid cell A's construction at curated's dev f and census's dev type-wrong recall (curated
    better by the SESOI on both of C2's components, the masked drafts at f + 0.05, recall MASKED_RECALL). Bank ICC
    BANK_ICC; the claims not studied at the calibration's alternatives."""
    dev = cal.DEV_BANK
    observed = Bank(structure=structure, curated=dev["curated"], census=dev["census"],
                    masked=tuple(dev[s] for s in ("generated-dc-1", "generated-dc-2", "generated-dc-3")), icc=BANK_ICC)
    f, r = dev["curated"].fpr, dev["census"].type_wrong
    tc, tn = r + SESOI["C2"], r
    sesoi = Bank(structure=structure, curated=SourceRates(tc, tc, f), census=SourceRates(tn, tn, f + 0.05),
                 masked=tuple(SourceRates(MASKED_RECALL, MASKED_RECALL, f + 0.05) for _ in range(3)), icc=BANK_ICC)
    return [Cell(name, Scenario(f"power {name}", c1=tuple(c1), c5=tuple(c5), bank=bank), BANK_STUDIED, note)
            for name, bank, note in ((DEV_BANK_CELLS[0], observed, "every source at its dev rates"),
                                     (DEV_BANK_CELLS[1], sesoi, f"grid cell A at f = {f:.4f}, r = {r:.4f} (dev)"))]


def reference_bank(structure: BankStructure) -> Bank:
    """The bank of the agent cells (C1, C5): cell A at f = 0.05, r = 0.5 (the bank claims are not studied there)."""
    return next(c.scenario.bank for c in bank_cells(structure) if c.name == "A f=0.05 r=0.5")


# --- C1 from the dev grid's prompted phase -------------------------------------------------------------------------


FALLBACK_RULE = ("c1-fallback/1 (science, design log: Gemma 20 Oct 19:33Z, DeepSeek V4.1 21 Oct 03:22Z, GLM 21 Oct "
                 "08:10Z): a model named on the command line, never one inferred from outcomes, whose dev prompted phase "
                 "is incomplete at the freeze gets as its C1 planning input its completed prompted records (GLM: those "
                 "ended before its cutoff, 05:45Z) plus its natural rate, stated: each prompted arm's accuracy and ICC "
                 "from its completed records alone, its natural-cohort dev accuracy beside them, and an arm without a "
                 "completed record takes the natural accuracy in its place, marked so")


def _arm_rows(qs: Mapping[str, Mapping[int, bool]], n_reps: int) -> dict[str, Any]:
    means = [float(np.mean(list(r.values()))) for r in qs.values()]
    full = np.array([[float(r[k]) for k in sorted(r)] for r in qs.values() if len(r) == n_reps])
    return {"accuracy": float(np.mean(means)), "questions": len(qs), "attempts": sum(len(r) for r in qs.values()),
            "icc": icc_estimate(full) if len(full) > 1 else None}


def c1_dev_rates(attempts: Iterable[Attempt], models: Sequence[str], n_reps: int,
                 arms: Sequence[str] = ("curated", "census"), split: str = "dev", fallback: Sequence[str] = (),
                 natural_cohort: str = "none") -> dict[str, Any]:
    """Per model and arm: the dev accuracy (each base question's mean over its repetitions, then the mean over base
    questions), the attempts and base questions it counts, and the ICC of the questions with all ``n_reps``
    repetitions. A missing attempt counts wrong (prereg §4.3). Refused: a model or arm with no attempt, or an attempt
    of another split. A model in ``fallback`` (FALLBACK_RULE; given, never inferred) is read from the attempts it has,
    its natural cohort's accuracy (the same estimator) stated beside and taken for an arm with none; it must be a model
    of P with a natural dev attempt."""
    stray = [m for m in fallback if m not in models]
    if stray:
        raise PowerInputError(f"{stray}: a fallback model that is not a model of P {list(models)}")
    cells: dict[tuple[str, str], dict[str, dict[int, bool]]] = defaultdict(lambda: defaultdict(dict))
    for a in attempts:
        read = a.model in models and (a.cohort in arms or (a.model in fallback and a.cohort == natural_cohort))
        if not read:
            continue
        if a.split != split:
            raise PowerInputError(f"{a.run_id}: split {a.split!r}, not {split!r}: dev rates read the dev split only")
        cells[(a.model, a.cohort)][a.base_question_id][a.repetition] = bool(a.correct)
    out: dict[str, Any] = {"schema": DEV_RATES_SCHEMA, "split": split, "arms": list(arms), "n_reps": n_reps,
                           "models": {}}
    if fallback:
        out["fallback"] = {"rule": FALLBACK_RULE, "natural_cohort": natural_cohort, "models": {}}
    for m in models:
        row, natural = {}, None
        if m in fallback:
            nq = cells.get((m, natural_cohort))
            if not nq:
                raise PowerInputError(f"{m}: no natural dev attempt for the fallback rule's natural rate")
            nat = _arm_rows(nq, n_reps)
            natural = {"accuracy": nat["accuracy"], "questions": nat["questions"], "attempts": nat["attempts"]}
        for arm in arms:
            qs = cells.get((m, arm))
            if qs:
                row[arm] = _arm_rows(qs, n_reps)
            elif natural is not None:
                row[arm] = {"accuracy": natural["accuracy"], "questions": 0, "attempts": 0, "icc": None,
                            "from": "natural rate (no completed prompted record in this arm)"}
            else:
                raise PowerInputError(f"{m}: no {split} attempt in arm {arm}")
        row["difference"] = row[arms[0]]["accuracy"] - row[arms[-1]]["accuracy"]
        out["models"][m] = row
        if natural is not None:
            out["fallback"]["models"][m] = {
                "natural": natural,
                "arms": {arm: ("natural rate" if "from" in row[arm] else "completed prompted records") for arm in arms},
                "prompted_attempts": {arm: row[arm]["attempts"] for arm in arms}}
    return out


def utc_instant(value: str, what: str) -> datetime:
    """An ISO time with its zone, as a UTC instant; refused without a zone (a zone-free time compares with nothing)."""
    t = datetime.fromisoformat(value)
    if t.tzinfo is None:
        raise PowerInputError(f"{what} {value!r} has no time zone: give it with a time zone (e.g. Z)")
    return t.astimezone(UTC)


def fallback_records(records: Iterable[Mapping[str, Any]], cohorts: Mapping[str, str],
                     fallback: Mapping[str, str | None], arms: Sequence[str] = ("curated", "census")
                     ) -> tuple[list[Mapping[str, Any]], dict[str, dict[str, dict[str, int]]]]:
    """The records the fallback rule reads: every record but a fallback model's prompted-arm records that are not
    completed: those that failed infrastructure (the review of 0fa4fa4), and, with a cutoff (``fallback``: model -> ISO
    cutoff with its zone, or None; GLM: 05:45Z), those that did not end before it, compared as UTC instants (a record
    without an end time is not completed before a cutoff; one without a zone is refused). Returns the kept records and,
    per fallback model, how many were left out of each arm, after the cutoff and for infrastructure."""
    cut = {m: (None if c is None else utc_instant(c, f"{m}'s cutoff")) for m, c in fallback.items()}
    kept: list[Mapping[str, Any]] = []
    dropped = {m: {"after_cutoff": dict.fromkeys(arms, 0), "infrastructure": dict.fromkeys(arms, 0)} for m in fallback}
    for r in records:
        m, arm = r.get("model"), cohorts.get(r.get("run_id"))
        if m in cut and arm in arms:
            if (r.get("outcome") or {}).get("failure_type") == "infrastructure":
                dropped[m]["infrastructure"][arm] += 1
                continue
            if cut[m] is not None:
                end = r.get("ended_at")
                if end is None or utc_instant(end, f"{r.get('run_id')}'s end time") >= cut[m]:
                    dropped[m]["after_cutoff"][arm] += 1
                    continue
        kept.append(r)
    return kept, dropped


def c1_scenario(rates: Mapping[str, Any], bank: Bank, margin: float, c5: Sequence[float] = OTHER_C5,
                icc_range: tuple[float, float] = (0.01, 0.95)) -> Scenario:
    """C1 at the margin from the dev rates: accuracy the equal-weight mean over P of the census arm's dev accuracy, ICC
    the mean of the per-model ICCs of both arms (clipped to ``icc_range``: the generator needs one in (0, 1)), every
    model's curated - census difference ``margin``. Refused when |P| is not the simulation's seven models."""
    models = rates["models"]
    if len(models) != len(cal.PANEL):
        raise PowerInputError(f"the dev rates give {len(models)} models; the simulation's P has {len(cal.PANEL)}")
    acc = float(np.mean([r["census"]["accuracy"] for r in models.values()]))
    iccs = [r[a]["icc"] for r in models.values() for a in ("curated", "census") if r[a]["icc"] is not None]
    icc = float(np.clip(np.mean(iccs), *icc_range)) if iccs else 0.3
    sc = Scenario(f"C1 dev rates, margin {margin:g}", accuracy=acc, icc=icc, c1=(margin,) * len(cal.PANEL),
                  c5=tuple(c5), bank=bank)
    params(sc, cal.simulation_config())  # refuses an infeasible scenario
    return sc


# --- C5 from the entry-77 replay -----------------------------------------------------------------------------------


def _count(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def read_c5_inputs(path: Path, panel: Sequence[str], graph: str) -> dict[str, Any]:
    """The entry-77 replay's counts (``c5-power-inputs/1``) for the models of P on the natural cohort's dev outputs:
    {"schema", "graph", "split": "dev", "source": "curated", "models": {route model: {"outputs", "correct", "flagged",
    "flagged_wrong"}}, "provenance": str}, optionally "curated_fix" and "corrupt" (retry rates). Refused: another
    schema, graph, split or source, a model set other than P, counts that cannot hold."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != C5_SCHEMA:
        raise PowerInputError(f"{path}: not a {C5_SCHEMA} file")
    if data.get("split") != "dev":
        raise PowerInputError(f"{path}: split {data.get('split')!r}: C5's power reads dev outputs only, never test")
    if data.get("graph") != graph:
        raise PowerInputError(f"{path}: graph {data.get('graph')!r}, not {graph!r}")
    if data.get("source") != "curated":
        raise PowerInputError(f"{path}: source {data.get('source')!r}: C5's flags are the curated checks' (F = 1)")
    if not isinstance(data.get("provenance"), str) or not data["provenance"].strip():
        raise PowerInputError(f"{path}: no provenance (which replay, commit and store)")
    models = data.get("models")
    if not isinstance(models, dict) or set(models) != set(panel):
        raise PowerInputError(f"{path}: models {sorted(models or {})} are not P {sorted(panel)}")
    for m, c in models.items():
        keys = ("outputs", "correct", "flagged", "flagged_wrong")
        if not isinstance(c, dict) or not all(_count(c.get(k)) for k in keys):
            raise PowerInputError(f"{path}: {m} needs non-negative integer {', '.join(keys)}")
        # every implied count is a count: flagged wrong <= wrong outputs, flagged correct (flagged - flagged wrong)
        # <= correct outputs, so flagged <= outputs and correct <= outputs follow but are checked first
        if not (0 < c["outputs"] and c["correct"] <= c["outputs"] and c["flagged"] <= c["outputs"]
                and c["flagged_wrong"] <= c["flagged"] and c["flagged_wrong"] <= c["outputs"] - c["correct"]
                and c["flagged"] - c["flagged_wrong"] <= c["correct"]):
            raise PowerInputError(f"{path}: {m}'s counts cannot hold together ({c})")
    for k in ("curated_fix", "corrupt"):
        if k in data and not (isinstance(data[k], int | float) and not isinstance(data[k], bool) and 0 <= data[k] <= 1):
            raise PowerInputError(f"{path}: {k} must be a number in [0, 1]")
    return data


PLANNING_SCHEMA = "c5-planning-inputs/1"  # the entry-77 offline replay's output (results/natural-checker-replay-*)


def c5_inputs_from_planning(doc: Mapping[str, Any], panel: Sequence[str], graph: str) -> dict[str, Any]:
    """The entry-77 replay's per-model counts (``c5-planning-inputs/1``: per model F, N_att, N_out, C, W, W_F, C_F)
    as C5_SCHEMA's: outputs = N_att (C5's opportunities are flagged outputs per attempt; an attempt without a scored
    output is wrong), correct = C, flagged = F, flagged_wrong = W_F; P's models only. Refused, never loosened: another
    schema or source, a description that is not the dev split, a model set other than P, and for any model of P a role
    outside must and panel, an incomplete or partly recorded design, F != W_F + C_F, N_out != C + W, or N_out > N_att."""
    if doc.get("schema") != PLANNING_SCHEMA:
        raise PowerInputError(f"not a {PLANNING_SCHEMA} file (schema {doc.get('schema')!r})")
    if doc.get("source") != "curated":
        raise PowerInputError(f"source {doc.get('source')!r}: C5's flags are the curated checks'")
    if not str(doc.get("what", "")).startswith("dev split"):
        raise PowerInputError(f"{doc.get('what')!r} is not the dev split: C5's power reads dev outputs only")
    if set(doc.get("P_models") or ()) != set(panel):
        raise PowerInputError(f"its P {sorted(doc.get('P_models') or ())} is not the config's P {sorted(panel)}")
    models = {}
    for m in panel:
        c = (doc.get("per_model") or {}).get(m)
        if not isinstance(c, dict):
            raise PowerInputError(f"{m}: no counts")
        keys = ("N_att", "N_out", "C", "W", "F", "W_F", "C_F")
        if not all(_count(c.get(k)) for k in keys):
            raise PowerInputError(f"{m}: {', '.join(keys)} must be non-negative integers")
        if c.get("role") not in ("must", "panel"):
            raise PowerInputError(f"{m}: role {c.get('role')!r} is not P's")
        if c.get("complete") is not True or c.get("recorded") != c.get("planned") or c["N_att"] != c.get("recorded"):
            raise PowerInputError(f"{m}: not a complete design (complete {c.get('complete')}, recorded "
                                  f"{c.get('recorded')} of {c.get('planned')}, N_att {c['N_att']})")
        if c["F"] != c["W_F"] + c["C_F"]:
            raise PowerInputError(f"{m}: F = W_F + C_F fails ({c['F']} != {c['W_F']} + {c['C_F']})")
        if c["N_out"] != c["C"] + c["W"] or c["N_out"] > c["N_att"]:
            raise PowerInputError(f"{m}: N_out = C + W <= N_att fails ({c['N_out']}, {c['C']} + {c['W']}, {c['N_att']})")
        models[m] = {"outputs": c["N_att"], "correct": c["C"], "flagged": c["F"], "flagged_wrong": c["W_F"]}
    return {"schema": C5_SCHEMA, "graph": graph, "split": "dev", "source": "curated", "models": models,
            "provenance": f"{doc.get('provenance')} [adapted from {PLANNING_SCHEMA} by weighted_power."
                          f"c5_inputs_from_planning: outputs = N_att, correct = C, flagged = F, flagged_wrong = W_F]"}


def c5_scenario(inputs: Mapping[str, Any], panel: Sequence[str], bank: Bank, sesoi: float = SESOI["C5"],
                c1: Sequence[float] = OTHER_C1) -> Scenario:
    """C5 at its SESOI for every model of P from the replay's counts: accuracy and flag precision pooled by count over
    P (correct outputs over attempts, wrong flagged outputs over flagged ones: the simulation has one of each), each
    model's curated flag rate its own (in the simulation's panel order: P's two must models first, Qwen then Sonnet as in
    calibration.PANEL, then the others in P's order). Refused when the generator cannot reach a model's rate at the
    pooled accuracy and precision."""
    models = inputs["models"]
    if len(panel) != len(cal.PANEL):
        raise PowerInputError(f"P has {len(panel)} models; the simulation's has {len(cal.PANEL)}")
    must = [m for m in cal.PANEL[:2] if m in panel]
    if len(must) != 2:
        raise PowerInputError(f"P must hold the two must models {cal.PANEL[:2]}")
    order = [*must, *(m for m in panel if m not in must)]
    acc = sum(models[m]["correct"] for m in order) / sum(models[m]["outputs"] for m in order)
    flagged = sum(models[m]["flagged"] for m in order)
    if flagged == 0:
        raise PowerInputError("no flagged output in any model: C5 has no estimable effect")
    precision = sum(models[m]["flagged_wrong"] for m in order) / flagged
    rates = tuple(models[m]["flagged"] / models[m]["outputs"] for m in order)
    kw = {k: float(inputs[k]) for k in ("curated_fix", "corrupt") if k in inputs}
    sc = Scenario(f"C5 entry-77 dev counts, effect {sesoi:g}", accuracy=acc, flag_precision=precision,
                  panel_flag_rates=rates, c1=tuple(c1), c5=(sesoi,) * len(order), bank=bank, **kw)
    try:
        params(sc, cal.simulation_config())
    except ValueError as exc:
        raise PowerInputError(f"the replay's counts give no feasible C5 scenario: {exc}") from None
    return replace(sc, name=f"{sc.name} ({', '.join(order)})")
