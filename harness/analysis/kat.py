"""Known answers of the inference code (design log): a fixed synthetic input set run through the registered inference by
the entry points the confirmatory analysis uses, every output serialised exactly (floats by their repr, keys sorted)
and hashed.

Scope. The known-answer test (KAT) detects accidental drift in what the inference code DOES on fixed inputs: an edited
function, a changed dependency (numpy, its BLAS, the maths library), or a name rebound before the run (a wrapper that
keeps the wrapped function's name, a generated method, a module attribute swapped, a builtin replaced), which the two
source digests (inference.py) do not see while no registered source changes. It does not show that the code behaves as
calibrated on other inputs: code that answers correctly on the known inputs and differently on real data needs a
deliberate change committed to the tree, which the freeze tag and the whole-tree code digest every runner checks
(pre-registration change-log entry 55) cover.

Two answers, each the sha256 of its payload's canonical JSON (``known_answers``):
- calibrated (``calibrated_kat_sha256`` in the config): the calibrated inference paths with every question its own
  cluster, where the entry-52 dispatcher is stats.studentise exactly: the six contrasts (C1a two-family and pooled, C1b,
  C2, C3, C4, C5) through clustered_contrasts, with their bootstrap-t and CR2-Satterthwaite p's and bounds, the
  calibration table's loader and alpha' application, Holm, the per-question rates and f_q weighting, and the numerical
  primitives (the resampling schemes, studentise, t_sf and betainc on a grid). The same payload computed through
  run_contrasts on the question-resample matrix is the calibration's path, which the calibrated commit 87c7126 also
  has: tests/test_analysis_kat.py runs this driver on 87c7126's own modules and the hashes are equal, so the value is
  frozen now.
- grouping (``grouping_kat_sha256``, [FILL] until the freeze commit): the cluster grouping added after the calibration,
  on a map that links questions (a chain of shared anchors within a template; an anchor shared across templates, which
  links nothing): the map's loader and self-check, the clusters and their resampling, and the contrasts on them.

The inputs include the cases the registered rules turn on: resamples drawing a single question (zero spread: t* is
undefined and never supports a claim), a statistic without spread, a C5 model with no flagged output and questions
without type-wrong items (empty denominators: p = 1), a family whose step-2 FWER check failed (every p = 1), questions
with and without rewrites, f = f_gold alone, missing attempts (the adverse analyses), and a bank with more questions
than the agent runs. They come from sha256 in counter mode over ``KAT_SEED`` (no random-number library), small enough to
run in about a second.

prereg.preflight recomputes both answers in the confirmatory run's own process, after every import, before it reads
any data, and refuses on a mismatch, naming what in ``environment()`` differs from the config's ``kat_environment``
(where the answers were computed), or that nothing does (then the code's behaviour changed).

The host (change-log entry 73). The answers are exact, so another machine's BLAS kernel or SIMD paths, or another
thread count, may move their last bits. The confirmatory analysis runs on one pinned host, the one ``kat_environment``
records (its CPU model and vector extensions, the BLAS kernel OpenBLAS chose there), single-threaded: the preflight
refuses unless OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1 (set before the process starts, so that OpenBLAS starts
with one thread; where OpenBLAS reports its thread count, it must be 1 too). If the host must change, the calibrated
answer is recomputed on the new host from 87c7126's own functions (a detached worktree, as
tests/test_analysis_kat.py does) and the current tree's, and the config records both hosts' values and environments;
a frozen value is never edited to match a host. The driver is in the grouping registry. Its calibrated part on the calibration's path (``clustered``
False) reads only names the calibrated commit has; the grouping objects are read from prereg only when used.
"""

from __future__ import annotations

import ctypes
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from harness.analysis import prereg, stats

KAT_SCHEMA = "inference-kat/1"
KAT_SEED = "www-2027 design log: the inference's known answers"
KAT_N_BOOT = 199
KAT_REPS = 3
KAT_MODELS = ("kat-model-a", "kat-model-b")  # C1's two families
KAT_PANEL = ("kat-model-a", "kat-model-b", "kat-model-c")  # C5's P (and the pooled C1a's), two models in one family
KAT_PANEL_FAMILIES = ("family-a", "family-b", "family-a")
KAT_AGENT_QUESTIONS = 12
KAT_BANK_QUESTIONS = 15  # a bank-only extension (entry 41): three questions the agents do not run
# per prompt arm (model, cohort): the base accuracy; per checker arm: the shares fixed and corrupted; per bank source:
# its flag rates on type-wrong, other wrong, untouched gold, rewritten and answer-neutral items; per panel model: its
# flag rate and the feedback and generic retries' success
KAT_ACCURACY = {("kat-model-a", "curated"): 0.66, ("kat-model-a", "census"): 0.48, ("kat-model-b", "curated"): 0.57,
                ("kat-model-b", "census"): 0.53, ("kat-model-c", "curated"): 0.61, ("kat-model-c", "census"): 0.47}
KAT_RETRY = {"union": (0.45, 0.10), "curated": (0.35, 0.12), "census": (0.30, 0.08)}
KAT_BANK_RATES = {"curated": (0.70, 0.45, 0.10, 0.20, 0.10), "census": (0.40, 0.50, 0.30, 0.35, 0.20),
                  "generated-masked-1": (0.55, 0.40, 0.20, 0.25, 0.10),
                  "generated-masked-2": (0.50, 0.45, 0.15, 0.30, 0.15),
                  "generated-masked-3": (0.60, 0.35, 0.25, 0.20, 0.05)}
KAT_C5 = {"kat-model-a": (0.45, 0.60, 0.35), "kat-model-b": (0.35, 0.50, 0.40), "kat-model-c": (0.40, 0.55, 0.30)}
# the masked draws' names in the KAT's inputs (KAT_BANK_RATES' keys; the calibrated answer's), kept when the registered
# names became generated-dc-k (spec trace B15): inputs of the known answer, not the paper's names
KAT_MASKED = ("generated-masked-1", "generated-masked-2", "generated-masked-3")
KAT_MISSING = 0.08  # each attempt's chance of being missing (and of a lost generic retry)
# the hand-built calibration table: per family and claim, the test chosen from step 1 (None: none passed) and the
# factor its alpha'(a) is of each Holm level a = 0.05 / k (generous for some claims, so that several enter Holm below 1)
KAT_TABLE = {"two-family": {"C1a": ("boot_t", 1.6), "C1b": ("cr2", 0.9), "C2": ("boot_t", 6.0), "C3": (None, 1.0),
                            "C4": ("boot_t", 4.0), "C5": ("cr2", 1.2)},
             "pooled C1a": {"C1a pooled": ("boot_t", 5.0), "C1b": ("boot_t", 0.8), "C2": ("cr2", 2.0),
                            "C3": ("boot_t", 3.0), "C4": (None, 1.0), "C5": ("boot_t", 1.5)}}
# step 1's alpha' (results/calibration-step1-20261019/step1.json: the two-family family, the bootstrap-t for every claim,
# factor 1; alpha'(0.05 / k) for k = 1..6), and six raw p's whose smallest, C3's, is above its alpha' at k = 6 and 5 and
# below it at k = 4, while C2's passes its alpha' at k = 6 (design log: Holm orders the calibrated p's, so C3 cannot
# block C2)
KAT_STEP1 = {"C1a": (0.057375, 0.0235875, 0.016191666666666667, 0.012993750000000002, 0.009195, 0.0077958333333333334),
             "C1b": (0.09187500000000001, 0.05, 0.03333333333333333, 0.025, 0.02, 0.016666666666666666),
             "C2": (0.040675, 0.017087500000000002, 0.010091666666666667, 0.006893750000000001, 0.004695,
                    0.0035958333333333333),
             "C3": (0.016975, 0.0027875, 0.0007916666666666666, 0.00039375, 0.000195, 0.00019166666666666667),
             "C4": (0.047775000000000005, 0.0241875, 0.016091666666666667, 0.01229375, 0.009595000000000001,
                    0.0077958333333333334),
             "C5": (0.047575000000000006, 0.0241875, 0.015291666666666667, 0.011193750000000002, 0.008595,
                    0.006695833333333333)}
KAT_STEP1_P = {"C1a": 0.5, "C1b": 0.9, "C2": 0.003, "C3": 0.0002, "C4": 0.3, "C5": 0.2}
KAT_WEIGHTED_STEP1 = {  # step 1's alpha' at the two-family's reachable levels (bootstrap-t), by s ascending
    "C1a": (0.057375, 0.0235875, 0.016191666666666667, 0.012993750000000002, 0.009195, 0.0077958333333333334,
            0.006396428571428573, 0.005796875, 0.005197222222222222, 0.0045975, 0.004197727272727273,
            0.003997916666666667, 0.0035980769230769235, 0.003398214285714286, 0.0031983333333333334,
            0.0029984375000000002),
    "C1b": (0.09187500000000001, 0.05, 0.03333333333333333, 0.025, 0.02, 0.016666666666666666, 0.014285714285714287,
            0.0125, 0.011111111111111112, 0.01, 0.009090909090909092, 0.008333333333333333, 0.007692307692307693,
            0.0071428571428571435, 0.006666666666666667, 0.00625),
    "C2": (0.040675, 0.03008, 0.024883333333333334, 0.020485714285714288, 0.017087500000000002, 0.010091666666666667,
           0.009092307692307693, 0.008192857142857144, 0.007693333333333333, 0.006893750000000001),
    "C3": (0.016975, 0.012977777777777777, 0.01038, 0.008181818181818182, 0.006383333333333334, 0.004984615384615385,
           0.004185714285714286, 0.0033866666666666667, 0.0027875),
    "C4": (0.047775000000000005, 0.0241875, 0.016091666666666667, 0.01229375, 0.009595000000000001,
           0.0077958333333333334, 0.0068964285714285724, 0.006096875000000001, 0.0053972222222222225, 0.0045975,
           0.004297727272727273, 0.0037979166666666665, 0.0033980769230769234, 0.003098214285714286,
           0.0027983333333333337, 0.0024984375000000002),
    "C5": (0.047575000000000006, 0.0241875, 0.015291666666666667, 0.011193750000000002, 0.008595, 0.006695833333333333,
           0.005596428571428572, 0.0046968750000000005, 0.004197222222222223, 0.0036975000000000003,
           0.003197727272727273, 0.002997916666666667, 0.0022980769230769235, 0.002098214285714286,
           0.0019983333333333337, 0.0018984375000000002)}
# fixed raw p's on it: C3 alone (0.0025 under its alpha'(s = 16), p* = 0.025, the first threshold exactly); C2 and C3
# (C3's raw p the smallest, both p* at their first level, a tie in ratio 0.003125: C2 first in family order)
KAT_WEIGHTED_P = {"C3 alone": {"C1a": 0.5, "C1b": 0.9, "C2": 0.3, "C3": 0.0025, "C4": 0.2, "C5": 0.02},
                  "C2 and C3": {"C1a": 0.5, "C1b": 0.9, "C2": 0.003, "C3": 0.0002, "C4": 0.2, "C5": 0.02}}
# the weighted Holm's hand table: per family and claim its test and the factor of its alpha' over each level
KAT_WEIGHTED_TABLE = {"two-family": {"C1a": ("boot_t", 2.0), "C1b": ("cr2", 1.0), "C2": ("boot_t", 6.0),
                                     "C3": ("boot_t", 1.5), "C4": (None, 1.0), "C5": ("cr2", 1.2)},
                      "pooled C1a": {"C1a pooled": ("boot_t", 5.0), "C1b": ("boot_t", 0.8), "C2": ("cr2", 2.0),
                                     "C3": ("boot_t", 3.0), "C4": ("boot_t", 1.0), "C5": ("boot_t", 1.5)}}
# the step-down on hand rows (weights 1, 1, 4, 8, 1, 1): C3 alone on its threshold; C3 needing a claim out first; C2
# then C3; a tie in family order; C2 first and above its threshold; every claim at its first level; C1a's p* on its
# level at s = 14 (C1b and C4 out first), where alpha * (w / s) is one ulp below alpha * w / s
# the frozen tables' refusals: (file, the path of a setting in it, its doctored value; "<drop>" deletes it)
KAT_DOCTORED = {"not an alpha' table": ("alpha", ("schema",), "weighted-holm-alpha/0"),
                "other rules": ("alpha", ("rules",), {}), "another alpha": ("alpha", ("alpha",), 0.1),
                "a test off the list": ("alpha", ("families", "two-family", "claims", "C3", "test"), "sign-flip"),
                "an alpha' off the levels": ("alpha", ("families", "two-family", "claims", "C3", "alpha_prime", "12"),
                                             "<drop>"),
                "an alpha' falling with the level": ("alpha", ("families", "two-family", "claims", "C3", "alpha_prime",
                                                               "16"), 0.5),
                "no FWER check recorded": ("usable", ("families", "two-family", "fwer_pass"), "<drop>"),
                "an unreachable usable level": ("usable", ("families", "two-family", "usable", "C3"), [3])}
# the producers' inputs (entries 74, 76, 91): a step-1 fingerprint, the calibration configurations with their null
# claims, the procedure digest the hand addenda record, their date; the producers' fixed p's (KAT_PRODUCER_N datasets
# per configuration, uniforms) and the chunk size they are stored in
KAT_FINGERPRINT = "kat-step1-fingerprint"
KAT_CONFIGURATIONS = (("kat-joint-null", ("C1a", "C1a pooled", "C1b", "C2", "C3", "C4", "C5")), ("kat-C3-null", ("C3",)),
                      ("kat-alternative", ()))
KAT_PROCEDURE = "kat-multiplicity-digest"
KAT_NO_OUTPUTS = "/dev/null/kat-step-2-outputs"  # a directory on no host (under a file), and the same in every run
KAT_DATE = "2026-10-26"
KAT_PRODUCER_N = 240
KAT_PRODUCER_CHUNK = 100
# step 3 (entry 106): the KAT's inference digest and fingerprint, its validation outputs on no host, and its date
KAT_INFERENCE = "kat-inference-digest"
KAT_STEP3_FINGERPRINT = "kat-step3-fingerprint"
KAT_NO_STEP3_OUTPUTS = "/dev/null/kat-step-3-outputs"
KAT_STEP3_DATE = "2026-10-27"
KAT_HOLM_WEIGHTS = (1, 1, 4, 8, 1, 1)
KAT_HOLM_ROWS = ((1.0, 1.0, 1.0, 0.025, 1.0, 1.0), (1.0, 1.0, 1.0, 0.4 / 15, 1.0, 1.0),
                 (1.0, 1.0, 0.0125, 0.4 / 12, 1.0, 1.0), (1.0, 1.0, 1.0, 0.025, 1.0, 0.05 / 16),
                 (1.0, 1.0, 0.2 / 13, 0.4 / 12, 1.0, 1.0), (0.003125, 0.003125, 0.0125, 0.025, 0.003125, 0.003125),
                 (0.05 * 1 / 14, 0.003125, 1.0, 1.0, 0.003125, 1.0))
# C3's equivalence reading (entry 85) on hand claim-level intervals: (estimate, lo, hi), calibrated; inside the margin, on
# it, outside it, an unbounded end, not estimable, and an uncertified interval inside it
KAT_C3_EQUIVALENCE = (((0.0, -0.04, 0.04), True), ((0.0, -0.05, 0.05), True), ((0.03, 0.0, 0.06), True),
                      ((0.0, None, 0.01), True), ((None, None, None), True), ((0.0, -0.01, 0.01), False))
# the grouping's map: declared anchors per question (the others none): kq00-kq03-kq06 a chain within template tA,
# kq01-kq04 and kq07-kq13 pairs within tB; kq09 (tA) and kq01 (tB), kq12 (tA) and kq02 (tC) share an anchor across
# templates, which links nothing
KAT_ANCHORS = {"kq00": ("x1",), "kq03": ("x1", "x2"), "kq06": ("x2",), "kq01": ("z", "y1"), "kq04": ("y1",),
               "kq09": ("z",), "kq07": ("y2",), "kq13": ("y2",), "kq12": ("w",), "kq02": ("w",)}
# the primitives' grid: the incomplete beta function at (a, b, x) on both sides of the continued fraction's switch and at
# the ends, and the t tail and quantile at small and large degrees of freedom
KAT_BETA = ((1.0, 1.0), (2.0, 3.0), (0.5, 0.5), (6.5, 0.5), (50.0, 0.5), (0.5, 7.0))
KAT_X = (-0.1, 0.0, 1e-9, 0.05, 0.2, 0.4, 0.5, 0.6, 0.8, 0.95, 1.0 - 1e-9, 1.0, 1.2)
KAT_T = (-30.0, -3.0, -1.0, -0.25, 0.0, 0.25, 1.0, 2.5, 4.0, 30.0)
KAT_DF = (1.0, 2.0, 5.0, 11.0, 30.0, 300.0)
KAT_Q = (0.9, 0.975, 0.995, 1 - 0.05 / 12)


def _uniforms(tag: str, questions: tuple[str, ...], width: int) -> np.ndarray:
    """(questions x width) uniforms in [0, 1): the first 56 bits of sha256(seed, tag, question, column)."""
    return np.array([[int.from_bytes(hashlib.sha256(f"{KAT_SEED}|{tag}|{q}|{j}".encode()).digest()[:7], "big") / 2.0**56
                      for j in range(width)] for q in questions])


def _hits(tag: str, questions: tuple[str, ...], p: np.ndarray | float, n: np.ndarray | int) -> np.ndarray:
    """(questions x max n) events: column j < n_q happens with the question's probability p_q."""
    n = np.broadcast_to(np.asarray(n, dtype=np.int64), (len(questions),))
    width = int(n.max()) if len(questions) else 0
    p = np.broadcast_to(np.asarray(p, dtype=np.float64), (len(questions),))
    return (_uniforms(tag, questions, width) < p[:, None]) & (np.arange(width)[None, :] < n[:, None])


def _spread(tag: str, questions: tuple[str, ...], base: float) -> np.ndarray:
    """Per-question probabilities around ``base`` (+-0.2), so that questions differ."""
    return np.clip(base + 0.4 * (_uniforms(tag, questions, 1)[:, 0] - 0.5), 0.02, 0.98)


def kat_index(n: int = KAT_AGENT_QUESTIONS) -> stats.QuestionIndex:
    """The KAT's base questions kq00.. with strata template|tier (templates tA, tB, tC by id, two tiers)."""
    return stats.QuestionIndex.from_strata({f"kq{i:02d}": f"t{'ABC'[i % 3]}|{('easy', 'hard')[i // 3 % 2]}"
                                            for i in range(n)})


def kat_config(family: str, rewrites_adopted: bool = True) -> prereg.PreregConfig:
    """The KAT's config: its models and panel, B = KAT_N_BOOT, the registered seed, the C1a family recorded with powers
    that choose it (settings the calibrated commit has too)."""
    powers = {prereg.POOLED: 0.6, prereg.TWO_FAMILY: 0.4} if family == prereg.POOLED else {
        prereg.POOLED: 0.4, prereg.TWO_FAMILY: 0.6}
    return prereg.PreregConfig(models=KAT_MODELS, panel=KAT_PANEL, panel_families=KAT_PANEL_FAMILIES,
                               c1a_family=family, c1a_power=powers, scorer_version="kat", n_boot=KAT_N_BOOT,
                               rewrites_adopted=rewrites_adopted, masked_sources=KAT_MASKED)


def kat_vectors(index: stats.QuestionIndex, bank_index: stats.QuestionIndex | None = None) -> prereg.Vectors:
    """The question vectors the contrasts read, built directly (no data preparation): prompt, checker and bank arms, C5's
    retry vectors and the bank's union-only counts, from item-level events."""
    cfg, qs = kat_config(prereg.TWO_FAMILY), index.questions
    bqs = (bank_index or index).questions
    arms: dict = {}
    for (model, cohort), base in KAT_ACCURACY.items():
        tag = f"prompt|{model}|{cohort}"
        missing = _hits(tag + "|missing", qs, KAT_MISSING, KAT_REPS).sum(axis=1)
        correct = _hits(tag + "|correct", qs, _spread(tag, qs, base), KAT_REPS - missing).sum(axis=1)
        att = np.full(len(qs), float(KAT_REPS))
        arms[cfg.prompt(model, cohort)] = {"n": att, "att": att, "correct": correct.astype(np.float64),
                                           "missing": missing.astype(np.float64)}
    for model in KAT_MODELS:
        for source, (fix, corrupt) in KAT_RETRY.items():
            tag = f"checker|{model}|{source}"
            fixed = _hits(tag + "|fixed", qs, _spread(tag, qs, fix), KAT_REPS).sum(axis=1)
            corrupted = _hits(tag + "|corrupted", qs, corrupt, KAT_REPS - fixed).sum(axis=1)
            att = np.full(len(qs), float(KAT_REPS))
            arms[cfg.checker(model, source)] = {"n": att, "att": att, "delta": (fixed - corrupted).astype(np.float64)}
    # the bank's design, the same for every source: 2-4 answer-changing wrong items (the first tw type-wrong, some
    # questions none), one untouched gold, 0-2 rewrites (k_q), 0-1 answer-neutral items
    wrong = 2 + _hits("bank|W", bqs, 0.5, 2).sum(axis=1)
    tw = _hits("bank|tw", bqs, 0.45, wrong).sum(axis=1)
    k = _hits("bank|k", bqs, 0.45, 2).sum(axis=1)
    neutral = _hits("bank|neutral", bqs, 0.5, 1).sum(axis=1)
    flags = {}
    for source, (r_tw, r_other, f_gold, f_rewrite, f_neutral) in KAT_BANK_RATES.items():
        tag = f"bank|{source}"
        is_tw = np.arange(int(wrong.max()))[None, :] < tw[:, None]
        w = _hits(tag + "|W", bqs, 1.0, wrong) & (_uniforms(tag + "|W rate", bqs, int(wrong.max()))
                                                  < np.where(is_tw, r_tw, r_other))
        f = {"W": w, "tw": w & is_tw, "gold": _hits(tag + "|gold", bqs, f_gold, 1),
             "rewrite": _hits(tag + "|rewrite", bqs, f_rewrite, k), "neutral": _hits(tag + "|neutral", bqs, f_neutral, neutral)}
        flags[source] = f
        arms[cfg.bank(source)] = {
            "W": wrong.astype(np.float64), "tw": tw.astype(np.float64), "C_untouched": np.ones(len(bqs)),
            "C_rewrite": k.astype(np.float64), "C_neutral": neutral.astype(np.float64),
            "W_F": f["W"].sum(axis=1).astype(np.float64), "W_F_tw": f["tw"].sum(axis=1).astype(np.float64),
            "C_F_untouched": f["gold"].sum(axis=1).astype(np.float64),
            "C_F_rewrite": f["rewrite"].sum(axis=1).astype(np.float64),
            "C_F_neutral": f["neutral"].sum(axis=1).astype(np.float64)}
    union = {}
    for over, other in (("curated", "census"), ("census", "curated")):
        a, b = flags[over], flags[other]
        union[over] = {"wrong": (b["W"] & ~a["W"]).sum(axis=1).astype(np.float64),
                       "gold": (b["gold"] & ~a["gold"]).sum(axis=1).astype(np.float64),
                       "rewrites": (b["rewrite"] & ~a["rewrite"]).sum(axis=1).astype(np.float64)}
    c5 = {}
    for model, (flag, feedback, generic) in KAT_C5.items():
        tag = f"c5|{model}"
        f = _hits(tag + "|flag", qs, _spread(tag, qs, flag), KAT_REPS)
        fb = _hits(tag + "|feedback", qs, feedback, KAT_REPS) & ~_hits(tag + "|feedback lost", qs, KAT_MISSING, KAT_REPS)
        lost = _hits(tag + "|generic lost", qs, KAT_MISSING, KAT_REPS)
        gen = _hits(tag + "|generic", qs, generic, KAT_REPS) & ~lost
        num = ((f & fb).sum(axis=1) - (f & gen).sum(axis=1)).astype(np.float64)
        c5[model] = {"num": num, "den": f.sum(axis=1).astype(np.float64), "n": np.full(len(qs), float(KAT_REPS)),
                     "num_adverse": num - (f & lost).sum(axis=1)}
    return prereg.Vectors(arms, c5, union)


def _empty(V: prereg.Vectors) -> prereg.Vectors:
    """V with empty denominators: a panel model with no flagged output (C5 non-estimable) and no type-wrong items in the
    bank (C2's recall component non-estimable)."""
    c5 = dict(V.c5)
    c5[KAT_PANEL[-1]] = {k: np.zeros_like(v) if k != "n" else v for k, v in V.c5[KAT_PANEL[-1]].items()}
    arms = {key: ({**vec, "tw": np.zeros_like(vec["tw"]), "W_F_tw": np.zeros_like(vec["W_F_tw"])} if "tw" in vec else vec)
            for key, vec in V.arms.items()}
    return prereg.Vectors(arms, c5, V.bank_union)


def kat_matrix(n_units: int, cfg: prereg.PreregConfig) -> np.ndarray:
    """The registered unstratified resample matrix over ``n_units`` units, its first resample drawing only the first
    unit and its second only the last (zero spread: t* undefined there)."""
    K = stats.resample_matrix(stats.QuestionIndex(tuple(f"u{i}" for i in range(n_units)), ("all",) * n_units),
                              cfg.n_boot, seed=cfg.seed, scheme="unstratified")
    K[0], K[1] = 0.0, 0.0
    K[0, 0], K[1, n_units - 1] = float(n_units), float(n_units)
    return K


def _attempt(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """fn's result, or the exception it raised (a refusal is an answer too)."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - recorded by class and message: a refusal is an answer
        return exc


def _statistic(s: stats.Studentised) -> dict[str, Any]:
    """A studentised statistic in full, with both tests' p's at both sides and its bounds."""
    est = 0.0 if s.estimate is None else s.estimate
    out: dict[str, Any] = {"s": s, "t_star": stats._t_star(s) if s.estimate is not None else None}
    for theta0 in (0.0, est, est - 0.05):
        for alt in ("greater", "less"):
            out[f"{alt} {theta0!r}"] = {"boot_t": stats.bootstrap_t_p(s, theta0, alt), "cr2": stats.cr2_p(s, theta0, alt)}
    out["unknown alternative"] = [_attempt(stats.bootstrap_t_p, s, 0.0, "two"), _attempt(stats.cr2_p, s, 0.0, "two")]
    out["intervals"] = {f"{sided} {alpha!r}": {t: prereg._interval(s, alpha, sided, t) for t in prereg.TESTS}
                        for sided, alpha in (("two", 0.05), ("greater", 0.05 / 6), ("equivalence", 0.05))}
    out["bootstrap-t bounds"] = {f"{side} {alpha!r}": stats.bootstrap_t_bound(s, alpha, side)
                                 for side in ("lower", "upper") for alpha in (0.05, 0.05 / 6)}
    return out


def _primitives(index: stats.QuestionIndex) -> dict[str, Any]:
    """The numerical primitives on their own: the resampling schemes and the registered matrix, the question index, the
    incomplete beta function, the t tail and quantile, Holm, the calibrated p and the C1a choice."""
    small = stats.QuestionIndex.from_strata({"p0": "s1", "p1": "s1", "p2": "s1", "p3": "s2", "p4": "s2", "p5": "s3"})
    pooled = kat_config(prereg.POOLED)
    out: dict[str, Any] = {
        "schemes": {s: stats.resample_matrix(small, 7, seed=20261026, scheme=s) for s in stats.SCHEMES},
        "resamples": prereg.resamples(index, pooled),
        "unknown scheme": _attempt(stats.resample_matrix, small, 7, seed=1, scheme="other"),
        "index": {"questions": index.questions, "strata": index.strata, "len": len(index),
                  "positions": [index.position(q) for q in index.questions], "contains": ["kq05" in index, "kq99" in index]},
        "betainc scalar": [[a, b, x, stats.betainc(a, b, x)] for a, b in KAT_BETA for x in KAT_X],
        "betainc array": [[a, b, stats.betainc(a, b, np.array(KAT_X))] for a, b in KAT_BETA],
        "betacf": [[a, b, stats._betacf_scalar(a, b, 0.3), stats._betacf(a, b, np.array([0.1, 0.3, 0.45]))]
                   for a, b in KAT_BETA],
        "t_sf scalar": [[t, df, stats.t_sf(t, df)] for t in KAT_T for df in KAT_DF],
        "t_sf array": [[df, stats.t_sf(np.array(KAT_T), df)] for df in KAT_DF],
        "t quantile": [[q, df, prereg._t_quantile(q, df)] for q in KAT_Q for df in KAT_DF[::2]],
    }
    ps = (0.01, 0.04, 0.04, 0.2, 0.003, 1.0)
    out["holm"] = {"adjusted": stats.holm(ps), "levels": stats.holm_levels(ps, 0.05), "ties": stats.holm([0.02] * 4)}
    ap = {k: 0.05 / k * (1.4 if k > 2 else 1.1) for k in prereg.HOLM_DIVISORS}
    grid = (0.0, 0.005, 0.0116, 0.0117, 0.015, 0.02, 0.0233, 0.03, 0.05, 0.055, 0.06, 1.0)
    out["calibrated p"] = {"alpha_prime": ap, "p": [[p, prereg.calibrated_p(p, ap, 0.05)] for p in grid],
                           "uncalibrated": prereg.calibrated_p(0.0, None),
                           "decreasing": _attempt(prereg.calibrated_p, 0.01, {**ap, 1: 0.001})}
    out["c1a choice"] = [prereg.choose_c1a(0.5, 0.5), prereg.choose_c1a(0.4, 0.6), _attempt(prereg.choose_c1a, 1.5, 0.2),
                         _attempt(prereg.choose_c1a, True, 0.2), pooled.require_c1a_choice(),
                         _attempt(dataclasses.replace(pooled, c1a_power={
                             prereg.POOLED: 0.1, prereg.TWO_FAMILY: 0.9}).require_c1a_choice),
                         _attempt(dataclasses.replace(pooled, c1a_power={prereg.POOLED: 0.6}).require_c1a_choice),
                         _attempt(dataclasses.replace(pooled, c1a_family=prereg.PLACEHOLDER).require_c1a_choice)]
    return out


def _statistics(V: prereg.Vectors, K: Any, cfg: prereg.PreregConfig) -> dict[str, Any]:
    """Statistics through the dispatcher on the resampling units ``K``: the pooled C1a over P (more than PAIRWISE_MAX
    columns), one family's C1a with the adverse scoring (column-pair products), a statistic without spread and one with
    an empty denominator; and the per-question rates and f_q."""
    n = len(V.c5[KAT_PANEL[0]]["den"])
    const, zero, ones = np.full(n, 0.5), np.zeros(n), np.ones(n)
    out: dict[str, Any] = {
        "C1a pooled": _statistic(prereg._c1_stat(K, V, cfg, KAT_PANEL, False, False)),
        "C1a one family, adverse": _statistic(prereg._c1_stat(K, V, cfg, KAT_MODELS[:1], True, False)),
        "no spread": _statistic(prereg.studentise(K, [stats.Term(1.0, const, ones)])),
        "empty denominator": _statistic(prereg.studentise(K, [stats.Term(1.0, zero, zero)])),
        "theta": [stats._theta([stats.Term(2.0, const, ones), stats.Term(-1.0, ones, 2 * ones)]),
                  stats._theta([stats.Term(1.0, zero, zero)])],
        "C5 without the descriptive part": prereg.c5(V, K, cfg, descriptive=False),
        "C5, each model its own family": prereg.c5(V, K, dataclasses.replace(cfg, panel_families=())),
        "C4 retry": prereg.c4_retry(V, K, cfg),
        "compared arms": prereg.compared_arms(cfg),
        "arm labels": [[key.label, key.to_json()] for key in V.arms],
    }
    rates = {}
    for adopted in (True, False):
        c = dataclasses.replace(cfg, rewrites_adopted=adopted)
        bank = {s: V.arms[c.bank(s)] for s in KAT_BANK_RATES}
        rates[str(adopted)] = {s: {"rates": prereg.source_rates(v, c), "f_q": prereg._f_q(v["C_F_untouched"], v[
            "C_F_rewrite"], v["C_rewrite"], c)} for s, v in bank.items()}
        rates[str(adopted)]["design"] = prereg._bank_design(V, c)
    out["rates"] = rates
    out["share"] = prereg._share(np.array([1.0, 0.0, 2.0, 3.0]), np.array([2.0, 0.0, 4.0, 3.0]))
    out["rate without a denominator"] = prereg._rate({"a": zero, "b": zero}, "a", "b")
    differ = {**V.arms[cfg.bank("census")], "W": V.arms[cfg.bank("census")]["W"] + 1.0}
    out["designs differ"] = _attempt(prereg._bank_design, prereg.Vectors({**V.arms, cfg.bank("census"): differ},
                                                                         V.c5, V.bank_union), cfg)
    two_golds = {key: {**vec, "C_untouched": vec["C_untouched"] * 2} if "C_untouched" in vec else vec
                 for key, vec in V.arms.items()}
    out["two untouched golds"] = _attempt(prereg._bank_design, prereg.Vectors(two_golds, V.c5, V.bank_union), cfg)
    return out


def _table_bytes(families: dict[str, Any], rules: Any = None) -> bytes:
    return (json.dumps({"schema": "calibration-table/2", "rules": prereg.CALIBRATION_RULES if rules is None else rules,
                        "families": families}, indent=1, sort_keys=True) + "\n").encode("utf-8")


def _tables(tmp: Path) -> dict[str, Any]:
    """The hand-built calibration table (KAT_TABLE) and its variants written under ``tmp``, and what the loader makes of
    them: both families, a family whose step-2 FWER check failed, and the refusals."""
    families = {fam: {"fwer_pass": True, "claims": {
        claim: {"test": test, "alpha_prime": {str(k): factor * 0.05 / k for k in prereg.HOLM_DIVISORS}}
        for claim, (test, factor) in claims.items()}} for fam, claims in KAT_TABLE.items()}
    files = {"table.json": _table_bytes(families),
             "failed.json": _table_bytes({**families, prereg.TWO_FAMILY: {**families[prereg.TWO_FAMILY],
                                                                          "fwer_pass": False}}),
             "rules.json": _table_bytes(families, rules={"other": "rules"}),
             "schema.json": _table_bytes(families).replace(b"calibration-table/2", b"calibration-table/1"),
             "step1.json": _table_bytes({prereg.TWO_FAMILY: {"fwer_pass": True, "claims": {
                 c: {"test": "boot_t", "alpha_prime": {str(k): a for k, a in enumerate(KAT_STEP1[c], start=1)}}
                 for c in prereg.CONTRASTS}}}),
             "incomplete.json": _table_bytes({prereg.TWO_FAMILY: {"fwer_pass": True, "claims": {
                 c: e for c, e in families[prereg.TWO_FAMILY]["claims"].items() if c != "C5"}}})}
    sha = {}
    for name, raw in files.items():
        (tmp / name).write_bytes(raw)
        sha[name] = hashlib.sha256(raw).hexdigest()
    two, pooled = kat_config(prereg.TWO_FAMILY), kat_config(prereg.POOLED)

    def load(name: str, cfg: prereg.PreregConfig, digest: str | None = None) -> Any:
        c = dataclasses.replace(cfg, calibration=name, calibration_sha256=digest or sha[name])
        return _attempt(prereg.load_calibration, c, tmp)

    out = {"two-family": load("table.json", two), "pooled": load("table.json", pooled),
           "failed step 2": load("failed.json", two), "failed step 2, the other family": load("failed.json", pooled),
           "other rules": load("rules.json", two), "incomplete": load("incomplete.json", two),
           "not a table": load("schema.json", two),
           "other sha256": load("table.json", two, "0" * 64), "not filled in": prereg.load_calibration(two, tmp)}
    out["entries"] = {fam: {c: out[key].entry(c) for c in prereg.FAMILY_CLAIMS[fam]}
                      for fam, key in ((prereg.TWO_FAMILY, "two-family"), (prereg.POOLED, "pooled"))}
    # step 1's table on fixed raw p's: each claim's calibrated p, then Holm over those (and, for contrast, over the raw)
    out["step 1"] = load("step1.json", two)
    calibrated = [prereg.calibrated_p(KAT_STEP1_P[c], out["step 1"].entry(c)[1], 0.05) for c in prereg.CONTRASTS]
    out["step 1 on fixed p's"] = {"calibrated": calibrated, "holm": stats.holm(calibrated),
                                  "levels": stats.holm_levels(calibrated, 0.05),
                                  "raw holm": stats.holm([KAT_STEP1_P[c] for c in prereg.CONTRASTS])}
    return out


def _contrasts(V: prereg.Vectors, index: stats.QuestionIndex, cfg: prereg.PreregConfig, calibration: Any, K: Any,
               bank_index: stats.QuestionIndex | None, clusters: Any) -> Any:
    """The registered contrasts with their intervals: through clustered_contrasts on ``clusters`` (a ClusterMap; the
    confirmatory path), or, with clusters None, through run_contrasts on the question-resample matrix (the calibration's
    path, which the calibrated commit has). ``K`` is the agent questions' (or clusters') resample matrix; a bank with
    other questions resamples its own from the seed."""
    if clusters is None:
        Kb = None if bank_index is None else prereg.resamples(bank_index, cfg)
        return _attempt(prereg.run_contrasts, V, K, cfg, K_bank=Kb, calibration=calibration)
    pre = prereg.Preflight(cfg=cfg, family=cfg.c1a_family, inference_digest="", grouping_digest="", clusters=clusters,
                           K=K, calibration=calibration, confirmatory=False, config_file=None, config_sha256=None)
    out = _attempt(prereg.clustered_contrasts, V, index, bank_index, pre)
    return out if isinstance(out, Exception) else out[0]


def _family(V: prereg.Vectors, cfg: prereg.PreregConfig, calibration: Any, K: Any, K_bank: Any = None) -> Any:
    """The six claims and Holm without the intervals and descriptive parts (run_contrasts as the calibration ran it), on
    the resampling units ``K`` (a matrix or Clusters)."""
    return _attempt(prereg.run_contrasts, V, K, cfg, K_bank=K_bank, calibration=calibration, intervals=False)


def calibrated_answers(clustered: bool = True) -> dict[str, Any]:
    """The calibrated paths' answers on the KAT inputs, every question its own cluster: through clustered_contrasts on
    singleton clusters (``clustered``, the confirmatory path), else through run_contrasts on the question-resample
    matrix (the calibration's path; the calibrated commit has only this one). The two payloads are equal."""
    index, bank_index = kat_index(), kat_index(KAT_BANK_QUESTIONS)
    V, V_bank = kat_vectors(index), kat_vectors(index, bank_index)
    two, pooled = kat_config(prereg.TWO_FAMILY), kat_config(prereg.POOLED)
    K = kat_matrix(len(index), two)
    singles = prereg.ClusterMap.singletons(bank_index.questions) if clustered else None
    # the dispatcher's input, Clusters or the matrix, for the agent questions and for the bank's own
    units = singles.resampling(index, two, K) if clustered else K
    bank_units = singles.resampling(bank_index, two) if clustered else prereg.resamples(bank_index, two)
    with tempfile.TemporaryDirectory(prefix="kat-") as tmp:
        root = Path(tmp).resolve()
        tables = _tables(root)
        two_table, pooled_table = tables["two-family"], tables["pooled"]
        contrasts = {
            "two-family": _contrasts(V, index, two, two_table, K, None, singles),
            "pooled": _contrasts(V, index, pooled, pooled_table, K, None, singles),
            "f = f_gold": _family(V, dataclasses.replace(two, rewrites_adopted=False), two_table, units),
            "failed step 2": _family(V, two, tables["failed step 2"], units),
            "uncalibrated": _family(V, two, prereg.UNCALIBRATED, units),
            "empty denominators": _family(_empty(V), two, two_table, units),
            "bank-only extension": _family(V_bank, two, two_table, units, bank_units),
            "pooled on the two-family table": _family(V, pooled, two_table, units),
            "step 1's table": _family(V, two, tables["step 1"], units)}
        payload = {"schema": KAT_SCHEMA, "part": "calibrated", "primitives": _primitives(index),
                   "statistics": _statistics(V, units, two), "tables": tables, "contrasts": contrasts}
        return _canon(payload, str(root))


def _map_bytes(index: stats.QuestionIndex, anchors: dict[str, tuple[str, ...]], clusters: dict[str, str]) -> bytes:
    """A cluster-map/2 file's bytes (prereg.write_cluster_map's format): per question its template, anchors, cluster."""
    templates = {q: s.split("|")[0] for q, s in zip(index.questions, index.strata)}
    entries = {q: {"template": templates[q], "anchors": sorted(anchors.get(q, ())), "cluster": clusters[q]}
               for q in sorted(index.questions)}
    return (json.dumps({"schema": prereg.CLUSTER_MAP_SCHEMA, "questions": entries}, indent=1) + "\n").encode("utf-8")


def _ledger_names(cfg: prereg.PreregConfig) -> dict[str, Any]:
    """The ledger's source names mapped to the config's arms (pipeline.ledger_source_names; the review of 9d9c88c): the
    checker loop's census-t5 and generic_retry/curated rows, every other name kept; its refusals (rows under a name and
    the name it maps to, a config naming other arms than the mapping's). And the dev grid's generic_retry/any rows
    (design log entry 91, spec trace B10), mapped to the generic arm only once their records show a checker-loop/2 generic
    cell that decided on its arm's curated sources (pipeline.curated_generic_rows), with its refusals (a row not so
    verified, a checker-loop/1 record, a decision that read another source, both generic controls in one ledger)."""
    from harness.analysis import inputs, pipeline

    row = inputs.CheckedOutput("kat-run", cfg.graph, KAT_MODELS[0], cfg.natural_cohort, "curated", cfg.checker_use,
                               "kq00", "kq00", True, "P", True)
    names = ("curated", "census-t5", "generic_retry/curated", "generic_retry/union", "union", "census-t5/replay")
    rows = [dataclasses.replace(row, source=name, run_id=f"kat-run-{i}") for i, name in enumerate(names)]
    any_row = dataclasses.replace(row, source="generic_retry/any", run_id="kat-run-parent")
    sources = [{"name": "curated", "kind": "curated", "digest": "0" * 64},
               {"name": "census-t5", "kind": "census", "digest": "1" * 64}]
    record = {"run_id": "kat-run-generic", "question_id": "kq00", "model": KAT_MODELS[0], "store": {"name": cfg.graph},
              "checker": {"branched_from": "kat-run-parent", "initial": {"reports": [{"source": "curated"}]},
                          "settings": {"mode": "generic", "version": "checker-loop/2",
                                       "arm": {"name": "any", "sources": sources}}}}
    full_arm = json.loads(json.dumps(record))
    full_arm["checker"]["settings"]["version"] = "checker-loop/1"
    census_read = json.loads(json.dumps(record))
    census_read["checker"]["initial"]["reports"].append({"source": "census-t5"})
    no_curated = json.loads(json.dumps(record))
    no_curated["checker"]["settings"]["arm"]["sources"] = sources[1:]
    verified = pipeline.curated_generic_rows([*rows, any_row], [record])
    return {"mapped": [r.source for r in pipeline.ledger_source_names(rows, cfg)],
            "both names": _attempt(pipeline.ledger_source_names, [*rows, dataclasses.replace(row, source="census")]),
            "another census arm": _attempt(pipeline.ledger_source_names, rows,
                                           dataclasses.replace(cfg, census_source="census-t5")),
            "generic on curated sources": {
                "verified": [list(v) for v in sorted(verified)],
                "mapped": [r.source for r in pipeline.ledger_source_names([*rows[:2], any_row], cfg, verified)]},
            "generic unverified": _attempt(pipeline.ledger_source_names, [any_row], cfg),
            "generic under checker-loop/1": _attempt(pipeline.curated_generic_rows, [any_row], [full_arm]),
            "generic read census": _attempt(pipeline.curated_generic_rows, [any_row], [census_read]),
            "generic without its record": _attempt(pipeline.curated_generic_rows, [any_row], []),
            "generic without a curated source": _attempt(pipeline.curated_generic_rows, [any_row], [no_curated]),
            "two generic controls": _attempt(pipeline.ledger_source_names, [*rows, any_row], cfg, verified)}


def grouping_answers() -> dict[str, Any]:
    """The cluster grouping's answers on the KAT inputs: the components of the declared anchors, the map file's loader
    and self-check, the clusters and their resampling, the dispatcher and the contrasts on a map that links questions."""
    index, bank_index = kat_index(), kat_index(KAT_BANK_QUESTIONS)
    V, V_bank = kat_vectors(index), kat_vectors(index, bank_index)
    two, pooled = kat_config(prereg.TWO_FAMILY), kat_config(prereg.POOLED)
    templates = {q: s.split("|")[0] for q, s in zip(bank_index.questions, bank_index.strata)}
    anchors = {q: KAT_ANCHORS.get(q, ()) for q in bank_index.questions}
    components = prereg.connected_clusters(anchors, templates)
    with tempfile.TemporaryDirectory(prefix="kat-") as tmp:
        root = Path(tmp).resolve()
        wrong = {**components, "kq06": "kq06"}  # a chain cut: the stated cluster is not the component
        files = {"map.json": _map_bytes(bank_index, anchors, components),
                 "wrong.json": _map_bytes(bank_index, anchors, wrong),
                 "singletons.json": _map_bytes(bank_index, {}, {q: q for q in bank_index.questions}),
                 "malformed.json": b'{"schema": "cluster-map/2", "questions": {"kq00": {"template": "tA"}}}\n',
                 "not a map.json": b'{"schema": "cluster-map/1", "clusters": {"kq00": "kq00"}}\n'}
        sha = {}
        for name, raw in files.items():
            (root / name).write_bytes(raw)
            sha[name] = hashlib.sha256(raw).hexdigest()
        load = {name: _attempt(prereg.ClusterMap.load, root / name, sha[name]) for name in files}
        load["other sha256"] = _attempt(prereg.ClusterMap.load, root / "map.json", "0" * 64)
        cmap, singles = load["map.json"], load["singletons.json"]
        cfg_file = dataclasses.replace(two, cluster_map="map.json", cluster_map_sha256=sha["map.json"])
        clusters = cmap.resampling(index, two)
        K = kat_matrix(len(clusters.ids), two)
        tables = _tables(root)
        terms = [stats.Term(1.0, V.c5[m]["num"], V.c5[m]["den"]) for m in KAT_PANEL]
        x = np.arange(len(index), dtype=np.float64) - 3.5
        unmapped = prereg.ClusterMap({q: c for q, c in components.items() if q != "kq05"}, "kat, kq05 unmapped")
        across = prereg.ClusterMap({**components, "kq09": "kq01"}, "kat, a cluster across templates")
        payload = {
            "schema": KAT_SCHEMA, "part": "grouping", "components": components, "load": load,
            "load_clusters": prereg.load_clusters(cfg_file, root),
            "load_clusters not filled in": _attempt(prereg.load_clusters, two, root),
            "a test run's split": [_attempt(prereg.check_test_run_split, ["dev"]),
                                   _attempt(prereg.check_test_run_split, ["dev", "test"])],
            "clusters": {"agent": clusters, "summary": clusters.summary(), "sum": clusters.sum(x),
                         "bank": cmap.resampling(bank_index, two).summary(),
                         "singletons": singles.resampling(index, two).summary(),
                         "singletons in memory": prereg.ClusterMap.singletons(index.questions).resampling(index, two),
                         "unmapped": _attempt(unmapped.resampling, index, two),
                         "across templates": _attempt(across.resampling, index, two),
                         "templates off": _attempt(dataclasses.replace(cmap, templates={**cmap.templates, "kq00": "tB"}
                                                                       ).resampling, index, two),
                         "matrix of another size": _attempt(cmap.resampling, index, two, K[:, :3])},
            "dispatcher": {"clusters": _statistic(prereg.studentise(clusters, terms)),
                           "own matrix": _statistic(prereg.studentise(dataclasses.replace(clusters, matrix=K), terms)),
                           "question matrix": _statistic(prereg.studentise(kat_matrix(len(index), two), terms))},
            "config": _config_answers(root),
            "ledger source names": _ledger_names(two),
            "contrasts": {
                "two-family": _contrasts(V, index, two, tables["two-family"], K, None, cmap),
                "bank-only extension, drawn from the seed": _contrasts(V_bank, index, two, tables["two-family"], None,
                                                                       bank_index, cmap),
                "pooled": _family(V, pooled, tables["pooled"], clusters),
                "failed step 2": _family(V, two, tables["failed step 2"], clusters),
                "empty denominators": _family(_empty(V), two, tables["two-family"], clusters)},
        }
        # entry 73's report: near-zero spread counted per claim (and the sensitivity analysis) on the clusters with
        # their single-cluster resamples, the same on the seed's matrix, and the bounds beyond +-1 shown unbounded
        hand = dataclasses.replace(clusters, matrix=K)
        singles_units = prereg.ClusterMap.singletons(index.questions).resampling(index, pooled,
                                                                                 kat_matrix(len(index), pooled))
        weighted_two = _weighted(root, "zero-two", prereg.TWO_FAMILY, KAT_WEIGHTED_TABLE)
        weighted_pooled = _weighted(root, "zero-pooled", prereg.POOLED, KAT_WEIGHTED_TABLE)
        payload["zero spread"] = {
            # every question its own cluster, the pooled family: its single-question resamples keep a rounding spread
            # in the pooled C1a and C5 (more than PAIRWISE_MAX columns), so the sensitivity analysis runs
            "single-question resamples, pooled": prereg.zero_spread(
                V, singles_units, None, pooled, weighted_pooled, _weighted_primary(
                    _family(V, pooled, prereg.UNCALIBRATED, singles_units), pooled, weighted_pooled)),
            "single-cluster resamples": prereg.zero_spread(V, hand, None, two, weighted_two, _weighted_primary(
                payload["contrasts"]["two-family"], two, weighted_two)),
            "drawn from the seed": prereg.zero_spread(V, clusters, None, two, weighted_two, _weighted_primary(
                _family(V, two, prereg.UNCALIBRATED, clusters), two, weighted_two)),
            "near zero": prereg.near_zero_spread(prereg.studentise(hand, terms)),
            "undefined": prereg.studentise(dataclasses.replace(hand, spread=prereg.spread_undefined), terms)}
        payload["bounded"] = prereg.bounded(payload["contrasts"])
        return _canon(payload, str(root))


# --- the multiplicity's known answers (design log, change-log entries 74 and 76) ----------------------------------------


def _configurations() -> list[dict[str, Any]]:
    return [{"name": name, "null": list(null)} for name, null in KAT_CONFIGURATIONS]


def _weighted(root: Path, name: str, family: str, table: Mapping[str, Mapping[str, tuple[str | None, float]]],
              alphas: Mapping[str, Mapping[int, float]] | None = None, usable: Mapping[str, list[int]] | None = None,
              fwer_pass: bool = True) -> Any:
    """A weighted Holm calibration of ``family`` written under ``root`` and read back by multiplicity.load_files: per
    claim its test and alpha'(s) = factor x its level (KAT_WEIGHTED_TABLE's (test, factor), capped at 1) unless
    ``alphas`` gives them, every level of a claim with a test usable unless ``usable`` says otherwise. The table records
    KAT_CONFIGURATIONS under KAT_FINGERPRINT, and the usable set is a step-2 addendum whose records agree with it on
    step 2's N2 (size 0 at a usable level, 1 at another; FWER 0 in every configuration, 1 in the first when it failed),
    computed by step 2's frozen procedure (prereg.STEP2_PROCEDURE_DIGEST), its outputs not on any host."""
    from harness.analysis import calibration, multiplicity

    members, weights = prereg.FAMILY_CLAIMS[family], dict(kat_config(family).holm_weights)
    tests = {c: table[family][c][0] for c in members}
    alphas = alphas or {c: {s: min(1.0, table[family][c][1] * multiplicity.level(0.05, weights[c], s))
                            for s in multiplicity.reachable_sums(c, members, weights)} for c in members}
    confs = _configurations()
    doc = multiplicity.table_document({family: {c: {"test": tests[c], "alpha_prime": alphas[c]} for c in members}},
                                      0.05, weights, inputs={"fingerprint": KAT_FINGERPRINT, "configurations": confs})
    raw = multiplicity.document_bytes(doc)
    (root / f"{name}-alpha.json").write_bytes(raw)
    table_sha = hashlib.sha256(raw).hexdigest()
    usable = {c: (sorted((usable or {}).get(c, alphas[c])) if tests[c] else []) for c in members}
    n = calibration.N2
    levels = {c: {str(s): {"alpha_prime": alphas[c][s], "size": 0.0 if s in usable[c] else 1.0,
                           "configuration": confs[0]["name"], "se": multiplicity._se(a, n),
                           "limit": a + 2 * multiplicity._se(a, n), "usable": s in usable[c]}
                  for s in (multiplicity.reachable_sums(c, members, weights) if tests[c] else [])
                  for a in (multiplicity.level(0.05, weights[c], s),)} for c in members}
    limit = 0.05 + 2 * multiplicity._se(0.05, n)
    fwer = [{"configuration": conf["name"], "fwer": 0.0 if fwer_pass or i else 1.0, "limit": limit,
             "pass": bool(fwer_pass or i), "null_claims": [c for c in members if c in conf["null"]]}
            for i, conf in enumerate(confs)]
    inputs = {"n": n, "seed": calibration.SEEDS[2], "seed_scheme": calibration.SEED_SCHEME,
              "fingerprint": KAT_FINGERPRINT, "configurations": [conf["name"] for conf in confs],
              "procedure": prereg.STEP2_PROCEDURE_DIGEST,
              "outputs": {"directory": KAT_NO_OUTPUTS, "chunks": 0, "sha256": "0" * 64}}
    raw = multiplicity.document_bytes(multiplicity.addendum_document(
        table_sha, {family: {"fwer_pass": fwer_pass, "usable": usable, "levels": levels, "fwer": fwer}}, inputs,
        KAT_DATE, 0.05, weights))
    (root / f"{name}-usable.json").write_bytes(raw)
    return multiplicity.load_files(root / f"{name}-alpha.json", table_sha, root / f"{name}-usable.json",
                                   hashlib.sha256(raw).hexdigest(), family, 0.05, weights)


def _producers(root: Path, weights: Mapping[str, int]) -> dict[str, Any]:
    """The producers of the frozen inputs on fixed p's (entries 74, 76, 91): step 1's table from KAT_PRODUCER_N
    uniforms per configuration, step 2 on others stored as calibrate.py's chunks, the addendum, and its check
    recomputing it from those chunks, or, with its outputs on no host, accepting it by the freeze check's record; its
    refusals (a doctored size, another procedure, other outputs, outputs on no host without a matching record)."""
    from harness.analysis import calibration, multiplicity

    confs, n = _configurations(), KAT_PRODUCER_N
    datasets = tuple(f"d{i:03d}" for i in range(n))
    width = len(calibration.CLAIMS) * len(prereg.TESTS)
    p1 = [_uniforms(f"step 1 {conf['name']}", datasets, width).reshape(n, len(calibration.CLAIMS), len(prereg.TESTS))
          for conf in confs]
    p2 = [_uniforms(f"step 2 {conf['name']}", datasets, width).reshape(n, len(calibration.CLAIMS), len(prereg.TESTS))
          for conf in confs]
    nulls = [{c: c in conf["null"] for c in calibration.CLAIMS} for conf in confs]
    families, record = multiplicity.step_one_table(p1, nulls, weights, 0.05)
    table = json.loads(multiplicity.document_bytes(multiplicity.table_document(
        families, 0.05, weights, inputs={"fingerprint": KAT_FINGERPRINT, "configurations": confs}, step1=record)))
    table_sha = hashlib.sha256(multiplicity.document_bytes(table)).hexdigest()
    outputs = root / "producer-outputs"
    (outputs / "step2").mkdir(parents=True)
    for i, conf in enumerate(confs):
        for lo in range(0, n, KAT_PRODUCER_CHUNK):
            hi = min(lo + KAT_PRODUCER_CHUNK, n)
            np.savez_compressed(outputs / "step2" / f"c{i:02d}-{lo // KAT_PRODUCER_CHUNK:04d}.npz", p=p2[i][lo:hi],
                                lo=lo, hi=hi, digest=KAT_FINGERPRINT, configuration=conf["name"], config_index=i,
                                step=2, seed=calibration.SEEDS[2], seed_scheme=calibration.SEED_SCHEME)
    result = multiplicity.step_two(lambda i: p2[i], confs, table, weights, 0.05, n)
    inputs = {"n": n, "seed": calibration.SEEDS[2], "seed_scheme": calibration.SEED_SCHEME,
              "fingerprint": KAT_FINGERPRINT, "configurations": [conf["name"] for conf in confs],
              "procedure": KAT_PROCEDURE,
              "outputs": {"directory": str(outputs), **multiplicity.outputs_digest(outputs, 2)}}
    addendum = json.loads(multiplicity.document_bytes(multiplicity.addendum_document(
        table_sha, result, inputs, KAT_DATE, 0.05, weights)))
    check = {"n": n, "seed": calibration.SEEDS[2], "seed_scheme": calibration.SEED_SCHEME, "alpha": 0.05,
             "weights": weights}
    # the same addendum naming its outputs on no host (the review of 9d9c88c): accepted only with the freeze check's
    # record of its recomputation, for its sha256, of the frozen procedure and the addendum's outputs
    away = {**addendum, "inputs": {**inputs, "outputs": {**inputs["outputs"], "directory": KAT_NO_OUTPUTS}}}
    away_sha = hashlib.sha256(multiplicity.document_bytes(away)).hexdigest()
    record = {"addendum_sha256": away_sha, "procedure": KAT_PROCEDURE,
              "outputs": {k: inputs["outputs"][k] for k in ("chunks", "sha256")}}
    doctored = json.loads(json.dumps(addendum))
    first = next(c for c, r in doctored["families"]["two-family"]["levels"].items() if r)
    s = next(iter(doctored["families"]["two-family"]["levels"][first]))
    doctored["families"]["two-family"]["levels"][first][s]["size"] = 0.0
    # chunks calibrate.py would not have written: no seed recorded, p-values of another shape, a gap in the datasets
    bad = root / "producer-doctored"
    (bad / "step2").mkdir(parents=True)
    np.savez_compressed(bad / "step2" / "c00-0000.npz", p=p2[0])
    meta = {"digest": KAT_FINGERPRINT, "configuration": confs[1]["name"], "config_index": 1, "step": 2,
            "seed": calibration.SEEDS[2], "seed_scheme": calibration.SEED_SCHEME}
    np.savez_compressed(bad / "step2" / "c01-0000.npz", p=p2[1][:, :3], lo=0, hi=n, **meta)
    np.savez_compressed(bad / "step2" / "c02-0000.npz", p=p2[2][:100], lo=0, hi=100, **{**meta, "configuration": confs[
        2]["name"], "config_index": 2})
    np.savez_compressed(bad / "step2" / "c02-0001.npz", p=p2[2][150:], lo=150, hi=n, **{**meta, "configuration": confs[
        2]["name"], "config_index": 2})
    chunk = {"fingerprint": KAT_FINGERPRINT, "n": n, "seed": calibration.SEEDS[2], "seed_scheme": calibration.SEED_SCHEME}
    loads = {"no seed recorded": (bad, 0, confs[0]["name"], chunk), "another shape": (bad, 1, confs[1]["name"], chunk),
             "a gap": (bad, 2, confs[2]["name"], chunk),
             "another fingerprint": (outputs, 0, confs[0]["name"], {**chunk, "fingerprint": "another"}),
             "another configuration": (outputs, 0, "another", chunk),
             "another seed": (outputs, 0, confs[0]["name"], {**chunk, "seed": calibration.SEEDS[1]}),
             "fewer datasets than n": (outputs, 0, confs[0]["name"], {**chunk, "n": n + 1})}
    # the common factor (ruling 3) where the unscaled FWER exceeds alpha (alpha' 1 everywhere): bisected on the
    # uniforms, none on p's of 0
    members = prereg.FAMILY_CLAIMS[prereg.TWO_FAMILY]
    ones = {c: dict.fromkeys(multiplicity.reachable_sums(c, members, weights), 1.0) for c in members}
    tests = dict.fromkeys(members, "boot_t")
    return {"step 1": {"families": families, "record": record}, "step 2": result, "addendum": addendum,
            "checked": multiplicity.check_addendum(addendum, table, procedure=KAT_PROCEDURE, **check),
            "checked by the freeze record": multiplicity.check_addendum(away, table, procedure=KAT_PROCEDURE,
                                                                        verified=[record], addendum_sha256=away_sha,
                                                                        **check),
            "chunk refusals": {name: _attempt(multiplicity.configuration_p, d, 2, i, conf_name, **kw)
                               for name, (d, i, conf_name, kw) in loads.items()},
            "joint factor": {"bisected": multiplicity.joint_factor(p1, nulls, members, tests, ones, weights, 0.05),
                             "none": multiplicity.joint_factor([np.zeros_like(p1[0])], nulls[:1], members, tests, ones,
                                                               weights, 0.05)},
            "p* without alpha' and falling": [multiplicity.p_star_array(np.array([0.0, 0.5]), 8, None, 0.05),
                                              _attempt(multiplicity.p_star_array, np.array([0.0]), 8,
                                                       {8: 0.01, 9: 0.02}, 0.05)],
            "refusals": {"a doctored size": _attempt(multiplicity.check_addendum, doctored, table,
                                                     procedure=KAT_PROCEDURE, **check),
                         "another procedure": _attempt(multiplicity.check_addendum, addendum, table, procedure="other",
                                                       **check),
                         "other outputs": _attempt(multiplicity.check_addendum, {**addendum, "inputs": {
                             **inputs, "outputs": {**inputs["outputs"], "sha256": "0" * 64}}}, table,
                                                   procedure=KAT_PROCEDURE, **check),
                         "no ISO date": _attempt(multiplicity.check_addendum, {**addendum, "date": "26 Oct"}, table,
                                                 procedure=KAT_PROCEDURE, **check),
                         **{f"outputs on no host, {name}": _attempt(
                             multiplicity.check_addendum, away, table, procedure=KAT_PROCEDURE, verified=verified,
                             addendum_sha256=away_sha, **check) for name, verified in (
                             ("no freeze record", []),
                             ("a record of another addendum", [{**record, "addendum_sha256": "0" * 64}]),
                             ("a record of another procedure", [{**record, "procedure": "other"}]),
                             ("a record of other outputs", [{**record, "outputs": {**record["outputs"],
                                                                                   "sha256": "0" * 64}}]),
                             ("two records", [record, record]))},
                         "fewer datasets than n": _attempt(multiplicity.step_two, lambda i: p2[i][:-1], confs, table,
                                                           weights, 0.05, n)}}


def _weighted_primary(results: Mapping[str, Mapping[str, Any]], cfg: prereg.PreregConfig, weighted: Any) -> dict:
    """The weighted family's blocks on contrasts' test p's, as zero_spread reads the primary results."""
    from harness.analysis import multiplicity

    blocks = multiplicity.decisions({cid: c["p_test"] for cid, c in results.items()}, cfg, weighted)
    return {cid: {"family": b} for cid, b in blocks.items()}


def _doctored(root: Path, name: str, which: str, path: tuple[str, ...], value: Any, weights: Mapping[str, int]) -> Any:
    """The two-family tables (``_weighted``'s "two") with one setting of one file doctored, loaded: the refusal."""
    from harness.analysis import multiplicity

    docs = {k: json.loads((root / f"two-{k}.json").read_text(encoding="utf-8")) for k in ("alpha", "usable")}
    node = docs[which]
    for key in path[:-1]:
        node = node[key]
    if value == "<drop>":
        del node[path[-1]]
    else:
        node[path[-1]] = value
    raw = {"alpha": multiplicity.document_bytes(docs["alpha"])}
    docs["usable"]["table_sha256"] = hashlib.sha256(raw["alpha"]).hexdigest()
    raw["usable"] = multiplicity.document_bytes(docs["usable"])
    for k, data in raw.items():
        (root / f"{name}-{k}.json").write_bytes(data)
    return _attempt(multiplicity.load_files, root / f"{name}-alpha.json", hashlib.sha256(raw["alpha"]).hexdigest(),
                    root / f"{name}-usable.json", hashlib.sha256(raw["usable"]).hexdigest(), prereg.TWO_FAMILY, 0.05,
                    weights)


def _freeze_registries(root: Path) -> list[str]:
    """Freeze registries the step-2 records are read from (multiplicity.freeze_records), written under ``root``: none,
    today's freeze-registry/1 (it records none), one recording a step-2 record, and two malformed ones."""
    from harness.analysis import multiplicity

    docs = {"today's": {"schema": "freeze-registry/1", "note": "", "entries": []},
            "recorded": {"schema": multiplicity.FREEZE_SCHEMA, "step2_outputs": [{"addendum_sha256": "0" * 64}]},
            "not a registry": [], "malformed records": {"schema": multiplicity.FREEZE_SCHEMA, "step2_outputs": ["x"]}}
    for name, doc in docs.items():
        (root / f"{name}.json").write_text(json.dumps(doc), encoding="utf-8")
    return ["absent", *docs]


def multiplicity_answers() -> dict[str, Any]:
    """The weighted Holm's answers on the KAT inputs: the reachable sums and levels of both families, p* on a grid (its
    usable levels only, its refusal), the step-down on hand rows (C3 rejected alone with its p* on the threshold, a tie
    in family order, a p* on the one level where dividing first would fail), the frozen tables' loader and its
    refusals (entry 76: unusable levels, a failed FWER check), step 1's own reachable-level alpha' on fixed p's, and the
    weighted family on the KAT's contrasts (both families, a claim whose first level is unusable, C3 rejected alone),
    C3's equivalence reading on hand intervals (entry 85), and the producers of the frozen inputs on fixed p's with the
    step-2 addendum's check (entries 74, 76, 91)."""
    from harness.analysis import calibration, multiplicity

    index = kat_index()
    V, two, pooled = kat_vectors(index), kat_config(prereg.TWO_FAMILY), kat_config(prereg.POOLED)
    weights = dict(two.holm_weights)
    units = prereg.ClusterMap.singletons(index.questions).resampling(index, two, kat_matrix(len(index), two))
    with tempfile.TemporaryDirectory(prefix="kat-") as tmp:
        root = Path(tmp).resolve()
        members = prereg.FAMILY_CLAIMS[prereg.TWO_FAMILY]
        step1 = {c: dict(zip(multiplicity.reachable_sums(c, members, weights), KAT_WEIGHTED_STEP1[c])) for c in members}
        lean = {c: {s: 1e-6 for s in multiplicity.reachable_sums(c, members, weights)} for c in members}
        alone = {**lean, "C3": {s: 0.9 for s in multiplicity.reachable_sums("C3", members, weights)}}
        tables = {
            "two-family": _weighted(root, "two", prereg.TWO_FAMILY, KAT_WEIGHTED_TABLE),
            "pooled, C3's first level unusable, C5 without one": _weighted(
                root, "pooled", prereg.POOLED, KAT_WEIGHTED_TABLE,
                usable={**{c: multiplicity.reachable_sums(c, prereg.FAMILY_CLAIMS[prereg.POOLED], weights)
                           for c in prereg.FAMILY_CLAIMS[prereg.POOLED]}, "C3": list(range(8, 16)), "C5": []}),
            "FWER failed": _weighted(root, "failed", prereg.TWO_FAMILY, KAT_WEIGHTED_TABLE, fwer_pass=False),
            "C3 alone": _weighted(root, "alone", prereg.TWO_FAMILY, KAT_WEIGHTED_TABLE, alphas=alone),
            "step 1": _weighted(root, "step1", prereg.TWO_FAMILY, {prereg.TWO_FAMILY: dict.fromkeys(members, (
                "boot_t", 1.0))}, alphas=step1)}
        good = root / "two-alpha.json", tables["two-family"].source.split("sha256 ")[1].split(")")[0]
        refusals = {
            "other sha256": _attempt(multiplicity.load_files, good[0], "0" * 64, root / "two-usable.json", "0" * 64,
                                     prereg.TWO_FAMILY, 0.05, weights),
            "another table's usable levels": _attempt(multiplicity.load_files, *good, root / "alone-usable.json",
                                                      hashlib.sha256((root / "alone-usable.json").read_bytes()).hexdigest(),
                                                      prereg.TWO_FAMILY, 0.05, weights),
            "the other family": _attempt(multiplicity.load_files, *good, root / "two-usable.json",
                                         hashlib.sha256((root / "two-usable.json").read_bytes()).hexdigest(),
                                         prereg.POOLED, 0.05, weights),
            "other weights": _attempt(multiplicity.load_files, *good, root / "two-usable.json",
                                      hashlib.sha256((root / "two-usable.json").read_bytes()).hexdigest(),
                                      prereg.TWO_FAMILY, 0.05, {**weights, "C3": 4}),
            "not filled in": _attempt(multiplicity.load, two, root),
            "a falling alpha'": _attempt(multiplicity.p_star, 0.01, 8, {8: 0.01, 9: 0.02}, 0.05),
            "the pooled config on the two-family tables": _attempt(multiplicity.decisions, {
                cid: {"boot_t": 0.5, "cr2": 0.5} for cid in prereg.CONTRASTS}, pooled, tables["two-family"]),
            "other weights in the config": _attempt(dataclasses.replace, two, holm_weights={**weights, "C2": 2}),
            "tables with other weights": _attempt(multiplicity.decisions, {
                cid: {"boot_t": 0.5, "cr2": 0.5} for cid in prereg.CONTRASTS}, two,
                dataclasses.replace(tables["two-family"], weights={**weights, "C2": 2})),
            **{name: _doctored(root, f"doctored-{i}", *how, weights) for i, (name, how) in enumerate(KAT_DOCTORED.items())}}
        usable_sha = hashlib.sha256((root / "two-usable.json").read_bytes()).hexdigest()
        step3_settings, sel_raw, step3_raw = _step3_pass(root, "two", prereg.TWO_FAMILY)
        # the freeze check's records of the step-2 addendum's outputs (by step 2's own procedure), of step 3's selection
        # from its chunks and steps 1 and 2's, and of step 3's validation outputs (on no host; the reviews of 9d9c88c
        # and 1e31f1f)
        step2_record = {"addendum_sha256": usable_sha, "procedure": prereg.STEP2_PROCEDURE_DIGEST,
                        "outputs": {"chunks": 0, "sha256": "0" * 64}}
        selection_record = {"selection_sha256": hashlib.sha256(sel_raw).hexdigest(), "procedure": KAT_PROCEDURE,
                            "outputs": {"selection": {"chunks": 0, "sha256": "0" * 64},
                                        "step1": step3_settings["step1_outputs"], "step2": {"chunks": 0, "sha256": "0" * 64}}}
        step3_record = {"addendum_sha256": hashlib.sha256(step3_raw).hexdigest(), "procedure": KAT_PROCEDURE,
                        "outputs": {"chunks": 0, "sha256": "0" * 64}}
        for registry, records in (("freeze-registry", {"step2_outputs": [step2_record],
                                                       "step3_selection_outputs": [selection_record],
                                                       "step3_outputs": [step3_record]}),
                                  ("step2-only-registry", {"step2_outputs": [step2_record]}),
                                  ("no-selection-registry", {"step2_outputs": [step2_record],
                                                             "step3_outputs": [step3_record]})):
            (root / f"{registry}.json").write_text(json.dumps({"schema": multiplicity.FREEZE_SCHEMA, **records}),
                                                   encoding="utf-8")
        filed = dataclasses.replace(two, multiplicity_table="two-alpha.json", multiplicity_table_sha256=good[1],
                                    step2_addendum="two-usable.json", step2_addendum_sha256=usable_sha,
                                    multiplicity_digest=KAT_PROCEDURE, freeze_registry="freeze-registry.json",
                                    calibrated_inference_digest=KAT_INFERENCE, **step3_settings)
        base = {"two-family": prereg.run_contrasts(V, units, two, intervals=False, keep_private=True),
                "pooled": prereg.run_contrasts(V, units, pooled, intervals=False, keep_private=True)}
        decide = {name: multiplicity.decisions({cid: c["p_test"] for cid, c in base[fam].items()}, cfg, tables[name])
                  for name, fam, cfg in (("two-family", "two-family", two), ("FWER failed", "two-family", two),
                                         ("C3 alone", "two-family", two))}
        decide["no table"] = multiplicity.decisions({cid: c["p_test"] for cid, c in base["pooled"].items()}, pooled,
                                                    multiplicity.NO_TABLE)
        fixed = {name: multiplicity.decisions({cid: {"boot_t": p, "cr2": p} for cid, p in ps.items()}, two,
                                              tables["step 1"]) for name, ps in KAT_WEIGHTED_P.items()}
        payload = {
            "schema": KAT_SCHEMA, "part": "multiplicity", "weights": weights,
            "levels": {fam: {c: [multiplicity.reachable_sums(c, ms, weights),
                                 multiplicity.reachable_levels(c, ms, weights, 0.05)] for c in ms}
                       for fam, ms in prereg.FAMILY_CLAIMS.items()},
            "loaded from the config's files": multiplicity.load(filed, root),
            "an addendum of another procedure": _attempt(multiplicity.load, dataclasses.replace(
                filed, multiplicity_digest="another"), root),
            "an addendum without the freeze record": _attempt(multiplicity.load, dataclasses.replace(
                filed, freeze_registry="no-registry.json"), root),
            "freeze records": {name: _attempt(multiplicity.freeze_records, root / f"{name}.json") for name in
                               _freeze_registries(root)},
            "producers": _producers(root, weights),
            "step 3": _step3_producers(root, weights),
            "step 3 not filled in": _attempt(multiplicity.load, dataclasses.replace(filed, step3_addendum="[FILL]"), root),
            "step 3 of another sha256": _attempt(multiplicity.load, dataclasses.replace(
                filed, step3_addendum_sha256="0" * 64), root),
            "step 3's selection as its addendum": _attempt(multiplicity.load, dataclasses.replace(
                filed, step3_addendum=filed.step3_selection, step3_addendum_sha256=filed.step3_selection_sha256), root),
            "step 3 without its freeze record": _attempt(multiplicity.load, dataclasses.replace(
                filed, freeze_registry="step2-only-registry.json"), root),
            "step 3's selection without its freeze record": _attempt(multiplicity.load, dataclasses.replace(
                filed, freeze_registry="no-selection-registry.json"), root),
            "step 3 under another fingerprint": _attempt(multiplicity.load, dataclasses.replace(
                filed, step3_fingerprint="another"), root),
            "step 3's configurations not bound": _attempt(multiplicity.load, dataclasses.replace(
                filed, step3_configurations="[FILL]"), root),
            "step 2 by the step-3 procedure": _attempt(multiplicity.check_addendum, json.loads(
                (root / "two-usable.json").read_bytes()), json.loads((root / "two-alpha.json").read_bytes()),
                n=calibration.N2, seed=calibration.SEEDS[2], seed_scheme=calibration.SEED_SCHEME,
                procedure=KAT_PROCEDURE, alpha=0.05, weights=weights,
                verified=multiplicity.freeze_records(root / "freeze-registry.json"), addendum_sha256=usable_sha),
            "both families": multiplicity.load_final(filed, root),
            "a chosen family the table lacks": _attempt(multiplicity.load, filed.with_c1a_choice(0.7, 0.3), root),
            "step 2 alone": multiplicity.load_step2(filed, root),
            "without alpha'": [multiplicity.p_star(0.0, 8, None, 0.05), multiplicity.p_star(0.0, 8, {}, 0.05)],
            "p_star": [[c, x, multiplicity.p_star(x, weights[c], step1[c], 0.05),
                        multiplicity.p_star(x, weights[c], {s: a for s, a in step1[c].items() if s % 2}, 0.05)]
                       for c in members for x in (0.0, *KAT_WEIGHTED_STEP1[c][::3], 0.004, 0.02, 0.06, 1.0)],
            "step-down": [multiplicity.step_down(row, KAT_HOLM_WEIGHTS, 0.05) for row in KAT_HOLM_ROWS],
            "tables": tables, "refusals": refusals, "finals": {name: {c: t.final(c) for c in t.claims}
                                                               for name, t in tables.items()},
            "decisions": decide, "step 1 on fixed p's": fixed,
            "C3 equivalence": [multiplicity.c3_equivalence({"estimate": e, "ci_weighted_bonferroni": {"lo": lo, "hi": hi}},
                                                           calibrated, two.c3_sesoi)
                               for (e, lo, hi), calibrated in KAT_C3_EQUIVALENCE],
            "families": {"two-family": multiplicity.weighted_family(base["two-family"], two, tables["two-family"]),
                         "pooled, C3's first level unusable, C5 without one": multiplicity.weighted_family(
                             base["pooled"], pooled, tables["pooled, C3's first level unusable, C5 without one"])}}
        return _canon(payload, str(root))


def _step3_selection_pass(t: Mapping[str, Any], u: Mapping[str, Any], family: str) -> dict[str, Any]:
    """One family's step-3 selection passing the calibration (``t`` its alpha' table, ``u`` its step-2 addendum) through
    as step3_select records a selection whose own values are the table's: C2 at every reachable level at the table's
    alpha' on its level's grid (calibration._snap: j x 0.0005 a, at most 2a) and at most the next larger level's, its
    test by entry 46's rule on size
    checks that pass exactly its table test; every level step 2 left unusable repaired with the table's alpha' (on its
    grid) as its own value (multiplicity.repair_records)."""
    from harness.analysis import calibration, multiplicity

    weights, claims = t["weights"], t["families"][family]["claims"]
    scope = multiplicity.step3_scope(t, u, family)

    def own(c: str, s: int) -> float:
        a = multiplicity.level(0.05, weights[c], s)
        return calibration._snap(float((claims[c]["alpha_prime"] or {}).get(str(s), a)), a, calibration.GRID_STEPS)

    test, c2, upper = claims["C2"]["test"], {}, math.inf
    for s in scope["C2"]:  # as cp_alpha_primes snaps: from the largest level down, at most the next larger's
        upper = calibration._snap(min(own("C2", s), upper), multiplicity.level(0.05, weights["C2"], s),
                                  calibration.GRID_STEPS)
        c2[str(s)] = upper
    sel: dict[str, Any] = {"C2": {"test": test, "alpha_prime": c2,
                                  "selection_size_ok": {x: x == test for x in prereg.TESTS}}, "repairs": {}}
    for c, levels in scope.items():
        if c != "C2":
            kept = {int(s): float(claims[c]["alpha_prime"][str(s)]) for s in u["families"][family]["usable"][c]}
            rec = multiplicity.repair_records({s: own(c, s) for s in levels}, c, weights, 0.05, kept)
            sel["repairs"][c] = {"test": claims[c]["test"], "levels": {str(s): r for s, r in rec.items()}}
    return sel


def _step3_pass(root: Path, name: str, family: str) -> tuple[dict[str, Any], bytes, bytes]:
    """Step 3's bound configurations, selection and addendum passing the single-family calibration ``name`` (its
    -alpha.json and -usable.json under ``root``) through as the frozen producers would record them: the table's
    configurations bound with their null claims; the selection _step3_selection_pass's, read from steps 1 and 2's
    frozen chunks (step 1's content as the settings bind it, step 2's as its addendum records it) and from a selection
    run on no host; every level it selected validated (size 0 in the first configuration where its claim is null, none
    where it is null in none), FWER 0, the outputs on no host. Returns the config settings naming the three files with
    the bound fingerprint and step 1's content, and the selection's and the addendum's bytes for the freeze records."""
    from harness.analysis import calibration, multiplicity

    t_raw, u_raw = (root / f"{name}-alpha.json").read_bytes(), (root / f"{name}-usable.json").read_bytes()
    t, u = json.loads(t_raw), json.loads(u_raw)
    weights, n, confs = t["weights"], calibration.N3, t["inputs"]["configurations"]
    members = prereg.FAMILY_CLAIMS[family]
    confs_raw = multiplicity.document_bytes(multiplicity.step3_configurations_document(confs, "0" * 64, 0))
    step1 = {"chunks": 0, "sha256": "1" * 64}
    sel = {family: _step3_selection_pass(t, u, family)}
    sel_inputs = {"table_sha256": hashlib.sha256(t_raw).hexdigest(), "step2_addendum_sha256": hashlib.sha256(u_raw).hexdigest(),
                  "fingerprint": KAT_STEP3_FINGERPRINT, "steps12_fingerprint": t["inputs"]["fingerprint"],
                  "n_selection": calibration.N3_SELECTION, "seed": prereg.STEP3_SEEDS["selection"],
                  "seed_scheme": calibration.SEED_SCHEME, "configurations": confs,
                  "selection_configurations": [c["name"] for c in confs if "C2" in c["null"]], "repair_data": {},
                  "outputs": {"directory": KAT_NO_STEP3_OUTPUTS, "chunks": 0, "sha256": "0" * 64},
                  "steps12_directory": KAT_NO_OUTPUTS,
                  "steps12_outputs": {"step1": step1, "step2": {k: u["inputs"]["outputs"][k] for k in ("chunks", "sha256")}},
                  "procedure": KAT_PROCEDURE, "inference": KAT_INFERENCE}
    sel_raw = multiplicity.document_bytes(multiplicity.step3_alpha_document(sel, sel_inputs, 0.05, weights))
    tests, alphas, check = multiplicity.step3_candidates(t, u, sel, family)

    def worst(c: str) -> str | None:
        return next((conf["name"] for conf in confs if c in conf["null"]), None)

    levels = {c: {str(s): {"alpha_prime": alphas[c][s], "size": 0.0 if worst(c) else None, "configuration": worst(c),
                           "se": multiplicity._se(a, n), "limit": a + 2 * multiplicity._se(a, n), "usable": True}
                  for s in check[c] for a in (multiplicity.level(0.05, weights[c], s),)} for c in check}
    limit = 0.05 + 2 * multiplicity._se(0.05, n)
    fwer = [{"configuration": conf["name"], "fwer": 0.0, "limit": limit, "pass": True,
             "null_claims": [c for c in members if c in conf["null"]]} for conf in confs]
    result = {family: {"tests": tests, "alpha_prime": {c: {str(k): v for k, v in sorted(alphas[c].items())}
                                                       for c in members},
                       "usable": {c: sorted(alphas[c]) for c in members}, "levels": levels, "fwer": fwer,
                       "fwer_pass": True}}
    inputs = {"selection_sha256": hashlib.sha256(sel_raw).hexdigest(), "table_sha256": hashlib.sha256(t_raw).hexdigest(),
              "step2_addendum_sha256": hashlib.sha256(u_raw).hexdigest(), "n": n,
              "seed": prereg.STEP3_SEEDS["validation"], "seed_scheme": calibration.SEED_SCHEME,
              "fingerprint": KAT_STEP3_FINGERPRINT, "configurations": confs, "procedure": KAT_PROCEDURE,
              "inference": KAT_INFERENCE, "outputs": {"directory": KAT_NO_STEP3_OUTPUTS, "chunks": 0, "sha256": "0" * 64}}
    step3_raw = multiplicity.document_bytes(multiplicity.step3_document(result, inputs, KAT_STEP3_DATE, 0.05, weights))
    (root / f"{name}-step3-configurations.json").write_bytes(confs_raw)
    (root / f"{name}-step3-alpha.json").write_bytes(sel_raw)
    (root / f"{name}-step3.json").write_bytes(step3_raw)
    return ({"step1_outputs": step1, "step3_configurations": f"{name}-step3-configurations.json",
             "step3_configurations_sha256": hashlib.sha256(confs_raw).hexdigest(),
             "step3_fingerprint": KAT_STEP3_FINGERPRINT,
             "step3_selection": f"{name}-step3-alpha.json", "step3_selection_sha256": hashlib.sha256(sel_raw).hexdigest(),
             "step3_addendum": f"{name}-step3.json", "step3_addendum_sha256": hashlib.sha256(step3_raw).hexdigest()},
            sel_raw, step3_raw)


def _step3_producers(root: Path, weights: Mapping[str, int]) -> dict[str, Any]:
    """Step 3's producers on fixed p's (entry 106): the binomial CDF and the Clopper-Pearson count with their edges,
    the margin rule's grid value, C2's selection and a repair (kept, capped, unusable), step 3's selection and
    validation on a step-1 table and a step-2 set with C5's s = 2 taken out (so it is repaired), a level failing
    validation, the documents, the chunk reader's phase, and the addendum's check recomputing it from its chunks or
    accepting it by the freeze record, with its refusals."""
    from harness.analysis import calibration, multiplicity

    confs, n = _configurations(), KAT_PRODUCER_N
    datasets = tuple(f"d{i:03d}" for i in range(n))
    shape = (n, len(calibration.CLAIMS), len(prereg.TESTS))
    width = shape[1] * shape[2]

    def draws(what: str) -> list[np.ndarray]:
        return [_uniforms(f"{what} {conf['name']}", datasets, width).reshape(shape) for conf in confs]

    p1, p2, sel_p, val_p = draws("step 1"), draws("step 2"), draws("step 3 selection"), draws("step 3 validation")
    nulls = [{c: c in conf["null"] for c in calibration.CLAIMS} for conf in confs]
    families, record = multiplicity.step_one_table(p1, nulls, weights, 0.05)
    table = json.loads(multiplicity.document_bytes(multiplicity.table_document(
        families, 0.05, weights, inputs={"fingerprint": KAT_FINGERPRINT, "configurations": confs}, step1=record)))
    two_step = multiplicity.step_two(lambda i: p2[i], confs, table, weights, 0.05, n)
    usable = json.loads(multiplicity.document_bytes(multiplicity.usable_document("kat", two_step, 0.05, weights)))
    usable["families"]["two-family"]["usable"]["C5"] = [
        s for s in usable["families"]["two-family"]["usable"]["C5"] if s != 2]  # repaired at step 3
    c2i = calibration.CLAIMS.index("C2")
    null_idx = {c: [i for i, conf in enumerate(confs) if c in conf["null"]] for c in calibration.CLAIMS}
    c2_null_p = {t: [sel_p[i][:, c2i, ti] for i in null_idx["C2"]] for ti, t in enumerate(prereg.TESTS)}
    repair_p = {fam: {c: [sel_p[i][:, calibration.CLAIMS.index(c), prereg.TESTS.index(
        table["families"][fam]["claims"][c]["test"])] for i in null_idx[c]]
        for c in multiplicity.step3_scope(table, usable, fam) if c != "C2"} for fam in prereg.FAMILY_CLAIMS}
    selection = multiplicity.step3_select(c2_null_p, repair_p, table, usable, 0.05)
    frozen12 = {"step1": {"chunks": 0, "sha256": "1" * 64}, "step2": {"chunks": 0, "sha256": "0" * 64}}
    sel_inputs = {"table_sha256": "kat", "step2_addendum_sha256": "kat", "fingerprint": KAT_STEP3_FINGERPRINT,
                  "steps12_fingerprint": KAT_FINGERPRINT, "n_selection": n, "seed": prereg.STEP3_SEEDS["selection"],
                  "seed_scheme": calibration.SEED_SCHEME, "configurations": confs,
                  "selection_configurations": [conf["name"] for conf in confs if "C2" in conf["null"]],
                  "repair_data": {}, "outputs": {"directory": KAT_NO_STEP3_OUTPUTS, "chunks": 0, "sha256": "0" * 64},
                  "steps12_directory": KAT_NO_OUTPUTS, "steps12_outputs": frozen12, "procedure": KAT_PROCEDURE,
                  "inference": KAT_INFERENCE}
    sel_doc = json.loads(multiplicity.document_bytes(multiplicity.step3_alpha_document(
        selection, sel_inputs, 0.05, weights)))
    sel_sha = hashlib.sha256(multiplicity.document_bytes(sel_doc)).hexdigest()
    outputs = root / "step3-outputs"
    (outputs / "step3-validation").mkdir(parents=True)
    for i, conf in enumerate(confs):
        np.savez_compressed(outputs / "step3-validation" / f"c{i:02d}-0000.npz", p=val_p[i], lo=0, hi=n,
                            digest=KAT_STEP3_FINGERPRINT, configuration=conf["name"], config_index=i, step=3,
                            phase="validation", seed=prereg.STEP3_SEEDS["validation"], seed_scheme=calibration.SEED_SCHEME)
    mixed = root / "step3-mixed"  # a selection chunk where the validation's are read
    (mixed / "step3-validation").mkdir(parents=True)
    np.savez_compressed(mixed / "step3-validation" / "c00-0000.npz", p=val_p[0], lo=0, hi=n, digest=KAT_STEP3_FINGERPRINT,
                        configuration=confs[0]["name"], config_index=0, step=3, phase="selection",
                        seed=prereg.STEP3_SEEDS["validation"], seed_scheme=calibration.SEED_SCHEME)
    repaired = json.loads(json.dumps(sel_doc))  # C5's s = 2 repaired usable at 0.01, by hand
    repaired["families"]["two-family"]["repairs"]["C5"]["levels"]["2"] = {
        "own": 0.01, "alpha_prime": 0.01, "next_smaller_level_alpha_prime": None, "usable": True}
    read = multiplicity.configuration_p(outputs, 3, 0, confs[0]["name"], KAT_STEP3_FINGERPRINT, n,
                                        prereg.STEP3_SEEDS["validation"], calibration.SEED_SCHEME, phase="validation")
    result = multiplicity.step_three(lambda i: val_p[i], confs, table, usable, sel_doc, 0.05, n)
    halved = [p.copy() for p in val_p]
    halved[0][:, c2i, :] /= 4  # C2's p's a quarter in the joint null: its sizes beyond a + 2 SE at some levels
    failing = multiplicity.step_three(lambda i: halved[i], confs, table, usable, sel_doc, 0.05, n)
    inputs = {"selection_sha256": sel_sha, "table_sha256": "kat", "step2_addendum_sha256": "kat", "n": n,
              "seed": prereg.STEP3_SEEDS["validation"], "seed_scheme": calibration.SEED_SCHEME,
              "fingerprint": KAT_STEP3_FINGERPRINT, "configurations": confs, "procedure": KAT_PROCEDURE,
              "inference": KAT_INFERENCE, "outputs": {"directory": str(outputs),
                                                      **multiplicity.outputs_digest(outputs, "3-validation")}}
    doc = json.loads(multiplicity.document_bytes(multiplicity.step3_document(result, inputs, KAT_STEP3_DATE, 0.05,
                                                                             weights)))
    away = {**doc, "inputs": {**inputs, "outputs": {**inputs["outputs"], "directory": KAT_NO_STEP3_OUTPUTS}}}
    away_sha = hashlib.sha256(multiplicity.document_bytes(away)).hexdigest()
    rec = {"addendum_sha256": away_sha, "procedure": KAT_PROCEDURE,
           "outputs": {k: inputs["outputs"][k] for k in ("chunks", "sha256")}}
    kw = {"configurations": confs, "fingerprint": KAT_STEP3_FINGERPRINT, "n": n,
          "seed": prereg.STEP3_SEEDS["validation"], "seed_scheme": calibration.SEED_SCHEME,
          "procedure": KAT_PROCEDURE, "inference": KAT_INFERENCE, "alpha": 0.05, "weights": weights,
          "selection_sha256": sel_sha}

    def check(d, **more):
        return _attempt(multiplicity.check_step3, d, table, usable, sel_doc, **{**kw, **more})

    # the review of 1e31f1f: every null label emptied and the validation recomputed on those labels (a vacuous pass)
    emptied = [{"name": conf["name"], "null": []} for conf in confs]
    vacuous = multiplicity.step_three(lambda i: val_p[i], emptied, table, usable, sel_doc, 0.05, n)
    vacuous_doc = json.loads(multiplicity.document_bytes(multiplicity.step3_document(
        vacuous, {**inputs, "configurations": emptied}, KAT_STEP3_DATE, 0.05, weights)))
    sel_kw = {"configurations": confs, "fingerprint": KAT_STEP3_FINGERPRINT, "table_sha256": "kat",
              "usable_sha256": "kat", "n": n, "seed": prereg.STEP3_SEEDS["selection"],
              "seed_scheme": calibration.SEED_SCHEME, "procedure": KAT_PROCEDURE, "inference": KAT_INFERENCE,
              "alpha": 0.05, "weights": weights, "steps12_outputs": frozen12, "selection_sha256": sel_sha}
    sel_rec = {"selection_sha256": sel_sha, "procedure": KAT_PROCEDURE,
               "outputs": {"selection": {"chunks": 0, "sha256": "0" * 64}, **frozen12}}

    def check_sel(d, **more):
        return _attempt(multiplicity.check_selection, d, table, usable, **{**sel_kw, "verified": [sel_rec], **more})

    # the selection from its chunks (the review of 1e31f1f): steps 1 and 2's for the configurations of theirs, pooled for
    # the repairs; the selection run's for C2's null configurations, one beyond theirs (C2 and C5 null, so C5's repair
    # reads it there); recomputed by the check from the chunks
    confs3 = [*confs, {"name": "kat-mirror", "null": ["C2", "C5"]}]
    sel_p3 = [*sel_p, _uniforms("step 3 selection kat-mirror", datasets, width).reshape(shape)]
    run = root / "step3-selection-run"
    for step, arrays, seed in ((1, p1, calibration.SEEDS[1]), (2, p2, calibration.SEEDS[2])):
        (run / f"step{step}").mkdir(parents=True)
        for i, conf in enumerate(confs):
            np.savez_compressed(run / f"step{step}" / f"c{i:02d}-0000.npz", p=arrays[i], lo=0, hi=n, digest=KAT_FINGERPRINT,
                                configuration=conf["name"], config_index=i, step=step, seed=seed,
                                seed_scheme=calibration.SEED_SCHEME)
    (run / "step3-selection").mkdir()
    c2_null3 = [i for i, conf in enumerate(confs3) if "C2" in conf["null"]]
    for i in c2_null3:
        np.savez_compressed(run / "step3-selection" / f"c{i:02d}-0000.npz", p=sel_p3[i], lo=0, hi=n,
                            digest=KAT_STEP3_FINGERPRINT, configuration=confs3[i]["name"], config_index=i, step=3,
                            phase="selection", seed=prereg.STEP3_SEEDS["selection"], seed_scheme=calibration.SEED_SCHEME)
    arrays3 = {"fingerprint": KAT_STEP3_FINGERPRINT, "n": n, "seed": prereg.STEP3_SEEDS["selection"],
               "seed_scheme": calibration.SEED_SCHEME, "steps12_n": (n, n)}
    c2_p3, repair_p3, where3 = multiplicity.step3_selection_arrays(table, usable, confs3, run, run, **arrays3)
    run12 = {"step1": multiplicity.outputs_digest(run, 1), "step2": multiplicity.outputs_digest(run, 2)}
    run_inputs = {**sel_inputs, "configurations": confs3, "selection_configurations": [confs3[i]["name"] for i in c2_null3],
                  "repair_data": where3, "outputs": {"directory": str(run), **multiplicity.outputs_digest(run, "3-selection")},
                  "steps12_directory": str(run), "steps12_outputs": run12}
    run_doc = json.loads(multiplicity.document_bytes(multiplicity.step3_alpha_document(
        multiplicity.step3_select(c2_p3, repair_p3, table, usable, 0.05), run_inputs, 0.05, weights)))
    run_kw = {**sel_kw, "configurations": confs3, "steps12_outputs": run12, "steps12_n": (n, n)}
    smallest = json.loads(json.dumps(run_doc))  # on the grid and monotone, but not what the chunks give
    for fam in smallest["families"].values():
        fam["C2"]["alpha_prime"] = {s: 0.0005 * multiplicity.level(0.05, weights["C2"], int(s))
                                    for s in fam["C2"]["alpha_prime"]}
    renamed = [confs3[0], {**confs3[1], "name": "kat-renamed"}, *confs3[2:]]  # C3 null, read from steps 1 and 2
    nowhere = [*confs3, {"name": "kat-nowhere", "null": ["C5"]}]
    (root / "step3-configurations.json").write_bytes(multiplicity.document_bytes(
        multiplicity.step3_configurations_document(confs3, "0" * 64, n)))
    (root / "step3-configurations-empty.json").write_text(json.dumps(
        {"schema": multiplicity.STEP3_CONFIGURATIONS_SCHEMA, "configurations": []}), encoding="utf-8")

    def read_confs(name: str, sha: str | None = None) -> Any:
        path = root / name
        return _attempt(multiplicity.read_step3_configurations, path, sha or hashlib.sha256(path.read_bytes()).hexdigest())

    off_grid = json.loads(json.dumps(sel_doc))  # the review of 1e31f1f: small, monotone, off-grid thresholds
    c2_levels = off_grid["families"]["two-family"]["C2"]["alpha_prime"]
    for i, key in enumerate(sorted(c2_levels, key=int)):
        c2_levels[key] = 1e-3 / (i + 1) + 1e-9
    moved = json.loads(json.dumps(sel_doc))
    moved["families"]["two-family"]["repairs"]["C5"]["levels"]["2"]["alpha_prime"] = 0.01

    doctored = json.loads(json.dumps(doc))
    lv = doctored["families"]["two-family"]["levels"]["C2"]
    first = next(iter(lv))
    lv[first]["size"] = (lv[first]["size"] or 0.0) / 2 if lv[first]["usable"] else 0.0
    lv[first]["usable"] = True
    bad_row = json.loads(json.dumps(doc))
    bad_row["families"]["two-family"]["fwer"][0]["pass"] = not bad_row["families"]["two-family"]["fwer"][0]["pass"]
    bad_pass = json.loads(json.dumps(doc))
    bad_pass["families"]["two-family"]["fwer_pass"] = not bad_pass["families"]["two-family"]["fwer_pass"]
    k = multiplicity.cp_max_count(200, 0.025)
    return {
        "binomial": [multiplicity.binom_cdf(-1, 10, 0.5), multiplicity.binom_cdf(10, 10, 0.5),
                     multiplicity.binom_cdf(5, 10, 0.5), multiplicity.binom_cdf(30, 120000, 0.0003)],
        "cp counts": [multiplicity.cp_max_count(100, 0.03), multiplicity.cp_max_count(100, 0.05),
                      multiplicity.cp_max_count(10, 0.03), multiplicity.cp_max_count(120000, 0.0125)],
        "cp grid": [multiplicity.cp_grid_alpha(np.array([0.0005, 0.0009, 0.002] + [0.5] * 97), 0.05, 4000),
                    multiplicity.cp_grid_alpha(np.array([1e-9] + [0.5] * 99), 0.03, 4000),
                    multiplicity.cp_grid_alpha(np.full(10, 0.5), 0.03, 4000)],
        "repairs by hand": [multiplicity.cp_repair([np.array([0.0001] * k + [0.0200125] + [0.9] * (199 - k))], "X",
                                                   ("X", "Y", "Z"), {"X": 1, "Y": 1, "Z": 1}, 0.05,
                                                   {1: 0.045, 3: 0.012}, [2]),
                            multiplicity.cp_repair([np.full(200, 0.9)], "X", ("X", "Y", "Z"), {"X": 1, "Y": 1, "Z": 1},
                                                   0.05, {1: 0.018, 3: 0.012}, [2]),
                            multiplicity.cp_repair([np.array([0.0001] * k + [0.0100125] + [0.9] * (199 - k))], "X",
                                                   ("X", "Y", "Z"), {"X": 1, "Y": 1, "Z": 1}, 0.05,
                                                   {1: 0.045, 3: 0.012}, [2])],
        "scope": {fam: multiplicity.step3_scope(table, usable, fam) for fam in prereg.FAMILY_CLAIMS},
        "selection": sel_doc, "validation": result, "a level failing validation": failing, "read back": read,
        "wrong n": _attempt(multiplicity.step_three, lambda i: val_p[i], confs, table, usable, sel_doc, 0.05, n + 1),
        "a chunk of another phase": _attempt(multiplicity.configuration_p, mixed, 3, 0, confs[0]["name"],
                                             KAT_STEP3_FINGERPRINT, n, prereg.STEP3_SEEDS["validation"],
                                             calibration.SEED_SCHEME, phase="validation"),
        "candidates with a repair": multiplicity.step3_candidates(table, usable, repaired, "two-family"),
        "addendum": doc,
        "checked": check(doc), "checked by the freeze record": check(away, verified=[rec], step3_sha256=away_sha),
        "refusals": {"no ISO date": check({**doc, "date": "27 Oct"}),
                     "another n": check({**doc, "inputs": {**inputs, "n": n + 1}}),
                     "another procedure": check(doc, procedure="other"),
                     "other inference code": check(doc, inference="other"),
                     "another selection": check(doc, selection_sha256="0" * 64),
                     "no configurations": check({**doc, "inputs": {**inputs, "configurations": None}}),
                     "no outputs": check({**doc, "inputs": {**inputs, "outputs": {}}}),
                     "a doctored decision": check(doctored), "a doctored FWER row": check(bad_row),
                     "a doctored FWER check": check(bad_pass),
                     "other outputs": check({**doc, "inputs": {**inputs, "outputs": {**inputs["outputs"],
                                                                                     "sha256": "0" * 64}}}),
                     "outputs on no host without a record": check(away, step3_sha256=away_sha),
                     "a record of another procedure": check(away, verified=[{**rec, "procedure": "other"}],
                                                            step3_sha256=away_sha),
                     "two records": check(away, verified=[rec, rec], step3_sha256=away_sha),
                     "every null label emptied": check(vacuous_doc),
                     "no configurations bound": check(doc, configurations=[]),
                     "another fingerprint": check(doc, fingerprint="other"),
                     "a level the selection did not select": check({**doc, "families": {**doc["families"], "two-family": {
                         **doc["families"]["two-family"], "levels": {**doc["families"]["two-family"]["levels"],
                                                                     "C1b": {"1": doc["families"]["two-family"]["levels"]
                                                                             ["C2"][first]}}}}})},
        "vacuous validation on emptied labels": vacuous,
        "no configurations": _attempt(multiplicity.step_three, lambda i: val_p[i], [], table, usable, sel_doc, 0.05, n),
        "selection checked by the freeze record": check_sel(sel_doc),
        "selection from its chunks": [run_doc, where3],
        "selection recomputed from its chunks": _attempt(multiplicity.check_selection, run_doc, table, usable, **run_kw),
        "selection refused from its chunks": {
            "thresholds the chunks do not give": _attempt(multiplicity.check_selection, smallest, table, usable, **run_kw),
            "chunks other than recorded": _attempt(multiplicity.check_selection, {**run_doc, "inputs": {
                **run_inputs, "outputs": {**run_inputs["outputs"], "sha256": "0" * 64}}}, table, usable, **run_kw),
            "another name at one of steps 1 and 2's indices": _attempt(
                multiplicity.step3_selection_arrays, table, usable, renamed, run, run, **arrays3),
            "a repaired claim null outside both runs": _attempt(
                multiplicity.step3_selection_arrays, table, usable, nowhere, run, run, **arrays3)},
        "bound configurations": {"read": read_confs("step3-configurations.json"),
                                 "another sha256": read_confs("step3-configurations.json", "0" * 64),
                                 "empty": read_confs("step3-configurations-empty.json")},
        "selection refusals": {"no freeze record": check_sel(sel_doc, verified=[]),
                               "required recomputed": check_sel(sel_doc, require_recomputed=True),
                               "off-grid thresholds": check_sel(off_grid),
                               "a repair moved off its chain": check_sel(moved),
                               "null labels emptied": check_sel({**sel_doc, "inputs": {**sel_inputs,
                                                                                       "configurations": emptied}}),
                               "no configurations bound": check_sel(sel_doc, configurations=[]),
                               "another fingerprint": check_sel(sel_doc, fingerprint="other"),
                               "other steps 1 and 2 chunks": check_sel(sel_doc, steps12_outputs={
                                   **frozen12, "step1": {"chunks": 1, "sha256": "1" * 64}}),
                               "a record of other outputs": check_sel(sel_doc, verified=[{
                                   **sel_rec, "outputs": {**sel_rec["outputs"], "step2": {"chunks": 9, "sha256": "9"}}}]),
                               "another table": check_sel(sel_doc, table_sha256="other")}}


def _eligibility_answers(two: prereg.PreregConfig) -> dict[str, Any]:
    kept, dropped = two.with_c1a_choice(0.7, 0.3), two.with_c1a_choice(0.7, 0.3, ("m2",))
    return {"kept": [kept.c1a_family, kept.c1a_power, kept.require_c1a_eligibility()],
            "dropped": [dropped.c1a_family, dropped.c1a_power, dropped.c1a_eligibility,
                        dropped.require_c1a_eligibility()],
            "refused": [_attempt(dataclasses.replace(kept, c1a_eligibility={}).require_c1a_eligibility),
                        _attempt(dataclasses.replace(kept, c1a_eligibility=dropped.c1a_eligibility).require_c1a_eligibility),
                        _attempt(prereg.c1a_choice, 0.7, 0.0, ("m2",)), _attempt(prereg.c1a_choice, 0.7, 0.3, "m2")]}


def _config_answers(root: Path) -> dict[str, Any]:
    """The config's construction (registered whole in the grouping registry): its refusals, the C1a choice it records,
    a file's load and the path and sha256 it records, the frozen question-set hash and the arms it requires."""
    two, pooled = kat_config(prereg.TWO_FAMILY), kat_config(prereg.POOLED)
    path, unknown = root / "config.json", root / "unknown.json"
    path.write_text(json.dumps(two.to_json(), indent=1) + "\n", encoding="utf-8")
    unknown.write_text(json.dumps({**two.to_json(), "unregistered": 1}) + "\n", encoding="utf-8")
    loaded = prereg.PreregConfig.load(path)
    filled = dataclasses.replace(two, questions_sha256={two.graph: "a" * 64, f"{two.graph}#bank": "b" * 64})
    # the host-change rule's record, bound by its sha256 (kat_hosts, kat_hosts_sha256), checked at load
    hosts = root / "kat-hosts.json"
    hosts.write_text('{"schema": "kat-hosts/1"}\n', encoding="utf-8")
    bound = dataclasses.replace(two, kat_hosts=str(hosts), kat_hosts_sha256=hashlib.sha256(hosts.read_bytes()).hexdigest())
    bound_path = root / "bound.json"
    bound_path.write_text(json.dumps(bound.to_json(), indent=1) + "\n", encoding="utf-8")
    return {"refusals": [_attempt(prereg.PreregConfig, panel=KAT_PANEL, panel_families=("one",)),
                         _attempt(prereg.PreregConfig, panel=("another",)),
                         _attempt(prereg.PreregConfig, c1a_family="another"),
                         _attempt(prereg.PreregConfig, step2_fwer_failure="another"),
                         _attempt(prereg.PreregConfig, refusal_rule="refused"),  # the runner's setting, not the rule
                         _attempt(prereg.PreregConfig, holm_weights={**two.holm_weights, "C3": 0}),
                         _attempt(prereg.PreregConfig, step3_seeds={"selection": 1, "validation": 2}),
                         _attempt(prereg.PreregConfig, step2_procedure_digest="another")],
            "with_c1a_choice": [two.with_c1a_choice(0.7, 0.3), two.with_c1a_choice(0.3, 0.3)],
            # the pooled C1a's eligibility (design log, 18:50; C1A_ELIGIBILITY): entry 81 taking a model out, and every
            # refusal of the record
            "c1a eligibility": _eligibility_answers(two),
            "loaded": {"to_json": loaded.to_json(), "loaded_from": loaded.loaded_from, "built in code": two.loaded_from,
                       "settings equal": loaded.to_json() == two.to_json()},
            "unknown setting": _attempt(prereg.PreregConfig.load, unknown),
            "frozen questions": [_attempt(two.frozen_questions), filled.frozen_questions(),
                                 filled.frozen_questions(bank=True)],
            "arms": {c.c1a_family: {"prompt models": c.prompt_models(), "required": c.required_arms()}
                     for c in (two, pooled)},
            "host record": {"bound": bound.check_kat_hosts(), "unbound": two.check_kat_hosts(),
                            "loaded": prereg.PreregConfig.load(bound_path).kat_hosts_sha256,
                            "drifted": _attempt(dataclasses.replace(bound, kat_hosts_sha256="0" * 64).check_kat_hosts),
                            "unreadable": _attempt(dataclasses.replace(bound, kat_hosts=str(root / "absent.json"))
                                                   .check_kat_hosts),
                            "half filled": _attempt(dataclasses.replace(two, kat_hosts=str(hosts)).check_kat_hosts)}}


def _key(k: Any) -> str:
    if isinstance(k, str):
        return k
    if isinstance(k, bool | np.bool_ | int | np.integer):
        return str(int(k))
    return repr(k)


def _canon(x: Any, tmp: str) -> Any:
    """``x`` as plain JSON values, exactly: floats as floats (written by their repr), arrays with dtype and shape,
    dataclasses with their class, exceptions by class and message, the temporary directory's path replaced; any other
    type refused."""
    if isinstance(x, str):
        return x.replace(tmp, "<kat>")
    if x is None:
        return None
    if isinstance(x, bool | np.bool_):
        return bool(x)
    if isinstance(x, int | np.integer):
        return int(x)
    if isinstance(x, float | np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return {"dtype": str(x.dtype), "shape": list(x.shape), "values": [_canon(v, tmp) for v in x.ravel().tolist()]}
    if isinstance(x, dict):
        return {_key(k): _canon(v, tmp) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_canon(v, tmp) for v in x]
    if isinstance(x, BaseException):
        return {"raises": type(x).__name__, "message": _canon(str(x), tmp)}
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {"class": type(x).__qualname__, "fields": _canon(dataclasses.asdict(x), tmp)}
    raise TypeError(f"the known-answer payload holds a {type(x).__qualname__}, which it cannot serialise exactly")


def canonical(payload: dict[str, Any]) -> str:
    """A canonical payload's JSON text: sorted keys, no spaces, floats by their repr (NaN and infinities included)."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=True, ensure_ascii=True)


def known_answers() -> dict[str, str]:
    """The sha256 of each answer's canonical text: "calibrated" (the confirmatory path on singleton clusters),
    "grouping" and "multiplicity" (the weighted Holm, entries 74 and 76)."""
    return {"calibrated": hashlib.sha256(canonical(calibrated_answers()).encode("ascii")).hexdigest(),
            "grouping": hashlib.sha256(canonical(grouping_answers()).encode("ascii")).hexdigest(),
            "multiplicity": hashlib.sha256(canonical(multiplicity_answers()).encode("ascii")).hexdigest()}


def _openblas() -> dict[str, str]:
    """The OpenBLAS numpy loaded, as it runs: the kernel its DYNAMIC_ARCH chose for this CPU and its thread count, read
    through its own functions in the loaded library; "unknown" where they cannot be read."""
    out = {"blas core (run time)": "unknown", "blas threads (run time)": "unknown"}
    try:
        with open("/proc/self/maps", encoding="utf-8") as maps:
            paths = sorted({line.split()[-1] for line in maps if "openblas" in line.lower() and "/" in line})
        lib = ctypes.CDLL(paths[0])  # already loaded: the same library and state
    except (OSError, IndexError):
        return out
    for prefix in ("scipy_openblas_", "openblas_"):  # numpy's wheels' prefixed symbols, then plain OpenBLAS
        for suffix in ("64_", ""):
            core = getattr(lib, f"{prefix}get_corename{suffix}", None)
            threads = getattr(lib, f"{prefix}get_num_threads{suffix}", None)
            if core is not None and threads is not None:
                core.restype = ctypes.c_char_p
                return {"blas core (run time)": core().decode("ascii", "replace"),
                        "blas threads (run time)": str(int(threads()))}
    return out


def _cpu() -> dict[str, str]:
    """This machine's CPU model and its vector extensions (avx*, fma) as /proc/cpuinfo states them, else unknown."""
    model, flags = "unknown", "unknown"
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError:
        text = ""
    for line in text.splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "model name" and model == "unknown":
            model = value.strip()
        elif key.strip() == "flags" and flags == "unknown":
            flags = " ".join(sorted(f for f in value.split() if f.startswith(("avx", "fma"))))
    return {"cpu model": model, "cpu flags": flags}


def environment() -> dict[str, str]:
    """What the answers' last bits may depend on besides the code: the Python, numpy and scipy versions, numpy's BLAS as
    built (numpy.show_config) and as it runs (its kernel and thread count, from OpenBLAS itself), the SIMD extensions
    numpy found, the CPU model and its vector extensions, and the two thread variables, read now (the preflight
    refuses unless both are 1)."""
    try:
        scipy = importlib.metadata.version("scipy")
    except importlib.metadata.PackageNotFoundError:
        scipy = "not installed"
    config = np.show_config(mode="dicts") or {}
    blas = (config.get("Build Dependencies") or {}).get("blas") or {}
    simd = config.get("SIMD Extensions") or {}
    return {"python": platform.python_version(), "machine": platform.machine(), "numpy": np.__version__, "scipy": scipy,
            "blas (build)": f"{blas.get('name')} {blas.get('version')} ({blas.get('openblas configuration')})",
            **_openblas(), "simd found": " ".join(simd.get("found") or ()), **_cpu(),
            **{name: os.environ.get(name, "unset") for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")}}
