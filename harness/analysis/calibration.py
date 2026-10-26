"""The size calibration of every confirmatory claim (prereg §4.2, change-log entry 40).

**Procedure:**
- **Step 1** (N1 = 20,000 datasets per configuration). For each claim, each test (the primary
  bootstrap-t and the CR2 fallback, computed on the same datasets) and each Holm level a = 0.05/k
  (k = 1..6), alpha'(a) is the largest value on the grid j x 0.0005 a whose simulated size, in the
  least-favourable configuration among those where the claim is null, is at most a.
  - **Grid endpoint:** j <= ``GRID_STEPS`` (4,000), so alpha'(a) <= 2a; prereg §4.2 names the step, not the
    endpoint. This is our reading, to confirm before the run. The values capped at a (j <= 2,000) are reported
    beside it as a diagnostic.
  - **Monotonicity:** Holm needs alpha' non-decreasing in a (prereg.calibrated_p). From the largest level down,
    alpha'(a) is the largest value on a's own grid at most min(its own value, alpha' of the next larger level): the
    largest values on each level's own grid that are non-decreasing in a, each with step-1 size <= a.
- **Two candidate families** (design log ruling 2): (C1a two-family, C1b, C2, C3, C4, C5) and (C1a pooled, C1b, C2,
  C3, C4, C5). Both are calibrated and checked; prereg.choose_c1a later picks the one that enters Holm.
- **The tests, from step 1 only** (change-log entry 46): a claim takes the primary if its step-1 size at alpha'(a) is
  <= a at every level in every null configuration, else the fallback by the same rule, else none (p = 1).
- **Joint step 1** (ruling 3), per family: with those tests, the six-claim FWER on the step-1 datasets must be <= 0.05
  in every configuration. When it is not, EVERY alpha' of the family is multiplied by one common factor c = k / 1000,
  the largest k in 1..1000 for which it is (``joint_step_one``); when no k reaches it, the family fails. (With the
  tests chosen as above, Holm's union bound keeps the step-1 FWER <= 0.05, so c = 1 in practice.)
- **Step 2** (a fresh N2 = 100,000 datasets per configuration, another seed), per family, verifies only.
  - **Per claim:** with its scaled alpha'(a), the size must be <= a + 2 SE at every level in every null
    configuration, where SE = sqrt(a (1 - a) / N2), the binomial SE at a with N2 (our reading of "SE"). A test that
    fails gives the claim p = 1, with no swap to the other test and no new factor.
  - **Joint:** the six-claim FWER must be <= 0.05 + 2 sqrt(0.05 x 0.95 / N2) in every configuration. The FWER is Holm
    over the six calibrated p's, with claims without a test at p = 1, counting rejections of claims null in that
    configuration. Nothing is adjusted after this check: a chosen family that fails it gives every claim p = 1
    (prereg.load_calibration).

**Configurations** (``configurations``, the registered list). Every one simulates all six claims:
- least-favourable partial nulls with the other claims strongly alternative;
- heterogeneous per-question bank rates and within-question correlation (Beta rates at ICC 0.2
  and 0.5, on the real provisional bank's per-question structure);
- rates near 0 and 1;
- rare discordance;
- sparse or concentrated denominators;
- missingness;
- for C5 and the pooled C1a, heterogeneous per-model effects averaging zero, and effects
  correlated with the flag rates (both signs), with dependence across models and repetitions.

The panel P is a placeholder of seven models, the two families first, until P is fixed at the
freeze. The pooled C1a is calibrated beside the two-family C1a, as the "C1a pooled" claim, so
that the §1 [FILL] can be decided either way. G0 is the joint null of the two-family family, G0p of
the pooled one; E1 and E2 have C1a and C1b both alternative (both variants), so that both can advance
Holm, with C3 null and C2, C4, C5 alternative; F1-F7 have each other claim alone null, with the other
five alternative, in both families.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import Any

import numpy as np

from harness.analysis import prereg as pr
from harness.analysis.simulate import (
    Bank,
    BankStructure,
    Scenario,
    SourceRates,
    dataset_vectors,
    generate,
    is_null,
    truths,
)
from harness.analysis.stats import QuestionIndex

CLAIMS = ("C1a", "C1a pooled", "C1b", "C2", "C3", "C4", "C5")
FAMILY = pr.FAMILY_CLAIMS  # the two candidate families (design log ruling 2)
FACTOR_STEPS = 1000  # ruling 3: the common step-1 factor's grid, k / 1000 for k = 1..1000
N1, N2 = 20_000, 100_000  # §4.2
SEEDS = {1: 20261101, 2: 20261102}  # registered: one per step, never the analysis's 20261026
# every dataset's generator (dataset_p); each chunk records it with its step and seed, and the aggregation checks them
SEED_SCHEME = "numpy default_rng([step seed, configuration index, dataset index])"
# alpha' = j * 0.0005 a for j = 1..GRID_STEPS: the registered grid's upper endpoint is 2a (prereg §4.2 names the step,
# not the endpoint; step 2's fresh check at a + 2 SE guards a conservative test's alpha' above a). Our reading, to
# confirm before the run; the values capped at a are reported beside it (CAPPED_STEPS)
GRID_STEPS = 4000
CAPPED_STEPS = 2000
DIVISORS = (1, 2, 3, 4, 5, 6)
PANEL = ("Qwen/Qwen3.8-27B", "claude-sonnet-5-5", "panel-3", "panel-4", "panel-5", "panel-6", "panel-7")
PANEL_FAMILIES = ("qwen", "claude", "f3", "f4", "f5", "f6", "f7")
STRONG_CURATED = SourceRates(0.8, 0.8, 0.02)  # the default bank: C2 strongly true (recall +30 points, f -18 points)
STRONG_CENSUS = SourceRates(0.5, 0.5, 0.20)


def simulation_config() -> pr.PreregConfig:
    """The registered analysis with the placeholder panel and both C1a variants available."""
    return pr.PreregConfig(scorer_version="simulation", panel=PANEL, panel_families=PANEL_FAMILIES, c1a_family=pr.POOLED)


@dataclass(frozen=True)
class Configuration:
    name: str
    scenario: Scenario
    tags: tuple[str, ...]


def configurations(structure: BankStructure | None) -> list[Configuration]:
    """The registered configuration list (§4.2). ``structure`` is the bank's per-question structure (the real
    provisional bank in the registered run; None gives the uniform test structure)."""
    strong_c1 = (0.15,) * 7
    strong_c5 = (0.25,) * 7

    def bank(cur=None, cen=None, masked=(0.10, 0.10, 0.10), icc=0.2):
        cur, cen = cur or STRONG_CURATED, cen or STRONG_CENSUS
        return Bank(structure=structure, curated=cur, census=cen,
                    masked=tuple(SourceRates(0.6, 0.6, f) for f in masked), icc=icc)

    def sc(name, tags, c1=strong_c1, c5=strong_c5, b=None, **kw):
        return Configuration(name, Scenario(name, c1=c1, c5=c5, bank=b or bank(), **kw), tags)

    het0 = (-0.06, 0.06, -0.03, 0.03, 0.0, -0.02, 0.02)
    c5_het0 = (-0.10, 0.10, -0.05, 0.05, 0.0, -0.08, 0.08)
    c5_corr = (0.10, 0.10, 0.05, 0.0, -0.05, -0.10, -0.10)  # high flag rates, negative effects
    rates_corr = (0.05, 0.05, 0.10, 0.10, 0.15, 0.20, 0.20)

    def c4_null(f_cur, f_cen, r_cur=0.5):
        r_cen = 3 * f_cen * (1 - f_cur) / (1 - r_cur)  # 0.25 r_cen (1 - r_cur) = 0.75 f_cen (1 - f_cur)
        return bank(cur=SourceRates(r_cur, r_cur, f_cur), cen=SourceRates(r_cen, r_cen, f_cen))

    def c4_null_census(f_cur, f_cen, r_cen=0.5):  # the mirror: Delta over census at 0, over curated strongly positive
        r_cur = 3 * f_cur * (1 - f_cen) / (1 - r_cen)
        return bank(cur=SourceRates(r_cur, r_cur, f_cur), cen=SourceRates(r_cen, r_cen, f_cen))

    def alone_alt():  # C2 (both components), C3 and C4 (both comparisons) alternative: the alone-null configurations
        return bank(cur=SourceRates(0.55, 0.55, 0.002), cen=SourceRates(0.45, 0.45, 0.03), masked=(0.04, 0.04, 0.04))

    inside = (0.0,) * 7  # C1a (both variants) null at 0 and C1b strictly inside its margins: C1b can advance Holm
    concentrated = None
    if structure is not None:
        n = len(structure.rewrites)
        k = tuple(3 if i % 10 == 0 else 0 for i in range(n))  # all rewrites on a tenth of the questions
        concentrated = replace(structure, rewrites=k)

    return [
        # the joint least-favourable configuration: C1a, C1b, C2, C3, C4 and C5 all null
        sc("G0 global null", ("joint",), c1=(0.0,) + (0.15,) * 6, c5=c5_het0, corrupt=0.2,
           b=replace(c4_null(0.02, 0.02), masked=(SourceRates(0.6, 0.6, 0.02),) * 3)),
        # the pooled family's joint null: C1 averaging zero over P with the second family at +0.15 (so the two-family
        # C1a and C1b are null too), everything else as in G0
        sc("G0p global null, pooled C1a", ("joint", "C1a pooled"), c1=(0.0, 0.15, -0.15, 0.0, 0.0, 0.0, 0.0), c5=c5_het0,
           corrupt=0.2, b=replace(c4_null(0.02, 0.02), masked=(SourceRates(0.6, 0.6, 0.02),) * 3)),
        # C1a: one family at 0, the other strongly on one side
        sc("A1 C1a LF up", ("C1a",), c1=(0.0, 0.15) + (0.15,) * 5),
        sc("A2 C1a LF down", ("C1a",), c1=(0.0, -0.15) + (-0.15,) * 5),
        sc("A3 C1a LF up, accuracy 0.5, ICC 0.6", ("C1a", "endpoint"), c1=(0.0, 0.15) + (0.15,) * 5, accuracy=0.5, icc=0.6),
        sc("A4 C1a LF up, rare discordance (accuracy 0.9, ICC 0.9)", ("C1a", "rare discordance"),
           c1=(0.0, 0.08) + (0.08,) * 5, accuracy=0.9, icc=0.9),
        sc("A5 C1a LF up, missingness 3%", ("C1a", "missingness"), c1=(0.0, 0.15) + (0.15,) * 5, missing=0.03,
           retry_missing=0.05),
        # pooled C1a: heterogeneous per-model differences averaging zero, with dependence across models
        sc("P1 pooled C1a, heterogeneous zero mean", ("C1a pooled", "heterogeneous"), c1=het0),
        sc("P2 pooled C1a, zero mean, cross-model dependence 0.8, ICC 0.6", ("C1a pooled", "dependence"), c1=het0,
           cross_model=0.8, icc=0.6),
        sc("P3 pooled C1a, zero mean, accuracy 0.8, missingness 3%", ("C1a pooled", "missingness", "endpoint"), c1=het0,
           accuracy=0.8, missing=0.03),
        # C1b: one family at a margin, the other at 0
        sc("B1 C1b at +3", ("C1b",), c1=(0.03, 0.0) + (0.0,) * 5),
        sc("B2 C1b at -3", ("C1b",), c1=(-0.03, 0.0) + (0.0,) * 5),
        sc("B3 C1b at +3, accuracy 0.8, ICC 0.6", ("C1b", "endpoint"), c1=(0.03, 0.0) + (0.0,) * 5, accuracy=0.8, icc=0.6),
        sc("B4 C1b at -3, missingness 3%", ("C1b", "missingness"), c1=(-0.03, 0.0) + (0.0,) * 5, missing=0.03),
        # C2: each component at its boundary with the other strongly true
        sc("C21 C2 (i) at 0, f = 0.02", ("C2",), c1=inside, b=bank(cur=SourceRates(0.8, 0.8, 0.02), cen=SourceRates(0.5, 0.5, 0.02))),
        sc("C22 C2 (i) at 0, f = 0.005 (rates near 0)", ("C2", "near 0"),
           b=bank(cur=SourceRates(0.8, 0.8, 0.005), cen=SourceRates(0.5, 0.5, 0.005))),
        sc("C23 C2 (i) at 0, f = 0.10, bank ICC 0.5", ("C2", "concentrated"),
           b=bank(cur=SourceRates(0.8, 0.8, 0.10), cen=SourceRates(0.5, 0.5, 0.10), icc=0.5)),
        sc("C24 C2 (ii) at 0, recall 0.5", ("C2",), c1=inside, b=bank(cur=SourceRates(0.5, 0.5, 0.02), cen=SourceRates(0.5, 0.5, 0.2))),
        sc("C25 C2 (ii) at 0, recall 0.95 (near 1)", ("C2", "near 1"),
           b=bank(cur=SourceRates(0.95, 0.95, 0.02), cen=SourceRates(0.95, 0.95, 0.2))),
        sc("C26 C2 (ii) at 0, recall 0.05 (near 0), bank ICC 0.5", ("C2", "near 0", "concentrated"),
           b=bank(cur=SourceRates(0.05, 0.05, 0.02), cen=SourceRates(0.05, 0.05, 0.2), icc=0.5)),
        # C3: curated equal to the mean of the draws
        sc("C31 C3 at 0, f = 0.02", ("C3",), c1=inside, b=bank(cur=SourceRates(0.8, 0.8, 0.02), masked=(0.02, 0.02, 0.02))),
        sc("C32 C3 at 0, f = 0.005 (rates near 0)", ("C3", "near 0"),
           b=bank(cur=SourceRates(0.8, 0.8, 0.005), masked=(0.005, 0.005, 0.005))),
        sc("C33 C3 at 0, unequal draws (0, 0.02, 0.04)", ("C3", "sparse"),
           b=bank(cur=SourceRates(0.8, 0.8, 0.02), masked=(0.0, 0.02, 0.04))),
        sc("C34 C3 at 0, f = 0.10, bank ICC 0.5", ("C3", "concentrated"),
           b=bank(cur=SourceRates(0.8, 0.8, 0.10), masked=(0.10, 0.10, 0.10), icc=0.5)),
        # C4: one comparison at Delta = 0, the other strongly positive
        sc("C41 C4 at 0 over curated (f 0.02, 0.05)", ("C4",), c1=inside, b=c4_null(0.02, 0.05)),
        sc("C42 C4 at 0 over curated, rates near 0 (f 0.005, 0.01)", ("C4", "near 0"), b=c4_null(0.005, 0.01)),
        sc("C43 C4 at 0 over curated (f 0.05, 0.10), bank ICC 0.5", ("C4", "concentrated"),
           b=replace(c4_null(0.05, 0.10), icc=0.5)),
        sc("C44 C4 at 0 over census (f 0.05, 0.02)", ("C4", "mirror"), b=c4_null_census(0.05, 0.02)),
        sc("C45 C4 at 0 over census, rates near 0 (f 0.01, 0.005)", ("C4", "mirror", "near 0"),
           b=c4_null_census(0.01, 0.005)),
        sc("C27 C2 (i) at 0, f = 0.02, rewrites concentrated on a tenth of the questions", ("C2", "concentrated"),
           b=replace(bank(cur=SourceRates(0.8, 0.8, 0.02), cen=SourceRates(0.5, 0.5, 0.02)),
                     structure=concentrated or BankStructure.uniform(210, rewrites=0))),
        # C5: pooled zero mean, heterogeneous, correlated with flag rates, sparse, rare discordance, dependence
        sc("D1 C5 all zero, independent retries", ("C5",), c1=inside, c5=(0.0,) * 7),
        sc("D2 C5 heterogeneous zero mean, corruption 0.2", ("C5", "heterogeneous"), c1=inside, c5=c5_het0, corrupt=0.2),
        sc("D3 C5 zero mean, effects falling with flag rate", ("C5", "correlated"), c5=c5_corr,
           panel_flag_rates=rates_corr, corrupt=0.2),
        sc("D4 C5 zero mean, effects rising with flag rate", ("C5", "correlated"), c5=tuple(reversed(c5_corr)),
           panel_flag_rates=rates_corr, corrupt=0.2),
        sc("D5 C5 zero mean, sparse flags (2%)", ("C5", "sparse"), c5=c5_het0, corrupt=0.2,
           panel_flag_rates=(0.02,) * 7, flag_rate=(0.02, 0.02), accuracy=0.8, icc=0.6),
        sc("D6 C5 zero mean, retry coupling 0.9 (rare discordance)", ("C5", "rare discordance"), c5=c5_het0,
           corrupt=0.2, retry_coupling=0.9),
        sc("D7 C5 zero mean, cross-model 0.8, effect spread 0.05", ("C5", "dependence"), c5=c5_het0, corrupt=0.2,
           cross_model=0.8, c5_spread=0.05, icc=0.6),
        sc("D8 C5 zero mean, missingness (3%, retries 5%)", ("C5", "missingness"), c5=c5_het0, corrupt=0.2,
           missing=0.03, retry_missing=0.05),
        # C1a and C1b both alternative (every C1 effect inside C1b's margins and away from 0, for the two-family and the
        # pooled C1a), so that both can advance Holm, with C3 null and C2, C4 and C5 alternative (second review)
        sc("E1 C1a and C1b alternative (+0.02 for every model), C3 null at f = 0.002", ("C3", "C1 alternative", "near 0"),
           c1=(0.02,) * 7, b=bank(cur=SourceRates(0.55, 0.55, 0.002), cen=SourceRates(0.45, 0.45, 0.03),
                                  masked=(0.002, 0.002, 0.002))),
        sc("E2 C1a and C1b alternative (heterogeneous over P, mean +0.02), C3 null at f = 0.02",
           ("C3", "C1 alternative", "heterogeneous"), c1=(0.02, 0.025, 0.06, -0.02, 0.0, 0.05, 0.005),
           b=bank(cur=SourceRates(0.55, 0.55, 0.02), cen=SourceRates(0.45, 0.45, 0.045), masked=(0.02, 0.02, 0.02))),
        # every other claim alone null, with the other five alternative, in both families (design log). The alternatives as
        # in E1: every C1 effect +0.02 (C1a, pooled C1a and C1b true), curated f = 0.002 against census 0.03 (C2 (i)
        # +2.8 points), recall 0.55 against 0.45 (C2 (ii) +10), C4's Delta +0.028 and +0.074, masked f = 0.04 (C3
        # -3.8 points), C5 +0.25. Each null at its least-favourable point, as for the existing configurations:
        # - C1a at zero: the first family at 0 and the second at +0.025 (inside C1b's margins), the pooled mean 0;
        sc("F1 C1a alone null (both variants: first family 0, second +0.025, pooled mean 0)", ("C1a", "C1a pooled", "alone"),
           c1=(0.0, 0.025, -0.025, 0.0, 0.0, 0.0, 0.0), b=alone_alt()),
        # - C1b at a margin, each sign, with C1a true in the same direction;
        sc("F2 C1b alone null at +3 (other models +0.02)", ("C1b", "alone"), c1=(0.03,) + (0.02,) * 6, b=alone_alt()),
        sc("F3 C1b alone null at -3 (other models -0.02)", ("C1b", "alone"), c1=(-0.03,) + (-0.02,) * 6, b=alone_alt()),
        # - C2's IUT at the boundary of each component with the other strongly true;
        sc("F4 C2 alone null, (i) at 0 (f = 0.02 both), (ii) +30 points", ("C2", "alone"), c1=(0.02,) * 7,
           b=bank(cur=SourceRates(0.7, 0.7, 0.02), cen=SourceRates(0.4, 0.4, 0.02), masked=(0.05, 0.05, 0.05))),
        sc("F5 C2 alone null, (ii) at 0 (recall 0.5 both), (i) +4.8 points", ("C2", "alone"), c1=(0.02,) * 7,
           b=bank(cur=SourceRates(0.5, 0.5, 0.002), cen=SourceRates(0.5, 0.5, 0.05), masked=(0.04, 0.04, 0.04))),
        # - C4 at Delta = 0 over curated, the comparison that binds while C2 is true (Delta over census = 0 with Delta over
        #   curated > 0 needs r_census > r_curated, so C2 (ii) false);
        sc("F6 C4 alone null at Delta = 0 over curated (f 0.002, 0.03)", ("C4", "alone"), c1=(0.02,) * 7,
           b=replace(c4_null(0.002, 0.03, r_cur=0.6), masked=tuple(SourceRates(0.6, 0.6, 0.04) for _ in range(3)))),
        # - C5 with heterogeneous per-model effects averaging zero.
        sc("F7 C5 alone null, heterogeneous zero mean, corruption 0.2", ("C5", "alone", "heterogeneous"), c1=(0.02,) * 7,
           c5=c5_het0, corrupt=0.2, b=alone_alt()),
    ]


# --- step 3 (prereg §4.2, change-log entry 106) ----------------------------------------------------------------------

STEP3_SEEDS = pr.STEP3_SEEDS  # registered in prereg (and the config's step3_seeds)
N3_SELECTION, N3 = 100_000, 100_000  # datasets per configuration: the selection run's and the validation's (§4.2)
STEP3_MIRRORED = ("C21", "C22", "C23", "C24", "C25", "C26", "C27", "F4", "F5")  # C2's one-sided least favourable
# The dev bank's rates (the dev bank replay's ledger on fix/bank-replay 4932e43, sha256 2cf7321c...; computed by the
# analysis's own code at 4932e43, results/power-weighted-20261020/c2-deflated/dev-bank-rates.json): per source the
# type-wrong recall, the other-wrong recall and f (prereg.source_rates' per-question means)
DEV_BANK = {"curated": SourceRates(0.5651515151515152, 0.7155225726654297, 0.15955473098330242),
            "census": SourceRates(0.7886518243661101, 0.7167594310451453, 0.0686456400742115),
            "generated-dc-1": SourceRates(0.49922696351267787, 0.6910946196660482, 0.0),
            "generated-dc-2": SourceRates(0.5184291898577612, 0.6910946196660482, 0.0),
            "generated-dc-3": SourceRates(0.49922696351267787, 0.6910946196660482, 0.0)}


def step3_configurations(structure: BankStructure | None) -> list[Configuration]:
    """Step 3's configuration list: steps 1 and 2's (``configurations``, unchanged, in order), then C2's least
    favourable configurations for the census direction (each of STEP3_MIRRORED with the curated and census sources
    swapped, so both of C2's components change sign and nothing else of the bank does), then four at the dev bank's
    baselines (DEV_BANK; the generated drafts at their dev rates; the other claims strongly true as in the C2
    configurations): one component at zero at the midpoint of the two sources' dev values, the other at its dev
    difference, census better (DV1, DV2) and curated better (DV3, DV4)."""
    base = configurations(structure)
    out = list(base)
    by_id = {c.name.split()[0]: c for c in base}
    for m in STEP3_MIRRORED:
        c = by_id[m]
        b = c.scenario.bank
        name = f"M{m} {c.name.split(' ', 1)[1]}, mirrored: curated <-> census (census better)"
        out.append(Configuration(name, replace(c.scenario, name=name, bank=replace(b, curated=b.census, census=b.curated)),
                                 (*c.tags, "census direction")))
    cur, cen = DEV_BANK["curated"], DEV_BANK["census"]
    masked = tuple(DEV_BANK[s] for s in ("generated-dc-1", "generated-dc-2", "generated-dc-3"))
    f0, r0 = (cur.fpr + cen.fpr) / 2, (cur.type_wrong + cen.type_wrong) / 2

    def dev(name: str, c: SourceRates, n: SourceRates) -> Configuration:
        return Configuration(name, Scenario(name, c1=(0.15,) * 7, c5=(0.25,) * 7,
                                            bank=Bank(structure=structure, curated=c, census=n, masked=masked, icc=0.2)),
                             ("C2", "dev baselines"))

    return out + [
        dev("DV1 C2 (i) at 0 at the dev midpoint f, (ii) census better as on dev",
            replace(cur, fpr=f0), replace(cen, fpr=f0)),
        dev("DV2 C2 (ii) at 0 at the dev midpoint type-wrong recall, (i) census better as on dev",
            replace(cur, type_wrong=r0), replace(cen, type_wrong=r0)),
        dev("DV3 C2 (i) at 0 at the dev midpoint f, (ii) curated better (dev difference reversed)",
            replace(cur, fpr=f0, type_wrong=cen.type_wrong), replace(cen, fpr=f0, type_wrong=cur.type_wrong)),
        dev("DV4 C2 (ii) at 0 at the dev midpoint type-wrong recall, (i) curated better (dev difference reversed)",
            replace(cur, type_wrong=r0, fpr=cen.fpr), replace(cen, type_wrong=r0, fpr=cur.fpr)),
    ]


def structure_from_record(record: Mapping[str, Any]) -> BankStructure:
    """A calibration's bank structure from its structure.json (calibrate.py ``_record``: per question W, C = gold +
    answer-neutral, T and k), exactly as it was simulated: type-wrong T, other wrong W - T, neutral C - 1, rewrites k."""
    rows = record["questions"]
    return BankStructure(tuple(int(r["T"]) for r in rows), tuple(int(r["W"]) - int(r["T"]) for r in rows),
                         tuple(int(r["C"]) - 1 for r in rows), tuple(int(r["k"]) for r in rows))


def step3_configuration_list(structure: BankStructure, n: int) -> list[dict[str, Any]]:
    """Step 3's configurations as the producers and the addendum record them: each one's name and its null claims
    (C2 two-sided), in step3_configurations' order (``n``: the analysis's questions, as setup reads truths)."""
    cfg, out = simulation_config(), []
    for c in step3_configurations(structure):
        nl = nulls(c.scenario, cfg, n, two_sided_c2=True)
        out.append({"name": c.name, "null": [k for k in CLAIMS if nl[k]]})
    return out


# --- one dataset ---------------------------------------------------------------------------------


def dataset_p(sc: Scenario, index: QuestionIndex, K: np.ndarray, cfg: pr.PreregConfig, seed: int, conf: int,
              i: int, K_bank: np.ndarray | None = None) -> np.ndarray:
    """(claims x tests) p-values of one simulated dataset, every claim by the registered code (prereg); ``K_bank`` is
    the resample matrix of a bank with more questions than the agent runs (a bank-only extension, entry 41). The
    generator is SEED_SCHEME's."""
    ds = generate(sc, index, cfg, np.random.default_rng([seed, conf, i]))
    V = dataset_vectors(ds, index, cfg)
    Kb = K if K_bank is None else K_bank
    res = {"C1a": pr.c1a(V, K, replace(cfg, c1a_family=pr.TWO_FAMILY)), "C1a pooled": pr.c1a(V, K, cfg),
           "C1b": pr.c1b(V, K, cfg), "C2": pr.c2(V, Kb, cfg), "C3": pr.c3(V, Kb, cfg), "C4": pr.c4(V, Kb, cfg),
           "C5": pr.c5(V, K, cfg, descriptive=False)}
    return np.array([[res[c]["p_test"][t] for t in pr.TESTS] for c in CLAIMS])


def nulls(sc: Scenario, cfg: pr.PreregConfig, n: int, two_sided_c2: bool = False) -> dict[str, bool]:
    """Which claims are null in a configuration (``cfg`` has the pooled C1a, so t["C1a"] is the pooled truth); C2 by
    its two-sided claim with ``two_sided_c2`` (step 3, entry 106), by the one-sided one steps 1 and 2 recorded without."""
    t = truths(sc, cfg, n)
    out = {c: is_null(c, t[c], cfg, two_sided_c2) for c in ("C1b", "C2", "C3", "C4", "C5")}
    out["C1a pooled"] = is_null("C1a", t["C1a"], cfg)
    out["C1a"] = is_null("C1a", dict(zip(cfg.models, sc.c1[:2])), cfg)
    return out


# --- step 1 and step 2 ---------------------------------------------------------------------------


def _grid_alpha(p: np.ndarray, a: float, steps: int) -> float:
    """The largest j * 0.0005 a (j <= steps) with mean(p <= alpha') <= a; 0 when none."""
    grid = np.arange(1, steps + 1) * 0.0005 * a
    sizes = np.searchsorted(np.sort(p), grid, side="right") / len(p)
    ok = np.nonzero(sizes <= a)[0]
    return float(grid[ok[-1]]) if len(ok) else 0.0


def _snap(x: float, a: float, steps: int) -> float:
    """The largest value j x 0.0005 a (j <= steps) at most x; 0 when none."""
    grid = np.arange(1, steps + 1) * 0.0005 * a
    i = int(np.searchsorted(grid, x, side="right")) - 1
    return float(grid[i]) if i >= 0 else 0.0


def step_one(p_by_conf: Sequence[np.ndarray], null_by_conf: Sequence[dict[str, bool]], alpha: float = 0.05) -> dict:
    """alpha'(a) per claim, test and Holm level from step-1 p-values (one (N1 x claims x tests) array per
    configuration). Each level's own value is the largest value on its grid whose size is <= a in every configuration
    where the claim is null. Holm needs alpha' non-decreasing in a (prereg.calibrated_p), so from the largest level
    down alpha'(a) is the largest value on a's own grid at most min(own value, alpha' of the next larger level): the
    largest values on each level's own grid that are non-decreasing in a, each with step-1 size <= a."""
    out: dict[str, Any] = {}
    for ci, claim in enumerate(CLAIMS):
        confs = [p[:, ci, :] for p, nl in zip(p_by_conf, null_by_conf) if nl[claim]]
        entry = {"null_configurations": sum(nl[claim] for nl in null_by_conf)}
        for ti, test in enumerate(pr.TESTS):
            own = {k: min([_grid_alpha(c[:, ti], alpha / k, GRID_STEPS) for c in confs] or [alpha / k]) for k in DIVISORS}
            chosen, upper = {}, math.inf
            for k in sorted(DIVISORS):  # the largest level first
                upper = _snap(min(own[k], upper), alpha / k, GRID_STEPS)
                chosen[str(k)] = upper
            capped = {str(k): min([_grid_alpha(c[:, ti], alpha / k, CAPPED_STEPS) for c in confs] or [alpha / k])
                      for k in DIVISORS}
            entry[test] = {"alpha_prime": dict(sorted(chosen.items())), "own": {str(k): v for k, v in own.items()},
                           "capped_at_a_diagnostic": capped}
        out[claim] = entry
    return out


def _se(a: float, n: int) -> float:
    return math.sqrt(a * (1 - a) / n)


# --- the joint FWER (design log ruling 3, change-log entry 46) -------------------------------------


def calibrated_ps(x: np.ndarray, alpha_prime: Mapping[int, float] | None, alpha: float = 0.05) -> np.ndarray:
    """prereg.calibrated_p over an array of test p's: the smallest Holm level a = alpha / k with x <= alpha'(a),
    else 1; alpha' falling with the level is refused, as there."""
    out = np.ones(len(x))
    if alpha_prime is None:
        return out
    levels = [alpha_prime[k] for k in pr.HOLM_DIVISORS]
    if any(b < a for a, b in pairwise(levels)):
        raise pr.PreregError(f"alpha' must be non-decreasing in the Holm level, got {levels} for a = alpha/6 .. alpha")
    for k in reversed(pr.HOLM_DIVISORS):  # the largest level first; each smaller level it reaches overwrites
        out = np.where(x <= alpha_prime[k], alpha / k, out)
    return out


def holm_rejects(P: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """stats.holm on every row of P (datasets x claims), as the matrix of Holm-adjusted p <= alpha: the same stable
    order, multiply-then-compare and running maximum, so it decides exactly as the registered code does."""
    m = P.shape[1]
    order = np.argsort(P, axis=1, kind="stable")
    ordered = np.take_along_axis(P, order, axis=1)
    running = np.maximum.accumulate(np.minimum(1.0, (m - np.arange(m)) * ordered), axis=1)
    adjusted = np.empty_like(P)
    np.put_along_axis(adjusted, order, running, axis=1)
    return adjusted <= alpha


def family_fwer(p_by_conf: Sequence[np.ndarray], null_by_conf: Sequence[dict[str, bool]], members: Sequence[str],
                tests: Mapping[str, str | None], alpha_primes: Mapping[str, Mapping[int, float]],
                alpha: float = 0.05) -> list[float]:
    """The six-claim FWER of a family in each configuration: the share of datasets in which Holm over the members'
    calibrated p's (a claim without a test at p = 1) rejects any claim that is null in that configuration."""
    rates = []
    for p, nl in zip(p_by_conf, null_by_conf):
        null_cols = [j for j, c in enumerate(members) if nl[c]]
        if not null_cols:
            rates.append(0.0)
            continue
        P = np.ones((len(p), len(members)))
        for j, c in enumerate(members):
            if tests[c] is not None:
                P[:, j] = calibrated_ps(p[:, CLAIMS.index(c), pr.TESTS.index(tests[c])], alpha_primes[c], alpha)
        rates.append(float(holm_rejects(P, alpha)[:, null_cols].any(axis=1).mean()))
    return rates


def _scale(alpha_prime: Mapping[int, float], factor: float) -> dict[int, float]:
    return {k: v * factor for k, v in alpha_prime.items()}


def joint_step_one(p_by_conf: Sequence[np.ndarray], null_by_conf: Sequence[dict[str, bool]], members: Sequence[str],
                   tests: Mapping[str, str | None], alpha_primes: Mapping[str, Mapping[int, float]],
                   alpha: float = 0.05) -> dict[str, Any]:
    """Ruling 3 on the step-1 datasets: the largest k in 1..FACTOR_STEPS for which, with EVERY alpha' of the family
    (``alpha_primes``, of each claim's test) multiplied by c = k / FACTOR_STEPS, the six-claim FWER is <= alpha in every
    configuration (k = FACTOR_STEPS, c = 1, when the unscaled levels already hold it; None when even k = 1 does not).
    The FWER is non-decreasing in k (calibrated_p is non-increasing in alpha', Holm is monotone in each p), so bisection
    finds the first k from the top at which it holds."""
    def fwer(k: int) -> list[float]:
        c = k / FACTOR_STEPS
        return family_fwer(p_by_conf, null_by_conf, members, tests,
                           {m: _scale(v, c) for m, v in alpha_primes.items()}, alpha)

    unscaled = fwer(FACTOR_STEPS)
    if max(unscaled, default=0.0) <= alpha:
        k = FACTOR_STEPS
    elif max(fwer(1), default=0.0) > alpha:
        k = None
    else:
        lo, hi = 1, FACTOR_STEPS  # fwer(lo) holds, fwer(hi) does not
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if max(fwer(mid), default=0.0) <= alpha:
                lo = mid
            else:
                hi = mid
        k = lo
    return {"k": k, "factor": None if k is None else k / FACTOR_STEPS, "fwer_unscaled": unscaled,
            "fwer": None if k is None else (unscaled if k == FACTOR_STEPS else fwer(k))}


def sizes_ok(p_by_conf: Sequence[np.ndarray], null_by_conf: Sequence[dict[str, bool]], claim: str, test: str,
             alpha_prime: Mapping[int, float], alpha: float = 0.05) -> bool:
    """Entry 46's step-1 rule: the test's size on the step-1 datasets at alpha'(a) is <= a at every level, in every
    configuration where the claim is null."""
    ci, ti = CLAIMS.index(claim), pr.TESTS.index(test)
    return all(float(np.mean(p[:, ci, ti] <= alpha_prime[k])) <= alpha / k
               for p, nl in zip(p_by_conf, null_by_conf) if nl[claim] for k in DIVISORS)


def _levels(alphas: dict, members: Sequence[str]) -> dict[str, dict[str, dict[int, float]]]:
    return {c: {t: {int(k): float(v) for k, v in alphas[c][t]["alpha_prime"].items()} for t in pr.TESTS} for c in members}


def family_step_one(members: Sequence[str], p1: Sequence[np.ndarray], null_by_conf: Sequence[dict[str, bool]],
                    alphas: dict, alpha: float = 0.05) -> dict[str, Any]:
    """A family's step 1 (change-log entry 46, ruling 3), from step-1 data only: each claim's test is the primary if its
    step-1 size at alpha'(a) is <= a at every level (``sizes_ok``), else the fallback by the same rule, else none
    (p = 1); then the common factor from the step-1 FWER with those tests (``joint_step_one``)."""
    levels = _levels(alphas, members)
    ok = {c: {t: sizes_ok(p1, null_by_conf, c, t, levels[c][t], alpha) for t in pr.TESTS} for c in members}
    tests = {c: next((t for t in pr.TESTS if ok[c][t]), None) for c in members}
    joint = joint_step_one(p1, null_by_conf, members, tests,
                           {c: levels[c][t] for c, t in tests.items() if t is not None}, alpha)
    return {"tests": tests, "step1_size_ok": ok, **joint}


def size_checks(p_by_conf: Sequence[np.ndarray], null_by_conf: Sequence[dict[str, bool]], names: Sequence[str],
                claim: str, test: str, alpha_prime: Mapping[int, float], alpha: float = 0.05) -> list[dict[str, Any]]:
    """Step 2's per-claim check: the size at alpha'(a) against a + 2 SE at every level in every null configuration."""
    ci, ti = CLAIMS.index(claim), pr.TESTS.index(test)
    checks = []
    for p, nl, name in zip(p_by_conf, null_by_conf, names):
        if not nl[claim]:
            continue
        for k in DIVISORS:
            a = alpha / k
            size = float(np.mean(p[:, ci, ti] <= alpha_prime[k]))
            limit = a + 2 * _se(a, len(p))
            checks.append({"configuration": name, "level": a, "alpha_prime": alpha_prime[k], "size": size,
                           "limit": limit, "pass": size <= limit})
    return checks


def _json_levels(alpha_prime: Mapping[int, float] | None) -> dict[str, float] | None:
    return None if alpha_prime is None else {str(k): float(v) for k, v in alpha_prime.items()}


def calibrate_family(members: Sequence[str], p1: Sequence[np.ndarray], p2: Sequence[np.ndarray],
                     null_by_conf: Sequence[dict[str, bool]], names: Sequence[str], alphas: dict,
                     alpha: float = 0.05) -> dict[str, Any]:
    """One candidate family (rulings 2-3, change-log entry 46) from step 1's per-claim alpha' (``alphas``,
    step_one's output) and the step-1 and step-2 p-values of the same configurations.

    Step 1 alone chooses each claim's test and the family's common factor (``family_step_one``). Step 2 only verifies:
    each claim's test at its scaled alpha' must pass the size check (a + 2 SE at every level), and a test that fails
    gives that claim p = 1, with no swap to the other test and no new factor; then the family FWER (<= alpha + 2 SE,
    every configuration), once, with nothing adjusted after it (a failed check gives every claim p = 1 when the family
    is chosen, prereg.load_calibration). Nothing from step 2 reaches step 1's choices."""
    step1 = family_step_one(members, p1, null_by_conf, alphas, alpha)
    levels = _levels(alphas, members)
    if step1["factor"] is None:  # no common factor holds step 1's FWER: the family fails (ruling 3)
        claims = {c: {"test": None, "alpha_prime": None, "step1_test": step1["tests"][c], "alpha_prime_step1": None,
                      "alpha_prime_unscaled": None, "step1_size_ok": step1["step1_size_ok"][c], "step2": None}
                  for c in members}
        return {"members": list(members), "claims": claims, "step1": step1,
                "step2": {"pass": False, "configurations": [], "note": "not run: no step-1 factor on the grid holds"},
                "fwer_pass": False}
    factor, claims, tests = step1["factor"], {}, {}
    for c in members:
        t = step1["tests"][c]
        scaled = None if t is None else _scale(levels[c][t], factor)
        checks = None if t is None else size_checks(p2, null_by_conf, names, c, t, scaled, alpha)
        verified = checks is not None and all(x["pass"] for x in checks)
        tests[c] = t if verified else None  # a step-2 failure: p = 1, no swap
        claims[c] = {"test": tests[c], "alpha_prime": _json_levels(scaled) if verified else None, "step1_test": t,
                     "alpha_prime_step1": _json_levels(scaled), "alpha_prime_unscaled": None if t is None else
                     _json_levels(levels[c][t]), "step1_size_ok": step1["step1_size_ok"][c],
                     "step2": None if checks is None else {"pass": verified, "checks": checks}}
    final = {c: _scale(levels[c][t], factor) for c, t in tests.items() if t is not None}
    rates = family_fwer(p2, null_by_conf, members, tests, final, alpha)
    rows = []
    for rate, p, nl, name in zip(rates, p2, null_by_conf, names):
        limit = alpha + 2 * _se(alpha, len(p))
        rows.append({"configuration": name, "fwer": rate, "limit": limit, "pass": rate <= limit,
                     "null_claims": [c for c in members if nl[c]]})
    passed = all(r["pass"] for r in rows)
    return {"members": list(members), "claims": claims, "step1": step1,
            "step2": {"pass": passed, "configurations": rows}, "fwer_pass": passed}


def calibrate_families(p1: Sequence[np.ndarray], p2: Sequence[np.ndarray], null_by_conf: Sequence[dict[str, bool]],
                       names: Sequence[str], alphas: dict, alpha: float = 0.05) -> dict[str, Any]:
    """Both candidate families (ruling 2), each through its step 1 and step 2 (``calibrate_family``)."""
    return {"families": {fam: calibrate_family(members, p1, p2, null_by_conf, names, alphas, alpha)
                         for fam, members in FAMILY.items()}}


def step_one_families(p1: Sequence[np.ndarray], null_by_conf: Sequence[dict[str, bool]], alphas: dict,
                      alpha: float = 0.05) -> dict[str, Any]:
    """Each family's step-1 outcome (its tests and common factor), reported after step 1: step 2 cannot change it."""
    return {fam: family_step_one(members, p1, null_by_conf, alphas, alpha) for fam, members in FAMILY.items()}


def table(result: dict, meta: dict[str, Any]) -> dict[str, Any]:
    """The calibration-table/2 that prereg.Calibration reads: per family, its claims' tests and scaled alpha' (the
    values step 2 verified), the common step-1 factor and step 2's FWER verdict, with the rule versions."""
    families = {}
    for fam, f in result["families"].items():
        families[fam] = {"claims": {c: {"test": v["test"], "alpha_prime": v["alpha_prime"],
                                        "alpha_prime_unscaled": v["alpha_prime_unscaled"]} for c, v in f["claims"].items()},
                         "step1_factor": f["step1"]["factor"], "fwer_pass": f["fwer_pass"]}
    return {"schema": "calibration-table/2", **meta, "rules": pr.CALIBRATION_RULES, "factor_steps": FACTOR_STEPS,
            "families": families}
