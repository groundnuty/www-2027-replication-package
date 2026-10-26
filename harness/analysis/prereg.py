"""The confirmatory analysis of the pre-registration (paper/preregistration.md, draft 4 after change-log
entry 40; "prereg" below).

This module is the executable specification of prereg §1–§4: one family of six claim-level p-values
(C1a, C1b, C2, C3, C4, C5), Holm at alpha = 0.05. A barred, uncalibrated or non-estimable claim gets
p = 1 and stays in the family (§1).

**Estimands** (§2, §4.1). Each is a per-question mean with equal weight per base question, except
C5's per-model ratios:
- **C1:** curated-prompt minus census-prompt accuracy (each question's mean over its attempts).
  - C1a: two-family IUT, or pooled over P when ``c1a_pooled``.
  - C1b: TOST at +-3 points, two-family.
- **C2:** IUT of (i) f_census - f_curated > 0 and (ii) r_curated - r_census > 0 (type-wrong).
  - f is the mean over questions of f_q = 0.5 f_gold + 0.5 f_rewrite when the question has rewrites
    (k_q >= 1; change-log entry 41), else f_q = f_gold. f_gold is the untouched gold's flag, 0 or 1;
    f_rewrite is the flagged share of the question's rewrites. f = f_gold alone when the rewrites
    failed their audit (``rewrites_adopted``).
  - r is the flagged share of a question's type-wrong items, over the questions that have any.
- **C3:** f_curated - mean(f over the three masked draws), two-sided.
- **C4:** Delta_s = pi (union-only recall over s) - (1 - pi) (union-only false-flag rate over s),
  with pi = 25%, for both s; an IUT.
- **C5:** the equal-weight mean over P of theta_m = sum F (Y_fb - Y_gen) / sum F; one-sided.

**Tests** (§4.2). Every statistic is ``stats.studentise`` of its per-question terms on one
UNSTRATIFIED resample matrix of complete question vectors (B = 10,000, seed 20261026).
- **Primary:** the paired, studentised cluster bootstrap-t (``stats.bootstrap_t_p``).
- **Fallback:** the CR2-Satterthwaite Wald test (``stats.cr2_p``).
- **Sidedness:** one-sided p's for one-sided claims; two-sided as 2 min (C1a, C3); C1b by TOST;
  IUTs take the maximum.

**Calibration** (§4.2, issue 19 rulings 1-3). Two candidate families are calibrated, with the
two-family and with the pooled C1a; ``c1a_family`` records which one enters Holm (``choose_c1a``,
from the two calibrated powers), and the confirmatory analysis refuses to run while it is [FILL]. A
claim enters Holm through its family's calibration: the test chosen from step 1 (primary, else
fallback; change-log entry 46) that step 2 verified, and its levels alpha'(a), a = 0.05/k, already
multiplied by the family's common step-1 factor. Its family p is the smallest a with p_test <= alpha'(a), or 1 (``calibrated_p``). Holm on
these p's reproduces the stage-wise Holm procedure with the alpha' values. Before the table exists,
and when the chosen family failed step 2's FWER check, every claim has p = 1.

**Missing data** (§4.3). Missing attempts count wrong (primary). The adverse assignment scores them
toward the null of each one-sided component: a census attempt correct for "curated higher", the
reverse for "census higher", and a missing generic retry correct for C5. Every component's p is the
maximum over the two analyses, because a claim must hold under both. Bank claims have no missing
outcomes: every frozen item has its row (inputs.check_bank_manifest).

The same ``run_contrasts`` serves the real records (``confirmatory``) and the calibration
simulations (calibration.py).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from harness.analysis.inputs import BANK, InputError
from harness.analysis.metrics import ArmKey, Unit, question_vectors
from harness.analysis.stats import (
    QuestionIndex,
    Studentised,
    Term,
    bootstrap_t_bound,
    bootstrap_t_p,
    cr2_p,
    holm,
    holm_levels,
    resample_matrix,
    t_sf,
)
from harness.analysis.stats import studentise as _stats_studentise

PLACEHOLDER = "[FILL]"
CONTRASTS = ("C1a", "C1b", "C2", "C3", "C4", "C5")
TESTS = ("boot_t", "cr2")  # §4.2: the primary and the fallback
HOLM_DIVISORS = (6, 5, 4, 3, 2, 1)  # the Holm levels alpha / k, smallest first
F_WEIGHTS = (0.5, 0.5)  # §4.1: f = 0.5 f_gold + 0.5 f_rewrite
TWO_FAMILY, POOLED = "two-family", "pooled C1a"  # the two candidate Holm families (issue 19 ruling 2)
C1A_FAMILIES = (TWO_FAMILY, POOLED)
STEP2_FWER_FAILURE = "every claim p = 1, no adjustment after step 2"  # issue 19 ruling 3: the only rule implemented
# calibration step 3 (entry 106): its registered seeds, the next two of the step seeds (20261101 step 1, 20261102 step
# 2), apart from the analysis's 20261026 and the power studies' 20261119-20261121; the selection run and the validation
# run each get their own, so the validation data never touch the selection
STEP3_SEEDS = {"selection": 20261103, "validation": 20261104}
# the multiplicity code calibration step 2 ran as: its dated addendum's procedure (42cd54d8..., 2026-10-20). The code has
# grown step 3 since (entry 106), so step 2's addendum is checked against this digest and step 3's against the
# config's multiplicity_digest (the review of 1e31f1f); registered here, so no config can name another
STEP2_PROCEDURE_DIGEST = "0294c59aeac46daf26439664c1a1fbf51936acbd38a61e1a26cd7bfb62348590"
# science ruling (issue 3, 20 Oct 18:50): the pooled C1a may enter the family only if entry 81 keeps every model of P.
# Applied on top of choose_c1a (c1a_choice), which stays as calibrated: it is in the calibrated inference digest.
C1A_ELIGIBILITY = ("c1a-eligibility/1: the pooled C1a is eligible only if entry 81 keeps all seven models of P (no model's "
                   "dev capped:context rate differs between the curated and census arms by more than 2 points); if entry "
                   "81 takes any model out, the two-family C1a enters the family, with no recalibration (the pooled "
                   "alpha' is calibrated for all of P); the ineligible pooled C1a's power is recorded as 0 beside its "
                   "measured value")
# issue 3 (science ruling), change-log entry 91: the refusal rule the registered runs ran under, frozen by name; the
# runner's REFUSAL_RULE and REFUSAL_PHRASES (feat/e0.1-skeleton 3675da5, harness/agent/loop.py). A confirmatory run
# refuses any record not run under it (pipeline.require_refusal_rule)
REFUSAL_RULE = "refusal-v1"
# change-log entry 74: the weighted Holm's weights (C1a, C1b, C2, C3, C4, C5) = (1, 1, 4, 8, 1, 1) / 16, the pooled C1a as
# C1a; a config with any other vector is refused (it would be another procedure)
HOLM_WEIGHTS = {"C1a": 1, "C1a pooled": 1, "C1b": 1, "C2": 4, "C3": 8, "C4": 1, "C5": 1}
REGISTERED_SPLIT = "test"  # prereg §1: the confirmatory population, which a test run never reads (issue 19 ruling)
# The calibration's rule versions (issue 19 rulings 1-3): the calibration table records them, the calibration's
# fingerprint binds them, and load_calibration refuses a table computed under any others.
CALIBRATION_RULES = {
    "grid": "grid/3: each level's own alpha'(a) is the largest j x 0.0005 a (j = 1..4000, upper end 2a) whose step-1 "
            "size is <= a in every null configuration; from the largest level down, alpha'(a) is the largest value on "
            "a's own grid at most min(own, alpha' of the next larger level), so non-decreasing in a and on each "
            "level's grid; the values capped at a (j <= 2000) are reported beside it (ruling 1)",
    "families": "families/1: both candidate families are calibrated, (C1a two-family, C1b, C2, C3, C4, C5) and "
                "(C1a pooled, C1b, C2, C3, C4, C5); one enters Holm (ruling 2)",
    "c1a_choice": "c1a-choice/1: pooled C1a if its calibrated power at the 3-point margin from the dev grid's rates is at "
                  "least the two-family C1a's, else two-family; recorded in the config before the freeze (ruling 2)",
    "fallback": "fallback/3 (change-log entry 46): each claim's test from step-1 data only: the primary if its step-1 "
                "size at alpha'(a) is <= a at every level in every null configuration, else the fallback by the same "
                "rule, else none (p = 1)",
    "joint_step1": "joint-step1/3: per family, with the tests chosen from step 1, one common factor c = k/1000 "
                   "multiplies every alpha' of the family: the largest k in 1..1000 whose step-1 six-claim FWER is "
                   "<= 0.05 in every configuration; when no k reaches it, the family fails (ruling 3)",
    "step2": f"step2/2 (change-log entry 46): step 2 only verifies: a claim's test must have size <= a + 2 SE at every "
             f"level in every null configuration at its scaled alpha', and a failure gives that claim p = 1 with no swap "
             f"and no new factor; then the family FWER must be <= 0.05 + 2 SE in every configuration, and a chosen "
             f"family failing it gives {STEP2_FWER_FAILURE} (ruling 3)",
    "chunks": "chunks/2: every chunk records its step, its seed and the seed scheme; the aggregation refuses a chunk of "
              "another step, seed or scheme",
}


class PreregError(ValueError):
    """The inputs cannot support the registered analysis (prereg §1-§5); nothing confirmatory is produced."""


@dataclass(frozen=True)
class PreregConfig:
    """Arm names and frozen numbers of the registered analysis (prereg §1, §2, §4, §5)."""

    graph: str = "uniprot-sp-2026_03"  # §1: the UniProt Swiss-Prot subset
    models: tuple[str, ...] = ("Qwen/Qwen3.8-27B", "claude-sonnet-5-5")  # §1: C1's two families
    panel: tuple[str, ...] = ()  # §2 C5: the model set P (exact ids, fixed at the freeze); empty = the two families
    panel_families: tuple[str, ...] = ()  # each panel model's family (leave-one-family-out); empty = its id
    # §1 [FILL]: the C1a variant in Holm, "two-family" or "pooled C1a" (over P, as C5 is), chosen before the freeze from
    # the two calibrated powers by choose_c1a and recorded with them (record_c1a_choice; issue 19 ruling 2)
    c1a_family: str = PLACEHOLDER
    c1a_power: dict[str, float] = field(default_factory=dict)
    # the eligibility rule the choice was made under (C1A_ELIGIBILITY) with entry 81's outcome and the measured powers:
    # {"rule", "entry81_dropped", "pooled_eligible", "measured_power"}; empty while the choice is [FILL]
    c1a_eligibility: dict[str, Any] = field(default_factory=dict)
    step2_fwer_failure: str = STEP2_FWER_FAILURE  # issue 19 ruling 3: a chosen family failing step 2's FWER check
    # issue 3, entry 91: the refusal rule, by name (REFUSAL_RULE: feat/e0.1-skeleton 3675da5's loop.py REFUSAL_RULE and
    # REFUSAL_PHRASES); a refused attempt is scored wrong (§2 "Refusals"), and a confirmatory run refuses any record
    # run under another rule or none (pipeline.require_refusal_rule)
    refusal_rule: str = REFUSAL_RULE
    natural_cohort: str = "none"
    curated_prompt_cohort: str = "curated"
    census_prompt_cohort: str = "census"
    checker_use: str = "checker"
    curated_source: str = "curated"
    census_source: str = "census"
    union_source: str = "union"  # the descriptive retry contrast (the old C4)
    generic_source: str = "generic"  # §2 C5: "Please re-check your answer." on the curated-flagged set
    bank_model: str = "bank"
    bank_use: str = "replay"
    # §2 C3: the three masked-input generated draws, named as their source files and OntologySource names (feat/sources
    # data/sources/<graph>/generated-dc-k.json; spec trace B15)
    masked_sources: tuple[str, ...] = ("generated-dc-1", "generated-dc-2", "generated-dc-3")
    scorer_version: str = PLACEHOLDER  # §3: the frozen scorer's rules version
    # §3: the frozen scorer's digest, the sha256 of its closed module set (feat/scorer-digest); a registered run refuses
    # a batch whose manifest records another (gate.require_scorer; spec trace C11)
    scorer_digest: str = PLACEHOLDER
    bank_manifest_sha256: str = PLACEHOLDER  # §5 "Fault bank"
    # the frozen question set per graph (ids are reused across calibration rounds, so every record keys on (id,
    # questions_sha256)); "<graph>#bank" may name a bank built on another question set (a bank-only extension)
    questions_sha256: dict[str, str] = field(default_factory=lambda: {"uniprot-sp-2026_03": PLACEHOLDER})
    calibration: str = PLACEHOLDER  # §4.2: the calibration table (path relative to the repository)
    calibration_sha256: str = PLACEHOLDER
    # change-log entry 52: the cluster map (base question id -> cluster id: connected components of shared anchors within
    # a template), written by the question generator; the confirmatory analysis resamples its clusters
    cluster_map: str = PLACEHOLDER
    cluster_map_sha256: str = PLACEHOLDER
    # prereg §5 "Prompt renders", change-log entry 91: C1's two compared prompts (prompt cohort -> {"file", "sha256"}, the
    # file under the repository), from which the leak exclusion, a descriptive second C1 report (harness/analysis/
    # leak.py), is computed at analysis time; a confirmatory run refuses without them
    c1_leak_prompts: dict[str, Any] = field(default_factory=dict)
    # issue 12: inference.inference_digest() of the code the calibration ran (87c7126); a confirmatory run refuses another
    calibrated_inference_digest: str = PLACEHOLDER
    # issue 12: inference.grouping_digest() of the cluster grouping added after the calibration, frozen at the freeze
    # commit; a confirmatory run refuses another, and refuses while it is [FILL]
    grouping_digest: str = PLACEHOLDER
    # issue 12: the inference's known answers (kat.known_answers(): the calibrated paths', equal to the calibrated
    # commit's own, and the cluster grouping's, [FILL] until the freeze commit), and the environment they were computed
    # in (kat.environment()); a confirmatory run recomputes both and refuses others, or either while it is [FILL]
    calibrated_kat_sha256: str = PLACEHOLDER
    grouping_kat_sha256: str = PLACEHOLDER
    kat_environment: dict[str, str] = field(default_factory=dict)
    # the host-change rule's record (data/analysis/kat-hosts.json: each host's environment and answers, the calibrated
    # one recomputed from 87c7126's own functions), under the repository, bound by its sha256 when the config is loaded
    # (check_kat_hosts), so the record cannot drift silently; a confirmatory run refuses while either is [FILL]
    kat_hosts: str = PLACEHOLDER
    kat_hosts_sha256: str = PLACEHOLDER
    # issue 19, change-log entries 74 and 76: the confirmatory multiplicity is weighted Holm (harness/analysis/
    # multiplicity.py) with these weights (x 16; the pooled C1a weighted as C1a), on step 1's alpha' at every reachable
    # level and step 2's usable levels, each frozen file read by its sha256 ([FILL]: a confirmatory run refuses); the
    # multiplicity code has its own digest and known answer, frozen after its review. Step 2's usable levels and FWER
    # check are the dated step-2 addendum (entry 91), refused unless the frozen step-2 procedure produced it
    holm_weights: dict[str, int] = field(default_factory=lambda: dict(HOLM_WEIGHTS))
    multiplicity_table: str = PLACEHOLDER
    multiplicity_table_sha256: str = PLACEHOLDER
    # the review of 9d9c88c: step 1's stored p-values (calibrate.py's OUT/step1 chunks) by their content, as
    # multiplicity.outputs_digest gives it (the chunk count, one sha256 over every chunk's arrays). step1.json records only
    # the fingerprint the chunks carry, so this is their content as hashed on SiteA on 2026-10-19; the alpha' producer
    # refuses chunks of any other content before using them
    step1_outputs: dict[str, Any] = field(default_factory=dict)
    step2_addendum: str = PLACEHOLDER
    step2_addendum_sha256: str = PLACEHOLDER
    # the multiplicity code step 2 ran as (its addendum's procedure, STEP2_PROCEDURE_DIGEST): the code has grown step 3
    # since (entry 106), so step 2's addendum is checked against the digest it was produced under, step 3's against
    # multiplicity_digest
    step2_procedure_digest: str = STEP2_PROCEDURE_DIGEST
    # the review of 9d9c88c: the freeze check's records of its recomputations on SiteA (path relative to the repository,
    # freeze-registry/2, apart from the citation check's registry.json): step2_outputs records the step-2 outputs'
    # content it recomputed for an addendum, step 3's lists beside it; without the chunks on this host a confirmatory
    # run accepts an addendum only with its record (multiplicity.check_addendum, check_selection, check_step3)
    freeze_registry: str = "data/freeze/recomputations.json"
    # calibration step 3 (entry 106): its seeds (STEP3_SEEDS, registered here), the selection's alpha' (C2 two-sided at
    # every level, the repaired levels) and the dated addendum of its validation, each by its sha256; C2's decision and
    # the repaired levels' wait for them (multiplicity.load)
    step3_seeds: dict[str, int] = field(default_factory=lambda: dict(STEP3_SEEDS))
    # its configurations and their two-sided null claims, bound before it runs (calibration.step3_configuration_list on
    # steps 1 and 2's structure), and the step-3 runs' fingerprint (calibrate.py fingerprint3, printed by plan3 on SiteA
    # before submission): the selection and the addendum must record exactly these (the review of 1e31f1f)
    step3_configurations: str = PLACEHOLDER
    step3_configurations_sha256: str = PLACEHOLDER
    step3_fingerprint: str = PLACEHOLDER
    step3_selection: str = PLACEHOLDER
    step3_selection_sha256: str = PLACEHOLDER
    step3_addendum: str = PLACEHOLDER
    step3_addendum_sha256: str = PLACEHOLDER
    multiplicity_digest: str = PLACEHOLDER
    multiplicity_kat_sha256: str = PLACEHOLDER
    rewrites_adopted: bool = True  # §4.1: f = 0.5 f_gold + 0.5 f_rewrite; f_gold alone if the rewrites fail their audit
    split: str = "test"  # §1
    n_reps: int = 3  # §4.4 "Fixed stopping"
    alpha: float = 0.05  # §1
    c1_margin: float = 0.03  # §2 C1b
    c2_min_effect: float = 0.05  # §2 C2: 5 points per component, "at least 5 points" from the 0.05/6 bound
    c3_sesoi: float = 0.05  # §2 C3
    c4_sesoi: float = 0.01  # §2 C4
    c5_sesoi: float = 0.10  # §2 C5
    prevalence: float = 0.25  # §4.1: C4 and the descriptive precision at pi = 25%
    prevalences: tuple[float, ...] = (0.10, 0.25, 0.50)  # descriptive precision
    n_boot: int = 10_000  # §4.2
    seed: int = 20261026  # §4.2
    missing_limit: float = 0.02  # §4.3: a compromised arm is reported

    def __post_init__(self) -> None:
        if self.panel_families and len(self.panel_families) != len(self.panel):
            raise PreregError("panel_families must name one family per panel model")
        if self.panel and not set(self.models) <= set(self.panel):
            raise PreregError("the panel P must contain the two families' models (§2 C5)")
        if self.c1a_family not in (PLACEHOLDER, *C1A_FAMILIES):
            raise PreregError(f"c1a_family {self.c1a_family!r} is none of {C1A_FAMILIES} (or {PLACEHOLDER})")
        if dict(self.holm_weights) != HOLM_WEIGHTS or any(isinstance(v, bool) for v in self.holm_weights.values()):
            raise PreregError(f"holm_weights must be the registered weights {HOLM_WEIGHTS} (change-log entry 74; the "
                              f"review of 70281e9: any other vector is another procedure), got {self.holm_weights}")
        if dict(self.step3_seeds) != STEP3_SEEDS:
            raise PreregError(f"step3_seeds must be the registered {STEP3_SEEDS} (entry 106), got {self.step3_seeds}")
        if self.step2_procedure_digest != STEP2_PROCEDURE_DIGEST:
            raise PreregError(f"step2_procedure_digest must be the registered {STEP2_PROCEDURE_DIGEST} (the procedure step "
                              f"2's dated addendum was produced by), got {self.step2_procedure_digest}")
        if self.step2_fwer_failure != STEP2_FWER_FAILURE:
            raise PreregError(f"step2_fwer_failure {self.step2_fwer_failure!r}: the only registered rule is "
                              f"{STEP2_FWER_FAILURE!r} (issue 19 ruling 3)")
        if self.refusal_rule != REFUSAL_RULE:
            raise PreregError(f"refusal_rule {self.refusal_rule!r}: the registered refusal rule is {REFUSAL_RULE!r} "
                              f"(issue 3, change-log entry 91)")

    @property
    def c1a_pooled(self) -> bool:
        """Whether C1a is pooled over P (False while the choice is [FILL]: confirmatory refuses then)."""
        return self.c1a_family == POOLED

    def with_c1a_choice(self, power_pooled: float, power_two_family: float,
                        dropped: Sequence[str] = ()) -> PreregConfig:
        """This config with the C1a choice made from the two calibrated powers and entry 81's outcome (c1a_choice), as
        record_c1a_choice records it in a config file."""
        family, power, record = c1a_choice(power_pooled, power_two_family, dropped)
        return dataclasses.replace(self, c1a_family=family, c1a_power=power, c1a_eligibility=record)

    def require_c1a_eligibility(self) -> str:
        """The recorded C1a family (require_c1a_choice), refused unless its eligibility record is C1A_ELIGIBILITY's
        and the recorded family and powers are what c1a_choice gives from the record's measured powers and entry 81's
        outcome (a choice set by hand, or one recorded without entry 81's outcome)."""
        family = self.require_c1a_choice()
        record = self.c1a_eligibility
        measured = record.get("measured_power") if isinstance(record, dict) else None
        if not isinstance(measured, dict) or set(measured) != set(C1A_FAMILIES) or record.get("rule") != C1A_ELIGIBILITY:
            raise PreregError(f"the C1a choice has no eligibility record under {C1A_ELIGIBILITY!r} (entry 81's outcome "
                              f"and the measured powers; record_c1a_choice writes it), got {record!r}")
        want = c1a_choice(measured[POOLED], measured[TWO_FAMILY], record.get("entry81_dropped"))
        if (family, self.c1a_power, record) != want:
            raise PreregError(f"the recorded C1a choice ({family}, powers {self.c1a_power}, {record}) is not the "
                              f"eligibility rule's ({want[0]}, powers {want[1]}, {want[2]})")
        return family

    def require_c1a_choice(self) -> str:
        """The recorded C1a family; PreregError while it is [FILL], or when it is not what choose_c1a gives from the
        powers recorded with it (a choice set by hand)."""
        if self.c1a_family == PLACEHOLDER:
            raise PreregError("the C1a family (pooled or two-family) is not recorded (prereg §1 [FILL], issue 19 ruling 2): "
                              "no confirmatory inference until record_c1a_choice has recorded it from the two calibrated "
                              "powers")
        if set(self.c1a_power) != set(C1A_FAMILIES):
            raise PreregError(f"c1a_power must give both calibrated powers {C1A_FAMILIES}, got {sorted(self.c1a_power)}")
        expected = choose_c1a(self.c1a_power[POOLED], self.c1a_power[TWO_FAMILY])
        if expected != self.c1a_family:
            raise PreregError(f"c1a_family {self.c1a_family!r} is not the rule's choice {expected!r} from the recorded "
                              f"powers {self.c1a_power}")
        return self.c1a_family

    @classmethod
    def load(cls, path: Path) -> PreregConfig:
        """The config file's settings, with the file's path and sha256 recorded on the config (``loaded_from``), which
        a confirmatory run requires and reports: the frozen values are the file's, whose hash the freeze binds."""
        raw = Path(path).read_bytes()
        data = json.loads(raw)
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - names - {"schema", "note"}
        if unknown:
            raise PreregError(f"{path}: unknown settings {sorted(unknown)}")
        values = {k: tuple(v) if isinstance(v, list) else v for k, v in data.items() if k in names}
        cfg = cls(**values)
        cfg.check_kat_hosts()  # the known answers' host record, by its bound sha256
        object.__setattr__(cfg, "_loaded_from", (str(Path(path).resolve()), hashlib.sha256(raw).hexdigest()))
        return cfg

    def check_kat_hosts(self) -> str | None:
        """The sha256 of the known answers' host record (``kat_hosts``, under the repository), when it is the bound
        ``kat_hosts_sha256``; None while both are [FILL] (a config built for a test run). Refused: one of the two [FILL],
        the record unreadable, or its sha256 another (it changed since the config bound it)."""
        if self.kat_hosts == PLACEHOLDER and self.kat_hosts_sha256 == PLACEHOLDER:
            return None
        if PLACEHOLDER in (self.kat_hosts, self.kat_hosts_sha256):
            raise PreregError("kat_hosts and kat_hosts_sha256 (the known answers' host record and its sha256) are filled "
                              "together or not at all")
        path = repo_root() / self.kat_hosts
        try:
            raw = path.read_bytes()
        except OSError:
            raise PreregError(f"the known answers' host record {path} is not readable") from None
        actual = hashlib.sha256(raw).hexdigest()
        if actual != self.kat_hosts_sha256:
            raise PreregError(f"the known answers' host record {path} has sha256 {actual}, not the frozen "
                              f"{self.kat_hosts_sha256}: it changed since the config bound it")
        return actual

    @property
    def loaded_from(self) -> tuple[str, str] | None:
        """(absolute path, sha256) of the file this config was loaded from; None for a config built in code, or built from
        it by dataclasses.replace. Not a setting, and no proof by itself that the settings are still the file's: the
        preflight reloads the file and compares."""
        return self.__dict__.get("_loaded_from")

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def frozen_questions(self, bank: bool = False) -> str:
        """The frozen question-set hash of the graph (or of its bank); PreregError while it is [FILL]."""
        value = (self.questions_sha256.get(f"{self.graph}#bank") if bank else None) or self.questions_sha256.get(self.graph)
        if value in (None, "", PLACEHOLDER):
            raise PreregError(f"the frozen question-set hash of {self.graph} is not filled in (questions_sha256 [FILL]); "
                              f"a confirmatory run keys every record on (id, questions_sha256)")
        return value

    @property
    def c5_models(self) -> tuple[str, ...]:
        """The model set P of C5 (and of a pooled C1a)."""
        return self.panel or self.models

    @property
    def c1a_models(self) -> tuple[str, ...]:
        return self.c5_models if self.c1a_pooled else self.models

    @property
    def families(self) -> dict[str, str]:
        if self.panel_families:
            return dict(zip(self.panel, self.panel_families))
        return {m: m for m in self.c5_models}

    def prompt(self, model: str, cohort: str) -> ArmKey:
        return ArmKey(self.graph, model, cohort, "prompt")

    def checker(self, model: str, source: str) -> ArmKey:
        return ArmKey(self.graph, model, self.natural_cohort, self.checker_use, source)

    def bank(self, source: str) -> ArmKey:
        return ArmKey(self.graph, self.bank_model, BANK, self.bank_use, source)

    def prompt_models(self) -> list[str]:
        return list(dict.fromkeys([*self.models, *(self.c1a_models if self.c1a_pooled else ())]))

    def required_arms(self) -> list[ArmKey]:
        arms = [self.bank(s) for s in (self.curated_source, self.census_source, *self.masked_sources)]
        for m in self.prompt_models():
            arms += [self.prompt(m, self.curated_prompt_cohort), self.prompt(m, self.census_prompt_cohort)]
        for m in self.models:
            arms += [self.checker(m, s) for s in (self.curated_source, self.census_source, self.union_source,
                                                  self.generic_source)]
        for m in self.c5_models:
            if m not in self.models:
                arms += [self.checker(m, s) for s in (self.curated_source, self.generic_source)]
        return arms


# --- calibration (§4.2) -------------------------------------------------------------------------


FAMILY_CLAIMS = {TWO_FAMILY: ("C1a", "C1b", "C2", "C3", "C4", "C5"),
                 POOLED: ("C1a pooled", "C1b", "C2", "C3", "C4", "C5")}  # the calibration's claim names per family


@dataclass(frozen=True)
class Calibration:
    """The calibration of the family in Holm (§4.2, issue 19, change-log entry 46): per claim, the test chosen from
    step 1 and verified by step 2 ("boot_t", "cr2", or None: no test passed step 1, or it failed step 2) and its
    calibrated levels alpha'(a) at every Holm level a = alpha / k (keys "1".."6"), already multiplied by the
    family's step-1 factor (the values step 2 verified). ``family`` is the
    candidate family the levels belong to (None: built by hand). A family whose step-2 FWER check failed has no
    claims: every p = 1 (ruling 3)."""

    claims: dict[str, dict[str, Any]]
    source: str = "none"
    family: str | None = None
    fwer_pass: bool | None = None  # step 2's six-claim FWER check of the family

    @classmethod
    def load(cls, path: Path, sha256: str, family: str) -> Calibration:
        """One family of a calibration-table/2 file, read by its frozen hash and refused under other rule versions."""
        raw = Path(path).read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != sha256:
            raise PreregError(f"{path}: sha256 {actual} is not the frozen calibration table's {sha256}")
        data = json.loads(raw)
        if data.get("schema") != "calibration-table/2":
            raise PreregError(f"{path}: not a calibration-table/2 file")
        if data.get("rules") != CALIBRATION_RULES:
            raise PreregError(f"{path}: computed under other calibration rule versions than this code's (issue 19)")
        fam = (data.get("families") or {}).get(family)
        if fam is None or set(fam.get("claims") or {}) != set(FAMILY_CLAIMS[family]):
            raise PreregError(f"{path}: no complete {family} family (claims {FAMILY_CLAIMS[family]})")
        if fam.get("fwer_pass") is not True:  # ruling 3: no adjustment after step 2
            return cls({}, f"{path}, {family}: step 2's six-claim FWER check failed, so {STEP2_FWER_FAILURE} "
                           f"(issue 19 ruling 3)", family, False)
        return cls(fam["claims"], f"{path}, {family}", family, True)

    def entry(self, claim: str) -> tuple[str | None, dict[int, float] | None]:
        e = self.claims.get(claim) or {}
        if e.get("test") not in TESTS:
            return None, None
        return e["test"], {int(k): float(v) for k, v in e["alpha_prime"].items()}


UNCALIBRATED = Calibration({}, "none (no claim calibrated yet: every p = 1)")


def choose_c1a(power_pooled: float, power_two_family: float) -> str:
    """Issue 19 ruling 2: the pooled C1a enters Holm if its calibrated power at the 3-point margin (from the dev
    grid's rates) is at least the two-family C1a's, else the two-family C1a. Each power is a number in [0, 1]."""
    for name, v in ((POOLED, power_pooled), (TWO_FAMILY, power_two_family)):
        if isinstance(v, bool) or not isinstance(v, int | float) or not 0 <= v <= 1:  # NaN fails the range too
            raise PreregError(f"the calibrated power of the {name} family must be a number in [0, 1], got {v!r}")
    return POOLED if power_pooled >= power_two_family else TWO_FAMILY


def _entry81_dropped(dropped: Any) -> list[str]:
    if isinstance(dropped, str) or not isinstance(dropped, list | tuple) or not all(
            isinstance(m, str) and m for m in dropped) or len(set(dropped)) != len(dropped):
        raise PreregError(f"entry 81's dropped models must be a list of distinct model ids, got {dropped!r}")
    return sorted(dropped)


def c1a_choice(power_pooled: float, power_two_family: float,
               dropped: Sequence[str]) -> tuple[str, dict[str, float], dict[str, Any]]:
    """C1A_ELIGIBILITY on top of ruling 2 (choose_c1a, unchanged): (family, recorded powers, eligibility record).
    Entry 81 kept every model of P (``dropped`` empty): choose_c1a on the measured powers, recorded as measured. It took
    any model out: the pooled C1a is not eligible, its power is recorded as 0 (the measured one kept in the record), so
    choose_c1a, as the confirmatory analysis re-applies it (require_c1a_choice), gives the two-family C1a; refused when
    the two-family power is 0 too, where ruling 2's tie rule would give the pooled C1a (a science ruling is needed)."""
    choose_c1a(power_pooled, power_two_family)  # the powers, checked as ruling 2 checks them
    out = _entry81_dropped(dropped)
    measured = {POOLED: float(power_pooled), TWO_FAMILY: float(power_two_family)}
    power = dict(measured) if not out else {POOLED: 0.0, TWO_FAMILY: float(power_two_family)}
    family = choose_c1a(power[POOLED], power[TWO_FAMILY])
    if out and family != TWO_FAMILY:
        raise PreregError(f"entry 81 took {', '.join(out)} out, so the pooled C1a is not eligible, but with the two-family "
                          f"C1a's power 0 ruling 2's tie rule gives the pooled C1a: a science ruling is needed")
    return family, power, {"rule": C1A_ELIGIBILITY, "entry81_dropped": out, "pooled_eligible": not out,
                           "measured_power": measured}


def record_c1a_choice(path: Path, power_pooled: float, power_two_family: float, *, dropped: Sequence[str]) -> str:
    """Records the C1a choice (c1a_choice: ruling 2 under C1A_ELIGIBILITY) with the two powers and entry 81's outcome
    (``dropped``: the models of P it took out, empty when it kept all) in the config file at ``path``, once, before the
    freeze. A recorded choice is never changed: the same powers and outcome again are a no-op, others are refused."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    family, power, record = c1a_choice(power_pooled, power_two_family, dropped)
    current = data.get("c1a_family", PLACEHOLDER)
    if current != PLACEHOLDER:
        if current == family and data.get("c1a_power") == power and data.get("c1a_eligibility") == record:
            return family
        raise PreregError(f"{path}: the C1a choice is already recorded ({current}, powers {data.get('c1a_power')}, "
                          f"eligibility {data.get('c1a_eligibility')}); it is never changed")
    data["c1a_family"], data["c1a_power"], data["c1a_eligibility"] = family, power, record
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    try:
        PreregConfig.load(tmp).require_c1a_eligibility()  # the file as written loads and holds the rule's choice
    except Exception:
        tmp.unlink()
        raise
    tmp.replace(path)  # atomic
    return family


def calibrated_p(p_test: float, alpha_prime: Mapping[int, float] | None, alpha: float = 0.05) -> float:
    """§4.2: the smallest Holm level a = alpha / k at which the claim's test rejects at its calibrated level
    (p_test <= alpha'(a)); 1 when none or uncalibrated. Holm over these p's makes the stage-wise calibrated
    decisions only if alpha'(a) is non-decreasing in a, so anything else is refused."""
    if alpha_prime is None:
        return 1.0
    levels = [alpha_prime[k] for k in HOLM_DIVISORS]
    if any(b < a for a, b in pairwise(levels)):
        raise PreregError(f"alpha' must be non-decreasing in the Holm level, got {levels} for a = alpha/6 .. alpha")
    for k in HOLM_DIVISORS:
        if p_test <= alpha_prime[k]:
            return alpha / k
    return 1.0


# --- clusters (change-log entry 52) -------------------------------------------------------------


def connected_clusters(anchors: Mapping[str, Iterable[str]], templates: Mapping[str, str]) -> dict[str, str]:
    """Entry 52: base questions linked by a shared anchor entity within their template, closed under chains (the
    connected components); each cluster is named by its smallest question id, and a question sharing nothing is its
    own cluster. The question generator owns the anchors and writes the map with this and write_cluster_map."""
    parent = {q: q for q in anchors}

    def root(q: str) -> str:
        while parent[q] != q:
            parent[q] = parent[parent[q]]
            q = parent[q]
        return q

    first: dict[tuple[str, str], str] = {}
    for q in sorted(anchors):
        for a in sorted(anchors[q]):
            other = first.setdefault((templates[q], a), q)
            ra, rb = root(other), root(q)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)  # the smaller id stays the root: the cluster's name
    return {q: root(q) for q in sorted(anchors)}


CLUSTER_MAP_SCHEMA = "cluster-map/2"


def write_cluster_map(path: Path, anchors: Mapping[str, Iterable[str]], templates: Mapping[str, str]) -> None:
    """The cluster-map/2 file the analysis reads (its sha256 is frozen in the config as cluster_map_sha256): per base
    question, its template, its declared anchors (sorted) and its cluster id from connected_clusters, so that the
    loader can recompute the clusters from the file itself."""
    clusters = connected_clusters(anchors, templates)
    questions = {q: {"template": templates[q], "anchors": sorted(set(anchors[q])), "cluster": clusters[q]}
                 for q in sorted(anchors)}
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"schema": CLUSTER_MAP_SCHEMA, "questions": questions}, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


@dataclass(frozen=True)
class Clusters:
    """The resampling units of one question index (entry 52): each question's cluster position (``group``), the
    clusters in order of first appearance along the index, and the resample matrix over the clusters."""

    group: np.ndarray
    matrix: np.ndarray
    ids: tuple[str, ...]
    # change-log entry 73, reporting only: what every statistic on these clusters passes through (zero_spread counts
    # near-zero spread with it, and its sensitivity analysis makes those resamples undefined); None in the inference
    spread: Callable[[Studentised], Studentised] | None = None

    def sum(self, x: np.ndarray) -> np.ndarray:
        """Per-question values summed per cluster."""
        return np.bincount(self.group, weights=np.asarray(x, dtype=np.float64), minlength=len(self.ids))

    def summary(self) -> dict[str, Any]:
        sizes = Counter(int(n) for n in np.bincount(self.group, minlength=len(self.ids)))
        return {"questions": len(self.group), "clusters": len(self.ids),
                "sizes": {str(k): sizes[k] for k in sorted(sizes)}, "largest": max(sizes, default=0)}


@dataclass(frozen=True)
class ClusterMap:
    """Base question id -> cluster id (entry 52), from the question generator's cluster-map/2 file, with each question's
    template as the file declares it (None for a map built in memory)."""

    clusters: Mapping[str, str]
    source: str = "in memory"
    templates: Mapping[str, str] | None = None

    @classmethod
    def load(cls, path: Path, sha256: str) -> ClusterMap:
        """A cluster-map/2 file, read by its frozen hash and checked against itself: every question declares its
        template and anchors, and the stated cluster ids must be the connected components of those anchors within the
        templates (connected_clusters, over every question in the file: one outside the analysed split, a dev question,
        may link two of its questions)."""
        raw = Path(path).read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != sha256:
            raise PreregError(f"{path}: sha256 {actual} is not the frozen cluster map's {sha256}")
        data = json.loads(raw)
        questions = data.get("questions") if isinstance(data, dict) else None
        if (not isinstance(questions, dict) or not questions or data.get("schema") != CLUSTER_MAP_SCHEMA
                or "clusters" in data):
            raise PreregError(f"{path}: not a {CLUSTER_MAP_SCHEMA} file (question id -> its template, declared anchors "
                              f"and cluster id; the clusters stated per question only)")
        malformed = sorted(q for q, e in questions.items() if not (
            isinstance(e, dict) and set(e) == {"template", "anchors", "cluster"}
            and isinstance(e["template"], str) and e["template"] and isinstance(e["cluster"], str) and e["cluster"]
            and isinstance(e["anchors"], list) and all(isinstance(a, str) and a for a in e["anchors"])))
        if malformed:
            raise PreregError(f"{path}: {len(malformed)} malformed questions (each needs a non-empty template, a list of "
                              f"non-empty anchor ids and a non-empty cluster id, and nothing else), e.g. {malformed[:3]}")
        stated = {q: e["cluster"] for q, e in questions.items()}
        templates = {q: e["template"] for q, e in questions.items()}
        components = connected_clusters({q: e["anchors"] for q, e in questions.items()}, templates)
        wrong = sorted(q for q in stated if stated[q] != components[q])
        if wrong:
            raise PreregError(f"{path}: {len(wrong)} questions' stated clusters are not the connected components of their "
                              f"declared anchors within their template (entry 52), e.g. (question, stated, computed) "
                              f"{[(q, stated[q], components[q]) for q in wrong[:3]]}")
        return cls(stated, f"{path} (sha256 {actual})", templates)

    @classmethod
    def singletons(cls, questions: Iterable[str]) -> ClusterMap:
        """Every question its own cluster (the resampling before entry 52)."""
        return cls({q: q for q in questions}, "every question its own cluster")

    def resampling(self, index: QuestionIndex, cfg: PreregConfig, K: np.ndarray | None = None) -> Clusters:
        """The index's clusters and their resample matrix (B unstratified draws of the clusters, seed 20261026, or
        ``K`` over the clusters). Every question must be in the map, and a cluster lies within one template."""
        missing = [q for q in index.questions if q not in self.clusters]
        if missing:
            raise PreregError(f"{len(missing)} base questions are not in the cluster map ({self.source}), e.g. "
                              f"{missing[:3]} (entry 52: an unmapped question is refused)")
        if self.templates is not None:  # the file's templates are the ones its clusters were computed within
            off = [q for q, s in zip(index.questions, index.strata) if self.templates[q] != s.split("|")[0]]
            if off:
                raise PreregError(f"{len(off)} questions' templates in the cluster map ({self.source}) are not their "
                                  f"question file's, e.g. {off[:3]}")
        position: dict[str, int] = {}
        templates: dict[str, set[str]] = {}
        group = []
        for q, stratum in zip(index.questions, index.strata):
            cid = self.clusters[q]
            group.append(position.setdefault(cid, len(position)))
            templates.setdefault(cid, set()).add(stratum.split("|")[0])
        across = sorted(c for c, t in templates.items() if len(t) > 1)
        if across:
            raise PreregError(f"clusters {across[:3]} span more than one template (entry 52: within a template)")
        ids = tuple(position)
        if K is None:
            K = resample_matrix(QuestionIndex(ids, ("all",) * len(ids)), cfg.n_boot, seed=cfg.seed, scheme="unstratified")
        elif K.shape[1] != len(ids):
            raise PreregError(f"a resample matrix over {K.shape[1]} units for {len(ids)} clusters")
        return Clusters(np.array(group, dtype=np.intp), K, ids)


def studentise(K: np.ndarray | Clusters, terms: Sequence[Term]) -> Studentised:
    """stats.studentise on the resampling units (entry 52). On a question-resample matrix it is stats.studentise
    itself; on Clusters, each term's per-question a and b are summed per cluster first and the clusters' resample
    matrix is used. Only the grouping key changes: the contrasts and stats.studentise are as calibrated. Clusters
    with a ``spread`` hook (the zero-spread report, entry 73) pass each statistic through it."""
    if isinstance(K, Clusters):
        s = _stats_studentise(K.matrix, [Term(t.c, K.sum(t.a), K.sum(t.b)) for t in terms])
        return s if K.spread is None else K.spread(s)
    return _stats_studentise(K, terms)


def load_clusters(cfg: PreregConfig, root: Path) -> ClusterMap:
    """The frozen cluster map (entry 52); PreregError while the config's entry is [FILL]."""
    if PLACEHOLDER in (cfg.cluster_map, cfg.cluster_map_sha256):
        raise PreregError("the cluster map (entry 52) is not frozen (cluster_map / cluster_map_sha256 [FILL]): a "
                          "confirmatory run resamples clusters of base questions and refuses without it")
    return ClusterMap.load(Path(root) / cfg.cluster_map, cfg.cluster_map_sha256)


@dataclass(frozen=True)
class Preflight:
    """What a run starts from, settled before any data is read (preflight); the inference takes its inputs only from
    here (clustered_contrasts)."""

    cfg: PreregConfig  # a confirmatory run's own copy, reloaded from the config file and compared with the caller's
    family: str  # the recorded C1a family ([FILL] in a test run before the choice is recorded)
    inference_digest: str
    grouping_digest: str
    clusters: ClusterMap
    K: np.ndarray | None  # a resample matrix over the clusters (test runs only), else drawn from the seed
    calibration: Calibration
    confirmatory: bool
    config_file: str | None  # the config file and its sha256 (None for a config built in code: test runs only)
    config_sha256: str | None
    known_answers: dict[str, str] = field(default_factory=dict)  # the recomputed known answers (check_known_answers)
    multiplicity_digest: str = ""  # inference.multiplicity_digest() (check_multiplicity_digest)
    multiplicity: Any = None  # the weighted Holm's frozen tables (multiplicity.WeightedCalibration; entries 74, 76)


def preflight(cfg: PreregConfig, root: Path, *, clusters: ClusterMap | None = None, K: np.ndarray | None = None,
              multiplicity: Any = None, testing: bool = False) -> Preflight:
    """The checks a run makes before it reads any data, in order: the recorded C1a choice (issue 19 ruling 2), the
    inference digests (issue 12) and the multiplicity's (entry 74), the inference's known answers, recomputed here, and
    the frozen inputs. A confirmatory run reads the cluster map (entry 52) and the weighted Holm's tables (step 1's
    alpha' at the reachable levels, step 2's usable levels; entries 74, 76) only from the config's files under
    ``root``, by their frozen sha256 (and the tables' rules), refuses while they are [FILL], refuses a step-2 addendum
    the frozen step-2 procedure did not produce (entry 91: multiplicity.check_addendum), and draws its own resample
    matrix from the seed; a map, matrix or tables passed in by the caller are refused. The plain Holm's calibration
    table is no longer read: the contrasts run uncalibrated and the weighted family replaces their plain Holm. It takes the config as loaded from its file: it reloads that file, refuses when its sha256 is not the one
    loaded or when the config it was handed differs from it in any setting or attribute (changed in place after
    loading), and gives the inference the fresh copy, which the caller never holds; the report records the file's path
    and sha256. A test run (``testing``) may pass a map, matrix or table, or a config built in code; what it passes is
    labelled so, and the report says the run is not confirmatory; it does not need the C1a choice, which stays [FILL]
    until calibration step 2 (it runs C1a as the two-family IUT and reports the choice as [FILL]; the dress rehearsal's
    F4). A confirmatory run refuses first, reading nothing,
    while a slot of the model set P (or its family) is [FILL] (issue 22, change-log entry 82: the DeepSeek model is
    decided by issue 21); a test run is not refused for it, its arms being its own."""
    if not testing and PLACEHOLDER in (*cfg.panel, *cfg.panel_families):
        raise PreregError("the model set P has a [FILL] slot (issue 22, change-log entry 82; the DeepSeek model is decided "
                          "by issue 21): no confirmatory run until every panel model is its exact id and family")
    # the C1a choice is [FILL] until calibration step 2: a test run (not confirmatory) runs C1a as the two-family IUT
    # and reports the choice as [FILL]; a recorded choice is checked in every run, and a confirmatory run needs one
    family = cfg.c1a_family if testing and cfg.c1a_family == PLACEHOLDER else cfg.require_c1a_choice()
    if family != PLACEHOLDER:  # a recorded choice holds its eligibility record (C1A_ELIGIBILITY)
        cfg.require_c1a_eligibility()
    inference, grouping = check_inference_digest(cfg)
    weighted_digest = check_multiplicity_digest(cfg)
    answers = check_known_answers(cfg)
    passed = [n for n, v in (("a cluster map", clusters), ("a resample matrix", K),
                             ("the weighted Holm's tables", multiplicity)) if v is not None]
    if not testing:
        if passed:
            raise PreregError(f"{' and '.join(passed)} passed in by the caller is refused in a confirmatory run: the map "
                              f"and the table are read only from the config's files by their frozen sha256, and the "
                              f"resampling drawn from the seed; a test run passes testing=True and is reported as not "
                              f"confirmatory")
        if cfg.loaded_from is None:
            raise PreregError("a config built or altered in code is refused in a confirmatory run: it takes the config "
                              "as loaded from its frozen file (PreregConfig.load), whose sha256 the report records; a test "
                              "run passes testing=True")
        path, sha = cfg.loaded_from
        fresh = PreregConfig.load(Path(path))
        if fresh.loaded_from[1] != sha:
            raise PreregError(f"the config file changed since it was loaded: {path} has sha256 {fresh.loaded_from[1]}, "
                              f"not the loaded {sha}")
        if fresh.to_json() != cfg.to_json() or set(cfg.__dict__) != set(fresh.__dict__):
            raise PreregError(f"the config differs from its file {path}: a setting changed in place, or an attribute set "
                              f"on it, after loading; a confirmatory run takes the file's settings only")
        cfg = fresh
    from harness.analysis import multiplicity as weighted_holm

    label = "passed in by a test run (not confirmatory): "
    cmap = load_clusters(cfg, root) if clusters is None else dataclasses.replace(clusters, source=label + clusters.source)
    # the tables are frozen only when every one is: step 1's, step 2's addendum and step 3's (entry 106), so a test run
    # keeps its labelled every-p*-=-1 tables once step 2 is bound and step 3 is not yet
    unfrozen = PLACEHOLDER in (cfg.multiplicity_table, cfg.multiplicity_table_sha256, cfg.step2_addendum,
                               cfg.step2_addendum_sha256, cfg.step3_configurations, cfg.step3_configurations_sha256,
                               cfg.step3_fingerprint, cfg.step3_selection, cfg.step3_selection_sha256,
                               cfg.step3_addendum, cfg.step3_addendum_sha256)
    if multiplicity is None and testing and unfrozen:  # a test run without tables: labelled so, every p* = 1
        weighted = dataclasses.replace(weighted_holm.NO_TABLE, source="a test run (not confirmatory), the config's "
                                                                       "tables [FILL]: every p* = 1")
    elif multiplicity is None and testing and family == PLACEHOLDER:
        # every table frozen, the C1a choice not yet (it waits for the powers on step 3's calibration): a test run runs
        # C1a as the two-family IUT, so it takes the two-family final calibration, labelled so (the review of 9d95e92)
        finals = weighted_holm.load_final(cfg, root)
        if TWO_FAMILY not in finals:
            raise PreregError(f"the alpha' table holds no {TWO_FAMILY} family for a test run before the C1a choice")
        weighted = dataclasses.replace(finals[TWO_FAMILY], source="a test run (not confirmatory), the C1a choice "
                                                                  "[FILL]: the two-family final calibration; "
                                                                  + finals[TWO_FAMILY].source)
    else:
        weighted = (weighted_holm.load(cfg, root) if multiplicity is None  # refused while [FILL] (entry 74, ruling 3)
                    else dataclasses.replace(multiplicity, source=label + multiplicity.source))
    if not testing and PLACEHOLDER in (cfg.kat_hosts, cfg.kat_hosts_sha256):  # bound at load when filled
        raise PreregError("the known answers' host record is not bound (kat_hosts, kat_hosts_sha256 [FILL]): a "
                          "confirmatory run takes the host-change rule's record (data/analysis/kat-hosts.json) only by "
                          "its frozen sha256")
    config_file, config_sha256 = cfg.loaded_from or (None, None)
    return Preflight(cfg, family, inference, grouping, cmap, K, UNCALIBRATED, not testing, config_file, config_sha256,
                     answers, weighted_digest, weighted)


def clustered_contrasts(V: Vectors, index: QuestionIndex, bank_index: QuestionIndex | None, pre: Preflight,
                        keep_private: bool = False) -> tuple[dict[str, dict[str, Any]], Clusters, Clusters]:
    """The registered contrasts on clusters (entry 52), with the preflight's config, map, matrix and table only: the
    agent index's clusters and their resample matrix (``pre.K`` over the clusters, test runs only), the bank's own when
    it holds other questions (a bank-only extension, entry 41), then run_contrasts as calibrated. Returns the contrasts
    and the agent and bank Clusters (``keep_private``: each component keeps its statistic, for the weighted family)."""
    agent = pre.clusters.resampling(index, pre.cfg, pre.K)
    same_bank = bank_index is None or bank_index == index
    bank = agent if same_bank else pre.clusters.resampling(bank_index, pre.cfg)
    return run_contrasts(V, agent, pre.cfg, K_bank=None if same_bank else bank, calibration=pre.calibration,
                         keep_private=keep_private), agent, bank


# change-log entry 73: in exact arithmetic a resample whose drawn clusters agree has se* = 0 (t* undefined, never
# support, §4.2), but rounding can leave a tiny positive se* and a finite, huge t*. The calibrated statistics stay as
# they ran (the calibration simulated the same code); the confirmatory report counts such resamples, and adds a
# sensitivity analysis treating them as undefined when there are any.
NEAR_ZERO_SPREAD = 1e-9  # with NEAR_ZERO_RELATIVE: a resample's 0 < se* < max(this, that x se) is residue of no spread
NEAR_ZERO_RELATIVE = 1e-6  # issue 12 ruling: relative to the statistic's own se (the column-pair residue is ~3e-9)
UNBOUNDED = "UNBOUNDED"  # a reported interval bound beyond +-1, the end of every registered estimand's range


def near_zero_spread(s: Studentised) -> np.ndarray:
    """Entry 73 and the issue 12 ruling: the resamples whose se* is positive but below max(NEAR_ZERO_SPREAD,
    NEAR_ZERO_RELATIVE se), se the statistic's observed standard error (NEAR_ZERO_SPREAD alone without one)."""
    limit = NEAR_ZERO_SPREAD if s.se is None else max(NEAR_ZERO_SPREAD, NEAR_ZERO_RELATIVE * s.se)
    return (s.se_star > 0) & (s.se_star < limit)


def spread_undefined(s: Studentised) -> Studentised:
    """``s`` with those resamples' se* set to 0, so their t* is undefined and never supports a claim (the sensitivity
    analysis of entry 73)."""
    return dataclasses.replace(s, se_star=np.where(near_zero_spread(s), 0.0, s.se_star))


def _claims(blocks: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """The weighted family's decision per claim (multiplicity.decisions' blocks)."""
    return {cid: {k: b[k] for k in ("test", "p_test", "p_star", "step", "level", "claim")} for cid, b in blocks.items()}


def zero_spread(V: Vectors, agent: Clusters, bank: Clusters | None, cfg: PreregConfig, weighted: Any,
                contrasts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Entry 73, reporting only: per claim, the resamples with near-zero spread (``near_zero_spread``) summed over the
    statistics its p is computed from (every component, primary and adverse), on the run's own clusters and resample
    matrices (``bank`` the bank's own, or None when the bank resamples the agent clusters); and, when there are any,
    the six claims' tests and the weighted family (``weighted``, its tables) again with those resamples undefined (no
    support), beside the primary results (``contrasts``, the weighted family's)."""
    from harness.analysis import multiplicity as weighted_holm

    banked = agent if bank is None else bank
    counts = {}
    for cid, units in (("C1a", agent), ("C1b", agent), ("C2", banked), ("C3", banked), ("C4", banked), ("C5", agent)):
        seen: list[int] = []

        def count(s: Studentised, seen: list[int] = seen) -> Studentised:
            seen.append(int(near_zero_spread(s).sum()))
            return s

        counting = dataclasses.replace(units, spread=count)
        if cid == "C5":
            c5(V, counting, cfg, descriptive=False)
        else:
            {"C1a": c1a, "C1b": c1b, "C2": c2, "C3": c3, "C4": c4}[cid](V, counting, cfg)
        counts[cid] = {"resamples": sum(seen), "statistics": len(seen)}
    out = {"threshold": {"absolute": NEAR_ZERO_SPREAD, "relative": NEAR_ZERO_RELATIVE,
                         "rule": "0 < se* < max(absolute, relative x se)"},
           "counts": counts, "primary": _claims({cid: c["family"] for cid, c in contrasts.items()}), "sensitivity": None}
    if any(c["resamples"] for c in counts.values()):
        undefined = run_contrasts(V, dataclasses.replace(agent, spread=spread_undefined), cfg,
                                  K_bank=None if bank is None else dataclasses.replace(bank, spread=spread_undefined),
                                  intervals=False)
        out["sensitivity"] = _claims(weighted_holm.decisions({cid: c["p_test"] for cid, c in undefined.items()}, cfg,
                                                             weighted))
    return out


def _bound_shown(value: Any, computed: bool, owner: tuple[Any] | None) -> Any:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return UNBOUNDED if abs(value) > 1 else value
    if value is None and computed and owner is not None and owner[0] is not None:
        return UNBOUNDED  # bootstrap_t_bound's None with an estimate: the quantile at infinity
    return value


def bounded(x: Any, owner: tuple[Any] | None = None) -> Any:
    """Entry 73: a copy of a report with every interval bound beyond +-1 shown as UNBOUNDED, never as a number: every
    registered estimand, and every descriptive one beside it, is a rate or a difference or mean of rates, so a bound
    outside [-1, 1] says only that the interval's quantile ran away (a resample with near-zero spread). So is a side
    the interval computes and the bootstrap-t left open (None: the undefined t*'s share put its quantile at infinity)
    when its statistic has an estimate (``owner``: the ``estimate`` beside it, ``unconditional`` for
    ``unconditional_ci95``); a statistic without one keeps None, as does the side a one-sided interval leaves out."""
    if isinstance(x, list):
        return [bounded(v, owner) for v in x]
    if not isinstance(x, dict):
        return x
    if {"lo", "hi", "sided"} <= set(x):
        sides = {"two": ("lo", "hi"), "equivalence": ("lo", "hi"), "greater": ("lo",), "less": ("hi",)}.get(
            x["sided"], ())
        return {k: _bound_shown(v, k in sides, owner) if k in ("lo", "hi") else v for k, v in x.items()}
    own = (x["estimate"],) if "estimate" in x else owner
    unconditional = (x["unconditional"],) if "unconditional" in x else None
    return {k: bounded(v, unconditional if k == "unconditional_ci95" else own) for k, v in x.items()}


def check_test_run_split(splits: Iterable[str]) -> None:
    """Issue 19 ruling: a test run never reads the registered split (prereg §1), whatever the freeze status; it runs on
    dev inputs. Refuses when any of the inputs' ``splits`` is it."""
    found = sorted(set(splits))
    if REGISTERED_SPLIT in found:
        raise PreregError(f"a test run never reads the registered split {REGISTERED_SPLIT!r} (prereg §1; issue 19 "
                          f"ruling), whatever the freeze status: its inputs hold {found}; run it on dev inputs, or run "
                          f"the confirmatory analysis without testing")


def repo_root() -> Path:
    """The repository: the config's relative paths (the cluster map, the calibration table) are under it."""
    return Path(__file__).resolve().parent.parent.parent


def check_inference_digest(cfg: PreregConfig) -> tuple[str, str]:
    """Issue 12: the inference code the calibrated alpha' apply to must be the code the calibration ran (the current
    inference.inference_digest() against the frozen calibrated_inference_digest), and the cluster grouping added after
    it the code frozen at the freeze (inference.grouping_digest() against grouping_digest); [FILL] is refused. Returns
    both digests."""
    from harness.analysis.inference import grouping_digest, inference_digest

    if cfg.calibrated_inference_digest in (PLACEHOLDER, "", None):
        raise PreregError("the calibrated inference digest is not filled in (calibrated_inference_digest [FILL])")
    actual = inference_digest()
    if actual != cfg.calibrated_inference_digest:
        raise PreregError(f"the inference code changed since the calibration: inference digest {actual}, not the "
                          f"calibrated {cfg.calibrated_inference_digest} (inference.INFERENCE_FUNCTIONS); the calibrated "
                          f"alpha' do not apply to it")
    if cfg.grouping_digest in (PLACEHOLDER, "", None):
        raise PreregError("the grouping digest is not filled in (grouping_digest [FILL]): the cluster grouping "
                          "(inference.GROUPING_FUNCTIONS) is frozen at the freeze commit, and a confirmatory run refuses "
                          "without it")
    grouping = grouping_digest()
    if grouping != cfg.grouping_digest:
        raise PreregError(f"the cluster-grouping code changed since the freeze: grouping digest {grouping}, not the "
                          f"frozen {cfg.grouping_digest} (inference.GROUPING_FUNCTIONS)")
    return actual, grouping


def check_multiplicity_digest(cfg: PreregConfig) -> str:
    """Change-log entry 74: the weighted Holm (inference.MULTIPLICITY_FUNCTIONS, new code apart from the calibrated
    registry) must be the code frozen after its review (inference.multiplicity_digest() against multiplicity_digest);
    [FILL] is refused. Returns the digest."""
    from harness.analysis.inference import multiplicity_digest

    if cfg.multiplicity_digest in (PLACEHOLDER, "", None):
        raise PreregError("the multiplicity digest is not filled in (multiplicity_digest [FILL]): the weighted Holm "
                          "(inference.MULTIPLICITY_FUNCTIONS) is frozen after its review, and a confirmatory run refuses "
                          "without it")
    actual = multiplicity_digest()
    if actual != cfg.multiplicity_digest:
        raise PreregError(f"the multiplicity code changed since it was frozen: multiplicity digest {actual}, not the frozen "
                          f"{cfg.multiplicity_digest} (inference.MULTIPLICITY_FUNCTIONS)")
    return actual


def check_known_answers(cfg: PreregConfig) -> dict[str, str]:
    """Issue 12: the inference's known answers (harness/analysis/kat.py), recomputed in this process after every
    import, single-threaded (change-log entry 73: OPENBLAS_NUM_THREADS and OMP_NUM_THREADS 1 in the environment, and
    the BLAS running one thread where OpenBLAS reports it; refused otherwise): the calibrated paths' must be the frozen calibrated_kat_sha256 (the calibrated commit's own answer) and the
    cluster grouping's the frozen grouping_kat_sha256; [FILL] is refused, and so is an answer that fails to compute. A
    refusal on a mismatch names what in kat.environment() differs from the environment the answers were computed in
    (kat_environment), or says that nothing does: then the inference code's behaviour changed. Returns both answers."""
    from harness.analysis import kat

    env = kat.environment()
    variables = {k: env[k] for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")}
    if set(variables.values()) != {"1"} or env["blas threads (run time)"] not in ("1", "unknown"):
        raise PreregError(f"the confirmatory analysis runs single-threaded (change-log entry 73: the BLAS's reductions in "
                          f"one order every run), with OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1 set before the "
                          f"process starts; here {', '.join(f'{k}={v!r}' for k, v in variables.items())}, and the BLAS "
                          f"runs {env['blas threads (run time)']} threads (blas threads (run time))")
    if cfg.calibrated_kat_sha256 in (PLACEHOLDER, "", None):
        raise PreregError("the calibrated known answer is not filled in (calibrated_kat_sha256 [FILL])")
    if cfg.grouping_kat_sha256 in (PLACEHOLDER, "", None):
        raise PreregError("the cluster grouping's known answer is not filled in (grouping_kat_sha256 [FILL]): it is "
                          "frozen at the freeze commit, and a confirmatory run refuses without it")
    if cfg.multiplicity_kat_sha256 in (PLACEHOLDER, "", None):
        raise PreregError("the multiplicity's known answer is not filled in (multiplicity_kat_sha256 [FILL]): it is "
                          "frozen after the weighted Holm's review, and a confirmatory run refuses without it")
    try:
        answers = kat.known_answers()
    except Exception as exc:
        raise PreregError(f"the inference's known-answer test failed to run ({type(exc).__name__}: {exc}); the "
                          f"inference code does not behave as frozen") from exc
    frozen = {"calibrated": cfg.calibrated_kat_sha256, "grouping": cfg.grouping_kat_sha256,
              "multiplicity": cfg.multiplicity_kat_sha256}
    changed = [f"{part} {answers[part]}, not the frozen {frozen[part]}" for part in frozen if answers[part] != frozen[part]]
    if changed:
        now, then = env, cfg.kat_environment
        differs = [f"{k} {now.get(k)!r} (frozen {then.get(k)!r})" for k in sorted(set(now) | set(then))
                   if now.get(k) != then.get(k)]
        if differs:
            where = f"the environment differs from the one they were computed in: {'; '.join(differs)}"
        elif env["blas core (run time)"] == "unknown":  # the review of 93d4aca: say what the record cannot exclude
            where = ("the environment's recorded parts are the ones they were computed in, but the BLAS kernel chosen at "
                     "run time is not recorded (blas core (run time) unknown), so an environment cause cannot be "
                     "excluded")
        else:
            where = ("the environment is the one they were computed in (kat_environment, the BLAS kernel at run time "
                     "included), so the inference code's behaviour changed")
        raise PreregError(f"the inference's known answers changed: {'; '.join(changed)} (harness/analysis/kat.py); "
                          f"{where}")
    return answers


# --- vectors -----------------------------------------------------------------------------------


@dataclass
class Vectors:
    """The question vectors the contrasts read (§4.2, complete question vectors): per arm; C5's retry vectors
    per panel model ("num", "den", "n", "num_adverse"); the bank's union-only counts over each single source."""

    arms: dict[ArmKey, dict[str, np.ndarray]]
    c5: dict[str, dict[str, np.ndarray]]
    bank_union: dict[str, dict[str, np.ndarray]]


def _share(num: np.ndarray, den: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(1_S num/den, 1_S) for S = {den > 0}: the terms of a per-question mean over the questions that have a
    denominator."""
    m = (den > 0).astype(np.float64)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0), m


def _test_ps(s: Studentised, theta0: float, alternative: str) -> dict[str, float]:
    return {"boot_t": bootstrap_t_p(s, theta0, alternative), "cr2": cr2_p(s, theta0, alternative)}


def _max_ps(*ps: Mapping[str, float]) -> dict[str, float]:
    return {t: max(p[t] for p in ps) for t in TESTS}


def _two_sided(up: Mapping[str, float], down: Mapping[str, float]) -> dict[str, float]:
    return {t: min(1.0, 2 * min(up[t], down[t])) for t in TESTS}


# --- C1 (§2, §4.2, §4.3) ------------------------------------------------------------------------


def _accuracy_terms(vec: Mapping[str, np.ndarray], c: float, missing_correct: bool) -> Term:
    correct = vec["correct"] + (vec["missing"] if missing_correct else 0.0)
    x, m = _share(correct, vec["att"])
    return Term(c, x * m, m)


def _c1_stat(K, V: Vectors, cfg: PreregConfig, models: Sequence[str], cur_missing: bool, cen_missing: bool):
    """The equal-weight mean over ``models`` of curated-minus-census accuracy, with missing attempts scored
    correct in the curated (``cur_missing``) or census (``cen_missing``) arm."""
    w = 1.0 / len(models)
    terms = []
    for m in models:
        terms.append(_accuracy_terms(V.arms[cfg.prompt(m, cfg.curated_prompt_cohort)], w, cur_missing))
        terms.append(_accuracy_terms(V.arms[cfg.prompt(m, cfg.census_prompt_cohort)], -w, cen_missing))
    return studentise(K, terms)


def c1a(V: Vectors, K: np.ndarray, cfg: PreregConfig) -> dict[str, Any]:
    """C1a (§2): curated-prompt minus census-prompt accuracy, two-sided; two-family IUT (p_up and p_down each the
    maximum over the families; p = min(1, 2 min)), or one pooled component over P (``c1a_pooled``). Each
    direction's p is the maximum of the primary and the adverse analysis (census missing correct for "up",
    curated missing correct for "down")."""
    groups = {"pooled over P": cfg.c1a_models} if cfg.c1a_pooled else {m: (m,) for m in cfg.models}
    comps = []
    for label, ms in groups.items():
        prim = _c1_stat(K, V, cfg, ms, False, False)
        up = _max_ps(_test_ps(prim, 0.0, "greater"), _test_ps(_c1_stat(K, V, cfg, ms, False, True), 0.0, "greater"))
        down = _max_ps(_test_ps(prim, 0.0, "less"), _test_ps(_c1_stat(K, V, cfg, ms, True, False), 0.0, "less"))
        comps.append({"label": label, "estimate": prim.estimate, "se": prim.se, "p_up": up, "p_down": down, "_s": prim})
    p_up = {t: max(c["p_up"][t] for c in comps) for t in TESTS}
    p_down = {t: max(c["p_down"][t] for c in comps) for t in TESTS}
    p = _two_sided(p_up, p_down)
    direction = {t: "curated higher" if p_up[t] < p_down[t] else "census higher" if p_down[t] < p_up[t] else "none"
                 for t in TESTS}
    return {"id": "C1a", "section": "§2 C1a, §4.2", "metric": "accuracy", "sided": "two", "pooled": cfg.c1a_pooled,
            "models": list(cfg.c1a_models), "p_up": p_up, "p_down": p_down, "p_test": p, "direction": direction,
            "components": comps}


def c1b(V: Vectors, K: np.ndarray, cfg: PreregConfig) -> dict[str, Any]:
    """C1b (§2): TOST at +-margin, an IUT over both families and both bounds: p = the maximum of four one-sided
    p's, each the maximum of its primary and adverse analysis (missing attempts assigned toward that bound)."""
    comps = []
    for m in cfg.models:
        prim = _c1_stat(K, V, cfg, (m,), False, False)
        lower = _max_ps(_test_ps(prim, -cfg.c1_margin, "greater"),
                        _test_ps(_c1_stat(K, V, cfg, (m,), False, True), -cfg.c1_margin, "greater"))
        upper = _max_ps(_test_ps(prim, cfg.c1_margin, "less"),
                        _test_ps(_c1_stat(K, V, cfg, (m,), True, False), cfg.c1_margin, "less"))
        comps.append({"label": m, "estimate": prim.estimate, "se": prim.se, "p_lower": lower, "p_upper": upper,
                      "_s": prim})
    p = {t: max(max(c["p_lower"][t], c["p_upper"][t]) for c in comps) for t in TESTS}
    return {"id": "C1b", "section": "§2 C1b, §4.2", "metric": "accuracy", "sided": "equivalence",
            "margin": cfg.c1_margin, "p_test": p, "components": comps}


# --- C5 (§2, §4.2, §4.3) ------------------------------------------------------------------------


def c5(V: Vectors, K: np.ndarray, cfg: PreregConfig, descriptive: bool = True) -> dict[str, Any]:
    """C5 (§2, entries 36-37): the equal-weight mean over P of theta_m = sum F (Y_fb - Y_gen) / sum F, one-sided,
    each theta_m recomputed in every resample and no model ever dropped. A model with no flagged output makes
    theta non-estimable: p = 1. p = max(primary, adverse), the adverse analysis scoring a missing generic retry
    correct (a missing feedback retry already counts wrong). Descriptive: each theta_m, the flag-weighted mean,
    leave-one-family-out means, the two-family IUT and the unconditional effects."""
    models = cfg.c5_models
    w = 1.0 / len(models)
    prim = studentise(K, [Term(w, V.c5[m]["num"], V.c5[m]["den"]) for m in models])
    adv = studentise(K, [Term(w, V.c5[m]["num_adverse"], V.c5[m]["den"]) for m in models])
    p = _max_ps(_test_ps(prim, 0.0, "greater"), _test_ps(adv, 0.0, "greater"))
    comp = {"label": "pooled over P", "estimate": prim.estimate, "se": prim.se, "p_greater": p, "_s": prim}
    out = {"id": "C5", "section": "§2 C5, §4.2", "metric": "retry with feedback - generic retry, pooled over P",
           "sided": "greater", "models": list(models), "p_test": p, "sesoi": cfg.c5_sesoi,
           "estimable": prim.estimate is not None, "components": [comp]}
    if not descriptive:
        return out
    per_model = {}
    for m in models:
        v = V.c5[m]
        s = studentise(K, [Term(1.0, v["num"], v["den"])])
        x, msk = _share(v["num"], v["n"])
        unc = studentise(K, [Term(1.0, x * msk, msk)])
        per_model[m] = {"estimate": s.estimate, "ci95": _interval(s, 0.05, "two", "boot_t"), "flagged": int(v["den"].sum()),
                        "p_greater": _test_ps(s, 0.0, "greater"), "unconditional": unc.estimate,
                        "unconditional_ci95": _interval(unc, 0.05, "two", "boot_t")}
    flag_weighted = studentise(K, [Term(1.0, sum(V.c5[m]["num"] for m in models), sum(V.c5[m]["den"] for m in models))])
    fam = cfg.families
    lofo = {}
    for f in sorted(set(fam.values())):
        rest = [m for m in models if fam.get(m, m) != f]
        if rest:
            s = studentise(K, [Term(1.0 / len(rest), V.c5[m]["num"], V.c5[m]["den"]) for m in rest])
            lofo[f] = {"models": rest, "estimate": s.estimate, "ci95": _interval(s, 0.05, "two", "boot_t")}
    two = [m for m in cfg.models if m in per_model]
    out["descriptive"] = {
        "per_model": per_model,
        "flag_weighted": {"estimate": flag_weighted.estimate, "ci95": _interval(flag_weighted, 0.05, "two", "boot_t")},
        "leave_one_family_out": lofo,
        "two_family_iut": {"models": two, "p": {t: max(per_model[m]["p_greater"][t] for m in two) for t in TESTS}
                           if two else None},
    }
    return out


def c4_retry(V: Vectors, K: np.ndarray, cfg: PreregConfig) -> dict[str, Any]:
    """The old C4 (net accuracy change of the union checker arm minus each single source, per family), descriptive
    since change-log entry 35: estimates and 95% percentile-t intervals."""
    comps = []
    for m in cfg.models:
        union = V.arms[cfg.checker(m, cfg.union_source)]
        for s in (cfg.curated_source, cfg.census_source):
            single = V.arms[cfg.checker(m, s)]
            xu, mu = _share(union["delta"], union["att"])
            xs, ms = _share(single["delta"], single["att"])
            st = studentise(K, [Term(1.0, xu * mu, mu), Term(-1.0, xs * ms, ms)])
            comps.append({"label": f"{m}: union - {s}", "estimate": st.estimate, "ci95": _interval(st, 0.05, "two", "boot_t")})
    return {"id": "C4 retry (descriptive)", "section": "change-log entry 35", "metric": "net_change", "components": comps}


# --- the bank contrasts (§2, §4.1) --------------------------------------------------------------


def _bank_design(V: Vectors, cfg: PreregConfig) -> dict[str, np.ndarray]:
    """The frozen bank's per-question item counts, the same for every bank source, with exactly one untouched gold
    per question (§4.1)."""
    sources = (cfg.curated_source, cfg.census_source, *cfg.masked_sources)
    first = V.arms[cfg.bank(cfg.curated_source)]
    for s in sources[1:]:
        other = V.arms[cfg.bank(s)]
        for k in ("W", "tw", "C_untouched", "C_rewrite", "C_neutral"):
            if not np.array_equal(first[k], other[k]):
                raise PreregError(f"bank sources {cfg.curated_source} and {s} differ in their {k} items per question")
    if not np.array_equal(first["C_untouched"], np.ones_like(first["C_untouched"])):
        raise PreregError("the bank must hold exactly one untouched gold per question (§4.1)")
    return {k: first[k] for k in ("W", "tw", "C_untouched", "C_rewrite", "C_neutral")}


def _f_q(gold: np.ndarray, rewrites: np.ndarray, k: np.ndarray, cfg: PreregConfig) -> np.ndarray:
    """Each question's false-flag rate (§4.1, change-log entry 41): 0.5 f_gold + 0.5 f_rewrite when the question has
    rewrites (k_q >= 1), else f_gold; f_gold alone everywhere when the rewrites were not adopted."""
    if not cfg.rewrites_adopted:
        return gold
    share = np.where(k > 0, rewrites / np.where(k > 0, k, 1.0), 0.0)
    return np.where(k > 0, F_WEIGHTS[0] * gold + F_WEIGHTS[1] * share, gold)


def _f_terms(gold: np.ndarray, rewrites: np.ndarray, k: np.ndarray, c: float, cfg: PreregConfig) -> list[Term]:
    """Terms of c f, f the mean over questions of f_q (§4.1)."""
    return [Term(c, _f_q(gold, rewrites, k, cfg), np.ones_like(gold))]


def _f_diff_terms(a: Mapping[str, np.ndarray], b: Mapping[str, np.ndarray], k: np.ndarray, cfg: PreregConfig) -> list[Term]:
    """f_a - f_b on the same questions: the mean of the per-question differences of f_q."""
    fa = _f_q(a["C_F_untouched"], a["C_F_rewrite"], k, cfg)
    fb = _f_q(b["C_F_untouched"], b["C_F_rewrite"], k, cfg)
    return [Term(1.0, fa - fb, np.ones_like(k))]


def _rate(vec: Mapping[str, np.ndarray], num: str, den: str) -> float | None:
    x, m = _share(vec[num], vec[den])
    return float((x * m).sum() / m.sum()) if m.sum() > 0 else None


def source_rates(vec: Mapping[str, np.ndarray], cfg: PreregConfig) -> dict[str, Any]:
    """A bank source's per-question-mean rates (§4.1): recall on all answer-changing items and on type-wrong items,
    f_gold, f_rewrite, f, and, descriptive, precision at pi (and at 10%, 50%) and the answer-neutral stratum."""
    r, r_tw = _rate(vec, "W_F", "W"), _rate(vec, "W_F_tw", "tw")
    fg, fr = _rate(vec, "C_F_untouched", "C_untouched"), _rate(vec, "C_F_rewrite", "C_rewrite")
    f = float(_f_q(vec["C_F_untouched"], vec["C_F_rewrite"], vec["C_rewrite"], cfg).mean())

    def precision(pi: float) -> float | None:
        if r is None or f is None or pi * r + (1 - pi) * f <= 0:
            return None
        return pi * r / (pi * r + (1 - pi) * f)

    return {"recall": r, "recall_type_wrong": r_tw, "f_gold": fg, "f_rewrite": fr, "f": f,
            "precision": precision(cfg.prevalence), "precision_at": {f"{pi:g}": precision(pi) for pi in cfg.prevalences},
            "false_flag_rate_neutral": _rate(vec, "C_F_neutral", "C_neutral"),
            "questions_without_type_wrong": int((vec["tw"] <= 0).sum()),
            "questions_without_rewrites": int((vec["C_rewrite"] <= 0).sum())}


def c2(V: Vectors, K: np.ndarray, cfg: PreregConfig) -> dict[str, Any]:
    """C2 (§2, entries 40 and 106): two-sided, the same construction as C1a. For each direction an IUT of (i) the
    false-flag rate difference and (ii) the type-wrong recall difference, each one-sided: up (curated better) is
    f_census - f_curated > 0 and r_curated - r_census > 0, p_up = max(p_i_up, p_ii_up); down (census better) is both
    differences < 0, p_down = max(p_i_down, p_ii_down); p = min(1, 2 min(p_up, p_down)). Opposite directions on the two
    components reject neither direction."""
    cur, cen = V.arms[cfg.bank(cfg.curated_source)], V.arms[cfg.bank(cfg.census_source)]
    d = _bank_design(V, cfg)
    s_f = studentise(K, _f_diff_terms(cen, cur, d["C_rewrite"], cfg))
    x, m = _share(cur["W_F_tw"] - cen["W_F_tw"], d["tw"])
    s_r = studentise(K, [Term(1.0, x * m, m)])
    comps = [{"label": "false-flag rate (census - curated)", "estimate": s_f.estimate, "se": s_f.se,
              "p_greater": _test_ps(s_f, 0.0, "greater"), "p_less": _test_ps(s_f, 0.0, "less"), "_s": s_f},
             {"label": "type-wrong recall (curated - census)", "estimate": s_r.estimate, "se": s_r.se,
              "p_greater": _test_ps(s_r, 0.0, "greater"), "p_less": _test_ps(s_r, 0.0, "less"), "_s": s_r}]
    up, down = _max_ps(*[c["p_greater"] for c in comps]), _max_ps(*[c["p_less"] for c in comps])
    direction = {t: "curated better" if up[t] < down[t] else "census better" if down[t] < up[t] else "none"
                 for t in TESTS}
    return {"id": "C2", "section": "§2 C2, §4.1", "metric": "false-flag rate, type-wrong recall", "sided": "two",
            "p_up": up, "p_down": down, "p_test": _two_sided(up, down), "direction": direction,
            "min_effect": cfg.c2_min_effect,
            "rates": {cfg.curated_source: source_rates(cur, cfg), cfg.census_source: source_rates(cen, cfg)},
            "components": comps}


def c2_at_least(components: Sequence[Mapping[str, Any]], direction: str, key: str, min_effect: float) -> dict[str, bool]:
    """C2's "at least 5 points" per component (§2, entry 106), read in C2's direction from its claim-level interval
    ``key`` (two-sided): curated better needs the lower bound above +min_effect, census better the upper bound below
    -min_effect; with no direction, neither."""
    up, down = direction == "curated better", direction == "census better"
    return {comp["label"]: bool((up and comp[key]["lo"] is not None and comp[key]["lo"] > min_effect)
                                or (down and comp[key]["hi"] is not None and comp[key]["hi"] < -min_effect))
            for comp in components}


def c3(V: Vectors, K: np.ndarray, cfg: PreregConfig) -> dict[str, Any]:
    """C3 (§2): f_curated - mean(f over the three masked draws), two-sided (p = 2 min)."""
    cur = V.arms[cfg.bank(cfg.curated_source)]
    draws = [V.arms[cfg.bank(s)] for s in cfg.masked_sources]
    d = _bank_design(V, cfg)
    mean_draw = {k: np.mean([v[k] for v in draws], axis=0) for k in ("C_F_untouched", "C_F_rewrite")}
    s = studentise(K, _f_diff_terms(cur, mean_draw, d["C_rewrite"], cfg))
    up, down = _test_ps(s, 0.0, "greater"), _test_ps(s, 0.0, "less")
    direction = {t: "curated higher" if up[t] < down[t] else "generated higher" if down[t] < up[t] else "none"
                 for t in TESTS}
    rates = {cfg.curated_source: source_rates(cur, cfg), **{n: source_rates(v, cfg) for n, v in zip(cfg.masked_sources, draws)}}
    return {"id": "C3", "section": "§2 C3", "metric": "false-flag rate", "sided": "two", "p_up": up, "p_down": down,
            "p_test": _two_sided(up, down), "direction": direction, "sesoi": cfg.c3_sesoi, "rates": rates,
            "components": [{"label": "curated - mean(masked)", "estimate": s.estimate, "se": s.se, "p_up": up,
                            "p_down": down, "_s": s}]}


def c4(V: Vectors, K: np.ndarray, cfg: PreregConfig) -> dict[str, Any]:
    """C4 (§2, §4.1): Delta_s = pi mean_q(G_q / W_q) - (1 - pi) f^u_s for s in {curated, census}, G the wrong items the
    union flags and s does not and f^u the union-only false-flag rate on the same strata; IUT, p = max."""
    d = _bank_design(V, cfg)
    comps = []
    for s in (cfg.curated_source, cfg.census_source):
        u = V.bank_union[s]
        x, m = _share(u["wrong"], d["W"])
        terms = [Term(cfg.prevalence, x * m, m)] + _f_terms(u["gold"], u["rewrites"], d["C_rewrite"],
                                                             -(1 - cfg.prevalence), cfg)
        st = studentise(K, terms)
        comps.append({"label": f"union - {s}", "estimate": st.estimate, "se": st.se,
                      "p_greater": _test_ps(st, 0.0, "greater"), "_s": st})
    return {"id": "C4", "section": "§2 C4, §4.1", "metric": "net detection benefit of the union", "sided": "greater",
            "p_test": _max_ps(*[c["p_greater"] for c in comps]), "sesoi": cfg.c4_sesoi, "components": comps}


# --- intervals and the family (§4.4) ------------------------------------------------------------


def _t_quantile(q: float, df: float) -> float:
    lo, hi = -1e3, 1e3
    for _ in range(200):
        mid = (lo + hi) / 2
        if float(t_sf(mid, df)) > 1 - q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _bound(s: Studentised, alpha: float, side: str, test: str) -> float | None:
    if s.estimate is None:
        return None
    if test == "boot_t":
        return bootstrap_t_bound(s, alpha, side)
    q = _t_quantile(1 - alpha, len(s.psi) - 1)
    return s.estimate - q * s.se if side == "lower" else s.estimate + q * s.se


def _interval(s: Studentised, alpha: float, sided: str, test: str) -> dict[str, Any]:
    """Bounds at level alpha: two-sided (each end at alpha / 2), "greater" (a lower bound), or "equivalence" (the
    (1 - 2 alpha) TOST interval, each end at alpha)."""
    if sided == "greater":
        return {"lo": _bound(s, alpha, "lower", test), "hi": None, "level": 1 - alpha, "sided": "greater"}
    each = alpha if sided == "equivalence" else alpha / 2
    return {"lo": _bound(s, each, "lower", test), "hi": _bound(s, each, "upper", test),
            "level": 1 - 2 * each, "sided": sided}


def run_contrasts(V: Vectors, K: np.ndarray, cfg: PreregConfig, *, K_bank: np.ndarray | None = None,
                  calibration: Calibration = UNCALIBRATED,
                  intervals: bool = True, keep_private: bool = False) -> dict[str, dict[str, Any]]:
    """The six claim-level p-values and Holm across them (§1, §4.2, §4.4). Each claim's family p is its
    calibrated p under the test its calibration selected (1 when uncalibrated, or when both tests failed). With
    ``intervals``, each component gets its unadjusted 95% interval and the Bonferroni claim-level interval at
    0.05/6 (at the calibrated alpha'(0.05/6) of the selected test when calibrated), and C2's "at least 5 points"
    per component from that lower bound; ``intervals=False`` (simulation) skips them and the descriptive parts."""
    if calibration.family is not None and calibration.family != (POOLED if cfg.c1a_pooled else TWO_FAMILY):
        raise PreregError(f"the calibration is the {calibration.family} family's, not the config's C1a family "
                          f"{cfg.c1a_family!r}")
    Kb = K if K_bank is None else K_bank  # the bank's own questions (a bank-only extension, entry 41)
    family = [c1a(V, K, cfg), c1b(V, K, cfg), c2(V, Kb, cfg), c3(V, Kb, cfg), c4(V, Kb, cfg),
              c5(V, K, cfg, descriptive=intervals)]
    bonferroni = cfg.alpha / len(CONTRASTS)
    for c in family:
        # every claim reads the chosen family's levels (the common step-1 factor differs between the families); a
        # pooled C1a is calibration.py's "C1a pooled"
        test, ap = calibration.entry("C1a pooled" if c["id"] == "C1a" and cfg.c1a_pooled else c["id"])
        c["calibration"] = {"test": test, "alpha_prime": ap, "table": calibration.source}
        c["p"] = calibrated_p(c["p_test"][test], ap, cfg.alpha) if test is not None else 1.0
    ps = [c["p"] for c in family]
    adjusted, levels = holm(ps), holm_levels(ps, cfg.alpha)
    # CR2 beside the primary, descriptive (issue 19, after change-log entry 46 left the fallback unused): its nominal
    # p's under a nominal Holm, never calibrated and never in the family
    cr2_adjusted = holm([c["p_test"]["cr2"] for c in family])
    out = {}
    for c, p_adj, level, cr2_adj in zip(family, adjusted, levels, cr2_adjusted):
        c["p_holm"], c["holm_level"] = p_adj, level
        c["reject"] = c["claim"] = p_adj <= cfg.alpha
        c["cross_check_cr2"] = {"descriptive": True, "test": "cr2", "p_test": c["p_test"]["cr2"],
                                "p_holm_nominal": cr2_adj, "claim_nominal": cr2_adj <= cfg.alpha,
                                "note": "CR2-Satterthwaite at nominal levels, a descriptive cross-check beside the "
                                        "primary: not calibrated, not in the Holm family, never a claim"}
        if intervals:
            test, ap = c["calibration"]["test"], c["calibration"]["alpha_prime"]
            shown = test or "boot_t"
            level_b = ap[len(CONTRASTS)] if ap is not None else bonferroni
            for comp in c["components"]:
                comp["ci95"] = _interval(comp["_s"], 0.05, "two", shown)  # §4.4: the unadjusted two-sided 95%
                comp["ci_bonferroni"] = _interval(comp["_s"], level_b, c["sided"], shown) | {"calibrated": ap is not None,
                                                                                            "test": shown}
                comp["cross_check_cr2"] = {"ci95": _interval(comp["_s"], 0.05, "two", "cr2"),
                                           "ci_bonferroni": _interval(comp["_s"], bonferroni, c["sided"], "cr2")}
            if c["id"] == "C2":  # entry 106: in C2's direction, by its primary (or chosen) test
                c["at_least_min_effect"] = c2_at_least(c["components"], c["direction"][shown], "ci_bonferroni",
                                                       cfg.c2_min_effect)
                c["both_at_least_min_effect"] = all(c["at_least_min_effect"].values())
        if not keep_private:
            for comp in c["components"]:
                comp.pop("_s", None)
        out[c["id"]] = c
    return out


# --- from units to vectors (§4.2, §4.3) ---------------------------------------------------------


def c5_vectors(curated: Sequence[Unit], generic: Sequence[Unit], index: QuestionIndex) -> dict[str, np.ndarray]:
    """C5's paired retry vectors for one model (§2 C5): per question sum F (Y_fb - Y_gen), sum F and the attempts;
    ``num_adverse`` scores a missing generic retry correct (§4.3). The generic arm must have retried exactly the
    curated-flagged outputs."""
    gen = {u.attempt.run_id: u for u in generic}
    if {u.attempt.run_id for u in curated} != set(gen):
        raise PreregError("C5: the curated and generic arms do not hold the same attempts")
    num, den, n, gen_missing = (np.zeros(len(index)) for _ in range(4))
    for u in curated:
        g = gen[u.attempt.run_id]
        flagged = u.checked is not None and u.checked.status == "F"
        g_flagged = g.checked is not None and g.checked.status == "F"
        if flagged != g_flagged or flagged != bool(g.checked and g.checked.retried):
            raise PreregError(f"C5: output {u.attempt.run_id} is not flagged and retried alike in the curated and "
                              f"generic arms; the generic control runs on the same flagged set")
        i = index.position(u.cluster)
        n[i] += 1
        if flagged:
            num[i] += float(bool(u.checked.retry_correct)) - float(bool(g.checked.retry_correct))
            den[i] += 1
            gen_missing[i] += float(g.checked.retry_missing)
    return {"num": num, "den": den, "n": n, "num_adverse": num - gen_missing}


def bank_union_vectors(curated: Sequence[Unit], census: Sequence[Unit], index: QuestionIndex) -> dict[str, dict]:
    """C4's union-only counts (§4.1): per question, the items the union (curated or census) flags and the single
    source s does not, over each s: wrong items, the gold and the rewrites."""
    cen = {u.attempt.run_id: u for u in census}
    if set(cen) != {u.attempt.run_id for u in curated}:
        raise PreregError("C4: the curated and census bank arms do not hold the same items")
    out = {s: {k: np.zeros(len(index)) for k in ("wrong", "gold", "rewrites")} for s in ("curated", "census")}
    kinds = {"answer_changing": "wrong", "untouched": "gold", "rewrite": "rewrites"}
    for u in curated:
        v = cen[u.attempt.run_id]
        kind = kinds.get(u.checked.bank_item)
        if kind is None:
            continue
        i = index.position(u.cluster)
        f_cur, f_cen = u.checked.status == "F", v.checked.status == "F"
        out["curated"][kind][i] += float(f_cen and not f_cur)
        out["census"][kind][i] += float(f_cur and not f_cen)
    return out


def _require_arms(arms: Mapping[ArmKey, Sequence[Unit]], cfg: PreregConfig) -> None:
    absent = [k.label for k in cfg.required_arms() if k not in arms]
    if absent:
        raise PreregError(f"the registered analysis needs arms that are not in the inputs: {absent}")


def vectors_from_units(arms: Mapping[ArmKey, Sequence[Unit]], index: QuestionIndex, cfg: PreregConfig,
                       bank_index: QuestionIndex | None = None) -> Vectors:
    """The registered arms' question vectors, C5's retry vectors per panel model and C4's union-only counts."""
    _require_arms(arms, cfg)
    bank_index = bank_index or index  # the bank may hold more questions than the agent runs (entry 41)
    vecs = {k: question_vectors(arms[k], bank_index if k.cohort == BANK else index) for k in cfg.required_arms()}
    c5v = {m: c5_vectors(arms[cfg.checker(m, cfg.curated_source)], arms[cfg.checker(m, cfg.generic_source)], index)
           for m in cfg.c5_models}
    union = bank_union_vectors(arms[cfg.bank(cfg.curated_source)], arms[cfg.bank(cfg.census_source)], bank_index)
    return Vectors(vecs, c5v, {cfg.curated_source: union["curated"], cfg.census_source: union["census"]})


def missing_shares(arms: Mapping[ArmKey, Sequence[Unit]], cfg: PreregConfig) -> dict[str, float]:
    """Per registered agent arm: (missing attempts + missing retries) / N_att (§4.3)."""
    out = {}
    for k in cfg.required_arms():
        if k.cohort == BANK:
            continue
        units = arms[k]
        lost = sum(1 for u in units if u.attempt.status == "missing") + sum(int(u.ind["retry_missing"]) for u in units)
        out[k.label] = lost / len(units) if units else 0.0
    return out


def compared_arms(cfg: PreregConfig) -> dict[str, list[ArmKey]]:
    """The arms each contrast compares (§4.3 "compromised")."""
    prompts = [cfg.prompt(m, c) for m in cfg.models for c in (cfg.curated_prompt_cohort, cfg.census_prompt_cohort)]
    pooled = [cfg.prompt(m, c) for m in cfg.c1a_models for c in (cfg.curated_prompt_cohort, cfg.census_prompt_cohort)]
    return {"C1a": pooled if cfg.c1a_pooled else prompts, "C1b": prompts, "C2": [], "C3": [], "C4": [],
            "C5": [cfg.checker(m, s) for m in cfg.c5_models for s in (cfg.curated_source, cfg.generic_source)]}


def _check_scorer(arms: Mapping[ArmKey, Sequence[Unit]], cfg: PreregConfig) -> None:
    if cfg.scorer_version in (PLACEHOLDER, "", None):
        raise PreregError("the frozen scorer version is not filled in (prereg §3 '[FILL] version hash')")
    for k in cfg.required_arms():
        if k.cohort == BANK:
            continue
        other = {u.attempt.rules_version for u in arms[k] if u.attempt.output} - {cfg.scorer_version}
        if other:
            raise PreregError(f"arm {k.label}: outputs scored under {sorted(map(str, other))}, not the frozen "
                              f"scorer {cfg.scorer_version}")


def _check_repetitions(arms: Mapping[ArmKey, Sequence[Unit]], index: QuestionIndex, cfg: PreregConfig) -> None:
    """§4.4 "Fixed stopping": in every registered agent arm, every question has exactly the repetitions 0..n_reps - 1
    (absent ones as missing attempts, inputs.complete_design), and every base question of the population is present."""
    want = list(range(cfg.n_reps))
    for k in cfg.required_arms():
        if k.cohort == BANK:
            continue
        reps: dict[str, list[int]] = {}
        for u in arms[k]:
            reps.setdefault(u.attempt.question_id, []).append(u.attempt.repetition)
        bad = sorted(q for q, r in reps.items() if sorted(r) != want)
        if bad:
            raise PreregError(f"arm {k.label}: question {bad[0]} has repetitions {sorted(reps[bad[0]])}, not exactly "
                              f"repetitions 0..{cfg.n_reps - 1} (prereg §4.4 'Fixed stopping'; {len(bad)} such questions)")
        absent = set(index.questions) - {u.cluster for u in arms[k]}
        if absent:
            raise PreregError(f"arm {k.label}: {len(absent)} base questions of the population have no attempts, "
                              f"e.g. {sorted(absent)[:3]}")


def load_calibration(cfg: PreregConfig, root: Path) -> Calibration:
    """The frozen calibration table's chosen family (§4.2, issue 19), or UNCALIBRATED while the config's entry is
    [FILL]. The C1a choice must be recorded; a chosen family whose step-2 FWER check failed gives every claim p = 1
    (ruling 3), whatever the other family did."""
    if cfg.calibration in (PLACEHOLDER, "", None):
        return UNCALIBRATED
    return Calibration.load(Path(root) / cfg.calibration, cfg.calibration_sha256, cfg.require_c1a_choice())


def resamples(index: QuestionIndex, cfg: PreregConfig) -> np.ndarray:
    """The registered resample matrix (§4.2): B unstratified resamples of the base questions, seed 20261026."""
    return resample_matrix(index, cfg.n_boot, seed=cfg.seed, scheme="unstratified")


def confirmatory(arms: Mapping[ArmKey, Sequence[Unit]], index: QuestionIndex, cfg: PreregConfig,
                 K: np.ndarray | None = None, multiplicity: Any = None,
                 code_digest: str | None = None, bank_index: QuestionIndex | None = None,
                 clusters: ClusterMap | None = None, simulated: Mapping[str, Any] | None = None, *,
                 root: Path | None = None, testing: bool = False) -> dict[str, Any]:
    """The registered analysis on real records (prereg §1-§4): the six contrasts under the weighted Holm (entries 74
    and 76: multiplicity.weighted_family on the contrasts run uncalibrated), the missing share of every compared arm,
    each contrast's conclusion (a compromised arm is reported, §4.3), C5's survivor mean (descriptive) and the
    descriptive retry contrast. No confirmatory inference until the C1a choice is recorded (issue 19 ruling 2), nor on
    inference code other than the calibrated, the frozen grouping and the frozen multiplicity (issue 12, entry 74),
    nor without the frozen cluster map (entry 52; the bootstrap and CR2 resample its clusters) and the weighted Holm's
    frozen tables. The registered preflight runs first, before any data is read: the map and the tables are read only
    from the config's files, under ``root`` (the repository), and the config must be the one loaded from its file;
    ``clusters``, ``K`` (over the clusters) and ``multiplicity`` (the tables) are taken only in a test run
    (``testing``), reported as not confirmatory. The report records the config file's sha256,
    and the real cluster counts and sizes beside what the calibration simulated (``simulated``: its table's
    description)."""
    pre = preflight(cfg, repo_root() if root is None else Path(root), clusters=clusters, K=K, multiplicity=multiplicity,
                    testing=testing)
    if not pre.confirmatory:  # a test run (issue 19 ruling): never the registered split, whatever the freeze status
        check_test_run_split(u.attempt.split for units in arms.values() for u in units)
    # the data preparation reads the attempt units with the caller's config (the preflight compared it with its file);
    # the inference and the report use the preflight's own copy, which nothing outside the two digests holds
    _require_arms(arms, cfg)
    _check_scorer(arms, cfg)
    _check_repetitions(arms, index, cfg)
    V = vectors_from_units(arms, index, cfg, bank_index)
    from harness.analysis import multiplicity as weighted_holm

    contrasts, K, K_bank = clustered_contrasts(V, index, bank_index, pre, keep_private=True)
    contrasts = weighted_holm.weighted_family(contrasts, pre.cfg, pre.multiplicity)  # entries 74, 76
    spread = zero_spread(V, K, None if K_bank is K else K_bank, pre.cfg, pre.multiplicity, contrasts)
    shares = missing_shares(arms, cfg)
    frozen = pre.cfg
    conclusions = {}
    for cid, keys in compared_arms(frozen).items():
        worst = max((shares[k.label] for k in keys), default=0.0)
        conclusions[cid] = {"claim": contrasts[cid]["family"]["claim"], "compromised": worst > frozen.missing_limit,
                            "max_missing_share": worst, "test": contrasts[cid]["family"]["test"]}
    survivors = [m for m in frozen.c5_models if max(shares[frozen.checker(m, s).label] for s in (
        frozen.curated_source, frozen.generic_source)) <= frozen.missing_limit]
    per_model = contrasts["C5"]["descriptive"]["per_model"]
    contrasts["C5"]["descriptive"]["survivors"] = {
        "models": survivors, "estimate": None if not survivors or any(per_model[m]["estimate"] is None for m in survivors)
        else float(np.mean([per_model[m]["estimate"] for m in survivors]))}
    return {"schema": "prereg-analysis/6", "document": "paper/preregistration.md, draft 4 (entry 40; issue 19 rulings)",
            "confirmatory": pre.confirmatory, "config": frozen.to_json(), "config_file": pre.config_file,
            "config_sha256": pre.config_sha256, "code_digest": code_digest, "inference_digest": pre.inference_digest,
            "grouping_digest": pre.grouping_digest, "multiplicity_digest": pre.multiplicity_digest,
            "known_answers": pre.known_answers, "c1a_family": pre.family,
            "multiplicity": {"procedure": weighted_holm.PROCEDURE, "tables": pre.multiplicity.source,
                             "weights": frozen.holm_weights, "step2_addendum": pre.multiplicity.step2},
            "resampling": "unstratified paired cluster bootstrap of complete cluster vectors (§4.2; clusters of base "
                          "questions, entry 52)",
            "clusters": {"map": pre.clusters.source, "agent": K.summary(), "bank": K_bank.summary(),
                         "calibration_simulated": {"unit": "base question, independent", "agent_questions": len(index),
                                                   **(simulated or {})}},
            "n_questions": len(index), "n_bank_questions": len(bank_index or index),
            "n_strata": len(set(index.strata)), "n_contrasts": len(contrasts),
            "contrasts": bounded(contrasts), "zero_spread": spread,
            "descriptive": {"C4 retry": bounded(c4_retry(V, K, frozen))}, "missing_shares": shares,
            "conclusions": conclusions}


def index_from_questions(questions: Mapping[str, dict], split: str) -> QuestionIndex:
    """The registered population (prereg §1): the split's base questions (their template x tier recorded as strata,
    for reporting; the resampling is unstratified, §4.2). Every question of a base question must agree on its stratum."""
    strata: dict[str, str] = {}
    for q in questions.values():
        if q.get("split") != split:
            continue
        base, label = q.get("base_question_id") or q["id"], f"{q['template']}|{q.get('tier', 'unrecorded')}"
        if strata.setdefault(base, label) != label:
            raise InputError(f"base question {base} has two strata ({strata[base]}, {label})")
    return QuestionIndex.from_strata(strata)
