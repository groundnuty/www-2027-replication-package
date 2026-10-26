"""The confirmatory multiplicity (design log, change-log entries 74 and 76): weighted Holm on calibrated p's.

- **Weights** (C1a, C1b, C2, C3, C4, C5) = (1, 1, 4, 8, 1, 1) / 16, the pooled C1a weighted as C1a (``holm_weights``
  in the config, as integers: the 1/16 cancels in every level and every ordering).
- **Levels.** Claim i can be tested at alpha w_i / s for every s = sum_{j in S} w_j over the subsets S of its family
  that hold it (``reachable_sums``). Every level and every threshold is computed by ``level``, alpha * w_i / s,
  multiply first, on the integer weights. A p* is a level, and the step-down's threshold at the same s is then the
  same float; divided first, 0.05 * (1 / 14) is one ulp below 0.05 * 1 / 14 and a p* on that level would fail.
- **p*** (``p_star``): the smallest of the claim's usable levels whose calibrated alpha' holds its test's p, else 1.
- **The step-down** (``step_down``): with S the claims not yet rejected, the claim with the smallest p*_i / w_i (the
  first in family order on a tie) is rejected if p*_i <= alpha w_i / sum_{j in S} w_j; it is removed and the step
  repeats, else the procedure stops. Claims are decided by this comparison only (an adjusted p, (sum_S w / w_i) p*
  against alpha, can differ from it by an ulp on a level, so none is reported).
- **The frozen inputs** (``load``), each by its sha256 in the config: step 1's alpha' at every reachable level, per
  family, claim and its step-1 test (``weighted-holm-alpha/1``, keyed by s); step 2's usable levels per claim (entry 76:
  a level whose size exceeds a + 2 SE is unusable for that claim) and the six-claim FWER check
  (``weighted-holm-usable/2``, naming the alpha' table's sha256): the dated step-2 addendum (entry 91). A claim with no
  test or no usable level, and every claim when the FWER check failed, has p* = 1. A confirmatory run refuses while
  either is [FILL], and refuses an addendum not produced by the frozen step-2 procedure (``check_addendum``): its
  recorded sizes bind nothing by themselves, so it is recomputed from the step-2 chunks where they are, and elsewhere
  accepted only when the freeze check's registry records that recomputation for it, on SiteA (``freeze_records``).
- **Their producers** (entries 74, 76, 91; scripts/weighted_holm_tables.py writes the files): ``step_one_table``
  computes the alpha' table from step 1's stored p-values (calibrate.py's chunks, ``configuration_p``) by step 1's grid
  rule at every reachable level, each claim's test by entry 46's rule at those levels, and the family's common factor
  from the step-1 FWER of this procedure (ruling 3); ``step_two`` verifies it on step 2's chunks, per level and then
  the six-claim FWER; ``addendum_document`` dates and binds the result with what it was computed from (the step-2
  outputs' digest, the fingerprint, N2, seed and seed scheme, the configurations, the procedure's digest), and
  ``check_addendum`` re-derives every decision from the recorded sizes, checks the recorded inputs against the frozen
  ones, and recomputes the whole result from the step-2 outputs when they are on this host, or else requires the freeze
  check's record of that recomputation (the review of 9d9c88c).
- **The family in the report** (``weighted_family``): run_contrasts' statistics as calibrated (run uncalibrated, its
  plain-Holm fields dropped), each claim's ``family`` block (weight, test, p, p*, the step and level it was rejected
  at, the claim), its intervals with the claim's test, the claim-level interval at the claim's first weighted level
  0.05 w_i / 16 with the calibrated alpha' there (ruling 1: weighted Bonferroni, levels summing to 0.05; where the claim
  has none there, the nominal level, its coverage "not certified" and C2's "at least 5 points" and C3's equivalence at
  5 points (entry 85, ``c3_equivalence``) not decided from it), and CR2 beside it under nominal weighted Holm,
  descriptive (ruling 2).

The reference implementation is results/weighted-holm-20261019/weighted_holm.py; tests/test_analysis_multiplicity.py
checks these functions against it bit for bit. This is new code: its own registry (inference.MULTIPLICITY_FUNCTIONS),
digest (``multiplicity_digest``) and known answer (``multiplicity_kat_sha256``), apart from the calibrated ones.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations, pairwise
from pathlib import Path
from typing import Any

import numpy as np

from harness.analysis import calibration as cal
from harness.analysis import prereg

REGISTERED_WEIGHTS = prereg.HOLM_WEIGHTS  # entry 74, x 16: the config refuses any other vector
PROCEDURE = "weighted Holm on calibrated p's (change-log entries 74 and 76)"
NOMINAL = ("nominal weighted Holm, the same weights on the raw CR2 p's without calibration: a descriptive cross-check, "
           "never a claim (entry 74, ruling 2)")
# /2: the usable-level set is the dated step-2 addendum (entry 91), with what it was computed from and every level's
# recorded size, so that its decisions can be re-derived and, where the outputs are, recomputed
ALPHA_SCHEMA, USABLE_SCHEMA = "weighted-holm-alpha/1", "weighted-holm-usable/2"
# the freeze check's registry once it records step 2's outputs (results/freeze-check/merge-order.md §4 on
# feat/freeze-check): "step2_outputs", the content of the step-2 chunks it recomputed on SiteA for an addendum
FREEZE_SCHEMA = "freeze-registry/2"
ADDENDUM = ("step 2's output (prereg §4.2, change-log entry 91): each claim's usable levels and the six-claim FWER check "
            "of the weighted Holm, bound by sha256 before any confirmatory inference runs")
RULES = {
    "procedure": "weighted-holm/1 (change-log entry 74): weights (1, 1, 4, 8, 1, 1)/16; with S the claims not yet "
                 "rejected, the smallest p*/w (family order on a tie) is rejected if p* <= alpha w / sum_S w, "
                 "multiply first",
    "levels": "levels/1: alpha' at every reachable level alpha w_i / s by step 1's grid rule, keyed by the integer s",
    "step2": "step2/3 (change-log entry 76): a level whose step-2 size exceeds a + 2 SE for a claim is unusable for it; "
             "p* over the usable levels, 1 with none; a failed six-claim FWER check gives every claim p = 1",
}
_PLAIN_HOLM = ("calibration", "p", "p_holm", "holm_level", "reject", "claim", "cross_check_cr2", "at_least_min_effect",
               "both_at_least_min_effect")  # run_contrasts' fields of the plain Holm, dropped from the report


def level(alpha: float, weight: int, total: int) -> float:
    """A level or threshold, alpha * w_i / s: multiplied first, the one expression every level and threshold uses."""
    return alpha * weight / total


def reachable_sums(claim: str, members: Sequence[str], weights: Mapping[str, int]) -> list[int]:
    """Every s = w_i + the weights of a subset of the family's other claims, ascending (its levels descending)."""
    others = [weights[c] for c in members if c != claim]
    return sorted({weights[claim] + sum(sub) for r in range(len(others) + 1) for sub in combinations(others, r)})


def reachable_levels(claim: str, members: Sequence[str], weights: Mapping[str, int], alpha: float) -> list[float]:
    """The claim's levels alpha w_i / s, largest first."""
    return [level(alpha, weights[claim], s) for s in reachable_sums(claim, members, weights)]


def p_star(x: float, weight: int, alphas: Mapping[int, float] | None, alpha: float) -> float:
    """The calibrated p of a raw p-value x (prereg.calibrated_p on the reachable levels): the smallest level alpha w / s,
    over the usable s in ``alphas`` (s -> alpha'), with x <= alpha'(s); 1 when none, or without alpha' (None). An alpha'
    that falls as the level rises is refused."""
    if not alphas:
        return 1.0
    sums = sorted(alphas)  # levels largest first
    if any(alphas[b] > alphas[a] for a, b in pairwise(sums)):
        raise prereg.PreregError(f"alpha' must be non-decreasing in the level, got {[alphas[s] for s in sums]} for s = "
                                 f"{sums}")
    out = 1.0
    for s in sums:  # each smaller level the p reaches replaces the larger one
        if x <= alphas[s]:
            out = level(alpha, weight, s)
    return out


def step_down(p: Sequence[float], weights: Sequence[int], alpha: float) -> list[dict[str, Any]]:
    """The weighted Bonferroni-Holm procedure on calibrated p's ``p`` (family order): per claim whether it is rejected,
    at which step (1 first) and at which level alpha w_i / sum_S w."""
    out: list[dict[str, Any]] = [{"rejected": False, "step": None, "level": None} for _ in p]
    left = list(range(len(p)))
    while left:
        j = min(left, key=lambda i: p[i] / weights[i])  # the first minimum: family order on a tie
        at = level(alpha, weights[j], sum(weights[i] for i in left))
        if not p[j] <= at:
            break
        out[j] = {"rejected": True, "step": len(p) - len(left) + 1, "level": at}
        left.remove(j)
    return out


@dataclass(frozen=True)
class WeightedCalibration:
    """The frozen inputs of one family: per claim its step-1 test and the alpha' of its usable levels by s (entry 76),
    the six-claim FWER check (step 2), the weights and alpha they were computed with, and where they were read."""

    family: str | None  # None: no table (a test run only)
    claims: Mapping[str, Mapping[str, Any]]  # claim -> {"test": str | None, "alpha_prime": {str(s): alpha'}}
    weights: Mapping[str, int]
    alpha: float
    fwer_pass: bool | None
    source: str
    step2: Mapping[str, Any] | None = None  # how the step-2 addendum was checked (check_addendum), when read by load
    step3: Mapping[str, Any] | None = None  # how the step-3 addendum was checked (check_step3), when read by load

    def final(self, claim: str) -> tuple[str | None, dict[int, float] | None]:
        """The claim's test and its usable levels' alpha' by s; (None, None), p* = 1, without a test or a usable level,
        when the FWER check failed, or without a table."""
        entry = self.claims.get(claim) or {}
        if self.fwer_pass is not True or entry.get("test") not in prereg.TESTS or not entry.get("alpha_prime"):
            return None, None
        return entry["test"], {int(s): float(a) for s, a in entry["alpha_prime"].items()}


NO_TABLE = WeightedCalibration(None, {}, REGISTERED_WEIGHTS, 0.05, None,
                               "none (every p* = 1): a test run's; a confirmatory run refuses without its frozen tables")


def _read(path: Path, sha256: str, schema: str, what: str) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != sha256:
        raise prereg.PreregError(f"{path}: sha256 {actual} is not the frozen {what}'s {sha256}")
    data = json.loads(raw)
    if not isinstance(data, dict) or data.get("schema") != schema:
        raise prereg.PreregError(f"{path}: not a {schema} file")
    if data.get("rules") != RULES:
        raise prereg.PreregError(f"{path}: computed under other rules than this code's (entries 74, 76)")
    return data


def _number(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool) and 0 <= v <= 1


def load_files(table_path: Path, table_sha256: str, usable_path: Path, usable_sha256: str, family: str, alpha: float,
               weights: Mapping[str, int]) -> WeightedCalibration:
    """One family's frozen inputs: the alpha' table and the usable-level set, each by its sha256 and refused under other
    rules, alpha or weights; the alpha' of every claim at exactly its reachable s, non-decreasing in the level, and its
    test the primary, the fallback or none; the usable set naming that table, its levels reachable ones."""
    table = _read(table_path, table_sha256, ALPHA_SCHEMA, "alpha' table")
    usable = _read(usable_path, usable_sha256, USABLE_SCHEMA, "usable-level set")
    for path, data in ((table_path, table), (usable_path, usable)):
        if data.get("alpha") != alpha:
            raise prereg.PreregError(f"{path}: alpha {data.get('alpha')!r}, not the config's {alpha!r}")
        if data.get("weights") != dict(weights):
            raise prereg.PreregError(f"{path}: weights {data.get('weights')!r}, not the config's {dict(weights)!r}")
    if usable.get("table_sha256") != table_sha256:
        raise prereg.PreregError(f"{usable_path}: the usable levels of another alpha' table "
                                 f"({usable.get('table_sha256')!r}, not {table_sha256})")
    members = prereg.FAMILY_CLAIMS[family]
    fam = (table.get("families") or {}).get(family) or {}
    ufam = (usable.get("families") or {}).get(family) or {}
    if set(fam.get("claims") or {}) != set(members) or set(ufam.get("usable") or {}) != set(members):
        raise prereg.PreregError(f"{table_path}, {usable_path}: no complete {family} family (claims {members})")
    if not isinstance(ufam.get("fwer_pass"), bool):
        raise prereg.PreregError(f"{usable_path}: the {family} family's six-claim FWER check is not recorded")
    claims = {}
    for c in members:
        entry, sums = fam["claims"][c], reachable_sums(c, members, weights)
        if entry.get("test") not in (*prereg.TESTS, None):
            raise prereg.PreregError(f"{table_path}: {c}'s test {entry.get('test')!r} is none of {prereg.TESTS} or none")
        ap = entry.get("alpha_prime") or {}
        if set(ap) != {str(s) for s in sums} or not all(_number(v) for v in ap.values()):
            raise prereg.PreregError(f"{table_path}: {c}'s alpha' must be numbers in [0, 1] at exactly its reachable s "
                                     f"{sums}, got {sorted(ap, key=int)}")
        alphas = {int(s): float(v) for s, v in ap.items()}
        if any(alphas[b] > alphas[a] for a, b in pairwise(sums)):
            raise prereg.PreregError(f"{table_path}: {c}'s alpha' must be non-decreasing in the level")
        used = ufam["usable"][c]
        if not isinstance(used, list) or len(set(used)) != len(used) or not set(used) <= set(sums) or any(
                isinstance(s, bool) or not isinstance(s, int) for s in used):
            raise prereg.PreregError(f"{usable_path}: {c}'s usable levels {used!r} are not reachable ones of {sums}")
        claims[c] = {"test": entry["test"], "alpha_prime": {str(s): alphas[s] for s in sorted(used)}}
    return WeightedCalibration(family, claims, dict(weights), alpha, ufam["fwer_pass"],
                               f"{table_path} (sha256 {table_sha256}), {usable_path} (sha256 {usable_sha256}), {family}")


def load_step2(cfg: prereg.PreregConfig, root: Path) -> WeightedCalibration:
    """The config's frozen inputs up to step 2 for its recorded C1a family (``load_files``), with how its addendum was
    checked (``step2_report``). A confirmatory run reads ``load``, which adds step 3."""
    report = step2_report(cfg, root)
    weighted = load_files(Path(root) / cfg.multiplicity_table, cfg.multiplicity_table_sha256,
                          Path(root) / cfg.step2_addendum, cfg.step2_addendum_sha256, cfg.require_c1a_choice(),
                          cfg.alpha, cfg.holm_weights)
    return dataclasses.replace(weighted, step2=report)


def step2_report(cfg: prereg.PreregConfig, root: Path) -> dict[str, Any]:
    """Step 2's frozen inputs checked, for either family: refused while any is [FILL] (entry 74, ruling 3), and the
    step-2 addendum refused unless the frozen step-2 procedure produced it (entry 91, ``check_addendum``: N2, step 2's
    seed and seed scheme, the table's fingerprint and configurations, and the frozen multiplicity digest; the result
    recomputed from the step-2 outputs when they are on this host, else the freeze registry's record of that
    recomputation required). Step 2's procedure is the multiplicity code it ran as (the config's
    step2_procedure_digest, prereg.STEP2_PROCEDURE_DIGEST), for its addendum and its registry record: the code has grown
    step 3 since (the review of 1e31f1f)."""
    values = (cfg.multiplicity_table, cfg.multiplicity_table_sha256, cfg.step2_addendum, cfg.step2_addendum_sha256,
              cfg.step2_procedure_digest)
    if any(v in (prereg.PLACEHOLDER, "", None) for v in values):
        raise prereg.PreregError("the weighted Holm's alpha' table and step 2's addendum are not frozen "
                                 "(multiplicity_table, step2_addendum and their sha256 [FILL]): a confirmatory run "
                                 "refuses without them (entry 74, ruling 3; entry 91: step 2's output is bound by a "
                                 "dated addendum before any confirmatory inference runs)")
    table_path, addendum_path = Path(root) / cfg.multiplicity_table, Path(root) / cfg.step2_addendum
    report = check_addendum(_read(addendum_path, cfg.step2_addendum_sha256, USABLE_SCHEMA, "step-2 addendum"),
                            _read(table_path, cfg.multiplicity_table_sha256, ALPHA_SCHEMA, "alpha' table"),
                            n=cal.N2, seed=cal.SEEDS[2], seed_scheme=cal.SEED_SCHEME,
                            procedure=cfg.step2_procedure_digest, alpha=cfg.alpha, weights=cfg.holm_weights,
                            verified=freeze_records(Path(root) / cfg.freeze_registry),
                            addendum_sha256=cfg.step2_addendum_sha256)
    return {"addendum": f"{addendum_path} (sha256 {cfg.step2_addendum_sha256})", **report}


def load(cfg: prereg.PreregConfig, root: Path) -> WeightedCalibration:
    """The config's frozen inputs for its recorded C1a family, as a confirmatory run reads them: ``load_final``'s
    calibration of that family (step 2 and step 3 verified)."""
    finals, family = load_final(cfg, root), cfg.require_c1a_choice()
    if family not in finals:  # the table holds no such family: load_files' refusal
        load_files(Path(root) / cfg.multiplicity_table, cfg.multiplicity_table_sha256, Path(root) / cfg.step2_addendum,
                   cfg.step2_addendum_sha256, family, cfg.alpha, cfg.holm_weights)
    return finals[family]


def load_final(cfg: prereg.PreregConfig, root: Path) -> dict[str, WeightedCalibration]:
    """The candidate families' final calibrations (both, from the registered alpha' table), every input verified once (entry 106), for a confirmatory run
    (``load``) and for the registered powers that make the C1a choice (scripts/power_weighted.py; the review of
    1e31f1f: no power or choice on the one-sided calibration):
    - step 2 as ``load_step2`` checks it (its addendum by its own frozen procedure, prereg.STEP2_PROCEDURE_DIGEST);
    - step 3's bound configurations (``read_step3_configurations``) and fingerprint (the config's step3_fingerprint);
    - the selection re-derived from its inputs (``check_selection``: from its chunks and steps 1 and 2's, or the freeze
      check's record of that recomputation);
    - the addendum the frozen step-3 procedure's output on those configurations (``check_step3``);
    - per family, C2 two-sided at its validated levels, the repaired levels, every other level as step 2 left it
      (``load_step3``)."""
    step2 = step2_report(cfg, root)
    step3_values = (cfg.step3_configurations, cfg.step3_configurations_sha256, cfg.step3_fingerprint,
                    cfg.step3_selection, cfg.step3_selection_sha256, cfg.step3_addendum, cfg.step3_addendum_sha256)
    if any(v in (prereg.PLACEHOLDER, "", None) for v in step3_values):
        raise prereg.PreregError("step 3's configurations, fingerprint, selection and addendum are not frozen "
                                 "(step3_configurations, step3_fingerprint, step3_selection, step3_addendum and their "
                                 "sha256 [FILL]): C2's decision and the repaired levels' wait for them (entry 106)")
    table_path, addendum_path = Path(root) / cfg.multiplicity_table, Path(root) / cfg.step2_addendum
    sel_path, step3_path = Path(root) / cfg.step3_selection, Path(root) / cfg.step3_addendum
    configurations = read_step3_configurations(Path(root) / cfg.step3_configurations, cfg.step3_configurations_sha256)
    selection = _read3(sel_path, cfg.step3_selection_sha256, STEP3_ALPHA_SCHEMA, "step-3 selection")
    step3 = _read3(step3_path, cfg.step3_addendum_sha256, STEP3_SCHEMA, "step-3 addendum")
    table = _read(table_path, cfg.multiplicity_table_sha256, ALPHA_SCHEMA, "alpha' table")
    usable = _read(addendum_path, cfg.step2_addendum_sha256, USABLE_SCHEMA, "step-2 addendum")
    registry = Path(root) / cfg.freeze_registry
    selected = check_selection(selection, table, usable, configurations=configurations,
                               fingerprint=cfg.step3_fingerprint, table_sha256=cfg.multiplicity_table_sha256,
                               usable_sha256=cfg.step2_addendum_sha256, n=cal.N3_SELECTION,
                               seed=prereg.STEP3_SEEDS["selection"], seed_scheme=cal.SEED_SCHEME,
                               procedure=cfg.multiplicity_digest, inference=cfg.calibrated_inference_digest,
                               alpha=cfg.alpha, weights=cfg.holm_weights,
                               steps12_outputs={"step1": cfg.step1_outputs, "step2": usable["inputs"]["outputs"]},
                               verified=freeze_records(registry, "step3_selection_outputs"),
                               selection_sha256=cfg.step3_selection_sha256)
    report3 = check_step3(step3, table, usable, selection, configurations=configurations,
                          fingerprint=cfg.step3_fingerprint, n=cal.N3, seed=prereg.STEP3_SEEDS["validation"],
                          seed_scheme=cal.SEED_SCHEME, procedure=cfg.multiplicity_digest,
                          inference=cfg.calibrated_inference_digest, alpha=cfg.alpha, weights=cfg.holm_weights,
                          verified=freeze_records(registry, "step3_outputs"), step3_sha256=cfg.step3_addendum_sha256,
                          selection_sha256=cfg.step3_selection_sha256)
    record3 = {"addendum": f"{step3_path} (sha256 {cfg.step3_addendum_sha256})",
               "selection": f"{sel_path} (sha256 {cfg.step3_selection_sha256})", "selection_checked": selected,
               **report3}
    return {fam: dataclasses.replace(load_step3(table_path, cfg.multiplicity_table_sha256, addendum_path,
                                                cfg.step2_addendum_sha256, step3, fam, cfg.alpha, cfg.holm_weights),
                                     step2=step2, step3=record3)
            for fam in prereg.FAMILY_CLAIMS if fam in table["families"]}


STEP3_CONFIGURATIONS_SCHEMA = "step3-configurations/1"


def step3_configurations_document(configurations: Sequence[Mapping[str, Any]], structure_sha256: str,
                                  n: int) -> dict[str, Any]:
    """The bound configurations file (STEP3_CONFIGURATIONS_SCHEMA): step 3's configurations in order, each its name and
    its null claims with C2 two-sided (calibration.step3_configuration_list), with the structure they were computed on
    (by sha256: steps 1 and 2's structure.json) and the analysis's question count."""
    return {"schema": STEP3_CONFIGURATIONS_SCHEMA, "structure_sha256": structure_sha256, "n": n,
            "producer": "harness.analysis.calibration.step3_configuration_list",
            "configurations": [{"name": c["name"], "null": list(c["null"])} for c in configurations]}


def read_step3_configurations(path: Path, sha256: str) -> list[dict[str, Any]]:
    """Step 3's configurations and their two-sided null claims, as bound by sha256 in the config (step3_configurations:
    step3_configurations_document); refused under another sha256, schema or shape, or empty."""
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise prereg.PreregError(f"{path}: sha256 {hashlib.sha256(raw).hexdigest()} is not the bound step-3 "
                                 f"configurations' {sha256}")
    data = json.loads(raw)
    confs = data.get("configurations") if isinstance(data, dict) else None
    if not isinstance(data, dict) or data.get("schema") != STEP3_CONFIGURATIONS_SCHEMA or not isinstance(confs, list) \
            or not confs or not all(isinstance(c, dict) and set(c) == {"name", "null"} and isinstance(c["name"], str)
                                    and isinstance(c["null"], list) and set(c["null"]) <= set(cal.CLAIMS)
                                    for c in confs) \
            or len({c["name"] for c in confs}) != len(confs):
        raise prereg.PreregError(f"{path}: not a {STEP3_CONFIGURATIONS_SCHEMA} file of uniquely named configurations")
    return confs


def _read3(path: Path, sha256: str, schema: str, what: str) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != sha256:
        raise prereg.PreregError(f"{path}: sha256 {actual} is not the frozen {what}'s {sha256}")
    data = json.loads(raw)
    if not isinstance(data, dict) or data.get("schema") != schema or data.get("rules") != STEP3_RULES:
        raise prereg.PreregError(f"{path}: not a {schema} file under step 3's rules")
    return data


def table_document(families: Mapping[str, Mapping[str, Mapping[str, Any]]], alpha: float,
                   weights: Mapping[str, int], inputs: Mapping[str, Any] | None = None,
                   step1: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """A weighted-holm-alpha/1 document: family -> claim -> {"test", "alpha_prime": {s: alpha'}}; with what it was
    computed from (``inputs``: step 1's fingerprint, N1, seed, configurations and their null claims, outputs) and each
    family's step-1 record (``step1``: the factor, the step-1 FWER, the size checks) when the producer wrote it."""
    doc = {"schema": ALPHA_SCHEMA, "rules": RULES, "alpha": alpha, "weights": dict(weights),
           "families": {fam: {"claims": {c: {"test": e["test"],
                                             "alpha_prime": {str(s): float(a) for s, a in sorted(e["alpha_prime"].items())}}
                                         for c, e in claims.items()}, **dict((step1 or {}).get(fam) or {})}
                        for fam, claims in families.items()}}
    return doc if inputs is None else {**doc, "inputs": dict(inputs)}


def usable_document(table_sha256: str, families: Mapping[str, Mapping[str, Any]], alpha: float,
                    weights: Mapping[str, int]) -> dict[str, Any]:
    """A weighted-holm-usable/1 document: family -> {"fwer_pass", "usable": claim -> [s, ...]}, naming its table."""
    return {"schema": USABLE_SCHEMA, "rules": RULES, "table_sha256": table_sha256, "alpha": alpha,
            "weights": dict(weights),
            "families": {fam: {"fwer_pass": bool(f["fwer_pass"]),
                               "usable": {c: sorted(int(s) for s in used) for c, used in f["usable"].items()}}
                         for fam, f in families.items()}}


def document_bytes(doc: Mapping[str, Any]) -> bytes:
    return (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode("utf-8")


def sums_from_levels(claim: str, members: Sequence[str], weights: Mapping[str, int], alpha: float,
                     by_level: Mapping[str, float]) -> dict[int, float]:
    """A table keyed by levels (the reference's repr keys) keyed by s instead: each key must read back as alpha w / s
    for exactly one reachable s, bit for bit; refused otherwise. (Writing the frozen tables, not read by the analysis.)"""
    at = {level(alpha, weights[claim], s): s for s in reachable_sums(claim, members, weights)}
    out = {}
    for key, value in by_level.items():
        if float(key) not in at:
            raise prereg.PreregError(f"{claim}: {key} is not a reachable level alpha w / s of it")
        out[at[float(key)]] = value
    return out


# --- the producers of the frozen inputs (entries 74, 76, 91) --------------------------------------------------------


def configuration_p(directory: Path, step: int, index: int, name: str, fingerprint: str, n: int, seed: int,
                    seed_scheme: str, phase: str | None = None) -> np.ndarray:
    """One configuration's stored p-values for one calibration step (n datasets x calibration.CLAIMS x prereg.TESTS),
    from calibrate.py's chunks ``<directory>/step<step>/c<index>-<chunk>.npz`` (step 3: ``step3-<phase>``, the phase
    recorded in each chunk), with calibrate._load's refusals: a chunk of another fingerprint, configuration, step,
    phase, seed or seed scheme, and chunks that do not cover datasets [0, n) exactly, without gaps or overlaps."""
    chunks = []
    sub = f"step{step}" if phase is None else f"step{step}-{phase}"
    for f in sorted((Path(directory) / sub).glob(f"c{index:02d}-*.npz")):
        with np.load(f) as z:
            fields = set(z.files)
            if not {"p", "lo", "hi", "digest", "configuration", "config_index", "step", "seed", "seed_scheme"} <= fields:
                raise prereg.PreregError(f"{f}: not a calibration chunk recording its step, seed and seed scheme")
            if phase is not None and ("phase" not in fields or str(z["phase"]) != phase):
                raise prereg.PreregError(f"{f}: not a chunk of step 3's {phase} run")
            if str(z["digest"]) != fingerprint:
                raise prereg.PreregError(f"{f}: produced under fingerprint {z['digest']}, not {fingerprint}")
            if str(z["configuration"]) != name or int(z["config_index"]) != index:
                raise prereg.PreregError(f"{f}: configuration {z['configuration']!s}, not {name}")
            if int(z["step"]) != step or int(z["seed"]) != seed or str(z["seed_scheme"]) != seed_scheme:
                raise prereg.PreregError(f"{f}: step {int(z['step'])}, seed {int(z['seed'])}, scheme "
                                         f"{z['seed_scheme']!s}; not step {step}'s {seed}, {seed_scheme}")
            lo, hi, p = int(z["lo"]), int(z["hi"]), np.asarray(z["p"], dtype=np.float64)
        if p.shape != (hi - lo, len(cal.CLAIMS), len(prereg.TESTS)):
            raise prereg.PreregError(f"{f}: p-values of shape {p.shape} for datasets {lo}..{hi}")
        chunks.append((lo, hi, p))
    chunks.sort(key=lambda c: c[0])
    edge = 0
    for lo, hi, _ in chunks:
        if lo != edge:
            raise prereg.PreregError(f"step {step} configuration {index}: datasets {edge}..{lo} missing or overlapping")
        edge = hi
    if edge != n:
        raise prereg.PreregError(f"step {step} configuration {index}: datasets cover 0..{edge}, not 0..{n}")
    return np.concatenate([c[2] for c in chunks])


def outputs_digest(directory: Path, step: int | str) -> dict[str, Any]:
    """A calibration step's stored outputs as a table or an addendum names them: the number of chunk files and one
    sha256 over their contents, file by file in name order and array by array in name order (each array's file name,
    name, dtype and shape as compact JSON, then its bytes). The arrays, not the files: an npz's zip entries carry the
    time they were written."""
    h, count = hashlib.sha256(), 0
    for f in sorted((Path(directory) / f"step{step}").glob("c*-*.npz")):
        with np.load(f) as z:
            for key in sorted(z.files):
                a = np.ascontiguousarray(z[key])
                h.update(json.dumps([f.name, key, str(a.dtype), list(a.shape)], separators=(",", ":")).encode())
                h.update(a.tobytes())
        count += 1
    return {"chunks": count, "sha256": h.hexdigest()}


CP_CONFIDENCE = 0.95  # step 3's margin (entry 106): the one-sided 95% Clopper-Pearson upper bound on the size


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p), 0 < p < 1, summed in logs from P(X = 0) up (no underflow at n = 120,000)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    log_pmf, log_cdf = n * math.log1p(-p), -math.inf
    step = math.log(p) - math.log1p(-p)
    for j in range(k + 1):
        log_cdf = np.logaddexp(log_cdf, log_pmf)
        log_pmf += math.log(n - j) - math.log(j + 1) + step
    return float(min(1.0, math.exp(log_cdf)))


def cp_max_count(n: int, a: float, confidence: float = CP_CONFIDENCE) -> int:
    """The largest count k of n whose one-sided Clopper-Pearson upper bound is <= a, -1 when none: the bound u(k)
    solves P(X <= k | n, u) = 1 - confidence and P(X <= k | n, p) falls as p rises, so u(k) <= a iff
    P(X <= k | n, a) <= 1 - confidence; that CDF rises with k, so the counts that qualify are 0..k_max."""
    tail = 1 - confidence
    j, log_pmf, log_cdf = 0, n * math.log1p(-a), -math.inf
    step = math.log(a) - math.log1p(-a)
    while True:  # P(X <= n) = 1 > tail, so this returns by j = n at the latest
        log_cdf = np.logaddexp(log_cdf, log_pmf)
        if math.exp(log_cdf) > tail:
            return j - 1
        log_pmf += math.log(n - j) - math.log(j + 1) + step
        j += 1


def cp_grid_alpha(p: np.ndarray, a: float, steps: int) -> float:
    """Step 3's own value at a level (entry 106): the largest grid value j x 0.0005 a (j <= steps) whose count of p's
    at or below it has a one-sided 95% Clopper-Pearson upper bound <= a (cp_max_count); 0 when none."""
    k_max = cp_max_count(len(p), a)
    if k_max < 0:
        return 0.0
    grid = np.arange(1, steps + 1) * 0.0005 * a
    counts = np.searchsorted(np.sort(p), grid, side="right")
    ok = np.nonzero(counts <= k_max)[0]
    return float(grid[ok[-1]]) if len(ok) else 0.0


def alpha_primes(null_p: Sequence[np.ndarray], claim: str, members: Sequence[str], weights: Mapping[str, int],
                 alpha: float) -> dict[int, float]:
    """Step 1's grid rule (calibration.step_one) at the claim's reachable levels a = alpha w / s, by s: each level's own
    value is the smallest, over the claim's null configurations (``null_p``: one test's p-values in each), of the
    largest j x 0.0005 a (j <= GRID_STEPS) whose size is <= a (a itself with none); then, from the largest level down,
    the largest value on the level's own grid at most min(its own value, the next larger level's alpha')."""
    out, upper = {}, math.inf
    for s in reachable_sums(claim, members, weights):  # s ascending: the largest level first
        a = level(alpha, weights[claim], s)
        own = min([cal._grid_alpha(p, a, cal.GRID_STEPS) for p in null_p] or [a])
        upper = cal._snap(min(own, upper), a, cal.GRID_STEPS)
        out[s] = upper
    return out


def cp_alpha_primes(null_p: Sequence[np.ndarray], claim: str, members: Sequence[str], weights: Mapping[str, int],
                    alpha: float) -> dict[int, float]:
    """Step 3's selection (entry 106) at every reachable level of the claim: each level's own value is the smallest,
    over the claim's null configurations (``null_p``: one test's p-values in each), of cp_grid_alpha (a itself with
    none); then, as step 1's rule, from the largest level down, the largest value on the level's grid at most min(its
    own value, the next larger level's alpha')."""
    out, upper = {}, math.inf
    for s in reachable_sums(claim, members, weights):  # s ascending: the largest level first
        a = level(alpha, weights[claim], s)
        own = min([cp_grid_alpha(p, a, cal.GRID_STEPS) for p in null_p] or [a])
        upper = cal._snap(min(own, upper), a, cal.GRID_STEPS)
        out[s] = upper
    return out


def cp_repair(null_p: Sequence[np.ndarray], claim: str, members: Sequence[str], weights: Mapping[str, int],
              alpha: float, kept: Mapping[int, float], repair: Sequence[int]) -> dict[int, dict[str, Any]]:
    """Step 3's repair of levels that failed step 2 (entry 106: the passing levels keep their alpha', ``kept``): each
    repaired s gets as its own value the smallest over the null configurations of cp_grid_alpha (a itself with none),
    then repair_records chains them."""
    own = {}
    for s in sorted(repair):
        a = level(alpha, weights[claim], s)
        own[s] = min([cp_grid_alpha(p, a, cal.GRID_STEPS) for p in null_p] or [a])
    return repair_records(own, claim, weights, alpha, kept)


def repair_records(own: Mapping[int, float], claim: str, weights: Mapping[str, int], alpha: float,
                   kept: Mapping[int, float]) -> dict[int, dict[str, Any]]:
    """The repaired levels' records from their own values (``own``: s -> value), largest level first: each at most the
    next larger level's alpha' (kept or repaired) and snapped to its own grid; it stays unusable when that falls below
    the next smaller kept level's alpha', since alpha' must not fall as the level rises and a passing level is not
    changed. Also what the selection's check re-derives each record from (check_selection)."""
    out: dict[int, dict[str, Any]] = {}
    chosen = {int(s): float(v) for s, v in kept.items()}
    for s in sorted(own):
        a = level(alpha, weights[claim], s)
        larger = [t for t in chosen if t < s]  # smaller s: larger levels
        value = cal._snap(min(own[s], chosen[max(larger)] if larger else math.inf), a, cal.GRID_STEPS)
        smaller = [int(t) for t in kept if int(t) > s]
        floor = float(chosen[min(smaller)]) if smaller else 0.0
        usable = value >= floor
        out[s] = {"own": own[s], "alpha_prime": value, "next_smaller_level_alpha_prime": floor if smaller else None,
                  "usable": usable}
        if usable:
            chosen[s] = value
    return out


def sizes_within(null_p: Sequence[np.ndarray], weight: int, alphas: Mapping[int, float], alpha: float) -> bool:
    """Entry 46's step-1 rule at the reachable levels: the size at alpha'(s) is <= alpha w / s at every level, in every
    null configuration."""
    return all(float(np.mean(p <= alphas[s])) <= level(alpha, weight, s) for p in null_p for s in alphas)


def p_star_array(x: np.ndarray, weight: int, alphas: Mapping[int, float] | None, alpha: float) -> np.ndarray:
    """``p_star`` on many raw p's at once (the same levels, the same comparisons): 1 without alpha'."""
    out = np.ones(len(x))
    if not alphas:
        return out
    sums = sorted(alphas)  # levels largest first: each smaller level the p reaches replaces the larger one
    if any(alphas[b] > alphas[a] for a, b in pairwise(sums)):
        raise prereg.PreregError(f"alpha' must be non-decreasing in the level, got {[alphas[s] for s in sums]}")
    for s in sums:
        out = np.where(x <= alphas[s], level(alpha, weight, s), out)
    return out


def step_down_rejections(P: np.ndarray, weights: Sequence[int], alpha: float) -> np.ndarray:
    """``step_down`` on every row of P (datasets x claims, family order) at once: the rejections. Each step takes the
    first smallest p*/w among the claims left and rejects it if p* <= alpha w / sum_S w (``level``, multiply first)."""
    n, m = P.shape
    w = np.asarray(weights, dtype=np.float64)
    rejected = np.zeros((n, m), dtype=bool)
    active = np.ones(n, dtype=bool)
    rows = np.arange(n)
    for _ in range(m):
        j = np.argmin(np.where(rejected, np.inf, P / w), axis=1)  # the first minimum: family order on a tie
        at = level(alpha, w[j], np.where(rejected, 0.0, w).sum(axis=1))
        ok = active & (P[rows, j] <= at)
        rejected[rows[ok], j[ok]] = True
        active = ok
    return rejected


def calibrated_matrix(p: np.ndarray, members: Sequence[str], tests: Mapping[str, str | None],
                      alphas: Mapping[str, Mapping[int, float]], weights: Mapping[str, int], alpha: float) -> np.ndarray:
    """(datasets x members) p*: each claim's test's p (calibration.CLAIMS x prereg.TESTS in ``p``) through its alpha';
    1 for a claim without a test."""
    return np.column_stack([np.ones(len(p)) if tests.get(c) is None else p_star_array(
        p[:, cal.CLAIMS.index(c), prereg.TESTS.index(tests[c])], weights[c], alphas[c], alpha) for c in members])


def family_fwer(p: np.ndarray, null: Mapping[str, bool], members: Sequence[str], tests: Mapping[str, str | None],
                alphas: Mapping[str, Mapping[int, float]], weights: Mapping[str, int], alpha: float) -> float:
    """The six-claim FWER of the weighted procedure in one configuration: the share of its datasets where it rejects a
    claim null there (0 with none null)."""
    cols = [j for j, c in enumerate(members) if null[c]]
    if not cols:
        return 0.0
    R = step_down_rejections(calibrated_matrix(p, members, tests, alphas, weights, alpha),
                             [weights[c] for c in members], alpha)
    return float(np.mean(np.any(R[:, cols], axis=1)))


def joint_factor(p_by_conf: Sequence[np.ndarray], nulls: Sequence[Mapping[str, bool]], members: Sequence[str],
                 tests: Mapping[str, str | None], alphas: Mapping[str, Mapping[int, float]], weights: Mapping[str, int],
                 alpha: float) -> dict[str, Any]:
    """Ruling 3 for the weighted procedure: the largest k in 1..FACTOR_STEPS for which, with every alpha' multiplied by
    k / FACTOR_STEPS, the step-1 FWER is <= alpha in every configuration (bisection: the FWER does not fall as k rises);
    None when no k reaches it (the family fails: every alpha' 0)."""
    steps = cal.FACTOR_STEPS

    def fwer(k: int) -> list[float]:
        scaled = {c: {s: v * (k / steps) for s, v in a.items()} for c, a in alphas.items()}
        return [family_fwer(p, nl, members, tests, scaled, weights, alpha) for p, nl in zip(p_by_conf, nulls)]

    unscaled = fwer(steps)
    if max(unscaled, default=0.0) <= alpha:
        k = steps
    elif max(fwer(1), default=0.0) > alpha:
        k = None
    else:
        lo, hi = 1, steps
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if max(fwer(mid), default=0.0) <= alpha:
                lo = mid
            else:
                hi = mid
        k = lo
    return {"k": k, "factor": None if k is None else k / steps, "fwer_unscaled": unscaled}


def step_one_table(p_by_conf: Sequence[np.ndarray], nulls: Sequence[Mapping[str, bool]], weights: Mapping[str, int],
                   alpha: float) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Step 1 of the weighted procedure (entry 74), per family, from step 1's stored p-values (one array per
    configuration, n x calibration.CLAIMS x prereg.TESTS) and each configuration's null claims: both tests' alpha' at
    every reachable level (``alpha_primes``); each claim's test by entry 46's rule at those levels (the primary, else
    the fallback, else none); the common factor from the step-1 FWER with those tests (``joint_factor``). Returns the
    table's families (claim -> its test and its alpha' times the factor; 0 at every level without a test or when the
    family fails) and each family's step-1 record (``table_document``'s ``step1``)."""
    families, record = {}, {}
    for family, members in prereg.FAMILY_CLAIMS.items():
        alphas, ok = {}, {}
        for c in members:
            ci = cal.CLAIMS.index(c)
            null_p = {t: [p[:, ci, ti] for p, nl in zip(p_by_conf, nulls) if nl[c]] for ti, t in enumerate(prereg.TESTS)}
            alphas[c] = {t: alpha_primes(null_p[t], c, members, weights, alpha) for t in prereg.TESTS}
            ok[c] = {t: sizes_within(null_p[t], weights[c], alphas[c][t], alpha) for t in prereg.TESTS}
        tests = {c: next((t for t in prereg.TESTS if ok[c][t]), None) for c in members}
        joint = joint_factor(p_by_conf, nulls, members, tests,
                             {c: alphas[c][t] for c, t in tests.items() if t is not None}, weights, alpha)
        factor = joint["factor"] or 0.0
        families[family] = {c: {"test": tests[c], "alpha_prime": {
            s: (v * factor if tests[c] is not None else 0.0) for s, v in alphas[c][tests[c] or prereg.TESTS[0]].items()}}
            for c in members}
        record[family] = {"factor": joint["factor"], "k": joint["k"], "fwer_unscaled": joint["fwer_unscaled"],
                          "fwer_max": max(joint["fwer_unscaled"], default=0.0), "step1_size_ok": ok}
    return families, record


def _se(a: float, n: int) -> float:
    return math.sqrt(a * (1 - a) / n)


def _table_levels(table: Mapping[str, Any], family: str) -> tuple[dict[str, str | None], dict[str, dict[int, float]]]:
    claims = table["families"][family]["claims"]
    tests = {c: claims[c]["test"] for c in prereg.FAMILY_CLAIMS[family]}
    return tests, {c: {int(s): float(a) for s, a in claims[c]["alpha_prime"].items()} for c in tests if tests[c]}


def step_two(p_of: Callable[[int], np.ndarray], configurations: Sequence[Mapping[str, Any]], table: Mapping[str, Any],
             weights: Mapping[str, int], alpha: float, n: int) -> dict[str, dict[str, Any]]:
    """Step 2 of the weighted procedure (entry 76) on step 2's p-values (``p_of(i)``: configuration i's n datasets x
    calibration.CLAIMS x prereg.TESTS; ``configurations``: each one's name and null claims), at the table's alpha', one
    configuration at a time in two passes:
    - per claim with a test and per reachable level: the largest size at alpha'(s) over the claim's null configurations
      (the first configuration on a tie) against a + 2 SE, SE = sqrt(a (1 - a) / n); the level is usable iff that size
      is within it (a level with no null configuration is usable);
    - the six-claim FWER with p* over the usable levels (a claim with none, or no test, at p = 1) against
      alpha + 2 SE(alpha) in every configuration; ``fwer_pass`` iff every one is within it.
    Per family: "usable" (claim -> its usable s), "levels" (claim -> s -> the record), "fwer" (one row per
    configuration), "fwer_pass"."""
    worst: dict[str, dict[str, dict[int, tuple[float, str] | None]]] = {}
    for family in table["families"]:
        tests, alphas = _table_levels(table, family)
        worst[family] = {c: dict.fromkeys(alphas[c]) for c in alphas}
    for i, conf in enumerate(configurations):
        p = p_of(i)
        if len(p) != n:
            raise prereg.PreregError(f"configuration {conf['name']}: {len(p)} datasets, not {n}")
        for family, by_claim in worst.items():
            tests, alphas = _table_levels(table, family)
            for c, levels in by_claim.items():
                if c not in conf["null"]:
                    continue
                x = p[:, cal.CLAIMS.index(c), prereg.TESTS.index(tests[c])]
                for s in levels:
                    size = float(np.mean(x <= alphas[c][s]))
                    if levels[s] is None or size > levels[s][0]:
                        levels[s] = (size, conf["name"])
    out = {}
    for family, by_claim in worst.items():
        tests, alphas = _table_levels(table, family)
        records: dict[str, dict[str, Any]] = {c: {} for c in tests}
        for c, levels in by_claim.items():
            for s, w in levels.items():
                a = level(alpha, weights[c], s)
                limit = a + 2 * _se(a, n)
                records[c][str(s)] = {"alpha_prime": alphas[c][s], "size": None if w is None else w[0],
                                      "configuration": None if w is None else w[1], "se": _se(a, n), "limit": limit,
                                      "usable": w is None or w[0] <= limit}
        out[family] = {"usable": {c: sorted(int(s) for s, r in records[c].items() if r["usable"]) for c in tests},
                       "levels": records, "fwer": []}
    for i, conf in enumerate(configurations):
        p = p_of(i)
        for family, fam in out.items():
            tests, alphas = _table_levels(table, family)
            members = prereg.FAMILY_CLAIMS[family]
            usable = {c: {s: alphas[c][s] for s in fam["usable"][c]} for c in alphas}
            entered = {c: (tests[c] if usable.get(c) else None) for c in members}
            rate = family_fwer(p, {c: c in conf["null"] for c in members}, members, entered, usable, weights, alpha)
            limit = alpha + 2 * _se(alpha, n)
            fam["fwer"].append({"configuration": conf["name"], "fwer": rate, "limit": limit, "pass": rate <= limit,
                                "null_claims": [c for c in members if c in conf["null"]]})
    for fam in out.values():
        fam["fwer_pass"] = all(r["pass"] for r in fam["fwer"])
    return out


# --- step 3 (prereg §4.2, change-log entry 106) ------------------------------------------------------------------------

STEP3_ALPHA_SCHEMA, STEP3_SCHEMA = "weighted-holm-step3-alpha/1", "weighted-holm-step3/1"
STEP3_RULES = {
    "scope": "step3-scope/1 (entry 106): every level of C2 (two-sided, a new statistic: nothing carries over from the "
             "one-sided calibration) and the levels that failed step 2 (each claim's reachable levels outside its "
             "step-2 usable set: C1a two-family s = 6, C5 s = 2 and 3); every other level keeps its alpha' and its step-2 "
             "verdict",
    "selection": "selection/2: alpha'(a) the largest grid value j x 0.0005 a (j <= 4000) whose one-sided 95% "
                 "Clopper-Pearson upper bound on the size is <= a in every null configuration of the claim, then step "
                 "1's monotone snap from the largest level down; a repaired level at most the next larger level's alpha' "
                 "and unusable below the next smaller kept level's; C2's test by entry 46's rule on the selection run, "
                 "a repaired level keeps its claim's step-1 test; C2 on the selection run's draws, a repaired claim in "
                 "each null configuration on every draw that applies, pooled (steps 1 and 2's and the selection run's)",
    "validation": "validation/1: on fresh draws (the validation seed, N3 per configuration) step 2's rule per selected "
                  "level (size <= a + 2 SE, else unusable), then the six-claim FWER <= alpha + 2 SE in every "
                  "configuration with the final usable levels; a failed FWER check gives every claim p = 1",
}


def step3_scope(table: Mapping[str, Any], usable: Mapping[str, Any], family: str) -> dict[str, list[int]]:
    """The levels step 3 selects in one family: every reachable s of C2, and for every other claim with a test its
    reachable s outside its step-2 usable set."""
    members, weights = prereg.FAMILY_CLAIMS[family], table["weights"]
    out = {"C2": reachable_sums("C2", members, weights)}
    for c in members:
        if c == "C2" or table["families"][family]["claims"][c]["test"] is None:
            continue
        failed = sorted(set(reachable_sums(c, members, weights)) - set(usable["families"][family]["usable"][c]))
        if failed:
            out[c] = failed
    return out


def step3_selection_arrays(table: Mapping[str, Any], usable: Mapping[str, Any], configurations: Sequence[Mapping[str, Any]],
                           selection_dir: Path, steps12_dir: Path, fingerprint: str, n: int, seed: int,
                           seed_scheme: str, steps12_n: tuple[int, int] | None = None) -> tuple[dict, dict, dict]:
    """The selection's inputs, as the producer and the check read them: C2's p-values per test in each of its two-sided
    null configurations from the selection run (``selection_dir``: calibrate.py's step-3 OUT, its step3-selection
    chunks under ``fingerprint``, n datasets, the selection seed); each repaired claim's in each of its null
    configurations, every draw that applies pooled (its statistic is unchanged since 87c7126): steps 1 and 2's chunks
    (``steps12_dir``, the table's fingerprint) where the configuration is one of theirs (an index below their count,
    which must carry their name), with the selection run's where it ran there (C2 null too; the review of 3b149eb),
    else the selection run's alone. Refused: a repaired claim null in a configuration of neither, and another name at
    one of their indices. Also
    returned: per family and repaired claim, where each array came from (the selection's repair_data). Steps 1 and 2's
    sizes are calibration.N1 and N2 unless ``steps12_n`` gives others (a known answer's small runs)."""
    old_confs, old_fp = table["inputs"]["configurations"], table["inputs"]["fingerprint"]
    n1, n2 = steps12_n or (cal.N1, cal.N2)
    null = {c: [i for i, conf in enumerate(configurations) if c in conf["null"]] for c in cal.CLAIMS}
    sel = {i: configuration_p(selection_dir, 3, i, configurations[i]["name"], fingerprint, n, seed, seed_scheme,
                              phase="selection") for i in null["C2"]}
    c2i = cal.CLAIMS.index("C2")
    c2_null_p = {t: [sel[i][:, c2i, ti] for i in null["C2"]] for ti, t in enumerate(prereg.TESTS)}
    pooled: dict[int, np.ndarray] = {}
    repair_null_p: dict[str, dict[str, list[np.ndarray]]] = {}
    where: dict[str, list[str]] = {}
    for fam in (f for f in prereg.FAMILY_CLAIMS if f in table["families"]):  # both, in the registered table
        repair_null_p[fam] = {}
        for c in step3_scope(table, usable, fam):
            if c == "C2":
                continue
            ci, ti = cal.CLAIMS.index(c), prereg.TESTS.index(table["families"][fam]["claims"][c]["test"])
            arrays, rows = [], []
            for i in null[c]:
                name = configurations[i]["name"]
                if i < len(old_confs):
                    if old_confs[i]["name"] != name:  # never a silent fallback to the selection run
                        raise prereg.PreregError(f"step 3's configuration {i} {name} is not steps 1 and 2's "
                                                 f"{old_confs[i]['name']}")
                    if i not in pooled:  # steps 1 and 2 pooled, with the selection run where it ran there
                        pooled[i] = np.concatenate([
                            configuration_p(steps12_dir, 1, i, name, old_fp, n1, cal.SEEDS[1], seed_scheme),
                            configuration_p(steps12_dir, 2, i, name, old_fp, n2, cal.SEEDS[2], seed_scheme),
                            *([sel[i]] if i in sel else [])])
                    arrays.append(pooled[i][:, ci, ti])
                    rows.append(f"{name}: steps 1 and 2 and the selection run ({n1 + n2} + {n})" if i in sel
                                else f"{name}: steps 1 and 2 ({n1 + n2})")
                elif i in sel:
                    arrays.append(sel[i][:, ci, ti])
                    rows.append(f"{name}: the selection run ({n})")
                else:
                    raise prereg.PreregError(f"{name}: {c} is null there, outside steps 1 and 2 and the selection run")
            repair_null_p[fam][c], where[f"{fam} {c}"] = arrays, rows
    return c2_null_p, repair_null_p, where


def step3_select(c2_null_p: Mapping[str, Sequence[np.ndarray]], repair_null_p: Mapping[str, Mapping[str, Sequence[np.ndarray]]],
                 table: Mapping[str, Any], usable: Mapping[str, Any], alpha: float) -> dict[str, dict[str, Any]]:
    """Step 3's selection per family: C2 at every level by cp_alpha_primes on the selection run (``c2_null_p``: per
    test, C2's p-values in each of its two-sided null configurations), its test by entry 46's rule (the primary if its
    selection-run size at alpha' is <= a at every level in every null configuration, else the fallback, else none);
    each failed level of another claim by cp_repair on its null configurations (``repair_null_p``: family -> claim ->
    the claim's test's p-values), the claim's step-1 test and its kept levels' alpha'."""
    weights = table["weights"]
    out = {}
    for family, members in ((f, m) for f, m in prereg.FAMILY_CLAIMS.items() if f in table["families"]):
        scope = step3_scope(table, usable, family)
        c2 = {t: cp_alpha_primes(c2_null_p[t], "C2", members, weights, alpha) for t in prereg.TESTS}
        ok = {t: sizes_within(c2_null_p[t], weights["C2"], c2[t], alpha) for t in prereg.TESTS}
        test = next((t for t in prereg.TESTS if ok[t]), None)
        fam: dict[str, Any] = {"C2": {"test": test, "alpha_prime": {str(s): v for s, v in c2[test or prereg.TESTS[0]].items()},
                                      "selection_size_ok": ok}, "repairs": {}}
        for c, levels in scope.items():
            if c == "C2":
                continue
            entry = table["families"][family]["claims"][c]
            kept = {int(s): float(entry["alpha_prime"][str(s)]) for s in usable["families"][family]["usable"][c]}
            rec = cp_repair(repair_null_p[family][c], c, members, weights, alpha, kept, levels)
            fam["repairs"][c] = {"test": entry["test"], "levels": {str(s): r for s, r in rec.items()}}
        out[family] = fam
    return out


def step3_candidates(table: Mapping[str, Any], usable: Mapping[str, Any], selection: Mapping[str, Any],
                     family: str) -> tuple[dict[str, str | None], dict[str, dict[int, float]], dict[str, list[int]]]:
    """One family's candidate calibration for step 3's validation: per claim its test, its alpha' at the levels in
    play (C2: every selected level; another claim: its step-2 usable levels with step 1's alpha' and its usable
    repaired levels with step 3's), and the levels step 3 must verify (C2's and the repaired ones)."""
    members = prereg.FAMILY_CLAIMS[family]
    sel = selection["families"][family] if "families" in selection else selection[family]
    tests, alphas, check = {}, {}, {}
    for c in members:
        if c == "C2":
            tests[c] = sel["C2"]["test"]
            alphas[c] = {int(s): float(v) for s, v in sel["C2"]["alpha_prime"].items()} if tests[c] else {}
            check[c] = sorted(alphas[c])
            continue
        entry = table["families"][family]["claims"][c]
        tests[c] = entry["test"]
        alphas[c] = {int(s): float(entry["alpha_prime"][str(s)]) for s in usable["families"][family]["usable"][c]}
        rep = (sel["repairs"].get(c) or {}).get("levels", {})
        fixed = {int(s): float(r["alpha_prime"]) for s, r in rep.items() if r["usable"]}
        alphas[c].update(fixed)
        if fixed:
            check[c] = sorted(fixed)
    return tests, alphas, check


def step_three(p_of: Callable[[int], np.ndarray], configurations: Sequence[Mapping[str, Any]], table: Mapping[str, Any],
               usable: Mapping[str, Any], selection: Mapping[str, Any], alpha: float, n: int) -> dict[str, dict[str, Any]]:
    """Step 3's validation (entry 106) on the fresh draws (``p_of(i)``: configuration i's n datasets x
    calibration.CLAIMS x prereg.TESTS; ``configurations``: each one's name and null claims, C2 two-sided), one
    configuration at a time in two passes: per level step 3 selected, its largest size over the claim's null
    configurations against a + 2 SE (usable iff within, or with no null configuration); then the six-claim FWER with the
    final usable levels (C2's validated ones; another claim's step-2 usable levels and its validated repaired ones)
    against alpha + 2 SE in every configuration. Per family: "tests", "alpha_prime" (claim -> s -> alpha' at its final
    usable levels), "usable", "levels" (claim -> s -> the record), "fwer", "fwer_pass"."""
    if not configurations:  # a vacuous pass otherwise: no size measured, no FWER row (the review of 1e31f1f)
        raise prereg.PreregError("step 3's validation has no configurations")
    weights = table["weights"]
    cand = {fam: step3_candidates(table, usable, selection, fam) for fam in prereg.FAMILY_CLAIMS}
    worst: dict[str, dict[str, dict[int, tuple[float, str] | None]]] = {
        fam: {c: dict.fromkeys(levels) for c, levels in check.items()} for fam, (_, _, check) in cand.items()}
    for i, conf in enumerate(configurations):
        p = p_of(i)
        if len(p) != n:
            raise prereg.PreregError(f"configuration {conf['name']}: {len(p)} datasets, not {n}")
        for fam, by_claim in worst.items():
            tests, alphas, _ = cand[fam]
            for c, levels in by_claim.items():
                if c not in conf["null"] or tests[c] is None:
                    continue
                x = p[:, cal.CLAIMS.index(c), prereg.TESTS.index(tests[c])]
                for s in levels:
                    size = float(np.mean(x <= alphas[c][s]))
                    if levels[s] is None or size > levels[s][0]:
                        levels[s] = (size, conf["name"])
    out = {}
    for fam, by_claim in worst.items():
        tests, alphas, _ = cand[fam]
        records: dict[str, dict[str, Any]] = {}
        final = {c: dict(v) for c, v in alphas.items()}
        for c, levels in by_claim.items():
            records[c] = {}
            for s, w in levels.items():
                a = level(alpha, weights[c], s)
                limit = a + 2 * _se(a, n)
                ok = w is None or w[0] <= limit
                records[c][str(s)] = {"alpha_prime": alphas[c][s], "size": None if w is None else w[0],
                                      "configuration": None if w is None else w[1], "se": _se(a, n), "limit": limit,
                                      "usable": ok}
                if not ok:
                    final[c].pop(s)
        out[fam] = {"tests": tests, "alpha_prime": {c: {str(s): v for s, v in sorted(final[c].items())} for c in final},
                    "usable": {c: sorted(final[c]) for c in final}, "levels": records, "fwer": []}
    for i, conf in enumerate(configurations):
        p = p_of(i)
        for fam, res in out.items():
            members = prereg.FAMILY_CLAIMS[fam]
            usable_a = {c: {int(s): v for s, v in res["alpha_prime"][c].items()} for c in members}
            entered = {c: (res["tests"][c] if usable_a[c] else None) for c in members}
            rate = family_fwer(p, {c: c in conf["null"] for c in members}, members, entered, usable_a, weights, alpha)
            limit = alpha + 2 * _se(alpha, n)
            res["fwer"].append({"configuration": conf["name"], "fwer": rate, "limit": limit, "pass": rate <= limit,
                                "null_claims": [c for c in members if c in conf["null"]]})
    for res in out.values():
        res["fwer_pass"] = all(r["pass"] for r in res["fwer"])
    return out


def step3_alpha_document(selection: Mapping[str, Any], inputs: Mapping[str, Any], alpha: float,
                         weights: Mapping[str, int]) -> dict[str, Any]:
    """The selection's output (STEP3_ALPHA_SCHEMA): per family C2's test and alpha' at every level and each repaired
    level's record, with what it was computed from (``inputs``)."""
    return {"schema": STEP3_ALPHA_SCHEMA, "rules": STEP3_RULES, "alpha": alpha, "weights": dict(weights),
            "families": {f: dict(v) for f, v in selection.items()}, "inputs": dict(inputs)}


def step3_document(result: Mapping[str, Mapping[str, Any]], inputs: Mapping[str, Any], date: str, alpha: float,
                   weights: Mapping[str, int]) -> dict[str, Any]:
    """The dated step-3 addendum (STEP3_SCHEMA, entry 106): per family every claim's test and alpha' at its final
    usable levels, the records of the levels step 3 verified, the six-claim FWER rows and the check, with what it was
    computed from (``inputs``: the alpha' table, the step-2 addendum and the selection by sha256, the validation's
    seed, N3, configurations, fingerprint and outputs, the procedure's and the inference code's digests)."""
    return {"schema": STEP3_SCHEMA, "rules": STEP3_RULES, "alpha": alpha, "weights": dict(weights), "date": date,
            "families": {f: {k: v for k, v in r.items()} for f, r in result.items()}, "inputs": dict(inputs)}


def _refuse3(ok: bool, why: str) -> None:
    if not ok:
        raise prereg.PreregError(f"the step-3 addendum is not the frozen step-3 procedure's output (entry 106): {why}")


def _monotone(alphas: Mapping[int, float]) -> bool:
    sums = sorted(alphas)
    return all(alphas[b] <= alphas[a] for a, b in pairwise(sums))


def load_step3(table_path: Path, table_sha256: str, usable_path: Path, usable_sha256: str, step3: Mapping[str, Any],
               family: str, alpha: float, weights: Mapping[str, int]) -> WeightedCalibration:
    """One family's frozen inputs after step 3 (entry 106): the step-3 addendum's final alpha' per claim (``step3``,
    read by its sha256), checked against the alpha' table and the step-2 usable set (each by its sha256, load_files):
    a claim outside step 3's scope exactly at its step-2 usable levels with step 1's alpha' and test; a repaired claim at
    those plus its repaired levels the addendum records usable; C2 at the levels it records usable, with their alpha';
    every claim's alpha' non-decreasing in the level. The FWER check is step 3's, on the final procedure, and step
    2's."""
    base = load_files(table_path, table_sha256, usable_path, usable_sha256, family, alpha, weights)
    table, usable = _read(table_path, table_sha256, ALPHA_SCHEMA, "alpha' table"), _read(
        usable_path, usable_sha256, USABLE_SCHEMA, "usable-level set")
    _refuse3(step3.get("schema") == STEP3_SCHEMA and step3.get("rules") == STEP3_RULES, "not a step-3 addendum")
    _refuse3(step3.get("alpha") == alpha and step3.get("weights") == dict(weights), "another alpha or other weights")
    rec = step3.get("inputs") or {}
    _refuse3(rec.get("table_sha256") == table_sha256 and rec.get("step2_addendum_sha256") == usable_sha256,
             "it names another alpha' table or step-2 addendum")
    fam = (step3.get("families") or {}).get(family) or {}
    scope = step3_scope(table, usable, family)
    claims = {}
    for c in prereg.FAMILY_CLAIMS[family]:
        test = (fam.get("tests") or {}).get(c)
        ap = {int(k): float(v) for k, v in ((fam.get("alpha_prime") or {}).get(c) or {}).items()}
        levels = (fam.get("levels") or {}).get(c) or {}
        kept = {int(k): float(v) for k, v in (base.claims[c]["alpha_prime"] or {}).items()}
        if c not in scope:
            _refuse3(test == base.claims[c]["test"] and ap == kept, f"{c} is outside step 3 but its alpha' changed")
        else:
            verified = {int(k): float(r["alpha_prime"]) for k, r in levels.items() if r.get("usable")}
            expect = verified if c == "C2" else {**kept, **verified}
            _refuse3(ap == expect, f"{c}'s alpha' are not its kept and validated levels'")
            _refuse3(c == "C2" or test == base.claims[c]["test"], f"{c}'s test changed")
        _refuse3(_monotone(ap), f"{c}'s alpha' fall as the level rises")
        _refuse3(sorted(ap) == list((fam.get("usable") or {}).get(c) or []), f"{c}'s usable levels disagree")
        claims[c] = {"test": test, "alpha_prime": {str(k): v for k, v in sorted(ap.items())}}
    _refuse3(isinstance(fam.get("fwer_pass"), bool), "the six-claim FWER check is not recorded")
    return WeightedCalibration(family, claims, dict(weights), alpha, bool(base.fwer_pass and fam["fwer_pass"]),
                               f"{base.source}; step 3 (entry 106)")


def _check_step3_records(step3: Mapping[str, Any], table: Mapping[str, Any], usable: Mapping[str, Any],
                         selection: Mapping[str, Any], configurations: Sequence[Mapping[str, Any]], alpha: float,
                         weights: Mapping[str, int], n: int) -> None:
    """The addendum's records as step_three writes them from these configurations and this selection, short of the
    sizes and rates (which only the validation chunks give): the alpha' table's families (both, in the registered
    table); per family the tests and the verified
    levels exactly the selection's candidates (step3_candidates) with their alpha'; per level its SE and limit, its
    worst configuration one of the claim's null configurations (none and no size iff the claim is null in none), and
    step 2's rule on its size; the FWER rows the configurations in order with each one's null claims in the family,
    each row and the check the rule's."""
    families = step3.get("families") or {}
    _refuse3(set(families) == set(table["families"]), "it does not record the alpha' table's families")
    lim = alpha + 2 * _se(alpha, n)
    for family, fam in families.items():
        members = prereg.FAMILY_CLAIMS[family]
        _refuse3(family in (selection.get("families") or {}), f"{family}: its selection records no {family}")
        tests, alphas, check = step3_candidates(table, usable, selection, family)
        _refuse3(fam.get("tests") == tests, f"{family}: its tests are not the selection's and step 1's")
        levels = fam.get("levels") or {}
        _refuse3({c: sorted(int(s) for s in v) for c, v in levels.items()} == check,
                 f"{family}: the levels it verified are not the selection's")
        for c, by_s in levels.items():
            null_in = [conf["name"] for conf in configurations if c in conf["null"]]
            for s, r in by_s.items():
                a = level(alpha, weights[c], int(s))
                limit = a + 2 * _se(a, n)
                size = r.get("size")
                _refuse3(r.get("alpha_prime") == alphas[c][int(s)] and r.get("se") == _se(a, n)
                         and r.get("limit") == limit,
                         f"{family} {c} s = {s}: its alpha', SE or limit is not the selection's and the rule's")
                _refuse3((size is None and r.get("configuration") is None) if not null_in else
                         (_number(size) and 0 <= size <= 1 and r.get("configuration") in null_in),
                         f"{family} {c} s = {s}: its worst configuration is not one where {c} is null")
                _refuse3(r.get("usable") == (size is None or size <= limit),
                         f"{family} {c} s = {s}: its decision is not step 2's rule on its recorded size")
        rows = fam.get("fwer") or []
        _refuse3([(r.get("configuration"), r.get("null_claims")) for r in rows]
                 == [(conf["name"], [c for c in members if c in conf["null"]]) for conf in configurations],
                 f"{family}: its FWER rows are not the bound configurations with their null claims")
        _refuse3(all(_number(r.get("fwer")) and 0 <= r["fwer"] <= 1 and r.get("limit") == lim
                     and r.get("pass") == (r["fwer"] <= lim) for r in rows), f"{family}: an FWER row is not the rule's")
        _refuse3(fam.get("fwer_pass") == all(r["pass"] for r in rows), f"{family}: the FWER check is not its rows'")


def check_step3(step3: Mapping[str, Any], table: Mapping[str, Any], usable: Mapping[str, Any],
                selection: Mapping[str, Any], *, configurations: Sequence[Mapping[str, Any]], fingerprint: str, n: int,
                seed: int, seed_scheme: str, procedure: str, inference: str, alpha: float,
                weights: Mapping[str, int], outputs: Callable[[Path], bool] = Path.is_dir,
                verified: Sequence[Mapping[str, Any]] = (), step3_sha256: str | None = None,
                selection_sha256: str | None = None) -> dict[str, Any]:
    """Entry 106: the step-3 addendum is the frozen step-3 procedure's output, or it is refused.
    - **Recorded inputs:** an ISO date; the validation's N3, seed and seed scheme (``n``, ``seed``, ``seed_scheme``:
      calibration.N3 and prereg.STEP3_SEEDS' validation seed for the frozen run); the selection by
      its sha256 (``selection_sha256``); the configurations with their two-sided null claims, exactly the bound ones
      (``configurations``: the config's step3_configurations, never empty), the fingerprint, the bound one
      (``fingerprint``: the config's step3_fingerprint) and the selection's, and the outputs (directory, chunks,
      sha256); the multiplicity code of the frozen digest (``procedure``) and the inference
      code of the calibrated digest (``inference``).
    - **Its records, re-derived** from the bound configurations and the selection (``_check_step3_records``): the levels
      it verified are exactly the selection's (tests, levels, alpha'), each level's worst configuration one where its
      claim is null (no size iff the claim is null in none), the FWER rows the configurations in order with their null
      claims; and its decisions from the recorded sizes: each verified level usable iff its size is <= a + 2 SE,
      SE = sqrt(a (1 - a) / N3); each FWER row passes iff its rate is <= alpha + 2 SE(alpha); the check passes iff all
      do.
    - **Recomputed from the validation outputs** when they are on this host (``outputs``): their digest is the
      recorded one and step_three from them, with the table, the step-2 set and the selection, gives every record;
      elsewhere the freeze check's record of that recomputation (``verified``: its registry's step3_outputs) for this
      addendum's sha256 is required."""
    rec = step3.get("inputs") or {}
    try:
        date = datetime.date.fromisoformat(step3.get("date"))
    except (TypeError, ValueError):
        date = None
    _refuse3(date is not None, f"its date {step3.get('date')!r} is not an ISO date")
    _refuse3(rec.get("n") == n and rec.get("seed") == seed and rec.get("seed_scheme") == seed_scheme,
             f"not {n} fresh datasets of step 3's validation seed {seed}")
    _refuse3(rec.get("procedure") == procedure, f"computed by multiplicity code {rec.get('procedure')}, not the frozen "
                                                f"{procedure}")
    _refuse3(rec.get("inference") == inference, f"run on inference code {rec.get('inference')}, not the calibrated "
                                                f"{inference}")
    _refuse3(selection_sha256 is not None and rec.get("selection_sha256") == selection_sha256,
             "it names another selection")
    _refuse3(bool(configurations), "no step-3 configurations are bound (step3_configurations)")
    confs = rec.get("configurations")
    _refuse3(confs == list(configurations),
             "its configurations and their null claims are not the bound ones (step3_configurations)")
    _refuse3(fingerprint not in (prereg.PLACEHOLDER, "", None) and rec.get("fingerprint") == fingerprint
             and (selection.get("inputs") or {}).get("fingerprint") == fingerprint,
             "its fingerprint or its selection's is not the bound step-3 fingerprint (step3_fingerprint)")
    out = rec.get("outputs") or {}
    _refuse3(isinstance(out.get("directory"), str) and isinstance(out.get("chunks"), int)
             and isinstance(out.get("sha256"), str), "it names no validation outputs (directory, chunks, sha256)")
    _check_step3_records(step3, table, usable, selection, confs, alpha, weights, n)
    mine = [r for r in verified if step3_sha256 is not None and r.get("addendum_sha256") == step3_sha256]
    _refuse3(len(mine) <= 1, f"the freeze registry records the step-3 addendum {len(mine)} times")
    recorded = {k: out[k] for k in ("chunks", "sha256")}
    for r in mine:
        _refuse3(r.get("procedure") == procedure and r.get("outputs") == recorded,
                 "the freeze registry's record is of another procedure or other outputs")
    directory = Path(out["directory"])
    if not outputs(directory / "step3-validation"):
        _refuse3(bool(mine), f"its validation outputs ({directory}) are not on this host and the freeze registry records "
                             f"no recomputation of them for this addendum")
        return {"date": step3["date"], "recomputed": False, "verified": dict(mine[0]),
                "checked": "the recorded inputs and every decision re-derived from the recorded sizes; the freeze "
                           "check's record of its recomputation from the validation chunks"}
    found = outputs_digest(directory, "3-validation")
    _refuse3(found == recorded, f"the validation outputs at {directory} are {found}, not the recorded {recorded}")
    result = step_three(lambda i: configuration_p(directory, 3, i, confs[i]["name"], rec["fingerprint"], n, rec["seed"],
                                                  rec["seed_scheme"], phase="validation"),
                        confs, table, usable, selection, alpha, n)
    for family, fam in result.items():
        _refuse3({k: fam[k] for k in ("tests", "alpha_prime", "usable", "levels", "fwer", "fwer_pass")} ==
                 {k: (step3["families"].get(family) or {}).get(k) for k in ("tests", "alpha_prime", "usable", "levels",
                                                                             "fwer", "fwer_pass")},
                 f"{family}: recomputed from the validation outputs, step 3 gives other records")
    return {"date": step3["date"], "recomputed": True,
            "checked": f"recomputed from the validation outputs at {directory} ({found['chunks']} chunks, sha256 "
                       f"{found['sha256']}): every record equal"}


def _refuse_selection(ok: bool, why: str) -> None:
    if not ok:
        raise prereg.PreregError(f"the step-3 selection is not the frozen step-3 procedure's output from its frozen inputs "
                                 f"(entry 106): {why}")


def _check_selection_records(selection: Mapping[str, Any], table: Mapping[str, Any], usable: Mapping[str, Any],
                             alpha: float) -> None:
    """The selection's records as step3_select writes them, short of the p-values (which only the chunks give): the
    alpha' table's families (both, in the registered table); C2 at exactly its reachable levels, each alpha' on its level's grid (j x 0.0005 a, or 0) and none falling
    as the level rises, its test entry 46's rule on its recorded size checks; per family exactly the repairs in step 3's
    scope, each with its claim's step-1 test at exactly the failed levels, its records re-derived from their recorded own
    values (repair_records)."""
    weights, families = table["weights"], selection.get("families") or {}
    _refuse_selection(set(families) == set(table["families"]), "it does not record the alpha' table's families")
    for family, fam in families.items():
        scope = step3_scope(table, usable, family)
        c2 = fam.get("C2") or {}
        ok = c2.get("selection_size_ok") or {}
        _refuse_selection(set(ok) == set(prereg.TESTS) and all(isinstance(v, bool) for v in ok.values())
                          and c2.get("test") == next((t for t in prereg.TESTS if ok[t]), None),
                          f"{family}: C2's test is not entry 46's rule on its recorded size checks")
        ap = {int(k): v for k, v in (c2.get("alpha_prime") or {}).items()}
        _refuse_selection(sorted(ap) == scope["C2"] and all(
            _number(v) and cal._snap(v, level(alpha, weights["C2"], k), cal.GRID_STEPS) == v for k, v in ap.items())
            and _monotone(ap), f"{family}: C2's alpha' are not on its levels' grids at its reachable levels, or fall")
        repairs = fam.get("repairs") or {}
        _refuse_selection(set(repairs) == set(scope) - {"C2"}, f"{family}: its repairs are not step 3's scope")
        for c, rep in repairs.items():
            entry = table["families"][family]["claims"][c]
            kept = {int(k): float(entry["alpha_prime"][str(k)]) for k in usable["families"][family]["usable"][c]}
            levels = rep.get("levels") or {}
            _refuse_selection(rep.get("test") == entry["test"] and sorted(int(k) for k in levels) == scope[c]
                              and all(isinstance(r, dict) and _number(r.get("own")) for r in levels.values()),
                              f"{family} {c}: not its step-1 test at exactly its failed levels, each with its own value")
            again = repair_records({int(k): r["own"] for k, r in levels.items()}, c, weights, alpha, kept)
            _refuse_selection({str(k): r for k, r in again.items()} == levels,
                              f"{family} {c}: its records are not cp_repair's chain from their own values")


def check_selection(selection: Mapping[str, Any], table: Mapping[str, Any], usable: Mapping[str, Any], *,
                    configurations: Sequence[Mapping[str, Any]], fingerprint: str, table_sha256: str,
                    usable_sha256: str, n: int, seed: int, seed_scheme: str, procedure: str, inference: str,
                    alpha: float, weights: Mapping[str, int], steps12_outputs: Mapping[str, Any],
                    outputs: Callable[[Path], bool] = Path.is_dir, verified: Sequence[Mapping[str, Any]] = (),
                    selection_sha256: str | None = None, require_recomputed: bool = False,
                    steps12_n: tuple[int, int] | None = None) -> dict[str, Any]:
    """Entry 106: step 3's selection is the frozen procedure's output from the frozen inputs, or it is refused (the review
    of 1e31f1f: a selection's sha256 alone binds no threshold to the data).
    - **Recorded inputs:** step 3's rules, alpha and weights; the alpha' table and the step-2 addendum by their sha256
      (``table_sha256``, ``usable_sha256``); the selection run's N3_SELECTION, seed and seed scheme (``n``, ``seed``,
      ``seed_scheme``); the bound fingerprint (``fingerprint``: the config's step3_fingerprint) and the table's as steps
      1 and 2's; the bound configurations (``configurations``, never empty) and, as its selection configurations,
      exactly their C2-null ones; the frozen multiplicity and inference digests; the selection outputs (directory,
      chunks, sha256) and steps 1 and 2's (``steps12_outputs``: the config's step1_outputs and the step-2 addendum's
      outputs, chunks and sha256) with their directory.
    - **Its records**, re-derived as far as they can be without the p-values (``_check_selection_records``).
    - **Recomputed** when the selection chunks and steps 1 and 2's are on this host: their digests are the recorded ones
      and step3_select on step3_selection_arrays from them gives the recorded families exactly; elsewhere the freeze
      check's record of that recomputation (``verified``: its registry's step3_selection_outputs, each
      {"selection_sha256", "procedure", "outputs": {"selection", "step1", "step2": {"chunks", "sha256"}}}) for this
      selection's sha256 is required, unless ``require_recomputed`` (the validation's producer, which runs where the
      chunks are). ``steps12_n``: as step3_selection_arrays (the registered N1 and N2 unless given)."""
    rec = selection.get("inputs") or {}
    _refuse_selection(selection.get("schema") == STEP3_ALPHA_SCHEMA and selection.get("rules") == STEP3_RULES,
                      "it is not a step-3 selection under step 3's rules")
    _refuse_selection(selection.get("alpha") == alpha and selection.get("weights") == dict(weights),
                      "another alpha or other weights")
    _refuse_selection(rec.get("table_sha256") == table_sha256 and rec.get("step2_addendum_sha256") == usable_sha256,
                      "it names another alpha' table or step-2 addendum")
    _refuse_selection(rec.get("n_selection") == n and rec.get("seed") == seed and rec.get("seed_scheme") == seed_scheme,
                      f"it is not {n} datasets of step 3's selection seed {seed}")
    _refuse_selection(fingerprint not in (prereg.PLACEHOLDER, "", None) and rec.get("fingerprint") == fingerprint,
                      "its fingerprint is not the bound step-3 fingerprint (step3_fingerprint)")
    _refuse_selection(rec.get("steps12_fingerprint") == table["inputs"]["fingerprint"],
                      "its steps 1 and 2 are not the alpha' table's")
    _refuse_selection(bool(configurations), "no step-3 configurations are bound (step3_configurations)")
    _refuse_selection(rec.get("configurations") == list(configurations),
                      "its configurations and their null claims are not the bound ones (step3_configurations)")
    _refuse_selection(rec.get("selection_configurations") == [c["name"] for c in configurations if "C2" in c["null"]],
                      "its run is not exactly C2's two-sided null configurations")
    _refuse_selection(rec.get("procedure") == procedure and rec.get("inference") == inference,
                      "it was computed by other multiplicity or inference code")
    out, out12 = rec.get("outputs") or {}, rec.get("steps12_outputs") or {}
    _refuse_selection(isinstance(out.get("directory"), str) and isinstance(out.get("chunks"), int)
                      and isinstance(out.get("sha256"), str) and isinstance(rec.get("steps12_directory"), str),
                      "it names no outputs (directory, chunks, sha256) or no steps 1 and 2 directory")
    frozen12 = {k: {kk: (steps12_outputs.get(k) or {}).get(kk) for kk in ("chunks", "sha256")} for k in ("step1", "step2")}
    _refuse_selection(all(isinstance(v["chunks"], int) and isinstance(v["sha256"], str) for v in frozen12.values())
                      and {k: {kk: (out12.get(k) or {}).get(kk) for kk in ("chunks", "sha256")}
                           for k in ("step1", "step2")} == frozen12,
                      "it did not read steps 1 and 2's frozen chunks")
    _check_selection_records(selection, table, usable, alpha)
    mine = [r for r in verified if selection_sha256 is not None and r.get("selection_sha256") == selection_sha256]
    _refuse_selection(len(mine) <= 1, f"the freeze registry records it {len(mine)} times")
    recorded = {"selection": {k: out[k] for k in ("chunks", "sha256")}, **frozen12}
    for r in mine:
        _refuse_selection(r.get("procedure") == procedure and r.get("outputs") == recorded,
                          "the freeze registry's record of it is of another procedure or other outputs")
    sel_dir, dir12 = Path(out["directory"]), Path(rec["steps12_directory"])
    if not (outputs(sel_dir / "step3-selection") and outputs(dir12 / "step1") and outputs(dir12 / "step2")):
        _refuse_selection(not require_recomputed, f"its chunks ({sel_dir}, {dir12}) are not on this host")
        _refuse_selection(bool(mine), f"its chunks ({sel_dir}, {dir12}) are not on this host and the freeze registry "
                                      f"records no recomputation of them for this selection")
        return {"recomputed": False, "verified": dict(mine[0]),
                "checked": "the recorded inputs and records; the freeze check's record of its recomputation from the "
                           "chunks"}
    found = {"selection": outputs_digest(sel_dir, "3-selection"), "step1": outputs_digest(dir12, 1),
             "step2": outputs_digest(dir12, 2)}
    _refuse_selection(found == recorded, f"its chunks are {found}, not the recorded {recorded}")
    c2_p, repair_p, where = step3_selection_arrays(table, usable, configurations, sel_dir, dir12, rec["fingerprint"],
                                                   n, seed, seed_scheme, steps12_n)
    again = step3_select(c2_p, repair_p, table, usable, alpha)
    _refuse_selection(json.loads(json.dumps(again)) == selection.get("families") and where == rec.get("repair_data"),
                      "recomputed from its chunks, it gives other alpha' or reads other data")
    return {"recomputed": True, "checked": f"recomputed from the selection chunks at {sel_dir} and steps 1 and 2's at "
                                           f"{dir12}: every alpha' equal"}


def addendum_document(table_sha256: str, result: Mapping[str, Mapping[str, Any]], inputs: Mapping[str, Any], date: str,
                      alpha: float, weights: Mapping[str, int]) -> dict[str, Any]:
    """The dated step-2 addendum (entry 91): a weighted-holm-usable/2 document (``usable_document``: per family the
    usable levels and the FWER check, naming the alpha' table's sha256) with ``step_two``'s record of every level and
    configuration, the date, and what it was computed from (``inputs``)."""
    doc = usable_document(table_sha256, result, alpha, weights)
    for family, fam in doc["families"].items():
        fam.update(levels=result[family]["levels"], fwer=result[family]["fwer"])
    return {**doc, "addendum": ADDENDUM, "date": date, "inputs": dict(inputs)}


def freeze_records(path: Path, key: str = "step2_outputs") -> list[dict[str, Any]]:
    """The freeze check's records of step 2's (``key`` "step2_outputs") or step 3's ("step3_outputs") outputs: its
    registry's (data/freeze/registry.json, ``FREEZE_SCHEMA``) list under ``key``, each {"addendum_sha256", "procedure", "outputs": {"chunks", "sha256"}}, written by the freeze
    check only after recomputing that addendum from the step-2 chunks on SiteA (``check_addendum`` with them there: their
    ``outputs_digest`` and ``step_two`` from them equal to the addendum's). No records when the registry is absent or
    of an earlier schema, which records none; refused when malformed."""
    path = Path(path)
    if not path.is_file():
        return []
    data = json.loads(path.read_bytes())
    if not isinstance(data, dict):
        raise prereg.PreregError(f"{path}: not a freeze registry")
    if data.get("schema") != FREEZE_SCHEMA:
        return []
    records = data.get(key, [])
    if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
        raise prereg.PreregError(f"{path}: its {key} is not a list of records")
    return records


def _refuse_unless(ok: bool, why: str) -> None:
    if not ok:
        raise prereg.PreregError(f"the step-2 addendum was not produced by the frozen step-2 procedure (entry 91): {why}")


def check_addendum(addendum: Mapping[str, Any], table: Mapping[str, Any], *, n: int, seed: int, seed_scheme: str,
                   procedure: str, alpha: float, weights: Mapping[str, int],
                   outputs: Callable[[Path], bool] = Path.is_dir, verified: Sequence[Mapping[str, Any]] = (),
                   addendum_sha256: str | None = None) -> dict[str, Any]:
    """Entry 91: the step-2 addendum is the frozen step-2 procedure's output, or it is refused.
    - **Recorded inputs:** its date is an ISO date; it was computed on n datasets per configuration with step 2's seed
      and seed scheme, under the alpha' table's step-1 fingerprint and configurations (and their null claims), by the
      multiplicity code of the frozen digest (``procedure``).
    - **Its decisions, re-derived:** every claim with a test has a record at exactly its reachable s with the table's
      alpha' there, the limit a + 2 SE recomputed and "usable" iff the recorded size is within it; the usable sets are
      those levels; one FWER row per configuration in order, its limit recomputed, "pass" iff within it; the FWER
      check iff every row passes.
    - **Recomputed from the step-2 outputs** when the chunks of the directory it names (calibrate.py's OUT, its step2/)
      are on this host (``outputs``): their digest must be the recorded one, and ``step_two`` on them must give every
      recorded record exactly.
    - **Otherwise refused unless the freeze check recomputed it** (the review of 9d9c88c: an internally consistent
      addendum naming outputs on no host passed): ``verified`` (``freeze_records``) must hold exactly one record for the
      addendum's sha256 (``addendum_sha256``), of the frozen procedure, whose step-2 outputs (chunks and content
      sha256) are the addendum's. A record for it must agree in either case.
    Returns how it was checked."""
    rec = addendum.get("inputs")
    t_inputs = table.get("inputs")
    _refuse_unless(isinstance(rec, dict) and isinstance(t_inputs, dict),
                   "the addendum or the alpha' table records no inputs")
    try:
        date = datetime.date.fromisoformat(addendum.get("date"))
    except (TypeError, ValueError):
        date = None
    _refuse_unless(date is not None, f"its date {addendum.get('date')!r} is not an ISO date")
    configurations = t_inputs.get("configurations") or []
    want = {"n": n, "seed": seed, "seed_scheme": seed_scheme, "fingerprint": t_inputs.get("fingerprint"),
            "configurations": [c["name"] for c in configurations], "procedure": procedure}
    for key, value in want.items():
        _refuse_unless(rec.get(key) == value, f"its {key} is {rec.get(key)!r}, not {value!r}")
    out = rec.get("outputs")
    _refuse_unless(isinstance(out, dict) and isinstance(out.get("directory"), str) and isinstance(out.get("chunks"), int)
                   and isinstance(out.get("sha256"), str), "it names no step-2 outputs (directory, chunks, sha256)")
    _refuse_unless(addendum.get("addendum") == ADDENDUM, "it does not state what it binds")
    families = addendum.get("families") or {}
    _refuse_unless(set(families) == set(table.get("families") or {}) and all(
        f in prereg.FAMILY_CLAIMS and set(table["families"][f].get("claims") or {}) == set(prereg.FAMILY_CLAIMS[f])
        for f in families), "not the alpha' table's families, or a family of the table incomplete")
    for family, fam in families.items():
        tests, alphas = _table_levels(table, family)
        members = prereg.FAMILY_CLAIMS[family]
        levels = fam.get("levels") or {}
        _refuse_unless(set(levels) == set(members), f"{family}: no record for every claim")
        for c in members:
            sums = reachable_sums(c, members, weights) if tests[c] else []
            _refuse_unless(set(levels[c]) == {str(s) for s in sums}, f"{family} {c}: records off its reachable levels")
            for s in sums:
                r, a = levels[c][str(s)], level(alpha, weights[c], s)
                _refuse_unless(r.get("alpha_prime") == alphas[c][s], f"{family} {c} s = {s}: not the table's alpha'")
                _refuse_unless(r.get("se") == _se(a, n) and r.get("limit") == a + 2 * _se(a, n),
                               f"{family} {c} s = {s}: its limit is not a + 2 SE")
                size = r.get("size")
                _refuse_unless((size is None) == (r.get("configuration") is None) and (size is None or _number(size)),
                               f"{family} {c} s = {s}: its size {size!r} and configuration {r.get('configuration')!r}")
                _refuse_unless(r.get("usable") == (size is None or size <= r["limit"]),
                               f"{family} {c} s = {s}: usable {r.get('usable')!r} with size {size!r}, limit {r['limit']!r}")
            _refuse_unless(fam["usable"].get(c) == [s for s in sums if levels[c][str(s)]["usable"]],
                           f"{family} {c}: its usable levels are not the levels within their limit")
        rows = fam.get("fwer") or []
        _refuse_unless([r.get("configuration") for r in rows] == want["configurations"],
                       f"{family}: not one FWER row per configuration")
        for r, conf in zip(rows, configurations):
            _refuse_unless(r.get("null_claims") == [c for c in members if c in conf["null"]] and _number(r.get("fwer"))
                           and r.get("limit") == alpha + 2 * _se(alpha, n) and r.get("pass") == (r["fwer"] <= r["limit"]),
                           f"{family} {r.get('configuration')}: the FWER row does not follow from its rate")
        _refuse_unless(fam.get("fwer_pass") == all(r["pass"] for r in rows), f"{family}: the FWER check")
    recorded = {"chunks": out["chunks"], "sha256": out["sha256"]}
    mine = [r for r in verified if addendum_sha256 is not None and r.get("addendum_sha256") == addendum_sha256]
    _refuse_unless(len(mine) <= 1, f"the freeze registry records the addendum (sha256 {addendum_sha256}) {len(mine)} "
                                   f"times")
    for r in mine:
        _refuse_unless(r.get("procedure") == procedure, f"the freeze check recomputed it with procedure "
                                                        f"{r.get('procedure')!r}, not the frozen {procedure!r}")
        _refuse_unless(r.get("outputs") == recorded, f"its step-2 outputs {recorded} are not the "
                                                     f"{r.get('outputs')!r} the freeze check recomputed")
    directory = Path(out["directory"])
    if not outputs(directory / "step2"):  # calibrate.py's OUT: the step-2 chunks under step2/
        _refuse_unless(bool(mine), f"the step-2 outputs it names ({directory}) are not on this host, and the freeze "
                                   f"registry records no recomputation of them for this addendum (sha256 "
                                   f"{addendum_sha256}): recorded sizes alone bind nothing")
        return {"date": addendum["date"], "recomputed": False, "verified": dict(mine[0]),
                "checked": f"the recorded inputs and every decision re-derived from the recorded sizes; the step-2 "
                           f"outputs ({directory}) are not on this host: their content ({recorded['chunks']} chunks, "
                           f"sha256 {recorded['sha256']}) and the result are the freeze check's recomputation from the "
                           f"step-2 chunks"}
    found = outputs_digest(directory, 2)
    _refuse_unless(found == {k: rec["outputs"].get(k) for k in found},
                   f"the step-2 outputs at {directory} are {found}, not the recorded {rec['outputs']}")
    result = step_two(lambda i: configuration_p(directory, 2, i, configurations[i]["name"], rec["fingerprint"], n, seed,
                                                seed_scheme),
                      configurations, table, weights, alpha, n)
    for family, fam in families.items():
        for key in ("usable", "levels", "fwer", "fwer_pass"):
            _refuse_unless(json.loads(json.dumps(result[family][key])) == fam[key],
                           f"{family}: its {key} differ from step 2's recomputed from the outputs at {directory}")
    return {"date": addendum["date"], "recomputed": True,
            "checked": f"recomputed from the step-2 outputs at {directory} ({found['chunks']} chunks, sha256 "
                       f"{found['sha256']}): every record equal"}


def _claim_name(cid: str, cfg: prereg.PreregConfig) -> str:
    return "C1a pooled" if cid == "C1a" and cfg.c1a_pooled else cid


def decisions(p_tests: Mapping[str, Mapping[str, float]], cfg: prereg.PreregConfig,
              weighted: WeightedCalibration) -> dict[str, dict[str, Any]]:
    """Per contrast id, in family order, the claim's family block: its weight, test, raw p, usable alpha', p*, and the
    step and level of the step-down it was rejected at (None: not rejected), and the claim."""
    family = prereg.POOLED if cfg.c1a_pooled else prereg.TWO_FAMILY
    if weighted.family is not None and weighted.family != family:
        raise prereg.PreregError(f"the weighted Holm's tables are the {weighted.family} family's, not the config's "
                                 f"{family!r}")
    if dict(weighted.weights) != dict(cfg.holm_weights) or weighted.alpha != cfg.alpha:
        raise prereg.PreregError("the weighted Holm's tables were computed with other weights or alpha than the config's")
    weights = [cfg.holm_weights[_claim_name(cid, cfg)] for cid in prereg.CONTRASTS]
    blocks, stars = {}, []
    for cid, w in zip(prereg.CONTRASTS, weights):
        test, alphas = weighted.final(_claim_name(cid, cfg))
        raw = None if test is None else p_tests[cid][test]
        star = 1.0 if test is None else p_star(raw, w, alphas, cfg.alpha)
        blocks[cid] = {"procedure": PROCEDURE, "table": weighted.source, "weight": w, "weights_total": sum(weights),
                       "test": test, "p_test": raw, "p_star": star,
                       "alpha_prime": None if alphas is None else {str(s): a for s, a in sorted(alphas.items())}}
        stars.append(star)
    for cid, r in zip(prereg.CONTRASTS, step_down(stars, weights, cfg.alpha)):
        blocks[cid].update(step=r["step"], level=r["level"], claim=r["rejected"])
    return blocks


def c3_equivalence(comp: Mapping[str, Any], calibrated: bool, margin: float) -> dict[str, Any]:
    """Change-log entry 85 (prereg §2 C3): curated and generated are equivalent in false-flag rate at the margin (C3's
    smallest effect of interest, 5 points) when C3's calibrated claim-level interval, ``comp``'s weighted-Bonferroni
    interval at its first level, lies strictly inside (-margin, +margin): no test is added to the family. Never read
    from an uncertified interval ("not certified": no calibrated alpha' at the first level), nor from a statistic
    that is not estimable; an unbounded end (None) does not lie inside."""
    rule = (f"equivalent when the calibrated claim-level interval lies strictly inside +-{margin:g} (entry 85); never "
            f"from an uncertified interval")
    if not calibrated:
        return {"margin": margin, "equivalent": None, "status": "not certified", "reading": "not certified", "rule": rule}
    if comp["estimate"] is None:
        return {"margin": margin, "equivalent": None, "status": "not estimable", "reading": None, "rule": rule}
    iv = comp["ci_weighted_bonferroni"]
    inside = iv["lo"] is not None and iv["hi"] is not None and -margin < iv["lo"] and iv["hi"] < margin
    points = f"{round(100 * margin, 6):g} points"
    return {"margin": margin, "equivalent": inside, "status": "decided",
            "reading": f"equivalent within {points}" if inside else f"not equivalent within {points}", "rule": rule}


def weighted_family(contrasts: Mapping[str, Mapping[str, Any]], cfg: prereg.PreregConfig,
                    weighted: WeightedCalibration) -> dict[str, dict[str, Any]]:
    """run_contrasts' output (uncalibrated, statistics kept: keep_private) with the weighted family in place of its plain
    Holm: each claim's ``family`` block (``decisions``); every component's 95% interval with the claim's test and its
    claim-level interval at the first weighted level 0.05 w_i / 16, at the calibrated alpha' there where the claim has
    one (ruling 1), else at the nominal level with the reason; C2's "at least 5 points" and C3's equivalence at its
    margin (entry 85) from that interval; and CR2 beside it, nominal weighted Holm on its raw p's and its intervals at
    the nominal levels (ruling 2)."""
    blocks = decisions({cid: c["p_test"] for cid, c in contrasts.items()}, cfg, weighted)
    weights = [blocks[cid]["weight"] for cid in prereg.CONTRASTS]
    nominal = dict(zip(prereg.CONTRASTS, step_down([contrasts[cid]["p_test"]["cr2"] for cid in prereg.CONTRASTS],
                                                   weights, cfg.alpha)))
    out = {}
    for cid, c in contrasts.items():
        fam, r = blocks[cid], nominal[cid]
        test, alphas = weighted.final(_claim_name(cid, cfg))
        shown, first = test or "boot_t", level(cfg.alpha, fam["weight"], fam["weights_total"])
        calibrated = alphas is not None and fam["weights_total"] in alphas
        at = alphas[fam["weights_total"]] if calibrated else first
        extra = {"coverage": "calibrated"} if calibrated else {
            "coverage": "not certified",  # design log ruling: never presented as calibrated
            "reason": f"no calibrated alpha' at s = {fam['weights_total']} (no test, the level unusable at step 2, or the "
                      f"FWER check failed): the nominal level, its coverage not certified"}
        comps = []
        for comp in c["components"]:
            s = comp["_s"]
            comp = {k: v for k, v in comp.items() if k not in ("_s", "ci95", "ci_bonferroni", "cross_check_cr2")}
            comp["ci95"] = prereg._interval(s, 0.05, "two", shown)
            comp["ci_weighted_bonferroni"] = prereg._interval(s, at, c["sided"], shown) | {
                "level_nominal": first, "calibrated": calibrated, "test": shown, **extra}
            comp["cross_check_cr2"] = {"ci95": prereg._interval(s, 0.05, "two", "cr2"),
                                       "ci_weighted_bonferroni": prereg._interval(s, first, c["sided"], "cr2")
                                       | {"level_nominal": first, "nominal": True, "coverage": "nominal"}}
            comps.append(comp)
        if cid == "C2":  # from a calibrated claim-level bound only (the review of 70281e9): else not decided, None;
            # in C2's direction by its test (entry 106: two-sided)
            reading = prereg.c2_at_least(comps, c["direction"][shown], "ci_weighted_bonferroni", cfg.c2_min_effect)
            fam["at_least_min_effect"] = {k: (v if calibrated else None) for k, v in reading.items()}
            fam["both_at_least_min_effect"] = all(fam["at_least_min_effect"].values()) if calibrated else None
        if cid == "C3":  # entry 85: equivalence from the calibrated claim-level interval only
            fam["equivalence"] = c3_equivalence(comps[0], calibrated, cfg.c3_sesoi)
        out[cid] = {**{k: v for k, v in c.items() if k not in _PLAIN_HOLM}, "components": comps, "family": fam,
                    "cross_check_cr2": {"descriptive": True, "test": "cr2", "procedure": NOMINAL,
                                        "p_test": c["p_test"]["cr2"], "step": r["step"], "level": r["level"],
                                        "rejected_nominal": r["rejected"]}}
    return out
