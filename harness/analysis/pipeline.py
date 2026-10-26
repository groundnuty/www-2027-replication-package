"""From run records and checker-ledger rows to the paper's numbers (design log, EA.1).

``analyse`` computes, on one stratified resample matrix of the analysis's base questions
(prereg §4 "Resampling"), for every arm (graph x model x cohort x use x source): raw counts
(the six cells, N_out, N_att, retries, missing attempts, side events), every metric that
exists for the arm under prereg §3's weighting with a 95% percentile interval, exact bounds
for binary question-level events (§3), flag precision per check type, recall per fault class
and false-flag rate per control type on the bank, bank precision at fixed prevalences (§3),
cost distributions, and the same per stratum (§6 "Strata"). These per-arm numbers are
exploratory. With a ``PreregConfig`` it adds the registered confirmatory analysis
(prereg.confirmatory). ``analyse_batches`` is the entry point for run records: it gates every
batch on its replay check and completes the registered design with missing attempts first.

The result is a plain JSON-able dict; the macros are generated from the written file only.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import statistics
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from harness.analysis import leak
from harness.analysis.ablations import ablation_table
from harness.analysis.digest import ROOT, code_digest
from harness.analysis.gate import (
    BATCH_DIRS,
    GRID,
    PLAIN,
    grid_cells,
    manifest_layout,
    require_questions_match,
    require_replay_check,
    require_scorer,
    select_questions,
)
from harness.analysis.inputs import (
    BANK,
    EXPLORATORY,
    REFUSAL_SETTINGS,
    REFUSAL_SIGNALS,
    SINGLE_SPLIT_LEDGER,
    UNRECORDED,
    Attempt,
    CheckedCallFn,
    CheckedOutput,
    InputError,
    attempts_from_batch,
    bank_items_bytes,
    check_bank_manifest,
    check_one_stack,
    complete_design,
    final_message,
    first_line,
    load_batch,
    load_checker_ledger,
    load_questions,
    output_cap_report,
    refusal_setting,
    store_name,
)
from harness.analysis.metrics import (
    BANK_METRICS,
    CHECKER_METRICS,
    METRICS,
    PROMPT,
    PROMPT_METRICS,
    PROPORTION,
    QUESTION_EVENTS,
    RATIO,
    ArmKey,
    Unit,
    arm_counts,
    build_arms,
    metric_arrays,
    metric_of,
    point_estimate,
    question_vectors,
)
from harness.analysis.model_settings import ModelSettings
from harness.analysis.prereg import (
    PLACEHOLDER,
    ClusterMap,
    PreregConfig,
    PreregError,
    check_test_run_split,
    compared_arms,
    confirmatory,
    index_from_questions,
    load_clusters,
    preflight,
)
from harness.analysis.scorer_changes import C1_UNDER_V4, ScorerChanges, c1_under_v4
from harness.analysis.scorer_changes import c1_table as scorer_changes_c1
from harness.analysis.stats import (
    QuestionIndex,
    clopper_pearson,
    interval,
    ratio_draws,
    resample_matrix,
)

# /4: the real-output ledger (prereg §7); /5: the refusals (design log, entry 91); /6: the verdicts the frozen scorer
# changes against scorer-rules-v4, per arm and for C1's prompted arms (change-log entries 120 and 127; scorer_changes)
RESULTS_SCHEMA = "analysis-results/6"
DEFAULT_SEED = 20261026  # prereg §4
DEFAULT_N_BOOT = 10_000  # prereg §4
DEFAULT_LEVEL = 0.95
SIDE_COUNTS = ("infra_tries", "failed_harness")
PREVALENCES = (0.10, 0.25, 0.50)  # prereg §3 "Bank metrics"


# The checker loop's names of the ledger's sources (feat/checker-loop harness/checkloop/ledger.py: a source is its arm's
# name, the census at threshold 5 the arm "census-t5", and the matched generic control's rows "generic_retry/<arm>") ->
# the config's (census_source, generic_source; the generic retry of C5 is the one on the curated arm's flags). Every
# other source keeps its name: generic_retry/union, a replay variant "<arm>/<variant>", the masked draws, the union.
# The dev grid's generic cell is gen:any (generic_retry/any), whose arm is the role's feedback arms; under checker-loop/2
# a generic cell decides on its arm's curated sources alone (spec trace B10, design log entry 91), so its rows are the
# generic arm's too, but only those whose record shows that decision (GENERIC_ON_CURATED: curated_generic_rows).
LEDGER_SOURCES = {"census-t5": "census", "generic_retry/curated": "generic", "generic_retry/any": "generic"}
GENERIC_ON_CURATED = ("generic_retry/any",)
# The checker versions whose generic cell decides on its arm's curated sources alone (harness/checkloop/online.py:
# checker-loop/1 decided on the whole arm, GENERIC_ON_FULL_ARM), and the source kind that decides (GENERIC_TRIGGER_KIND).
CURATED_GENERIC_VERSIONS = ("checker-loop/2",)
CURATED_KIND = "curated"


def curated_generic_rows(rows: Iterable[CheckedOutput], records: Iterable[dict]) -> set[tuple[str, str]]:
    """The (run id, source) of each GENERIC_ON_CURATED ledger row whose generic cell's record shows it decided on its
    arm's curated sources alone: the one record of that generic cell (its checker's mode generic and arm the row's) for
    the row's question and model, the row keyed by the record's own run id or by its parent's (checker.branched_from),
    of the row's graph; its checker version one of CURATED_GENERIC_VERSIONS, its arm holding a curated source, and every
    check report its decision read a curated source's. Any other such row is refused, naming the row, the record and the
    fields; every other row is left alone."""
    records = [r for r in records if isinstance(r.get("checker"), dict)
               and isinstance(r["checker"].get("settings"), dict) and r["checker"]["settings"].get("mode") == "generic"]
    verified = set()
    for row in rows:
        if row.source not in GENERIC_ON_CURATED:
            continue
        arm = row.source.partition("/")[2]
        found = [r for r in records if (r["checker"]["settings"].get("arm") or {}).get("name") == arm
                 and row.run_id in (r.get("run_id"), r["checker"].get("branched_from"))
                 and (r.get("question_id"), r.get("model"), (r.get("store") or {}).get("name"))
                 == (row.question_id, row.model, row.graph)]
        where = f"ledger row {row.run_id} ({row.source})"
        if len(found) != 1:
            raise InputError(f"{where}: {'no record' if not found else f'{len(found)} records'} of its generic cell "
                             f"(arm {arm!r}, its question and model) in the analysed batches, so what it decided on "
                             f"cannot be verified")
        record = found[0]
        settings = record["checker"]["settings"]
        where = f"{where}: its generic cell's record {record.get('run_id')}"
        if settings.get("version") not in CURATED_GENERIC_VERSIONS:
            raise InputError(f"{where} decided under {settings.get('version')}, not {' or '.join(CURATED_GENERIC_VERSIONS)}"
                             f" (on its whole arm, or under an unknown rule)")
        curated = {s.get("name") for s in (settings.get("arm") or {}).get("sources") or [] if s.get("kind") == CURATED_KIND}
        if not curated:
            raise InputError(f"{where}: its arm holds no curated source, so no curated flag decided it")
        decided = {rep.get("source") for rep in (record["checker"].get("initial") or {}).get("reports") or []}
        if not decided <= curated:
            raise InputError(f"{where}: its decision read {sorted(decided - curated, key=str)}, not only its arm's curated "
                             f"sources {sorted(curated, key=str)}")
        verified.add((row.run_id, row.source))
    return verified


def ledger_source_names(rows: Iterable[CheckedOutput], prereg: PreregConfig | None = None,
                        curated_generic: Iterable[tuple[str, str]] = ()) -> list[CheckedOutput]:
    """The ledger's rows with the checker loop's source names mapped to the analysis's (``LEDGER_SOURCES``), so that the
    registered arms of C2, C4 and C5 find their rows. A GENERIC_ON_CURATED row is mapped only when ``curated_generic``
    names it (curated_generic_rows: its record shows its generic cell decided on its arm's curated sources). Refused:
    rows holding both a name and the name it maps to, or two names mapped to one arm (two sources would merge into one
    arm), a GENERIC_ON_CURATED row not so verified, and a config whose census, curated or generic source is not the
    mapping's (the mapping and the config would name different arms)."""
    rows = list(rows)
    if prereg is not None:
        expected = {"census-t5": prereg.census_source, f"generic_retry/{prereg.curated_source}": prereg.generic_source,
                    **{name: prereg.generic_source for name in GENERIC_ON_CURATED}}
        if expected != LEDGER_SOURCES:
            raise PreregError(f"the config's sources {expected} are not the ledger mapping's {LEDGER_SOURCES}")
    present = {r.source for r in rows}
    both = sorted(f"{a} and {b}" for a, b in LEDGER_SOURCES.items() if a in present and b in present)
    both += sorted(f"{a} and {b}" for a in sorted(present & set(LEDGER_SOURCES)) for b in sorted(present & set(LEDGER_SOURCES))
                   if a < b and LEDGER_SOURCES[a] == LEDGER_SOURCES[b])
    if both:
        raise InputError(f"the ledger holds both {', '.join(both)}: mapping the checker loop's names would merge two "
                         f"sources into one arm")
    verified = set(curated_generic)
    unverified = sorted(r.run_id for r in rows if r.source in GENERIC_ON_CURATED and (r.run_id, r.source) not in verified)
    if unverified:
        raise InputError(f"the ledger's {', '.join(GENERIC_ON_CURATED)} row {unverified[0]} (and {len(unverified) - 1} "
                         f"more) is mapped to the generic arm only once its record shows its generic cell decided on its "
                         f"arm's curated sources alone (curated_generic_rows; spec trace B10, checker-loop/2)")
    return [dataclasses.replace(r, source=LEDGER_SOURCES[r.source]) if r.source in LEDGER_SOURCES else r for r in rows]


def _checked_query(u: Unit) -> str:
    if u.checked is not None and u.checked.checked_query != UNRECORDED:
        return u.checked.checked_query
    return u.attempt.alignment


def _flag_kind(u: Unit) -> str:
    if u.checked is None:
        return "no output"
    if u.checked.branch_local_only:
        return "branch-local only"
    return "other flag" if u.checked.status == "F" else "not flagged"


def _bank_item(u: Unit) -> str:
    c = u.checked
    if c is None or c.bank_item is None:
        return UNRECORDED
    return c.fault_class if c.bank_item == "answer_changing" else c.bank_item


# Strata reported beside the totals (design log AC3; prereg §6 "Strata"); a dimension whose units
# are all "unrecorded" is left out.
STRATA: dict[str, Callable[[Unit], str]] = {
    "template": lambda u: u.attempt.template,
    "tier": lambda u: u.attempt.tier,
    "engine_sensitive": lambda u: u.attempt.engine_sensitive,
    "nested_graph": lambda u: u.attempt.nested_graph,
    "checked_query": _checked_query,
    "sib": lambda u: u.attempt.sib_stratum,
    "gold_empty": lambda u: u.attempt.gold_empty,
    "split": lambda u: u.attempt.split,
}
CHECKER_STRATA: dict[str, Callable[[Unit], str]] = {"branch_local": _flag_kind}
BANK_STRATA: dict[str, Callable[[Unit], str]] = {"template": STRATA["template"], "tier": STRATA["tier"],
                                                  "bank_item": _bank_item}


def _summary(vec: Mapping[str, np.ndarray], num: np.ndarray, den: np.ndarray, K: np.ndarray, level: float,
             events: tuple[str, str] | None = None) -> dict[str, Any]:
    est = point_estimate(num, den)
    out: dict[str, Any] = {"estimate": est, "ci": None, "undefined_share": None}
    if est is not None:
        iv = interval(ratio_draws(K, num, den), level, "two")
        out["ci"] = [iv["lo"], iv["hi"]] if iv["lo"] is not None else None
        out["undefined_share"] = iv["undefined_share"]
    if events is not None:  # prereg §3: exact bounds for binary question-level events only
        k, n = int((vec[events[0]] > 0).sum()), int((vec[events[1]] > 0).sum())
        bounds = clopper_pearson(k, n, level)
        out["question_event"] = {"k": k, "n": n, "exact": list(bounds) if bounds else None,
                                 "event": f"questions with at least one {events[0]} among those with a {events[1]}"}
    return out


def _metrics(units: Sequence[Unit], names: Sequence[str], index: QuestionIndex, K: np.ndarray,
             level: float) -> dict[str, Any]:
    vec = question_vectors(units, index)
    return {name: _summary(vec, *metric_of(vec, name), K, level, QUESTION_EVENTS.get(name)) for name in names}


def _by_check_type(units: Sequence[Unit], index: QuestionIndex, K: np.ndarray, level: float) -> dict[str, Any]:
    types = sorted({t for u in units if u.checked for t in u.checked.flagged_check_types})
    out = {}
    for t in types:
        rows = [Unit(u.cluster, u.attempt, u.checked,
                     {"F_t": float(bool(u.checked and t in u.checked.flagged_check_types)),
                      "W_F_t": float(bool(u.checked and t in u.checked.flagged_check_types) and u.ind["W"] > 0)})
                for u in units]
        vec = question_vectors(rows, index, names=("F_t", "W_F_t"))
        w, f = int(vec["W_F_t"].sum()), int(vec["F_t"].sum())
        out[t] = {"W_F": w, "C_F": f - w, "flag_precision": _summary(vec, *metric_arrays(vec, "W_F_t", "F_t", "ratio"),
                                                                     K, level)}
    return out


def _precision_at_prevalence(units: Sequence[Unit], index: QuestionIndex, K: np.ndarray,
                             level: float) -> dict[str, Any]:
    """Bank precision at a fixed prevalence pi (prereg §3): pi r / (pi r + (1 - pi) f), with
    recall r = W_F / W and false-flag rate f = C_F / C recomputed in every resample."""
    vec = question_vectors(units, index)
    r_est, f_est = point_estimate(vec["W_F"], vec["W"]), point_estimate(vec["C_F"], vec["C"])
    r, f = ratio_draws(K, vec["W_F"], vec["W"]), ratio_draws(K, vec["C_F"], vec["C"])
    out = {}
    for pi in PREVALENCES:
        with np.errstate(invalid="ignore", divide="ignore"):
            draws = pi * r / (pi * r + (1 - pi) * f)
        est = None
        if r_est is not None and f_est is not None and pi * r_est + (1 - pi) * f_est > 0:
            est = pi * r_est / (pi * r_est + (1 - pi) * f_est)
        iv = interval(draws, level, "two")
        out[f"{pi:g}"] = {"estimate": est, "ci": [iv["lo"], iv["hi"]] if est is not None and iv["lo"] is not None
                          else None, "undefined_share": iv["undefined_share"]}
    return out


def _dist(values: Iterable[float | int | None]) -> dict[str, Any]:
    xs = [float(x) for x in values if x is not None]
    if not xs:
        return {"n": 0, "mean": None, "median": None, "total": None}
    return {"n": len(xs), "mean": statistics.fmean(xs), "median": statistics.median(xs), "total": sum(xs)}


def _costs(units: Sequence[Unit], checker: bool) -> dict[str, Any]:
    present = [u for u in units if u.attempt.status != "missing"]
    out = {
        "agent_usd": _dist(u.attempt.cost_usd for u in present),
        "wall_s": _dist(u.attempt.wall_s for u in present),
        "input_tokens": _dist(u.attempt.input_tokens for u in present),
        "output_tokens": _dist(u.attempt.output_tokens for u in present),
        "tool_calls": _dist(u.attempt.tool_calls for u in present),
    }
    if checker:
        checked = [u.checked for u in units if u.checked is not None]
        out.update({
            "check_s": _dist(c.check_s for c in checked),
            "evidence_triples": _dist(c.evidence_triples for c in checked),
            "retry_usd": _dist(c.retry_usd for c in checked if c.retried),
        })
    return out


def _versions(units: Sequence[Unit]) -> dict[str, int]:
    return dict(sorted(Counter(u.attempt.rules_version or UNRECORDED for u in units if u.attempt.output).items()))


def _arm_result(key: ArmKey, units: Sequence[Unit], index: QuestionIndex, K: np.ndarray, level: float,
                side: Mapping[str, int], strata: bool) -> dict[str, Any]:
    bank = key.cohort == BANK
    checker = key.use != PROMPT and not bank
    names = BANK_METRICS if bank else CHECKER_METRICS if checker else PROMPT_METRICS
    result: dict[str, Any] = {
        "key": key.to_json(),
        "counts": {**arm_counts(units), **({} if bank else {k: int(side.get(k, 0)) for k in SIDE_COUNTS})},
        "rules_versions": {} if bank else _versions(units),
        "metrics": _metrics(units, names, index, K, level),
    }
    if not bank:
        result["costs"] = _costs(units, checker)
    if checker or bank:
        result["by_check_type"] = _by_check_type(units, index, K, level)
    if bank:
        result["precision_at_prevalence"] = _precision_at_prevalence(units, index, K, level)
    if strata:
        dims = BANK_STRATA if bank else {**STRATA, **(CHECKER_STRATA if checker else {})}
        result["strata"] = {}
        for dim, label_of in dims.items():
            levels: dict[str, list[Unit]] = {}
            for u in units:
                levels.setdefault(label_of(u), []).append(u)
            if set(levels) == {UNRECORDED}:
                continue
            result["strata"][dim] = {
                name: {"counts": {k: v for k, v in arm_counts(us).items() if k in ("N_att", "N_out", "n_questions")},
                       "metrics": _metrics(us, names, index, K, level)}
                for name, us in sorted(levels.items())}
    return result


# --- the ledger on real outputs beside the bank (prereg §7; change-log entries 85 and 86): descriptive ------------------

REFERENCE_SOURCE = "empty-retry"  # retry on empty, the reference row (feat/baselines: checkloop/matrix.py EMPTY_RETRY_ARM)
# per attempt on real outputs: each a proportion over N_att under prereg §3's weighting
LEDGER_PER_ATTEMPT = {"catches": "W_F", "false_flags": "C_F", "fixes": "fixed", "corruptions": "corrupted"}
RESULT_STRATA = ("returns_rows", "empty", "no_row_count")
LEDGER_DEFINITIONS = {
    "status": "descriptive (prereg §7, change-log entries 85 and 86): it frames the bank claims and is never confirmatory",
    "rows": "one per graph x cohort x use x source, its arms' units pooled over models, then prereg §3's weighting; "
            "'by_model' gives the same row per model (the appendix table, no predictions)",
    "per_attempt": "catches W_F, false flags C_F, fixes (wrong, flagged, right after the retry) and corruptions (right, "
                   "flagged, wrong after it), each over N_att: a question's mean over its attempts, then over questions",
    "not_flagged": "only status F is a flag: an unknown (U) and a vacuous pass count as not flagged",
    "unknown_rate": "U / N_out, pooled counts",
    "non_empty": "outputs (items) whose checked query returned rows (result_rows > 0, feat/baselines' 'returns_rows' "
                 "stratum): catches and false flags on them over N_att; on the bank, recall and false-flag rate over the "
                 "non-empty items; not estimable when no row count is recorded",
    "bank": "the same source's bank arm: recall W_F / W, false-flag rate C_F / C, unknown rate U / N_out, pooled counts",
    "reference": f"retry on empty ({REFERENCE_SOURCE!r}): it flags exactly the outputs whose result had no rows",
}


def result_stratum(c: CheckedOutput | None) -> str:
    """Whether the checked query returned rows, as feat/baselines' result_stratum reads it: "returns_rows", "empty",
    or "no_row_count" (not recorded)."""
    n = None if c is None else c.result_rows
    return "no_row_count" if n is None else "empty" if n == 0 else "returns_rows"


def _non_empty_units(units: Sequence[Unit]) -> list[Unit]:
    """The units with the non-empty indicators: each of W_F, C_F, W, C on an output whose result returned rows."""
    out = []
    for u in units:
        ne = float(u.attempt.output and result_stratum(u.checked) == "returns_rows")
        out.append(Unit(u.cluster, u.attempt, u.checked,
                        {**u.ind, **{f"{k}_ne": u.ind[k] * ne for k in ("W_F", "C_F", "W", "C")}}))
    return out


_LEDGER_NAMES = ("att", "out", "U", "W", "C", "W_F", "C_F", "fixed", "corrupted", "W_F_ne", "C_F_ne", "W_ne", "C_ne")


def _strata(units: Sequence[Unit]) -> dict[str, int]:
    found = Counter(result_stratum(u.checked) for u in units if u.attempt.output)
    return {s: found.get(s, 0) for s in RESULT_STRATA}


def _real_output_row(units: Sequence[Unit], index: QuestionIndex, K: np.ndarray, level: float) -> dict[str, Any]:
    vec = question_vectors(_non_empty_units(units), index, names=_LEDGER_NAMES)
    n = arm_counts(units)
    out: dict[str, Any] = {
        "counts": {"N_att": n["N_att"], "N_out": n["N_out"], "catches": n["W_F"], "false_flags": n["C_F"],
                   "fixes": n["fixed"], "corruptions": n["corrupted"], "unknown": n["W_U"] + n["C_U"],
                   "vacuous_passes": n["P_vacuous"], "retried": n["retried"]},
        "per_attempt": {name: _summary(vec, *metric_arrays(vec, ind, "att", PROPORTION), K, level)
                        for name, ind in LEDGER_PER_ATTEMPT.items()},
        "unknown_rate": _summary(vec, *metric_of(vec, "unknown_rate"), K, level),
    }
    outputs = _strata(units)
    out["non_empty"] = {"outputs": outputs, "estimable": outputs["no_row_count"] < n["N_out"]}
    if out["non_empty"]["estimable"]:
        out["non_empty"].update({name: _summary(vec, *metric_arrays(vec, f"{ind}_ne", "att", PROPORTION), K, level)
                                 for name, ind in (("catches", "W_F"), ("false_flags", "C_F"))})
    else:
        out["non_empty"]["reason"] = "no output's result row count is recorded (result_rows)"
    return out


def _bank_rates(key: ArmKey, units: Sequence[Unit], index: QuestionIndex, K: np.ndarray, level: float) -> dict[str, Any]:
    vec = question_vectors(_non_empty_units(units), index, names=_LEDGER_NAMES)
    n = arm_counts(units)
    items = _strata(units)
    out: dict[str, Any] = {
        "arm": key.label,
        "counts": {"items": n["N_out"], "W": n["W_F"] + n["W_P"] + n["W_U"], "W_F": n["W_F"],
                   "C": n["C_F"] + n["C_P"] + n["C_U"], "C_F": n["C_F"], "unknown": n["W_U"] + n["C_U"]},
        **{name: _summary(vec, *metric_of(vec, name), K, level) for name in ("recall", "false_flag_rate", "unknown_rate")},
        "non_empty": {"items": items, "estimable": items["no_row_count"] < n["N_out"]},
    }
    if out["non_empty"]["estimable"]:
        out["non_empty"].update({"recall": _summary(vec, *metric_arrays(vec, "W_F_ne", "W_ne", RATIO), K, level),
                                 "false_flag_rate": _summary(vec, *metric_arrays(vec, "C_F_ne", "C_ne", RATIO), K,
                                                             level)})
    else:
        out["non_empty"]["reason"] = "no item's result row count is recorded (result_rows)"
    return out


def real_output_ledger(arms: Mapping[ArmKey, Sequence[Unit]], index: QuestionIndex, K: np.ndarray, level: float,
                       bank_index: QuestionIndex | None = None, K_bank: np.ndarray | None = None) -> dict[str, Any]:
    """The ledger on real outputs beside the bank (prereg §7, change-log entries 85 and 86; LEDGER_DEFINITIONS), on
    the per-arm resample matrices: per graph x cohort x use x source of the checker arms (models pooled), its counts,
    catches, false flags, fixes and corruptions per attempt, unknown rate, non-empty column and the source's bank rates;
    the same per model ('by_model', the appendix); retry on empty (REFERENCE_SOURCE) marked as the reference row, or
    its absence stated. Descriptive only: nothing here enters the confirmatory analysis."""
    pooled: dict[tuple[str, str, str, str], list[Unit]] = {}
    by_model: dict[tuple[str, str, str, str, str], Sequence[Unit]] = {}
    banks: dict[tuple[str, str], list[ArmKey]] = {}
    for key, units in sorted(arms.items()):
        if key.cohort == BANK:
            banks.setdefault((key.graph, key.source), []).append(key)
        elif key.use != PROMPT:
            pooled.setdefault((key.graph, key.cohort, key.use, key.source), []).extend(units)
            by_model[(key.graph, key.model, key.cohort, key.use, key.source)] = units
    bank_rates: dict[tuple[str, str], dict[str, Any] | None] = {}
    for (graph, source), keys in banks.items():
        if len(keys) > 1:
            raise InputError(f"two bank arms of source {source} on {graph}: {[k.label for k in keys]}")
        bank_rates[(graph, source)] = _bank_rates(keys[0], arms[keys[0]], index if bank_index is None else bank_index,
                                                  K if K_bank is None else K_bank, level)
    rows = [{"graph": g, "cohort": c, "use": use, "source": s, "reference": s == REFERENCE_SOURCE,
             **_real_output_row(units, index, K, level), "bank": bank_rates.get((g, s))}
            for (g, c, use, s), units in sorted(pooled.items())]
    present = any(r["reference"] for r in rows)
    return {
        "descriptive": True, "definitions": LEDGER_DEFINITIONS, "reference_source": REFERENCE_SOURCE,
        "reference_present": present,
        "reference_note": (f"retry on empty ({REFERENCE_SOURCE!r}) is the reference row" if present else
                           f"no ledger rows of the reference source {REFERENCE_SOURCE!r} (retry on empty) on real "
                           f"outputs: no ledger writer exports a retry-on-empty cell yet (merge-order §4)"),
        "rows": rows,
        "by_model": [{"graph": g, "model": m, "cohort": c, "use": use, "source": s, "reference": s == REFERENCE_SOURCE,
                      **_real_output_row(units, index, K, level)}
                     for (g, m, c, use, s), units in sorted(by_model.items())],
    }


def index_from_units(arms: Mapping[ArmKey, Sequence[Unit]]) -> QuestionIndex:
    """Prereg §4 "Resampling": the base questions present, stratified by template x tier (for
    analyses without a
    registered population, e.g. tests)."""
    return QuestionIndex.from_strata({u.cluster: f"{u.attempt.template}|{u.attempt.tier}"
                                      for units in arms.values() for u in units})


def require_refusal_rule(records: Iterable[dict], rule: str, where: Path | str) -> None:
    """A confirmatory run's records were all run under the frozen refusal rule (``refusal_rule``, design log, entry 91):
    each record's runner setting (agent_config.refusal_rule; "failed", no refusal rule, before agent-run/17) is the
    rule's (inputs.REFUSAL_SETTINGS), and each refused outcome names the rule and a provider or phrase signal; any other
    is refused. A test run and an exploratory analysis accept every record (their refusal report counts the settings)."""
    want = REFUSAL_SETTINGS[rule]
    for r in records:
        setting = refusal_setting(r)
        if setting != want:
            raise PreregError(f"{where}: record {r['run_id']} ran under the runner's refusal setting {setting!r}, not "
                              f"{want!r}, the frozen refusal rule {rule}'s (a record before agent-run/17 ran under "
                              f"none): a confirmatory run refuses it (design log)")
        refusal = r["outcome"].get("refusal") or {}
        if r["outcome"]["status"] == "refused" and (refusal.get("rule") != rule or not str(
                refusal.get("signal") or "").startswith(REFUSAL_SIGNALS)):
            raise PreregError(f"{where}: record {r['run_id']} was refused under {refusal.get('rule')!r} with signal "
                              f"{refusal.get('signal')!r}, not under the frozen refusal rule {rule} with a provider or "
                              f"phrase signal")


def refusal_report(attempts: Sequence[Attempt], prereg: PreregConfig | None,
                   messages: Mapping[str, Mapping[str, str | None]] | None) -> dict[str, Any]:
    """The refusals (prereg §2 "Refusals", §6 "refusal-excluded"; change-log entries 34 and 91), descriptive and in no
    registry. A refused attempt is scored wrong in the primary analysis; here they are counted per arm, the secondary
    analysis's exclusion is named (per contrast, the base questions refused in any arm it compares), and beside it the
    count of each refusal signal (provider:<stop reason>, phrase:<name>) and every refused attempt's final message
    (``messages``, by run id: read only from the run's own split's records, so a test run's only from dev; None when no
    records were read)."""
    refused = sorted((a for a in attempts if a.status == "refused"),
                     key=lambda a: (a.graph, a.model, a.cohort, a.question_id, a.repetition, a.run_id))
    out: dict[str, Any] = {
        "descriptive": True, "rule": prereg.refusal_rule if prereg is not None else None,
        "reading": "refused attempts are scored wrong in every arm (prereg §2); the secondary analysis excludes any "
                   "base question refused in any compared arm",
        "refused": len(refused),
        "per_arm": dict(sorted(Counter(f"{a.graph}|{a.model}|{a.cohort}" for a in refused).items())),
        "signals": dict(sorted(Counter(a.refusal_signal for a in refused).items())),
        "final_messages_read": messages is not None,
        "attempts": [{"run_id": a.run_id, "graph": a.graph, "model": a.model, "cohort": a.cohort,
                      "question_id": a.question_id, "base_question_id": a.base_question_id,
                      "repetition": a.repetition, "signal": a.refusal_signal,
                      "final_message": None if messages is None else messages.get(a.run_id)} for a in refused]}
    if prereg is not None:
        out["secondary_excludes"] = {cid: sorted({a.base_question_id for a in refused if any(
            (k.graph, k.model, k.cohort) == (a.graph, a.model, a.cohort) for k in keys)})
            for cid, keys in compared_arms(prereg).items() if keys}
    return out


def analyse(attempts: Sequence[Attempt], ledger: Iterable[CheckedOutput] = (), *,
            questions: Mapping[str, dict] | None = None, index: QuestionIndex | None = None,
            prereg: PreregConfig | None = None, seed: int = DEFAULT_SEED, n_boot: int = DEFAULT_N_BOOT,
            level: float = DEFAULT_LEVEL, side_counts: Mapping[tuple[str, str, str], Mapping[str, int]] | None = None,
            strata: bool = True, settings: dict[str, Any] | None = None,
            multiplicity: Any = None, bank_index: QuestionIndex | None = None,
            clusters: ClusterMap | None = None, testing: bool = False, preflighted: bool = False,
            refusal_messages: Mapping[str, Mapping[str, str | None]] | None = None,
            scorer_changes: Mapping[str, Any] | None = None,
            v4_verdicts: Mapping[str, tuple[bool, float]] | None = None,
            model_settings: Mapping[str, Any] | None = None,
            output_cap_hits: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Every arm's counts, metrics and strata on one stratified resample (prereg §3, §4, §6;
    exploratory), the ledger on real outputs beside the bank (prereg §7, descriptive: real_output_ledger), plus the
    registered analysis when ``prereg`` is given (prereg §1-§4). The registered
    analysis reads the cluster map and the weighted Holm's tables only from the config's frozen files, and
    takes the config as loaded from its file; a ``clusters`` map or the tables (``multiplicity``) are taken
    only in a test run (``testing``), which the report marks as not confirmatory (prereg.preflight). With
    ``prereg``, the registered preflight runs first, before any attempt is read (``preflighted``: the caller,
    analyse_batches, ran it before reading the records); confirmatory runs it again for the inference it binds. The
    ledger's source names are the config's (``ledger_source_names``), and beside the family the leak exclusion's
    descriptive C1 report (leak.report: prereg §5, entry 91) is computed from the config's frozen prompts and the
    question records' gold, which a confirmatory run requires. The refusals are reported beside the arms
    (``refusal_report``, with the refused attempts' final messages when the caller read them: ``refusal_messages``).
    The verdicts the frozen scorer changes against scorer-rules-v4 (``scorer_changes``, a ScorerChanges table the
    caller computed from the records, as analyse_batches does; None when it was not computed) are reported per arm, and
    beside the family for C1's prompted arms (scorer_changes.c1_table), descriptive. With ``v4_verdicts`` (run id ->
    (exact, F1) under scorer-rules-v4: ScorerChanges.v4_verdicts, the same rescoring) and ``prereg``, C1a and C1b are
    also reported with each output's v4 verdict, a descriptive sensitivity beside the family (change-log entry 129
    item 5); an output without one (never rescored) keeps its stored verdict, counted. ``model_settings`` (a
    model_settings.ModelSettings table the caller computed from the records, as analyse_batches does) is reported
    beside the arms: how often each per-model harness setting fired per arm (change-log entry 129 item 4),
    descriptive."""
    if prereg is not None and not preflighted:
        preflight(prereg, ROOT, clusters=clusters, multiplicity=multiplicity, testing=testing)
    # the leak exclusion's frozen prompts (prereg §5, entry 91), by their sha256: a confirmatory run refuses without them
    prompts = leak.frozen_prompts(prereg, ROOT, required=not testing) if prereg is not None else None
    if prereg is not None and not testing and questions is None:
        raise PreregError("the leak exclusion (prereg §5, entry 91) is computed from the question records' gold: a "
                          "confirmatory run needs them")
    if prereg is not None and testing:  # design log ruling: a test run never reads the registered split
        check_test_run_split(a.split for a in attempts)
    named = ledger_source_names(ledger, prereg)
    arms = build_arms(attempts, named, questions)
    index = index or index_from_units(arms)
    K = resample_matrix(index, n_boot, seed=seed)
    K_bank = K if bank_index is None else resample_matrix(bank_index, n_boot, seed=seed)
    side_counts = side_counts or {}
    result: dict[str, Any] = {
        "schema": RESULTS_SCHEMA,
        "settings": {
            "seed": seed, "n_boot": n_boot, "level": level, "n_questions": len(index),
            "strata": dict(sorted(Counter(index.strata).items())),
            "resampling": "complete question vectors, stratified by template x tier, one matrix for every arm "
                          "(prereg §4)",
            "weighting": "proportions over N_att averaged per question then over questions; ratios as pooled "
                         "counts (prereg §3)",
            "interval": "percentile over defined resamples, inverted-CDF quantiles (prereg §4)",
            "exact_bounds": "Clopper-Pearson for binary question-level events only (prereg §3)",
            "status": "per-arm numbers are exploratory (prereg §6); the confirmatory analysis is under 'confirmatory' "
                      "(its own unstratified resampling, prereg §4.2)",
            **(settings or {}),
        },
        "arms": {key.label: _arm_result(key, units, *((bank_index, K_bank) if bank_index is not None and key.cohort == BANK
                                                      else (index, K)), level,
                                        side_counts.get((key.graph, key.model, key.cohort), {}), strata)
                 for key, units in sorted(arms.items())},
        "real_output_ledger": real_output_ledger(arms, index, K, level, bank_index, K_bank),
        "refusals": refusal_report(attempts, prereg, refusal_messages),
        "scorer_changes": scorer_changes,
        "model_settings": model_settings,
        "confirmatory": None,
    }
    if output_cap_hits is not None:  # entry 161: the output-cap hits, labelled from the records (inputs.OUTPUT_CAP_RULE)
        result["output_cap_hits"] = output_cap_hits
    result["settings"]["code_digest"] = code_digest()  # plan D: one code hash binds every results file
    if prereg is not None:
        # the registered resampling is unstratified (prereg §4.2), never the per-arm matrix above
        result["confirmatory"] = confirmatory(arms, index, prereg, multiplicity=multiplicity,
                                              code_digest=result["settings"]["code_digest"], bank_index=bank_index,
                                              clusters=clusters, simulated=calibration_simulation(prereg, ROOT),
                                              root=ROOT, testing=testing)
        if v4_verdicts is not None:  # entry 129 item 5: the same attempts, each output's verdict under v4
            v4_attempts = [dataclasses.replace(a, correct=v4_verdicts[a.run_id][0], f1=v4_verdicts[a.run_id][1])
                           if a.output and a.run_id in v4_verdicts else a for a in attempts]
            outputs = sum(a.output for a in attempts)
            rescored = sum(a.output and a.run_id in v4_verdicts for a in attempts)
            # only the prompt arms take the v4 verdicts (C1 reads no other): a checker row keeps the frozen scorer's
            # initial_correct, so the checker and bank arms stay the primary's (External review of d02717f)
            v4_arms = {**arms, **{k: u for k, u in build_arms(v4_attempts, (), questions).items() if k.use == PROMPT}}
            result["confirmatory"]["descriptive"][C1_UNDER_V4] = {
                **c1_under_v4(v4_arms, index, prereg, bank_index, root=ROOT, clusters=clusters,
                              multiplicity=multiplicity, testing=testing),
                "verdicts": {"outputs": outputs, "v4_verdict": rescored, "stored_verdict_kept": outputs - rescored}}
        # descriptive, beside the family (entry 91): C1 without the questions whose gold value one compared prompt holds
        result["confirmatory"]["descriptive"]["C1 leak exclusion"] = leak.report(
            arms, index, prereg, questions, prompts, load_clusters(prereg, ROOT) if clusters is None else clusters)
        # descriptive, beside C2 (prereg §6's source-layer ablations; change-log entry 108): one table of the bank arms
        result["confirmatory"]["descriptive"]["C2 source-layer ablations"] = ablation_table(
            arms, index if bank_index is None else bank_index, prereg)
        if scorer_changes is not None:  # descriptive, beside C1 (change-log entries 120 and 127)
            result["confirmatory"]["descriptive"]["C1 verdicts changed by the frozen scorer against v4"] = \
                scorer_changes_c1(scorer_changes, prereg)
    return result


def calibration_simulation(prereg: PreregConfig, root: Path) -> dict[str, Any]:
    """What the frozen calibration table says it simulated (its bank structure and sizes), reported beside the real
    cluster counts (entry 52); empty while the calibration is [FILL]. Descriptive: the confirmatory run itself reads
    the table by its frozen hash and refuses, producing nothing, when the file is not the frozen one; so this reads
    it only when it is (empty otherwise)."""
    path = Path(root) / prereg.calibration if prereg.calibration not in (PLACEHOLDER, "", None) else None
    if path is None or not path.is_file():
        return {}
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != prereg.calibration_sha256:
        return {}
    data = json.loads(raw)
    return {k: data.get(k) for k in ("structure", "per_template", "N1", "N2")}


DEFAULT_N_REPS = 3  # prereg §4 "Fixed stopping", for analyses without a registered config
# question fields the analysis reads (strata, the design, the population): they must agree across the
# question files of different batches; other fields (guard metadata, change-log entry 15) may differ
QUESTION_FIELDS = ("base_question_id", "template", "tier", "split", "engine_sensitive", "sib_stratum", "gold_row_count")


def design_repetitions(n_reps: int | None, prereg: PreregConfig | None) -> int:
    """The design's repetitions per question (prereg §4 "Fixed stopping"): the frozen config's, and a
    different value given by the caller is refused; without a config, the caller's or 3."""
    if prereg is None:
        return DEFAULT_N_REPS if n_reps is None else n_reps
    if n_reps is not None and n_reps != prereg.n_reps:
        raise PreregError(f"n_reps {n_reps} differs from the frozen {prereg.n_reps} (prereg §4 'Fixed stopping')")
    return prereg.n_reps


def branch_arm(record: Mapping[str, Any]) -> str:
    """A checker branch-cell record's arm for the per-model settings table: its store, model and branch arm (an
    unreadable store reads "unknown", so the record is still counted)."""
    try:
        graph = store_name(record)
    except InputError:
        graph = "unknown"
    return f"{graph}|{record.get('model')}|branch:{(record.get('load') or {}).get('arm')}"


def merge_questions(seen: dict[str, dict], batch_questions: Mapping[str, dict], source: str) -> dict[str, dict]:
    """Add one batch's questions to those already read, refusing a question whose analysed fields
    (``QUESTION_FIELDS``) differ from an earlier file's, since the bootstrap would use one stratum
    for both batches."""
    for qid, q in batch_questions.items():
        if qid in seen:
            for f in QUESTION_FIELDS:
                if seen[qid].get(f) != q.get(f):
                    raise InputError(f"question {qid}: {f} is {seen[qid].get(f)!r} in an earlier questions file but "
                                     f"{q.get(f)!r} in {source}")
        else:
            seen[qid] = q
    return seen


def analyse_batches(batches: Sequence[Path], questions: Sequence[Path], *, split: str, n_reps: int | None = None,
                    ledgers: Sequence[Path] = (), prereg: PreregConfig | None = None,
                    bank_manifest: Path | None = None, checked_call: CheckedCallFn | None = None,
                    tier_if_missing: str | None = None, bank_questions: Sequence[str] | None = None,
                    **kw: Any) -> dict[str, Any]:
    """Gate, load and analyse run-record batches on the registered design: every batch needs a
    passing, current replay check and the questions file it ran on (by sha256 against its
    manifest), and the questions files must agree on every analysed field; the design is every
    question of ``split`` with the frozen config's repetitions per cohort (``design_repetitions``),
    and a registered attempt with no record is missing (prereg §4). With a registered config the
    fault bank's rows must equal the frozen bank manifest (``bank_manifest``, checked against the
    config's ``bank_manifest_sha256``; inputs.check_bank_manifest). The registered preflight runs first, before any
    record is read (the review of 93d4aca: the digests, the single-threaded BLAS and the known answers refuse a run
    before it touches the data).

    ``bank_questions`` declares a subset of the split's bank questions (a dress rehearsal's slice): accepted in a test
    run only, where the bank rows must then be exactly those questions' items and the subset is recorded in the
    results (``settings.bank.question_subset``); a confirmatory run refuses it, reading nothing, since the registered
    bank metrics are over every item of the split."""
    n_reps = design_repetitions(n_reps, prereg)
    config = None
    if bank_questions is not None and not (prereg is not None and kw.get("testing", False)):
        raise PreregError("a bank question subset is accepted in a test run only: a confirmatory run checks every item "
                          "of the frozen bank's split")
    frozen = prereg.frozen_questions() if prereg is not None else None  # refused while [FILL]
    if prereg is not None:
        if prereg.bank_manifest_sha256 in (PLACEHOLDER, "", None):
            raise PreregError("the frozen bank manifest sha256 is not filled in (prereg §5 'Fault bank' [FILL])")
        if bank_manifest is None:
            raise PreregError("the registered analysis needs the frozen fault-bank manifest (prereg §5 'Fault bank')")
        # reading nothing either: a confirmatory run's config is the frozen one, by its file's sha256 (design ruling,
        # design log, 20 Oct); a test run records both values
        config = frozen_config(prereg, testing=kw.get("testing", False))
        # the config's own refusals above read nothing, nor does a test run's split check (design log ruling: it never
        # reads the registered split); the preflight next, before the first record is read
        if kw.get("testing", False):
            check_test_run_split([split])
        preflight(prereg, ROOT, clusters=kw.get("clusters"), multiplicity=kw.get("multiplicity"),
                  testing=kw.get("testing", False))
        leak.frozen_prompts(prereg, ROOT, required=not kw.get("testing", False))  # entry 91's prompts, before any record
        kw = {**kw, "preflighted": True}
    testing = prereg is not None and kw.get("testing", False)
    for path in ledgers if testing else ():
        _test_run_ledger(path, split)
    attempts: list[Attempt] = []
    side: dict[tuple[str, str, str], Counter] = {}
    inputs, all_questions = [], {}
    settings_seen: Counter = Counter()  # the runner's refusal setting of each record in the split
    messages: dict[str, dict[str, str | None]] = {}  # each refused attempt's final message in the split, by run id
    cap_records: list[tuple[dict, str]] = []  # each in-split record with its arm, for the output-cap hits (entry 161)
    checker_records: list[dict] = []  # every record with a checker decision: a generic row's join is verified on them
    stack_inputs: list[tuple] = []  # each OUT's records and deployment identity: one stack per (model, graph)
    changes = ScorerChanges()  # the verdicts the frozen scorer changes against v4, per arm (entries 120 and 127)
    fired = ModelSettings()  # how often each per-model setting fired, per arm (entry 129 item 4)
    for path in batches:
        if testing:  # decided from file names and claims before any record is read, even hashed by the replay check
            _test_run_batch(path, questions, split)
        scorer = None
        if prereg is not None:  # from the manifest alone, before any record is hashed or read (spec trace C11)
            if prereg.scorer_digest in (PLACEHOLDER, "", None):
                raise PreregError("the frozen scorer digest is not filled in (prereg §3: the sha256 of the scorer's "
                                  "closed module set)")
            scorer = require_scorer(path, prereg.scorer_digest)
        check = require_replay_check(path)
        qpath = select_questions(path, questions)
        manifest_sha = require_questions_match(path, qpath)
        if frozen is not None and manifest_sha != frozen:
            raise PreregError(f"{path}: ran on questions {manifest_sha}, not the frozen question set {frozen} (ids are "
                              f"keyed on (id, questions_sha256))")
        batch_questions = load_questions(qpath)
        if tier_if_missing:
            for q in batch_questions.values():
                q.setdefault("tier", tier_if_missing)
        merge_questions(all_questions, batch_questions, str(qpath))
        batch = load_batch(path, grid_cells(path))  # a grid's records by their cells; a plain run's by condition
        identity_path = Path(path) / "deployment-identity.json"
        stack_inputs.append((path, [*batch.records, *batch.branch],
                             json.loads(identity_path.read_text(encoding="utf-8")) if identity_path.exists() else None))
        if prereg is not None and not testing:  # design log, entry 91: every record under the frozen refusal rule
            require_refusal_rule([*batch.records, *batch.branch], prereg.refusal_rule, path)
        rows = [dataclasses.replace(a, questions_sha256=manifest_sha)
                for a in attempts_from_batch(batch, batch_questions, checked_call=checked_call,
                                             tier_if_missing=tier_if_missing)]
        attempts.extend(a for a in rows if a.split == split)
        in_split = {a.run_id: a for a in rows if a.split == split}
        for r in batch.records:
            if r["run_id"] in in_split:
                a = in_split[r["run_id"]]
                changes.add(r, batch_questions.get(r["question_id"]), a)
                fired.add(r, f"{a.graph}|{a.model}|{a.cohort}")
        for r in batch.branch:  # a grid's checker branch cells: each its own arm, the analysed split only
            if (batch_questions.get(r.get("question_id")) or {}).get("split") == split:
                fired.add(r, branch_arm(r))
                cap_records.append((r, branch_arm(r)))  # a checker retry the output cap cut counts too (entry 161)
        settings_seen.update(refusal_setting(r) for r in batch.records if r["run_id"] in in_split)
        cap_records.extend((r, f"{in_split[r['run_id']].graph}|{r['model']}|{in_split[r['run_id']].cohort}")
                           for r in batch.records if r["run_id"] in in_split)
        messages.update({r["run_id"]: final_message(r) for r in batch.records
                         if r["run_id"] in in_split and in_split[r["run_id"]].status == "refused"})
        for key, counts in batch.side_counts.items():
            side.setdefault(key, Counter()).update(counts)
        checker_records.extend(r for r in (*batch.records, *batch.branch) if r.get("checker"))
        inputs.append({"batch": str(path), "records_sha256": check["records_sha256"], "n_records": check["n_records"],
                       "replay_check_code": check.get("code"), "questions_sha256": manifest_sha, "scorer_digest": scorer,
                       "records_outside_split": sum(1 for a in rows if a.split != split),
                       # a grid's branch-cell records: checker arms' (their rows come from the checker ledger)
                       "branch_records": dict(sorted(Counter(r["load"]["arm"] for r in batch.branch).items())),
                       "branch_side": {cell: dict(c) for cell, c in sorted(batch.branch_side.items())}})
    # change-log entry 144: one stack per (model, graph), never pooled across stacks; a confirmatory analysis also
    # needs every record's route base URL
    check_one_stack(stack_inputs, require_route=prereg is not None and not testing)
    design = {qid: q for qid, q in all_questions.items() if q.get("split") == split}
    attempts = complete_design(attempts, design, n_reps)
    ledger: list[CheckedOutput] = []
    for path in ledgers:  # the header decides alone; a /2 file's rows each checked for the split first (inputs)
        ledger.extend(load_checker_ledger(path, split, single_split=testing, registered=prereg is not None))
    # the checker loop's source names -> the config's, gen:any's generic rows once their records show a curated decision
    ledger = ledger_source_names(ledger, prereg, curated_generic_rows(ledger, checker_records))
    bank, bank_index = None, None
    if prereg is not None:
        check_questions_hash(ledger, prereg)
        bank_rows = [r for r in ledger if r.cohort == BANK and r.graph == prereg.graph]
        # the bank's confirmatory population is the manifest's split: the agent design's questions, plus any
        # bank-only extension questions (change-log entry 41); a test run's declared subset narrows it to those
        # questions (check_bank_manifest refuses a row outside it); bank_question_index decides its index
        split_questions = bank_split_questions(bank_manifest, split)
        checked = check_bank_manifest([r for r in bank_rows if r.question_id in split_questions], bank_manifest,
                                      prereg.bank_manifest_sha256, split, questions=bank_questions)
        population = (split_questions if bank_questions is None
                      else {q: s for q, s in split_questions.items() if q in set(bank_questions)})
        bank_index = bank_question_index(population, design)
        ledger = [r for r in ledger if r.cohort != BANK or (r.graph == prereg.graph and r.question_id in population)]
        bank = {"manifest": str(bank_manifest), "sha256": prereg.bank_manifest_sha256, **checked,
                "rows_outside_split": sum(1 for r in bank_rows if r.question_id not in split_questions),
                "bank_only_questions": len(set(population) - set(design))}
        if bank_questions is not None:
            bank["question_subset"]["ids"] = sorted(set(bank_questions))
    settings = {"gated": True, "inputs": inputs, "ledgers": [str(p) for p in ledgers], "split": split,
                "n_reps": n_reps, "tier_if_missing": tier_if_missing, "bank": bank, "frozen_config": config,
                "refusal_settings": dict(sorted(settings_seen.items()))}
    cap_arms = {r["run_id"]: arm for r, arm in cap_records}
    return analyse(attempts, ledger, questions=all_questions, index=index_from_questions(design, split), prereg=prereg,
                   side_counts={k: dict(v) for k, v in side.items()}, settings=settings, bank_index=bank_index,
                   refusal_messages=messages, scorer_changes=changes.table(), v4_verdicts=changes.v4_verdicts,
                   model_settings=fired.table(), output_cap_hits=output_cap_report(
                       [r for r, _ in cap_records], lambda r: cap_arms[r["run_id"]]), **kw)


# The frozen analysis config's pin (design ruling, design log, 20 Oct): the sha256 of the config file a confirmatory run
# must load, filled at the freeze; scripts/freeze_binding_check.py --bind-ref checks it against the file it names.
FROZEN_CONFIG_PIN = ROOT / "data" / "freeze" / "frozen-config.json"
FROZEN_CONFIG_SCHEMA = "frozen-config/1"


def frozen_config(prereg: PreregConfig, *, testing: bool, pin: Path | None = None) -> dict[str, Any]:
    """The config against the frozen one (``FROZEN_CONFIG_PIN``): what the run records, the pin's value and the loaded
    config's file and sha256. A confirmatory run refuses unless the config was loaded from a file whose sha256 is the
    pinned one; a [FILL] pin, an unreadable pin, or a config built in code is refused too. A test run is never refused
    here (it may load another config); the record says whether the config is the frozen one."""
    pin = FROZEN_CONFIG_PIN if pin is None else Path(pin)
    try:
        data = json.loads(pin.read_text(encoding="utf-8"))
        pinned = data["sha256"] if data.get("schema") == FROZEN_CONFIG_SCHEMA else None
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        data, pinned = None, None
    config_file, config_sha256 = prereg.loaded_from or (None, None)
    record = {"pin": str(pin), "frozen_sha256": pinned, "config_file": None if config_file is None else str(config_file),
              "config_sha256": config_sha256, "is_frozen": pinned not in (None, PLACEHOLDER) and pinned == config_sha256}
    if testing:
        return record
    if data is None or pinned is None:
        raise PreregError(f"{pin}: not a {FROZEN_CONFIG_SCHEMA} pin of the frozen config; a confirmatory run takes the "
                          f"config only by its frozen sha256")
    if pinned == PLACEHOLDER:
        raise PreregError(f"the frozen config's sha256 is not pinned ({pin} [FILL]): a confirmatory run refuses until "
                          f"the freeze fills it")
    if config_sha256 is None:
        raise PreregError("a config built in code is refused in a confirmatory run: it is taken from its file, by its "
                          "frozen sha256")
    if config_sha256 != pinned:
        raise PreregError(f"the config {config_file} has sha256 {config_sha256}, not the frozen config's {pinned} ({pin})")
    return record


# The runner's claim (scripts/run_pilot.py, _lease): an attempt's key and its lease, never an outcome; the runner since
# feat/e0.1-skeleton 1b7b7ec adds the attempt's split (CLAIM_SPLIT), earlier ones do not.
CLAIM_FIELDS = ("key", "pid", "host", "claimed_at", "heartbeat_at")
CLAIM_SPLIT = "split"
# The runner's attempt marker (feat/e0.1-skeleton 1b7b7ec, write_attempt_marker): claims/<claim digest>.attempt, its key
# and split, written before the attempt's first try and never deleted (an aborted attempt's claim is, its marker not).
MARKER_SCHEMA = "attempt-marker/1"
MARKER_SUFFIX = ".attempt"
MARKER_FIELDS = ("schema", "key", "split")
# A run file's name (the runner's _RUN_FILE_NAME): "<claim digest>.<run id>.json", a set-aside sidecar
# "<claim digest>.<run id>.setaside.json"; a run before attempt markers named its files "<run id>.json".
RUN_FILE_NAME = re.compile(r"(?P<digest>[0-9a-f]{40})\.(?P<run_id>[^./]+)(?:\.setaside)?\.json")
# checker-ledger schemas whose writer guarantees a file of one split, named in its header line (merge-order §4: the
# feat/checker-loop writer's checker-ledger/2); a checker-ledger/1 header proves nothing about its rows
SINGLE_SPLIT_LEDGERS = (SINGLE_SPLIT_LEDGER,)


def _number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


# The runner's attempt key (scripts/run_pilot.py attempt_key), which names an attempt's claim, marker and files: (question,
# route model, repetition) in a run without the grid, (arm, question, route model, repetition) in a grid run, whose OUT
# holds every cell. The batch's manifest decides which (gate.manifest_layout); a key of the other shape, or of neither,
# is refused. Every element is a name but the repetition, an integer.
ATTEMPT_KEYS = {PLAIN: ("question", "route model", "repetition"), GRID: ("arm", "question", "route model", "repetition")}


def _attempt_key(key: Any, layout: str) -> bool:
    return (isinstance(key, list) and len(key) == len(ATTEMPT_KEYS[layout])
            and all(isinstance(k, str) for k in key[:-1]) and isinstance(key[-1], int) and not isinstance(key[-1], bool))


def _key_refusal(where: str, layout: str) -> PreregError:
    return PreregError(f"{where}: its key is not ({', '.join(ATTEMPT_KEYS[layout])}), the {layout} run's attempt key "
                       f"its manifest calls for")


def key_question(key: tuple, layout: str) -> str:
    """The question of an attempt key of the layout."""
    return key[ATTEMPT_KEYS[layout].index("question")]


def _key_digest(key: list) -> str:
    """The runner's claim digest of an attempt key (claim_digest): it names the claim, the marker and every run file."""
    return hashlib.sha256(json.dumps(key, ensure_ascii=False).encode("utf-8")).hexdigest()[:40]


def _read_json_object(path: Path, where: str) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise PreregError(f"{where}: it is unreadable, so its split is undecided") from None
    if not isinstance(data, dict):
        raise PreregError(f"{where}: not a JSON object")
    return data


def read_claim(path: Path, layout: str) -> tuple[tuple[str | int, ...], str | None]:
    """One runner claim's attempt key (the layout's, ATTEMPT_KEYS: a plain run's (question, route model, repetition), a
    grid run's (arm, question, route model, repetition)) and split (None for a runner before 1b7b7ec), taken only from
    a claim that is exactly the runner's lease (scripts/run_pilot.py: _lease, claim_path): the fields CLAIM_FIELDS, plus
    CLAIM_SPLIT or not, no others, so a claim carrying an outcome (or any other field) is refused, never used; its key
    the layout's exactly, its lease's types the runner's, and its file name the runner's digest of its key. Anything
    else is PreregError, whose message names fields, never their values (the review of 2ff8f40, blocker 4)."""
    path = Path(path)
    where = f"a test run does not use the claim {path.parent.parent.name}/claims/{path.name}"
    data = _read_json_object(path, where)
    if set(data) not in (set(CLAIM_FIELDS), {*CLAIM_FIELDS, CLAIM_SPLIT}):
        allowed = {*CLAIM_FIELDS, CLAIM_SPLIT}
        raise PreregError(f"{where}: its fields are not the runner's claim schema {list(CLAIM_FIELDS)} (and "
                          f"{CLAIM_SPLIT!r}) (beyond it: {sorted(set(data) - allowed)}; missing: "
                          f"{sorted(set(CLAIM_FIELDS) - set(data))}); a claim holds an attempt's key and lease, never "
                          f"an outcome")
    if not _attempt_key(data["key"], layout):
        raise _key_refusal(where, layout)
    if not (isinstance(data["pid"], int) and not isinstance(data["pid"], bool) and isinstance(data["host"], str)
            and _number(data["claimed_at"]) and _number(data["heartbeat_at"])
            and isinstance(data.get(CLAIM_SPLIT, ""), str)):
        raise PreregError(f"{where}: its lease fields are not of the runner's types")
    if path.name != f"{_key_digest(data['key'])}.claim":  # claim_path: a claim is named by its key's digest
        raise PreregError(f"{where}: its name is not the runner's digest of its key")
    return tuple(data["key"]), data.get(CLAIM_SPLIT)


def read_marker(path: Path, layout: str) -> tuple[tuple[str | int, ...], str]:
    """One attempt marker's key and split, taken only from a marker that is exactly the runner's (MARKER_SCHEMA): the
    fields MARKER_FIELDS, no more and no fewer (a marker holds no outcome), its key the layout's exactly (ATTEMPT_KEYS:
    a plain run's (question, route model, repetition), a grid run's (arm, question, route model, repetition)), its
    split a split name, and its file name the runner's digest of its key. Anything else is PreregError naming fields,
    never their values."""
    path = Path(path)
    where = f"a test run does not use the marker {path.parent.parent.name}/claims/{path.name}"
    data = _read_json_object(path, where)
    if set(data) != set(MARKER_FIELDS) or data.get("schema") != MARKER_SCHEMA:
        raise PreregError(f"{where}: not the runner's {MARKER_SCHEMA} schema {list(MARKER_FIELDS)} (beyond it: "
                          f"{sorted(set(data) - set(MARKER_FIELDS))}; missing: {sorted(set(MARKER_FIELDS) - set(data))}"
                          f"; a marker holds an attempt's key and split, never an outcome)")
    if not _attempt_key(data["key"], layout):
        raise _key_refusal(where, layout)
    split = data["split"]
    if not (isinstance(split, str) and split and all(c.isascii() and (c.isalnum() or c in "_-") for c in split)):
        raise PreregError(f"{where}: its split is not a split name")
    if path.name != f"{_key_digest(data['key'])}{MARKER_SUFFIX}":
        raise PreregError(f"{where}: its name is not the runner's digest of its key")
    return tuple(data["key"]), split


def attempt_file_names(batch: Path) -> list[str]:
    """Every ``*.json`` name, set-aside sidecars included, in the directories the replay check hashes (gate.BATCH_DIRS:
    records/, infrastructure/, harness/, aborted/), which hold everything the loaders read: names only, from the
    directory listing (os.listdir), no file opened or stat-ed. A superset of gate.batch_files and of load_batch's."""
    out = []
    for sub in BATCH_DIRS:
        try:
            names = os.listdir(Path(batch) / sub)
        except (FileNotFoundError, NotADirectoryError):
            continue
        out.extend(f"{sub}/{n}" for n in names if n.endswith(".json"))
    return sorted(out)


def _by_directory(files: Iterable[str]) -> str:
    return ", ".join(f"{n} in {d}/" for d, n in sorted(Counter(f.split("/", 1)[0] for f in files).items()))


def _test_run_batch(batch: Path, questions: Sequence[Path], split: str) -> None:
    """Design log ruling, the reviews of 2ff8f40 and 87c5c65: a test run opens a batch only when every file the replay
    check or the loaders would touch is placed in the run's split by its name, before any of them is read. The runner
    (feat/e0.1-skeleton 1b7b7ec) names every file of an attempt "<claim digest>.<run id>.json" (RUN_FILE_NAME; a
    set-aside sidecar ".setaside.json") and keeps the attempt's marker claims/<claim digest>.attempt (its key and
    split; never deleted, so an aborted attempt keeps it). So a file is placed by its name's digest and that marker,
    read alone (read_marker; it holds no outcome). Refused, with no file opened: a file named without a digest (a run
    before attempt markers), and a file whose digest has no marker. Then every marker and every claim in claims/
    (read_claim; with the split since 1b7b7ec) is read, so an attempt that left no file is placed too, each with the
    attempt key of the batch's layout (its manifest's schema, gate.manifest_layout: a grid run's keys hold the arm
    first): each one's split (the marker's, the claim's when it has one) must be the split the batch's questions file
    gives its question, and together they must place every attempt in the run's split alone. A batch with no markers
    and no claims, or a mixed one, is refused. A replay-check/2 then hashes the batch's concurrency-events.jsonl too,
    which no marker places: it is read only after every attempt has been placed in the run's split here, and holds the
    runner's concurrency settings and their reasons per model, never an attempt's outcome."""
    files = attempt_file_names(batch)
    unnamed = [f for f in files if RUN_FILE_NAME.fullmatch(f.split("/", 1)[1]) is None]
    if unnamed:
        raise PreregError(f"a test run does not open {batch}: its files ({_by_directory(unnamed)}) are named without a "
                          f"claim digest (a run before attempt markers), so their split cannot be established without "
                          f"reading them (design log ruling: a test run never reads the registered split)")
    claims = Path(batch) / "claims"
    markers = {p.name.removesuffix(MARKER_SUFFIX): p for p in sorted(claims.glob(f"*{MARKER_SUFFIX}"))}
    unmarked = [f for f in files if RUN_FILE_NAME.fullmatch(f.split("/", 1)[1])["digest"] not in markers]
    if unmarked:
        raise PreregError(f"a test run does not open {batch}: its files ({_by_directory(unmarked)}) have no attempt "
                          f"marker (claims/<claim digest>{MARKER_SUFFIX}), so their split cannot be established without "
                          f"reading them (design log ruling: a test run never reads the registered split)")
    by_id = load_questions(select_questions(batch, questions))
    layout = manifest_layout(batch)  # from the manifest alone, which holds no outcome
    placed = [(p.name, *read_marker(p, layout)) for p in markers.values()]
    placed += [(p.name, *read_claim(p, layout)) for p in sorted(claims.glob("*.claim"))]
    if not placed:
        raise PreregError(f"a test run does not open {batch}: no markers or claims decide its split (design log ruling: a "
                          f"test run never reads the registered split)")
    splits = set()
    for name, key, named in placed:
        own = str((by_id.get(key_question(key, layout)) or {}).get("split", "unknown"))
        if named is not None and named != own:
            kind = "marker" if name.endswith(MARKER_SUFFIX) else "claim"
            raise PreregError(f"a test run does not open {batch}: the {kind} {name} names the split {named!r}, but the "
                              f"batch's questions file puts its question in {own!r}")
        splits.add(own)
    if splits != {split}:
        raise PreregError(f"a test run does not open {batch}: its markers and claims place attempts in "
                          f"{sorted(splits)}, not only in {split!r} (design log ruling: a test run never reads the "
                          f"registered split; a mixed batch is refused, not opened)")


def _test_run_ledger(path: Path, split: str) -> None:
    """Design log ruling, the review of 2ff8f40 (blocker 3): a checker-ledger row holds its split only through its
    question, so no row's split can be established without reading it, and a header naming a split says nothing about
    the rows below it unless the writer guarantees a file of one split. A test run opens a ledger only when its first
    line, the header, is of a SINGLE_SPLIT_LEDGERS schema and names the run's split; only that line is read to decide
    (inputs.first_line). Every checker-ledger/1 file is refused, whatever its header names. A /2 file passing here is
    then read by inputs.load_checker_ledger row by row, each row's split checked first (the review of 87c5c65): its
    writer guarantees one split per file, and that check is the backstop."""
    raw = first_line(path)
    try:
        header = json.loads(raw) if raw is not None else None
    except ValueError:
        header = None
    if not isinstance(header, dict):
        raise PreregError(f"a test run does not open the checker ledger {path}: its first line is not a header "
                          f"(design log ruling: a test run never reads the registered split)")
    schema, named = header.get("schema"), header.get("split")
    if schema not in SINGLE_SPLIT_LEDGERS:
        raise PreregError(f"a test run does not open the checker ledger {path}: its schema {schema!r} has no writer that "
                          f"guarantees a file of one split, and its rows hold their split only through their questions; "
                          f"a test run opens only a {' or '.join(SINGLE_SPLIT_LEDGERS)} ledger whose header names the "
                          f"split (design log ruling: a test run never reads the registered split)")
    if named != split:
        raise PreregError(f"a test run does not open the checker ledger {path}: its header "
                          f"{'names no split' if named is None else f'names the split {named!r}'}, not {split!r} "
                          f"(design log ruling: a test run never reads the registered split)")
    if EXPLORATORY in header:
        raise PreregError(f"a test run does not open the checker ledger {path}: an exploratory export (its header says "
                          f"so), which a registered analysis does not read; export it without --exploratory")


def check_questions_hash(ledger: Iterable[CheckedOutput], prereg: PreregConfig) -> None:
    """Every checker-ledger and bank row of the graph names the frozen question set (ids are reused across calibration
    rounds: a row from another question set would join silently on its id); anything else is refused."""
    for r in ledger:
        if r.graph != prereg.graph:
            continue
        want = prereg.frozen_questions(bank=r.cohort == BANK)
        if r.questions_sha256 != want:
            raise PreregError(f"ledger row {r.run_id} ({r.source}): questions_sha256 {r.questions_sha256}, not the frozen "
                              f"question set {want}")


def bank_split_questions(manifest: Path, split: str) -> dict[str, str]:
    """The bank's questions in the split and their template x tier strata, from the manifest's items.jsonl (read
    source-free: question, split, template, tier; inputs.bank_items_bytes, which check_bank_manifest checks against the
    manifest's sha256)."""
    out: dict[str, str] = {}
    path, raw = bank_items_bytes(manifest)
    if not raw and not path.is_file():
        raise FileNotFoundError(f"{path}: the bank's items are not beside its manifest (items.jsonl or items.jsonl.gz)")
    for line in raw.decode("utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if item.get("split") == split:
                out.setdefault(item["question_id"], f"{item.get('template')}|{item.get('tier', 'unrecorded')}")
    return out


def bank_question_index(population: Mapping[str, str], design: Iterable[str]) -> QuestionIndex | None:
    """The bank's own question index over ``population`` (question -> template x tier stratum), or None when the bank
    holds exactly the agent design's questions and so shares the design's index. The bank contrasts count exactly one
    untouched gold per question of the index they run on (prereg §4.1), so any other population gets its own: a full
    bank with bank-only extension questions (change-log entry 41), and a test run's declared subset of the design's
    questions (dress rehearsal 2, whose six-question slice took the design's 91-question index and was refused). An
    empty population carries no bank contrast and is refused."""
    if not population:
        raise InputError("the bank's population holds no bank question: no bank contrast can be computed")
    if set(population) == set(design):
        return None
    return QuestionIndex.from_strata(population)


__all__ = ["METRICS", "analyse", "analyse_batches", "index_from_units"]
