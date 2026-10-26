"""C2's source-layer ablations on the fault bank, one descriptive table reported beside C2 (prereg §6, "Exploratory
analyses, registered": source-layer ablations; change-log entry 108: "curated with and without vocabulary flags, and
census type checks against its join and pattern checks. No new analysis.").

Descriptive only: no estimand changes and no test of a hypothesis. Every rate is the registered per-question mean
(prereg §4.1: prereg.source_rates, unchanged) with its pooled counts, computed from the bank arms the analysis already
loaded (build_arms over the checker ledger: the matrix's primary variant, both rule-23 switches on), with some of a
source's flags left out; the dev bank's descriptive tables (results/c2-descriptive-20261020, c2_descriptive.py, tables
A to C) computed the same numbers from the matrix and the ledger outside the analysis. An item counts as flagged under a
row's checks when one of those checks fired on it (the ledger's flagged_check_types: rule 9's check types,
harness/checks/outcomes.py CHECKS); an item a row no longer counts as flagged keeps its other outcome (unknown when any
check was unknown, else a pass), which no rate here reads.

Rows:
- curated, every check, and curated without its vocabulary flags (vocabulary existence left out);
- census, every check; census by its type checks (class, domain/range, disjointness), and by its co-occurrence join
  and pattern checks alone (a census check fired and no type check): the two parts of its catches;
- retry on empty, entry 59's reference: an item is flagged exactly when its query returned no rows when the bank was
  built (result_rows 0), unknown when that count is not recorded.

Columns: type-wrong recall (C2 ii's rate) on all type-wrong answer-changing mutations and split by whether the mutant's
query returns rows (the registered empty-result stratum, entry 59: pipeline.result_stratum); f_gold, f_rewrite and f
(C2 i's rate). The reference flags no mutant that returns rows and every empty one, by construction.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from harness.analysis.inputs import CheckedOutput
from harness.analysis.metrics import ArmKey, Unit, indicators, question_vectors
from harness.analysis.prereg import PreregConfig, source_rates
from harness.analysis.stats import QuestionIndex

ABLATIONS_SCHEMA = "source-layer-ablations/1"
VOCABULARY = "vocabulary"
TYPE_CHECKS = frozenset({"class", "domain_range", "disjointness"})
CENSUS_CHECKS = frozenset({"census"})  # census_join.static and census_pattern.static
REFERENCE_CHECK = "empty_result"  # retry on empty's one trigger (outcomes.CHECKS)
SPLITS = ("returns_rows", "empty")


def _types(c: CheckedOutput) -> tuple[str, ...]:
    return tuple(c.flagged_check_types) if c.status == "F" else ()


def _keep(drop: frozenset[str] = frozenset(), only: frozenset[str] | None = None,
          unless: frozenset[str] = frozenset()) -> Callable[[CheckedOutput], tuple[str, ...]]:
    """The check types of an item's flag that a row counts: its own, without ``drop``, within ``only`` (when given),
    and none at all when one of ``unless`` fired."""
    def kept(c: CheckedOutput) -> tuple[str, ...]:
        types = _types(c)
        if set(types) & unless:
            return ()
        return tuple(t for t in types if t not in drop and (only is None or t in only))
    return kept


def _rows(cfg: PreregConfig) -> list[tuple[str, str, str, Callable[[CheckedOutput], tuple[str, ...]]]]:
    """(label, the bank source it reads, its checks, the check types of an item's flag it counts)."""
    return [
        ("curated", cfg.curated_source, "every check", _keep()),
        ("curated without vocabulary flags", cfg.curated_source, "every check but vocabulary existence",
         _keep(drop=frozenset({VOCABULARY}))),
        ("census", cfg.census_source, "every check", _keep()),
        ("census type checks", cfg.census_source, "class, domain/range and disjointness checks",
         _keep(only=TYPE_CHECKS)),
        ("census join and pattern checks alone", cfg.census_source,
         "a co-occurrence join or pattern check, when no type check fired",
         _keep(only=CENSUS_CHECKS, unless=TYPE_CHECKS)),
    ]


def _recounted(u: Unit, status: str, types: tuple[str, ...]) -> Unit:
    """The unit with its item's status and flagging check types as a row counts them."""
    c = u.checked
    if (status, types) == (c.status, tuple(c.flagged_check_types)):
        return u
    flagged = status == "F"
    new = dataclasses.replace(c, status=status, flagged_check_types=types,
                              branch_local_only=c.branch_local_only and flagged, data_fault=c.data_fault and flagged)
    return Unit(u.cluster, u.attempt, new, indicators(u.attempt, new))


def _ablated(u: Unit, kept: Callable[[CheckedOutput], tuple[str, ...]]) -> Unit:
    """An item a row no longer counts as flagged keeps its other outcome: unknown when a check was unknown, else a pass
    (no rate here reads which)."""
    c = u.checked
    types = kept(c)
    if types:
        return _recounted(u, "F", types)
    return _recounted(u, ("U" if c.any_unknown else "P") if c.status == "F" else c.status, ())


def _reference(u: Unit) -> Unit:
    """Retry on empty on a bank item (entry 59): flagged exactly when its query returned no rows when the bank was
    built, unknown when that count is not recorded (as matrix.empty_retry_bank_rows reads the bank)."""
    n = u.checked.result_rows
    if n is None:
        return _recounted(u, "U", ())
    return _recounted(u, "F", (REFERENCE_CHECK,)) if n == 0 else _recounted(u, "P", ())


def _count(units: Sequence[Unit], num: str, den: str) -> dict[str, Any]:
    k, n = int(sum(u.ind[num] for u in units)), int(sum(u.ind[den] for u in units))
    return {"flagged": k, "items": n, "pooled": k / n if n else None}


def _questions(vec: Mapping[str, Any], den: str) -> int:
    return int((vec[den] > 0).sum())


def ablation_table(arms: Mapping[ArmKey, Sequence[Unit]], index: QuestionIndex, cfg: PreregConfig) -> dict[str, Any]:
    """The table on the bank arms of the config's graph (prereg §6; the module docstring), from ``index``, the bank's
    question index."""
    # imported here, since the pipeline imports this module
    from harness.analysis.pipeline import result_stratum

    missing = [s for s in (cfg.curated_source, cfg.census_source) if cfg.bank(s) not in arms]
    if missing:
        return {"descriptive": True, "schema": ABLATIONS_SCHEMA, "computed": False,
                "reason": f"no bank arm of source(s) {missing} on {cfg.graph}"}
    tables = [(label, source, checks, [_ablated(u, kept) for u in arms[cfg.bank(source)]])
              for label, source, checks, kept in _rows(cfg)]
    # the reference reads the bank's items, which every bank arm holds alike (inputs.check_bank_pairing)
    tables.append(("retry on empty (reference)", None, "the query returned no rows when the bank was built",
                   [_reference(u) for u in arms[cfg.bank(cfg.curated_source)]]))
    rows = []
    for label, source, checks, units in tables:
        vec = question_vectors(units, index)
        rates = source_rates(vec, cfg)
        row: dict[str, Any] = {
            "label": label, "source": source, "checks": checks,
            "type_wrong_recall": {**_count(units, "W_F_tw", "tw"), "per_question": rates["recall_type_wrong"],
                                  "questions": _questions(vec, "tw")},
        }
        for stratum in SPLITS:
            part = [u for u in units if u.ind["tw"] and result_stratum(u.checked) == stratum]
            pvec = question_vectors(part, index)
            row[f"type_wrong_recall_{stratum}"] = {**_count(part, "W_F_tw", "tw"),
                                                   "per_question": source_rates(pvec, cfg)["recall_type_wrong"],
                                                   "questions": _questions(pvec, "tw")}
        row["f_gold"] = {**_count(units, "C_F_untouched", "C_untouched"), "per_question": rates["f_gold"],
                         "questions": _questions(vec, "C_untouched")}
        row["f_rewrite"] = {**_count(units, "C_F_rewrite", "C_rewrite"), "per_question": rates["f_rewrite"],
                            "questions": _questions(vec, "C_rewrite")}
        row["f"] = rates["f"]
        rows.append(row)
    return {"descriptive": True, "schema": ABLATIONS_SCHEMA, "computed": True, "graph": cfg.graph,
            "section": "prereg §6 source-layer ablations, beside C2 (change-log entry 108)",
            "rates": "per-question means (prereg §4.1, prereg.source_rates) with pooled counts", "rows": rows}
