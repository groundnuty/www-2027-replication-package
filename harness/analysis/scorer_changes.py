"""The verdicts the frozen scorer changes against scorer-rules-v4, per arm (change-log entries 120 and 127): descriptive,
beside the analysis, never confirmatory.

The frozen scorer's rules (scorer-rules-v6) read some outputs that v4 scored wrong as exact: an answer with extra
columns where exactly one choice of columns has the gold's kinds, an anchor's quoted name for its IRI, a plain-literal
date. Each scored output of the analysed split is scored again through the runner's own path (loop.score_outcome
against harness.score.question_gold, under the question's rules) twice: with every v5 rule off (v4) and as the rules now
read (the frozen version). The stored score is checked against its own rules version (v4, v5 or the frozen one): a
stored score those rules do not reproduce is counted (``not_reproduced``), never silently replaced. Per arm (graph,
model, cohort): the outputs scored, exact under v4 and under the frozen rules, gained (wrong under v4, exact now), lost
(the reverse) and unscored ones (no score, or a question without gold rows). A record or question the runner's path
cannot score again (malformed stored rules, question rules or gold, no outcome) is counted (``rescore_errors``) and
listed, never raised (External review of 13741b8). The analysis pipeline reports it for every arm and, beside the
registered family, for C1's prompted arms (``c1_table``). Each rescored record's v4 verdict (``v4_verdicts``) is what
the C1 sensitivity under v4 reads (change-log entry 129 item 5), so the two agree by construction.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

FIELDS = ("scored", "exact_v4", "exact_frozen", "gained", "lost", "unscored")
STATUS = ("descriptive (change-log entries 120 and 127): the verdicts the frozen scorer changes against "
          "scorer-rules-v4; it is never confirmatory and changes no arm's numbers")


class ScorerChanges:
    """The counts, added one record at a time (``add``) and read with ``table``."""

    def __init__(self) -> None:
        self.counts: dict[str, Counter] = defaultdict(Counter)
        self.versions: dict[str, Counter] = defaultdict(Counter)
        self.errors: list[str] = []
        self.v4_verdicts: dict[str, tuple[bool, float]] = {}  # run id -> (exact, F1) under v4, rescored records only

    def add(self, record: Mapping[str, Any], question: Mapping[str, Any] | None, attempt: Any) -> None:
        from harness.agent.loop import score_outcome
        from harness.score import ScoringRules, question_gold
        from harness.score.rules import RULES_VERSION, RULES_VERSION_V4, RULES_VERSION_V5

        arm = f"{attempt.graph}|{attempt.model}|{attempt.cohort}"
        counts = self.counts[arm]
        score = record.get("score")
        stored = score.get("rules") if isinstance(score, Mapping) else None
        version = stored.get("version") if isinstance(stored, Mapping) else None
        if isinstance(score, Mapping):  # every stored score's version, rescored or not
            self.versions[arm][str(version)] += 1
        if not isinstance(score, Mapping) or not question or "gold_rows" not in question or \
                "scoring_rules" not in question:
            counts["unscored"] += 1
            return
        try:
            if not isinstance(stored, Mapping):
                raise TypeError(f"the stored score's rules are {type(stored).__name__}, not an object")
            base, gold = ScoringRules(**question["scoring_rules"]), question_gold(question)
            outcome, rows = record["outcome"], record.get("final_rows")
            v4 = score_outcome(outcome, rows, gold, base.as_v4().to_json())
            frozen = score_outcome(outcome, rows, gold, base.to_json())
            own = {RULES_VERSION_V4: v4, RULES_VERSION: frozen}.get(version)
            if version == RULES_VERSION_V5:
                own = score_outcome(outcome, rows, gold, replace(base, extra_columns_by_kind=False).to_json())
        except (AttributeError, KeyError, TypeError, ValueError) as exc:  # the runner's path cannot score it: counted
            counts["rescore_errors"] += 1
            self.errors.append(f"{record.get('run_id')}: {type(exc).__name__}: {str(exc)[:200]}")
            return
        counts["not_reproduced"] += own != dict(score)
        self.v4_verdicts[attempt.run_id] = (v4["exact"], v4["f1"])
        counts["scored"] += 1
        counts["exact_v4"] += v4["exact"]
        counts["exact_frozen"] += frozen["exact"]
        counts["gained"] += frozen["exact"] and not v4["exact"]
        counts["lost"] += v4["exact"] and not frozen["exact"]

    def table(self) -> dict[str, Any]:
        from harness.score.rules import RULES_VERSION

        arms = {arm: {**{f: c[f] for f in FIELDS}, "stored_versions": dict(sorted(self.versions[arm].items())),
                      "not_reproduced": c["not_reproduced"], "rescore_errors": c["rescore_errors"]}
                for arm, c in sorted(self.counts.items())}
        totals = {f: sum(row[f] for row in arms.values()) for f in (*FIELDS, "not_reproduced", "rescore_errors")}
        return {"descriptive": True, "status": STATUS, "frozen_version": RULES_VERSION, "arms": arms, "totals": totals,
                "rescore_errors": self.errors[:50], "rescore_errors_omitted": max(0, len(self.errors) - 50)}


def c1_table(table: Mapping[str, Any], prereg: Any) -> dict[str, Any]:
    """The table's rows of C1's prompted arms (each model of C1, its curated and census cohorts), zeros where an arm has
    no output."""
    rows = []
    for model in prereg.models:
        for cohort in (prereg.curated_prompt_cohort, prereg.census_prompt_cohort):
            key = prereg.prompt(model, cohort)
            row = table["arms"].get(f"{key.graph}|{key.model}|{key.cohort}") or {
                **dict.fromkeys(FIELDS, 0), "stored_versions": {}, "not_reproduced": 0, "rescore_errors": 0}
            rows.append({"arm": key.label, "model": model, "cohort": cohort, **row})
    return {"descriptive": True, "status": STATUS, "frozen_version": table["frozen_version"], "rows": rows}


C1_UNDER_V4 = "C1 under scorer-rules-v4 (sensitivity)"
C1_UNDER_V4_STATUS = ("descriptive (change-log entry 129 item 5): C1a and C1b with each output's verdict as "
                      "scorer-rules-v4 scores it, beside the family under the frozen scorer; the same clusters and "
                      "resamples, the recorded C1a form, p unadjusted; no Holm, never a claim")


def c1_under_v4(arms: Mapping[Any, Any], index: Any, prereg: Any, bank_index: Any = None, *, root: Any,
                clusters: Any = None, multiplicity: Any = None, testing: bool = False) -> dict[str, Any]:
    """C1a and C1b on ``arms``, the analysed arms with each output's verdict under scorer-rules-v4 (change-log entry 129
    item 5), descriptive beside the family. It runs outside the frozen inference (prereg.confirmatory and its closure
    are unchanged) and reads only what the registered run reads: the registered preflight's config, cluster map,
    resample matrix and weighted-Holm tables (the primary's clusters and resamples: prereg.Clusters.resampling with
    the same seed), the recorded C1a form, the test the weighted family selected for each claim (as
    multiplicity.weighted_family shows it) with its p unadjusted, and each component's unadjusted 95% interval by that
    test. No Holm, never a claim. ``arms`` must hold the v4 verdicts in the prompt arms only (C1 reads no other)."""
    from harness.analysis.multiplicity import _claim_name
    from harness.analysis.prereg import _interval, bounded, c1a, c1b, preflight, vectors_from_units

    pre = preflight(prereg, root, clusters=clusters, multiplicity=multiplicity, testing=testing)
    K = pre.clusters.resampling(index, pre.cfg, pre.K)  # as prereg.clustered_contrasts draws the primary's
    V = vectors_from_units(arms, index, prereg, bank_index)
    out: dict[str, Any] = {"descriptive": True, "primary_scorer": pre.cfg.scorer_version, "status": C1_UNDER_V4_STATUS}
    for c in (c1a(V, K, pre.cfg), c1b(V, K, pre.cfg)):
        # the test the weighted family selected (multiplicity.weighted_family; boot_t shown where none is selected)
        selected, _ = pre.multiplicity.final(_claim_name(c["id"], pre.cfg))
        shown = selected or "boot_t"
        c["test"], c["test_selected"], c["p_unadjusted"] = shown, selected is not None, c["p_test"][shown]
        for comp in c["components"]:
            comp["ci95"] = _interval(comp.pop("_s"), 0.05, "two", shown)
        out[c["id"]] = bounded(c)
    return out
