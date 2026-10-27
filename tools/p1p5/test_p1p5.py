"""Tests of p1p5.py on synthetic matrices (no real data): python3 -I test_p1p5.py -v

The synthetic world (graph uniprot-sp-2026_03, nine attempts) has hand-counted cells; see WORLD below.
"""

from __future__ import annotations

import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I leaves the script directory off the import search list
import p1p5  # noqa: E402

G = "uniprot-sp-2026_03"
SONNET, QWEN, HAIKU = "claude-sonnet-5-5", "Qwen/Qwen3.8-27B", "claude-haiku-5-5"
FP, VP, UNK, FL = "substantive_pass", "vacuous_pass", "unknown", "flagged"

# run, model, template, correct (None = no output), curated, census-t5, union
WORLD = [
    ("r1", QWEN, "aggregation", False, FL, UNK, FL),
    ("r2", QWEN, "aggregation", False, VP, FP, FP),
    ("r3", QWEN, "numeric_filter", False, UNK, FL, FL),
    ("r4", SONNET, "type_members", False, FP, FP, FP),
    ("r5", SONNET, "type_members", True, FL, FP, FL),
    ("r6", SONNET, "aggregation", True, FP, FP, FP),
    ("r7", QWEN, "aggregation", True, UNK, VP, UNK),
    ("r8", QWEN, "aggregation", None, None, None, None),
    ("r9", SONNET, "aggregation", None, None, None, None),
]
ARMS = ("curated", "census-t5", "union")


def mk_rows(world=WORLD, graph=G, split="dev", with_other_variants=True) -> list[dict]:
    rows = []
    for run, model, tmpl, correct, *st in world:
        by_arm = dict(zip(ARMS, st))
        for arm in ARMS:
            for variant in (("primary", "both_off") if with_other_variants else ("primary",)):
                out = correct is not None
                status = by_arm[arm] if variant == "primary" else (FL if out else None)  # both_off differs on purpose
                row = {"run_id": run, "graph": graph, "question_id": f"q-{run}", "template": tmpl, "split": split,
                       "model": model, "cohort": "none", "variant": variant, "arm": arm, "output": out,
                       "correct": correct, "status": status}
                if out and arm == "union":
                    row["per_source"] = {"curated": by_arm["curated"], "census-t5": by_arm["census-t5"]}
                elif out:
                    row["per_source"] = {arm: status}
                rows.append(row)
    return rows


def units_of(rows=None) -> dict:
    return p1p5.build_units(rows if rows is not None else mk_rows(), "dev")


def pop_of(units=None, digests=None, result=None, judge=None, judges=(), accounting=None, design_log=None,
           registered=True) -> p1p5.Population:
    return p1p5.Population(
        "syn", "synthetic", registered, units if units is not None else units_of(),
        digests if digests is not None else {G: {"curated": "d-cur", "census-t5": "d-other"}},
        {G: {"union": ["curated", "census-t5"]}}, result or {}, judge, list(judges), accounting or {}, design_log)


# -- the bank ---------------------------------------------------------------------------------------------------------

BANK_ITEMS = [  # item, correct, result_rows, curated status, census status
    ("b1", False, 5, FL, FP), ("b2", False, 3, FP, FP), ("b3", False, 0, FL, FP), ("b4", False, None, UNK, FP),
    ("b5", True, 2, FL, FP), ("b6", True, 2, FP, FP)]


def bank_rows(split="dev", extra=()) -> list[dict]:
    rows = []
    for item, correct, rr, cur, cen in BANK_ITEMS:
        for arm, st in (("curated", cur), ("census-t5", cen), ("generated-dc-1", FP)):
            for variant in ("primary", "both_off"):
                rows.append({"run_id": item, "graph": G, "split": split, "variant": variant, "arm": arm,
                             "output": True, "correct": correct, "status": st if variant == "primary" else FL,
                             "result_rows": rr, "cohort": "bank"})
    return rows + list(extra)


def bank_of() -> dict:
    counts = p1p5.bank_counts(bank_rows(), "dev")
    return {"graph": G, "counts": counts,
            "digests": {"curated": "d-cur", "census-t5": "d-cen", "generated-dc-1": "d-dc1"}, "bank_rates": {},
            "plan_code": None, "items": 6}


# -- the judge ---------------------------------------------------------------------------------------------------------

def judge_data() -> p1p5.JudgeData:
    keys = {f"output:r{i}": ("wrong" if i <= 4 else "right") for i in range(1, 8)}
    model = {"r1": QWEN, "r2": QWEN, "r3": QWEN, "r4": SONNET, "r5": SONNET, "r6": SONNET, "r7": QWEN}
    result = {"r1": "rows", "r2": "rows", "r3": "empty", "r4": "rows", "r5": "rows", "r6": "rows", "r7": "rows"}
    selection = {k: {"key": k, "model": model[k.split(":")[1]], "result": result[k.split(":")[1]]} for k in keys}
    sonnet = {"r1": "reject", "r2": "accept", "r3": None, "r4": "reject", "r5": "reject", "r6": "accept", "r7": "ERR"}
    haiku = {"r1": "reject", "r2": "reject"}
    verdict, vocab = {}, {}
    for judge, table in ((SONNET, sonnet), (HAIKU, haiku)):
        for r, v in table.items():
            k = f"output:{r}"
            verdict[(k, judge)] = "error" if v == "ERR" else (v or "unparsed")
    raw = {pair: ("<failed>" if v == "error" else "<unparsed>" if v == "unparsed" else v) for pair, v in verdict.items()}
    return p1p5.JudgeData(keys, selection, verdict, raw)


class Cells(unittest.TestCase):
    def setUp(self):
        self.us = list(units_of().values())

    def test_cells_unknown_in_denominator_vacuous_is_pass_no_output_excluded(self):
        c = p1p5.cells(self.us, "curated")
        self.assertEqual(c, {"W_F": 1, "W_P": 2, "W_U": 1, "C_F": 1, "C_P": 1, "C_U": 1})  # r2 vacuous = P
        b = p1p5.rates_block(self.us, "curated")
        self.assertEqual((b["recall"]["num"], b["recall"]["den"]), (1, 4))  # U in the denominator, not a flag
        self.assertEqual((b["false_flag"]["num"], b["false_flag"]["den"]), (1, 3))
        self.assertEqual(len(p1p5.outputs(self.us)), 7)  # the two no-output attempts enter no denominator

    def test_union_row_is_read_not_recomputed(self):
        rows = mk_rows()
        for r in rows:  # make the union row disagree with an OR of its members on r2
            if r["run_id"] == "r2" and r["arm"] == "union" and r["variant"] == "primary":
                r["status"] = FL
        us = list(units_of(rows).values())
        self.assertEqual(p1p5.cells(us, "union")["W_F"], 3)  # r1, r2, r3 flagged by the union's own rows

    def test_only_primary_variant(self):
        us = list(units_of(mk_rows(with_other_variants=True)).values())
        us2 = list(units_of(mk_rows(with_other_variants=False)).values())
        for a in ARMS:
            self.assertEqual(p1p5.cells(us, a), p1p5.cells(us2, a))

    def test_zero_wrong_is_not_estimable_not_zero(self):
        correct_only = [w for w in WORLD if w[3] is not False]
        us = list(units_of(mk_rows(correct_only)).values())
        r = p1p5.rates_block(us, "curated")["recall"]
        self.assertEqual((r["num"], r["den"], r["rate"]), (0, 0, None))
        self.assertIn("not estimable", p1p5.fmt_ratio(r))

    def test_input_errors(self):
        with self.assertRaises(p1p5.InputError):  # a test-split row is refused
            p1p5.build_units(mk_rows(split="test"), "dev")
        rows = mk_rows()
        with self.assertRaises(p1p5.InputError):  # duplicate (run, arm)
            p1p5.build_units(rows + [dict(rows[0])], "dev")
        bad = mk_rows()
        bad[0]["status"] = "weird"
        with self.assertRaises(p1p5.InputError):
            p1p5.build_units(bad, "dev")
        bad = mk_rows()
        for r in bad:
            if r["run_id"] == "r8" and r["variant"] == "primary" and r["arm"] == "curated":
                r["status"] = FL  # a status on a no-output row
        with self.assertRaises(p1p5.InputError):
            p1p5.build_units(bad, "dev")
        bad = mk_rows()
        for r in bad:
            if r["run_id"] == "r1" and r["arm"] == "census-t5":
                r["correct"] = True  # arms disagree on correctness
        with self.assertRaises(p1p5.InputError):
            p1p5.build_units(bad, "dev")
        short = [r for r in mk_rows() if not (r["run_id"] == "r1" and r["arm"] == "union")]
        with self.assertRaises(p1p5.InputError):  # one output lacks an arm the others have
            p1p5.build_units(short, "dev")


class Predictions(unittest.TestCase):
    def test_p2_template_table_and_focus(self):
        r = p1p5.p2(pop_of())[G]
        self.assertEqual(r["templates"]["aggregation"]["wrong_outputs"], 2)
        self.assertEqual(r["templates"]["aggregation"]["sources"]["curated"]["W_F"], 1)
        self.assertEqual(r["templates"]["numeric_filter"]["sources"]["curated"]["W_U"], 1)
        f = r["focus"]
        self.assertEqual(f["templates"], ["aggregation", "numeric_filter"])
        self.assertEqual(f["wrong_outputs"], 3)  # r1, r2, r3; r4 (type_members) is outside
        self.assertEqual(f["sources"]["curated"]["W_F"], 1)
        self.assertEqual(f["sources"]["union"]["W_F"], 2)

    def test_p3_not_replayed_literal_and_rates(self):
        r = p1p5.p3(pop_of())[G]
        self.assertEqual(r["correct_outputs"], 3)
        self.assertEqual(r["sources"]["census-t5"]["C_F"], 0)
        self.assertEqual(r["sources"]["census-t5"]["false_flag"]["rate"], 0.0)
        self.assertEqual(r["not_replayed"]["curated+doc"], "not replayed")
        self.assertNotIn("curated", r["not_replayed"])
        self.assertNotIn("union", r["not_replayed"])
        self.assertEqual(r["sources"]["curated"]["by_agent"][SONNET]["C_F"], 1)

    def test_p4_union_against_each_source_and_tie_rule(self):
        r = p1p5.p4(pop_of())[G]["unions"]["union"]
        self.assertEqual(r["members"], ["curated", "census-t5"])
        cur = r["singles"]["curated"]
        self.assertAlmostEqual(cur["union_minus_recall"], 0.25)
        self.assertAlmostEqual(cur["union_minus_false_flag"], 0.0)
        self.assertEqual(cur["wrong_flagged_by_union_not_by_source"], 1)  # r3
        self.assertEqual(cur["wrong_flagged_by_source_not_by_union"], 0)
        cen = r["singles"]["census-t5"]
        self.assertEqual(cen["wrong_flagged_by_union_not_by_source"], 1)  # r1 (census unknown)
        self.assertAlmostEqual(cen["union_minus_false_flag"], 1 / 3)
        bm = r["best_member"]  # curated and census-t5 tie on recall 1/4: lower false-flag rate wins -> census-t5
        self.assertEqual(bm["tied_on_recall"], ["census-t5", "curated"])
        self.assertEqual(bm["source"], "census-t5")

    def test_p1_bank_pairing_by_digest(self):
        pop = pop_of(result={"r1": "rows", "r2": "rows", "r3": "empty", "r4": "rows"})
        r = p1p5.p1(pop, bank_of())[G]
        srcs = {e["source"]: e for e in r["sources"]}
        cur = srcs["curated"]
        self.assertEqual(cur["bank"]["bank_arm"], "curated")
        rr = cur["bank"]["returns_rows"]  # non-empty answer-changing items b1 (F), b2 (P)
        self.assertEqual((rr["W"], rr["W_F"]), (2, 1))
        self.assertEqual(cur["bank"]["all_items"]["W"], 4)
        self.assertEqual(cur["real_wrong_all"]["den"], 4)
        self.assertEqual((cur["real_wrong_result_rows"]["num"], cur["real_wrong_result_rows"]["den"]), (1, 3))
        self.assertAlmostEqual(cur["ratio_real_all_over_bank_returns_rows"], 0.25 / 0.5)
        # census-t5: the replay's digest differs from every bank arm's: not replayed (never another arm)
        self.assertTrue(srcs["census-t5"]["bank"].startswith("not replayed"))
        self.assertTrue(srcs["union"]["bank"].startswith("not replayed"))
        self.assertEqual(r["wrong_with_result_rows"], 3)
        # no bank at all
        self.assertTrue(p1p5.p1(pop, None)[G]["sources"][0]["bank"].startswith("not replayed"))
        # a graph without a bank replay
        self.assertTrue(p1p5.bank_pair(pop, bank_of(), "rhea-142", "curated").startswith("not replayed"))

    def test_bank_counts_strata_and_refusals(self):
        c = p1p5.bank_counts(bank_rows(), "dev")["curated"]
        self.assertEqual((c["returns_rows"]["W_F"], c["returns_rows"]["W_P"], c["returns_rows"]["C_F"],
                          c["returns_rows"]["C_P"]), (1, 1, 1, 1))
        self.assertEqual((c["empty"]["W_F"], c["no_row_count"]["W_U"]), (1, 1))
        self.assertEqual(sum(c["all"].values()), 6)
        with self.assertRaises(p1p5.InputError):
            p1p5.bank_counts(bank_rows(split="test"), "dev")
        with self.assertRaises(p1p5.InputError):
            p1p5.bank_counts(bank_rows(extra=[bank_rows()[0]]), "dev")  # an item twice

    def test_p5_judges_complete_partial_not_run_and_unparsed(self):
        pop = pop_of(judge=judge_data(), judges=(SONNET, HAIKU, QWEN))
        q = p1p5.p5(pop)[QWEN][G]
        self.assertEqual(q["wrong_outputs"], 3)
        s = q["judges"][SONNET]
        self.assertEqual(s["coverage"], "complete")
        self.assertEqual((s["recall"]["num"], s["recall"]["den"], s["unparsed_or_failed"]), (1, 3, 1))
        self.assertEqual(s["role"], "cross")
        self.assertEqual(s["sources_on_the_judged_outputs"]["union"]["num"], 2)
        h = q["judges"][HAIKU]  # judged r1, r2 of Qwen's 3 wrong: reported as partial, no whole-cell recall
        self.assertEqual(h["coverage"], "partial 2 of 3")
        self.assertIsNone(h["recall"])
        self.assertEqual((h["recall_on_judged_subset"]["num"], h["recall_on_judged_subset"]["den"]), (2, 2))
        self.assertEqual((h["sources_on_the_judged_outputs"]["curated"]["num"],
                          h["sources_on_the_judged_outputs"]["curated"]["den"]), (1, 2))
        self.assertEqual(h["sources_on_the_judged_outputs"]["census-t5"]["num"], 0)
        n = q["judges"][QWEN]
        self.assertEqual((n["coverage"], n["role"], n["recall"]), ("not run", "self", None))
        self.assertEqual(q["sources_on_all_wrong"]["curated"]["num"], 1)
        cov = p1p5.judge_coverage(pop)
        self.assertEqual(cov[QWEN]["status"], "not run")
        self.assertEqual(cov[HAIKU]["status"], "partial")
        self.assertEqual(cov[SONNET]["status"], "complete")
        self.assertEqual(p1p5.p5(pop_of())["status"], "no judge data for this population")

    def test_best_of_with_nothing_estimable(self):
        rows = {"a": {"recall": {"rate": None}, "false_flag": {"rate": None}}}
        self.assertEqual(p1p5.best_of(["a"], rows, "union", {}), p1p5.NOT_ESTIMABLE)


class Checks(unittest.TestCase):
    def test_population_check_against_registered(self):
        pc = p1p5.population_check(pop_of(judge=judge_data(), judges=(SONNET,)),
                                   {"outputs": 7, "wrong": 4, "wrong_by_agent": {SONNET: 1, QWEN: 3}})
        self.assertEqual((pc["attempts"], pc["no_output"], pc["outputs"], pc["wrong"]), (9, 2, 7, 4))
        self.assertTrue(pc["against_registered"]["all_equal"])
        self.assertTrue(pc["against_judge_calibration"]["equal"])
        pc = p1p5.population_check(pop_of(), {"outputs": 8, "wrong": 5, "wrong_by_agent": {SONNET: 1, QWEN: 3}})
        a = pc["against_registered"]
        self.assertFalse(a["all_equal"])
        self.assertEqual((a["outputs"]["difference"], a["wrong"]["difference"]), (-1, -1))
        self.assertTrue(a["wrong_by_agent"][SONNET]["equal"])
        pc = p1p5.population_check(pop_of(registered=False), {"outputs": 8, "wrong": 5, "wrong_by_agent": {}})
        self.assertEqual(pc["against_registered"]["note"], "not the registered population")

    def test_population_check_judge_set_and_label_differences(self):
        jd = judge_data()
        del jd.truth["output:r7"]
        jd.truth["output:r9x"] = "wrong"
        jd.truth["output:r5"] = "wrong"  # the matrix says correct
        pc = p1p5.population_check(pop_of(judge=jd, judges=(SONNET,)), None)["against_judge_calibration"]
        self.assertEqual((pc["only_in_matrix"], pc["only_in_truth"], pc["truth_label_disagrees_with_matrix_correct"]),
                         (1, 1, 1))
        self.assertFalse(pc["equal"])

    def test_union_consistency(self):
        self.assertEqual(p1p5.union_consistency(pop_of())["disagreeing"], 0)
        self.assertEqual(p1p5.union_consistency(pop_of())["member_statuses_checked"], 14)
        rows = mk_rows()
        for r in rows:
            if r["run_id"] == "r1" and r["arm"] == "union" and r["variant"] == "primary":
                r["per_source"]["curated"] = FP  # the curated row says flagged
        self.assertEqual(p1p5.union_consistency(pop_of(units_of(rows)))["disagreeing"], 1)

    def accounting(self):
        def cell(**kw):
            base = {k: 0 for k in ("C_F", "C_P", "C_U", "W_F", "W_P", "W_U")}
            base.update(kw)
            return base
        by = {QWEN: {"curated": (cell(W_F=1, W_P=1, W_U=1, C_U=1), 4), "census-t5": (cell(W_U=1, W_P=1, W_F=1, C_P=1), 4),
                     "union": (cell(W_F=2, W_P=1, C_U=1), 4)},
              SONNET: {"curated": (cell(W_P=1, C_F=1, C_P=1), 3), "census-t5": (cell(W_P=1, C_P=2), 3),
                       "union": (cell(W_P=1, C_F=1, C_P=1), 3)}}
        return {"accounting": {m: {"none": {a: {"primary": {"cells": c, "N_out": n}} for a, (c, n) in arms.items()}}
                               for m, arms in by.items()}, "meta": {}}

    def test_reconcile_accounting_matches_and_detects(self):
        acc = self.accounting()
        r = p1p5.reconcile_accounting(pop_of(accounting={G: acc}))
        self.assertEqual((r["compared"], r["mismatches"], r["match"]), (6, 0, True))
        acc["accounting"][QWEN]["none"]["curated"]["primary"]["cells"]["W_F"] = 2
        r = p1p5.reconcile_accounting(pop_of(accounting={G: acc}))
        self.assertEqual((r["mismatches"], r["match"]), (1, False))
        acc = self.accounting()
        acc["accounting"][SONNET]["none"]["union"]["primary"]["N_out"] = 4
        self.assertFalse(p1p5.reconcile_accounting(pop_of(accounting={G: acc}))["match"])
        acc = self.accounting()
        del acc["accounting"][SONNET]["none"]["union"]
        self.assertFalse(p1p5.reconcile_accounting(pop_of(accounting={G: acc}))["match"])

    def test_reconcile_bank(self):
        bank = bank_of()
        bank["bank_rates"] = {}
        for arm, strata in bank["counts"].items():
            bank["bank_rates"][arm] = {"primary": {s: {"W": p1p5.cells_ratios(c)["W"], "W_F": c["W_F"],
                                                      "C": p1p5.cells_ratios(c)["C"], "C_F": c["C_F"]}
                                                  for s, c in strata.items()}}
        r = p1p5.reconcile_bank(bank)
        self.assertEqual((r["compared"], r["mismatches"]), (9, 0))  # 3 arms x (all, returns_rows, empty)
        bank["bank_rates"]["curated"]["primary"]["returns_rows"]["W_F"] = 9
        self.assertEqual(p1p5.reconcile_bank(bank)["mismatches"], 1)

    def test_reconcile_judge_against_tables(self):
        pop = pop_of(judge=judge_data(), judges=(SONNET, HAIKU, QWEN))
        t = []
        for g in ("uniprot", "all"):
            t.append({"graph": g, "agent": QWEN, "judge": "Sonnet", "wrong": 3, "right": 1,
                      "recall": {"reject": 1, "unparsed": 1}, "false_reject": {"reject": 0}})
            t.append({"graph": g, "agent": SONNET, "judge": "Sonnet", "wrong": 1, "right": 2,
                      "recall": {"reject": 1, "unparsed": 0}, "false_reject": {"reject": 1}})
        r = p1p5.reconcile_judge(pop, {"table_1_by_graph_round_agent_judge": t})
        self.assertEqual((r["compared"], r["mismatches"]), (4, 0))
        t[0]["recall"]["reject"] = 2
        self.assertEqual(p1p5.reconcile_judge(pop, {"table_1_by_graph_round_agent_judge": t})["mismatches"], 1)

    def test_reconcile_design_log(self):
        summary = {"attempts": 9, "groups": {"P": [QWEN, SONNET]},
                   "per_template_P": {"curated": {"aggregation": {"W": 2, "W_F": 1, "C": 2, "C_F": 0}},
                                      "union": {"type_members": {"W": 1, "W_F": 0, "C": 1, "C_F": 1}}}}
        pop = pop_of()
        self.assertEqual(p1p5.reconcile_design_log(pop, summary)["mismatches"], 0)
        summary["per_template_P"]["curated"]["aggregation"]["C_F"] = 1
        summary["attempts"] = 10
        self.assertEqual(p1p5.reconcile_design_log(pop, summary)["mismatches"], 2)



# -- a two-graph, two-agent world with a refused result and bank strata that differ ------------------------------------

GA, GB = "uniprot-sp-2026_03", "rhea-142"
# graph, run, model, template, correct (None = no output), curated, census-t5, union
WORLD2 = [
    (GA, "a1", QWEN, "aggregation", False, FL, FP, FL),
    (GA, "a2", QWEN, "aggregation", False, FP, FP, FP),
    (GA, "a3", SONNET, "numeric_filter", False, FL, FL, FL),
    (GA, "a6", SONNET, "aggregation", False, UNK, FP, UNK),
    (GA, "a4", SONNET, "numeric_filter", True, FL, FP, FL),
    (GA, "a5", QWEN, "aggregation", True, FP, FP, FP),
    (GB, "b1", QWEN, "aggregation", False, FP, FP, FP),
    (GB, "b2", SONNET, "type_members", False, FL, FP, FL),
    (GB, "b3", SONNET, "type_members", True, FL, FL, FL),
    (GB, "b4", QWEN, "type_members", True, FP, FP, FP),
    (GB, "b5", QWEN, "type_members", None, None, None, None),
]
RESULT2 = {"a1": "rows", "a2": "refused", "a3": "rows", "a6": "rows", "b1": "empty", "b2": "rows"}
BANK2 = [  # item, correct, result_rows, curated, census-t5
    ("b1", False, 5, FL, FP), ("b2", False, 3, FP, FP), ("b3", False, 0, FP, FP), ("b4", False, None, FP, FP),
    ("b5", True, 2, FL, FP), ("b6", True, 2, FP, FP)]


def mk_rows2(world=WORLD2) -> list[dict]:
    rows = []
    for g, run, model, tmpl, correct, cur, cen, un in world:
        for arm, st in (("curated", cur), ("census-t5", cen), ("union", un)):
            out = correct is not None
            rows.append({"run_id": run, "graph": g, "question_id": f"q-{run}", "template": tmpl, "split": "dev",
                         "model": model, "cohort": "none", "variant": "primary", "arm": arm, "output": out,
                         "correct": correct, "status": st})
    return rows


def acc_from_world(world, graph) -> dict:
    """accounting.json as another program would write it, counted here with plain loops from the world's table."""
    def letter(st):
        return {FL: "F", FP: "P", VP: "P", UNK: "U"}[st]
    acc: dict = {}
    for g, run, model, tmpl, correct, cur, cen, un in world:
        if g != graph or correct is None:
            continue
        for arm, st in (("curated", cur), ("census-t5", cen), ("union", un)):
            slot = acc.setdefault(model, {}).setdefault("none", {}).setdefault(arm, {}).setdefault(PRIM, {
                "cells": {k: 0 for k in ("C_F", "C_P", "C_U", "W_F", "W_P", "W_U")}, "strata": {"template": {}}})
            key = ("C_" if correct else "W_") + letter(st)
            slot["cells"][key] += 1
            tc = slot["strata"]["template"].setdefault(
                tmpl, {"cells": {k: 0 for k in ("C_F", "C_P", "C_U", "W_F", "W_P", "W_U")}})
            tc["cells"][key] += 1
    return {"accounting": acc, "meta": {}}


PRIM = "primary"


def bank2_rows() -> list[dict]:
    rows = []
    for item, correct, rr, cur, cen in BANK2:
        for arm, st in (("curated", cur), ("census-t5", cen)):
            rows.append({"run_id": item, "graph": GA, "split": "dev", "variant": "primary", "arm": arm, "output": True,
                         "correct": correct, "status": st, "result_rows": rr, "cohort": "bank"})
    return rows


def pop2(with_judge=False, with_accounting=False, result=True, excluded=None) -> p1p5.Population:
    units = p1p5.build_units(mk_rows2(), "dev")
    jd = None
    judges: list[str] = []
    if with_judge:
        truth, selection, verdict, raw = {}, {}, {}, {}
        for g, run, model, tmpl, correct, *_ in WORLD2:
            if correct is None:
                continue
            k = f"output:{run}"
            truth[k] = "right" if correct else "wrong"
            selection[k] = {"key": k, "model": model, "result": RESULT2.get(run, "rows")}
            verdict[(k, SONNET)] = "reject" if run in ("a1", "a3", "b2", "a4") else "accept"
        verdict[("output:a1", HAIKU)] = "reject"
        raw = {pair: v for pair, v in verdict.items()}
        jd = p1p5.JudgeData(truth, selection, verdict, raw)
        judges = [SONNET, HAIKU, QWEN]
    acc = {GA: acc_from_world(WORLD2, GA), GB: acc_from_world(WORLD2, GB)} if with_accounting else {}
    return p1p5.Population("w2", "two-graph world", True, units,
                           {GA: {"curated": "d-cur", "census-t5": "d-other"}, GB: {"curated": "d-cur", "census-t5": "d-other"}},
                           {GA: {"union": ["curated", "census-t5"]}, GB: {"union": ["curated", "census-t5"]}},
                           dict(RESULT2) if result else {}, jd, judges, acc, None, excluded or {})


def bank2() -> dict:
    counts = p1p5.bank_counts(bank2_rows(), "dev")
    return {"graph": GA, "counts": counts, "digests": {"curated": "d-cur", "census-t5": "d-cen"}, "bank_rates": {},
            "plan_code": None, "items": 6}


def blocks(pop, bank=None) -> dict:
    return {"P1": p1p5.p1(pop, bank), "P2": p1p5.p2(pop), "P3": p1p5.p3(pop), "P4": p1p5.p4(pop), "P5": p1p5.p5(pop)}


def e77_from_acc(acc: dict, group: list[str]) -> dict:
    """An design_log_summary.json as the analysis would write it: counts per model and source, and per template pooled
    over the group of models (built here from the test-side accounting)."""
    e77 = {"sources": {}, "per_template_P": {}, "attempts": 0, "groups": {"P": group}}
    for model, per_cohort in acc["accounting"].items():
        for arm, per_var in per_cohort["none"].items():
            c = per_var[PRIM]["cells"]
            e77["sources"].setdefault(arm, {"by_model": {}})["by_model"][model] = {"counts": {
                "W": c["W_F"] + c["W_P"] + c["W_U"], "W_F": c["W_F"], "C": c["C_F"] + c["C_P"] + c["C_U"],
                "C_F": c["C_F"]}}
            if model not in group:
                continue
            for t, d in per_var[PRIM]["strata"]["template"].items():
                x = d["cells"]
                slot = e77["per_template_P"].setdefault(arm, {}).setdefault(t, {k: 0 for k in ("W", "W_F", "C", "C_F")})
                slot["W"] += x["W_F"] + x["W_P"] + x["W_U"]
                slot["W_F"] += x["W_F"]
                slot["C"] += x["C_F"] + x["C_P"] + x["C_U"]
                slot["C_F"] += x["C_F"]
    return e77


class TwoGraphs(unittest.TestCase):
    def test_p1_denominators_refused_and_two_stratum_bank(self):
        r = p1p5.p1(pop2(), bank2())
        a = r[GA]
        self.assertEqual((a["wrong_outputs"], a["wrong_with_result_rows"]), (4, 3))  # the refused output is not non-empty
        self.assertEqual(a["wrong_by_result"], {"rows": 3, "empty": 0, "refused": 1})
        self.assertEqual(a["wrong_by_result_by_agent"], {QWEN: {"rows": 1, "empty": 0, "refused": 1},
                                                         SONNET: {"rows": 2, "empty": 0, "refused": 0}})
        cur = next(e for e in a["sources"] if e["source"] == "curated")
        self.assertEqual((cur["real_wrong_all"]["num"], cur["real_wrong_all"]["den"]), (2, 4))  # U in the denominator
        self.assertEqual((cur["real_wrong_result_rows"]["num"], cur["real_wrong_result_rows"]["den"]), (2, 3))
        bank = cur["bank"]
        self.assertEqual((bank["returns_rows"]["W"], bank["returns_rows"]["W_F"]), (2, 1))  # 0.5
        self.assertEqual((bank["all_items"]["W"], bank["all_items"]["W_F"]), (4, 1))  # 0.25: a different number
        self.assertAlmostEqual(cur["ratio_real_all_over_bank_returns_rows"], 1.0)  # 0.5 / 0.5 (0.25 would give 2.0)
        self.assertAlmostEqual(cur["ratio_real_result_rows_over_bank_returns_rows"], (2 / 3) / 0.5)
        self.assertEqual(cur["real_wrong_by_agent"][QWEN]["den"], 2)
        b = r[GB]
        self.assertEqual((b["wrong_outputs"], b["wrong_with_result_rows"], b["wrong_by_result"]["empty"]), (2, 1, 1))
        self.assertTrue(all(e["bank"].startswith("not replayed") for e in b["sources"]))  # no bank for Rhea
        self.assertEqual(set(a["not_replayed"]), set(p1p5.SOURCE_UNIVERSE) - {"curated", "census-t5"})
        self.assertEqual(set(b["not_replayed"]), set(a["not_replayed"]))
        # a bank recall of 0 gives 'not estimable', not a missing cell
        bk = bank2()
        bk["counts"]["curated"]["returns_rows"]["W_F"] = 0
        cur0 = next(e for e in p1p5.p1(pop2(), bk)[GA]["sources"] if e["source"] == "curated")
        self.assertEqual(cur0["ratio_real_all_over_bank_returns_rows"], p1p5.NOT_ESTIMABLE)

    def test_p1_missing_or_unknown_result_is_an_error(self):
        short = dict(RESULT2)
        del short["a6"]
        p = pop2()
        p.result = short
        with self.assertRaises(p1p5.InputError):
            p1p5.p1(p, None)
        p.result = dict(RESULT2, a1="weird")
        with self.assertRaises(p1p5.InputError):
            p1p5.p1(p, None)

    def test_pooled_rows_equal_the_sum_of_the_graphs_and_graphs_stay_apart(self):
        p3 = p1p5.p3(pop2())
        for arm in ("curated", "census-t5", "union"):
            for k in ("C", "C_F"):
                self.assertEqual(p3[p1p5.ALL_GRAPHS]["sources"][arm][k], p3[GA]["sources"][arm][k] + p3[GB]["sources"][arm][k])
        self.assertEqual((p3[p1p5.ALL_GRAPHS]["sources"]["curated"]["C"], p3[p1p5.ALL_GRAPHS]["sources"]["curated"]["C_F"]), (4, 2))
        p4 = p1p5.p4(pop2())
        u = {g: p4[g]["unions"]["union"]["union"] for g in p4}
        self.assertEqual((u[GA]["W"], u[GA]["W_F"], u[GB]["W"], u[GB]["W_F"]), (4, 2, 2, 1))
        self.assertEqual((u[p1p5.ALL_GRAPHS]["W"], u[p1p5.ALL_GRAPHS]["W_F"]), (6, 3))
        self.assertEqual(set(p4[p1p5.ALL_GRAPHS]["not_replayed"]), set(p4[GA]["not_replayed"]))

    def test_p2_graphs_are_not_mixed(self):
        r = p1p5.p2(pop2())
        self.assertEqual(set(r[GA]["templates"]), {"aggregation", "numeric_filter"})
        self.assertEqual(set(r[GB]["templates"]), {"type_members", "aggregation"})
        self.assertEqual(r[GA]["templates"]["aggregation"]["wrong_outputs"], 3)  # GB's wrong aggregation output stays out
        self.assertEqual((r[GA]["templates"]["aggregation"]["wrong_outputs"], r[GA]["templates"]["aggregation"]["sources"]["curated"]["W_F"]), (3, 1))
        self.assertEqual(r[GA]["templates"]["aggregation"]["sources"]["curated"]["W"], 3)  # a6's unknown stays in W
        f = r[GA]["focus"]
        self.assertEqual((f["wrong_outputs"], f["sources"]["curated"]["W"], f["sources"]["curated"]["W_F"]), (4, 4, 2))
        self.assertEqual(r[GB]["focus"]["templates"], ["type_members"])
        self.assertEqual((r[GB]["focus"]["wrong_outputs"], r[GB]["focus"]["sources"]["curated"]["W_F"]), (1, 1))
        self.assertIn("curated+doc", r[GA]["not_replayed"])

    def test_p5_filters_by_agent_and_graph_and_names_not_replayed_sources(self):
        r = p1p5.p5(pop2(with_judge=True))
        self.assertEqual(r[QWEN][p1p5.ALL_GRAPHS]["wrong_outputs"], 3)  # a1, a2, b1
        self.assertEqual(r[SONNET][p1p5.ALL_GRAPHS]["wrong_outputs"], 3)  # a3, a6, b2
        q = r[QWEN][p1p5.ALL_GRAPHS]["sources_on_all_wrong"]["curated"]
        self.assertEqual((q["num"], q["den"]), (1, 3))
        s = r[SONNET][p1p5.ALL_GRAPHS]["sources_on_all_wrong"]["curated"]
        self.assertEqual((s["num"], s["den"]), (2, 3))
        g = r[QWEN][GA]["sources_on_all_wrong"]["curated"]
        self.assertEqual((g["num"], g["den"]), (1, 2))
        self.assertIn("generated-2", r[QWEN][GA]["not_replayed"])
        jq = r[QWEN][GA]["judges"][SONNET]
        self.assertEqual((jq["coverage"], jq["recall"]["num"], jq["recall"]["den"]), ("complete", 1, 2))  # a1 rejected
        hq = r[QWEN][GA]["judges"][HAIKU]
        self.assertEqual(hq["coverage"], "partial 1 of 2")

    def test_population_gate_catches_a_short_selection_and_stray_calls(self):
        p = pop2(with_judge=True)
        self.assertTrue(p1p5.population_check(p, None)["against_judge_calibration"]["equal"])
        del p.judge.selection["output:a4"]  # the selection no longer covers every output
        jc = p1p5.population_check(p, None)["against_judge_calibration"]
        self.assertFalse(jc["selection_keys_equal_matrix_outputs"])
        self.assertFalse(jc["equal"])
        p = pop2(with_judge=True)
        p.judge.verdict[("output:zzz", SONNET)] = "reject"
        jc = p1p5.population_check(p, None)["against_judge_calibration"]
        self.assertEqual(jc["judge_calls_for_keys_outside_the_population"][SONNET], 1)
        self.assertFalse(jc["equal"])
        p = pop2(with_judge=True)
        p.judge.selection["output:a1"]["question_id"] = "some-other-question"
        self.assertFalse(p1p5.population_check(p, None)["against_judge_calibration"]["equal"])

    def test_judge_coverage_counts_only_population_outputs(self):
        p = pop2(with_judge=True)
        p.judge.verdict[("output:zzz", SONNET)] = "reject"
        p.judge.raw[("output:zzz", SONNET)] = "reject"
        c = p1p5.judge_coverage(p)[SONNET]
        self.assertEqual(c["calls_for_population_outputs"], 10)
        self.assertEqual(c["raw_verdicts"], {"accept": 6, "reject": 4})

    def test_reconcile_printed_matches_then_catches_every_kind_of_change(self):
        import copy
        p = pop2(with_judge=True, with_accounting=True)
        pr = blocks(p, bank2())
        good = p1p5.reconcile_printed(p, pr)
        self.assertEqual((good["mismatches"], good["match"]), (0, True))
        self.assertGreater(good["compared"], 100)

        def broken(mutate):
            b = copy.deepcopy(pr)
            mutate(b)
            r = p1p5.reconcile_printed(p, b)
            return r["mismatches"]

        def p1_all(b):
            b["P1"][GA]["sources"][0]["real_wrong_all"]["num"] += 1

        def p1_agent(b):
            b["P1"][GB]["sources"][0]["real_wrong_by_agent"][SONNET]["den"] += 1

        def p1_subset(b):
            b["P1"][GA]["sources"][0]["real_wrong_result_rows"]["den"] = 99

        def p2_cell(b):
            b["P2"][GA]["templates"]["aggregation"]["sources"]["curated"]["W_F"] += 1

        def p2_focus(b):
            b["P2"][GA]["focus"]["sources"]["curated"]["W"] -= 1

        def p3_pooled(b):
            b["P3"][p1p5.ALL_GRAPHS]["sources"]["curated"]["C"] += 1

        def p3_agent(b):
            b["P3"][GA]["sources"]["union"]["by_agent"][QWEN]["C_F"] += 1

        def p4_union(b):
            b["P4"][GB]["unions"]["union"]["union"]["W_F"] += 1

        def p4_one_sided(b):
            b["P4"][GA]["unions"]["union"]["singles"]["curated"]["wrong_flagged_by_union_not_by_source"] += 1

        def p5_source(b):
            b["P5"][QWEN][GA]["sources_on_all_wrong"]["curated"]["num"] += 1

        def p5_subset(b):
            b["P5"][QWEN][GA]["judges"][SONNET]["sources_on_the_judged_outputs"]["curated"]["num"] += 1

        def p1_wrong_count(b):
            b["P1"][GA]["wrong_outputs"] += 1

        def p3_correct_count(b):
            b["P3"][GB]["correct_outputs"] += 1

        def p5_wrong_count(b):
            b["P5"][SONNET][GA]["wrong_outputs"] -= 1

        for f in (p1_all, p1_agent, p1_subset, p2_cell, p2_focus, p3_pooled, p3_agent, p4_union, p4_one_sided,
                  p5_source, p5_subset, p1_wrong_count, p3_correct_count, p5_wrong_count):
            self.assertGreaterEqual(broken(f), 1, f.__name__)

    def test_nonempty_denominators_and_structure_are_reconciled(self):
        import copy
        p = pop2(with_judge=True, with_accounting=True)
        pr = blocks(p, bank2())
        tables = {"table_2_by_result": [
            {"graph": "uniprot", "agent": QWEN, "result": "rows", "judge": "Sonnet", "wrong": 1},
            {"graph": "uniprot", "agent": QWEN, "result": "refused", "judge": "Sonnet", "wrong": 1},
            {"graph": "uniprot", "agent": SONNET, "result": "rows", "judge": "Sonnet", "wrong": 2},
            {"graph": "rhea", "agent": QWEN, "result": "empty", "judge": "Sonnet", "wrong": 1},
            {"graph": "rhea", "agent": SONNET, "result": "rows", "judge": "Sonnet", "wrong": 1}]}
        self.assertTrue(p1p5.reconcile_nonempty(p, pr, tables)["match"])
        b = copy.deepcopy(pr)
        b["P1"][GA]["wrong_with_result_rows"] = 4  # as if the refused output were counted as non-empty
        self.assertGreaterEqual(p1p5.reconcile_nonempty(p, b, tables)["mismatches"], 1)
        b = copy.deepcopy(pr)
        b["P1"][GA]["sources"][0]["real_wrong_result_rows"]["den"] = 4
        self.assertGreaterEqual(p1p5.reconcile_nonempty(p, b, tables)["mismatches"], 1)
        # a pooled row that is missing is a mismatch
        b = copy.deepcopy(pr)
        del b["P3"][p1p5.ALL_GRAPHS]
        self.assertGreaterEqual(p1p5.reconcile_printed(p, b)["mismatches"], 1)
        b = copy.deepcopy(pr)
        del b["P5"][QWEN][p1p5.ALL_GRAPHS]
        self.assertGreaterEqual(p1p5.reconcile_printed(p, b)["mismatches"], 1)

    def test_p1_bank_columns_and_ratios_are_reconciled(self):
        import copy
        bk = bank2()
        for arm, strata in bk["counts"].items():
            bk["bank_rates"][arm] = {"primary": {st: {k: p1p5.cells_ratios(c)[k] for k in ("W", "W_F", "C", "C_F")}
                                                 for st, c in strata.items()}}
        pr = blocks(pop2(), bk)
        self.assertEqual(p1p5.reconcile_p1_bank(pr, bk)["mismatches"], 0)
        self.assertEqual(p1p5.reconcile_p1_bank(pr, bk)["compared"], 4)
        for mutate in (lambda b: b["P1"][GA]["sources"][0].__setitem__("ratio_real_all_over_bank_returns_rows", 2.0),
                       lambda b: b["P1"][GA]["sources"][0]["bank"]["returns_rows"].__setitem__("W_F", 2),
                       lambda b: b["P1"][GA]["sources"][0]["bank"]["all_items"].__setitem__("W", 5)):
            b = copy.deepcopy(pr)
            mutate(b)
            self.assertGreaterEqual(p1p5.reconcile_p1_bank(b, bk)["mismatches"], 1)

    def test_reconcile_printed_against_an_design_log_summary(self):
        # a one-graph, panel-only population with the reference given as design_log_summary.json would give it
        units = p1p5.build_units([r for r in mk_rows2() if r["graph"] == GA], "dev")
        p = pop2()
        p.units = units
        e77 = e77_from_acc(acc_from_world(WORLD2, GA), [SONNET, QWEN])
        p.design_log = e77
        r = p1p5.reconcile_printed(p, blocks(p))
        self.assertEqual((r["mismatches"], r["match"], r["against"]), (0, True, "design_log_summary.json"))
        e77["per_template_P"]["curated"]["aggregation"]["W_F"] += 1
        self.assertGreaterEqual(p1p5.reconcile_printed(p, blocks(p))["mismatches"], 1)

    def test_reconcile_nonempty_against_tables_table_2(self):
        p = pop2(with_judge=True)
        rows = [{"graph": "uniprot", "agent": QWEN, "result": "rows", "judge": "Sonnet", "wrong": 1},
                {"graph": "uniprot", "agent": QWEN, "result": "refused", "judge": "Sonnet", "wrong": 1},
                {"graph": "uniprot", "agent": SONNET, "result": "rows", "judge": "Sonnet", "wrong": 2},
                {"graph": "rhea", "agent": QWEN, "result": "empty", "judge": "Sonnet", "wrong": 1},
                {"graph": "rhea", "agent": SONNET, "result": "rows", "judge": "Sonnet", "wrong": 1}]
        pr = blocks(p, None)
        r = p1p5.reconcile_nonempty(p, pr, {"table_2_by_result": rows})
        self.assertEqual((r["mismatches"], r["match"]), (0, True))
        rows[1]["result"] = "rows"  # the tables say the refused output returned rows
        r = p1p5.reconcile_nonempty(p, pr, {"table_2_by_result": rows})
        self.assertGreaterEqual(r["mismatches"], 1)

    def test_selection_without_result_is_refused_at_load(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            w = EndToEnd()
            cfg = w.world(d)
            sel = json.loads((d / "selection.json").read_text())
            del sel["outputs"][0]["result"]
            (d / "selection.json").write_text(json.dumps(sel))
            self.assertEqual(p1p5.main(["--config", str(cfg), "--out", str(d / "out")]), 2)

    def test_judge_verdicts_other_than_accept_reject_are_refused(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            w = EndToEnd()
            w.world(d)
            (d / "calls.jsonl").write_text(json.dumps({"item": {"key": "output:r1"}, "judge_model": SONNET,
                                                       "judge": {"verdict": "maybe"}}) + "\n")
            with self.assertRaises(p1p5.InputError):
                p1p5.load_judge(d / "truth.jsonl", d / "selection.json", d / "calls.jsonl")


class EndToEnd(unittest.TestCase):
    def write(self, d: Path, name: str, obj, gz=False, lines=False):
        p = d / name
        data = ("".join(json.dumps(x) + "\n" for x in obj) if lines else json.dumps(obj))
        if gz or name.endswith(".gz"):
            with gzip.open(p, "wt") as f:
                f.write(data)
        else:
            p.write_text(data)
        return name

    def world(self, d: Path, split="dev", bad_accounting=False):
        acc = acc_from_world([(G,) + w for w in WORLD], G)
        for m in acc["accounting"].values():
            for arms in m["none"].values():
                arms["primary"]["N_out"] = 0  # not read by the tool's reconciliations of the printed blocks
        for run in ("curated", "census-t5", "union"):
            for model in (SONNET, QWEN):
                acc["accounting"][model]["none"][run]["primary"]["N_out"] = sum(
                    1 for w in WORLD if w[1] == model and w[3] is not None)
        acc["meta"] = {"source_digests": {"curated": "d-cur", "census-t5": "d-other"},
                       "unions": {"union": ["curated", "census-t5"]}}
        if bad_accounting:
            acc["accounting"][QWEN]["none"]["curated"]["primary"]["cells"]["W_F"] = 5
        self.write(d, "m.jsonl.gz", mk_rows(split=split), lines=True)
        self.write(d, "acc.json", acc)
        bank = bank_of()
        self.write(d, "bank.jsonl.gz", bank_rows(), lines=True)
        self.write(d, "plan.json.gz", {"split": "dev", "code": {"git_commit": "x"},
                                       "sources": {a: {"digest": g} for a, g in bank["digests"].items()}})
        rates = {arm: {"primary": {s: {"W": p1p5.cells_ratios(c)["W"], "W_F": c["W_F"], "C": p1p5.cells_ratios(c)["C"],
                                       "C_F": c["C_F"]} for s, c in strata.items()}}
                 for arm, strata in bank["counts"].items()}
        self.write(d, "run.json.gz", {"bank_rates": rates, "items": 6})
        jd = judge_data()
        self.write(d, "truth.jsonl", [{"key": k, "truth": v, "stratum": None} for k, v in jd.truth.items()], lines=True)
        self.write(d, "selection.json", {"outputs": list(jd.selection.values())})
        calls = []
        for (k, j), v in jd.verdict.items():
            calls.append({"item": {"key": k}, "judge_model": j,
                          "judge": None if v == "error" else {"verdict": None if v == "unparsed" else v}})
        self.write(d, "calls.jsonl", calls, lines=True)
        t = []
        for g in ("uniprot", "all"):
            t.append({"graph": g, "agent": QWEN, "judge": "Sonnet", "wrong": 3, "right": 1,
                      "recall": {"reject": 1, "unparsed": 1}, "false_reject": {"reject": 0}})
            t.append({"graph": g, "agent": SONNET, "judge": "Sonnet", "wrong": 1, "right": 2,
                      "recall": {"reject": 1, "unparsed": 0}, "false_reject": {"reject": 1}})
        t2 = [{"graph": "uniprot", "agent": QWEN, "result": "rows", "judge": "Sonnet", "wrong": 2},
              {"graph": "uniprot", "agent": QWEN, "result": "empty", "judge": "Sonnet", "wrong": 1},
              {"graph": "uniprot", "agent": SONNET, "result": "rows", "judge": "Sonnet", "wrong": 1}]
        self.write(d, "tables.json", {"table_1_by_graph_round_agent_judge": t, "table_2_by_result": t2})
        cfg = {"expected_split": "dev", "registered": {"outputs": 7, "wrong": 4, "wrong_by_agent": {SONNET: 1, QWEN: 3}},
               "populations": [{"name": "syn", "title": "synthetic", "registered": True,
                                "matrices": [{"matrix": "m.jsonl.gz", "accounting": "acc.json"}],
                                "judge": {"truth": "truth.jsonl", "selection": "selection.json", "calls": "calls.jsonl",
                                          "tables": "tables.json", "judges": [SONNET, HAIKU, QWEN]}}],
               "bank": {"graph": G, "matrix": "bank.jsonl.gz", "plan": "plan.json.gz", "run": "run.json.gz"}}
        self.write(d, "cfg.json", cfg)
        return d / "cfg.json"

    def test_run_ok_and_outputs(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            cfg = self.world(d)
            self.assertEqual(p1p5.main(["--config", str(cfg), "--out", str(d / "out")]), 0)
            res = json.loads((d / "out" / "p1p5-dev.json").read_text())
            self.assertEqual(res["result"], "OK")
            self.assertEqual(len(res["tool"]["sha256"]), 64)
            self.assertIn("not replayed", (d / "out" / "p1p5-dev.txt").read_text())
            self.assertNotIn("holds", (d / "out" / "p1p5-dev.txt").read_text().lower().replace("no verdict", ""))

    def frozen_world(self, d: Path, tree=None):
        """The synthetic world plus a frozen_sources block (inputs file; dir `sources`, created when `tree` is given: {file name: bytes} under the graph's directory)."""
        cfg = json.loads(self.world(d).read_text())
        data = b"abc"
        self.write(d, "audit-inputs.json", {"graphs": {G: {"sources": {"curated": {"sha256": p1p5.hashlib.sha256(data).hexdigest()}}}}})
        cfg["frozen_sources"] = {"inputs": "audit-inputs.json", "dir": "sources"}
        if tree is not None:
            (d / "sources" / G).mkdir(parents=True)
            for name, blob in tree.items(): (d / "sources" / G / name).write_bytes(blob)
        self.write(d, "cfg2.json", cfg)
        return d / "cfg2.json"

    def test_frozen_sources_absent_are_reported_not_checked_and_do_not_fail(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t); cfg = self.frozen_world(d)
            self.assertEqual(p1p5.main(["--config", str(cfg), "--out", str(d / "out")]), 0)
            res = json.loads((d / "out" / "p1p5-dev.json").read_text())
            self.assertEqual(res["result"], "OK")
            self.assertEqual(res["reconciliation"]["source files against the frozen tree"], p1p5.NOT_CHECKED)
            self.assertIn("source files against the frozen tree: NOT CHECKED (sources not released)", (d / "out" / "p1p5-dev.txt").read_text())

    def test_frozen_sources_present_stay_strict(self):
        with tempfile.TemporaryDirectory() as t:   # a source that differs from the replay's still FAILS the run (exit 1)
            d = Path(t); cfg = self.frozen_world(d, {"curated.json.gz": b"different"})
            self.assertEqual(p1p5.main(["--config", str(cfg), "--out", str(d / "out")]), 1)
            self.assertIn("DIFFERENT", (d / "out" / "p1p5-dev.txt").read_text())
        with tempfile.TemporaryDirectory() as t:   # an empty frozen tree is not 'absent': the sources are missing from it, so it fails too
            d = Path(t); cfg = self.frozen_world(d, {})
            self.assertEqual(p1p5.main(["--config", str(cfg), "--out", str(d / "out")]), 1)
        with tempfile.TemporaryDirectory() as t:   # equal sources pass and are reported as equal
            d = Path(t); cfg = self.frozen_world(d, {"curated.json.gz": b"abc"})
            self.assertEqual(p1p5.main(["--config", str(cfg), "--out", str(d / "out")]), 0)
            self.assertIn("=equal", (d / "out" / "p1p5-dev.txt").read_text())

    def test_mismatch_exit_1_and_test_split_exit_2(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            cfg = self.world(d, bad_accounting=True)
            self.assertEqual(p1p5.main(["--config", str(cfg), "--out", str(d / "out")]), 1)
            self.assertEqual(json.loads((d / "out" / "p1p5-dev.json").read_text())["result"], "FAIL")
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            cfg = self.world(d, split="test")
            self.assertEqual(p1p5.main(["--config", str(cfg), "--out", str(d / "out")]), 2)
            self.assertFalse((d / "out").exists())

    def test_registered_population_difference_makes_the_run_fail(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            cfg = self.world(d)
            c = json.loads(cfg.read_text())
            c["registered"]["outputs"] = 8
            cfg.write_text(json.dumps(c))
            self.assertEqual(p1p5.main(["--config", str(cfg), "--out", str(d / "out")]), 1)
            res = json.loads((d / "out" / "p1p5-dev.json").read_text())
            self.assertEqual(res["populations"]["syn"]["population_check"]["against_registered"]["outputs"]["difference"], -1)

    def test_model_group_filter_and_reconciled_attempts(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            cfg = self.world(d)
            # the summary says: 9 attempts in all, group P = Qwen only; the analysed set keeps Qwen's 5 attempts
            e77 = e77_from_acc(acc_from_world([(G,) + w for w in WORLD], G), [QWEN])
            e77["attempts"] = 9
            self.write(d, "e77.json", e77)
            c = json.loads(cfg.read_text())
            c["registered"] = None
            c["populations"][0].update({"model_group": "P", "design_log_summary": "e77.json", "registered": False})
            c["populations"][0].pop("judge")
            cfg.write_text(json.dumps(c))
            # the accounting file covers Sonnet too: drop its reconciliation, which the filter makes inapplicable
            c["populations"][0]["matrices"][0]["reconcile_accounting"] = False
            cfg.write_text(json.dumps(c))
            self.assertEqual(p1p5.main(["--config", str(cfg), "--out", str(d / "out")]), 0)
            res = json.loads((d / "out" / "p1p5-dev.json").read_text())
            pc = res["populations"]["syn"]["population_check"]
            self.assertEqual(pc["models_left_out_of_the_analysed_set_attempts"], {SONNET: 4})
            self.assertEqual((pc["attempts"], pc["wrong_by_agent"]), (5, {QWEN: 3}))
            self.assertEqual(res["reconciliation"]["syn: pooled counts against design_log_summary.json"]["mismatches"], 0)

    def test_frozen_source_check(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            (d / "g").mkdir()
            (d / "g" / "curated.json.gz").write_bytes(b"abc")
            (d / "g" / "curated+doc.json.gz").write_bytes(b"other")
            sha = p1p5.sha256_file(d / "g" / "curated.json.gz")
            inputs = {"graphs": {"g": {"sources": {"curated": {"sha256": sha}, "census-t5": {"sha256": "0" * 64}}}}}
            r = {(e["graph"], e["source"]): e["equal"] for e in p1p5.frozen_source_check(inputs, d)}
            self.assertEqual(r, {("g", "curated"): True, ("g", "census-t5"): False})



class Sensitivity(unittest.TestCase):
    """The registered population without the records the audit's population file labels with one frame label."""

    def build(self, d: Path, labelled=("r1", "r5"), expect=2, frame="bad"):
        w = EndToEnd()
        cfg_path = w.world(d)
        rows = []
        for i in range(1, 10):
            run = f"r{i}"
            rows.append({"run_id": run, "graph": G, "frame": frame if run in labelled else "dev", "source": "curated"})
        w.write(d, "population.jsonl.gz", rows, lines=True)
        cfg = json.loads(cfg_path.read_text())
        sens = json.loads(json.dumps(cfg["populations"][0]))
        sens.update({"name": "sens", "title": "sensitivity", "registered": False, "kind": "sensitivity",
                     "drop_runs": {"file": "population.jsonl.gz", "frame": "bad", "graph": G, "expect_records": expect}})
        cfg["populations"].append(sens)
        cfg_path.write_text(json.dumps(cfg))
        return cfg_path

    def test_sensitivity_counts_and_every_reconciliation(self):
        with tempfile.TemporaryDirectory() as t_:
            d = Path(t_)
            cfg = self.build(d)
            self.assertEqual(p1p5.main(["--config", str(cfg), "--out", str(d / "out")]), 0)
            res = json.loads((d / "out" / "p1p5-dev.json").read_text())
            self.assertEqual(res["result"], "OK")
            full, sens = res["populations"]["syn"], res["populations"]["sens"]
            self.assertEqual(full["population_check"]["outputs"], 7)  # the registered population is untouched
            pc = sens["population_check"]
            self.assertEqual((pc["attempts"], pc["outputs"], pc["wrong"], pc["wrong_by_agent"]), (7, 5, 3, {QWEN: 2, SONNET: 1}))
            dr = pc["dropped"]
            self.assertEqual((dr["records"], dr["outputs"], dr["wrong"], dr["wrong_by_agent"]), (2, 2, 1, {QWEN: 1}))
            self.assertTrue(pc["registered_minus_dropped"]["all_equal"])
            self.assertEqual(pc["against_registered"]["note"], "a sensitivity of the registered population")
            self.assertEqual(pc["against_judge_calibration"]["keys_removed_with_the_dropped_records"], 2)
            self.assertTrue(pc["against_judge_calibration"]["equal"])
            p1_ = sens["P1"][G]
            self.assertEqual((p1_["wrong_outputs"], p1_["wrong_by_result"]), (3, {"rows": 2, "empty": 1, "refused": 0}))
            self.assertEqual(full["P1"][G]["wrong_outputs"], 4)
            q = sens["P5"][QWEN][G]
            self.assertEqual(q["wrong_outputs"], 2)  # r2, r3 (r1 is dropped)
            self.assertEqual(q["judges"][SONNET]["of"], 2)
            for k in ("printed P1-P5 blocks against accounting.json less the dropped records",
                      "non-empty denominators against tables.json table 2", "judge counts against tables.json",
                      "six cells against accounting.json"):
                self.assertEqual(res["reconciliation"][f"sens: {k}"]["mismatches"], 0, k)
                self.assertTrue(res["reconciliation"][f"sens: {k}"]["match"], k)
            self.assertIn("DROPPED from the registered population", (d / "out" / "p1p5-dev.txt").read_text())

    def test_wrong_record_count_or_unknown_record_is_refused(self):
        with tempfile.TemporaryDirectory() as t_:
            d = Path(t_)
            cfg = self.build(d, expect=3)
            self.assertEqual(p1p5.main(["--config", str(cfg), "--out", str(d / "out")]), 2)
        with tempfile.TemporaryDirectory() as t_:
            d = Path(t_)
            cfg = self.build(d, labelled=("r1", "r5", "r99"), expect=3)
            self.assertEqual(p1p5.main(["--config", str(cfg), "--out", str(d / "out")]), 2)

    def test_dropped_outputs_must_be_in_the_judge_calibration(self):
        with tempfile.TemporaryDirectory() as t_:
            d = Path(t_)
            cfg = self.build(d)
            truth = [json.loads(x) for x in (d / "truth.jsonl").read_text().splitlines()]
            (d / "truth.jsonl").write_text("".join(json.dumps(x) + "\n" for x in truth if x["key"] != "output:r5"))
            self.assertEqual(p1p5.main(["--config", str(cfg), "--out", str(d / "out")]), 2)

    def test_the_subtraction_is_what_makes_the_reconciliations_match(self):
        with tempfile.TemporaryDirectory() as t_:
            d = Path(t_)
            cfg = self.build(d)
            c = json.loads(cfg.read_text())
            shas: dict = {}
            pop = p1p5.load_population(c["populations"][1], d, "dev", shas, c["populations"][1]["judge"])
            pr = {"P1": p1p5.p1(pop, None), "P2": p1p5.p2(pop), "P3": p1p5.p3(pop), "P4": p1p5.p4(pop),
                  "P5": p1p5.p5(pop)}
            self.assertEqual(p1p5.reconcile_printed(pop, pr)["mismatches"], 0)
            pop.dropped = []  # without the dropped records' counts taken out of the reference
            self.assertGreater(p1p5.reconcile_printed(pop, pr)["mismatches"], 0)


class Text(unittest.TestCase):
    def test_render_covers_every_prediction(self):
        pop = pop_of(judge=judge_data(), judges=(SONNET, HAIKU, QWEN),
                     result={"r1": "rows", "r2": "rows", "r3": "empty", "r4": "rows"})
        res = {"tool": {"version": "x"}, "result": "OK",
               "populations": {"syn": {"title": "t", "population_check": p1p5.population_check(pop, None),
                                       "P1": p1p5.p1(pop, bank_of()), "P2": p1p5.p2(pop), "P3": p1p5.p3(pop),
                                       "P4": p1p5.p4(pop), "P5": p1p5.p5(pop), "judge_coverage": p1p5.judge_coverage(pop)}},
               "reconciliation": {}}
        text = p1p5.render_text(res)
        for tag in ("P1 ", "P2 ", "P3 ", "P4 ", "P5 ", "not replayed", "partial 2 of 3", "not run"):
            self.assertIn(tag, text)


if __name__ == "__main__":
    unittest.main()
