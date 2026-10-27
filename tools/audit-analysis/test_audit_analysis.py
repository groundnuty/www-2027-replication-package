"""The audit's analysis (harness/analysis/audit_analysis.py): Cohen's kappa against known answers and an independent
reference, the counts and shares against the codes that built the synthetic sheets, the denominators, the gate (a
sheet that is not locked, an invalid adjudication or a mismatched key refuses, and the key file is never opened before
every other check has passed: an audit hook records each open of its path), and the output's modes. Every package here
is built by the real builder from SYNTHETIC records (test_audit_package.World); the real audit's sheets are never read.
Expectations are computed from the codes the fixtures were filled with, never from the tool's own output."""

import csv
import io
import json
import os
import stat
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import pytest
from test_audit_package import World, store  # noqa: F401  (store is a fixture)

from harness.analysis import audit_analysis as aa
from harness.analysis import audit_package as ap
from harness.analysis import stats

UNI, RHEA = "uniprot-sp-2026_03", "rhea-142"

# --- the audit hook: which paths were opened -----------------------------------------------------------------------

WATCHED: set[str] = set()
OPENED: list[str] = []


def _hook(event, args):
    if event == "open" and WATCHED:
        try:
            p = os.fsdecode(args[0])
        except (TypeError, ValueError):
            return
        if p in WATCHED:
            OPENED.append(p)


sys.addaudithook(_hook)


@pytest.fixture(autouse=True)
def _clear_hook():
    WATCHED.clear()
    OPENED.clear()
    yield
    WATCHED.clear()
    OPENED.clear()


# --- an independent reference kappa --------------------------------------------------------------------------------

def ref_kappa(a, b):
    """kappa from the contingency table with exact fractions (an implementation apart from the tool's)."""
    labels = sorted(set(a) | set(b))
    n = len(a)
    table = {(x, y): Fraction(0) for x in labels for y in labels}
    for x, y in zip(a, b, strict=True):
        table[(x, y)] += 1
    po = sum(table[(x, x)] for x in labels) / n
    pe = sum(sum(table[(x, y)] for y in labels) * sum(table[(y, x)] for y in labels) for x in labels) / (n * n)
    return None if pe == 1 else (po - pe) / (1 - pe)


def seq(table):
    """Two label sequences from {(a_label, b_label): count}."""
    a, b = [], []
    for (x, y), k in table.items():
        a += [x] * k
        b += [y] * k
    return a, b


# --- kappa --------------------------------------------------------------------------------------------------------

def test_kappa_two_by_two_known_answer():
    a, b = seq({("y", "y"): 20, ("y", "n"): 5, ("n", "y"): 10, ("n", "n"): 15})
    r = aa.cohen_kappa(a, b)
    assert (r["n"], r["agree"]) == (50, 35)
    assert r["observed_agreement"] == 0.7 and r["expected_agreement"] == 0.5 and r["kappa"] == 0.4
    assert r["reason"] is None


@pytest.mark.parametrize("table,expected", [
    ({("x", "x"): 7, ("y", "y"): 5}, 1.0),  # perfect
    ({("x", "x"): 1, ("x", "y"): 1, ("y", "x"): 1, ("y", "y"): 1}, 0.0),  # chance
    ({("x", "y"): 2, ("y", "x"): 2}, -1.0),  # complete disagreement on balanced margins
    ({("x", "x"): 1, ("x", "y"): 3, ("y", "x"): 3, ("y", "y"): 1}, -0.5),
])
def test_kappa_perfect_chance_and_negative(table, expected):
    a, b = seq(table)
    assert aa.cohen_kappa(a, b)["kappa"] == expected
    assert Fraction(expected).limit_denominator(1000) == ref_kappa(a, b)


def test_kappa_three_labels_by_hand():
    # 4 of 6 agree; margins a: 2,2,2 and b: 2,3,1 give p_e = 12/36, so kappa = (2/3 - 1/3) / (1 - 1/3) = 0.5
    r = aa.cohen_kappa(list("112233"), list("122231"))
    assert (r["agree"], r["n"], r["kappa"]) == (4, 6, 0.5)
    assert r["expected_agreement"] == round(1 / 3, 6)


def test_kappa_is_null_with_a_reason_when_undefined():
    r = aa.cohen_kappa(["x"] * 5, ["x"] * 5)  # p_e = 1: not 1, not a crash
    assert r["kappa"] is None and "p_e = 1" in r["reason"] and r["observed_agreement"] == 1.0
    assert aa.cohen_kappa([], [])["kappa"] is None
    assert "no item" in aa.cohen_kappa([], [])["reason"]
    one_sided = aa.cohen_kappa(["x"] * 4, ["x", "x", "x", "y"])  # defined: only a's margin is one label
    assert one_sided["kappa"] == 0.0


def test_kappa_matches_the_reference_on_many_tables():
    import random
    rng = random.Random(3)
    for _ in range(200):
        labels = list("abcd")[: rng.randint(2, 4)]
        n = rng.randint(1, 30)
        a, b = [rng.choice(labels) for _ in range(n)], [rng.choice(labels) for _ in range(n)]
        ref, got = ref_kappa(a, b), aa.cohen_kappa(a, b)["kappa"]
        assert (got is None) == (ref is None)
        if ref is not None:
            assert got == pytest.approx(float(ref), abs=1e-6)


def test_cell_shares_and_intervals():
    assert aa.cell(0, 0) == {"count": 0, "n": 0, "share": None, "ci95": None}
    c = aa.cell(8, 32)
    lo, hi = stats.clopper_pearson(8, 32)
    assert c["share"] == 0.25 and c["ci95"] == [round(lo, 6), round(hi, 6)] and c["count"] == 8 and c["n"] == 32
    z = aa.cell(0, 5)
    assert z["share"] == 0.0 and z["ci95"][0] == 0.0
    with pytest.raises(aa.AnalysisError):
        aa.count_table(["nope"], ["a", "b"])


# --- a synthetic audit ---------------------------------------------------------------------------------------------

N_A, N_B, N_C = 10, 8, 6
KINDS = ["A"] * N_A + ["B"] * N_B + ["C"] * N_C
A_ERR = ["aggregation", "constant", "filter_scope", "path", "shape", "other", "scorer", "gold_or_question",
         "aggregation", "aggregation"]
A_DET = ["yes", "no", "no", "unsure", "no", "yes", "no", "no", "yes", "no"]
B_CAUSE = ["data_clash", "closed_world", "closed_world", "undeclared_term", "undeclared_term", "undeclared_term",
           "checker_defect", "other"]
B_REAL = ["yes", "yes", "yes", "yes", "yes", "yes", "no", "unsure"]
C_FOLLOWS = ["yes", "yes", "no", "unsure", "yes", "no"]
C_WRONG = ["yes", "no", "", "", "yes", ""]
B_RULES = ["rule-vocab", "rule-range", "rule-range", "rule-vocab", "rule-vocab", "rule-vocab", "rule-disj",
           "rule-disj"]


def truth_codes(n, kinds=KINDS):
    """The true (adjudicated) codes of the world's item n (the j-th of its kind)."""
    kind = kinds[n]
    j = n - kinds.index(kind)
    if kind == "A":
        return "A", {"error_type": A_ERR[j], "error_type_note": "a note" if A_ERR[j] == "other" else "",
                     "structurally_detectable": A_DET[j]}
    if kind == "B":
        return "B", {"cause": B_CAUSE[j], "cause_note": "a note" if B_CAUSE[j] == "other" else "",
                     "answer_really_correct": B_REAL[j]}
    return "C", {"follows_from_ontology": C_FOLLOWS[j], "witness": "quoted" if C_FOLLOWS[j] == "yes" else "",
                 "statement_wrong_or_misleading": C_WRONG[j]}


def second_annotator(n, codes):
    """Annotator 2's codes: the truth with a few deliberate differences."""
    c = dict(codes)
    flips = {1: {"error_type": "path"}, 3: {"error_type": "constant"}, 0: {"structurally_detectable": "no"},
             2: {"structurally_detectable": "unsure"},
             N_A + 1: {"cause": "data_clash"}, N_A + 4: {"cause": "closed_world"},
             N_A + 7: {"answer_really_correct": "yes"},
             N_A + N_B + 2: {"follows_from_ontology": "unsure"},
             N_A + N_B + 3: {"follows_from_ontology": "yes", "witness": "other quote",
                             "statement_wrong_or_misleading": "no"},
             N_A + N_B + 1: {"statement_wrong_or_misleading": "yes"}}
    c.update(flips.get(n, {}))
    return c


def write_sheet(pkg, annotator, dest, codes_by_iid, *, shuffle=False):
    rows = list(csv.DictReader((pkg / "annotators" / annotator / "sheet.csv").open(newline="", encoding="utf-8")))
    if shuffle:
        rows.sort(key=lambda r: r["item_id"])
    for r in rows:
        r.update(codes_by_iid[r["item_id"]])
    buf = io.StringIO()
    wr = csv.DictWriter(buf, fieldnames=ap.SHEET_COLUMNS, lineterminator="\n")
    wr.writeheader()
    wr.writerows(rows)
    dest.write_text(buf.getvalue(), encoding="utf-8")


class Audit:
    """A synthetic audit: a package built by the real builder, both sheets filled, the adjudicated codes, the key."""

    def __init__(self, tmp_path, blocks, *, kinds=KINDS, flip=True, seed="00ff" * 8):
        w = World(kinds, split="test", block_root=blocks)
        for n in range(len(kinds)):
            graph, group = (UNI if n % 2 == 0 else RHEA), ("claude" if n % 3 else "open-weight")
            w.items[n]["graph"], w.items[n]["model_group"] = graph, group
            w.update_record(n, lambda rec, g=graph: rec["store"].update(name=g))
        files, key_text = w.build(seed=seed)
        self.tmp = tmp_path
        self.pkg, self.key = tmp_path / "pkg", tmp_path / "key.json"
        ap.write_package(self.pkg, self.key, files, key_text, False)
        self.key_doc = json.loads(key_text)
        self.manifest = json.loads((self.pkg / "manifest.json").read_text())
        self.index = {iid: int(it["sample_item"]["run_id"].rsplit("-", 1)[1])
                      for iid, it in self.key_doc["items"].items()}
        self.meta = {iid: it["sample_item"] for iid, it in self.key_doc["items"].items()}
        self.truth = {iid: truth_codes(n, kinds)[1] for iid, n in self.index.items()}
        self.form = {iid: truth_codes(n, kinds)[0] for iid, n in self.index.items()}
        self.sheets = {}
        for a, mk in (("annotator-1", lambda n, c: c),
                      ("annotator-2", second_annotator if flip else lambda n, c: c)):
            dest = tmp_path / f"{a}.csv"
            write_sheet(self.pkg, a, dest, {i: mk(self.index[i], self.truth[i]) for i in self.truth})
            self.sheets[a] = dest
        self.adj = tmp_path / "adjudicated.csv"
        write_sheet(self.pkg, "annotator-1", self.adj, self.truth, shuffle=True)
        self.out = tmp_path / "out" / "result"

    def args(self, **over):
        a = {"package": self.pkg, "sheets": dict(self.sheets), "adjudicated": self.adj, "key": self.key,
             "out": self.out, "flags": None}
        a.update(over)
        return a

    def argv(self, **over):
        a = self.args(**over)
        argv = ["--package", str(a["package"]), "--adjudicated", str(a["adjudicated"]), "--key", str(a["key"]),
                "--out", str(a["out"])]
        for name, p in a["sheets"].items():
            argv += ["--sheet", f"{name}={p}"]
        if a["flags"]:
            argv += ["--flags", str(a["flags"])]
        return argv

    def run(self, **over):
        a = self.args(**over)
        return aa.analyse(a["package"], a["sheets"], a["adjudicated"], a["key"], a["flags"])

    def result(self, **over):
        return self.run(**over)[0]


@pytest.fixture
def audit(tmp_path, store):  # noqa: F811
    return Audit(tmp_path, store)


def codes_of(audit_, col, forms=None):
    return [c[col] for i, c in audit_.truth.items() if col in c and (forms is None or audit_.form[i] in forms)]


def test_a_full_audit_gives_the_kappas_of_the_sheets(audit):
    r = audit.result()
    for form, field in (("A", "error_type"), ("A", "structurally_detectable"), ("B", "cause"),
                        ("B", "answer_really_correct"), ("C", "follows_from_ontology")):
        ids = sorted(i for i in audit.truth if audit.form[i] == form)
        a1 = [audit.truth[i][field] for i in ids]
        a2 = [second_annotator(audit.index[i], audit.truth[i])[field] for i in ids]
        got = r["agreement"][form]["fields"][field]
        assert got["n"] == len(ids) and got["agree"] == sum(x == y for x, y in zip(a1, a2))
        assert got["kappa"] == pytest.approx(float(ref_kappa(a1, a2)), abs=1e-6)
        assert got["primary"] == (field in ("error_type", "cause", "follows_from_ontology"))
        assert got["disagreements"] == got["n"] - got["agree"]
    assert r["agreement"]["A"]["fields"]["error_type"]["disagreements"] == 2
    assert r["items"] == {"total": 24, "by_form": {"A": 10, "B": 8, "C": 6}}


def test_the_conditional_code_is_over_the_items_both_annotators_answered_yes_to(audit):
    f = audit.result()["agreement"]["C"]["fields"]["statement_wrong_or_misleading"]
    # follows: truth yes at j = 0, 1, 4; annotator 2 also says yes at j = 3 (unsure for 1) and unsure at j = 2
    by_j = {audit.index[i] - N_A - N_B: i for i in audit.truth if audit.form[i] == "C"}
    a1 = [audit.truth[by_j[j]]["statement_wrong_or_misleading"] for j in (0, 1, 4)]
    a2 = [second_annotator(N_A + N_B + j, audit.truth[by_j[j]])["statement_wrong_or_misleading"] for j in (0, 1, 4)]
    assert f["n"] == 3 and f["coded_by_one_only"] == 1  # j = 3 was coded by annotator 2 alone
    assert f["agree"] == sum(x == y for x, y in zip(a1, a2))
    assert f["kappa"] == pytest.approx(float(ref_kappa(a1, a2)), abs=1e-6)


def test_per_label_kappa_is_one_versus_rest(audit):
    ids = sorted(i for i in audit.truth if audit.form[i] == "A")
    a1 = [audit.truth[i]["error_type"] for i in ids]
    a2 = [second_annotator(audit.index[i], audit.truth[i])["error_type"] for i in ids]
    per = audit.result()["agreement"]["A"]["fields"]["error_type"]["per_label"]
    for lab in ap.CODEBOOK["A"][0].labels:
        ref = ref_kappa([x == lab for x in a1], [y == lab for y in a2])
        assert per[lab]["annotator_1"] == a1.count(lab) and per[lab]["annotator_2"] == a2.count(lab)
        if ref is None:
            assert per[lab]["kappa"] is None and "p_e = 1" in per[lab]["reason"]  # a label neither used
        else:
            assert per[lab]["kappa"] == pytest.approx(float(ref), abs=1e-6)


def test_h4_counts_and_shares_with_their_denominators(audit):
    h4 = audit.result()["H4_residue"]
    err = Counter(codes_of(audit, "error_type", "A"))
    allv = h4["error_type"]["overall"]["all_items"]
    assert allv["n"] == 10 and set(allv["labels"]) == set(ap.CODEBOOK["A"][0].labels)
    for lab, c in allv["labels"].items():
        assert (c["count"], c["n"], c["share"]) == (err[lab], 10, round(err[lab] / 10, 6))
        assert c["ci95"] == [round(x, 6) for x in stats.clopper_pearson(err[lab], 10)]
    # misses only: the wrong gold / ambiguous question and the scorer error are no agent's miss: 10 - 2 = 8
    miss = h4["error_type"]["overall"]["misses_only"]
    assert miss["n"] == 8
    assert set(miss["labels"]) == {"aggregation", "constant", "filter_scope", "path", "shape", "other"}
    assert miss["labels"]["aggregation"]["share"] == round(3 / 8, 6)
    assert sum(c["count"] for c in miss["labels"].values()) == 8
    det = h4["structurally_detectable"]["overall"]
    assert det["all_items"]["labels"]["no"]["count"] == 6 and det["all_items"]["labels"]["no"]["n"] == 10
    # misses only drops j = 6 (scorer, no) and j = 7 (gold_or_question, no): 4 'no' of 8
    assert (det["misses_only"]["labels"]["no"]["count"], det["misses_only"]["n"]) == (4, 8)
    assert h4["other_notes"] == [{"item_id": next(i for i in audit.truth if audit.truth[i]["error_type"] == "other"),
                                  "note": "a note"}]


def test_h4_by_graph_and_by_model_group_partition_the_items(audit):
    h4 = audit.result()["H4_residue"]["error_type"]
    for part, key in (("by_graph", "graph"), ("by_model_group", "model_group")):
        want = Counter(audit.meta[i][key] for i in audit.truth if audit.form[i] == "A")
        assert {g: v["all_items"]["n"] for g, v in h4[part].items()} == dict(want)
        total = Counter()
        for v in h4[part].values():
            for lab, c in v["all_items"]["labels"].items():
                total[lab] += c["count"]
        assert total == Counter(codes_of(audit, "error_type", "A"))


def test_false_alarm_causes_over_all_items_and_over_really_correct_ones(audit):
    fa = audit.result()["false_alarms"]
    assert fa["source_kind_basis"].startswith("'curated'") and fa["by_rule"] is None
    assert list(fa["by_source_kind"]) == ["curated"]
    cause = fa["by_source_kind"]["curated"]["cause"]["overall"]
    allv, real = cause["all_items"], cause["answer_really_correct_yes"]
    assert allv["n"] == 8 and real["n"] == 6  # B items 6 and 7 were not really correct (no, unsure)
    c = Counter(B_CAUSE)
    assert {k: v["count"] for k, v in allv["labels"].items()} == {
        k: c.get(k, 0) for k in ("undeclared_term", "closed_world", "data_clash", "schema_too_narrow",
                                 "checker_defect", "other")}
    assert allv["labels"]["closed_world"]["share"] == round(2 / 8, 6)
    assert real["labels"]["undeclared_term"]["share"] == round(3 / 6, 6)
    assert real["labels"]["checker_defect"]["count"] == 0 and real["labels"]["other"]["count"] == 0
    # the three causes the task names, each its own row, and the rest kept apart
    assert {"data_clash", "closed_world", "undeclared_term", "schema_too_narrow", "checker_defect",
            "other"} == set(allv["labels"])
    rc = fa["by_source_kind"]["curated"]["answer_really_correct"]["overall"]["all_items"]
    assert (rc["labels"]["yes"]["count"], rc["n"]) == (6, 8)
    assert fa["other_notes"] == [{"item_id": next(i for i in audit.truth if audit.truth[i].get("cause") == "other"),
                                  "note": "a note"}]


def test_h6_denominators(audit):
    h6 = audit.result()["H6_shared_premise"]
    f = h6["follows_from_ontology"]["overall"]["all_items"]
    assert f["n"] == 6 and f["labels"]["yes"]["count"] == 3 and f["labels"]["no"]["count"] == 2
    s = h6["statement_wrong_or_misleading"]["overall"]["follows_yes"]
    assert s["n"] == 3 and s["labels"]["yes"]["count"] == 2 and s["labels"]["no"]["count"] == 1


def test_a_form_with_no_items_gives_null_shares(tmp_path, store):  # noqa: F811
    a = Audit(tmp_path, store, kinds=["A"] * 4 + ["B"] * 2, flip=False)
    r = a.result()
    assert r["items"]["by_form"] == {"A": 4, "B": 2}
    h6 = r["H6_shared_premise"]
    assert h6["follows_from_ontology"]["overall"]["all_items"]["n"] == 0
    assert all(c["share"] is None and c["ci95"] is None
               for c in h6["follows_from_ontology"]["overall"]["all_items"]["labels"].values())
    assert r["agreement"]["C"]["fields"]["follows_from_ontology"]["kappa"] is None


# --- the flag table -----------------------------------------------------------------------------------------------

def flags_file(audit_, path, rows=None, header="run_id,source,rule"):
    if rows is None:
        rows = [(audit_.meta[i]["run_id"], "curated", B_RULES[audit_.index[i] - N_A])
                for i in audit_.truth if audit_.form[i] == "B"]
    path.write_text(header + "\n" + "".join(",".join(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_the_flag_table_adds_cause_by_rule_within_a_source_kind(audit):
    fa = audit.result(flags=flags_file(audit, audit.tmp / "flags.csv"))["false_alarms"]
    by_rule = fa["by_rule"]["curated"]
    assert set(by_rule) == set(B_RULES)
    for rule in set(B_RULES):
        want = Counter(B_CAUSE[j] for j in range(N_B) if B_RULES[j] == rule)
        got = by_rule[rule]["all_items"]
        assert got["n"] == sum(want.values())
        assert {k: v["count"] for k, v in got["labels"].items() if v["count"]} == dict(want)
    assert by_rule["rule-disj"]["answer_really_correct_yes"]["n"] == 0  # both its items were not really correct
    assert by_rule["rule-disj"]["answer_really_correct_yes"]["labels"]["other"]["share"] is None
    assert fa["source_kind_basis"] == "the flag table"


@pytest.mark.parametrize("what", ["missing", "extra", "duplicate", "empty rule", "header", "a non-B run"])
def test_a_flag_table_that_does_not_fit_the_b_items_is_refused(audit, what):
    rows = [(audit.meta[i]["run_id"], "curated", "r") for i in audit.truth if audit.form[i] == "B"]
    header = "run_id,source,rule"
    if what == "missing":
        rows = rows[1:]
    elif what == "extra":
        rows.append(("SENTINEL-RUN-NOPE", "curated", "r"))
    elif what == "duplicate":
        rows.append(rows[0])
    elif what == "empty rule":
        rows[0] = (rows[0][0], "curated", "")
    elif what == "header":
        header = "run_id,source"
    elif what == "a non-B run":
        rows[0] = (next(audit.meta[i]["run_id"] for i in audit.truth if audit.form[i] == "A"), "curated", "r")
    with pytest.raises(aa.AnalysisError):
        audit.result(flags=flags_file(audit, audit.tmp / "f.csv", rows, header))


# --- the gate ------------------------------------------------------------------------------------------------------

def refused(audit_, match, **over):
    WATCHED.add(str(audit_.key))
    with pytest.raises(aa.AnalysisError, match=match):
        audit_.run(**over)
    return list(OPENED)


def test_the_key_is_opened_once_and_only_after_everything_else_passed(audit):
    WATCHED.add(str(audit.key))
    audit.run()
    assert OPENED == [str(audit.key)]


def test_a_blank_sheet_is_not_locked_and_the_key_is_not_read(audit):
    blank = audit.tmp / "blank.csv"
    blank.write_text((audit.pkg / "annotators/annotator-2/sheet.csv").read_text(), encoding="utf-8")
    assert refused(audit, "annotator-2's sheet is not locked", sheets={**audit.sheets, "annotator-2": blank}) == []


@pytest.mark.parametrize("mutation,match", [
    (lambda r: r.update(error_type="mystery") if r["form"] == "A" else None, "not a codebook label"),
    (lambda r: r.update(error_type="other", error_type_note="") if r["form"] == "A" else None, "other needs a note"),
    (lambda r: r.update(cause="") if r["form"] == "B" else None, "is empty"),
    (lambda r: r.update(witness="") if r["form"] == "C" and r["follows_from_ontology"] == "yes" else None,
     "without a quoted witness"),
    (lambda r: r.update(cause="closed_world") if r["form"] == "A" else None, "is not a Form A column"),
])
def test_an_invalid_sheet_is_refused_and_the_key_is_not_read(audit, mutation, match):
    rows = list(csv.DictReader(audit.sheets["annotator-1"].open(newline="", encoding="utf-8")))
    for r in rows:
        mutation(r)
    buf = io.StringIO()
    wr = csv.DictWriter(buf, fieldnames=ap.SHEET_COLUMNS, lineterminator="\n")
    wr.writeheader()
    wr.writerows(rows)
    bad = audit.tmp / "bad.csv"
    bad.write_text(buf.getvalue(), encoding="utf-8")
    assert refused(audit, match, sheets={**audit.sheets, "annotator-1": bad}) == []


def test_a_sheet_of_another_order_or_missing_rows_is_refused_before_the_key(audit):
    lines = audit.sheets["annotator-1"].read_text().splitlines()
    short = audit.tmp / "short.csv"
    short.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    assert refused(audit, "not locked", sheets={**audit.sheets, "annotator-1": short}) == []
    swapped = audit.tmp / "swapped.csv"
    swapped.write_text("\n".join([lines[0], lines[2], lines[1], *lines[3:]]) + "\n", encoding="utf-8")
    assert refused(audit, "not locked", sheets={**audit.sheets, "annotator-1": swapped}) == []


def test_a_malformed_sheet_is_refused_not_crashed(audit):
    ragged = audit.tmp / "ragged.csv"
    ragged.write_text(audit.sheets["annotator-1"].read_text() + "T999,x\n", encoding="utf-8")
    assert refused(audit, "annotator-1's sheet", sheets={**audit.sheets, "annotator-1": ragged}) == []
    binary = audit.tmp / "binary.csv"
    binary.write_bytes(b"\xff\xfe\x00bad")
    assert refused(audit, "annotator-1's sheet", sheets={**audit.sheets, "annotator-1": binary}) == []
    assert refused(audit, "cannot be read", sheets={**audit.sheets, "annotator-1": audit.tmp / "nope.csv"}) == []


def test_both_annotators_sheets_are_needed_and_they_must_be_two_files(audit):
    assert refused(audit, "exactly the package's annotators", sheets={"annotator-1": audit.sheets["annotator-1"]}) == []
    assert refused(audit, "exactly the package's annotators",
                   sheets={**audit.sheets, "annotator-3": audit.sheets["annotator-2"]}) == []
    both = {"annotator-1": audit.sheets["annotator-1"], "annotator-2": audit.sheets["annotator-1"]}
    assert refused(audit, "one file", sheets=both) == []


def test_a_practice_package_is_refused(tmp_path, store):  # noqa: F811
    w = World(("A", "B", "C"), split="dev", block_root=store)
    files, key_text = w.build(seed="00ff" * 8)
    pkg = tmp_path / "pkg"
    ap.write_package(pkg, tmp_path / "key.json", files, key_text, False)
    WATCHED.add(str(tmp_path / "key.json"))
    with pytest.raises(aa.AnalysisError, match="practice"):
        aa.analyse(pkg, {"annotator-1": tmp_path / "a", "annotator-2": tmp_path / "b"}, tmp_path / "c",
                   tmp_path / "key.json")
    assert OPENED == []


def test_an_edited_manifest_or_package_sheet_is_refused_before_the_key(audit):
    sheet = audit.pkg / "annotators/annotator-1/sheet.csv"
    sheet.write_text(sheet.read_text() + "\n", encoding="utf-8")  # no longer the one the manifest names
    assert refused(audit, "not locked") == []


@pytest.mark.parametrize("what,match", [
    ("missing", "have no row"), ("extra", "not an item of the package"), ("duplicate", "two rows"),
    ("form", "is not the package's"), ("label", "not a codebook label"), ("empty", "is empty"),
    ("other form column", "is not a Form"), ("columns", "columns are not the sheet's"),
    ("note", "other needs a note"), ("witness", "without a quoted witness"),
    ("second code", "coded only after a yes"),
])
def test_an_invalid_adjudication_is_refused_before_the_key_is_read(audit, what, match):
    rows = list(csv.DictReader(audit.adj.open(newline="", encoding="utf-8")))
    fields = ap.SHEET_COLUMNS
    a = next(r for r in rows if r["form"] == "A" and r["error_type"] != "other")
    c_no = next(r for r in rows if r["form"] == "C" and r["follows_from_ontology"] == "no")
    c_yes = next(r for r in rows if r["form"] == "C" and r["follows_from_ontology"] == "yes")
    if what == "missing":
        rows.remove(a)
    elif what == "extra":
        rows.append({**a, "item_id": "T999"})
    elif what == "duplicate":
        rows.append(dict(a))
    elif what == "form":
        a["form"] = "B"
    elif what == "label":
        a["error_type"] = "mystery"
    elif what == "empty":
        a["structurally_detectable"] = ""
    elif what == "other form column":
        a["cause"] = "closed_world"
    elif what == "columns":
        fields = fields[:-1]
        rows = [{k: v for k, v in r.items() if k != "comment"} for r in rows]
    elif what == "note":
        a["error_type"], a["error_type_note"] = "other", ""
    elif what == "witness":
        c_yes["witness"] = ""
    elif what == "second code":
        c_no["statement_wrong_or_misleading"] = "yes"
    buf = io.StringIO()
    wr = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    wr.writeheader()
    wr.writerows(rows)
    bad = audit.tmp / "bad-adj.csv"
    bad.write_text(buf.getvalue(), encoding="utf-8")
    assert refused(audit, match, adjudicated=bad) == []


def swap_run_ids(doc):
    """Two items exchange their run ids: the same set of outputs, but not the ids the seed gives them."""
    a, b = list(doc["items"])[:2]
    ia, ib = doc["items"][a]["sample_item"], doc["items"][b]["sample_item"]
    ia["run_id"], ib["run_id"] = ib["run_id"], ia["run_id"]


def rewrite_key(audit_, edit, *, keep_manifest_hash=False):
    doc = json.loads(audit_.key.read_text())
    edit(doc)
    text = ap._dump(doc)
    bad = audit_.tmp / "bad-key.json"
    bad.write_text(text, encoding="utf-8")
    if not keep_manifest_hash:  # a tampered key the manifest has been made to name: only the deeper checks refuse it
        m = json.loads((audit_.pkg / "manifest.json").read_text())
        m["key_sha256"] = ap._sha256_bytes(text.encode())
        (audit_.pkg / "manifest.json").write_text(ap._dump(m), encoding="utf-8")
    return bad


def test_a_key_the_manifest_does_not_name_is_refused(audit):
    bad = rewrite_key(audit, lambda d: d.update(split="dev"), keep_manifest_hash=True)
    WATCHED.add(str(bad))
    with pytest.raises(aa.AnalysisError, match="key_sha256"):
        audit.run(key=bad)
    assert OPENED == [str(bad)]  # it was opened (last), and refused


@pytest.mark.parametrize("edit,match", [
    (lambda d: d.update(schema="h4-audit-key/0"), "is not an"),
    (lambda d: d.update(practice=True), "practice key"),
    (lambda d: d.update(private=True), "privacy"),
    (lambda d: d.update(shuffle_seed="f" * 32), "seed"),
    (lambda d: d["files_sha256"].update({"HANDLING.md": "0" * 64}), "file hashes"),
    (lambda d: d["items"].pop(next(iter(d["items"]))), "not the sheets' items"),
    (lambda d: d["items"][next(iter(d["items"]))]["sample_item"].update(kind="C"), "kind is not"),
    (lambda d: swap_run_ids(d), "item ids are not"),
    (lambda d: d["orders"].update({"annotator-1": list(reversed(d["orders"]["annotator-1"]))}), "orders"),
])
def test_a_key_that_does_not_match_the_package_or_sheets_is_refused(audit, edit, match):
    bad = rewrite_key(audit, edit)
    with pytest.raises(aa.AnalysisError, match=match):
        audit.run(key=bad)


def test_the_key_of_another_build_is_refused(tmp_path, store):  # noqa: F811
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    one = Audit(tmp_path / "one", store)
    other = Audit(tmp_path / "two", store, seed="11aa" * 8)  # the same sample, another shuffle: another key
    assert one.key.read_bytes() != other.key.read_bytes()
    with pytest.raises(aa.AnalysisError, match="key_sha256"):
        one.run(key=other.key)


def test_a_public_package_with_a_private_graphs_item_in_its_key_is_refused(audit):
    bad = rewrite_key(audit, lambda d: d["items"][next(iter(d["items"]))]["sample_item"].update(graph="a15"))
    with pytest.raises(aa.AnalysisError, match="private graph"):
        audit.run(key=bad)


# --- adjudication --------------------------------------------------------------------------------------------------

def test_an_override_of_an_agreed_code_is_counted_and_a_resolved_disagreement_is_not(audit):
    # the adjudicated codes equal annotator 1's; change one agreed code and settle one disagreement differently
    agreed = next(i for i in audit.truth if audit.form[i] == "A" and audit.index[i] == 4)  # both: shape
    split = next(i for i in audit.truth if audit.form[i] == "A" and audit.index[i] == 1)  # 1: constant, 2: path
    codes = {i: dict(c) for i, c in audit.truth.items()}
    codes[agreed]["error_type"] = "path"
    codes[split]["error_type"] = "shape"  # neither annotator's: allowed
    write_sheet(audit.pkg, "annotator-1", audit.adj, codes)
    adj = audit.result()["adjudication"]
    assert adj["overrides_of_agreed_codes"] == 1
    assert adj["overrides"] == [{"item_id": agreed, "field": "error_type", "annotators": "shape",
                                 "adjudicated": "path"}]
    # the tables use the adjudicated codes
    assert audit.result()["H4_residue"]["error_type"]["overall"]["all_items"]["labels"]["shape"]["count"] == 1
    assert "path" in audit.result()["H4_residue"]["error_type"]["overall"]["all_items"]["labels"]


def c_item(audit_, j):
    return next(i for i in audit_.truth if audit_.form[i] == "C" and audit_.index[i] == N_A + N_B + j)


def test_a_form_c_override_counts_once_and_an_empty_cell_is_no_agreed_code(tmp_path, store):  # noqa: F811
    a = Audit(tmp_path, store, flip=False)  # both annotators code alike everywhere
    # follows = no (j = 2: both coded it, neither coded the second code): the adjudication makes it a yes
    no_item = c_item(a, 2)
    codes = {i: dict(c) for i, c in a.truth.items()}
    codes[no_item].update(follows_from_ontology="yes", witness="q", statement_wrong_or_misleading="no")
    write_sheet(a.pkg, "annotator-1", a.adj, codes)
    adj = a.result()["adjudication"]
    assert adj["overrides_of_agreed_codes"] == 1  # one decision; the empty second code was nobody's
    assert adj["overrides"] == [{"item_id": no_item, "field": "follows_from_ontology", "annotators": "no",
                                 "adjudicated": "yes"}]
    # follows = yes with a second code (j = 0: both "yes" + "yes"): the adjudication makes the first a no
    yes_item = c_item(a, 0)
    codes = {i: dict(c) for i, c in a.truth.items()}
    codes[yes_item].update(follows_from_ontology="no", witness="", statement_wrong_or_misleading="")
    write_sheet(a.pkg, "annotator-1", a.adj, codes)
    adj = a.result()["adjudication"]
    assert adj["overrides_of_agreed_codes"] == 1  # the dropped second code follows its parent's override
    assert adj["overrides"] == [{"item_id": yes_item, "field": "follows_from_ontology", "annotators": "yes",
                                 "adjudicated": "no"}]
    # the parent kept, the second code overridden: that one counts
    codes = {i: dict(c) for i, c in a.truth.items()}
    codes[yes_item].update(statement_wrong_or_misleading="no")
    write_sheet(a.pkg, "annotator-1", a.adj, codes)
    adj = a.result()["adjudication"]
    assert adj["overrides"] == [{"item_id": yes_item, "field": "statement_wrong_or_misleading",
                                 "annotators": "yes", "adjudicated": "no"}]
    # no override at all
    write_sheet(a.pkg, "annotator-1", a.adj, a.truth)
    assert a.result()["adjudication"] == {"overrides_of_agreed_codes": 0, "overrides": []}


def test_the_flag_table_is_checked_before_the_key_is_opened(audit):
    bad = audit.tmp / "bad-flags.csv"
    bad.write_text("run_id,source\n", encoding="utf-8")
    assert refused(audit, "columns are not", flags=bad) == []


def test_check_codes_agrees_with_check_sheet_on_the_same_mutations(audit):
    """The adjudicated file's validator (own code) and the builder's check_sheet find problems in the same cases."""
    base = list(csv.DictReader(audit.sheets["annotator-1"].open(newline="", encoding="utf-8")))
    want = {r["item_id"]: r["form"] for r in base}
    mutations = [
        lambda rs: None,
        lambda rs: rs[0].update(error_type="x", cause="x", follows_from_ontology="x"),
        lambda rs: [r.update(structurally_detectable="") for r in rs if r["form"] == "A"],
        lambda rs: [r.update(error_type="other", error_type_note="") for r in rs if r["form"] == "A"][:1],
        lambda rs: [r.update(witness="") for r in rs if r["form"] == "C" and r["follows_from_ontology"] == "yes"],
        lambda rs: [r.update(statement_wrong_or_misleading="yes") for r in rs
                    if r["form"] == "C" and r["follows_from_ontology"] == "no"],
        lambda rs: [r.update(cause="closed_world") for r in rs if r["form"] == "A"],
        lambda rs: [r.update(answer_really_correct="") for r in rs if r["form"] == "B"],
    ]
    for n, mutate in enumerate(mutations):
        rows = [dict(r) for r in base]
        mutate(rows)
        buf = io.StringIO()
        wr = csv.DictWriter(buf, fieldnames=ap.SHEET_COLUMNS, lineterminator="\n")
        wr.writeheader()
        wr.writerows(rows)
        path = audit.tmp / f"m{n}.csv"
        path.write_text(buf.getvalue(), encoding="utf-8")
        sheet_problems = ap.check_sheet(path, audit.pkg, "annotator-1", complete=True)
        assert bool(aa.check_codes(rows, want)) == bool(sheet_problems), n


# --- output ---------------------------------------------------------------------------------------------------------

def test_main_writes_json_and_text_with_closed_modes_and_prints_nothing_of_them(audit, capsys):
    assert aa.main(audit.argv()) == 0
    printed = capsys.readouterr().out
    assert "kappa" not in printed and "share" not in printed and printed.startswith("wrote ")
    assert stat.S_IMODE(audit.out.stat().st_mode) == 0o700
    for name in ("audit-analysis.json", "audit-analysis.txt"):
        assert stat.S_IMODE((audit.out / name).stat().st_mode) == 0o600
    doc = json.loads((audit.out / "audit-analysis.json").read_text())
    assert doc["schema"] == aa.RESULT_SCHEMA and doc["private"] is False
    assert doc["tool"]["audit_analysis_sha256"] == ap._sha256_bytes(Path(aa.__file__).read_bytes())
    assert doc["inputs"]["sheets_sha256"] == {a: ap._sha256_bytes(p.read_bytes()) for a, p in audit.sheets.items()}
    assert doc["inputs"]["key_sha256"] == audit.manifest["key_sha256"]
    text = (audit.out / "audit-analysis.txt").read_text()
    assert "Cohen's kappa" in text and "H4 residue" in text and "False alarms" in text and "H6 shared premise" in text
    assert "aggregation" in text and "closed_world" in text


def test_print_shows_the_text_table_of_a_public_package(audit, capsys):
    assert aa.main([*audit.argv(), "--print"]) == 0
    assert "Cohen's kappa" in capsys.readouterr().out


def test_a_refusal_exits_2_and_writes_nothing(audit, capsys):
    blank = audit.tmp / "blank.csv"
    blank.write_text((audit.pkg / "annotators/annotator-2/sheet.csv").read_text(), encoding="utf-8")
    assert aa.main(audit.argv(sheets={**audit.sheets, "annotator-2": blank})) == 2
    assert "audit analysis refused" in capsys.readouterr().err
    assert not audit.out.exists()
    assert not list(audit.out.parent.glob(".audit-analysis-*"))


def test_the_output_must_be_new_or_empty_and_outside_the_repository(audit, capsys):
    audit.out.mkdir(parents=True)
    (audit.out / "x").write_text("1")
    assert aa.main(audit.argv()) == 2
    assert aa.main(audit.argv(out=aa.ROOT / "audit-analysis-out")) == 2
    assert not (aa.ROOT / "audit-analysis-out").exists()
    capsys.readouterr()


def test_a_failed_write_leaves_nothing(audit, monkeypatch):
    def boom(*a, **k):
        raise OSError("disk")
    monkeypatch.setattr(aa.os, "rename", boom)
    with pytest.raises(OSError):
        aa.write_outputs(audit.out, {"a": "1"}, False)
    assert not audit.out.exists() and not list(audit.out.parent.glob(".audit-analysis-*"))


def test_the_text_report_names_an_undefined_kappa_and_the_flag_table_absence(tmp_path, store):  # noqa: F811
    a = Audit(tmp_path, store, kinds=["A"] * 3, flip=False)
    for name in ("annotator-1", "annotator-2"):  # both annotators code every item alike: p_e = 1 for error_type
        write_sheet(a.pkg, name, a.sheets[name], {i: {"error_type": "path", "error_type_note": "",
                                                      "structurally_detectable": "yes"} for i in a.truth})
    write_sheet(a.pkg, "annotator-1", a.adj, {i: {"error_type": "path", "error_type_note": "",
                                                  "structurally_detectable": "yes"} for i in a.truth})
    r, text = a.run()
    f = r["agreement"]["A"]["fields"]["error_type"]
    assert f["kappa"] is None and f["observed_agreement"] == 1.0 and "undefined" in text
    assert "no flag table" in text


# --- a private package -----------------------------------------------------------------------------------------------

def test_a_private_package_is_analysed_into_a_closed_directory_and_prints_nothing(
        tmp_path, store, capsys):  # noqa: F811
    kinds = ["A"] * 4 + ["B"] * 3
    w = World(kinds, graph="a15", split="test", private=True, block_root=store)
    files, key_text = w.build(seed="00ff" * 8, private=True)
    base = tmp_path / "private"
    base.mkdir(mode=0o700)
    os.chmod(base, 0o700)
    pkg, key = base / "pkg", base / "key.json"
    ap.write_package(pkg, key, files, key_text, True)
    doc = json.loads(key_text)
    idx = {iid: int(it["sample_item"]["run_id"].rsplit("-", 1)[1]) for iid, it in doc["items"].items()}

    def codes(n):
        if n < 4:
            return {"error_type": "path", "error_type_note": "", "structurally_detectable": "no"}
        return {"cause": "closed_world", "cause_note": "", "answer_really_correct": "yes"}
    sheets = {}
    for a in ("annotator-1", "annotator-2"):
        sheets[a] = base / f"{a}.csv"
        write_sheet(pkg, a, sheets[a], {i: codes(n) for i, n in idx.items()})
    adj = base / "adj.csv"
    write_sheet(pkg, "annotator-1", adj, {i: codes(n) for i, n in idx.items()})
    argv = ["--package", str(pkg), "--adjudicated", str(adj), "--key", str(key), "--print",
            *[x for a, p in sheets.items() for x in ("--sheet", f"{a}={p}")]]
    open_parent = tmp_path / "open"
    open_parent.mkdir()
    os.chmod(open_parent, 0o755)
    assert aa.main([*argv, "--out", str(open_parent / "o")]) == 2  # not under a closed directory
    assert not (open_parent / "o").exists()
    capsys.readouterr()
    assert aa.main([*argv, "--out", str(base / "result")]) == 0
    out = capsys.readouterr().out
    assert out.strip() == "wrote the private package's analysis (mode 600; nothing of it printed)"
    res = json.loads((base / "result" / "audit-analysis.json").read_text())
    assert res["private"] is True and res["items"]["by_form"] == {"A": 4, "B": 3}
    assert stat.S_IMODE((base / "result" / "audit-analysis.json").stat().st_mode) == 0o600
